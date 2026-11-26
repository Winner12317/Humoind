"""Small CPU checks for common style metrics and immutable paired provenance."""
from __future__ import annotations

from collections import OrderedDict
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/analyze_amp_recovery_style.py"
SPEC = importlib.util.spec_from_file_location("amp_recovery_style", SCRIPT)
STYLE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(STYLE)
torch.set_num_threads(1)


class StyleMathTest(unittest.TestCase):
    def test_nearest_distance_is_rms_not_dimension_dependent_l2(self):
        q = np.array([[0., 0.], [2., 2.]])
        ref = np.array([[1., 1.], [2., 2.]])
        np.testing.assert_allclose(STYLE.nearest_rmse(q, ref, batch_size=1), [1., 0.], atol=1e-12)
        np.testing.assert_allclose(STYLE.nearest_rmse(q, q), 0., atol=1e-12)
        self.assertEqual(STYLE.nearest_rmse(np.empty((0, 2)), ref).size, 0)

    def test_sampling_is_capped_repeatable_and_respects_phase(self):
        mask = np.arange(100) % 2 == 0
        first = STYLE.sample_indices(mask, 12, 7)
        np.testing.assert_array_equal(first, STYLE.sample_indices(mask, 12, 7))
        self.assertEqual(len(first), 12)
        self.assertTrue(mask[first].all())

    def test_action_second_difference_masks_boundaries(self):
        t = np.arange(5, dtype=np.float32)[:, None, None]
        action = np.broadcast_to(t ** 2, (5, 1, 29)).copy()
        previous = np.broadcast_to((t - 1) ** 2, action.shape).copy()
        boundary = np.zeros((5, 1), dtype=bool)
        boundary[1] = True
        result = STYLE.smoothness(action, previous, np.ones((5, 1), dtype=bool), boundary, .5)
        self.assertEqual(result["second_difference_rms"]["count"], 3)
        self.assertAlmostEqual(result["second_difference_rms"]["mean"], 2.)
        self.assertAlmostEqual(result["second_derivative_jerk_proxy_rms_per_second_squared"]["mean"], 8.)

    def test_phases_use_transition_start_and_exclude_invalid(self):
        state = np.zeros((110, 1, 71), dtype=np.float32)
        state[..., 2], state[..., 3] = .7, 1.
        state[0, 0, 2] = .4
        valid = np.ones((110, 1), dtype=bool)
        valid[-1] = False
        masks = STYLE.phase_masks(state, valid, .02)
        self.assertEqual(masks["first_2_seconds"].sum(), 100)
        self.assertEqual(masks["low_or_tilted"].sum(), 1)
        self.assertEqual(masks["upright"].sum(), 108)
        self.assertEqual(masks["intermediate"].sum(), 0)

    def test_common_features_reconstruct_ground_height_without_padding(self):
        adapter = STYLE.make_cpu_adapter(STYLE.REPO / "src/assets/robots/unitree_g1/xmls/g1.xml")
        state = np.zeros((2, 71), dtype=np.float32)
        state[:, 3] = 1.
        state[:, 2] = [.4, .8]
        original = state.copy()
        feature = STYLE.features(adapter, state)
        self.assertEqual(feature.shape, (2, 203))
        np.testing.assert_allclose(feature[:, 198], [.4, .8], atol=1e-6)
        np.testing.assert_allclose(feature[0, :195], feature[1, :195], atol=1e-6)
        np.testing.assert_array_equal(state, original)

    def test_frozen_scorer_uses_relu_and_saved_normalizer(self):
        network = torch.nn.Sequential(OrderedDict(
            trunk=torch.nn.Sequential(torch.nn.Linear(406, 2), torch.nn.ReLU()),
            amp_linear=torch.nn.Linear(2, 1)))
        with torch.no_grad():
            for parameter in network.parameters():
                parameter.zero_()
            network.trunk[0].weight[0, 0] = 1.
            network.amp_linear.weight[0, 0] = 1.
        norm = STYLE.RunningNorm(203)
        norm.mean.fill_(1.)
        norm.var.fill_(4.)
        payload = {"format": "amp-wmac-v1", "amp_feature_schema": "grounded_v2",
                   "config": {"discriminator_hidden": [2], "amp_reward_coef": .1},
                   "discriminator": network.state_dict(), "amp_norm": norm.state_dict(),
                   "amp_normalizer_settings": {"eps": 0., "variance_epsilon": True,
                                               "clip": 10., "max_count": float("inf")}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "d.pt"
            torch.save(payload, path)
            scorer = STYLE.load_scorer(path)
            pairs = np.ones((2, 406), dtype=np.float32)
            pairs[:, 0] = [-3., 3.]
            score, reward = STYLE.score_pairs(pairs, scorer)
            np.testing.assert_allclose(score, [0., 1.])
            np.testing.assert_allclose(reward, [.075, .1], atol=1e-7)
            self.assertTrue(all(not p.requires_grad for p in scorer[0].parameters()))
            json.dumps(STYLE.json_clean(scorer[3]), allow_nan=False)


class PairingTest(unittest.TestCase):
    def test_native_schemas_differ_but_physical_initial_arrays_must_match(self):
        with tempfile.TemporaryDirectory() as directory:
            roots = [Path(directory) / name for name in ("old", "new")]
            arrays = {key: np.zeros((2, width), dtype=np.float32) for key, width in
                      (("state", 71), ("context", 35), ("command", 3), ("actor_history", 384))}
            evaluations = []
            for root, task, schema in zip(roots, ("Unitree-G1-AMP-Flat", "Unitree-G1-AMP-Flat-Aligned"), ("legacy", "grounded_v2")):
                root.mkdir()
                np.savez_compressed(root / "initial_recovery.npz", **arrays)
                evaluations.append((root, {"arguments": {"seed": 2030}, "results": {"recovery": {}},
                                           "protocol": {"task": task, "amp_feature_schema": schema}}))
            audit = STYLE.audit_pairing(evaluations)
            self.assertTrue(audit["all_paired_comparisons_passed"])
            arrays["state"][0, 0] = .01
            np.savez_compressed(roots[1] / "initial_recovery.npz", **arrays)
            with self.assertRaisesRegex(ValueError, "pairing failed"):
                STYLE.audit_pairing(evaluations)


if __name__ == "__main__":
    unittest.main()
