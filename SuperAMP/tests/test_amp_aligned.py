"""Cross-arm AMP inputs, versioning, and opt-in PPO preprocessing fixes."""

from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch

from rsl_rl.algorithms.amp_ppo import AMPPPO
from rsl_rl.modules import ActorCritic
from rsl_rl.modules.discriminator import Discriminator
from rsl_rl.runners.amp_on_policy_runner import AmpOnPolicyRunner, _validate_amp_checkpoint_schema
from rsl_rl.utils import Normalizer
from rsl_rl.utils.motion_loader import AMPLoader
from src.amp_features import build_amp_features
from src.tasks.amp_loco.config.g1.env_cfgs import g1_amp_flat_aligned_env_cfg, g1_amp_flat_env_cfg
from src.tasks.amp_loco.config.g1.rl_cfg import g1_amp_aligned_ppo_runner_cfg, g1_amp_ppo_runner_cfg


BODY_NAMES = ("pelvis", "left_ankle_roll_link", "right_ankle_roll_link") + tuple(
    f"body_{i}" for i in range(10)
) + ("torso_link",)


def write_clip(path: Path, height: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    p = np.zeros((4, len(BODY_NAMES), 3), dtype=np.float32)
    p[..., 2] = height + np.arange(4, dtype=np.float32)[:, None] * 0.1
    q = np.zeros((4, len(BODY_NAMES), 4), dtype=np.float32)
    q[..., 0] = 1
    q[:, 1] = (0.5, 0.5, 0.5, 0.5)
    v = np.zeros_like(p)
    w = np.zeros_like(p)
    np.savez(path, fps=np.array(50), joint_pos=np.zeros((4, 29)), joint_vel=np.zeros((4, 29)),
             body_pos_w=p, body_quat_w=q, body_lin_vel_w=v, body_ang_vel_w=w)
    return tuple(torch.from_numpy(x) for x in (p, q, v, w))


class AlignedInputsTest(unittest.TestCase):
    def test_grounded_loader_has_shared_features_balanced_pairs_without_boundary_duplicates(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            recovery = write_clip(root / "Recovery" / "rise.npz", 10.)
            walk = write_clip(root / "WalkandRun" / "walk.npz", 1.)
            loader = AMPLoader(str(root), BODY_NAMES[:13], "torso_link", BODY_NAMES, device="cpu",
                               feature_schema="grounded_v2", expert_recovery_fraction=.5)
            self.assertEqual(loader.observation_dim, 203)
            for group, arrays in zip(loader._pair_groups, (walk, recovery)):
                expected = build_amp_features(*arrays, body_ids=list(range(13)), anchor_id=13,
                    pelvis_id=0, foot_ids=(1, 2), schema="grounded_v2")
                torch.testing.assert_close(group[:, 0], expected[:-1])
                torch.testing.assert_close(group[:, 1], expected[1:])
            for first, second in loader.feed_forward_generator(3, 20):
                self.assertEqual(first.shape, (20, 203))
                self.assertEqual(int((first[:, 198] >= 10).sum()), 10)
                torch.testing.assert_close(second[:, 198] - first[:, 198], torch.full((20,), .1))

    def test_default_loader_retains_legacy_width_and_rotation_convention(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()), patch(
            "rsl_rl.utils.motion_loader.tqdm", side_effect=lambda x, **_: x
        ):
            path = Path(directory) / "walk.npz"
            arrays = write_clip(path, 1.)
            loader = AMPLoader(str(path), BODY_NAMES[:13], "torso_link", BODY_NAMES, device="cpu")
            self.assertEqual(loader.observation_dim, 195)
            expected = build_amp_features(*arrays, body_ids=list(range(13)), anchor_id=13)
            # This relative cyclic-axis rotation distinguishes rows from columns.
            actual = torch.cat([loader._body_pos_b.flatten(1), loader._body_ori_b.flatten(1),
                                loader._body_lin_vel_b.flatten(1), loader._body_ang_vel_b.flatten(1)], -1)
            torch.testing.assert_close(actual, expected)
            first, second = next(loader.feed_forward_generator(1, 3))
            self.assertEqual(first.shape, second.shape)

    def test_new_config_is_explicit_and_does_not_mutate_legacy_task(self):
        legacy = g1_amp_ppo_runner_cfg()
        aligned = g1_amp_aligned_ppo_runner_cfg()
        self.assertEqual(legacy.amp_feature_schema, "legacy")
        self.assertEqual(legacy.actor.distribution_cfg["init_std"], 1.)
        self.assertEqual(aligned.amp_feature_schema, "grounded_v2")
        self.assertEqual(aligned.actor.distribution_cfg["init_std"], .25)
        self.assertEqual(aligned.amp_expert_recovery_fraction, .5)
        self.assertTrue(aligned.amp_normalizer_use_raw)
        self.assertTrue(aligned.amp_gp_normalized)
        self.assertEqual(aligned.algorithm, legacy.algorithm)
        old_env, new_env = g1_amp_flat_env_cfg(), g1_amp_flat_aligned_env_cfg()
        self.assertEqual(old_env.observations["critic"].history_length, 4)
        self.assertEqual(new_env.observations["critic"].history_length, 1)
        self.assertEqual(old_env.observations["actor"], new_env.observations["actor"])
        self.assertIn("body_pos_b", old_env.observations["amp"].terms)
        self.assertEqual(new_env.observations["amp"].terms["grounded_features"].params["schema"], "grounded_v2")

    def test_schema_guard_runs_before_loading_any_network(self):
        _validate_amp_checkpoint_schema({}, "legacy", 195)
        _validate_amp_checkpoint_schema({"amp_feature_schema": "grounded_v2", "amp_feature_dim": 203},
                                        "grounded_v2", 203)
        owner = SimpleNamespace(amp_feature_schema="grounded_v2", amp_feature_dim=203,
                                alg=SimpleNamespace(policy=Mock()))
        with patch("torch.load", return_value={"model_state_dict": {}}):
            with self.assertRaisesRegex(ValueError, "incompatible"):
                AmpOnPolicyRunner.load(owner, "old.pt")
        owner.alg.policy.load_state_dict.assert_not_called()


class RecordingNormalizer(Normalizer):
    def __init__(self):
        super().__init__(2)
        self.mean[:] = 10
        self.var[:] = 4
        self.updates = []

    def update(self, array):
        self.updates.append(array.copy())


class PpoPreprocessingTest(unittest.TestCase):
    def run_update(self, aligned):
        normalizer = RecordingNormalizer()
        data = SimpleNamespace(feed_forward_generator=lambda n, size: iter([
            (torch.full((size, 2), 30.), torch.full((size, 2), 31.)) for _ in range(n)
        ]))
        with contextlib.redirect_stdout(io.StringIO()):
            policy = ActorCritic(3, 4, 2, actor_hidden_dims=[8], critic_hidden_dims=[8])
            disc = Discriminator(4, .1, [8], "cpu", .75)
            alg = AMPPPO(policy, disc, data, normalizer, learning_rate=0.,
                         num_learning_epochs=1, num_mini_batches=1,
                         amp_normalizer_use_raw=aligned, amp_gp_normalized=aligned)
        alg.init_storage("rl", 2, 2, [3], [4], [2])
        for step in range(2):
            alg.act(torch.randn(2, 3), torch.randn(2, 4), torch.full((2, 2), 20.))
            alg.process_env_step(torch.tensor([1., 2.]) + step, torch.zeros(2, dtype=torch.bool),
                                 {}, torch.full((2, 2), 21.))
        alg.compute_returns(torch.randn(2, 4))
        with patch.object(disc, "compute_grad_pen", wraps=disc.compute_grad_pen) as penalty:
            result = alg.update()
        self.assertTrue(all(np.isfinite(float(x)) for x in result.values() if x is not None))
        return normalizer, penalty.call_args.args

    def test_aligned_uses_raw_moments_and_normalized_penalty(self):
        normalizer, penalty = self.run_update(True)
        np.testing.assert_allclose(normalizer.updates[0], 20.)
        np.testing.assert_allclose(normalizer.updates[1], 21.)
        np.testing.assert_allclose(normalizer.updates[2], 30.)
        np.testing.assert_allclose(normalizer.updates[3], 31.)
        torch.testing.assert_close(penalty[0], normalizer.normalize_torch(torch.full((4, 2), 30.), "cpu"))

    def test_legacy_preprocessing_stays_unchanged(self):
        normalizer, penalty = self.run_update(False)
        np.testing.assert_allclose(normalizer.updates[0], 10 / np.sqrt(4.0001), rtol=1e-6)
        np.testing.assert_allclose(normalizer.updates[1], 20 / np.sqrt(4.0001), rtol=1e-6)
        torch.testing.assert_close(penalty[0], torch.full((4, 2), 30.))


if __name__ == "__main__":
    unittest.main()
