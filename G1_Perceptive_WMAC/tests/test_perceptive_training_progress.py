"""Training-protocol regressions for visual validation and policy progress.

These checks exercise real imagination/optimizer code where useful and keep
the old fixed-horizon fixture explicitly on protocol v1. They do not measure
physical locomotion or recovery success.
"""
from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
from pathlib import Path
import tempfile
from unittest.mock import patch
import unittest
import warnings

import torch

from src.wmac_perceptive.runner import MODULES, OPTIMIZERS
from test_perceptive_runner import cloned_state, critic_noop, runner_fixture
from test_perceptive_runner_restore import restore_fixture, snapshot
from test_wmac_optimization import fix_update_noise


def progress_fixture():
    runner, batch, noise = runner_fixture(horizon=8)
    runner.cfg = replace(runner.cfg, perception_training_version=2, wm_window=8,
        imagination_horizon_start=1, horizon_growth_updates=100,
        depth_rollout_error_ratio=1.5, actor_stall_warning_iterations=20,
        model_warmup_iterations=0)
    runner.cfg.validate()
    runner._active_horizon = 1
    runner.actor_updates_accepted = 0
    runner.actor_updates_rejected = 0
    runner.iteration = 0
    runner._last_actor_update_iteration = 0
    runner._progress_start_iteration = 0
    runner._last_stall_warning_iteration = None
    runner.validation = dict(validation_model_error=.01,
        validation_contact_error=.01, validation_reward_gap=.01,
        validation_walk_error=.01, validation_recovery_error=.01,
        validation_depth_rmse=1., validation_depth_mask_error=.9,
        validation_walk_depth_rmse=1., validation_recovery_depth_rmse=1.,
        validation_walk_samples=16., validation_recovery_samples=16.)
    row = dict(wm_gate_error=.01, wm_updates_applied_fraction=1.)
    for horizon in range(1, 9):
        row[f"wm_h{horizon}_model_error"] = .01
        for phase in ("walk", "recovery"):
            prefix = f"validation_{phase}_h{horizon}_"
            runner.validation.update({prefix+"model_error": .01,
                prefix+"contact_error": .01, prefix+"actor_depth_rmse": .1,
                prefix+"actor_mask_error": .01, prefix+"samples": 16.,
                prefix+"depth_samples": 16., prefix+"finite_fraction": 1.})
    return runner, batch, noise, row


class PerceptiveProgressTest(unittest.TestCase):
    def test_large_full_image_error_does_not_disable_short_policy_learning(self):
        runner, _, _, row = progress_fixture()
        self.assertGreater(runner.validation["validation_depth_rmse"],
                           runner.cfg.max_validation_depth_error)
        self.assertTrue(runner._actor_gate(row))
        self.assertEqual(runner._effective_horizon(), 1)

    def test_short_horizon_validation_does_not_use_long_window_dynamics_gate(self):
        runner, _, _, row = progress_fixture()
        row["wm_gate_error"] = 1000.
        for name in ("validation_model_error", "validation_contact_error",
                     "validation_walk_error", "validation_recovery_error"):
            runner.validation[name] = 1000.
        row["wm_h8_model_error"] = 1000.
        self.assertTrue(runner._actor_gate(row))
        self.assertEqual(runner._effective_horizon(), 1)

    def test_short_horizon_still_rejects_nonfinite_or_unreliable_physics(self):
        cases = (("model_error", float("nan")), ("model_error", 1e9),
                 ("contact_error", 1e9), ("actor_depth_rmse", float("inf")),
                 ("actor_mask_error", float("nan")), ("finite_fraction", .99),
                 ("samples", 0.))
        for phase in ("walk", "recovery"):
            for name, value in cases:
                with self.subTest(phase=phase, name=name):
                    runner, _, _, row = progress_fixture()
                    runner.validation[f"validation_{phase}_h1_{name}"] = value
                    self.assertFalse(runner._actor_gate(row))
        runner, _, _, row = progress_fixture()
        row["wm_updates_applied_fraction"] = 0.
        self.assertFalse(runner._actor_gate(row))

    def test_horizon_progression_uses_retained_updates_and_can_fall_back(self):
        runner, _, _, row = progress_fixture()
        for accepted, expected in ((0, 1), (99, 1), (100, 2), (199, 2),
                                   (200, 4), (300, 8)):
            runner.actor_updates_accepted = accepted
            with self.subTest(accepted=accepted):
                self.assertTrue(runner._actor_gate(row))
                self.assertEqual(runner._effective_horizon(), expected)
        runner.validation["validation_recovery_h8_actor_depth_rmse"] = .4
        self.assertTrue(runner._actor_gate(row))
        self.assertEqual(runner._effective_horizon(), 4)
        runner.validation["validation_recovery_h4_actor_depth_rmse"] = .4
        runner.validation["validation_recovery_h2_actor_depth_rmse"] = .4
        self.assertTrue(runner._actor_gate(row))
        self.assertEqual(runner._effective_horizon(), 1)

    def test_promotion_needs_real_depth_and_transition_coverage_in_both_domains(self):
        for field in ("depth_samples", "samples"):
            runner, _, _, row = progress_fixture()
            runner.actor_updates_accepted = 100
            runner.validation[f"validation_recovery_h2_{field}"] = 0.
            with self.subTest(field=field):
                self.assertTrue(runner._actor_gate(row))
                self.assertEqual(runner._effective_horizon(), 1)

    def test_legitimate_no_hit_images_do_not_freeze_the_minimum_horizon(self):
        runner, _, _, row = progress_fixture()
        runner.actor_updates_accepted = 300
        for horizon in range(1, 9):
            for phase in ("walk", "recovery"):
                runner.validation[f"validation_{phase}_h{horizon}_depth_samples"] = 0.
                runner.validation[f"validation_{phase}_h{horizon}_actor_depth_rmse"] = 0.
        self.assertTrue(runner._actor_gate(row))
        self.assertEqual(runner._effective_horizon(), 1)

    def test_validation_preserves_late_error_and_counts_only_pre_reset_samples(self):
        runner, _, _, _ = progress_fixture()
        shape = (4, 8)
        values = {name: torch.full(shape, .01) for name in (
            "model_error", "contact_error", "full_rmse", "actor_rmse",
            "outside_rmse", "full_mask_error", "actor_mask_error")}
        values.update({name: torch.ones(shape) for name in (
            "actor_valid_pixels", "full_valid_pixels", "outside_valid_pixels", "finite")})
        mask = torch.ones(shape)
        # One start in each domain ends after its first transition. Its later
        # labels are unavailable and must not count toward a long rollout.
        mask[1, 1:] = 0.
        mask[3, 1:] = 0.
        values["actor_rmse"][0, 7] = .4
        values["model_error"][1, 1:] = float("nan")
        values["model_error"][3, 1:] = float("nan")
        measured = runner._validation_diagnostics(values, mask,
                                                   torch.tensor([False, False, True, True]))
        self.assertEqual(measured["validation_walk_h1_samples"], 2.)
        self.assertEqual(measured["validation_walk_h8_samples"], 1.)
        self.assertAlmostEqual(measured["validation_walk_h8_actor_depth_rmse"], .4, places=6)
        self.assertAlmostEqual(measured["validation_walk_h8_model_error"], .01, places=6)
        self.assertEqual(measured["validation_recovery_h8_finite_fraction"], 1.)

    def test_imagination_and_update_account_for_runtime_horizon(self):
        runner, batch, noise, _ = progress_fixture()
        runner._active_horizon = 2
        with patch.object(runner.world_model, "step_with_perception",
                          wraps=runner.world_model.step_with_perception) as step:
            result = runner.imagine(batch, noises=noise)
        self.assertEqual(step.call_count, 2)
        self.assertEqual(result["x"].shape[0], 2)
        self.assertTrue(bool(torch.isfinite(result["objective"]).all()))
        fix_update_noise(runner, noise)
        before = runner.imagination_steps
        with patch.object(runner, "_fit_critic", return_value=critic_noop()):
            updated = runner.update_actor_critic()
        self.assertEqual(updated["actor_update_rejected"], 0.)
        self.assertEqual(runner.imagination_steps-before, 2*len(batch["state"]))
        self.assertEqual(runner.actor_updates_accepted, 1)
        self.assertEqual(runner.actor_updates_rejected, 0)

    def test_rejected_updates_restore_actor_and_do_not_advance_curriculum(self):
        runner, _, noise, _ = progress_fixture()
        runner.cfg.max_action_shift = 0.
        before = cloned_state(runner.actor)
        fix_update_noise(runner, noise)
        with patch.object(runner, "_fit_critic", return_value=critic_noop()):
            result = runner.update_actor_critic()
        self.assertEqual(result["actor_update_rejected"], 1.)
        self.assertEqual(runner.actor_updates_accepted, 0)
        self.assertEqual(runner.actor_updates_rejected, 1)
        for name, tensor in runner.actor.state_dict().items():
            torch.testing.assert_close(tensor, before[name], rtol=0, atol=0)

    def test_critic_calibration_is_not_counted_as_policy_progress(self):
        runner, _, noise, _ = progress_fixture()
        fix_update_noise(runner, noise)
        with patch.object(runner, "_fit_critic", return_value=critic_noop()):
            runner.update_actor_critic(update_actor=False)
        self.assertEqual(runner.actor_updates_accepted, 0)
        self.assertEqual(runner.actor_updates_rejected, 0)

    def test_stall_alert_counts_absence_of_accepted_updates_and_is_rate_limited(self):
        runner, _, _, _ = progress_fixture()
        runner.iteration = 20
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            runner._check_actor_stall()
            runner.iteration = 21
            runner._check_actor_stall()
            self.assertEqual(len(caught), 1)
            self.assertTrue(issubclass(caught[0].category, RuntimeWarning))
            runner.iteration = 40
            runner._check_actor_stall()
            self.assertEqual(len(caught), 2)
            runner._last_actor_update_iteration = 40
            runner.iteration = 41
            runner._check_actor_stall()
            self.assertEqual(len(caught), 2)


def progress_restore_fixture(steps):
    runner = restore_fixture(steps)
    runner.cfg = replace(runner.cfg, perception_training_version=2, wm_window=3,
                         imagination_horizon_start=1)
    runner.cfg.validate()
    runner._ensure_training_progress()
    return runner


class PerceptiveProgressRestoreTest(unittest.TestCase):
    def assert_tree_equal(self, actual, expected):
        if isinstance(expected, torch.Tensor):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        elif isinstance(expected, dict):
            self.assertEqual(actual.keys(), expected.keys())
            for key in expected: self.assert_tree_equal(actual[key], expected[key])
        elif isinstance(expected, (list, tuple)):
            self.assertEqual(len(actual), len(expected))
            for left, right in zip(actual, expected): self.assert_tree_equal(left, right)
        elif hasattr(expected, "dtype") and hasattr(expected, "shape"):
            import numpy as np
            np.testing.assert_array_equal(actual, expected)
        else:
            self.assertEqual(actual, expected)

    def test_explicit_v1_to_v2_load_preserves_weights_adam_and_records_migration(self):
        source, restored = restore_fixture(1), restore_fixture(2)
        self.assertEqual(source.cfg.perception_training_version, 1)
        self.assertEqual((source.cfg.imagination_horizon, source.cfg.wm_window), (3, 2))
        source._ensure_training_progress()
        source.actor_updates_accepted, source.actor_updates_rejected = 1, 3
        source._active_horizon = 3
        source._progress_start_iteration = 0
        source._last_actor_update_iteration = source.iteration-1
        source._last_stall_warning_iteration = 10
        source.perception_training_history = [{"kind": "fixture_v1_history"}]
        restored.cfg = replace(restored.cfg, perception_training_version=2,
            migrate_perception_training=True, imagination_horizon_start=1,
            imagination_horizon=2)
        restored.cfg.validate()
        with tempfile.TemporaryDirectory() as directory, patch("torch.cuda.is_available", return_value=False):
            path = Path(directory)/"legacy_source.pt"
            source.save(path, export=False)
            original_bytes = path.read_bytes()
            digest = hashlib.sha256(original_bytes).hexdigest()
            payload = torch.load(path, map_location="cpu", weights_only=False)
            with patch("src.wmac_perceptive.runner.read_depth_frame", return_value=restored.current_depth):
                restored.load(path)
            self.assertEqual(path.read_bytes(), original_bytes)
        for name in (*MODULES, *OPTIMIZERS):
            with self.subTest(module=name):
                self.assert_tree_equal(getattr(restored, name).state_dict(), payload[name])
        self.assertGreater(payload["critic_calibration_updates"], 0)
        self.assertEqual(restored.critic_calibration_updates, 0)
        self.assertEqual(restored.actor_updates_accepted, 0)
        self.assertEqual(restored.actor_updates_rejected, 0)
        self.assertEqual(restored._effective_horizon(), 1)
        self.assertEqual(restored._progress_start_iteration, payload["iteration"])
        self.assertIsNone(restored._last_actor_update_iteration)
        for name in ("iteration", "sim_steps", "imagination_steps"):
            self.assertEqual(getattr(restored, name), payload[name])
        self.assertFalse(torch.isfinite(torch.tensor(restored.validation["validation_model_error"])))
        self.assertEqual(restored.perception_training_history[:-1], payload["perception_training_history"])
        migration = restored.perception_training_history[-1]
        self.assertEqual((migration["from_version"], migration["to_version"]), (1, 2))
        self.assertEqual(migration["changes"]["perception_training_version"], {"before": 1, "after": 2})
        self.assertEqual(migration["changes"]["imagination_horizon"], {"before": 3, "after": 2})
        self.assertEqual(migration["source_sha256"], digest)
        self.assertEqual(migration["source_iteration"], payload["iteration"])
        self.assertTrue(migration["network_and_optimizer_states_preserved"])
        self.assertTrue(migration["critic_calibration_reset"])
        self.assertTrue(migration["replay_and_validation_reset"])

    def test_v2_resume_preserves_policy_progress_and_invalidates_old_validation(self):
        source, restored = progress_restore_fixture(1), progress_restore_fixture(2)
        source.actor_updates_accepted, source.actor_updates_rejected = 12, 3
        source._active_horizon = 2
        source._progress_start_iteration = 0
        source._last_actor_update_iteration = source.iteration-1
        source._last_stall_warning_iteration = 10
        with tempfile.TemporaryDirectory() as directory, patch("torch.cuda.is_available", return_value=False):
            path = Path(directory)/"source.pt"
            source.save(path, export=False)
            payload = torch.load(path, map_location="cpu", weights_only=False)
            with patch("src.wmac_perceptive.runner.read_depth_frame", return_value=restored.current_depth):
                restored.load(path)
        for key, value in payload["perception_training_progress"].items():
            self.assertEqual(getattr(restored, key), value)
        self.assertEqual(restored.critic_calibration_updates, payload["critic_calibration_updates"])
        self.assertFalse(torch.isfinite(torch.tensor(restored.validation["validation_model_error"])))

    def test_malformed_progress_and_history_are_rejected_before_loading_weights(self):
        source, restored = progress_restore_fixture(1), progress_restore_fixture(2)
        with tempfile.TemporaryDirectory() as directory, patch("torch.cuda.is_available", return_value=False):
            source_path = Path(directory)/"source.pt"
            source.save(source_path, export=False)
            payload = torch.load(source_path, map_location="cpu", weights_only=False)
            for problem in ("history_none", "progress_list", "future_progress_start", "missing_progress_key"):
                bad = copy.deepcopy(payload)
                if problem == "history_none": bad["perception_training_history"] = None
                elif problem == "progress_list": bad["perception_training_progress"] = []
                elif problem == "future_progress_start":
                    bad["perception_training_progress"]["_progress_start_iteration"] = bad["iteration"]+1
                else: bad["perception_training_progress"].pop("_last_actor_update_iteration")
                path = Path(directory)/f"{problem}.pt"
                torch.save(bad, path)
                before = snapshot(restored)
                with self.subTest(problem=problem), self.assertRaises(ValueError):
                    restored.load(path)
                self.assert_tree_equal(snapshot(restored), before)


if __name__ == "__main__":
    unittest.main()
