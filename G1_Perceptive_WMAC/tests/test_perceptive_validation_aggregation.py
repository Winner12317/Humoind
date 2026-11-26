"""Independent known-value checks for policy-view validation aggregation."""
from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from src.wmac_perceptive.runner import PerceptiveRunner


class PerceptiveValidationAggregationTest(unittest.TestCase):
    def fixture(self):
        runner = object.__new__(PerceptiveRunner)
        runner.cfg = SimpleNamespace(wm_window=3)
        # The second walk window has only its first real transition; the last
        # recovery window is pure padding. Their padded errors are sentinels.
        mask = torch.tensor([[1, 1, 1], [1, 0, 0], [1, 1, 1], [0, 0, 0]])
        recovery = torch.tensor([False, False, True, True])
        actor = torch.tensor([[.1, .5, .2], [.3, 999., 999.], [.4, .6, .8], [999., 999., 999.]], dtype=torch.float64)
        values = {
            "actor_rmse": actor,
            "full_rmse": actor+1,
            "outside_rmse": actor+2,
            "model_error": actor/10,
            "contact_error": actor/20,
            "actor_mask_error": actor/100,
            "actor_valid_pixels": torch.full((4, 3), 10.),
            "full_valid_pixels": torch.full((4, 3), 20.),
            "outside_valid_pixels": torch.full((4, 3), 10.),
            "finite": torch.ones(4, 3),
        }
        return runner, values, mask, recovery

    def test_padding_and_motion_domains_are_separate_and_horizons_use_worst_step(self):
        runner, values, mask, recovery = self.fixture()
        result = runner._validation_diagnostics(values, mask, recovery)
        self.assertAlmostEqual(result["validation_walk_step1_actor_depth_rmse"], .2)
        self.assertAlmostEqual(result["validation_walk_step2_actor_depth_rmse"], .5)
        self.assertAlmostEqual(result["validation_walk_step3_actor_depth_rmse"], .2)
        self.assertAlmostEqual(result["validation_walk_h1_actor_depth_rmse"], .2)
        self.assertAlmostEqual(result["validation_walk_h2_actor_depth_rmse"], .5)
        self.assertAlmostEqual(result["validation_walk_h3_actor_depth_rmse"], .5)
        self.assertAlmostEqual(result["validation_recovery_h3_actor_depth_rmse"], .8)
        self.assertEqual(result["validation_walk_step1_actor_depth_samples"], 2.)
        self.assertEqual(result["validation_walk_step2_actor_depth_samples"], 1.)
        self.assertEqual(result["validation_recovery_step1_actor_depth_samples"], 1.)
        self.assertEqual(result["validation_actor_depth_samples"], 7.)
        self.assertAlmostEqual(result["validation_actor_depth_rmse"], 2.9/7)
        self.assertAlmostEqual(result["validation_full_depth_rmse"], 2.9/7+1)
        self.assertAlmostEqual(result["validation_outside_depth_rmse"], 2.9/7+2)

    def test_empty_depth_view_has_no_certifying_samples_and_padding_nan_is_ignored(self):
        runner, values, mask, recovery = self.fixture()
        values["actor_valid_pixels"][0, 2] = 0.
        values["actor_rmse"][0, 2] = 0.
        values["finite"][1, 1:] = float("nan")
        result = runner._validation_diagnostics(values, mask, recovery)
        self.assertEqual(result["validation_walk_step3_actor_depth_samples"], 0.)
        self.assertEqual(result["validation_walk_step3_actor_depth_rmse"], 0.)
        self.assertEqual(result["validation_walk_h3_depth_samples"], 0.)
        self.assertEqual(result["validation_walk_h3_finite_fraction"], 1.)

    def prefix_fixture(self, valid_count, device="cpu"):
        runner = object.__new__(PerceptiveRunner)
        runner.cfg = SimpleNamespace(wm_window=8, motion_scope="forward_walk_turn_v1",
            imagination_horizon_start=1, imagination_horizon=8, horizon_growth_updates=100,
            max_model_loss_for_actor=20., max_validation_model_error=20.,
            max_validation_contact_error=2., validation_min_walk_samples=1,
            depth_rollout_error_ratio=1.5)
        runner.actor_updates_accepted = 300
        shape = (256, 8)
        mask = torch.zeros(shape, dtype=torch.bool, device=device)
        mask.flatten()[:valid_count] = True
        recovery = torch.zeros(shape[0], dtype=torch.bool, device=device)
        values = {name: torch.full(shape, .1, device=device)
                  for name in ("actor_rmse", "full_rmse", "outside_rmse",
                               "model_error", "contact_error", "actor_mask_error")}
        values.update({name: torch.ones(shape, device=device)
                       for name in ("actor_valid_pixels", "full_valid_pixels", "outside_valid_pixels", "finite")})
        # Nonfinite padding must never count against a real transition.
        values["finite"][~mask] = float("nan")
        row = {f"wm_h{horizon}_model_error": .1 for horizon in (1, 2, 4, 8)}
        row["wm_updates_applied_fraction"] = 1.
        return runner, values, mask, recovery, row

    def check_exact_finite_counts(self, device):
        for count in (255, 2039):
            runner, values, mask, recovery, row = self.prefix_fixture(count, device)
            with self.subTest(device=device, count=count, case="all_finite"):
                result = runner._validation_diagnostics(values, mask, recovery)
                self.assertEqual(result["validation_walk_h8_finite_fraction"], 1.)
                runner.validation = result
                self.assertTrue(runner._adaptive_actor_gate(dict(row)))
                self.assertEqual(runner._active_horizon, 8)
            # A failure in the final step must still prevent H8 while leaving
            # the genuinely finite shorter prefixes available to the Actor.
            for invalid in (0., float("nan"), float("inf")):
                with self.subTest(device=device, count=count, case="one_nonfinite", invalid=invalid):
                    values["finite"][0, 7] = invalid
                    result = runner._validation_diagnostics(values, mask, recovery)
                    self.assertEqual(result["validation_walk_h8_finite_fraction"], (count-1)/count)
                    self.assertLess(result["validation_walk_h8_finite_fraction"], 1.)
                    self.assertEqual(result["validation_walk_h4_finite_fraction"], 1.)
                    runner.validation = result
                    self.assertTrue(runner._adaptive_actor_gate(dict(row)))
                    self.assertEqual(runner._active_horizon, 4)
        runner, values, mask, recovery, row = self.prefix_fixture(0, device)
        with self.subTest(device=device, case="empty_prefix"):
            result = runner._validation_diagnostics(values, mask, recovery)
            self.assertEqual(result["validation_walk_h1_finite_fraction"], 0.)
            self.assertEqual(result["validation_walk_h8_finite_fraction"], 0.)
            runner.validation = result
            self.assertFalse(runner._adaptive_actor_gate(dict(row)))

    def test_integer_finite_counts_keep_exact_one_and_reject_real_failures_on_cpu(self):
        self.check_exact_finite_counts("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is optional for validation aggregation checks")
    def test_integer_finite_counts_keep_exact_one_and_reject_real_failures_on_cuda(self):
        self.check_exact_finite_counts("cuda")

    def test_known_cuda_float_mean_rounding_cannot_falsely_shorten_the_horizon(self):
        runner, values, mask, recovery, row = self.prefix_fixture(2039)
        original_mean = runner._masked_mean
        finite_pointer = values["finite"].data_ptr()

        def rounded_cuda_mean(value, valid, *, empty=float("inf")):
            if value.data_ptr() == finite_pointer:
                # Observed CUDA reduction of all-one flags with 2039 active
                # entries; reproduce it on CPU without disturbing training.
                return value.new_tensor(.9999999403953552)
            return original_mean(value, valid, empty=empty)

        with patch.object(runner, "_masked_mean", side_effect=rounded_cuda_mean):
            result = runner._validation_diagnostics(values, mask, recovery)
        self.assertEqual(result["validation_walk_h8_finite_fraction"], 1.)
        runner.validation = result
        self.assertTrue(runner._adaptive_actor_gate(dict(row)))
        self.assertEqual(runner._active_horizon, 8)


if __name__ == "__main__":
    unittest.main()
