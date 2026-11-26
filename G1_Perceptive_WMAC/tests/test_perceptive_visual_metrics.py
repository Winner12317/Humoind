"""Policy-view training and diagnostics must describe what the Actor sees."""
from __future__ import annotations

import math
import unittest

import torch

from src.wmac_perceptive.models import DepthEncoder, TerrainPerceptiveWorldModel
from src.wmac_perceptive.perception import PerceptionSpec
from src.wmac_perceptive.visual_metrics import depth_metrics, policy_view, policy_view_loss_terms


class PolicyViewMetricsTest(unittest.TestCase):
    def setUp(self):
        self.spec = PerceptionSpec()

    def frame(self, *, depth=.4, valid=1., shape=(2,)):
        frame = torch.full((*shape, *self.spec.frame_shape), depth, dtype=torch.float64)
        frame[..., 1, :, :] = valid
        return frame

    def test_actor_encoder_and_standalone_view_agree_for_history_and_batch_prefixes(self):
        frames = self.frame(shape=(2, 3, 8))
        frames[..., 0, 24, 30] = 1.
        frames[..., 1, 25, 31] = 0.
        encoder = DepthEncoder(16, 36, 64, 128, self.spec).double()
        actual = encoder.preprocess(frames.flatten(-4, -3))
        expected = policy_view(frames, self.spec).flatten(-4, -3)
        self.assertEqual(actual.shape, (2, 3, 16, 18, 32))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(actual[..., 1::2, :, :], frames[..., 1, 18:, 16:-16], rtol=0, atol=0)

    def test_blur_preserves_constant_depth_and_spreads_an_interior_impulse(self):
        frame = self.frame(depth=0., shape=())
        frame[0, 24, 30] = 1.
        view = policy_view(frame, self.spec)
        gaussian_center = 1/(1+2*math.exp(-.5))
        self.assertAlmostEqual(float(view[0, 6, 14]), gaussian_center**2, places=12)
        self.assertEqual(int((view[0] > 0).sum()), 9)
        self.assertAlmostEqual(float(view[0].sum()), 1., places=12)
        constant = self.frame(shape=())
        torch.testing.assert_close(policy_view(constant, self.spec)[0],
                                   torch.full((18, 32), .4, dtype=torch.float64))

    def test_outside_errors_are_diagnostic_and_cannot_change_policy_view_loss(self):
        target = self.frame()
        predicted = target.clone()
        predicted[..., 0, :18, :] = .9
        predicted[..., 0, 18:, :16] = .9
        predicted[..., 0, 18:, -16:] = .9
        baseline = policy_view_loss_terms(target, target, self.spec)
        loss = policy_view_loss_terms(predicted, target, self.spec)
        for name in loss:
            torch.testing.assert_close(loss[name], baseline[name], rtol=0, atol=0)
        metrics = depth_metrics(predicted, target, self.spec)
        torch.testing.assert_close(metrics["actor_rmse"], torch.zeros(2, dtype=torch.float64))
        torch.testing.assert_close(metrics["outside_rmse"], torch.full((2,), 1.25, dtype=torch.float64))
        self.assertTrue(bool((metrics["full_rmse"] > 0).all()))
        torch.testing.assert_close(metrics["actor_valid_pixels"], torch.full((2,), 18*32, dtype=torch.float64))
        torch.testing.assert_close(metrics["outside_valid_pixels"], torch.full((2,), 36*64-18*32, dtype=torch.float64))

    def test_in_view_error_is_measured_in_meters_and_has_no_outside_gradient(self):
        target = self.frame()
        predicted = self.frame(depth=.6, valid=.8).requires_grad_()
        terms = policy_view_loss_terms(predicted, target, self.spec)
        torch.testing.assert_close(terms["depth"], torch.full((2,), .02, dtype=torch.float64))
        torch.testing.assert_close(terms["valid"], torch.full((2,), -math.log(.8), dtype=torch.float64))
        metrics = depth_metrics(predicted, target, self.spec)
        torch.testing.assert_close(metrics["actor_rmse"], torch.full((2,), .5, dtype=torch.float64))
        torch.testing.assert_close(metrics["actor_mask_error"], torch.full((2,), .04, dtype=torch.float64))
        (terms["depth"]+terms["valid"]).sum().backward()
        self.assertTrue(bool(torch.isfinite(predicted.grad).all()))
        self.assertGreater(float(predicted.grad[..., 0, 18:, 16:-16].abs().sum()), 0.)
        self.assertGreater(float(predicted.grad[..., 1, 18:, 16:-16].abs().sum()), 0.)
        self.assertEqual(float(predicted.grad[..., :18, :].abs().sum()), 0.)
        self.assertEqual(float(predicted.grad[..., 18:, :16].abs().sum()), 0.)
        self.assertEqual(float(predicted.grad[..., 18:, -16:].abs().sum()), 0.)

    def test_target_validity_controls_depth_loss_even_if_prediction_claims_no_hits(self):
        target = self.frame()
        predicted = self.frame(depth=.6, valid=0.)
        terms = policy_view_loss_terms(predicted, target, self.spec)
        torch.testing.assert_close(terms["depth"], torch.full((2,), .02, dtype=torch.float64))
        self.assertTrue(bool((terms["valid"] > 10).all()))
        torch.testing.assert_close(depth_metrics(predicted, target, self.spec)["actor_rmse"],
                                   torch.full((2,), .5, dtype=torch.float64))

    def test_empty_hit_regions_report_zero_depth_with_explicit_zero_counts(self):
        target = self.frame(valid=0., shape=(2, 3))
        predicted = self.frame(depth=.9, valid=.8, shape=(2, 3))
        metrics = depth_metrics(predicted, target, self.spec)
        for region in ("full", "actor", "outside"):
            torch.testing.assert_close(metrics[f"{region}_rmse"], torch.zeros((2, 3), dtype=torch.float64))
            torch.testing.assert_close(metrics[f"{region}_valid_pixels"], torch.zeros((2, 3), dtype=torch.float64))
        terms = policy_view_loss_terms(predicted, target, self.spec)
        torch.testing.assert_close(terms["depth"], torch.zeros((2, 3), dtype=torch.float64))
        torch.testing.assert_close(terms["valid"], torch.full((2, 3), -math.log(.2), dtype=torch.float64))

    def test_uncropped_spec_has_explicitly_empty_outside_region(self):
        spec = PerceptionSpec(height=4, width=6, crop_top=0, crop_left=0, crop_right=0, blur_kernel=1)
        frame = torch.zeros(2, *spec.frame_shape)
        frame[:, 1] = 1.
        metrics = depth_metrics(frame, frame, spec)
        for key in ("outside_pixel_count", "outside_valid_pixels", "outside_rmse", "outside_mask_error"):
            torch.testing.assert_close(metrics[key], torch.zeros(2), rtol=0, atol=0)
        torch.testing.assert_close(policy_view(frame, spec), frame, rtol=0, atol=0)

    def test_nonfinite_measurements_are_not_silently_sanitized(self):
        target = self.frame()
        predicted = target.clone()
        predicted[..., 0, 24, 30] = float("nan")
        metrics = depth_metrics(predicted, target, self.spec)
        self.assertTrue(bool(torch.isnan(metrics["actor_rmse"]).all()))
        self.assertTrue(bool(torch.isnan(metrics["full_rmse"]).all()))

    def test_mismatched_frame_shapes_are_rejected(self):
        target = self.frame()
        with self.assertRaises(ValueError):
            depth_metrics(target[..., :-1], target, self.spec)
        with self.assertRaises(ValueError):
            policy_view_loss_terms(target[..., :1, :, :], target[..., :1, :, :], self.spec)

    def test_replay_precision_is_converted_to_prediction_precision_before_bce(self):
        predicted = self.frame(depth=.6, valid=.8).requires_grad_()
        target = self.frame().float()
        terms = policy_view_loss_terms(predicted, target, self.spec)
        legacy = TerrainPerceptiveWorldModel.pixel_loss_terms(predicted, target)
        for values in (terms, legacy):
            self.assertEqual(values["depth"].dtype, torch.float64)
            self.assertEqual(values["valid"].dtype, torch.float64)
            self.assertTrue(bool(torch.isfinite(values["depth"]+values["valid"]).all()))
        (terms["depth"]+terms["valid"]+legacy["depth"]+legacy["valid"]).sum().backward()
        self.assertTrue(bool(torch.isfinite(predicted.grad).all()))
        self.assertEqual(depth_metrics(predicted, target, self.spec)["actor_rmse"].dtype, torch.float64)


if __name__ == "__main__":
    unittest.main()
