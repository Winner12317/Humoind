"""Measured-depth geometry and raw units, independent of the simulator backend."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from src.wmac_perceptive.camera_calibration import (
    G1_DEPTH_CALIBRATION_PATH, calibrated_optical_rays,
    load_g1_depth_calibration, raw_depth_to_metres,
)


class G1CameraCalibrationTest(unittest.TestCase):
    def setUp(self):
        self.calibration = load_g1_depth_calibration()

    def test_active_depth_stream_is_used_without_factory_rgb_extrinsics(self):
        profile = json.loads(G1_DEPTH_CALIBRATION_PATH.read_text())
        c = self.calibration
        self.assertEqual((c.width, c.height, c.fps), (848, 480, 30))
        self.assertEqual((c.fx, c.fy), (431.9433288574219, 431.9433288574219))
        self.assertEqual((c.cx, c.cy), (432.1478576660156, 235.75416564941406))
        self.assertEqual(c.depth_scale_m_per_raw_unit, .0010000000474974513)
        self.assertEqual(c.source_report_sha256, "712c48441b847bf01eafb02cba55b4084520b8ac4142972c0044028cf3368ec6")
        self.assertEqual(profile["selected_profile_path"], "capture.active_profile")
        self.assertIsNone(profile["camera_mounting_extrinsics"])
        self.assertNotIn("depth_optical_to_color_optical", profile)
        self.assertNotIn("devices", profile)

    def test_full_profile_rays_match_independent_matrix_deprojection(self):
        c = self.calibration
        actual, cosine = calibrated_optical_rays(c, dtype=torch.float64)
        u, v = np.meshgrid(np.arange(c.width), np.arange(c.height))
        pixels = np.stack((u, v, np.ones_like(u)), -1).reshape(-1, 3)
        k = np.array([[c.fx, 0, c.cx], [0, c.fy, c.cy], [0, 0, 1]])
        optical = pixels @ np.linalg.inv(k).T
        camera = optical[:, [2, 0, 1]] * np.array([1., -1., -1.])
        expected = camera / np.linalg.norm(camera, axis=-1, keepdims=True)
        np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(cosine.numpy(), expected[:, 0], rtol=1e-13, atol=1e-13)

    def test_resize_preserves_source_pixel_centre_rays(self):
        c = self.calibration
        width, height = 64, 36
        actual, cosine = calibrated_optical_rays(c, width=width, height=height, dtype=torch.float64)
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        source_u = (u+.5)*c.width/width-.5
        source_v = (v+.5)*c.height/height-.5
        x = (source_u-c.cx)/c.fx
        y = (source_v-c.cy)/c.fy
        expected = np.stack((np.ones_like(x), -x, -y), -1).reshape(-1, 3)
        expected /= np.linalg.norm(expected, axis=-1, keepdims=True)
        np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-13, atol=1e-13)
        torch.testing.assert_close(cosine, actual[:, 0])
        scaled = c.resized(width, height)
        self.assertGreater(abs(scaled.cx-(width-1)/2), .5)
        # A naive principal-point scale would rotate this ray by half a pixel.
        naive_x = (0-c.cx*width/c.width)/(c.fx*width/c.width)
        self.assertGreater(abs(naive_x-float(-actual[0, 1]/actual[0, 0])), .01)

    def test_crop_then_resize_preserves_original_sensor_projection(self):
        c = self.calibration
        cropped = c.cropped(100, 50, 600, 300).resized(60, 30)
        actual, _ = calibrated_optical_rays(cropped, dtype=torch.float64)
        # Destination pixel (20, 10) originates at sensor centre (304.5,154.5).
        expected = np.array([1., -(304.5-c.cx)/c.fx, -(154.5-c.cy)/c.fy])
        expected /= np.linalg.norm(expected)
        np.testing.assert_allclose(actual[10*60+20].numpy(), expected, atol=1e-14)

    def test_optical_depth_is_not_euclidean_ray_range(self):
        rays, cosine = calibrated_optical_rays(self.calibration, width=64, height=36)
        optical_depth = torch.full_like(cosine, 2.)
        ranges = optical_depth/cosine
        xyz = rays*ranges[:, None]
        torch.testing.assert_close(xyz[:, 0], optical_depth)
        self.assertGreater(float(ranges[0]), 2.5)
        self.assertTrue(bool((rays[:, 0] > 0).all()))
        self.assertGreater(float(rays[0, 1]), 0)
        self.assertGreater(float(rays[0, 2]), 0)

    def test_raw_z16_scale_and_invalid_zero(self):
        raw = torch.tensor([[0, 1, 1000, 2500, 65535]], dtype=torch.uint16)
        metres, valid = raw_depth_to_metres(raw, self.calibration)
        self.assertEqual(valid.tolist(), [[False, True, True, True, True]])
        expected = raw.to(torch.float64)*self.calibration.depth_scale_m_per_raw_unit
        torch.testing.assert_close(metres.double(), expected, rtol=1e-7, atol=1e-7)
        self.assertEqual(metres.dtype, torch.float32)

    def test_invalid_depth_and_explicit_application_range_are_not_near_obstacles(self):
        raw = torch.tensor([float("nan"), float("inf"), -1., 0., 1000., 3000., 65536.])
        metres, valid = raw_depth_to_metres(raw, self.calibration, max_depth=2.5)
        self.assertEqual(valid.tolist(), [False, False, False, False, True, False, False])
        self.assertEqual(metres[~valid].tolist(), [0.]*6)
        self.assertTrue(bool(torch.isfinite(metres).all()))

    def test_rejects_nonzero_distortion_instead_of_silently_ignoring_it(self):
        with self.assertRaisesRegex(ValueError, "rectification"):
            replace(self.calibration, distortion_coefficients=(.01, 0., 0., 0., 0.))
        with self.assertRaisesRegex(ValueError, "distortion model"):
            replace(self.calibration, distortion_model="F-Theta")
        profile = json.loads(G1_DEPTH_CALIBRATION_PATH.read_text())
        profile["distortion_coefficients"][1] = .001
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"calibration.json"
            path.write_text(json.dumps(profile))
            with self.assertRaisesRegex(ValueError, "rectification"):
                load_g1_depth_calibration(path)

    def test_rejects_rgb_aligned_or_unknown_calibration_conventions(self):
        profile = json.loads(G1_DEPTH_CALIBRATION_PATH.read_text())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"calibration.json"
            for key, value in (("aligned_to_color", True), ("stream", "Color"),
                               ("schema_version", 99), ("depth_convention", "euclidean_range")):
                modified = {**profile, key: value}
                path.write_text(json.dumps(modified))
                with self.assertRaises(ValueError):
                    load_g1_depth_calibration(path)

    def test_invalid_resize_crop_and_range_parameters(self):
        for width, height in ((0, 36), (64, -1), (True, 36), (64., 36)):
            with self.assertRaises(ValueError):
                self.calibration.resized(width, height)
        with self.assertRaises(ValueError):
            self.calibration.cropped(800, 0, 100, 100)
        with self.assertRaises(ValueError):
            calibrated_optical_rays(self.calibration, width=64)
        for bound in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                raw_depth_to_metres(torch.ones(1), max_depth=bound)


if __name__ == "__main__":
    unittest.main()
