"""Capture cadence, terminal/reset separation and G1 deployment parity."""
from __future__ import annotations

from dataclasses import replace
import unittest

import torch
from torch.nn import functional as F

from src.wmac_perceptive.camera_calibration import load_g1_depth_calibration
from src.wmac_perceptive.rpl_camera_stream import RPLCameraStream, preprocess_real_g1_z16
from src.wmac_perceptive.rpl_depth import RPLDepthConfig, normalize_rpl_depth, preprocess_rpl_depth


class RPLCameraStreamTests(unittest.TestCase):
    def setUp(self):
        self.cfg = RPLDepthConfig(native_width=24, native_height=15,
                                  output_width=8, output_height=5)
        self.stones = torch.tensor([False, True])

    def depth(self, value):
        return torch.full((2, 1, 15, 24), float(value))

    def stream(self, **kwargs):
        return RPLCameraStream(2, cfg=self.cfg, **kwargs)

    def test_capture_delay_is_two_camera_ticks_not_control_ticks(self):
        stream = self.stream(enabled_randomization=False)
        captured = {}
        for step in range(16):
            packet = stream.step(self.depth(.2+.1*step), control_step=step,
                                 terrain_is_stepping_stones=self.stones)
            if step in (0, 5, 10, 15):
                captured[step] = packet.clone()
            if step < 10:
                self.assertLess(packet[:, 0].abs().max().item(), 1e-6)
        self.assertEqual(stream.capture_count, 4)
        self.assertTrue(torch.allclose(captured[10][:, :1],
            normalize_rpl_depth(torch.full((2, 1, 5, 8), .2), self.cfg), atol=1e-6))
        self.assertTrue(torch.allclose(captured[15][:, :1],
            normalize_rpl_depth(torch.full((2, 1, 5, 8), .7), self.cfg), atol=1e-6))

    def test_repeated_reads_do_not_advance_rng_or_history(self):
        generator = torch.Generator().manual_seed(73)
        stream = self.stream(generator=generator)
        packet = stream.step(self.depth(1.), control_step=0,
                             terrain_is_stepping_stones=self.stones)
        rng = generator.get_state().clone()
        history = stream.delay.buffer.clone()
        again = stream.step(self.depth(1.9), control_step=0,
                            terrain_is_stepping_stones=self.stones)
        self.assertEqual(packet.data_ptr(), again.data_ptr())
        self.assertTrue(torch.equal(history, stream.delay.buffer))
        self.assertTrue(torch.equal(rng, generator.get_state()))
        # Four intermediate policy reads likewise leave capture randomness alone.
        for step in range(1, 5):
            stream.step(self.depth(1.9), control_step=step,
                        terrain_is_stepping_stones=self.stones)
        self.assertTrue(torch.equal(rng, generator.get_state()))
        self.assertEqual(stream.capture_count, 1)

    def test_terminal_copy_survives_selective_same_step_reset(self):
        stream = self.stream(enabled_randomization=False)
        for step in (0, 5, 10):
            stream.step(self.depth(.4), control_step=step,
                        terrain_is_stepping_stones=self.stones)
        terminal = stream.packet.clone()
        other_history = stream.delay.buffer[1].clone()
        stream.reset(torch.tensor([True, False]))
        again = stream.step(self.depth(1.8), control_step=10,
                            terrain_is_stepping_stones=self.stones)
        self.assertEqual(stream.capture_count, 3)
        self.assertTrue(torch.equal(terminal[1], again[1]))
        self.assertGreater(terminal[0, 0].abs().mean().item(), .1)
        self.assertEqual(again[0].abs().max().item(), 0.)
        self.assertTrue(torch.equal(other_history, stream.delay.buffer[1]))
        self.assertTrue(torch.allclose(stream.delay.buffer[0],
                                      torch.full_like(stream.delay.buffer[0], 1.05)))
        # Reset fill remains for two captures; old episode .4m never leaks back.
        for step in (15, 20):
            packet = stream.step(self.depth(1.8), control_step=step,
                                 terrain_is_stepping_stones=self.stones)
            self.assertLess(packet[0, 0].abs().max().item(), 1e-6)
        packet = stream.step(self.depth(1.8), control_step=25,
                             terrain_is_stepping_stones=self.stones)
        self.assertGreater(packet[0, 0].mean().item(), .3)

    def test_reset_argument_is_immediate_without_new_capture(self):
        stream = self.stream(enabled_randomization=False)
        for step in (0, 5, 10):
            stream.step(self.depth(.4), control_step=step,
                        terrain_is_stepping_stones=self.stones)
        packet = stream.step(self.depth(1.8), control_step=11,
            terrain_is_stepping_stones=self.stones, reset_mask=torch.tensor([True, False]))
        self.assertEqual(packet[0].abs().max().item(), 0.)
        self.assertGreater(packet[1, 0].abs().mean().item(), .1)
        self.assertEqual(stream.capture_count, 3)

    def test_nonfinite_zero_and_far_inputs_are_safe_and_diagnostic(self):
        stream = self.stream(enabled_randomization=False)
        depth = self.depth(float('inf'))
        depth[1] = 0.
        for step in (0, 5, 10):
            packet = stream.step(depth, control_step=step,
                                 terrain_is_stepping_stones=self.stones)
        self.assertTrue(torch.isfinite(packet).all())
        self.assertTrue(torch.allclose(packet[:, 0], torch.full_like(packet[:, 0], .5)))
        self.assertEqual(packet[:, 1].count_nonzero().item(), 0)

    def test_default_packet_dimensions(self):
        stream = RPLCameraStream(1, enabled_randomization=False)
        self.assertEqual(tuple(stream.packet.shape), (1, 2, 27, 48))
        self.assertEqual(stream.capture_interval, 5)
        self.assertEqual(stream.cfg.delay, 2)

    def test_bad_clock_or_capture_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            self.stream(capture_period=.03)
        stream = self.stream(enabled_randomization=False)
        with self.assertRaises(ValueError):
            stream.step(torch.ones(2, 1, 2, 2), control_step=0,
                        terrain_is_stepping_stones=self.stones)
        # A failed input must not mark its control step as already captured.
        stream.step(self.depth(1.), control_step=0, terrain_is_stepping_stones=self.stones)
        self.assertEqual(stream.capture_count, 1)
        with self.assertRaises(ValueError):
            stream.step(self.depth(1.), control_step=-1, terrain_is_stepping_stones=self.stones)
        stream.step(self.depth(1.), control_step=2, terrain_is_stepping_stones=self.stones)
        with self.assertRaises(ValueError):
            stream.step(self.depth(1.), control_step=1, terrain_is_stepping_stones=self.stones)
        with self.assertRaises(ValueError):
            stream.reset(torch.tensor([True]))


class RealG1DepthPreprocessingTests(unittest.TestCase):
    def test_native_real_adapter_matches_explicit_deterministic_pipeline(self):
        calibration = load_g1_depth_calibration()
        raw = torch.full((1, 1, calibration.height, calibration.width), 1000, dtype=torch.int32)
        raw[..., :70, :] = 0
        raw[..., 70:140, :] = 65535
        raw[..., 140:210, :] = 200
        actual = preprocess_real_g1_z16(raw, calibration)
        metres = raw.float()*calibration.depth_scale_m_per_raw_unit
        metres = torch.where(raw > 0, metres.clamp(max=2.5), torch.full_like(metres, 2.5))
        resized = F.interpolate(metres, size=(135, 240), mode='bicubic',
                                align_corners=False, antialias=True)
        expected = preprocess_rpl_depth(resized, training=False)
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(tuple(actual.shape), (1, 1, 27, 48))
        self.assertTrue(torch.isfinite(actual).all())

    def test_adapter_accepts_single_and_batch_inputs_without_randomness(self):
        calibration = replace(load_g1_depth_calibration(), width=24, height=15)
        cfg = RPLDepthConfig(native_width=24, native_height=15, output_width=8, output_height=5)
        raw = torch.full((15, 24), 1000, dtype=torch.int32)
        before = torch.random.get_rng_state().clone()
        single = preprocess_real_g1_z16(raw, calibration, cfg=cfg)
        batched = preprocess_real_g1_z16(raw[None], calibration, cfg=cfg)
        self.assertTrue(torch.equal(single, batched))
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        metric = 1000*calibration.depth_scale_m_per_raw_unit
        self.assertTrue(torch.allclose(single, torch.full_like(single, (metric-.1)/1.9-.5), atol=1e-6))

    def test_wrong_real_dimensions_are_rejected(self):
        with self.assertRaises(ValueError):
            preprocess_real_g1_z16(torch.ones(10, 10))


if __name__ == '__main__':
    unittest.main()
