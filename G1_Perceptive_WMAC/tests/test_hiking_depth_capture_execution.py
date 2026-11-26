"""CPU stochastic equivalence against the published d56489a6 pixel formulas.

The frozen oracle deliberately retains the original full-batch advanced
indexing and per-call strip construction. Tests compare pixels, RNG state,
capture timing and partial-reset terminal snapshots, rather than distribution
averages alone. No simulator or GPU is started.
"""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch

import src.wmac_perceptive.hiking_depth_randomization as randomization
from src.wmac_perceptive.hiking_depth_randomization import (
    HikingDepthCapture, HikingDepthRandomizationCfg, augment_hiking_depth_packet,
)
from src.tasks.amp_perceptive.sensors import _hiking_stepping_stones_mask


def frozen_augmentation(packet, stones, cfg, enabled, generator):
    """Published numeric/RNG oracle; no production augmentation helpers."""
    if not enabled:
        return packet
    n = len(packet)
    ys, xs = cfg.roi_slices
    result = packet.clone()
    depth = packet[:, 0, ys, xs] * cfg.max_depth
    valid = (packet[:, 1, ys, xs] > .5) & (depth >= 0.) & (depth <= cfg.max_depth)
    if cfg.noise_enabled:
        noisy = torch.normal(depth, depth.abs() * cfg.std_multiplier, generator=generator)
        depth = torch.where(valid, noisy, depth)
        dropped = torch.bernoulli(torch.full_like(depth, cfg.dropout_probability), generator=generator).bool() & valid
        depth = torch.where(dropped, torch.full_like(depth, cfg.dropout_depth_m), depth)
        result[:, 1, ys, xs] = torch.where(dropped, torch.zeros_like(depth), result[:, 1, ys, xs])
    if cfg.occlusion_enabled:
        small_probability = cfg.p_small + torch.where(stones, cfg.terrain_bias, -cfg.terrain_bias)
        choice = torch.rand(n, device=packet.device, generator=generator)
        small = (choice >= cfg.p_none) & (choice < cfg.p_none + small_probability)
        large = choice >= cfg.p_none + small_probability
        apply = torch.rand(n, device=packet.device, generator=generator) < cfg.occlusion_probability
        ll, lr, sl, sr = cfg.occlusion_widths
        cols = torch.arange(cfg.roi_width, device=packet.device)
        large_strip = (cols < ll) | (cols >= cfg.roi_width - lr)
        small_strip = (cols < sl) | (cols >= cfg.roi_width - sr)
        mask = apply[:, None, None] & ((large[:, None, None] & large_strip[None, None])
            | (small[:, None, None] & small_strip[None, None]))
        synthetic = cfg.max_depth * torch.rand(n, cfg.roi_height, cfg.roi_width,
            device=packet.device, dtype=packet.dtype, generator=generator)
        depth = torch.where(mask, synthetic, depth)
        result[:, 1, ys, xs] = torch.where(mask, torch.zeros_like(depth), result[:, 1, ys, xs])
    result[:, 0, ys, xs] = depth / cfg.max_depth
    return result


def packet(count=5):
    values = torch.linspace(.1, 1., count * 36 * 64).reshape(count, 36, 64)
    result = torch.stack((values, torch.ones_like(values)), 1)
    result[0, 0, 20, 20] = 1e6 / 2.5
    result[0, 1, 20, 20] = 0.
    result[-1, 0, 22, 22] = 4. / 2.5
    result[-1, 1, 22, 22] = 0.
    return result


class FrozenCapture:
    def __init__(self, count, cfg, enabled, seed):
        self.packet = torch.empty(count, 2, cfg.height, cfg.width)
        self.pending = torch.ones(count, dtype=torch.bool)
        self.last = None
        self.counts = torch.zeros(count, dtype=torch.long)
        self.calls = 0
        self.cfg, self.enabled = cfg, enabled
        self.generator = torch.Generator().manual_seed(seed)

    def step(self, value, stones, tick):
        if self.last == tick and not self.pending.any():
            return self.packet
        ids = self.pending.nonzero().flatten() if self.last == tick else torch.arange(len(self.packet))
        self.packet[ids] = frozen_augmentation(value[ids], stones[ids], self.cfg, self.enabled, self.generator)
        self.pending[ids] = False
        self.counts[ids] += 1
        self.calls += 1
        self.last = tick
        return self.packet


class HikingCaptureExecutionTest(unittest.TestCase):
    def test_pixels_rng_and_terminal_partial_reset_match_published_capture(self):
        base = HikingDepthRandomizationCfg()
        for cfg, enabled in ((base, True), (base, False),
                (replace(base, noise_enabled=False), True),
                (replace(base, occlusion_enabled=False), True)):
            for strided in (False, True):
                with self.subTest(cfg=cfg, enabled=enabled, strided=strided):
                    actual = HikingDepthCapture(5, cfg=cfg, enabled_randomization=enabled,
                        generator=torch.Generator().manual_seed(1423))
                    reference = FrozenCapture(5, cfg, enabled, 1423)
                    stones = torch.tensor([False, True, False, True, False])
                    raw = packet()
                    if strided:
                        storage = torch.empty(10, *raw.shape[1:])
                        storage[::2] = raw
                        raw = storage[::2]
                        self.assertFalse(raw.is_contiguous())
                    terminal_snapshots = []
                    for tick in (0, 1, 2):
                        a = actual.step(raw, control_step=tick, terrain_is_stepping_stones=stones)
                        b = reference.step(raw, stones, tick)
                        torch.testing.assert_close(a, b, rtol=0., atol=0.)
                        self.assertTrue(torch.equal(actual.generator.get_state(), reference.generator.get_state()))
                        self.assertIs(actual.step(None, control_step=tick), actual.packet)
                        terminal_snapshots.append((a.clone(), a.clone()))
                        reset = torch.tensor([False, True, False, False, True])
                        actual.reset(reset)
                        reference.pending[reset] = True
                        changed = raw.clone(); changed[:, 0].clamp_(max=1.).mul_(.8)
                        a = actual.step(changed, control_step=tick, terrain_is_stepping_stones=stones)
                        b = reference.step(changed, stones, tick)
                        torch.testing.assert_close(a, b, rtol=0., atol=0.)
                        self.assertTrue(torch.equal(actual.generator.get_state(), reference.generator.get_state()))
                        torch.testing.assert_close(actual.captures_per_env, reference.counts, rtol=0., atol=0.)
                        self.assertEqual(actual.capture_count, reference.calls)
                    for snapshot, saved in terminal_snapshots:
                        torch.testing.assert_close(snapshot, saved, rtol=0., atol=0.)

    def test_full_capture_checks_packet_once_and_reuses_fixed_strips_without_arange(self):
        capture = HikingDepthCapture(5, generator=torch.Generator().manual_seed(92))
        source = packet()
        stones = torch.zeros(5, dtype=torch.bool)
        strips = tuple(value.data_ptr() for value in capture._strips)
        with patch.object(randomization, "_check_packet", wraps=randomization._check_packet) as validate, \
                patch("torch.arange", side_effect=AssertionError("full frame rebuilt its indices")):
            capture.step(source, control_step=0, terrain_is_stepping_stones=stones)
            capture.step(source, control_step=1, terrain_is_stepping_stones=stones)
        self.assertEqual(validate.call_count, 2)
        self.assertEqual(tuple(value.data_ptr() for value in capture._strips), strips)

    def test_validation_priority_and_unselected_invalid_rows_remain_checked(self):
        capture = HikingDepthCapture(5, enabled_randomization=False)
        capture.step(packet(), control_step=0)
        capture.reset([1])
        broken = packet()
        broken[4, 1, 0, 0] = 2.
        with self.assertRaisesRegex(ValueError, "validity"):
            capture.step(broken, control_step=0)
        broken[0, 0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            capture.step(broken, control_step=0)
        with self.assertRaisesRegex(ValueError, "non-finite"):
            augment_hiking_depth_packet(broken, enabled=False)

    def test_static_terrain_constants_reused_but_live_types_and_config_changes_are_honored(self):
        sub = {"rough": NS(proportion=.2), "stepping_stones": NS(proportion=.4), "stairs": NS(proportion=.4)}
        generator = NS(sub_terrains=sub, num_cols=20)
        terrain = NS(cfg=NS(terrain_generator=generator), terrain_types=torch.arange(20))
        image = torch.zeros(20, 2, 36, 64)
        def original():
            keys = list(generator.sub_terrains)
            p = image.new_tensor([generator.sub_terrains[key].proportion for key in keys])
            thresholds = p.cumsum(0) / p.sum()
            fraction = (terrain.terrain_types.to(image.dtype) + .5) / generator.num_cols
            family = torch.searchsorted(thresholds, fraction).clamp_max(len(keys) - 1)
            return torch.tensor(["stepping" in key for key in keys])[family]
        torch.testing.assert_close(_hiking_stepping_stones_mask(terrain, image), original())
        cached = terrain._hiking_depth_family_constants
        terrain.terrain_types = terrain.terrain_types.flip(0)
        torch.testing.assert_close(_hiking_stepping_stones_mask(terrain, image), original())
        self.assertIs(terrain._hiking_depth_family_constants, cached)
        generator.sub_terrains["rough"].proportion = .5
        torch.testing.assert_close(_hiking_stepping_stones_mask(terrain, image), original())
        self.assertIsNot(terrain._hiking_depth_family_constants, cached)


if __name__ == "__main__":
    unittest.main()
