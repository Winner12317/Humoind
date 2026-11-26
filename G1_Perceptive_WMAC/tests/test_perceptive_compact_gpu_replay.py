"""Shared-frame uint8 replay against the existing full-stack replay oracle."""

from __future__ import annotations

import copy
from dataclasses import replace
import unittest

import torch

from src.wmac_perceptive.compact_gpu_replay import CompressedLatentReplay
from src.wmac_perceptive.latent_perception import processed_depth_stack, unpack_processed_actor_input
from src.wmac_perceptive.latent_replay import LatentPerceptiveReplay
from src.wmac_perceptive.perception import PerceptionSpec, advance_depth_ring, select_depth_frames


SPEC = PerceptionSpec(height=2, width=3, ring_length=4, frames=2, stride=2, delay=1,
    crop_top=0, crop_bottom=0, crop_left=0, crop_right=0, blur_kernel=1)


def frame(generator, count=4, device="cpu"):
    depth = torch.rand(count, 1, 2, 3, generator=generator)
    valid = torch.randint(0, 2, (count, 1, 2, 3), generator=generator).float()
    return torch.cat((depth, valid), 1).to(device)


def native_batch(step, history, next_history, *, invalid=False, bad_image=False):
    count, device = len(history), history.device
    state = torch.zeros(count, 71, device=device)
    state[:, 0] = step
    state[:, 1] = torch.arange(count, device=device)
    state[:, 2], state[:, 3] = .8, 1
    context = torch.zeros(count, 35, device=device)
    context[2, 34] = 3
    successor = state.clone()
    successor[:, 0] += .01
    if invalid:
        successor[1, 0] = float("nan")
    current = select_depth_frames(history, SPEC).flatten(-4, -3)
    following = select_depth_frames(next_history, SPEC).flatten(-4, -3)
    if bad_image:
        following = following.clone()
        following[2, 0, 0, 0] = float("nan")
    terminal = torch.zeros(count, dtype=torch.bool, device=device)
    timeout = torch.zeros_like(terminal)
    boundary = torch.zeros_like(terminal)
    if step in (1, 9, 17):
        terminal[0] = True
    if step in (4, 11, 23):
        timeout[1] = True
    if step in (2, 12):
        boundary[2] = True  # Boundary alone does not reset the camera history.
    return dict(state=state, next_state=successor, context=context,
        action=torch.zeros(count, 29, device=device),
        actor_history=torch.arange(count*384, device=device).reshape(count, 384).float()*1e-5 + step*.01,
        relative_height=torch.tensor([.2, .8, .8, .8], device=device),
        depth_stack=current, next_depth_stack=following,
        terminated=terminal, timeout=timeout, boundary=boundary)


def replay_pair(steps=12, capacity=6, device="cpu", invalid=True):
    generator = torch.Generator().manual_seed(432)
    history = torch.stack([frame(generator, device=device) for _ in range(SPEC.ring_length)], 1)
    compressed = CompressedLatentReplay(capacity, 4, device, .25, spec=SPEC)
    reference = LatentPerceptiveReplay(capacity, 4, device, .25, spec=SPEC)
    compressed.bootstrap_processed_history(history)
    for step in range(steps):
        next_frame = frame(generator, device=device)
        next_history = advance_depth_ring(history, next_frame, SPEC)
        batch = native_batch(step, history, next_history, invalid=invalid and step == 3,
                             bad_image=invalid and step == 5)
        accepted = compressed.add(batch)
        expected = reference.add(batch)
        torch.testing.assert_close(accepted, expected, rtol=0, atol=0)
        done = batch["terminated"] | batch["timeout"]
        reset = frame(generator, device=device)
        history = torch.where(done[:, None, None, None, None], reset[:, None].expand_as(next_history), next_history)
    return compressed, reference


class CompressedReplayOracleMixin:
    def assert_samples_equal(self, compressed, reference, *, windows=False, split="train", recovery=0.):
        fields = ("state", "next_state", "context", "actor_history", "depth_stack", "next_depth_stack", "terminated", "timeout")
        torch.manual_seed(882)
        actual = (compressed.sample_windows(128, 3, recovery, fields, split) if windows
                  else compressed.sample_starts(128, recovery, fields, split))
        torch.manual_seed(882)
        expected = (reference.sample_windows(128, 3, recovery, fields, split) if windows
                    else reference.sample_starts(128, recovery, fields, split))
        for name in actual:
            if name in ("depth_stack", "next_depth_stack"):
                self.assertLessEqual(float((actual[name] - expected[name].float()).abs().max()), 1/255)
                torch.testing.assert_close(actual[name][..., 1::2, :, :], expected[name][..., 1::2, :, :].float(), rtol=0, atol=0)
            elif name == "actor_history" and not windows:
                proprio, image = unpack_processed_actor_input(actual[name], SPEC)
                truth, truth_image = unpack_processed_actor_input(expected[name], SPEC)
                torch.testing.assert_close(proprio, truth, rtol=0, atol=0)
                self.assertLessEqual(float((image-truth_image).abs().max()), 1/255)
            else:
                torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0, equal_nan=True)


class CompressedReplayCpuTest(CompressedReplayOracleMixin, unittest.TestCase):
    def test_reference_model_requires_complete_physics_windows_and_uniform_envs(self):
        replay, _ = replay_pair(steps=12, capacity=12, invalid=False)
        batch = replay.sample_reference_model_windows(12000, 3, fields=("terminated", "timeout", "boundary"))
        self.assertTrue(batch["mask"].all())
        self.assertFalse((batch["terminated"] | batch["timeout"]).any())
        self.assertTrue(batch["boundary"].any())  # Goal changes do not reset physical dynamics.
        self.assertLessEqual(int(batch["time_index"].max()), 10)  # Eighth state must be present.
        counts = batch["env_index"][:, 0].bincount(minlength=4)
        self.assertTrue(((counts[:3] - 4000).abs() < 400).all())
        self.assertEqual(int(counts[3]), 0)

    def test_initial_history_reset_and_quantization_match_full_stack_oracle(self):
        compressed, reference = replay_pair(steps=6, capacity=8)
        self.assert_samples_equal(compressed, reference)
        self.assert_samples_equal(compressed, reference, windows=True)
        self.assertEqual(compressed.invalid_transitions, reference.invalid_transitions)
        self.assertNotIn("depth_stack", compressed.data)
        self.assertNotIn("next_depth_stack", compressed.data)
        self.assertEqual(compressed.frame_cache.dtype, torch.uint8)
        self.assertEqual(compressed.data["actor_history"].dtype, torch.float32)
        self.assertFalse(any("latent" in name or name == "z" for name in compressed.data))

    def test_many_ring_wraps_retain_oldest_record_history_and_pre_reset_successor(self):
        compressed, reference = replay_pair(steps=31, capacity=6, invalid=False)
        self.assertEqual((compressed.ptr, compressed.size, compressed.total_steps), (1, 6, 31))
        self.assert_samples_equal(compressed, reference)
        self.assert_samples_equal(compressed, reference, windows=True)
        for age in range(compressed.size):
            row = compressed._row(torch.tensor(age))
            env = torch.arange(4)
            rows = row.expand_as(env)
            for name, successor in (("depth_stack", False), ("next_depth_stack", True)):
                actual = compressed._depth(rows, env, successor)
                truth = reference.data[name][row].float()
                self.assertLessEqual(float((actual-truth).abs().max()), 1/255)

    def test_held_out_split_strata_boundaries_and_invalid_padding_preserve_sampling(self):
        compressed, reference = replay_pair(steps=6, capacity=8)
        for split in ("train", "validation", "all"):
            for fraction in (0., .5, 1.):
                self.assert_samples_equal(compressed, reference, split=split, recovery=fraction)
                self.assert_samples_equal(compressed, reference, windows=True, split=split, recovery=fraction)
        self.assertEqual(compressed.num_valid("train"), reference.num_valid("train"))
        torch.testing.assert_close(compressed._valid, reference._valid)
        torch.testing.assert_close(compressed._recovery, reference._recovery)

    def test_bootstrap_is_explicit_reset_shortcut_and_storage_marker_are_unambiguous(self):
        generator = torch.Generator().manual_seed(233)
        first = frame(generator)
        history = first[:, None].expand(-1, SPEC.ring_length, -1, -1, -1).clone()
        following = advance_depth_ring(history, frame(generator), SPEC)
        batch = native_batch(0, history, following)
        replay = CompressedLatentReplay(5, 4, "cpu", spec=SPEC)
        with self.assertRaisesRegex(ValueError, "bootstrap"):
            replay.add(batch)
        replay.bootstrap_reset(first)
        replay.add(batch)
        values = replay.sample_starts(4, fields=("depth_stack",))
        for image, env in zip(values["depth_stack"], values["env_index"]):
            self.assertLessEqual(float((image-first[env].repeat(SPEC.frames, 1, 1)).abs().max()), 1/255)
        with self.assertRaises(ValueError):
            replay.bootstrap_processed_history(history)
        self.assertEqual(replay.state_dict()["perceptive_format"], "frame_uint8_v1")
        with self.assertRaisesRegex(ValueError, "one-frame delay"):
            CompressedLatentReplay(5, 4, "cpu", spec=replace(SPEC, delay=0))

    def test_bootstrap_on_first_add_and_reject_nonbinary_or_nonfinite_history(self):
        generator = torch.Generator().manual_seed(433)
        history = torch.stack([frame(generator) for _ in range(SPEC.ring_length)], 1)
        batch = native_batch(0, history, advance_depth_ring(history, frame(generator), SPEC))
        replay = CompressedLatentReplay(5, 4, "cpu", spec=SPEC)
        replay.add(batch, processed_history=history)
        self.assertEqual(replay.size, 1)
        for bad in (float("nan"), .4):
            invalid = history.clone()
            invalid[0, 0, 1, 0, 0] = bad
            fresh = CompressedLatentReplay(5, 4, "cpu", spec=SPEC)
            with self.assertRaises(ValueError):
                fresh.bootstrap_processed_history(invalid)
            self.assertFalse(fresh._bootstrapped)

    def test_native_schema_rejection_does_not_mutate_frame_cache_or_time(self):
        compressed, _ = replay_pair(steps=2, capacity=5, invalid=False)
        history = torch.zeros(4, SPEC.ring_length, 2, 2, 3)
        history[:, :, 1] = 1
        bad = native_batch(2, history, history)
        bad["action"] = torch.zeros(4, 30)
        cache, ptr, total = compressed.frame_cache.clone(), compressed.ptr, compressed.total_steps
        with self.assertRaises(ValueError):
            compressed.add(bad)
        torch.testing.assert_close(compressed.frame_cache, cache, rtol=0, atol=0)
        self.assertEqual((compressed.ptr, compressed.total_steps), (ptr, total))

    def test_compressed_serialization_is_explicit_and_rejects_full_stack_replay(self):
        compressed, reference = replay_pair(steps=12, capacity=6)
        payload = copy.deepcopy(compressed.state_dict())
        restored = CompressedLatentReplay(6, 4, "cpu", .25, spec=SPEC)
        restored.load_state_dict(payload)
        self.assert_samples_equal(restored, reference)
        for foreign in (reference.state_dict(), {**payload, "perceptive_format": "instinct-depth-stack-replay-v1"}):
            with self.assertRaisesRegex(ValueError, "protocol"):
                restored.load_state_dict(foreign)

    def test_default_budget_estimate_allocates_no_full_stack_tensor(self):
        expected = (1000 + 37) * 4096 * 2 * 18 * 32
        self.assertEqual(CompressedLatentReplay.estimated_frame_cache_bytes(1000, 4096), expected)
        self.assertLess(expected, 5 * 1024**3)
        replay = CompressedLatentReplay(5, 4, "cpu", spec=SPEC)
        self.assertEqual(tuple(replay.frame_cache.shape), (9, 4, 2, 2, 3))

    def test_restore_marks_first_new_camera_history_as_a_fresh_episode(self):
        compressed, _ = replay_pair(steps=3, capacity=6, invalid=False)
        restored = CompressedLatentReplay(6, 4, "cpu", .25, spec=SPEC)
        restored.load_state_dict(copy.deepcopy(compressed.state_dict()))
        generator = torch.Generator().manual_seed(434)
        reset = frame(generator)
        history = reset[:, None].expand(-1, SPEC.ring_length, -1, -1, -1).clone()
        batch = native_batch(3, history, advance_depth_ring(history, frame(generator), SPEC))
        restored.add(batch)
        row, env = torch.full((4,), 3, dtype=torch.long), torch.arange(4)
        current = restored._depth(row, env)
        self.assertLessEqual(float((current-batch["depth_stack"]).abs().max()), 1/255)

    def test_float64_gaussian_roundoff_bootstraps_without_accepting_real_outliers_or_bad_masks(self):
        spec = PerceptionSpec()
        raw = torch.ones(1, *spec.ring_shape, dtype=torch.float64)
        processed = processed_depth_stack(raw, spec)
        self.assertTrue(bool(CompressedLatentReplay._visual_valid(processed).all()))
        # Use the actual Gaussian result, retaining any accumulation roundoff.
        pair = processed[:, -2:]
        history = pair[:, None].expand(-1, spec.ring_length, -1, -1, -1).clone()
        replay = CompressedLatentReplay(2, 1, "cpu", spec=spec)
        replay.bootstrap_processed_history(history)
        self.assertTrue(replay._bootstrapped)
        for dtype in (torch.float16, torch.float32, torch.float64):
            for value in (1.001, -.001, float("nan"), float("inf"), float("-inf")):
                bad = history.to(dtype).clone()
                bad[0, 0, 0, 0, 0] = value
                fresh = CompressedLatentReplay(2, 1, "cpu", spec=spec)
                with self.subTest(dtype=dtype, depth=value), self.assertRaises(ValueError):
                    fresh.bootstrap_processed_history(bad)
            for value in (.5, 1.001, float("nan"), float("inf")):
                bad = history.to(dtype).clone()
                bad[0, 0, 1, 0, 0] = value
                fresh = CompressedLatentReplay(2, 1, "cpu", spec=spec)
                with self.subTest(dtype=dtype, valid=value), self.assertRaises(ValueError):
                    fresh.bootstrap_processed_history(bad)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
class CompressedReplayCudaTest(CompressedReplayOracleMixin, unittest.TestCase):
    def test_cuda_wrap_reset_padding_splits_and_reconstruction_remain_on_gpu(self):
        compressed, reference = replay_pair(steps=25, capacity=6, device="cuda:0")
        self.assert_samples_equal(compressed, reference)
        self.assert_samples_equal(compressed, reference, windows=True)
        self.assert_samples_equal(compressed, reference, windows=True, split="validation")
        values = compressed.sample_windows(8, 3, fields=("depth_stack", "next_depth_stack", "state"))
        self.assertTrue(all(value.is_cuda for value in values.values()))


if __name__ == "__main__":
    unittest.main()
