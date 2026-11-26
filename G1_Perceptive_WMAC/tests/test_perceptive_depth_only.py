"""Depth-only visual inputs and shared replay preserve the reference observation.

These checks exercise the removed validity input, Hiking's blur-before-clip
ordering, the encoder gradient boundary, and replay reconstruction across
terminal resets and ring wraps. Historical two-channel behavior stays covered.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, replace
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from src.wmac_perceptive.compact_gpu_replay import CompressedLatentReplay
from src.wmac_perceptive.instinct_latent import InstinctLatentDecoder
from src.wmac_perceptive.latent_models import LatentPerceptiveActor
from src.wmac_perceptive.latent_perception import (
    pack_latent_actor_input, pack_processed_actor_input, processed_depth_frame,
    processed_depth_history, processed_depth_stack, processed_stack_shape,
    unpack_processed_actor_input,
)
from src.wmac_perceptive.latent_replay import LatentPerceptiveReplay
from src.wmac_perceptive.perception import (
    DepthHistory, PerceptionSpec, advance_depth_ring, pack_actor_input, select_depth_frames,
)
from test_perceptive_compact_gpu_replay import SPEC as REPLAY_SPEC, frame, native_batch


class DepthOnlyObservationTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(570)
        self.spec = PerceptionSpec(include_valid=False)

    def test_depth_only_encoder_and_decoder_have_eight_channels_and_no_validity_input(self):
        actor = LatentPerceptiveActor(self.spec).double()
        decoder = InstinctLatentDecoder(128, self.spec.frames).double()
        self.assertEqual(actor.vision.conv[0].in_channels, 8)
        ring = torch.rand(2, *self.spec.ring_shape, dtype=torch.float64)
        altered = ring.clone()
        altered[:, :, 1] = 1 - altered[:, :, 1]
        stack = processed_depth_stack(ring, self.spec)
        self.assertEqual(stack.shape, (2, 8, 18, 32))
        torch.testing.assert_close(stack, processed_depth_stack(altered, self.spec), rtol=0, atol=0)
        torch.testing.assert_close(actor.encode_history(ring), actor.encode_history(altered), rtol=0, atol=0)
        self.assertEqual(decoder(actor.encode_processed(stack)).shape, stack.shape)
        history = processed_depth_history(ring, self.spec)
        self.assertEqual(history.shape, (2, self.spec.ring_length, 1, 18, 32))
        torch.testing.assert_close(history[:, list(self.spec.selected_indices), 0], stack, rtol=0, atol=1e-15)

    def test_gaussian_blur_precedes_depth_clipping(self):
        spec = replace(self.spec, height=4, width=4, crop_top=0, crop_left=0, crop_right=0)
        raw = torch.zeros(1, 2, 4, 4, dtype=torch.float64)
        # A beyond-range pixel influences its neighbours before the image is
        # clipped, just as the reference crop -> GaussianBlur -> clip modifiers.
        raw[0, 0, 1, 1] = 2.0
        raw[0, 0, 2, 2] = -.1
        raw[:, 1] = 1
        axis = torch.tensor([-1., 0., 1.], dtype=torch.float64)
        gaussian = (-axis.square()/2).exp()
        gaussian /= gaussian.sum()
        kernel = (gaussian[:, None] * gaussian[None, :])[None, None]
        expected = F.conv2d(F.pad(raw[:, :1], (1, 1, 1, 1), mode="reflect"), kernel).clamp(0, 1)
        actual = processed_depth_frame(raw, spec)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        prematurely_clipped = F.conv2d(F.pad(raw[:, :1].clamp(0, 1), (1, 1, 1, 1), mode="reflect"), kernel)
        self.assertGreater(float((actual-prematurely_clipped).abs().max()), .1)

    def test_raw_processed_and_latent_actor_actions_match_and_policy_gradient_stops_at_encoder(self):
        actor = LatentPerceptiveActor(self.spec).double()
        proprio = torch.randn(2, 384, dtype=torch.float64)
        ring = torch.rand(2, *self.spec.ring_shape, dtype=torch.float64, requires_grad=True)
        stack = processed_depth_stack(ring, self.spec)
        latent = actor.encode_processed(stack)
        raw = pack_actor_input(proprio, ring, self.spec)
        processed = pack_processed_actor_input(proprio, stack, self.spec)
        compact = pack_latent_actor_input(proprio, latent, self.spec)
        torch.testing.assert_close(actor(raw), actor(processed), rtol=0, atol=0)
        torch.testing.assert_close(actor(raw), actor(compact), rtol=0, atol=0)
        actor(raw).square().sum().backward()
        self.assertTrue(ring.grad is None or not bool(ring.grad.any()))
        self.assertTrue(all(p.grad is None for p in actor.vision.parameters()))

    def test_reconstruction_updates_encoder_without_reading_validity(self):
        actor = LatentPerceptiveActor(self.spec)
        decoder = InstinctLatentDecoder(128, self.spec.frames)
        raw = torch.rand(2, *self.spec.ring_shape, requires_grad=True)
        stack = processed_depth_stack(raw, self.spec)
        prediction = decoder(actor.encode_processed(stack))
        (prediction-stack.detach()).abs().mean().backward()
        self.assertGreater(float(raw.grad[:, :, 0].abs().sum()), 0.)
        self.assertEqual(float(raw.grad[:, :, 1].abs().sum()), 0.)
        self.assertTrue(any(p.grad is not None and bool(p.grad.any()) for p in actor.vision.parameters()))
        self.assertTrue(all(p.grad is None for p in actor.policy.parameters()))

    def test_eight_channel_decoder_matches_pinned_instinct_forward_and_gradients(self):
        from test_instinct_latent_reference import reference_decoder_and_stats
        reference_class, _ = reference_decoder_and_stats()
        reference = reference_class(128, channels=8).double()
        decoder = InstinctLatentDecoder(128, 8).double()
        decoder.load_state_dict(reference.state_dict(), strict=True)
        first = torch.randn(2, 128, dtype=torch.float64, requires_grad=True)
        second = first.detach().clone().requires_grad_()
        expected, actual = reference(first), decoder(second)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        weights = torch.randn_like(expected)
        (expected*weights).sum().backward()
        (actual*weights).sum().backward()
        torch.testing.assert_close(second.grad, first.grad, rtol=0, atol=0)
        for left, right in zip(decoder.parameters(), reference.parameters()):
            torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)

    def test_legacy_shape_and_weights_cannot_silently_become_depth_only(self):
        legacy = LatentPerceptiveActor()
        current = LatentPerceptiveActor(self.spec)
        self.assertEqual(legacy.vision.conv[0].in_channels, 16)
        self.assertEqual(processed_stack_shape(legacy.spec), (16, 18, 32))
        with self.assertRaises(RuntimeError):
            current.load_state_dict(legacy.state_dict(), strict=True)
        with self.assertRaises(ValueError):
            PerceptionSpec(include_valid=1)


class DepthOnlyReplayTest(unittest.TestCase):
    def build_replays(self):
        spec = replace(REPLAY_SPEC, include_valid=False)
        generator = torch.Generator().manual_seed(571)
        history = torch.stack([frame(generator) for _ in range(spec.ring_length)], 1)
        compressed = CompressedLatentReplay(6, 4, "cpu", .25, spec=spec)
        reference = LatentPerceptiveReplay(6, 4, "cpu", .25, spec=spec)
        compressed.bootstrap_processed_history(processed_depth_history(history, spec))
        for step in range(31):
            successor = advance_depth_ring(history, frame(generator), spec)
            batch = native_batch(step, history, successor)
            batch.update(depth_stack=processed_depth_stack(history, spec),
                         next_depth_stack=processed_depth_stack(successor, spec))
            torch.testing.assert_close(compressed.add(batch), reference.add(batch), rtol=0, atol=0)
            done = batch["terminated"] | batch["timeout"]
            reset = frame(generator)
            history = torch.where(done[:, None, None, None, None], reset[:, None].expand_as(successor), successor)
        return compressed, reference, spec

    def test_reset_wrap_and_sampling_match_full_stack_oracle_without_dummy_valid_channels(self):
        compressed, reference, spec = self.build_replays()
        self.assertEqual(compressed.frame_cache.shape, (10, 4, 1, 2, 3))
        for split in ("train", "validation", "all"):
            for windows in (False, True):
                torch.manual_seed(572)
                actual = (compressed.sample_windows(128, 3, 0., split=split) if windows
                          else compressed.sample_starts(128, 0., split=split))
                torch.manual_seed(572)
                expected = (reference.sample_windows(128, 3, 0., split=split) if windows
                            else reference.sample_starts(128, 0., split=split))
                for key in actual:
                    if key in ("depth_stack", "next_depth_stack"):
                        self.assertEqual(actual[key].shape[-3], spec.frames)
                        self.assertLessEqual(float((actual[key]-expected[key].float()).abs().max()), 1/255)
                    elif key == "actor_history" and not windows:
                        proprio, visual = unpack_processed_actor_input(actual[key], spec)
                        truth, truth_visual = unpack_processed_actor_input(expected[key], spec)
                        torch.testing.assert_close(proprio, truth, rtol=0, atol=0)
                        self.assertLessEqual(float((visual-truth_visual).abs().max()), 1/255)
                    else:
                        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
        restored = CompressedLatentReplay(6, 4, "cpu", .25, spec=spec)
        restored.load_state_dict(copy.deepcopy(compressed.state_dict()))
        torch.testing.assert_close(restored.frame_cache, compressed.frame_cache, rtol=0, atol=0)
        self.assertNotIn("depth_stack", restored.data)

    def test_all_depth_channels_are_continuous_values_and_receive_range_validation(self):
        spec = replace(REPLAY_SPEC, include_valid=False)
        replay = CompressedLatentReplay(6, 4, "cpu", spec=spec)
        depth = torch.full((4, 1, 2, 3), .43)
        replay.bootstrap_reset(depth)  # .43 must not be mistaken for a binary validity channel.
        for bad in (-.001, 1.001, float("nan"), float("inf")):
            stack = torch.full((4, spec.frames, 2, 3), .43)
            stack[1, 1, 0, 0] = bad
            accepted = replay._visual_valid(stack, include_valid=False)
            self.assertEqual(accepted.tolist(), [True, False, True, True])

    def test_only_depth_channel_is_allocated_and_cache_budget_halves(self):
        legacy = CompressedLatentReplay.estimated_frame_cache_bytes(1000, 4096)
        current = CompressedLatentReplay.estimated_frame_cache_bytes(1000, 4096, PerceptionSpec(include_valid=False))
        self.assertEqual(current * 2, legacy)
        self.assertEqual(current, 1037 * 4096 * 18 * 32)


class DepthOnlyRunnerWiringTest(unittest.TestCase):
    def make_runner(self, spec, *, processed=True):
        from src.wmac_perceptive.runner import PerceptiveRunner
        runner = object.__new__(PerceptiveRunner)
        runner.spec = spec
        runner.cfg = SimpleNamespace(depth_noise_std_m=0., depth_dropout_probability=0.,
            exploration_noise=0., action_limit=20., replay_storage="frame_uint8",
            training_schedule="supermimic_v1")
        runner._processed_history = processed
        return runner

    def test_processed_two_channel_ring_and_raw_ring_produce_the_same_depth_only_observation(self):
        spec = PerceptionSpec(include_valid=False)
        runner = self.make_runner(spec)
        torch.manual_seed(573)
        raw = torch.rand(2, *spec.ring_shape)
        raw[:, :, 0, 21, 20] = 4e5  # Hiking's 1e6 m no-hit sentinel divided by 2.5 m.
        raw[:, :, 0, 22, 20] = 1.4  # Beyond policy range must enter the blur before clipping.
        raw[:, :, 1] = torch.randint(0, 2, raw[:, :, 1].shape).float()
        prepared = runner._policy_depth_frame(raw.flatten(0, 1))
        prepared = prepared.reshape(2, spec.ring_length, 2, 18, 32)
        history_spec = replace(spec, height=18, width=32, crop_top=0, crop_bottom=0,
                               crop_left=0, crop_right=0, blur_kernel=1)
        selected = select_depth_frames(prepared, history_spec).flatten(-4, -3)[..., 0::2, :, :]
        expected = processed_depth_stack(raw, spec)
        self.assertTrue(bool(torch.isfinite(selected).all()))
        self.assertTrue(bool(((selected >= 0) & (selected <= 1)).all()))
        torch.testing.assert_close(selected, expected, rtol=1e-6, atol=1e-7)
        # Diagnostic validity remains separate, but cannot change the encoded image.
        changed = raw.clone()
        changed[:, :, 1] = 1 - changed[:, :, 1]
        changed_prepared = runner._policy_depth_frame(changed.flatten(0, 1))
        torch.testing.assert_close(changed_prepared[:, 0], prepared.flatten(0, 1)[:, 0], rtol=0, atol=0)
        slow = self.make_runner(spec, processed=False)
        slow_raw = slow._policy_depth_frame(raw.flatten(0, 1)).reshape_as(raw)
        self.assertEqual(float(slow_raw[:, :, 0].max()), 4e5)
        torch.testing.assert_close(processed_depth_stack(slow_raw, spec), expected, rtol=0, atol=0)

    def test_depth_only_diagnostics_measure_all_depth_pixels_and_do_not_invent_validity_accuracy(self):
        from src.wmac_perceptive.latent_training import decoded_stack_metrics
        target = torch.ones(2, 8, 3, 4)
        target[:, 1] = .25  # This is another depth frame, not a binary valid mask.
        prediction = torch.zeros_like(target)
        metrics = decoded_stack_metrics(prediction, target, 2.5, include_valid=False)
        expected = target.square().mean((-3, -2, -1)).sqrt() * 2.5
        torch.testing.assert_close(metrics["actor_rmse"], expected, rtol=0, atol=0)
        torch.testing.assert_close(metrics["actor_valid_pixels"], torch.full((2,), 8*3*4.))
        for key in ("actor_mask_error", "full_mask_error", "full_rmse", "outside_rmse"):
            self.assertTrue(bool(torch.isnan(metrics[key]).all()), key)
        self.assertEqual(float(metrics["full_valid_pixels"].sum()), 0.)
        self.assertEqual(float(metrics["outside_valid_pixels"].sum()), 0.)

    def test_collector_reaches_uint8_directly_without_fp16_rounding_and_never_stores_validity(self):
        from src.wmac_perceptive.latent_training import collect_latent
        spec = replace(REPLAY_SPEC, include_valid=False)
        count = 2
        runner = self.make_runner(spec)
        runner.actor = LatentPerceptiveActor(spec)
        runner.replay = CompressedLatentReplay(5, count, "cpu", spec=spec)
        runner._train_env_mask = torch.ones(count, dtype=torch.bool)
        runner.world_model = SimpleNamespace(update_normalizers_latent=lambda *args: None)
        runner.sim_steps = 0
        initial = torch.full((count, 2, 2, 3), .25)
        initial[:, 1] = 1
        runner.current_depth = initial
        runner.depth_history = DepthHistory(spec, count)
        runner.depth_history.reset(initial)
        runner.replay.bootstrap_reset(initial[:, :1])
        # This value quantizes to 90 from FP32 but 91 after an FP16 detour.
        marker = torch.tensor((90.5)/255-1e-6, dtype=torch.float32)
        next_frame = torch.full_like(initial, marker)
        next_frame[:, 1] = 0  # Diagnostic mask is deliberately unlike a depth value.
        state = torch.zeros(count, 71)
        state[:, 2], state[:, 3] = .8, 1
        env = SimpleNamespace(num_envs=count, obs_buf={"actor": torch.zeros(count, spec.proprio_dim)})

        def step(action):
            env.transition = dict(state=state.clone(), next_state=state.clone(), action=action,
                context=torch.zeros(count, 35), actor_history=env.obs_buf["actor"],
                relative_height=torch.full((count,), .8), next_depth_frame=next_frame,
                reward=torch.zeros(count), terminated=torch.zeros(count, dtype=torch.bool),
                timeout=torch.zeros(count, dtype=torch.bool), boundary=torch.zeros(count, dtype=torch.bool))

        env.step = step
        runner.env = env
        with patch("src.wmac_perceptive.latent_training.read_depth_frame", return_value=next_frame), \
                patch.object(runner.replay, "add", wraps=runner.replay.add) as add:
            collect_latent(runner, 2)  # The fixed one-frame camera delay exposes marker on the second transition.
        saved = add.call_args.args[0]
        self.assertEqual(saved["depth_stack"].dtype, torch.float32)
        self.assertEqual(saved["next_depth_stack"].dtype, torch.float32)
        self.assertEqual(saved["next_depth_stack"].shape[-3], spec.frames)
        self.assertNotIn("next_depth_frame", saved)
        expected = (marker * 255).round().to(torch.uint8)
        premature = (marker.half().float() * 255).round().to(torch.uint8)
        self.assertNotEqual(int(expected), int(premature))
        torch.testing.assert_close(runner.replay.frame_cache[1], torch.full((count, 1, 2, 3), expected, dtype=torch.uint8))

    def test_saved_visual_schema_is_checked_by_resume_and_playback_with_legacy_default_compatibility(self):
        from scripts.evaluate_wmac_perceptive import load_actor
        from src.wmac_perceptive.checkpoints import architecture_metadata, validate_checkpoint
        from src.wmac_perceptive.config import FORMAT, LATENT_VISUAL_MODE, PerceptiveConfig, perception_spec_from_config
        configs = (replace(PerceptiveConfig.defaults(), visual_mode=LATENT_VISUAL_MODE),
                   PerceptiveConfig.defaults("supermimic"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"visual-schema.pt"
            for cfg in configs:
                spec = perception_spec_from_config(cfg)
                actor = LatentPerceptiveActor(spec,gate_hidden=cfg.moe_gate_hidden)
                payload = dict(format=FORMAT, config=asdict(cfg), architecture=architecture_metadata(cfg),
                    visual_spec=asdict(spec), amp_grounding=cfg.ground_height_semantics,
                    terrain_fingerprint="fixed-terrain", motion_manifest_sha256="fixed-motion",
                    actor=actor.state_dict())
                if spec.include_valid:
                    # Files predating this field still reconstruct the exact
                    # original 16-channel Actor and historical camera settings.
                    payload["config"].pop("sensor_protocol")
                    payload["visual_spec"].pop("include_valid")
                self.assertEqual(validate_checkpoint(payload, cfg), {})
                torch.save(payload, path)
                loaded, restored_cfg, _ = load_actor(path, "cpu")
                self.assertEqual(restored_cfg.sensor_protocol, cfg.sensor_protocol)
                self.assertEqual(loaded.spec.include_valid, spec.include_valid)
                for key, value in (("include_valid", not spec.include_valid),
                                   ("stride", spec.stride-1), ("crop_left", spec.crop_left-1)):
                    altered = copy.deepcopy(payload)
                    altered["visual_spec"][key] = value
                    with self.subTest(protocol=cfg.sensor_protocol, field=key):
                        with self.assertRaises(ValueError):
                            validate_checkpoint(altered, cfg)
                        torch.save(altered, path)
                        with self.assertRaises(ValueError):
                            load_actor(path, "cpu")
                if not spec.include_valid:
                    missing = copy.deepcopy(payload)
                    missing["visual_spec"].pop("include_valid")
                    with self.assertRaises(ValueError):
                        validate_checkpoint(missing, cfg)
                    torch.save(missing, path)
                    with self.assertRaises(ValueError):
                        load_actor(path, "cpu")


if __name__ == "__main__":
    unittest.main()
