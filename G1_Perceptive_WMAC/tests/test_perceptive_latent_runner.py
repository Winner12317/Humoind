"""G1/FK integration of online stack latents, replay and strict checkpoints.

The fixture is a fresh latent experiment, built from the existing real G1
kinematics fixture. It performs no historical checkpoint migration and creates
no GPU simulator. Collection tests substitute only environment I/O; history,
replay, encoding, normalization, losses and optimizers execute real code.
"""

from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from src.wmac.networks import frozen_parameters
from src.wmac.runner import WmacRunner
from src.wmac_perceptive.config import GENCOORD_DYNAMICS, LATENT_VISUAL_MODE
from src.wmac_perceptive.instinct_latent import InstinctLatentDecoder
from src.wmac_perceptive.latent_models import LatentPerceptiveActor, LatentPerceptiveWorldModel
from src.wmac_perceptive.latent_perception import (
    pack_processed_actor_input, processed_depth_stack, processed_stack_shape, unpack_processed_actor_input,
)
from src.wmac_perceptive.latent_replay import LatentPerceptiveReplay
from src.wmac_perceptive.latent_training import rollout_latent_model_metrics
from src.wmac_perceptive.models import PerceptiveActor, TerrainPerceptiveWorldModel
from src.wmac_perceptive.perception import DepthHistory, PerceptionSpec, advance_depth_ring, unpack_actor_input
from src.wmac_perceptive.runner import MODULES, OPTIMIZERS
from test_perceptive_runner import cloned_state, critic_noop, runner_fixture, tiny_spec
import test_perceptive_runner_restore as restore_checks
from test_wmac_optimization import FixedReplay, fix_update_noise


def latent_runner_fixture(batch_size=3, horizon=3, spec=None, adam_steps=0):
    """Fresh walking-only GenCoord/v2 runner with the production optimizer split.

    Pass ``spec=PerceptionSpec()`` for production-compatible checkpoint tests.
    The default tiny sensor keeps integration tests inexpensive. Returned
    actor starts contain processed pixels, never cached encoder latents.
    """
    runner, previous_batch, noise = runner_fixture(batch_size, horizon)
    proprio, _ = unpack_actor_input(previous_batch["actor_history"], runner.spec)
    runner.cfg = replace(runner.cfg, visual_mode=LATENT_VISUAL_MODE, latent_dim=128,
        dynamics_parameterization=GENCOORD_DYNAMICS, perception_training_version=2,
        motion_scope="forward_walk_turn_v1", recovery_fraction=0., expert_recovery_fraction=0.,
        recovery_flat_fraction=0., validation_min_recovery_samples=0,
        imagination_horizon_start=horizon, wm_window=max(2, horizon), wm_updates=1, latent_recon_images=batch_size*2,
        migrate_world_model=False, migrate_perception_training=False)
    runner.cfg.validate()
    runner.spec = tiny_spec() if spec is None else spec
    runner.adapter.motion_scope = runner.cfg.motion_scope
    runner.adapter.max_delay_steps = 0
    runner.adapter.reward_cfg.pop("track_root_height", None)
    runner.actor = LatentPerceptiveActor(runner.spec).double()
    runner.world_model = LatentPerceptiveWorldModel(runner.adapter.terrain, runner.spec,
        hidden=runner.cfg.model_hidden, dt=runner.cfg.dynamics_dt,
        decimation=runner.cfg.dynamics_decimation).double()
    with torch.no_grad():
        runner.world_model.net[-1].weight.normal_(0., .002)
    shape = processed_stack_shape(runner.spec)
    runner.latent_decoder = InstinctLatentDecoder(128, shape[0], shape[1:]).double()
    runner._model_training_modules = nn.ModuleList((runner.world_model, runner.actor.vision, runner.latent_decoder))
    runner.opt_actor = torch.optim.Adam(runner.actor.policy.parameters(), lr=.0002,
                                       betas=runner.cfg.actor_betas, fused=False)
    runner.opt_model = torch.optim.Adam(runner._model_training_modules.parameters(), lr=.001, fused=False)
    for name in OPTIMIZERS:
        for group in getattr(runner, name).param_groups:
            group["fused"] = False
    runner.WM_FIELDS = (*WmacRunner.WM_FIELDS, "depth_stack", "next_depth_stack", "relative_height")
    runner.VALIDATION_FIELDS = (*runner.WM_FIELDS, "previous_action", "command", "reward")
    runner.ACTOR_FIELDS = (*WmacRunner.ACTOR_FIELDS, "navigation_target", "desired_speed", "depth_stack", "relative_height")
    depth = torch.full((batch_size, *runner.spec.frame_shape), .2, dtype=torch.float64)
    depth[:, 1] = 1
    ring = depth[:, None].expand(batch_size, *runner.spec.ring_shape).clone()
    stack = processed_depth_stack(ring, runner.spec)
    batch = {key: value.clone() for key, value in previous_batch.items() if key != "depth_frame"}
    batch["context"][:, 33:] = 0
    batch.update(actor_history=pack_processed_actor_input(proprio, stack, runner.spec), depth_stack=stack)
    runner.replay = FixedReplay(batch)
    runner.env = SimpleNamespace(num_envs=batch_size, common_step_counter=0, obs_buf={"actor": proprio.clone()})
    runner.started, runner.wall_offset = time.perf_counter(), 0.
    runner.sim_steps, runner.imagination_steps, runner.critic_calibration_updates = 0, 0, 0
    runner._collect_accumulator = 0.
    runner._cost_trajectory_id, runner._cost_history_complete = "fresh-latent-fixture", True
    runner.actor_initialization_history, runner.perception_training_history = [], []
    runner._fingerprints = lambda: ("terrain-fixture", "motion-fixture")
    runner.current_depth = depth
    runner.depth_history = DepthHistory(runner.spec, batch_size, "cpu", torch.float64)
    runner.depth_history.reset(depth)
    runner.validation = runner._empty_validation()
    runner._ensure_training_progress()
    runner._active_horizon = horizon
    if adam_steps:
        for name in OPTIMIZERS:
            optimizer = getattr(runner, name)
            for _ in range(adam_steps):
                for group in optimizer.param_groups:
                    for parameter in group["params"]:
                        parameter.grad = torch.full_like(parameter, .01)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
    return runner, batch, noise


def latent_transition(runner, actor_batch, time_index=0):
    """Finite raw replay row, with no packed pixels or cached z in storage."""
    proprio, stack = unpack_processed_actor_input(actor_batch["actor_history"], runner.spec)
    count = len(proprio)
    next_stack = stack.clone()
    next_stack[..., 0::2, :, :] = (.25 + time_index * .03 + torch.rand_like(next_stack[..., 0::2, :, :]) * .1)
    state = actor_batch["state"].clone()
    state[:, 0] += time_index * .002
    successor = state.clone()
    successor[:, 0] += .002
    return {**{key: value.clone() for key, value in actor_batch.items() if key != "actor_history"},
        "state": state, "next_state": successor, "action": torch.zeros_like(actor_batch["previous_action"]),
        "actor_history": proprio.clone(), "depth_stack": stack.clone(), "next_depth_stack": next_stack,
        "next_context": actor_batch["context"].clone(), "contact_cost": state.new_zeros(count, 2),
        "reward": state.new_zeros(count), "terminated": torch.zeros(count, dtype=torch.bool),
        "timeout": torch.zeros(count, dtype=torch.bool), "boundary": torch.zeros(count, dtype=torch.bool)}


def populated_latent_replay(runner, actor_batch, steps=4):
    replay = LatentPerceptiveReplay(runner.cfg.replay_steps, len(actor_batch["state"]), spec=runner.spec)
    for index in range(steps):
        replay.add(latent_transition(runner, actor_batch, index))
    return replay


def latent_snapshot(runner):
    names = (*MODULES, "latent_decoder", *OPTIMIZERS)
    return copy.deepcopy({**{name: getattr(runner, name).state_dict() for name in names},
        "clocks": (runner.iteration, runner.sim_steps, runner.imagination_steps, runner.critic_calibration_updates),
        "depth_history": runner.depth_history.snapshot(), "torch_rng": torch.get_rng_state()})


class LatentRunnerTrainingTest(unittest.TestCase):
    def test_imagination_encodes_only_initial_stack_once_and_never_calls_decoder(self):
        runner, batch, noise = latent_runner_fixture()
        with patch.object(runner.latent_decoder, "forward", side_effect=AssertionError("decoder entered imagination")), \
                patch.object(runner.actor, "encode_processed", wraps=runner.actor.encode_processed) as encode, \
                frozen_parameters(runner.world_model, runner.discriminator, runner.critic):
            result = runner.imagine(batch, noises=noise)
        self.assertEqual(encode.call_count, 1)
        self.assertEqual(result["task"].shape[0], 3)
        self.assertTrue(bool(torch.isfinite(result["objective"]).all()))

    def test_actor_optimizer_changes_policy_only_without_any_world_or_visual_parameter_gradients(self):
        runner, _, noise = latent_runner_fixture()
        frozen = (runner.world_model, runner.actor.vision, runner.latent_decoder, runner.discriminator, runner.critic)
        before = [cloned_state(module) for module in frozen]
        policy_before = cloned_state(runner.actor.policy)
        self.assertEqual({id(p) for group in runner.opt_actor.param_groups for p in group["params"]},
                         {id(p) for p in runner.actor.policy.parameters()})
        fix_update_noise(runner, noise)
        with patch.object(runner, "_fit_critic", return_value=critic_noop()):
            result = runner.update_actor_critic()
        self.assertEqual(result["actor_update_rejected"], 0.)
        self.assertTrue(any(not torch.equal(value, policy_before[name]) for name, value in runner.actor.policy.state_dict().items()))
        for module, saved in zip(frozen, before):
            for name, value in module.state_dict().items():
                torch.testing.assert_close(value, saved[name], rtol=0, atol=0)
            self.assertTrue(all(parameter.grad is None for parameter in module.parameters()))

    def test_world_model_fit_updates_encoder_decoder_and_physics_in_joint_adam(self):
        runner, batch, _ = latent_runner_fixture()
        runner.replay = populated_latent_replay(runner, batch)
        modules = (runner.world_model, runner.actor.vision, runner.latent_decoder)
        before = [cloned_state(module) for module in modules]
        policy_before = cloned_state(runner.actor.policy)
        self.assertEqual({id(p) for group in runner.opt_model.param_groups for p in group["params"]},
                         {id(p) for module in modules for p in module.parameters()})
        result = runner.fit_model()
        self.assertEqual(result["wm_updates_applied_fraction"], 1.)
        self.assertTrue(torch.isfinite(torch.tensor(result["wm_latent_recon_l1"])))
        for module, saved in zip(modules, before):
            self.assertTrue(any(not torch.equal(value, saved[name]) for name, value in module.state_dict().items()))
        for name, value in runner.actor.policy.state_dict().items():
            torch.testing.assert_close(value, policy_before[name], rtol=0, atol=0)
        self.assertTrue(all(parameter.grad is None for parameter in runner.actor.policy.parameters()))

    def test_latent_prediction_supervision_does_not_update_encoder_or_decoder(self):
        runner, batch, _ = latent_runner_fixture()
        replay = populated_latent_replay(runner, batch)
        windows = replay.sample_windows(3, runner.cfg.wm_window, 0., fields=runner.WM_FIELDS)
        runner.opt_model.zero_grad(set_to_none=True)
        before = cloned_state(runner.actor.vision)
        with patch.object(runner.latent_decoder, "forward", side_effect=AssertionError("latent MSE decoded pixels")):
            values = rollout_latent_model_metrics(runner, windows, update_statistics=True)
            self.assertFalse(values["target_latents"].requires_grad)
            (values["latent_error"] * windows["mask"]).sum().backward()
        self.assertTrue(any(parameter.grad is not None and bool((parameter.grad != 0).any()) for parameter in runner.world_model.parameters()))
        for module in (runner.actor.vision, runner.latent_decoder):
            self.assertTrue(all(parameter.grad is None for parameter in module.parameters()))
        runner.opt_model.step()
        for name, value in runner.actor.vision.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)


class LatentReplayCollectionTest(unittest.TestCase):
    def test_replay_stores_processed_images_and_reencodes_current_weights_at_sampled_starts(self):
        runner, batch, _ = latent_runner_fixture()
        replay = populated_latent_replay(runner, batch)
        for forbidden in ("latent", "z", "latent_ring", "depth_ring", "depth_frame"):
            self.assertNotIn(forbidden, replay.data)
        self.assertEqual(replay.data["actor_history"].shape[-1], 384)
        self.assertEqual(replay.data["actor_history"].dtype, torch.float32)
        self.assertEqual(replay.data["depth_stack"].dtype, torch.float16)
        starts = replay.sample_starts(3, 0., fields=runner.ACTOR_FIELDS)
        proprio, stack = unpack_processed_actor_input(starts["actor_history"], runner.spec)
        self.assertEqual(starts["actor_history"].shape[-1], runner.actor.processed_obs_dim)
        self.assertNotEqual(starts["actor_history"].shape[-1], runner.actor.latent_obs_dim)
        first_z = runner.actor.encode_processed(stack).detach()
        with torch.no_grad():
            runner.actor.vision.projection[-1].bias.add_(.1)
        second_z = runner.actor.encode_processed(stack).detach()
        torch.testing.assert_close(second_z - first_z, torch.full_like(first_z, .1))
        self.assertEqual(proprio.shape, (3, 384))

    def test_collect_labels_pre_reset_successor_stack_with_delay_and_fills_reset_history(self):
        runner, actor_batch, _ = latent_runner_fixture()
        count, spec = runner.env.num_envs, runner.spec
        ring = runner.depth_history.ring.clone()
        for index in range(spec.ring_length):
            ring[:, index, 0] = .1 * (index + 1)
        runner.depth_history.buffer = ring
        transition = latent_transition(runner, actor_batch)
        transition.pop("depth_stack")
        transition.pop("next_depth_stack")
        end = torch.full((count, *spec.frame_shape), .8, dtype=torch.float64)
        end[:, 1] = 1
        reset = torch.full_like(end, .95)
        reset[:, 1] = 1
        transition["next_depth_frame"] = end
        transition["terminated"][0] = True
        transition["timeout"][1] = True
        def step(action):
            transition["action"] = action
            runner.env.transition = transition
        runner.env.step = step
        runner.replay = LatentPerceptiveReplay(8, count, spec=spec)
        runner._train_env_mask = runner.replay.train_env_mask
        expected_successor = processed_depth_stack(advance_depth_ring(ring, end, spec), spec).half()
        with patch("src.wmac_perceptive.latent_training.read_depth_frame", return_value=reset):
            result = runner.collect(1)
        torch.testing.assert_close(runner.replay.data["next_depth_stack"][0], expected_successor, rtol=0, atol=0)
        # With delay=1, the latest output channel is the previous real .3,
        # rather than the successor .8 or the reset episode's .95 image.
        torch.testing.assert_close(expected_successor[:, -2], torch.full_like(expected_successor[:, -2], .3), atol=2e-4, rtol=0)
        torch.testing.assert_close(runner.depth_history.ring[:2], reset[:2, None].expand_as(ring[:2]), rtol=0, atol=0)
        torch.testing.assert_close(runner.depth_history.ring[2], advance_depth_ring(ring, end.float().double(), spec)[2], rtol=0, atol=0)
        self.assertTrue(bool(runner.replay.data["terminated"][0, 0]))
        self.assertTrue(bool(runner.replay.data["timeout"][0, 1]))
        self.assertEqual(result["recovery_fraction"], 0.)
        self.assertEqual(runner.sim_steps, count)

    def test_replay_roundtrip_requires_latent_stack_schema_and_preserves_fp32_proprio(self):
        runner, batch, _ = latent_runner_fixture()
        replay = populated_latent_replay(runner, batch)
        saved = replay.state_dict()
        restored = LatentPerceptiveReplay(8, 3, spec=runner.spec)
        restored.load_state_dict(saved)
        for name, value in replay.data.items():
            torch.testing.assert_close(restored.data[name][:replay.size], value[:replay.size], rtol=0, atol=0)
        bad = copy.deepcopy(saved)
        bad["perceptive_format"] = "fp32-proprio-fp16-depth-v1"
        with self.assertRaises(ValueError):
            restored.load_state_dict(bad)


class LatentRunnerCheckpointTest(unittest.TestCase):
    assert_tree_equal = restore_checks.PerceptiveRunnerRestoreTest.assert_tree_equal

    def test_actual_save_load_restores_decoder_encoder_and_joint_adam_strictly(self):
        source, _, _ = latent_runner_fixture(spec=PerceptionSpec(), adam_steps=1)
        restored, _, _ = latent_runner_fixture(spec=PerceptionSpec(), adam_steps=2)
        source.iteration, source.sim_steps, source.imagination_steps = 17, 101, 303
        with tempfile.TemporaryDirectory() as directory, patch("torch.cuda.is_available", return_value=False):
            path = Path(directory) / "latent.pt"
            source.save(path, export=False)
            payload = torch.load(path, map_location="cpu", weights_only=False)
            self.assertIn("latent_decoder", payload)
            self.assertEqual(payload["config"]["visual_mode"], LATENT_VISUAL_MODE)
            fresh = torch.full_like(restored.current_depth, .35)
            fresh[:, 1] = 1
            with patch("src.wmac_perceptive.runner.read_depth_frame", return_value=fresh):
                restored.load(path)
        for name in (*MODULES, "latent_decoder", *OPTIMIZERS):
            self.assert_tree_equal(getattr(restored, name).state_dict(), payload[name])
        self.assertEqual(restored.iteration, 17)
        self.assertEqual(restored.sim_steps, 101)
        self.assertEqual(restored.imagination_steps, 303)
        self.assertEqual(len(restored.opt_model.state), len(tuple(restored._model_training_modules.parameters())))
        self.assertEqual(len(restored.opt_actor.state), len(tuple(restored.actor.policy.parameters())))

    def test_bad_decoder_or_joint_adam_rejects_before_any_runtime_mutation(self):
        source, _, _ = latent_runner_fixture(spec=PerceptionSpec(), adam_steps=1)
        restored, _, _ = latent_runner_fixture(spec=PerceptionSpec(), adam_steps=2)
        with tempfile.TemporaryDirectory() as directory, patch("torch.cuda.is_available", return_value=False):
            path = Path(directory) / "latent.pt"
            source.save(path, export=False)
            payload = torch.load(path, map_location="cpu", weights_only=False)
            for problem in ("missing_decoder", "decoder_shape", "joint_adam_shape", "joint_adam_nan"):
                bad = copy.deepcopy(payload)
                if problem == "missing_decoder":
                    bad.pop("latent_decoder")
                elif problem == "decoder_shape":
                    key = next(iter(bad["latent_decoder"]))
                    bad["latent_decoder"][key] = torch.zeros(1)
                else:
                    moment = next(reversed(bad["opt_model"]["state"].values()))
                    if problem == "joint_adam_shape":
                        moment["exp_avg"] = torch.zeros(1)
                    else:
                        moment["exp_avg"].flatten()[0] = float("nan")
                invalid = Path(directory) / f"{problem}.pt"
                torch.save(bad, invalid)
                before = latent_snapshot(restored)
                with self.subTest(problem=problem), self.assertRaises(ValueError):
                    restored.load(invalid)
                self.assert_tree_equal(latent_snapshot(restored), before)

    def test_pixel_checkpoint_cannot_silently_change_to_online_latent_protocol(self):
        source, _, _ = latent_runner_fixture(spec=PerceptionSpec())
        source.cfg = replace(source.cfg, visual_mode="pixel_v2")
        source.actor = PerceptiveActor(source.spec).double()
        source.world_model = TerrainPerceptiveWorldModel(source.adapter.terrain, source.spec,
            hidden=source.cfg.model_hidden, dynamics_parameterization=GENCOORD_DYNAMICS).double()
        source.opt_actor = torch.optim.Adam(source.actor.parameters(), lr=.0002)
        source.opt_model = torch.optim.Adam(source.world_model.parameters(), lr=.001)
        restored, _, _ = latent_runner_fixture(spec=PerceptionSpec())
        with tempfile.TemporaryDirectory() as directory, patch("torch.cuda.is_available", return_value=False):
            path = Path(directory) / "pixels.pt"
            source.save(path, export=False)
            before = latent_snapshot(restored)
            with self.assertRaisesRegex(ValueError, "architecture|visual"):
                restored.load(path)
            self.assert_tree_equal(latent_snapshot(restored), before)


if __name__ == "__main__":
    unittest.main()
