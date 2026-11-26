"""CPU integration of perceptive BPTT, terrain grounding and recovery gates.

Uses the real G1 MuJoCo/FK fixture and actual neural modules, without creating
a physical environment or allocating a GPU. These are implementation checks,
not a terrain traversal or get-up success-rate experiment.
"""
from __future__ import annotations

import copy
from dataclasses import replace
from unittest.mock import patch
import unittest

import torch

from rsl_rl.modules.discriminator import Discriminator
from src.amp_features import build_amp_features, G1_FOOT_REFERENCE_OFFSET
from src.wmac.config import WmacConfig
from src.wmac.networks import Critic, RunningNorm, frozen_parameters
from src.wmac.state import quat_apply
from src.wmac_perceptive.config import PerceptiveConfig
from src.wmac_perceptive.models import PerceptiveActor, TerrainPerceptiveWorldModel
from src.wmac_perceptive.perception import PerceptionSpec, pack_actor_input, unpack_actor_input
from src.wmac_perceptive.runner import PerceptiveRunner
from src.wmac_perceptive.state import PerceptiveStateAdapter
from test_wmac_optimization import make_runner, fix_update_noise, FixedReplay


class PlaneOracle:
    patch_dim = 81

    def __init__(self, offset=0., slope=(0., 0.)):
        self.offset, self.slope = offset, slope

    def height_at(self, xy):
        return xy[..., 0]*self.slope[0]+xy[..., 1]*self.slope[1]+self.offset

    def local_patch(self, state):
        return (self.height_at(state[..., :2])-state[..., 2]).unsqueeze(-1).expand(*state.shape[:-1], self.patch_dim)

    def edge_distance_at(self, xy):
        return xy.new_full(xy.shape[:-1], 1.)


def tiny_spec():
    return PerceptionSpec(height=4, width=4, ring_length=3, frames=2, stride=1, delay=1,
                          crop_top=0, crop_bottom=0, crop_left=0, crop_right=0)


def runner_fixture(batch_size=3, horizon=3):
    old, batch, noise = make_runner(batch_size=batch_size, horizon=horizon)
    runner = object.__new__(PerceptiveRunner)
    runner.__dict__.update(old.__dict__)
    runner.iteration = 0
    runner.cfg = replace(PerceptiveConfig.defaults(), perception_training_version=1,
        dynamics_parameterization="independent_residual_v1",
        motion_scope="joint_recovery_v1",recovery_flat_fraction=.25,
        recovery_fraction=.5,expert_recovery_fraction=.5,
        device="cpu", replay_device="cpu", precision="float32",
        compile=False, num_envs=batch_size, actor_batch_size=batch_size, actor_microbatch_size=0,
        wm_batch_size=batch_size, imagination_horizon=horizon, wm_window=2, warmup_steps=8, replay_steps=8,
        model_hidden=(16,), critic_hidden=(16, 8), discriminator_hidden=(12, 6),
        critic_epochs=1, critic_minibatches=1, reward_chunk_size=4,
        max_grad_norm=1e6, actor_grad_value_clip=1e6, max_action_shift=100., gamma=.5, gae_lambda=.5,
        validation_min_walk_samples=1, validation_min_recovery_samples=1)
    runner.cfg.depth_noise_std_m=0.
    runner.cfg.depth_dropout_probability=0.
    runner.cfg.validate()
    runner.spec = tiny_spec()
    runner.adapter.__class__ = PerceptiveStateAdapter
    runner.adapter.terrain = PlaneOracle()
    runner.adapter.recovery_counter_semantics = "hard_forward_v2"
    runner.adapter.amp_feature_schema = "grounded_v2"
    runner.adapter.reward_cfg["track_root_height"].params["upright_power"] = 2.
    runner.actor = PerceptiveActor(runner.spec).double()
    runner.world_model = TerrainPerceptiveWorldModel(runner.adapter.terrain, runner.spec,
                                                    hidden=runner.cfg.model_hidden).double()
    with torch.no_grad():
        # The zero-residual initializer would hide some action→physical-state
        # derivatives. A small learned residual exercises the actual BPTT path.
        runner.world_model.net[-1].weight.normal_(0., .002)
    runner.critic = Critic(217, hidden=runner.cfg.critic_hidden).double()
    runner.discriminator = Discriminator(406, .1, [12, 6], "cpu", .75).double()
    runner.amp_norm = RunningNorm(203).double()
    with torch.no_grad():
        runner.discriminator.amp_linear.weight.mul_(.1)
        runner.discriminator.amp_linear.bias.zero_()
    runner.opt_actor = torch.optim.SGD(runner.actor.parameters(), lr=.002)
    runner.opt_model = torch.optim.Adam(runner.world_model.parameters(), lr=1e-3)
    runner.opt_critic = torch.optim.Adam(runner.critic.parameters(), lr=1e-3)
    runner.opt_disc = torch.optim.Adam(runner.discriminator.parameters(), lr=1e-3)
    runner._value_fn = runner.critic.forward
    depth = torch.full((batch_size, *runner.spec.frame_shape), .2, dtype=torch.float64)
    depth[:, 1].fill_(1.)
    ring = depth[:, None].expand(batch_size, *runner.spec.ring_shape).clone()
    batch = dict(batch)
    batch.update(actor_history=pack_actor_input(batch["actor_history"], ring, runner.spec),
                 depth_frame=depth, relative_height=batch["state"][:, 2].clone(),
                 navigation_target=batch["state"][:, :3].clone()+torch.tensor([2., .5, 0.], dtype=torch.float64),
                 desired_speed=torch.full((batch_size,), .5, dtype=torch.float64))
    runner.replay = FixedReplay(batch)
    return runner, batch, noise


def cloned_state(module):
    return {key: value.detach().clone() for key, value in module.state_dict().items()}


def critic_noop():
    return dict(critic_loss=0., real_critic_loss=0., critic_grad_norm=0., critic_updates_applied_fraction=0.)


class PerceptiveRunnerTest(unittest.TestCase):
    def test_imagination_applies_the_same_outer_action_clip_as_real_collection(self):
        runner,batch,noise=runner_fixture()
        runner.cfg.action_limit=.02
        with torch.no_grad():
            for expert in runner.actor.experts:
                expert[-1].weight.zero_();expert[-1].bias.fill_(.4)
        self.assertGreater(float(runner.actor(batch['actor_history']).abs().max()),.02)
        actions=[]
        original=runner.world_model.step_with_perception
        def record(state,action,*args):
            actions.append(action.detach().clone())
            return original(state,action,*args)
        with patch.object(runner.world_model,'step_with_perception',side_effect=record):
            output=runner.imagine(batch,noises=noise)
        for action in actions:
            torch.testing.assert_close(action,torch.full_like(action,.02),rtol=0,atol=0)
        torch.testing.assert_close(output['first_action'],actions[0],rtol=0,atol=0)

    def test_predicted_future_view_reaches_delayed_actor_and_has_direct_action_gradient(self):
        runner, batch, noise = runner_fixture()
        predicted, inputs, actions = [], [], []
        original = runner.world_model.step_with_perception
        def capture_model(*args):
            state, contact, frame = original(*args)
            predicted.append(frame)
            return state, contact, frame
        def capture_actor(module, args, output):
            inputs.append(args[0]); actions.append(output)
        handle = runner.actor.register_forward_hook(capture_actor)
        try:
            with patch.object(runner.world_model, "step_with_perception", side_effect=capture_model), \
                 frozen_parameters(runner.world_model, runner.discriminator, runner.critic):
                result = runner.imagine(batch, noises=noise)
        finally:
            handle.remove()
        self.assertEqual(len(predicted), 3)
        self.assertEqual(len(inputs), 3)
        initial_ring = unpack_actor_input(batch["actor_history"], runner.spec)[1]
        third_ring = unpack_actor_input(inputs[2], runner.spec)[1]
        # With a one-step camera delay, the first predicted image is selected
        # by the third Actor call. A stale first-image implementation fails.
        torch.testing.assert_close(third_ring[:, 1], predicted[0], rtol=0, atol=0)
        self.assertGreater(float((third_ring[:, 1]-initial_ring[:, 1]).abs().max()), .01)
        partial = torch.autograd.grad(actions[2].sum(), inputs[2], retain_graph=True)[0]
        visual_partial = partial[:, runner.spec.proprio_dim:].reshape(-1, *runner.spec.ring_shape)[:, 1]
        self.assertTrue(bool(torch.isfinite(visual_partial).all()))
        self.assertGreater(float(visual_partial.abs().sum()), 0.)
        task_image_grad = torch.autograd.grad(result["objective"].sum(), predicted[0], allow_unused=True)[0]
        self.assertIsNotNone(task_image_grad)
        self.assertTrue(bool(torch.isfinite(task_image_grad).all()))
        self.assertGreater(float(task_image_grad.abs().sum()), 0.)

    def test_actual_actor_update_trains_cnn_router_and_all_experts_without_frozen_parameter_leaks(self):
        runner, _, noise = runner_fixture()
        frozen = (runner.world_model, runner.discriminator, runner.critic)
        frozen_before = [cloned_state(module) for module in frozen]
        actor_before = cloned_state(runner.actor)
        fix_update_noise(runner, noise)
        with patch.object(runner, "_fit_critic", return_value=critic_noop()):
            result = runner.update_actor_critic()
        self.assertEqual(result["actor_update_rejected"], 0.)
        self.assertEqual(result["actor_router_aux_loss"], 0.)
        for module in (runner.actor.vision, runner.actor.router, *runner.actor.experts):
            gradients = [p.grad for p in module.parameters()]
            self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in gradients))
            self.assertGreater(sum(float(g.abs().sum()) for g in gradients), 0.)
        self.assertTrue(any(not torch.equal(value, actor_before[key]) for key, value in runner.actor.state_dict().items()))
        for module, before in zip(frozen, frozen_before):
            for key, value in module.state_dict().items():
                torch.testing.assert_close(value, before[key], rtol=0, atol=0)
            self.assertTrue(all(p.grad is None for p in module.parameters()))

    def test_critic_calibration_does_not_change_actor_or_sensor_encoder(self):
        runner, _, noise = runner_fixture()
        actor_before, critic_before = cloned_state(runner.actor), cloned_state(runner.critic)
        fix_update_noise(runner, noise)
        result = runner.update_actor_critic(update_actor=False)
        for key, value in runner.actor.state_dict().items():
            torch.testing.assert_close(value, actor_before[key], rtol=0, atol=0)
        self.assertTrue(all(p.grad is None for p in runner.actor.parameters()))
        self.assertEqual(result["actor_router_aux_loss"], 0.)
        self.assertTrue(any(not torch.equal(value, critic_before[key]) for key, value in runner.critic.state_dict().items()))

    def test_imagination_recomputes_navigation_command_for_its_target(self):
        outputs = []
        for direction in (-1., 1.):
            runner, batch, noise = runner_fixture()
            batch["navigation_target"][:, 1] = direction*2.
            commands = []
            original = runner.adapter.actor_frame
            def observe(state, action, command, context, **kwargs):
                commands.append(command.detach().clone())
                return original(state, action, command, context, **kwargs)
            with patch.object(runner.adapter, "actor_frame", side_effect=observe):
                result = runner.imagine(batch, noises=noise)
            self.assertTrue(bool((commands[0][:, 2]*direction > 0).all()))
            outputs.append(commands[0])
            self.assertTrue(bool(torch.isfinite(result["objective"]).all()))
        self.assertGreater(float((outputs[0]-outputs[1]).abs().max()), .1)

    def test_raising_robot_and_terrain_together_preserves_recovery_amp_value_features_and_rewards(self):
        runner, batch, _ = runner_fixture()
        state = batch["state"].clone(); state[0, 2] = .2
        context = batch["context"].clone(); context[:, 33] = 1.; context[:, 34] = 5.
        previous, command = batch["previous_action"], batch["command"]
        action, contact = torch.zeros_like(previous), state.new_zeros(len(state), 2)
        outputs = []
        for lift in (0., 2.):
            runner.adapter.terrain = PlaneOracle(offset=lift)
            lifted = state.clone(); lifted[:, 2] += lift
            advanced, terminal = runner.adapter.advance_context(lifted, context)
            outputs.append((runner.adapter.relative_height(lifted), advanced, terminal,
                            runner.adapter.amp_features(lifted),
                            runner.adapter.critic_features(lifted, previous, command, context),
                            runner.adapter.task_reward(lifted, lifted, action, previous, command, advanced, terminal, contact)))
        for initial, raised in zip(outputs[0], outputs[1]):
            torch.testing.assert_close(raised, initial, rtol=1e-12, atol=1e-12)
        self.assertEqual(float(outputs[1][1][0, 34]), 6.)
        self.assertAlmostEqual(float(outputs[1][0][0]), .2, places=12)
        self.assertEqual(outputs[1][3].shape[-1], 203)
        self.assertEqual(outputs[1][4].shape[-1], 217)

    def test_amp_sole_clearance_uses_each_foot_local_ground(self):
        runner, batch, _ = runner_fixture()
        runner.adapter.terrain = PlaneOracle(offset=1., slope=(.2, -.1))
        state = batch["state"].clone(); state[:, 2] += 1.
        p, q, v, w = runner.adapter.kinematics(state)
        raw = build_amp_features(p, q, v, w, body_ids=runner.adapter.amp_ids,
            anchor_id=runner.adapter.anchor_id, pelvis_id=runner.adapter.pelvis_id,
            foot_ids=runner.adapter.foot_ids, schema="grounded_v2")
        offset = p.new_tensor((G1_FOOT_REFERENCE_OFFSET, G1_FOOT_REFERENCE_OFFSET))
        sole = p[:, runner.adapter.foot_ids]+quat_apply(q[:, runner.adapter.foot_ids], offset)
        ground = runner.adapter.terrain.height_at(sole[..., :2])
        actual = runner.adapter.amp_features(state)
        torch.testing.assert_close(actual[:, 199:201], raw[:, 199:201]-ground, rtol=0, atol=1e-12)
        self.assertGreater(float((ground[:, 0]-ground[:, 1]).abs().max()), 1e-3)

    def test_actor_gate_requires_finite_good_dynamics_and_depth_in_both_motion_phases(self):
        runner, _, _ = runner_fixture()
        valid = dict(validation_model_error=.01, validation_contact_error=.01,
                     validation_walk_error=.01, validation_recovery_error=.01,
                     validation_depth_mask_error=.01,
                     validation_walk_depth_rmse=.01, validation_recovery_depth_rmse=.01,
                     validation_walk_samples=16., validation_recovery_samples=16.)
        row = dict(wm_gate_error=.01, wm_updates_applied_fraction=1.)
        runner.validation = valid.copy()
        self.assertTrue(runner._actor_gate(row))
        for field, value in (("validation_walk_samples", 0.), ("validation_recovery_samples", 0.),
                             ("validation_walk_depth_rmse", 10.), ("validation_recovery_depth_rmse", float("inf")),
                             ("validation_depth_mask_error", float("nan")),
                             ("validation_walk_error", float("inf")), ("validation_recovery_error", float("inf")),
                             ("validation_walk_samples", float("nan"))):
            runner.validation = {**valid, field: value}
            with self.subTest(field=field, value=value):
                self.assertFalse(runner._actor_gate(row))

    def test_perceptive_config_excludes_recovery_and_flat_configuration_stays_isolated(self):
        cfg = PerceptiveConfig.defaults(); cfg.validate()
        self.assertEqual(cfg.recovery_counter_semantics, "hard_forward_v2")
        self.assertEqual(cfg.recovery_height_reward, "upright_v2")
        self.assertEqual(cfg.utd, .4)
        self.assertEqual(cfg.motion_scope,"forward_walk_turn_v1")
        self.assertEqual(cfg.recovery_flat_fraction,0.)
        self.assertEqual(cfg.recovery_fraction,0.)
        self.assertEqual(cfg.expert_recovery_fraction,0.)
        self.assertEqual(cfg.validation_min_recovery_samples,0)
        old = WmacConfig.for_profile("ppo_aligned")
        self.assertEqual(old.actor_type, "mlp")
        old.task = cfg.task
        with self.assertRaises(ValueError): old.validate()
        with self.assertRaises(ValueError): replace(cfg, recovery_counter_semantics="soft_v1").validate()


if __name__ == "__main__":
    unittest.main()
