"""Reuse step-local FK and fixed Critic data without changing the learner."""
from __future__ import annotations

import unittest
from unittest.mock import patch

import torch
from torch import nn

from src.wmac.networks import frozen_parameters
from src.wmac_perceptive.hiking_observations import pack_hiking_proprio_history
from src.wmac_perceptive.macro_execution import ActorCriticMacroExecution
from src.wmac_perceptive.reference_policy_step import HikingWorldModelStep
from test_hiking_training_v2 import v2_reference_fixture
from test_perceptive_macro_execution import (
    g1_runner, independent_policy_update, assert_optimizer_equal,
)
from test_reference_policy_step import step_inputs, scalar_objective


class OriginalHikingStep(nn.Module):
    """Independent pre-reuse step, including its repeated FK/command calls."""
    def __init__(self, runner):
        super().__init__()
        self.actor, self.world_model = runner.actor, runner.world_model
        object.__setattr__(self, "adapter", runner.adapter)
        self.action_limit = runner.cfg.action_limit

    def forward(self, state, context, history, latent, target, desired_speed, noise,
                standing_mask=None, random_yaw_rate=None, random_velocity_mask=None, *, previous_action):
        options = dict(standing_mask=standing_mask, random_yaw_rate=random_yaw_rate,
                       random_velocity_mask=random_velocity_mask)
        command = self.adapter.navigation_command(state, target, desired_speed, **options)
        action = self.actor.forward_latent(pack_hiking_proprio_history(history), latent).clamp(
            -self.action_limit, self.action_limit)
        next_state, contact, next_latent = self.world_model.step_with_latent(state, action, context, latent)
        next_context, terminated = self.adapter.advance_context(next_state, context, action=action, contact=contact)
        next_command = self.adapter.navigation_command(next_state, target, desired_speed, **options)
        observation = self.adapter.actor_frame(next_state, action, next_command, next_context)+noise
        next_history = torch.cat((history[:, 1:], observation[:, None]), 1)
        kinematics = self.adapter.analyze_state(next_state)
        task = self.adapter.task_reward(state, next_state, action, previous_action, command,
                                       next_context, terminated, contact, kinematics=kinematics)
        amp_next = self.adapter.amp_features(next_state, kinematics=kinematics)
        next_command = self.adapter.navigation_command(next_state, target, desired_speed, **options)
        critic = self.adapter.critic_features(next_state, action, next_command, next_context)
        return (next_state, next_context, next_history, next_latent, action, command, contact,
                terminated, task, amp_next, critic)


class ImaginationReuseTest(unittest.TestCase):
    def test_step_reuses_only_current_successor_fk_and_matches_all_input_and_actor_gradients(self):
        torch.manual_seed(20261117)
        runner, batch, noise = v2_reference_fixture()
        old_inputs, new_inputs = step_inputs(runner, batch, noise[0]), step_inputs(runner, batch, noise[0])
        for inputs in (old_inputs, new_inputs):
            inputs["previous_action"] = inputs["previous_action"].clone().requires_grad_(True)
        kernel = HikingWorldModelStep(runner)
        parameters = tuple(runner.actor.parameters())
        names = ("state", "context", "history", "latent", "previous_action")
        with frozen_parameters(runner.world_model, runner.discriminator, runner.critic):
            with patch.object(runner.adapter, "analyze_state", wraps=runner.adapter.analyze_state) as fk:
                expected = OriginalHikingStep(runner)(**old_inputs)
                self.assertEqual(fk.call_count, 6)
            with patch.object(runner.adapter, "analyze_state", wraps=runner.adapter.analyze_state) as fk:
                actual = kernel(**new_inputs)
                self.assertEqual(fk.call_count, 2)
            expected_grad = torch.autograd.grad(scalar_objective(expected),
                tuple(old_inputs[name] for name in names)+parameters, allow_unused=True)
            actual_grad = torch.autograd.grad(scalar_objective(actual),
                tuple(new_inputs[name] for name in names)+parameters, allow_unused=True, retain_graph=True)
            action_latent_grad, = torch.autograd.grad(actual[4].sum(), new_inputs["latent"],
                                                     allow_unused=True, retain_graph=True)
            latent_recurrence_grad, = torch.autograd.grad(actual[3].sum(), new_inputs["latent"])
        for value, truth in zip(actual, expected):
            torch.testing.assert_close(value, truth, rtol=0, atol=0)
        for value, truth in zip(actual_grad, expected_grad):
            if truth is None:self.assertIsNone(value)
            else:torch.testing.assert_close(value, truth, rtol=2e-9, atol=2e-11)
        self.assertTrue(action_latent_grad is None or bool((action_latent_grad == 0).all()))
        self.assertGreater(float(latent_recurrence_grad.abs().sum()), 0.)
        self.assertFalse(hasattr(runner.adapter, "_cached_kinematics"))

    def test_aot_step_and_legacy_defaults_keep_their_gradient_boundaries(self):
        runner, batch, noise = v2_reference_fixture()
        kernel = HikingWorldModelStep(runner)
        compiled = torch.compile(kernel, backend="aot_eager", fullgraph=True, dynamic=False)
        a, b = step_inputs(runner, batch, noise[0]), step_inputs(runner, batch, noise[0])
        names = ("state", "context", "history", "latent")
        with frozen_parameters(runner.world_model, runner.discriminator, runner.critic):
            actual, expected = compiled(**a), OriginalHikingStep(runner)(**b)
            ga = torch.autograd.grad(scalar_objective(actual), tuple(a[name] for name in names), allow_unused=True)
            gb = torch.autograd.grad(scalar_objective(expected), tuple(b[name] for name in names), allow_unused=True)
        for value, truth in zip(actual, expected):
            torch.testing.assert_close(value, truth, rtol=2e-9, atol=2e-11)
        for value, truth in zip(ga, gb):
            if truth is None:self.assertIsNone(value)
            else:torch.testing.assert_close(value, truth, rtol=2e-8, atol=2e-10)
        runner.cfg.hiking_training_version = 1
        legacy = HikingWorldModelStep(runner)
        self.assertFalse(legacy.share_hiking_kinematics)
        with patch.object(runner.adapter, "analyze_state", wraps=runner.adapter.analyze_state) as fk:
            legacy(**step_inputs(runner, batch, noise[0]))
            self.assertEqual(fk.call_count, 6)

    def test_H32_actor_and_twenty_critic_updates_match_original_fk_and_gather_order(self):
        runner, batch, noise = g1_runner()
        reference, _, _ = g1_runner()
        for name in ("actor", "world_model", "critic", "discriminator", "amp_norm"):
            getattr(reference, name).load_state_dict(getattr(runner, name).state_dict())
        for value in (runner, reference):
            value.cfg.imagination_horizon = 32
        noise = noise.repeat((32+len(noise)-1)//len(noise), 1, 1)[:32]
        reference._reference_step_fn = OriginalHikingStep(reference)
        execution = ActorCriticMacroExecution(runner, cuda_graphs=False)
        inputs = []
        loss_fn = runner._latent_critic_loss_fn
        def critic_loss(features, target):
            inputs.append((features.data_ptr(), target.data_ptr(), features.detach().clone(), target.detach().clone()))
            return loss_fn(features, target)
        runner._latent_critic_loss_fn = critic_loss
        rng = torch.get_rng_state().clone()
        expected, expected_norm, expected_critic = independent_policy_update(reference, batch, noise)
        expected_rng = torch.get_rng_state().clone()
        torch.set_rng_state(rng)
        actual, norm, accepted, critic = execution.step(batch, noise)
        self.assertTrue(bool(accepted))
        self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))
        for name in expected:
            torch.testing.assert_close(actual[name], expected[name], rtol=2e-9, atol=2e-11)
        torch.testing.assert_close(norm, expected_norm, rtol=2e-9, atol=2e-11)
        torch.testing.assert_close(critic, critic.new_tensor(tuple(expected_critic.values())), rtol=2e-9, atol=2e-11)
        for name in ("actor", "critic"):
            for value, truth in zip(getattr(runner, name).parameters(), getattr(reference, name).parameters()):
                torch.testing.assert_close(value, truth, rtol=2e-9, atol=2e-11)
        assert_optimizer_equal(self, runner.opt_actor, reference.opt_actor, rtol=2e-8, atol=2e-10)
        assert_optimizer_equal(self, runner.opt_critic, reference.opt_critic, rtol=2e-8, atol=2e-10)
        self.assertEqual(len(inputs), 20)
        for index, value in enumerate(inputs):
            original = inputs[index % 4]
            self.assertEqual(value[:2], original[:2])
            for observed, truth in zip(value[2:], original[2:]):
                torch.testing.assert_close(observed, truth, rtol=0, atol=0)
        for optimizer, count in ((runner.opt_actor, 1), (runner.opt_critic, 20)):
            self.assertTrue(all(float(value["step"]) == count for value in optimizer.state.values() if "step" in value))
        for module in (runner.world_model, runner.discriminator, runner.critic, runner.actor.vision):
            self.assertTrue(all(parameter.grad is None for parameter in module.parameters()))


if __name__ == "__main__":
    unittest.main()
