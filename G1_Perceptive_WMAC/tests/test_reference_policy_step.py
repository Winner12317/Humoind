"""Single-step reward, observation and gradient parity on the real G1 FK fixture.

The oracle retains the existing trajectory reward implementation for a single
transition. No simulator, data downloads or training run is started.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from rsl_rl.modules.discriminator import Discriminator
from src.tasks.amp_perceptive.hiking_geometry import HikingEdgeCylinders
from src.wmac.networks import frozen_parameters
from src.wmac_perceptive.compiled_imagination import LatentImaginedStep, LatentTrajectoryTerms
from src.wmac_perceptive.hiking_observations import unpack_hiking_proprio_history
from src.wmac_perceptive.latent_perception import unpack_processed_actor_input
from src.wmac_perceptive.reference_policy_step import HikingWorldModelStep
from src.wmac_perceptive.reference_training import reference_value_ceiling
from src.wmac_perceptive.runner import PerceptiveRunner
from src.wmac_perceptive.safety import hiking_foot_edge_cost
from test_hiking_full_world_model import full_fixture


def step_inputs(runner, batch, noise):
    proprio, stack = unpack_processed_actor_input(batch["actor_history"], runner.spec)
    with torch.no_grad():
        latent = runner.actor.encode_processed(stack)
    return dict(state=batch["state"].clone().requires_grad_(True),
        context=batch["context"].clone().requires_grad_(True),
        history=unpack_hiking_proprio_history(proprio).clone().requires_grad_(True),
        latent=latent.detach().requires_grad_(True), target=batch["navigation_target"],
        desired_speed=batch["desired_speed"], noise=noise,
        standing_mask=batch["command_standing"], random_yaw_rate=batch["random_yaw_rate"],
        random_velocity_mask=batch["random_velocity_mask"], previous_action=batch["previous_action"])


def trajectory_oracle(runner, inputs):
    """Existing whole-trajectory computation specialized to one transition."""
    result = LatentImaginedStep(runner)(**{key: value for key, value in inputs.items()
        if key != "previous_action"})
    state, context, _, _, action, command, contact, terminated = result
    tasks, amp = LatentTrajectoryTerms(runner)(torch.stack((inputs["state"], state)),
        action[None], inputs["previous_action"][None], command[None],
        torch.stack((inputs["context"], context)), terminated[None], contact[None])
    next_command = runner.adapter.navigation_command(state, inputs["target"], inputs["desired_speed"],
        standing_mask=inputs["standing_mask"], random_yaw_rate=inputs["random_yaw_rate"],
        random_velocity_mask=inputs["random_velocity_mask"])
    critic = runner.adapter.critic_features(state, action, next_command, context)
    return (*result, tasks[0], amp[1], critic)


def scalar_objective(outputs):
    # Include each path returned to the rollout, including reward and bootstrap
    # features, so parity checks detect an accidental detach in the new boundary.
    return sum((index + 1) * value.square().mean() for index, value in enumerate(outputs))


def reference_loop_fixture():
    """Real FK/edge geometry with the task's ten-frame Hiking AMP adapter.

    The small CPU model/batch is a computation oracle, not a training preset.
    Official SuperMimic has no AMP; this fixture verifies that reorganizing its
    policy loop preserves the existing Hiking task and style objectives.
    """
    runner, batch, noises = full_fixture()
    runner.cfg.execution_profile = "supermimic"
    runner.cfg.training_schedule = "supermimic_v1"
    runner.cfg.amp_feature_schema = "hiking_v1"
    runner.adapter.amp_feature_schema = "hiking_v1"
    runner.adapter.amp_dim = 67
    torso_id = int(runner.adapter.robot.indexing.body_ids[runner.adapter.anchor_id])
    runner.adapter._hiking_torso_com_offset = batch["state"].new_tensor(
        runner.adapter.env.sim.mj_model.body_ipos[torso_id].copy())
    runner.discriminator = Discriminator(670, .25, [12, 6], "cpu").double()
    runner.amp_norm = nn.Identity()
    runner._critic_value_ceiling = reference_value_ceiling(runner.adapter, runner.cfg)
    batch["amp_state_history"] = batch["state"][:, None].expand(-1, 10, -1).clone()
    # Real predecessors differ from the initial state; the test must exercise
    # the ten-frame shift, rather than a repeated-frame AMP special case.
    offsets = torch.arange(-9, 1, dtype=batch["state"].dtype)
    batch["amp_state_history"][..., 7] += .002 * offsets
    batch["amp_state_history"][..., 42] += .01 * offsets

    position, _, _, _ = runner.adapter.kinematics(batch["state"])
    feet = position[:, runner.adapter.foot_ids].detach().numpy().reshape(-1, 3)
    edges = []
    for point in feet:
        center = point + np.array((0., .011, -.016))
        edges.append((center + (-.2, 0., 0.), center + (.3, 0., 0.)))
    runner.adapter.terrain.hiking_edges = HikingEdgeCylinders(np.asarray(edges))
    runner._latent_step_fn = LatentImaginedStep(runner)
    runner._latent_trajectory_fn = LatentTrajectoryTerms(runner)
    runner._reference_step_fn = HikingWorldModelStep(runner)
    return runner, batch, noises


class ReferencePolicyStepTest(unittest.TestCase):
    def test_reference_policy_loop_preserves_hiking_trajectory_values_and_actor_gradients(self):
        torch.manual_seed(20261115)
        runner, batch, noises = reference_loop_fixture()
        self.assertTrue(runner.uses_reference_policy_step)
        parameters = tuple(runner.actor.parameters())
        with frozen_parameters(runner.world_model, runner.discriminator, runner.critic):
            # Restore the published whole-trajectory formulas as an independent
            # oracle. This does not call the reverted per-step wrapper.
            with patch.object(PerceptiveRunner, "uses_reference_policy_step", property(lambda self: False)):
                expected = runner.imagine(batch, noises=noises)
                expected_gradients = torch.autograd.grad(expected["objective"].sum(), parameters,
                    allow_unused=True)
            penetration = hiking_foot_edge_cost(runner.adapter.kinematics(batch["state"]),
                runner.adapter.foot_ids, runner.adapter.terrain)
            self.assertTrue(bool((penetration > 0.).all()))
            self.assertGreater(float(expected["style"].abs().sum()), 0.)
            self.assertFalse(expected["x"].requires_grad)
            for backend in (None, "aot_eager"):
                with self.subTest(backend=backend or "eager"):
                    kernel = HikingWorldModelStep(runner)
                    runner._reference_step_fn = kernel if backend is None else torch.compile(
                        kernel, backend=backend, fullgraph=True, dynamic=False)
                    actual = runner.imagine(batch, noises=noises)
                    actual_gradients = torch.autograd.grad(actual["objective"].sum(), parameters,
                        allow_unused=True)
                    self.assertEqual(actual.keys(), expected.keys())
                    for name in expected:
                        torch.testing.assert_close(actual[name], expected[name], rtol=2e-8, atol=2e-10)
                    compared = 0
                    for observed, oracle in zip(actual_gradients, expected_gradients):
                        if oracle is None:
                            self.assertIsNone(observed)
                        else:
                            self.assertTrue(torch.isfinite(observed).all())
                            torch.testing.assert_close(observed, oracle, rtol=2e-7, atol=2e-9)
                            compared += 1
                    self.assertGreater(compared, 0)
                    self.assertFalse(actual["x"].requires_grad)
                    self.assertFalse(actual["targets"].requires_grad)
                    self.assertTrue(actual["objective"].requires_grad)

    def test_single_step_matches_trajectory_oracle_values_and_input_actor_gradients(self):
        torch.manual_seed(20261113)
        runner, batch, noises = full_fixture()
        expected_inputs = step_inputs(runner, batch, noises[0])
        actual_inputs = step_inputs(runner, batch, noises[0])
        parameters = tuple(runner.actor.parameters())
        input_names = ("state", "context", "history", "latent")
        with frozen_parameters(runner.world_model, runner.discriminator, runner.critic):
            expected = trajectory_oracle(runner, expected_inputs)
            expected_gradients = torch.autograd.grad(scalar_objective(expected),
                tuple(expected_inputs[name] for name in input_names) + parameters, allow_unused=True)
            actual = HikingWorldModelStep(runner)(**actual_inputs)
            actual_gradients = torch.autograd.grad(scalar_objective(actual),
                tuple(actual_inputs[name] for name in input_names) + parameters, allow_unused=True)
        self.assertEqual(len(actual), 11)
        for observed, oracle in zip(actual, expected):
            torch.testing.assert_close(observed, oracle, rtol=1e-10, atol=1e-11)
        for observed, oracle in zip(actual_gradients, expected_gradients):
            if oracle is None:
                self.assertIsNone(observed)
            else:
                self.assertTrue(torch.isfinite(observed).all())
                torch.testing.assert_close(observed, oracle, rtol=1e-8, atol=1e-10)

    def test_aot_step_matches_eager_and_keeps_instinct_visual_gradient_boundary(self):
        torch.manual_seed(20261114)
        runner, batch, noises = full_fixture()
        eager_inputs = step_inputs(runner, batch, noises[0])
        aot_inputs = step_inputs(runner, batch, noises[0])
        kernel = HikingWorldModelStep(runner)
        compiled = torch.compile(kernel, backend="aot_eager", fullgraph=True, dynamic=False)
        with frozen_parameters(runner.world_model, runner.discriminator, runner.critic), \
                patch.object(runner.latent_decoder, "forward", side_effect=AssertionError("decoder entered step")):
            eager = kernel(**eager_inputs)
            actual = compiled(**aot_inputs)
            for observed, oracle in zip(actual, eager):
                torch.testing.assert_close(observed, oracle, rtol=1e-8, atol=1e-10)
            parameters = tuple(runner.actor.parameters())
            names = ("state", "context", "history", "latent")
            expected_gradients = torch.autograd.grad(scalar_objective(eager),
                tuple(eager_inputs[name] for name in names) + parameters,
                allow_unused=True, retain_graph=True)
            actual_gradients = torch.autograd.grad(scalar_objective(actual),
                tuple(aot_inputs[name] for name in names) + parameters,
                allow_unused=True, retain_graph=True)
            for observed, oracle in zip(actual_gradients, expected_gradients):
                if oracle is None:
                    self.assertIsNone(observed)
                else:
                    self.assertTrue(torch.isfinite(observed).all())
                    torch.testing.assert_close(observed, oracle, rtol=1e-7, atol=1e-9)
            # Instinct's policy reads detached z; WM's predicted z retains its
            # derivative for the WM rollout. The step must preserve both rules.
            action_gradient, = torch.autograd.grad(actual[4].sum(), aot_inputs["latent"],
                allow_unused=True, retain_graph=True)
            latent_gradient, = torch.autograd.grad(actual[3].sum(), aot_inputs["latent"], retain_graph=True)
            self.assertTrue(action_gradient is None or bool((action_gradient == 0).all()))
            self.assertGreater(float(latent_gradient.abs().sum()), 0.)
            self.assertTrue(actual[10].requires_grad)
            state_gradient, = torch.autograd.grad(actual[10].sum(), aot_inputs["state"])
            self.assertTrue(torch.isfinite(state_gradient).all())
            self.assertGreater(float(state_gradient.abs().sum()), 0.)


if __name__ == "__main__":
    unittest.main()
