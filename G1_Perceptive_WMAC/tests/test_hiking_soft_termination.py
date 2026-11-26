"""CPU checks of coherent Hiking imagination termination on actual G1 FK.

The H32 case repeats one posture with unit rewards to isolate the survival
objective. It is not a terrain rollout or a trained-policy performance check.
"""
from __future__ import annotations

import unittest

import torch

from src.tasks.amp_perceptive.hiking_rewards import hiking_termination_terms
from src.wmac.state import quat_apply, quat_inv
from test_hiking_full_world_model import full_fixture


class HikingSoftTerminationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        runner, batch, _ = full_fixture()
        cls.adapter = runner.adapter
        cls.adapter.hiking_training_version = 2
        cls.state = batch["state"][:1].clone()
        cls.context = batch["context"][:1].clone()
        cls.context[..., 35:38] = 0.
        cls.action = cls.state.new_zeros(1, 29)

    def inputs(self, parameters):
        """Set torso height, root pitch and torso contact force independently."""
        height, pitch, force = parameters.unbind()
        zero = torch.zeros_like(pitch)
        quaternion = torch.stack((torch.cos(pitch / 2), zero,
                                  torch.sin(pitch / 2), zero))[None]
        state = torch.cat((self.state[..., :3], quaternion, self.state[..., 7:]), -1)
        position, _, _, _ = self.adapter.kinematics(state)
        z = state[..., 2] + height - position[..., self.adapter.anchor_id, 2]
        state = torch.cat((state[..., :2], z[..., None], state[..., 3:]), -1)
        auxiliary = torch.cat((self.state.new_zeros(1, 5), force.reshape(1, 1),
                               self.state.new_zeros(1, 7)), -1)
        return state, auxiliary

    def probability(self, parameters):
        state, auxiliary = self.inputs(parameters)
        _, probability = self.adapter.advance_context(state, self.context,
            action=self.action, contact=auxiliary)
        return probability

    def components(self, parameters):
        state, auxiliary = self.inputs(parameters)
        position, quaternion, _, _ = self.adapter.kinematics(state)
        gravity = quat_apply(quat_inv(quaternion[..., self.adapter.anchor_id, :]),
                             state.new_tensor((0., 0., -1.)))
        height = position[..., self.adapter.anchor_id, 2] - self.context[..., 37].clamp_max(0.)
        return (torch.sigmoid((.5 - height) / .025),
                torch.sigmoid((gravity[..., 2] + state.new_tensor(1.).cos()) / .025),
                torch.sigmoid((auxiliary[..., 5] - 1.) / .1))

    def finite_difference(self, function, parameters, index, epsilon=1e-6):
        delta = torch.zeros_like(parameters)
        delta[index] = epsilon
        with torch.no_grad():
            return (function(parameters + delta) - function(parameters - delta)) / (2 * epsilon)

    @staticmethod
    def h32_objective(probability):
        alive = torch.ones_like(probability)
        value = torch.zeros_like(probability)
        for step in range(32):
            value = value + .99 ** step * alive
            alive = alive * (1 - probability)
        return value.sum(), alive

    def test_forward_is_the_existing_hiking_soft_or_at_all_three_thresholds(self):
        for values in ((.49, 0., 0.), (.51, 0., 0.),
                       (.8, .98, 0.), (.8, 1.02, 0.),
                       (.8, 0., .9), (.8, 0., 1.1)):
            with self.subTest(values=values):
                parameters = self.state.new_tensor(values)
                actual = self.probability(parameters)
                height, tilt, contact = self.components(parameters)
                expected = 1 - (1 - height) * (1 - tilt) * (1 - contact)
                torch.testing.assert_close(actual, expected, rtol=0., atol=0.)
                self.assertTrue(bool(((actual > 0.) & (actual < 1.)).all()))

    def test_height_tilt_and_contact_autograd_match_forward_finite_differences(self):
        for values, index in (((.51, 0., 0.), 0),
                              ((.8, 1.02, 0.), 1),
                              ((.8, 0., 1.02), 2)):
            with self.subTest(values=values, derivative_index=index):
                parameters = self.state.new_tensor(values).requires_grad_()
                derivative = torch.autograd.grad(self.probability(parameters).sum(), parameters)[0][index]
                numerical = self.finite_difference(lambda x: self.probability(x).sum(), parameters, index)
                self.assertGreater(float(derivative.abs()), .1)
                torch.testing.assert_close(derivative, numerical, rtol=1e-6, atol=1e-8)

    def test_h32_survival_value_and_gradient_use_the_same_soft_probability(self):
        parameters = self.state.new_tensor((.51, 0., 0.)).requires_grad_()
        probability = self.probability(parameters)
        value, alive = self.h32_objective(probability)
        expected = sum(.99 ** step * (1 - probability) ** step for step in range(32)).sum()
        torch.testing.assert_close(value, expected)
        torch.testing.assert_close(alive, (1 - probability) ** 32)
        self.assertGreater(float(value.detach()), 1.)
        self.assertLess(float(value.detach()), 3.)
        self.assertLess(float(alive.detach().max()), 1e-6)
        derivative = torch.autograd.grad(value, parameters)[0][0]
        numerical = self.finite_difference(lambda x: self.h32_objective(self.probability(x))[0], parameters, 0)
        torch.testing.assert_close(derivative, numerical, rtol=1e-6, atol=1e-7)

    def test_fullgraph_forward_and_backward_match_eager(self):
        parameters = self.state.new_tensor((.51, .98, .95)).requires_grad_()
        expected = self.probability(parameters)
        expected_gradient = torch.autograd.grad(expected.sum(), parameters)[0]
        compiled = torch.compile(self.probability, backend="aot_eager", fullgraph=True)
        actual = compiled(parameters)
        actual_gradient = torch.autograd.grad(actual.sum(), parameters)[0]
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual_gradient, expected_gradient)

    def test_real_hiking_failures_remain_hard_and_keep_their_strict_thresholds(self):
        position = self.state.new_tensor(((0., 0., .5), (0., 0., .499),
                                         (0., 0., .8), (0., 0., .8)))
        gravity = self.state.new_tensor(((0., 0., -1.), (0., 0., -1.),
                                        (0., 0., -1.), (0., 0., -1.)))
        origins = torch.zeros_like(position)
        force = self.state.new_tensor((0., 0., 1., 1.001))
        terms = hiking_termination_terms(position, gravity, origins, force)
        self.assertEqual(terms["root_height"].tolist(), [False, True, False, False])
        self.assertEqual(terms["base_contact"].tolist(), [False, False, False, True])
        for term in terms.values():
            self.assertEqual(term.dtype, torch.bool)
        tilt = self.state.new_tensor((.98, 1.02))
        tilted_gravity = torch.stack((torch.sin(tilt), torch.zeros_like(tilt), -torch.cos(tilt)), -1)
        terms = hiking_termination_terms(position[:2] + self.state.new_tensor((0., 0., 1.)),
            tilted_gravity, origins[:2], force[:2])
        self.assertEqual(terms["bad_orientation"].tolist(), [False, True])

    def test_explicit_legacy_protocol_preserves_old_checkpoint_forward_values(self):
        try:
            self.adapter.hiking_training_version = 1
            for height, expected in ((.49, 1.), (.51, 0.)):
                probability = self.probability(self.state.new_tensor((height, 0., 0.)))
                torch.testing.assert_close(probability, torch.full_like(probability, expected), rtol=0., atol=0.)
        finally:
            self.adapter.hiking_training_version = 2

    def test_v3_uses_worst_normalized_threshold_for_survival_and_critic(self):
        from src.wmac.gencoord import gencoord_features
        from src.wmac_perceptive.hiking_context import normalize_hiking_sim_context

        self.adapter.hiking_training_version = 3
        try:
            # At two simultaneous thresholds the reference max still gives
            # probability 0.5. A probabilistic OR would give at least 0.75.
            at_threshold = self.state.new_tensor((.5, 0., 1.))
            torch.testing.assert_close(self.probability(at_threshold), self.state.new_tensor([.5]))
            for values, index in (((.49, 0., .4), 0), ((.8, 1.01, .3), 1), ((.8, 0., 1.02), 2)):
                parameters = self.state.new_tensor(values).requires_grad_()
                gradient = torch.autograd.grad(self.probability(parameters).sum(), parameters)[0][index]
                numerical = self.finite_difference(self.probability, parameters, index)
                torch.testing.assert_close(gradient, numerical.squeeze(), rtol=1e-6, atol=1e-8)
                self.assertGreater(gradient.abs().item(), .1)
            materials = self.state.new_full((1, 29), 1.6)
            context = torch.cat((self.context, materials), -1)
            context[:, 38:67] = torch.arange(29, dtype=context.dtype)
            command = self.state.new_tensor([[.45, .1, .2]])
            expected = torch.cat((gencoord_features(self.state), command,
                                  normalize_hiking_sim_context(context), self.adapter.terrain.local_patch(self.state)), -1)
            features = self.adapter.critic_features(self.state, self.action, command, context)
            torch.testing.assert_close(features, expected)
            self.assertEqual(features.shape[-1], 68 + 3 + 129 + self.adapter.terrain.patch_dim)
        finally:
            self.adapter.hiking_training_version = 2


if __name__ == "__main__":
    unittest.main()
