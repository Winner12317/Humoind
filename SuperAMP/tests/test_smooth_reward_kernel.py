"""Smooth reward arithmetic, genuine derivatives and actuator-contract checks."""
import copy
from types import SimpleNamespace
import unittest

import torch

from mjlab.actuator import ActuatorCmd
from src.assets.robots.unitree_g1.unitree_actuators import UnitreeActuator
from src.wmac.smooth_rewards import SmoothRewardTerms, TERM_NAMES


def kernel(dtype=torch.float64):
    values = dict(action_scale=.4, action_offset=.1, kp=3., kd=.2,
                  effort_limit=10., y1=8., y2=9., x1=5., x2=10.,
                  friction_static=.1, friction_dynamic=.02, activation_velocity=.5)
    return SmoothRewardTerms(values).to(dtype=dtype)


def inputs():
    generator = torch.Generator().manual_seed(91)
    state = torch.randn(2, 71, dtype=torch.float64, generator=generator) * .1
    state[:, 42:71] += .4
    following = state + .03
    action = torch.randn(2, 29, dtype=torch.float64, generator=generator) * .1
    previous, previous_previous = action + .02, action - .03
    context = torch.randn(2, 35, dtype=torch.float64, generator=generator) * .001
    return state, following, action, previous, previous_previous, context


class SmoothRewardKernelTest(unittest.TestCase):
    def test_actual_unitree_pd_motor_clipping_and_friction_equivalence(self):
        module = kernel()
        state, following, action, previous, previous_previous, context = inputs()
        # Cover both motor quadrants, torque saturation and speed taper.
        state[:, 42:71] = torch.linspace(-12., 12., 29)
        action[:] = torch.linspace(-40., 40., 29)
        request, clipped, command = module.motor_effort(state, action, context)
        native = object.__new__(UnitreeActuator)
        native.cfg = SimpleNamespace(stiffness=float(module.kp[0]), damping=float(module.kd[0]), effort_limit=10.)
        for attr, value in (("_effort_y1", module.y1), ("_effort_y2", module.y2),
                            ("_velocity_x1", module.x1), ("_velocity_x2", module.x2),
                            ("_friction_static", module.friction_static),
                            ("_friction_dynamic", module.friction_dynamic),
                            ("_activation_vel", module.activation_velocity)):
            setattr(native, attr, value.expand(2, -1).clone())
        native._joint_vel = state[:, 42:71].clone()
        torch.testing.assert_close(clipped, native._clip_effort(request), rtol=0, atol=0)
        target = module.action_offset + module.action_scale * action - context[:, 1:30]
        cmd = ActuatorCmd(position_target=target, velocity_target=torch.zeros_like(action),
                          effort_target=torch.zeros_like(action), pos=state[:, 13:42],
                          vel=state[:, 42:71])
        control = native.compute(cmd)
        reconstructed = native.cfg.stiffness * (control - cmd.pos) - native.cfg.damping * cmd.vel
        torch.testing.assert_close(command, reconstructed, rtol=1e-14, atol=2e-15)

    def test_full_cost_gradient_matches_finite_difference(self):
        module = kernel()
        values = tuple(x.clone().requires_grad_(True) for x in inputs())
        def total(*args):
            terms = module(*args)
            return sum(terms.values())
        self.assertTrue(torch.autograd.gradcheck(total, values, atol=1e-5, rtol=1e-4))
        self.assertEqual(list(module.parameters()), [])

    def test_second_difference_keeps_both_previous_action_gradients(self):
        values = list(inputs())
        for i in (2, 3, 4):
            values[i] = values[i].clone().requires_grad_(True)
        term = kernel()(*values)["action_second_difference"].sum()
        action_grad, previous_grad, previous_previous_grad = torch.autograd.grad(term, values[2:5])
        torch.testing.assert_close(action_grad, previous_previous_grad)
        torch.testing.assert_close(previous_grad, -2 * action_grad)
        self.assertGreater(float(action_grad.abs().max()), 0)

    def test_source_terms_are_zero_safe_and_nonnegative(self):
        module = kernel()
        state = torch.zeros(3, 71, dtype=torch.float64)
        action = torch.zeros(3, 29, dtype=torch.float64, requires_grad=True)
        # Make target=stateq, so requested torque and velocity are also zero.
        state[:, 13:42] = module.action_offset
        context = torch.zeros(3, 35, dtype=torch.float64)
        terms = module(state, state, action, action, action, context)
        self.assertEqual(tuple(terms), TERM_NAMES)
        for value in terms.values():
            self.assertTrue(bool((value == 0).all()))
        gradient, = torch.autograd.grad(sum(terms.values()).sum(), action)
        self.assertTrue(bool(torch.isfinite(gradient).all()))
        self.assertTrue(bool((gradient == 0).all()))

    def test_request_torque_has_real_gradient_outside_motor_saturation(self):
        module = kernel()
        state, following, action, previous, previous_previous, context = inputs()
        action = torch.full_like(action, 100.).requires_grad_(True)
        request, clipped, _ = module.motor_effort(state, action, context)
        self.assertTrue(bool((request.abs() > clipped.abs()).all()))
        costs = module(state, following, action, previous, previous_previous, context)
        gradient, = torch.autograd.grad(costs["requested_torque"].sum(), action)
        self.assertTrue(bool(torch.isfinite(gradient).all()))
        self.assertTrue(bool((gradient.abs() > 0).all()))

    def test_encoder_bias_matches_target_subtraction(self):
        module = kernel()
        state, following, action, previous, previous_previous, context = inputs()
        original = module.motor_effort(state, action, context)
        shifted = state.clone(); shifted[:, 13:42] += .03
        compensated = context.clone(); compensated[:, 1:30] -= .03
        changed = module.motor_effort(shifted, action, compensated)
        for first, second in zip(original, changed):
            torch.testing.assert_close(first, second, atol=2e-16, rtol=1e-14)

    def test_arbitrary_trajectory_prefix_and_no_statistical_state_updates(self):
        module = kernel()
        original = copy.deepcopy(module.state_dict())
        values = tuple(x.unsqueeze(0).expand(4, *x.shape) for x in inputs())
        terms = module(*values)
        self.assertTrue(all(value.shape == (4, 2) for value in terms.values()))
        for key, value in original.items():
            torch.testing.assert_close(value, module.state_dict()[key], rtol=0, atol=0)

    def test_metadata_and_validation_make_motor_contract_explicit(self):
        module = kernel()
        meta = module.metadata()
        self.assertIn("not_actual_actuator_force", meta["torque_semantics"])
        self.assertEqual(module.metadata(), meta)
        with torch.no_grad():
            module.action_scale[0] *= 2
        self.assertNotEqual(module.metadata()["sha256"], meta["sha256"])
        values = {key: getattr(module, key) for key in meta["constants"]}
        with self.assertRaises(ValueError):
            SmoothRewardTerms(dict(values, kp=-1))
        with self.assertRaises(ValueError):
            SmoothRewardTerms(dict(values, action_scale=torch.randn(2, 29)))


if __name__ == '__main__':
    unittest.main()
