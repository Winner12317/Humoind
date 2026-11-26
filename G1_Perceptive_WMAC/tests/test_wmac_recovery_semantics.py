"""CPU oracles for versioned recovery clocks and upright-gated height rewards.

The v2 delayed-done derivative is an explicitly biased straight-through
surrogate. Tests verify its forward contract and finite/local backward signal;
they deliberately do not gradcheck the discontinuous physical clock.
"""
from __future__ import annotations

import copy
import math
from types import SimpleNamespace
import unittest

import torch

from src.recovery_rewards import (
    configure_recovery_height_reward, recovery_height_reward,
    recovery_root_up_z, recovery_upright_gate,
)
from src.tasks.amp_loco.mdp.rewards import track_root_height
from src.wmac.state import StateAdapter, quat_apply, quat_from_rotvec, quat_inv, quat_mul
from test_wmac_state_fast import make_adapter


def pose(adapter, angles, heights=None):
    angles = torch.as_tensor(angles, dtype=torch.float64).reshape(-1)
    state = torch.zeros(len(angles), 71, dtype=angles.dtype)
    state[:, 2] = .72 if heights is None else torch.as_tensor(heights, dtype=angles.dtype)
    rotations = torch.zeros(len(angles), 3, dtype=angles.dtype)
    rotations[:, 0] = torch.deg2rad(angles)
    state[:, 3:7] = quat_from_rotvec(rotations)
    state[:, 13:42] = adapter.default_joint_pos
    return state


def context(count, delay=1., counter=0.):
    result = torch.zeros(count, 35, dtype=torch.float64)
    result[:, 33] = torch.as_tensor(delay)
    result[:, 34] = torch.as_tensor(counter)
    return result


def legacy_context_oracle(adapter, state, values):
    """Independent copy of the complete pre-v2 clock/hazard expression."""
    gravity = quat_apply(quat_inv(state[..., 3:7]), state.new_tensor([0., 0., -1.]))
    height = torch.sigmoid((adapter.height_threshold-state[..., 2])/.025)
    tilt = torch.sigmoid((gravity[..., 2]+adapter.tilt_cos)/.025)
    bad = 1-(1-height)*(1-tilt)
    delay = values[..., 33]
    counter = delay*(values[..., 34]+1)*bad
    delayed_done = bad*torch.sigmoid((counter-adapter.max_delay_steps+.5)/.25)
    terminated = (1-delay)*bad+delay*delayed_done
    counter = counter*(1-delayed_done)
    return torch.cat((values[..., :34], counter[..., None]), -1), terminated, bad


class RecoveryClockSemanticsTest(unittest.TestCase):
    def test_default_and_explicit_versions_are_validated(self):
        adapter, _ = make_adapter()
        self.assertEqual(adapter.recovery_counter_semantics, "hard_forward_v2")
        old = StateAdapter(adapter.env, recovery_counter_semantics="soft_v1")
        self.assertEqual(old.recovery_counter_semantics, "soft_v1")
        with self.assertRaisesRegex(ValueError, "counter semantics"):
            StateAdapter(adapter.env, recovery_counter_semantics="unknown")

    def test_hard_forward_matches_strict_physical_bad_pose_predicate(self):
        adapter, _ = make_adapter()
        state = pose(adapter, [0., 69., 70., 71., 90., 180., 0.],
                     [.5, .72, .72, .72, .72, .72, .499999])
        values = context(len(state), counter=4.)
        gravity = quat_apply(quat_inv(state[:, 3:7]), state.new_tensor([0., 0., -1.]))
        limit = adapter.env.cfg.terminations["bad_orientation"].params["limit_angle"]
        expected_bad = ((state[:, 2] < adapter.height_threshold) |
                        (torch.acos(-gravity[:, 2]).abs() > limit))
        advanced, done = adapter.advance_context(state, values)
        torch.testing.assert_close(advanced[:, 34], expected_bad.double()*5., rtol=0, atol=0)
        torch.testing.assert_close(advanced[:, :34], values[:, :34], rtol=0, atol=0)
        torch.testing.assert_close(done, torch.zeros_like(done), rtol=0, atol=0)
        self.assertFalse(bool(expected_bad[0]))  # Equal height is not below the threshold.
        self.assertTrue(bool(expected_bad[-1]))

        # Set the angle threshold to the exact computed boundary value, avoiding
        # assumptions that a decimal-degree quaternion has no rounding error.
        boundary_state = pose(adapter, [71.])
        boundary_gravity = quat_apply(quat_inv(boundary_state[:, 3:7]), boundary_state.new_tensor([0., 0., -1.]))
        exact_angle = float(torch.acos(-boundary_gravity[:, 2]).abs())
        cfg = copy.deepcopy(adapter.env.cfg)
        cfg.terminations["bad_orientation"].params["limit_angle"] = exact_angle
        env = copy.copy(adapter.env)
        env.cfg = cfg
        boundary_adapter = StateAdapter(env)
        advanced, _ = boundary_adapter.advance_context(boundary_state, context(1, counter=4.))
        self.assertEqual(float(advanced[0, 34]), 0.)  # The physical condition is >, not >=.

    def test_continuous_bad_pose_reaches_250_then_clears_and_recovery_resets(self):
        adapter, _ = make_adapter()
        state = pose(adapter, [71.])
        values = context(1)
        for step in range(1, 251):
            values, done = adapter.advance_context(state, values)
            self.assertEqual(float(values[0, 34]), float(step if step < 250 else 0))
            self.assertEqual(float(done[0]), float(step == 250))
        recovered, done = adapter.advance_context(pose(adapter, [69.]), context(1, counter=125.))
        self.assertEqual(float(recovered[0, 34]), 0.)
        self.assertEqual(float(done[0]), 0.)
        bad_again, _ = adapter.advance_context(state, recovered)
        self.assertEqual(float(bad_again[0, 34]), 1.)

    def test_nondelay_and_disabled_clocks_preserve_counter_and_soft_termination(self):
        adapter, _ = make_adapter()
        state = pose(adapter, [71., 0.], [.52, .8])
        values = context(2, delay=0., counter=[7.25, 9.])
        _, _, soft_bad = legacy_context_oracle(adapter, state, values)
        advanced, done = adapter.advance_context(state, values)
        torch.testing.assert_close(advanced, values, rtol=0, atol=0)
        torch.testing.assert_close(done, soft_bad, rtol=0, atol=0)
        adapter.max_delay_steps = 0
        values = context(2, delay=[1., 0.], counter=[11., 5.5])
        advanced, done = adapter.advance_context(state, values)
        torch.testing.assert_close(advanced, values, rtol=0, atol=0)
        torch.testing.assert_close(done, soft_bad, rtol=0, atol=0)

    def test_detached_hard_clock_and_biased_st_hazard_have_expected_backward_contract(self):
        adapter, _ = make_adapter()
        state = pose(adapter, [71.], [.52]).requires_grad_(True)
        values = context(1, counter=249.).requires_grad_(True)
        advanced, done = adapter.advance_context(state, values)
        self.assertEqual(float(done.detach()[0]), 1.)
        self.assertEqual(float(advanced.detach()[0, 34]), 0.)
        counter_grads = torch.autograd.grad(advanced[:, 34].sum(), (state, values),
                                           allow_unused=True, retain_graph=True)
        for gradient in counter_grads:
            if gradient is not None:
                torch.testing.assert_close(gradient, torch.zeros_like(gradient), rtol=0, atol=0)
        state_grad, context_grad = torch.autograd.grad(done.sum(), (state, values))
        self.assertTrue(torch.isfinite(state_grad).all())
        self.assertTrue(torch.isfinite(context_grad).all())
        self.assertGreater(float(state_grad[:, 2:7].abs().sum()), 0.)
        self.assertEqual(float(context_grad[0, 34]), 0.)

    def test_explicit_soft_v1_matches_old_expression_values_and_gradients(self):
        template, _ = make_adapter()
        adapter = StateAdapter(template.env, recovery_counter_semantics="soft_v1")
        for maximum in (250, 0):
            with self.subTest(maximum=maximum):
                adapter.max_delay_steps = maximum
                state = pose(adapter, [0., 69., 71., 120.], [.49, .8, .52, .7]).requires_grad_(True)
                values = context(4, delay=[0., 1., 1., 1.], counter=[7., 3., 249., 100.]).requires_grad_(True)
                actual_context, actual_done = adapter.advance_context(state, values)
                expected_context, expected_done, _ = legacy_context_oracle(adapter, state, values)
                torch.testing.assert_close(actual_context, expected_context, rtol=0, atol=0)
                torch.testing.assert_close(actual_done, expected_done, rtol=0, atol=0)
                actual = torch.autograd.grad(actual_context[:, 34].sum()+actual_done.sum(), (state, values), retain_graph=True)
                expected = torch.autograd.grad(expected_context[:, 34].sum()+expected_done.sum(), (state, values))
                for first, second in zip(actual, expected):
                    torch.testing.assert_close(first, second, rtol=1e-12, atol=1e-12)


def physical_reward_env(adapter, state, values):
    count = len(state)
    desired = torch.zeros(count, 13, dtype=state.dtype)
    desired[:, 2] = adapter.desired_height
    # The extra torso-like body intentionally has another orientation. The MDP
    # must read root index0, not an anchor or a cached gravity observation.
    other_q = state.new_tensor([1., 0., 0., 0.]).expand(count, 4)
    robot = SimpleNamespace(data=SimpleNamespace(
        default_root_state=desired,
        body_link_pos_w=torch.stack((state[:, :3], state[:, :3]+1.), 1),
        body_link_quat_w=torch.stack((state[:, 3:7], other_q), 1)))
    return SimpleNamespace(scene={"robot": robot}, termination_manager=SimpleNamespace(
        _delay_env_mask=values[:, 33].bool(), _delay_counters=values[:, 34].long()))


class RecoveryHeightRewardSemanticsTest(unittest.TestCase):
    def test_upright_gate_and_height_reward_are_orientation_aware_and_yaw_invariant(self):
        up = torch.tensor([1., 0., -1., math.cos(math.radians(71.))], dtype=torch.float64)
        gate = recovery_upright_gate(up)
        torch.testing.assert_close(gate[:3], torch.tensor([1., 0., 0.], dtype=up.dtype), rtol=0, atol=0)
        self.assertAlmostEqual(float(gate[3]), math.cos(math.radians(71.))**2)
        self.assertLess(float(gate[3]), .11)
        tilt = torch.zeros(7, 3, dtype=up.dtype)
        tilt[:, 0] = torch.deg2rad(torch.tensor([89., 80., 71., 60., 45., 30., 0.], dtype=up.dtype))
        quaternion = quat_from_rotvec(tilt)
        reward = recovery_height_reward(torch.full((7,), .75, dtype=up.dtype), .75, .3,
                                        up_z=recovery_root_up_z(quaternion), upright_power=2.)
        self.assertTrue(bool((reward[1:] > reward[:-1]).all()))
        yaw = torch.zeros(7, 3, dtype=up.dtype)
        yaw[:, 2] = torch.linspace(-math.pi, math.pi, 7, dtype=up.dtype)
        rotated = quat_mul(quat_from_rotvec(yaw), quaternion)
        torch.testing.assert_close(recovery_root_up_z(rotated), recovery_root_up_z(quaternion),
                                   rtol=1e-13, atol=1e-13)
        legacy = recovery_height_reward(torch.tensor([.75, .75, .75]), .75, .3)
        torch.testing.assert_close(legacy, torch.ones_like(legacy), rtol=0, atol=0)

    def test_configuration_changes_only_the_selected_term_and_requires_a_version(self):
        original = {"std": .3, "mask_delay": True, "delay_env_rew_ratio": 3.5}
        unrelated = SimpleNamespace(weight=-200., params={"key": 9})
        cfg = SimpleNamespace(rewards={"track_root_height": SimpleNamespace(weight=1., params=original),
                                       "is_terminated": unrelated})
        self.assertIs(configure_recovery_height_reward(cfg, "upright_v2"), cfg)
        self.assertEqual(cfg.rewards["track_root_height"].params["upright_power"], 2.)
        self.assertNotIn("upright_power", original)
        self.assertIs(cfg.rewards["is_terminated"], unrelated)
        configure_recovery_height_reward(cfg, "height_only_v1")
        self.assertEqual(cfg.rewards["track_root_height"].params["upright_power"], 0.)
        with self.assertRaisesRegex(ValueError, "semantics"):
            configure_recovery_height_reward(cfg, "unknown")
        with self.assertRaisesRegex(ValueError, "track_root_height"):
            configure_recovery_height_reward(SimpleNamespace(rewards={}), "upright_v2")

    def test_physical_and_adapter_height_values_and_raw_state_gradients_match(self):
        adapter, _ = make_adapter()
        adapter.reward_cfg = copy.deepcopy(adapter.reward_cfg)
        for name, term in adapter.reward_cfg.items():
            term.weight = 1.7 if name == "track_root_height" else 0.
        for semantics in ("height_only_v1", "upright_v2"):
            with self.subTest(semantics=semantics):
                configure_recovery_height_reward(SimpleNamespace(rewards=adapter.reward_cfg), semantics)
                state = pose(adapter, [0., 25., 71., 110.], [.72, .72, .72, .72]).requires_grad_(True)
                values = context(4, delay=[0., 1., 1., 1.], counter=[5., 0., 2., 9.])
                env = physical_reward_env(adapter, state, values)
                term = adapter.reward_cfg["track_root_height"]
                physical = track_root_height(env, **term.params)*term.weight*adapter.reward_scale
                imagined = adapter.task_reward(state.detach(), state, state.new_zeros(4, 29),
                    state.new_zeros(4, 29), state.new_zeros(4, 3), values,
                    state.new_zeros(4), state.new_zeros(4, 2))
                torch.testing.assert_close(imagined, physical, rtol=1e-12, atol=1e-12)
                actual_grad = torch.autograd.grad(imagined.sum(), state, retain_graph=True)[0]
                physical_grad = torch.autograd.grad(physical.sum(), state)[0]
                self.assertTrue(torch.isfinite(actual_grad).all())
                torch.testing.assert_close(actual_grad, physical_grad, rtol=1e-10, atol=1e-11)
                torch.testing.assert_close(physical[:2], torch.zeros_like(physical[:2]), rtol=0, atol=0)

    def test_walking_mask_and_other_reward_contributions_are_unchanged(self):
        adapter, _ = make_adapter()
        adapter.reward_cfg = copy.deepcopy(adapter.reward_cfg)
        state = pose(adapter, [0., 71., 110.])
        prior = state.clone()
        action, previous = state.new_full((3, 29), .02), state.new_zeros(3, 29)
        command, contact = state.new_zeros(3, 3), state.new_full((3, 2), .01)
        def reward(values, done):
            return adapter.task_reward(prior, state, action, previous, command, values, done, contact)
        walking = context(3, delay=[0., 1., 0.], counter=[0., 0., 10.])
        active = context(3, counter=4.)
        configure_recovery_height_reward(SimpleNamespace(rewards=adapter.reward_cfg), "height_only_v1")
        old_walk, old_active = reward(walking, state.new_zeros(3)), reward(active, state.new_zeros(3))
        configure_recovery_height_reward(SimpleNamespace(rewards=adapter.reward_cfg), "upright_v2")
        new_walk, new_active = reward(walking, state.new_zeros(3)), reward(active, state.new_zeros(3))
        torch.testing.assert_close(new_walk, old_walk, rtol=0, atol=0)
        height_cfg = adapter.reward_cfg["track_root_height"]
        raw_height = torch.exp(-(state[:, 2]-adapter.desired_height).square()/height_cfg.params["std"]**2)
        delta = raw_height*(recovery_upright_gate(recovery_root_up_z(state[:, 3:7]))-1.)
        expected_delta = delta*height_cfg.params["delay_env_rew_ratio"]*height_cfg.weight*adapter.reward_scale
        torch.testing.assert_close(new_active-old_active, expected_delta, rtol=1e-12, atol=1e-12)
        terminal_delta = reward(active, state.new_ones(3))-new_active
        expected_terminal = state.new_full((3,), adapter.reward_cfg["is_terminated"].weight*adapter.reward_scale)
        torch.testing.assert_close(terminal_delta, expected_terminal, rtol=1e-12, atol=1e-12)
        height_cfg.params["mask_delay"] = False
        env = physical_reward_env(adapter, state, active)
        torch.testing.assert_close(track_root_height(env, **height_cfg.params), state.new_zeros(3), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
