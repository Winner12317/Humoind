"""Geometry, local gradient, and reward-scope checks for imagined foot slip."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import mujoco
import numpy as np
import torch

from src.wmac.foot_slip import G1FootSlipGeometry
from test_wmac_state_fast import make_adapter


def geometry_for(adapter, model, *, contact_height_m=0.0, softness_m=.001, differentiate_support=False):
    return G1FootSlipGeometry.from_model(
        model, adapter.body_ids, adapter.foot_ids, contact_height_m=contact_height_m,
        softness_m=softness_m, differentiate_support=differentiate_support, dtype=torch.float64,
    )


def setup_reward(adapter, terrain_type="plane"):
    adapter.env.cfg.scene = SimpleNamespace(terrain=SimpleNamespace(terrain_type=terrain_type))
    term = adapter.reward_cfg["foot_slip"]
    term.params = dict(command_name="twist", command_threshold=.1,
                       asset_cfg=SimpleNamespace(site_names=("left_foot", "right_foot")))


class RecoveryFootSlipGeometryTest(unittest.TestCase):
    def test_geometry_and_site_velocity_match_mujoco_at_tilted_poses(self):
        adapter, model = make_adapter()
        geometry = geometry_for(adapter, model)
        data = mujoco.MjData(model)
        states, heights, velocities = [], [], []
        rng = np.random.default_rng(288)
        for _ in range(6):
            qpos = model.qpos0.copy()
            qpos[2] = rng.uniform(.25, .65)
            qpos[3:7] = rng.normal(size=4)
            qpos[3:7] /= np.linalg.norm(qpos[3:7])
            qpos[7:] += rng.normal(scale=.2, size=29)
            data.qpos[:] = qpos
            data.qvel[:] = rng.normal(size=35)
            mujoco.mj_forward(model, data)
            rotation = data.xmat[adapter.body_ids[0]].reshape(3, 3)
            states.append(np.concatenate((qpos[:7], data.qvel[:3], rotation @ data.qvel[3:6],
                                          qpos[7:], data.qvel[6:])))
            frame_heights, frame_velocities = [], []
            for side in ("left", "right"):
                indices = [model.geom(f"{side}_foot{index}_collision").id for index in range(1, 8)]
                axial_vertical = np.abs(data.geom_xmat[indices].reshape(7, 3, 3)[:, 2, 2])
                lowest = (data.geom_xpos[indices, 2] - model.geom_size[indices, 0]
                          - axial_vertical * model.geom_size[indices, 1])
                frame_heights.append(lowest.min())
                site_velocity = np.empty(6, dtype=np.float64)
                mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_SITE,
                                         model.site(f"{side}_foot").id, site_velocity, 0)
                frame_velocities.append(site_velocity[3:5])
            heights.append(frame_heights)
            velocities.append(frame_velocities)
        states = torch.as_tensor(np.asarray(states)).reshape(2, 3, 71)
        actual_height, actual_velocity = geometry.kinematic_terms(adapter.analyze_state(states))
        torch.testing.assert_close(actual_height, torch.as_tensor(np.asarray(heights)).reshape(2, 3, 2),
                                   atol=2e-7, rtol=0)
        torch.testing.assert_close(actual_velocity, torch.as_tensor(np.asarray(velocities)).reshape(2, 3, 2, 2),
                                   atol=8e-7, rtol=0)

    def test_support_stops_height_gradient_but_preserves_slip_and_command_gate(self):
        adapter, model = make_adapter()
        geometry = geometry_for(adapter, model)
        size = len(adapter.body_names)
        positions = torch.zeros(2, size, 3, dtype=torch.float64)
        positions[..., 2] = .035
        positions.requires_grad_()
        orientations = torch.zeros(2, size, 4, dtype=torch.float64)
        orientations[..., 0] = 1
        linear = torch.ones(2, size, 3, dtype=torch.float64, requires_grad=True)
        angular = torch.zeros_like(linear)
        commands = torch.tensor([[0., 0., -.12], [.1, 0., 0.]], dtype=torch.float64)
        args = (positions, orientations, linear, angular)
        cost = geometry(args, commands)
        height_grad, speed_grad = torch.autograd.grad(cost.sum(), (positions, linear), allow_unused=True)
        self.assertIsNone(height_grad)
        torch.testing.assert_close(cost, torch.tensor([2., 0.], dtype=torch.float64))
        expected_grad = torch.zeros_like(speed_grad)
        expected_grad[0, adapter.foot_ids, :2] = 1.
        torch.testing.assert_close(speed_grad, expected_grad)
        compiled = torch.compile(geometry, backend="eager", fullgraph=True)
        torch.testing.assert_close(compiled(args, commands), cost)

    def test_instantamp_full_gradient_preserves_reward_values_at_the_same_parameters(self):
        adapter, model = make_adapter()
        full = geometry_for(adapter, model, contact_height_m=.001, differentiate_support=True)
        detached = geometry_for(adapter, model, contact_height_m=.001)
        size = len(adapter.body_names)
        positions = torch.zeros(2, size, 3, dtype=torch.float64)
        positions[..., 2] = .036
        positions.requires_grad_()
        orientations = torch.zeros(2, size, 4, dtype=torch.float64)
        orientations[..., 0] = 1.
        linear = torch.ones(2, size, 3, dtype=torch.float64, requires_grad=True)
        angular = torch.zeros_like(linear)
        commands = torch.tensor([[0., 0., -.12], [.1, 0., 0.]], dtype=torch.float64)
        args = (positions, orientations, linear, angular)
        cost = full(args, commands)
        torch.testing.assert_close(cost, detached(args, commands), rtol=0, atol=0)
        height_grad, speed_grad = torch.autograd.grad(cost.sum(), (positions, linear))
        expected_height_grad = torch.zeros_like(height_grad)
        expected_height_grad[0, adapter.foot_ids, 2] = -.5/.001
        expected_speed_grad = torch.zeros_like(speed_grad)
        expected_speed_grad[0, adapter.foot_ids, :2] = 1.
        torch.testing.assert_close(height_grad, expected_height_grad)
        torch.testing.assert_close(speed_grad, expected_speed_grad)
        compiled = torch.compile(full, backend="eager", fullgraph=True)
        torch.testing.assert_close(compiled(args, commands), cost)

    def test_instantamp_support_gradient_follows_tilted_foot_kinematics(self):
        adapter, model = make_adapter()
        geometry = geometry_for(adapter, model, contact_height_m=.001, differentiate_support=True)
        size = len(adapter.body_names)
        positions = torch.zeros(1, size, 3, dtype=torch.float64)
        orientations = torch.zeros(1, size, 4, dtype=torch.float64)
        orientations[..., 0] = 1.
        orientations[:, adapter.foot_ids] = torch.tensor([.92, .14, -.28, .03], dtype=torch.float64)
        linear = torch.ones_like(positions)
        angular = torch.zeros_like(positions)
        height, _ = geometry.kinematic_terms((positions, orientations, linear, angular))
        positions[:, adapter.foot_ids, 2] = .001 - height
        orientations.requires_grad_()
        cmd = torch.tensor([[.2, 0., 0.]], dtype=torch.float64)
        value = geometry((positions, orientations, linear, angular), cmd)
        gradient, = torch.autograd.grad(value.sum(), orientations)
        direction = torch.zeros_like(orientations)
        direction[:, adapter.foot_ids[0], 2] = 1.
        with torch.no_grad():
            upper = geometry((positions, orientations + 1e-7*direction, linear, angular), cmd)
            lower = geometry((positions, orientations - 1e-7*direction, linear, angular), cmd)
        expected = ((upper - lower) / (2e-7)).sum()
        self.assertGreater(abs(float(expected)), 1.)
        torch.testing.assert_close((gradient * direction).sum(), expected, rtol=1e-5, atol=1e-7)

    def test_adapter_changes_only_recovery_slip_and_reuses_kinematics(self):
        adapter, _ = make_adapter(dtype=torch.float32)
        setup_reward(adapter)
        state = torch.zeros(4, 71)
        state[:, 2] = torch.tensor([.4, .6, .4, .6])
        state[:, 3] = 1
        next_state = state.clone()
        context = torch.zeros(4, 35)
        context[:, 33] = torch.tensor([1., 1., 0., 1.])
        context[1, 34] = 1.
        next_context = context.clone()
        next_context[:, 34] = 0.
        cmd = torch.tensor([[.5, 0., 0.]]).repeat(4, 1)
        action = torch.zeros(4, 29)
        done = torch.zeros(4)
        contact = torch.tensor([[2., 3.], [2., 4.], [2., 5.], [2., 6.]], requires_grad=True)
        fk = adapter.analyze_state(next_state)
        args = (state, next_state, action, action, cmd, next_context, done, contact, fk)
        before = adapter.task_reward(*args)
        adapter.configure_recovery_foot_slip(contact_height_m=0.0, softness_m=.001)
        with patch.object(adapter, "kinematics", side_effect=AssertionError("FK already supplied")):
            after = adapter.task_reward(*args, current_context=context)
        expected = before.clone()
        replacement = adapter.recovery_foot_slip(fk, cmd)
        selected = torch.tensor([0, 1])
        expected[selected] -= .25 * adapter.dt * (replacement[selected] - contact[selected, 0])
        torch.testing.assert_close(after, expected)
        torch.testing.assert_close(after[2:], before[2:], rtol=0, atol=0)
        contact_gradient, = torch.autograd.grad(after.sum(), contact)
        expected_gradient = torch.tensor([[0., -.1], [0., -.1], [-.25, -.1], [-.25, -.1]]) * adapter.dt
        torch.testing.assert_close(contact_gradient, expected_gradient)
        compiled = torch.compile(adapter.task_reward, backend="eager", fullgraph=True)
        torch.testing.assert_close(compiled(*args, current_context=context), after)
        adapter.configure_recovery_foot_slip(contact_height_m=0.0, softness_m=.001, differentiate_support=True)
        torch.testing.assert_close(adapter.task_reward(*args, current_context=context), after, rtol=0, atol=0)
        self.assertEqual(adapter.reward_cfg["foot_slip"].weight, -.25)
        with self.assertRaisesRegex(ValueError, "pre-transition"):
            adapter.task_reward(*args)

    def test_configuration_rejects_nonflat_or_changed_reward(self):
        adapter, _ = make_adapter(dtype=torch.float32)
        setup_reward(adapter, terrain_type="generator")
        with self.assertRaisesRegex(ValueError, "flat plane"):
            adapter.configure_recovery_foot_slip(contact_height_m=0.0, softness_m=.001)
        setup_reward(adapter)
        adapter.reward_cfg["foot_slip"].weight = -.2
        with self.assertRaisesRegex(ValueError, "original G1"):
            adapter.configure_recovery_foot_slip(contact_height_m=0.0, softness_m=.001)


if __name__ == "__main__":
    unittest.main()
