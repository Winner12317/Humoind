"""Small deterministic checks for return semantics, gradient ownership and replay."""

from __future__ import annotations

import unittest

import torch
from torch import nn

from src.wmac.config import WmacConfig
from src.wmac.networks import Actor, Critic, RunningNorm, WorldModel, frozen_parameters
from src.wmac.replay import ReplayBuffer
from src.wmac.runner import differentiable_amp_reward, lambda_returns
from src.wmac.state import heading_quat, quat_from_rotvec


def physical_state(batch_size: int) -> torch.Tensor:
    state = torch.zeros(batch_size, 71)
    state[:, 2] = 0.8
    state[:, 3] = 1.0
    return state


def transition(time: int, num_envs: int = 2) -> dict[str, torch.Tensor]:
    state = physical_state(num_envs)
    state[0, 2] = 0.3
    return {
        "state": state,
        "next_state": state.clone(),
        "action": torch.full((num_envs, 29), float(time)),
        "previous_action": torch.zeros(num_envs, 29),
        "context": torch.zeros(num_envs, 35),
        "next_context": torch.zeros(num_envs, 35),
        "command": torch.zeros(num_envs, 3),
        "actor_history": torch.zeros(num_envs, 384),
        "reward": torch.zeros(num_envs),
        "terminated": torch.zeros(num_envs, dtype=torch.bool),
        "timeout": torch.zeros(num_envs, dtype=torch.bool),
        "boundary": torch.zeros(num_envs, dtype=torch.bool),
        "contact_cost": torch.zeros(num_envs, 2),
    }


class LambdaReturnTest(unittest.TestCase):
    def test_fixed_targets_include_terminal_reward_and_detach_every_source(self):
        rewards = torch.tensor([[1., 2.], [3., 4.]], requires_grad=True)
        continuation = torch.tensor([[1., 0.], [0., 1.]], requires_grad=True)
        values = torch.tensor([[10., 20.], [30., 40.], [50., 60.]], requires_grad=True)
        targets = lambda_returns(rewards, continuation, values, gamma=.9, lam=.5)
        expected = torch.tensor([[15.85, 2.], [3., 58.]])
        torch.testing.assert_close(targets, expected)
        self.assertFalse(targets.requires_grad)
        self.assertIsNone(targets.grad_fn)
        self.assertIsNone(rewards.grad)
        self.assertIsNone(continuation.grad)
        self.assertIsNone(values.grad)

    def test_time_limit_bootstraps_but_true_terminal_does_not(self):
        # Both samples end a replay window, but only physical death removes V.
        rewards = torch.tensor([[5., 5.]])
        terminated = torch.tensor([[True, False]])
        timeout = torch.tensor([[False, True]])
        values = torch.tensor([[2., 2.], [7., 7.]])
        self.assertTrue((terminated | timeout).all())
        targets = lambda_returns(rewards, (~terminated).float(), values, gamma=.9)
        torch.testing.assert_close(targets, torch.tensor([[5., 11.3]]))

    def test_lambda_zero_and_one_have_expected_bootstrap_depth(self):
        rewards = torch.tensor([[1.], [2.]])
        continuation = torch.ones_like(rewards)
        values = torch.tensor([[0.], [10.], [20.]])
        one_step = lambda_returns(rewards, continuation, values, gamma=.5, lam=0.)
        full_horizon = lambda_returns(rewards, continuation, values, gamma=.5, lam=1.)
        torch.testing.assert_close(one_step, torch.tensor([[6.], [12.]]))
        torch.testing.assert_close(full_horizon, torch.tensor([[7.], [12.]]))


class DynamicsGradientTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def test_identity_rotation_and_state_loss_have_finite_gradients(self):
        model = WorldModel(hidden=(16, 16))
        state = physical_state(3).requires_grad_(True)
        action = torch.randn(3, 29, requires_grad=True)
        context = torch.zeros(3, 35)
        predicted = model.step(state, action, context)
        target = physical_state(3)
        target[:, 3:7] = quat_from_rotvec(torch.tensor([[.01, 0., 0.]]).expand(3, -1))
        target[:, 42] = .2
        loss = model.loss(predicted, target)
        self.assertEqual(loss.shape, (3,))
        torch.testing.assert_close(predicted[:, 3:7].norm(dim=-1), torch.ones(3))
        loss.mean().backward()
        self.assertTrue(torch.isfinite(state.grad).all())
        self.assertTrue(torch.isfinite(action.grad).all())
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(g).all() for g in gradients))

    def test_quaternion_sign_does_not_change_the_physical_loss(self):
        model = WorldModel(hidden=(8,))
        state = physical_state(2)
        equivalent = state.clone()
        equivalent[:, 3:7].neg_()
        torch.testing.assert_close(model.loss(equivalent, state), torch.zeros(2))

    def test_vertical_forward_axis_has_finite_heading_and_dynamics_gradients(self):
        # This unit quaternion has a vertical forward axis: the ordinary yaw
        # formula becomes atan2(0, 0), a real corner case during recovery.
        quaternion = torch.tensor([[.5, .5, .5, -.5]], requires_grad=True)
        heading_quat(quaternion).sum().backward()
        self.assertTrue(torch.isfinite(quaternion.grad).all())
        model = WorldModel(hidden=(8,))
        state = physical_state(1)
        state[:, 3:7] = quaternion.detach()
        state.requires_grad_(True)
        action = torch.zeros(1, 29, requires_grad=True)
        predicted = model.step(state, action, torch.zeros(1, 35))
        target = state.detach().clone()
        target[:, 2] += .01
        model.loss(predicted, target).sum().backward()
        self.assertTrue(torch.isfinite(state.grad).all())
        self.assertTrue(torch.isfinite(action.grad).all())

    def test_actor_receives_reward_and_terminal_value_gradients_through_frozen_networks(self):
        actor = Actor(obs_dim=4, action_dim=29, hidden=(8,), action_limit=None)
        model = WorldModel(hidden=(16,))
        # A fitted dynamics model must have action sensitivity. Fresh dynamics
        # deliberately initializes its residual head to zero before fitting.
        with torch.no_grad():
            model.net[-1].weight.normal_(0, .01)
        discriminator = nn.Linear(2, 1)
        critic = Critic(1, hidden=())
        with torch.no_grad():
            discriminator.weight.copy_(torch.tensor([[0., .5]]))
            discriminator.bias.zero_()
            critic.net[0].weight.fill_(2.)
            critic.net[0].bias.zero_()
        normalizer = nn.Identity()
        observation = torch.randn(4, 4)
        parameters = tuple(actor.parameters())
        original_flags = [[p.requires_grad for p in net.parameters()]
                          for net in (model, discriminator, critic)]
        with frozen_parameters(model, discriminator, critic):
            action = actor(observation)
            predicted = model.step(physical_state(4), action, torch.zeros(4, 35))
            task = predicted[:, 2]
            style, _ = differentiable_amp_reward(discriminator, normalizer,
                torch.zeros(4, 1), predicted[:, 2:3], coefficient=.1)
            terminal_value = critic(predicted[:, 2:3])
            for name, objective in (("task", task), ("style", style), ("value", terminal_value)):
                gradients = torch.autograd.grad(objective.mean(), parameters,
                    retain_graph=True, allow_unused=True)
                norm = sum(float(g.square().sum()) for g in gradients if g is not None)
                self.assertGreater(norm, 0., msg=f"Missing actor gradient from {name}")
            (-(task + style + .99 * terminal_value).mean()).backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in parameters))
        for net, flags in zip((model, discriminator, critic), original_flags):
            self.assertTrue(all(p.grad is None for p in net.parameters()))
            self.assertEqual([p.requires_grad for p in net.parameters()], flags)

    def test_normalizer_statistics_are_fixed_during_differentiable_forward(self):
        normalizer = RunningNorm(2)
        normalizer.update(torch.tensor([[1., 2.], [3., 4.]]))
        count = normalizer.count.clone()
        value = torch.tensor([[9., 10.]], requires_grad=True)
        output = normalizer(value)
        output.sum().backward()
        torch.testing.assert_close(normalizer.count, count)
        torch.testing.assert_close(normalizer.mean, torch.tensor([2., 3.]))
        self.assertTrue(torch.isfinite(value.grad).all())
        self.assertTrue((value.grad != 0).all())


class RewardAblationTest(unittest.TestCase):
    def setUp(self):
        self.discriminator = nn.Linear(2, 1)
        with torch.no_grad():
            self.discriminator.weight.copy_(torch.tensor([[0., .5]]))
            self.discriminator.bias.zero_()

    def objective(self, preset):
        cfg = WmacConfig(preset=preset).apply_preset()
        x = torch.tensor([[.2]], requires_grad=True)
        style, _ = differentiable_amp_reward(self.discriminator, nn.Identity(),
            torch.zeros_like(x), x, coefficient=cfg.amp_reward_coef)
        direct_style = style if cfg.direct_style_grad else style.detach()
        task = x[:, 0].square()
        objective = cfg.task_weight * task + cfg.style_weight * direct_style
        gradient = torch.autograd.grad(objective.sum(), x)[0]
        return cfg, objective.detach(), gradient

    def test_task_only_removes_style_value_and_gradient_from_reward_target(self):
        cfg, objective, gradient = self.objective("task_only")
        self.assertEqual((cfg.task_weight, cfg.style_weight), (1., 0.))
        self.assertFalse(cfg.direct_style_grad)
        torch.testing.assert_close(objective, torch.tensor([.04]))
        torch.testing.assert_close(gradient, torch.tensor([[.4]]))

    def test_direct_style_ablation_preserves_reward_values_but_changes_local_gradient(self):
        _, full_value, full_gradient = self.objective("full")
        cfg, ablated_value, ablated_gradient = self.objective("no_direct_style_grad")
        self.assertFalse(cfg.direct_style_grad)
        torch.testing.assert_close(full_value, ablated_value)
        torch.testing.assert_close(ablated_gradient, torch.tensor([[.3]]))
        torch.testing.assert_close(full_gradient, torch.tensor([[.305625]]))

    def test_clipped_amp_reward_has_a_real_dead_gradient_region(self):
        second = torch.tensor([[-4.]], requires_grad=True)
        reward, score = differentiable_amp_reward(self.discriminator, nn.Identity(),
            torch.zeros_like(second), second, coefficient=.1)
        torch.testing.assert_close(score, torch.tensor([-2.]))
        torch.testing.assert_close(reward, torch.zeros(1))
        reward.sum().backward()
        torch.testing.assert_close(second.grad, torch.zeros_like(second))


class ReplayTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)

    def test_ring_order_and_boundary_step_is_valid_but_its_successors_are_not(self):
        replay = ReplayBuffer(capacity_steps=4, num_envs=2)
        for time in range(7):
            batch = transition(time)
            batch["boundary"][0] = time == 4
            replay.add(batch)
        self.assertEqual(replay.size, 4)
        window = replay.sample_windows(batch_size=32, length=4, recovery_fraction=.5)
        expected_time = torch.tensor([3, 4, 5, 6]).expand(32, -1)
        torch.testing.assert_close(window["time_index"], expected_time)
        torch.testing.assert_close(window["action"][:, :, 0], expected_time.float())
        recovery = window["state"][:, 0, 2] < .55
        self.assertEqual(int(recovery.sum()), 16)
        torch.testing.assert_close(window["mask"][recovery], torch.tensor([1., 1., 0., 0.]).expand(16, -1))
        torch.testing.assert_close(window["mask"][~recovery], torch.ones(16, 4))

    def test_delayed_recovery_counter_is_stratified_even_after_height_recovers(self):
        replay = ReplayBuffer(4, 2)
        for time in range(4):
            batch = transition(time)
            batch["state"][:, 2] = .8
            batch["context"][0, 34] = 3
            replay.add(batch)
        sample = replay.sample_starts(100, recovery_fraction=.5)
        self.assertEqual(int((sample["context"][:, 34] > 0).sum()), 50)

    def test_uniform_zero_does_not_exclude_or_oversample_the_recovery_class(self):
        replay = ReplayBuffer(2, 8)
        replay.add(transition(0, num_envs=8))
        sample = replay.sample_starts(800, recovery_fraction=0.)
        count = int((sample["state"][:, 2] < .55).sum())
        self.assertGreater(count, 0)
        self.assertLess(count, 200)  # Uniform: ~100; a forced 50/50 split: 400.

    def test_missing_recovery_or_normal_class_falls_back_without_resampling_loop(self):
        for height in (.3, .8):
            with self.subTest(height=height):
                replay = ReplayBuffer(2, 2)
                batch = transition(0)
                batch["state"][:, 2] = height
                replay.add(batch)
                sample = replay.sample_starts(100, recovery_fraction=.5)
                self.assertEqual(sample["state"].shape, (100, 71))
                torch.testing.assert_close(sample["state"][:, 2], torch.full((100,), height))

    def test_checkpoint_tail_is_cut_before_new_simulator_data(self):
        original = ReplayBuffer(5, 2)
        for time in range(8):
            original.add(transition(time))
        resumed = ReplayBuffer(5, 2)
        resumed.load_state_dict(original.state_dict())
        resumed.mark_boundary()
        # A restarted simulator has new states, despite consecutive replay IDs.
        resumed.add(transition(100))
        resumed.add(transition(101))
        windows = resumed.sample_windows(64, length=4, recovery_fraction=0.)
        for times, mask in zip(windows["time_index"], windows["mask"]):
            boundary_index = int((times == 7).nonzero()[0])
            self.assertEqual(float(mask[boundary_index]), 1.)
            self.assertFalse(mask[boundary_index + 1:].any())


if __name__ == "__main__":
    unittest.main()
