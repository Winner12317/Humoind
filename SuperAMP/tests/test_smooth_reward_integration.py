"""Reward-only changes across actual WMAC imagination, restore and history paths."""
from __future__ import annotations

import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from src.wmac.networks import frozen_parameters
from src.wmac.replay import ReplayBuffer
from src.wmac.smooth_rewards import SmoothRewardTerms
from test_centered_amp_integration import assert_tree_equal
from test_wmac_optimization import FixedReplay, fix_update_noise, make_runner
from test_wmac_schedule_restore import checkpoint_runner


MODE = "host_humanup_v1"
NETWORKS = ("actor", "critic", "world_model", "discriminator", "amp_norm")
OPTIMIZERS = ("opt_actor", "opt_critic", "opt_model", "opt_disc")
WEIGHTS = {
    "action_second_difference": "smooth_action_second_coef",
    "joint_velocity": "smooth_joint_velocity_coef",
    "requested_torque": "smooth_torque_coef",
    "commanded_power": "smooth_power_coef",
}


def smoothed(runner):
    runner.cfg.smooth_reward_mode = MODE
    constants = dict(action_scale=.25, action_offset=runner.adapter.default_joint_pos,
                     kp=40., kd=2., effort_limit=88., y1=60., y2=80., x1=10., x2=30.,
                     friction_static=1., friction_dynamic=.02, activation_velocity=.2)
    dtype = next(runner.actor.parameters()).dtype
    runner.smooth_reward_terms = SmoothRewardTerms(constants).to(device=runner.device, dtype=dtype)
    return runner


def zero_base_reward(runner):
    runner.adapter.reward_cfg = copy.deepcopy(runner.adapter.reward_cfg)
    for term in runner.adapter.reward_cfg.values():
        term.weight = 0.
    runner.cfg.style_weight = 0.
    runner.cfg.task_weight = 1.
    runner.cfg.critic_bootstrap = False
    runner.cfg.survival_grad = False


class SmoothRewardIntegrationTest(unittest.TestCase):
    def test_reward_only_delta_matches_terms_once_and_does_not_change_dynamics(self):
        baseline, batch, noise = make_runner(batch_size=3)
        runner, _, _ = make_runner(batch_size=3)
        smoothed(runner)
        captured = []
        handle = runner.smooth_reward_terms.register_forward_hook(
            lambda module, args, result: captured.append(result))
        with frozen_parameters(baseline.world_model, baseline.discriminator, baseline.critic), \
             frozen_parameters(runner.world_model, runner.discriminator, runner.critic):
            old = baseline.imagine(batch, noises=noise)
            new = runner.imagine(batch, noises=noise)
        handle.remove()
        self.assertEqual(len(captured), 1)
        cost = sum(getattr(runner.cfg, weight) * captured[0][key]
                   for key, weight in WEIGHTS.items()) * runner.adapter.reward_scale
        torch.testing.assert_close(new["task"], old["task"] - cost.detach())
        for key in ("score", "style", "x", "first_action", "survival"):
            torch.testing.assert_close(new[key], old[key], rtol=0, atol=0)
        self.assertTrue((new["task"] <= old["task"]).all())
        self.assertGreater(float(cost.mean()), 0.)

    def test_imagination_bootstraps_second_action_difference_from_true_history(self):
        runner, batch, noise = make_runner(batch_size=3, horizon=4)
        smoothed(runner)
        history = batch["actor_history"].reshape(3, 4, 96)
        history[:, -2, 67:96] = -.73
        history[:, -1, 67:96] = batch["previous_action"]
        action_outputs, inputs = [], []
        actor_hook = runner.actor.register_forward_hook(
            lambda module, args, result: action_outputs.append(result))
        reward_hook = runner.smooth_reward_terms.register_forward_pre_hook(
            lambda module, args: inputs.append(args))
        with frozen_parameters(runner.world_model, runner.discriminator, runner.critic):
            runner.imagine(batch, noises=noise)
            state, next_state, action, previous, older, context = inputs[0]
            torch.testing.assert_close(older[0], history[:, -2, 67:96])
            torch.testing.assert_close(older[1], batch["previous_action"])
            torch.testing.assert_close(older[2], action_outputs[0])
            torch.testing.assert_close(older[3], action_outputs[1])
            # Later imagined histories must carry gradients to prior actions,
            # rather than becoming detached replay-like constants.
            carried, = torch.autograd.grad(older[2].sum(), action_outputs[0], retain_graph=True)
            torch.testing.assert_close(carried, torch.ones_like(carried))
            carried, = torch.autograd.grad(previous[1].sum(), action_outputs[0])
            torch.testing.assert_close(carried, torch.ones_like(carried))
        actor_hook.remove(); reward_hook.remove()

    def test_each_added_cost_has_finite_actor_gradient_with_frozen_world_model(self):
        for selected in WEIGHTS:
            with self.subTest(term=selected):
                runner, batch, noise = make_runner(batch_size=3, horizon=3)
                smoothed(runner); zero_base_reward(runner)
                batch["state"][:, 42:71] = .2
                for key, weight in WEIGHTS.items():
                    setattr(runner.cfg, weight, .01 if key == selected else 0.)
                before = {name: copy.deepcopy(getattr(runner, name).state_dict())
                          for name in ("world_model", "critic", "discriminator")}
                with frozen_parameters(runner.world_model, runner.discriminator, runner.critic):
                    result = runner.imagine(batch, noises=noise)
                    gradients = torch.autograd.grad(result["objective"].sum(), tuple(runner.actor.parameters()))
                self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
                self.assertGreater(sum(float(g.square().sum()) for g in gradients), 1e-20)
                for name, expected in before.items():
                    assert_tree_equal(self, expected, getattr(runner, name).state_dict())
                    self.assertTrue(all(p.grad is None for p in getattr(runner, name).parameters()))

    def test_actor_and_lambda_target_use_same_reward_and_no_added_network_pass(self):
        runner, batch, noise = make_runner(batch_size=3)
        smoothed(runner)
        runner.cfg.gamma = 0.
        runner.cfg.style_weight = 0.
        runner.cfg.task_weight = 1.
        runner.cfg.critic_bootstrap = False
        counts = {"model": 0, "disc": 0}
        model_call = runner.world_model.step_with_contact
        def step(*args, **kwargs):
            counts["model"] += 1
            return model_call(*args, **kwargs)
        disc_hook = runner.discriminator.register_forward_hook(
            lambda *args: counts.__setitem__("disc", counts["disc"] + 1))
        with patch.object(runner.world_model, "step_with_contact", side_effect=step), \
             frozen_parameters(runner.world_model, runner.discriminator, runner.critic):
            result = runner.imagine(batch, noises=noise)
        disc_hook.remove()
        torch.testing.assert_close(result["targets"], result["task"])
        torch.testing.assert_close(result["objective"], result["task"][0])
        self.assertEqual(counts, {"model": 3, "disc": 3})
        self.assertEqual(list(runner.smooth_reward_terms.parameters()), [])

    def test_zero_added_coefficients_preserve_historical_forward_and_gradient(self):
        baseline, batch, noise = make_runner(batch_size=3)
        runner, _, _ = make_runner(batch_size=3)
        smoothed(runner)
        for weight in WEIGHTS.values():
            setattr(runner.cfg, weight, 0.)
        with frozen_parameters(baseline.world_model, baseline.discriminator, baseline.critic), \
             frozen_parameters(runner.world_model, runner.discriminator, runner.critic):
            old = baseline.imagine(batch, noises=noise)
            new = runner.imagine(batch, noises=noise)
            old_grad = torch.autograd.grad(old["objective"].sum(), tuple(baseline.actor.parameters()))
            new_grad = torch.autograd.grad(new["objective"].sum(), tuple(runner.actor.parameters()))
        for key in ("objective", "task", "targets"):
            torch.testing.assert_close(new[key], old[key], rtol=0, atol=0)
        for first, second in zip(old_grad, new_grad):
            torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_actor_update_preserves_world_model_optimizer_and_reward_constants(self):
        runner, _, noise = make_runner(batch_size=3)
        smoothed(runner)
        fix_update_noise(runner, noise)
        before = {name: copy.deepcopy(getattr(runner, name).state_dict())
                  for name in ("world_model", "opt_model", "discriminator", "opt_disc", "smooth_reward_terms")}
        result = runner.update_actor_critic()
        self.assertEqual(result["actor_update_rejected"], 0.)
        for name, expected in before.items():
            assert_tree_equal(self, expected, getattr(runner, name).state_dict())

    def test_optional_real_critic_uses_already_relabelled_reward_without_double_penalty(self):
        runner, values, _ = make_runner(batch_size=3)
        smoothed(runner)
        runner.cfg.gamma = 0.
        runner.cfg.task_weight = 1.
        runner.cfg.style_weight = 0.
        values.update(next_state=values["state"]+.001,
                      next_context=values["context"].clone(),
                      action=torch.full_like(values["previous_action"], .2),
                      reward=torch.tensor([.031, -.04, .12], dtype=values["state"].dtype),
                      terminated=torch.zeros(3, dtype=torch.bool))
        runner.replay = FixedReplay(values)
        # The physical reward manager already included the negative cost.
        # Optional real TD must consume that label exactly once.
        with patch.object(runner.smooth_reward_terms, "forward", side_effect=AssertionError("double penalty")):
            _, target = runner._real_critic_batch(3)
        torch.testing.assert_close(target, values["reward"], rtol=0, atol=0)


class SmoothRewardRestoreIntegrationTest(unittest.TestCase):
    def test_explicit_reward_migration_preserves_all_networks_optimizers_and_clocks(self):
        source = checkpoint_runner()
        for name in OPTIMIZERS:
            optimizer = getattr(source, name)
            for group in optimizer.param_groups:
                for parameter in group["params"]:
                    parameter.grad = torch.full_like(parameter, .001)
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
        restored = smoothed(checkpoint_runner())
        restored.cfg.allow_smooth_reward_migration = True
        expected = {name: copy.deepcopy(getattr(source, name).state_dict())
                    for name in NETWORKS + OPTIMIZERS}
        with tempfile.TemporaryDirectory() as folder, patch("torch.cuda.is_available", return_value=False):
            path = Path(folder)/"source.pt"
            source.save(path, export=False)
            restored.load(path)
        for name, saved in expected.items():
            assert_tree_equal(self, saved, getattr(restored, name).state_dict())
        self.assertEqual((restored.iteration, restored.sim_steps, restored.imagination_steps),
                         (source.iteration, source.sim_steps, source.imagination_steps))
        self.assertEqual(restored.critic_calibration_updates, source.critic_calibration_updates)
        self.assertEqual(restored.cfg.model_warmup_iterations, source.cfg.model_warmup_iterations)
        self.assertEqual(restored.cfg.critic_warmup_iterations, source.cfg.critic_warmup_iterations)
        self.assertEqual(restored._collect_accumulator, source._collect_accumulator)

    def test_reward_migration_discards_saved_old_reward_labels(self):
        source = checkpoint_runner()
        source.cfg.checkpoint_replay = True
        source.replay = ReplayBuffer(2, 2, "cpu")
        state = torch.zeros(2, 71); state[:, 3] = 1.; state[:, 2] = .7
        source.replay.add(dict(state=state, next_state=state+.001, reward=torch.tensor([1., 2.]),
                               terminated=torch.zeros(2, dtype=torch.bool)))
        restored = smoothed(checkpoint_runner())
        restored.cfg.allow_smooth_reward_migration = True
        restored.replay = ReplayBuffer(2, 2, "cpu")
        with tempfile.TemporaryDirectory() as folder, patch("torch.cuda.is_available", return_value=False):
            path = Path(folder)/"old_rewards.pt"
            source.save(path, export=False); restored.load(path)
        self.assertEqual(restored.replay.size, 0)
        self.assertEqual(restored.sim_steps, source.sim_steps)

    def test_new_mode_roundtrip_and_malformed_motor_metadata_reject_before_network_mutation(self):
        source = smoothed(checkpoint_runner())
        restored = smoothed(checkpoint_runner())
        with tempfile.TemporaryDirectory() as folder, patch("torch.cuda.is_available", return_value=False):
            path = Path(folder)/"new_rewards.pt"
            source.save(path, export=False)
            payload = torch.load(path, weights_only=False)
            self.assertEqual(payload["implementation_version"], 12)
            restored.load(path)
            for name in NETWORKS + OPTIMIZERS:
                assert_tree_equal(self, getattr(source, name).state_dict(), getattr(restored, name).state_dict())
            payload["smooth_reward_model"] = {"corrupt": True}
            # If checks run after loading networks, this sentinel will change.
            with torch.no_grad():
                next(restored.world_model.parameters()).add_(.1)
            expected = copy.deepcopy(restored.world_model.state_dict())
            torch.save(payload, path)
            with self.assertRaises(ValueError):
                restored.load(path)
            assert_tree_equal(self, expected, restored.world_model.state_dict())

    def test_ordinary_new_mode_resume_keeps_saved_new_reward_labels(self):
        source = smoothed(checkpoint_runner())
        source.cfg.checkpoint_replay = True
        source.replay = ReplayBuffer(2, 2, "cpu")
        state = torch.zeros(2, 71); state[:, 3] = 1.; state[:, 2] = .7
        labels = torch.tensor([-.01, .027])
        source.replay.add(dict(state=state, next_state=state+.001, reward=labels,
                               terminated=torch.zeros(2, dtype=torch.bool)))
        restored = smoothed(checkpoint_runner())
        restored.replay = ReplayBuffer(2, 2, "cpu")
        with tempfile.TemporaryDirectory() as folder, patch("torch.cuda.is_available", return_value=False):
            path = Path(folder)/"new_reward_replay.pt"
            source.save(path, export=False); restored.load(path)
        self.assertEqual(restored.replay.size, 1)
        torch.testing.assert_close(restored.replay.data["reward"][0], labels, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
