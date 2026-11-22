"""Production-path checks for the source-derived centered AMP reward.

These use real G1 differentiable geometry and a small existing WMAC runner,
without starting a simulator or adding a training branch.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from src.wmac.amp_rewards import CenteredAmpReward
from src.wmac.networks import frozen_parameters
from src.wmac.replay import ReplayBuffer
from test_wmac_optimization import FixedReplay, control_runner, fix_update_noise, make_runner
from test_wmac_schedule_restore import checkpoint_runner


MODE = "instantamp_centered_tanh_v1"
NETWORKS = ("actor", "critic", "world_model", "discriminator", "amp_norm")
OPTIMIZERS = ("opt_actor", "opt_critic", "opt_model", "opt_disc")


def centered(runner, *, mean=-2.):
    runner.cfg.amp_reward_mapping = MODE
    runner.amp_reward_stats = CenteredAmpReward(
        coefficient=runner.cfg.amp_reward_coef,
        scale=runner.cfg.amp_score_scale,
        decay=runner.cfg.amp_score_ema_decay,
    ).to(device=runner.device)
    if mean is not None:
        runner.amp_reward_stats.update(torch.tensor([mean]), torch.tensor([1.]))
    return runner


def assert_tree_equal(test, first, second):
    if isinstance(first, torch.Tensor):
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    elif isinstance(first, dict):
        test.assertEqual(set(first), set(second))
        for key in first:
            assert_tree_equal(test, first[key], second[key])
    elif isinstance(first, (list, tuple)):
        test.assertEqual(len(first), len(second))
        for a, b in zip(first, second):
            assert_tree_equal(test, a, b)
    else:
        test.assertEqual(first, second)


class CenteredAmpIntegrationTest(unittest.TestCase):
    def test_first_failed_d_step_defers_actor_without_extra_sampling_or_calibration(self):
        runner, records, updates = control_runner(3, critic_warmup=0)
        runner.cfg.amp_reward_mapping = MODE
        valid_model = {"wm_gate_error": 1., "wm_updates_applied_fraction": 1.}
        self.assertFalse(runner._actor_gate(valid_model))
        for invalid in (0., -1., float("nan"), float("inf")):
            self.assertFalse(runner._actor_gate({**valid_model, "amp_score_initialized": invalid}))
        self.assertTrue(runner._actor_gate({**valid_model, "amp_score_initialized": 1.}))
        calls = []
        def discriminator():
            calls.append(runner.iteration)
            # First ordinary discriminator update failed, so no real score
            # statistic exists yet. The next ordinary update initializes it.
            return {"amp_score_initialized": float(runner.iteration >= 1)}
        runner.update_discriminator = discriminator
        runner.learn()
        self.assertEqual(calls, [0, 1, 2])
        self.assertEqual(updates, [(1, True), (2, True)])
        self.assertEqual([row["actor_active"] for row in records], [0., 1., 1.])
        self.assertEqual([row["collect_steps"] for row in records], [2, 3, 2])
        self.assertEqual(runner.sim_steps, 14)
        self.assertEqual(runner.critic_calibration_updates, 0)

    def test_low_scoring_getup_style_has_actor_gradient_through_frozen_wm_and_discriminator(self):
        gradients = {}
        for mode in ("clipped_lsgan_v1", MODE):
            runner, batch, noise = make_runner(batch_size=3)
            runner.cfg.task_weight = 0.
            runner.cfg.style_weight = 1.
            runner.cfg.critic_bootstrap = False
            runner.cfg.survival_grad = False
            if mode == MODE:
                centered(runner)
            # The discriminator's state-dependent slope remains nonzero while
            # all scores sit below the old hard reward threshold of -1.
            with torch.no_grad():
                runner.discriminator.amp_linear.bias.fill_(-2.)
            before = {name: copy.deepcopy(getattr(runner, name).state_dict())
                      for name in ("world_model", "discriminator", "critic")}
            calls = []
            handle = runner.discriminator.register_forward_hook(lambda *args: calls.append(1))
            with frozen_parameters(runner.world_model, runner.discriminator, runner.critic):
                result = runner.imagine(batch, noises=noise)
                self.assertTrue((result["score"] < -1.).all())
                grads = torch.autograd.grad(result["objective"].sum(), tuple(runner.actor.parameters()))
                self.assertTrue(all(torch.isfinite(g).all() for g in grads))
                gradients[mode] = sum(float(g.square().sum()) for g in grads)
            handle.remove()
            self.assertEqual(len(calls), math.ceil(9 / runner.cfg.reward_chunk_size))
            for name, expected in before.items():
                assert_tree_equal(self, expected, getattr(runner, name).state_dict())
                self.assertTrue(all(parameter.grad is None for parameter in getattr(runner, name).parameters()))
        self.assertEqual(gradients["clipped_lsgan_v1"], 0.)
        self.assertGreater(gradients[MODE], 1e-20)

    def test_actor_and_cached_critic_target_use_the_same_centered_forward_reward(self):
        runner, batch, noise = make_runner(batch_size=3)
        centered(runner, mean=-.7)
        runner.cfg.task_weight = 0.
        runner.cfg.style_weight = 1.
        runner.cfg.gamma = 0.  # Isolates the actual reward in every TD target.
        runner.cfg.critic_bootstrap = False
        with frozen_parameters(runner.world_model, runner.discriminator, runner.critic):
            result = runner.imagine(batch, noises=noise)
            with torch.no_grad():
                cold = runner.imagine(batch, noises=noise)
        expected_style = runner.amp_reward_stats(result["score"])
        torch.testing.assert_close(result["style"], expected_style)
        torch.testing.assert_close(result["targets"], expected_style)
        torch.testing.assert_close(result["objective"], expected_style[0])
        for key in ("objective", "style", "score", "targets"):
            torch.testing.assert_close(result[key], cold[key])
        self.assertFalse(result["targets"].requires_grad)
        self.assertEqual(int(runner.amp_reward_stats.updates), 1)

    def test_discriminator_reuses_existing_real_scores_without_additional_forward_or_optimizer_steps(self):
        runner, values, _ = make_runner(batch_size=3, dtype=torch.float32)
        centered(runner, mean=None)
        runner.cfg.amp_normalizer_per_update = True
        runner.cfg.discriminator_updates = 2
        runner.cfg.discriminator_batch_size = 3
        values["next_state"] = values["state"].clone()
        values["next_state"][:, 2] += .03
        runner.replay = FixedReplay(values)
        ef = torch.linspace(-1., 2., 3 * 195).reshape(3, 195)
        runner.experts = SimpleNamespace(sample=lambda size: (ef, ef + .2))
        scores = []
        handle = runner.discriminator.register_forward_hook(
            lambda module, args, output: scores.append(output.detach().clone()))
        with patch.object(runner.opt_disc, "step", wraps=runner.opt_disc.step) as step:
            runner.update_discriminator()
        handle.remove()
        self.assertEqual(len(scores), 4)  # Robot + expert, twice; no calibration pass.
        self.assertEqual(step.call_count, 2)
        self.assertEqual(int(runner.amp_reward_stats.updates), 2)
        decay = runner.cfg.amp_score_ema_decay
        expected_robot = scores[0].mean() * decay + scores[2].mean() * (1 - decay)
        expected_expert = scores[1].mean() * decay + scores[3].mean() * (1 - decay)
        torch.testing.assert_close(runner.amp_reward_stats.robot_mean, expected_robot)
        torch.testing.assert_close(runner.amp_reward_stats.expert_mean, expected_expert)
        self.assertTrue(bool(runner.amp_reward_stats.initialized))

    def test_rejected_discriminator_update_does_not_advance_score_statistics(self):
        runner, values, _ = make_runner(batch_size=3, dtype=torch.float32)
        centered(runner)
        runner.cfg.amp_normalizer_per_update = True
        runner.cfg.discriminator_updates = 1
        runner.cfg.discriminator_batch_size = 3
        values["next_state"] = values["state"].clone()
        runner.replay = FixedReplay(values)
        expert = torch.zeros(3, 195)
        runner.experts = SimpleNamespace(sample=lambda size: (expert, expert))
        before = copy.deepcopy(runner.amp_reward_stats.state_dict())
        with patch.object(runner, "_optimizer_step", return_value=(torch.tensor(0.), False)):
            runner.update_discriminator()
        assert_tree_equal(self, before, runner.amp_reward_stats.state_dict())

    def test_actor_update_does_not_train_world_model_or_modify_score_ema(self):
        runner, _, noises = make_runner(batch_size=3)
        centered(runner)
        fix_update_noise(runner, noises)
        before = {name: copy.deepcopy(getattr(runner, name).state_dict())
                  for name in ("world_model", "discriminator", "amp_norm", "opt_model", "opt_disc", "amp_reward_stats")}
        result = runner.update_actor_critic()
        self.assertEqual(result["actor_update_rejected"], 0.)
        self.assertTrue(math.isfinite(result["actor_grad_norm"]))
        self.assertGreater(result["actor_grad_norm"], 0.)
        for name, expected in before.items():
            assert_tree_equal(self, expected, getattr(runner, name).state_dict())


class CenteredAmpCheckpointIntegrationTest(unittest.TestCase):
    def test_mapping_migration_keeps_real_replay_task_labels_and_cuts_old_simulator_tail(self):
        source = checkpoint_runner()
        source.cfg.checkpoint_replay = True
        source.replay = ReplayBuffer(2, 2, "cpu")
        state = torch.zeros(2, 71)
        state[:, 2] = torch.tensor([.3, .7])
        state[:, 3] = 1.
        for step in range(2):
            current = state.clone()
            current[:, 2] += step * .01
            source.replay.add({"state": current, "next_state": current + .001,
                "context": torch.zeros(2, 35), "reward": torch.tensor([.02, .06]) + step * .001})
        restored = centered(checkpoint_runner(), mean=None)
        restored.cfg.allow_amp_reward_migration = True
        restored.replay = ReplayBuffer(2, 2, "cpu")
        with tempfile.TemporaryDirectory() as directory, patch("torch.cuda.is_available", return_value=False):
            path = Path(directory) / "real_replay.pt"
            source.save(path, export=False)
            restored.load(path)
        self.assertEqual((restored.replay.size, restored.replay.ptr, restored.replay.total_steps), (2, 0, 2))
        assert_tree_equal(self, source.replay.data, restored.replay.data)
        self.assertTrue(restored.replay._cut_rows[-1].all())
        self.assertTrue(restored.amp_reward_migration_history[-1]["replay_preserved_if_saved"])

    def test_explicit_mapping_migration_preserves_all_networks_optimizers_and_counters(self):
        source = checkpoint_runner()
        # Populate every optimizer's real state when that optimizer has one.
        for name in OPTIMIZERS:
            optimizer = getattr(source, name)
            for group in optimizer.param_groups:
                for parameter in group["params"]:
                    parameter.grad = torch.full_like(parameter, .01)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        expected = {name: copy.deepcopy(getattr(source, name).state_dict()) for name in NETWORKS + OPTIMIZERS}
        restored = centered(checkpoint_runner(), mean=None)
        restored.cfg.allow_amp_reward_migration = True
        with tempfile.TemporaryDirectory() as directory, patch("torch.cuda.is_available", return_value=False):
            path = Path(directory) / "legacy.pt"
            source.save(path, export=False)
            restored.load(path)
        for name, state in expected.items():
            assert_tree_equal(self, state, getattr(restored, name).state_dict())
        self.assertEqual((restored.iteration, restored.sim_steps, restored.imagination_steps), (17, 128, 45))
        self.assertEqual(restored.critic_calibration_updates, 3)
        self.assertEqual(restored._collect_accumulator, source._collect_accumulator)
        self.assertFalse(bool(restored.amp_reward_stats.initialized))
        self.assertEqual(int(restored.amp_reward_stats.updates), 0)
        self.assertEqual(len(restored.amp_reward_migration_history), 1)

    def test_new_checkpoint_restores_exact_ema_and_cannot_silently_change_scale(self):
        source = centered(checkpoint_runner(), mean=-.8)
        source.amp_reward_stats.update(torch.tensor([-.1, -.2]), torch.tensor([.6, .7]))
        restored = centered(checkpoint_runner(), mean=None)
        with tempfile.TemporaryDirectory() as directory, patch("torch.cuda.is_available", return_value=False):
            path = Path(directory) / "centered.pt"
            source.save(path, export=False)
            payload = torch.load(path, weights_only=False)
            self.assertEqual(payload["implementation_version"], 11)
            self.assertIn("amp_reward_state", payload)
            restored.load(path)
            assert_tree_equal(self, source.amp_reward_stats.state_dict(), restored.amp_reward_stats.state_dict())
            restored.cfg.amp_score_scale = 2.
            before = copy.deepcopy(restored.world_model.state_dict())
            with self.assertRaises(ValueError):
                restored.load(path)
            assert_tree_equal(self, before, restored.world_model.state_dict())

    def test_corrupt_score_statistics_rejected_before_model_mutation(self):
        source = centered(checkpoint_runner(), mean=-.8)
        restored = centered(checkpoint_runner(), mean=None)
        with torch.no_grad():
            next(restored.world_model.parameters()).add_(.123)
        before = copy.deepcopy(restored.world_model.state_dict())
        with tempfile.TemporaryDirectory() as directory, patch("torch.cuda.is_available", return_value=False):
            path = Path(directory) / "corrupt.pt"
            source.save(path, export=False)
            payload = torch.load(path, weights_only=False)
            payload["amp_reward_state"]["robot_mean"].fill_(float("nan"))
            torch.save(payload, path)
            with self.assertRaises(ValueError):
                restored.load(path)
        assert_tree_equal(self, before, restored.world_model.state_dict())


if __name__ == "__main__":
    unittest.main()
