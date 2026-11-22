"""CPU checks of optimized imagination, update ordering and initialization."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import shutil
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
import tyro

from rsl_rl.modules.discriminator import Discriminator
from rsl_rl.utils.utils import Normalizer
from src.wmac.config import WmacConfig
from src.wmac.networks import Actor, Critic, RunningNorm, WorldModel, frozen_parameters
from src.wmac.runner import WmacRunner, differentiable_amp_reward
from test_wmac_state_fast import legacy_kinematics, make_adapter


class FixedReplay:
    def __init__(self, values):
        self.values = values

    def sample_starts(self, size, recovery_fraction, fields=None, split="train"):
        if size != len(self.values["state"]):
            raise AssertionError("The test's paired batch was unexpectedly resampled")
        keys = tuple(self.values) if fields is None else fields
        return {key: self.values[key] for key in keys}


def make_runner(batch_size=5, horizon=3, dtype=torch.float64):
    """Actual G1 geometry and networks, without constructing a physics environment."""
    torch.manual_seed(43)
    adapter, _ = make_adapter(dtype=dtype)
    runner = object.__new__(WmacRunner)
    runner.cfg = WmacConfig(device="cpu", replay_device="cpu", precision="float32", compile=False,
        actor_batch_size=batch_size, imagination_horizon=horizon, actor_microbatch_size=0,
        model_hidden=(16,), policy_hidden=(16, 8), critic_hidden=(16, 8),
        discriminator_hidden=(12, 6), real_critic_coef=0., critic_epochs=1, critic_minibatches=2,
        reward_chunk_size=4, max_grad_norm=1e6, actor_grad_value_clip=1e6,
        max_action_shift=100., gamma=.5, gae_lambda=.5)
    runner.adapter = adapter
    runner.device = torch.device("cpu")
    runner.actor = Actor(hidden=(16, 8), action_limit=5.).to(dtype=dtype)
    runner.world_model = WorldModel(hidden=(16,)).to(dtype=dtype)
    runner.critic = Critic(136, hidden=(16, 8)).to(dtype=dtype)
    runner.discriminator = Discriminator(390, .1, [12, 6], "cpu", .75).to(dtype=dtype)
    runner.amp_norm = RunningNorm(195).to(dtype=dtype)
    with torch.no_grad():
        # A nonzero fitted residual is necessary to test the dynamics derivative.
        runner.world_model.net[-1].weight.normal_(0., .002)
        runner.discriminator.amp_linear.weight.mul_(.1)
        runner.discriminator.amp_linear.bias.zero_()
    runner.opt_actor = torch.optim.SGD(runner.actor.parameters(), lr=.02)
    runner.opt_model = torch.optim.Adam(runner.world_model.parameters(), lr=1e-3)
    runner.opt_disc = torch.optim.Adam(runner.discriminator.parameters(), lr=1e-3)
    runner.opt_critic = torch.optim.Adam(runner.critic.parameters(), lr=1e-3)
    runner._imag_step_fn = runner._imagined_step
    runner._reward_fn = runner._trajectory_rewards
    runner._value_fn = runner.critic.forward
    runner.imagination_steps = 0
    generator = torch.Generator().manual_seed(28)
    state = torch.zeros(batch_size, 71, dtype=dtype)
    state[:, 2] = torch.linspace(.6, .8, batch_size, dtype=dtype)
    state[:, 3] = 1.
    state[:, 7] = torch.linspace(.05, .15, batch_size, dtype=dtype)
    state[:, 13:42] = adapter.default_joint_pos
    context = torch.zeros(batch_size, 35, dtype=dtype)
    context[:, 0] = torch.linspace(.4, 1., batch_size, dtype=dtype)
    context[-1, 33:35] = context.new_tensor([1., 2.])
    previous = torch.randn(batch_size, 29, dtype=dtype, generator=generator) * .01
    command = torch.randn(batch_size, 3, dtype=dtype, generator=generator) * .1
    history = adapter.actor_frame(state, previous, command, context).unsqueeze(1).repeat(1, 4, 1)
    values = {"state": state, "context": context, "previous_action": previous,
              "command": command, "actor_history": history.flatten(1)}
    runner.replay = FixedReplay(values)
    noises = torch.randn(horizon, batch_size, 96, dtype=dtype, generator=generator) * .001
    return runner, values, noises


def step_oracle(runner, batch, noises):
    """Original per-step computation, using independent body-loop FK.

    The oracle deliberately does not call imagine, the shared reward head or
    combined step_with_contact. It integrates rewards before each survival
    update and bootstraps only after the final transition.
    """
    cfg, adapter = runner.cfg, runner.adapter
    state, context = batch["state"], batch["context"]
    command, previous = batch["command"], batch["previous_action"]
    history = batch["actor_history"].reshape(-1, 4, 96)
    objective = torch.zeros(len(state), dtype=state.dtype)
    alive = torch.ones_like(objective)
    tasks, styles, scores = [], [], []
    for step in range(cfg.imagination_horizon):
        action = runner.actor(history.flatten(1))
        next_state = runner.world_model.step(state, action, context)
        contact = runner.world_model.contact_cost(state, action, context)
        next_context, terminated = adapter.advance_context(next_state, context)
        first_fk, next_fk = legacy_kinematics(adapter, state), legacy_kinematics(adapter, next_state)
        task = adapter.task_reward(state, next_state, action, previous, command,
            next_context, terminated, contact, kinematics=next_fk)
        style, score = differentiable_amp_reward(runner.discriminator, runner.amp_norm,
            adapter.amp_features(state, kinematics=first_fk),
            adapter.amp_features(next_state, kinematics=next_fk), cfg.amp_reward_coef)
        direct = style if cfg.direct_style_grad else style.detach()
        objective = objective + cfg.gamma**step * alive * (cfg.task_weight * task + cfg.style_weight * direct)
        continuation = 1 - terminated
        alive = alive * (continuation if cfg.survival_grad else continuation.detach())
        observation = adapter.actor_frame(next_state, action, command, next_context) + noises[step]
        history = torch.cat((history[:, 1:], observation[:, None]), 1)
        state, context, previous = next_state, next_context, action
        tasks.append(task); styles.append(style); scores.append(score)
    if cfg.critic_bootstrap:
        terminal = runner.critic(adapter.critic_features(state, previous, command, context))
        objective = objective + cfg.gamma**cfg.imagination_horizon * alive * terminal
    return objective, torch.stack(tasks), torch.stack(styles), torch.stack(scores)


def fix_update_noise(runner, full_noise):
    cursor = 0
    def paired_noise(count):
        nonlocal cursor
        result = full_noise[:, cursor:cursor + count]
        cursor += count
        return result
    runner._observation_noise = paired_noise


class OptimizedImaginationTest(unittest.TestCase):
    def test_batched_rewards_and_actor_gradients_match_per_step_geometry_oracle(self):
        for direct_style, survival in ((True, True), (False, True), (True, False)):
            with self.subTest(direct_style=direct_style, survival=survival):
                runner, batch, noises = make_runner(batch_size=3)
                runner.cfg.direct_style_grad = direct_style
                runner.cfg.survival_grad = survival
                with frozen_parameters(runner.world_model, runner.discriminator, runner.critic):
                    with patch.object(runner.adapter, "analyze_state", wraps=runner.adapter.analyze_state) as fk:
                        actual = runner.imagine(batch, noises=noises)
                    self.assertEqual(fk.call_count, 1)
                    expected, task, style, score = step_oracle(runner, batch, noises)
                    for new, old in ((actual["objective"], expected), (actual["task"], task),
                                     (actual["style"], style), (actual["score"], score)):
                        torch.testing.assert_close(new, old, rtol=2e-10, atol=2e-11)
                    parameters = tuple(runner.actor.parameters())
                    actual_grad = torch.autograd.grad(actual["objective"].sum(), parameters)
                    expected_grad = torch.autograd.grad(expected.sum(), parameters)
                    self.assertGreater(sum(float(g.square().sum()) for g in actual_grad), 0.)
                    for new, old in zip(actual_grad, expected_grad):
                        self.assertTrue(torch.isfinite(new).all())
                        torch.testing.assert_close(new, old, rtol=3e-8, atol=3e-10)
                for module in (runner.world_model, runner.discriminator, runner.critic):
                    self.assertTrue(all(p.grad is None for p in module.parameters()))

    def test_uneven_microbatches_match_one_whole_batch_optimizer_step(self):
        whole, _, whole_noise = make_runner(batch_size=5)
        micro, _, micro_noise = make_runner(batch_size=5)
        micro.cfg.actor_microbatch_size = 2  # 2 + 2 + 1: incorrect unweighted means fail.
        fix_update_noise(whole, whole_noise)
        fix_update_noise(micro, micro_noise)
        whole_result = whole.update_actor_critic()
        micro_result = micro.update_actor_critic()
        self.assertEqual(whole_result["actor_update_rejected"], 0.)
        self.assertEqual(micro_result["actor_update_rejected"], 0.)
        self.assertAlmostEqual(whole_result["actor_loss"], micro_result["actor_loss"], places=11)
        for expected, actual in zip(whole.actor.parameters(), micro.actor.parameters()):
            torch.testing.assert_close(actual, expected, rtol=2e-10, atol=2e-11)
        self.assertEqual(whole.imagination_steps, 15)
        self.assertEqual(micro.imagination_steps, 15)

    def test_targets_are_fixed_before_actor_step_and_other_networks_stay_frozen(self):
        runner, _, noises = make_runner()
        fix_update_noise(runner, noises)
        events, targets = [], []
        original_imagine, original_step = runner.imagine, runner.opt_actor.step
        def observed_imagine(*args, **kwargs):
            for module in (runner.world_model, runner.discriminator, runner.critic):
                self.assertTrue(all(not p.requires_grad for p in module.parameters()))
            result = original_imagine(*args, **kwargs)
            self.assertFalse(result["x"].requires_grad)
            self.assertFalse(result["targets"].requires_grad)
            self.assertIsNone(result["targets"].grad_fn)
            targets.append(result["targets"].clone())
            events.append("cached_targets")
            return result
        def observed_step(*args, **kwargs):
            self.assertTrue(targets)
            self.assertEqual(events[-1], "cached_targets")
            for module in (runner.world_model, runner.discriminator, runner.critic):
                self.assertTrue(all(p.grad is None for p in module.parameters()))
            events.append("actor_step")
            return original_step(*args, **kwargs)
        def observed_fit(x, y):
            events.append("fit_critic")
            self.assertEqual(events[-2], "actor_step")
            self.assertFalse(x.requires_grad)
            self.assertFalse(y.requires_grad)
            torch.testing.assert_close(y, torch.cat(targets, 1))
            return {"critic_loss": 0., "real_critic_loss": 0., "critic_grad_norm": 0.,
                    "critic_updates_applied_fraction": 0.}
        with patch.object(runner, "imagine", side_effect=observed_imagine), \
             patch.object(runner.opt_actor, "step", side_effect=observed_step), \
             patch.object(runner, "_fit_critic", side_effect=observed_fit):
            runner.update_actor_critic()
        self.assertEqual(events, ["cached_targets", "actor_step", "fit_critic"])

    def test_critic_only_calibration_updates_value_but_leaves_actor_untouched(self):
        runner, _, noises = make_runner()
        fix_update_noise(runner, noises)
        actor_before = {k: v.clone() for k, v in runner.actor.state_dict().items()}
        critic_before = {k: v.clone() for k, v in runner.critic.state_dict().items()}
        result = runner.update_actor_critic(update_actor=False)
        for key, value in runner.actor.state_dict().items():
            torch.testing.assert_close(value, actor_before[key], rtol=0, atol=0)
        self.assertTrue(any(not torch.equal(value, critic_before[key])
                            for key, value in runner.critic.state_dict().items()))
        self.assertEqual(result["actor_grad_norm"], 0.)
        self.assertEqual(result["actor_action_shift"], 0.)
        self.assertEqual(result["critic_updates_applied_fraction"], 1.)


class WarmstartAndScheduleTest(unittest.TestCase):
    def test_baseline_discriminator_and_variance_epsilon_clipping_are_preserved(self):
        runner, _, _ = make_runner(dtype=torch.float32)
        source_actor = copy.deepcopy(runner.actor)
        source_disc = copy.deepcopy(runner.discriminator)
        source_norm = Normalizer(195, epsilon=.04, clip_obs=2.5)
        source_norm.mean = np.linspace(-.5, .8, 195)
        source_norm.var = np.linspace(.001, 4., 195)
        source_norm.count = 12345.
        payload = {"model_state_dict": {f"actor.{k}": v.clone() for k, v in source_actor.net.state_dict().items()},
                   "discriminator_state_dict": source_disc.state_dict(), "amp_normalizer": source_norm}
        with torch.no_grad():
            for parameter in runner.discriminator.parameters():
                parameter.add_(1.)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "baseline.pt"
            torch.save(payload, path)
            runner.initialize_from_policy(path)
        first = torch.randn(7, 195) * 4
        second = torch.randn(7, 195) * 4
        task = torch.linspace(-1., 1., 7)
        expected_normalized = source_norm.normalize_torch(first, "cpu")
        torch.testing.assert_close(runner.amp_norm(first), expected_normalized, rtol=1e-6, atol=5e-7)
        self.assertTrue(runner.amp_norm.variance_epsilon)
        self.assertEqual(runner.amp_norm.clip, 2.5)
        self.assertEqual(float(runner.amp_norm.count), 12345.)
        old_reward, _ = source_disc.predict_amp_reward(first, second, task, source_norm)
        new_style, _ = runner._amp_reward(first, second)
        new_reward = runner.cfg.task_weight * task + runner.cfg.style_weight * new_style
        torch.testing.assert_close(new_reward, old_reward, rtol=1e-6, atol=1e-7)
        for key, value in source_disc.state_dict().items():
            torch.testing.assert_close(runner.discriminator.state_dict()[key], value, rtol=0, atol=0)

    def test_utd_point_four_collects_two_then_three_steps_without_drift(self):
        runner = object.__new__(WmacRunner)
        runner.cfg = WmacConfig(utd=.4, collect_steps=0)
        runner._collect_accumulator = 0.
        counts = [runner._next_collect_count() for _ in range(100)]
        self.assertEqual(counts[:6], [2, 3, 2, 3, 2, 3])
        self.assertEqual(sum(counts), 250)
        self.assertAlmostEqual(runner._collect_accumulator, 0.)
        runner.cfg.collect_steps = 8
        self.assertEqual([runner._next_collect_count() for _ in range(4)], [8, 8, 8, 8])

    def test_cosine_schedule_has_correct_endpoints_and_updates_every_optimizer(self):
        runner, _, _ = make_runner()
        runner.cfg.lr_decay_iterations = 10
        runner.cfg.lr_final_fraction = .03
        self.assertEqual(runner._lr_factor(0), 1.)
        self.assertAlmostEqual(runner._lr_factor(5), .515)
        self.assertEqual(runner._lr_factor(10), .03)
        self.assertEqual(runner._lr_factor(20), .03)
        runner.iteration = 5
        runner._set_learning_rates()
        for optimizer, base in ((runner.opt_actor, runner.cfg.actor_lr), (runner.opt_model, runner.cfg.world_model_lr),
                                (runner.opt_critic, runner.cfg.critic_lr), (runner.opt_disc, runner.cfg.discriminator_lr)):
            for group in optimizer.param_groups:
                self.assertAlmostEqual(group["lr"], base * .515)

    def test_explicit_cli_options_override_supermimic_profile_defaults(self):
        cfg = tyro.cli(WmacConfig, default=WmacConfig.for_profile("supermimic"), args=[
            "--profile", "supermimic", "--wm-window", "4", "--imagination-horizon", "7",
            "--num-envs", "12", "--policy-hidden", "8", "4", "--no-compile", "--precision", "float32",
        ])
        cfg.apply_preset(); cfg.validate()
        self.assertEqual((cfg.wm_window, cfg.imagination_horizon, cfg.num_envs), (4, 7, 12))
        self.assertEqual(cfg.policy_hidden, (8, 4))
        self.assertFalse(cfg.compile)
        self.assertEqual(cfg.precision, "float32")
        self.assertEqual(cfg.model_hidden, (2048, 2048, 2048))


def control_runner(iterations, model_warmup=0, critic_warmup=2, gate_after=0,
                   failed_calibration=None):
    """Run the real learn scheduler against deterministic inexpensive stages."""
    runner = object.__new__(WmacRunner)
    runner.cfg = WmacConfig(iterations=iterations, model_warmup_iterations=model_warmup,
        critic_warmup_iterations=critic_warmup, collect_steps=0, utd=.4,
        compile=False, device="cpu", replay_device="cpu", evaluation_interval=0)
    runner.device = torch.device("cpu")
    runner.iteration = runner.sim_steps = runner.imagination_steps = 0
    runner.critic_calibration_updates = 0
    runner._collect_accumulator = 0.
    runner.replay = SimpleNamespace(size=runner.cfg.wm_window)
    runner.env = SimpleNamespace(num_envs=2)
    runner.writer = SimpleNamespace(close=lambda: None)
    runner.csv_file = None
    runner.log_dir = Path("unused_control_flow_test")
    runner.validation = {"validation_model_error": 1., "validation_contact_error": .1,
        "validation_walk_error": 1., "validation_recovery_error": 1., "validation_reward_gap": .1}
    for name in ("opt_actor", "opt_critic", "opt_model", "opt_disc"):
        setattr(runner, name, SimpleNamespace(param_groups=[{"lr": 0.}]))
    records, updates = [], []
    def collect(count):
        runner.sim_steps += count * runner.env.num_envs
        return {"real_task_reward": 0.}
    runner.collect = collect
    runner.fit_model = lambda: {"wm_gate_error": 1. if runner.iteration >= gate_after else float("inf"),
                               "wm_updates_applied_fraction": 1., "wm_loss": 1.}
    runner.update_discriminator = lambda: {}
    runner.validate_model = lambda: runner.validation
    runner._log = lambda row: records.append(dict(row))
    runner.save = lambda *args, **kwargs: None
    def update(update_actor):
        updates.append((runner.iteration, update_actor))
        result = {key: 0. for key in ("actor_loss", "actor_grad_norm", "actor_update_rejected",
            "actor_action_shift", "critic_loss", "real_critic_loss", "critic_grad_norm",
            "critic_updates_applied_fraction", "imagined_task_reward", "imagined_style_reward",
            "style_zero_gradient_fraction", "imagined_survival")}
        result["critic_loss"] = 1.
        result["critic_updates_applied_fraction"] = 1.
        if runner.iteration == 0 and failed_calibration:
            result.update(failed_calibration)
        return result
    runner.update_actor_critic = update
    return runner, records, updates


class SchedulerAndCheckpointRegressionTest(unittest.TestCase):
    def test_nonfinite_overall_validation_cannot_be_hidden_by_finite_phases(self):
        runner, _, _ = control_runner(1)
        row = {"wm_gate_error": 1., "wm_updates_applied_fraction": 1.}
        self.assertTrue(runner._actor_gate(row))
        for key in ("validation_model_error", "validation_contact_error"):
            for invalid in (float("nan"), float("inf")):
                with self.subTest(key=key, invalid=invalid):
                    before = runner.validation[key]
                    runner.validation[key] = invalid
                    self.assertFalse(runner._actor_gate(row))
                    runner.validation[key] = before

    def test_zero_model_warmup_still_requires_actual_critic_calibration(self):
        runner, records, updates = control_runner(3, model_warmup=0, critic_warmup=2)
        runner.learn()
        self.assertEqual(updates, [(0, False), (1, False), (2, True)])
        self.assertEqual([row["actor_active"] for row in records], [0., 0., 1.])
        self.assertEqual([row["critic_calibration_updates"] for row in records], [1, 2, 2])

    def test_late_model_gate_does_not_bypass_the_unfinished_critic_warmup(self):
        runner, records, updates = control_runner(5, model_warmup=1, critic_warmup=2, gate_after=2)
        runner.learn()
        self.assertEqual(updates, [(2, False), (3, False), (4, True)])
        self.assertEqual([row["critic_calibration_updates"] for row in records], [0, 0, 1, 2, 2])
        self.assertEqual([row["actor_active"] for row in records], [0., 0., 0., 0., 1.])

    def test_rejected_or_nonfinite_critic_calibration_is_not_counted(self):
        for failure in ({"critic_updates_applied_fraction": 0.}, {"critic_loss": float("nan")}):
            with self.subTest(failure=failure):
                runner, records, updates = control_runner(3, critic_warmup=1,
                                                          failed_calibration=failure)
                runner.learn()
                self.assertEqual(updates, [(0, False), (1, False), (2, True)])
                self.assertEqual([row["critic_calibration_updates"] for row in records], [0, 1, 1])

    def test_resume_preserves_fractional_collection_and_calibration_progress(self):
        source, _, _ = make_runner()
        restored, _, _ = make_runner()
        for runner in (source, restored):
            runner.env = SimpleNamespace(common_step_counter=42)
            runner.started, runner.wall_offset = time.perf_counter(), 0.
            runner.iteration, runner.sim_steps = 17, 128
            runner.best_eval_score = (.5, .5, .5, -.1, -.4)
            runner.validation = {"validation_model_error": float("inf"),
                "validation_contact_error": float("inf"), "validation_walk_error": float("inf"),
                "validation_recovery_error": float("inf"), "validation_reward_gap": float("inf")}
        source._collect_accumulator = .5
        source.critic_calibration_updates = 3
        source.imagination_steps = 45
        restored._collect_accumulator = 0.
        restored.critic_calibration_updates = 0
        with tempfile.TemporaryDirectory() as directory, patch("torch.cuda.is_available", return_value=False):
            path = Path(directory) / "roundtrip.pt"
            source.save(path, export=False)
            payload = torch.load(path, map_location="cpu", weights_only=False)
            self.assertEqual(payload["critic_calibration_updates"], 3)
            self.assertEqual(payload["collect_accumulator"], .5)
            restored.load(path)
        self.assertEqual(restored.critic_calibration_updates, 3)
        self.assertEqual(restored._collect_accumulator, .5)
        self.assertEqual(restored.imagination_steps, 45)
        self.assertEqual(restored.best_eval_score, source.best_eval_score)
        self.assertEqual(restored._next_collect_count(), 3)
        self.assertTrue(math.isinf(restored.validation["validation_model_error"]))

    def test_new_best_score_is_saved_before_the_best_checkpoint_is_copied(self):
        runner = object.__new__(WmacRunner)
        runner.cfg = WmacConfig(device="cpu", evaluation_envs=4, evaluation_steps=10)
        runner.device = torch.device("cpu")
        runner.env = SimpleNamespace(step_dt=.02)
        runner.iteration, runner.best_eval_score = 12, None
        expected_score = (.5, .75, .875, -.2, -.4)
        events = []
        original_copy = shutil.copy2
        with tempfile.TemporaryDirectory() as directory:
            runner.log_dir = Path(directory)
            checkpoint = runner.log_dir / "model_12.pt"
            torch.save({"best_eval_score": None}, checkpoint)
            def evaluate(command, **kwargs):
                self.assertEqual(command[command.index("--device") + 1], "cpu")
                output = Path(command[command.index("--output") + 1])
                output.mkdir(parents=True, exist_ok=True)
                result = {"results": {
                    "recovery": {"recovery_success_fraction": .5, "final_upright_fraction": .75,
                                 "first_stable_recovery_seconds_mean": .4},
                    "walk": {"first_episode_survival_fraction": .875, "velocity_xy_error_mean": .2}}}
                (output / "metrics.json").write_text(json.dumps(result))
                events.append("evaluate")
            def save(path, export=True):
                self.assertFalse(export)
                self.assertEqual(runner.best_eval_score, expected_score)
                torch.save({"best_eval_score": runner.best_eval_score}, path)
                events.append("save_updated_score")
            def copy_checkpoint(source, target):
                self.assertEqual(torch.load(source, weights_only=False)["best_eval_score"], expected_score)
                events.append("copy_best")
                return original_copy(source, target)
            with patch("src.wmac.runner.subprocess.run", side_effect=evaluate), \
                 patch.object(runner, "save", side_effect=save), \
                 patch("src.wmac.runner.shutil.copy2", side_effect=copy_checkpoint) as copying:
                runner.evaluate_checkpoint()
                self.assertEqual(events, ["evaluate", "save_updated_score", "copy_best"])
                self.assertEqual(torch.load(runner.log_dir / "best_model.pt", weights_only=False)["best_eval_score"], expected_score)
                selection = json.loads((runner.log_dir / "best_selection.json").read_text())
                self.assertEqual(selection["score"], list(expected_score))
                self.assertEqual(selection["iteration"], 12)
                runner.evaluate_checkpoint()  # An equal score must not overwrite the best.
                self.assertEqual(copying.call_count, 1)


if __name__ == "__main__":
    unittest.main()
