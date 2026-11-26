"""Opt-in recovery-style options: historical behaviour by default, guarded on resume."""
from __future__ import annotations

import copy
from dataclasses import asdict
import importlib.util
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from src.robot_profiles import G1
from src.wmac.actors import actor_architecture
from src.wmac.checkpoints import resolve_checkpoint_config
from src.wmac.config import WmacConfig
from src.wmac.networks import frozen_parameters
from src.wmac.recovery_style import (
    DEFAULT_PROTOCOL, RECOVERY_REFRESH_TERM, RECOVERY_STYLE_VERSION, RECOVERY_TEMPO_REWARD_TERM,
    RecoveryTempoCost, changes_reward, checkpoint_recovery_style_protocol, configure_recovery_style_env,
    delayed_recovery_refresh, motion_root_fingerprint, recovery_start_mask, recovery_style_enabled,
    recovery_style_protocol, reference_joint_speed_limits, scale_recovery_gradient,
    validate_recovery_style_resume,
)
from src.wmac.recovery_style import RECOVERY_PATIENCE_VERSION, standing_reward_ceiling, weighted_style
from src.wmac.recovery_style import implementation_version as recovery_style_implementation_version
from src.recovery_rewards import patient_recovery_height_reward, recovery_height_reward, recovery_root_up_z
from src.wmac.diagnostics import objective_gradient_probe
from src.wmac.reference import amp_reward_protocol, amp_value_upper_bound, config_protocol, smooth_reward_protocol
from test_centered_amp_integration import centered
from test_wmac_optimization import FixedReplay, make_runner
from test_wmac_schedule_restore import checkpoint_runner

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load_script(name):
    spec = importlib.util.spec_from_file_location(f"recovery_style_{name}", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def payload(cfg=None, *, version=12, record=True):
    cfg = cfg or WmacConfig.for_profile("supermimic_official")
    result = dict(
        format="amp-wmac-v1", implementation_version=version, config=asdict(cfg),
        training_protocol=config_protocol(cfg), amp_reward_protocol=amp_reward_protocol(cfg),
        smooth_reward_protocol=smooth_reward_protocol(cfg), actor_type=cfg.actor_type,
        actor_architecture=actor_architecture(cfg),
        recovery_counter_semantics=cfg.recovery_counter_semantics,
        recovery_height_reward=cfg.recovery_height_reward)
    if record and recovery_style_enabled(cfg):
        result["recovery_style_protocol"] = recovery_style_protocol(cfg)
    return result


def fixed(**changes):
    cfg = WmacConfig.for_profile("supermimic_official")
    for key, value in changes.items():
        setattr(cfg, key, value)
    return cfg


class RecoveryStyleProtocolTest(unittest.TestCase):
    def test_every_profile_defaults_to_the_historical_objective(self):
        for profile in ("balanced", "supermimic", "legacy", "ppo_aligned", "supermimic_official"):
            cfg = WmacConfig.for_profile(profile)
            self.assertEqual(recovery_style_protocol(cfg), DEFAULT_PROTOCOL)
            self.assertFalse(recovery_style_enabled(cfg))
        self.assertNotIn("recovery_tempo_base_rise_limit", DEFAULT_PROTOCOL)

    def test_checkpoint_written_before_the_options_resolves_to_defaults(self):
        source = payload()
        for key in tuple(source["config"]):
            if key in recovery_style_protocol(fixed(recovery_tempo_mode="reference_limits_v1")) \
                    or key in ("reference_motion_root", "allow_recovery_style_migration"):
                source["config"].pop(key)
        self.assertEqual(checkpoint_recovery_style_protocol(source), DEFAULT_PROTOCOL)
        restored = resolve_checkpoint_config(source)
        self.assertEqual(restored.recovery_value_grad_scale, 1.0)
        self.assertEqual(validate_recovery_style_resume(source, restored), {})

    def test_active_tempo_adds_its_parameters_and_invalid_values_are_rejected(self):
        protocol = recovery_style_protocol(fixed(recovery_tempo_mode="reference_limits_v1"))
        self.assertEqual(protocol["recovery_tempo_joint_speed_percentile"], 99.0)
        self.assertTrue(recovery_style_enabled(fixed(reference_motion_root="motions/clean")))
        for key, value in (("recovery_value_grad_scale", 1.5), ("recovery_value_grad_scale", -0.1),
                           ("recovery_height_reward_ratio", -1.0), ("recovery_tempo_mode", "fast"),
                           ("discriminator_imagined_fraction", 1.2), ("discriminator_penalty_points", "both"),
                           ("recovery_refresh_seconds", 0.0), ("recovery_max_action_shift", 0.0),
                           ("recovery_model_error_gate", float("nan")), ("wm_recovery_fraction", 2.0)):
            with self.assertRaises(ValueError, msg=key):
                fixed(**{key: value}).validate()

    def test_enabled_checkpoint_needs_its_version_and_recorded_protocol(self):
        cfg = fixed(recovery_value_grad_scale=0.1)
        self.assertEqual(checkpoint_recovery_style_protocol(payload(cfg, version=RECOVERY_STYLE_VERSION)),
                         recovery_style_protocol(cfg))
        with self.assertRaisesRegex(ValueError, "implementation_version"):
            checkpoint_recovery_style_protocol(payload(cfg, version=12))
        with self.assertRaisesRegex(ValueError, "metadata"):
            checkpoint_recovery_style_protocol(payload(cfg, version=RECOVERY_STYLE_VERSION, record=False))
        conflicting = payload(cfg, version=RECOVERY_STYLE_VERSION)
        conflicting["recovery_style_protocol"] = dict(DEFAULT_PROTOCOL)
        with self.assertRaisesRegex(ValueError, "conflicts"):
            checkpoint_recovery_style_protocol(conflicting)

    def test_resume_keeps_saved_settings_unless_the_fork_is_authorised(self):
        target = fixed(recovery_value_grad_scale=0.1, recovery_height_reward_ratio=2.0)
        with self.assertRaisesRegex(ValueError, "allow-recovery-style-migration"):
            validate_recovery_style_resume(payload(), target)
        target.allow_recovery_style_migration = True
        changes = validate_recovery_style_resume(payload(), target)
        self.assertEqual(changes["recovery_value_grad_scale"], {"source": 1.0, "target": 0.1})
        self.assertTrue(changes_reward(changes))
        self.assertFalse(changes_reward({"recovery_value_grad_scale": {}}))
        # The authorisation belongs to one invocation and is never restored from a checkpoint.
        saved = payload(target, version=RECOVERY_STYLE_VERSION)
        restored = resolve_checkpoint_config(saved)
        self.assertFalse(restored.allow_recovery_style_migration)
        self.assertEqual(restored.recovery_height_reward_ratio, 2.0)
        self.assertEqual(validate_recovery_style_resume(saved, restored), {})

    def test_migration_flag_requires_resume(self):
        cfg = fixed(allow_recovery_style_migration=True)
        with self.assertRaisesRegex(ValueError, "requires --resume"):
            cfg.validate()
        cli = load_script("train_wmac")
        with self.assertRaises(ValueError):
            cli.parse_config(["--profile", "supermimic_official", "--allow-recovery-style-migration"])

    def test_reference_root_gets_its_own_mirror_cache(self):
        cfg = fixed(reference_motion_root="src/assets/motions/g1/amp_clean")
        cfg.resolve_robot_defaults()
        self.assertEqual(cfg.motion_cache_root, G1.mirrored_motion_cache + "__amp_clean")
        cfg.resolve_robot_defaults()  # a restored configuration is not suffixed again
        self.assertEqual(cfg.motion_cache_root, G1.mirrored_motion_cache + "__amp_clean")


class RecoveryGradientScaleTest(unittest.TestCase):
    def test_values_are_unchanged_and_only_recovery_rows_are_scaled(self):
        features = torch.randn(4, 6, dtype=torch.float64, requires_grad=True)
        recovery = torch.tensor([False, True, False, True])
        weights = torch.randn(4, 6, dtype=torch.float64)
        record = []
        scaled = scale_recovery_gradient(features * 1.0, recovery, 0.25, record)
        torch.testing.assert_close(scaled, features)
        (scaled * weights).sum().backward()
        expected = weights * torch.tensor([1., .25, 1., .25], dtype=torch.float64)[:, None]
        torch.testing.assert_close(features.grad, expected)
        norms = weights.norm(dim=-1)
        torch.testing.assert_close(record[0], torch.stack((norms[[0, 2]].mean(), norms[[1, 3]].mean())).float())
        # Without a graph there is nothing to scale.
        detached = features.detach()
        self.assertIs(scale_recovery_gradient(detached, recovery, 0.25), detached)

    def test_start_mask_matches_replay_stratification(self):
        state = torch.zeros(3, 71); state[:, 2] = torch.tensor([.3, .7, .7])
        context = torch.zeros(3, 35); context[2, -1] = 4.
        self.assertEqual(recovery_start_mask(state, context, .55).tolist(), [True, False, True])

    def test_imagined_objective_keeps_its_value_and_scales_the_terminal_value_gradient(self):
        runner, values, noise = make_runner()
        recovery = recovery_start_mask(values["state"], values["context"], runner.recovery_height)
        self.assertEqual(recovery.tolist(), [False, False, False, False, True])

        def gradient(scale):
            runner.cfg.recovery_value_grad_scale = scale
            with frozen_parameters(runner.world_model, runner.discriminator, runner.critic):
                rollout = runner.imagine(values, noises=noise)
                objective = rollout["objective"]
                grads = torch.autograd.grad(objective.sum(), list(runner.actor.parameters()), allow_unused=True)
            vector = torch.cat([g.flatten() for g in grads if g is not None])
            return objective.detach(), rollout["targets"], vector

        full_objective, full_targets, full = gradient(1.0)
        none_objective, _, none = gradient(0.0)
        part_objective, part_targets, part = gradient(0.25)
        torch.testing.assert_close(full_objective, part_objective, rtol=0, atol=0)
        torch.testing.assert_close(full_objective, none_objective, rtol=0, atol=0)
        torch.testing.assert_close(full_targets, part_targets, rtol=0, atol=0)
        self.assertGreater(float((full - none).norm()), 0.)
        # The scaled path is linear: s interpolates between the two extremes.
        torch.testing.assert_close(part, none + 0.25 * (full - none), rtol=1e-9, atol=1e-12)


class RecoveryTempoCostTest(unittest.TestCase):
    def setUp(self):
        self.cost = RecoveryTempoCost(torch.full((29,), 4.0), base_rise_limit=0.9, base_rise_coef=10.,
                                      joint_speed_coef=0.02, upright_height=0.65)

    def states(self):
        state = torch.zeros(4, 71); state[:, 3] = 1.
        state[:, 2] = torch.tensor([.3, .3, .8, .8])
        context = torch.zeros(4, 35)
        context[:, -2] = torch.tensor([1., 0., 1., 1.])  # delayed-recovery flag
        context[3, -1] = 5.  # upright but still inside a bad-pose countdown
        following = state.clone()
        following[:, 9] = 1.4  # base rise speed, m/s
        following[:, 13 + 29] = 7.0  # one joint speed, rad/s
        following[:, 13 + 30] = -6.0
        return state, following, context

    def test_cost_is_active_only_for_a_delayed_environment_that_is_down_or_counting(self):
        state, following, context = self.states()
        self.assertEqual(self.cost.active(state, context).tolist(), [True, False, False, True])
        expected = 10. * (1.4 - 0.9) ** 2 + 0.02 * ((7. - 4.) ** 2 + (6. - 4.) ** 2)
        torch.testing.assert_close(self.cost(state, following, context),
                                   torch.tensor([expected, 0., 0., expected]))

    def test_motion_inside_the_limits_costs_nothing_and_falling_is_not_penalised(self):
        state, following, context = self.states()
        following[:, 9] = -3.0
        following[:, 13 + 29:] = 3.9
        self.assertEqual(float(self.cost(state, following, context).abs().max()), 0.)

    def test_gradient_reaches_the_predicted_state(self):
        state, following, context = self.states()
        following.requires_grad_(True)
        self.cost(state, following, context).sum().backward()
        self.assertGreater(float(following.grad[0, 9]), 0.)
        self.assertLess(float(following.grad[0, 13 + 30]), 0.)
        self.assertEqual(float(following.grad[1].abs().max()), 0.)

    def test_limits_come_from_the_recovery_clips_with_left_right_pairs_sharing_the_larger(self):
        names = ["left_knee_joint", "right_knee_joint", "waist_yaw_joint"]
        with tempfile.TemporaryDirectory() as folder:
            velocity = np.zeros((101, 3), dtype=np.float32)
            velocity[:, 0] = np.linspace(0, 2, 101); velocity[:, 1] = -np.linspace(0, 6, 101)
            velocity[:, 2] = 1.0
            np.savez(Path(folder) / "clip.npz", joint_vel=velocity)
            limits, source = reference_joint_speed_limits(folder, 50.0, profile=G1, joint_names=names)
            self.assertEqual(limits.tolist(), [3.0, 3.0, 1.0])
            self.assertEqual(set(source["clips_sha256"]), {"clip.npz"})
            with self.assertRaisesRegex(ValueError, "joints"):
                reference_joint_speed_limits(folder, 50.0, profile=G1, joint_names=names[:2])
        metadata = self.cost.metadata()
        self.assertEqual(len(metadata["joint_speed_limit_rad_s"]), 29)
        self.assertEqual(metadata, self.cost.metadata())

    def test_imagined_task_reward_subtracts_the_cost_once_at_the_task_scale(self):
        runner, values, noise = make_runner()
        values["context"][:, -2] = 1.  # every start is a delayed-recovery environment ...
        values["state"][:, 2] = .3  # ... on the ground
        with torch.no_grad():
            before = runner.imagine(values, noises=noise)
            runner.recovery_tempo = RecoveryTempoCost(torch.full((29,), 1e-3), base_rise_limit=0.,
                base_rise_coef=10., joint_speed_coef=0.02, upright_height=0.65).double()
            after = runner.imagine(values, noises=noise)
        difference = before["task"] - after["task"]
        self.assertTrue(bool((difference > 0).all()))
        # First transition: reconstruct the cost from the same imagined step.
        with torch.no_grad():
            history = values["actor_history"].reshape(len(values["state"]), 4, -1)
            predicted, *_ = runner._imagined_step(values["state"], values["context"], history,
                                                  values["command"], noise[0])
            expected = runner.adapter.reward_scale * runner.recovery_tempo(
                values["state"], predicted, values["context"])
        torch.testing.assert_close(difference[0], expected, rtol=1e-9, atol=1e-12)


class RecoveryEnvironmentTest(unittest.TestCase):
    def env_cfg(self):
        return SimpleNamespace(
            rewards={"track_root_height": SimpleNamespace(params={"std": .3, "mask_delay": True,
                                                                  "delay_env_rew_ratio": 3.5})},
            terminations={}, scale_rewards_by_dt=True)

    def test_default_configuration_leaves_the_task_untouched(self):
        env_cfg = self.env_cfg()
        configure_recovery_style_env(fixed(), env_cfg)
        self.assertEqual(env_cfg.rewards["track_root_height"].params["delay_env_rew_ratio"], 3.5)
        self.assertEqual(list(env_cfg.rewards), ["track_root_height"])
        self.assertEqual(env_cfg.terminations, {})

    def test_options_set_the_height_ratio_and_add_one_reward_and_one_timeout_term(self):
        env_cfg = self.env_cfg()
        shared = env_cfg.rewards["track_root_height"].params
        cfg = fixed(recovery_height_reward_ratio=2.0, recovery_tempo_mode="reference_limits_v1",
                    recovery_refresh_seconds=2.0)
        configure_recovery_style_env(cfg, env_cfg)
        self.assertEqual(env_cfg.rewards["track_root_height"].params["delay_env_rew_ratio"], 2.0)
        self.assertEqual(shared["delay_env_rew_ratio"], 3.5)  # another task's mapping is not edited
        self.assertEqual(env_cfg.rewards[RECOVERY_TEMPO_REWARD_TERM].weight, -1.)
        term = env_cfg.terminations[RECOVERY_REFRESH_TERM]
        self.assertTrue(term.time_out)
        self.assertEqual(term.params, {"seconds": 2.0, "recovery_height": G1.recovery_height})
        with self.assertRaisesRegex(ValueError, "twice"):
            configure_recovery_style_env(cfg, env_cfg)

    def test_refresh_fires_after_a_delayed_environment_has_stayed_up(self):
        delayed = torch.tensor([True, True, False, True])
        manager = SimpleNamespace(_delay_env_mask=delayed, _delay_counters=torch.tensor([0, 0, 0, 3]))
        heights = torch.tensor([.8, .3, .8, .8])
        robot = SimpleNamespace(data=SimpleNamespace(
            root_link_pos_w=torch.stack((torch.zeros(4), torch.zeros(4), heights), -1)))

        class Scene(dict):
            env_origins = torch.zeros(4, 3)

        env = SimpleNamespace(termination_manager=manager, num_envs=4, device="cpu", step_dt=.02,
                              scene=Scene(robot=robot), episode_length_buf=torch.ones(4, dtype=torch.long))
        fired = []
        for step in range(1, 5):
            env.episode_length_buf = torch.full((4,), step, dtype=torch.long)
            fired.append(delayed_recovery_refresh(env, seconds=.06, recovery_height=.55).tolist())
        # Up for 3 steps (0.06 s): only the delayed, upright, non-counting environment times out.
        self.assertEqual(fired, [[False] * 4, [False] * 4, [True, False, False, False], [True, False, False, False]])
        env.episode_length_buf = torch.ones(4, dtype=torch.long)  # reset: the count starts again
        self.assertEqual(delayed_recovery_refresh(env, seconds=.06, recovery_height=.55).tolist(), [False] * 4)
        self.assertEqual(env._wmac_up_steps.tolist(), [1, 0, 1, 0])

    def test_motion_root_fingerprint_follows_content_not_path(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            for folder in (first, second):
                for group in ("WalkandRun", "Recovery"):
                    (Path(folder) / group).mkdir()
                    np.savez(Path(folder) / group / "clip.npz", joint_vel=np.ones((2, 3)))
            self.assertEqual(motion_root_fingerprint(first), motion_root_fingerprint(second))
            np.savez(Path(second) / "Recovery" / "clip.npz", joint_vel=np.zeros((2, 3)))
            self.assertNotEqual(motion_root_fingerprint(first), motion_root_fingerprint(second))


class RecoveryRunnerTest(unittest.TestCase):
    def runner(self, batch_size=4):
        runner, values, noise = make_runner(batch_size=batch_size, dtype=torch.float32)
        runner._reward_pairs_fn = runner._trajectory_reward_terms
        values["next_state"] = values["state"].clone()
        values["next_state"][:, 2] += .03
        runner.replay = FixedReplay(values)
        return runner, values, noise

    def test_default_discriminator_update_is_the_historical_one(self):
        results = []
        for explicit in (False, True):
            runner, values, _ = self.runner()
            runner.cfg.discriminator_batch_size = 4
            if explicit:
                runner.cfg.discriminator_imagined_fraction, runner.cfg.discriminator_penalty_points = 0., "expert"
                runner._imagined_pairs = (torch.ones(8, 195), torch.ones(8, 195))  # must be ignored
            expert = torch.linspace(-1., 2., 4 * 195).reshape(4, 195)
            runner.experts = SimpleNamespace(sample=lambda size: (expert, expert + .2))
            torch.manual_seed(5)
            metrics = runner.update_discriminator()
            results.append((metrics, copy.deepcopy(runner.discriminator.state_dict())))
        self.assertEqual(results[0][0], results[1][0])
        self.assertEqual(set(results[0][0]), {"discriminator_loss", "discriminator_grad_penalty",
                                              "discriminator_grad_norm"})
        for key, value in results[0][1].items():
            torch.testing.assert_close(value, results[1][1][key], rtol=0, atol=0)

    def test_actor_update_leaves_imagined_pairs_that_replace_half_of_the_negatives(self):
        runner, values, _ = self.runner()
        centered(runner, mean=None)
        cfg = runner.cfg
        cfg.discriminator_imagined_fraction, cfg.discriminator_penalty_points = .5, "interpolated"
        cfg.discriminator_batch_size, cfg.discriminator_updates = 8, 2
        cfg.amp_normalizer_per_update = True
        self.assertIsNone(getattr(runner, "_imagined_pairs", None))
        runner.update_actor_critic(update_actor=True)
        first, second = runner._imagined_pairs
        self.assertEqual(first.shape, (4 * 2, 195))  # half a batch per discriminator update
        self.assertEqual(second.shape, first.shape)
        self.assertFalse(first.requires_grad)
        self.assertIsNone(runner._imagined_pair_sink)

        expert = torch.linspace(-1., 2., 8 * 195).reshape(8, 195)
        runner.experts = SimpleNamespace(sample=lambda size: (expert[:size], expert[:size] + .2))
        batches, scores = [], []
        def observe(module, args, output):
            batches.append(len(args[0])); scores.append(output.detach().clone())

        handle = runner.discriminator.register_forward_hook(observe)
        normalizer_count = float(runner.amp_norm.count)
        metrics = runner.update_discriminator()
        handle.remove()
        # Negatives (4 simulator + 4 imagined), experts, penalty points: same batch size as before.
        self.assertEqual(batches, [8, 8, 8] * 2)
        self.assertEqual(metrics["discriminator_imagined_share"], .5)
        self.assertTrue(np.isfinite(metrics["discriminator_imagined_score"]))
        # The reward stays centred on simulator transitions, and so does the normaliser.
        decay = cfg.amp_score_ema_decay
        expected = scores[0][:4].mean() * decay + scores[3][:4].mean() * (1 - decay)
        torch.testing.assert_close(runner.amp_reward_stats.robot_mean, expected)
        self.assertEqual(float(runner.amp_norm.count) - normalizer_count, 2 * (4 + 4 + 8 + 8))

    def test_first_discriminator_update_without_a_pool_uses_simulator_negatives_only(self):
        runner, values, _ = self.runner()
        runner.cfg.discriminator_imagined_fraction, runner.cfg.discriminator_batch_size = .5, 4
        expert = torch.zeros(4, 195)
        runner.experts = SimpleNamespace(sample=lambda size: (expert, expert))
        metrics = runner.update_discriminator()
        self.assertEqual(metrics["discriminator_imagined_share"], 0.)
        self.assertTrue(np.isnan(metrics["discriminator_imagined_score"]))

    def test_closed_gate_removes_recovery_starts_from_the_actor_objective(self):
        runner, values, _ = self.runner()
        values["context"][:, -1] = 3.  # every start is in recovery
        before = [parameter.detach().clone() for parameter in runner.actor.parameters()]
        runner._recovery_gate_closed = True
        result = runner.update_actor_critic(update_actor=True)
        self.assertEqual(result["recovery_gate_closed"], 1.)
        for saved, parameter in zip(before, runner.actor.parameters()):
            torch.testing.assert_close(saved, parameter, rtol=0, atol=0)
        runner._recovery_gate_closed = False
        result = runner.update_actor_critic(update_actor=True)
        self.assertEqual(result["recovery_gate_closed"], 0.)
        self.assertTrue(any(not torch.equal(saved, parameter)
                            for saved, parameter in zip(before, runner.actor.parameters())))

    def test_gate_reads_the_training_and_held_out_recovery_error(self):
        runner, _, _ = self.runner()
        runner.validation = {"validation_recovery_error": float("inf")}
        self.assertFalse(runner._recovery_gate({"wm_recovery_error": 9.}))  # disabled by default
        runner.cfg.recovery_model_error_gate = 2.
        self.assertTrue(runner._recovery_gate({"wm_recovery_error": 9.}))
        self.assertFalse(runner._recovery_gate({"wm_recovery_error": 1.}))
        self.assertFalse(runner._recovery_gate({"wm_recovery_error": float("nan")}))  # no recovery data yet
        runner.validation = {"validation_recovery_error": 5.}
        self.assertTrue(runner._recovery_gate({"wm_recovery_error": 1.}))

    def test_recovery_trust_region_rejects_and_restores_a_large_step(self):
        runner, values, _ = self.runner()
        values["context"][-1, -1] = 3.
        before = [parameter.detach().clone() for parameter in runner.actor.parameters()]
        runner.cfg.recovery_max_action_shift = 1e-12
        result = runner.update_actor_critic(update_actor=True)
        self.assertEqual(result["actor_update_rejected"], 1.)
        self.assertGreater(result["actor_recovery_action_shift"], 1e-12)
        for saved, parameter in zip(before, runner.actor.parameters()):
            torch.testing.assert_close(saved, parameter, rtol=0, atol=0)
        runner.cfg.recovery_max_action_shift = 1e6
        self.assertEqual(runner.update_actor_critic(update_actor=True)["actor_update_rejected"], 0.)

    def test_terminal_value_gradient_is_logged_per_phase(self):
        runner, values, _ = self.runner()
        values["context"][-1, -1] = 3.
        result = runner.update_actor_critic(update_actor=True)
        self.assertGreater(result["terminal_value_grad_walk"], 0.)
        self.assertGreater(result["terminal_value_grad_recovery"], 0.)


class RecoveryCheckpointTest(unittest.TestCase):
    def save(self, directory, **settings):
        runner = checkpoint_runner()
        for key, value in settings.items():
            setattr(runner.cfg, key, value)
        path = Path(directory) / "model.pt"
        with patch("torch.cuda.is_available", return_value=False):
            runner.save(path, export=False)
        return path, torch.load(path, weights_only=False)

    def test_default_checkpoint_is_written_exactly_as_before(self):
        with tempfile.TemporaryDirectory() as directory:
            _, saved = self.save(directory)
        self.assertEqual(saved["implementation_version"], 4)
        for key in ("recovery_style_protocol", "recovery_style_migration_history",
                    "reference_motion_dataset", "recovery_tempo_model"):
            self.assertNotIn(key, saved)

    def test_enabled_checkpoint_records_its_protocol_and_resumes_only_under_it(self):
        with tempfile.TemporaryDirectory() as directory, patch("torch.cuda.is_available", return_value=False):
            path, saved = self.save(directory, recovery_value_grad_scale=.1, discriminator_imagined_fraction=.5)
            self.assertEqual(saved["implementation_version"], RECOVERY_STYLE_VERSION)
            self.assertEqual(saved["recovery_style_protocol"]["recovery_value_grad_scale"], .1)
            self.assertIsNone(saved["reference_motion_dataset"])
            plain = checkpoint_runner()
            before = copy.deepcopy(plain.actor.state_dict())
            with self.assertRaisesRegex(ValueError, "allow-recovery-style-migration"):
                plain.load(path)
            for key, value in before.items():  # a refused resume loads nothing
                torch.testing.assert_close(plain.actor.state_dict()[key], value, rtol=0, atol=0)
            same = checkpoint_runner()
            same.cfg.recovery_value_grad_scale, same.cfg.discriminator_imagined_fraction = .1, .5
            same.load(path)
            self.assertEqual(same.recovery_style_migration_history, [])
            self.assertEqual(same._cost_trajectory_id, "wmac-test-trajectory")

    def test_authorised_fork_keeps_the_networks_and_records_what_changed(self):
        with tempfile.TemporaryDirectory() as directory, patch("torch.cuda.is_available", return_value=False):
            path, saved = self.save(directory)
            fork = checkpoint_runner()
            fork.cfg.recovery_value_grad_scale, fork.cfg.recovery_height_reward_ratio = .1, 2.
            fork.cfg.allow_recovery_style_migration = True
            fork.log_dir = Path(directory)
            fork.load(path)
            for key, value in saved["actor"].items():
                torch.testing.assert_close(fork.actor.state_dict()[key], value, rtol=0, atol=0)
            record, = fork.recovery_style_migration_history
            self.assertEqual(record["kind"], "explicit_recovery_style_migration")
            self.assertEqual(set(record["changes"]), {"recovery_value_grad_scale", "recovery_height_reward_ratio"})
            self.assertTrue((Path(directory) / "recovery_style_migration.json").is_file())
            self.assertNotEqual(fork._cost_trajectory_id, "wmac-test-trajectory")
            # The fork saves as an enabled checkpoint that then resumes without the flag.
            fork.cfg.allow_recovery_style_migration = False
            forked = Path(directory) / "forked.pt"
            fork.save(forked, export=False)
            again = checkpoint_runner()
            again.cfg.recovery_value_grad_scale, again.cfg.recovery_height_reward_ratio = .1, 2.
            again.load(forked)
            self.assertEqual(len(again.recovery_style_migration_history), 1)

    def test_new_options_save_as_version_15(self):
        with tempfile.TemporaryDirectory() as directory:
            _, saved = self.save(directory, recovery_style_weight=.5)
        self.assertEqual(saved["implementation_version"], RECOVERY_PATIENCE_VERSION)
        self.assertEqual(saved["recovery_style_protocol"]["recovery_style_weight"], .5)

    def test_reward_change_refuses_a_replay_that_holds_old_reward_labels(self):
        with tempfile.TemporaryDirectory() as directory, patch("torch.cuda.is_available", return_value=False):
            path, _ = self.save(directory)
            fork = checkpoint_runner()
            fork.cfg.recovery_height_reward_ratio, fork.cfg.allow_recovery_style_migration = 2., True
            fork.replay.size = 5
            with self.assertRaisesRegex(ValueError, "empty target replay"):
                fork.load(path)


class RecoveryPatienceTest(unittest.TestCase):
    """recovery_patience_seconds: a constant reward for the first seconds of a recovery."""

    def env_cfg(self, delay_ratio=.4, max_delay=250):
        def term(weight, **params):
            return SimpleNamespace(weight=weight, params=params)
        return SimpleNamespace(
            rewards={"track_anchor_linear_velocity": term(1., std=1., mask_delay=True, delay_env_rew_ratio=0.),
                     "track_anchor_angular_velocity": term(1., std=3.14, mask_delay=True, delay_env_rew_ratio=0.),
                     "track_root_height": term(1., std=.3, mask_delay=True, delay_env_rew_ratio=3.5),
                     "body_ang_vel_xy_l2": term(.5, std=3.14, mask_delay=True, delay_env_rew_ratio=0.),
                     "is_terminated": term(-200.), "action_rate_l2": term(-.01)},
            terminations={}, scale_rewards_by_dt=True, decimation=4,
            sim=SimpleNamespace(mujoco=SimpleNamespace(timestep=.005)),
            events={"init_motion_loader": SimpleNamespace(params={"delay_reset_env_ratio": delay_ratio,
                                                                  "max_delay_steps": max_delay})})

    def test_protocol_records_the_reward_only_when_the_window_is_on(self):
        self.assertIsNone(DEFAULT_PROTOCOL["recovery_patience_seconds"])
        self.assertIsNone(DEFAULT_PROTOCOL["recovery_style_weight"])
        self.assertNotIn("recovery_patience_reward", DEFAULT_PROTOCOL)
        # The unused reward value is not part of the objective.
        self.assertEqual(recovery_style_protocol(fixed(recovery_patience_reward=1.0)), DEFAULT_PROTOCOL)
        protocol = recovery_style_protocol(fixed(recovery_patience_seconds=3.5))
        self.assertEqual((protocol["recovery_patience_seconds"], protocol["recovery_patience_reward"]), (3.5, 2.0))
        for changes in ({"recovery_patience_seconds": 0.}, {"recovery_patience_seconds": -1.},
                        {"recovery_patience_seconds": 3.5, "recovery_patience_reward": 0.},
                        {"recovery_style_weight": -.1}, {"recovery_style_weight": float("inf")}):
            with self.assertRaises(ValueError, msg=changes):
                fixed(amp_reward_mapping="instantamp_centered_tanh_v1", **changes).validate()

    def test_new_options_need_implementation_version_15(self):
        self.assertIsNone(recovery_style_implementation_version(fixed()))
        self.assertEqual(recovery_style_implementation_version(fixed(recovery_value_grad_scale=.1)),
                         RECOVERY_STYLE_VERSION)
        for changes in ({"recovery_patience_seconds": 3.5}, {"recovery_style_weight": .5}):
            cfg = fixed(amp_reward_mapping="instantamp_centered_tanh_v1", **changes)
            self.assertEqual(recovery_style_implementation_version(cfg), RECOVERY_PATIENCE_VERSION)
            self.assertEqual(checkpoint_recovery_style_protocol(payload(cfg, version=RECOVERY_PATIENCE_VERSION)),
                             recovery_style_protocol(cfg))
            with self.assertRaisesRegex(ValueError, "implementation_version"):
                checkpoint_recovery_style_protocol(payload(cfg, version=RECOVERY_STYLE_VERSION))
            target = copy.deepcopy(cfg)
            target.allow_recovery_style_migration = True
            self.assertTrue(changes_reward(validate_recovery_style_resume(payload(), target)))

    def test_version_14_checkpoint_recorded_before_the_options_still_resumes(self):
        cfg = fixed(recovery_value_grad_scale=.1, recovery_tempo_mode="reference_limits_v1")
        source = payload(cfg, version=RECOVERY_STYLE_VERSION)
        for key in ("recovery_patience_seconds", "recovery_patience_reward", "recovery_style_weight"):
            source["config"].pop(key, None)
            source["recovery_style_protocol"].pop(key, None)
        self.assertEqual(checkpoint_recovery_style_protocol(source), recovery_style_protocol(cfg))
        restored = resolve_checkpoint_config(source)
        self.assertEqual(validate_recovery_style_resume(source, restored), {})
        target = copy.deepcopy(restored)
        target.recovery_patience_seconds, target.allow_recovery_style_migration = 3.5, True
        self.assertEqual(set(validate_recovery_style_resume(source, target)),
                         {"recovery_patience_seconds", "recovery_patience_reward"})

    def test_shared_formula(self):
        height = torch.tensor([.2, .2, .2, .2])
        counter = torch.tensor([1, 5, 6, 40])
        torch.testing.assert_close(patient_recovery_height_reward(height, counter, 2., None, 5), height * 2.)
        torch.testing.assert_close(patient_recovery_height_reward(height, counter, 2., 1.5, 5),
                                   torch.tensor([1.5, 1.5, .4, .4]))

    def test_environment_configuration(self):
        env_cfg = self.env_cfg()
        shared = env_cfg.rewards["track_root_height"].params
        configure_recovery_style_env(fixed(recovery_patience_seconds=3.5), env_cfg)
        params = env_cfg.rewards["track_root_height"].params
        self.assertEqual((params["patience_reward"], params["patience_steps"]), (2.0, 175))
        self.assertNotIn("patience_reward", shared)  # another task's mapping is not edited
        self.assertEqual(standing_reward_ceiling(env_cfg.rewards), 2.5)
        for cfg, env_cfg, message in (
                (fixed(recovery_patience_seconds=5.0), self.env_cfg(), "deadline"),
                (fixed(recovery_patience_seconds=3.5, recovery_patience_reward=2.5), self.env_cfg(), "standing"),
                (fixed(recovery_patience_seconds=3.5), self.env_cfg(delay_ratio=0.), "delayed-recovery"),
                (fixed(recovery_patience_seconds=.001), self.env_cfg(), "one control step")):
            with self.assertRaisesRegex(ValueError, message):
                configure_recovery_style_env(cfg, env_cfg)

    def test_simulator_term(self):
        from src.tasks.amp_loco.mdp.rewards import track_root_height
        heights = torch.tensor([.8, .3, .3, .3, .3])
        robot = SimpleNamespace(data=SimpleNamespace(
            default_root_state=torch.tensor([[0., 0., .8] + [1., 0., 0., 0.] + [0.] * 6] * 5),
            body_link_pos_w=torch.stack((torch.zeros(5), torch.zeros(5), heights), -1)[:, None],
            body_link_quat_w=torch.tensor([[1., 0., 0., 0.]] * 5)[:, None]))
        manager = SimpleNamespace(_delay_env_mask=torch.tensor([True, True, True, True, False]),
                                  _delay_counters=torch.tensor([0, 1, 5, 6, 3]))
        env = SimpleNamespace(scene={"robot": robot}, termination_manager=manager)
        height = math.exp(-(.5 / .3) ** 2)
        plain = track_root_height(env, std=.3, mask_delay=True, delay_env_rew_ratio=2., upright_power=2.)
        torch.testing.assert_close(plain, torch.tensor([0., 2 * height, 2 * height, 2 * height, 0.]))
        patient = track_root_height(env, std=.3, mask_delay=True, delay_env_rew_ratio=2., upright_power=2.,
                                    patience_reward=1.5, patience_steps=5)
        torch.testing.assert_close(patient, torch.tensor([0., 1.5, 1.5, 2 * height, 0.]))

    def test_imagined_term_is_the_same_expression(self):
        runner, values, _ = make_runner(batch_size=4)
        adapter = runner.adapter
        state = values["state"].clone(); state[:, 2] = .3
        following = state.clone()
        context = values["context"].clone(); context[:, -2] = torch.tensor([1., 1., 1., 0.])
        following_context = context.clone(); following_context[:, -1] = torch.tensor([3., 200., 0., 3.])
        args = (state, following, torch.zeros(4, 29, dtype=state.dtype), torch.zeros(4, 29, dtype=state.dtype),
                values["command"], following_context, torch.zeros(4, dtype=state.dtype),
                torch.zeros(4, 2, dtype=state.dtype))
        term = adapter.reward_cfg["track_root_height"]
        original = term.params
        with torch.no_grad():
            before = adapter.task_reward(*args)
            term.params = {**original, "patience_reward": 1.5, "patience_steps": 175}
            try:
                after = adapter.task_reward(*args)
            finally:
                term.params = original
        height = recovery_height_reward(following[:, 2], adapter.desired_height, .3,
                                        up_z=recovery_root_up_z(following[:, 3:7]), upright_power=2.)
        expected = adapter.reward_scale * torch.stack((1.5 - 3.5 * height[0], torch.zeros_like(height[0]),
                                                       torch.zeros_like(height[0]), torch.zeros_like(height[0])))
        torch.testing.assert_close(after - before, expected)

    def test_critic_bound_covers_both_options(self):
        runner, _, _ = make_runner()
        base = amp_value_upper_bound(runner.adapter, runner.cfg)
        runner.cfg.recovery_style_weight = runner.cfg.style_weight + .5
        raised = amp_value_upper_bound(runner.adapter, runner.cfg)
        self.assertAlmostEqual((raised - base) * (1 - runner.cfg.gamma), .25 * runner.cfg.amp_reward_coef)
        runner.cfg.recovery_style_weight = None
        term = runner.adapter.reward_cfg["track_root_height"]
        original = term.params
        try:
            term.params = {**original, "patience_reward": 1.5, "patience_steps": 175}  # below the 3.5 ratio
            self.assertEqual(amp_value_upper_bound(runner.adapter, runner.cfg), base)
            term.params = {**original, "delay_env_rew_ratio": 1.}
            low_ratio = amp_value_upper_bound(runner.adapter, runner.cfg)
            term.params = {**original, "delay_env_rew_ratio": 1., "patience_reward": 2., "patience_steps": 175}
            self.assertAlmostEqual(amp_value_upper_bound(runner.adapter, runner.cfg) - low_ratio,
                                   runner.cfg.task_weight * runner.adapter.reward_scale * 1. / (1 - runner.cfg.gamma))
        finally:
            term.params = original


class RecoveryStyleWeightTest(unittest.TestCase):
    def recovering(self, runner, values):
        values["context"][:, -2] = 1.
        values["state"][:, 2] = .3
        values["actor_history"] = runner.adapter.actor_frame(values["state"], values["previous_action"],
            values["command"], values["context"]).unsqueeze(1).repeat(1, 4, 1).flatten(1)

    def test_extra_weight_is_centred_on_the_robot_average(self):
        cfg = fixed(amp_reward_mapping="instantamp_centered_tanh_v1")
        state = torch.zeros(4, 71); state[:, 2] = torch.tensor([.3, .3, .9, .3])
        context = torch.zeros(4, 35); context[:, -2] = torch.tensor([1., 1., 1., 0.])
        style = torch.tensor([.08, .05, .08, .02])  # the centre is amp_reward_coef / 2 = .05
        torch.testing.assert_close(weighted_style(cfg, style, state, context, .65), cfg.style_weight * style)
        cfg.recovery_style_weight = .75
        expected = cfg.style_weight * style + torch.tensor([.5 * .03, 0., 0., 0.])
        torch.testing.assert_close(weighted_style(cfg, style, state, context, .65), expected)
        cfg.amp_reward_mapping = "clipped_lsgan_v1"
        with self.assertRaisesRegex(ValueError, "centred"):
            cfg.validate()

    def test_recovering_rollouts_use_the_recovery_weight_and_walking_ones_do_not(self):
        runner, values, noise = make_runner()
        values["context"][:, -2:] = 0.  # no delayed-recovery environment
        with torch.no_grad():
            walking = runner.imagine(values, noises=noise)
            runner.cfg.recovery_style_weight = .9
            unchanged = runner.imagine(values, noises=noise)
        torch.testing.assert_close(unchanged["objective"], walking["objective"], rtol=0, atol=0)
        torch.testing.assert_close(unchanged["targets"], walking["targets"], rtol=0, atol=0)
        runner, values, noise = make_runner()
        self.recovering(runner, values)
        base = runner.cfg.style_weight
        with torch.no_grad():
            runner.cfg.recovery_style_weight = .9
            weighted = runner.imagine(values, noises=noise)
            runner.cfg.recovery_style_weight, runner.cfg.style_weight = None, .9
            reference = runner.imagine(values, noises=noise)
        self.assertEqual(float(weighted["survival"].min()), 1.)  # every imagined step stays a recovery step
        # The same weight on style, less the centre on every recovering step.
        offset = (.9 - base) * .5 * runner.cfg.amp_reward_coef * sum(
            runner.cfg.gamma ** k for k in range(runner.cfg.imagination_horizon))
        torch.testing.assert_close(weighted["objective"], reference["objective"] - offset)

    def test_gradient_probe_components_still_add_up(self):
        runner, values, noise = make_runner(batch_size=2, horizon=2)
        self.recovering(runner, values)
        runner.cfg.recovery_style_weight = .9
        result = objective_gradient_probe(runner, values, noise)
        self.assertLess(result["decomposition"]["objective_absolute_residual"], 1e-12)
        self.assertLess(result["decomposition"]["gradient_relative_residual"], 1e-9)
        self.assertEqual(runner.cfg.recovery_style_weight, .9)


class CleanRecoveryReferenceTest(unittest.TestCase):
    XML = """
    <mujoco>
      <compiler angle="radian"/>
      <worldbody>
        <geom name="floor" type="plane" size="0 0 .01"/>
        <body name="robot/base" pos="0 0 .3">
          <freejoint name="robot/root"/>
          <geom name="robot/base_collision" type="sphere" size=".1" mass="1"/>
          <body name="robot/left_arm" pos="0 .15 0">
            <joint name="robot/left_shoulder" type="hinge" axis="1 0 0" range="-2 2"/>
            <geom name="robot/left_arm_collision" type="capsule" fromto="0 .06 0 0 .3 0" size=".03" mass="1"/>
          </body>
          <body name="robot/right_arm" pos="0 -.15 0">
            <joint name="robot/right_shoulder" type="hinge" axis="1 0 0" range="-2 2"/>
            <geom name="robot/right_arm_collision" type="capsule" fromto="0 -.06 0 0 -.3 0" size=".03" mass="1"/>
          </body>
        </body>
      </worldbody>
    </mujoco>
    """

    @classmethod
    def setUpClass(cls):
        import mujoco
        cls.tool = load_script("clean_recovery_reference")
        cls.robot = cls.tool.Robot(mujoco.MjModel.from_xml_string(cls.XML), contact_band=.01,
                                   floor_clearance=0., self_clearance=0., iterations=40)

    def test_constrained_step_is_the_closest_feasible_point(self):
        step = self.tool.closest_feasible_step(np.eye(2), np.zeros(2), np.array([[1., 0.]]), np.array([1.]))
        np.testing.assert_allclose(step, [1., 0.], atol=1e-9)
        free = self.tool.closest_feasible_step(np.eye(2), np.array([3., 1.]), np.array([[1., 0.]]), np.array([1.]))
        np.testing.assert_allclose(free, [3., 1.], atol=1e-9)
        weighted = self.tool.closest_feasible_step(np.diag([1., 100.]), np.zeros(2),
                                                   np.array([[1., 1.]]), np.array([1.]))
        np.testing.assert_allclose(weighted, [100. / 101., 1. / 101.], atol=1e-9)

    def test_feasible_pose_is_returned_unchanged(self):
        pose = self.robot.qpos([0., 0., .3], np.array([1., 0., 0., 0.]), [.2, -.1])
        self.assertEqual(self.robot.penetration(pose)["floor_depth"], 0.)
        repaired = self.robot.repair(pose, pose, self.robot.floor_targets(pose))
        np.testing.assert_allclose(repaired, pose, atol=1e-9)

    def test_floor_penetration_is_removed_without_lifting_a_resting_contact(self):
        # The base rests on the floor; the left arm points 0.9 rad down, into it.
        pose = self.robot.qpos([0., 0., .1], np.array([1., 0., 0., 0.]), [-.9, 0.])
        before = self.robot.penetration(pose)
        self.assertGreater(before["floor_depth"], .05)
        self.assertEqual(before["floor_geom"], "left_arm_collision")
        repaired = self.robot.repair(pose, pose, self.robot.floor_targets(pose))
        after = self.robot.penetration(repaired)
        self.assertLess(after["floor_depth"], 1e-4)
        # Lifting the base alone would take the full depth; the arm bends instead.
        self.assertLess(repaired[2] - .1, .4 * before["floor_depth"])
        self.assertGreater(repaired[self.robot.qadr[0]], -.8)
        self.assertEqual(repaired[self.robot.qadr[1]], 0.)

    def test_a_flat_capsule_is_constrained_at_both_ends(self):
        self.robot.forward(self.robot.qpos([0., 0., .1], np.array([1., 0., 0., 0.]), [0., 0.]))
        counts = [len(points) for points in self.robot.floor_points()]
        self.assertEqual(counts, [1, 2, 2])

    def test_export_reproduces_finite_difference_velocities(self):
        frames = 6
        poses = np.stack([self.robot.qpos([.1 * t, 0., .3], np.array([1., 0., 0., 0.]), [.05 * t, 0.])
                          for t in range(frames)])
        motion = self.robot.motion(poses, fps=50.)
        self.assertEqual(motion["body_pos_w"].shape, (frames, 3, 3))
        np.testing.assert_allclose(motion["joint_vel"][:, 0], 2.5, atol=1e-5)
        np.testing.assert_allclose(motion["body_lin_vel_w"][:, 0, 0], 5., atol=1e-5)
        np.testing.assert_allclose(motion["body_pos_w"][:, 0], poses[:, :3], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
