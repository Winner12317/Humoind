"""CPU safeguards for explicit coefficient-only forks of an active objective."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from src.wmac.checkpoints import resolve_checkpoint_config
from src.wmac.config import WmacConfig
from src.wmac.reference import (
    SMOOTH_REWARD_COEFFICIENT_KEYS, smooth_reward_resume_transition,
    validate_smooth_reward_resume, validate_smooth_reward_retune,
    validate_training_protocol_resume,
)
from src.wmac.replay import ReplayBuffer
from test_centered_amp_integration import centered, assert_tree_equal
from test_smooth_reward_integration import smoothed, NETWORKS, OPTIMIZERS
from test_smooth_reward_protocol import payload, CLI
from test_wmac_official_protocol import protocol_runner


def runner_fixture():
    runner, _, _ = protocol_runner(hybrid=True, checkpoint=True)
    runner = centered(smoothed(runner))
    runner.replay = ReplayBuffer(2, 2, "cpu")
    return runner


def add_reward_labels(runner, labels):
    state = torch.zeros(2, 71)
    state[:, 2], state[:, 3] = .7, 1.
    runner.replay.add(dict(state=state, next_state=state+.001,
                           reward=torch.tensor(labels), terminated=torch.zeros(2, dtype=torch.bool)))


class SmoothRewardRetuneProtocolTest(unittest.TestCase):
    def test_each_active_coefficient_needs_new_authorization_and_keeps_enable_contract(self):
        source = payload(smooth=True)
        for key in SMOOTH_REWARD_COEFFICIENT_KEYS:
            target = resolve_checkpoint_config(source)
            target.allow_smooth_reward_migration = True
            setattr(target, key, getattr(target, key)*2)
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'allow-smooth-reward-retune'):
                validate_training_protocol_resume(source, target)
            target.allow_smooth_reward_migration = False
            target.allow_smooth_reward_retune = True
            self.assertEqual(smooth_reward_resume_transition(source, target), 'retune')
            self.assertTrue(validate_smooth_reward_retune(source, target))
            self.assertFalse(validate_smooth_reward_resume(source, target))
            self.assertFalse(validate_training_protocol_resume(source, target))  # AMP unchanged.

    def test_enable_and_downgrade_authorization_remain_separate(self):
        source = payload()
        target = resolve_checkpoint_config(source)
        target.smooth_reward_mode = 'host_humanup_v1'
        target.allow_smooth_reward_retune = True
        with self.assertRaisesRegex(ValueError, 'allow-smooth-reward-migration'):
            smooth_reward_resume_transition(source, target)
        target.allow_smooth_reward_migration = True
        self.assertEqual(smooth_reward_resume_transition(source, target), 'enable')
        self.assertTrue(validate_smooth_reward_resume(source, target))
        self.assertFalse(validate_smooth_reward_retune(source, target))
        source = payload(smooth=True)
        target = resolve_checkpoint_config(source)
        target.allow_smooth_reward_migration = target.allow_smooth_reward_retune = True
        target.smooth_reward_mode = 'none'
        with self.assertRaises(ValueError):
            smooth_reward_resume_transition(source, target)

    def test_same_objective_is_noop_and_flag_is_not_inherited(self):
        source = payload(smooth=True)
        source['config']['allow_smooth_reward_retune'] = True
        target = resolve_checkpoint_config(source)
        self.assertFalse(target.allow_smooth_reward_retune)
        self.assertEqual(smooth_reward_resume_transition(source, target), 'unchanged')
        target.allow_smooth_reward_retune = True
        self.assertEqual(smooth_reward_resume_transition(source, target), 'unchanged')
        cfg = WmacConfig()
        cfg.allow_smooth_reward_retune = 1
        with self.assertRaisesRegex(ValueError, 'must be a bool'):
            cfg.validate()

    def test_retune_locks_learning_rules_distributions_and_other_authorizations(self):
        source = payload(smooth=True)
        changes = {
            'gradient_penalty': 5., 'expert_recovery_fraction': .25, 'recovery_fraction': .25,
            'utd': .8, 'actor_lr': 1e-4, 'actor_betas': (.8, .9),
            'wm_window': 6, 'wm_updates': 5, 'wm_batch_size': 1024,
            'model_hidden': (512, 512), 'amp_normalizer_clip': 5.,
            'validation_env_fraction': .2, 'mirror_walk_motions': False,
            'allow_amp_reward_migration': True, 'allow_recovery_semantics_migration': True,
            'allow_smooth_reward_migration': True,
        }
        for key, value in changes.items():
            target = resolve_checkpoint_config(source)
            target.smooth_torque_coef *= 2
            target.allow_smooth_reward_retune = True
            setattr(target, key, value)
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                smooth_reward_resume_transition(source, target)

    def test_retune_accepts_explicit_pilot_execution_overrides_without_relaxing_learning(self):
        source = payload(smooth=True)
        target = resolve_checkpoint_config(source)
        target.smooth_torque_coef *= 3
        target.allow_smooth_reward_retune = True
        overrides = {
            'resume': '/local/source.pt', 'run_name': 'candidate', 'log_root': '/local/results',
            'iterations': 32500, 'num_envs': 512, 'actor_batch_size': 1024,
            'actor_microbatch_size': 256, 'compile': False,
            'precision': 'float32', 'save_interval': 500, 'log_interval': 20,
            'evaluation_interval': 0,
            'device': 'cpu', 'replay_device': 'cpu', 'checkpoint_replay': True,
            'benchmark_sync': True,
        }
        for key, value in overrides.items():
            setattr(target, key, value)
        self.assertEqual(smooth_reward_resume_transition(source, target), 'retune')
        self.assertFalse(validate_training_protocol_resume(source, target))

    def test_retune_cannot_combine_amp_or_either_recovery_change(self):
        source = payload(smooth=True)
        for key, value in (('amp_reward_mapping', 'clipped_lsgan_v1'),
                           ('recovery_counter_semantics', 'soft_v1'),
                           ('recovery_height_reward', 'height_only_v1')):
            target = resolve_checkpoint_config(source)
            target.smooth_torque_coef *= 2
            target.allow_smooth_reward_retune = True
            target.allow_amp_reward_migration = target.allow_recovery_semantics_migration = True
            setattr(target, key, value)
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'cannot also change'):
                validate_training_protocol_resume(source, target)

    def test_cli_requires_resume_and_new_flag_and_restores_new_weights_without_permission(self):
        with self.assertRaisesRegex(ValueError, 'requires --resume'):
            CLI.parse_config(['--allow-smooth-reward-retune'])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'active.pt'
            source = payload(smooth=True)
            torch.save(source, path)
            args = ['--resume', str(path), '--smooth-torque-coef', '.003']
            with self.assertRaises(ValueError):
                CLI.parse_config(args)
            cfg = CLI.parse_config([*args, '--allow-smooth-reward-retune'])
            self.assertEqual(cfg.smooth_torque_coef, .003)
            self.assertEqual(cfg.wm_updates, source['config']['wm_updates'])
            self.assertEqual(cfg.utd, source['config']['utd'])
            self.assertEqual(cfg.imagination_horizon, source['config']['imagination_horizon'])
            for flag, value in (('--gradient-penalty', '5'), ('--expert-recovery-fraction', '.25')):
                with self.subTest(flag=flag), self.assertRaisesRegex(ValueError, 'forbidden configuration changes'):
                    CLI.parse_config([*args, '--allow-smooth-reward-retune', flag, value])
            for flag, value in (('--amp-reward-mapping', 'clipped_lsgan_v1'),
                                ('--recovery-counter-semantics', 'soft_v1'),
                                ('--recovery-height-reward', 'height_only_v1')):
                with self.subTest(flag=flag), self.assertRaises(ValueError):
                    CLI.parse_config([*args, '--allow-smooth-reward-retune', flag, value])


class SmoothRewardRetuneRestoreTest(unittest.TestCase):
    def test_retune_preserves_actual_muon_adam_networks_ema_rng_and_clocks(self):
        source, target = runner_fixture(), runner_fixture()
        for name in OPTIMIZERS:
            optimizer = getattr(source, name)
            for group in optimizer.param_groups:
                for parameter in group['params']:
                    parameter.grad = torch.full_like(parameter, .001)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        source.cfg.checkpoint_replay = True
        source._collect_accumulator = .375
        source.best_eval_score = (1., 1., 1., -.1, -1.)
        add_reward_labels(source, [12., 34.])
        target.cfg.allow_smooth_reward_retune = True
        target.cfg.smooth_torque_coef *= 4
        target.cfg.smooth_power_coef *= 2
        target.validation = {key: .123 for key in target.validation}
        target.last_model_loss = .1
        with tempfile.TemporaryDirectory() as directory, patch('torch.cuda.is_available', return_value=False):
            path = Path(directory)/'source.pt'
            source.save(path, export=False)
            saved = torch.load(path, weights_only=False)
            target.log_dir = Path(directory)/'target'; target.log_dir.mkdir()
            target.load(path)
            for name in (*NETWORKS, *OPTIMIZERS):
                assert_tree_equal(self, saved[name], getattr(target, name).state_dict())
            assert_tree_equal(self, saved['amp_reward_state'], target.amp_reward_stats.state_dict())
            torch.testing.assert_close(torch.get_rng_state(), saved['torch_rng'], rtol=0, atol=0)
            self.assertEqual(random.getstate(), saved['python_rng'])
            current_numpy = np.random.get_state(); saved_numpy = saved['numpy_rng']
            self.assertEqual(current_numpy[0], saved_numpy[0])
            np.testing.assert_array_equal(current_numpy[1], saved_numpy[1])
            self.assertEqual(current_numpy[2:], saved_numpy[2:])
            for attr, key in (('iteration', 'iteration'), ('sim_steps', 'sim_steps'),
                              ('imagination_steps', 'imagination_steps'), ('wall_offset', 'wall_seconds'),
                              ('_collect_accumulator', 'collect_accumulator'),
                              ('critic_calibration_updates', 'critic_calibration_updates')):
                self.assertEqual(getattr(target, attr), saved[key])
            self.assertEqual(target.env.common_step_counter, saved['env_common_step_counter'])
            self.assertEqual(target.replay.size, 0)
            self.assertIsNone(target.best_eval_score)
            self.assertTrue(all(math.isinf(v) for v in target.validation.values()))
            self.assertTrue(math.isinf(target.last_model_loss))
            self.assertNotEqual(target._cost_trajectory_id, saved['training_cost']['trajectory_id'])
            record = target.smooth_reward_retune_history[-1]
            self.assertEqual(record['source_sha256'], hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(record['source_protocol'], saved['smooth_reward_protocol'])
            self.assertEqual(set(record['coefficient_changes']), {'smooth_torque_coef', 'smooth_power_coef'})
            self.assertTrue(record['saved_replay_discarded'])
            self.assertTrue(record['amp_ema_preserved'])
            self.assertEqual(json.loads((target.log_dir/'smooth_reward_retune.json').read_text()), record)
            self.assertEqual((target.cfg.model_warmup_iterations, target.cfg.critic_warmup_iterations),
                             (source.cfg.model_warmup_iterations, source.cfg.critic_warmup_iterations))

            # The new objective becomes ordinary saved behavior, with new labels retained.
            target.cfg.checkpoint_replay = True
            add_reward_labels(target, [-.05, -.03])
            new_path = Path(directory)/'retuned.pt'; target.save(new_path, export=False)
            updated = torch.load(new_path, weights_only=False)
            self.assertEqual(updated['implementation_version'], 12)
            ordinary = runner_fixture()
            ordinary.cfg = resolve_checkpoint_config(updated)
            self.assertFalse(ordinary.cfg.allow_smooth_reward_retune)
            ordinary.load(new_path)
            self.assertEqual(ordinary.replay.size, 1)
            torch.testing.assert_close(ordinary.replay.data['reward'][0], torch.tensor([-.05, -.03]))
            self.assertEqual(ordinary.smooth_reward_retune_history, target.smooth_reward_retune_history)
            ordinary.cfg.smooth_torque_coef *= 2
            with self.assertRaisesRegex(ValueError, 'allow-smooth-reward-retune'):
                ordinary.load(new_path)

    def test_retune_failures_precede_all_network_optimizer_rng_and_clock_mutation(self):
        source = runner_fixture()
        with tempfile.TemporaryDirectory() as directory, patch('torch.cuda.is_available', return_value=False):
            path = Path(directory)/'source.pt'; source.save(path, export=False)
            original = torch.load(path, weights_only=False)
            problems = ('motor', 'formula', 'nonempty_replay', 'amp', 'counter', 'height',
                        'gradient_penalty', 'expert_recovery_fraction',
                        'task_weight', 'actor_architecture', 'motion_dataset')
            for problem in problems:
                target = runner_fixture()
                target.cfg.allow_smooth_reward_retune = True
                target.cfg.smooth_torque_coef *= 4
                broken = copy.deepcopy(original)
                if problem == 'motor':
                    broken['smooth_reward_model'] = {'corrupt': True}
                elif problem == 'formula':
                    broken['smooth_reward_protocol']['formula_version'] = 'other'
                elif problem == 'nonempty_replay':
                    add_reward_labels(target, [1., 2.])
                elif problem == 'amp':
                    target.cfg.amp_reward_mapping = 'clipped_lsgan_v1'
                    target.cfg.allow_amp_reward_migration = True
                elif problem in ('counter', 'height'):
                    setattr(target.cfg, 'recovery_counter_semantics' if problem == 'counter' else 'recovery_height_reward',
                            'soft_v1' if problem == 'counter' else 'height_only_v1')
                    target.cfg.allow_recovery_semantics_migration = True
                elif problem == 'task_weight':
                    target.cfg.task_weight = .5
                elif problem == 'gradient_penalty':
                    target.cfg.gradient_penalty = 5.
                elif problem == 'expert_recovery_fraction':
                    target.cfg.expert_recovery_fraction = .25
                elif problem == 'actor_architecture':
                    broken['actor_architecture']['policy_hidden'] = [1, 1]
                elif problem == 'motion_dataset':
                    broken['motion_dataset'] = {'corrupt': True}
                torch.save(broken, path)
                before = {name: copy.deepcopy(getattr(target, name).state_dict()) for name in (*NETWORKS, *OPTIMIZERS)}
                clocks = (target.iteration, target.sim_steps, target.imagination_steps, target._collect_accumulator)
                rng = torch.get_rng_state().clone()
                with self.subTest(problem=problem), self.assertRaises(ValueError):
                    target.load(path)
                for name, values in before.items():
                    assert_tree_equal(self, values, getattr(target, name).state_dict())
                torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
                self.assertEqual((target.iteration, target.sim_steps, target.imagination_steps, target._collect_accumulator), clocks)


if __name__ == '__main__':
    unittest.main()
