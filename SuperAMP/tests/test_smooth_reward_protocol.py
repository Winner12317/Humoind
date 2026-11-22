"""Reward-only migration guards keep old checkpoint objectives reproducible."""
from __future__ import annotations

import copy
from dataclasses import asdict
import importlib.util
from pathlib import Path
import tempfile
import unittest

import torch

from src.wmac.actors import actor_architecture
from src.wmac.checkpoints import resolve_checkpoint_config
from src.wmac.config import WmacConfig
from src.wmac.reference import (
    amp_reward_protocol, smooth_reward_protocol, checkpoint_smooth_reward_protocol,
    checkpoint_protocol, config_protocol, validate_training_protocol_resume,
    validate_smooth_reward_resume,
)


def payload(*, smooth=False, centered=True):
    cfg = WmacConfig.for_profile('supermimic_official')
    if centered:
        cfg.amp_reward_mapping = 'instantamp_centered_tanh_v1'
    if smooth:
        cfg.smooth_reward_mode = 'host_humanup_v1'
    return dict(
        format='amp-wmac-v1', implementation_version=12 if smooth else (11 if centered else 7),
        config=asdict(cfg), training_protocol=config_protocol(cfg),
        amp_reward_protocol=amp_reward_protocol(cfg), smooth_reward_protocol=smooth_reward_protocol(cfg),
        actor_type=cfg.actor_type, actor_architecture=actor_architecture(cfg),
        recovery_counter_semantics=cfg.recovery_counter_semantics,
        recovery_height_reward=cfg.recovery_height_reward,
    )


SPEC = importlib.util.spec_from_file_location(
    'smooth_wmac_cli', Path(__file__).resolve().parents[1]/'scripts/train_wmac.py')
CLI = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CLI)


class SmoothRewardProtocolTest(unittest.TestCase):
    def test_historical_none_keeps_exact_three_or_six_training_keys(self):
        for centered, count in ((False, 3), (True, 6)):
            source = payload(centered=centered)
            self.assertEqual(len(source['training_protocol']), count)
            source.pop('smooth_reward_protocol')
            for key in tuple(source['config']):
                if key.startswith('smooth_') or key == 'allow_smooth_reward_migration':
                    source['config'].pop(key)
            self.assertEqual(checkpoint_smooth_reward_protocol(source), {'smooth_reward_mode': 'none'})
            self.assertEqual(len(checkpoint_protocol(source)), count)
            self.assertEqual(resolve_checkpoint_config(source).smooth_reward_mode, 'none')

    def test_active_mode_requires_full_config_and_versioned_metadata(self):
        source = payload(smooth=True)
        self.assertEqual(checkpoint_protocol(source), source['training_protocol'])
        for key in ('smooth_reward_mode', 'smooth_action_second_coef', 'smooth_joint_velocity_coef',
                    'smooth_torque_coef', 'smooth_power_coef'):
            broken = copy.deepcopy(source)
            broken['config'].pop(key)
            with self.subTest(key=key), self.assertRaises(ValueError):
                checkpoint_protocol(broken)
        for key in ('training_protocol', 'smooth_reward_protocol'):
            broken = copy.deepcopy(source)
            broken.pop(key)
            with self.subTest(key=key), self.assertRaises(ValueError):
                checkpoint_protocol(broken)
        broken = copy.deepcopy(source)
        broken['implementation_version'] = 11
        with self.assertRaisesRegex(ValueError, 'version >= 12'):
            checkpoint_protocol(broken)

    def test_formulas_proxy_and_no_curriculum_are_part_of_protocol(self):
        source = payload(smooth=True)
        expected = source['smooth_reward_protocol']
        self.assertFalse(expected['action_amplitude_curriculum'])
        self.assertFalse(expected['world_model_changed'])
        self.assertIn('request_proxy', expected['torque_semantics'])
        for key, replacement in (('formula_version', 'other'),
                                 ('action_amplitude_curriculum', True),
                                 ('world_model_changed', True),
                                 ('scaling', 'unscaled')):
            broken = copy.deepcopy(source)
            broken['smooth_reward_protocol'][key] = replacement
            with self.subTest(key=key), self.assertRaises(ValueError):
                checkpoint_protocol(broken)

    def test_migration_requires_its_own_one_way_authorization(self):
        source = payload()
        target = resolve_checkpoint_config(source)
        target.smooth_reward_mode = 'host_humanup_v1'
        target.allow_amp_reward_migration = True
        with self.assertRaisesRegex(ValueError, 'allow-smooth-reward-migration'):
            validate_training_protocol_resume(source, target)
        target.allow_smooth_reward_migration = True
        self.assertTrue(validate_smooth_reward_resume(source, target))
        self.assertFalse(validate_training_protocol_resume(source, target))  # AMP did not change.
        new = payload(smooth=True)
        new['config']['allow_smooth_reward_migration'] = True
        restored = resolve_checkpoint_config(new)
        self.assertFalse(restored.allow_smooth_reward_migration)
        self.assertFalse(validate_smooth_reward_resume(new, restored))
        restored.smooth_reward_mode = 'none'
        restored.allow_smooth_reward_migration = True
        with self.assertRaises(ValueError):
            validate_smooth_reward_resume(new, restored)

    def test_active_resume_locks_all_coefficients_even_with_migration_flag(self):
        source = payload(smooth=True)
        for key in ('smooth_action_second_coef', 'smooth_joint_velocity_coef',
                    'smooth_torque_coef', 'smooth_power_coef'):
            target = resolve_checkpoint_config(source)
            target.allow_smooth_reward_migration = True
            setattr(target, key, getattr(target, key) * 2 + .001)
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'ordinary smooth resume'):
                validate_training_protocol_resume(source, target)

    def test_both_migrations_independent_and_cannot_change_base_method(self):
        source = payload(centered=False)
        target = resolve_checkpoint_config(source)
        target.smooth_reward_mode = 'host_humanup_v1'
        target.amp_reward_mapping = 'instantamp_centered_tanh_v1'
        target.allow_smooth_reward_migration = True
        with self.assertRaisesRegex(ValueError, 'AMP reward protocol mismatch'):
            validate_training_protocol_resume(source, target)
        target.allow_amp_reward_migration = True
        self.assertTrue(validate_training_protocol_resume(source, target))
        for key, value in (('optimizer_algorithm', 'adam'), ('mirror_walk_motions', False)):
            changed = copy.deepcopy(target)
            setattr(changed, key, value)
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'Training protocol mismatch'):
                validate_training_protocol_resume(source, changed)

    def test_nonnegative_finite_coefficients_and_bool_flags_are_validated(self):
        for key in ('smooth_action_second_coef', 'smooth_joint_velocity_coef',
                    'smooth_torque_coef', 'smooth_power_coef'):
            for value in (-.1, float('nan'), float('inf'), True, '1'):
                cfg = WmacConfig.for_profile('supermimic_official')
                setattr(cfg, key, value)
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    cfg.validate()
        for key, value in (('smooth_reward_mode', 'unknown'), ('allow_smooth_reward_migration', 1)):
            cfg = WmacConfig.for_profile('supermimic_official')
            setattr(cfg, key, value)
            with self.subTest(key=key), self.assertRaises(ValueError):
                cfg.validate()

    def test_old_reader_rejects_new_training_markers(self):
        source = payload(smooth=True)
        previous = {key: value for key, value in source['training_protocol'].items()
                    if not key.startswith('smooth_')}
        self.assertNotEqual(source['training_protocol'], previous)
        self.assertEqual(len(previous), 6)

    def test_cli_explicit_migration_then_ordinary_resume_and_height_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'centered.pt'
            torch.save(payload(), path)
            flags = ['--resume', str(path), '--smooth-reward-mode', 'host_humanup_v1']
            with self.assertRaises(ValueError):
                CLI.parse_config(flags)
            migrated = CLI.parse_config([*flags, '--allow-smooth-reward-migration'])
            self.assertEqual(migrated.smooth_reward_mode, 'host_humanup_v1')
            self.assertEqual(migrated.wm_updates, 10)
            self.assertEqual(migrated.real_critic_coef, 0.)
            with self.assertRaisesRegex(ValueError, 'cannot also change'):
                CLI.parse_config([*flags, '--allow-smooth-reward-migration',
                                  '--recovery-height-reward', 'height_only_v1',
                                  '--allow-recovery-semantics-migration'])
            smooth = Path(directory)/'smooth.pt'
            torch.save(payload(smooth=True), smooth)
            restored = CLI.parse_config(['--resume', str(smooth)])
            self.assertFalse(restored.allow_smooth_reward_migration)
            with self.assertRaises(ValueError):
                CLI.parse_config(['--resume', str(smooth), '--smooth-torque-coef', '1'])


if __name__ == '__main__':
    unittest.main()
