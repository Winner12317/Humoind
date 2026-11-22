from dataclasses import asdict, replace
import unittest

from src.wmac.checkpoints import resolve_checkpoint_config
from src.wmac.config import WmacConfig
from src.wmac.reference import (
    IMAGINED_FOOT_SLIP_GEOMETRY,
    IMAGINED_FOOT_SLIP_INSTANTAMP,
    amp_reward_protocol,
    checkpoint_protocol,
    config_protocol,
    imagined_foot_slip_protocol,
    smooth_reward_protocol,
    validate_imagined_foot_slip_resume,
    validate_training_protocol_resume,
)


def fixture(config, version):
    payload = {
        'format': 'amp-wmac-v1', 'implementation_version': version,
        'config': asdict(config), 'training_protocol': config_protocol(config),
        'amp_reward_protocol': amp_reward_protocol(config),
        'smooth_reward_protocol': smooth_reward_protocol(config),
        'actor_type': config.actor_type, 'recovery_counter_semantics': config.recovery_counter_semantics,
        'recovery_height_reward': config.recovery_height_reward,
    }
    from src.wmac.actors import actor_architecture
    payload['actor_architecture'] = actor_architecture(config)
    if version >= 13:
        payload['imagined_foot_slip_protocol'] = imagined_foot_slip_protocol(config)
    return payload


class ImaginedFootSlipProtocolTests(unittest.TestCase):
    def setUp(self):
        self.old = replace(WmacConfig.for_profile('supermimic_official'), smooth_reward_mode='host_humanup_v1',
                           amp_reward_mapping='instantamp_centered_tanh_v1')
        self.new = replace(self.old, imagined_recovery_foot_slip=IMAGINED_FOOT_SLIP_GEOMETRY,
                           recovery_foot_slip_contact_height_m=0., recovery_foot_slip_softness_m=.001)

    def test_version_12_without_new_fields_preserves_old_protocol(self):
        payload = fixture(self.old, 12)
        for field in ('imagined_recovery_foot_slip', 'recovery_foot_slip_contact_height_m',
                      'recovery_foot_slip_softness_m', 'allow_imagined_foot_slip_migration'):
            payload['config'].pop(field)
        restored = resolve_checkpoint_config(payload)
        self.assertEqual(restored.imagined_recovery_foot_slip, 'learned_v1')
        self.assertNotIn('imagined_recovery_foot_slip', checkpoint_protocol(payload))
        self.assertFalse(validate_training_protocol_resume(payload, restored))

    def test_fork_requires_permission_and_cannot_change_other_learning_settings(self):
        source = fixture(self.old, 12)
        target = replace(self.new, resume='/tmp/model_46000.pt')
        with self.assertRaisesRegex(ValueError, 'Imagined foot slip protocol mismatch'):
            validate_training_protocol_resume(source, target)
        target.allow_imagined_foot_slip_migration = True
        self.assertFalse(validate_training_protocol_resume(source, target))
        with self.assertRaisesRegex(ValueError, 'separate reward experiment'):
            validate_training_protocol_resume(source, replace(target, task_weight=.5))

    def test_geometry_resume_locks_geometry_and_does_not_inherit_authorization(self):
        source = fixture(replace(self.new, allow_imagined_foot_slip_migration=True), 13)
        restored = resolve_checkpoint_config(source)
        self.assertFalse(restored.allow_imagined_foot_slip_migration)
        self.assertEqual(restored.recovery_foot_slip_contact_height_m, 0.)
        self.assertFalse(validate_training_protocol_resume(source, restored))
        with self.assertRaisesRegex(ValueError, 'Imagined foot slip protocol mismatch'):
            validate_training_protocol_resume(source, replace(restored, recovery_foot_slip_softness_m=.002))
        source['config']['recovery_foot_slip_softness_m'] = .002
        with self.assertRaisesRegex(ValueError, 'conflicts with its saved config'):
            checkpoint_protocol(source)

    def test_geometry_is_explicit_and_limited_to_flat_hard_recovery(self):
        self.new.validate()
        with self.assertRaisesRegex(ValueError, 'explicit finite contact height'):
            replace(self.new, recovery_foot_slip_softness_m=None).validate()
        with self.assertRaisesRegex(ValueError, 'flat G1'):
            replace(self.new, task='Unitree-G1-AMP-Perceptive').validate()
        with self.assertRaisesRegex(ValueError, 'requires --resume'):
            replace(self.new, allow_imagined_foot_slip_migration=True).validate()

    def test_instantamp_pins_reference_geometry_and_retains_its_own_checkpoint_mode(self):
        instantamp = replace(self.old, imagined_recovery_foot_slip=IMAGINED_FOOT_SLIP_INSTANTAMP)
        self.assertEqual(imagined_foot_slip_protocol(instantamp), {
            'imagined_recovery_foot_slip': IMAGINED_FOOT_SLIP_INSTANTAMP,
            'recovery_foot_slip_contact_height_m': .001, 'recovery_foot_slip_softness_m': .001,
        })
        with self.assertRaisesRegex(ValueError, 'pins contact height and softness'):
            replace(instantamp, recovery_foot_slip_contact_height_m=0.).validate()
        older = fixture(self.old, 12)
        with self.assertRaisesRegex(ValueError, 'Imagined foot slip protocol mismatch'):
            validate_imagined_foot_slip_resume(older, instantamp)
        fork = replace(instantamp, resume='/tmp/model_46000.pt', allow_imagined_foot_slip_migration=True)
        self.assertTrue(validate_imagined_foot_slip_resume(older, fork))
        payload = fixture(fork, 13)
        restored = resolve_checkpoint_config(payload)
        self.assertEqual(restored.imagined_recovery_foot_slip, IMAGINED_FOOT_SLIP_INSTANTAMP)
        self.assertEqual(restored.recovery_foot_slip_contact_height_m, .001)
        self.assertFalse(restored.allow_imagined_foot_slip_migration)
        self.assertFalse(validate_imagined_foot_slip_resume(payload, restored))
        with self.assertRaisesRegex(ValueError, 'Imagined foot slip protocol mismatch'):
            validate_imagined_foot_slip_resume(payload, self.new)


if __name__ == '__main__':
    unittest.main()
