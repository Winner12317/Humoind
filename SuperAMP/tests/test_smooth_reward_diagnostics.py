"""Historical objective probes must not silently omit the new reward terms."""
from __future__ import annotations

from dataclasses import asdict
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from src.wmac.actors import actor_architecture
from src.wmac.checkpoints import resolve_checkpoint_config
from src.wmac.reference import config_protocol, amp_reward_protocol, smooth_reward_protocol
from test_wmac_schedule_restore import checkpoint_runner


def script(name):
    spec = importlib.util.spec_from_file_location(
        'smooth_diagnostic_' + name, Path(__file__).resolve().parents[1]/'scripts'/(name+'.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PHYSICS = script('check_wmac_physics')
DIAGNOSE = script('diagnose_wmac')
PHASES = script('probe_wmac_recorded_phases')
EVALUATE = script('evaluate_wmac')


def smooth_actor_payload():
    runner = checkpoint_runner()
    runner.cfg.smooth_reward_mode = 'host_humanup_v1'
    cfg = runner.cfg
    payload = dict(format='amp-wmac-v1', implementation_version=12, config=asdict(cfg),
                   training_protocol=config_protocol(cfg),
                   amp_reward_protocol=amp_reward_protocol(cfg),
                   smooth_reward_protocol=smooth_reward_protocol(cfg),
                   actor_type=cfg.actor_type, actor_architecture=actor_architecture(cfg),
                   actor=runner.actor.state_dict(),
                   recovery_counter_semantics=cfg.recovery_counter_semantics,
                   recovery_height_reward=cfg.recovery_height_reward)
    return runner, payload


class SmoothRewardDiagnosticTest(unittest.TestCase):
    def test_historical_physics_loader_rejects_smooth_even_with_amp_mapper_opt_in(self):
        _, payload = smooth_actor_payload()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'smooth.pt'
            torch.save(payload, path)
            for include_mapper in (False, True):
                with self.subTest(include_mapper=include_mapper), self.assertRaisesRegex(
                        ValueError, 'do not support the new smooth reward objective'):
                    PHYSICS.load_checkpoint(path, 'cpu', include_reward_mapper=include_mapper)

    def test_manual_direction_probe_cannot_bypass_objective_guard(self):
        modules = (SimpleNamespace(smooth_reward_mode='host_humanup_v1'), None, None, None, None, None)
        with self.assertRaisesRegex(ValueError, 'new smooth reward objective'):
            PHYSICS.check_directions(None, modules, 2, 1, .01)

    def test_component_and_phase_probes_reject_missing_smooth_objective(self):
        _, payload = smooth_actor_payload()
        with self.assertRaisesRegex(ValueError, 'new smooth reward objective'):
            DIAGNOSE.resolved_config(payload)
        cfg = resolve_checkpoint_config(payload)
        with self.assertRaisesRegex(ValueError, 'new smooth reward objective'):
            PHASES.validate_bundles({22000: (cfg,)})

    def test_actor_only_loader_keeps_actions_and_labels_eval_baseline_reward(self):
        runner, payload = smooth_actor_payload()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'smooth.pt'
            torch.save(payload, path)
            with patch('src.wmac.networks.WorldModel', side_effect=AssertionError('No model is needed for Actor evaluation')):
                actor, kind = EVALUATE.load_actor(path, 'cpu')
        self.assertEqual(kind, 'amp-wmac-v1')
        observation = torch.randn(3, 384)
        torch.testing.assert_close(actor(observation), runner.actor.float()(observation))
        metadata = EVALUATE.evaluation_source_metadata(actor)
        self.assertEqual(metadata['checkpoint_training_smooth_reward_protocol'], smooth_reward_protocol(runner.cfg))
        self.assertEqual(metadata['evaluation_task_reward_scope'],
                         'baseline_physical_task_without_training_smooth_terms')
        self.assertFalse(metadata['source_provenance']['training_smooth_terms_used_during_physical_evaluation'])

    def test_old_actor_metadata_keeps_same_common_physical_reward_scope(self):
        metadata = EVALUATE.evaluation_source_metadata(SimpleNamespace())
        self.assertEqual(metadata['checkpoint_training_smooth_reward_protocol'], {'smooth_reward_mode': 'none'})
        self.assertEqual(metadata['evaluation_task_reward_scope'],
                         'baseline_physical_task_without_training_smooth_terms')


if __name__ == '__main__':
    unittest.main()
