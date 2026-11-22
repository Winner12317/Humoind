"""Centered checkpoints must be scored with their saved EMA in diagnostics."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

from src.wmac.amp_rewards import CenteredAmpReward
from src.wmac.runner import differentiable_amp_reward
from test_centered_amp_protocol import labelled_payload
from test_wmac_schedule_restore import checkpoint_runner


REPO = Path(__file__).resolve().parents[1]


def script_module(name):
    spec = importlib.util.spec_from_file_location("centered_diagnostic_" + name, REPO / "scripts" / (name + ".py"))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


PHYSICS = script_module("check_wmac_physics")
DIAGNOSE = script_module("diagnose_wmac")
PHASES = script_module("probe_wmac_recorded_phases")


class CenteredAmpDiagnosticTest(unittest.TestCase):
    def test_optional_mapper_preserves_score_graph_and_historical_default(self):
        discriminator = nn.Linear(4, 1)
        with torch.no_grad():
            discriminator.weight.fill_(1.)
            discriminator.bias.fill_(-2.)
        first = torch.zeros(3, 2, requires_grad=True)
        second = torch.zeros(3, 2, requires_grad=True)
        normalizer = nn.Identity()
        mapper = CenteredAmpReward()
        mapper.update(torch.tensor([-2.]), torch.tensor([1.]))
        old, old_score = differentiable_amp_reward(discriminator, normalizer, first, second, .1)
        reward, score = differentiable_amp_reward(discriminator, normalizer, first, second, .1,
                                                 reward_mapper=mapper)
        torch.testing.assert_close(old_score, score)
        torch.testing.assert_close(old, torch.zeros(3))
        torch.testing.assert_close(reward, torch.full((3,), .05))
        gradient = torch.autograd.grad(reward.sum(), first)[0]
        torch.testing.assert_close(gradient, torch.full_like(first, .05))

    def test_loader_preserves_six_tuple_for_legacy_and_restores_saved_mapper_when_opted_in(self):
        source = checkpoint_runner()
        with tempfile.TemporaryDirectory() as directory, patch("torch.cuda.is_available", return_value=False):
            path = Path(directory) / "policy.pt"
            source.save(path, export=False)
            self.assertEqual(len(PHYSICS.load_checkpoint(path, "cpu")), 6)
            self.assertEqual(len(PHYSICS.load_checkpoint(path, "cpu", include_reward_mapper=True)), 7)
            source.cfg.amp_reward_mapping = "instantamp_centered_tanh_v1"
            source.amp_reward_stats = CenteredAmpReward()
            source.amp_reward_stats.update(torch.tensor([-2.]), torch.tensor([.8]))
            source.save(path, export=False)
            with self.assertRaisesRegex(ValueError, "does not support centered AMP"):
                PHYSICS.load_checkpoint(path, "cpu")
            cfg, *modules, mapper = PHYSICS.load_checkpoint(path, "cpu", include_reward_mapper=True)
            self.assertEqual(cfg.amp_reward_mapping, "instantamp_centered_tanh_v1")
            torch.testing.assert_close(mapper.robot_mean, torch.tensor(-2.))
            torch.testing.assert_close(mapper(torch.tensor([-2.])), torch.tensor([.05]))
            self.assertEqual(int(mapper.updates), 1)

    def test_mapper_rejects_uninitialized_or_corrupt_saved_statistics(self):
        cfg = checkpoint_runner().cfg
        cfg.amp_reward_mapping = "instantamp_centered_tanh_v1"
        mapper = CenteredAmpReward()
        with self.assertRaisesRegex(ValueError, "initialized saved robot-score EMA"):
            PHYSICS.checkpoint_reward_mapper({"amp_reward_state": mapper.state_dict()}, cfg, "cpu")
        mapper.update(torch.tensor([-2.]), torch.tensor([1.]))
        bad = mapper.state_dict()
        bad["robot_mean"] = torch.tensor(1e300, dtype=torch.float64)
        with self.assertRaises(ValueError):
            PHYSICS.checkpoint_reward_mapper({"amp_reward_state": bad}, cfg, "cpu")

    def test_historical_component_and_recorded_phase_probes_reject_centered_explicitly(self):
        payload = labelled_payload(centered=True)
        with self.assertRaisesRegex(ValueError, "do not support centered AMP"):
            DIAGNOSE.resolved_config(payload)
        from src.wmac.checkpoints import resolve_checkpoint_config
        cfg = resolve_checkpoint_config(payload)
        with self.assertRaisesRegex(ValueError, "do not support centered AMP"):
            PHASES.validate_bundles({22000: (cfg,)})


if __name__ == "__main__":
    unittest.main()
