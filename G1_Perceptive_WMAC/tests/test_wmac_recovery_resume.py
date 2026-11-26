"""Recovery interventions are explicit, versioned forks of a saved experiment."""

import copy
import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import torch

from src.wmac.checkpoints import resolve_checkpoint_config
from test_wmac_schedule_restore import checkpoint_runner


class RecoveryResumeTest(unittest.TestCase):
    def test_new_checkpoint_saves_both_independent_semantics(self):
        runner = checkpoint_runner()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"new.pt"
            runner.save(path, export=False)
            payload = torch.load(path, weights_only=False)
        self.assertEqual(payload["implementation_version"], 4)
        for key in ("recovery_counter_semantics", "recovery_height_reward"):
            self.assertEqual(payload[key], payload["config"][key])
        self.assertEqual(payload["recovery_counter_semantics"], "hard_forward_v2")
        self.assertEqual(payload["recovery_height_reward"], "upright_v2")
        self.assertIn("biased_straight_through", payload["recovery_derivative"])

    def test_cross_semantics_rejects_before_any_saved_state_is_applied(self):
        for field, old_value in (("recovery_counter_semantics", "soft_v1"),
                                 ("recovery_height_reward", "height_only_v1")):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                source, restored = checkpoint_runner(), checkpoint_runner()
                setattr(source.cfg, field, old_value)
                path = Path(directory)/"old.pt"
                source.save(path, export=False)
                with torch.no_grad():
                    next(restored.actor.parameters()).add_(.25)
                restored.iteration, restored.sim_steps = 1234, 5678
                before = copy.deepcopy(restored.actor.state_dict())
                rng_before = torch.get_rng_state().clone()
                opt_before = copy.deepcopy(restored.opt_model.state_dict())
                with self.assertRaisesRegex(ValueError, "allow-recovery-semantics-migration"):
                    restored.load(path)
                self.assertEqual((restored.iteration, restored.sim_steps), (1234, 5678))
                torch.testing.assert_close(torch.get_rng_state(), rng_before, rtol=0, atol=0)
                self.assertEqual(restored.opt_model.state_dict()["param_groups"], opt_before["param_groups"])
                for key, value in before.items():
                    torch.testing.assert_close(restored.actor.state_dict()[key], value, rtol=0, atol=0)

    def test_explicit_migration_preserves_weights_adam_and_records_source(self):
        source, restored = checkpoint_runner(), checkpoint_runner()
        source.cfg.recovery_counter_semantics = "soft_v1"
        source.cfg.recovery_height_reward = "height_only_v1"
        for parameter in source.world_model.parameters():
            parameter.grad = torch.full_like(parameter, .01)
        source.opt_model.step()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"source.pt"
            source.save(path, export=False)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            restored.load(path, allow_recovery_semantics_migration=True)
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)
            record = restored.recovery_semantics_history[-1]
            self.assertEqual(record["source_sha256"], digest)
            self.assertEqual(record["source_iteration"], source.iteration)
            self.assertEqual(set(record["changes"]), {"recovery_counter_semantics", "recovery_height_reward"})
            self.assertTrue(record["networks_and_optimizers_preserved"])
            self.assertEqual(restored.critic_calibration_updates, source.critic_calibration_updates)
            for key, value in source.world_model.state_dict().items():
                torch.testing.assert_close(restored.world_model.state_dict()[key], value, rtol=0, atol=0)
            before, after = source.opt_model.state_dict(), restored.opt_model.state_dict()
            self.assertEqual(after["param_groups"], before["param_groups"])
            for index, values in before["state"].items():
                for key, value in values.items():
                    torch.testing.assert_close(after["state"][index][key], value, rtol=0, atol=0)
            output = Path(directory)/"migrated.pt"
            restored.save(output, export=False)
            saved = torch.load(output, weights_only=False)
            self.assertEqual(saved["recovery_semantics_history"][-1], record)

    def test_old_unlabelled_resume_retains_original_rules_without_migration(self):
        source, restored = checkpoint_runner(), checkpoint_runner()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"historical.pt"
            source.save(path, export=False)
            payload = torch.load(path, weights_only=False)
            payload["implementation_version"] = 3
            for key in ("recovery_counter_semantics", "recovery_height_reward"):
                payload.pop(key)
                payload["config"].pop(key)
            torch.save(payload, path)
            restored.cfg = resolve_checkpoint_config(payload)
            restored.load(path)
        self.assertEqual(restored.cfg.recovery_counter_semantics, "soft_v1")
        self.assertEqual(restored.cfg.recovery_height_reward, "height_only_v1")
        self.assertEqual(restored.adapter.recovery_counter_semantics, "soft_v1")
        self.assertEqual(restored.recovery_semantics_history, [])

    def test_height_migration_does_not_restore_replay_with_old_reward_labels(self):
        source, restored = checkpoint_runner(), checkpoint_runner()
        source.cfg.recovery_height_reward = "height_only_v1"
        source.cfg.checkpoint_replay = True
        source.replay = SimpleNamespace(state_dict=lambda: {"old_reward": 1.})
        restored.replay = SimpleNamespace(load_state_dict=Mock(), mark_boundary=Mock(), size=0)
        restored.cfg.allow_recovery_semantics_migration = True
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"with_replay.pt"
            source.save(path, export=False)
            restored.load(path)
        restored.replay.load_state_dict.assert_not_called()
        self.assertTrue(restored.recovery_semantics_history[-1]["saved_replay_discarded"])

    def test_counter_only_migration_keeps_compatible_physical_reward_labels(self):
        source, restored = checkpoint_runner(), checkpoint_runner()
        source.cfg.recovery_counter_semantics = "soft_v1"
        source.cfg.checkpoint_replay = True
        source.replay = SimpleNamespace(state_dict=lambda: {"physical_reward": 1.})
        restored.replay = SimpleNamespace(load_state_dict=Mock(), mark_boundary=Mock())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"with_replay.pt"
            source.save(path, export=False)
            restored.load(path, allow_recovery_semantics_migration=True)
        restored.replay.load_state_dict.assert_called_once_with({"physical_reward": 1.})
        restored.replay.mark_boundary.assert_called_once()
        self.assertFalse(restored.recovery_semantics_history[-1]["saved_replay_discarded"])

    def test_height_migration_rejects_nonempty_target_replay_before_loading_weights(self):
        source, restored = checkpoint_runner(), checkpoint_runner()
        source.cfg.recovery_height_reward = "height_only_v1"
        restored.replay = SimpleNamespace(size=3)
        restored.iteration = 999
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"source.pt"
            source.save(path, export=False)
            with self.assertRaisesRegex(ValueError, "fresh empty target replay"):
                restored.load(path, allow_recovery_semantics_migration=True)
        self.assertEqual(restored.iteration, 999)

    def test_malformed_warmstart_semantics_rejects_before_loading_actor(self):
        source, restored = checkpoint_runner(), checkpoint_runner()
        before = copy.deepcopy(restored.actor.state_dict())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"malformed.pt"
            source.save(path, export=False)
            payload = torch.load(path, weights_only=False)
            payload["config"]["recovery_counter_semantics"] = "unknown_rule"
            payload["recovery_counter_semantics"] = "unknown_rule"
            with torch.no_grad():
                for key, value in payload["actor"].items():
                    if value.is_floating_point():
                        value.add_(.2)
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, "Unknown checkpoint"):
                restored.initialize_from_policy(path)
        for key, value in before.items():
            torch.testing.assert_close(restored.actor.state_dict()[key], value, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
