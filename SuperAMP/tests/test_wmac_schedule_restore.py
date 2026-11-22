"""Schedule migration must preserve weights, optimizer state and resource history."""

import copy
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from src.training_cost import WALL_TIME_SCOPE
from test_wmac_optimization import control_runner, make_runner


def checkpoint_runner():
    runner, _, _ = make_runner()
    runner.env = SimpleNamespace(common_step_counter=42, num_envs=2)
    runner.started, runner.wall_offset = time.perf_counter(), 20.
    runner._cost_history_complete = True
    runner._cost_trajectory_id = "wmac-test-trajectory"
    runner.iteration, runner.sim_steps, runner.imagination_steps = 17, 128, 45
    runner.best_eval_score = None
    runner._collect_accumulator = 0.
    runner.critic_calibration_updates = 3
    runner.validation = {"validation_model_error": float("inf"),
        "validation_contact_error": float("inf"), "validation_walk_error": float("inf"),
        "validation_recovery_error": float("inf"), "validation_reward_gap": float("inf")}
    return runner


class WmacScheduleRestoreTest(unittest.TestCase):
    def test_physical_budget_counts_initial_and_resume_warmup_without_changing_utd_phase(self):
        for start_iteration, initial_steps in ((0, 0), (17, 128)):
            with self.subTest(start_iteration=start_iteration):
                runner, records, _ = control_runner(start_iteration + 3)
                runner.iteration, runner.sim_steps = start_iteration, initial_steps
                runner.replay.size = 0  # New simulator; checkpoint omitted replay.
                runner.learn()
                self.assertEqual([row["collect_steps"] for row in records], [2, 3, 2])
                self.assertEqual(runner.sim_steps, initial_steps + (64 + 2 + 3 + 2) * 2)
                self.assertEqual(runner._collect_accumulator, .5)

    def test_explicit_migration_preserves_optimizer_weights_counters_and_cost_history(self):
        source = checkpoint_runner()
        source.cfg.collect_steps = 24
        source.cfg.discriminator_updates, source.cfg.discriminator_batch_size = 20, 0
        # Exercise nonempty Adam state, not just an untrained optimizer shell.
        for parameter in source.world_model.parameters():
            parameter.grad = torch.full_like(parameter, .01)
        source.opt_model.step()
        expected = copy.deepcopy(source.world_model.state_dict())
        restored = checkpoint_runner()
        restored.cfg.collect_steps, restored.cfg.utd = 0, .4
        restored.cfg.discriminator_updates, restored.cfg.discriminator_batch_size = 2, 512
        restored.wall_offset, restored.sim_steps = 0., 0
        with tempfile.TemporaryDirectory() as directory, patch("torch.cuda.is_available", return_value=False):
            path = Path(directory) / "fixed_schedule.pt"
            source.save(path, export=False)
            payload = torch.load(path, weights_only=False)
            restored.load(path)
        self.assertEqual([restored._next_collect_count() for _ in range(4)], [2, 3, 2, 3])
        self.assertEqual((restored.iteration, restored.sim_steps, restored.imagination_steps), (17, 128, 45))
        self.assertEqual(restored.critic_calibration_updates, 3)
        self.assertTrue(restored._cost_history_complete)
        self.assertEqual(restored._cost_trajectory_id, source._cost_trajectory_id)
        self.assertEqual(payload["training_cost"]["wall_time_scope"], WALL_TIME_SCOPE)
        self.assertEqual(payload["training_cost"]["physical_transitions"], 128)
        self.assertEqual(payload["training_cost"]["wall_seconds"], payload["wall_seconds"])
        self.assertGreaterEqual(restored.wall_time, payload["wall_seconds"])
        for key, value in expected.items():
            torch.testing.assert_close(restored.world_model.state_dict()[key], value, rtol=0, atol=0)
        before, after = source.opt_model.state_dict(), restored.opt_model.state_dict()
        self.assertEqual(before["param_groups"], after["param_groups"])
        for index, values in before["state"].items():
            for key, value in values.items():
                torch.testing.assert_close(after["state"][index][key], value, rtol=0, atol=0)

    def test_legacy_cost_keeps_known_counts_but_does_not_claim_complete_comparability(self):
        source, restored = checkpoint_runner(), checkpoint_runner()
        with tempfile.TemporaryDirectory() as directory, patch("torch.cuda.is_available", return_value=False):
            path = Path(directory) / "legacy_cost.pt"
            source.save(path, export=False)
            payload = torch.load(path, weights_only=False)
            payload.pop("training_cost")
            torch.save(payload, path)
            restored.load(path)
            restored.save(path, export=False)
            updated = torch.load(path, weights_only=False)
        self.assertFalse(updated["training_cost"]["history_complete"])
        self.assertEqual(updated["training_cost"]["physical_transitions"], 128)
        self.assertGreaterEqual(updated["training_cost"]["wall_seconds"], payload["wall_seconds"])


if __name__ == "__main__":
    unittest.main()
