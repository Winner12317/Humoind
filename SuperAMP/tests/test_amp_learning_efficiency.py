"""Read-only, protocol-checked physical performance/resource comparisons."""

from __future__ import annotations

import importlib.util
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

from src.training_cost import WALL_TIME_SCOPE

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/compare_amp_learning_efficiency.py"
SPEC = importlib.util.spec_from_file_location("amp_learning_efficiency_test_cli", SCRIPT)
CLI = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CLI
SPEC.loader.exec_module(CLI)


def write_metrics(root, arm, index, *, success=.9, rise_time=1.8, transitions=None, wall=None,
                  cost_metadata=True, complete=True, gpus=1, seed=2026, initial="identical-state", scope=WALL_TIME_SCOPE):
    directory = root / arm / f"eval_{index}"
    directory.mkdir(parents=True)
    payload = {
        "checkpoint": f"/runs/{arm}/model_{index}.pt",
        "protocol": {"first_episode_only": True, "task": "Unitree-G1-AMP-Flat-Aligned"},
        "arguments": {"seed": seed, "steps": 500, "num_envs": 32,
                      "checkpoint": f"/runs/{arm}/model_{index}.pt", "output": str(directory)},
        "results": {"recovery": {
            "initial_state_sha256": initial, "num_envs": 32, "steps_requested": 500, "control_dt": .02,
            "recovery_success_fraction": success, "first_stable_recovery_seconds_mean": rise_time,
        }},
    }
    if cost_metadata:
        payload["training_cost"] = {
            "physical_transitions": (index + 1) * 100 if transitions is None else transitions,
            "wall_seconds": (index + 1) * 60 if wall is None else wall,
            "gpu_count": gpus, "wall_time_scope": scope, "history_complete": complete,
        }
    path = directory / "metrics.json"
    path.write_text(json.dumps(payload))
    return path


class LearningEfficiencyTest(unittest.TestCase):
    def compare(self, root, **kwargs):
        return CLI.compare_learning_efficiency(
            [root / "wm"], [root / "ppo"],
            minimum=["recovery.recovery_success_fraction=.8"],
            maximum=["recovery.first_stable_recovery_seconds_mean=2"], **kwargs)

    def test_first_observed_crossing_requires_all_thresholds_and_preserves_sampling_costs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_metrics(root, "wm", 0, success=.4)
            write_metrics(root, "wm", 1, rise_time=2.4)
            wm_pass = write_metrics(root, "wm", 2, transitions=700, wall=240)
            write_metrics(root, "wm", 3, success=.1, transitions=800, wall=300)
            write_metrics(root, "ppo", 0, success=.6)
            ppo_pass = write_metrics(root, "ppo", 1, transitions=500, wall=360)
            report = self.compare(root)
            self.assertTrue(report["training_cost_comparison_available"])
            self.assertEqual(report["arms"]["wm"]["first_threshold_crossing"]["metrics_path"], str(wm_pass))
            self.assertEqual(report["arms"]["ppo"]["first_threshold_crossing"]["metrics_path"], str(ppo_pass))
            self.assertAlmostEqual(report["ratios"]["wm_over_ppo_physical_transitions"], 1.4)
            self.assertAlmostEqual(report["ratios"]["wm_over_ppo_wall_seconds"], 2 / 3)
            self.assertIn("not stable convergence", report["interpretation"])

    def test_not_reached_and_null_success_time_are_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_metrics(root, "wm", 0, rise_time=None)
            write_metrics(root, "ppo", 0)
            report = self.compare(root)
            self.assertEqual(report["arms"]["wm"]["status"], "not_reached")
            self.assertIsNone(report["arms"]["wm"]["first_threshold_crossing"])
            self.assertFalse(report["training_cost_comparison_available"])
            self.assertIsNone(report["ratios"])

    def test_old_or_incomplete_history_is_never_used_as_trusted_cost(self):
        for missing, complete, scope in ((True, True, WALL_TIME_SCOPE),
                                          (False, False, WALL_TIME_SCOPE),
                                          (False, True, "learning updates only")):
            with self.subTest(missing=missing, complete=complete, scope=scope), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                write_metrics(root, "wm", 0, cost_metadata=not missing, complete=complete, scope=scope)
                write_metrics(root, "wm", 1)
                write_metrics(root, "ppo", 0)
                report = self.compare(root)
                self.assertEqual(report["arms"]["wm"]["status"], "cost_unavailable")
                self.assertIsNone(report["arms"]["wm"]["first_threshold_crossing"])
                self.assertFalse(report["training_cost_comparison_available"])

    def test_protocol_seed_and_initial_hash_mismatches_are_rejected(self):
        for values in ({"seed": 2030}, {"initial": "different-state"}):
            with self.subTest(values=values), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                write_metrics(root, "wm", 0)
                write_metrics(root, "ppo", 0, **values)
                with self.assertRaisesRegex(ValueError, "protocol/arguments/initial-state"):
                    self.compare(root)

    def test_different_gpu_counts_allow_resource_hours_with_hardware_caveat(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_metrics(root, "wm", 0, wall=1800, gpus=2)
            write_metrics(root, "ppo", 0, wall=3600, gpus=1)
            report = self.compare(root)
            self.assertTrue(report["different_gpu_counts"])
            self.assertAlmostEqual(report["ratios"]["wm_over_ppo_wall_seconds"], .5)
            self.assertAlmostEqual(report["ratios"]["wm_over_ppo_gpu_hours"], 1.)

    def test_costs_must_be_monotonic_within_one_training_trajectory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_metrics(root, "wm", 0, transitions=100, wall=30)
            write_metrics(root, "wm", 1, transitions=200, wall=20)
            write_metrics(root, "ppo", 0)
            with self.assertRaisesRegex(ValueError, "not monotonic"):
                self.compare(root)

    def test_no_threshold_and_metric_typos_fail_clearly(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            CLI.compare_learning_efficiency([], [])
        with self.assertRaisesRegex(ValueError, "Invalid"):
            CLI.parse_condition("recovery.bad=nan", ">=")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = write_metrics(root, "wm", 0)
            write_metrics(root, "ppo", 0)
            with self.assertRaisesRegex(ValueError, "Missing metric"):
                CLI.compare_learning_efficiency([path], [root / "ppo"], minimum=["recovery.typo=1"])

    def test_matching_trajectory_id_allows_checkpoint_directories_to_change_on_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index in (0, 1):
                path = write_metrics(root, "wm", index)
                payload = json.loads(path.read_text())
                payload["checkpoint"] = f"/runs/wm-resume-{index}/model_{index}.pt"
                payload["training_cost"]["trajectory_id"] = "one-wm-trajectory"
                path.write_text(json.dumps(payload))
            write_metrics(root, "ppo", 0)
            self.assertTrue(self.compare(root)["training_cost_comparison_available"])

    def test_different_trajectory_ids_are_rejected_even_in_one_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index in (0, 1):
                path = write_metrics(root, "wm", index)
                payload = json.loads(path.read_text())
                payload["training_cost"]["trajectory_id"] = f"different-wm-trajectory-{index}"
                path.write_text(json.dumps(payload))
            write_metrics(root, "ppo", 0)
            with self.assertRaisesRegex(ValueError, "trajectory_id"):
                self.compare(root)

    def test_missing_trajectory_id_keeps_directory_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = write_metrics(root, "wm", 0)
            second = write_metrics(root, "wm", 1)
            payload = json.loads(second.read_text())
            payload["checkpoint"] = "/some-other-run/model_1.pt"
            second.write_text(json.dumps(payload))
            write_metrics(root, "ppo", 0)
            with self.assertRaisesRegex(ValueError, "checkpoint directories"):
                self.compare(root)

    def test_cli_output_never_overwrites_existing_report_or_input_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wm_path = write_metrics(root, "wm", 0)
            write_metrics(root, "ppo", 0)
            report_path = root / "existing_report.json"
            report_path.write_text("original report\n")
            for output in (wm_path, report_path):
                before = output.read_bytes()
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
                    CLI.main(["--wm", str(root / "wm"), "--ppo", str(root / "ppo"),
                              "--minimum", "recovery.recovery_success_fraction=.8", "--output", str(output)])
                self.assertEqual(raised.exception.code, 2)
                self.assertEqual(output.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
