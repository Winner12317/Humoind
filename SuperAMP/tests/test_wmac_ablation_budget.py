"""Ablation budgets preserve WMAC UTD and full 24-step PPO rollouts."""

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_wmac_ablations.py"
SPEC = importlib.util.spec_from_file_location("wmac_ablation_launcher", SCRIPT)
LAUNCHER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LAUNCHER)


def option(command, name):
    return command[command.index(name) + 1]


class AblationBudgetTest(unittest.TestCase):
    def plan(self, directory, extra=()):
        arguments = [str(SCRIPT), "--profile", "balanced", "--presets", "full",
                     "--seeds", "42", "--num-envs", "2", "--iterations", "3",
                     "--include-baseline", "--log-root", directory, *extra]
        output = io.StringIO()
        with patch("sys.argv", arguments), contextlib.redirect_stdout(output):
            LAUNCHER.main()
        manifests = list(Path(directory).glob("*/manifest.json"))
        self.assertEqual(len(manifests), 1)
        return json.loads(manifests[0].read_text()), output.getvalue()

    def test_utd_budget_includes_warmup_and_ppo_rounding_is_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest, output = self.plan(directory)
        # WMAC uses 64 warmup + floor(3 / .4) = 71 vectorized steps.
        self.assertEqual(manifest["wmac_vectorized_steps_per_run"], 71)
        self.assertEqual(manifest["wmac_physical_transitions_per_run"], 142)
        self.assertEqual(manifest["ppo_vectorized_steps_per_run"], 72)
        self.assertEqual(manifest["ppo_physical_transitions_per_run"], 144)
        self.assertEqual(manifest["ppo_extra_vectorized_steps"], 1)
        self.assertEqual(manifest["ppo_extra_physical_transitions"], 2)
        self.assertIn("extra: 2", output)
        ppo, wmac = manifest["jobs"]
        self.assertEqual(ppo["physical_transitions"], 144)
        self.assertEqual(wmac["physical_transitions"], 142)
        self.assertEqual(option(ppo["command"], "--agent.num-steps-per-env"), "24")
        self.assertEqual(option(ppo["command"], "--agent.max-iterations"), "3")
        self.assertEqual(option(wmac["command"], "--collect-steps"), "0")
        self.assertEqual(option(wmac["command"], "--utd"), "0.4")
        self.assertEqual(option(wmac["command"], "--iterations"), "3")
        self.assertEqual(wmac["resolved_config"]["collect_steps"], 0)
        self.assertEqual(wmac["resolved_config"]["utd"], .4)
        self.assertTrue(all(job["status"] == "planned" for job in manifest["jobs"]))

    def test_exact_budget_needs_no_extra_ppo_interactions(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest, _ = self.plan(directory, ("--iterations", "32", "--strict-baseline-budget"))
        self.assertEqual(manifest["wmac_vectorized_steps_per_run"], 144)
        self.assertEqual(manifest["ppo_vectorized_steps_per_run"], 144)
        self.assertEqual(manifest["ppo_extra_physical_transitions"], 0)
        self.assertEqual(manifest["jobs"][0]["physical_transitions"], manifest["jobs"][1]["physical_transitions"])

    def test_strict_budget_rejects_a_partial_rollout_without_changing_wmac_cadence(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stderr(io.StringIO()) as errors:
            with self.assertRaises(SystemExit):
                self.plan(directory, ("--strict-baseline-budget",))
            self.assertFalse(list(Path(directory).glob("*/manifest.json")))
        self.assertIn("retaining WMAC's sampling cadence", errors.getvalue())
        self.assertIn("preserves the WMAC UTD and PPO rollout length", errors.getvalue())
        self.assertNotIn("--collect-steps 8", errors.getvalue())

    def test_explicit_fixed_collection_override_retains_its_own_real_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest, _ = self.plan(directory, ("--collect-steps", "8"))
        self.assertEqual(manifest["wmac_vectorized_steps_per_run"], 64 + 3 * 8)
        self.assertEqual(manifest["ppo_vectorized_steps_per_run"], 96)
        self.assertEqual(manifest["ppo_extra_vectorized_steps"], 8)
        self.assertLess(manifest["ppo_extra_vectorized_steps"], 24)
        self.assertEqual(option(manifest["jobs"][1]["command"], "--collect-steps"), "8")
        self.assertEqual(option(manifest["jobs"][0]["command"], "--agent.num-steps-per-env"), "24")


if __name__ == "__main__":
    unittest.main()
