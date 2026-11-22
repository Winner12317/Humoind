"""Detect configuration drift across the opt-in PPO/WMAC comparison arms."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

import torch

from src.wmac.config import WmacConfig

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "compare_amp_parameters.py"
SPEC = importlib.util.spec_from_file_location("amp_parameter_comparison", SCRIPT)
COMPARISON = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COMPARISON)


class ParameterAlignmentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = COMPARISON.build_comparison()

    def test_actual_shared_network_counts_and_parameter_contract(self):
        report = self.report
        self.assertTrue(report["all_common_parameters_match"],
                        [row for row in report["common_parameters"] if not row["matched"]])
        self.assertEqual(report["network_parameters"]["actor"], {"wmac": 365085, "ppo": 365085})
        self.assertEqual(report["network_parameters"]["critic"], {"wmac": 234497, "ppo": 234497})
        self.assertEqual(report["network_parameters"]["discriminator"]["wmac"],
                         report["network_parameters"]["discriminator"]["ppo"])
        self.assertEqual(report["network_parameters"]["ppo_distribution_extra"], 29)
        rows = {row["parameter"]: row for row in report["common_parameters"]}
        self.assertEqual(rows["discriminator_input_dim"]["wmac"], 406)
        self.assertEqual(rows["critic_observation_dim"]["ppo"], 136)
        self.assertEqual(len(rows), 39)
        self.assertFalse({"training_total_iterations", "physical_collection_steps_per_iteration",
                          "discriminator_batch_size", "discriminator_updates_per_iteration"} & rows.keys())
        wm, ppo = report["method_budgets"]["wmac"], report["method_budgets"]["ppo"]
        self.assertEqual(wm["configured_collect_steps"], 0)
        self.assertEqual(wm["configured_utd"], .4)
        self.assertEqual(wm["possible_collection_steps_per_outer_iteration"], [2, 3])
        self.assertEqual(wm["real_transitions_per_outer_iteration_mean"], 10240)
        self.assertEqual(ppo["real_transitions_per_outer_iteration_mean"], 98304)
        self.assertEqual(wm["discriminator_batch_size_per_class"], 512)
        self.assertEqual(wm["discriminator_policy_pairs_per_outer_iteration"], 1024)
        self.assertEqual(wm["discriminator_expert_pairs_per_outer_iteration"], 1024)
        self.assertEqual(ppo["discriminator_policy_pairs_per_outer_iteration"], 491520)
        self.assertEqual(ppo["discriminator_expert_pairs_per_outer_iteration"], 491520)
        self.assertEqual(wm["maximum_outer_iterations"], 320000)
        self.assertEqual(ppo["maximum_outer_iterations"], 100001)

    def test_drift_is_reported_instead_of_assuming_equivalence(self):
        cfg = WmacConfig.for_profile("ppo_aligned")
        cfg.actor_lr = 0.002
        cfg.critic_hidden = (256, 128)
        cfg.amp_normalizer_clip = 5.
        report = COMPARISON.build_comparison(wmac_config=cfg)
        self.assertFalse(report["all_common_parameters_match"])
        different = {row["parameter"] for row in report["common_parameters"] if not row["matched"]}
        self.assertTrue({"actor_optimizer", "critic_hidden", "critic_network_parameters", "amp_normalizer"} <= different)
        self.assertNotEqual(report["network_parameters"]["critic"]["wmac"],
                            report["network_parameters"]["critic"]["ppo"])

    def test_env_override_resolves_actual_d_batches_without_mutating_input_or_rng(self):
        cfg = WmacConfig.for_profile("ppo_aligned")
        rng = torch.random.get_rng_state().clone()
        report = COMPARISON.build_comparison(16, wmac_config=cfg)
        torch.testing.assert_close(torch.random.get_rng_state(), rng)
        self.assertEqual(cfg.num_envs, 4096)
        rows = {row["parameter"]: row for row in report["common_parameters"]}
        self.assertEqual(rows["num_envs"]["wmac"], 16)
        self.assertEqual(report["method_budgets"]["wmac"]["discriminator_batch_size_per_class"], 512)
        self.assertEqual(report["method_budgets"]["ppo"]["discriminator_batch_size_per_class"], 96)
        self.assertTrue(report["all_common_parameters_match"])

    def test_independent_training_budgets_do_not_break_the_shared_parameter_contract(self):
        cfg = WmacConfig.for_profile("ppo_aligned")
        cfg.iterations = 60000
        cfg.collect_steps = 5
        cfg.discriminator_batch_size = 128
        cfg.discriminator_updates = 3
        report = COMPARISON.build_comparison(wmac_config=cfg)
        self.assertTrue(report["all_common_parameters_match"])
        budget = report["method_budgets"]["wmac"]
        self.assertEqual(budget["maximum_outer_iterations"], 60000)
        self.assertEqual(budget["collection_mode"], "fixed_steps_override")
        self.assertEqual(budget["configured_utd"], .4)
        self.assertEqual(budget["effective_outer_updates_per_vectorized_collection_step"], .2)
        self.assertEqual(budget["possible_collection_steps_per_outer_iteration"], [5])
        self.assertEqual(budget["discriminator_policy_pairs_per_outer_iteration"], 384)
        self.assertEqual(budget["discriminator_expert_pairs_per_outer_iteration"], 384)

    def test_report_keeps_algorithmic_differences_explicit(self):
        differences = {row["parameter"]: row for row in self.report["remaining_differences"]}
        self.assertEqual(differences["actor_update_budget"]["wmac"]["imagined_horizon"], 32)
        self.assertEqual(differences["actor_update_budget"]["ppo"]["physical_rollout_steps"], 24)
        self.assertEqual(differences["action_output"]["wmac"]["soft_tanh_limit"], 20.)
        self.assertIsNone(differences["action_output"]["ppo"]["clip_actions"])
        self.assertEqual(differences["action_exploration"]["ppo"]["entropy_coef"], .005)
        self.assertIn("different moments", " ".join(self.report["limits"]))
        self.assertIn("total_training_wall_time_and_real_interactions", self.report["comparison_protocol"]["primary_endpoint"])
        self.assertIn("target-not-reached", self.report["comparison_protocol"]["target_not_reached"])

    def test_cli_writes_new_json_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()) as output:
            path = Path(directory) / "parameters.json"
            self.assertEqual(COMPARISON.main(["--num-envs", "8", "--output", str(path)]), 0)
            report = json.loads(path.read_text())
            before = path.read_bytes()
            self.assertTrue(report["all_common_parameters_match"])
            self.assertEqual(json.loads(output.getvalue())["remaining_difference_count"], 13)
            differences = {row["parameter"]: row for row in report["remaining_differences"]}
            self.assertEqual(differences["recovery_height_objective"]["wmac"], "upright_v2")
            self.assertEqual(differences["recovery_height_objective"]["ppo"], "height_only_v1")
            self.assertEqual(differences["imagined_recovery_counter"]["wmac"], "hard_forward_v2")
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as rejected:
                COMPARISON.main(["--output", str(path)])
            self.assertEqual(rejected.exception.code, 2)
            self.assertEqual(path.read_bytes(), before)

    def test_invalid_environment_count_is_rejected(self):
        for value in (0, -2, 1.5, True):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "positive integer"):
                COMPARISON.build_comparison(value)


if __name__ == "__main__":
    unittest.main()
