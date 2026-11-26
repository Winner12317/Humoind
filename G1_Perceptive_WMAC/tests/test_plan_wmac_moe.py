"""CPU-only checks for real parameter capacity and executable plan contracts."""
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest

import torch

from scripts.plan_wmac_moe import ACTOR_FIELDS, build_plan, write_plan
from scripts.train_wmac import parse_config


class MoELongPlanTest(unittest.TestCase):
    def test_actual_capacity_budget_commands_and_cpu_rng_are_consistent(self):
        before = torch.get_rng_state().clone()
        cuda_initialized = torch.cuda.is_initialized()
        plan = build_plan()
        torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)
        self.assertEqual(torch.cuda.is_initialized(), cuda_initialized)
        self.assertEqual((plan["arm_count"], plan["run_count"]), (6, 18))
        arms = {item["name"]: item for item in plan["arms"]}
        self.assertEqual(arms["mlp"]["parameters"]["total_parameters"], 365085)
        self.assertEqual(arms["moe_top2"]["parameters"]["total_parameters"], 1559928)
        self.assertEqual(arms["moe_top2"]["parameters"]["active_parameters_per_sample"], 829758)
        for label, widths, count in (("mlp_total_matched", [1024, 1008, 128], 1560333),
                                     ("mlp_active_matched", [768, 591, 128], 829676)):
            self.assertEqual(arms[label]["actor_architecture"]["policy_hidden"], widths)
            self.assertEqual(arms[label]["parameters"]["total_parameters"], count)
            self.assertFalse(arms[label]["parameter_match"]["exact_match"])
            self.assertLess(arms[label]["parameter_match"]["absolute_relative_error"], .0003)
        self.assertEqual(plan["training_budget_per_run"]["physical_training_transitions"], 614662144)
        self.assertEqual(len({run["run_name"] for run in plan["runs"]}), 18)
        for run in plan["runs"]:
            restored = json.loads(json.dumps(asdict(parse_config(run["argv"][2:]))))
            self.assertEqual(restored, run["config"])
            self.assertEqual(restored["recovery_counter_semantics"], "hard_forward_v2")
            self.assertEqual(restored["recovery_height_reward"], "upright_v2")
            common = {key: value for key, value in restored.items()
                      if key not in (*ACTOR_FIELDS, "seed", "run_name")}
            self.assertEqual(common, plan["common_training_config"])

    def test_plan_has_no_overwrite_or_duplicate_seed_aliases(self):
        with self.assertRaisesRegex(ValueError, "distinct"):
            build_plan(seeds=[42, 42])
        with self.assertRaisesRegex(ValueError, "positive"):
            build_plan(iterations=0)
        plan = build_plan(iterations=3, seeds=[7], num_envs=4)
        self.assertEqual(plan["training_budget_per_run"]["physical_training_steps_per_env"], 71)
        self.assertEqual(plan["run_count"], 6)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/"plan.json"
            write_plan(plan, path)
            original = path.read_bytes()
            with self.assertRaises(FileExistsError):
                write_plan(plan, path)
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(json.loads(original)["status"], "planned_only_not_started")


if __name__ == "__main__":
    unittest.main()
