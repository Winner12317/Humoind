"""Training review distinguishes actual updates, comparable outcomes and convergence."""
from __future__ import annotations

import contextlib
import copy
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from scripts.check_wmac_perceptive_training import (
    compare_evaluations, main, parse_args, training_summary,
)


def evaluation():
    return {"status": "completed", "protocol": {"seed": 2032, "num_envs": 16, "steps": 500},
            "results": {
                "walk": {"initial_state_sha256": "same-walk", "first_episode_survival_fraction": .25},
                "recovery": {"initial_state_sha256": "same-recovery", "first_episode_survival_fraction": .125,
                             "recovery_success_fraction": .125, "success_then_walk_fraction_all_starts": 0.}}}


def checkpoint(path):
    torch.save({"format": "amp-wmac-perceptive-v1", "iteration": 3,
                "config": {"perception_training_version": 2, "actor_updates": 1,
                           "actor_stall_warning_iterations": 200, "imagination_horizon": 8},
                "perception_training_progress": {"actor_updates_accepted": 12, "actor_updates_rejected": 6,
                                                  "_last_actor_update_iteration": 3, "_progress_start_iteration": 0},
                "opt_actor": {"state": {0: {"step": torch.tensor(20.)}}}}, path)


class PerceptiveTrainingReviewTest(unittest.TestCase):
    def test_cli_requires_existing_checkpoint_and_valid_counts(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stderr(io.StringIO()):
            path = Path(directory)/"model.pt"; checkpoint(path)
            for argv in ([], ["--checkpoint", str(path)+".missing"],
                         ["--checkpoint", str(path), "--num-envs", "-1"],
                         ["--checkpoint", str(path), "--steps", "0"],
                         ["--checkpoint", str(path), "--window", "0"],
                         ["--checkpoint", str(path), "--seed", "-1"],
                         ["--checkpoint", str(path), "--baseline-checkpoint", str(path)+".missing"]):
                with self.subTest(argv=argv), self.assertRaises(SystemExit): parse_args(argv)
            args = parse_args(["--checkpoint", str(path), "--baseline-checkpoint", str(path)])
            self.assertEqual(args.checkpoint, path.resolve())
            self.assertEqual(args.baseline_checkpoint, path.resolve())

    def test_csv_counts_ignore_future_rows_and_optimizer_history_can_predate_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"model_3.pt"; checkpoint(path)
            fields = ("iteration", "actor_active", "actor_update_rejected", "actor_updates_accepted",
                      "actor_updates_rejected", "active_imagination_horizon", "actor_update_stall_iterations", "gate_reason")
            with (path.parent/"metrics.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
                for values in ((1, 1, 0, 11, 5, 1, 0, "ready"), (2, 1, 1, 11, 6, 1, 1, "ready"),
                               (3, 1, 0, 12, 6, 2, 0, "ready"), (4, 1, 0, 13, 6, 2, 0, "ready")):
                    writer.writerow(dict(zip(fields, values)))
            result = training_summary(path, window=2)
            self.assertEqual(result["csv_rows_to_checkpoint"], 3)
            self.assertEqual(result["csv_observed_updates_accepted"], 2)
            self.assertEqual(result["csv_observed_updates_rejected"], 1)
            self.assertEqual(result["recent_updates_accepted"], 1)
            self.assertEqual(result["protocol_updates_accepted"], 12)
            self.assertEqual(result["actor_optimizer_steps"], [20.])
            self.assertEqual(result["horizon_distribution"], {"1": 2, "2": 1})
            self.assertEqual(result["status"], "updating_recently")

    def test_recorded_stall_does_not_claim_updates_without_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"model.pt"; checkpoint(path)
            result = training_summary(path)
            self.assertEqual(result["status"], "insufficient_metrics")
            self.assertEqual(result["recent_updates_accepted"], 0)
            payload = torch.load(path, weights_only=False)
            payload["iteration"] = 1000
            torch.save(payload, path)
            self.assertEqual(training_summary(path)["status"], "stalled")

    def test_comparison_reports_primary_improvement_regression_and_mixed_outcomes(self):
        baseline = evaluation()
        result = compare_evaluations(evaluation(), baseline)
        self.assertEqual(result["verdict"], "no_observed_primary_improvement")
        better = evaluation(); better["results"]["recovery"]["recovery_success_fraction"] = .5
        self.assertEqual(compare_evaluations(better, baseline)["verdict"], "observed_improvement")
        worse = evaluation(); worse["results"]["walk"]["first_episode_survival_fraction"] = 0.
        self.assertEqual(compare_evaluations(worse, baseline)["verdict"], "observed_regression")
        mixed = copy.deepcopy(better); mixed["results"]["walk"]["first_episode_survival_fraction"] = 0.
        self.assertEqual(compare_evaluations(mixed, baseline)["verdict"], "mixed_outcomes")
        self.assertEqual(baseline, evaluation())

    def test_different_seed_or_initial_states_and_failed_evaluations_are_not_comparable(self):
        baseline = evaluation()
        for change in ("seed", "initial", "missing", "failed"):
            candidate = evaluation()
            if change == "seed": candidate["protocol"]["seed"] += 1
            elif change == "initial": candidate["results"]["walk"]["initial_state_sha256"] = "different"
            elif change == "missing": candidate["results"]["recovery"].pop("initial_state_sha256")
            else: candidate["status"] = "failed"
            with self.subTest(change=change):
                self.assertFalse(compare_evaluations(candidate, baseline)["comparable"])

    def test_main_writes_a_new_review_and_calls_both_evaluations_without_claiming_convergence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); candidate = root/"model.pt"; checkpoint(candidate)
            output = root/"review"
            with patch("scripts.check_wmac_perceptive_training._evaluate", return_value=evaluation()) as evaluate:
                with contextlib.redirect_stdout(io.StringIO()):
                    code = main(["--checkpoint", str(candidate), "--baseline-checkpoint", str(candidate),
                                 "--output", str(output)])
            self.assertEqual(code, 0)
            self.assertEqual(evaluate.call_count, 2)
            result = json.loads((output/"review.json").read_text())
            self.assertEqual(result["status"], "completed")
            self.assertFalse(result["convergence_established"])
            with self.assertRaises(FileExistsError):
                main(["--checkpoint", str(candidate), "--output", str(output)])


if __name__ == "__main__":
    unittest.main()
