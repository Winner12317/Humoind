"""Pure NumPy checks for finite-scale response classification above noise."""
from __future__ import annotations

import importlib.util
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/probe_wmac_physical_directions.py"
SPEC = importlib.util.spec_from_file_location("wmac_physical_direction_stats", SCRIPT)
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)


class FiniteScaleStatsTest(unittest.TestCase):
    def test_large_response_resolves_and_sign_disagreement_is_visible(self):
        baseline = np.zeros((5, 2))
        plus = np.array([[.01, -.01], [.01, -.01]])
        minus = -plus
        rows = PROBE.finite_scale_response(baseline, plus, minus, np.ones(2), .01)
        self.assertTrue(all(row["resolved"] for row in rows))
        self.assertTrue(rows[0]["sign_agrees"])
        self.assertFalse(rows[1]["sign_agrees"])
        summary = PROBE.summarize_responses(rows)
        self.assertEqual(summary["sign_agreement_fraction_resolved"], .5)

    def test_noise_dominated_signal_is_unresolved_not_counted_as_agreement(self):
        baseline = np.array([[-.001], [.001], [-.001], [.001], [0.]])
        plus = np.array([[.0001], [.0001]])
        minus = -plus
        row = PROBE.finite_scale_response(baseline, plus, minus, np.ones(1), .01)[0]
        self.assertFalse(row["resolved"])
        self.assertIsNone(row["sign_agrees"])
        self.assertGreater(row["required_return_difference"], abs(row["physical_return_difference"]))
        summary = PROBE.summarize_responses([row])
        self.assertEqual(summary["resolved"], 0)
        self.assertIsNone(summary["sign_agreement_fraction_resolved"])

    def test_perturbation_noise_can_invalidate_an_apparently_quiet_baseline(self):
        baseline = np.zeros((5, 1))
        plus = np.array([[.1], [-.09]])
        minus = np.array([[0.], [0.]])
        row = PROBE.finite_scale_response(baseline, plus, minus, np.ones(1), .03)[0]
        self.assertFalse(row["resolved"])
        self.assertGreaterEqual(row["noise_reference_amplitude"], .19)

    def test_absolute_floor_and_zero_gradient_keep_tiny_response_unresolved(self):
        baseline = np.zeros((5, 2))
        plus, minus = np.full((2, 2), 1e-8), np.full((2, 2), -1e-8)
        rows = PROBE.finite_scale_response(baseline, plus, minus, np.array([1., 0.]), .1)
        self.assertTrue(all(not row["resolved"] for row in rows))
        self.assertIn("negligible gradient", rows[1]["reason"])

    def test_nonfinite_measurements_remain_explicit_and_json_safe(self):
        baseline = np.zeros((5, 2))
        baseline[0, 0] = np.nan
        plus = np.ones((2, 2))
        minus = np.zeros((2, 2))
        rows = PROBE.finite_scale_response(baseline, plus, minus, np.array([1., np.inf]), .01)
        self.assertTrue(all(not row["resolved"] for row in rows))
        self.assertTrue(all("nonfinite" in row["reason"] for row in rows))
        serialized = json.dumps(PROBE.clean(rows), allow_nan=False)
        self.assertIn("null", serialized)
        stats = PROBE.noise_summary(baseline)
        self.assertEqual(stats["nonfinite_count"].tolist(), [1, 0])

    def test_invalid_shapes_and_thresholds_are_rejected(self):
        with self.assertRaises(ValueError):
            PROBE.noise_summary(np.zeros((1, 2)))
        with self.assertRaises(ValueError):
            PROBE.finite_scale_response(np.zeros((5, 2)), np.zeros((2, 3)),
                                        np.zeros((2, 2)), np.ones(2), .01)
        with self.assertRaises(ValueError):
            PROBE.finite_scale_response(np.zeros((5, 2)), np.zeros((2, 2)),
                                        np.zeros((2, 2)), np.ones(2), 0.)
        with self.assertRaises(ValueError):
            PROBE.finite_scale_response(np.zeros((5, 2)), np.zeros((2, 2)),
                                        np.zeros((2, 2)), np.ones(2), .01, noise_multiplier=.5)


class PreludeCliTest(unittest.TestCase):
    def test_default_and_boundary_preludes_are_in_the_physical_step_budget(self):
        # At H=32 with one epsilon, nominal generation + five baselines +
        # twelve component/sign repeats require 576 physical control steps.
        # A prelude happens once and must be added outside those repetitions.
        for prelude, expected_steps in ((None, 576), (32, 608), (128, 704)):
            with self.subTest(prelude=prelude), tempfile.TemporaryDirectory() as root:
                checkpoint = Path(root) / "input.pt"
                checkpoint.write_bytes(b"CPU test checkpoint; probe is stubbed")
                output = Path(root) / "output"
                arguments = ["--checkpoint", str(checkpoint), "--output", str(output),
                             "--horizon", "32", "--epsilons", ".03"]
                if prelude is not None:
                    arguments += ["--prelude-steps", str(prelude)]
                def no_physics(args, directory, report):
                    self.assertEqual(args.prelude_steps, prelude or 0)
                    report["status"] = "completed"
                with mock.patch.object(PROBE, "probe", side_effect=no_physics) as probe:
                    with mock.patch.dict(PROBE.os.environ), redirect_stdout(io.StringIO()):
                        PROBE.main(arguments)
                probe.assert_called_once()
                report = json.loads((output / "report.json").read_text())
                self.assertEqual(report["protocol"]["maximum_environment_step_calls"], expected_steps)
                self.assertEqual(report["protocol"]["prelude_steps"], prelude or 0)
                self.assertEqual(checkpoint.read_bytes(), b"CPU test checkpoint; probe is stubbed")

    def test_invalid_prelude_is_rejected_before_creating_output_or_running_physics(self):
        for prelude in (-1, 129):
            with self.subTest(prelude=prelude), tempfile.TemporaryDirectory() as root:
                checkpoint = Path(root) / "input.pt"
                checkpoint.write_bytes(b"untouched")
                output = Path(root) / "output"
                with mock.patch.object(PROBE, "probe") as probe, redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as failure:
                        PROBE.main(["--checkpoint", str(checkpoint), "--output", str(output),
                                    "--prelude-steps", str(prelude)])
                self.assertEqual(failure.exception.code, 2)
                probe.assert_not_called()
                self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
