"""CPU-only phase sampling and window-boundary regressions."""
from __future__ import annotations

import importlib.util
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.wmac.config import WmacConfig


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/probe_wmac_recorded_phases.py"
SPEC = importlib.util.spec_from_file_location("wmac_recorded_phase_probe", SCRIPT)
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)


def recording(steps=10, environments=2):
    values = {key: np.zeros((steps, environments) if width is None else (steps, environments, width),
                           dtype=bool if key in PROBE.BOOL_FIELDS else np.float32)
              for key, width in PROBE.WIDTHS.items()}
    values["first_episode_valid"].fill(True)
    values["state"][..., 2] = .8
    values["state"][..., 3] = 1.
    values["next_state"][:] = values["state"]
    values["context"][..., 34] = np.arange(steps)[:, None]
    values["next_context"][..., 34] = np.arange(1, steps + 1)[:, None]
    return values


class RecordedPhaseSamplingTest(unittest.TestCase):
    def test_cross_checkpoint_phase_probes_reject_different_recovery_rules(self):
        source = WmacConfig()
        for key, value in (("recovery_counter_semantics", "soft_v1"),
                           ("recovery_height_reward", "height_only_v1")):
            with self.subTest(key=key):
                bundles = {22000: (source,), 24000: (replace(source, **{key: value}),)}
                with self.assertRaisesRegex(ValueError, "incompatible semantics"):
                    PROBE.validate_bundles(bundles)

    def test_complete_windows_do_not_cross_push_reset_or_invalid_padding(self):
        values = recording(steps=10, environments=2)
        values["boundary"][3, 0] = True  # A push; later windows may start anew.
        values["terminated"][5, 1] = values["boundary"][5, 1] = True
        values["first_episode_valid"][6:, 1] = False
        candidates = PROBE.eligible_starts(values, "upright", length=4, max_start_step=7)
        self.assertIn([0, 0], candidates.tolist())  # Boundary on the final step is valid.
        self.assertNotIn([1, 0], candidates.tolist())
        self.assertIn([4, 0], candidates.tolist())
        self.assertIn([2, 1], candidates.tolist())  # True terminal at final step.
        self.assertNotIn([3, 1], candidates.tolist())
        self.assertFalse(any(env == 1 and time >= 6 for time, env in candidates))
        values["actor_history"][4, 0, 0] = np.nan
        after = PROBE.eligible_starts(values, "upright", length=4, max_start_step=7)
        self.assertNotIn([4, 0], after.tolist())

    def test_start_phase_uses_height_and_tilt_and_respects_early_guard(self):
        values = recording()
        values["state"][1, 0, 2] = .5
        values["state"][2, 1, 3:7] = [0., 1., 0., 0.]
        fallen = PROBE.eligible_starts(values, "nonupright", length=3, max_start_step=3)
        self.assertEqual(fallen.tolist(), [[1, 0], [2, 1]])
        upright = PROBE.eligible_starts(values, "upright", length=3, max_start_step=3)
        self.assertTrue(all(time < 3 for time, _ in upright))
        self.assertNotIn([1, 0], upright.tolist())
        values["next_context"][2, 0, 34] = 250.
        initial = PROBE.eligible_starts(values, "initial", length=3, max_start_step=3)
        self.assertEqual(initial.tolist(), [[0, 1]])

    def test_scarce_initial_starts_use_explicit_deterministic_replacement(self):
        values = recording()
        before = {key: value.copy() for key, value in values.items()}
        windows, first = PROBE.sample_phase(values, "initial", batch_size=8, length=4, seed=17)
        np.random.seed(991)
        again, second = PROBE.sample_phase(values, "initial", batch_size=8, length=4, seed=17)
        self.assertEqual(first, second)
        self.assertEqual(first["candidate_count"], 2)
        self.assertTrue(first["sample_with_replacement"])
        self.assertTrue(all(time == 0 for time, _ in first["indices_time_env"]))
        self.assertEqual(windows["state"].shape, (8, 4, 71))
        self.assertEqual(PROBE.array_sha256(windows), PROBE.array_sha256(again))
        for key in values:
            np.testing.assert_array_equal(values[key], before[key])
        skipped, metadata = PROBE.sample_phase(values, "initial", batch_size=8,
                                               length=4, scarce_policy="skip")
        self.assertIsNone(skipped)
        self.assertEqual(metadata["status"], "skipped")
        empty, metadata = PROBE.sample_phase(values, "nonupright", length=4)
        self.assertIsNone(empty)
        self.assertEqual(metadata["candidate_count"], 0)

    def test_recording_validation_rejects_reset_leak_and_complex_values(self):
        values = recording()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.npz"
            np.savez_compressed(path, **values)
            loaded, metadata = PROBE.read_recording(path)
            self.assertEqual(metadata["environments"], 2)
            self.assertEqual(loaded["state"].dtype, np.float32)
            values["terminated"][2, 0] = values["boundary"][2, 0] = True
            np.savez_compressed(path, **values)
            with self.assertRaisesRegex(ValueError, "exclude every reset"):
                PROBE.read_recording(path)
            values["first_episode_valid"][3:, 0] = False
            values["reward"] = values["reward"].astype(np.complex64)
            np.savez_compressed(path, **values)
            with self.assertRaisesRegex(ValueError, "real floating"):
                PROBE.read_recording(path)


if __name__ == "__main__":
    unittest.main()
