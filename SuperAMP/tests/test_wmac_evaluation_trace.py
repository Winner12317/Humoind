"""CPU checks for optional physical evaluation recordings."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_wmac.py"
SPEC = importlib.util.spec_from_file_location("wmac_evaluation_cli", SCRIPT)
CLI = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CLI)


def physical_transition(num_envs=2):
    widths = {
        "state": 71, "next_state": 71, "action": 29, "previous_action": 29,
        "context": 35, "next_context": 35, "command": 3,
        "actor_history": 384, "contact_cost": 2,
    }
    result = {key: torch.zeros(num_envs, width) for key, width in widths.items()}
    result["reward"] = torch.zeros(num_envs)
    for key in ("terminated", "timeout", "boundary"):
        result[key] = torch.zeros(num_envs, dtype=torch.bool)
    return result


def physical_trace(num_envs=2):
    return {key: torch.zeros(num_envs, dtype=torch.bool if key in (
        "recovered", "terminated", "timeout") else torch.float32)
        for key in CLI.EvaluationRecorder.TRACE_FIELDS}


class EvaluationRecordingTest(unittest.TestCase):
    def test_snapshots_preserve_terminal_successor_and_exclude_reset_episodes(self):
        recorder = CLI.EvaluationRecorder(2, 3, save_transitions=True, save_trace=True)
        transition, trace = physical_transition(), physical_trace()
        active = torch.tensor([True, True])
        transition["state"][:, 2] = torch.tensor([0.2, 0.3])
        transition["next_state"][:, 2] = torch.tensor([0.1, 0.7])
        transition["terminated"][0] = True
        transition["boundary"][0] = True
        trace["root_height"].copy_(transition["next_state"][:, 2])
        recorder.append(transition, active, trace)

        # Mimic buffers overwritten by an automatic reset and the next step.
        active[0] = False
        transition["state"][:, 2] = torch.tensor([100., 0.7])
        transition["next_state"][:, 2] = torch.tensor([101., 0.8])
        transition["terminated"].zero_()
        transition["boundary"].zero_()
        trace["root_height"].copy_(transition["next_state"][:, 2])
        recorder.append(transition, active, trace)
        transition["next_state"].fill_(999.)
        trace["root_height"].fill_(999.)
        active.zero_()

        with tempfile.TemporaryDirectory() as directory:
            summary = recorder.finish(directory, "recovery", .02)
            with np.load(Path(directory) / "transitions_recovery.npz", allow_pickle=False) as data:
                np.testing.assert_array_equal(data["first_episode_valid"], [[True, True], [False, True]])
                np.testing.assert_allclose(data["next_state"][:, :, 2], [[.1, .7], [101., .8]])
                self.assertTrue(data["terminated"][0, 0])
                self.assertTrue(data["boundary"][0, 0])
                # Post-reset values are available for audit, but never valid
                # for the first-episode diagnostic sample or window selection.
                np.testing.assert_allclose(data["next_state"][:, :, 2][data["first_episode_valid"]], [.1, .7, .8])
                self.assertEqual(data["actor_history"].shape, (2, 2, 384))
            with np.load(Path(directory) / "trace_recovery.npz", allow_pickle=False) as data:
                np.testing.assert_allclose(data["root_height"], [[.1, .7], [101., .8]])
                np.testing.assert_allclose(data["control_time_seconds"], [.02, .04])
            self.assertEqual(summary["transitions_recording"]["first_episode_transitions"], 3)
            self.assertEqual(summary["trace_recording"]["steps_recorded"], 2)

    def test_protocol_and_content_hashes_are_reproducible_without_pickle(self):
        protocol = {"seed": 2026, "initial_state_sha256": "test-initial-state"}
        results = []
        with tempfile.TemporaryDirectory() as directory:
            for index in range(2):
                output = Path(directory) / str(index)
                output.mkdir()
                recorder = CLI.EvaluationRecorder(2, 1, save_transitions=True, protocol=protocol)
                recorder.append(physical_transition(), torch.ones(2, dtype=torch.bool))
                results.append(recorder.finish(output, "walk", .02)["transitions_recording"])
                with np.load(output / "transitions_walk.npz", allow_pickle=False) as data:
                    saved = json.loads(str(data["recording_protocol_json"].item()))
                    self.assertEqual(saved["seed"], 2026)
                    self.assertFalse(saved["restart_snapshot"])
                    self.assertEqual(saved["successor"], "physical next state before automatic reset")
                    self.assertTrue(all(not data[key].dtype.hasobject for key in data.files))
                self.assertFalse((output / "transitions_walk.npz.tmp").exists())
            self.assertEqual(results[0]["protocol_sha256"], results[1]["protocol_sha256"])
            self.assertEqual(results[0]["content_sha256"], results[1]["content_sha256"])

    def test_invalid_step_is_rejected_without_partially_appending(self):
        recorder = CLI.EvaluationRecorder(2, 2, save_transitions=True)
        invalid = physical_transition()
        invalid["contact_cost"] = torch.zeros(3, 2)
        with self.assertRaisesRegex(ValueError, "contact_cost"):
            recorder.append(invalid, torch.ones(2, dtype=torch.bool))
        self.assertEqual(len(recorder.valid), 0)
        self.assertEqual(len(recorder.transitions), 0)
        with self.assertRaisesRegex(ValueError, "boolean"):
            recorder.append(physical_transition(), torch.ones(2))
        with self.assertRaisesRegex(ValueError, "empty"):
            recorder.finish("unused", "walk", .02)

    def test_recording_is_bounded_by_requested_steps(self):
        recorder = CLI.EvaluationRecorder(2, 1, save_trace=True)
        recorder.append(None, torch.ones(2, dtype=torch.bool), physical_trace())
        with self.assertRaisesRegex(ValueError, "bound"):
            recorder.append(None, torch.ones(2, dtype=torch.bool), physical_trace())
        self.assertEqual(len(recorder.valid), 1)
        self.assertEqual(len(recorder.traces), 1)

    def test_failed_atomic_write_keeps_existing_artifact_and_cleans_temporary_file(self):
        recorder = CLI.EvaluationRecorder(2, 1, save_trace=True)
        recorder.append(None, torch.ones(2, dtype=torch.bool), physical_trace())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace_walk.npz"
            path.write_bytes(b"previous artifact")

            def failed_write(handle, **arrays):
                handle.write(b"incomplete new archive")
                raise OSError("controlled write failure")

            with patch("numpy.savez_compressed", side_effect=failed_write):
                with self.assertRaisesRegex(OSError, "controlled"):
                    recorder.finish(directory, "walk", .02)
            self.assertEqual(path.read_bytes(), b"previous artifact")
            self.assertFalse(path.with_suffix(".npz.tmp").exists())


if __name__ == "__main__":
    unittest.main()
