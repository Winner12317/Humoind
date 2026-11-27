"""Regression tests for reference and independent simulated PM01 recovery starts."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from sim2sim import Controller, HERE, apply_key


class RecoveryStartTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.clip_dir = self.root / "motions"
        self.clip_dir.mkdir()
        self.enterContext(patch.dict(os.environ, PM01_MOTION_ROOT=str(self.clip_dir)))
        meta = json.loads((HERE / "policy_wamp_v3_70400/policy.json").read_text())
        self.policy = SimpleNamespace(directory=self.root, meta=meta)

    def test_missing_reference_only_affects_explicit_reference_recovery(self):
        controller = Controller(self.policy, "sdk", recovery_poses="reference")
        controller.reset("stand")
        self.assertNotIn("recovery_starts", controller.__dict__)
        with self.assertRaises(FileNotFoundError):
            controller.reset("fallen")

    def test_generated_falls_work_for_keyboard_and_both_models(self):
        for dynamics in ("sdk", "training"):
            with self.subTest(dynamics=dynamics):
                controller = Controller(self.policy, dynamics)
                self.assertEqual(controller.recovery_starts[0], "generic_simulated")
                self.assertEqual(controller.start_poses.shape, (8, 30))
                self.assertEqual(len(np.unique(controller.start_poses, axis=0)), 8)
                state = {}
                for _ in range(8):
                    self.assertIn("generic_simulated", apply_key("f", controller, state))
                    self.assertTrue(controller.status.fallen)
                    self.assertTrue(np.isfinite(controller.data.qpos).all())
                    self.assertTrue((controller.data.qvel == 0).all())
                self.assertEqual(state["pose"], 8)
                controller.reset("stand")
                self.assertTrue(controller.status.upright)

    def test_archive_precedence_and_exact_reference_frames(self):
        rng = np.random.default_rng(4)
        positions = rng.normal(size=(10, 1, 3)).astype("float32")
        quaternions = rng.normal(size=(10, 1, 4)).astype("float32")
        joints = rng.normal(size=(10, 23)).astype("float32")
        frames = [9, 0, 6, 2, 1, 8, 3, 5]
        clip = self.clip_dir / "test.npz"
        np.savez(clip, body_pos_w=positions, body_quat_w=quaternions, joint_pos=joints)
        manifest = {"clip": clip.name, "frames": frames, "clip_sha256": hashlib.sha256(clip.read_bytes()).hexdigest()}
        reference = self.root / "start_poses.reference.json"
        reference.write_text(json.dumps(manifest))
        expected = np.column_stack((positions[frames, 0], quaternions[frames, 0], joints[frames])).astype("float64")
        expected[:, :2] = 0

        reference_controller = Controller(self.policy, "sdk")
        self.assertEqual(reference_controller.recovery_starts[0], "reference_motion")
        np.testing.assert_array_equal(reference_controller.start_poses, expected)
        manifest["clip_sha256"] = "0" * 64
        reference.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "differs"):
            _ = Controller(self.policy, "sdk").start_poses

        archive_poses = np.ones((8, 30))
        np.savez(self.root / "start_poses.npz", qpos=archive_poses)
        np.testing.assert_array_equal(Controller(self.policy, "sdk").start_poses, archive_poses)
        self.assertEqual(Controller(self.policy, "sdk", recovery_poses="generic").recovery_starts[0],
                         "generic_simulated")


if __name__ == "__main__":
    unittest.main()
