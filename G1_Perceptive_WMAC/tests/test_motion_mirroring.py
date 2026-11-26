"""Real-model kinematic and isolated-dataset checks for G1 motion reflection."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import mujoco
import numpy as np

from src.wmac_perceptive.mirroring import (
    AXIAL_SIGN, DEFAULT_G1_XML, POLAR_SIGN, build_g1_mirror_map,
    mirror_motion, reflect_axial, reflect_polar, reflect_quaternion_wxyz,
    turn_statistics,
)
from src.wmac_perceptive.motions import (
    prepare_mirrored_dataset, read_motion, verify_mirrored_dataset,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/"src/assets/motions/g1/amp"


def rotations(quaternions):
    result = np.empty((*quaternions.shape[:-1], 3, 3))
    for out, quat in zip(result.reshape(-1, 3, 3), quaternions.reshape(-1, 4)):
        mujoco.mju_quat2Mat(out.reshape(-1), np.asarray(quat, dtype=np.float64))
    return result


def geometric_clip(model, joints, speeds, root_position, root_rotation, step=1e-5):
    """Two frames and link velocities derived from CPU MuJoCo FK, not mirror code."""
    poses, orientations = [], []
    data = mujoco.MjData(model)
    for delta in (0., step):
        data.qpos[:3] = root_position
        data.qpos[3:7] = root_rotation
        data.qpos[7:] = joints + delta*speeds
        mujoco.mj_forward(model, data)
        poses.append(data.xpos[1:].copy())
        orientations.append(data.xquat[1:].copy())
    positions, quaternions = np.stack(poses), np.stack(orientations)
    linear = (positions[1]-positions[0])/step
    r = rotations(quaternions)
    delta = r[1] @ r[0].swapaxes(-1, -2)
    angular = np.stack((delta[:, 2, 1]-delta[:, 1, 2],
                        delta[:, 0, 2]-delta[:, 2, 0],
                        delta[:, 1, 0]-delta[:, 0, 1]), axis=-1)/(2*step)
    return {"fps": np.asarray([1/step]), "joint_pos": np.stack((joints, joints+step*speeds)),
            "joint_vel": np.stack((speeds, speeds)), "body_pos_w": positions,
            "body_quat_w": quaternions, "body_lin_vel_w": np.stack((linear, linear)),
            "body_ang_vel_w": np.stack((angular, angular))}


class MotionMirroringTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mapping = build_g1_mirror_map()
        cls.model = mujoco.MjModel.from_xml_path(str(DEFAULT_G1_XML))

    def test_joint_axes_and_rest_geometry_determine_signs_not_just_names(self):
        mapping = self.mapping
        self.assertEqual(len(mapping.joint_names), 29)
        self.assertEqual(len(mapping.body_names), 30)
        self.assertLessEqual(mapping.maximum_rest_position_asymmetry, 2e-5)
        for i, name in enumerate(mapping.joint_names):
            expected = -1 if any(part in name for part in ("_roll_", "_yaw_")) else 1
            self.assertEqual(mapping.joint_sign[i], expected, name)
            if name.startswith("left_"):
                self.assertEqual(mapping.joint_names[mapping.joint_permutation[i]], "right_"+name[5:])
        np.testing.assert_array_equal(mapping.joint_permutation[mapping.joint_permutation], np.arange(29))
        np.testing.assert_array_equal(mapping.body_permutation[mapping.body_permutation], np.arange(30))

    def test_quaternion_is_mrm_and_angular_velocity_is_axial(self):
        generator = np.random.default_rng(124)
        q = generator.normal(size=(11, 4)); q /= np.linalg.norm(q, axis=-1, keepdims=True)
        reflection = np.diag(POLAR_SIGN)
        expected = reflection @ rotations(q) @ reflection
        actual = rotations(reflect_quaternion_wxyz(q))
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-14)
        a, b = generator.normal(size=(2, 11, 3))
        np.testing.assert_allclose(np.cross(reflect_polar(a), reflect_polar(b)),
                                   reflect_axial(np.cross(a, b)), rtol=1e-14, atol=1e-14)
        np.testing.assert_array_equal(reflect_axial(np.asarray([[1., 2., 3.]])), [[-1., 2., -3.]])

    def test_mirror_twice_retains_time_fps_quaternions_and_every_original_array(self):
        path = sorted((SOURCE/"WalkandRun").glob("*.npz"))[0]
        original = read_motion(path)
        before = {key: value.copy() for key, value in original.items()}
        mirrored = mirror_motion(original, self.mapping)
        restored = mirror_motion(mirrored, self.mapping)
        for name, value in original.items():
            np.testing.assert_array_equal(value, before[name])
            self.assertEqual(value.shape, mirrored[name].shape)
            self.assertEqual(value.dtype, mirrored[name].dtype)
            np.testing.assert_allclose(restored[name], value, rtol=0, atol=3e-7)
        np.testing.assert_array_equal(mirrored["fps"], original["fps"])
        # Reflection of the path is about its own first-frame pelvis y, not y=0.
        self.assertEqual(mirrored["body_pos_w"][0, 0, 1], original["body_pos_w"][0, 0, 1])
        np.testing.assert_array_equal(mirrored["body_ang_vel_w"][:, 0, 2], -original["body_ang_vel_w"][:, 0, 2])

    def test_reflected_joint_pose_and_velocity_match_independent_mujoco_kinematics(self):
        rng = np.random.default_rng(93)
        reflection = np.diag(POLAR_SIGN)
        mapping = self.mapping
        foot_ids = [mapping.body_names.index(name) for name in ("left_ankle_roll_link", "right_ankle_roll_link")]
        for _ in range(7):
            joints, speeds = rng.normal(0., .2, (2, 29))
            root_q = rng.normal(size=4); root_q /= np.linalg.norm(root_q)
            original = geometric_clip(self.model, joints, speeds, np.asarray([.2, .3, .9]), root_q)
            mirrored = mirror_motion(original, mapping)
            independent = geometric_clip(self.model, mirrored["joint_pos"][0], mirrored["joint_vel"][0],
                                         mirrored["body_pos_w"][0, 0], mirrored["body_quat_w"][0, 0])
            # The exact published XML has one 10um shoulder offset mismatch.
            np.testing.assert_allclose(mirrored["body_pos_w"], independent["body_pos_w"], rtol=0, atol=2e-5)
            np.testing.assert_allclose(rotations(mirrored["body_quat_w"]), rotations(independent["body_quat_w"]),
                                       rtol=0, atol=1e-12)
            np.testing.assert_allclose(mirrored["body_pos_w"][:, foot_ids, 2],
                                       independent["body_pos_w"][:, foot_ids, 2], rtol=0, atol=1e-12)
            np.testing.assert_allclose(mirrored["body_lin_vel_w"], independent["body_lin_vel_w"], rtol=0, atol=2e-5)
            np.testing.assert_allclose(mirrored["body_ang_vel_w"], independent["body_ang_vel_w"], rtol=0, atol=2e-8)

    def test_actual_walking_distribution_becomes_balanced_without_recovery_changes(self):
        originals = [read_motion(p) for p in sorted((SOURCE/"WalkandRun").glob("*.npz"))]
        pairs = [motion for original in originals for motion in (original, mirror_motion(original, self.mapping))]
        before, after = turn_statistics(originals), turn_statistics(pairs)
        self.assertEqual(before["clips"], 17)
        self.assertEqual(after["clips"], 34)
        self.assertEqual(after["frames"], 2*before["frames"])
        self.assertGreater(after["positive_yaw_frames"], 0)
        self.assertEqual(after["positive_yaw_frames"], after["negative_yaw_frames"])
        self.assertAlmostEqual(after["positive_yaw_seconds"], after["negative_yaw_seconds"], places=10)
        self.assertAlmostEqual(after["yaw_angular_velocity_integral_rad"], 0., places=12)

    def fixture_source(self, root):
        for group in ("WalkandRun", "Recovery"):
            (root/group).mkdir(parents=True)
            values = read_motion(sorted((SOURCE/group).glob("*.npz"))[0])
            short = {name: value if name == "fps" else value[:8] for name, value in values.items()}
            np.savez_compressed(root/group/"fixture.npz", **short)

    def test_preparation_preserves_source_and_recovery_and_reuses_only_verified_cache(self):
        with tempfile.TemporaryDirectory() as temp:
            source, output = Path(temp)/"source", Path(temp)/"output"
            self.fixture_source(source)
            original_hashes = {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in source.rglob("*.npz")}
            paths = prepare_mirrored_dataset(source, output)
            manifest = verify_mirrored_dataset(output, source_root=source)
            self.assertEqual(manifest["walking_total_clips"], 2)
            self.assertEqual(manifest["recovery_original_clips"], 1)
            self.assertTrue(manifest["recovery_copied_byte_for_byte"])
            self.assertEqual(len(list(paths.walk.glob("*.npz"))), 2)
            self.assertEqual((paths.recovery/"fixture.npz").read_bytes(), (source/"Recovery/fixture.npz").read_bytes())
            for name, digest in original_hashes.items():
                self.assertEqual(hashlib.sha256((source/name).read_bytes()).hexdigest(), digest)
            manifest_bytes = paths.manifest.read_bytes()
            self.assertEqual(prepare_mirrored_dataset(source, output), paths)
            self.assertEqual(paths.manifest.read_bytes(), manifest_bytes)
            with (paths.walk/"fixture__mirror_y.npz").open("ab") as stream:
                stream.write(b"modified")
            with self.assertRaisesRegex(ValueError, "modified"):
                prepare_mirrored_dataset(source, output)

    def test_double_mirroring_and_nested_output_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp)/"source"
            self.fixture_source(source)
            with self.assertRaisesRegex(ValueError, "separate"):
                prepare_mirrored_dataset(source, source/"nested")
            (source/"WalkandRun/fixture.npz").rename(source/"WalkandRun/fixture__mirror_y.npz")
            with self.assertRaisesRegex(ValueError, "twice"):
                prepare_mirrored_dataset(source, Path(temp)/"output")


if __name__ == "__main__":
    unittest.main()
