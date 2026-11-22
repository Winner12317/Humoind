"""The PM01 reference motion set and its left/right mirroring (CPU only)."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import mujoco
import numpy as np

from src.assets.robots.engineai_pm01 import pm01_constants as pm01
from src.robot_profiles import G1, PM01
from src.wmac.mirroring import AXIAL_SIGN, build_g1_mirror_map, build_mirror_map, mirror_motion
from src.wmac.motions import MIRROR_SUFFIX, prepare_mirrored_dataset, read_motion, verify_mirrored_dataset

ROOT = Path(__file__).resolve().parents[1]
PM01_MOTIONS = ROOT / "src/assets/motions/pm01/amp"
G1_MOTIONS = ROOT / "src/assets/motions/g1/amp"
GROUPS = ("WalkandRun", "Recovery")


def clips(root: Path) -> dict[str, Path]:
    return {f"{group}/{path.name}": path for group in GROUPS for path in sorted((root / group).glob("*.npz"))}


class Pm01MotionSetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pm01, cls.g1 = clips(PM01_MOTIONS), clips(G1_MOTIONS)
        cls.model = mujoco.MjModel.from_xml_path(str(pm01.PM01_XML))
        cls.data = mujoco.MjData(cls.model)

    def test_every_g1_clip_has_a_pm01_clip_with_the_same_frames(self):
        self.assertEqual(set(self.pm01), set(self.g1))
        self.assertEqual(len(self.pm01), 18)
        total = 0
        for name, path in self.pm01.items():
            ours, theirs = read_motion(path), read_motion(self.g1[name])
            self.assertEqual(set(ours), set(theirs), name)
            frames = len(theirs["joint_pos"])
            total += frames
            self.assertEqual(float(np.asarray(ours["fps"]).reshape(-1)[0]), 50., name)
            self.assertEqual(ours["joint_pos"].shape, (frames, PM01.num_joints), name)
            self.assertEqual(ours["joint_vel"].shape, (frames, PM01.num_joints), name)
            for key, width in (("body_pos_w", 3), ("body_quat_w", 4), ("body_lin_vel_w", 3), ("body_ang_vel_w", 3)):
                self.assertEqual(ours[key].shape, (frames, PM01.num_bodies, width), (name, key))
                self.assertEqual(ours[key].dtype, theirs[key].dtype, (name, key))
                self.assertTrue(np.isfinite(ours[key]).all(), (name, key))
        self.assertEqual(total, 9992)

    def test_body_arrays_are_the_forward_kinematics_of_the_stored_joints(self):
        model, data = self.model, self.data
        for name, path in self.pm01.items():
            motion = read_motion(path)
            for frame in np.linspace(0, len(motion["joint_pos"]) - 1, 12).astype(int):
                data.qpos[:3] = motion["body_pos_w"][frame, 0]
                data.qpos[3:7] = motion["body_quat_w"][frame, 0]
                data.qpos[7:] = motion["joint_pos"][frame]
                mujoco.mj_kinematics(model, data)
                np.testing.assert_allclose(motion["body_pos_w"][frame], data.xpos[1:], atol=2e-6, err_msg=name)
                alignment = np.abs(np.sum(motion["body_quat_w"][frame] * data.xquat[1:], axis=-1))
                self.assertLess(float(np.max(1 - alignment)), 1e-6, name)

    def test_velocities_are_central_differences_of_the_stored_poses(self):
        motion = read_motion(self.pm01["WalkandRun/walk_forward_loop_002__A022.npz"])
        expected = (motion["joint_pos"][2:] - motion["joint_pos"][:-2]) * 25.
        np.testing.assert_allclose(motion["joint_vel"][1:-1], expected, atol=2e-3)
        root = (motion["body_pos_w"][2:, 0] - motion["body_pos_w"][:-2, 0]) * 25.
        np.testing.assert_allclose(motion["body_lin_vel_w"][1:-1, 0], root, atol=2e-3)

    def test_joints_stay_inside_the_model_limits_and_speeds_inside_the_actuators(self):
        low, high = self.model.jnt_range[1:, 0], self.model.jnt_range[1:, 1]
        speed_limit = np.asarray([26.3 if any(token in name for token in ("HIP_PITCH", "HIP_ROLL", "KNEE_PITCH"))
                                  else 35.2 for name in pm01.PM01_JOINT_NAMES])
        outside_soft = frames = 0
        mid, half = (low + high) / 2, (high - low) / 2 * 0.9
        for name, path in self.pm01.items():
            motion = read_motion(path)
            q = motion["joint_pos"]
            self.assertTrue(((q >= low - 1e-6) & (q <= high + 1e-6)).all(), name)
            self.assertLess(float((np.abs(motion["joint_vel"]) / speed_limit).max()), 0.8, name)
            outside_soft += int(((q < mid - half) | (q > mid + half)).any(axis=1).sum())
            frames += len(q)
        # Some joint leaves the 0.9 soft range used by the limit penalty in 17.3 % of
        # the frames (26.7 % for the G1 references), mostly the ankle roll.
        self.assertAlmostEqual(outside_soft / frames, 0.173, delta=0.01)

    def test_reference_heights_fit_the_task_thresholds(self):
        walk = np.concatenate([read_motion(path)["body_pos_w"][:, 0, 2] for name, path in self.pm01.items()
                               if name.startswith("WalkandRun")])
        g1_walk = np.concatenate([read_motion(path)["body_pos_w"][:, 0, 2] for name, path in self.g1.items()
                                  if name.startswith("WalkandRun")])
        # No walking or running reference frame would be terminated or called fallen.
        self.assertGreater(float(walk.min()), PM01.recovery_height)
        self.assertGreater(float(walk.min()), pm01.PM01_MIN_BASE_HEIGHT)
        # The thresholds are the G1's, scaled by how much higher the PM01 base walks.
        ratio = float(np.median(walk) / np.median(g1_walk))
        self.assertAlmostEqual(ratio, 1.035, delta=0.005)
        self.assertAlmostEqual(pm01.PM01_MIN_BASE_HEIGHT, 0.5 * ratio, delta=0.005)
        self.assertAlmostEqual(PM01.recovery_height, G1.recovery_height * ratio, delta=0.005)
        self.assertAlmostEqual(PM01.upright_height, G1.upright_height * ratio, delta=0.005)
        self.assertAlmostEqual(float(np.percentile(walk, 99.5)), pm01.PM01_STANDING_HEIGHT, delta=0.02)
        recovery = read_motion(self.pm01["Recovery/fallAndGetUp1_subject1.npz"])["body_pos_w"][:, 0, 2]
        self.assertLess(float(recovery.min()), 0.3)
        self.assertGreater(float((recovery < PM01.recovery_height).mean()), 0.4)

    def test_feet_rest_on_the_floor_in_the_walking_references(self):
        model, data = self.model, self.data
        soles = [model.geom(f"{side}_foot_{part}_collision").id for side in ("left", "right") for part in ("sole", "toe")]
        lowest = []
        for name, path in self.pm01.items():
            if not name.startswith("WalkandRun"):
                continue
            motion = read_motion(path)
            for frame in range(0, len(motion["joint_pos"]), 5):
                data.qpos[:3] = motion["body_pos_w"][frame, 0]
                data.qpos[3:7] = motion["body_quat_w"][frame, 0]
                data.qpos[7:] = motion["joint_pos"][frame]
                mujoco.mj_kinematics(model, data)
                lowest.append(min(data.geom_xpos[g][2] - float(np.abs(data.geom_xmat[g].reshape(3, 3)[2]) @ model.geom_size[g])
                                  for g in soles))
        lowest = np.asarray(lowest)
        self.assertGreater(float(lowest.min()), -0.011)
        self.assertLess(abs(float(np.median(lowest))), 0.005)


class Pm01MirroringTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mapping = build_mirror_map(PM01)
        cls.model = mujoco.MjModel.from_xml_path(str(pm01.PM01_XML))

    def test_mirror_map_pairs_left_and_right_with_pitch_kept_and_roll_yaw_negated(self):
        mapping = self.mapping
        self.assertEqual((mapping.robot, mapping.joint_names), ("pm01", pm01.PM01_JOINT_NAMES))
        names = mapping.joint_names
        for index, name in enumerate(names):
            partner = names[mapping.joint_permutation[index]]
            self.assertEqual(partner, PM01.counterpart(name, names))
            self.assertEqual(mapping.joint_sign[index], 1. if "PITCH" in name else -1., name)
        self.assertEqual(names[mapping.joint_permutation[names.index("J12_WAIST_YAW")]], "J12_WAIST_YAW")
        body = mapping.body_names
        self.assertEqual(body[mapping.body_permutation[body.index("LINK_FOOT_L")]], "LINK_FOOT_R")
        self.assertEqual(body[mapping.body_permutation[body.index("LINK_HEAD_YAW")]], "LINK_HEAD_YAW")
        np.testing.assert_array_equal(mapping.joint_permutation[mapping.joint_permutation], np.arange(23))

    def test_measured_asymmetries_of_the_official_model_are_recorded(self):
        mapping = self.mapping
        # Elbow-pitch axes lean the same way on both arms: 5.4e-3 from a mirror pair.
        self.assertAlmostEqual(mapping.maximum_axis_asymmetry, 5.45e-3, delta=2e-4)
        self.assertLess(mapping.maximum_axis_asymmetry, PM01.mirror_axis_tolerance)
        self.assertLess(mapping.maximum_rest_position_asymmetry, 5e-6)
        axes = self.model.jnt_axis[1:]
        difference = np.linalg.norm(axes[mapping.joint_permutation] * AXIAL_SIGN - axes * mapping.joint_sign[:, None], axis=1)
        worst = [mapping.joint_names[i] for i in np.argsort(difference)[-2:]]
        self.assertEqual(set(worst), {"J16_ELBOW_PITCH_L", "J21_ELBOW_PITCH_R"})
        self.assertLess(float(np.sort(difference)[-3]), 3e-5)
        # The waist yaw range is one-sided, so limits are checked per clip instead.
        self.assertIsNotNone(mapping.joint_range)
        waist = mapping.joint_names.index("J12_WAIST_YAW")
        self.assertNotAlmostEqual(-mapping.joint_range[waist, 0], mapping.joint_range[waist, 1], places=2)
        metadata = mapping.metadata()
        self.assertIn("robot_xml_sha256", metadata)
        self.assertNotIn("g1_xml_sha256", metadata)
        self.assertEqual((metadata["robot"], metadata["symmetric_joint_limits"]), ("pm01", False))
        # The measured differences themselves go into the manifest.
        angles = metadata["hinge_axis_asymmetry_deg"]
        self.assertEqual(tuple(angles), mapping.joint_names)
        self.assertAlmostEqual(angles["J16_ELBOW_PITCH_L"], 0.31, delta=0.005)
        self.assertEqual(angles["J16_ELBOW_PITCH_L"], angles["J21_ELBOW_PITCH_R"])
        self.assertEqual(metadata["maximum_hinge_axis_asymmetry_deg"], angles["J16_ELBOW_PITCH_L"])
        self.assertLess(max(angle for name, angle in angles.items() if "ELBOW_PITCH" not in name), 1e-3)
        self.assertEqual(metadata["asymmetric_joint_ranges"], {"J12_WAIST_YAW": {
            "counterpart": "J12_WAIST_YAW", "range_rad": [-4.014, 1.57],
            "reflected_counterpart_range_rad": [-1.57, 4.014]}})

    def test_g1_map_and_metadata_are_unchanged(self):
        mapping = build_g1_mirror_map()
        self.assertEqual((mapping.robot, len(mapping.joint_names), mapping.joint_range), ("g1", 29, None))
        self.assertEqual(set(mapping.metadata()), {
            "joint_names", "body_names", "joint_permutation", "joint_sign", "joint_reference",
            "body_permutation", "g1_xml_sha256", "maximum_rest_position_asymmetry_m", "convention"})
        self.assertEqual(build_mirror_map(G1).metadata(), mapping.metadata())
        with self.assertRaisesRegex(ValueError, "G1 29-hinge"):
            build_mirror_map(G1, pm01.PM01_XML)
        with self.assertRaisesRegex(ValueError, "PM01 23-hinge"):
            build_mirror_map(PM01, G1.xml_path)

    def test_mirroring_twice_returns_the_clip_and_swaps_the_turn_direction(self):
        motion = read_motion(PM01_MOTIONS / "WalkandRun/arc_walk_left_loop_001__A029.npz")
        mirrored = mirror_motion(motion, self.mapping)
        restored = mirror_motion(mirrored, self.mapping, plane_y=float(motion["body_pos_w"][0, 0, 1]))
        for key in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"):
            np.testing.assert_allclose(restored[key], motion[key], atol=1e-6, err_msg=key)
        self.assertGreater(float(motion["body_ang_vel_w"][:, 0, 2].mean()), 0.1)
        np.testing.assert_allclose(mirrored["body_ang_vel_w"][:, 0, 2], -motion["body_ang_vel_w"][:, 0, 2], atol=1e-7)
        left, right = self.mapping.body_names.index("LINK_FOOT_L"), self.mapping.body_names.index("LINK_FOOT_R")
        np.testing.assert_allclose(mirrored["body_pos_w"][:, left, 2], motion["body_pos_w"][:, right, 2], atol=1e-7)

    def test_a_mirrored_clip_outside_the_one_sided_waist_range_is_rejected(self):
        motion = read_motion(PM01_MOTIONS / "WalkandRun/idle_turn_270_001__A048.npz")
        motion["joint_pos"] = motion["joint_pos"].copy()
        motion["joint_pos"][:, self.mapping.joint_names.index("J12_WAIST_YAW")] = -2.0  # allowed; +2.0 is not
        with self.assertRaisesRegex(ValueError, "not mirror-symmetric"):
            mirror_motion(motion, self.mapping)

    def test_prepared_dataset_is_balanced_consistent_with_the_model_and_robot_specific(self):
        model, data = self.model, mujoco.MjData(self.model)
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "pm01_mirrored"
            paths = prepare_mirrored_dataset(PM01_MOTIONS, output, profile=PM01)
            manifest = json.loads(paths.manifest.read_text())
            self.assertEqual(manifest["format"], "pm01-sagittal-mirrored-motions-v1")
            recorded = manifest["robot_mapping"]
            self.assertAlmostEqual(recorded["hinge_axis_asymmetry_deg"]["J21_ELBOW_PITCH_R"], 0.31, delta=0.005)
            self.assertEqual(recorded["asymmetric_joint_ranges"]["J12_WAIST_YAW"]["range_rad"], [-4.014, 1.57])
            self.assertIn("J16_ELBOW_PITCH_L and J21_ELBOW_PITCH_R differ by 0.31 deg", manifest["limits"][1])
            self.assertIn("J12_WAIST_YAW is one-sided (-4.014 to 1.57 rad)", manifest["limits"][1])
            self.assertEqual((manifest["walking_original_clips"], manifest["walking_total_clips"],
                              manifest["recovery_original_clips"]), (17, 34, 1))
            after = manifest["turn_statistics"]["after"]
            self.assertEqual(after["positive_yaw_frames"], after["negative_yaw_frames"])
            self.assertEqual((paths.recovery / "fallAndGetUp1_subject1.npz").read_bytes(),
                             (PM01_MOTIONS / "Recovery/fallAndGetUp1_subject1.npz").read_bytes())
            verify_mirrored_dataset(output, source_root=PM01_MOTIONS)
            self.assertEqual(prepare_mirrored_dataset(PM01_MOTIONS, output, profile=PM01), paths)
            # A cache made for one robot is never reused for the other.
            with self.assertRaises(ValueError):
                prepare_mirrored_dataset(PM01_MOTIONS, output)
            # Mirrored joints on the real, slightly asymmetric model reproduce the mirrored
            # bodies to within a millimetre.
            worst = 0.
            for path in sorted(paths.walk.glob(f"*{MIRROR_SUFFIX}.npz")):
                motion = read_motion(path)
                for frame in range(0, len(motion["joint_pos"]), 25):
                    data.qpos[:3] = motion["body_pos_w"][frame, 0]
                    data.qpos[3:7] = motion["body_quat_w"][frame, 0]
                    data.qpos[7:] = motion["joint_pos"][frame]
                    mujoco.mj_kinematics(model, data)
                    worst = max(worst, float(np.abs(data.xpos[1:] - motion["body_pos_w"][frame]).max()))
            self.assertLess(worst, 1.5e-3)


if __name__ == "__main__":
    unittest.main()
