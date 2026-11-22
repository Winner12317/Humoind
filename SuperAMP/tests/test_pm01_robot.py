"""EngineAI PM01 model, actuators, robot profile and task wiring (CPU only).

    python -m unittest discover -s tests -p "test_pm01_*.py" -v
"""

from __future__ import annotations

import math
import re
import unittest

import mujoco
import numpy as np

from src.assets.robots.engineai_pm01 import pm01_actuators as actuators
from src.assets.robots.engineai_pm01 import pm01_constants as pm01
from src.robot_profiles import G1, PM01, profile_for_bodies, profile_for_joint_count, profile_for_task


def lowest_collision_point(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    """Lowest world z of the collision primitives (spheres, boxes, cylinders)."""
    lowest = math.inf
    for geom in range(model.ngeom):
        kind, size = int(model.geom_type[geom]), model.geom_size[geom]
        if kind == mujoco.mjtGeom.mjGEOM_MESH:
            continue
        rotation, z = data.geom_xmat[geom].reshape(3, 3), data.geom_xpos[geom][2]
        if kind == mujoco.mjtGeom.mjGEOM_SPHERE:
            reach = size[0]
        elif kind == mujoco.mjtGeom.mjGEOM_BOX:
            reach = float(np.abs(rotation[2]) @ size)
        elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
            reach = abs(rotation[2, 2]) * size[1] + size[0] * math.sqrt(max(0., 1 - rotation[2, 2] ** 2))
        else:
            raise AssertionError(f"unexpected collision geom type {kind}")
        lowest = min(lowest, z - reach)
    return lowest


def default_qpos(model: mujoco.MjModel) -> np.ndarray:
    qpos = model.qpos0.copy()
    qpos[:3] = pm01.HOME_KEYFRAME.pos
    for joint in range(1, model.njnt):
        name = model.joint(joint).name
        for pattern, value in pm01.HOME_KEYFRAME.joint_pos.items():
            if re.fullmatch(pattern, name):
                qpos[model.jnt_qposadr[joint]] = value
    return qpos


class Pm01ModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = mujoco.MjModel.from_xml_path(str(pm01.PM01_XML))
        cls.data = mujoco.MjData(cls.model)

    def test_joints_are_the_23_sdk_joints_in_sdk_order_and_the_head_is_welded(self):
        model = self.model
        self.assertEqual(model.jnt_type[0], mujoco.mjtJoint.mjJNT_FREE)
        hinges = tuple(model.joint(j).name for j in range(1, model.njnt))
        self.assertEqual(hinges, pm01.PM01_JOINT_NAMES)
        self.assertEqual((model.nq, model.nv, model.nu), (30, 29, 0))
        self.assertTrue((model.jnt_type[1:] == mujoco.mjtJoint.mjJNT_HINGE).all())
        self.assertFalse(model.jnt_pos.any(), "PM01 hinges sit at their body origins")
        head = model.body("LINK_HEAD_YAW")
        self.assertEqual(model.body_jntnum[head.id], 0)
        self.assertEqual(model.body(model.body_parentid[head.id]).name, "LINK_TORSO_YAW")

    def test_bodies_mass_and_passive_joint_terms(self):
        model = self.model
        self.assertEqual(model.nbody - 1, PM01.num_bodies)
        self.assertEqual(model.body(1).name, pm01.PM01_ROOT_BODY)
        self.assertAlmostEqual(float(model.body_mass.sum()), 40.923, places=2)
        # Armature, damping and friction belong to the actuator configuration.
        self.assertFalse(model.dof_armature.any())
        self.assertFalse(model.dof_damping.any())
        self.assertFalse(model.dof_frictionloss.any())

    def test_collision_geoms_are_named_and_match_the_configuration_patterns(self):
        model = self.model
        collision = [g for g in range(model.ngeom) if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH]
        names = [model.geom(g).name for g in collision]
        self.assertEqual(len(names), 32)
        self.assertEqual(len(set(names)), 32)
        self.assertTrue(all(re.fullmatch(".*_collision", name) for name in names))
        feet = [name for name in names if re.fullmatch(pm01._FOOT_GEOMS, name)]
        self.assertEqual(tuple(feet), pm01.PM01_FOOT_GEOMS)
        for name in feet:
            self.assertEqual(model.geom_type[model.geom(name).id], mujoco.mjtGeom.mjGEOM_BOX)
        # Visual meshes never collide.
        meshes = [g for g in range(model.ngeom) if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH]
        self.assertEqual(len(meshes), 25)
        self.assertFalse(model.geom_contype[meshes].any() or model.geom_conaffinity[meshes].any())

    def test_foot_sites_are_the_centre_of_the_sole_underside(self):
        model = self.model
        for side, body, site, offset in zip(("left", "right"), pm01.PM01_FOOT_BODIES, pm01.PM01_FOOT_SITES,
                                           PM01.foot_reference_offsets):
            site_id = model.site(site).id
            self.assertEqual(model.body(model.site_bodyid[site_id]).name, body)
            np.testing.assert_allclose(model.site_pos[site_id], offset, atol=1e-9)
            sole = model.geom(f"{side}_foot_sole_collision")
            # The sole box hangs on LINK_FOOT_*, which shares the ankle-roll frame.
            foot_body = model.geom_bodyid[sole.id]
            self.assertEqual(model.body(model.body_parentid[foot_body]).name, body)
            self.assertFalse(model.body_pos[foot_body].any())
            expected = model.geom_pos[sole.id] - [0., 0., model.geom_size[sole.id][2]]
            np.testing.assert_allclose(model.site_pos[site_id], expected, atol=1e-9)

    def test_imu_site_and_sensors(self):
        model = self.model
        imu = model.site(pm01.PM01_IMU_SITE)
        self.assertEqual(model.body(model.site_bodyid[imu.id]).name, pm01.PM01_ROOT_BODY)
        np.testing.assert_allclose(model.site_pos[imu.id], (0.02311, 0., -0.09631), atol=1e-9)
        sensors = {model.sensor(i).name for i in range(model.nsensor)}
        self.assertLessEqual({"imu_ang_vel", "imu_lin_vel", "imu_lin_acc"}, sensors)

    def test_neutral_poses_have_no_self_contact_and_hand_thigh_pairs_are_excluded(self):
        model, data = self.model, self.data
        excluded = set()
        for pair in range(model.nexclude):
            signature = int(model.exclude_signature[pair])
            excluded.add(frozenset((model.body(signature >> 16).name, model.body(signature & 0xFFFF).name)))
        self.assertEqual(excluded, {
            frozenset(("LINK_ELBOW_END_L", "LINK_HIP_YAW_L")), frozenset(("LINK_ELBOW_END_L", "LINK_HIP_ROLL_L")),
            frozenset(("LINK_ELBOW_END_R", "LINK_HIP_YAW_R")), frozenset(("LINK_ELBOW_END_R", "LINK_HIP_ROLL_R")),
        })
        for qpos in (model.qpos0.copy(), default_qpos(model)):
            qpos[2] = 2.
            data.qpos[:] = qpos
            mujoco.mj_forward(model, data)
            self.assertEqual(data.ncon, 0)

    def test_default_pose_stands_on_its_soles_at_the_configured_height(self):
        model, data = self.model, self.data
        data.qpos[:] = default_qpos(model)
        mujoco.mj_forward(model, data)
        clearance = lowest_collision_point(model, data)
        # The base is spawned within 2 mm of where the soles touch the floor.
        self.assertGreaterEqual(clearance, 0.)
        self.assertLess(clearance, 2e-3)
        soles = [data.geom_xpos[model.geom(name).id][2] - model.geom_size[model.geom(name).id][2]
                 for name in ("left_foot_sole_collision", "right_foot_sole_collision")]
        # Both soles are flat on the floor (the tilted hip-pitch axes leave 0.2 mm of roll).
        np.testing.assert_allclose(soles, clearance, atol=5e-4)


class Pm01ActuatorTest(unittest.TestCase):
    def test_gains_follow_the_ten_hertz_rule_on_the_published_armature(self):
        omega = 2 * math.pi * 10
        self.assertAlmostEqual(pm01.STIFFNESS_Q90, actuators.ARMATURE_Q90 * omega**2, places=6)
        self.assertAlmostEqual(pm01.STIFFNESS_Q25, actuators.ARMATURE_Q25 * omega**2, places=6)
        self.assertAlmostEqual(pm01.DAMPING_Q90, 4 * actuators.ARMATURE_Q90 * omega, places=6)
        self.assertAlmostEqual(pm01.DAMPING_Q25, 4 * actuators.ARMATURE_Q25 * omega, places=6)
        self.assertAlmostEqual(pm01.STIFFNESS_Q90, 178.84, places=2)
        self.assertAlmostEqual(pm01.STIFFNESS_Q25, 26.45, places=2)

    def test_every_joint_has_exactly_one_actuator_with_its_module_limits(self):
        expected = {}
        for name in pm01.PM01_JOINT_NAMES:
            high_torque = any(token in name for token in ("HIP_PITCH", "HIP_ROLL", "KNEE_PITCH"))
            ankle = "ANKLE" in name
            expected[name] = dict(
                stiffness=pm01.STIFFNESS_Q90 if high_torque else pm01.STIFFNESS_Q25,
                damping=pm01.DAMPING_Q90 if high_torque else pm01.DAMPING_ANKLE if ankle else pm01.DAMPING_Q25,
                effort=164. if high_torque else 52., speed=26.3 if high_torque else 35.2,
                armature=0.0453 if high_torque else 0.0067,
            )
        seen = {}
        for cfg in pm01.PM01_ARTICULATION.actuators:
            for name in pm01.PM01_JOINT_NAMES:
                if any(re.fullmatch(pattern, name) for pattern in cfg.target_names_expr):
                    self.assertNotIn(name, seen)
                    seen[name] = cfg
        self.assertEqual(set(seen), set(pm01.PM01_JOINT_NAMES))
        for name, cfg in seen.items():
            want = expected[name]
            self.assertAlmostEqual(cfg.stiffness, want["stiffness"], places=9, msg=name)
            self.assertAlmostEqual(cfg.damping, want["damping"], places=9, msg=name)
            self.assertEqual((cfg.effort_limit, cfg.Y1, cfg.Y2), (want["effort"],) * 3, name)
            # Peak torque up to the published maximum speed, none above it.
            self.assertEqual((cfg.X1, cfg.X2), (want["speed"],) * 2, name)
            self.assertEqual(cfg.armature, want["armature"], name)
            self.assertEqual((cfg.Fs, cfg.Fd), (0.8, 0.6), name)
        self.assertEqual(pm01.PM01_ARTICULATION.soft_joint_pos_limit_factor, 0.9)

    def test_action_scale_is_a_quarter_of_the_saturating_position_error(self):
        for cfg in pm01.PM01_ARTICULATION.actuators:
            for pattern in cfg.target_names_expr:
                self.assertAlmostEqual(pm01.PM01_ACTION_SCALE[pattern], 0.25 * cfg.effort_limit / cfg.stiffness)
        values = sorted(set(round(v, 4) for v in pm01.PM01_ACTION_SCALE.values()))
        self.assertEqual(values, [0.2293, 0.4915])
        for name in pm01.PM01_JOINT_NAMES:
            self.assertEqual(sum(bool(re.fullmatch(p, name)) for p in pm01.PM01_ACTION_SCALE), 1, name)


class Pm01ProfileTest(unittest.TestCase):
    def test_profile_agrees_with_the_robot_constants_and_the_model(self):
        model = mujoco.MjModel.from_xml_path(str(PM01.xml_path))
        self.assertEqual(PM01.xml_path, pm01.PM01_XML)
        self.assertEqual(PM01.num_joints, len(pm01.PM01_JOINT_NAMES))
        self.assertEqual((PM01.root_body, PM01.torso_body), (pm01.PM01_ROOT_BODY, pm01.PM01_TORSO_BODY))
        self.assertEqual(PM01.foot_bodies, pm01.PM01_FOOT_BODIES)
        self.assertEqual(PM01.foot_sites, pm01.PM01_FOOT_SITES)
        self.assertEqual(PM01.imu_site, pm01.PM01_IMU_SITE)
        self.assertEqual(PM01.amp_body_names, pm01.PM01_AMP_BODY_NAMES)
        self.assertEqual(PM01.recovery_height, pm01.PM01_RECOVERY_HEIGHT)
        body_names = [model.body(i).name for i in range(1, model.nbody)]
        indices = [body_names.index(name) for name in PM01.amp_body_names]
        # Model order: the simulator's observation and the expert loader then agree.
        self.assertEqual(indices, sorted(indices))
        self.assertEqual(len(PM01.amp_body_names), len(G1.amp_body_names))

    def test_profiles_are_found_from_tasks_bodies_and_joint_counts(self):
        model = mujoco.MjModel.from_xml_path(str(PM01.xml_path))
        self.assertIs(profile_for_bodies(model.body(i).name for i in range(1, model.nbody)), PM01)
        self.assertIs(profile_for_bodies(f"robot/{model.body(i).name}" for i in range(1, model.nbody)), PM01)
        g1 = mujoco.MjModel.from_xml_path(str(G1.xml_path))
        self.assertIs(profile_for_bodies(g1.body(i).name for i in range(1, g1.nbody)), G1)
        self.assertIs(profile_for_task("EngineAI-PM01-AMP-Flat"), PM01)
        self.assertIs(profile_for_task("Unitree-G1-AMP-Flat-Aligned"), G1)
        self.assertEqual(PM01.task_schema("EngineAI-PM01-AMP-Flat-Aligned"), "grounded_v2")
        self.assertEqual(PM01.task_schema("EngineAI-PM01-AMP-Flat"), "legacy")
        self.assertIs(profile_for_joint_count(23), PM01)
        self.assertIs(profile_for_joint_count(29), G1)
        with self.assertRaises(ValueError):
            profile_for_task("EngineAI-PM01-AMP-Rough")
        with self.assertRaises(ValueError):
            profile_for_bodies(("base", "trunk"))

    def test_g1_profile_keeps_the_original_constants(self):
        from src.amp_features import G1_FOOT_REFERENCE_OFFSET
        self.assertEqual(G1.foot_reference_offsets, (G1_FOOT_REFERENCE_OFFSET,) * 2)
        self.assertIsNone(G1.amp_foot_offsets)
        self.assertEqual(PM01.amp_foot_offsets, PM01.foot_reference_offsets)
        self.assertEqual((G1.recovery_height, G1.upright_height, G1.num_joints, G1.num_bodies), (.55, .65, 29, 30))
        self.assertEqual(G1.mirrored_motion_cache, ".wmac_cache/motions/g1_mirrored")

    def test_left_right_counterparts(self):
        joints = pm01.PM01_JOINT_NAMES
        self.assertEqual(PM01.counterpart("J00_HIP_PITCH_L", joints), "J06_HIP_PITCH_R")
        self.assertEqual(PM01.counterpart("J22_ELBOW_YAW_R", joints), "J17_ELBOW_YAW_L")
        self.assertEqual(PM01.counterpart("J12_WAIST_YAW", joints), "J12_WAIST_YAW")
        bodies = ("LINK_BASE", "LINK_HIP_ROLL_L", "LINK_HIP_ROLL_R", "LINK_TORSO_YAW")
        self.assertEqual(PM01.counterpart("LINK_HIP_ROLL_L", bodies), "LINK_HIP_ROLL_R")
        self.assertEqual(PM01.counterpart("LINK_TORSO_YAW", bodies), "LINK_TORSO_YAW")
        with self.assertRaises(ValueError):
            PM01.counterpart("LINK_KNEE_PITCH_L", bodies)
        g1_joints = ("left_knee_joint", "right_knee_joint", "waist_yaw_joint")
        self.assertEqual(G1.counterpart("left_knee_joint", g1_joints), "right_knee_joint")
        self.assertEqual(G1.counterpart("waist_yaw_joint", g1_joints), "waist_yaw_joint")


class Pm01TaskConfigTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from src.tasks.amp_loco.config.pm01 import env_cfgs, rl_cfg
        cls.env_cfgs, cls.rl_cfg = env_cfgs, rl_cfg
        cls.model = mujoco.MjModel.from_xml_path(str(pm01.PM01_XML))
        cls.bodies = {cls.model.body(i).name for i in range(1, cls.model.nbody)}

    def test_three_tasks_are_registered_with_the_amp_runner(self):
        import mjlab.tasks  # noqa: F401
        import src.tasks  # noqa: F401
        from mjlab.tasks.registry import list_tasks, load_rl_cfg, load_runner_cls
        from src.tasks.amp_loco.rl import AMPOnPolicyRunner
        for task in ("EngineAI-PM01-AMP-Rough", *PM01.flat_tasks):
            self.assertIn(task, list_tasks())
            self.assertIs(load_runner_cls(task), AMPOnPolicyRunner)
            self.assertEqual(load_rl_cfg(task).amp_body_names, PM01.amp_body_names)

    def test_flat_task_names_only_things_the_model_has(self):
        cfg = self.env_cfgs.pm01_amp_flat_env_cfg()
        model = self.model
        sites = {model.site(i).name for i in range(model.nsite)}
        geoms = {model.geom(i).name for i in range(model.ngeom)}
        self.assertNotIn("height_scan", cfg.observations["actor"].terms)
        for group in ("critic", "amp"):
            for term in cfg.observations[group].terms.values():
                if "body_cfg" in term.params:
                    self.assertEqual(tuple(term.params["body_cfg"].body_names), PM01.amp_body_names)
                    self.assertEqual(tuple(term.params["anchor_cfg"].body_names), (PM01.torso_body,))
        self.assertLessEqual(set(cfg.rewards["foot_slip"].params["asset_cfg"].site_names), sites)
        self.assertLessEqual(set(cfg.events["foot_friction"].params["asset_cfg"].geom_names), geoms)
        self.assertLessEqual(set(cfg.events["base_com"].params["asset_cfg"].body_names), self.bodies)
        for name in ("track_anchor_linear_velocity", "track_anchor_angular_velocity"):
            self.assertEqual(tuple(cfg.rewards[name].params["anchor_cfg"].body_names), (PM01.torso_body,))
        self.assertEqual(tuple(cfg.rewards["body_ang_vel_xy_l2"].params["body_cfg"].body_names), (PM01.root_body,))
        self.assertEqual(cfg.viewer.body_name, PM01.torso_body)
        sensors = {sensor.name: sensor for sensor in cfg.scene.sensors}
        self.assertEqual(set(sensors), {"feet_ground_contact", "self_collision"})
        feet = re.compile(sensors["feet_ground_contact"].primary.pattern)
        self.assertEqual({name for name in self.bodies if feet.fullmatch(name)}, set(PM01.foot_bodies))
        self.assertEqual(sensors["self_collision"].primary.pattern, PM01.root_body)
        self.assertEqual(cfg.actions["joint_pos"].scale, pm01.PM01_ACTION_SCALE)
        self.assertEqual(cfg.scene.terrain.terrain_type, "plane")

    def test_heights_follow_the_robot_not_the_g1(self):
        cfg = self.env_cfgs.pm01_amp_flat_env_cfg()
        self.assertEqual(cfg.terminations["bad_base_height"].params["minimum_height"], pm01.PM01_MIN_BASE_HEIGHT)
        self.assertEqual(cfg.scene.entities["robot"].init_state.pos[2], pm01.PM01_STANDING_HEIGHT)
        self.assertLess(pm01.PM01_MIN_BASE_HEIGHT, pm01.PM01_RECOVERY_HEIGHT)
        self.assertLess(pm01.PM01_RECOVERY_HEIGHT, PM01.upright_height)
        self.assertLess(PM01.upright_height, pm01.PM01_STANDING_HEIGHT)
        from src.tasks.amp_loco.config.g1.env_cfgs import g1_amp_flat_env_cfg
        self.assertEqual(g1_amp_flat_env_cfg().terminations["bad_base_height"].params["minimum_height"], 0.5)

    def test_motion_directories_and_runner_configuration(self):
        cfg = self.env_cfgs.pm01_amp_flat_env_cfg()
        loader = cfg.events["init_motion_loader"].params
        self.assertTrue(loader["motion_dir"].endswith("assets/motions/pm01/amp/WalkandRun"))
        self.assertTrue(loader["recovery_dir"].endswith("assets/motions/pm01/amp/Recovery"))
        self.assertEqual(cfg.events["reset_from_motion"].params["motion_dir"], loader["motion_dir"])
        self.assertEqual((loader["delay_reset_env_ratio"], loader["max_delay_steps"]), (0.4, 250))
        runner = self.rl_cfg.pm01_amp_ppo_runner_cfg()
        self.assertTrue(runner.amp_motion_files.endswith("assets/motions/pm01/amp"))
        self.assertEqual(runner.min_normalized_std, [0.05] * 23)
        self.assertEqual((runner.amp_anchor_name, runner.amp_feature_schema), (PM01.torso_body, "legacy"))
        self.assertEqual(runner.experiment_name, "pm01_amp_locomotion")
        aligned = self.rl_cfg.pm01_amp_aligned_ppo_runner_cfg()
        self.assertEqual((aligned.amp_feature_schema, aligned.experiment_name),
                         ("grounded_v2", "pm01_amp_locomotion_aligned"))

    def test_aligned_task_uses_grounded_features_on_pm01_bodies(self):
        cfg = self.env_cfgs.pm01_amp_flat_aligned_env_cfg()
        term = cfg.observations["amp"].terms["grounded_features"]
        self.assertEqual(term.params["schema"], "grounded_v2")
        self.assertEqual(tuple(term.params["pelvis_cfg"].body_names), (PM01.root_body,))
        self.assertEqual(tuple(term.params["foot_cfg"].body_names), PM01.foot_bodies)
        self.assertEqual(tuple(term.params["body_cfg"].body_names), PM01.amp_body_names)
        self.assertEqual(set(cfg.observations["critic"].terms), {"physical_context"})
        self.assertEqual(cfg.scene.num_envs, 4096)


if __name__ == "__main__":
    unittest.main()
