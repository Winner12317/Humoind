"""Unitree-convention deployment interface: conventions checked against MuJoCo FK."""
from __future__ import annotations

import re
import unittest

import mujoco
import numpy as np
import torch

from src.wmac_perceptive import unitree_interface as ui

# Unitree MJCF waist chain (unitree_mujoco g1_29dof.xml body offsets).
_UNITREE_WAIST = """
<mujoco>
  <worldbody>
    <body name="pelvis">
      <freejoint/>
      <geom type="sphere" size=".05" mass="3"/>
      <body name="waist_yaw_link">
        <joint name="waist_yaw_joint" axis="0 0 1"/>
        <geom type="sphere" size=".02" mass=".2"/>
        <body name="waist_roll_link" pos="-0.0039635 0 0.035">
          <joint name="waist_roll_joint" axis="1 0 0"/>
          <geom type="sphere" size=".02" mass=".05"/>
          <body name="torso_link" pos="0 0 0.019">
            <joint name="waist_pitch_joint" axis="0 1 0"/>
            <geom type="sphere" size=".1" mass="9"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def _random_quat(rng):
    q = rng.normal(size=4)
    return q / np.linalg.norm(q)


class UnitreeInterfaceTest(unittest.TestCase):
    def test_torso_from_pelvis_matches_unitree_forward_kinematics(self):
        model = mujoco.MjModel.from_xml_string(_UNITREE_WAIST)
        data = mujoco.MjData(model)
        torso = model.body("torso_link").id
        rng = np.random.default_rng(3)
        for _ in range(20):
            pelvis = _random_quat(rng)
            waist = rng.uniform(-.5, .5, 3)
            waist_rate = rng.uniform(-3, 3, 3)
            pelvis_gyro = rng.uniform(-3, 3, 3)
            data.qpos[:] = 0
            data.qpos[3:7] = pelvis
            data.qpos[7:10] = waist
            data.qvel[:] = 0
            data.qvel[3:6] = pelvis_gyro          # free-joint angular velocity is in the body frame
            data.qvel[6:9] = waist_rate
            mujoco.mj_forward(model, data)
            velocity = np.zeros(6)
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, torso, velocity, 1)
            quat, gyro = ui.torso_from_pelvis(pelvis, pelvis_gyro, waist, waist_rate)
            same = min(np.abs(quat - data.xquat[torso]).max(), np.abs(quat + data.xquat[torso]).max())
            self.assertLess(same, 1e-9)
            np.testing.assert_allclose(gyro, velocity[:3], atol=1e-9)

    def test_waist_sign_maps_training_robot_onto_unitree_orientation(self):
        from src.tasks.amp_perceptive.hiking_robot import get_hiking_robot_spec
        train = get_hiking_robot_spec().compile()
        unitree = mujoco.MjModel.from_xml_string(_UNITREE_WAIST)
        rng = np.random.default_rng(5)
        names = ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")
        sign = np.array([ui.UNITREE_JOINT_SIGN[ui.POLICY_JOINT_NAMES.index(n)] for n in names])
        self.assertTrue(np.all(sign == -1))
        for _ in range(10):
            q_policy = rng.uniform(-.5, .5, 3)
            poses = []
            for model, q in ((train, q_policy), (unitree, sign * q_policy)):
                data = mujoco.MjData(model)
                data.qpos[:] = 0
                data.qpos[3] = 1
                for name, value in zip(names, q):
                    data.qpos[model.jnt_qposadr[model.joint(name).id]] = value
                mujoco.mj_kinematics(model, data)
                poses.append(data.xquat[model.body("torso_link").id].copy())
            same = min(np.abs(poses[0] - poses[1]).max(), np.abs(poses[0] + poses[1]).max())
            self.assertLess(same, 1e-9)
            # Without the conversion the torso would lean the other way.
            data = mujoco.MjData(unitree)
            data.qpos[3] = 1
            data.qpos[7:10] = q_policy
            mujoco.mj_kinematics(unitree, data)
            wrong = min(np.abs(poses[0] - data.xquat[-1]).max(), np.abs(poses[0] + data.xquat[-1]).max())
            self.assertGreater(wrong, 1e-3)

    def test_proprio_history_matches_training_packing(self):
        from src.wmac_perceptive.hiking_observations import pack_hiking_proprio_history
        rng = np.random.default_rng(1)
        frames = rng.normal(size=(8, 96)).astype(np.float32)
        iface = ui.UnitreeG1PolicyInterface()
        iface._proprio = frames
        expected = pack_hiking_proprio_history(torch.from_numpy(frames)[None])[0].numpy()
        np.testing.assert_array_equal(iface.proprio_history(), expected)

    def test_actions_defaults_and_gains_match_training_actuators(self):
        from src.assets.robots.unitree_g1.g1_constants import KNEES_BENT_KEYFRAME
        from src.wmac_perceptive.hiking_actuators import _GROUP_EXPRESSIONS, _joint_parameters
        defaults = ui.default_joint_positions()
        for i, name in enumerate(ui.POLICY_JOINT_NAMES):
            expected = 0.
            for pattern, value in KNEES_BENT_KEYFRAME.joint_pos.items():
                if re.fullmatch(pattern, name):
                    expected = value
            self.assertAlmostEqual(defaults[i], expected)
            parameters = _joint_parameters(name)
            self.assertAlmostEqual(ui.joint_parameters()["action_scale"][i],
                                   .25 * parameters["effort_limit"] / parameters["kp"])
        grouped = {name for patterns in _GROUP_EXPRESSIONS.values() for pattern in patterns
                   for name in ui.POLICY_JOINT_NAMES if re.fullmatch(pattern, name)}
        self.assertEqual(grouped, set(ui.POLICY_JOINT_NAMES))
        iface = ui.UnitreeG1PolicyInterface()
        action = np.random.default_rng(2).normal(size=29)
        targets = iface.unitree_targets(action)
        np.testing.assert_allclose(ui.UNITREE_JOINT_SIGN * targets,
                                   defaults + ui.joint_parameters()["action_scale"] * action)

    def test_policy_input_width_and_depth_semantics(self):
        iface = ui.UnitreeG1PolicyInterface()
        depth = np.full((36, 64), 1.25, np.float32)
        depth[0, 0], depth[0, 1], depth[0, 2] = np.inf, .05, 3.
        frame = ui.depth_frame(depth)
        self.assertEqual(frame.shape, (2, 36, 64))
        self.assertEqual(frame[0, 0, 0], np.float32(ui.DEPTH_MISSING_M / 2.5))
        self.assertEqual(frame[0, 0, 1], np.float32(ui.DEPTH_MISSING_M / 2.5))   # closer than 0.1 m
        self.assertAlmostEqual(float(frame[0, 0, 2]), 3. / 2.5, places=6)         # kept unclipped
        self.assertEqual(frame[1, 0, 2], 0.)
        self.assertAlmostEqual(float(frame[0, 5, 5]), .5, places=6)
        iface.reset(torso_quat=np.array([1., 0, 0, 0]), torso_gyro=np.zeros(3), q_unitree=np.zeros(29),
                    dq_unitree=np.zeros(29), command=np.zeros(3), depth_frame=frame)
        self.assertEqual(iface.policy_input().shape, (1, 768 + 37 * 2 * 36 * 64))
        np.testing.assert_allclose(iface.proprio_history()[24:48].reshape(8, 3), [[0., 0., -1.]] * 8)

    def test_contract_records_conventions(self):
        contract = ui.deployment_contract()
        self.assertEqual(contract["joints"]["order"], list(ui.POLICY_JOINT_NAMES))
        self.assertEqual([n for n, s in zip(contract["joints"]["order"], contract["joints"]["unitree_sign"]) if s < 0],
                         list(ui.WAIST_JOINT_NAMES))


if __name__ == "__main__":
    unittest.main()
