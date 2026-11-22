"""WMAC state adapter, world model, smooth costs and sole geometry on the PM01 (CPU only).

The G1 versions of these checks live in test_wmac_state_fast.py,
test_smooth_reward_kernel.py and test_recovery_foot_slip_geometry.py.
"""

from __future__ import annotations

from dataclasses import replace
import re
from types import SimpleNamespace
import unittest

import mujoco
import numpy as np
import torch

from mjlab.actuator import ActuatorCmd

from src.amp_features import amp_feature_metadata, build_amp_features
from src.assets.robots.engineai_pm01 import pm01_constants as pm01
from src.assets.robots.engineai_pm01.pm01_actuators import Pm01ActuatorCfg
from src.assets.robots.unitree_g1.unitree_actuators import UnitreeActuator
from src.robot_profiles import PM01
from src.wmac.config import WmacConfig
from src.wmac.foot_slip import FootSlipGeometry
from src.wmac.networks import WorldModel, invariant_features
from src.wmac.replay import ReplayBuffer
from src.wmac.reward_integration import previous_previous_action
from src.wmac.smooth_rewards import SmoothRewardTerms
from src.wmac.state import (StateAdapter, actor_frame_dim, context_dim, joint_count, joint_positions,
                            joint_velocities, quat_normalize, recovery_foot_slip_mask, state_dim)

JOINTS = 23
STATE, CONTEXT, FRAME = 59, 29, 78


def make_pm01_adapter(dtype=torch.float64):
    """The real PM01 MuJoCo model with only the CPU env metadata the adapter reads."""
    model = mujoco.MjModel.from_xml_path(str(pm01.PM01_XML))
    body_ids = np.arange(1, model.nbody)
    joint_ids = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_HINGE)
    default_q = torch.tensor(model.qpos0[model.jnt_qposadr[joint_ids]], dtype=torch.float32)[None]
    limits = torch.tensor(model.jnt_range[joint_ids], dtype=torch.float32)[None]
    root = torch.zeros(1, 13)
    root[0, :7] = torch.tensor(model.qpos0[:7], dtype=torch.float32)
    indexing = SimpleNamespace(
        body_ids=torch.tensor(body_ids), joint_ids=torch.tensor(joint_ids),
        geom_ids=torch.arange(model.ngeom),
        geoms=[SimpleNamespace(id=i, name=model.geom(i).name) for i in range(model.ngeom)],
        sites=[SimpleNamespace(id=i, name=model.site(i).name) for i in range(model.nsite)],
    )
    robot = SimpleNamespace(
        body_names=tuple(model.body(i).name for i in body_ids), indexing=indexing,
        data=SimpleNamespace(default_joint_pos=default_q, default_joint_vel=torch.zeros(1, JOINTS),
                             soft_joint_pos_limits=limits, default_root_state=root),
    )

    def term(weight, **params):
        return SimpleNamespace(weight=weight, params=params)

    rewards = {
        "track_anchor_linear_velocity": term(1., std=1., mask_delay=True, delay_env_rew_ratio=0.),
        "track_anchor_angular_velocity": term(1., std=3.14, mask_delay=True, delay_env_rew_ratio=0.),
        "track_root_height": term(1., std=.3, mask_delay=True, delay_env_rew_ratio=3.5),
        "body_ang_vel_xy_l2": term(.5, std=3.14, mask_delay=True, delay_env_rew_ratio=0.),
        "is_terminated": term(-200.), "joint_acc_l2": term(-2.5e-7),
        "joint_pos_limits": term(-10.), "action_rate_l2": term(-.01),
        "foot_slip": term(-.25), "self_collisions": term(-.1),
    }
    grounded = SimpleNamespace(terms={"grounded_features": SimpleNamespace(params={"schema": "grounded_v2"})})
    env = SimpleNamespace(
        scene={"robot": robot}, device="cpu", step_dt=.02,
        sim=SimpleNamespace(mj_model=model),
        cfg=SimpleNamespace(rewards=rewards, scale_rewards_by_dt=True, observations={"amp": grounded}, terminations={
            "bad_base_height": term(0., minimum_height=pm01.PM01_MIN_BASE_HEIGHT),
            "bad_orientation": term(0., limit_angle=np.deg2rad(70.)),
        }), termination_manager=SimpleNamespace(_max_delay_steps=250),
    )
    adapter = StateAdapter(env)
    for name, value in vars(adapter).items():
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            setattr(adapter, name, value.to(dtype))
    adapter._fk_levels = tuple(replace(level, **{
        name: getattr(level, name).to(dtype)
        for name in ("offsets", "orientations", "axes", "pivots", "qrefs")
    }) for level in adapter._fk_levels)
    return adapter, model


def sample_state(count=6, dtype=torch.float64):
    generator = torch.Generator().manual_seed(2301)
    state = torch.randn(count, STATE, generator=generator, dtype=dtype) * .2
    state[:, 2] += .7
    state[:, 3:7] = quat_normalize(torch.randn(count, 4, generator=generator, dtype=dtype))
    state[0, 3:7] = state.new_tensor([.5, .5, .5, -.5])  # lying, forward axis vertical
    state[1, 3:7] = state.new_tensor([1., 0., 0., 0.])
    return state


def mujoco_bodies(model, state):
    """Positions, quaternions and world velocities of every body from MuJoCo itself."""
    data = mujoco.MjData(model)
    data.qpos[:7] = state[:7].numpy()
    data.qpos[7:] = state[13:13 + JOINTS].numpy()
    # qvel holds the root angular velocity in the body frame; the state holds it in the world frame.
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, state[3:7].numpy().copy())
    rotation = rotation.reshape(3, 3)
    data.qvel[:3] = state[7:10].numpy()
    data.qvel[3:6] = rotation.T @ state[10:13].numpy()
    data.qvel[6:] = state[13 + JOINTS:].numpy()
    mujoco.mj_forward(model, data)
    linear, angular = [], []
    for body in range(1, model.nbody):
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_XBODY, body, velocity, 0)
        angular.append(velocity[:3].copy())
        linear.append(velocity[3:].copy())
    return data, data.xpos[1:].copy(), data.xquat[1:].copy(), np.asarray(linear), np.asarray(angular)


class Pm01StateAdapterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter, cls.model = make_pm01_adapter()

    def test_layout_widths_come_from_the_joint_count(self):
        adapter = self.adapter
        self.assertIs(adapter.profile, PM01)
        self.assertEqual((adapter.num_joints, adapter.state_dim, adapter.context_dim, adapter.actor_frame_dim),
                         (JOINTS, STATE, CONTEXT, FRAME))
        self.assertEqual((state_dim(29), context_dim(29), actor_frame_dim(29)), (71, 35, 96))
        state = sample_state()
        self.assertEqual(joint_count(state), JOINTS)
        torch.testing.assert_close(joint_positions(state), state[:, 13:36])
        torch.testing.assert_close(joint_velocities(state), state[:, 36:59])
        self.assertEqual(adapter.amp_dim, 203)
        self.assertEqual(adapter.recovery_height, 0.57)
        self.assertEqual(adapter.height_threshold, 0.52)
        self.assertEqual(len(adapter.friction_geom_ids), 4)
        np.testing.assert_allclose(adapter.imu_offset.numpy(), (0.02311, 0., -0.09631), atol=1e-7)

    def test_names_resolve_to_pm01_bodies(self):
        adapter = self.adapter
        names = adapter.body_names
        self.assertEqual(names[adapter.anchor_id], "LINK_TORSO_YAW")
        self.assertEqual(names[adapter.pelvis_id], "LINK_BASE")
        self.assertEqual(tuple(names[i] for i in adapter.foot_ids), PM01.foot_bodies)
        self.assertEqual(tuple(names[i] for i in adapter.amp_ids), PM01.amp_body_names)
        # Feet, hand ends and the welded head are bodies without a hinge.
        jointless = {names[i] for i, joint in enumerate(adapter.joint_indices) if joint < 0}
        self.assertEqual(jointless, {"LINK_BASE", "LINK_FOOT_L", "LINK_FOOT_R", "LINK_ELBOW_END_L",
                                     "LINK_ELBOW_END_R", "LINK_HEAD_YAW"})

    def test_fk_matches_mujoco(self):
        states = sample_state()
        p, q, v, w = self.adapter.analyze_state(states)
        for i, state in enumerate(states):
            _, positions, quaternions, linear, angular = mujoco_bodies(self.model, state)
            np.testing.assert_allclose(p[i].numpy(), positions, atol=3e-7)
            alignment = np.abs(np.sum(q[i].numpy() * quaternions, axis=-1))
            self.assertLess(float(np.max(1 - alignment)), 1e-10)
            np.testing.assert_allclose(v[i].numpy(), linear, atol=3e-6)
            np.testing.assert_allclose(w[i].numpy(), angular, atol=3e-6)

    def test_grounded_features_use_the_pm01_sole_point_and_match_the_expert_builder(self):
        adapter = self.adapter
        states = sample_state()
        kinematics = adapter.analyze_state(states)
        features = adapter.amp_features(states, kinematics=kinematics)
        self.assertEqual(features.shape, (len(states), 203))
        expert = build_amp_features(*kinematics, body_ids=adapter.amp_ids, anchor_id=adapter.anchor_id,
                                    pelvis_id=adapter.pelvis_id, foot_ids=adapter.foot_ids, schema="grounded_v2",
                                    foot_local_offsets=PM01.foot_reference_offsets)
        torch.testing.assert_close(features, expert, rtol=0, atol=0)
        default = build_amp_features(*kinematics, body_ids=adapter.amp_ids, anchor_id=adapter.anchor_id,
                                     pelvis_id=adapter.pelvis_id, foot_ids=adapter.foot_ids, schema="grounded_v2")
        self.assertGreater(float((features[:, 199:203] - default[:, 199:203]).abs().max()), 1e-3)
        # The foot heights are those of the MuJoCo foot sites.
        for i, state in enumerate(states):
            data, *_ = mujoco_bodies(self.model, state)
            sites = [data.site_xpos[self.model.site(name).id][2] for name in PM01.foot_sites]
            np.testing.assert_allclose(features[i, 199:201].numpy(), sites, atol=3e-7)
        metadata = amp_feature_metadata("grounded_v2", 13, PM01.foot_reference_offsets)
        self.assertEqual(metadata["foot_reference_local_offset"], [list(o) for o in PM01.foot_reference_offsets])
        self.assertEqual(amp_feature_metadata("grounded_v2")["foot_reference_local_offset"], [0.04, 0.0, -0.035])

    def test_actor_and_critic_feature_widths(self):
        adapter = self.adapter
        state = sample_state(3)
        action = torch.zeros(3, JOINTS, dtype=torch.float64)
        command = torch.zeros(3, 3, dtype=torch.float64)
        context = torch.zeros(3, CONTEXT, dtype=torch.float64)
        self.assertEqual(adapter.actor_frame(state, action, command, context).shape, (3, FRAME))
        # height 1, orientation 4, velocities 6, joints 2n, previous action n, command 3, context n + 6
        self.assertEqual(adapter.critic_features(state, action, command, context).shape, (3, 11 + 3 * JOINTS + 3 + CONTEXT))

    def test_recovery_clock_reads_the_last_two_context_columns(self):
        adapter = self.adapter
        state = sample_state(4)
        state[:, 3:7] = state.new_tensor([1., 0., 0., 0.])
        state[:, 2] = torch.tensor([.3, .3, .9, .3], dtype=torch.float64)
        context = torch.zeros(4, CONTEXT, dtype=torch.float64)
        context[:, -2] = torch.tensor([1., 1., 1., 0.])  # delay flag
        context[:, -1] = torch.tensor([3., 249., 5., 0.])  # consecutive bad steps
        following, terminated = adapter.advance_context(state, context)
        torch.testing.assert_close(following[:, -1], torch.tensor([4., 0., 0., 0.], dtype=torch.float64))
        torch.testing.assert_close(following[:, :-1], context[:, :-1])
        self.assertEqual(terminated[1].item(), 1.)  # the delay ran out
        self.assertEqual(terminated[0].item(), 0.)
        self.assertGreater(terminated[3].item(), .99)  # an ordinary environment below 0.52 m
        mask = recovery_foot_slip_mask(state, context, adapter.recovery_height)
        self.assertEqual(mask.tolist(), [True, True, True, False])
        # Between the two robots' recovery heights only the PM01 threshold counts it as fallen.
        state[:, 2] = .56
        context[:, -1] = 0.
        self.assertTrue(recovery_foot_slip_mask(state, context, 0.57)[:3].all())
        self.assertFalse(recovery_foot_slip_mask(state, context).any())

    def test_task_reward_is_finite_and_differentiable(self):
        adapter = self.adapter
        state = sample_state(5).requires_grad_(True)
        following = sample_state(5) + .01
        following = torch.cat((following[:, :3], quat_normalize(following[:, 3:7]), following[:, 7:]), -1)
        action = torch.full((5, JOINTS), .03, dtype=torch.float64, requires_grad=True)
        context = torch.zeros(5, CONTEXT, dtype=torch.float64)
        context[:, -2], context[:, -1] = 1., .6
        reward = adapter.task_reward(state, following, action, torch.zeros_like(action),
                                     torch.full((5, 3), .05, dtype=torch.float64), context,
                                     torch.zeros(5, dtype=torch.float64), torch.zeros(5, 2, dtype=torch.float64))
        self.assertEqual(reward.shape, (5,))
        gradients = torch.autograd.grad(reward.sum(), (state, action))
        self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
        self.assertGreater(float(gradients[0][:, 36:].abs().max()), 0.)  # joint acceleration term


class Pm01WorldModelTest(unittest.TestCase):
    def test_world_model_sizes_and_zero_increment_identity(self):
        model = WorldModel(context_dim=CONTEXT, hidden=(32, 32), num_joints=JOINTS).double()
        self.assertEqual(model.in_dim, STATE + JOINTS + CONTEXT)
        self.assertEqual(model.net[-1].out_features, 12 + 2 * JOINTS)
        self.assertEqual(model.error_floor.shape, (12 + 2 * JOINTS,))
        state = sample_state(4)
        action = torch.zeros(4, JOINTS, dtype=torch.float64)
        context = torch.zeros(4, CONTEXT, dtype=torch.float64)
        self.assertEqual(invariant_features(state).shape, (4, STATE))
        self.assertEqual(model.inputs(state, action, context).shape, (4, model.in_dim))
        # Zero-initialised output and unit normaliser: the model predicts no change.
        torch.testing.assert_close(model.step(state, action, context), state, atol=1e-12, rtol=0)
        following, contact = model.step_with_contact(state, action, context)
        self.assertEqual((following.shape, contact.shape), ((4, STATE), (4, 2)))

    def test_targets_and_integration_are_inverse_for_23_joints(self):
        model = WorldModel(context_dim=CONTEXT, hidden=(8,), num_joints=JOINTS).double()
        state, following = sample_state(4), sample_state(4) * 1.1
        following = torch.cat((following[:, :3], quat_normalize(following[:, 3:7]), following[:, 7:]), -1)
        increment = model.targets(state, following)
        self.assertEqual(increment.shape, (4, 12 + 2 * JOINTS))
        restored = model.integrate(state, (increment - model.out_norm.mean) / model.out_norm.std())
        torch.testing.assert_close(restored[:, :3], following[:, :3], atol=1e-9, rtol=0)
        torch.testing.assert_close(restored[:, 7:], following[:, 7:], atol=1e-9, rtol=0)
        self.assertLess(float((1 - (restored[:, 3:7] * following[:, 3:7]).sum(-1).abs()).max()), 1e-9)

    def test_default_world_model_is_still_the_g1_size(self):
        model = WorldModel(hidden=(8,))
        self.assertEqual((model.in_dim, model.net[-1].out_features, model.num_joints), (135, 70, 29))


class Pm01SmoothRewardTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from src.tasks.amp_loco.config.pm01.env_cfgs import pm01_amp_flat_env_cfg
        cls.terms = SmoothRewardTerms.from_config(pm01_amp_flat_env_cfg(), pm01.PM01_JOINT_NAMES).double()

    def test_constants_are_read_from_the_pm01_actuators(self):
        terms = self.terms
        self.assertEqual(terms.num_joints, JOINTS)
        names = pm01.PM01_JOINT_NAMES
        knee, ankle, elbow = names.index("J03_KNEE_PITCH_L"), names.index("J04_ANKLE_PITCH_L"), names.index("J16_ELBOW_PITCH_L")
        self.assertAlmostEqual(float(terms.kp[knee]), pm01.STIFFNESS_Q90, places=4)
        self.assertAlmostEqual(float(terms.kd[knee]), pm01.DAMPING_Q90, places=4)
        self.assertAlmostEqual(float(terms.kd[ankle]), pm01.DAMPING_ANKLE, places=6)
        self.assertAlmostEqual(float(terms.kd[elbow]), pm01.DAMPING_Q25, places=5)
        self.assertEqual((float(terms.effort_limit[knee]), float(terms.effort_limit[elbow])), (164., 52.))
        np.testing.assert_allclose((float(terms.x1[knee]), float(terms.x2[knee]), float(terms.x1[ankle])),
                                   (26.3, 26.3, 35.2), rtol=1e-6)
        self.assertLess(float((terms.friction_static - .8).abs().max()), 1e-6)
        self.assertLess(float((terms.friction_dynamic - .6).abs().max()), 1e-6)
        self.assertAlmostEqual(float(terms.action_scale[knee]), 0.25 * 164. / pm01.STIFFNESS_Q90, places=5)
        self.assertAlmostEqual(float(terms.action_offset[knee]), 0.12, places=6)
        self.assertAlmostEqual(float(terms.action_offset[names.index("J19_SHOULDER_ROLL_R")]), -0.15, places=6)
        self.assertEqual(len(terms.metadata()["joint_names"]), JOINTS)

    def test_motor_proxy_equals_the_actuator_and_cuts_torque_above_the_speed_limit(self):
        terms = self.terms
        generator = torch.Generator().manual_seed(7)
        state = torch.randn(3, STATE, dtype=torch.float64, generator=generator) * .1
        state[:, 36:] = torch.linspace(-40., 40., JOINTS)  # beyond both speed limits at the ends
        action = torch.linspace(-30., 30., JOINTS, dtype=torch.float64).expand(3, -1).clone()
        context = torch.randn(3, CONTEXT, dtype=torch.float64, generator=generator) * .001
        request, clipped, command = terms.motor_effort(state, action, context)
        velocity = state[:, 36:]
        above = velocity.abs() >= terms.x1
        self.assertTrue(above.any() and (~above).any())
        self.assertTrue((clipped[above] == 0).all())
        self.assertTrue((clipped.abs() <= terms.effort_limit + 1e-12).all())
        self.assertTrue(((clipped.abs() - terms.effort_limit).abs() < 1e-9).any())  # torque saturation occurs
        # The same numbers through the actuator class the simulator runs, one group at a time.
        for cfg in pm01.PM01_ARTICULATION.actuators:
            self.assertIsInstance(cfg, Pm01ActuatorCfg)
            ids = [i for i, name in enumerate(pm01.PM01_JOINT_NAMES)
                   if any(re.fullmatch(p, name) for p in cfg.target_names_expr)]
            native = object.__new__(UnitreeActuator)
            native.cfg = cfg
            shape = (3, len(ids))
            for attr, value in (("_effort_y1", cfg.Y1), ("_effort_y2", cfg.Y2), ("_velocity_x1", cfg.X1),
                                ("_velocity_x2", cfg.X2), ("_friction_static", cfg.Fs),
                                ("_friction_dynamic", cfg.Fd), ("_activation_vel", cfg.Va)):
                setattr(native, attr, torch.full(shape, value, dtype=torch.float64))
            native._joint_vel = torch.zeros(shape, dtype=torch.float64)
            target = (terms.action_offset + terms.action_scale * action - context[:, 1:1 + JOINTS])[:, ids]
            cmd = ActuatorCmd(position_target=target, velocity_target=torch.zeros(shape, dtype=torch.float64),
                              effort_target=torch.zeros(shape, dtype=torch.float64),
                              pos=state[:, 13:36][:, ids], vel=velocity[:, ids])
            control = native.compute(cmd)
            reconstructed = cfg.stiffness * (control - cmd.pos) - cfg.damping * cmd.vel
            # The proxy stores the configuration's constants in single precision.
            torch.testing.assert_close(command[:, ids], reconstructed, rtol=1e-6, atol=1e-5)

    def test_costs_have_one_value_per_sample_and_use_all_23_joints(self):
        terms = self.terms
        state, following = sample_state(4), sample_state(4)
        action = torch.ones(4, JOINTS, dtype=torch.float64)
        context = torch.zeros(4, CONTEXT, dtype=torch.float64)
        costs = terms(state, following, action, torch.zeros_like(action), torch.zeros_like(action), context)
        self.assertEqual({name: tuple(value.shape) for name, value in costs.items()},
                         {name: (4,) for name in ("action_second_difference", "joint_velocity",
                                                  "requested_torque", "commanded_power")})
        torch.testing.assert_close(costs["action_second_difference"], torch.full((4,), float(JOINTS), dtype=torch.float64))
        torch.testing.assert_close(costs["joint_velocity"], following[:, 36:].square().sum(-1))

    def test_previous_previous_action_is_read_from_78_wide_frames(self):
        history = torch.arange(4 * FRAME, dtype=torch.float32).reshape(1, 4 * FRAME)
        older = previous_previous_action(history)
        torch.testing.assert_close(older, history.reshape(1, 4, FRAME)[:, 2, FRAME - JOINTS:])
        g1 = torch.arange(384, dtype=torch.float32).reshape(1, 384)
        torch.testing.assert_close(previous_previous_action(g1), g1.reshape(1, 4, 96)[:, 2, 67:96])


class Pm01SoleGeometryTest(unittest.TestCase):
    def test_sole_boxes_give_the_mujoco_clearance_and_site_velocity(self):
        adapter, model = make_pm01_adapter()
        geometry = FootSlipGeometry.from_model(model, adapter.body_ids, adapter.foot_ids, contact_height_m=0.,
                                               softness_m=.001, dtype=torch.float64, site_names=PM01.foot_sites)
        # Two boxes per foot, eight corners each, no radius.
        self.assertEqual(geometry.endpoints.shape, (2, 16, 3))
        self.assertFalse(geometry.radii.any())
        np.testing.assert_allclose(geometry.site_offsets.numpy(), PM01.foot_reference_offsets, atol=1e-9)
        states = sample_state()
        clearance, xy_velocity = geometry.kinematic_terms(adapter.analyze_state(states))
        for i, state in enumerate(states):
            data, *_ = mujoco_bodies(model, state)
            for foot, side in enumerate(("left", "right")):
                lowest = np.inf
                for name in (f"{side}_foot_sole_collision", f"{side}_foot_toe_collision"):
                    geom = model.geom(name).id
                    rotation = data.geom_xmat[geom].reshape(3, 3)
                    lowest = min(lowest, data.geom_xpos[geom][2] - float(np.abs(rotation[2]) @ model.geom_size[geom]))
                self.assertAlmostEqual(float(clearance[i, foot]), lowest, places=6)
                velocity = np.zeros(6)
                mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_SITE,
                                         model.site(PM01.foot_sites[foot]).id, velocity, 0)
                np.testing.assert_allclose(xy_velocity[i, foot].numpy(), velocity[3:5], atol=3e-6)

    def test_adapter_configures_the_geometry_on_a_flat_pm01_task(self):
        adapter, _ = make_pm01_adapter()
        adapter.env.cfg.scene = SimpleNamespace(terrain=SimpleNamespace(terrain_type="plane"))
        adapter.reward_cfg["foot_slip"].params = dict(command_name="twist", command_threshold=.1,
                                                      asset_cfg=SimpleNamespace(site_names=PM01.foot_sites))
        adapter.configure_recovery_foot_slip(contact_height_m=.001, softness_m=.001)
        self.assertIsInstance(adapter.recovery_foot_slip, FootSlipGeometry)
        adapter.reward_cfg["foot_slip"].params["asset_cfg"] = SimpleNamespace(site_names=("a", "b"))
        with self.assertRaisesRegex(ValueError, "original PM01"):
            adapter.configure_recovery_foot_slip(contact_height_m=.001, softness_m=.001)


class Pm01ConfigurationTest(unittest.TestCase):
    def test_pm01_flat_tasks_validate_with_the_matching_schema(self):
        cfg = WmacConfig.for_profile("supermimic_official")
        cfg.task = "EngineAI-PM01-AMP-Flat-Aligned"
        cfg.validate()
        cfg.task, cfg.amp_feature_schema = "EngineAI-PM01-AMP-Flat", "legacy"
        cfg.validate()
        cfg.amp_feature_schema = "grounded_v2"
        with self.assertRaisesRegex(ValueError, "grounded_v2"):
            cfg.validate()
        cfg.task = "EngineAI-PM01-AMP-Rough"
        with self.assertRaisesRegex(ValueError, "flat ground"):
            cfg.validate()

    def test_mirrored_cache_follows_the_robot_unless_chosen_explicitly(self):
        cfg = WmacConfig.for_profile("supermimic_official")
        cfg.resolve_robot_defaults()
        self.assertEqual(cfg.motion_cache_root, ".wmac_cache/motions/g1_mirrored")
        cfg.task = "EngineAI-PM01-AMP-Flat-Aligned"
        cfg.resolve_robot_defaults()
        self.assertEqual(cfg.motion_cache_root, ".wmac_cache/motions/pm01_mirrored")
        cfg.motion_cache_root = "/somewhere/else"
        cfg.resolve_robot_defaults()
        self.assertEqual(cfg.motion_cache_root, "/somewhere/else")

    def test_geometric_foot_slip_is_allowed_on_flat_pm01(self):
        from src.wmac.reference import imagined_foot_slip_protocol, IMAGINED_FOOT_SLIP_INSTANTAMP
        protocol = imagined_foot_slip_protocol({
            "imagined_recovery_foot_slip": IMAGINED_FOOT_SLIP_INSTANTAMP,
            "task": "EngineAI-PM01-AMP-Flat-Aligned", "recovery_counter_semantics": "hard_forward_v2"})
        self.assertEqual(protocol["recovery_foot_slip_contact_height_m"], .001)

    def test_replay_stratifies_pm01_states_by_its_own_height_and_counter_column(self):
        replay = ReplayBuffer(4, 6, "cpu", recovery_height=PM01.recovery_height)
        state = torch.zeros(6, STATE)
        state[:, 2] = torch.tensor([.9, .56, .9, .9, .58, .9])
        context = torch.zeros(6, CONTEXT)
        context[2, -1] = 3.  # active recovery clock
        context[3, 20] = 5.  # an encoder-bias column must not count
        batch = {"state": state, "next_state": state.clone(), "context": context, "next_context": context.clone(),
                 "action": torch.zeros(6, JOINTS), "previous_action": torch.zeros(6, JOINTS),
                 "command": torch.zeros(6, 3), "actor_history": torch.zeros(6, 4 * FRAME),
                 "reward": torch.zeros(6), "terminated": torch.zeros(6, dtype=torch.bool),
                 "timeout": torch.zeros(6, dtype=torch.bool), "boundary": torch.zeros(6, dtype=torch.bool),
                 "contact_cost": torch.zeros(6, 2)}
        replay.add(batch)
        self.assertEqual(replay._recovery[0].tolist(), [False, True, True, False, False, False])


if __name__ == "__main__":
    unittest.main()
