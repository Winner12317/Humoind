"""Pin the ordinary G1 topology and joint convention used on unshod hardware."""
from __future__ import annotations

import hashlib
import struct
import unittest
from types import SimpleNamespace

import mujoco
import numpy as np
import torch
from mjlab.entity import Entity

from src.tasks.amp_perceptive.env_cfg import g1_amp_perceptive_hiking_env_cfg
from src.tasks.amp_perceptive.commands import FlatPatchVelocityCommand
from src.tasks.amp_perceptive.hiking_robot import (
    HIKING_HEAD, HIKING_URDF, get_hiking_robot_spec, source_hiking_model,
)
from src.wmac_perceptive.hiking_materials import hiking_material_geom_names, sample_hiking_material_buckets


def build_config(version, *, play=False):
    return g1_amp_perceptive_hiking_env_cfg(
        play=play, sensor_protocol="hiking_depth_only_v1", motion_data_protocol="hiking_walk_v1",
        locomotion_objective="hiking_full_v1", hiking_training_version=version,
    )


class HikingReferenceRobotTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = source_hiking_model()
        cls.local = get_hiking_robot_spec().compile()

    def test_pinned_raw_assets(self):
        self.assertEqual(hashlib.sha256(HIKING_URDF.read_bytes()).hexdigest(),
                         "5e99d930af64f3f0bfd34235239e288cc380ba62c8b01fcfcf8cc9f0996e7fa4")
        self.assertEqual(hashlib.sha256(HIKING_HEAD.read_bytes()).hexdigest(),
                         "d297328b3569fc85074355e76cba6c889d19de5361224bcd3c5b8ce82e13b9b6")
        triangles = struct.unpack_from("<I", HIKING_HEAD.read_bytes(), 80)[0]
        self.assertEqual(int(self.local.mesh("head_link").facenum[0]), triangles)

    def test_urdf_and_gen_coord_frames_match_on_every_link(self):
        # Compare TWO independent kinematic trees with opposite root direction.
        # A waist axis sign error is invisible at the nominal zero waist pose.
        original, local = self.original, self.local
        source_data, local_data = mujoco.MjData(original), mujoco.MjData(local)
        rng = np.random.default_rng(74)
        for random_pose in (False, True, True, True):
            source_data.qpos[:3] = rng.normal(size=3)
            source_data.qvel[:] = rng.normal(size=original.nv)
            quat = rng.normal(size=4)
            source_data.qpos[3:7] = quat / np.linalg.norm(quat)
            for j in range(1, original.njnt):
                angle = rng.uniform(*original.jnt_range[j]) if random_pose else 0.
                src = original.joint(j)
                dest = local.joint(src.name)
                np.testing.assert_allclose(src.range, dest.range)
                source_data.qpos[src.qposadr] = local_data.qpos[dest.qposadr] = angle
                local_data.qvel[dest.dofadr] = source_data.qvel[src.dofadr]
            mujoco.mj_forward(original, source_data)
            local_data.qpos[:3] = source_data.body("pelvis").xpos
            local_data.qpos[3:7] = source_data.body("pelvis").xquat
            root_linear, root_angular = self._link_twist(original, source_data, "pelvis")
            local_data.qvel[:3] = root_linear
            local_data.qvel[3:6] = source_data.body("pelvis").xmat.reshape(3, 3).T @ root_angular
            mujoco.mj_forward(local, local_data)
            for i in range(1, original.nbody):
                name = original.body(i).name
                np.testing.assert_allclose(original.body(name).mass, local.body(name).mass)
                np.testing.assert_allclose(original.body(name).inertia, local.body(name).inertia)
                for field in ("xpos", "xmat", "xipos", "ximat"):
                    np.testing.assert_allclose(getattr(source_data.body(name), field),
                                               getattr(local_data.body(name), field), atol=1e-12)
                np.testing.assert_allclose(self._link_twist(original, source_data, name),
                                           self._link_twist(local, local_data, name), atol=1e-12)
                collision = [g for g in range(*self._geom_range(local.body(name)))
                             if local.geom_contype[g] | local.geom_conaffinity[g]]
                native = range(*self._geom_range(original.body(name)))
                self.assertEqual(len(native), len(collision))
                for a, b in zip(native, collision):
                    kind = original.geom_type[a]
                    if kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
                        kind = mujoco.mjtGeom.mjGEOM_CAPSULE
                    self.assertEqual(int(kind), int(local.geom_type[b]))
                    np.testing.assert_allclose(original.geom_size[a], local.geom_size[b])
                    np.testing.assert_allclose(source_data.geom_xpos[a], local_data.geom_xpos[b], atol=1e-12)
                    np.testing.assert_allclose(source_data.geom_xmat[a], local_data.geom_xmat[b], atol=1e-12)
        self.assertEqual(local.nexclude, 0)
        self.assertEqual(original.ngeom, 29)

    @staticmethod
    def _link_twist(model, data, name):
        position_jacobian = np.empty((3, model.nv))
        rotation_jacobian = np.empty_like(position_jacobian)
        mujoco.mj_jacBody(model, data, position_jacobian, rotation_jacobian, model.body(name).id)
        return position_jacobian @ data.qvel, rotation_jacobian @ data.qvel

    @staticmethod
    def _geom_range(body):
        start = int(body.geomadr[0])
        return start, start + int(body.geomnum[0])

    def test_command_partial_resamples_match_upstream_draws_and_timing(self):
        # Golden values from running de6b92f PoseVelocityCommand._resample_command
        # with these terrain types, seed and three differently ordered batches.
        n = 8
        terrain = SimpleNamespace(
            flat_patches={"target": torch.arange(72).reshape(2, 4, 3, 3).float()},
            terrain_levels=torch.arange(n) % 2, terrain_types=torch.arange(n) % 4)
        turn = torch.arange(n) % 4 == 1
        env = SimpleNamespace(cfg=SimpleNamespace(_hiking_training_version=3),
                              scene=SimpleNamespace(terrain=terrain))
        cfg = SimpleNamespace(task_protocol="hiking_full_v1", rel_standing_envs=.05,
                              ranges=SimpleNamespace(ang_vel_z=(-1., 1.)))
        cmd = SimpleNamespace(_env=env, device="cpu", cfg=cfg, target_pos_w=torch.zeros(n, 3),
                              vel_command_b=torch.ones(n, 3), random_velocity_mask=turn,
                              desired_speed=torch.zeros(n), random_yaw_rate=torch.zeros(n),
                              _hiking_speed_low=.45*(~turn), _hiking_speed_high=.8*(~turn))
        for name in ("is_standing_env", "is_heading_env", "is_in_place_turn", "goal_is_traversal"):
            setattr(cmd, name, torch.zeros(n, dtype=torch.bool))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(123)
            for ids in (torch.arange(n), torch.tensor([5, 2, 0, 1]), torch.tensor([0, 3])):
                FlatPatchVelocityCommand._resample_command(cmd, ids)
            torch.testing.assert_close(torch.rand(2), torch.tensor([.5983095169, .4526599646]))
        torch.testing.assert_close(cmd.desired_speed, torch.tensor(
            [.6976691484, 0., .6630293131, .6289302111, .476472348, 0., .5607441664, .5906090736]))
        torch.testing.assert_close(cmd.random_yaw_rate,
                                   torch.tensor([0., .8777432442, 0., 0., 0., .6112383604, 0., 0.]))
        torch.testing.assert_close(cmd.target_pos_w[:, 0], torch.tensor([3., 45., 24., 66., 3., 48., 18., 69.]))
        self.assertEqual(cmd.is_standing_env.nonzero(as_tuple=True)[0].tolist(), [5])
        torch.testing.assert_close(cmd.vel_command_b, torch.ones(n, 3))

    def test_play_and_training_builders_select_source_model(self):
        cfg, play = build_config(3), build_config(3, play=True)
        self.assertIs(cfg.scene.entities["robot"].spec_fn, get_hiking_robot_spec)
        self.assertEqual(cfg.scene.env_spacing, 2.5)
        self.assertEqual(len(cfg.rewards), 26)
        self.assertEqual({sensor.name for sensor in cfg.scene.sensors}, {"hiking_depth", "hiking_contact"})
        self.assertEqual((cfg.decimation, cfg.sim.mujoco.timestep), (4, .005))
        self.assertEqual(len(hiking_material_geom_names(3)), 29)
        self.assertEqual(len(hiking_material_geom_names(2)), 33)
        model = Entity(play.scene.entities["robot"]).spec.compile()
        colliding = (model.geom_contype | model.geom_conaffinity) != 0
        np.testing.assert_allclose(model.geom_friction[colliding, 0], 1.)
        self.assertNotIn("physics_material", play.events)
        self.assertNotIn("bad_base_height", play.terminations)
        self.assertIn("terrain_levels", play.curriculum)
        self.assertTrue(play.observations["actor"].enable_corruption)
        self.assertIn("bad_base_height", cfg.terminations)
        self.assertIn("physics_material", cfg.events)
        self.assertEqual(play.scene.num_envs, 10)
        self.assertEqual(play.episode_length_s, 10.)
        terrain = play.scene.terrain.terrain_generator
        self.assertEqual((terrain.num_rows, terrain.num_cols), (4, 10))
        self.assertTrue(all(t.wall_probability == 0. for t in terrain.sub_terrains.values()))
        self.assertIsNot(build_config(2).scene.entities["robot"].spec_fn, get_hiking_robot_spec)

    def test_material_samples_follow_isaaclab_cpu_matrix_draw(self):
        ranges = torch.tensor([[.3, 1.6], [.3, 1.6], [.05, .5]])
        generator = torch.Generator().manual_seed(59)
        expected = ranges[:, 0] + torch.rand((64, 3), generator=generator) * (ranges[:, 1] - ranges[:, 0])
        expected[:, 1] = expected[:, 0:2].amin(dim=-1)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(59)
            actual = sample_hiking_material_buckets(hiking_training_version=3)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
