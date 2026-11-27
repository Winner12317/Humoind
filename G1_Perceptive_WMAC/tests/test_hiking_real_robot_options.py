"""Real-robot options on top of Hiking training version 4.

* ``collision_horizontal_scale=0.025``: the collision heightfield samples the
  upper surface of Hiking's vertical-riser mesh on a 2.5 cm grid.
* ``depth_geometry="source_mesh"``: the depth camera sees that mesh itself.
* ``RealSenseDepthAdapter``: raw RealSense image -> the training camera's frame.
"""
from __future__ import annotations

import json
import math
import unittest

import mujoco
import numpy as np
import torch

from mjlab.terrains.terrain_generator import TerrainGenerator

from src.tasks.amp_perceptive.hiking_geometry import isaaclab_height_field_mesh, source_mesh_height_grid
from src.tasks.amp_perceptive.hiking_terrain_source import source_height_array
from src.tasks.amp_perceptive.sensors import HikingDepthPatternCfg
from src.tasks.amp_perceptive.terrain import (
    HIKING_COLLISION_ONLY_GROUP, TerrainHeightField, hiking_terrain_generator,
)
from src.wmac_perceptive import unitree_interface as ui
from src.wmac_perceptive.camera_calibration import load_g1_depth_calibration
from src.wmac_perceptive.config import HIKING_SENSOR_PROTOCOL, PerceptiveConfig, saved_training_config

HFIELD, MESH, BOX = (int(kind) for kind in (mujoco.mjtGeom.mjGEOM_HFIELD, mujoco.mjtGeom.mjGEOM_MESH,
                                             mujoco.mjtGeom.mjGEOM_BOX))


def group_mask(*groups):
    mask = np.zeros(6, np.uint8)
    mask[list(groups)] = 1
    return mask


def downward_heights(model, data, points, groups):
    """Height and geom type of the first surface below each xy point (NaN where nothing is hit)."""
    geom = np.zeros(1, np.int32)
    heights, kinds = np.full(len(points), np.nan), np.full(len(points), -1)
    for i, (x, y) in enumerate(points):
        distance = mujoco.mj_ray(model, data, np.array([x, y, 30.]), np.array([0., 0., -1.]), groups, 1, -1, geom)
        if distance >= 0:
            heights[i], kinds[i] = 30. - distance, model.geom_type[geom[0]]
    return heights, kinds


class SourceMeshHeightGridTest(unittest.TestCase):
    def test_a_riser_keeps_its_plateau_and_drops_within_one_fine_cell(self):
        grid = np.zeros((12, 12), dtype=np.int64)
        grid[:7] = 30                                   # 0.15 m plateau up to x index 6 (x = 0.30 m)
        vertices, triangles = isaaclab_height_field_mesh(grid)
        # IsaacLab also slides the displaced vertices one cell along the riser,
        # which leaves a sliver at the tile border uncovered (hence the fallback).
        fallback = np.repeat(np.repeat(grid, 2, 0), 2, 1)[:23, :23] * .005
        fine = source_mesh_height_grid(vertices, triangles, (23, 23), .025, fallback)[:, :20]
        np.testing.assert_allclose(fine[:13], .15, atol=1e-7)       # x <= 0.300 m
        np.testing.assert_allclose(fine[13:], 0., atol=1e-7)        # x >= 0.325 m
        # The 5 cm heightfield would still be at half height there.
        self.assertAlmostEqual(float(np.interp(.325, [.30, .35], [.15, 0.])), .075)

    def test_grid_equals_the_mesh_surface_on_generated_terrain(self):
        for family in ("pyramid_stairs", "square_gaps", "boxes", "pyramid_stairs_inv_high"):
            with self.subTest(family=family):
                height, _ = source_height_array(family, .8, np.random.default_rng(3))
                grid = np.rint(height.T / .005).astype(np.int64)
                vertices, triangles = isaaclab_height_field_mesh(grid)
                fine = source_mesh_height_grid(vertices, triangles, (321, 321), .025)
                # Unchanged at every source sample that IsaacLab left in place,
                # so plateaus keep their source extent.
                nominal = np.stack(np.meshgrid(np.arange(161) * .05, np.arange(161) * .05, indexing="ij"), -1)
                displaced = (np.abs(vertices[:, :2].reshape(161, 161, 2) - nominal) > 1e-4).any(-1)
                self.assertGreater(int(displaced.sum()), 100)
                np.testing.assert_allclose(fine[::2, ::2][~displaced], (grid * .005)[~displaced], atol=1e-6)
                spec = mujoco.MjSpec()
                spec.add_mesh(name="tile", uservert=vertices.astype(np.float64).flatten().tolist(),
                              userface=triangles.flatten().tolist(), inertia=mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL)
                spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname="tile", contype=0, conaffinity=0)
                model = spec.compile()
                data = mujoco.MjData(model)
                mujoco.mj_forward(model, data)
                rng = np.random.default_rng(0)
                ix, iy = rng.integers(1, 320, 1500), rng.integers(1, 320, 1500)
                # A sample on a riser takes its top: the highest surface just around the sample.
                best = np.full(len(ix), -np.inf)
                for dx, dy in ((1e-4, 1.3e-4), (-1e-4, 1.3e-4), (1e-4, -1.3e-4), (-1e-4, -1.3e-4)):
                    heights, _ = downward_heights(model, data, np.stack((ix * .025 + dx, iy * .025 + dy), -1), None)
                    best = np.fmax(best, heights)
                np.testing.assert_allclose(fine[ix, iy], best, atol=1e-3)

    def test_uncovered_samples_need_a_fallback(self):
        vertices = np.array([[0, 0, 0], [.05, 0, 0], [0, .05, 0]], dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "does not cover"):
            source_mesh_height_grid(vertices, np.array([[0, 1, 2]]), (3, 3), .025)
        filled = source_mesh_height_grid(vertices, np.array([[0, 1, 2]]), (3, 3), .025, np.full((3, 3), 7.))
        self.assertEqual(filled[2, 2], 7.)
        self.assertEqual(filled[0, 0], 0.)


class RealRobotTerrainTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models = {}
        for key, options in (("base", {}), ("refined", {"collision_horizontal_scale": .025}),
                             ("both", {"collision_horizontal_scale": .025, "depth_source_mesh": True})):
            spec = mujoco.MjSpec()
            TerrainGenerator(hiking_terrain_generator(rows=2, cols=20, task_protocol="hiking_full_v1",
                                                      **options)).compile(spec)
            cls.models[key] = spec.compile()
        cls.terrains = {key: TerrainHeightField.from_model(model, source_mesh_edges=True)
                        for key, model in cls.models.items()}
        rng = np.random.default_rng(0)
        low, high = cls.terrains["base"].grid_min.numpy(), cls.terrains["base"].grid_max.numpy()
        cls.points = rng.uniform(low + .01, high - .01, (6000, 2))

    def test_edges_do_not_depend_on_the_collision_grid(self):
        segments = {key: terrain.hiking_edges.edges[:terrain.hiking_edges.edge_count].numpy()
                    for key, terrain in self.terrains.items()}
        self.assertGreater(len(segments["base"]), 1000)
        np.testing.assert_array_equal(segments["base"], segments["refined"])
        np.testing.assert_array_equal(segments["base"], segments["both"])

    def test_grid_sizes_and_stored_source_heights(self):
        self.assertEqual(tuple(self.terrains["base"].heights.shape), (40, 161, 161))
        self.assertEqual(tuple(self.terrains["both"].heights.shape), (40, 321, 321))
        model = self.models["both"]
        # Every tile except the flat mesh-box ground keeps its 161 x 161 source integers.
        names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_HFIELD, i) for i in range(model.nhfield)]
        stored = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_NUMERIC, name) >= 0 for name in names]
        self.assertEqual(stored, [not name.startswith("hiking_full_mesh_boxes_") for name in names])
        self.assertTrue(all(model.numeric_size[i] == 161 * 161 for i in range(model.nnumeric)))
        self.assertEqual(self.models["base"].nnumeric, 0)

    def test_depth_geometry_is_seen_but_never_touched(self):
        model, base = self.models["both"], self.models["base"]
        kinds, groups = model.geom_type, model.geom_group
        colliding = (model.geom_contype != 0) | (model.geom_conaffinity != 0)
        self.assertTrue((groups[kinds == HFIELD] == HIKING_COLLISION_ONLY_GROUP).all())
        self.assertTrue(colliding[kinds == HFIELD].all())
        self.assertTrue((groups[kinds == MESH] == 0).all())
        self.assertFalse(colliding[kinds == MESH].any())
        # One depth surface per tile: a mesh, or a flat box where the ground is level.
        depth_only = ~colliding
        self.assertEqual(int(depth_only.sum()), 40)
        self.assertEqual(int((kinds[depth_only] == BOX).sum()) + int((kinds == MESH).sum()), 40)
        self.assertEqual(int((base.geom_group != 0).sum()), 0)
        # Walls, rotated boxes and the border are unchanged, and the depth-only
        # boxes are not terrain to stand on.
        self.assertEqual(int((kinds[colliding] == BOX).sum()), int((base.geom_type == BOX).sum()))
        self.assertEqual(len(self.terrains["both"].hiking_boxes.boxes), len(self.terrains["base"].hiking_boxes.boxes))

    def test_camera_sees_risers_and_refinement_halves_the_ramp(self):
        both, base = self.models["both"], self.models["base"]
        data_both, data_base = mujoco.MjData(both), mujoco.MjData(base)
        mujoco.mj_forward(both, data_both)
        mujoco.mj_forward(base, data_base)
        seen, kinds = downward_heights(both, data_both, self.points, group_mask(0))
        touched, _ = downward_heights(both, data_both, self.points, group_mask(HIKING_COLLISION_ONLY_GROUP))
        coarse, _ = downward_heights(base, data_base, self.points, group_mask(0))
        self.assertFalse(np.isnan(seen).any())
        self.assertNotIn(HFIELD, set(kinds.tolist()))
        on_mesh = kinds == MESH
        self.assertGreater(on_mesh.mean(), .5)
        # Collision never undercuts the seen surface, and departs from it over
        # about half the area the 5 cm heightfield does.
        self.assertGreater(float((touched - seen)[on_mesh].min()), -2e-3)
        fine_ramp = float(((touched - seen)[on_mesh] > .01).mean())
        coarse_ramp = float(((coarse - seen)[on_mesh] > .01).mean())
        self.assertGreater(coarse_ramp, .015)
        self.assertLess(fine_ramp, .65 * coarse_ramp)
        self.assertGreater(fine_ramp, .35 * coarse_ramp)

    def test_foot_height_queries_follow_the_refined_collision_surface(self):
        model = self.models["both"]
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        # Everything that collides: the refined heightfields plus walls, boxes and border.
        hfield, _ = downward_heights(model, data, self.points, group_mask(HIKING_COLLISION_ONLY_GROUP))
        seen, kinds = downward_heights(model, data, self.points, group_mask(0))
        colliding_box = (kinds == BOX) & np.array([
            bool(model.geom_contype[g] or model.geom_conaffinity[g]) for g in self._first_geoms(model, data, self.points)])
        expected = np.fmax(hfield, np.where(colliding_box, seen, -np.inf))
        starts = torch.tensor(np.c_[self.points, np.full(len(self.points), 30.)], dtype=torch.float32)
        np.testing.assert_allclose(self.terrains["both"].hiking_ray_height_at(starts).numpy(), expected, atol=2e-4)

    @staticmethod
    def _first_geoms(model, data, points):
        geom, result = np.zeros(1, np.int32), []
        for x, y in points:
            mujoco.mj_ray(model, data, np.array([x, y, 30.]), np.array([0., 0., -1.]), group_mask(0), 1, -1, geom)
            result.append(int(geom[0]))
        return result

    def test_depth_mesh_is_for_the_full_hiking_terrain_only(self):
        with self.assertRaisesRegex(ValueError, "full Hiking terrain"):
            hiking_terrain_generator(depth_source_mesh=True)


class RealRobotConfigurationTest(unittest.TestCase):
    def test_version_five_defaults_train_for_the_physical_robot(self):
        cfg = PerceptiveConfig.defaults("supermimic")
        self.assertEqual((cfg.hiking_training_version, cfg.collision_horizontal_scale, cfg.depth_geometry,
                          cfg.robot_dynamics_randomization, cfg.robot_variant, cfg.sensor_protocol, cfg.num_envs),
                         (5, .025, "source_mesh", "unitree_g1_v1", "unshod", HIKING_SENSOR_PROTOCOL, 4096))
        cfg.validate()
        aligned = PerceptiveConfig.defaults("supermimic", hiking_training_version=4)
        self.assertEqual((aligned.collision_horizontal_scale, aligned.depth_geometry,
                          aligned.robot_dynamics_randomization), (.05, "collision", "none"))
        aligned.validate()

    def test_options_validate_for_the_hiking_task_only(self):
        cfg = PerceptiveConfig.defaults("supermimic")
        cfg.collision_horizontal_scale, cfg.depth_geometry = .025, "source_mesh"
        cfg.validate()
        cfg.collision_horizontal_scale = .0125
        with self.assertRaisesRegex(ValueError, "collides on the original"):
            cfg.validate()
        # Version 3 policies can be evaluated on this geometry; earlier tasks cannot use it.
        version3 = PerceptiveConfig.defaults("supermimic", hiking_training_version=3)
        version3.collision_horizontal_scale, version3.depth_geometry = .025, "source_mesh"
        version3.validate()
        for field, value, message in (("collision_horizontal_scale", .025, "collides on the original"),
                                      ("depth_geometry", "source_mesh", "requires the Hiking task")):
            old = PerceptiveConfig.defaults("supermimic", hiking_training_version=2)
            setattr(old, field, value)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, message):
                old.validate()
        cfg = PerceptiveConfig.defaults("supermimic")
        cfg.depth_geometry = "heightfield"
        with self.assertRaisesRegex(ValueError, "depth_geometry must be one of"):
            cfg.validate()

    def test_earlier_checkpoints_keep_collision_depth(self):
        from dataclasses import asdict
        saved = asdict(PerceptiveConfig.defaults("supermimic", hiking_training_version=3))
        saved.pop("depth_geometry")
        saved.pop("robot_dynamics_randomization")
        restored = saved_training_config(saved)
        self.assertEqual((restored["depth_geometry"], restored["robot_dynamics_randomization"]), ("collision", "none"))

    def test_environment_configuration_receives_both_options(self):
        from src.tasks.amp_perceptive.env_cfg import g1_amp_perceptive_hiking_env_cfg
        cfg = PerceptiveConfig.defaults("supermimic", hiking_training_version=4)
        common = dict(motion_scope=cfg.motion_scope, sensor_protocol=cfg.sensor_protocol,
                      motion_data_protocol=cfg.motion_data_protocol, locomotion_objective=cfg.locomotion_objective,
                      hiking_training_version=cfg.hiking_training_version)
        plain = g1_amp_perceptive_hiking_env_cfg(**common)
        real = g1_amp_perceptive_hiking_env_cfg(**common, collision_horizontal_scale=.025, depth_geometry="source_mesh")
        tiles = lambda env: list(env.scene.terrain.terrain_generator.sub_terrains.values())
        self.assertTrue(all(t.collision_horizontal_scale == .05 and not t.depth_source_mesh for t in tiles(plain)))
        self.assertTrue(all(t.collision_horizontal_scale == .025 and t.depth_source_mesh for t in tiles(real)))
        self.assertEqual((plain.sim.njmax, real.sim.njmax), (700, 1024))


class RealSenseAdapterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.calibration = json.loads(ui.G1_DEPLOYMENT_DEPTH_CALIBRATION.read_text())
        cls.adapter = ui.RealSenseDepthAdapter()
        cls.crop = (ui.DEPTH_CROP_ROWS, ui.DEPTH_CROP_COLS)
        pattern = HikingDepthPatternCfg(sensor_protocol=HIKING_SENSOR_PROTOCOL)
        cls.training_rays = pattern.optical_rays()[0].numpy().astype(np.float64).reshape(36, 64, 3)
        cls.training_cosines = pattern.optical_rays()[1].numpy().astype(np.float64).reshape(36, 64)

    def real_image(self, rotation_wxyz):
        """Raw Z16 image of a floor with one step, from the real camera at the given torso orientation."""
        c = self.calibration
        u, v = np.meshgrid(np.arange(c["width"]), np.arange(c["height"]), indexing="xy")
        x, y = (u - c["cx"]) / c["fx"], (v - c["cy"]) / c["fy"]
        rays = np.stack((np.ones_like(x), -x, -y), -1)
        norm = np.linalg.norm(rays, axis=-1)
        z = self.scene(rays / norm[..., None], rotation_wxyz) / norm
        return np.where(np.isfinite(z) & (z < 60), np.rint(z / c["depth_scale_m_per_raw_unit"]), 0).astype(np.uint16)

    @staticmethod
    def scene(rays, rotation_wxyz, height=1.25, step_x=1.2, step_height=.17):
        """Distance along unit camera-frame rays to a floor that rises by ``step_height`` at ``step_x``."""
        world = rays @ ui._rotation_matrix(rotation_wxyz).T
        with np.errstate(divide="ignore", invalid="ignore"):
            floor, top, riser = -height / world[..., 2], -(height - step_height) / world[..., 2], step_x / world[..., 0]
            distance = np.where(floor * world[..., 0] < step_x, floor,
                                np.where(riser * world[..., 2] > -(height - step_height), top, riser))
        return np.where((world[..., 2] < 0) & (distance > 0), distance, np.inf)

    def training_image(self):
        return self.scene(self.training_rays, ui.HIKING_CAMERA_ROTATION_WXYZ) * self.training_cosines

    def test_constants_are_the_training_camera(self):
        pattern = HikingDepthPatternCfg(sensor_protocol=HIKING_SENSOR_PROTOCOL)
        self.assertEqual(ui.HIKING_CAMERA_FOV_DEGREES, (pattern.fovx, pattern.fovy))
        self.assertEqual(ui.HIKING_CAMERA_POSITION, pattern.position)
        self.assertEqual(ui.HIKING_CAMERA_ROTATION_WXYZ, pattern.rotation)
        self.assertEqual((ui.DEPTH_HEIGHT, ui.DEPTH_WIDTH), (pattern.height, pattern.width))

    def test_bundled_stream_is_the_supplied_480x270_profile(self):
        c = self.calibration
        self.assertEqual((c["width"], c["height"], c["fps"], c["format"]), (480, 270, 60, "Z16"))
        self.assertEqual((c["fx"], c["fy"]), (244.4962158203125, 244.4962158203125))
        self.assertEqual((c["cx"], c["cy"]), (244.61199951171875, 132.59669494628906))
        self.assertIsNone(c["camera_mounting_extrinsics"])
        self.assertNotIn("serial", json.dumps(c).replace("device serial number", ""))
        # Same camera as the 848 x 480 capture, at RealSense's scaled resolution; the training-side loader reads it too.
        loaded, capture = load_g1_depth_calibration(ui.G1_DEPLOYMENT_DEPTH_CALIBRATION), load_g1_depth_calibration()
        self.assertAlmostEqual(loaded.fx, capture.fx * 480 / 848, places=3)
        self.assertEqual(loaded.depth_scale_m_per_raw_unit, capture.depth_scale_m_per_raw_unit)

    def test_lookup_reproduces_the_training_image(self):
        truth = self.training_image()
        got = self.adapter.image_plane_depth(self.real_image(ui.HIKING_CAMERA_ROTATION_WXYZ))
        near = np.isfinite(truth[self.crop]) & (truth[self.crop] < 2.5)
        self.assertGreater(near.mean(), .9)
        error = np.abs(got - truth)[self.crop][near]
        self.assertLess(error.max(), .004)
        # A plain resize, as if the RealSense were the training pinhole, is measurably worse.
        raw = self.real_image(ui.HIKING_CAMERA_ROTATION_WXYZ)
        resized = raw[(np.arange(36)[:, None] * 7.5 + 3.75).astype(int),
                      (np.arange(64)[None] * 7.5 + 3.75).astype(int)] * self.calibration["depth_scale_m_per_raw_unit"]
        self.assertGreater(np.abs(resized - truth)[self.crop][near].mean(), 3 * error.mean())

    def test_only_the_cropped_away_border_is_outside_the_real_image(self):
        self.assertEqual(self.adapter.crop_coverage, 1.)
        self.assertEqual(sorted(set(np.where(~self.adapter.covered)[1].tolist())), [63])
        np.testing.assert_array_equal(self.adapter.depth_per_z, np.ones((36, 64), dtype=np.float32))

    def test_measured_rotation_is_compensated_exactly(self):
        half = math.radians(1.5)   # camera pitched 3 degrees further down than nominal
        w1, x1, y1, z1 = ui.HIKING_CAMERA_ROTATION_WXYZ
        w2, x2, y2, z2 = math.cos(half), 0., math.sin(half), 0.
        measured = (w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                    w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2)
        raw, truth = self.real_image(measured), self.training_image()
        near = np.isfinite(truth[self.crop]) & (truth[self.crop] < 2.5)
        compensated = ui.RealSenseDepthAdapter(mount_rotation_wxyz=measured).image_plane_depth(raw)
        self.assertLess(np.abs(compensated - truth)[self.crop][near].max(), .004)
        self.assertGreater(np.abs(self.adapter.image_plane_depth(raw) - truth)[self.crop][near].mean(), .02)

    def test_a_mount_that_loses_the_policy_region_is_refused(self):
        half = math.radians(10.)
        w1, x1, y1, z1 = ui.HIKING_CAMERA_ROTATION_WXYZ
        up = (w1 * math.cos(half) + y1 * math.sin(half), x1 * math.cos(half) - z1 * math.sin(half),
              y1 * math.cos(half) - w1 * math.sin(half), z1 * math.cos(half) + x1 * math.sin(half))
        with self.assertRaisesRegex(ValueError, "cannot be converted by a lookup"):
            ui.RealSenseDepthAdapter(mount_rotation_wxyz=up)
        with self.assertRaisesRegex(ValueError, "Rectify"):
            ui.RealSenseDepthAdapter(dict(self.calibration, distortion_coefficients=[.1, 0, 0, 0, 0]))
        with self.assertRaisesRegex(ValueError, "270x480"):
            self.adapter.frame(np.zeros((480, 848), np.uint16))

    def test_holes_are_filled_inside_the_policy_region(self):
        raw = self.real_image(ui.HIKING_CAMERA_ROTATION_WXYZ)
        clean = self.adapter.frame(raw)
        holed = raw.copy()
        holed[np.random.default_rng(0).random(raw.shape) < .15] = 0
        holed[230:250, 200:240] = 0                     # a patch of floor without stereo match
        frame = self.adapter.frame(holed)
        self.assertEqual((frame.shape, frame.dtype), ((2, 36, 64), np.float32))
        self.assertLess(np.abs(frame[0] - clean[0])[self.crop].max(), .03)       # of the 2.5 m range
        self.assertTrue((frame[0][self.crop] <= 1.).all())
        # Without filling, the same holes read as "nothing within range".
        unfilled = ui.RealSenseDepthAdapter(fill_holes=False).frame(holed)
        self.assertTrue((unfilled[0][self.crop] > 1.).any())
        # An image without any measurement stays missing instead of being invented.
        empty = self.adapter.frame(np.zeros_like(raw))
        self.assertTrue((empty[0] == ui.DEPTH_MISSING_M / ui.DEPTH_MAX_M).all())
        self.assertFalse(empty[1].any())

    def test_hole_fill_rule(self):
        image = np.full((5, 5), 1., dtype=np.float32)
        image[2, 2], image[0, 0], image[4, 4] = 0., .15, np.nan
        image[2, 3] = 2.
        filled, holes = ui.fill_depth_holes(image)
        self.assertEqual(int(holes.sum()), 3)
        self.assertAlmostEqual(float(filled[2, 2]), 9. / 8.)       # mean of its eight measured neighbours
        self.assertEqual((float(filled[0, 0]), float(filled[4, 4])), (1., 1.))
        np.testing.assert_array_equal(filled[~holes], image[~holes])

    def test_description_repeats_the_conversion(self):
        raw = self.real_image(ui.HIKING_CAMERA_ROTATION_WXYZ)
        described = json.loads(json.dumps(self.adapter.description()))
        index, factor = np.array(described["source_index"]), np.array(described["depth_per_z"], dtype=np.float32)
        z = raw.reshape(-1)[np.clip(index, 0, None)].astype(np.float32) * np.float32(described["depth_scale_m_per_raw_unit"])
        np.testing.assert_array_equal(np.where((index >= 0) & (z > 0), z * factor, 0.).astype(np.float32),
                                      self.adapter.image_plane_depth(raw))
        self.assertEqual(described["uncovered_training_pixels"], 36)


if __name__ == "__main__":
    unittest.main()
