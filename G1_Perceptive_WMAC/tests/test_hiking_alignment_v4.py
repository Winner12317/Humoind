"""Training version 4: the task realigned with the original Hiking implementation.

Covers the parts version 4 changes relative to version 3: terrain-edge
detection on Hiking's own terrain mesh, the optional shod robot variant, the
first-reset terrain level and the fresh-run defaults. The edge set was also
compared with InstinctLab's and IsaacLab's own code executed on the same
multi-tile terrain when the change was made; that needs both checkouts, so the
checks here use the bundled IsaacLab mesh function and fixed geometry.
"""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import mujoco
import numpy as np
import torch
import trimesh
from mjlab.terrains.terrain_generator import TerrainGenerator

from scripts.train_wmac_perceptive import parse_config
from src.tasks.amp_perceptive import hiking_geometry as geometry
from src.tasks.amp_perceptive.commands import hiking_tracking_terrain_curriculum
from src.tasks.amp_perceptive.env_cfg import g1_amp_perceptive_hiking_env_cfg
from src.tasks.amp_perceptive.hiking_rewards import hiking_reward_terms
from src.tasks.amp_perceptive.hiking_robot import (
    HIKING_SHOE_URDF, HIKING_SHOE_URDF_SHA256, get_hiking_robot_spec, get_hiking_shoe_robot_spec,
    hiking_foot_geometry,
)
from src.tasks.amp_perceptive.terrain import TerrainHeightField, hiking_terrain_generator
from src.wmac_perceptive.config import PerceptiveConfig
from src.wmac_perceptive.hiking_actuators import HikingDelayedPdActuatorCfg, ResetFixedDelay
from src.wmac_perceptive.safety import hiking_foot_point_grid, hiking_foot_volume_points

ROOT = Path(__file__).resolve().parents[1]
ISAACLAB_HEIGHT_FIELD = ROOT / "reproduce/references/IsaacLab/source/isaaclab/isaaclab/terrains/height_field/utils.py"


def canonical(edges):
    """Direction-independent, rounded edge rows for set comparison."""
    edges = np.round(np.asarray(edges, dtype=np.float64).reshape(-1, 2, 3), 4) + 0.
    swap = [tuple(a) > tuple(b) for a, b in edges]
    edges[swap] = edges[swap][:, ::-1]
    return {tuple(row.reshape(-1)) for row in edges}


def step_grid(rise_cells, cells=12):
    """Integer heights [x, y] with one straight step of ``rise_cells`` * 5 mm at x index 6."""
    grid = np.zeros((cells, cells), dtype=np.int64)
    grid[6:] = rise_cells
    return grid


class SourceMeshEdgeTest(unittest.TestCase):
    def test_height_field_mesh_equals_isaaclab_source(self):
        spec = importlib.util.spec_from_file_location("isaaclab_height_field_utils", ISAACLAB_HEIGHT_FIELD)
        source = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(source)
        rng = np.random.default_rng(4)
        for threshold in (1., None):
            grid = rng.integers(-30, 31, (23, 17)).astype(np.int16)
            vertices, triangles = source.convert_height_field_to_mesh(grid, .05, .005, threshold)
            local_vertices, local_triangles = geometry.isaaclab_height_field_mesh(grid, .05, .005, threshold)
            np.testing.assert_array_equal(local_vertices, vertices)
            np.testing.assert_array_equal(local_triangles, triangles.astype(np.int64))

    def test_low_steps_have_edges_only_on_the_source_mesh(self):
        for rise_cells, rise in ((11, .055), (20, .10), (26, .13), (46, .23)):
            with self.subTest(rise=rise):
                grid = step_grid(rise_cells)
                source = geometry.hiking_source_mesh_edges([(grid, np.zeros(2), (.55, .55))], [])
                # A vertical riser at x = 0.30: lip and crease share that x.
                x = source[:, [0, 3]]
                np.testing.assert_allclose(x, .30, atol=1e-6)
                self.assertEqual(sorted({round(float(z), 4) for z in source[:, [2, 5]].reshape(-1)}), [0., rise])
                # The lip spans the tile. IsaacLab's corner rule also shifts the
                # lower vertices one cell along the step, so the crease is one cell shorter.
                for z, length in ((0., .50), (rise, .55)):
                    row = source[np.isclose(source[:, 2], z)]
                    self.assertAlmostEqual(float(np.abs(row[:, 4] - row[:, 1]).sum()), length, places=5)
                # The MuJoCo heightfield turns the riser into a one-cell ramp of
                # atan(rise / 0.05); it passes the 70 degree test only above ~0.137 m.
                ramp = geometry.sharp_edges(geometry.heightfield_mesh(
                    grid.T * .005, np.array([.275, .275]), np.array([.275, .275])))
                self.assertEqual(len(ramp) > 0, rise > .137)

    def test_slopes_below_45_degrees_and_flat_ground_have_no_edges(self):
        ramp = np.arange(12)[:, None].repeat(12, 1) * 9          # 45 mm per 50 mm cell
        self.assertEqual(len(geometry.hiking_source_mesh_edges([(ramp, np.zeros(2), (.55, .55))], [])), 0)
        self.assertEqual(len(geometry.hiking_source_mesh_edges([(step_grid(0), np.zeros(2), (.55, .55))], [])), 0)

    def test_merged_mesh_drops_edges_shared_by_more_than_two_faces(self):
        # Two walls meeting at a tile corner share their vertical corner edge,
        # and a wall standing on the zero-height mesh-box ground shares its
        # inner base edge with it. trimesh keeps an edge only when exactly two
        # faces meet there, which is what Hiking's detector sees.
        walls = [(-.025, 4., 2.5, .025, 4., 2.5, 0.), (4., -.025, 2.5, 4., .025, 2.5, 0.)]
        merged = canonical(geometry.hiking_source_mesh_edges([(None, np.zeros(2), (8., 8.))], walls))
        separate = canonical(np.concatenate([geometry.sharp_edges(trimesh.creation.box(
            extents=(2 * hx, 2 * hy, 2 * hz), transform=trimesh.transformations.translation_matrix((x, y, z))))
            for x, y, z, hx, hy, hz, _ in walls]))
        corner = (0., 0., 0., 0., 0., 5.)
        base = (0., 0., 0., 0., 8., 0.)
        self.assertIn(corner, separate)
        self.assertIn(base, separate)
        self.assertNotIn(corner, merged)
        self.assertNotIn(base, merged)
        self.assertIn((-.05, 0., 5., -.05, 8., 5.), merged)

    def test_pruned_detection_equals_full_mesh_detection(self):
        rng = np.random.default_rng(9)
        tiles, boxes = [], []
        for row in range(2):
            for col in range(2):
                grid = np.zeros((161, 161), dtype=np.int64)
                for _ in range(6):
                    x, y = rng.integers(5, 120, 2)
                    grid[x:x + rng.integers(4, 30), y:y + rng.integers(4, 30)] = rng.integers(-40, 60)
                grid[1:-1, 1:-1] += rng.integers(-4, 5, (159, 159))
                corner = np.array([row * 8. - 8., col * 8. - 8.])
                tiles.append((None if (row, col) == (1, 1) else grid, corner, (8., 8.)))
                boxes.append((corner[0] - .025, corner[1] + 4., 2.5, .025, 4., 2.5, 0.))
                boxes.append((corner[0] + 4., corner[1] - .025, 2.5, 4., .025, 2.5, 0.))
        for _ in range(12):
            boxes.append((rng.uniform(1, 7), rng.uniform(1, 7), .15, .2, .22, .15, rng.uniform(0, 6.28)))
        mesh = geometry.hiking_source_terrain_mesh(tiles, boxes)
        full = canonical(geometry._sharp_edge_coordinates(mesh, prune=False))
        pruned = canonical(geometry._sharp_edge_coordinates(mesh, prune=True))
        self.assertGreater(len(full), 500)
        self.assertEqual(full, pruned)

    def test_component_merge_covers_the_same_lines_as_one_global_pass(self):
        grid = np.zeros((40, 40), dtype=np.int64)
        grid[10:30, 10:30] = 30
        grid[15:25, 15:25] = 60
        mesh = geometry.hiking_source_terrain_mesh([(grid, np.zeros(2), (1.95, 1.95))], [])
        raw = geometry._sharp_edge_coordinates(mesh)
        settings = SimpleNamespace(cfg=SimpleNamespace(adjacent_angle_threshold=30., min_points=2),
                                   rng=__import__("random").Random(0))
        global_pass = geometry.merge_sharp_edges(settings, raw).reshape(-1, 6)
        by_component = geometry._merge_by_component(raw)
        length = lambda rows: float(np.linalg.norm(rows[:, 3:] - rows[:, :3], axis=1).sum())
        self.assertAlmostEqual(length(global_pass), length(by_component), places=4)
        # Two square rings (1.0 m and 0.5 m sides), lip and crease each.
        self.assertGreater(length(by_component), 11.5)
        self.assertLessEqual(length(by_component), 12. + 1e-4)
        points = torch.as_tensor(raw.reshape(-1, 3), dtype=torch.float32) + torch.tensor([.01, -.01, .02])
        torch.testing.assert_close(
            geometry.HikingEdgeCylinders(global_pass).penetration_depth(points),
            geometry.HikingEdgeCylinders(by_component).penetration_depth(points))


class SourceMeshTerrainOracleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cfg = hiking_terrain_generator(rows=2, cols=20, task_protocol="hiking_full_v1")
        spec = mujoco.MjSpec()
        TerrainGenerator(cfg).compile(spec)
        cls.model = spec.compile()
        cls.heightfield = TerrainHeightField.from_model(cls.model)
        cls.source = TerrainHeightField.from_model(cls.model, source_mesh_edges=True)

    @staticmethod
    def segments(terrain):
        return terrain.hiking_edges.edges[:terrain.hiking_edges.edge_count].numpy()

    def test_every_stair_tile_has_edges_at_every_difficulty(self):
        source, heightfield = self.segments(self.source), self.segments(self.heightfield)
        families = {}
        for gid in range(self.model.ngeom):
            if int(self.model.geom_type[gid]) != int(mujoco.mjtGeom.mjGEOM_HFIELD):
                continue
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_HFIELD, int(self.model.geom_dataid[gid]))
            family = name[len("hiking_full_"):].rsplit("_", 1)[0]
            centre = self.model.geom_pos[gid, :2]

            def length(edges):
                middle = edges.mean(1)[:, :2]
                inside = (np.abs(middle - centre) < 3.9).all(1) & (edges[..., 2].max(1) < 2.)
                return float(np.linalg.norm(edges[inside, 1] - edges[inside, 0], axis=1).sum())
            families.setdefault(family, []).append((length(source), length(heightfield)))
        self.assertEqual(len(families), 10)
        for family in ("pyramid_stairs", "pyramid_stairs_inv", "boxes"):
            for source_length, _ in families[family]:
                self.assertGreater(source_length, 100., family)
        # Row 0 is the easiest half of the difficulty range: stairs below 0.14 m.
        easy_stairs = [pair for family in ("pyramid_stairs", "pyramid_stairs_inv") for pair in families[family][::2]]
        self.assertGreater(sum(a for a, _ in easy_stairs), 3. * sum(b for _, b in easy_stairs))
        for family in ("perlin_rough", "perlin_rough_stand"):
            self.assertTrue(all(source_length < 1. for source_length, _ in families[family]), family)

    def test_outer_border_contributes_no_edge(self):
        half = np.array([2 * 8., 20 * 8.]) / 2
        for terrain, expect_border in ((self.source, False), (self.heightfield, True)):
            edges = self.segments(terrain)
            outside = (np.abs(edges[..., :2]) > half + .5).any(-1).any(-1)
            self.assertEqual(bool(outside.any()), expect_border)

    def test_height_and_ray_queries_are_unchanged(self):
        xy = torch.rand(4000, 2) * torch.tensor([16., 160.]) - torch.tensor([8., 80.])
        torch.testing.assert_close(self.source.height_at(xy), self.heightfield.height_at(xy))
        start = torch.cat((xy, torch.full((4000, 1), 20.)), -1)
        torch.testing.assert_close(self.source.hiking_ray_height_at(start), self.heightfield.hiking_ray_height_at(start))

    def test_environment_version_selects_the_edge_source(self):
        for version, expected in ((3, self.heightfield), (4, self.source)):
            env = SimpleNamespace(sim=SimpleNamespace(mj_model=self.model), device="cpu",
                                  cfg=SimpleNamespace(_hiking_training_version=version))
            self.assertEqual(TerrainHeightField.from_env(env).hiking_edges.edge_count, expected.hiking_edges.edge_count)
        self.assertNotEqual(self.source.hiking_edges.edge_count, self.heightfield.hiking_edges.edge_count)


class ShoeVariantTest(unittest.TestCase):
    def test_shoe_urdf_is_the_pinned_upstream_file_and_only_lowers_the_soles(self):
        self.assertEqual(hashlib.sha256(HIKING_SHOE_URDF.read_bytes()).hexdigest(), HIKING_SHOE_URDF_SHA256)
        self.assertEqual(HIKING_SHOE_URDF_SHA256, "ef88a3094280b966624f60f7179aa59b7b2536fc282cf9fefa0c71c75a0a40fb")
        unshod, shod = get_hiking_robot_spec().compile(), get_hiking_shoe_robot_spec().compile()
        self.assertEqual((unshod.nbody, unshod.ngeom, unshod.njnt), (shod.nbody, shod.ngeom, shod.njnt))
        for name in ("body_mass", "body_ipos", "body_inertia", "jnt_range", "geom_size", "geom_quat"):
            np.testing.assert_array_equal(getattr(unshod, name), getattr(shod, name), err_msg=name)
        # Only the free root's default height differs: the rest pose stands on the lower soles.
        difference = shod.body_pos - unshod.body_pos
        pelvis = shod.body("pelvis").id
        np.testing.assert_allclose(difference[pelvis], (0., 0., .023), atol=1e-12)
        np.testing.assert_array_equal(np.delete(difference, pelvis, axis=0), 0.)
        moved = np.flatnonzero(np.abs(unshod.geom_pos - shod.geom_pos).max(1) > 0)
        names = sorted(shod.geom(int(i)).name for i in moved)
        self.assertEqual(names, sorted(f"{side}_foot{k}_collision" for side in ("left", "right") for k in range(1, 8)))
        np.testing.assert_allclose(unshod.geom_pos[moved, 2], -.025, atol=1e-12)
        np.testing.assert_allclose(shod.geom_pos[moved, 2], -.048, atol=1e-12)
        np.testing.assert_array_equal(unshod.geom_pos[moved, :2], shod.geom_pos[moved, :2])

    def test_foot_measurements_follow_shoe_config_mixin(self):
        self.assertEqual(hiking_foot_geometry("unshod")["volume_points_z"], (-.04, 0.))
        self.assertEqual(hiking_foot_geometry("unshod")["plane_height_offset"], .035)
        self.assertEqual(hiking_foot_geometry("shoe")["volume_points_z"], (-.063, -.023))
        self.assertEqual(hiking_foot_geometry("shoe")["plane_height_offset"], .058)
        with self.assertRaises(ValueError):
            hiking_foot_geometry("boots")
        grid = hiking_foot_point_grid((-.063, -.023))
        torch.testing.assert_close(grid.amin(0), torch.tensor([-.025, -.03, -.063]))
        torch.testing.assert_close(grid.amax(0), torch.tensor([.12, .03, -.023]))
        position = torch.zeros(1, 2, 3)
        quaternion = torch.zeros(1, 2, 4)
        quaternion[..., 0] = 1.
        kinematics = (position, quaternion, torch.zeros_like(position), torch.zeros_like(position))
        default_points, _ = hiking_foot_volume_points(kinematics, (0, 1))
        shod_points, _ = hiking_foot_volume_points(kinematics, (0, 1), foot_points=grid)
        torch.testing.assert_close(shod_points, default_points - torch.tensor([0., 0., .023]))

    def test_task_builder_selects_robot_and_actuator_delay(self):
        def build(variant):
            return g1_amp_perceptive_hiking_env_cfg(
                sensor_protocol="hiking_depth_only_v1", motion_data_protocol="hiking_walk_v1",
                locomotion_objective="hiking_full_v1", hiking_training_version=4, robot_variant=variant)
        unshod, shod = build("unshod"), build("shoe")
        self.assertIs(unshod.scene.entities["robot"].spec_fn, get_hiking_robot_spec)
        self.assertIs(shod.scene.entities["robot"].spec_fn, get_hiking_shoe_robot_spec)
        for cfg, lag in ((unshod, 2), (shod, 0)):
            actuators = cfg.scene.entities["robot"].articulation.actuators
            self.assertEqual(len(actuators), 5)
            self.assertTrue(all(isinstance(a, HikingDelayedPdActuatorCfg) and a.max_delay_substeps == lag
                                for a in actuators))
        self.assertEqual(unshod.actions["joint_pos"].scale, shod.actions["joint_pos"].scale)
        self.assertAlmostEqual(shod.scene.entities["robot"].init_state.pos[2]
                               - unshod.scene.entities["robot"].init_state.pos[2], .023)
        self.assertEqual(set(unshod.rewards), set(shod.rewards))
        with self.assertRaises(ValueError):
            g1_amp_perceptive_hiking_env_cfg(
                sensor_protocol="hiking_depth_only_v1", motion_data_protocol="hiking_walk_v1",
                locomotion_objective="hiking_full_v1", hiking_training_version=4, robot_variant="boots")

    def test_undelayed_actuators_never_sample_a_lag(self):
        delay = ResetFixedDelay(5000, 1, "cpu", max_lag=0)
        delay.reset()
        self.assertEqual(int(delay.lags.max()), 0)
        delayed = ResetFixedDelay(5000, 1, "cpu")
        delayed.reset()
        self.assertEqual(sorted(delayed.lags.unique().tolist()), [0, 1, 2])
        with self.assertRaises(ValueError):
            ResetFixedDelay(1, 1, "cpu", max_lag=3)

    def test_plane_reward_uses_the_variant_offset(self):
        from tests.test_hiking_full_rewards import fixture
        inputs, auxiliary, _, _ = fixture()
        auxiliary = auxiliary.clone()
        auxiliary[..., 6:8] = 5.                              # both feet in contact
        inputs = dict(inputs)
        inputs["foot_ground_height"] = inputs["foot_position_w"][..., :, 2, None].expand(-1, -1, 2) - .05
        unshod = hiking_reward_terms(inputs, auxiliary)["feet_at_plane"]
        torch.testing.assert_close(unshod, torch.full_like(unshod, 4 * (.05 - .035)))
        inputs["foot_plane_height_offset"] = .058
        shod = hiking_reward_terms(inputs, auxiliary)["feet_at_plane"]
        torch.testing.assert_close(shod, torch.zeros_like(shod))


class Version4ConfigurationTest(unittest.TestCase):
    def test_fresh_defaults_follow_hiking_and_version_3_is_preserved(self):
        fresh, previous = parse_config(["--hiking-training-version", "4"]), parse_config(["--hiking-training-version", "3"])
        self.assertEqual((fresh.hiking_training_version, fresh.sensor_protocol, fresh.walking_expert_source,
                          fresh.discriminator_trunk_weight_decay, fresh.discriminator_head_weight_decay,
                          fresh.robot_variant, fresh.amp_reward_coef, fresh.gradient_penalty),
                         (4, "hiking_depth_only_v1", "none", 3e-4, 1., "unshod", .25, 5.))
        self.assertEqual((previous.sensor_protocol, previous.walking_expert_source,
                          previous.discriminator_head_weight_decay),
                         ("g1_hiking_calibrated_v1", "pre_hiking_local_walk_turn_v2", .04))
        # Both remain selectable on a version 4 run.
        measured = parse_config(["--hiking-training-version", "4", "--sensor-protocol", "g1_hiking_calibrated_v1",
                                 "--walking-expert-source", "pre_hiking_local_walk_turn_v2"])
        self.assertEqual(measured.hiking_training_version, 4)

    def test_robot_variant_is_validated_and_needs_version_4(self):
        self.assertEqual(parse_config(["--robot-variant", "shoe"]).robot_variant, "shoe")
        with self.assertRaisesRegex(ValueError, "robot_variant"):
            parse_config(["--robot-variant", "boots"])
        with self.assertRaisesRegex(ValueError, "version 4"):
            parse_config(["--hiking-training-version", "3", "--robot-variant", "shoe"])
        self.assertEqual(PerceptiveConfig.defaults("supermimic", hiking_training_version=2).robot_variant, "unshod")

    def test_transformer_actor_is_available_on_versions_3_and_4(self):
        for version in ("3", "4"):
            cfg = parse_config(["--hiking-training-version", version, "--actor-type", "transformer"])
            self.assertEqual(cfg.actor_type, "transformer")
        with self.assertRaises(ValueError):
            parse_config(["--hiking-training-version", "2", "--actor-type", "transformer"])


class InitialCurriculumTest(unittest.TestCase):
    @staticmethod
    def environment(version, step=0):
        updates = []
        terrain = SimpleNamespace(terrain_levels=torch.tensor([3, 0, 5]),
                                  update_env_origins=lambda ids, up, down: updates.append((up.clone(), down.clone())))
        command = SimpleNamespace(metrics={"tracking_exp_vel_xy": torch.zeros(3), "tracking_exp_vel_yaw": torch.zeros(3)})
        env = SimpleNamespace(num_envs=3, device="cpu", scene=SimpleNamespace(terrain=terrain),
                              command_manager=SimpleNamespace(get_term=lambda name: command),
                              episode_length_buf=torch.zeros(3, dtype=torch.long), common_step_counter=step,
                              cfg=SimpleNamespace(_hiking_training_version=version))
        return env, updates, command

    def test_first_reset_moves_every_robot_down_once(self):
        env, updates, command = self.environment(4)
        hiking_tracking_terrain_curriculum(env, None)
        self.assertFalse(updates[0][0].any())
        self.assertTrue(updates[0][1].all())
        # A second reset before any step, and ordinary resets later, follow the scores.
        hiking_tracking_terrain_curriculum(env, None)
        self.assertFalse(updates[1][1].any())
        env.common_step_counter = 40
        env.episode_length_buf = torch.tensor([40, 0, 40])
        command.metrics["tracking_exp_vel_xy"] = torch.tensor([.7, .7, .1])
        command.metrics["tracking_exp_vel_yaw"] = torch.tensor([.2, .2, .2])
        hiking_tracking_terrain_curriculum(env, None)
        self.assertEqual(updates[2][0].tolist(), [True, False, False])
        self.assertEqual(updates[2][1].tolist(), [False, False, True])

    def test_version_3_keeps_its_sampled_initial_levels(self):
        env, updates, _ = self.environment(3)
        hiking_tracking_terrain_curriculum(env, None)
        self.assertFalse(updates[0][0].any())
        self.assertFalse(updates[0][1].any())


if __name__ == "__main__":
    unittest.main()
