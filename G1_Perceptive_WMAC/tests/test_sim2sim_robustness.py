"""Heightfield guard, robot-dynamics randomization and per-terrain curriculum metrics."""
from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest import mock

import mujoco
import numpy as np
import torch

from src.wmac_perceptive import physics_checks, robot_dynamics
from src.wmac_perceptive.hiking_context import normalize_hiking_sim_context


def _training_model():
    """Training robot with the armature its Hiking actuators install in the real environment."""
    from src.tasks.amp_perceptive.hiking_robot import get_hiking_robot_spec
    from src.wmac_perceptive.hiking_actuators import _joint_parameters
    model = get_hiking_robot_spec().compile()
    for i in range(model.njnt):
        if model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE:
            model.dof_armature[model.jnt_dofadr[i]] = _joint_parameters(model.joint(i).name)["armature"]
    return model


class ContextNormalizationTest(unittest.TestCase):
    def test_default_is_unchanged_and_dynamics_columns_pass_through(self):
        context = torch.randn(5, 133)
        old = torch.cat(((context[:, :1] - .95) / .65, context[:, 1:67], context[:, 67:96] - 1.,
                         context[:, 96:100], (context[:, 100:129] - .95) / .65), -1)
        self.assertTrue(torch.equal(normalize_hiking_sim_context(context[:, :129]), old))
        widened = normalize_hiking_sim_context(context, dynamics_columns=4)
        self.assertTrue(torch.equal(widened[:, :129], old))
        self.assertTrue(torch.equal(widened[:, 129:], context[:, 129:]))


class ConfigAndMetadataTest(unittest.TestCase):
    def _cfg(self, randomization="none"):
        from src.wmac_perceptive.config import PerceptiveConfig
        cfg = PerceptiveConfig.defaults("supermimic")
        cfg.robot_dynamics_randomization = randomization
        cfg.validate()
        return cfg

    def test_validation(self):
        from src.wmac_perceptive.config import PerceptiveConfig
        cfg = PerceptiveConfig.defaults("supermimic")
        cfg.robot_dynamics_randomization = "unknown"
        with self.assertRaises(ValueError):
            cfg.validate()
        legacy = PerceptiveConfig.defaults()
        legacy.robot_dynamics_randomization = "unitree_g1_v1"
        with self.assertRaises(ValueError):
            legacy.validate()

    def test_metadata_widens_context_only_when_enabled(self):
        from src.wmac_perceptive.checkpoints import architecture_metadata
        base = architecture_metadata(self._cfg())
        wide = architecture_metadata(self._cfg("unitree_g1_v1"))
        self.assertNotIn("robot_dynamics_randomization", base)
        self.assertEqual(wide["context_dim"], base["context_dim"] + 4)
        self.assertEqual(wide["context_layout"]["robot_dynamics"], [base["context_dim"], base["context_dim"] + 4])
        self.assertEqual(wide["world_model"]["input_dim"], base["world_model"]["input_dim"] + 4)
        self.assertEqual(wide["critic"]["input_dim"], base["critic"]["input_dim"] + 4)
        replay, wide_replay = base["execution_protocol"]["replay"], wide["execution_protocol"]["replay"]
        self.assertEqual(wide_replay["fixed_context_columns"], replay["fixed_context_columns"] + 4)
        self.assertEqual(wide["world_model"]["input_normalizer_dim"], base["world_model"]["input_normalizer_dim"])

    def test_old_payload_loads_without_randomization_and_never_inherits_hfield_override(self):
        from src.wmac_perceptive.config import config_from_payload, FORMAT
        cfg = self._cfg()
        saved = {k: v for k, v in vars(cfg).items()
                 if k not in ("robot_dynamics_randomization", "allow_unpatched_hfield")}
        loaded = config_from_payload({"format": FORMAT, "config": saved})
        self.assertEqual(loaded.robot_dynamics_randomization, "none")
        saved["allow_unpatched_hfield"] = True
        self.assertFalse(config_from_payload({"format": FORMAT, "config": saved}).allow_unpatched_hfield)


class HfieldGuardTest(unittest.TestCase):
    def test_refuses_unpatched_unless_allowed(self):
        for status, allowed, raises in (("needs_patch", False, True), ("needs_patch", True, False),
                                        ("patched", False, False), ("upstream_fixed", False, False)):
            report = {"status": status, "mujoco_warp_version": "3.8.1", "path": "x"}
            with mock.patch.object(physics_checks, "hfield_collision_status", return_value=report):
                if raises:
                    with self.assertRaises(RuntimeError):
                        physics_checks.require_fixed_hfield_collision(allowed)
                else:
                    self.assertEqual(physics_checks.require_fixed_hfield_collision(allowed), report)


class _FakeModel:
    def __init__(self, mj, worlds):
        for field in (*robot_dynamics.PROFILE_BODY_FIELDS, *robot_dynamics.PROFILE_DOF_FIELDS):
            value = torch.as_tensor(getattr(mj, field).copy(), dtype=torch.float32)
            setattr(self, field, value.expand(worlds, *value.shape).clone())


class RobotDynamicsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mj = _training_model()
        mj = cls.mj
        bodies = [mj.body(i).name for i in range(1, mj.nbody)]
        joints = [mj.joint(i).name for i in range(mj.njnt) if mj.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE]
        cls.robot = SimpleNamespace(body_names=bodies, joint_names=joints, indexing=SimpleNamespace(
            body_ids=torch.arange(1, mj.nbody),
            joint_v_adr=torch.tensor([int(mj.jnt_dofadr[mj.joint(n).id]) for n in joints])))

    def _env(self, worlds=3):
        mj = self.mj
        defaults = {field: torch.as_tensor(getattr(mj, field).copy(), dtype=torch.float32)
                    for field in (*robot_dynamics.PROFILE_BODY_FIELDS, *robot_dynamics.PROFILE_DOF_FIELDS)}
        sim = SimpleNamespace(model=_FakeModel(mj, worlds), get_default_field=defaults.__getitem__)
        return SimpleNamespace(num_envs=worlds, device=torch.device("cpu"), sim=sim, scene={"robot": self.robot})

    def test_profile_describes_training_robot(self):
        profile = robot_dynamics.load_robot_dynamics_profile("unitree_mujoco_g1_29dof")
        self.assertEqual(set(profile["bodies"]), set(self.robot.body_names))
        self.assertEqual(set(profile["joints"]), set(self.robot.joint_names))
        self.assertAlmostEqual(profile["training_total_mass"], float(self.mj.body_mass.sum()), places=4)
        self.assertAlmostEqual(profile["total_mass"], 35.112, places=3)

    def _sample(self, value):
        bounds = (value, value + 1e-9)   # a degenerate range samples the endpoint itself
        settings = {"profile": "unitree_mujoco_g1_29dof", "inertial_interpolation": bounds,
                    "armature_interpolation": bounds, "damping_scale": bounds, "frictionloss_scale": bounds}
        env = self._env()
        with mock.patch.dict(robot_dynamics.ROBOT_DYNAMICS_RANDOMIZATION_SETTINGS, {"fixed": settings}):
            robot_dynamics.randomize_robot_dynamics(env, None, randomization="fixed")
        return env

    @staticmethod
    def _tensors(model, world):
        rotation = torch.stack([torch.as_tensor(_quat_matrix(q)) for q in model.body_iquat[world].double()])
        return rotation @ torch.diag_embed(model.body_inertia[world].double()) @ rotation.transpose(-1, -2)

    def test_endpoints_reproduce_training_robot_and_profile(self):
        training = self._sample(0.)
        reference = _FakeModel(self.mj, 1)
        for field in ("body_mass", "body_ipos", "dof_armature"):
            torch.testing.assert_close(getattr(training.sim.model, field)[:, 1:] if field.startswith("body")
                                       else getattr(training.sim.model, field),
                                       getattr(reference, field)[:, 1:].expand_as(getattr(training.sim.model, field)[:, 1:])
                                       if field.startswith("body") else getattr(reference, field).expand_as(
                                           getattr(training.sim.model, field)), atol=1e-5, rtol=0)
        torch.testing.assert_close(self._tensors(training.sim.model, 0)[1:], self._tensors(reference, 0)[1:],
                                   atol=1e-7, rtol=1e-4)
        self.assertLess(float(training.sim.model.dof_damping.abs().max()), 1e-9)
        nominal = robot_dynamics.nominal_robot_dynamics_context("unitree_g1_v1", 2, "cpu")
        torch.testing.assert_close(nominal, torch.tensor([[-2/3, -2/3, -1., -1.]]).expand(2, 4))
        unitree = self._sample(1.)
        profile = robot_dynamics.profile_tensors(self.robot, robot_dynamics.load_robot_dynamics_profile(
            "unitree_mujoco_g1_29dof"), "cpu")
        torch.testing.assert_close(unitree.sim.model.body_mass[0, 1:], profile["body_mass"])
        torch.testing.assert_close(unitree.sim.model.dof_armature[0, profile["dof_ids"]], profile["dof_armature"])
        torch.testing.assert_close(unitree.sim.model.dof_frictionloss[0, profile["dof_ids"]], profile["dof_frictionloss"])
        expected = torch.stack([torch.as_tensor(_quat_matrix(q)) for q in profile["body_iquat"].double()])
        expected = expected @ torch.diag_embed(profile["body_inertia"].double()) @ expected.transpose(-1, -2)
        torch.testing.assert_close(self._tensors(unitree.sim.model, 0)[1:], expected, atol=1e-7, rtol=1e-4)

    def test_random_samples_stay_physical(self):
        env = self._env(256)
        robot_dynamics.randomize_robot_dynamics(env, None)
        model = env.sim.model
        self.assertTrue(torch.all(model.body_mass[:, 1:] > 0))
        self.assertTrue(torch.all(model.body_inertia[:, 1:] > 0))
        self.assertTrue(torch.all(model.dof_armature[:, self.robot.indexing.joint_v_adr] > 0))
        self.assertTrue(torch.all(env._robot_dynamics_context.abs() <= 1 + 1e-6))
        total = model.body_mass[:, 1:].sum(-1)
        self.assertGreater(float(total.max()), 35.)
        self.assertLess(float(total.min()), 33.5)
        # The context encodes the sample: decoding lambda predicts each world's total mass.
        lam = (env._robot_dynamics_context[:, 0] + 1) / 2 * 1.5 - .25
        predicted = (1 - lam) * float(self.mj.body_mass.sum()) + lam * 35.112
        torch.testing.assert_close(total, predicted, atol=2e-3, rtol=0)


def _quat_matrix(q):
    matrix = np.zeros(9)
    mujoco.mju_quat2Mat(matrix, np.asarray(q, float))
    return matrix.reshape(3, 3)


class TerrainCurriculumMetricsTest(unittest.TestCase):
    def test_family_means_follow_mjlab_column_allocation(self):
        from src.tasks.amp_perceptive.terrain import hiking_terrain_generator
        from src.wmac_perceptive.runner import PerceptiveRunner
        generator = hiking_terrain_generator(rows=10, cols=20, task_protocol="hiking_full_v1")
        terrain = SimpleNamespace(cfg=SimpleNamespace(terrain_generator=generator),
                                  terrain_types=torch.tensor([0, 0, 7, 9, 19]),
                                  terrain_levels=torch.tensor([1, 3, 4, 6, 9]))
        fake = SimpleNamespace(env=SimpleNamespace(scene=SimpleNamespace(terrain=terrain)))
        metrics = PerceptiveRunner._terrain_curriculum_metrics(fake)
        self.assertEqual(len(metrics), len(generator.sub_terrains))
        # 20 columns: 0 perlin_rough, 1 perlin_rough_stand, 2-3 gaps, 4-6 stairs, 7-8 stairs_high,
        # 9-11 stairs_inv, 12-13 stairs_inv_high, 14-15 boxes, 16-17 mesh_boxes, 18-19 slope.
        self.assertEqual(metrics["terrain_level_perlin_rough"], 2.)
        self.assertEqual(metrics["terrain_level_pyramid_stairs_high"], 4.)
        self.assertEqual(metrics["terrain_level_pyramid_stairs_inv"], 6.)
        self.assertEqual(metrics["terrain_level_hf_pyramid_slope_inv"], 9.)
        self.assertEqual(metrics["terrain_level_pyramid_stairs"], 0.)
        self.assertEqual(metrics["terrain_level_boxes"], 0.)


if __name__ == "__main__":
    unittest.main()
