"""Actual perceptive save/load on CPU, including late-optimizer rejection."""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import random
import tempfile
import time
from types import SimpleNamespace
from unittest.mock import patch
import unittest

import numpy as np
import torch

from src.wmac_perceptive.models import PerceptiveActor, TerrainPerceptiveWorldModel
from src.wmac_perceptive.perception import DepthHistory, PerceptionSpec
from src.wmac_perceptive.runner import MODULES, OPTIMIZERS
from test_perceptive_runner import runner_fixture


def restore_fixture(steps=1):
    runner, _, _ = runner_fixture(batch_size=3)
    runner.spec = PerceptionSpec()
    runner.actor = PerceptiveActor(runner.spec).double()
    runner.world_model = TerrainPerceptiveWorldModel(runner.adapter.terrain, runner.spec,
                                                    hidden=runner.cfg.model_hidden).double()
    # Enable real augmentation during restore so the test detects RNG restored
    # too early, before creating the fresh post-reset camera history.
    runner.cfg.depth_noise_std_m = .005
    runner.cfg.depth_dropout_probability = .01
    runner.cfg.validate()
    runner.opt_actor = torch.optim.Adam(runner.actor.parameters(), lr=runner.cfg.actor_lr,
                                        betas=runner.cfg.actor_betas, fused=False)
    runner.opt_model = torch.optim.Adam(runner.world_model.parameters(), lr=runner.cfg.world_model_lr, fused=False)
    runner.opt_critic = torch.optim.Adam(runner.critic.parameters(), lr=runner.cfg.critic_lr, fused=False)
    optimizer_class = torch.optim.AdamW if runner.cfg.discriminator_optimizer == "adamw" else torch.optim.Adam
    runner.opt_disc = optimizer_class([
        {"params": runner.discriminator.trunk.parameters(), "name": "amp_trunk",
         "weight_decay": runner.cfg.discriminator_trunk_weight_decay},
        {"params": runner.discriminator.amp_linear.parameters(), "name": "amp_head",
         "weight_decay": runner.cfg.discriminator_head_weight_decay},
    ], lr=runner.cfg.discriminator_lr, fused=False)
    for module_name, optimizer_name in (("actor", "opt_actor"), ("world_model", "opt_model"),
                                        ("critic", "opt_critic"), ("discriminator", "opt_disc")):
        module, optimizer = getattr(runner, module_name), getattr(runner, optimizer_name)
        for _ in range(steps):
            for parameter in module.parameters(): parameter.grad = torch.full_like(parameter, .01)
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
        if len(optimizer.state) != len(tuple(module.parameters())):
            raise AssertionError("Every trainable parameter must have populated Adam moments")
    runner.env = SimpleNamespace(common_step_counter=73+steps, num_envs=3)
    runner.started, runner.wall_offset = time.perf_counter(), 12.+steps
    runner.iteration, runner.sim_steps, runner.imagination_steps = 17+steps, 100+steps, 300+steps
    runner.critic_calibration_updates, runner._collect_accumulator = 20+steps, .375
    runner._cost_trajectory_id, runner._cost_history_complete = f"fixture-{steps}", True
    runner.actor_initialization_history = [{"kind": "test_perceptive", "step": steps}]
    runner.validation = {"validation_model_error": float("inf"), "validation_walk_samples": 0.,
                         "validation_recovery_samples": 0.}
    runner._fingerprints = lambda: ("terrain-fixture", "motion-fixture")
    runner.current_depth = torch.full((3, *runner.spec.frame_shape), .2, dtype=torch.float64)
    runner.current_depth[:, 1].fill_(1.)
    runner.depth_history = DepthHistory(runner.spec, 3, "cpu")
    runner.depth_history.reset(runner.current_depth)
    with torch.no_grad():
        runner.amp_norm.mean.fill_(steps*.1)
        runner.amp_norm.var.fill_(1.+steps*.2)
        runner.amp_norm.count.fill_(50+steps)
    runner.amp_norm.eps, runner.amp_norm.clip = .04, 3.
    return runner


def snapshot(runner):
    return copy.deepcopy({
        **{name: getattr(runner, name).state_dict() for name in (*MODULES, *OPTIMIZERS)},
        "clocks": (runner.iteration, runner.sim_steps, runner.imagination_steps,
                   runner.critic_calibration_updates, runner._collect_accumulator,
                   runner.wall_offset, runner.env.common_step_counter),
        "cost": (runner._cost_trajectory_id, runner._cost_history_complete),
        "initialization": runner.actor_initialization_history,
        "amp_settings": {key: getattr(runner.amp_norm, key) for key in ("eps", "variance_epsilon", "clip", "max_count")},
        "depth_ring": runner.depth_history.snapshot(), "depth_initialized": runner.depth_history.initialized.clone(),
        "current_depth": runner.current_depth.clone(),
        "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state(), "python_rng": random.getstate(),
    })


class PerceptiveRunnerRestoreTest(unittest.TestCase):
    def assert_tree_equal(self, actual, expected):
        if isinstance(expected, torch.Tensor):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        elif isinstance(expected, np.ndarray):
            np.testing.assert_array_equal(actual, expected)
        elif isinstance(expected, dict):
            self.assertEqual(actual.keys(), expected.keys())
            for key in expected: self.assert_tree_equal(actual[key], expected[key])
        elif isinstance(expected, (list, tuple)):
            self.assertEqual(len(actual), len(expected))
            for left, right in zip(actual, expected): self.assert_tree_equal(left, right)
        else: self.assertEqual(actual, expected)

    def test_actual_save_load_restores_all_weights_adam_clocks_and_rng_after_camera_reset(self):
        source = restore_fixture(steps=1)
        random.seed(101); np.random.seed(102); torch.manual_seed(103)
        with tempfile.TemporaryDirectory() as directory, patch("torch.cuda.is_available", return_value=False):
            path = Path(directory)/"source.pt"
            source.save(path, export=False)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            payload = torch.load(path, map_location="cpu", weights_only=False)
            restored = restore_fixture(steps=2)
            random.seed(201); np.random.seed(202); torch.manual_seed(203)
            fresh = torch.full((3, *restored.spec.frame_shape), .35, dtype=torch.float64)
            fresh[:, 1].fill_(1.)
            with patch("src.wmac_perceptive.runner.read_depth_frame", return_value=fresh):
                restored.load(path)
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)
        for name in (*MODULES, *OPTIMIZERS): self.assert_tree_equal(getattr(restored, name).state_dict(), payload[name])
        for key in ("iteration", "sim_steps", "imagination_steps", "critic_calibration_updates"):
            self.assertEqual(getattr(restored, key), payload[key])
        self.assertEqual(restored.wall_offset, payload["wall_seconds"])
        self.assertEqual(restored._collect_accumulator, payload["collect_accumulator"])
        self.assertEqual(restored.env.common_step_counter, payload["env_common_step_counter"])
        self.assertEqual(restored.actor_initialization_history, payload["actor_initialization_history"])
        for key, value in payload["amp_normalizer_settings"].items(): self.assertEqual(getattr(restored.amp_norm, key), value)
        self.assert_tree_equal(torch.get_rng_state(), payload["torch_rng"])
        self.assert_tree_equal(np.random.get_state(), payload["numpy_rng"])
        self.assert_tree_equal(random.getstate(), payload["python_rng"])
        self.assertTrue(bool(restored.depth_history.initialized.all()))
        # Resume starts a fresh simulator episode/history, not source images.
        torch.testing.assert_close(restored.current_depth, fresh, rtol=0, atol=0)
        ring = restored.depth_history.ring
        torch.testing.assert_close(ring, ring[:, :1].expand_as(ring), rtol=0, atol=0)
        self.assertFalse(restored._cost_history_complete)
        self.assertTrue(all(group["fused"] is False for name in OPTIMIZERS for group in getattr(restored, name).param_groups))

    def test_bad_late_discriminator_optimizer_and_flat_format_reject_before_any_mutation(self):
        source, restored = restore_fixture(steps=1), restore_fixture(steps=2)
        with tempfile.TemporaryDirectory() as directory, patch("torch.cuda.is_available", return_value=False):
            source_path = Path(directory)/"source.pt"
            source.save(source_path, export=False)
            payload = torch.load(source_path, map_location="cpu", weights_only=False)
            for problem in ("shape", "nan", "flat_format"):
                bad = copy.deepcopy(payload)
                if problem == "flat_format": bad["format"] = "amp-wmac-v1"
                else:
                    first = next(iter(bad["opt_disc"]["state"].values()))
                    if problem == "shape": first["exp_avg"] = torch.zeros(1, dtype=first["exp_avg"].dtype)
                    else: first["exp_avg"].reshape(-1)[0] = float("nan")
                path = Path(directory)/f"bad_{problem}.pt"
                torch.save(bad, path)
                before = snapshot(restored)
                with self.subTest(problem=problem), self.assertRaises(ValueError): restored.load(path)
                self.assert_tree_equal(snapshot(restored), before)


if __name__ == "__main__":
    unittest.main()
