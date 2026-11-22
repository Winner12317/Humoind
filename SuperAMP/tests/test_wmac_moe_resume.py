"""Actual CPU runner checkpoint/resume and MLP-to-MoE warmstart contracts."""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from src.wmac.actors import actor_architecture, create_actor
from test_wmac_schedule_restore import checkpoint_runner


MODULES = ("actor", "critic", "world_model", "discriminator", "amp_norm")
OPTIMIZERS = ("opt_actor", "opt_critic", "opt_model", "opt_disc")


def runner_fixture(kind="moe", **changes):
    runner = checkpoint_runner()
    # The shared legacy fixture has cfg.action_limit=20 but an actual bound of
    # 5. Match the saved architecture to its actual function in these tests.
    runner.cfg.action_limit = runner.actor.action_limit
    runner.cfg.actor_type = kind
    runner.cfg.moe_variant = "legacy"
    runner.cfg.moe_num_experts = 4
    runner.cfg.moe_top_k = 2
    runner.cfg.moe_gate_hidden = (8,)
    runner.cfg.moe_temperature = 1.
    runner.cfg.moe_load_balance_coef, runner.cfg.moe_router_z_coef = .01, .001
    for name, value in changes.items():
        setattr(runner.cfg, name, value)
    runner.actor = create_actor(runner.cfg).to(dtype=torch.float64)
    runner.opt_actor = torch.optim.Adam(runner.actor.parameters(), lr=runner.cfg.actor_lr,
                                        betas=runner.cfg.actor_betas)
    return runner


def exercise_actor_adam(runner):
    """Populate every Actor Adam slot through a real differentiable loss."""
    observations = torch.randn(7, 384, dtype=torch.float64)
    loss = runner.actor(observations).square().mean()
    # Tiny parameter regularization also covers currently unselected experts;
    # sparse dispatch itself is tested separately in test_wmac_moe.py.
    loss = loss+1e-3*sum(p.square().mean() for p in runner.actor.parameters())
    if runner.cfg.actor_type == "moe":
        loss = loss+.01*runner.actor.route(observations).logits.square().mean()
    runner.opt_actor.zero_grad(set_to_none=True)
    loss.backward()
    runner.opt_actor.step()
    runner.opt_actor.zero_grad(set_to_none=True)


def stamp_source(runner):
    exercise_actor_adam(runner)
    # Distinct trained auxiliary state makes accidental loading during an
    # actor-only warmstart observable, including nonempty optimizer moments.
    for name, optimizer in (("critic", runner.opt_critic), ("world_model", runner.opt_model),
                            ("discriminator", runner.opt_disc)):
        for parameter in getattr(runner, name).parameters():
            parameter.grad = torch.full_like(parameter, .02)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    with torch.no_grad():
        runner.actor.norm.mean.copy_(torch.linspace(-.3, .3, 384, dtype=torch.float64))
        runner.actor.norm.var.copy_(torch.linspace(.7, 1.8, 384, dtype=torch.float64))
        runner.actor.norm.count.fill_(71.)
        runner.amp_norm.mean.fill_(.2)
        runner.amp_norm.var.fill_(1.5)
        runner.amp_norm.count.fill_(55.)
    runner.amp_norm.eps, runner.amp_norm.clip = .04, 3.
    runner._collect_accumulator = .375
    runner.actor_initialization_history = [{"kind": "test_initialization", "source_sha256": "fixture"}]


def mutable_snapshot(runner):
    return copy.deepcopy({
        **{name: getattr(runner, name).state_dict() for name in (*MODULES, *OPTIMIZERS)},
        "clocks": (runner.iteration, runner.sim_steps, runner.imagination_steps,
                   runner.env.common_step_counter, runner.critic_calibration_updates,
                   runner._collect_accumulator, runner.wall_offset),
        "adapter_counter": runner.adapter.recovery_counter_semantics,
        "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(),
    })


class MoERunnerResumeTest(unittest.TestCase):
    def assert_tree_equal(self, actual, expected):
        if isinstance(expected, torch.Tensor):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        elif isinstance(expected, np.ndarray):
            np.testing.assert_array_equal(actual, expected)
        elif isinstance(expected, dict):
            self.assertEqual(actual.keys(), expected.keys())
            for key in expected:
                self.assert_tree_equal(actual[key], expected[key])
        elif isinstance(expected, (list, tuple)):
            self.assertEqual(len(actual), len(expected))
            for first, second in zip(actual, expected):
                self.assert_tree_equal(first, second)
        else:
            self.assertEqual(actual, expected)

    def test_actual_save_load_roundtrips_moe_weights_moments_adam_and_clocks(self):
        source, restored = runner_fixture(), runner_fixture()
        stamp_source(source)
        self.assertEqual(len(source.opt_actor.state), len(tuple(source.actor.parameters())))
        restored.iteration, restored.sim_steps, restored.imagination_steps = 999, 1000, 1001
        restored.env.common_step_counter, restored.critic_calibration_updates = 999, 999
        restored.wall_offset, restored._collect_accumulator = 0., 0.
        with tempfile.TemporaryDirectory() as directory, patch("torch.cuda.is_available", return_value=False):
            path = Path(directory)/"moe.pt"
            source.save(path, export=False)
            payload = torch.load(path, weights_only=False)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            restored.load(path)
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)
        self.assertEqual(payload["implementation_version"], 5)
        self.assertEqual(payload["actor_type"], "moe")
        self.assertEqual(payload["actor_architecture"], actor_architecture(source.cfg))
        self.assertTrue(any(key.startswith("experts.") for key in payload["actor"]))
        self.assertTrue(any(key.startswith("router.") for key in payload["actor"]))
        self.assertTrue(any(key.startswith("norm.") for key in payload["actor"]))
        for name in (*MODULES, *OPTIMIZERS):
            self.assert_tree_equal(getattr(restored, name).state_dict(), payload[name])
        for key in ("iteration", "sim_steps", "imagination_steps", "critic_calibration_updates"):
            self.assertEqual(getattr(restored, key), payload[key])
        self.assertEqual(restored.env.common_step_counter, payload["env_common_step_counter"])
        self.assertEqual(restored._collect_accumulator, payload["collect_accumulator"])
        self.assertEqual(restored.wall_offset, payload["wall_seconds"])
        self.assertEqual(restored.actor_initialization_history, source.actor_initialization_history)
        self.assertEqual(restored.amp_norm.eps, .04)
        self.assertEqual(restored.amp_norm.clip, 3.)
        self.assert_tree_equal(torch.get_rng_state(), payload["torch_rng"])
        self.assert_tree_equal(np.random.get_state(), payload["numpy_rng"])
        self.assert_tree_equal(random.getstate(), payload["python_rng"])

    def test_hiking_actual_runner_checkpoint_roundtrip_and_mlp_warmstart_rejection(self):
        options = dict(moe_variant="hiking", moe_top_k=4, moe_gate_hidden=(256,),
                       moe_load_balance_coef=0., moe_router_z_coef=0.)
        source, restored = runner_fixture(**options), runner_fixture(**options)
        stamp_source(source)
        with tempfile.TemporaryDirectory() as directory, patch("torch.cuda.is_available", return_value=False):
            path = Path(directory)/"hiking.pt"
            source.save(path, export=False)
            payload = torch.load(path, weights_only=False)
            self.assertEqual(payload["implementation_version"], 6)
            self.assertEqual(payload["actor_architecture"]["moe_variant"], "hiking")
            restored.load(path)
            for name in (*MODULES, *OPTIMIZERS):
                self.assert_tree_equal(getattr(restored, name).state_dict(), payload[name])
            observations = torch.randn(5, 384, dtype=torch.float64)
            torch.testing.assert_close(restored.actor(observations), source.actor(observations), rtol=0, atol=0)
            mlp = runner_fixture("mlp")
            mlp_path = Path(directory)/"mlp.pt"
            mlp.save(mlp_path, export=False)
            before = mutable_snapshot(restored)
            with self.assertRaisesRegex(ValueError, "Hiking warmstart requires"):
                restored.initialize_from_policy(mlp_path)
            self.assert_tree_equal(mutable_snapshot(restored), before)

    def test_incompatible_architectures_reject_in_both_directions_before_any_mutation(self):
        pairs = (
            (("mlp", {}), ("moe", {})),
            (("moe", {}), ("moe", {"moe_variant": "hiking", "moe_top_k": 4,
                                     "moe_gate_hidden": (), "moe_load_balance_coef": 0.,
                                     "moe_router_z_coef": 0.})),
            (("moe", {"moe_variant": "hiking", "moe_top_k": 4,
                       "moe_gate_hidden": (), "moe_load_balance_coef": 0., "moe_router_z_coef": 0.}),
             ("moe", {"moe_variant": "hiking", "moe_top_k": 4,
                       "moe_gate_hidden": (256,), "moe_load_balance_coef": 0., "moe_router_z_coef": 0.})),
            (("moe", {}), ("moe", {"moe_num_experts": 5})),
            (("moe", {}), ("moe", {"moe_top_k": 3})),
            (("moe", {}), ("moe", {"moe_gate_hidden": (7,)})),
            (("moe", {}), ("moe", {"moe_temperature": .7})),
        )
        for first, second in pairs:
            for source_spec, target_spec in ((first, second), (second, first)):
                with self.subTest(source=source_spec, target=target_spec), \
                     tempfile.TemporaryDirectory() as directory, \
                     patch("torch.cuda.is_available", return_value=False):
                    source = runner_fixture(source_spec[0], **source_spec[1])
                    restored = runner_fixture(target_spec[0], **target_spec[1])
                    exercise_actor_adam(restored)
                    restored.iteration, restored.sim_steps = 1234, 5678
                    # Recovery authorization must not bypass the separate
                    # architecture guard, even if both recovery modes differ.
                    restored.cfg.recovery_counter_semantics = "soft_v1"
                    restored.cfg.recovery_height_reward = "height_only_v1"
                    restored.cfg.allow_recovery_semantics_migration = True
                    path = Path(directory)/"source.pt"
                    source.save(path, export=False)
                    digest = hashlib.sha256(path.read_bytes()).hexdigest()
                    before = mutable_snapshot(restored)
                    with self.assertRaisesRegex(ValueError, "Actor architecture mismatch"):
                        restored.load(path, allow_recovery_semantics_migration=True)
                    self.assert_tree_equal(mutable_snapshot(restored), before)
                    self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)

    def test_actual_mlp_warmstart_clones_experts_but_keeps_other_networks_and_adams_fresh(self):
        source, restored = runner_fixture("mlp"), runner_fixture()
        stamp_source(source)
        restored.cfg.init_discriminator = False
        restored.cfg.moe_init_noise_std = 0.
        restored.iteration = restored.sim_steps = restored.imagination_steps = 0
        other_names = ("world_model", "critic", "discriminator", "amp_norm", *OPTIMIZERS)
        fresh = copy.deepcopy({name: getattr(restored, name).state_dict() for name in other_names})
        fresh_router = copy.deepcopy(restored.actor.router.state_dict())
        self.assertTrue(source.opt_actor.state)
        self.assertTrue(source.opt_model.state)
        self.assertTrue(source.opt_critic.state)
        self.assertEqual(restored.opt_actor.state_dict()["state"], {})
        with tempfile.TemporaryDirectory() as directory, patch("torch.cuda.is_available", return_value=False):
            path = Path(directory)/"mlp_source.pt"
            source.save(path, export=False)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            restored.initialize_from_policy(path)
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)
            provenance = restored.actor_initialization_history[-1]
            self.assertEqual(provenance["source_checkpoint"], str(path.resolve()))
            self.assertEqual(provenance["source_sha256"], digest)
        for expert in restored.actor.experts:
            self.assert_tree_equal(expert.state_dict(), source.actor.net.state_dict())
        self.assert_tree_equal(restored.actor.norm.state_dict(), source.actor.norm.state_dict())
        self.assert_tree_equal(restored.actor.router.state_dict(), fresh_router)
        for name in other_names:
            self.assert_tree_equal(getattr(restored, name).state_dict(), fresh[name])
        self.assertEqual((restored.iteration, restored.sim_steps, restored.imagination_steps), (0, 0, 0))
        self.assertEqual(provenance["source_architecture"], actor_architecture(source.cfg))
        self.assertEqual(provenance["target_architecture"], actor_architecture(restored.cfg))
        self.assertEqual(provenance["expert_head_noise_std"], 0.)
        self.assertFalse(provenance["actor_optimizer_loaded"])
        semantic_record = restored.recovery_semantics_history[-1]
        self.assertEqual(semantic_record["source_sha256"], digest)
        self.assertEqual(semantic_record["source_iteration"], source.iteration)
        self.assertEqual(semantic_record["scope"], "actor_only")
        self.assertFalse(semantic_record["critic_model_optimizer_states_loaded"])
        observations = torch.randn(9, 384, dtype=torch.float64)
        torch.testing.assert_close(restored.actor(observations), source.actor(observations),
                                   rtol=2e-12, atol=2e-13)


if __name__ == "__main__":
    unittest.main()
