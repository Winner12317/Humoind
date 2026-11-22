"""CPU protocol checks for bounded official details and flat AMP preservation."""
from __future__ import annotations

import copy
from dataclasses import asdict
import hashlib
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from src.wmac.actors import actor_architecture, create_actor
from src.wmac.checkpoints import resolve_checkpoint_config
from src.wmac.config import WmacConfig
from src.wmac.networks import Critic, WorldModel
from src.wmac.optimizers import build_reference_optimizer
from src.wmac.reference import amp_value_upper_bound, checkpoint_protocol, config_protocol
from test_wmac_optimization import make_runner
from test_wmac_schedule_restore import checkpoint_runner


MODULES = ("actor", "critic", "world_model", "discriminator", "amp_norm")
OPTIMIZERS = ("opt_actor", "opt_critic", "opt_model", "opt_disc")


def protocol_runner(*, official=True, hybrid=False, checkpoint=False):
    if checkpoint:
        runner = checkpoint_runner()
        batch = noises = None
    else:
        runner, batch, noises = make_runner(batch_size=5, horizon=3)
    cfg = runner.cfg
    cfg.reference_alignment = "official_details_v1" if official else "legacy_v1"
    cfg.optimizer_algorithm = "muon_adam" if hybrid else "adam"
    cfg.mirror_walk_motions = False  # No filesystem dataset is needed by this fixture.
    if official:
        cfg.policy_output_init_scale, cfg.policy_output_zero_bias = 1., False
    runner.actor = create_actor(cfg).double()
    runner.world_model = WorldModel(hidden=cfg.model_hidden,
        activation="gelu" if official else "silu", output_zero_init=not official).double()
    runner.critic = Critic(136, hidden=cfg.critic_hidden,
        activation="gelu" if official else "elu", layer_norm=official).double()
    runner._value_fn = runner.critic.forward
    runner._critic_vmax = amp_value_upper_bound(runner.adapter, cfg) if official else None
    with torch.no_grad():
        runner.world_model.net[-1].weight.normal_(0., .002)
        runner.world_model.net[-1].bias.zero_()
    for name, module_name, rate in (("opt_actor", "actor", cfg.actor_lr),
                                    ("opt_critic", "critic", cfg.critic_lr),
                                    ("opt_model", "world_model", cfg.world_model_lr)):
        module = getattr(runner, module_name)
        optimizer = (build_reference_optimizer([module], rate,
                     betas=cfg.actor_betas if name == "opt_actor" else (.9, .999), device="cpu")
                     if hybrid else torch.optim.Adam(module.parameters(), lr=rate,
                         betas=cfg.actor_betas if name == "opt_actor" else (.9, .999)))
        setattr(runner, name, optimizer)
    return runner, batch, noises


def mutable_state(runner):
    return copy.deepcopy({
        "modules": {name: getattr(runner, name).state_dict() for name in MODULES},
        "optimizers": {name: getattr(runner, name).state_dict() for name in OPTIMIZERS},
        "clocks": {name: getattr(runner, name) for name in
                   ("iteration", "sim_steps", "imagination_steps", "wall_offset",
                    "_collect_accumulator", "critic_calibration_updates")},
        "env_step": runner.env.common_step_counter,
        "counter_semantics": runner.adapter.recovery_counter_semantics,
        "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(),
    })


class OfficialProtocolTest(unittest.TestCase):
    def assert_tree_equal(self, before, after):
        if isinstance(before, torch.Tensor):
            torch.testing.assert_close(before, after, rtol=0, atol=0)
        elif isinstance(before, np.ndarray):
            np.testing.assert_array_equal(before, after)
        elif isinstance(before, dict):
            self.assertEqual(set(before), set(after))
            for key in before:
                self.assert_tree_equal(before[key], after[key])
        elif isinstance(before, (tuple, list)):
            self.assertEqual(type(before), type(after))
            self.assertEqual(len(before), len(after))
            for first, second in zip(before, after):
                self.assert_tree_equal(first, second)
        else:
            self.assertEqual(before, after)

    def test_official_profile_maps_eight_state_frames_to_seven_transitions(self):
        cfg = WmacConfig.for_profile("supermimic_official")
        cfg.validate()
        expected = {
            "task": "Unitree-G1-AMP-Flat-Aligned", "amp_feature_schema": "grounded_v2",
            "reference_alignment": "official_details_v1", "optimizer_algorithm": "muon_adam",
            "mirror_walk_motions": True, "num_envs": 4096, "replay_steps": 1000,
            "wm_window": 7, "wm_updates": 10, "wm_batch_size": 2048,
            "actor_batch_size": 4096, "actor_updates": 1, "imagination_horizon": 32,
            "actor_lr": 2e-4, "critic_lr": 2e-4, "world_model_lr": 2e-4,
            "actor_betas": (.9, .95), "actor_grad_value_clip": .1, "max_grad_norm": 1.,
            "model_warmup_iterations": 0, "critic_warmup_iterations": 0, "warmup_steps": 8,
            "collect_steps": 0, "utd": .4, "real_critic_coef": 0., "exploration_noise": .1,
            "recovery_counter_semantics": "hard_forward_v2", "recovery_height_reward": "upright_v2",
        }
        for name, value in expected.items():
            self.assertEqual(getattr(cfg, name), value, name)
        self.assertEqual(cfg.wm_window+1, 8)
        self.assertEqual(cfg.model_hidden, (2048, 2048, 2048))
        self.assertEqual(cfg.policy_hidden, (2048, 1024, 512, 256))
        self.assertEqual(cfg.critic_hidden, (256, 128))
        self.assertEqual((cfg.critic_epochs, cfg.critic_minibatches), (5, 4))
        architecture = actor_architecture(cfg)
        self.assertEqual((architecture["activation"], architecture["layer_norm"],
                          architecture["action_limit"]), ("gelu", True, None))

    def test_existing_profiles_keep_historical_alignment_sampling_and_architecture(self):
        for profile in ("balanced", "supermimic", "legacy", "ppo_aligned"):
            cfg = WmacConfig.for_profile(profile)
            self.assertEqual(config_protocol(cfg), dict(reference_alignment="legacy_v1",
                             optimizer_algorithm="adam", mirror_walk_motions=False))
            self.assertEqual(cfg.wm_window, 8)
            architecture = actor_architecture(cfg)
            self.assertNotIn("activation", architecture)
            self.assertNotIn("layer_norm", architecture)
        cfg = WmacConfig.for_profile("ppo_aligned")
        self.assertEqual((cfg.actor_lr, cfg.actor_betas, cfg.model_warmup_iterations,
                          cfg.critic_warmup_iterations), (1e-3, (.9, .999), 500, 20))

    def test_unlabelled_checkpoint_cannot_inherit_new_official_protocol(self):
        old = {"format": "amp-wmac-v1", "config": {"profile": "supermimic"}}
        cfg = resolve_checkpoint_config(old)
        self.assertEqual(config_protocol(cfg), dict(reference_alignment="legacy_v1",
                         optimizer_algorithm="adam", mirror_walk_motions=False))
        self.assertEqual((cfg.recovery_counter_semantics, cfg.recovery_height_reward),
                         ("soft_v1", "height_only_v1"))
        for missing in ("training_protocol", "optimizer_algorithm", "reference_alignment"):
            payload = {"format": "amp-wmac-v1", "implementation_version": 7,
                       "config": asdict(WmacConfig.for_profile("supermimic_official")),
                       "training_protocol": config_protocol(WmacConfig.for_profile("supermimic_official"))}
            if missing == "training_protocol":
                payload.pop(missing)
            else:
                payload["config"].pop(missing)
            with self.assertRaisesRegex(ValueError, "Version7 checkpoint"):
                checkpoint_protocol(payload)

    def test_official_learning_rates_stay_constant_while_amp_discriminator_keeps_schedule(self):
        runner, _, _ = protocol_runner(hybrid=True)
        runner.cfg.actor_lr, runner.cfg.critic_lr = .01, .02
        runner.cfg.world_model_lr, runner.cfg.discriminator_lr = .03, .04
        runner.iteration = runner.cfg.lr_decay_iterations // 2
        pointers = {name: [group["lr"].data_ptr() for group in getattr(runner, name).param_groups]
                    for name in ("opt_actor", "opt_critic", "opt_model")}
        runner._set_learning_rates()
        for name, expected in (("opt_actor", .01), ("opt_critic", .02), ("opt_model", .03)):
            self.assertEqual(pointers[name], [group["lr"].data_ptr() for group in getattr(runner, name).param_groups])
            for group in getattr(runner, name).param_groups:
                self.assertAlmostEqual(float(group["lr"]), expected, places=7)
        self.assertAlmostEqual(runner.opt_disc.param_groups[0]["lr"], .04*runner._lr_factor(runner.iteration))

    def test_official_gate_retains_finite_step_protection_without_validation_error_gate(self):
        runner, _, _ = protocol_runner()
        runner.validation = dict(validation_model_error=float("inf"), validation_contact_error=float("inf"))
        row = dict(wm_gate_error=1e6, wm_updates_applied_fraction=1.)
        self.assertTrue(runner._actor_gate(row))
        for bad in (float("nan"), float("inf"), -float("inf")):
            self.assertFalse(runner._actor_gate({**row, "wm_gate_error": bad}))
        self.assertFalse(runner._actor_gate({**row, "wm_updates_applied_fraction": .5}))
        runner.cfg.reference_alignment = "legacy_v1"
        self.assertFalse(runner._actor_gate(row))

    def test_nonfinite_optimizer_input_cannot_change_official_policy_or_optimizer(self):
        runner, _, _ = protocol_runner(hybrid=True)
        before_model = copy.deepcopy(runner.actor.state_dict())
        before_opt = copy.deepcopy(runner.opt_actor.state_dict())
        parameter = next(runner.actor.parameters())
        loss = parameter.sum()*float("nan")
        _, accepted = runner._optimizer_step(loss, runner.opt_actor, runner.actor, actor=True)
        self.assertFalse(accepted)
        for current in runner.actor.parameters():
            current.grad = torch.full_like(current, float("inf"))
        _, accepted = runner._optimizer_step(None, runner.opt_actor, runner.actor, actor=True, backward=False)
        self.assertFalse(accepted)
        self.assert_tree_equal(before_model, runner.actor.state_dict())
        self.assert_tree_equal(before_opt, runner.opt_actor.state_dict())

    def test_critic_cap_uses_retained_amp_reward_bound_and_preserves_negative_values(self):
        term = lambda weight, **params: SimpleNamespace(weight=weight, params=params)
        adapter = SimpleNamespace(reward_scale=.02, reward_cfg={
            "track_anchor_linear_velocity": term(2., mask_delay=True, delay_env_rew_ratio=0.),
            "track_anchor_angular_velocity": term(.5),
            "track_root_height": term(2., mask_delay=True, delay_env_rew_ratio=3.),
            "body_ang_vel_xy_l2": term(.5), "joint_acc_l2": term(-.2),
        })
        cfg = WmacConfig(task_weight=.75, style_weight=.25, amp_reward_coef=.1, gamma=.9)
        self.assertAlmostEqual(amp_value_upper_bound(adapter, cfg), 1.6)
        adapter.reward_cfg["joint_acc_l2"].weight = .2
        with self.assertRaisesRegex(ValueError, "needs a bound"):
            amp_value_upper_bound(adapter, cfg)
        runner, _, _ = protocol_runner()
        runner._critic_vmax = 1.6
        values = torch.tensor([-100., 1., 1.6, 100.], dtype=torch.float64)
        torch.testing.assert_close(runner._cap_values(values), torch.tensor([-100., 1., 1.6, 1.6], dtype=torch.float64))
        runner.cfg.reference_alignment = "legacy_v1"
        self.assertIs(runner._cap_values(values), values)

    def test_imagined_targets_and_terminal_value_use_the_same_upper_bound(self):
        runner, batch, noises = protocol_runner()
        runner._critic_vmax = .2
        shape = (runner.cfg.imagination_horizon, runner.cfg.actor_batch_size)
        rewards = (torch.full(shape, 1e5, dtype=torch.float64),
                   torch.zeros(shape, dtype=torch.float64), torch.zeros(shape, dtype=torch.float64))
        with torch.no_grad(), patch.object(runner, "_trajectory_rewards", return_value=rewards):
            with patch.object(runner, "_value", side_effect=lambda x: torch.full(x.shape[:-1], 1e6, dtype=x.dtype)):
                oversized = runner.imagine(batch, noises=noises)
            with patch.object(runner, "_value", side_effect=lambda x: torch.full(x.shape[:-1], .2, dtype=x.dtype)):
                bounded = runner.imagine(batch, noises=noises)
        self.assertTrue(bool((oversized["targets"] <= .2).all()))
        torch.testing.assert_close(oversized["objective"], bounded["objective"], rtol=0, atol=0)

    def test_critic_reuses_one_permutation_across_epochs_only_for_official_protocol(self):
        for official, expected_permutations in ((True, 1), (False, 5)):
            runner, _, _ = protocol_runner(official=official)
            runner.cfg.critic_epochs, runner.cfg.critic_minibatches = 5, 4
            x = torch.zeros(2, 5, 136, dtype=torch.float64)
            x[..., 0] = torch.arange(10).reshape(2, 5)
            observed = []
            input_batches = []
            def evaluate(features):
                observed.append(features[:, 0].clone())
                input_batches.append(features)
                return runner.critic(features)
            runner._value_fn = evaluate
            with patch("torch.randperm", wraps=torch.randperm) as permutations:
                runner._fit_critic(x, torch.zeros(2, 5, dtype=torch.float64))
            self.assertEqual(permutations.call_count, expected_permutations)
            self.assertEqual(len(observed), 20)
            if official:
                for epoch in range(1, 5):
                    for chunk in range(4):
                        torch.testing.assert_close(observed[chunk], observed[4*epoch+chunk], rtol=0, atol=0)
                        self.assertIs(input_batches[chunk], input_batches[4*epoch+chunk])
                self.assertEqual(sum(map(len, observed[:4])), 8)
            else:
                self.assertEqual(sum(map(len, observed[:4])), 10)

    def test_whole_imagination_uses_one_actor_step_without_action_shift_rollback(self):
        runner, _, noises = protocol_runner()
        runner.cfg.critic_bootstrap = False
        runner.cfg.max_action_shift = 1e-30
        runner._observation_noise = lambda count: noises
        with patch.object(runner.opt_actor, "step", wraps=runner.opt_actor.step) as step:
            metrics = runner.update_actor_critic(update_actor=True)
        self.assertEqual(step.call_count, 1)
        self.assertGreater(metrics["actor_action_shift"], runner.cfg.max_action_shift)
        self.assertEqual(metrics["actor_update_rejected"], 0.)
        self.assertEqual(runner.imagination_steps, 3*5)

    def test_checkpoint_roundtrip_restores_actual_muon_adam_and_rejects_late_corruption(self):
        source, _, _ = protocol_runner(hybrid=True, checkpoint=True)
        for name in ("opt_actor", "opt_critic", "opt_model"):
            optimizer = getattr(source, name)
            for group in optimizer.param_groups:
                for parameter in group["params"]:
                    parameter.grad = torch.full_like(parameter, .01)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        with tempfile.TemporaryDirectory() as directory, patch("torch.cuda.is_available", return_value=False):
            path = Path(directory)/"source.pt"
            source.save(path, export=False)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            payload = torch.load(path, weights_only=False)
            self.assertEqual(payload["implementation_version"], 7)
            self.assertEqual(payload["training_protocol"], config_protocol(source.cfg))
            self.assertEqual(payload["optimizer_algorithms"]["opt_model"], "hybrid_supermimic_muon_adam_v1")
            restored, _, _ = protocol_runner(hybrid=True, checkpoint=True)
            restored.load(path)
            for name in (*MODULES, *OPTIMIZERS):
                self.assert_tree_equal(payload[name], getattr(restored, name).state_dict())
            self.assertEqual(restored.iteration, payload["iteration"])
            self.assertEqual(restored.sim_steps, payload["sim_steps"])
            for problem in ("missing_protocol", "changed_protocol", "late_optimizer_shape", "late_optimizer_nan"):
                bad = copy.deepcopy(payload)
                if problem == "missing_protocol":
                    bad.pop("training_protocol")
                elif problem == "changed_protocol":
                    bad["training_protocol"]["mirror_walk_motions"] = True
                else:
                    state = next(iter(bad["opt_model"]["state"].values()))
                    if problem.endswith("shape"):
                        state["exp_avg"] = torch.zeros(1)
                    else:
                        state["exp_avg"].flatten()[0] = float("nan")
                broken = Path(directory)/f"{problem}.pt"
                torch.save(bad, broken)
                before = mutable_state(restored)
                with self.assertRaises(ValueError, msg=problem):
                    restored.load(broken)
                self.assert_tree_equal(before, mutable_state(restored))
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)

    def test_same_shape_world_model_cannot_implicitly_change_activation_protocol(self):
        # Hiking Actor architecture stays identical while WM activation changes.
        # This makes the protocol guard observable independently of MLP/LN keys.
        source, _, _ = protocol_runner(official=False, checkpoint=True)
        source.cfg.actor_type, source.cfg.moe_variant = "moe", "hiking"
        source.actor = create_actor(source.cfg).double()
        source.opt_actor = torch.optim.Adam(source.actor.parameters(), lr=source.cfg.actor_lr)
        restored, _, _ = protocol_runner(official=True, checkpoint=True)
        restored.cfg.actor_type, restored.cfg.moe_variant = "moe", "hiking"
        restored.actor = create_actor(restored.cfg).double()
        restored.opt_actor = torch.optim.Adam(restored.actor.parameters(), lr=restored.cfg.actor_lr)
        self.assertEqual(actor_architecture(source.cfg), actor_architecture(restored.cfg))
        self.assertEqual(tuple(source.world_model.state_dict()), tuple(restored.world_model.state_dict()))
        for key, value in source.world_model.state_dict().items():
            self.assertEqual(value.shape, restored.world_model.state_dict()[key].shape)
        with tempfile.TemporaryDirectory() as directory, patch("torch.cuda.is_available", return_value=False):
            path = Path(directory)/"legacy.pt"
            source.save(path, export=False)
            before = mutable_state(restored)
            with self.assertRaisesRegex(ValueError, "Training protocol mismatch"):
                restored.load(path)
            self.assert_tree_equal(before, mutable_state(restored))


if __name__ == "__main__":
    unittest.main()
