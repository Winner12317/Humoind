"""Real perceptive GenCoord heads and explicit legacy-checkpoint migration."""
from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
from pathlib import Path
import tempfile
from unittest.mock import patch
import unittest

import torch
from torch import nn

from src.wmac.state import quat_apply, quat_from_rotvec, quat_inv, quat_to_matrix
from src.wmac_perceptive.config import GENCOORD_DYNAMICS, LEGACY_DYNAMICS
from src.wmac_perceptive.models import TerrainPerceptiveWorldModel
from test_perceptive_models import PlaneOracle, SMALL, state_batch
from test_perceptive_runner_restore import restore_fixture, snapshot


def model_fixture(**kwargs):
    return TerrainPerceptiveWorldModel(PlaneOracle(), SMALL, hidden=(24,),
        dynamics_parameterization=GENCOORD_DYNAMICS, **kwargs).double()


def inputs_fixture(prefix=(2,)):
    state = state_batch(1).expand(*prefix, 71).clone()
    state[..., 3:7] = quat_from_rotvec(state.new_tensor([.2, -.3, .7])).expand(*prefix, 4)
    state[..., 7:10] = state.new_tensor([.6, .1, -.03])
    state[..., 10:13] = state.new_tensor([.1, -.2, .3])
    state[..., 42:71] = .2
    action = torch.zeros(*prefix, 29, dtype=state.dtype)
    context = torch.zeros(*prefix, 35, dtype=state.dtype)
    frame = torch.rand(*prefix, *SMALL.frame_shape, dtype=state.dtype)
    return state, action, context, frame


class PerceptiveGenCoordModelTest(unittest.TestCase):
    def test_full_body_features_and_raw_velocity_head_have_explicit_shapes(self):
        model = model_fixture()
        state, action, context, frame = inputs_fixture((2, 3))
        values = model.inputs(state, action, context, frame)
        self.assertEqual(values.shape, (2, 3, 68+29+35+64+3))
        self.assertEqual(model.net[-1].out_features, 35)
        self.assertTrue(any(isinstance(layer, nn.GELU) for layer in model.net))
        self.assertFalse(any(isinstance(layer, nn.SiLU) for layer in model.net))
        inv_q = quat_inv(state[..., 3:7])
        down = torch.zeros_like(state[..., 7:10]); down[..., 2] = -1
        expected = torch.cat(((state[..., 2]-model.terrain_oracle.height_at(state[..., :2])).unsqueeze(-1),
            quat_apply(inv_q, down), quat_apply(inv_q, state[..., 7:10]),
            quat_apply(inv_q, state[..., 10:13]), state[..., 13:]), -1)
        torch.testing.assert_close(values[..., :68], expected, rtol=1e-13, atol=1e-13)
        predicted, contact, future = model.step_with_perception(state, action, context, frame)
        self.assertEqual(predicted.shape, state.shape)
        self.assertEqual(contact.shape, (2, 3, 2))
        self.assertEqual(future.shape, frame.shape)
        self.assertTrue(bool(torch.isfinite(predicted).all()))
        self.assertTrue(bool((contact >= 0).all()))
        self.assertTrue(bool(((future >= 0) & (future <= 1)).all()))

    def test_zero_head_advances_pose_and_keeps_world_velocities_without_output_moments(self):
        model = model_fixture()
        state, action, context, frame = inputs_fixture()
        self.assertFalse(hasattr(model, "out_norm"))
        self.assertFalse(hasattr(model, "error_floor"))
        self.assertFalse(any(key.startswith("out_norm.") for key in model.state_dict()))
        # Updating input statistics must never turn a zero physical head into
        # a learned mean velocity increment.
        next_state = state.clone(); next_state[..., 7:13] += 3.
        model.update_normalizers(state, action, context, next_state, frame)
        self.assertGreater(float(model.in_norm.count), 0)
        predicted = model.step(state, action, context, frame)
        torch.testing.assert_close(predicted[..., :3], state[..., :3]+state[..., 7:10]*.02,
                                   rtol=1e-13, atol=1e-13)
        torch.testing.assert_close(predicted[..., 13:42], state[..., 13:42]+state[..., 42:71]*.02,
                                   rtol=1e-13, atol=1e-13)
        torch.testing.assert_close(predicted[..., 7:13], state[..., 7:13], rtol=1e-13, atol=1e-13)
        torch.testing.assert_close(predicted[..., 42:71], state[..., 42:71], rtol=0, atol=0)
        self.assertGreater(float((predicted[..., 3:7]-state[..., 3:7]).abs().sum()), 0.)
        self.assertEqual(model.targets(state, next_state).shape, (2, 35))

    def test_model_control_duration_and_decimation_reach_the_integrator(self):
        for dt, decimation in ((.02, 4), (.04, 2), (.01, 1)):
            with self.subTest(dt=dt, decimation=decimation):
                model = model_fixture(dt=dt, decimation=decimation)
                state, action, context, frame = inputs_fixture()
                delta = state.new_zeros(35)
                delta[:3] = state.new_tensor([.3, -.1, .2]); delta[6:] = .4
                with torch.no_grad(): model.net[-1].bias.copy_(delta)
                predicted = model.step(state, action, context, frame)
                coeff = (decimation+1)/(2*decimation)
                world_dv = quat_apply(state[..., 3:7], delta[:3].expand(2, 3))
                torch.testing.assert_close(predicted[..., :3], state[..., :3]+dt*(state[..., 7:10]+coeff*world_dv),
                                           rtol=1e-13, atol=1e-13)
                torch.testing.assert_close(predicted[..., 7:10], state[..., 7:10]+world_dv,
                                           rtol=1e-13, atol=1e-13)
                torch.testing.assert_close(predicted[..., 13:42], state[..., 13:42]+dt*(state[..., 42:71]+coeff*.4),
                                           rtol=1e-13, atol=1e-13)
                torch.testing.assert_close(predicted[..., 42:71], state[..., 42:71]+.4, rtol=0, atol=0)

    def test_integrated_model_loss_uses_official_pose_and_joint_channels(self):
        model = model_fixture()
        state, _, _, _ = inputs_fixture()
        predicted = state.clone()
        predicted[..., :3] += predicted.new_tensor([.01, -.02, .03])
        predicted[..., 3:7] = quat_from_rotvec(predicted.new_tensor([-.3, .2, .1])).expand(2, 4)
        predicted[..., 7:13] += 100.  # No direct root twist loss upstream.
        predicted[..., 13:42] += .1; predicted[..., 42:71] += .5
        rotation_error = quat_to_matrix(predicted[..., 3:7])-quat_to_matrix(state[..., 3:7])
        expected = (((predicted[..., :3]-state[..., :3])/.01).square().sum(-1)
            + .5*rotation_error.square().sum((-2, -1))/.02**2
            + ((predicted[..., 13:42]-state[..., 13:42])/.1).square().sum(-1)
            + ((predicted[..., 42:71]-state[..., 42:71])/3.14).square().sum(-1))/70
        torch.testing.assert_close(model.loss(predicted, state), expected, rtol=1e-13, atol=1e-13)
        sign_flipped = predicted.clone(); sign_flipped[..., 3:7].neg_()
        torch.testing.assert_close(model.loss(sign_flipped, state), expected, rtol=1e-13, atol=1e-13)

    def test_real_physical_contact_and_depth_heads_retain_input_and_parameter_gradients(self):
        torch.manual_seed(893)
        model = model_fixture()
        with torch.no_grad():
            model.net[-1].weight.normal_(0., .01)
            model.contact_net[-1].weight.normal_(0., .01)
        state, action, context, frame = [value.requires_grad_() for value in inputs_fixture()]
        predicted, contact, future = model.step_with_perception(state, action, context, frame)
        loss = model.loss(predicted, state.detach()).mean()+contact.square().mean()+future.square().mean()
        loss.backward()
        for label, value in (("state", state), ("action", action), ("context", context), ("image", frame)):
            with self.subTest(input=label):
                self.assertIsNotNone(value.grad)
                self.assertTrue(bool(torch.isfinite(value.grad).all()))
                self.assertGreater(float(value.grad.abs().sum()), 0.)
        for name in ("net", "contact_net", "depth_net", "depth_encoder"):
            gradients = [p.grad for p in getattr(model, name).parameters()]
            with self.subTest(module=name):
                self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in gradients))
                self.assertGreater(sum(float(g.abs().sum()) for g in gradients), 0.)

    def test_legacy_model_retains_the_71_feature_70_normalized_residual_contract(self):
        model = TerrainPerceptiveWorldModel(PlaneOracle(), SMALL, hidden=(24,),
            dynamics_parameterization=LEGACY_DYNAMICS).double()
        state, action, context, frame = inputs_fixture()
        self.assertEqual(model.inputs(state, action, context, frame).shape[-1], 71+29+35+64+3)
        self.assertEqual(model.net[-1].out_features, 70)
        self.assertTrue(hasattr(model, "out_norm"))
        self.assertTrue(any(isinstance(layer, nn.SiLU) for layer in model.net))
        # The old zero residual remains unchanged for strict historical resume.
        torch.testing.assert_close(model.step(state, action, context, frame), state, rtol=1e-13, atol=1e-13)


def migration_receiver(steps=2, *, migrate=True):
    runner = restore_fixture(steps)
    runner.cfg = replace(runner.cfg, dynamics_parameterization=GENCOORD_DYNAMICS,
                         migrate_world_model=migrate)
    runner.cfg.validate()
    if not migrate:
        runner.world_model = TerrainPerceptiveWorldModel(runner.adapter.terrain, runner.spec,
            hidden=runner.cfg.model_hidden, dynamics_parameterization=GENCOORD_DYNAMICS,
            dt=runner.cfg.dynamics_dt, decimation=runner.cfg.dynamics_decimation).double()
        runner.opt_model = torch.optim.Adam(runner.world_model.parameters(),
                                           lr=runner.cfg.world_model_lr, fused=False)
    return runner


class PerceptiveGenCoordMigrationTest(unittest.TestCase):
    def assert_tree_equal(self, actual, expected):
        if isinstance(expected, torch.Tensor):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        elif isinstance(expected, dict):
            self.assertEqual(actual.keys(), expected.keys())
            for key in expected: self.assert_tree_equal(actual[key], expected[key])
        elif isinstance(expected, (list, tuple)):
            self.assertEqual(len(actual), len(expected))
            for left, right in zip(actual, expected): self.assert_tree_equal(left, right)
        elif hasattr(expected, "dtype") and hasattr(expected, "shape"):
            import numpy as np
            np.testing.assert_array_equal(actual, expected)
        else: self.assertEqual(actual, expected)

    def test_actual_migration_preserves_actor_amp_and_adam_but_rebuilds_world_model_and_critic(self):
        source, restored = restore_fixture(1), migration_receiver(2)
        source.iteration = 900  # Already beyond the absolute 500-step warmup.
        source._ensure_training_progress()
        source.actor_updates_accepted, source.actor_updates_rejected = 55, 7
        source._last_actor_update_iteration = 899
        source.perception_training_history = [{"kind": "source_history"}]
        old_modules = restored.world_model, restored.critic
        old_optimizers = restored.opt_model, restored.opt_critic
        old_parameter_ids = {id(parameter) for module in old_modules for parameter in module.parameters()}
        with tempfile.TemporaryDirectory() as directory, patch("torch.cuda.is_available", return_value=False):
            path = Path(directory)/"legacy.pt"
            source.save(path, export=False)
            original_bytes = path.read_bytes()
            payload = torch.load(path, map_location="cpu", weights_only=False)
            fresh = torch.full_like(restored.current_depth, .35); fresh[:, 1].fill_(1.)
            with patch("src.wmac_perceptive.runner.read_depth_frame", return_value=fresh):
                restored.load(path)
            self.assertEqual(path.read_bytes(), original_bytes)
        for name in ("actor", "discriminator", "amp_norm", "opt_actor", "opt_disc"):
            with self.subTest(preserved=name):
                self.assert_tree_equal(getattr(restored, name).state_dict(), payload[name])
        for name, old in zip(("world_model", "critic"), old_modules):
            with self.subTest(rebuilt=name):
                self.assertIsNot(getattr(restored, name), old)
        for name, old in zip(("opt_model", "opt_critic"), old_optimizers):
            with self.subTest(reset=name):
                optimizer = getattr(restored, name)
                self.assertIsNot(optimizer, old)
                self.assertEqual(len(optimizer.state), 0)
        new_parameter_ids = {id(parameter) for module in (restored.world_model, restored.critic)
                             for parameter in module.parameters()}
        self.assertTrue(old_parameter_ids.isdisjoint(new_parameter_ids))
        self.assertEqual({id(p) for group in restored.opt_model.param_groups for p in group["params"]},
                         {id(p) for p in restored.world_model.parameters()})
        self.assertEqual({id(p) for group in restored.opt_critic.param_groups for p in group["params"]},
                         {id(p) for p in restored.critic.parameters()})
        self.assertEqual(restored.world_model.prediction_dim, 35)
        self.assertFalse(hasattr(restored.world_model, "out_norm"))
        self.assertEqual(float(restored.world_model.in_norm.count), 0.)
        self.assertEqual(float(restored.critic.norm.count), 0.)
        self.assertEqual(float(restored.world_model.net[-1].weight.abs().sum()), 0.)
        self.assertEqual(float(restored.world_model.net[-1].bias.abs().sum()), 0.)
        critic_inputs = torch.randn(2, 217, dtype=torch.float64)
        torch.testing.assert_close(restored._value_fn(critic_inputs), restored.critic(critic_inputs), rtol=0, atol=0)
        self.assertEqual(restored.iteration, payload["iteration"])
        self.assertEqual(restored.critic_calibration_updates, 0)
        self.assertEqual(restored.actor_updates_accepted, 0)
        self.assertEqual(restored.actor_updates_rejected, 0)
        self.assertIsNone(restored._last_actor_update_iteration)
        self.assertEqual(restored.world_model_warmup_until, 900+restored.cfg.model_warmup_iterations)
        self.assertFalse(torch.isfinite(torch.tensor(restored.validation["validation_model_error"])))
        self.assertEqual(restored.perception_training_history[:-1], payload["perception_training_history"])
        record = restored.perception_training_history[-1]
        self.assertEqual(record["world_model_migration"]["from_parameterization"], LEGACY_DYNAMICS)
        self.assertEqual(record["world_model_migration"]["to_parameterization"], GENCOORD_DYNAMICS)
        self.assertEqual(record["reset_modules"], ["world_model", "critic"])
        self.assertEqual(record["reset_optimizers"], ["opt_model", "opt_critic"])
        self.assertEqual(record["source_sha256"], hashlib.sha256(original_bytes).hexdigest())
        self.assertEqual(record["source_iteration"], 900)

    def test_warmup_anchor_blocks_high_iteration_actor_and_survives_new_checkpoint_resume(self):
        source, restored = restore_fixture(1), migration_receiver(2)
        source.iteration = 900
        with tempfile.TemporaryDirectory() as directory, patch("torch.cuda.is_available", return_value=False):
            path, new_path = Path(directory)/"legacy.pt", Path(directory)/"gencoord.pt"
            source.save(path, export=False)
            with patch("src.wmac_perceptive.runner.read_depth_frame", return_value=restored.current_depth):
                restored.load(path)
            until = restored.world_model_warmup_until
            restored.validation = {"validation_model_error": 0., "validation_contact_error": 0.,
                "validation_walk_error": 0., "validation_recovery_error": 0.,
                "validation_walk_samples": 100., "validation_recovery_samples": 100.,
                "validation_walk_depth_rmse": 0., "validation_recovery_depth_rmse": 0.,
                "validation_depth_mask_error": 0.}
            row = {"wm_gate_error": 0., "wm_updates_applied_fraction": 1.}
            for iteration in (900, until-restored.cfg.critic_warmup_iterations-1):
                restored.iteration = iteration
                gated = dict(row)
                self.assertFalse(restored._actor_gate(gated))
                self.assertEqual(gated["gate_reason"], "world_model_warmup")
                self.assertEqual(restored._check_actor_stall(), 0)
            restored.iteration = until-restored.cfg.critic_warmup_iterations
            self.assertTrue(restored._actor_gate(dict(row)))  # Critic calibration can start.
            needs_calibration = restored.critic_calibration_updates < restored.cfg.critic_warmup_iterations
            self.assertTrue(needs_calibration)
            self.assertFalse(restored.iteration >= restored.cfg.model_warmup_iterations
                             and restored._actor_gate(dict(row)) and not needs_calibration)
            restored.critic_calibration_updates = restored.cfg.critic_warmup_iterations
            self.assertFalse(restored._actor_gate(dict(row)))  # Still before the new anchor.
            restored.iteration = until
            self.assertTrue(restored._actor_gate(dict(row)))
            restored.save(new_path, export=False)
            payload = torch.load(new_path, map_location="cpu", weights_only=False)
            self.assertEqual(payload["world_model_warmup_until"], until)
            resumed = migration_receiver(3, migrate=False)
            with patch("src.wmac_perceptive.runner.read_depth_frame", return_value=resumed.current_depth):
                resumed.load(new_path)
            self.assertEqual(resumed.world_model_warmup_until, until)
            self.assertEqual(resumed.perception_training_history, restored.perception_training_history)
            self.assert_tree_equal(resumed.world_model.state_dict(), payload["world_model"])

    def test_malformed_anchor_or_preserved_adam_rejects_before_rebuild_or_state_mutation(self):
        source, restored = restore_fixture(1), migration_receiver(2)
        with tempfile.TemporaryDirectory() as directory, patch("torch.cuda.is_available", return_value=False):
            original = Path(directory)/"legacy.pt"; source.save(original, export=False)
            payload = torch.load(original, map_location="cpu", weights_only=False)
            for problem in ("bool", "negative", "float", "future", "adam_shape", "adam_nan"):
                bad = copy.deepcopy(payload)
                if problem == "bool": bad["world_model_warmup_until"] = True
                elif problem == "negative": bad["world_model_warmup_until"] = -1
                elif problem == "float": bad["world_model_warmup_until"] = 4.5
                elif problem == "future":
                    bad["world_model_warmup_until"] = bad["iteration"]+restored.cfg.model_warmup_iterations+1
                else:
                    moment = next(iter(bad["opt_disc"]["state"].values()))
                    if problem == "adam_shape": moment["exp_avg"] = moment["exp_avg"].new_zeros(1)
                    else: moment["exp_avg"].reshape(-1)[0] = float("nan")
                path = Path(directory)/f"bad_{problem}.pt"; torch.save(bad, path)
                before = snapshot(restored)
                old = restored.world_model, restored.critic, restored.opt_model, restored.opt_critic
                with self.subTest(problem=problem), self.assertRaises(ValueError): restored.load(path)
                self.assert_tree_equal(snapshot(restored), before)
                for current, previous in zip((restored.world_model, restored.critic, restored.opt_model, restored.opt_critic), old):
                    self.assertIs(current, previous)

    def test_migrated_resume_rejects_lost_or_changed_anchor_atomically_but_accepts_expired_anchor(self):
        source, migrated = restore_fixture(1), migration_receiver(2)
        source.iteration = 900
        with tempfile.TemporaryDirectory() as directory, patch("torch.cuda.is_available", return_value=False):
            legacy_path, migrated_path = Path(directory)/"legacy.pt", Path(directory)/"migrated.pt"
            source.save(legacy_path, export=False)
            with patch("src.wmac_perceptive.runner.read_depth_frame", return_value=migrated.current_depth):
                migrated.load(legacy_path)
            migrated.save(migrated_path, export=False)
            payload = torch.load(migrated_path, map_location="cpu", weights_only=False)
            until = payload["world_model_warmup_until"]
            # Other training records can follow a migration; the last physical
            # migration, rather than the last history item, owns the anchor.
            payload["perception_training_history"].insert(0, {
                "kind": "world_model_parameterization_migration", "source_iteration": 100,
                "world_model_warmup_until": 600})
            payload["perception_training_history"].append({"kind": "later_training_note"})
            restored = migration_receiver(3, migrate=False)
            for problem in ("missing", "none", "mismatch", "history_bool", "history_float", "history_before_source"):
                bad = copy.deepcopy(payload)
                latest = bad["perception_training_history"][-2]
                if problem == "missing": bad.pop("world_model_warmup_until")
                elif problem == "none": bad["world_model_warmup_until"] = None
                elif problem == "mismatch": bad["world_model_warmup_until"] = until-1
                elif problem == "history_bool": latest["world_model_warmup_until"] = True
                elif problem == "history_float": latest["world_model_warmup_until"] = float(until)
                else: latest["world_model_warmup_until"] = latest["source_iteration"]-1
                path = Path(directory)/f"migrated_bad_{problem}.pt"; torch.save(bad, path)
                before = snapshot(restored)
                old = restored.world_model, restored.critic, restored.opt_model, restored.opt_critic
                anchor_before = getattr(restored, "world_model_warmup_until", None)
                history_before = copy.deepcopy(getattr(restored, "perception_training_history", None))
                with self.subTest(problem=problem), self.assertRaisesRegex(ValueError, "warmup"):
                    restored.load(path)
                self.assert_tree_equal(snapshot(restored), before)
                self.assertEqual(getattr(restored, "world_model_warmup_until", None), anchor_before)
                self.assertEqual(getattr(restored, "perception_training_history", None), history_before)
                for current, previous in zip((restored.world_model, restored.critic, restored.opt_model, restored.opt_critic), old):
                    self.assertIs(current, previous)
            expired = copy.deepcopy(payload)
            expired["iteration"] = until+7
            expired_path = Path(directory)/"expired_anchor.pt"; torch.save(expired, expired_path)
            with patch("src.wmac_perceptive.runner.read_depth_frame", return_value=restored.current_depth):
                restored.load(expired_path)
            self.assertEqual(restored.world_model_warmup_until, until)
            self.assertGreater(restored.iteration, restored.world_model_warmup_until)
            self.assert_tree_equal(restored.world_model.state_dict(), payload["world_model"])
            self.assertEqual(restored.perception_training_history, payload["perception_training_history"])


if __name__ == "__main__":
    unittest.main()
