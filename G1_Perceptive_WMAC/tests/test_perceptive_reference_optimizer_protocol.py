"""Reference Muon/Adam preflight and actual atomic Runner save/load."""
from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

from src.wmac_perceptive.checkpoints import validate_optimizer_states
from src.wmac_perceptive.optimizers import ReferenceMuonAdam, build_reference_optimizer
from src.wmac_perceptive.perception import DepthHistory, PerceptionSpec
from src.wmac_perceptive.runner import MODULES, OPTIMIZERS
from test_perceptive_latent_runner import latent_runner_fixture
import test_perceptive_runner_restore as restore_checks


def optimizer_fixture():
    model = nn.Sequential(nn.Linear(3, 3), nn.ELU(), nn.Linear(3, 3), nn.ELU(),
                          nn.Linear(3, 3), nn.ELU(), nn.Linear(3, 3))
    optimizer = build_reference_optimizer([model], lr=2e-4, device="cpu")
    model(torch.ones(4, 3)).square().mean().backward()
    optimizer.step(); optimizer.zero_grad(set_to_none=True)
    return model, optimizer


def reference_runner_fixture(steps):
    runner, _, _ = latent_runner_fixture(spec=PerceptionSpec())
    runner.cfg = replace(runner.cfg, execution_profile="supermimic", execution_optimizer="muon_adam",
        validation_horizon=runner.cfg.imagination_horizon, replay_storage="legacy",
        compile=False, cuda_graphs=False, actor_lr=2e-4, critic_lr=2e-4,
        world_model_lr=2e-4, actor_betas=(.9, .95), lr_final_fraction=1.)
    runner.cfg.validate()
    # Use the same optimizer factory/module order as PerceptiveRunner.__init__.
    runner.opt_actor = build_reference_optimizer([runner.actor.policy], lr=runner.cfg.actor_lr,
        betas=runner.cfg.actor_betas, device="cpu")
    runner.opt_model = build_reference_optimizer(list(runner._model_training_modules),
        lr=runner.cfg.world_model_lr, device="cpu")
    runner.opt_critic = build_reference_optimizer([runner.critic], lr=runner.cfg.critic_lr, device="cpu")
    for name in OPTIMIZERS:
        optimizer = getattr(runner, name)
        for _ in range(steps):
            for group in optimizer.param_groups:
                for parameter in group["params"]:
                    parameter.grad = torch.full_like(parameter, .01)
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
    runner._processed_history = True
    history_spec = replace(runner.spec,
        height=runner.spec.height-runner.spec.crop_top-runner.spec.crop_bottom,
        width=runner.spec.width-runner.spec.crop_left-runner.spec.crop_right,
        crop_top=0, crop_bottom=0, crop_left=0, crop_right=0, blur_kernel=1)
    runner.depth_history = DepthHistory(history_spec, runner.env.num_envs, "cpu", torch.float64)
    runner.depth_history.reset(runner._policy_depth_frame(runner.current_depth))
    runner._setup_fast_execution()
    runner.iteration, runner.sim_steps, runner.imagination_steps = 17+steps, 101+steps, 303+steps
    runner.critic_calibration_updates = steps
    return runner


def full_snapshot(runner):
    value = restore_checks.snapshot(runner)
    value["latent_decoder"] = copy.deepcopy(runner.latent_decoder.state_dict())
    value["execution_identity"] = id(runner._latent_wm_execution)
    value["progress"] = copy.deepcopy({key: getattr(runner, key) for key in (
        "actor_updates_accepted", "actor_updates_rejected", "_active_horizon",
        "_progress_start_iteration", "_last_actor_update_iteration")})
    return value


class ReferenceOptimizerPreflightTest(unittest.TestCase):
    assert_tree_equal = restore_checks.PerceptiveRunnerRestoreTest.assert_tree_equal

    def assert_pure_rejection(self, model, optimizer, bad):
        before = copy.deepcopy(optimizer.state_dict())
        weights, rng = copy.deepcopy(model.state_dict()), torch.get_rng_state().clone()
        with self.assertRaises(ValueError):
            validate_optimizer_states({"opt": bad}, {"opt": optimizer})
        self.assert_tree_equal(optimizer.state_dict(), before)
        self.assert_tree_equal(model.state_dict(), weights)
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)

    def test_real_reference_state_and_scheduled_scalar_lr_pass_without_mutation(self):
        model, optimizer = optimizer_fixture()
        saved = copy.deepcopy(optimizer.state_dict())
        self.assertIsNone(validate_optimizer_states({"opt": saved}, {"opt": optimizer}))
        for group in saved["param_groups"]:
            group["lr"].fill_(1e-5)
        self.assertIsNone(validate_optimizer_states({"opt": saved}, {"opt": optimizer}))

    def test_reference_with_only_adam_group_cannot_be_relabelled_as_native_adam(self):
        model = nn.Linear(3, 3)  # No middle Sequential weights: one Adam group only.
        reference = build_reference_optimizer([model], lr=2e-4, device="cpu")
        native = torch.optim.Adam(model.parameters(), lr=2e-4)
        before = copy.deepcopy(native.state_dict())
        with self.assertRaisesRegex(ValueError, "algorithm marker"):
            validate_optimizer_states({"opt": reference.state_dict()}, {"opt": native})
        self.assert_tree_equal(native.state_dict(), before)

    def test_algorithm_named_group_and_same_shape_order_are_strict(self):
        model, optimizer = optimizer_fixture()
        good = copy.deepcopy(optimizer.state_dict())
        for problem in ("algorithm", "missing_algorithm", "kind", "ids", "names", "decay", "lr"):
            bad = copy.deepcopy(good)
            group = bad["param_groups"][-1]
            if problem == "algorithm": bad["algorithm_name"] = "adam"
            elif problem == "missing_algorithm": bad.pop("algorithm_name")
            elif problem == "kind": group["kind"] = "adam"
            elif problem == "ids": group["params"].reverse()
            elif problem == "names": group["param_names"].reverse()
            elif problem == "decay": group["weight_decay"] = 0.
            else: group["lr"] = torch.tensor([.001])
            with self.subTest(problem=problem): self.assert_pure_rejection(model, optimizer, bad)

    def test_adam_and_muon_state_shapes_dtypes_finiteness_and_fields_are_strict(self):
        model, optimizer = optimizer_fixture()
        good = copy.deepcopy(optimizer.state_dict())
        for kind, problems in (("adam", ("missing", "negative_second", "step", "dtype", "sparse")),
                               ("muon", ("missing", "nan", "shape", "dtype", "sparse"))):
            for problem in problems:
                bad = copy.deepcopy(good)
                group = next(group for group in bad["param_groups"] if group["kind"] == kind)
                state = bad["state"][group["params"][0]]
                key = "exp_avg_sq" if kind == "adam" else "momentum_buffer"
                if problem == "missing": state.pop(key)
                elif problem == "negative_second": state[key].flatten()[0] = -.1
                elif problem == "step": state["step"] = torch.tensor(1.5)
                elif problem == "nan": state[key].flatten()[0] = float("nan")
                elif problem == "shape": state[key] = torch.zeros(1)
                elif problem == "dtype": state[key] = state[key].double()
                else: state[key] = state[key].to_sparse()
                with self.subTest(kind=kind, problem=problem): self.assert_pure_rejection(model, optimizer, bad)


class ReferenceRunnerRestoreTest(unittest.TestCase):
    assert_tree_equal = restore_checks.PerceptiveRunnerRestoreTest.assert_tree_equal

    def test_actual_runner_roundtrip_preserves_three_reference_optimizers_and_all_moments(self):
        source, restored = reference_runner_fixture(1), reference_runner_fixture(2)
        old_execution = restored._latent_wm_execution
        with tempfile.TemporaryDirectory() as directory, patch("torch.cuda.is_available", return_value=False):
            path = Path(directory)/"reference.pt"
            source.save(path, export=False)
            payload = torch.load(path, map_location="cpu", weights_only=False)
            fresh = restored.current_depth.clone()
            with patch("src.wmac_perceptive.runner.read_depth_frame", return_value=fresh):
                restored.load(path)
        for name in (*MODULES, "latent_decoder", *OPTIMIZERS):
            self.assert_tree_equal(getattr(restored, name).state_dict(), payload[name])
        for name in ("opt_actor", "opt_model", "opt_critic"):
            self.assertIsInstance(getattr(restored, name), ReferenceMuonAdam)
            self.assertEqual({group["kind"] for group in getattr(restored, name).param_groups}, {"adam", "muon"})
        self.assertIsNot(restored._latent_wm_execution, old_execution)
        self.assertEqual(restored.iteration, payload["iteration"])
        self.assertEqual(restored.sim_steps, payload["sim_steps"])

    def test_late_optimizer_rejection_and_algorithm_swaps_are_atomic(self):
        source, restored = reference_runner_fixture(1), reference_runner_fixture(2)
        with tempfile.TemporaryDirectory() as directory, patch("torch.cuda.is_available", return_value=False):
            original = Path(directory)/"reference.pt"
            source.save(original, export=False)
            payload = torch.load(original, map_location="cpu", weights_only=False)
            for problem in ("late_native_adam_nan", "model_muon_nan", "critic_algorithm", "config_algorithm"):
                bad = copy.deepcopy(payload)
                if problem == "late_native_adam_nan":
                    state = next(iter(bad["opt_disc"]["state"].values()))
                    state["exp_avg"].flatten()[0] = float("nan")
                elif problem == "model_muon_nan":
                    group = next(group for group in bad["opt_model"]["param_groups"] if group["kind"] == "muon")
                    bad["opt_model"]["state"][group["params"][0]]["momentum_buffer"].flatten()[0] = float("nan")
                elif problem == "critic_algorithm":
                    bad["opt_critic"]["algorithm_name"] = "adam"
                else: bad["config"]["execution_optimizer"] = "adam"
                path = Path(directory)/(problem+".pt")
                torch.save(bad, path)
                before = full_snapshot(restored)
                with self.subTest(problem=problem), self.assertRaises(ValueError):
                    restored.load(path)
                self.assert_tree_equal(full_snapshot(restored), before)


if __name__ == "__main__":
    unittest.main()
