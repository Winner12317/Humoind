"""Diagnostic isolation, gradient accounting and physical-boundary checks."""
from __future__ import annotations

import copy
import json
import math
import random
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from src.wmac.diagnostics import (
    objective_gradient_probe, open_loop_probe, parameter_gradient_vector, safe_cosine,
)
from src.wmac.networks import RunningNorm
from test_wmac_optimization import make_runner


def snapshot(runner):
    roots = (runner.actor, runner.world_model, runner.critic, runner.discriminator, runner.amp_norm)
    return {
        "cfg": copy.deepcopy(vars(runner.cfg)),
        "states": [{key: value.clone() for key, value in module.state_dict().items()} for module in roots],
        "modes": [[child.training for child in module.modules()] for module in roots],
        "flags": [[parameter.requires_grad for parameter in module.parameters()] for module in roots],
        "grads": [[parameter.grad.clone() if parameter.grad is not None else None
                   for parameter in module.parameters()] for module in roots],
        "torch": torch.get_rng_state().clone(), "python": random.getstate(),
        "numpy": np.random.get_state(),
    }


class ObjectiveGradientProbeTest(unittest.TestCase):
    def assert_preserved(self, before, runner):
        after = snapshot(runner)
        self.assertEqual(before["cfg"], after["cfg"])
        self.assertEqual(before["modes"], after["modes"])
        self.assertEqual(before["flags"], after["flags"])
        torch.testing.assert_close(before["torch"], after["torch"], rtol=0, atol=0)
        self.assertEqual(before["python"], after["python"])
        self.assertEqual(before["numpy"][0], after["numpy"][0])
        np.testing.assert_array_equal(before["numpy"][1], after["numpy"][1])
        self.assertEqual(before["numpy"][2:], after["numpy"][2:])
        for expected, actual in zip(before["states"], after["states"]):
            for key in expected:
                torch.testing.assert_close(expected[key], actual[key], rtol=0, atol=0)
        for expected, actual in zip(before["grads"], after["grads"]):
            for first, second in zip(expected, actual):
                if first is None:
                    self.assertIsNone(second)
                else:
                    torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_exact_gradient_partition_with_original_survival_and_style_settings(self):
        for direct_style, survival in ((True, True), (False, True), (False, False)):
            with self.subTest(direct_style=direct_style, survival=survival):
                runner, batch, noises = make_runner(batch_size=2, horizon=2)
                runner.cfg.direct_style_grad, runner.cfg.survival_grad = direct_style, survival
                expected = float(runner.imagine(batch, noises=noises)["objective"].mean())
                first_parameter = next(runner.actor.parameters())
                first_parameter.grad = torch.full_like(first_parameter, .125)
                next(runner.world_model.parameters()).requires_grad_(False)
                runner.actor.norm.eval()
                original_inputs = {key: value.clone() for key, value in batch.items()}
                before = snapshot(runner)
                result = objective_gradient_probe(runner, batch, noises, return_gradients=True)
                self.assert_preserved(before, runner)
                self.assertAlmostEqual(result["components"]["total"]["objective_mean"], expected, places=12)
                self.assertLess(result["decomposition"]["objective_absolute_residual"], 1e-12)
                self.assertLess(result["decomposition"]["gradient_relative_residual"], 1e-9)
                total = result["_gradients"]["total"]
                components = sum(result["_gradients"][key] for key in ("task", "style", "bootstrap"))
                torch.testing.assert_close(total, components, rtol=1e-9, atol=1e-12)
                self.assertGreater(result["components"]["total"]["gradient_norm"], 0.)
                if not direct_style and not survival:
                    self.assertEqual(result["components"]["style"]["gradient_norm"], 0.)
                    self.assertIsNone(result["cosines"]["task_style"])
                del result["_gradients"]
                json.dumps(result, allow_nan=False)
                for key, value in batch.items():
                    torch.testing.assert_close(original_inputs[key], value, rtol=0, atol=0)

    def test_generated_noise_and_exception_restore_rng_buffers_grads_and_configuration(self):
        runner, batch, _ = make_runner(batch_size=2, horizon=2)
        before = snapshot(runner)
        objective_gradient_probe(runner, batch)
        self.assert_preserved(before, runner)
        original = runner.imagine
        count = 0

        def faulty(*args, **kwargs):
            nonlocal count
            count += 1
            if count == 2:
                runner.amp_norm.mean.add_(5.)
                next(runner.actor.parameters()).grad = torch.ones_like(next(runner.actor.parameters()))
                random.random()
                np.random.rand()
                torch.rand(3)
                raise RuntimeError("controlled probe failure")
            return original(*args, **kwargs)

        with patch.object(runner, "imagine", side_effect=faulty):
            with self.assertRaisesRegex(RuntimeError, "controlled probe failure"):
                objective_gradient_probe(runner, batch)
        self.assert_preserved(before, runner)

    def test_gradient_vector_keeps_zero_slots_and_preserves_existing_grads(self):
        active = nn.Parameter(torch.tensor([1., 2.]))
        unused = nn.Parameter(torch.tensor([3.]))
        frozen = nn.Parameter(torch.tensor([4.]), requires_grad=False)
        active.grad = torch.tensor([8., 9.])
        result = parameter_gradient_vector(active.square().sum(), (active, unused, frozen))
        torch.testing.assert_close(result, torch.tensor([2., 4., 0., 0.]))
        torch.testing.assert_close(active.grad, torch.tensor([8., 9.]))
        self.assertIsNone(unused.grad)
        self.assertAlmostEqual(safe_cosine(result, result), 1.)
        self.assertAlmostEqual(safe_cosine(result, -result), -1.)
        self.assertIsNone(safe_cosine(result, torch.zeros_like(result)))
        self.assertIsNone(safe_cosine(result, torch.full_like(result, float("nan"))))


class ToyDynamics(nn.Module):
    def step_with_contact(self, state, action, context):
        result = state.clone()
        result[:, 0] += 1.
        return result, state.new_zeros((len(state), 2))

    def loss(self, predicted, target):
        return (predicted - target).square().mean(-1)


class ToyCritic(nn.Module):
    def forward(self, features):
        return features[..., 0]


class ToyAdapter:
    def advance_context(self, state, context):
        return context, state.new_zeros(len(state))

    def task_reward(self, state, next_state, *args):
        return next_state[..., 0]

    def amp_features(self, state):
        return state[..., :1]

    def critic_features(self, state, *args):
        return state[..., :1]


def toy_physical_windows():
    state = torch.zeros(2, 3, 71)
    state[:, :, 0] = torch.arange(3.)
    state[:, :, 3] = 1.
    successor = state.clone()
    successor[:, :, 0] += 1.
    windows = {
        "state": state, "next_state": successor,
        "action": torch.zeros(2, 3, 29), "previous_action": torch.zeros(2, 3, 29),
        "context": torch.zeros(2, 3, 35), "next_context": torch.zeros(2, 3, 35),
        "command": torch.zeros(2, 3, 3), "contact_cost": torch.zeros(2, 3, 2),
        "reward": successor[:, :, 0].clone(), "terminated": torch.zeros(2, 3, dtype=torch.bool),
        "timeout": torch.zeros(2, 3, dtype=torch.bool),
        "mask": torch.tensor([[1., 0., 1.], [1., 1., 0.]]),
    }
    windows["terminated"][0, 0] = True
    windows["timeout"][1, 1] = True
    runner = SimpleNamespace(
        actor=nn.Linear(1, 1), world_model=ToyDynamics(), discriminator=nn.Linear(1, 1),
        critic=ToyCritic(), amp_norm=nn.Identity(), adapter=ToyAdapter(),
        cfg=SimpleNamespace(gamma=.5, task_weight=1., style_weight=0.),
        _amp_reward=lambda first, second: (torch.zeros_like(first[..., 0]), torch.zeros_like(first[..., 0])),
    )
    return runner, windows


def model_matched_windows(runner, starts, length=2):
    """Synthetic exact-model reference to isolate diagnostic alignment errors."""
    rows = []
    state, context = starts["state"], starts["context"]
    previous, command = starts["previous_action"], starts["command"]
    with torch.no_grad():
        action = runner.actor(starts["actor_history"])
        for _ in range(length):
            successor, contact = runner.world_model.step_with_contact(state, action, context)
            next_context, terminal = runner.adapter.advance_context(successor, context)
            reward = runner.adapter.task_reward(state, successor, action, previous,
                command, next_context, terminal, contact)
            rows.append({"state": state, "next_state": successor, "action": action,
                "previous_action": previous, "context": context, "next_context": next_context,
                "command": command, "contact_cost": contact, "reward": reward,
                "terminated": torch.zeros(len(state), dtype=torch.bool),
                "mask": torch.ones(len(state), dtype=state.dtype)})
            state, context, previous = successor, next_context, action
    return {key: torch.stack([row[key] for row in rows], 1) for key in rows[0]}


def grounded_probe_runner():
    runner, starts, _ = make_runner(batch_size=2, horizon=2)
    runner.adapter.amp_feature_schema = runner.cfg.amp_feature_schema = "grounded_v2"
    runner.adapter.amp_dim = 203
    runner.amp_norm = RunningNorm(203).double()
    runner.amp_norm.mean.fill_(.1)
    runner.amp_norm.var.fill_(4.)
    runner.amp_norm.count.fill_(123.)
    runner.amp_norm.clip = 10.
    # A linear D that sees only the predicted successor's grounded pelvis
    # height makes the sign and size of score error independently checkable.
    runner.discriminator = nn.Linear(406, 1, bias=False).double()
    with torch.no_grad():
        runner.discriminator.weight.zero_()
        runner.discriminator.weight[0, 203 + 198] = .2
    return runner, starts


class OpenLoopProbeTest(unittest.TestCase):
    def test_matching_transition_has_zero_amp_feature_score_and_style_gap(self):
        for schema in ("legacy", "grounded_v2"):
            with self.subTest(schema=schema):
                if schema == "grounded_v2":
                    runner, starts = grounded_probe_runner()
                else:
                    runner, starts, _ = make_runner(batch_size=2, horizon=2)
                windows = model_matched_windows(runner, starts)
                before = snapshot(runner)
                result = open_loop_probe(runner, windows, horizons=(1, 2))
                ObjectiveGradientProbeTest.assert_preserved(self, before, runner)
                for h in ("1", "2"):
                    amp = result["horizons"][h]["amp_transition"]
                    self.assertEqual(amp["feature_dim"], 203 if schema == "grounded_v2" else 195)
                    self.assertEqual(amp["valid_count"], 2)
                    for summary in amp["feature_errors"].values():
                        self.assertAlmostEqual(summary["mean"], 0., places=12)
                    for key in ("discriminator_raw_score", "style_reward"):
                        self.assertAlmostEqual(amp[key]["signed_gap"]["mean"], 0., places=12)
                        self.assertAlmostEqual(amp[key]["absolute_gap"]["mean"], 0., places=12)
                        self.assertEqual(amp[key]["predicted"], amp[key]["recorded"])
                    for frame in ("prior", "next"):
                        self.assertEqual(amp["grounded_feature_errors"][frame]["available"], schema == "grounded_v2")
                json.dumps(result, allow_nan=False)

    def test_model_height_bias_produces_grounded_feature_and_fixed_discriminator_gap(self):
        runner, starts = grounded_probe_runner()
        windows = model_matched_windows(runner, starts)
        # Inject a known 20cm model error after making the recorded reference.
        # This is a diagnostic fixture, not a change to a trained checkpoint.
        runner.world_model.out_norm.mean[2].sub_(.2)
        before = snapshot(runner)
        result = open_loop_probe(runner, windows, horizons=(1,))
        ObjectiveGradientProbeTest.assert_preserved(self, before, runner)
        amp = result["horizons"]["1"]["amp_transition"]
        self.assertEqual(amp["feature_errors"]["prior_raw_rmse"]["mean"], 0.)
        self.assertGreater(amp["feature_errors"]["next_raw_rmse"]["mean"], 0.)
        expected_score_gap = -.2 / 2.01 * .2
        self.assertAlmostEqual(amp["discriminator_raw_score"]["signed_gap"]["mean"], expected_score_gap, places=12)
        self.assertAlmostEqual(amp["discriminator_raw_score"]["absolute_gap"]["mean"], abs(expected_score_gap), places=12)
        self.assertLess(amp["style_reward"]["signed_gap"]["mean"], 0.)
        self.assertGreater(amp["style_reward"]["absolute_gap"]["mean"], 0.)
        extra = amp["grounded_feature_errors"]["next"]["fields"]
        pelvis = extra["pelvis_ground_relative_height"]["components"]["pelvis"]
        self.assertAlmostEqual(pelvis["signed_gap"]["mean"], -.2, places=12)
        self.assertAlmostEqual(pelvis["absolute_gap"]["mean"], .2, places=12)
        for foot in ("left", "right"):
            height = extra["foot_reference_ground_relative_height"]["components"][foot]
            speed = extra["foot_reference_world_vertical_velocity"]["components"][foot]
            self.assertAlmostEqual(height["absolute_gap"]["mean"], .2, places=12)
            self.assertAlmostEqual(speed["absolute_gap"]["mean"], 0., places=12)
        for axis in ("x", "y", "z"):
            gravity = extra["pelvis_projected_gravity"]["components"][axis]
            self.assertAlmostEqual(gravity["absolute_gap"]["mean"], 0., places=12)

    def test_raw_errors_have_physical_units_and_quaternion_sign_invariance(self):
        runner, windows = toy_physical_windows()
        windows["mask"].zero_()
        windows["mask"][0, 0] = 1.
        target = windows["next_state"][0, 0]
        target[1:3] = torch.tensor([3., 4.])
        target[3:7] = torch.tensor([math.cos(math.pi / 4), 0., 0., math.sin(math.pi / 4)])
        target[7:10] = torch.tensor([3., 4., 0.])
        target[10:13] = torch.tensor([0., 0., 2.])
        target[13:42] = .2
        target[42:71] = -.4
        expected = {
            "root_position_l2_m": 5., "root_orientation_geodesic_rad": math.pi / 2,
            "root_linear_velocity_l2": 5., "root_angular_velocity_l2": 2.,
            "joint_position_rmse_rad": .2, "joint_velocity_rmse_rad_s": .4,
        }
        for sign in (1., -1.):
            with self.subTest(quaternion_sign=sign):
                target[3:7].mul_(sign)
                result = open_loop_probe(runner, windows, horizons=(1,))
                raw = result["horizons"]["1"]["raw_state_errors"]
                for name, value in expected.items():
                    self.assertEqual(raw[name]["valid_count"], 1)
                    self.assertEqual(raw[name]["nonfinite_count"], 0)
                    self.assertAlmostEqual(raw[name]["mean"], value, places=6)

    def test_model_loss_scales_are_reported_separately_from_raw_errors(self):
        runner, starts, _ = make_runner(batch_size=2, horizon=1)
        runner.world_model.out_norm.var.fill_(4.)
        runner.world_model.out_norm.eps = .1
        runner.world_model.out_norm.count.fill_(123.)
        with torch.no_grad():
            action = runner.actor(starts["actor_history"])
            successor, contact = runner.world_model.step_with_contact(starts["state"], action, starts["context"])
            context, terminal = runner.adapter.advance_context(successor, starts["context"])
            reward = runner.adapter.task_reward(starts["state"], successor, action,
                starts["previous_action"], starts["command"], context, terminal, contact)
        windows = {key: value[:, None] for key, value in {
            "state": starts["state"], "next_state": successor, "action": action,
            "previous_action": starts["previous_action"], "context": starts["context"],
            "next_context": context, "command": starts["command"], "contact_cost": contact,
            "reward": reward, "terminated": terminal.bool(), "mask": torch.ones(2),
        }.items()}
        result = open_loop_probe(runner, windows, horizons=(1,))
        scaling = result["world_model_loss_normalization"]
        self.assertEqual(scaling["output_count"], 123.)
        self.assertEqual(scaling["output_epsilon"], .1)
        self.assertFalse(scaling["output_variance_epsilon"])
        self.assertEqual(scaling["output_variance"], [4.] * 70)
        for scale in scaling["loss_scale"]:
            self.assertAlmostEqual(scale, 2.1)
        raw = result["horizons"]["1"]["raw_state_errors"]
        for metrics in raw.values():
            self.assertAlmostEqual(metrics["mean"], 0., places=12)

    def test_mask_does_not_reopen_and_timeout_bootstraps_true_successor(self):
        runner, windows = toy_physical_windows()
        # Padding after a cut must not contaminate aggregates.
        windows["next_state"][0, 1:] = float("nan")
        windows["next_state"][1, 2] = float("nan")
        result = open_loop_probe(runner, windows, horizons=(1, 2, 3))
        counts = [result["horizons"][str(h)]["state_error"]["valid_count"] for h in (1, 2, 3)]
        self.assertEqual(counts, [2, 1, 0])
        for h in (1, 2):
            for key in ("state_error", "contact_log_error", "task_reward_absolute_gap"):
                self.assertEqual(result["horizons"][str(h)][key]["mean"], 0.)
                self.assertEqual(result["horizons"][str(h)][key]["nonfinite_count"], 0)
        self.assertIsNone(result["horizons"]["3"]["state_error"]["mean"])
        td = result["physical_critic_td"]["signed_residual"]
        self.assertEqual(td["valid_count"], 3)
        self.assertAlmostEqual(td["mean"], 1.5)
        for metrics in result["physical_amp"].values():
            self.assertEqual(metrics["valid_count"], 3)
            self.assertEqual(metrics["nonfinite_count"], 0)
            self.assertEqual(metrics["mean"], 0.)
        json.dumps(result, allow_nan=False)

    def test_no_valid_physical_rows_and_invalid_horizons_are_explicit(self):
        runner, windows = toy_physical_windows()
        windows["mask"].zero_()
        result = open_loop_probe(runner, windows, horizons=(1,))
        self.assertEqual(result["physical_critic_td"]["absolute_residual"]["valid_count"], 0)
        self.assertIsNone(result["physical_critic_td"]["absolute_residual"]["mean"])
        for horizons in ((), (0,), (4,)):
            with self.subTest(horizons=horizons), self.assertRaises(ValueError):
                open_loop_probe(runner, windows, horizons=horizons)


if __name__ == "__main__":
    unittest.main()
