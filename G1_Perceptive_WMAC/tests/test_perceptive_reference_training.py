"""Pinned official critic, cumulative survival, bootstrap and GAE parity.

SUPERMIMIC_REFERENCE_CHECKOUT may point to the audited upstream commit. The
source-backed tests skip when absent and verify hashes before executing the
selected upstream functions; ordinary behavior tests need no checkout.
"""
from __future__ import annotations

import ast
import hashlib
import math
import os
from pathlib import Path
from types import MethodType, SimpleNamespace
import unittest

import torch
from torch import nn

from src.wmac_perceptive.reference_training import (
    ReferenceCritic,
    build_reference_critic,
    reference_bootstrap,
    reference_gae_returns,
    reference_value_ceiling,
)


REFERENCE_COMMIT = "abc124c1bd5976341034668af3d12c087f2f1f55"
SOURCE_HASHES = {
    "models.py": "fbfbd5c836c2670a68776f9009b9463451d5115c753d0e592fba919f2d44b0ad",
    "learners/policy_learner.py": "58b29adc8495685fc4dbdff5ecf7b3bcaaa1601689c4e3ba4074cadda48dbd2f",
    "envs/world_model_env.py": "359fcab19230c5448644a5a30e626767c0eb1a3078c57437f468cf953d627319",
}


def official_functions():
    root = Path(os.environ.get("SUPERMIMIC_REFERENCE_CHECKOUT", "/tmp/supermimic-latest-audit-20261111-1720"))
    trees = {}
    for relative, expected in SOURCE_HASHES.items():
        path = root / "src/supermimic" / relative
        if not path.exists():
            raise unittest.SkipTest("Pinned official SuperMimic source unavailable")
        source = path.read_bytes()
        if hashlib.sha256(source).hexdigest() != expected:
            raise AssertionError(f"Official source hash differs: {relative}")
        trees[relative] = ast.parse(source, filename=str(path))
    selected = []
    for node in trees["models.py"].body:
        if isinstance(node, ast.FunctionDef) and node.name == "build_mlp":
            selected.append(node)
    for node in trees["learners/policy_learner.py"].body:
        if isinstance(node, ast.FunctionDef) and node.name == "gae_returns":
            selected.append(node)
        if isinstance(node, ast.ClassDef):
            for method in node.body:
                if isinstance(method, ast.FunctionDef) and (
                    (node.name == "ActorCriticPolicyLearner" and method.name == "_bootstrap")
                    or (node.name == "BpttPolicyLearner" and method.name == "_train_batch")
                ):
                    selected.append(method)

    class TensorDict(dict):
        def __init__(self, values, batch_size=None):
            super().__init__(values)

        def detach(self):
            return TensorDict({key: value.detach() for key, value in self.items()})

    namespace = dict(torch=torch, nn=nn, TensorDict=TensorDict)
    exec(compile(ast.Module(body=selected, type_ignores=[]), "pinned_supermimic", "exec"), namespace)
    return SimpleNamespace(**namespace)


def reward_fixture(**overrides):
    adapter = SimpleNamespace(reward_scale=.02, reward_cfg={
        "tracking": SimpleNamespace(weight=2.0),
        "upright": SimpleNamespace(weight=.5),
        "penalty": SimpleNamespace(weight=-100.0),
        "zero": SimpleNamespace(weight=0.0),
    })
    cfg = SimpleNamespace(gamma=.99, task_weight=.75, style_weight=.25, amp_reward_coef=.1)
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return adapter, cfg


class ReferenceCriticAndCeilingTest(unittest.TestCase):
    def test_default_critic_architecture_and_output_shape(self):
        critic = build_reference_critic(17)
        self.assertIsInstance(critic, ReferenceCritic)
        self.assertEqual([type(layer) for layer in critic.net], [
            nn.Linear, nn.LayerNorm, nn.GELU, nn.Linear, nn.LayerNorm, nn.GELU, nn.Linear,
        ])
        self.assertEqual([(layer.in_features, layer.out_features) for layer in critic.net if isinstance(layer, nn.Linear)],
                         [(17, 256), (256, 128), (128, 1)])
        self.assertEqual(critic(torch.randn(2, 3, 17)).shape, (2, 3))
        self.assertEqual(critic.precision, "float32")

    def test_input_norm_is_dormant_and_input_gradient_is_preserved(self):
        critic = build_reference_critic(5, (8, 4)).double()
        features = torch.randn(7, 5, dtype=torch.float64, requires_grad=True)
        expected = critic.net(features).squeeze(-1)
        with torch.no_grad():
            critic.norm.mean.fill_(1000)
            critic.norm.var.fill_(0)
            critic.norm.count.fill_(12345)
        actual = critic(features)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        gradient, = torch.autograd.grad(actual.sum(), features)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(float(gradient.abs().sum()), 0)

    def test_positive_task_terms_and_amp_ceiling_are_scaled_once(self):
        adapter, cfg = reward_fixture()
        ceiling = reference_value_ceiling(adapter, cfg)
        self.assertIs(type(ceiling), float)
        self.assertAlmostEqual(ceiling, (.75 * .02 * 2.5 + .25 * .1) / .01)
        cfg.style_weight = 0
        cfg.task_weight = 1
        self.assertAlmostEqual(reference_value_ceiling(adapter, cfg), .02 * 2.5 / .01)
        adapter.reward_scale = 1
        self.assertAlmostEqual(reference_value_ceiling(adapter, cfg), 2.5 / .01)

    def test_amp_only_and_no_positive_task_rewards(self):
        adapter, cfg = reward_fixture(task_weight=0, style_weight=1, gamma=0)
        self.assertEqual(reference_value_ceiling(adapter, cfg), .1)
        cfg.task_weight, cfg.style_weight = 1, 0
        adapter.reward_cfg = {"penalty": SimpleNamespace(weight=-1)}
        self.assertEqual(reference_value_ceiling(adapter, cfg), 0)

    def test_invalid_ceiling_configuration_fails_before_training(self):
        for gamma in (-.1, 1.0, math.inf, math.nan):
            adapter, cfg = reward_fixture(gamma=gamma)
            with self.assertRaises(ValueError):
                reference_value_ceiling(adapter, cfg)
        adapter, cfg = reward_fixture()
        adapter.reward_cfg["tracking"].weight = math.nan
        with self.assertRaises(ValueError):
            reference_value_ceiling(adapter, cfg)

    def test_bootstrap_caps_only_above_and_keeps_survival_gradient(self):
        value = torch.tensor([100., -10., 2.], requires_grad=True)
        survival = torch.tensor([.5, 1., 0.], requires_grad=True)
        actual = reference_bootstrap(value, survival, .9, 2, 3.)
        torch.testing.assert_close(actual, torch.tensor([1.215, -8.1, 0.]))
        value_gradient, survival_gradient = torch.autograd.grad(actual.sum(), (value, survival))
        torch.testing.assert_close(value_gradient, torch.tensor([0., .81, 0.]))
        torch.testing.assert_close(survival_gradient, torch.tensor([2.43, -8.1, 1.62]))

    def test_gae_caps_final_targets_and_leaves_negative_values_unbounded(self):
        values = torch.tensor([[10., -10.], [10., -10.]])
        last_value = torch.tensor([100., -100.])
        rewards = torch.tensor([[0., -10.], [0., -10.]])
        dones = torch.tensor([[0., 0.], [0., 1.]])
        actual = reference_gae_returns(values, last_value, rewards, dones, .9, .8, 3.)
        torch.testing.assert_close(actual, torch.tensor([[3., -19.], [3., -10.]]))


class PinnedOfficialReferenceTrainingParityTest(unittest.TestCase):
    def test_critic_parameters_outputs_and_all_gradients_match_official_mlp(self):
        official = official_functions()
        torch.manual_seed(831)
        critic = build_reference_critic(11, (256, 128)).double()
        torch.manual_seed(831)
        expected = official.build_mlp(11, (256, 128), 1).double()
        for actual_parameter, expected_parameter in zip(critic.net.parameters(), expected.parameters()):
            torch.testing.assert_close(actual_parameter, expected_parameter, rtol=0, atol=0)
        features = torch.randn(2, 3, 11, dtype=torch.float64, requires_grad=True)
        reference_features = features.detach().clone().requires_grad_()
        actual_output = critic(features)
        expected_output = expected(reference_features).squeeze(-1)
        torch.testing.assert_close(actual_output, expected_output, rtol=0, atol=0)
        actual_gradients = torch.autograd.grad(actual_output.square().sum(), (features, *critic.net.parameters()))
        expected_gradients = torch.autograd.grad(expected_output.square().sum(), (reference_features, *expected.parameters()))
        for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
            torch.testing.assert_close(actual_gradient, expected_gradient, rtol=0, atol=0)

    def test_gae_uses_raw_bootstrap_and_clamps_after_official_recursion(self):
        official = official_functions()
        torch.manual_seed(719)
        for shape in ((7, 5), (7, 5, 1)):
            values = (torch.randn(shape, dtype=torch.float64) * 4).requires_grad_()
            last_value = (torch.randn(shape[1:], dtype=torch.float64) * 20).requires_grad_()
            rewards = (torch.randn(shape, dtype=torch.float64) * 2).requires_grad_()
            dones = torch.rand(shape, dtype=torch.float64).requires_grad_()
            expected = official.gae_returns(values, last_value, rewards, dones, .99, .95).clamp(max=3.)
            actual = reference_gae_returns(values, last_value, rewards, dones, .99, .95, 3.)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            leaves = (values, last_value, rewards, dones)
            expected_gradients = torch.autograd.grad(expected.sum(), leaves, retain_graph=True)
            actual_gradients = torch.autograd.grad(actual.sum(), leaves)
            for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
                torch.testing.assert_close(actual_gradient, expected_gradient, rtol=0, atol=0)

    def test_survival_and_bootstrap_return_gradients_match_actual_official_train_batch(self):
        official = official_functions()
        horizon, batch, gamma, ceiling = 4, 3, .99, 3.
        rewards = torch.tensor([[1., 1., 1.], [2., 2., 2.], [-1., -1., -1.], [.5, .5, .5]],
                               dtype=torch.float64, requires_grad=True)
        dones = torch.tensor([[.1, 0., 0.], [.2, 1., 0.], [.3, 0., 0.], [.4, 0., 1.]],
                             dtype=torch.float64, requires_grad=True)
        last_value = torch.tensor([20., -5., 2.], dtype=torch.float64, requires_grad=True)
        captured = {}

        def optimize(optimizers, loss, parameters, max_norm, grad_clip_value):
            captured["loss"] = loss
            return loss.new_zeros(())

        official._train_batch.__globals__["_optimize_step"] = optimize

        class WorldModelEnv:
            def reset(self, start):
                self.time = 0
                return {"critic": last_value[:, None]}

            def step(self, action):
                t = self.time
                self.time += 1
                return {"critic": last_value[:, None]}, rewards[t], dones[t], {}

        learner = SimpleNamespace(
            actor=nn.Identity(), critic=nn.Identity(), wm_env=WorldModelEnv(),
            _rollout_actor=nn.Identity(), actor_params=[], actor_optimizers=[], _v_max=ceiling,
            cfg=SimpleNamespace(horizon=horizon, gamma=gamma, survival_masking=True, grad_clip_value=.1),
            _fit_critic=lambda rollout, obs: {},
        )
        learner._bootstrap = MethodType(official._bootstrap, learner)
        stats = official._train_batch(learner, None)
        survival = torch.cat((torch.ones(1, batch, dtype=dones.dtype), (1-dones).cumprod(0)), 0)
        discounted = gamma**torch.arange(horizon, dtype=rewards.dtype)[:, None]
        actual_return = (discounted * survival[:-1] * rewards).sum(0)
        actual_return = actual_return + reference_bootstrap(last_value, survival[-1], gamma, horizon, ceiling)
        actual_loss = -actual_return.mean()
        torch.testing.assert_close(actual_return.mean(), stats["return"], rtol=0, atol=2e-16)
        leaves = (rewards, dones, last_value)
        expected_gradients = torch.autograd.grad(captured["loss"], leaves, retain_graph=True)
        actual_gradients = torch.autograd.grad(actual_loss, leaves)
        for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
            torch.testing.assert_close(actual_gradient, expected_gradient, rtol=0, atol=5e-16)


if __name__ == "__main__":
    unittest.main()
