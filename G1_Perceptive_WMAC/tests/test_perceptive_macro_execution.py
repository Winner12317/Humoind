"""Whole learner macros: exact update counts and shared capture ownership.

CPU comparisons include the actual G1 v2 FK/soft-termination rollout. Optional
CUDA tests use tiny models only, without a simulator or production allocation.
"""
from __future__ import annotations

import copy
import os
from types import SimpleNamespace
import unittest

import torch
from torch import nn

from src.wmac.networks import frozen_parameters
from src.wmac_perceptive.captured_updates import _ActorLoss
from src.wmac_perceptive.compiled_imagination import CriticLoss, fit_critic_fast
from src.wmac_perceptive.macro_execution import CapturedMacroUpdate, ActorCriticMacroExecution
from src.wmac_perceptive.optimizers import build_reference_optimizer


CUDA = os.environ.get("PERCEPTIVE_TEST_MACRO_CUDA") == "1" and torch.cuda.is_available()


def independent_policy_update(runner, batch, noise):
    """Existing separate Actor then Critic update, with no macro executor."""
    runner.opt_actor.zero_grad(set_to_none=True)
    with frozen_parameters(runner.world_model, runner.discriminator, runner.critic):
        loss, rollout = _ActorLoss(runner)({**batch, "_noise": noise,
            "_router_load": noise.new_ones(getattr(runner.actor, "num_experts", 1))})
        loss.backward()
        parameters = tuple(parameter for group in runner.opt_actor.param_groups for parameter in group["params"])
        norm = nn.utils.clip_grad_norm_(parameters, runner.cfg.max_grad_norm, foreach=True)
        if runner.cfg.actor_grad_value_clip:
            nn.utils.clip_grad_value_(parameters, runner.cfg.actor_grad_value_clip, foreach=True)
        runner.opt_actor.step()
    critic = fit_critic_fast(runner, rollout["x"], rollout["targets"])
    runner.opt_critic.zero_grad(set_to_none=True)
    return rollout, norm, critic


def assert_optimizer_equal(test, left, right, *, rtol=1e-10, atol=1e-12):
    for lg, rg in zip(left.param_groups, right.param_groups):
        for lp, rp in zip(lg["params"], rg["params"]):
            test.assertEqual(left.state[lp].keys(), right.state[rp].keys())
            for name in left.state[lp]:
                torch.testing.assert_close(left.state[lp][name], right.state[rp][name], rtol=rtol, atol=atol)


def g1_runner():
    from test_hiking_training_v2 import v2_reference_fixture
    runner, batch, noise = v2_reference_fixture()
    runner.cfg.critic_epochs, runner.cfg.critic_minibatches = 5, 4
    runner.cfg.graph_warmup = 3
    runner.cfg.execution_optimizer = "muon_adam"
    runner.opt_actor = build_reference_optimizer([runner.actor.policy], runner.cfg.actor_lr,
        betas=runner.cfg.actor_betas)
    runner.opt_critic = build_reference_optimizer([runner.critic], runner.cfg.critic_lr)
    runner._latent_critic_execution = None
    runner._latent_critic_loss_fn = CriticLoss(runner.critic)
    return runner, batch, noise


class MacroCpuTest(unittest.TestCase):
    def test_g1_v2_actor_then_twenty_critic_updates_match_existing_execution(self):
        runner, batch, noise = g1_runner()
        reference, _, _ = g1_runner()
        for name in ("actor", "world_model", "critic", "discriminator", "amp_norm"):
            getattr(reference, name).load_state_dict(getattr(runner, name).state_dict())
        execution = ActorCriticMacroExecution(runner, cuda_graphs=False)
        state = torch.get_rng_state()
        actual, norm, accepted, critic = execution.step(batch, noise)
        torch.set_rng_state(state)
        expected, expected_norm, expected_critic = independent_policy_update(reference, batch, noise)
        for name in expected:
            torch.testing.assert_close(actual[name], expected[name], rtol=1e-10, atol=1e-12)
        torch.testing.assert_close(norm, expected_norm, rtol=1e-10, atol=1e-12)
        self.assertTrue(bool(accepted))
        torch.testing.assert_close(critic, critic.new_tensor(tuple(expected_critic.values())), rtol=1e-10, atol=1e-12)
        for name in ("actor", "critic"):
            for actual_parameter, expected_parameter in zip(getattr(runner, name).parameters(),
                                                           getattr(reference, name).parameters()):
                torch.testing.assert_close(actual_parameter, expected_parameter, rtol=1e-10, atol=1e-12)
        assert_optimizer_equal(self, runner.opt_actor, reference.opt_actor)
        assert_optimizer_equal(self, runner.opt_critic, reference.opt_critic)
        for optimizer, count in ((runner.opt_actor, 1), (runner.opt_critic, 20)):
            steps = [value["step"] for value in optimizer.state.values() if "step" in value]
            self.assertTrue(steps)
            self.assertTrue(all(float(step) == count for step in steps))
        for module in (runner.world_model, runner.discriminator, runner.critic, runner.actor.vision):
            self.assertTrue(all(parameter.grad is None for parameter in module.parameters()))
        self.assertTrue(any(parameter.grad is not None for parameter in runner.actor.policy.parameters()))

    def test_macro_warmup_counts_real_iterations_and_keeps_unused_parameters_unused(self):
        model = nn.Linear(2, 1).double()
        unused = nn.Parameter(torch.tensor(1., dtype=torch.float64))
        optimizer = torch.optim.Adam((*model.parameters(), unused), lr=.01)
        counts = torch.zeros((), dtype=torch.float64)
        execution = None
        def update(batch):
            losses = []
            for _ in range(10):
                execution.zero_grad(optimizer)
                loss = (model(batch["input"]) - batch["target"]).square().mean()
                loss.backward()
                optimizer.step()
                counts.add_(1.)
                losses.append(loss.detach())
            return {"loss": torch.stack(losses).mean(), "updates": counts.clone()}
        execution = CapturedMacroUpdate(update, (model,), (optimizer,), cuda_graphs=False)
        batch = {"input": torch.ones(3, 2, dtype=torch.float64), "target": torch.zeros(3, 1, dtype=torch.float64)}
        snapshots = [execution.step(batch) for _ in range(5)]
        self.assertEqual(execution._calls, 5)
        self.assertEqual(float(counts), 50.)
        self.assertEqual([float(item["updates"]) for item in snapshots], [10., 20., 30., 40., 50.])
        self.assertIsNone(unused.grad)
        self.assertFalse(optimizer.state[unused])
        for state in optimizer.state.values():
            if "step" in state:
                self.assertEqual(float(state["step"]), 50.)
        execution.invalidate()
        self.assertEqual(execution._calls, 0)
        self.assertIsNone(execution._static_inputs)

    def test_cuda_capture_is_explicit_and_invalid_inputs_fail_clearly(self):
        model = nn.Linear(2, 1)
        optimizer = torch.optim.Adam(model.parameters())
        with self.assertRaisesRegex(ValueError, "CUDA parameters"):
            CapturedMacroUpdate(lambda batch: batch, (model,), (optimizer,))
        for warmup in (True, 0, -1, 3.):
            with self.subTest(warmup=warmup), self.assertRaisesRegex(ValueError, "warmup"):
                CapturedMacroUpdate(lambda batch: batch, (model,), (optimizer,),
                    warmup=warmup, cuda_graphs=False)

    def test_policy_warmup_counts_three_global_macros_with_multiple_actor_updates(self):
        runner = TinyRunner(device="cpu")
        runner.cfg.training_schedule = "supermimic_v1"
        runner.cfg.actor_updates = 2
        execution = ActorCriticMacroExecution(runner, cuda_graphs=False)
        self.assertEqual(execution.execution.warmup, 6)


class TinyActor(nn.Module):
    num_experts = 1
    def __init__(self):
        super().__init__()
        self.policy = nn.Sequential(nn.Linear(7, 8), nn.ELU(), nn.Linear(8, 8), nn.ELU(), nn.Linear(8, 3))


class TinyRunner:
    """Differentiable dynamics and bootstrap exercising both optimizer owners."""
    uses_reference_schedule = True
    def __init__(self, device="cuda"):
        torch.manual_seed(20261117)
        self.cfg = SimpleNamespace(graph_warmup=3, max_grad_norm=1., actor_grad_value_clip=.2,
            moe_load_balance_coef=0., moe_router_z_coef=0., critic_bootstrap=True,
            critic_epochs=5, critic_minibatches=4, execution_optimizer="muon_adam")
        self.actor = TinyActor().to(device)
        self.world_model = nn.Linear(3, 7).to(device)
        self.discriminator = nn.Linear(7, 1).to(device)
        self.critic = nn.Sequential(nn.Linear(7, 8), nn.GELU(), nn.Linear(8, 8), nn.GELU(), nn.Linear(8, 1)).to(device)
        self.amp_norm = nn.Identity()
        self.opt_actor = build_reference_optimizer((self.actor.policy,), 2e-4, device=device)
        self.opt_critic = build_reference_optimizer((self.critic,), 2e-4, device=device)
        self._latent_critic_execution = None
        self._latent_critic_loss_fn = lambda x, y: (self.critic(x).squeeze(-1) - y).square().mean()

    def imagine(self, batch, noises):
        state = batch["state"]
        value, survival = state.new_zeros(len(state)), state.new_ones(len(state))
        features, tasks, styles, scores = [], [], [], []
        for step in range(3):
            features.append(state.detach())
            action = self.actor.policy(state)
            state = state + .05 * self.world_model(action) + noises[step]
            task = -state.square().mean(-1)
            score = self.discriminator(state).squeeze(-1)
            style = .25 * (1 - .25 * (score - 1).square()).clamp_min(0.)
            value = value + .99 ** step * survival * (task + style)
            survival = survival * (1 - torch.sigmoid(state[:, 0] - 3))
            tasks.append(task.detach()); styles.append(style.detach()); scores.append(score.detach())
        value = value + .99 ** 3 * survival * self.critic(state).squeeze(-1)
        x = torch.stack(features)
        with torch.no_grad():
            targets = self.critic(x).squeeze(-1) + torch.stack(tasks)
        return dict(objective=value, x=x, targets=targets, task=torch.stack(tasks), style=torch.stack(styles),
            score=torch.stack(scores), survival=survival.detach(), first_action=self.actor.policy(batch["state"]))


@unittest.skipUnless(CUDA, "set PERCEPTIVE_TEST_MACRO_CUDA=1 for tiny CUDA macro checks")
class MacroCudaTest(unittest.TestCase):
    def test_ten_online_targets_and_statistics_update_once_per_actual_macro(self):
        model = nn.Linear(3, 3).cuda()
        model.register_parameter("unused", nn.Parameter(torch.ones(2, device="cuda")))
        model.register_buffer("statistic", torch.zeros(3, device="cuda"))
        model.register_buffer("updates", torch.zeros((), device="cuda"))
        reference = copy.deepcopy(model)
        executors = []
        def make(module, capture):
            optimizer = build_reference_optimizer((module,), 1e-3, device="cuda")
            execution = None
            def update(batch):
                targets, losses = [], []
                for _ in range(10):
                    # Each target must see the encoder after the previous
                    # optimizer step, not a code cached before the macro.
                    with torch.no_grad():
                        target = module(batch["input"]) * .8
                        targets.append(target.mean().clone())
                        module.statistic.lerp_(target.mean(0), .1)
                        module.updates.add_(1.)
                    execution.zero_grad(optimizer)
                    loss = (module(batch["input"]) - target).square().mean()
                    loss.backward()
                    nn.utils.clip_grad_norm_(module.parameters(), 1., foreach=True)
                    optimizer.step()
                    losses.append(loss.detach())
                return dict(loss=torch.stack(losses).mean(), online_targets=torch.stack(targets),
                    statistic=module.statistic, updates=module.updates)
            execution = CapturedMacroUpdate(update, (module,), (optimizer,), cuda_graphs=capture)
            executors.append((execution, optimizer))
            return execution, optimizer
        captured, optimizer = make(model, True)
        eager, expected_optimizer = make(reference, False)
        captures = []
        for macro in range(1, 7):
            batch = {"input": torch.randn(8, 3, device="cuda")}
            actual, expected = captured.step(batch), eager.step(batch)
            torch.cuda.synchronize()
            for name in actual:
                torch.testing.assert_close(actual[name], expected[name], rtol=1e-5, atol=1e-7)
            self.assertEqual(float(model.updates), macro * 10)
            self.assertGreater(float((actual["online_targets"][1:] - actual["online_targets"][:-1]).abs().max()), 0.)
            self.assertIsNone(model.unused.grad)
            self.assertEqual(float(optimizer.state[model.unused]["step"]), 0.)
            self.assertTrue(all(float(optimizer.state[p]["step"]) == macro * 10
                for p in (model.weight, model.bias)))
            assert_optimizer_equal(self, optimizer, expected_optimizer, rtol=1e-5, atol=1e-7)
            captures.append(captured._graph is not None)
            optimizer.zero_grad(set_to_none=True)
        self.assertEqual(captures, [False, False, False, True, True, True])

    def test_actor_and_twenty_critic_steps_capture_once_and_match_eager_across_six_macros(self):
        runner, reference = TinyRunner(), TinyRunner()
        resources = torch.cuda.Stream(), torch.cuda.graph_pool_handle()
        execution = ActorCriticMacroExecution(runner, graph_resources=resources)
        captures, retained = [], []
        for macro in range(1, 7):
            batch = {"state": torch.randn(8, 7, device="cuda") * .1}
            noise = torch.randn(3, 8, 7, device="cuda") * .001
            random_state = torch.cuda.get_rng_state()
            actual, norm, accepted, summary = execution.step(batch, noise)
            torch.cuda.set_rng_state(random_state)
            expected, expected_norm, expected_summary = independent_policy_update(reference, batch, noise)
            torch.cuda.synchronize()
            for name in expected:
                torch.testing.assert_close(actual[name], expected[name], rtol=2e-5, atol=2e-6)
            torch.testing.assert_close(norm, expected_norm, rtol=2e-5, atol=2e-6)
            torch.testing.assert_close(summary, summary.new_tensor(tuple(expected_summary.values())), rtol=2e-5, atol=2e-6)
            self.assertTrue(bool(accepted))
            assert_optimizer_equal(self, runner.opt_actor, reference.opt_actor, rtol=2e-5, atol=2e-6)
            assert_optimizer_equal(self, runner.opt_critic, reference.opt_critic, rtol=2e-5, atol=2e-6)
            captures.append(execution.execution._graph is not None)
            retained.append((actual["objective"], actual["objective"].clone()))
            for optimizer, count in ((runner.opt_actor, macro), (runner.opt_critic, macro * 20)):
                self.assertTrue(all(float(state["step"]) == count for state in optimizer.state.values() if "step" in state))
            # Reproduce the real runner clearing bindings between phases.
            runner.opt_actor.zero_grad(set_to_none=True)
            runner.opt_critic.zero_grad(set_to_none=True)
        self.assertEqual(captures, [False, False, False, True, True, True])
        for output, saved in retained:
            torch.testing.assert_close(output, saved, rtol=0., atol=0.)
        runner.opt_actor.param_groups[0]["lr"] = runner.opt_actor.param_groups[0]["lr"].clone()
        with self.assertRaisesRegex(RuntimeError, "state moved or was rebound"):
            execution.step(batch, noise)
        execution.invalidate()
        self.assertIsNone(execution.execution._graph)
        execution.step(batch, noise)


if __name__ == "__main__":
    unittest.main()
