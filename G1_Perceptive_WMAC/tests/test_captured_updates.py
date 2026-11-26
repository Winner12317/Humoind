"""Small CUDA Actor/Critic full-step capture and safety-boundary checks."""
from __future__ import annotations

import copy
import ast
from dataclasses import fields, replace
import hashlib
import inspect
import os
from pathlib import Path
import textwrap
from types import SimpleNamespace
import unittest

import torch

from src.wmac.networks import Critic, frozen_parameters
from src.wmac_perceptive.captured_updates import ActorExecution, CapturedTensorUpdate, make_critic_execution
from src.wmac_perceptive.compiled_imagination import LatentImaginedStep, LatentTrajectoryTerms
from src.wmac_perceptive.optimizers import build_reference_optimizer
from src.wmac_perceptive.latent_execution import graph_warmup_steps
from src.wmac_perceptive.runner import PerceptiveRunner
from test_perceptive_latent_runner import latent_runner_fixture


CUDA = os.environ.get("PERCEPTIVE_TEST_CAPTURED_UPDATES") == "1" and torch.cuda.is_available()
OFFICIAL_POLICY = Path("/tmp/supermimic-official-audit-20261111/src/supermimic/learners/policy_learner.py")


def production_survival(termination):
    source = ast.parse(textwrap.dedent(inspect.getsource(PerceptiveRunner.imagine)))
    block = next(node for node in ast.walk(source) if isinstance(node, ast.If)
        and any(isinstance(statement, ast.Assign) and
                any(isinstance(target, ast.Name) and target.id == "alive_steps" for target in statement.targets)
                for statement in node.body))
    namespace = dict(torch=torch, continuation=1-termination, horizon=len(termination))
    exec(compile(ast.fix_missing_locations(ast.Module(body=block.body, type_ignores=[])),
                 "production_survival", "exec"), namespace)
    return namespace["alive"]


class OfficialSurvivalTest(unittest.TestCase):
    def test_original_warmup_counts_whole_training_macros(self):
        cfg = SimpleNamespace(training_schedule="supermimic_v1", graph_warmup=3,
                              wm_updates=10, actor_updates=2, critic_epochs=5, critic_minibatches=4)
        self.assertEqual(graph_warmup_steps(cfg, "wm"), 30)
        self.assertEqual(graph_warmup_steps(cfg, "actor"), 6)
        self.assertEqual(graph_warmup_steps(cfg, "critic"), 120)
        cfg.training_schedule = "validated_v2"
        self.assertEqual([graph_warmup_steps(cfg, kind) for kind in ("wm", "actor", "critic")], [3]*3)
    def test_unrolled_survival_values_and_gradients_handle_zero_termination(self):
        generator = torch.Generator().manual_seed(81)
        for horizon in (1, 3, 8, 32):
            termination = torch.rand(horizon, 5, generator=generator, dtype=torch.float64)
            termination[:, 1] = 0
            termination[:min(3, horizon), 0] = 1
            actual, expected = (termination.clone().requires_grad_() for _ in range(2))
            first, second = production_survival(actual), (1-expected).cumprod(0)
            gradient = torch.randn(termination.shape, generator=generator, dtype=termination.dtype)
            torch.testing.assert_close(first, second, rtol=0, atol=0)
            torch.testing.assert_close(torch.autograd.grad(first, actual, gradient)[0],
                                       torch.autograd.grad(second, expected, gradient)[0], rtol=1e-12, atol=1e-12)

    @unittest.skipUnless(OFFICIAL_POLICY.exists(), "fixed official source checkout is optional")
    def test_production_survival_matches_fixed_official_source_multiplication(self):
        self.assertEqual(hashlib.sha256(OFFICIAL_POLICY.read_bytes()).hexdigest(),
                         "58b29adc8495685fc4dbdff5ecf7b3bcaaa1601689c4e3ba4074cadda48dbd2f")
        source = ast.parse(OFFICIAL_POLICY.read_text())
        statement = next(node for node in ast.walk(source) if isinstance(node, ast.Assign)
            and ast.unparse(node) == "survival = survival * (1 - term_prob)")
        code = compile(ast.fix_missing_locations(ast.Module(body=[statement], type_ignores=[])),
                       "official_survival", "exec")
        termination = torch.tensor([[0., .2, 1.], [0., 1., 0.], [1., .4, 1.]], requires_grad=True)
        namespace, alive = dict(survival=1.), []
        for term_prob in termination:
            namespace["term_prob"] = term_prob
            exec(code, namespace)
            alive.append(namespace["survival"])
        actual, expected = production_survival(termination), torch.stack(alive)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(torch.autograd.grad(actual.sum(), termination)[0],
                                   torch.autograd.grad(expected.sum(), termination)[0], rtol=0, atol=0)


def cuda_actor_fixture():
    runner, batch, noise = latent_runner_fixture(batch_size=3, horizon=3)
    runner.cfg.precision = "float32"
    runner.cfg.training_schedule = "validated_v2"
    runner.cfg.moe_load_balance_coef, runner.cfg.moe_router_z_coef = .01, .001
    for module in (runner.actor, runner.world_model, runner.critic, runner.discriminator, runner.amp_norm):
        module.cuda().float()
    runner.actor.precision = runner.world_model.precision = "float32"
    runner.device = runner.adapter.device = torch.device("cuda")
    for name, value in vars(runner.adapter).items():
        if isinstance(value, torch.Tensor):
            setattr(runner.adapter, name, value.cuda().float() if value.is_floating_point() else value.cuda())
    runner.adapter._fk_levels = tuple(replace(level, **{
        field.name: getattr(level, field.name).cuda().float() if getattr(level, field.name).is_floating_point()
                    else getattr(level, field.name).cuda()
        for field in fields(level) if isinstance(getattr(level, field.name), torch.Tensor)
    }) for level in runner.adapter._fk_levels)
    runner.opt_actor = build_reference_optimizer([runner.actor.policy], lr=2e-4, device="cuda")
    runner.cfg.execution_profile = "supermimic"
    runner.cfg.compile, runner.cfg.compile_fullgraph = True, True
    runner._latent_step_module = LatentImaginedStep(runner)
    runner._latent_trajectory_module = LatentTrajectoryTerms(runner)
    runner._latent_step_fn = runner._compile(runner._latent_step_module, "test latent step")
    runner._latent_trajectory_fn = runner._compile(runner._latent_trajectory_module, "test trajectory")
    batch = {name: value.cuda().float() if value.is_floating_point() else value.cuda()
             for name, value in batch.items()}
    return runner, batch, noise.cuda().float()


@unittest.skipUnless(CUDA, "optional small CUDA full-update capture checks")
class CapturedUpdateTest(unittest.TestCase):
    def test_shared_pool_first_captures_macro_four_in_update_order_and_matches_eager(self):
        cfg = SimpleNamespace(training_schedule="supermimic_v1", graph_warmup=3,
                              wm_updates=2, actor_updates=1, critic_epochs=2, critic_minibatches=2,
                              max_grad_norm=1., compile_mode="default")
        resources = torch.cuda.Stream(), torch.cuda.graph_pool_handle()
        names, counts = ("wm", "actor", "critic"), (2, 1, 4)
        captures, executions, references = [], {}, {}
        for name in names:
            module = Critic(17, hidden=(16, 8)).cuda()
            reference = copy.deepcopy(module)
            optimizer = build_reference_optimizer([module], lr=2e-4, device="cuda")
            reference_optimizer = build_reference_optimizer([reference], lr=2e-4, device="cuda")
            holder = SimpleNamespace(critic=module, opt_critic=optimizer, cfg=cfg)
            eager_holder = SimpleNamespace(critic=reference, opt_critic=reference_optimizer, cfg=cfg)
            execution = make_critic_execution(holder, graph_resources=resources)
            expected = make_critic_execution(eager_holder)
            # The same scalar tensor objective exercises all three capture
            # scopes without allocating a simulator or production vision batch.
            execution.warmup = expected.warmup = graph_warmup_steps(cfg, name)
            expected.mode, expected.loss_function = "eager", expected.kernel
            executions[name], references[name] = execution, expected
        retained = []
        for macro in range(1, 7):
            for name, count in zip(names, counts):
                execution, reference = executions[name], references[name]
                for optimizer in (execution.optimizer, reference.optimizer):
                    optimizer.zero_grad(set_to_none=True)
                    for group in optimizer.param_groups:
                        group["lr"].fill_(2e-4*(1-.03*macro))
                for micro in range(count):
                    batch = {"features": torch.randn(8, 17, device="cuda")+.01*macro,
                             "target": torch.randn(8, device="cuda")+.02*micro}
                    was_captured = execution._graph is not None
                    actual, norm, applied = execution.step(batch)
                    expected, expected_norm, expected_applied = reference.step(batch)
                    if not was_captured and execution._graph is not None:
                        captures.append((macro, name))
                    self.assertEqual(bool(applied), bool(expected_applied))
                    torch.testing.assert_close(actual["loss"], expected["loss"], rtol=2e-5, atol=2e-6)
                    torch.testing.assert_close(norm, expected_norm, rtol=2e-5, atol=2e-6)
                    retained.append((actual["loss"], expected["loss"].clone()))
                for parameter, expected in zip(execution._parameters, reference._parameters):
                    torch.testing.assert_close(parameter, expected, rtol=2e-5, atol=2e-6)
                if macro < 4:
                    self.assertIsNone(execution._graph)
                else:
                    self.assertEqual(execution._graph.pool(), resources[1])
                    self.assertIs(execution._capture_stream, resources[0])
            for actual, expected in retained:
                torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
        self.assertEqual(captures, [(4, "wm"), (4, "actor"), (4, "critic")])

    def test_official_critic_capture_applies_nan_update_and_rejects_guard_change(self):
        critic = Critic(17, hidden=(16, 8)).cuda()
        optimizer = build_reference_optimizer([critic], lr=2e-4, device="cuda")
        runner = SimpleNamespace(critic=critic, opt_critic=optimizer,
            cfg=SimpleNamespace(max_grad_norm=1., graph_warmup=3, compile_mode="default",
                                training_schedule="supermimic_v1"))
        execution = make_critic_execution(runner)
        self.assertFalse(execution.guard_nonfinite)
        batch = {"features": torch.randn(8, 17, device="cuda"), "target": torch.randn(8, device="cuda")}
        for _ in range(4):
            self.assertTrue(bool(execution.step(batch)[2]))
        self.assertIsNotNone(execution._graph)
        batch["target"][0] = float("nan")
        _, norm, applied = execution.step(batch)
        self.assertTrue(bool(applied))
        self.assertFalse(bool(torch.isfinite(norm)))
        self.assertTrue(any(not bool(torch.isfinite(parameter).all()) for parameter in critic.parameters()))
        execution.guard_nonfinite = True
        with self.assertRaisesRegex(RuntimeError, "guard_nonfinite"):
            execution.step(batch)

    def test_critic_capture_survives_external_grad_clear_freeze_restore_and_bad_target(self):
        critic = Critic(17, hidden=(16, 8)).cuda()
        optimizer = build_reference_optimizer([critic], lr=2e-4, device="cuda")
        runner = SimpleNamespace(critic=critic, opt_critic=optimizer,
            cfg=SimpleNamespace(max_grad_norm=1., graph_warmup=3, compile_mode="default"))
        execution = make_critic_execution(runner)
        batch = {"features": torch.randn(8, 17, device="cuda"), "target": torch.randn(8, device="cuda")}
        for index in range(7):
            if index == 5:
                optimizer.zero_grad(set_to_none=True)
                with frozen_parameters(critic):
                    self.assertTrue(all(not parameter.requires_grad for parameter in critic.parameters()))
                    with self.assertRaisesRegex(RuntimeError, "moved or was rebound"):
                        execution.step(batch)
            _, norm, ok = execution.step(batch)
            self.assertTrue(bool(ok))
            self.assertTrue(norm.is_cuda)
            self.assertGreater(float(norm), 0.)
        state = copy.deepcopy(optimizer.state_dict())
        parameters = [parameter.detach().clone() for parameter in critic.parameters()]
        broken = {**batch, "target": batch["target"].clone()}
        broken["target"][0] = float("nan")
        _, _, ok = execution.step(broken)
        self.assertFalse(bool(ok))
        for before, after in zip(parameters, critic.parameters()):
            torch.testing.assert_close(after, before, rtol=0, atol=0)
        for identifier, values in state["state"].items():
            for name, before in values.items():
                torch.testing.assert_close(optimizer.state_dict()["state"][identifier][name], before, rtol=0, atol=0)
        self.assertTrue(bool(execution.step(batch)[2]))

    def test_actor_capture_optimizes_policy_only_and_invalidates_after_optimizer_rollback_and_horizon_change(self):
        runner, batch, noise = cuda_actor_fixture()
        execution = ActorExecution(runner)
        load = noise.new_full((4,), .25)
        frozen = (runner.world_model, runner.actor.vision, runner.critic, runner.discriminator, runner.amp_norm)
        before = [copy.deepcopy(module.state_dict()) for module in frozen]
        original = [parameter.detach().clone() for parameter in runner.actor.policy.parameters()]
        for index in range(7):
            if index == 5:
                runner.opt_actor.zero_grad(set_to_none=True)
            rollout, norm, ok = execution.step(batch, noise, load)
            self.assertTrue(bool(ok))
            self.assertTrue(norm.is_cuda)
            self.assertGreater(float(norm), 0.)
            self.assertEqual(rollout["x"].shape[:2], (3, 3))
            self.assertIn("auxiliary_loss", rollout)
            self.assertTrue(all(parameter.requires_grad for parameter in runner.world_model.parameters()))
        self.assertTrue(any(not torch.equal(parameter, saved)
                            for parameter, saved in zip(runner.actor.policy.parameters(), original)))
        for module, saved in zip(frozen, before):
            for name, value in module.state_dict().items():
                torch.testing.assert_close(value, saved[name], rtol=0, atol=0)
            self.assertTrue(all(parameter.grad is None for parameter in module.parameters()))
        weights = [parameter.detach().clone() for parameter in runner.actor.policy.parameters()]
        state = copy.deepcopy(runner.opt_actor.state_dict())
        broken_noise = noise.clone()
        broken_noise[0, 0, 0] = float("nan")
        self.assertFalse(bool(execution.step(batch, broken_noise, load)[2]))
        for actual, expected in zip(runner.actor.policy.parameters(), weights):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for identifier, values in state["state"].items():
            for name, expected in values.items():
                torch.testing.assert_close(runner.opt_actor.state_dict()["state"][identifier][name],
                                           expected, rtol=0, atol=0)
        self.assertTrue(bool(execution.step(batch, noise, load)[2]))
        saved = copy.deepcopy(runner.opt_actor.state_dict())
        runner.opt_actor.load_state_dict(saved)
        with self.assertRaisesRegex(RuntimeError, "moved or was rebound"):
            execution.step(batch, noise, load)
        execution.invalidate()
        self.assertTrue(bool(execution.step(batch, noise, load)[2]))
        old = execution.execution
        runner._active_horizon = 2
        rollout, _, ok = execution.step(batch, noise[:2], load)
        self.assertTrue(bool(ok))
        self.assertEqual(rollout["x"].shape[0], 2)
        self.assertIsNot(old, execution.execution)
        self.assertIsNone(old._graph)


if __name__ == "__main__":
    unittest.main()
