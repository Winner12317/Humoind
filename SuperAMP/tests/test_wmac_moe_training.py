"""CPU checks of MoE regularization, matched microbatches and frozen BPTT.

These use the real runner update and G1 adapter with a fixed replay batch. They
test training semantics, not whether sparse routing improves a trained policy.
"""
from __future__ import annotations

import copy
import unittest
from unittest.mock import patch

import torch

from src.wmac.actors import create_actor
from src.wmac.networks import frozen_parameters
from src.wmac.routing import load_fraction, router_losses
from test_wmac_optimization import fix_update_noise, make_runner


def make_moe_runner(batch_size=5, horizon=3):
    runner, batch, noises = make_runner(batch_size=batch_size, horizon=horizon)
    original = runner.actor
    runner.cfg.actor_type = "moe"
    runner.cfg.moe_variant = "legacy"
    runner.cfg.moe_num_experts = 4
    runner.cfg.moe_top_k = 2
    runner.cfg.moe_gate_hidden = (8,)
    runner.cfg.moe_temperature = .8
    runner.cfg.moe_load_balance_coef = .2
    runner.cfg.moe_router_z_coef = .03
    runner.actor = create_actor(runner.cfg).to(dtype=torch.float64)
    runner.actor.load_mlp_state_dict(original.state_dict(), noise_std=.01)
    runner.opt_actor = torch.optim.SGD(runner.actor.parameters(), lr=.02)

    # Split the real replay observations across expert pairs. Uneven chunks
    # then have distinct loads, so recomputing each chunk's own fraction fails
    # the full-batch gradient oracle below rather than passing by symmetry.
    observation = runner.actor.norm(batch["actor_history"])
    spread = observation.amax(0)-observation.amin(0)
    column = int(spread.argmax())
    scale = 4./float(spread[column])
    middle = float((observation[:, column].amax()+observation[:, column].amin())/2)
    first, last = runner.actor.router[0], runner.actor.router[-1]
    with torch.no_grad():
        first.weight.zero_(); first.bias.zero_()
        first.weight[0, column] = scale; first.bias[0] = -middle*scale
        first.weight[1, column] = -scale; first.bias[1] = middle*scale
        last.weight.zero_(); last.bias.zero_()
        last.weight[0, 0] = 1.; last.weight[1, 1] = 1.
        last.bias[2:] = .4
        last.weight[3, 0] = .3; last.weight[3, 1] = -.3
    return runner, batch, noises


def critic_noop():
    return dict(critic_loss=0., real_critic_loss=0., critic_grad_norm=0.,
                critic_updates_applied_fraction=0.)


def cloned_state(module):
    return {key: value.detach().clone() for key, value in module.state_dict().items()}


class MoEActorTrainingTest(unittest.TestCase):
    def test_hiking_actual_wmac_bptt_updates_all_experts_without_aux_or_frozen_parameter_leaks(self):
        runner, batch, noise = make_runner(batch_size=4, horizon=3)
        runner.cfg.actor_type = "moe"  # New defaults select exact Hiking.
        runner.cfg.validate()
        runner.actor = create_actor(runner.cfg).double()
        runner.opt_actor = torch.optim.SGD(runner.actor.parameters(), lr=.02)
        actor_before = cloned_state(runner.actor)
        frozen = (runner.world_model, runner.discriminator, runner.critic)
        frozen_before = [cloned_state(module) for module in frozen]
        fix_update_noise(runner, noise)
        with patch.object(runner, "_fit_critic", return_value=critic_noop()):
            result = runner.update_actor_critic()
        self.assertEqual(result["actor_update_rejected"], 0.)
        self.assertEqual(result["actor_router_aux_loss"], 0.)
        self.assertEqual(result["actor_loss"], result["actor_task_loss"])
        for name, module in (("router", runner.actor.router),
                             *((f"experts.{i}", expert) for i, expert in enumerate(runner.actor.experts))):
            gradients = [p.grad for p in module.parameters()]
            self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in gradients))
            self.assertGreater(sum(float(g.abs().sum()) for g in gradients), 0.)
            self.assertTrue(any(not torch.equal(value, actor_before[f"{name}.{key}"])
                                for key, value in module.state_dict().items()))
        for module, before in zip(frozen, frozen_before):
            self.assert_state_equal(module, before)
            self.assertTrue(all(p.grad is None for p in module.parameters()))

    def assert_state_equal(self, module, expected):
        for key, value in module.state_dict().items():
            torch.testing.assert_close(value, expected[key], rtol=0, atol=0)

    def test_detached_full_batch_load_preserves_uneven_chunk_loss_and_gradient(self):
        runner, batch, _ = make_moe_runner(batch_size=7)
        actor, observations = runner.actor, batch["actor_history"]
        balance, z, route = router_losses(actor, observations)
        full_load = load_fraction(route.indices, actor.num_experts, route.probabilities.dtype)
        self.assertFalse(full_load.requires_grad)
        self.assertIsNone(full_load.grad_fn)
        self.assertEqual(int((full_load > 0).sum()), 4)
        full_loss = .2*balance+.03*z
        parameters = tuple(actor.router.parameters())
        expected_gradients = torch.autograd.grad(full_loss, parameters)

        chunk_loss, incorrect_local_load_loss = [], []
        for start in range(0, len(observations), 2):
            chunk = observations[start:start+2]
            weight = len(chunk)/len(observations)
            b, regularizer, _ = router_losses(actor, chunk, full_load)
            chunk_loss.append(weight*(.2*b+.03*regularizer))
            local_b, local_z, _ = router_losses(actor, chunk)
            incorrect_local_load_loss.append(weight*(.2*local_b+.03*local_z))
        actual_loss = sum(chunk_loss)
        actual_gradients = torch.autograd.grad(actual_loss, parameters)
        torch.testing.assert_close(actual_loss, full_loss, rtol=2e-14, atol=2e-14)
        for actual, expected in zip(actual_gradients, expected_gradients):
            torch.testing.assert_close(actual, expected, rtol=2e-12, atol=2e-13)
        incorrect_gradients = torch.autograd.grad(sum(incorrect_local_load_loss), parameters)
        # The fixture must detect the tempting but incorrect chunk-local loss.
        self.assertGreater(sum(float((a-b).abs().sum()) for a, b in
                               zip(incorrect_gradients, expected_gradients)), 1e-4)

    def test_whole_and_uneven_microbatches_make_the_same_actor_update(self):
        whole, _, whole_noise = make_moe_runner()
        micro, _, micro_noise = make_moe_runner()
        micro.cfg.actor_microbatch_size = 2  # 2+2+1, not equal-sized chunk means.
        fix_update_noise(whole, whole_noise)
        fix_update_noise(micro, micro_noise)
        with patch.object(whole, "_fit_critic", return_value=critic_noop()), \
             patch.object(micro, "_fit_critic", return_value=critic_noop()):
            whole_result = whole.update_actor_critic()
            micro_result = micro.update_actor_critic()
        self.assertEqual(whole_result["actor_update_rejected"], 0.)
        self.assertEqual(micro_result["actor_update_rejected"], 0.)
        for key in ("actor_loss", "actor_task_loss", "actor_router_aux_loss"):
            self.assertAlmostEqual(whole_result[key], micro_result[key], places=10)
        for expected, actual in zip(whole.actor.parameters(), micro.actor.parameters()):
            torch.testing.assert_close(actual, expected, rtol=2e-9, atol=2e-11)
            if expected.grad is None:
                self.assertIsNone(actual.grad)
            else:
                torch.testing.assert_close(actual.grad, expected.grad, rtol=2e-9, atol=2e-10)
        self.assertEqual(whole.imagination_steps, 15)
        self.assertEqual(micro.imagination_steps, 15)

    def test_auxiliary_uses_real_starts_and_does_not_change_returns_or_critic_targets(self):
        regularized, batch, noises = make_moe_runner()
        pure, _, pure_noises = make_moe_runner()
        pure.cfg.moe_load_balance_coef = pure.cfg.moe_router_z_coef = 0.
        with frozen_parameters(regularized.world_model, regularized.discriminator, regularized.critic), \
             frozen_parameters(pure.world_model, pure.discriminator, pure.critic):
            actual = regularized.imagine(batch, noises=noises)
            expected = pure.imagine(batch, noises=pure_noises)
        for key in actual:
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
        del actual, expected

        targets, router_inputs = [], []
        def capture_targets(x, target):
            self.assertFalse(x.requires_grad)
            self.assertFalse(target.requires_grad)
            self.assertIsNone(target.grad_fn)
            targets.append((x.clone(), target.clone()))
            return critic_noop()
        def observe_aux(actor, observations, full_batch_load):
            self.assertFalse(observations.requires_grad)
            self.assertFalse(full_batch_load.requires_grad)
            router_inputs.append(observations.clone())
            return router_losses(actor, observations, full_batch_load)
        fix_update_noise(regularized, noises)
        fix_update_noise(pure, pure_noises)
        with patch("src.wmac.runner.router_losses", side_effect=observe_aux), \
             patch.object(regularized, "_fit_critic", side_effect=capture_targets):
            result = regularized.update_actor_critic()
        self.assertEqual(len(router_inputs), 1)
        torch.testing.assert_close(router_inputs[0], batch["actor_history"], rtol=0, atol=0)
        with patch.object(pure, "_fit_critic", side_effect=capture_targets):
            pure_result = pure.update_actor_critic()
        for actual, expected in zip(targets[0], targets[1]):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertGreater(result["actor_router_aux_loss"], 0.)
        self.assertEqual(pure_result["actor_router_aux_loss"], 0.)
        self.assertEqual(result["actor_task_loss"], pure_result["actor_task_loss"])
        self.assertAlmostEqual(result["actor_loss"],
                               result["actor_task_loss"]+result["actor_router_aux_loss"], places=12)
        # With SGD and the same task graph, aux changes the router update only.
        for expert, reference in zip(regularized.actor.experts, pure.actor.experts):
            self.assert_state_equal(expert, reference.state_dict())
        self.assertTrue(any(not torch.equal(a, b) for a, b in
                            zip(regularized.actor.router.parameters(), pure.actor.router.parameters())))

    def test_zero_auxiliary_coefficients_match_the_pure_imagination_gradient(self):
        runner, _, noises = make_moe_runner()
        oracle, batch, oracle_noise = make_moe_runner()
        runner.cfg.moe_load_balance_coef = runner.cfg.moe_router_z_coef = 0.
        oracle.cfg.moe_load_balance_coef = oracle.cfg.moe_router_z_coef = 0.
        with frozen_parameters(oracle.world_model, oracle.discriminator, oracle.critic):
            loss = -oracle.imagine(batch, noises=oracle_noise)["objective"].mean()
            loss.backward()
        oracle.opt_actor.step()
        fix_update_noise(runner, noises)
        with patch.object(runner, "_fit_critic", return_value=critic_noop()):
            result = runner.update_actor_critic()
        self.assertEqual(result["actor_router_aux_loss"], 0.)
        self.assertEqual(result["actor_loss"], result["actor_task_loss"])
        self.assertGreater(result["moe_router_grad_norm"], 0.)
        for actual, expected in zip(runner.actor.parameters(), oracle.actor.parameters()):
            torch.testing.assert_close(actual, expected, rtol=2e-10, atol=2e-11)
            torch.testing.assert_close(actual.grad, expected.grad, rtol=2e-9, atol=2e-10)

    def test_critic_only_calibration_keeps_router_and_experts_frozen_with_aux_enabled(self):
        runner, _, noises = make_moe_runner()
        fix_update_noise(runner, noises)
        actor_before, critic_before = cloned_state(runner.actor), cloned_state(runner.critic)
        world_before, discriminator_before = cloned_state(runner.world_model), cloned_state(runner.discriminator)
        optimizer_before = copy.deepcopy(runner.opt_actor.state_dict())
        with patch("src.wmac.runner.router_losses", side_effect=AssertionError("Actor aux during calibration")), \
             patch.object(runner.opt_actor, "step", wraps=runner.opt_actor.step) as step:
            result = runner.update_actor_critic(update_actor=False)
        step.assert_not_called()
        self.assert_state_equal(runner.actor, actor_before)
        self.assert_state_equal(runner.world_model, world_before)
        self.assert_state_equal(runner.discriminator, discriminator_before)
        self.assertEqual(runner.opt_actor.state_dict(), optimizer_before)
        self.assertTrue(all(p.grad is None for p in runner.actor.parameters()))
        self.assertTrue(any(not torch.equal(value, critic_before[key])
                            for key, value in runner.critic.state_dict().items()))
        self.assertEqual(result["actor_router_aux_loss"], 0.)
        self.assertEqual(result["actor_grad_norm"], 0.)
        self.assertEqual(result["moe_router_grad_norm"], 0.)
        self.assertEqual(result["moe_expert_grad_norm_max"], 0.)
        self.assertEqual(result["actor_action_shift"], 0.)
        self.assertEqual(result["critic_updates_applied_fraction"], 1.)

    def test_disabled_z_loss_cannot_reject_an_otherwise_finite_actor_update(self):
        for balance_coef in (0., .2):
            with self.subTest(balance_coef=balance_coef):
                runner, batch, noises = make_moe_runner()
                runner.cfg.moe_load_balance_coef = balance_coef
                runner.cfg.moe_router_z_coef = 0.
                with torch.no_grad():
                    # A common finite logit offset leaves the route valid while
                    # its unused squared z regularizer overflows. A disabled
                    # term must not introduce 0*inf, including when another
                    # router regularizer remains enabled.
                    runner.actor.router[-1].bias.fill_(1e200)
                with frozen_parameters(runner.world_model, runner.discriminator, runner.critic):
                    pure_loss = -runner.imagine(batch, noises=noises)["objective"].mean()
                    gradients = torch.autograd.grad(pure_loss, tuple(runner.actor.parameters()), allow_unused=True)
                self.assertTrue(torch.isfinite(pure_loss))
                self.assertTrue(all(g is None or torch.isfinite(g).all() for g in gradients))
                balance, z, _ = router_losses(runner.actor, batch["actor_history"])
                self.assertTrue(torch.isinf(z))
                expected_aux = balance_coef*float(balance)
                fix_update_noise(runner, noises)
                with patch.object(runner, "_fit_critic", return_value=critic_noop()):
                    result = runner.update_actor_critic()
                self.assertEqual(result["actor_router_aux_loss"], expected_aux)
                self.assertEqual(result["actor_update_rejected"], 0.)
                self.assertAlmostEqual(result["actor_loss"], float(pure_loss)+expected_aux, places=12)

    def test_actor_backward_does_not_leak_to_frozen_networks_or_change_normalizers(self):
        runner, batch, noises = make_moe_runner()
        frozen = (runner.world_model, runner.discriminator, runner.critic)
        # Preserve an existing nontrainable parameter as well as ordinary ones.
        next(runner.world_model.parameters()).requires_grad_(False)
        requires_before = [[p.requires_grad for p in module.parameters()] for module in frozen]
        network_before = [cloned_state(module) for module in frozen]
        actor_norm_before = cloned_state(runner.actor.norm)
        amp_norm_before = cloned_state(runner.amp_norm)
        batch_before = {key: value.clone() for key, value in batch.items()}
        original_step = runner.opt_actor.step
        def check_no_leak():
            for module in frozen:
                self.assertTrue(all(p.grad is None for p in module.parameters()))
            for module, flags in zip(frozen, requires_before):
                self.assertEqual([p.requires_grad for p in module.parameters()], flags)
        def observed_step(*args, **kwargs):
            check_no_leak()
            for module in (runner.actor.router, *runner.actor.experts):
                gradients = [p.grad for p in module.parameters() if p.grad is not None]
                self.assertTrue(gradients)  # This fixture routes tokens to all E.
                self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
                self.assertGreater(sum(float(g.abs().sum()) for g in gradients), 0.)
            return original_step(*args, **kwargs)
        def observed_fit(x, target):
            check_no_leak()
            return critic_noop()
        fix_update_noise(runner, noises)
        with patch.object(runner.opt_actor, "step", side_effect=observed_step), \
             patch.object(runner, "_fit_critic", side_effect=observed_fit):
            result = runner.update_actor_critic()
        self.assertEqual(result["actor_update_rejected"], 0.)
        check_no_leak()
        for module, before in zip(frozen, network_before):
            self.assert_state_equal(module, before)
        self.assert_state_equal(runner.actor.norm, actor_norm_before)
        self.assert_state_equal(runner.amp_norm, amp_norm_before)
        for key, value in batch.items():
            torch.testing.assert_close(value, batch_before[key], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
