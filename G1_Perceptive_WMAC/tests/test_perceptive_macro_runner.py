"""Bounded CPU equivalence through the public v2 policy runner entrypoint."""
from __future__ import annotations

import math
import unittest
from unittest.mock import patch, sentinel

import torch

from src.wmac_perceptive.macro_execution import ActorCriticMacroExecution
from test_perceptive_macro_execution import g1_runner, assert_optimizer_equal
from test_wmac_optimization import FixedReplay, fix_update_noise


class MacroRunnerTest(unittest.TestCase):
    def test_setup_keeps_policy_microbatch_fallback_while_wm_uses_complete_macro(self):
        runner, _, _ = g1_runner()
        runner.cfg.cuda_graphs = True
        runner.cfg.execution_optimizer = "muon_adam"
        runner.device = torch.device("cuda:0")
        width = runner.cfg.actor_batch_size
        for microbatch in (0, width, width + 2, 1):
            with self.subTest(actor_microbatch_size=microbatch), \
                    patch.object(runner, "_compile", side_effect=lambda module, label: module), \
                    patch("torch.cuda.Stream", return_value=sentinel.stream), \
                    patch("torch.cuda.graph_pool_handle", return_value=sentinel.pool), \
                    patch("src.wmac_perceptive.latent_execution.WMExecution") as wm, \
                    patch("src.wmac_perceptive.wm_macro_execution.WMMacroExecution") as wm_macro, \
                    patch("src.wmac_perceptive.macro_execution.ActorCriticMacroExecution") as policy_macro, \
                    patch("src.wmac_perceptive.captured_updates.ActorExecution") as actor, \
                    patch("src.wmac_perceptive.captured_updates.make_critic_execution") as critic:
                runner.cfg.actor_microbatch_size = microbatch
                runner._setup_fast_execution()
                resources = (sentinel.stream, sentinel.pool)
                wm.assert_called_once()
                wm_macro.assert_called_once_with(runner, graph_resources=resources)
                self.assertIs(runner._latent_wm_execution, wm_macro.return_value)
                if not microbatch or microbatch >= width:
                    policy_macro.assert_called_once_with(runner, graph_resources=resources)
                    self.assertIs(runner._latent_actor_critic_macro_execution, policy_macro.return_value)
                    self.assertIsNone(runner._latent_actor_execution)
                    self.assertIsNone(runner._latent_critic_execution)
                    actor.assert_not_called()
                    critic.assert_not_called()
                else:
                    policy_macro.assert_not_called()
                    self.assertIsNone(runner._latent_actor_critic_macro_execution)
                    actor.assert_called_once_with(runner, graph_resources=resources)
                    critic.assert_called_once_with(runner, graph_resources=resources)
                    self.assertIs(runner._latent_actor_execution, actor.return_value)
                    self.assertIs(runner._latent_critic_execution, critic.return_value)

    def test_public_update_matches_separate_path_metrics_parameters_and_counters(self):
        actual, batch, noise = g1_runner()
        expected, _, _ = g1_runner()
        for name in ("actor", "world_model", "critic", "discriminator", "amp_norm"):
            getattr(expected, name).load_state_dict(getattr(actual, name).state_dict())
        for runner in (actual, expected):
            runner.replay = FixedReplay(batch)
            runner.ACTOR_FIELDS = tuple(batch)
            # The model/AMP modules have been replaced by the coupled G1
            # fixture. Bind their actual parameters to the clearing owners.
            runner.opt_model = torch.optim.Adam(runner._model_training_modules.parameters(), lr=.001)
            runner.opt_disc = torch.optim.Adam(runner.discriminator.parameters(), lr=.0001)
            for module in (runner.world_model, runner.actor.vision, runner.discriminator):
                for parameter in module.parameters():
                    parameter.grad = torch.ones_like(parameter)
            fix_update_noise(runner, noise)
        actual._latent_actor_critic_macro_execution = ActorCriticMacroExecution(actual, cuda_graphs=False)
        expected._latent_actor_critic_macro_execution = None
        state = torch.get_rng_state()
        observed = actual.update_actor_critic()
        torch.set_rng_state(state)
        reference = expected.update_actor_critic()
        self.assertEqual(observed.keys(), reference.keys())
        for key in observed:
            with self.subTest(metric=key):
                if math.isnan(reference[key]):
                    self.assertTrue(math.isnan(observed[key]))
                else:
                    torch.testing.assert_close(torch.tensor(observed[key], dtype=torch.float64),
                        torch.tensor(reference[key], dtype=torch.float64), rtol=1e-8, atol=1e-10)
        for name in ("actor", "critic", "world_model", "discriminator"):
            for parameter, old in zip(getattr(actual, name).parameters(), getattr(expected, name).parameters()):
                torch.testing.assert_close(parameter, old, rtol=1e-10, atol=1e-12)
        assert_optimizer_equal(self, actual.opt_actor, expected.opt_actor)
        assert_optimizer_equal(self, actual.opt_critic, expected.opt_critic)
        for name in ("actor_updates_accepted", "actor_updates_rejected", "imagination_steps",
                     "_last_actor_update_iteration", "critic_calibration_updates"):
            self.assertEqual(getattr(actual, name), getattr(expected, name))
        for module in (actual.world_model, actual.actor.vision, actual.discriminator, actual.critic):
            self.assertTrue(all(parameter.grad is None for parameter in module.parameters()))
        self.assertEqual(actual.actor_updates_accepted, 1)
        self.assertEqual(actual.imagination_steps, 9)


if __name__ == "__main__":
    unittest.main()
