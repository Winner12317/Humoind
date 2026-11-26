"""Coupled v2 WM/Actor regression on real G1 FK, without a simulator run."""
from __future__ import annotations

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch

from src.wmac.networks import Critic, frozen_parameters
from src.wmac_perceptive.latent_execution import LatentWorldModelKernel
from src.wmac_perceptive.latent_models import LatentPerceptiveWorldModel
from src.wmac_perceptive.reference_policy_step import HikingWorldModelStep
from src.wmac_perceptive.runner import PerceptiveRunner
from test_reference_policy_step import reference_loop_fixture


def v2_reference_fixture():
    runner, batch, noises = reference_loop_fixture()
    runner.cfg.hiking_training_version = 2
    runner.adapter.hiking_training_version = 2
    runner.adapter.context_dim = 133
    materials = torch.linspace(.3, 1.7, 33, dtype=batch["context"].dtype)
    batch["context"] = torch.cat((batch["context"], materials.expand(len(batch["state"]), -1)), -1)
    runner.world_model = LatentPerceptiveWorldModel(runner.adapter.terrain, runner.spec,
        hidden=(16,), context_dim=133, contact_dim=13, continuous_normalization=True).double()
    with torch.no_grad():
        runner.world_model.net[-1].weight.normal_(0., .002)
    runner.critic = Critic(315, hidden=(16, 8)).double()
    runner._model_training_modules = torch.nn.ModuleList((runner.world_model, runner.actor.vision,
        runner.latent_decoder))
    runner._value_fn = runner.critic.forward
    runner._reference_step_fn = HikingWorldModelStep(runner)
    return runner, batch, noises


class HikingTrainingV2Test(unittest.TestCase):
    def test_imagination_with_all_materials_has_finite_actor_gradients_and_compile_parity(self):
        torch.manual_seed(20261117)
        runner, batch, noises = v2_reference_fixture()
        parameters = tuple(runner.actor.parameters())
        with frozen_parameters(runner.world_model, runner.discriminator, runner.critic):
            expected = runner.imagine(batch, noises=noises)
            gradients = torch.autograd.grad(expected["objective"].sum(), parameters, allow_unused=True)
            runner._reference_step_fn = torch.compile(HikingWorldModelStep(runner), backend="aot_eager",
                fullgraph=True, dynamic=False)
            actual = runner.imagine(batch, noises=noises)
            actual_gradients = torch.autograd.grad(actual["objective"].sum(), parameters, allow_unused=True)
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], rtol=2e-8, atol=2e-10)
        norm = 0.
        for left, right in zip(gradients, actual_gradients):
            if left is None:
                self.assertIsNone(right)
            else:
                self.assertTrue(bool(torch.isfinite(right).all()))
                torch.testing.assert_close(left, right, rtol=2e-7, atol=2e-9)
                norm += float(right.abs().sum())
        self.assertGreater(norm, 0.)
        self.assertFalse(actual["x"].requires_grad)
        self.assertEqual(runner.world_model.in_norm.mean.numel(), 311)
        self.assertEqual(runner.world_model.net[0].in_features, 439)
        self.assertEqual(runner.critic.norm.mean.numel(), 315)

    def test_wm_multistep_loss_keeps_material_conditions_and_continuous_statistics(self):
        runner, starts, _ = v2_reference_fixture()
        steps, size = runner.cfg.wm_window, len(starts["state"])
        expand = lambda value: value[:, None].expand(size, steps, *value.shape[1:]).clone()
        batch = dict(state=expand(starts["state"]), next_state=expand(starts["state"]),
            action=expand(starts["previous_action"]), context=expand(starts["context"]),
            contact_cost=torch.zeros(size, steps, 13, dtype=torch.float64),
            depth_stack=expand(starts["depth_stack"]), next_depth_stack=expand(starts["depth_stack"]),
            mask=torch.ones(size, steps, dtype=torch.bool))
        runner.world_model.in_norm.count.fill_(2e8)
        runner.world_model.update_normalizers(starts["state"], starts["previous_action"], starts["context"], starts["state"])
        kernel = LatentWorldModelKernel(runner)
        prepared = kernel.prepare(batch, update_statistics=True)
        self.assertGreater(float(runner.world_model.in_norm.count), 2e8)
        seen = []
        original = runner.world_model.step_with_latent
        def record(state, action, context, latent):
            seen.append(context.detach().clone())
            return original(state, action, context, latent)
        from unittest.mock import patch
        with patch.object(runner.world_model, "step_with_latent", side_effect=record):
            loss, _ = kernel(prepared)
        self.assertTrue(bool(torch.isfinite(loss).all()))
        loss.sum().backward()
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in runner.world_model.parameters()
            if p.grad is not None), 0.)
        self.assertEqual(len(seen), steps)
        for context in seen:
            torch.testing.assert_close(context[:, 100:], starts["context"][:, 100:], rtol=0, atol=0)

    def test_resume_rejects_populated_replay_before_changing_parameters(self):
        runner, _, _ = v2_reference_fixture()
        runner.replay = SimpleNamespace(total_steps=1)
        before = {name: value.clone() for name, value in runner.world_model.state_dict().items()}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"model.pt"
            torch.save({}, path)
            with self.assertRaisesRegex(ValueError, "fresh runner with empty Replay"):
                PerceptiveRunner.load(runner, path)
        for name, value in runner.world_model.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
