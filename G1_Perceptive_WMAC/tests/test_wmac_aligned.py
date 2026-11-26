"""Alignment and schema-safety checks on the actual WMAC update path."""

import copy
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np
import torch

from rsl_rl.modules.discriminator import Discriminator
from rsl_rl.utils.utils import Normalizer
from src.wmac.config import WmacConfig
from src.wmac.networks import Actor, RunningNorm, mlp
from test_wmac_optimization import FixedReplay, make_runner


class AlignedWmacTest(unittest.TestCase):
    def test_new_profile_preserves_wmac_schedule_and_explicit_discriminator_budget(self):
        cfg = WmacConfig.for_profile("ppo_aligned")
        cfg.validate()
        self.assertEqual(cfg.collect_steps, 0)
        self.assertEqual(cfg.utd, .4)
        self.assertEqual(cfg.warmup_steps, 64)
        self.assertEqual(cfg.iterations, 320000)
        self.assertEqual(cfg.discriminator_updates, 2)
        self.assertEqual(cfg.effective_discriminator_batch_size, 512)
        cfg.num_envs = 8
        self.assertEqual(cfg.effective_discriminator_batch_size, 512)
        cfg.discriminator_batch_size = 7
        self.assertEqual(cfg.effective_discriminator_batch_size, 7)
        # The old fixed-rollout auto-batch option remains explicit and valid.
        cfg.collect_steps, cfg.discriminator_batch_size = 24, 0
        cfg.validate()
        self.assertEqual(cfg.effective_discriminator_batch_size, 48)
        cfg.collect_steps = 0
        with self.assertRaisesRegex(ValueError, "fixed collect_steps"):
            cfg.validate()
        self.assertTrue(cfg.direct_style_grad and cfg.critic_bootstrap and cfg.survival_grad)
        self.assertEqual(cfg.real_critic_coef, 0.)
        self.assertEqual(cfg.amp_feature_schema, "grounded_v2")
        self.assertEqual(WmacConfig.for_profile("supermimic").amp_feature_schema, "legacy")

    def test_actor_initialization_can_match_the_ppo_linear_initialization_law(self):
        torch.manual_seed(72)
        expected = mlp(384, (16, 8), 29)
        torch.manual_seed(72)
        actual = Actor(hidden=(16, 8), output_init_scale=1., zero_output_bias=False)
        for key, value in expected.state_dict().items():
            torch.testing.assert_close(actual.net.state_dict()[key], value, rtol=0, atol=0)

    def test_grounded_actor_bptt_is_finite_and_does_not_update_other_networks(self):
        runner, batch, noises = make_runner(batch_size=3, dtype=torch.float32)
        runner.adapter.amp_feature_schema = "grounded_v2"
        runner.adapter.amp_dim = 203
        runner.cfg.amp_feature_schema = "grounded_v2"
        runner.cfg.task = "Unitree-G1-AMP-Flat-Aligned"
        runner.discriminator = Discriminator(406, .1, [12, 6], "cpu", .75)
        runner.amp_norm = RunningNorm(203)
        original = {name: copy.deepcopy(getattr(runner, name).state_dict())
                    for name in ("world_model", "discriminator", "critic")}
        # Use the same production update entry point and paired observation noise.
        original_imagine = runner.imagine
        runner.imagine = lambda values: original_imagine(values, noises=noises)
        runner._fit_critic = lambda *args: {"critic_loss": 0., "real_critic_loss": 0.,
            "critic_grad_norm": 0., "critic_updates_applied_fraction": 0.}
        result = runner.update_actor_critic()
        self.assertEqual(result["actor_update_rejected"], 0.)
        self.assertTrue(np.isfinite(result["actor_grad_norm"]))
        for name, before in original.items():
            for key, value in before.items():
                torch.testing.assert_close(getattr(runner, name).state_dict()[key], value, rtol=0, atol=0)

    def test_discriminator_updates_moments_from_all_four_raw_203d_batches(self):
        runner, values, _ = make_runner(batch_size=3, dtype=torch.float32)
        runner.adapter.amp_feature_schema = "grounded_v2"
        runner.adapter.amp_dim = 203
        runner.cfg.amp_normalizer_per_update = True
        runner.cfg.discriminator_updates, runner.cfg.discriminator_batch_size = 1, 3
        runner.discriminator = Discriminator(406, .1, [12, 6], "cpu", .75)
        runner.opt_disc = torch.optim.Adam(runner.discriminator.parameters(), lr=1e-4)
        runner.amp_norm = RunningNorm(203, eps=1e-4, max_count=float("inf"))
        runner.amp_norm.variance_epsilon, runner.amp_norm.clip = True, 10.
        runner.amp_norm.count.fill_(1e-4)
        values["next_state"] = values["state"].clone()
        values["next_state"][:, 2] += .03
        runner.replay = FixedReplay(values)
        ef = torch.linspace(-2., 3., 3 * 203).reshape(3, 203)
        es = ef + .4
        runner.experts = SimpleNamespace(sample=lambda size: (ef, es))
        first = runner.adapter.amp_features(values["state"])
        second = runner.adapter.amp_features(values["next_state"])
        expected = Normalizer(203)
        expected.update(torch.cat((first, second, ef, es)).numpy())
        result = runner.update_discriminator()
        self.assertTrue(np.isfinite(result["discriminator_loss"]))
        np.testing.assert_allclose(runner.amp_norm.mean.numpy(), expected.mean, rtol=2e-6, atol=2e-6)
        np.testing.assert_allclose(runner.amp_norm.var.numpy(), expected.var, rtol=2e-6, atol=2e-6)
        self.assertAlmostEqual(float(runner.amp_norm.count), expected.count, places=6)
        sample = torch.cat((first, second))
        torch.testing.assert_close(runner.amp_norm(sample), expected.normalize_torch(sample, "cpu"),
                                   rtol=2e-6, atol=2e-6)

    def test_checkpoint_schema_is_rejected_before_mutating_network_weights(self):
        runner, _, _ = make_runner()
        actor_before = copy.deepcopy(runner.actor.state_dict())
        payload = {"format": "amp-wmac-v1", "config": asdict(runner.cfg),
                   "amp_feature_schema": "grounded_v2", "amp_feature_dim": 203}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "grounded.pt"
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, "schema mismatch"):
                runner.load(path)
        for key, value in actor_before.items():
            torch.testing.assert_close(runner.actor.state_dict()[key], value, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
