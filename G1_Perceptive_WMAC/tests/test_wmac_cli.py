"""CPU checks for checkpoint configuration provenance and explicit overrides."""

import importlib.util
from pathlib import Path
import tempfile
import unittest

import torch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "train_wmac.py"
SPEC = importlib.util.spec_from_file_location("wmac_training_cli", SCRIPT)
CLI = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CLI)


class ResumeConfigurationTest(unittest.TestCase):
    def checkpoint(self, directory, config):
        path = Path(directory) / "model.pt"
        torch.save({"format": "amp-wmac-v1", "config": config}, path)
        return str(path)

    def test_resume_rejects_a_conflicting_explicit_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = self.checkpoint(directory, {"profile": "balanced"})
            with self.assertRaisesRegex(ValueError, "--init-policy"):
                CLI.parse_config(["--resume", checkpoint, "--profile", "supermimic"])

    def test_unlabelled_legacy_checkpoint_cannot_be_relabelled(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = self.checkpoint(directory, {"imagination_horizon": 16})
            with self.assertRaisesRegex(ValueError, "profile 'legacy'"):
                CLI.parse_config(["--resume", checkpoint, "--profile", "balanced"])

    def test_matching_profile_keeps_saved_values_and_explicit_iteration_override(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = self.checkpoint(directory, {
                "profile": "balanced", "iterations": 300, "num_envs": 64,
                "imagination_horizon": 24, "init_policy": "/previous/source.pt",
            })
            config = CLI.parse_config([
                "--resume", checkpoint, "--profile", "balanced", "--iterations", "500",
            ])
            self.assertEqual(config.profile, "balanced")
            self.assertEqual(config.iterations, 500)
            self.assertEqual(config.num_envs, 64)
            self.assertEqual(config.imagination_horizon, 24)
            self.assertIsNone(config.init_policy)
            self.assertEqual(config.resume, checkpoint)

    def test_legacy_resume_allows_individual_explicit_parameter_overrides(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = self.checkpoint(directory, {
                "iterations": 200, "imagination_horizon": 16, "action_limit": 5.0,
            })
            config = CLI.parse_config([
                "--resume", checkpoint, "--imagination-horizon", "32", "--iterations", "300",
            ])
            self.assertEqual(config.profile, "legacy")
            self.assertEqual(config.imagination_horizon, 32)
            self.assertEqual(config.action_limit, 5.0)
            self.assertEqual(config.iterations, 300)

    def test_new_aligned_run_restores_wmac_schedule_without_overriding_explicit_cli(self):
        config = CLI.parse_config(["--profile", "ppo_aligned"])
        config.validate()
        self.assertEqual((config.collect_steps, config.utd, config.warmup_steps), (0, .4, 64))
        self.assertEqual((config.discriminator_updates, config.discriminator_batch_size), (2, 512))
        overridden = CLI.parse_config(["--profile", "ppo_aligned", "--collect-steps", "24",
            "--warmup-steps", "72", "--discriminator-updates", "20", "--discriminator-batch-size", "0"])
        overridden.validate()
        self.assertEqual(overridden.collect_steps, 24)
        self.assertEqual(overridden.effective_discriminator_batch_size, 24576)

    def test_saved_fixed_schedule_is_preserved_until_explicitly_migrated(self):
        from dataclasses import asdict
        from src.wmac.config import WmacConfig

        saved = asdict(WmacConfig.for_profile("ppo_aligned"))
        saved.update(collect_steps=24, warmup_steps=72, discriminator_updates=20,
                     discriminator_batch_size=0, iterations=60000)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = self.checkpoint(directory, saved)
            original = CLI.parse_config(["--resume", checkpoint])
            original.validate()
            self.assertEqual((original.collect_steps, original.warmup_steps), (24, 72))
            self.assertEqual((original.discriminator_updates, original.effective_discriminator_batch_size), (20, 24576))
            migrated = CLI.parse_config(["--resume", checkpoint, "--collect-steps", "0",
                "--utd", "0.4", "--warmup-steps", "64", "--discriminator-updates", "2",
                "--discriminator-batch-size", "512", "--iterations", "320000"])
            migrated.validate()
            self.assertEqual((migrated.collect_steps, migrated.utd, migrated.warmup_steps), (0, .4, 64))
            self.assertEqual((migrated.discriminator_updates, migrated.effective_discriminator_batch_size), (2, 512))
            self.assertEqual(migrated.resume, checkpoint)
            self.assertEqual(migrated.actor_lr, original.actor_lr)
            self.assertEqual(migrated.amp_feature_schema, "grounded_v2")

    def test_unlabelled_resume_keeps_old_recovery_modes_and_discards_saved_authorization(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = self.checkpoint(directory, {
                "profile": "ppo_aligned", "allow_recovery_semantics_migration": True,
            })
            restored = CLI.parse_config(["--resume", checkpoint])
            self.assertEqual(restored.recovery_counter_semantics, "soft_v1")
            self.assertEqual(restored.recovery_height_reward, "height_only_v1")
            self.assertFalse(restored.allow_recovery_semantics_migration)
            for flags in (["--recovery-counter-semantics", "hard_forward_v2"],
                          ["--recovery-height-reward", "upright_v2"]):
                with self.subTest(flags=flags), self.assertRaisesRegex(ValueError, "explicitly pass"):
                    CLI.parse_config(["--resume", checkpoint, *flags])
                migrated = CLI.parse_config(["--resume", checkpoint, *flags,
                                            "--allow-recovery-semantics-migration"])
                self.assertTrue(migrated.allow_recovery_semantics_migration)
                self.assertEqual(migrated.recovery_counter_semantics,
                                 "hard_forward_v2" if "--recovery-counter-semantics" in flags else "soft_v1")
                self.assertEqual(migrated.recovery_height_reward,
                                 "upright_v2" if "--recovery-height-reward" in flags else "height_only_v1")


if __name__ == "__main__":
    unittest.main()
