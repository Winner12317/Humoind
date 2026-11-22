"""Reward migrations must be explicit and cannot change the underlying method."""
from __future__ import annotations

import copy
from dataclasses import asdict
import importlib.util
from pathlib import Path
import tempfile
import unittest

import torch

from src.wmac.actors import actor_architecture
from src.wmac.checkpoints import resolve_checkpoint_config
from src.wmac.config import WmacConfig
from src.wmac.reference import (amp_reward_protocol, checkpoint_amp_reward_protocol,
                                checkpoint_protocol, config_protocol,
                                validate_training_protocol_resume)


def labelled_payload(*, centered=False):
    cfg = WmacConfig.for_profile("supermimic_official")
    if centered:
        cfg.amp_reward_mapping = "instantamp_centered_tanh_v1"
    return {
        "format": "amp-wmac-v1", "implementation_version": 11 if centered else 7,
        "config": asdict(cfg), "training_protocol": config_protocol(cfg),
        "amp_reward_protocol": amp_reward_protocol(cfg),
        "actor_type": cfg.actor_type, "actor_architecture": actor_architecture(cfg),
        "recovery_counter_semantics": cfg.recovery_counter_semantics,
        "recovery_height_reward": cfg.recovery_height_reward,
    }


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/train_wmac.py"
SPEC = importlib.util.spec_from_file_location("centered_wmac_cli", SCRIPT)
CLI = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CLI)


class CenteredAmpProtocolTest(unittest.TestCase):
    def test_legacy_protocol_and_unlabelled_checkpoint_remain_clipped(self):
        cfg = WmacConfig.for_profile("supermimic_official")
        expected = dict(reference_alignment="official_details_v1",
                        optimizer_algorithm="muon_adam", mirror_walk_motions=True)
        self.assertEqual(config_protocol(cfg), expected)
        payload = labelled_payload()
        payload.pop("amp_reward_protocol")
        for key in ("amp_reward_mapping", "amp_score_scale", "amp_score_ema_decay"):
            payload["config"].pop(key)
        self.assertEqual(checkpoint_protocol(payload), expected)
        self.assertEqual(checkpoint_amp_reward_protocol(payload),
                         {"amp_reward_mapping": "clipped_lsgan_v1"})
        self.assertEqual(resolve_checkpoint_config(payload).amp_reward_mapping, "clipped_lsgan_v1")

    def test_centered_requires_exact_explicit_metadata_and_new_version(self):
        source = labelled_payload(centered=True)
        self.assertEqual(checkpoint_protocol(source), source["training_protocol"])
        for key in ("amp_reward_mapping", "amp_score_scale", "amp_score_ema_decay"):
            broken = copy.deepcopy(source)
            broken["config"].pop(key)
            with self.subTest(key=key), self.assertRaises(ValueError):
                checkpoint_protocol(broken)
        for key in ("amp_reward_protocol", "training_protocol"):
            broken = copy.deepcopy(source)
            broken.pop(key)
            with self.subTest(key=key), self.assertRaises(ValueError):
                checkpoint_protocol(broken)
        broken = copy.deepcopy(source)
        broken["implementation_version"] = 7
        with self.assertRaisesRegex(ValueError, "version >= 11"):
            checkpoint_protocol(broken)

    def test_migration_is_one_way_and_authorization_is_not_saved(self):
        legacy = labelled_payload()
        target = resolve_checkpoint_config(legacy)
        target.amp_reward_mapping = "instantamp_centered_tanh_v1"
        with self.assertRaisesRegex(ValueError, "allow-amp-reward-migration"):
            validate_training_protocol_resume(legacy, target)
        target.allow_amp_reward_migration = True
        self.assertTrue(validate_training_protocol_resume(legacy, target))
        new = labelled_payload(centered=True)
        new["config"]["allow_amp_reward_migration"] = True
        restored = resolve_checkpoint_config(new)
        self.assertFalse(restored.allow_amp_reward_migration)
        self.assertFalse(validate_training_protocol_resume(new, restored))
        restored.amp_reward_mapping = "clipped_lsgan_v1"
        restored.allow_amp_reward_migration = True
        with self.assertRaises(ValueError):
            validate_training_protocol_resume(new, restored)

    def test_migration_flag_cannot_change_optimizer_or_motions(self):
        source = labelled_payload()
        for key, value in (("optimizer_algorithm", "adam"),
                           ("mirror_walk_motions", False),
                           ("reference_alignment", "legacy_v1")):
            target = resolve_checkpoint_config(source)
            target.amp_reward_mapping = "instantamp_centered_tanh_v1"
            target.allow_amp_reward_migration = True
            setattr(target, key, value)
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "Training protocol mismatch"):
                validate_training_protocol_resume(source, target)

    def test_centered_resume_locks_scale_and_decay(self):
        source = labelled_payload(centered=True)
        for key, value in (("amp_score_scale", 2.), ("amp_score_ema_decay", .9)):
            target = resolve_checkpoint_config(source)
            target.allow_amp_reward_migration = True
            setattr(target, key, value)
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "AMP reward protocol mismatch"):
                validate_training_protocol_resume(source, target)

    def test_invalid_scale_decay_and_mapping_are_rejected(self):
        for key, values in (("amp_score_scale", (0., -1., float("inf"), float("nan"), True)),
                            ("amp_score_ema_decay", (-.01, 1., float("nan"), True)),
                            ("amp_reward_mapping", ("unknown",))):
            for value in values:
                cfg = WmacConfig.for_profile("supermimic_official")
                setattr(cfg, key, value)
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    cfg.validate()

    def test_new_training_marker_is_rejected_by_historical_protocol_comparison(self):
        payload = labelled_payload(centered=True)
        historical = {key: payload["config"][key] for key in
                      ("reference_alignment", "optimizer_algorithm", "mirror_walk_motions")}
        self.assertNotEqual(payload["training_protocol"], historical)

    def test_cli_can_migrate_explicitly_and_locks_ordinary_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.pt"
            torch.save(labelled_payload(), path)
            original = CLI.parse_config(["--resume", str(path)])
            self.assertEqual(original.amp_reward_mapping, "clipped_lsgan_v1")
            flags = ["--resume", str(path), "--amp-reward-mapping", "instantamp_centered_tanh_v1"]
            with self.assertRaises(ValueError):
                CLI.parse_config(flags)
            migrated = CLI.parse_config([*flags, "--allow-amp-reward-migration"])
            self.assertTrue(migrated.allow_amp_reward_migration)
            self.assertEqual(migrated.wm_updates, original.wm_updates)
            self.assertEqual(migrated.real_critic_coef, 0.)
            new_path = Path(directory) / "centered.pt"
            torch.save(labelled_payload(centered=True), new_path)
            restored = CLI.parse_config(["--resume", str(new_path)])
            self.assertEqual(restored.amp_reward_mapping, "instantamp_centered_tanh_v1")
            self.assertFalse(restored.allow_amp_reward_migration)
            with self.assertRaises(ValueError):
                CLI.parse_config(["--resume", str(new_path), "--amp-score-scale", "5"])


if __name__ == "__main__":
    unittest.main()
