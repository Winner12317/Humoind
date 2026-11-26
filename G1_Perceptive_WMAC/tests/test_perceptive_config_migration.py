"""Versioned training objectives never change implicitly on resume."""
from __future__ import annotations

import copy
from dataclasses import asdict, replace
from pathlib import Path
import tempfile
import unittest

import torch

from scripts.train_wmac_perceptive import parse_config
from src.wmac_perceptive.checkpoints import architecture_metadata, validate_checkpoint
from src.wmac_perceptive.config import (
    DYNAMICS_FIELDS, FORMAT, LEGACY_DYNAMICS, PERCEPTION_TRAINING_FIELDS,
    PerceptiveConfig, config_from_payload,
)


def legacy_payload():
    cfg = replace(PerceptiveConfig.defaults(),motion_scope="joint_recovery_v1",
        dynamics_parameterization=LEGACY_DYNAMICS,
        perception_training_version=1,imagination_horizon_start=8,
        recovery_flat_fraction=.25,recovery_fraction=.5,expert_recovery_fraction=.5,
        validation_min_recovery_samples=8,run_name="hiking_depth_recovery",
        mirrored_motion_root=".perceptive_cache/motions/g1_mirrored")
    saved = asdict(cfg)
    for name in PERCEPTION_TRAINING_FIELDS | DYNAMICS_FIELDS | {"migrate_perception_training","motion_scope","migrate_world_model"}:
        saved.pop(name)
    return {"format": FORMAT, "config": saved,
            "architecture": architecture_metadata(cfg),
            "terrain_fingerprint": "same-terrain", "motion_manifest_sha256": "same-motions"}


class PerceptiveConfigMigrationTest(unittest.TestCase):
    def test_historical_payload_keeps_original_training_protocol_and_architecture(self):
        payload = legacy_payload()
        cfg = config_from_payload(payload)
        self.assertEqual(cfg.perception_training_version, 1)
        self.assertEqual(cfg.motion_scope,"joint_recovery_v1")
        self.assertEqual(cfg.imagination_horizon_start, cfg.imagination_horizon)
        self.assertFalse(cfg.migrate_perception_training)
        self.assertEqual(architecture_metadata(cfg), payload["architecture"])
        self.assertEqual(validate_checkpoint(payload, cfg), {})
        cfg.imagination_horizon = cfg.wm_window+1
        # Historical files could use a horizon beyond the model-fit window.
        cfg.validate()

    def test_new_defaults_use_short_start_and_reject_invalid_curriculum_configuration(self):
        cfg = PerceptiveConfig.defaults()
        self.assertEqual(cfg.perception_training_version, 2)
        self.assertEqual(cfg.imagination_horizon_start, 1)
        cfg.validate()
        for values in ({"perception_training_version": 3}, {"perception_training_version": 2.0},
                       {"imagination_horizon_start": 0}, {"imagination_horizon_start": 9},
                       {"imagination_horizon": 9}, {"horizon_growth_updates": 0},
                       {"horizon_growth_updates": True}, {"actor_stall_warning_iterations": 0},
                       {"depth_rollout_error_ratio": True},
                       {"depth_rollout_error_ratio": .99}, {"depth_rollout_error_ratio": float("nan")},
                       {"depth_mask_loss_coef": 0.}, {"migrate_perception_training": "yes"}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                replace(cfg, **values).validate()

    def test_migration_requires_flag_and_returns_exact_configuration_changes(self):
        payload = legacy_payload()
        repaired = replace(config_from_payload(payload),perception_training_version=2,imagination_horizon_start=1)
        with self.assertRaisesRegex(ValueError, "strict training"):
            validate_checkpoint(payload, repaired)
        repaired.migrate_perception_training = True
        repaired.actor_lr *= .1
        repaired.max_action_shift *= .5
        before = copy.deepcopy(payload)
        result = validate_checkpoint(payload, repaired)
        self.assertEqual(result["from_version"], 1)
        self.assertEqual(result["to_version"], 2)
        self.assertEqual(result["changes"]["perception_training_version"], {"before": 1, "after": 2})
        self.assertEqual(result["changes"]["imagination_horizon_start"], {"before": 8, "after": 1})
        self.assertEqual(result["changes"]["actor_lr"],
                         {"before": payload["config"]["actor_lr"], "after": repaired.actor_lr})
        self.assertEqual(payload, before)

    def test_migration_does_not_authorize_other_training_or_geometry_changes(self):
        payload = legacy_payload()
        repaired = replace(config_from_payload(payload),perception_training_version=2,
                           imagination_horizon_start=1,migrate_perception_training=True)
        for values in ({"seed": repaired.seed+1}, {"utd": .5}, {"actor_batch_size": 65},
                       {"recovery_fraction": .25}, {"depth_noise_std_m": .01}):
            with self.subTest(values=values), self.assertRaisesRegex(ValueError, "unsupported strict"):
                validate_checkpoint(payload, replace(repaired, **values))
        for values in ({"terrain_fingerprint": "different"}, {"motion_fingerprint": "different"}):
            with self.subTest(values=values), self.assertRaisesRegex(ValueError, "differs"):
                validate_checkpoint(payload, repaired, **values)

    def test_incomplete_v2_records_and_repeated_migration_are_rejected(self):
        cfg = PerceptiveConfig.defaults()
        payload = {**legacy_payload(), "config": asdict(cfg), "architecture":architecture_metadata(cfg)}
        for name in PERCEPTION_TRAINING_FIELDS:
            bad = copy.deepcopy(payload); bad["config"].pop(name)
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "missing"):
                validate_checkpoint(bad, cfg)
        with self.assertRaisesRegex(ValueError, "version 1 -> 2"):
            validate_checkpoint(payload, replace(cfg, migrate_perception_training=True))

    def test_cli_preserves_legacy_resume_and_only_explicit_migration_sets_v2_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"legacy.pt"
            torch.save(legacy_payload(), path)
            old = parse_config(["--resume", str(path)])
            self.assertEqual(old.perception_training_version, 1)
            self.assertEqual(old.imagination_horizon_start, 8)
            migrated = parse_config(["--resume", str(path), "--migrate-perception-training",
                                     "--horizon-growth-updates", "20", "--actor-lr", "0.0001"])
            self.assertEqual(migrated.perception_training_version, 2)
            self.assertEqual(migrated.imagination_horizon_start, 1)
            self.assertEqual(migrated.horizon_growth_updates, 20)
            self.assertEqual(migrated.actor_lr, .0001)
            self.assertTrue(migrated.migrate_perception_training)
            validate_checkpoint(legacy_payload(), migrated)
            # Saving the migration command does not request migration again
            # when that repaired checkpoint is later resumed or evaluated.
            repaired_payload = {**legacy_payload(), "config": asdict(migrated)}
            resumed = config_from_payload(repaired_payload)
            self.assertEqual(resumed.perception_training_version, 2)
            self.assertFalse(resumed.migrate_perception_training)
            self.assertEqual(validate_checkpoint(repaired_payload, resumed), {})
        with self.assertRaisesRegex(ValueError, "requires --resume"):
            parse_config(["--migrate-perception-training"])

    def test_migration_caps_a_historical_horizon_beyond_its_validation_window(self):
        payload = legacy_payload()
        payload["config"].update(imagination_horizon=3, wm_window=2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"legacy_h3_k2.pt"
            torch.save(payload, path)
            cfg = parse_config(["--resume", str(path), "--migrate-perception-training"])
            self.assertEqual(cfg.imagination_horizon, 2)
            result = validate_checkpoint(payload, cfg)
            self.assertEqual(result["changes"]["imagination_horizon"], {"before": 3, "after": 2})
            cfg.imagination_horizon = 1
            validate_checkpoint(payload, cfg)


if __name__ == "__main__":
    unittest.main()
