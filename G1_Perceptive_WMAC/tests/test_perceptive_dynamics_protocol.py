"""Physical parameterization changes require an explicit, bounded migration."""
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
    DYNAMICS_FIELDS, FORMAT, GENCOORD_DYNAMICS, LEGACY_DYNAMICS,
    PerceptiveConfig, config_from_payload,
)


def checkpoint(cfg):
    return {"format": FORMAT, "config": asdict(cfg),
            "architecture": architecture_metadata(cfg),
            "terrain_fingerprint": "fixed-terrain", "motion_manifest_sha256": "fixed-motions"}


def legacy_checkpoint():
    cfg = replace(PerceptiveConfig.defaults(), dynamics_parameterization=LEGACY_DYNAMICS)
    payload = checkpoint(cfg)
    for name in DYNAMICS_FIELDS | {"migrate_world_model"}:
        payload["config"].pop(name)
    return payload


class PerceptiveDynamicsProtocolTest(unittest.TestCase):
    def test_new_defaults_and_architecture_name_the_actual_physical_contract(self):
        cfg = PerceptiveConfig.defaults()
        cfg.validate()
        self.assertEqual(cfg.dynamics_parameterization, GENCOORD_DYNAMICS)
        physical = architecture_metadata(cfg)["world_model"]
        self.assertEqual(physical["input_physical_dim"], 68)
        self.assertEqual(physical["output_physical_dim"], 35)
        self.assertEqual(physical["integration"], {"dt": .02, "decimation": 4, "substep_coeff": .625})
        self.assertEqual(physical["activation"], "GELU")
        self.assertEqual(physical["output_normalization"], "none")
        self.assertEqual(physical["supervision"], "integrated_pose_joint_bryson_v1")
        self.assertEqual(validate_checkpoint(checkpoint(cfg), cfg), {})

    def test_unmarked_historical_payload_keeps_legacy_model_and_exact_old_metadata(self):
        payload = legacy_checkpoint()
        old = config_from_payload(payload)
        self.assertEqual(old.dynamics_parameterization, LEGACY_DYNAMICS)
        self.assertEqual(old.dynamics_dt, .02)
        self.assertEqual(old.dynamics_decimation, 4)
        self.assertFalse(old.migrate_world_model)
        self.assertEqual(architecture_metadata(old), payload["architecture"])
        self.assertEqual(payload["architecture"]["world_model"], {
            "kind": "terrain_and_current_depth_conditioned_physical_residual",
            "hidden": list(old.model_hidden), "depth_embedding_dim": 64,
            "next_view": "predicted normalized depth and validity, advanced every imagined step",
            "contact_outputs": 2,
        })
        self.assertEqual(validate_checkpoint(payload, old), {})

    def test_parameterization_and_integration_never_change_on_ordinary_resume(self):
        old = legacy_checkpoint()
        with self.assertRaisesRegex(ValueError, "architecture"):
            validate_checkpoint(old, PerceptiveConfig.defaults())
        new = PerceptiveConfig.defaults()
        payload = checkpoint(new)
        for values in ({"dynamics_parameterization": LEGACY_DYNAMICS},
                       {"dynamics_dt": .04}, {"dynamics_decimation": 8}):
            with self.subTest(values=values), self.assertRaisesRegex(ValueError, "architecture"):
                validate_checkpoint(payload, replace(new, **values))

    def test_explicit_migration_returns_exact_changes_without_mutation(self):
        payload = legacy_checkpoint()
        before = copy.deepcopy(payload)
        cfg = replace(config_from_payload(payload), dynamics_parameterization=GENCOORD_DYNAMICS,
                      migrate_world_model=True)
        migration = validate_checkpoint(payload, cfg)
        self.assertEqual(migration, {"world_model_migration": {
            "from_parameterization": LEGACY_DYNAMICS,
            "to_parameterization": GENCOORD_DYNAMICS,
            "changes": {"dynamics_parameterization": {"before": LEGACY_DYNAMICS,
                                                       "after": GENCOORD_DYNAMICS}},
        }})
        self.assertEqual(payload, before)

    def test_migration_does_not_allow_learning_rate_data_or_actor_changes(self):
        payload = legacy_checkpoint()
        cfg = replace(config_from_payload(payload), dynamics_parameterization=GENCOORD_DYNAMICS,
                      migrate_world_model=True)
        for values in ({"actor_lr": cfg.actor_lr*.5}, {"seed": cfg.seed+1},
                       {"depth_loss_coef": cfg.depth_loss_coef*2}, {"imagination_horizon": 4}):
            with self.subTest(values=values), self.assertRaisesRegex(ValueError, "unsupported strict"):
                validate_checkpoint(payload, replace(cfg, **values))
        bad = copy.deepcopy(payload)
        bad["architecture"]["actor"]["visual_embedding_dim"] = 64
        with self.assertRaisesRegex(ValueError, "architecture"):
            validate_checkpoint(bad, cfg)
        with self.assertRaisesRegex(ValueError, "differs"):
            validate_checkpoint(payload, cfg, terrain_fingerprint="another-terrain")
        with self.assertRaisesRegex(ValueError, "differs"):
            validate_checkpoint(payload, cfg, motion_fingerprint="another-motion-pool")

    def test_unsupported_or_incomplete_protocol_values_are_rejected(self):
        cfg = PerceptiveConfig.defaults()
        for values in ({"dynamics_parameterization": "automatic"},
                       {"dynamics_dt": 0.}, {"dynamics_dt": float("nan")},
                       {"dynamics_dt": float("inf")}, {"dynamics_dt": True},
                       {"dynamics_dt": "0.02"}, {"dynamics_decimation": 0},
                       {"dynamics_decimation": 4.}, {"dynamics_decimation": True},
                       {"migrate_world_model": "yes"},
                       {"migrate_world_model": True, "dynamics_parameterization": LEGACY_DYNAMICS},
                       {"migrate_world_model": True, "migrate_perception_training": True}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                replace(cfg, **values).validate()
        for name in DYNAMICS_FIELDS:
            bad = checkpoint(cfg)
            bad["config"].pop(name)
            with self.subTest(missing=name), self.assertRaisesRegex(ValueError, "missing"):
                config_from_payload(bad)
        with self.assertRaisesRegex(ValueError, "only supports"):
            validate_checkpoint(checkpoint(cfg), replace(cfg, migrate_world_model=True))

    def test_cli_preserves_old_resume_and_migrates_only_with_the_explicit_flag(self):
        payload = legacy_checkpoint()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"legacy.pt"
            torch.save(payload, path)
            old = parse_config(["--resume", str(path)])
            self.assertEqual(old.dynamics_parameterization, LEGACY_DYNAMICS)
            new = parse_config(["--resume", str(path), "--migrate-world-model"])
            self.assertEqual(new.dynamics_parameterization, GENCOORD_DYNAMICS)
            self.assertTrue(new.migrate_world_model)
            self.assertEqual(new.actor_lr, old.actor_lr)
            validate_checkpoint(payload, new)
            repaired = checkpoint(new)
            resumed = config_from_payload(repaired)
            self.assertFalse(resumed.migrate_world_model)
            self.assertEqual(resumed.dynamics_parameterization, GENCOORD_DYNAMICS)
            self.assertEqual(validate_checkpoint(repaired, resumed), {})
            new_path = Path(directory)/"new.pt"
            torch.save(repaired, new_path)
            with self.assertRaisesRegex(ValueError, "requires an independent_residual_v1"):
                parse_config(["--resume", str(new_path), "--migrate-world-model"])
        with self.assertRaisesRegex(ValueError, "requires --resume"):
            parse_config(["--migrate-world-model"])
        with self.assertRaisesRegex(ValueError, "separate explicit resumes"):
            parse_config(["--migrate-world-model", "--migrate-perception-training"])


if __name__ == "__main__":
    unittest.main()
