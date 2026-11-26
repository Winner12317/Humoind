"""Instinct joint-stack latent semantics never change implicitly on resume."""
from __future__ import annotations

import copy
from dataclasses import asdict, fields, replace
from pathlib import Path
import tempfile
import unittest

import torch

from scripts.evaluate_wmac_perceptive import load_actor
from scripts.train_wmac_perceptive import parse_config
from src.wmac_perceptive.checkpoints import architecture_metadata, build_perceptive_actor, validate_checkpoint
from src.wmac_perceptive.config import (
    FORMAT, LATENT_VISUAL_MODE, LEGACY_DYNAMICS, PIXEL_VISUAL_MODE,
    VISUAL_FIELDS, PerceptiveConfig, config_from_payload, perception_spec_from_config,
)
from src.wmac_perceptive.latent_models import LatentPerceptiveActor
from src.wmac_perceptive.models import PerceptiveActor
from src.wmac_perceptive.perception import PerceptionSpec


def latent_config(**values):
    return replace(PerceptiveConfig.defaults(), visual_mode=LATENT_VISUAL_MODE, **values)


def checkpoint(cfg):
    return {"format": FORMAT, "config": asdict(cfg), "architecture": architecture_metadata(cfg),
            "visual_spec": asdict(perception_spec_from_config(cfg)), "amp_grounding": cfg.ground_height_semantics,
            "terrain_fingerprint": "fixed-terrain", "motion_manifest_sha256": "fixed-motions"}


class PerceptiveLatentProtocolTest(unittest.TestCase):
    def test_pixel_default_and_historical_config_keep_identical_architecture(self):
        cfg = PerceptiveConfig.defaults()
        self.assertEqual(cfg.visual_mode, PIXEL_VISUAL_MODE)
        self.assertEqual(cfg.latent_dim, 128)
        self.assertIsNone(cfg.max_validation_latent_error)
        self.assertNotIn("visual_codec", {field.name for field in fields(cfg)})
        payload = checkpoint(cfg)
        for name in VISUAL_FIELDS:
            payload["config"].pop(name)
        restored = config_from_payload(payload)
        self.assertEqual(restored.visual_mode, PIXEL_VISUAL_MODE)
        self.assertEqual(architecture_metadata(restored), payload["architecture"])
        self.assertNotIn("visual_mode", payload["architecture"])
        self.assertNotIn("visual_protocol", payload["architecture"])
        self.assertNotIn("latent_decoder", payload["architecture"])
        self.assertEqual(payload["architecture"]["actor"]["kind"], "depth_cnn_hiking_moe")
        self.assertEqual(validate_checkpoint(payload, restored), {})

    def test_latent_metadata_defines_online_joint_stack_encoder_and_shared_head(self):
        cfg, spec = latent_config(), PerceptionSpec()
        metadata = architecture_metadata(cfg)
        protocol = metadata["visual_protocol"]
        self.assertEqual(LATENT_VISUAL_MODE, "instinct_latent_v1")
        self.assertEqual(protocol["encoder_training"], "online_stack_reconstruction_only")
        self.assertIn("stop_gradient", protocol["actor_latent_input"])
        self.assertIn("detached", protocol["latent_targets"])
        self.assertFalse(protocol["pixel_decoder_in_rollout"])
        self.assertEqual(protocol["selected_indices"], list(spec.selected_indices))
        self.assertEqual((protocol["selected_frames"], protocol["delay"], protocol["stride"]), (8, 1, 5))
        self.assertIn("disabled_by_default", protocol["absolute_validation_veto"])
        self.assertIn("target_variance", protocol["validation_error"])
        self.assertEqual(metadata["actor"]["depth_encoder"]["input_channels"], 16)
        self.assertEqual(metadata["actor"]["visual_embedding_dim"], 128)
        self.assertEqual(metadata["actor"]["compact_input_dim"], 512)
        self.assertEqual(metadata["world_model"]["input_normalizer_dim"], 213)
        self.assertEqual(metadata["world_model"]["input_dim"], 341)
        self.assertEqual(metadata["world_model"]["output_dim"], 165)
        self.assertEqual(metadata["world_model"]["shared_mlp_outputs"],
                         {"physical_velocity_increment": 35, "contact": 2, "latent_delta": 128})
        self.assertEqual(metadata["world_model"]["latent_statistics"]["buffers"], ["z_mean", "z_std", "dz_std"])
        self.assertEqual(metadata["latent_decoder"]["objective"], "processed_depth_valid_stack_l1")
        self.assertEqual(metadata["optimizer_partition"]["opt_actor"], ["actor.policy"])
        self.assertEqual(metadata["optimizer_partition"]["opt_model"], ["world_model", "actor.vision", "latent_decoder"])
        self.assertEqual(validate_checkpoint(checkpoint(cfg), cfg), {})

    def test_recorded_visual_marker_requires_all_visual_training_fields(self):
        for cfg in (PerceptiveConfig.defaults(), latent_config()):
            for name in VISUAL_FIELDS:
                payload = checkpoint(cfg)
                payload["config"].pop(name)
                with self.subTest(mode=cfg.visual_mode, missing=name), self.assertRaisesRegex(ValueError, "missing"):
                    config_from_payload(payload)

    def test_latent_config_requires_current_task_dynamics_and_valid_online_training(self):
        cfg = latent_config()
        cfg.validate()
        replace(cfg, max_validation_latent_error=.25, latent_stat_rate=1.).validate()
        invalid = ({"visual_mode": "frozen_latent_v1"}, {"latent_dim": 0}, {"latent_dim": 64},
                   {"latent_dim": True}, {"latent_dim": 128.},
                   {"latent_loss_coef": 0.}, {"latent_loss_coef": float("nan")},
                   {"latent_recon_coef": 0.}, {"latent_recon_images": 0},
                   {"latent_recon_images": True}, {"latent_recon_images": 512.},
                   {"latent_stat_rate": 0.}, {"latent_stat_rate": 1.01},
                   {"latent_stat_rate": True}, {"latent_stat_rate": float("nan")},
                   {"max_validation_latent_error": 0.}, {"max_validation_latent_error": True},
                   {"max_validation_latent_error": float("inf")},
                   {"latent_rollout_error_ratio": .9}, {"latent_rollout_error_ratio": float("inf")},
                   {"dynamics_parameterization": LEGACY_DYNAMICS},
                   {"init_policy": "old-pixel.pt"}, {"migrate_world_model": True},
                   {"migrate_perception_training": True}, {"perception_training_version": 1},
                   {"motion_scope": "joint_recovery_v1", "recovery_flat_fraction": .25,
                    "validation_min_recovery_samples": 8})
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValueError):
                replace(cfg, **values).validate()

    def test_resume_cannot_change_visual_semantics_or_online_training_protocol(self):
        pixel, latent = PerceptiveConfig.defaults(), latent_config()
        for source, target in ((pixel, latent), (latent, pixel)):
            with self.subTest(source=source.visual_mode), self.assertRaisesRegex(ValueError, "architecture"):
                validate_checkpoint(checkpoint(source), target)
        payload = checkpoint(latent)
        before = copy.deepcopy(payload)
        for values in ({"latent_loss_coef": .5}, {"latent_recon_coef": .5},
                       {"latent_recon_images": 256}, {"latent_stat_rate": .02},
                       {"max_validation_latent_error": .5}, {"latent_rollout_error_ratio": 2.}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                validate_checkpoint(payload, replace(latent, **values))
        self.assertEqual(validate_checkpoint(payload, replace(latent, run_name="new-output-directory")), {})
        self.assertEqual(payload, before)

    def test_cli_starts_online_latent_without_artifact_and_refuses_pixel_conversion(self):
        fresh = parse_config(["--visual-mode", LATENT_VISUAL_MODE])
        self.assertEqual(fresh.visual_mode, LATENT_VISUAL_MODE)
        self.assertEqual(fresh.latent_dim, 128)
        self.assertEqual(fresh.latent_recon_images, 512)
        self.assertEqual(fresh.latent_stat_rate, .01)
        self.assertIsNone(fresh.max_validation_latent_error)
        explicit = parse_config(["--visual-mode", LATENT_VISUAL_MODE, "--max-validation-latent-error", "0.25"])
        self.assertEqual(explicit.max_validation_latent_error, .25)
        with tempfile.TemporaryDirectory() as directory:
            latent_path, pixel_path = Path(directory)/"latent.pt", Path(directory)/"pixel.pt"
            torch.save(checkpoint(fresh), latent_path)
            torch.save(checkpoint(PerceptiveConfig.defaults()), pixel_path)
            restored = parse_config(["--resume", str(latent_path)])
            self.assertEqual(restored.visual_mode, LATENT_VISUAL_MODE)
            self.assertEqual(validate_checkpoint(checkpoint(fresh), restored), {})
            for path, mode in ((pixel_path, LATENT_VISUAL_MODE), (latent_path, PIXEL_VISUAL_MODE)):
                with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "cannot change visual_mode"):
                    parse_config(["--resume", str(path), "--visual-mode", mode])

    def test_actor_only_load_needs_neither_world_model_nor_training_decoder(self):
        torch.manual_seed(85)
        cfg, spec = latent_config(), PerceptionSpec()
        actor = LatentPerceptiveActor(spec=spec, latent_dim=cfg.latent_dim).eval()
        payload = {**checkpoint(cfg), "actor": actor.state_dict(),
                   "world_model": {"not_constructed_by_evaluation": True},
                   "latent_decoder": {"not_constructed_by_evaluation": True}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"latent.pt"
            torch.save(payload, path)
            restored, restored_cfg, _ = load_actor(path)
            self.assertIsInstance(restored, LatentPerceptiveActor)
            self.assertEqual(restored_cfg.visual_mode, LATENT_VISUAL_MODE)
            for name, value in actor.state_dict().items():
                torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)
            from scripts.play_wmac_perceptive import load_actor as playback_load_actor
            self.assertIs(playback_load_actor, load_actor)
            pixel = build_perceptive_actor(checkpoint(PerceptiveConfig.defaults()))
            self.assertIsInstance(pixel, PerceptiveActor)
            self.assertNotIsInstance(pixel, LatentPerceptiveActor)


if __name__ == "__main__":
    unittest.main()
