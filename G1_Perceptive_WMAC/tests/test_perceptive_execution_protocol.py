"""Execution scale/storage is opt-in and never changes historical resumes."""
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
    EXECUTION_FIELDS, FORMAT, LATENT_VISUAL_MODE, LEGACY_EXECUTION_PROFILE,
    SUPERMIMIC_EXECUTION_PROFILE, HIKING_SENSOR_PROTOCOL, G1_RPL_SENSOR_PROTOCOL, G1_HIKING_DR_SENSOR_PROTOCOL,
    G1_HIKING_CALIBRATED_SENSOR_PROTOCOL,
    PerceptiveConfig, config_from_payload,
)


def checkpoint(cfg):
    return {"format": FORMAT, "config": asdict(cfg), "architecture": architecture_metadata(cfg),
            "terrain_fingerprint": "fixed-terrain", "motion_manifest_sha256": "fixed-motions"}


class PerceptiveExecutionProtocolTest(unittest.TestCase):
    def test_legacy_defaults_and_unmarked_records_preserve_architecture(self):
        cfg = PerceptiveConfig.defaults()
        self.assertEqual(cfg.execution_profile, LEGACY_EXECUTION_PROFILE)
        self.assertEqual((cfg.num_envs, cfg.wm_batch_size, cfg.actor_batch_size), (128, 64, 64))
        self.assertEqual((cfg.replay_storage, cfg.validation_horizon, cfg.cuda_graphs), ("legacy", 0, False))
        for source in (cfg, replace(cfg, visual_mode=LATENT_VISUAL_MODE)):
            payload = checkpoint(source)
            for name in EXECUTION_FIELDS:
                payload["config"].pop(name)
            restored = config_from_payload(payload)
            self.assertEqual(restored.execution_profile, LEGACY_EXECUTION_PROFILE)
            self.assertEqual(architecture_metadata(restored), payload["architecture"])
            self.assertNotIn("execution_protocol", payload["architecture"])
            self.assertEqual(validate_checkpoint(payload, restored), {})

    def test_supermimic_profile_sets_training_scale_before_explicit_cli_overrides(self):
        cfg = parse_config(["--execution-profile", SUPERMIMIC_EXECUTION_PROFILE])
        self.assertEqual(asdict(parse_config([])), asdict(cfg))
        legacy_cli = parse_config(["--execution-profile", LEGACY_EXECUTION_PROFILE])
        self.assertEqual((legacy_cli.visual_mode, legacy_cli.execution_optimizer, legacy_cli.num_envs),
                         ("pixel_v2", "adam", 128))
        self.assertEqual(cfg.visual_mode, LATENT_VISUAL_MODE)
        self.assertEqual((cfg.num_envs, cfg.wm_batch_size, cfg.actor_batch_size), (4096, 2048, 4096))
        self.assertEqual((cfg.wm_updates, cfg.wm_window, cfg.imagination_horizon, cfg.validation_horizon), (10, 7, 32, 32))
        self.assertEqual((cfg.actor_microbatch_size, cfg.replay_steps, cfg.replay_device), (0, 1000, "cuda:0"))
        self.assertEqual((cfg.replay_storage, cfg.compile, cfg.precision), ("frame_uint8", True, "tf32"))
        # Version 4 is Hiking's own camera and reference motion; version 3 keeps
        # the measured intrinsics and the supplemented reference pool.
        self.assertEqual((cfg.sensor_protocol, cfg.walking_expert_source, cfg.discriminator_head_weight_decay,
                          cfg.robot_variant), ("hiking_depth_only_v1", "none", 1., "unshod"))
        previous = parse_config(["--hiking-training-version", "3"])
        self.assertEqual((previous.sensor_protocol, previous.walking_expert_source,
                          previous.discriminator_head_weight_decay),
                         (G1_HIKING_CALIBRATED_SENSOR_PROTOCOL, "pre_hiking_local_walk_turn_v2", .04))
        # Version 4 is the Hiking-aligned task; version 5 adds the real-robot defaults on top.
        aligned = parse_config(["--hiking-training-version", "4"])
        self.assertEqual({name: value for name, value in asdict(previous).items() if value != asdict(aligned)[name]}.keys(),
                         {"hiking_training_version", "sensor_protocol", "walking_expert_source",
                          "discriminator_head_weight_decay"})
        self.assertEqual({name: (asdict(aligned)[name], value) for name, value in asdict(cfg).items()
                          if value != asdict(aligned)[name]},
                         {"hiking_training_version": (4, 5), "collision_horizontal_scale": (.05, .025),
                          "depth_geometry": ("collision", "source_mesh"),
                          "robot_dynamics_randomization": ("none", "unitree_g1_v1")})
        self.assertIsNone(cfg.action_limit)
        self.assertEqual((cfg.hiking_training_version, cfg.iterations, cfg.validation_env_fraction), (5, 1_000_000, 0.))
        historical = parse_config(["--hiking-training-version", "2"])
        self.assertEqual((historical.sensor_protocol, historical.action_limit), (G1_HIKING_DR_SENSOR_PROTOCOL, 20.))
        self.assertGreater(historical.validation_env_fraction, 0.)
        self.assertEqual((cfg.imagination_horizon_start, cfg.cuda_graphs, cfg.graph_warmup), (32, True, 3))
        self.assertEqual(cfg.training_schedule, "supermimic_v1")
        self.assertEqual((cfg.model_warmup_iterations, cfg.critic_warmup_iterations, cfg.warmup_steps), (0, 0, 8))
        self.assertEqual(cfg.exploration_noise, .1)
        self.assertEqual((cfg.execution_optimizer, cfg.actor_lr, cfg.lr_final_fraction), ("muon_adam", 2e-4, 1.))
        self.assertEqual((cfg.critic_lr, cfg.actor_betas), (2e-4, (.9, .95)))
        self.assertEqual(cfg.critic_hidden, (256, 128))
        smaller = parse_config(["--execution-profile", SUPERMIMIC_EXECUTION_PROFILE,
            "--num-envs", "4", "--wm-batch-size", "8", "--actor-batch-size", "16",
            "--imagination-horizon", "8", "--validation-horizon", "8",
            "--sensor-protocol", HIKING_SENSOR_PROTOCOL,
            "--replay-steps", "64",
            "--no-compile", "--no-cuda-graphs"])
        self.assertEqual((smaller.num_envs, smaller.wm_batch_size, smaller.actor_batch_size), (4, 8, 16))
        self.assertFalse(smaller.compile)
        self.assertEqual((smaller.validation_horizon, smaller.imagination_horizon), (8, 8))
        self.assertAlmostEqual(smaller.utd, .1)
        scaled = parse_config(["--num-envs", "2048"])
        self.assertAlmostEqual(scaled.utd, .2)
        explicit = parse_config(["--num-envs", "2048", "--utd=0.4"])
        self.assertAlmostEqual(explicit.utd, .4)
        protocol = architecture_metadata(explicit)["execution_protocol"]["update_schedule"]
        self.assertAlmostEqual(protocol["realized_replay_ratio"], 32.)

    def test_recorded_execution_marker_requires_complete_settings(self):
        for cfg in (PerceptiveConfig.defaults(), PerceptiveConfig.defaults(SUPERMIMIC_EXECUTION_PROFILE)):
            for name in EXECUTION_FIELDS:
                if name in ("execution_optimizer", "training_schedule"):
                    continue  # Later fields retain historical algorithm/schedule defaults.
                payload = checkpoint(cfg)
                payload["config"].pop(name)
                with self.subTest(profile=cfg.execution_profile, missing=name), self.assertRaisesRegex(ValueError, "missing"):
                    config_from_payload(payload)

    def test_validation_horizon_is_independent_of_wm_fit_and_rejects_uncovered_windows(self):
        cfg = PerceptiveConfig.defaults(SUPERMIMIC_EXECUTION_PROFILE)
        cfg.validate()
        replace(cfg, imagination_horizon=16, imagination_horizon_start=16).validate()
        for values in ({"validation_horizon": 0}, {"validation_horizon": 31},
                       {"imagination_horizon": 6}, {"imagination_horizon_start": 33},
                       {"replay_steps": 16}, {"warmup_steps": 6},
                       {"validation_horizon": True}, {"validation_horizon": 32.}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                replace(cfg, **values).validate()
        validated = replace(PerceptiveConfig.defaults(SUPERMIMIC_EXECUTION_PROFILE, hiking_training_version=1),
                            training_schedule="validated_v2", imagination_horizon_start=1,
                            model_warmup_iterations=500, critic_warmup_iterations=20, warmup_steps=64)
        validated.validate()
        with self.assertRaisesRegex(ValueError, "warmup_steps"):
            replace(validated, warmup_steps=16).validate()
        legacy = PerceptiveConfig.defaults()
        for values in ({"compile": True}, {"replay_device": "cuda:0"},
                       {"validation_horizon": 32}, {"replay_storage": "frame_uint8"}):
            with self.subTest(legacy=values), self.assertRaises(ValueError):
                replace(legacy, **values).validate()

    def test_graphs_require_compiled_cuda_latent_execution_and_storage_is_explicit(self):
        cfg = replace(PerceptiveConfig.defaults(SUPERMIMIC_EXECUTION_PROFILE),
                      sensor_protocol=HIKING_SENSOR_PROTOCOL, replay_storage="frame_uint8")
        replace(cfg, cuda_graphs=True).validate()
        for values in ({"cuda_graphs": "yes"}, {"cuda_graphs": True, "compile": False},
                       {"cuda_graphs": True, "device": "cpu"}, {"graph_warmup": 0},
                       {"graph_warmup": True}, {"replay_storage": "stack_uint8"},
                       {"replay_device": "cpu"}, {"visual_mode": "pixel_v2"},
                       {"execution_profile": "official"}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                replace(cfg, **values).validate()
        metadata = architecture_metadata(cfg)["execution_protocol"]
        self.assertEqual(metadata["world_model_optimizer"], "ReferenceMuonAdam")
        self.assertEqual(metadata["validation_horizon"], 32)
        self.assertEqual(metadata["replay"]["storage_protocol"], "hiking_lossless_frame_uint8_v3")
        self.assertEqual(metadata["replay"]["cache_frames"], 1037)
        self.assertIn("not_complete_stacks", metadata["replay"]["representation"])
        self.assertFalse(metadata["replay"]["latent_cache"])
        fresh = architecture_metadata(PerceptiveConfig.defaults(SUPERMIMIC_EXECUTION_PROFILE))["execution_protocol"]
        self.assertEqual(fresh["replay"]["storage_protocol"], "hiking_lossless_frame_uint8_v3")
        self.assertEqual(fresh["replay"]["cache_frames"],1037)
        self.assertEqual(fresh["replay"]["channels_per_frame"], 1)
        self.assertFalse(fresh["replay"]["latent_cache"])
        self.assertNotIn("valid",fresh["replay"])
        rpl=replace(PerceptiveConfig.defaults(SUPERMIMIC_EXECUTION_PROFILE, hiking_training_version=1),
                    sensor_protocol=G1_RPL_SENSOR_PROTOCOL,replay_storage="frame_fp16")
        historical=architecture_metadata(rpl)["execution_protocol"]["replay"]
        self.assertEqual(historical["storage_protocol"],"centered_frame_fp16_v1")
        self.assertEqual(historical["frames_per_transition"],2)
        self.assertEqual(historical["terminal_reset_semantics"],
                         "distinct_pre_reset_successor_and_post_reset_current_frames")

    def test_missing_later_optimizer_field_preserves_old_optimized_adam_and_metadata(self):
        cfg = replace(PerceptiveConfig.defaults(SUPERMIMIC_EXECUTION_PROFILE, hiking_training_version=1),
                      sensor_protocol=HIKING_SENSOR_PROTOCOL, replay_storage="frame_uint8",
                      execution_optimizer="adam", actor_lr=.001, critic_lr=.001,
                      actor_betas=(.9, .999), critic_hidden=(512, 256, 128),
                      wm_window=8, lr_final_fraction=.03, training_schedule="validated_v2",
                      imagination_horizon_start=1, model_warmup_iterations=500,
                      critic_warmup_iterations=20, warmup_steps=64, cuda_graphs=False)
        payload = checkpoint(cfg)
        payload["config"].pop("execution_optimizer")
        payload["config"].pop("training_schedule")
        old_metadata = copy.deepcopy(payload["architecture"])
        restored = config_from_payload(payload)
        self.assertEqual(restored.execution_optimizer, "adam")
        self.assertEqual((restored.actor_lr, restored.lr_final_fraction), (.001, .03))
        self.assertEqual((restored.wm_window, restored.critic_hidden), (8, (512, 256, 128)))
        self.assertEqual(architecture_metadata(restored), old_metadata)
        protocol = old_metadata["execution_protocol"]
        self.assertEqual(protocol["world_model_optimizer"], "Adam")
        self.assertNotIn("optimizer_protocol", protocol)
        self.assertNotIn("cuda_graph_scope", protocol)
        self.assertEqual(validate_checkpoint(payload, restored), {})

    def test_optimizer_selection_is_explicit_and_legacy_adam_never_becomes_muon_on_resume(self):
        cfg = PerceptiveConfig.defaults(SUPERMIMIC_EXECUTION_PROFILE)
        metadata = architecture_metadata(cfg)["execution_protocol"]
        self.assertEqual(metadata["optimizer_protocol"]["muon"]["weight_decay"], .1)
        self.assertEqual(metadata["optimizer_protocol"]["adam"]["weight_decay"], 0.)
        self.assertEqual(metadata["learning_rate_schedule"], "constant")
        self.assertEqual(metadata["learning_rates"]["actor"], cfg.actor_lr)
        self.assertEqual(metadata["optimizer_protocol"]["adam"]["actor_betas"], list(cfg.actor_betas))
        self.assertEqual(metadata["cuda_graph_components"]["actor"], "full_update")
        self.assertEqual(metadata["critic_protocol"]["activation"], "GELU")
        self.assertTrue(metadata["critic_protocol"]["official_value_ceiling_clamp"])
        self.assertEqual(metadata["optimizer_protocol"]["nonfinite_guard"], "disabled_official_unconditional_step")
        self.assertFalse(metadata["update_schedule"]["actor_quality_gate"])
        self.assertFalse(metadata["update_schedule"]["actor_drift_rollback"])
        self.assertAlmostEqual(metadata["update_schedule"]["realized_replay_ratio"], 16.)
        micro = architecture_metadata(replace(cfg, actor_microbatch_size=16))["execution_protocol"]
        self.assertEqual(micro["cuda_graph_components"]["actor"], "microbatch_uncaptured_update")
        self.assertEqual(metadata["world_model_window"],
                         {"reference_default_state_frames": 8, "predicted_transitions": 7})
        self.assertIn("optimizer_step", metadata["cuda_graph_scope"])
        with self.assertRaises(ValueError):
            replace(cfg, execution_optimizer="muon").validate()
        with self.assertRaisesRegex(ValueError, "Legacy execution retains Adam"):
            replace(PerceptiveConfig.defaults(), execution_optimizer="muon_adam").validate()
        old = replace(PerceptiveConfig.defaults(SUPERMIMIC_EXECUTION_PROFILE, hiking_training_version=1),
                      execution_optimizer="adam", training_schedule="validated_v2",
                      imagination_horizon_start=1, warmup_steps=64)
        payload = checkpoint(old)
        payload["config"].pop("execution_optimizer")
        with self.assertRaisesRegex(ValueError, "architecture"):
            validate_checkpoint(payload, cfg)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"old_optimized_adam.pt"
            torch.save(payload, path)
            restored = parse_config(["--resume", str(path), "--execution-profile", "supermimic"])
            self.assertEqual(restored.execution_optimizer, "adam")
            with self.assertRaisesRegex(ValueError, "cannot change execution_optimizer"):
                parse_config(["--resume", str(path), "--execution-optimizer", "muon_adam"])

    def test_resume_is_strict_but_same_protocol_device_relocation_is_allowed(self):
        cfg = PerceptiveConfig.defaults(SUPERMIMIC_EXECUTION_PROFILE)
        payload = checkpoint(cfg)
        before = copy.deepcopy(payload)
        relocated = replace(cfg, device="cuda:1", replay_device="cuda:1", run_name="relocated")
        self.assertEqual(architecture_metadata(cfg), architecture_metadata(relocated))
        self.assertEqual(validate_checkpoint(payload, relocated), {})
        for values in ({"num_envs": 128}, {"wm_batch_size": 64}, {"actor_microbatch_size": 16},
                       {"cuda_graphs": False}, {"compile": False}, {"validation_horizon": 64},
                       {"replay_storage": "legacy"}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                validate_checkpoint(payload, replace(cfg, **values))
        self.assertEqual(payload, before)

    def test_cli_preserves_saved_profile_without_reapplying_large_defaults(self):
        cfg = replace(PerceptiveConfig.defaults(SUPERMIMIC_EXECUTION_PROFILE), num_envs=4,
                      wm_batch_size=8, actor_batch_size=16)
        legacy = replace(PerceptiveConfig.defaults(), visual_mode=LATENT_VISUAL_MODE)
        with tempfile.TemporaryDirectory() as directory:
            path, old_path = Path(directory)/"optimized.pt", Path(directory)/"legacy.pt"
            torch.save(checkpoint(cfg), path)
            torch.save(checkpoint(legacy), old_path)
            for args in (["--resume", str(path)],
                         ["--resume", str(path), "--execution-profile", SUPERMIMIC_EXECUTION_PROFILE]):
                restored = parse_config(args)
                self.assertEqual((restored.num_envs, restored.wm_batch_size, restored.actor_batch_size), (4, 8, 16))
                self.assertEqual(restored.utd, cfg.utd)  # Saved budgets are never freshly derived.
                self.assertEqual(validate_checkpoint(checkpoint(cfg), restored), {})
            for args in (["--resume", str(old_path), "--execution-profile", SUPERMIMIC_EXECUTION_PROFILE],
                         ["--resume", str(path), "--execution-profile", LEGACY_EXECUTION_PROFILE]):
                with self.subTest(args=args), self.assertRaisesRegex(ValueError, "cannot change execution_profile"):
                    parse_config(args)

    def test_missing_later_training_schedule_preserves_validated_muon_history(self):
        cfg = replace(PerceptiveConfig.defaults(SUPERMIMIC_EXECUTION_PROFILE, hiking_training_version=1),
                      sensor_protocol=HIKING_SENSOR_PROTOCOL, replay_storage="frame_uint8",
                      training_schedule="validated_v2", imagination_horizon_start=1,
                      model_warmup_iterations=500, critic_warmup_iterations=20,
                      warmup_steps=64, cuda_graphs=False)
        payload = checkpoint(cfg)
        payload["config"].pop("training_schedule")
        restored = config_from_payload(payload)
        self.assertEqual(restored.training_schedule, "validated_v2")
        self.assertEqual((restored.imagination_horizon_start, restored.model_warmup_iterations), (1, 500))
        self.assertEqual(architecture_metadata(restored), payload["architecture"])
        self.assertEqual(validate_checkpoint(payload, restored), {})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"old_validated_muon.pt"
            torch.save(payload, path)
            self.assertEqual(parse_config(["--resume", str(path)]).training_schedule, "validated_v2")
            with self.assertRaisesRegex(ValueError, "cannot change training_schedule"):
                parse_config(["--resume", str(path), "--training-schedule", "supermimic_v1"])


if __name__ == "__main__":
    unittest.main()
