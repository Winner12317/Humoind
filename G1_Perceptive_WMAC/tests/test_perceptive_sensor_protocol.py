"""Depth-only camera changes must not silently reinterpret saved visual policies."""
from __future__ import annotations

import copy
from dataclasses import asdict, replace
from pathlib import Path
import tempfile
import unittest

import torch

from scripts.evaluate_wmac_perceptive import load_actor
from scripts.train_wmac_perceptive import parse_config
from src.wmac_perceptive.checkpoints import (
    architecture_metadata, build_perceptive_actor, validate_checkpoint,
)
from src.wmac_perceptive.config import (
    G1_HIKING_CALIBRATED_SENSOR_PROTOCOL,
    FORMAT, HIKING_SENSOR_PROTOCOL, G1_RPL_SENSOR_PROTOCOL, G1_HIKING_DR_SENSOR_PROTOCOL, LEGACY_SENSOR_PROTOCOL, LATENT_VISUAL_MODE,
    LEGACY_LOCOMOTION_OBJECTIVE,HIKING_FULL_OBJECTIVE,
    PerceptiveConfig, config_from_payload, perception_spec_from_config,
)
from src.wmac_perceptive.instinct_latent import InstinctLatentDecoder
from src.wmac_perceptive.perception import PerceptionSpec
from src.wmac_perceptive.latent_perception import processed_stack_shape


def record(cfg):
    return {
        "format": FORMAT,
        "config": asdict(cfg),
        "architecture": architecture_metadata(cfg),
        "visual_spec": asdict(perception_spec_from_config(cfg)),
        "amp_grounding": cfg.ground_height_semantics,
        "terrain_fingerprint": "camera-test-terrain",
        "motion_manifest_sha256": "camera-test-motions",
    }


def old_optimized_config():
    # A saved, graph-enabled Muon run predating the camera/channel change is
    # still a 16-channel policy; the execution profile alone cannot upgrade it.
    return replace(PerceptiveConfig.defaults("supermimic", hiking_training_version=1),
                   locomotion_objective=LEGACY_LOCOMOTION_OBJECTIVE,
                   standing_expert_source="none",
                   mirrored_motion_root=".perceptive_cache/motions/g1_hiking_walk_v1",
                   collision_horizontal_scale=.1,
                   moe_gate_hidden=(256,),
                   sensor_protocol=LEGACY_SENSOR_PROTOCOL,
                   replay_storage="frame_uint8",
                   depth_noise_std_m=.005, depth_dropout_probability=.01)


class PerceptiveSensorProtocolTest(unittest.TestCase):
    def test_fresh_cli_selects_depth_only_but_legacy_api_keeps_old_sensor(self):
        fresh = parse_config([])
        self.assertEqual(fresh.sensor_protocol, G1_HIKING_CALIBRATED_SENSOR_PROTOCOL)
        self.assertEqual(fresh.visual_mode, LATENT_VISUAL_MODE)
        self.assertFalse(perception_spec_from_config(fresh).include_valid)
        spec = perception_spec_from_config(fresh)
        self.assertEqual((spec.height, spec.width, spec.delay, spec.depth_protocol), (36, 64, 1, "hiking_v1"))
        self.assertEqual(fresh.replay_storage, "frame_uint8")
        self.assertEqual(processed_stack_shape(spec),(8,18,32))
        self.assertEqual(architecture_metadata(fresh)["architecture_version"],9)
        self.assertEqual((fresh.depth_noise_std_m, fresh.depth_dropout_probability), (0., 0.))
        self.assertEqual((fresh.execution_optimizer, fresh.lr_final_fraction), ("muon_adam", 1.))
        for old in (PerceptiveConfig.defaults(), parse_config(["--execution-profile", "legacy"])):
            self.assertEqual(old.sensor_protocol, LEGACY_SENSOR_PROTOCOL)
            self.assertTrue(perception_spec_from_config(old).include_valid)

    def test_checkpoint_without_sensor_marker_retains_optimized_legacy_vision(self):
        original = old_optimized_config()
        payload = record(original)
        payload["config"].pop("sensor_protocol")
        payload["visual_spec"].pop("include_valid")
        before = copy.deepcopy(payload)
        restored = config_from_payload(payload)
        self.assertEqual(restored.sensor_protocol, LEGACY_SENSOR_PROTOCOL)
        self.assertTrue(perception_spec_from_config(restored).include_valid)
        self.assertEqual((restored.execution_optimizer, restored.cuda_graphs), ("muon_adam", True))
        self.assertEqual((restored.depth_noise_std_m, restored.depth_dropout_probability), (.005, .01))
        self.assertEqual(validate_checkpoint(payload, restored), {})
        self.assertEqual(payload, before)

    def test_new_checkpoint_cannot_masquerade_as_unmarked_old_protocol(self):
        payload = record(parse_config([]))
        payload["config"].pop("sensor_protocol")
        with self.assertRaisesRegex(ValueError, "architecture|FP16"):
            restored = config_from_payload(payload)
            validate_checkpoint(payload, restored)

    def test_ordinary_resume_cannot_switch_camera_protocol_in_either_direction(self):
        cases = ((old_optimized_config(), HIKING_SENSOR_PROTOCOL),
                 (parse_config([]), LEGACY_SENSOR_PROTOCOL),
                 (parse_config([]), HIKING_SENSOR_PROTOCOL),
                 (parse_config(["--sensor-protocol", HIKING_SENSOR_PROTOCOL]), G1_HIKING_CALIBRATED_SENSOR_PROTOCOL),
                 (parse_config(["--hiking-training-version", "2"]), HIKING_SENSOR_PROTOCOL),
                 (parse_config(["--hiking-training-version", "2", "--sensor-protocol",G1_RPL_SENSOR_PROTOCOL,
                                "--replay-storage","frame_fp16"]),G1_HIKING_DR_SENSOR_PROTOCOL))
        with tempfile.TemporaryDirectory() as directory:
            for index, (cfg, other) in enumerate(cases):
                path = Path(directory)/f"sensor_{index}.pt"
                payload = record(cfg)
                torch.save(payload, path)
                loaded = parse_config(["--resume", str(path)])
                self.assertEqual(loaded.sensor_protocol, cfg.sensor_protocol)
                with self.assertRaisesRegex(ValueError, "cannot change sensor_protocol"):
                    parse_config(["--resume", str(path), "--sensor-protocol", other])
                with self.assertRaisesRegex(ValueError, "architecture|FP16"):
                    validate_checkpoint(payload, replace(cfg, sensor_protocol=other,
                        replay_storage="frame_fp16" if other==G1_RPL_SENSOR_PROTOCOL else "frame_uint8",
                        depth_noise_std_m=0., depth_dropout_probability=0.))

    def test_actual_encoder_decoder_dimensions_agree_with_saved_protocol(self):
        torch.manual_seed(119)
        hiking = replace(parse_config([]), sensor_protocol=HIKING_SENSOR_PROTOCOL, replay_storage="frame_uint8")
        rpl = parse_config(["--hiking-training-version", "2", "--sensor-protocol",G1_RPL_SENSOR_PROTOCOL,"--replay-storage","frame_fp16"])
        for cfg, channels in ((old_optimized_config(), 16), (hiking, 8), (rpl, 8), (parse_config([]), 8),
                              (parse_config(["--hiking-training-version", "2"]), 8)):
            with self.subTest(protocol=cfg.sensor_protocol):
                payload = record(cfg)
                actor = build_perceptive_actor(payload)
                metadata = payload["architecture"]
                self.assertEqual(actor.vision.conv[0].in_channels, channels)
                gate_linears=[layer for layer in actor.policy.router.modules() if isinstance(layer,torch.nn.Linear)]
                self.assertEqual(len(gate_linears),1 if cfg.locomotion_objective==HIKING_FULL_OBJECTIVE else 2)
                self.assertEqual(metadata["actor"]["depth_encoder"]["input_channels"], channels)
                self.assertEqual(metadata["latent_decoder"]["output_channels"], channels)
                spec = perception_spec_from_config(cfg)
                _, height, width = processed_stack_shape(spec)
                decoder = InstinctLatentDecoder(cfg.latent_dim, channels, (height, width),
                    bounded_output=spec.depth_protocol!="g1_rpl_v1")
                image = torch.rand(2, channels, height, width)
                latent = actor.encode_processed(image)
                self.assertEqual(latent.shape, (2, 128))
                self.assertEqual(decoder(latent).shape, image.shape)
                self.assertEqual(metadata["world_model"]["shared_mlp_outputs"],
                                 {"physical_velocity_increment": 35,
                                  "contact": 13 if cfg.locomotion_objective==HIKING_FULL_OBJECTIVE else 2,
                                  "latent_delta": 128})
                if channels == 8:
                    self.assertFalse(metadata["perception"]["include_valid"])
                    self.assertEqual(metadata["latent_decoder"]["objective"],
                        "centered_processed_depth_stack_l1" if cfg.sensor_protocol==G1_RPL_SENSOR_PROTOCOL
                        else "processed_depth_only_stack_l1")
                    if cfg.sensor_protocol==G1_RPL_SENSOR_PROTOCOL:
                        self.assertEqual(metadata["latent_decoder"]["output_activation"], "linear")
                    self.assertNotIn("valid", metadata["execution_protocol"]["replay"])
                else:
                    self.assertNotIn("include_valid", metadata["perception"])
                    self.assertNotIn("sensor_protocol", metadata)
                    self.assertEqual(metadata["camera"]["pixel_coordinates"],
                                     "inclusive_linspace_minus1_to_plus1")

    def test_conflicting_explicit_channel_spec_is_rejected(self):
        for cfg in (old_optimized_config(), parse_config([])):
            wrong = replace(perception_spec_from_config(cfg),
                            include_valid=not perception_spec_from_config(cfg).include_valid)
            with self.subTest(protocol=cfg.sensor_protocol), self.assertRaisesRegex(ValueError, "sensor|channel|valid"):
                architecture_metadata(cfg, wrong)

    def test_saved_legacy_actor_with_prechange_spec_loads_identical_actions(self):
        torch.manual_seed(120)
        cfg = old_optimized_config()
        payload = record(cfg)
        source = build_perceptive_actor(payload).eval()
        payload["actor"] = source.state_dict()
        payload["config"].pop("sensor_protocol")
        payload["visual_spec"].pop("include_valid")
        features = torch.randn(3, 512)
        with torch.no_grad():
            expected = source(features)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"legacy_actor.pt"
            torch.save(payload, path)
            actual, restored, _ = load_actor(path)
            self.assertEqual(restored.sensor_protocol, LEGACY_SENSOR_PROTOCOL)
            self.assertTrue(actual.spec.include_valid)
            self.assertEqual(actual.vision.conv[0].in_channels, 16)
            with torch.no_grad():
                torch.testing.assert_close(actual(features), expected, rtol=0, atol=0)

    def test_new_actor_saved_spec_cannot_claim_legacy_channels(self):
        payload = record(parse_config([]))
        payload["actor"] = build_perceptive_actor(payload).state_dict()
        payload["visual_spec"]["include_valid"] = True
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"wrong_channels.pt"
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, "visual_spec"):
                load_actor(path)

    def test_unknown_sensor_and_accidental_pixel_depth_only_configuration_fail(self):
        cfg = parse_config([])
        for values in ({"sensor_protocol": "hiking"}, {"visual_mode": "pixel_v2"},
                       {"depth_noise_std_m": .005}, {"depth_dropout_probability": .01}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                replace(cfg, **values).validate()
        with self.assertRaises(ValueError):
            PerceptionSpec(include_valid="false")


if __name__ == "__main__":
    unittest.main()
