"""Historical v2 camera corruption retains Hiking image semantics."""
from dataclasses import asdict,replace
import unittest

import torch

from scripts.train_wmac_perceptive import parse_config
from src.tasks.amp_perceptive.sensors import HikingDepthPatternCfg
from src.wmac_perceptive.config import FORMAT,perception_spec_from_config
from src.wmac_perceptive.checkpoints import architecture_metadata,validate_checkpoint
from src.wmac_perceptive.instinct_latent import InstinctLatentDecoder
from src.wmac_perceptive.latent_perception import processed_depth_stack,processed_stack_shape
from src.wmac_perceptive.perception import PerceptionSpec


class HikingCameraRandomizationProtocolTests(unittest.TestCase):
    def test_v2_default_keeps_hiking_shape_range_and_timing(self):
        cfg=parse_config(["--hiking-training-version", "2"]);spec=perception_spec_from_config(cfg)
        self.assertEqual(cfg.sensor_protocol,'g1_hiking_dr_v1')
        self.assertEqual(cfg.replay_storage,'frame_uint8')
        self.assertEqual(spec,PerceptionSpec(include_valid=False,proprio_dim=768))
        self.assertEqual(processed_stack_shape(spec),(8,18,32))
        self.assertEqual(cfg.model_hidden,(2048,2048,2048))

    def test_metadata_records_body_mount_and_no_rpl_timing(self):
        cfg=parse_config(["--hiking-training-version", "2"]);meta=architecture_metadata(cfg);cam=meta['camera']
        self.assertEqual(meta['architecture_version'],8)
        self.assertEqual(cam['image_shape'],[36,64])
        self.assertEqual(cam['processed_image_shape'],[18,32])
        self.assertEqual(cam['position'],list(HikingDepthPatternCfg.position))
        self.assertEqual(cam['rotation_wxyz'],list(HikingDepthPatternCfg.rotation))
        self.assertEqual(cam['capture_hz'],50.)
        self.assertEqual(cam['hardware_stream_hz'],30.)
        self.assertEqual(cam['delay_policy_steps'],1)
        self.assertNotIn('delay_capture_ticks',cam)
        self.assertEqual(meta['context_layout']['padding'],[34,35])
        self.assertEqual(meta['latent_decoder']['output_activation'],'sigmoid')
        self.assertEqual(meta['execution_protocol']['replay']['storage_protocol'],'hiking_lossless_frame_uint8_v3')

    def test_deterministic_actor_preprocessing_is_exact_hiking(self):
        spec=perception_spec_from_config(parse_config(["--hiking-training-version", "2"]))
        original=PerceptionSpec(include_valid=False,proprio_dim=768)
        ring=torch.rand(2,*spec.ring_shape)
        ring[:,3,0,20,24]=400000.  # Preserve no-hit before blur/clipping.
        torch.testing.assert_close(processed_depth_stack(ring,spec),
                                   processed_depth_stack(ring,original),rtol=0,atol=0)

    def test_decoder_restores_bounded_hiking_reconstruction(self):
        decoder=InstinctLatentDecoder(128,8,(18,32))
        output=decoder(torch.randn(2,128))
        self.assertEqual(tuple(output.shape),(2,8,18,32))
        self.assertTrue(bool(((output>=0)&(output<=1)).all()))

    def test_checkpoint_round_trip_and_sensor_conversion_are_strict(self):
        cfg=parse_config(["--hiking-training-version", "2"])
        payload=dict(format=FORMAT,config=asdict(cfg),architecture=architecture_metadata(cfg),
            visual_spec=asdict(perception_spec_from_config(cfg)),terrain_fingerprint='terrain',
            motion_manifest_sha256='motion')
        self.assertEqual(validate_checkpoint(payload,cfg),{})
        old=replace(cfg,sensor_protocol='hiking_depth_only_v1')
        with self.assertRaisesRegex(ValueError,'architecture'):
            validate_checkpoint(payload,old)


if __name__=='__main__':unittest.main()
