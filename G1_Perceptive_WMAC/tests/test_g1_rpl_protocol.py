"""Historical full-RPL schema and signed-value compatibility guards."""
import copy
from dataclasses import asdict,replace
import unittest

import torch

from scripts.train_wmac_perceptive import parse_config
from src.wmac_perceptive.config import (
    FORMAT,G1_RPL_SENSOR_PROTOCOL,HIKING_SENSOR_PROTOCOL,perception_spec_from_config,
)
from src.wmac_perceptive.checkpoints import architecture_metadata,validate_checkpoint
from src.wmac_perceptive.instinct_latent import InstinctLatentDecoder
from src.wmac_perceptive.latent_models import LatentPerceptiveActor
from src.wmac_perceptive.latent_perception import processed_depth_stack
from src.wmac_perceptive.perception import pack_actor_input


def historical_rpl_config():
    return parse_config(["--hiking-training-version","1",
                         "--sensor-protocol",G1_RPL_SENSOR_PROTOCOL,"--replay-storage","frame_fp16"])


class G1RPLProtocolTests(unittest.TestCase):
    def test_capture_clock_advances_without_new_state_dimensions(self):
        from src.wmac_perceptive.hiking_context import advance_hiking_context
        c=torch.zeros(5,100);c[:,34]=torch.arange(5)
        action=torch.zeros(5,29);aux=torch.zeros(5,13)
        new=advance_hiking_context(c,action,aux,camera_capture_steps=5)
        torch.testing.assert_close(new[:,34],torch.tensor([1.,2.,3.,4.,0.]))
        torch.testing.assert_close(advance_hiking_context(c,action,aux)[:,34],c[:,34])

    def test_historical_rpl_schema_names_calibration_and_synthetic_delay(self):
        cfg=historical_rpl_config();spec=perception_spec_from_config(cfg)
        meta=architecture_metadata(cfg)
        self.assertEqual(cfg.sensor_protocol,G1_RPL_SENSOR_PROTOCOL)
        self.assertEqual((spec.height,spec.width,spec.delay,spec.frames),(27,48,0,8))
        self.assertEqual((cfg.model_hidden,cfg.latent_dim),((2048,2048,2048),128))
        self.assertEqual(meta['architecture_version'],6)
        self.assertEqual(meta['camera']['capture_hz'],10.)
        self.assertEqual(meta['camera']['delay_capture_ticks'],2)
        self.assertFalse(meta['camera']['mounting_calibrated'])
        self.assertEqual(meta['latent_decoder']['output_activation'],'linear')
        self.assertEqual(meta['execution_protocol']['replay']['storage_protocol'],'centered_frame_fp16_v1')

    def test_negative_pixels_and_overshoot_survive_every_actor_entry(self):
        spec=perception_spec_from_config(historical_rpl_config())
        actor=LatentPerceptiveActor(spec,gate_hidden=()).eval()
        ring=torch.zeros(2,*spec.ring_shape)
        ring[:,:,0]=torch.linspace(-.7,.7,spec.height*spec.width).reshape(spec.height,spec.width)
        stack=processed_depth_stack(ring,spec)
        self.assertLess(float(stack.min()),-.5)
        self.assertGreater(float(stack.max()),.5)
        proprio=torch.randn(2,spec.proprio_dim)
        with torch.no_grad():
            raw=actor(pack_actor_input(proprio,ring,spec))
            latent=actor.encode_processed(stack)
            torch.testing.assert_close(raw,actor.forward_latent(proprio,latent))
            torch.testing.assert_close(actor.vision(stack),latent)

    def test_decoder_can_reconstruct_negative_rpl_targets(self):
        decoder=InstinctLatentDecoder(128,8,(27,48),bounded_output=False)
        with torch.no_grad():
            decoder.dec[-1].weight.zero_();decoder.dec[-1].bias.fill_(-.7)
            output=decoder(torch.zeros(2,128))
        torch.testing.assert_close(output,torch.full_like(output,-.7))

    def test_signed_spec_is_not_silently_accepted_as_old_hiking(self):
        cfg=historical_rpl_config();spec=perception_spec_from_config(cfg)
        with self.assertRaisesRegex(ValueError,'protocol'):
            architecture_metadata(cfg,replace(spec,depth_protocol='hiking_v1'))

    def test_historical_rpl_resume_preserves_camera_but_rejects_conversion(self):
        cfg=historical_rpl_config();spec=perception_spec_from_config(cfg)
        payload=dict(format=FORMAT,config=asdict(cfg),architecture=architecture_metadata(cfg),
            visual_spec=asdict(spec),amp_grounding=cfg.ground_height_semantics,
            terrain_fingerprint='terrain',motion_manifest_sha256='motion')
        self.assertEqual(validate_checkpoint(payload,cfg),{})
        old=replace(cfg,sensor_protocol=HIKING_SENSOR_PROTOCOL,replay_storage='frame_uint8')
        with self.assertRaisesRegex(ValueError,'architecture'):
            validate_checkpoint(payload,old)


if __name__=='__main__':unittest.main()
