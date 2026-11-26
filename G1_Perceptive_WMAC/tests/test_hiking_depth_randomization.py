"""Hiking pixel-domain adaptation, distinct from the full RPL camera preset."""
from dataclasses import replace
import unittest

import torch

from src.wmac_perceptive.camera_calibration import load_g1_depth_calibration
from src.wmac_perceptive.hiking_depth_randomization import (
    HikingDepthCapture, HikingDepthRandomizationCfg, augment_hiking_depth_packet,
    preprocess_real_g1_hiking_z16,
)
from src.wmac_perceptive.perception import PerceptionSpec
from src.wmac_perceptive.visual_metrics import preprocess_depth_channels


def packet(n=4, value=1.):
    depth = torch.full((n, 1, 36, 64), value/2.5)
    return torch.cat((depth, torch.ones_like(depth)), 1)


def process(value):
    return preprocess_depth_channels(value[:, :1], PerceptionSpec(include_valid=False),
        channels=1, include_valid=False).clamp(0, 1)


class HikingDepthRandomizationTests(unittest.TestCase):
    def test_hiking_geometry_ranges_and_roi_relative_mask_widths(self):
        cfg=HikingDepthRandomizationCfg()
        self.assertEqual((cfg.height,cfg.width,cfg.roi_height,cfg.roi_width),(36,64,18,32))
        self.assertEqual(cfg.occlusion_widths,(12,7,8,3))
        self.assertEqual((cfg.max_depth,cfg.missing_depth_m,cfg.dropout_depth_m),(2.5,1e6,2.5))
        with self.assertRaises(ValueError):replace(cfg,p_none=.2)
        with self.assertRaises(ValueError):replace(cfg,dropout_depth_m=-2.5)
        with self.assertRaises(ValueError):replace(cfg,dropout_depth_m=1e6)
        with self.assertRaises(ValueError):replace(cfg,crop_left=63)

    def test_disabled_is_exact_identity_and_draws_no_random_numbers(self):
        value=packet();value[0,0,25,24]=1e6/2.5;value[0,1,25,24]=0.
        rng=torch.Generator().manual_seed(1);before=rng.get_state().clone()
        actual=augment_hiking_depth_packet(value,enabled=False,generator=rng)
        self.assertIs(actual,value)
        self.assertTrue(torch.equal(before,rng.get_state()))
        torch.testing.assert_close(process(actual),process(value),rtol=0,atol=0)

    def test_relative_noise_only_modifies_valid_retained_pixels(self):
        cfg=replace(HikingDepthRandomizationCfg(),dropout_probability=0.,occlusion_enabled=False)
        value=packet(128);value[64:,0]*=2
        value[0,0,20,20]=1e6/2.5;value[0,1,20,20]=0.
        value[1,0,20,20]=4./2.5;value[1,1,20,20]=0.
        original=value.clone()
        out=augment_hiking_depth_packet(value,cfg=cfg,generator=torch.Generator().manual_seed(2))
        for sl,truth in [(slice(2,64),1.),(slice(64,128),2.)]:
            roi=out[sl,0,18:,16:48]*2.5
            self.assertAlmostEqual(float(roi.mean()),truth,delta=.006)
            self.assertAlmostEqual(float(roi.std()),truth*.1,delta=.006)
        self.assertEqual(out[0,0,20,20],value[0,0,20,20])
        self.assertEqual(out[1,0,20,20],value[1,0,20,20])
        outside=torch.ones(36,64,dtype=torch.bool);outside[18:,16:48]=False
        self.assertTrue(torch.equal(out[:,:,outside],value[:,:,outside]))
        self.assertTrue(torch.equal(value,original))

    def test_synthetic_dropout_is_far_endpoint_not_physical_no_hit(self):
        cfg=replace(HikingDepthRandomizationCfg(),std_multiplier=0.,dropout_probability=1.,occlusion_enabled=False)
        value=packet();value[0,0,20,20]=1e6/2.5;value[0,1,20,20]=0.
        out=augment_hiking_depth_packet(value,cfg=cfg,generator=torch.Generator().manual_seed(3))
        roi=out[:,0,18:,16:48]
        self.assertEqual(roi[0,2,4],1e6/2.5)
        expected=torch.ones_like(roi);expected[0,2,4]=1e6/2.5
        self.assertTrue(torch.equal(roi,expected))
        self.assertTrue(torch.equal(out[:,1,18:,16:48],torch.zeros_like(roi)))

    def test_five_percent_synthetic_dropout_does_not_saturate_blur_neighborhoods(self):
        cfg=replace(HikingDepthRandomizationCfg(),std_multiplier=0.,occlusion_enabled=False)
        value=packet(128)
        out=augment_hiking_depth_packet(value,cfg=cfg,generator=torch.Generator().manual_seed(71))
        missing=(out[:,1,18:,16:48]==0)
        self.assertAlmostEqual(float(missing.float().mean()),.05,delta=.004)
        processed=process(out)
        # A lone2.5m artificial hole blends with its1m neighbors rather than
        # propagating a1e6m sentinel into nine clipped-far output pixels.
        self.assertLess(float((processed>=1.-1e-6).float().mean()),.001)
        self.assertLess(float(processed.max()),.99)

    def test_large_and_small_masks_survive_the_hiking_crop(self):
        for small,widths in [(False,(12,7)),(True,(8,3))]:
            cfg=replace(HikingDepthRandomizationCfg(),noise_enabled=False,p_none=0.,
                p_small=float(small),p_large=float(not small),terrain_bias=0.)
            value=packet(2)
            out=augment_hiking_depth_packet(value,torch.tensor([False,True]),cfg,
                generator=torch.Generator().manual_seed(4))
            left,right=widths
            roi=out[:,0,18:,16:48]
            self.assertTrue(bool((roi[:,:,:left]>=0).all() & (roi[:,:,:left]<=1).all()))
            self.assertTrue(bool((roi[:,:,-right:]>=0).all() & (roi[:,:,-right:]<=1).all()))
            self.assertTrue(torch.equal(roi[:,:,left:32-right],value[:,0,18:,16+left:48-right]))
            valid=out[:,1,18:,16:48]
            self.assertTrue(bool((valid[:,:,:left]==0).all()))
            self.assertTrue(bool((valid[:,:,-right:]==0).all()))
            outside=torch.ones(36,64,dtype=torch.bool);outside[18:,16:48]=False
            self.assertTrue(torch.equal(out[:,:,outside],value[:,:,outside]))

    def test_terrain_bias_and_capture_seed_are_reproducible(self):
        cfg=replace(HikingDepthRandomizationCfg(),noise_enabled=False)
        value=packet(2000);stones=torch.arange(2000)<1000
        first=augment_hiking_depth_packet(value,stones,cfg,generator=torch.Generator().manual_seed(5))
        second=augment_hiking_depth_packet(value,stones,cfg,generator=torch.Generator().manual_seed(5))
        self.assertTrue(torch.equal(first,second))
        # ROI column9 is covered by large(left12), not small(left8).
        large=(first[:,1,20,16+9]==0)
        self.assertAlmostEqual(float(large[:1000].float().mean()),.3,delta=.05)
        self.assertAlmostEqual(float(large[1000:].float().mean()),.6,delta=.05)
        with self.assertRaises(ValueError):augment_hiking_depth_packet(value,cfg=cfg)

    def test_same_control_step_shares_packet_and_rng_state(self):
        rng=torch.Generator().manual_seed(6)
        stream=HikingDepthCapture(3,generator=rng)
        raw=packet(3);stones=torch.zeros(3,dtype=torch.bool)
        first=stream.step(raw,control_step=0,terrain_is_stepping_stones=stones)
        frozen=first.clone();state=rng.get_state().clone()
        second=stream.step(None,control_step=0)
        self.assertEqual(first.data_ptr(),second.data_ptr())
        self.assertTrue(torch.equal(frozen,second))
        self.assertTrue(torch.equal(state,rng.get_state()))
        for tick in [1,2]:
            stream.step(raw,control_step=tick,terrain_is_stepping_stones=stones)
        self.assertEqual(stream.capture_count,3)
        torch.testing.assert_close(stream.captures_per_env,torch.full((3,),3,dtype=torch.long))
        self.assertFalse(hasattr(stream,'delay'))

    def test_reset_refreshes_only_new_physical_rows_and_preserves_terminal_copy(self):
        stream=HikingDepthCapture(3,enabled_randomization=False)
        first=stream.step(packet(3,1.),control_step=7).clone()
        terminal=first.clone()
        stream.reset(torch.tensor([False,True,False]))
        self.assertTrue(torch.equal(stream.packet,terminal))
        reset=stream.step(packet(3,2.),control_step=7)
        self.assertTrue(torch.equal(reset[[0,2]],terminal[[0,2]]))
        self.assertTrue(torch.equal(reset[1],packet(1,2.)[0]))
        self.assertTrue(torch.equal(terminal,first))
        torch.testing.assert_close(stream.captures_per_env,torch.tensor([1,2,1]))
        self.assertIs(stream.step(None,control_step=7),stream.packet)
        stream.reset([0,2]);self.assertTrue(stream._has_pending)

    def test_invalid_data_shape_and_counter_are_rejected(self):
        value=packet();value[0,0,20,20]=float('nan')
        with self.assertRaisesRegex(ValueError,'non-finite'):
            augment_hiking_depth_packet(value,enabled=False)
        with self.assertRaises(ValueError):augment_hiking_depth_packet(torch.ones(2,1,36,64))
        with self.assertRaises(ValueError):augment_hiking_depth_packet(packet(),torch.tensor([True]))
        stream=HikingDepthCapture(4,enabled_randomization=False)
        stream.step(packet(),control_step=2)
        with self.assertRaises(ValueError):stream.step(packet(),control_step=1)
        with self.assertRaises(ValueError):stream.reset(torch.zeros(3,dtype=torch.bool))
        with self.assertRaises(ValueError):stream.reset([4])

    def test_real_z16_uses_measured_scale_and_original_hiking_output(self):
        calibration=load_g1_depth_calibration()
        raw=torch.full((calibration.height,calibration.width),1000,dtype=torch.uint16)
        result=preprocess_real_g1_hiking_z16(raw)
        self.assertEqual(tuple(result.shape),(1,1,18,32))
        torch.testing.assert_close(result,torch.full_like(result,1000*calibration.depth_scale_m_per_raw_unit/2.5))
        self.assertTrue(bool((result>=0).all() & (result<=1).all()))

    def test_real_invalid_is_not_preclipped_before_original_blur(self):
        calibration=load_g1_depth_calibration().resized(64,36)
        raw=torch.full((1,36,64),1000,dtype=torch.int32);raw[0,25,25]=0
        actual=preprocess_real_g1_hiking_z16(raw,calibration)
        depth=raw[:,None].float()*calibration.depth_scale_m_per_raw_unit
        depth=torch.where(raw[:,None]>0,depth,torch.full_like(depth,1e6))
        expected=preprocess_depth_channels(depth/2.5,PerceptionSpec(include_valid=False),
            channels=1,include_valid=False).clamp(0,1)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        self.assertEqual(float(actual[0,0,25-18,25-16]),1.)
        with self.assertRaises(ValueError):preprocess_real_g1_hiking_z16(raw)


if __name__=='__main__':unittest.main()
