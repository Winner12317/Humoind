"""RPL depth-processing parity and reset/temporal regression checks."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import ast
import copy
import unittest

import torch

from src.wmac_perceptive.rpl_depth import (
    RPLDepthConfig, RPLDepthDelay, normalize_rpl_depth, preprocess_rpl_depth,
    resize_rpl_depth, rpl_depth_noise, rpl_depth_occlusion,
    sample_rpl_horizontal_fov, sample_rpl_mount_offsets,
)


class RPLDepthTests(unittest.TestCase):
    def test_reference_defaults(self):
        c = RPLDepthConfig()
        self.assertEqual((c.native_width, c.native_height, c.output_width, c.output_height), (240,135,48,27))
        self.assertEqual((c.std_multiplier,c.dropout_probability,c.dropout_value), (.1,.05,-2.5))
        self.assertEqual((c.delay,c.delay_buffer_length,c.frame_rate), (2,4,10.))
        with self.assertRaises(ValueError):
            replace(c, p_none=.2)
        with self.assertRaises(ValueError):
            replace(c, delay=4)

    def test_noise_is_relative_to_depth_and_dropout_is_near(self):
        c = RPLDepthConfig()
        depth = torch.ones(64, 1, 27, 48)
        depth[32:] *= 2
        noisy = rpl_depth_noise(depth, c, generator=torch.Generator().manual_seed(0))
        dropped = noisy == c.dropout_value
        self.assertAlmostEqual(float(dropped.float().mean()), .05, delta=.003)
        for sl, value in [(slice(0,32),1.),(slice(32,None),2.)]:
            kept = noisy[sl][~dropped[sl]]
            self.assertAlmostEqual(float(kept.mean()), value, delta=.006)
            self.assertAlmostEqual(float(kept.std()), value*.1, delta=.006)
        self.assertTrue(torch.equal(depth[:32],torch.ones_like(depth[:32])))

    def test_reference_large_and_small_strip_widths(self):
        depth = torch.full((2, 2, 135, 240), 5.)
        for small, bounds in [(False,(90,50)),(True,(60,20))]:
            c = replace(RPLDepthConfig(), p_none=0., p_small=float(small), p_large=float(not small), terrain_bias=0.)
            out = rpl_depth_occlusion(depth, torch.tensor([False,True]), c,
                                      generator=torch.Generator().manual_seed(9))
            left,right = bounds
            self.assertTrue(torch.all(out[..., :left] <= 2.))
            self.assertTrue(torch.all(out[..., :left] >= .1))
            self.assertTrue(torch.all(out[..., -right:] <= 2.))
            self.assertTrue(torch.equal(out[..., left:240-right],depth[...,left:240-right]))
            self.assertTrue(torch.equal(out[:,0],out[:,1]))

    def test_scaled_strips_and_terrain_bias(self):
        depth = torch.full((2000, 1, 2, 24), 5.)
        stones = torch.arange(2000) < 1000
        out = rpl_depth_occlusion(depth,stones,generator=torch.Generator().manual_seed(23))
        # At scale .1, small covers left6/right2 and large left9/right5.
        large = out[:,0,0,7] != 5.
        small_or_large = out[:,0,0,1] != 5.
        self.assertAlmostEqual(float(large[:1000].float().mean()),.3,delta=.05)
        self.assertAlmostEqual(float(large[1000:].float().mean()),.6,delta=.05)
        self.assertAlmostEqual(float(small_or_large.float().mean()),.9,delta=.03)
        with self.assertRaises(ValueError):
            rpl_depth_occlusion(depth,None)

    def test_clip_before_bicubic_and_keep_overshoot(self):
        c = replace(RPLDepthConfig(),noise_enabled=False,occlusion_enabled=False)
        raw = torch.full((1,1,135,240),1000.)
        raw[...,120:] = -2.5
        metric = resize_rpl_depth(raw,c)
        norm = preprocess_rpl_depth(raw,c,training=False)
        self.assertEqual(tuple(norm.shape),(1,1,27,48))
        self.assertTrue(torch.equal(norm,normalize_rpl_depth(metric,c)))
        self.assertTrue(bool((metric < .1).any() | (metric > 2.).any()))
        self.assertTrue(bool((norm < -.5).any() | (norm > .5).any()))

    def test_noise_reproducibility_and_evaluation_bypass(self):
        raw = torch.ones(3,1,135,240)
        stones = torch.tensor([True,False,False])
        a = preprocess_rpl_depth(raw,terrain_is_stepping_stones=stones,generator=torch.Generator().manual_seed(3))
        b = preprocess_rpl_depth(raw,terrain_is_stepping_stones=stones,generator=torch.Generator().manual_seed(3))
        self.assertTrue(torch.equal(a,b))
        clean = preprocess_rpl_depth(raw,training=False)
        self.assertTrue(torch.allclose(clean,torch.full_like(clean,(1.-.1)/1.9-.5)))

    def test_delay_and_reset_do_not_leak_previous_episode(self):
        c = RPLDepthConfig()
        d = RPLDepthDelay(3,c,generator=torch.Generator().manual_seed(4))
        self.assertTrue(torch.allclose(d.normalized,torch.zeros_like(d.normalized),atol=1e-7))
        for index,value in enumerate([.5,1.,1.5,1.8]):
            out = d.push(torch.full((3,1,27,48),value))
            expected = [1.05,1.05,.5,1.][index]
            self.assertTrue(torch.allclose(out,torch.full_like(out,expected)))
        d.push(torch.full((3,1,27,48),.3),reset_ids=[1])
        self.assertTrue(torch.allclose(d.output_depth[1],torch.full_like(d.output_depth[1],1.05)))
        self.assertTrue(torch.allclose(d.buffer[1,0],torch.full_like(d.buffer[1,0],.3)))
        self.assertTrue(torch.allclose(d.output_depth[0],torch.full_like(d.output_depth[0],1.5)))

    def test_mount_and_fov_ranges_without_changing_nominal_rig(self):
        g = torch.Generator().manual_seed(17)
        position, rotation = sample_rpl_mount_offsets(2000,generator=g)
        self.assertLessEqual(float(position.abs().max()),.025)
        limits = torch.tensor([2.5,3.,2.5])
        self.assertTrue(bool((rotation.abs() <= limits).all()))
        fov = sample_rpl_horizontal_fov(89.,2000,generator=g)
        self.assertGreaterEqual(float(fov.min()),84.)
        self.assertLessEqual(float(fov.max()),94.)
        self.assertGreater(float(position.std()),.01)


class RPLUpstreamOracleTests(unittest.TestCase):
    """Execute pinned upstream methods directly when the audit copy is present."""
    source = Path('/tmp/rpl-warp-depth-reference-20261113.py')

    @classmethod
    def setUpClass(cls):
        if not cls.source.is_file():
            raise unittest.SkipTest('Optional pinned Holosoma source oracle unavailable')
        tree = ast.parse(cls.source.read_text())
        camera = next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='WarpDepthCamera')
        methods = [copy.deepcopy(n) for n in camera.body if isinstance(n,ast.FunctionDef) and
                   n.name in ('_apply_noise','_setup_occlusion','_apply_random_depth_occlusion')]
        cls.namespace = {'torch':torch}
        exec(compile(ast.fix_missing_locations(ast.Module(body=methods,type_ignores=[])),str(cls.source),'exec'),cls.namespace)

    def make_reference(self, depth):
        c = RPLDepthConfig()
        ref = SimpleNamespace(depth_buffer=depth.clone(),noise_cfg={
            'std_multiplier':.1,'dropout_prob':.05,'dropout_value':-2.5},
            occlusion_cfg={'box':[90,50,0,0],'diff':30,'mask':[1,2,3],'prob':1.,
                           'terrain_bias':.15,'p_none':.1,'p_small':.45,'p_large':.45,'share_across_cameras':True},
            ray_width=depth.shape[-1],ray_height=depth.shape[-2],device=depth.device,
            props=SimpleNamespace(width=240,height=135,near_clip=.1,far_clip=2.),
            _terrain_state=SimpleNamespace(terrain_type_names=['stones','slopes','up','down'],
                terrain_type_idx=torch.tensor([0,1,2,3,0,1])))
        self.namespace['_setup_occlusion'](ref)
        return ref,c

    def test_seeded_noise_exactly_matches_upstream(self):
        raw = torch.linspace(.1,2.5,6*2*135*240).reshape(6,2,135,240)
        ref,c = self.make_reference(raw)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(51)
            self.namespace['_apply_noise'](ref)
            torch.manual_seed(51)
            actual = rpl_depth_noise(raw,c)
        self.assertTrue(torch.equal(actual,ref.depth_buffer))

    def test_seeded_terrain_occlusion_exactly_matches_upstream(self):
        for size in [(135,240),(36,64)]:
            raw = torch.full((6,2,*size),5.)
            ref,c = self.make_reference(raw)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(1)
                self.namespace['_apply_random_depth_occlusion'](ref)
                torch.manual_seed(1)
                actual = rpl_depth_occlusion(raw,ref._terrain_state.terrain_type_idx==0,c)
            self.assertTrue(torch.equal(actual,ref.depth_buffer),size)


if __name__ == '__main__':
    unittest.main()
