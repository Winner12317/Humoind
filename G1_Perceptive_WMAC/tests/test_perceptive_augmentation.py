"""Observation corruption never rewrites clean sensor/model targets."""
import unittest
import torch
from src.wmac_perceptive.augmentation import augment_depth_frame


class DepthAugmentationTest(unittest.TestCase):
    def test_metric_noise_scale_and_nohit_convention(self):
        torch.manual_seed(53)
        clean=torch.full((32,2,36,64),.5)
        clean[:,1]=1.
        before=clean.clone()
        noisy=augment_depth_frame(clean,std_m=.005,dropout_probability=0.)
        self.assertAlmostEqual(float((noisy[:,0]-clean[:,0]).std())*2.5,.005,delta=.0002)
        torch.testing.assert_close(clean,before,rtol=0,atol=0)
        missing=clean.clone();missing[:,0]=1.;missing[:,1]=0.
        torch.testing.assert_close(augment_depth_frame(missing,std_m=.005,dropout_probability=0.),missing,rtol=0,atol=0)

    def test_dropout_is_far_and_invalid_with_no_fake_close_obstacle(self):
        torch.manual_seed(71)
        clean=torch.full((3,2,8,8),.4,requires_grad=True)
        with torch.no_grad():clean[:,1]=1.
        noisy=augment_depth_frame(clean,std_m=0.,dropout_probability=.5)
        dropped=noisy[:,1]==0
        self.assertGreater(int(dropped.sum()),0)
        self.assertLess(int(dropped.sum()),dropped.numel())
        torch.testing.assert_close(noisy[:,0][dropped],torch.ones_like(noisy[:,0][dropped]),rtol=0,atol=0)
        noisy[:,0].sum().backward()
        torch.testing.assert_close(clean.grad[:,0][dropped],torch.zeros_like(clean.grad[:,0][dropped]),rtol=0,atol=0)
        torch.testing.assert_close(clean.grad[:,0][~dropped],torch.ones_like(clean.grad[:,0][~dropped]),rtol=0,atol=0)

    def test_disabled_corruption_preserves_exact_sensor_tensor(self):
        frame=torch.rand(3,2,4,4,requires_grad=True)
        self.assertIs(augment_depth_frame(frame,std_m=0.,dropout_probability=0.),frame)


if __name__=="__main__":unittest.main()
