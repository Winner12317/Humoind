"""Explicit observation noise, separate from clean world-model pixel targets.

This lightweight range noise/dropout is an initial migration setting, not the
complete Hiking disparity/blur/OOD sim-to-real pipeline. Corruption is sampled
once on insertion into real/imagined image histories, never inside Actor
forward; trust-region action comparisons therefore see identical inputs.
"""
from __future__ import annotations

import torch


def augment_depth_frame(frame,*,std_m=.005,dropout_probability=.01,max_depth=2.5):
    if frame.shape[-3]!=2:raise ValueError("Expected depth and validity channels")
    if std_m<0 or not 0<=dropout_probability<1 or max_depth<=0:raise ValueError("Invalid depth augmentation settings")
    if std_m==0 and dropout_probability==0:return frame
    depth,valid=frame.unbind(-3)
    if std_m:
        depth=(depth+torch.randn_like(depth)*(std_m/max_depth)*valid).clamp(0,1)
    if dropout_probability:
        dropped=torch.rand_like(valid)<dropout_probability
        depth=torch.where(dropped,torch.ones_like(depth),depth)
        valid=torch.where(dropped,torch.zeros_like(valid),valid)
    return torch.stack((depth,valid),-3)
