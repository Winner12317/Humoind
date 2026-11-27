"""Deterministic history updates alongside the unchanged GenCoord state."""
from __future__ import annotations

import torch


def normalize_hiking_sim_context(context):
    """Normalize the static Hiking DR inputs using the sampled parameter ranges."""
    # The mean foot coefficient occupies slot 0. The full per-geom vector
    # follows the 100 action/timer/kinematic protocol fields.
    lo, hi = .3, 1.6
    mean, scale = (lo + hi) / 2., (hi - lo) / 2.
    # The actuator lags are sampled uniformly over 0, 1, 2 physics substeps.
    return torch.cat(((context[..., :1] - mean) / scale, context[..., 1:67],
                      context[..., 67:96] - 1., context[..., 96:100],
                      (context[..., 100:] - mean) / scale), -1)


def advance_hiking_context(context,action,auxiliary,*,camera_capture_steps=0):
    """Advance action memory and learned substep timers without future labels.

    Delay, origin and material parameters are fixed during an episode. A full
    policy step fills the0..2-substep target buffers because decimation is4.
    The next four foot timers are supervised physical quantities. Both WM
    multistep fitting/validation and Actor imagination use this recurrence.
    """
    if context.shape[-1]<100 or action.shape[-1]!=29 or auxiliary.shape[-1]!=13:
        raise ValueError("Hiking context/action/auxiliary widths must be at least100/29/13")
    clock=(torch.remainder(context[...,34:35]+1,camera_capture_steps)
           if camera_capture_steps else context[...,34:35])
    return torch.cat((context[...,:33],torch.ones_like(context[...,33:34]),
        clock,context[...,35:38],action,context[...,67:96],auxiliary[...,9:13],context[...,100:]),-1)
