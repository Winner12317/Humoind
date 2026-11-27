"""Hiking's 67-dimensional AMP frame and chronological ten-frame histories.

Matches InstinctLab de6b92f's five AMP observation terms. Policy linear
velocity is measured at the torso COM, as IsaacLab's root_lin_vel_b is a COM
velocity. The reference loader differentiates the torso origin trajectory.
This deliberate source distinction is retained. Neither path needs a terrain
height estimate, foot-contact inference or a running observation normalizer.
"""
from __future__ import annotations

import torch
from src.wmac.state import quat_apply, quat_inv

HISTORY_LENGTH = 10
FRAME_DIM = 67
TERM_WIDTHS = (3, 29, 29, 3, 3)


def frame(joint_pos, joint_vel, torso_quat, torso_lin_vel_w, torso_ang_vel_w,
          default_joint_pos, default_joint_vel=None):
    inverse = quat_inv(torso_quat)
    gravity_w = torch.cat((torch.zeros_like(torso_quat[..., :2]),
                           -torch.ones_like(torso_quat[..., :1])), -1)
    gravity = quat_apply(inverse, gravity_w)
    if default_joint_vel is None:
        default_joint_vel = torch.zeros_like(default_joint_pos)
    return torch.cat((gravity, joint_pos-default_joint_pos,
                      .05*(joint_vel-default_joint_vel),
                      quat_apply(inverse, torso_lin_vel_w),
                      quat_apply(inverse, torso_ang_vel_w)), -1)


def flatten_history(history):
    """[...,10,67] → [...,670], preserving the official term-major order."""
    if history.shape[-2:] != (HISTORY_LENGTH, FRAME_DIM):
        raise ValueError("Hiking AMP requires ten chronological 67-dimensional frames")
    return torch.cat(tuple(term.flatten(-2) for term in history.split(TERM_WIDTHS, -1)), -1)


def sliding_histories(initial_history, future_frames):
    """History at each imagined successor; initial_history is [B,10,67]."""
    # future_frames is [H,B,67]; retain real predecessors for the first H<10.
    sequence = torch.cat((initial_history[:, 1:].transpose(0, 1), future_frames), 0)
    # unfold appends the window dimension at the end.
    windows = sequence.unfold(0, HISTORY_LENGTH, 1).movedim(-1, -2)
    return windows


def configure_replay(replay):
    """Retain only nine overwritten states, avoiding cached 10-frame copies."""
    if getattr(replay, "_hiking_amp_history", False):
        return
    replay._hiking_amp_history = True
    n, c, device = replay.num_envs, replay.capacity_steps, replay.device
    replay._amp_overflow_states = torch.empty(HISTORY_LENGTH-1, n, 71, device=device)
    replay._amp_episode_starts = torch.empty(c, n, dtype=torch.long, device=device)
    replay._amp_episode_start = torch.zeros(n, dtype=torch.long, device=device)
    replay._amp_pending_reset = torch.zeros(n, dtype=torch.bool, device=device)


@torch.no_grad()
def before_replay_add(replay):
    if not getattr(replay, "_hiking_amp_history", False):
        return
    row, time = replay.ptr, replay.total_steps
    if replay.size == replay.capacity_steps:
        old_time = time-replay.capacity_steps
        replay._amp_overflow_states[old_time % (HISTORY_LENGTH-1)].copy_(replay.data["state"][row])
    start = torch.where(replay._amp_pending_reset, time, replay._amp_episode_start)
    replay._amp_episode_starts[row].copy_(start)
    replay._amp_episode_start.copy_(start)


@torch.no_grad()
def after_replay_add(replay, batch, accepted):
    if not getattr(replay, "_hiking_amp_history", False):
        return
    reset = ~accepted.to(device=replay.device).bool()
    for name in ("terminated", "timeout"):
        if name in batch:
            reset = reset | batch[name].to(device=replay.device).reshape(replay.num_envs, -1).bool().any(-1)
    replay._amp_pending_reset.copy_(reset)


def add_sample_history(replay, batch):
    if not getattr(replay, "_hiking_amp_history", False):
        return batch
    time, env = batch["time_index"], batch["env_index"]
    if time.ndim != 1:
        # WM training windows never consume discriminator histories.
        return batch
    row = time.remainder(replay.capacity_steps)
    start = replay._amp_episode_starts[row, env]
    # At a reset the first observed frame fills the complete CircularBuffer.
    times = torch.maximum(time[..., None]+torch.arange(1-HISTORY_LENGTH, 1, device=replay.device),
                          start[..., None])
    rows = times.remainder(replay.capacity_steps)
    states = replay.data["state"][rows, env[..., None]]
    overflow = replay._amp_overflow_states[times.remainder(HISTORY_LENGTH-1), env[..., None]]
    old = times < replay.total_steps-replay.size
    batch["amp_state_history"] = torch.where(old[..., None], overflow, states)
    return batch
