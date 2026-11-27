"""Shared real/imagined smooth-cost semantics; no changes to dynamics."""
from __future__ import annotations

import torch


def smooth_rewards_enabled(cfg) -> bool:
    return cfg.smooth_reward_mode == 'host_humanup_v1'


def previous_previous_action(actor_history: torch.Tensor) -> torch.Tensor:
    """History is time-major: frame t contains action a_(t-1).

    Therefore the preceding frame's last 29 channels are a_(t-2). This uses
    existing observations/replay, adding no WM input or persistent history.
    Observation noise has zero scale in these action channels.
    """
    return actor_history.reshape(*actor_history.shape[:-1], 4, 96)[..., -2, 67:96]


def smooth_cost_rate(terms: dict[str, torch.Tensor], cfg) -> torch.Tensor:
    return (cfg.smooth_action_second_coef * terms['action_second_difference']
            + cfg.smooth_joint_velocity_coef * terms['joint_velocity']
            + cfg.smooth_torque_coef * terms['requested_torque']
            + cfg.smooth_power_coef * terms['commanded_power'])


def physical_smooth_cost(env) -> torch.Tensor:
    """One reward-manager term evaluates all four rates once, before reset."""
    start = getattr(env, '_wmac_smooth_start', None)
    module = getattr(env, '_wmac_smooth_reward_terms', None)
    cfg = getattr(env, '_wmac_smooth_reward_cfg', None)
    if start is None or module is None or cfg is None:
        raise RuntimeError('Smooth rewards require CaptureEnv.step and initialized WMAC reward constants')
    with torch.no_grad():
        raw = module(start['state'], env.adapter.state(), env.action_manager.action,
                     start['previous_action'], previous_previous_action(start['actor_history']),
                     start['context'])
        env._wmac_last_smooth_terms = raw
        return smooth_cost_rate(raw, cfg)


def configure_smooth_reward_env(cfg, env_cfg) -> None:
    """A single negative cost rate; RewardManager applies the existing dt once."""
    if not smooth_rewards_enabled(cfg):
        return
    from mjlab.managers.reward_manager import RewardTermCfg
    if not env_cfg.scale_rewards_by_dt:
        raise ValueError('Smooth reward protocol requires existing dt-scaled task rewards')
    if 'wmac_smooth_penalty' in env_cfg.rewards:
        raise ValueError('Smooth reward term was configured twice')
    env_cfg.rewards['wmac_smooth_penalty'] = RewardTermCfg(func=physical_smooth_cost, weight=-1.)
