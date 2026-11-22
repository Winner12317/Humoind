"""Validate live action history/reset alignment and real reward dt exactly once."""
from pathlib import Path
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('OMP_NUM_THREADS', '1')

import torch
import mjlab.tasks
import src.tasks
from mjlab.tasks.registry import load_env_cfg
from src.recovery_rewards import configure_recovery_height_reward
from src.wmac.config import WmacConfig
from src.wmac.env import CaptureEnv
from src.wmac.state import StateAdapter
from src.wmac.smooth_rewards import SmoothRewardTerms
from src.wmac.reward_integration import (configure_smooth_reward_env,
                                         previous_previous_action, smooth_cost_rate)


def main():
    torch.manual_seed(173)
    cfg = WmacConfig.for_profile('supermimic_official')
    cfg.smooth_reward_mode = 'host_humanup_v1'
    env_cfg = load_env_cfg(cfg.task)
    configure_recovery_height_reward(env_cfg, cfg.recovery_height_reward)
    configure_smooth_reward_env(cfg, env_cfg)
    env_cfg.scene.num_envs, env_cfg.seed = 4, 173
    env_cfg.events.pop('push_robot', None)
    for group in env_cfg.observations.values():
        group.enable_corruption = False
    env = CaptureEnv(cfg=env_cfg, device='cuda:0')
    checks = []
    try:
        env.adapter = StateAdapter(env, recovery_counter_semantics=cfg.recovery_counter_semantics)
        env.reset(seed=173)
        module = SmoothRewardTerms.from_env(env)
        env._wmac_smooth_reward_terms, env._wmac_smooth_reward_cfg = module, cfg
        index = list(env.reward_manager.active_terms).index('wmac_smooth_penalty')
        with torch.no_grad():
            for step in range(8):
                if step == 4:
                    # Public reset exercises all episode/action/history buffers.
                    env.reset(seed=174)
                history = env.observation_manager.compute()['actor'].clone()
                older = previous_previous_action(history)
                torch.testing.assert_close(older, env.action_manager.prev_action, rtol=0, atol=0)
                torch.testing.assert_close(history.reshape(4, 4, 96)[:, -1, 67:96],
                                           env.action_manager.action, rtol=0, atol=0)
                action = torch.sin(torch.arange(4 * 29, device=env.device).reshape(4, 29) * .07 + step) * .15
                env.step(action)
                row = env.transition
                torch.testing.assert_close(previous_previous_action(row['actor_history']),
                                           env.action_manager.prev_prev_action, rtol=0, atol=0)
                raw = module(row['state'], row['next_state'], row['action'], row['previous_action'],
                             older, row['context'])
                rate = smooth_cost_rate(raw, cfg)
                manager_term = env.reward_manager._step_reward[:, index]
                torch.testing.assert_close(manager_term, -rate, rtol=2e-6, atol=2e-7)
                torch.testing.assert_close(row['reward'], env.reward_manager._step_reward.sum(-1) * env.step_dt,
                                           rtol=2e-6, atol=2e-7)
                checks.append(dict(step=step, reset_before_step=step == 4,
                                   history_max_error=float((previous_previous_action(row['actor_history'])
                                                            - env.action_manager.prev_prev_action).abs().max()),
                                   reward_rate_max_error=float((manager_term + rate).abs().max())))
        result = dict(physical_steps=8, num_envs=4, optimizer_steps=0,
                      history_and_reward_checks=checks,
                      registered_terms=list(env.reward_manager.active_terms),
                      motor_metadata_sha256=module.metadata()['sha256'])
        output = ROOT/'docs/validation/smooth_reward_physical_validation.json'
        output.write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result, indent=2))
    finally:
        env.close()


if __name__ == '__main__':
    main()
