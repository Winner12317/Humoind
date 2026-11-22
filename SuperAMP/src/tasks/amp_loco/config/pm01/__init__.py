from mjlab.tasks.registry import register_mjlab_task
from src.tasks.amp_loco.rl import AMPOnPolicyRunner

from .env_cfgs import (
  pm01_amp_flat_env_cfg,
  pm01_amp_flat_aligned_env_cfg,
  pm01_amp_rough_env_cfg,
)
from .rl_cfg import pm01_amp_ppo_runner_cfg, pm01_amp_aligned_ppo_runner_cfg

register_mjlab_task(
  task_id="EngineAI-PM01-AMP-Rough",
  env_cfg=pm01_amp_rough_env_cfg(),
  play_env_cfg=pm01_amp_rough_env_cfg(play=True),
  rl_cfg=pm01_amp_ppo_runner_cfg(),
  runner_cls=AMPOnPolicyRunner,
)

register_mjlab_task(
  task_id="EngineAI-PM01-AMP-Flat",
  env_cfg=pm01_amp_flat_env_cfg(),
  play_env_cfg=pm01_amp_flat_env_cfg(play=True),
  rl_cfg=pm01_amp_ppo_runner_cfg(),
  runner_cls=AMPOnPolicyRunner,
)

register_mjlab_task(
  task_id="EngineAI-PM01-AMP-Flat-Aligned",
  env_cfg=pm01_amp_flat_aligned_env_cfg(),
  play_env_cfg=pm01_amp_flat_aligned_env_cfg(play=True),
  rl_cfg=pm01_amp_aligned_ppo_runner_cfg(),
  runner_cls=AMPOnPolicyRunner,
)
