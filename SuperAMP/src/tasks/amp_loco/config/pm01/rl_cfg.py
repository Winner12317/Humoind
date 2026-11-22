"""RL configuration for the EngineAI PM01 AMP locomotion tasks.

The algorithm settings are the G1 ones; only what names the robot differs.
"""

import os

from src.assets.robots.engineai_pm01.pm01_constants import (
  PM01_AMP_BODY_NAMES,
  PM01_JOINT_NAMES,
  PM01_TORSO_BODY,
)
from src.tasks.amp_loco.config.g1.rl_cfg import (
  RslRlAmpRunnerCfg,
  g1_amp_aligned_ppo_runner_cfg,
  g1_amp_ppo_runner_cfg,
)

# AMP motion data directory (npz files), retargeted from the G1 set.
_MOTION_DATA_DIR = os.path.join(
  os.path.dirname(os.path.abspath(__file__)),
  os.pardir, os.pardir, os.pardir, os.pardir, os.pardir,
  "src", "assets", "motions", "pm01", "amp",
)


def _for_pm01(cfg: RslRlAmpRunnerCfg, experiment_name: str) -> RslRlAmpRunnerCfg:
  cfg.experiment_name = experiment_name
  cfg.amp_motion_files = os.path.normpath(_MOTION_DATA_DIR)
  cfg.min_normalized_std = [0.05] * len(PM01_JOINT_NAMES)
  cfg.amp_body_names = PM01_AMP_BODY_NAMES
  cfg.amp_anchor_name = PM01_TORSO_BODY
  return cfg


def pm01_amp_ppo_runner_cfg() -> RslRlAmpRunnerCfg:
  """Create RL runner configuration for the PM01 AMP locomotion task."""
  return _for_pm01(g1_amp_ppo_runner_cfg(), "pm01_amp_locomotion")


def pm01_amp_aligned_ppo_runner_cfg() -> RslRlAmpRunnerCfg:
  """Comparable PPO arm with grounded AMP inputs; keep the PPO update method."""
  return _for_pm01(g1_amp_aligned_ppo_runner_cfg(), "pm01_amp_locomotion_aligned")
