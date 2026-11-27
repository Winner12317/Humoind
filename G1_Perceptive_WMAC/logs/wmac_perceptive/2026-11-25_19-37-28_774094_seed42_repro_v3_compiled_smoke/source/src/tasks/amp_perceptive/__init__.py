"""New task registration; the original flat/recovery task IDs are unchanged."""
from mjlab.tasks.registry import register_mjlab_task
from src.tasks.amp_loco.config.g1.rl_cfg import g1_amp_aligned_ppo_runner_cfg
from .env_cfg import g1_amp_perceptive_hiking_env_cfg
from src.wmac_perceptive.config import PerceptiveConfig
from .entrypoint import PerceptiveTrainingEntrypointRequired


def _registered_environment(play=False):
    cfg = PerceptiveConfig.defaults("supermimic")
    env = g1_amp_perceptive_hiking_env_cfg(play=play, motion_scope=cfg.motion_scope,
        sensor_protocol=cfg.sensor_protocol, motion_data_protocol=cfg.motion_data_protocol,
        locomotion_objective=cfg.locomotion_objective, hiking_training_version=cfg.hiking_training_version)
    if not play:
        env.scene.num_envs = cfg.num_envs
    return env


register_mjlab_task(task_id="Unitree-G1-AMP-Perceptive-Hiking",
                    env_cfg=_registered_environment(), play_env_cfg=_registered_environment(play=True),
                    rl_cfg=g1_amp_aligned_ppo_runner_cfg(),runner_cls=PerceptiveTrainingEntrypointRequired)
