from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  subtract_frame_transforms,
  quat_apply_inverse,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")

def robot_body_pos_b(
    env: ManagerBasedRlEnv,
    anchor_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
    body_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
) -> torch.Tensor:
    asset: Entity = env.scene[anchor_cfg.name]
    
    anchor_pos_w = asset.data.body_link_pos_w[:, anchor_cfg.body_ids[0]]   # (num_envs, 3)
    anchor_quat_w = asset.data.body_link_quat_w[:, anchor_cfg.body_ids[0]]  # (num_envs, 4)
    
    body_pos_w = asset.data.body_link_pos_w[:, body_cfg.body_ids]     # (num_envs, num_bodies, 3)
    body_quat_w = asset.data.body_link_quat_w[:, body_cfg.body_ids]   # (num_envs, num_bodies, 4)

    num_bodies = body_pos_w.shape[1]
    pos_b, _ = subtract_frame_transforms(
        anchor_pos_w[:, None, :].expand(-1, num_bodies, -1),
        anchor_quat_w[:, None, :].expand(-1, num_bodies, -1),
        body_pos_w,
        body_quat_w,
    )
    return pos_b.reshape(env.num_envs, -1)

def robot_body_ori_b(
    env: ManagerBasedRlEnv,
    anchor_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
    body_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
) -> torch.Tensor:
    asset: Entity = env.scene[anchor_cfg.name]
    
    anchor_pos_w = asset.data.body_link_pos_w[:, anchor_cfg.body_ids[0]]   # (num_envs, 3)
    anchor_quat_w = asset.data.body_link_quat_w[:, anchor_cfg.body_ids[0]]  # (num_envs, 4)
    
    body_pos_w = asset.data.body_link_pos_w[:, body_cfg.body_ids]     # (num_envs, num_bodies, 3)
    body_quat_w = asset.data.body_link_quat_w[:, body_cfg.body_ids]   # (num_envs, num_bodies, 4)

    num_bodies = body_pos_w.shape[1]
    _, ori_b = subtract_frame_transforms(
        anchor_pos_w[:, None, :].expand(-1, num_bodies, -1),
        anchor_quat_w[:, None, :].expand(-1, num_bodies, -1),
        body_pos_w,
        body_quat_w,
    )
    mat = matrix_from_quat(ori_b)
    return mat[..., :2].reshape(mat.shape[0], -1)

def robot_body_lin_vel_b(
    env: ManagerBasedRlEnv,
    anchor_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
    body_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
) -> torch.Tensor:
    asset: Entity = env.scene[anchor_cfg.name]
    
    body_lin_vel_w = asset.data.body_link_lin_vel_w[:, body_cfg.body_ids]   # (num_envs, num_bodies, 3)
    body_quat_w = asset.data.body_link_quat_w[:, body_cfg.body_ids]       # (num_envs, num_bodies, 4)

    num_bodies = body_lin_vel_w.shape[1]

    body_lin_vel_b = quat_apply_inverse(
        body_quat_w.reshape(-1, 4),
        body_lin_vel_w.reshape(-1, 3),
    ).reshape(env.num_envs, num_bodies, 3)

    return body_lin_vel_b.reshape(env.num_envs, -1)

def robot_body_ang_vel_b(
    env: ManagerBasedRlEnv,
    anchor_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
    body_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
) -> torch.Tensor:
    asset: Entity = env.scene[anchor_cfg.name]
    
    body_ang_vel_w = asset.data.body_link_ang_vel_w[:, body_cfg.body_ids]   # (num_envs, num_bodies, 3)
    body_quat_w = asset.data.body_link_quat_w[:, body_cfg.body_ids]       # (num_envs, num_bodies, 4)

    num_bodies = body_ang_vel_w.shape[1]

    body_ang_vel_b = quat_apply_inverse(
        body_quat_w.reshape(-1, 4),
        body_ang_vel_w.reshape(-1, 3),
    ).reshape(env.num_envs, num_bodies, 3)

    return body_ang_vel_b.reshape(env.num_envs, -1)


def robot_grounded_amp_features(
    env: ManagerBasedRlEnv,
    anchor_cfg: SceneEntityCfg,
    body_cfg: SceneEntityCfg,
    pelvis_cfg: SceneEntityCfg,
    foot_cfg: SceneEntityCfg,
    schema: str = "grounded_v2",
) -> torch.Tensor:
    """Shared expert/real/model AMP layout, including flat-ground pose cues."""
    # Entity derived body buffers can lag qpos by one physics substep. Read
    # the true generalized state and apply the same FK used in imagination.
    adapter = _aligned_state_adapter(env)
    if adapter.amp_feature_schema != schema:
        raise ValueError("AMP observation and state-adapter schemas must agree")
    return adapter.amp_features(adapter.state())


def _aligned_state_adapter(env: ManagerBasedRlEnv):
    from src.wmac.state import StateAdapter

    adapter = getattr(env, "_aligned_critic_state_adapter", None)
    if adapter is None:
        adapter = StateAdapter(env)
        env._aligned_critic_state_adapter = adapter
    return adapter


def aligned_critic_features(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Use the same 136 physical/context inputs as the WMAC critic."""
    adapter = _aligned_state_adapter(env)
    return adapter.critic_features(
        adapter.state(), env.action_manager.action, adapter.command(), adapter.context()
    )
