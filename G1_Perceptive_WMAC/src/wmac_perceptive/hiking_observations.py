"""Hiking torso-frame proprioception, shared by collection and imagination.

The source policy has eight samples of each of six terms, flattened term by
term by InstinctLab's VecEnv wrapper. Imagined rollouts use chronological
frames internally; the pack/unpack functions preserve the source input order.
"""
from __future__ import annotations

import torch

from src.wmac.state import quat_apply, quat_inv

HIKING_PROPRIO_HISTORY=8
HIKING_PROPRIO_FRAME=96
HIKING_PROPRIO_TERM_WIDTHS=(3,3,3,29,29,29)


def pack_hiking_proprio_history(history):
    """[..., 8, 96] chronological frames -> source term-major [..., 768]."""
    return torch.cat(tuple(term.flatten(-2) for term in history.split(HIKING_PROPRIO_TERM_WIDTHS,-1)),-1)


def unpack_hiking_proprio_history(observation):
    """Source term-major [..., 768] -> chronological [..., 8, 96]."""
    widths=tuple(HIKING_PROPRIO_HISTORY*width for width in HIKING_PROPRIO_TERM_WIDTHS)
    return torch.cat(tuple(term.reshape(*observation.shape[:-1],HIKING_PROPRIO_HISTORY,width)
                     for term,width in zip(observation.split(widths,-1),HIKING_PROPRIO_TERM_WIDTHS)),-1)


def hiking_angular_velocity(env,asset_cfg):
    data=env.scene[asset_cfg.name].data
    body=asset_cfg.body_ids[0]
    return quat_apply(quat_inv(data.body_link_quat_w[:,body]),data.body_link_ang_vel_w[:,body])


def hiking_projected_gravity(env,asset_cfg):
    quaternion=env.scene[asset_cfg.name].data.body_link_quat_w[:,asset_cfg.body_ids[0]]
    return quat_apply(quat_inv(quaternion),quaternion.new_tensor((0.,0.,-1.)))


def hiking_proprio_frame(torso_quaternion,torso_angular_velocity,joint_position,joint_velocity,
                        previous_action,command,default_joint_position,default_joint_velocity,
                        *,noise=False):
    omega=quat_apply(quat_inv(torso_quaternion),torso_angular_velocity)
    gravity=quat_apply(quat_inv(torso_quaternion),torso_quaternion.new_tensor((0.,0.,-1.)))
    joints=joint_position-default_joint_position
    velocity=joint_velocity-default_joint_velocity
    if noise:
        # IsaacLab observation managers apply corruption before scaling.
        omega=omega+torch.empty_like(omega).uniform_(-.2,.2)
        gravity=gravity+torch.empty_like(gravity).uniform_(-.05,.05)
        joints=joints+torch.empty_like(joints).uniform_(-.01,.01)
        velocity=velocity+torch.empty_like(velocity).uniform_(-.5,.5)
    return torch.cat((omega*.25,gravity,command,joints,velocity*.05,previous_action),dim=-1)


def hiking_actor_frame(adapter,state,previous_action,command,context=None,noise=False,kinematics=None):
    del context
    _,quaternions,_,angular_velocities=(adapter.kinematics(state) if kinematics is None else kinematics)
    torso=adapter.anchor_id
    return hiking_proprio_frame(quaternions[...,torso,:],angular_velocities[...,torso,:],
        state[...,13:42],state[...,42:71],previous_action,command,
        adapter.default_joint_pos,adapter.default_joint_vel,noise=noise)


def hiking_proprio_noise_amplitudes(*,device=None,dtype=torch.float32):
    """Half-widths after scaling, for callers adding one fused noise tensor."""
    return torch.tensor([.05]*3+[.05]*3+[0.]*3+[.01]*29+[.025]*29+[0.]*29,
                        device=device,dtype=dtype)


def configure_hiking_observations(cfg):
    from mjlab.managers.scene_entity_config import SceneEntityCfg
    from mjlab.utils.noise import UniformNoiseCfg
    actor=cfg.observations["actor"]
    actor.history_length=HIKING_PROPRIO_HISTORY
    actor.flatten_history_dim=True
    actor.concatenate_terms=True
    actor.history_ordering="term"
    terms=actor.terms
    terms["base_ang_vel"].func=hiking_angular_velocity
    terms["base_ang_vel"].params={"asset_cfg":SceneEntityCfg("robot",body_names=("torso_link",))}
    terms["base_ang_vel"].scale=.25
    terms["projected_gravity"].func=hiking_projected_gravity
    terms["projected_gravity"].params={"asset_cfg":SceneEntityCfg("robot",body_names=("torso_link",))}
    terms["joint_vel"].scale=.05
    for name,amplitude in (("base_ang_vel",.2),("projected_gravity",.05),("joint_pos",.01),("joint_vel",.5)):
        terms[name].noise=UniformNoiseCfg(n_min=-amplitude,n_max=amplitude)
    return cfg
