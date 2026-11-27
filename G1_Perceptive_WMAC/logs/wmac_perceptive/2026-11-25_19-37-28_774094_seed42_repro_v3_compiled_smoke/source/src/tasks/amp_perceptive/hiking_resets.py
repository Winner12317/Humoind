"""Hiking torso-root resets expressed in the local pelvis-root GenCoord state.

The upstream G1ParkourRoughEnvCfg initializes the torso link at 0.9 m and
samples its COM velocity. Sampling a pelvis at the same height or sampling
pelvis velocities would describe a different reset distribution. The inverse
FK conversion below preserves the original torso pose and velocities after
the independent joint (including waist) perturbation has been sampled.
"""
from __future__ import annotations

import torch

from src.wmac.state import quat_apply,quat_inv,quat_mul


def torso_state_to_pelvis(adapter,torso_pose,torso_com_velocity,joint_position,
                          joint_velocity,torso_com_offset):
    """Convert link pose + COM twist to the unchanged 71D pelvis-link state."""
    count=joint_position.shape[0]
    free=joint_position.new_zeros(count,13)
    free[:,3]=1.
    relative=torch.cat((free,joint_position,joint_velocity),-1)
    position,quaternion,linear,angular=adapter.kinematics(relative)
    torso=adapter.anchor_id
    root_quaternion=quat_mul(torso_pose[:,3:7],quat_inv(quaternion[:,torso]))
    lever=quat_apply(root_quaternion,position[:,torso])
    root_position=torso_pose[:,:3]-lever
    torso_omega=torso_com_velocity[:,3:6]
    root_omega=torso_omega-quat_apply(root_quaternion,angular[:,torso])
    com_lever=quat_apply(torso_pose[:,3:7],torso_com_offset)
    torso_link_velocity=torso_com_velocity[:,:3]-torch.cross(torso_omega,com_lever,dim=-1)
    root_velocity=(torso_link_velocity-torch.cross(root_omega,lever,dim=-1)
                   -quat_apply(root_quaternion,linear[:,torso]))
    return torch.cat((root_position,root_quaternion,root_velocity,root_omega,
                      joint_position,joint_velocity),-1)


def _uniform(limits,shape,like):
    bounds=like.new_tensor(limits)
    return bounds[...,0]+torch.rand(shape,device=like.device,dtype=like.dtype)*(bounds[...,1]-bounds[...,0])


def sample_hiking_reset_state(adapter,env_origins,default_joint_position,
                              default_joint_velocity,joint_position_limits,
                              torso_com_offset,*,joint_position_range=(-.15,.15),
                              pose_range=None,velocity_range=None):
    """Original independent reset draws, followed by a change of root frame.

    The optional zero ranges support controlled zero-speed standing/startup
    evaluation; training's defaults are precisely the original random ranges.
    """
    pose_range=({"x":(-.1,.1),"y":(-.1,.1),"yaw":(-.1,.1)} if pose_range is None else pose_range)
    velocity_range=({key:(-.2,.2) for key in ("x","y","z","roll","pitch","yaw")}
                    if velocity_range is None else velocity_range)
    count=len(env_origins);like=default_joint_position
    axes=("x","y","z","roll","pitch","yaw")
    pose=_uniform([pose_range.get(key,(0.,0.)) for key in axes],(count,6),like)
    from mjlab.utils.lab_api.math import quat_from_euler_xyz
    orientation=quat_from_euler_xyz(pose[:,3],pose[:,4],pose[:,5])
    position=env_origins+like.new_tensor((0.,0.,.9))+pose[:,:3]
    velocity=_uniform([velocity_range.get(key,(0.,0.)) for key in axes],(count,6),like)
    q=default_joint_position+_uniform(joint_position_range,default_joint_position.shape,like)
    q=q.clamp(joint_position_limits[...,0],joint_position_limits[...,1])
    # Source reset_joints_by_offset samples the zero velocity interval as
    # well. Preserve the RNG draw rather than introducing phase sampling.
    dq=default_joint_velocity+_uniform((0.,0.),default_joint_velocity.shape,like)
    return torso_state_to_pelvis(adapter,torch.cat((position,orientation),-1),
        velocity,q,dq,torso_com_offset)


def reset_hiking_state(env,env_ids=None,joint_position_range=(-.15,.15),
                       pose_range=None,velocity_range=None):
    from src.wmac_perceptive.hiking_rewards import _real_adapter
    adapter=_real_adapter(env)
    robot=env.scene["robot"]
    if env_ids is None:env_ids=torch.arange(env.num_envs,device=env.device)
    body_offset=env.sim.model.body_ipos
    if body_offset.ndim==2:body_offset=body_offset.unsqueeze(0)
    torso_offset=body_offset.expand(env.num_envs,-1,-1)[env_ids,adapter.torso_gid]
    state=sample_hiking_reset_state(adapter,env.scene.env_origins[env_ids],
        robot.data.default_joint_pos[env_ids],robot.data.default_joint_vel[env_ids],
        robot.data.soft_joint_pos_limits[env_ids],torso_offset,
        joint_position_range=joint_position_range,pose_range=pose_range,velocity_range=velocity_range)
    robot.write_joint_state_to_sim(state[:,13:42],state[:,42:71],env_ids=env_ids)
    robot.write_root_state_to_sim(state[:,:13],env_ids=env_ids)
    # A reset/evaluation may occur without increasing the physics-step count.
    env._hiking_reward_cache=None
    env._hiking_termination_cache=None


def configure_hiking_resets(cfg):
    from mjlab.managers.event_manager import EventTermCfg
    cfg.events.pop("reset_base",None)
    cfg.events.pop("reset_robot_joints",None)
    cfg.events["reset_hiking_state"]=EventTermCfg(func=reset_hiking_state,mode="reset")
    return cfg
