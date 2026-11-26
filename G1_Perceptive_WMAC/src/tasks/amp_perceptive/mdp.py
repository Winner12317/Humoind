"""Terrain-relative task semantics with explicit legacy recovery support."""
from __future__ import annotations

import torch
from src.amp_features import build_amp_features,G1_FOOT_REFERENCE_OFFSET
from src.recovery_rewards import recovery_height_reward,recovery_root_up_z
from src.tasks.amp_loco.mdp.rewards import _apply_delay_env_reward_mask_only
from src.tasks.amp_loco.mdp.terminations import DelayedTerminationManager
from src.wmac.state import StateAdapter,quat_apply,quat_inv
from .terrain import TerrainHeightField


def hiking_reset_protocol_metadata():
    """Hiking's independent standing reset, expressed in the local root model.

    The official G1 URDF has a torso free root; this MJCF has a pelvis free
    root. Each uses its own model's default root state on the terrain origin.
    In particular, the official torso's 0.9 m height is not a pelvis height.
    Reference trajectories supply AMP examples only, never reset poses.
    """
    return dict(
        reference_commit="de6b92f01eb155d51f30e8ed1ea7aecd2eb1f057",
        reference_file="source/instinctlab/instinctlab/tasks/parkour/config/parkour_env_cfg.py",
        reference_root="torso_link", local_root="pelvis",
        initial_state="local_model_default_standing_state_plus_terrain_origin",
        pose_range={"x": (-.1,.1), "y": (-.1,.1), "yaw": (-.1,.1)},
        velocity_range={key: (-.2,.2) for key in ("x","y","z","roll","pitch","yaw")},
        joint_position_offset_range=(-.15,.15),
        joint_velocity_offset_range=(0.,0.),
        reference_motion_used_for_reset=False,
    )


def hiking_stand_still_term(joint_pos,default_joint_pos,command,threshold=.15,offset=4.):
    """Unweighted Hiking stand_still term, shared by physics and imagination.

    The upstream L1 posture term and strict command thresholds are retained
    exactly. The offset makes the default standing posture receive a positive
    contribution after applying the configured negative weight; it is not an
    unconditional posture or joint-velocity penalty.
    """
    deviation=(joint_pos-default_joint_pos).abs().sum(-1)
    return ((deviation-offset)
            *(torch.norm(command[...,:2],dim=-1)<threshold)
            *(command[...,2].abs()<threshold))


def hiking_stand_still(env,command_name="twist",threshold=.15,offset=4.,asset_cfg=None):
    """Read physical joint positions; encoder bias belongs only to observation."""
    asset=env.scene["robot" if asset_cfg is None else asset_cfg.name]
    return hiking_stand_still_term(asset.data.joint_pos,asset.data.default_joint_pos,
        env.command_manager.get_command(command_name),threshold=threshold,offset=offset)


def get_terrain_oracle(env):
    return TerrainHeightField.from_env(env)


class PerceptiveDelayedTerminationManager(DelayedTerminationManager):
    """Keep the 250-step rule but never carry its clock into a new episode."""
    def reset(self,env_ids=None):
        extras=super().reset(env_ids)
        self._delay_counters[slice(None) if env_ids is None else env_ids]=0
        return extras


def root_pose_from_asset(asset):
    """Read the true generalized root pose, avoiding pre-forward derived lag."""
    indexing=getattr(asset,"indexing",None)
    if indexing is not None and hasattr(indexing,"free_joint_q_adr"):
        return asset.data.data.qpos[:,indexing.free_joint_q_adr.long()]
    # Pure-data fixtures and supported external wrappers may expose only pose
    # components; production G1 entities always take the generalized path.
    data=asset.data
    position=(data.root_link_pos_w if hasattr(data,"root_link_pos_w")
              else data.body_link_pos_w[:,0])
    quaternion=(data.root_link_quat_w if hasattr(data,"root_link_quat_w")
                else data.body_link_quat_w[:,0])
    return torch.cat((position,quaternion),-1)


def _adapter(env):
    adapter=getattr(env,"adapter",None)
    if adapter is None:
        adapter=getattr(env,"_perceptive_real_adapter",None)
        if adapter is None:
            adapter=StateAdapter(env)
            env._perceptive_real_adapter=adapter
    return adapter


def terrain_relative_amp_features_from_state(adapter,state,oracle,kinematics=None):
    p,q,v,w=adapter.kinematics(state) if kinematics is None else kinematics
    result=build_amp_features(p,q,v,w,body_ids=adapter.amp_ids,anchor_id=adapter.anchor_id,
                              pelvis_id=adapter.pelvis_id,foot_ids=adapter.foot_ids,schema=adapter.amp_feature_schema)
    if adapter.amp_feature_schema=="legacy":return result
    offsets=p.new_tensor((G1_FOOT_REFERENCE_OFFSET,G1_FOOT_REFERENCE_OFFSET))
    feet=p[...,adapter.foot_ids,:]+quat_apply(q[...,adapter.foot_ids,:],offsets)
    pelvis_ground=oracle.height_at(p[...,adapter.pelvis_id,:2])
    foot_ground=oracle.height_at(feet[...,:2])
    return torch.cat((result[...,:-5],result[...,-5:-4]-pelvis_ground[...,None],
                      result[...,-4:-2]-foot_ground,result[...,-2:]),-1)


def terrain_relative_amp_observation(env,**unused):
    adapter=_adapter(env)
    return terrain_relative_amp_features_from_state(adapter,adapter.state(),get_terrain_oracle(env))


def terrain_relative_critic_observation(env):
    adapter=_adapter(env);state=adapter.state()
    value=StateAdapter.critic_features(adapter,state,env.action_manager.action,adapter.command(),adapter.context())
    return torch.cat((value[:,:1]-get_terrain_oracle(env).height_at(state[:,:2])[:,None],value[:,1:]),-1)


def terrain_recovery_height_reward(env,std,mask_delay=False,delay_env_rew_ratio=1.,asset_cfg=None,upright_power=0.):
    asset=env.scene["robot" if asset_cfg is None else asset_cfg.name]
    pose=root_pose_from_asset(asset);position=pose[:,:3]
    height=position[:,2]-get_terrain_oracle(env).height_at(position[:,:2])
    up=recovery_root_up_z(pose[:,3:7]) if upright_power else None
    reward=recovery_height_reward(height,asset.data.default_root_state[:,2],std,up_z=up,upright_power=upright_power)
    return _apply_delay_env_reward_mask_only(env,reward,mask_delay,delay_env_rew_ratio)


def terrain_root_height_below_minimum(env,minimum_height,asset_cfg=None):
    asset=env.scene["robot" if asset_cfg is None else asset_cfg.name]
    position=root_pose_from_asset(asset)[:,:3]
    return position[:,2]-get_terrain_oracle(env).height_at(position[:,:2])<minimum_height


def terrain_bad_orientation(env,limit_angle,asset_cfg=None):
    """Same strict qpos/acos predicate as hard_forward_v2 imagination.

    Deliberately use StateAdapter's quaternion routines and its exact acos
    expression, rather than changing clamp/boundary behavior independently.
    """
    asset=env.scene["robot" if asset_cfg is None else asset_cfg.name]
    quaternion=root_pose_from_asset(asset)[:,3:7]
    gravity=quat_apply(quat_inv(quaternion),quaternion.new_tensor((0.,0.,-1.)))
    return torch.acos(-gravity[:,2]).abs()>limit_angle


def outside_training_patch(env,margin=.1):
    position=root_pose_from_asset(env.scene["robot"])[:,:2]
    center=env.scene.env_origins[:,:2]
    return ((position-center).abs()>4.-margin).any(-1)


def reset_on_flat_patch(env,env_ids,motion_dir,asset_cfg=None):
    """Reuse original motion pose/velocity sampling on a clear terrain pad.

    Motion paths remain the user's walk/recovery data. The base reset places
    relative motion root height above env_origin.z, the sampled spawn surface.
    This wrapper uses a manager local to the environment (no singleton cache
    shared with the untouched flat task).
    """
    manager=env._perceptive_motion_reset_manager
    kwargs={} if asset_cfg is None else {"asset_cfg":asset_cfg}
    manager.reset(env,env_ids,motion_dir,**kwargs)


def init_perceptive_motion_loader(env,env_ids,motion_dir,recovery_dir=None,
                                  delay_reset_env_ratio=0.,max_delay_steps=0,
                                  motion_scope="forward_walk_turn_v1"):
    if motion_scope not in ("forward_walk_turn_v1","joint_recovery_v1"):
        raise ValueError(f"Unsupported perceptive motion scope: {motion_scope}")
    if motion_scope=="forward_walk_turn_v1" and (recovery_dir is not None or delay_reset_env_ratio or max_delay_steps):
        raise ValueError("Walking-only motion initialization cannot load Recovery data or delay falls")
    from src.tasks.amp_loco.mdp.events import MotionResetManager
    manager=MotionResetManager();manager.init(env,motion_dir,recovery_dir)
    env._perceptive_motion_reset_manager=manager
    count=int(env.num_envs*delay_reset_env_ratio)
    if count and max_delay_steps>0:
        mask=torch.zeros(env.num_envs,device=env.device,dtype=torch.bool)
        mask[torch.randperm(env.num_envs,device=env.device)[:count]]=True
        env.termination_manager=PerceptiveDelayedTerminationManager(env.termination_manager,mask,max_delay_steps)
