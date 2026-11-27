"""Physical-state assembly for the shared Hiking objective.

The GenCoord state remains unchanged. Contact, actuator and acceleration labels
are supervised auxiliary predictions, and the four contact timers are carried
in the full-objective context. Geometry remains an explicit terrain query.
"""
from __future__ import annotations

import torch

from src.wmac.state import quat_apply, quat_inv, heading_quat
from src.tasks.amp_perceptive.hiking_rewards import (
    hiking_auxiliary_targets, hiking_reward_terms, hiking_termination_terms,
    hiking_terrain_out_of_bounds, _hiking_contact_layout,
)


def initialize_hiking_reward_adapter(adapter):
    from .hiking_actuators import get_hiking_joint_parameters
    names = tuple(adapter.robot.joint_names)
    adapter.hiking_hip_joint_ids = tuple(i for i, name in enumerate(names)
        if "_hip_yaw_joint" in name or "_hip_roll_joint" in name)
    adapter.hiking_upper_body_joint_ids = tuple(i for i, name in enumerate(names)
        if any(part in name for part in ("_shoulder_", "_elbow_", "_wrist", "waist_")))
    parameters = get_hiking_joint_parameters(names, device=adapter.device)
    adapter.hiking_joint_velocity_limits = parameters["velocity_limit"]
    model = adapter.env.sim.mj_model
    adapter.hiking_foot_com_offsets = torch.as_tensor(
        model.body_ipos[adapter.body_ids.detach().cpu().numpy()[list(adapter.foot_ids)]].copy(),
        dtype=adapter.default_joint_pos.dtype, device=adapter.device)
    adapter.hiking_foot_scanner_starts = adapter.default_joint_pos.new_tensor(
        [[-.02, 0., 20.], [.10, 0., 20.]])
    # Unshod or shod foot measurements (ShoeConfigMixin lowers both by 23 mm).
    from src.tasks.amp_perceptive.hiking_robot import hiking_foot_geometry
    from .safety import hiking_foot_point_grid
    geometry = hiking_foot_geometry(
        getattr(getattr(adapter.env, "cfg", None), "_hiking_robot_variant", "unshod"))
    adapter.hiking_foot_points = hiking_foot_point_grid(
        geometry["volume_points_z"], device=adapter.device, dtype=adapter.default_joint_pos.dtype)
    adapter.hiking_foot_plane_offset = geometry["plane_height_offset"]


def hiking_foot_ground_height(kinematics, foot_ids, terrain, scanner_starts):
    position, quaternion, _, _ = kinematics
    starts = position[..., foot_ids, None, :] + quat_apply(
        heading_quat(quaternion[..., foot_ids, None, :]), scanner_starts)
    # A bilinear height raster is not the original ray/triangle intersection.
    # Require the precise full-objective implementation instead of a fallback.
    return terrain.hiking_ray_height_at(starts)


def hiking_reward_inputs(adapter, state, next_state, action, previous_action,
                         command, next_context, terminated, kinematics=None):
    del state
    from .safety import hiking_foot_edge_cost
    fk = adapter.kinematics(next_state) if kinematics is None else kinematics
    position, quaternion, velocity, omega = fk
    torso, pelvis, feet = adapter.anchor_id, adapter.pelvis_id, adapter.foot_ids
    tq = quaternion[..., torso, :]
    torso_com = adapter.default_torso_com + next_context[..., 30:33]
    torso_velocity = velocity[..., torso, :] + torch.cross(omega[..., torso, :], quat_apply(tq, torso_com), dim=-1)
    foot_offset = quat_apply(quaternion[..., feet, :], adapter.hiking_foot_com_offsets)
    foot_velocity = velocity[..., feet, :] + torch.cross(omega[..., feet, :], foot_offset, dim=-1)
    gravity = next_state.new_tensor((0., 0., -1.))
    heading = heading_quat(tq)
    return {
        "command": command,
        "torso_linear_velocity_b": quat_apply(quat_inv(tq), torso_velocity),
        "torso_angular_velocity_b": quat_apply(quat_inv(tq), omega[..., torso, :]),
        "torso_projected_gravity": quat_apply(quat_inv(tq), gravity),
        "pelvis_projected_gravity": quat_apply(quat_inv(quaternion[..., pelvis, :]), gravity),
        "foot_projected_gravity": quat_apply(quat_inv(quaternion[..., feet, :]), gravity),
        "torso_heading": 2. * torch.atan2(heading[..., 3], heading[..., 0]),
        "foot_position_w": position[..., feet, :],
        "foot_linear_velocity_w": foot_velocity,
        "foot_ground_height": hiking_foot_ground_height(fk, feet, adapter.terrain, adapter.hiking_foot_scanner_starts),
        # Upstream VolumePoints starts from PhysX COM velocity, then adds the
        # cross product about the LINK origin. Keep that exact source formula.
        "volume_points_penetration": hiking_foot_edge_cost(fk, feet, adapter.terrain,
            foot_points=getattr(adapter, "hiking_foot_points", None), foot_linear_velocity=foot_velocity),
        "foot_plane_height_offset": getattr(adapter, "hiking_foot_plane_offset", .035),
        "joint_position": next_state[..., 13:42],
        "default_joint_position": adapter.default_joint_pos,
        "joint_velocity": next_state[..., 42:71],
        "joint_position_limits": adapter.joint_limits,
        "joint_velocity_limits": adapter.hiking_joint_velocity_limits,
        "hip_joint_ids": adapter.hiking_hip_joint_ids,
        "upper_body_joint_ids": adapter.hiking_upper_body_joint_ids,
        "action": action, "previous_action": previous_action,
        "terminated": terminated,
    }


def _real_adapter(env):
    adapter = getattr(env, "adapter", None)
    if adapter is None:
        adapter = getattr(env, "_hiking_reward_adapter", None)
    if adapter is None:
        from .state import PerceptiveStateAdapter
        adapter = PerceptiveStateAdapter(env, motion_scope="forward_walk_turn_v1", amp_feature_schema="hiking_v1")
        env._hiking_reward_adapter = adapter
    if not hasattr(adapter, "hiking_hip_joint_ids"):
        initialize_hiking_reward_adapter(adapter)
    return adapter


def hiking_real_reward_terms(env):
    step = env.common_step_counter
    cached = getattr(env, "_hiking_reward_cache", None)
    if cached is not None and cached[0] == step:
        return cached[1]
    adapter = _real_adapter(env)
    state = adapter.state()
    inputs = hiking_reward_inputs(adapter, state, state, env.action_manager.action,
        env.action_manager.prev_action, env.command_manager.get_command("twist"),
        adapter.context(), env.termination_manager.terminated)
    result = hiking_reward_terms(inputs, hiking_auxiliary_targets(env))
    env._hiking_reward_cache = (step, result)
    return result


def hiking_real_termination_terms(env):
    step=env.common_step_counter
    cached=getattr(env,"_hiking_termination_cache",None)
    if cached is not None and cached[0]==step:
        return cached[1]
    adapter = _real_adapter(env)
    state = adapter.state()
    position, quaternion, _, _ = adapter.kinematics(state)
    torso = adapter.anchor_id
    _, torso_sensor = _hiking_contact_layout(env)
    data = env.scene["hiking_contact"].data
    force = torch.linalg.vector_norm(data.force_history[..., torso_sensor, :, :], dim=-1).max(-1).values
    gravity = quat_apply(quat_inv(quaternion[..., torso, :]), state.new_tensor((0., 0., -1.)))
    result = hiking_termination_terms(position[..., torso, :], gravity, env.scene.env_origins, force)
    terrain = env.cfg.scene.terrain
    if terrain.terrain_type == "plane":
        boundary = torch.zeros_like(result["base_contact"])
    else:
        cfg = terrain.terrain_generator
        size = state.new_tensor((cfg.num_rows * cfg.size[0] + 2. * cfg.border_width,
                                 cfg.num_cols * cfg.size[1] + 2. * cfg.border_width)) / 2.
        boundary = hiking_terrain_out_of_bounds(position[..., torso, :], size)
    result["terrain_out_bound"] = boundary
    env._hiking_termination_cache=(step,result)
    return result
