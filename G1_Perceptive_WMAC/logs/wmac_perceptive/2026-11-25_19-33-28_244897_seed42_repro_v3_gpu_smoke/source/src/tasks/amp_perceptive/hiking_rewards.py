"""Hiking's 26 reward formulas, shared by physics and GenCoord imagination.

Reference: InstinctLab de6b92f, parkour/config/parkour_env_cfg.py (plain G1,
without the separate shoe mixin), and IsaacLab f73c331738.  Values returned
here are unweighted; the reward manager or ``hiking_weighted_reward`` applies
the listed weights and the control-step duration exactly once.

Inputs are physical values, never biased encoder observations. Unavailable
physics quantities are explicit supervised auxiliary labels; none are silently
replaced by zeros or a finite-difference/kinematic proxy.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from mjlab.sensor.contact_sensor import ContactSensor, ContactSensorCfg


HIKING_REWARD_WEIGHTS = {
    "track_lin_vel_xy_exp": 2., "track_ang_vel_z_exp": 2.,
    "heading_error": -1., "dont_wait": -.5, "is_alive": 3.,
    "stand_still": -.3, "volume_points_penetration": -4.,
    "feet_air_time": .5, "feet_slide": -.4, "joint_deviation_hip": -.5,
    "ang_vel_xy_l2": -.05, "dof_torques_l2": -1.5e-7,
    "dof_acc_l2": -1.25e-7, "dof_vel_l2": -1e-4,
    "action_rate_l2": -.005, "flat_orientation_l2": -3.,
    "pelvis_orientation_l2": -3., "feet_flat_ori": -.4,
    "feet_at_plane": -.1, "feet_close_xy": .4, "energy": -5e-5,
    "freeze_upper_body": -.004, "dof_pos_limits": -1.,
    "dof_vel_limits": -1., "torque_limits": -.01,
    "undesired_contacts": -1.,
}

# The first nine slots are reward/termination labels. Four next-state sensor
# timers are additional WM targets, carried into the next imagined context.
HIKING_AUX_NAMES = (
    "dof_torques_l2", "dof_acc_l2", "energy", "torque_limits",
    "undesired_contacts", "torso_contact_force_max",
    "left_foot_contact_force_max", "right_foot_contact_force_max",
    "feet_air_time_ungated", "left_current_air_time", "right_current_air_time",
    "left_current_contact_time", "right_current_contact_time",
)


def hiking_air_time_ungated(air_time, contact_time):
    """Original single-stance minimum; command gating is applied separately."""
    in_contact = contact_time > 0.
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = in_contact.int().sum(-1) == 1
    return torch.where(single_stance[..., None], in_mode_time, 0.).min(-1).values


def hiking_volume_penetration(penetration_offset, points_velocity):
    depth = torch.linalg.vector_norm(penetration_offset, dim=-1)
    speed = torch.linalg.vector_norm(points_velocity, dim=-1)
    return ((depth > 0.) * (speed + 1e-6) * depth).flatten(-2).sum(-1)


def hiking_auxiliary_from_measurements(*, applied_torque, joint_velocity,
        joint_acceleration, stiffness, effort_limits, leg_joint_ids,
        contact_force_max, foot_body_ids, torso_body_id, current_air_time,
        current_contact_time):
    """Pack true post-step labels; contact forces already max over 3 substeps."""
    torque_cost = applied_torque[..., leg_joint_ids].square().sum(-1)
    acceleration_cost = joint_acceleration.square().sum(-1)
    power = applied_torque * joint_velocity / stiffness
    energy = power[..., leg_joint_ids].square().sum(-1)
    torque_limit_cost = (applied_torque.abs() - .8 * effort_limits).clamp_min(0.).square().sum(-1)
    foot_forces = contact_force_max[..., foot_body_ids]
    # Exclude exactly ankle-roll bodies, not their ancestor subtrees.
    undesired = (contact_force_max > 1.).sum(-1) - (foot_forces > 1.).sum(-1)
    air = current_air_time[..., foot_body_ids]
    contact = current_contact_time[..., foot_body_ids]
    return torch.stack((torque_cost, acceleration_cost, energy, torque_limit_cost,
        undesired.to(applied_torque.dtype), contact_force_max[..., torso_body_id],
        foot_forces[..., 0], foot_forces[..., 1], hiking_air_time_ungated(air, contact),
        air[..., 0], air[..., 1], contact[..., 0], contact[..., 1]), -1)


def hiking_reward_terms(inputs: Mapping[str, torch.Tensor], auxiliary: torch.Tensor):
    """All 26 unweighted terms; required keys are deliberately indexed strictly.

    ``foot_ground_height`` is [*,2,2] from the two original downward scanner
    rays per foot, and ``volume_points_penetration`` is the original virtual
    edge-cylinder cost. ``foot_linear_velocity_w`` is COM velocity, whereas
    ``foot_position_w`` is link-origin position, matching IsaacLab body data.
    """
    command = inputs["command"]
    velocity = inputs["torso_linear_velocity_b"]
    omega = inputs["torso_angular_velocity_b"]
    q = inputs["joint_position"]
    qdefault = inputs["default_joint_position"]
    dq = inputs["joint_velocity"]
    contact = auxiliary[..., 6:8] > 1.
    moving = (torch.linalg.vector_norm(command[..., :2], dim=-1) > .15) | (command[..., 2].abs() > .15)
    standing = (torch.linalg.vector_norm(command[..., :2], dim=-1) < .15) & (command[..., 2].abs() < .15)
    feet = inputs["foot_position_w"]
    heading = inputs["torso_heading"]
    # Only the difference is required; translating both feet before rotating
    # has no effect and is algebraically identical to the upstream expression.
    separation = feet[..., 0, :2] - feet[..., 1, :2]
    distance_y = (-heading.sin() * separation[..., 0] + heading.cos() * separation[..., 1]).abs()
    hits = inputs["foot_ground_height"]
    hits = torch.where(torch.isinf(hits), 0., hits)
    plane = (feet[..., :, 2, None] - hits - .035).clamp(0., .3).sum(-1)
    limits = inputs["joint_position_limits"]
    deviation = q - qdefault
    raw = {
        "track_lin_vel_xy_exp": torch.exp(-(command[..., :2] - velocity[..., :2]).square().sum(-1) / .25),
        "track_ang_vel_z_exp": torch.exp(-(command[..., 2] - omega[..., 2]).square() / .25),
        "heading_error": command[..., 2].abs(),
        "dont_wait": (command[..., 0] > .3) * ((velocity[..., 0] < .15).to(q.dtype)
            + (velocity[..., 0] < 0.).to(q.dtype) + (velocity[..., 0] < -.15).to(q.dtype)),
        "is_alive": 1. - inputs["terminated"].to(q.dtype),
        "stand_still": (deviation.abs().sum(-1) - 4.) * standing,
        "volume_points_penetration": inputs["volume_points_penetration"],
        "feet_air_time": auxiliary[..., 8] * moving,
        "feet_slide": (torch.linalg.vector_norm(inputs["foot_linear_velocity_w"][..., :2], dim=-1) * contact).sum(-1),
        "joint_deviation_hip": deviation[..., inputs["hip_joint_ids"]].square().sum(-1),
        "ang_vel_xy_l2": omega[..., :2].square().sum(-1),
        "dof_torques_l2": auxiliary[..., 0],
        "dof_acc_l2": auxiliary[..., 1],
        "dof_vel_l2": dq.square().sum(-1),
        "action_rate_l2": (inputs["action"] - inputs["previous_action"]).square().sum(-1),
        "flat_orientation_l2": inputs["torso_projected_gravity"][..., :2].square().sum(-1),
        "pelvis_orientation_l2": inputs["pelvis_projected_gravity"][..., :2].square().sum(-1),
        "feet_flat_ori": (torch.linalg.vector_norm(inputs["foot_projected_gravity"][..., :2], dim=-1) * contact).sum(-1),
        "feet_at_plane": (plane * contact).sum(-1),
        "feet_close_xy": torch.exp(-(.12 - distance_y).clamp_min(0.) / .05) - 1.,
        "energy": auxiliary[..., 2],
        "freeze_upper_body": deviation[..., inputs["upper_body_joint_ids"]].abs().sum(-1),
        "dof_pos_limits": ((limits[..., 0] - q).clamp_min(0.) + (q - limits[..., 1]).clamp_min(0.)).sum(-1),
        "dof_vel_limits": (dq.abs() - .9 * inputs["joint_velocity_limits"]).clamp(0., 1.).sum(-1),
        "torque_limits": auxiliary[..., 3],
        "undesired_contacts": auxiliary[..., 4],
    }
    return raw


def hiking_weighted_reward(inputs, auxiliary, reward_scale=1.):
    terms = hiking_reward_terms(inputs, auxiliary)
    return sum(terms[name] * weight for name, weight in HIKING_REWARD_WEIGHTS.items()) * reward_scale


def hiking_termination_terms(torso_position_w, torso_projected_gravity, env_origins, torso_contact_force):
    """Original three physical failures; dataset/time/terrain timeouts are separate."""
    return {
        "base_contact": torso_contact_force > 1.,
        "bad_orientation": torch.acos(-torso_projected_gravity[..., 2]).abs() > 1.,
        "root_height": torso_position_w[..., 2] - env_origins[..., 2].clamp_max(0.) < .5,
    }


def hiking_terrain_out_of_bounds(torso_position_w, map_half_extent, distance_buffer=2.):
    return (torso_position_w[..., :2].abs() > map_half_extent - distance_buffer).any(-1)


class HikingContactSensor(ContactSensor):
    """Keep mjlab net-force history, use IsaacLab's 1 N timer contact predicate."""

    def _update_air_time_tracking(self):
        state = self._air_time_state
        assert state is not None and self._data is not None
        contact_data = self._extract_sensor_data()
        if contact_data.force is None:
            raise RuntimeError("Hiking contact timers require net world forces")
        current_time = self._data.time
        elapsed_time = (current_time - state.last_time).unsqueeze(-1)
        in_contact = torch.linalg.vector_norm(contact_data.force, dim=-1) > 1.
        first_contact = (state.current_air_time > 0.) & in_contact
        first_detached = (state.current_contact_time > 0.) & ~in_contact
        state.last_air_time[:] = torch.where(first_contact, state.current_air_time + elapsed_time, state.last_air_time)
        state.current_air_time[:] = torch.where(~in_contact, state.current_air_time + elapsed_time, 0.)
        state.last_contact_time[:] = torch.where(first_detached, state.current_contact_time + elapsed_time, state.last_contact_time)
        state.current_contact_time[:] = torch.where(in_contact, state.current_contact_time + elapsed_time, 0.)
        state.last_time[:] = current_time


@dataclass
class HikingContactSensorCfg(ContactSensorCfg):
    def build(self):
        if self.reduce != "netforce" or self.history_length != 3 or not self.track_air_time:
            raise ValueError("Hiking requires netforce, three physics-step history slots, and air/contact timers")
        return HikingContactSensor(self)


def _hiking_contact_layout(env):
    cached = getattr(env, "_hiking_contact_layout", None)
    if cached is None:
        sensor = env.scene["hiking_contact"]
        names = tuple(dict.fromkeys(slot.primary_name.split("/")[-1] for slot in sensor._slots))
        feet = tuple(names.index(f"{side}_ankle_roll_link") for side in ("left", "right"))
        cached = (feet, names.index("torso_link"))
        env._hiking_contact_layout = cached
    return cached


def hiking_airtime_state(env):
    feet, _ = _hiking_contact_layout(env)
    data = env.scene["hiking_contact"].data
    return torch.cat((data.current_air_time[..., feet], data.current_contact_time[..., feet]), -1)


def hiking_auxiliary_targets(env):
    """True labels captured after physics and before resetting terminated envs."""
    from src.wmac_perceptive.hiking_actuators import read_hiking_actuator_parameters
    robot = env.scene["robot"]
    params = read_hiking_actuator_parameters(robot)
    leg_ids = [i for i, name in enumerate(robot.joint_names)
               if any(part in name for part in ("_hip_", "_knee_", "_ankle_"))]
    feet, torso = _hiking_contact_layout(env)
    data = env.scene["hiking_contact"].data
    force_max = torch.linalg.vector_norm(data.force_history, dim=-1).max(-1).values
    return hiking_auxiliary_from_measurements(
        applied_torque=robot.data.data.qfrc_actuator[:, robot.indexing.joint_v_adr.long()],
        joint_velocity=robot.data.joint_vel, joint_acceleration=robot.data.joint_acc,
        stiffness=params["kp"], effort_limits=params["effort_limit"],
        leg_joint_ids=leg_ids, contact_force_max=force_max,
        foot_body_ids=feet, torso_body_id=torso, current_air_time=data.current_air_time,
        current_contact_time=data.current_contact_time)


def hiking_real_reward(env, term_name):
    """RewardManager wrapper; expensive FK/geometry is evaluated once per step."""
    from src.wmac_perceptive.hiking_rewards import hiking_real_reward_terms
    return hiking_real_reward_terms(env)[term_name]


def hiking_real_termination(env, term_name):
    from src.wmac_perceptive.hiking_rewards import hiking_real_termination_terms
    return hiking_real_termination_terms(env)[term_name]


def initialize_hiking_reward_adapter(adapter):
    from src.wmac_perceptive.hiking_rewards import initialize_hiking_reward_adapter as initialize
    return initialize(adapter)


def hiking_reward_inputs(adapter, state, next_state, action, previous_action,
                         command, next_context, terminated, kinematics=None):
    from src.wmac_perceptive.hiking_rewards import hiking_reward_inputs as build
    return build(adapter, state, next_state, action, previous_action, command,
                 next_context, terminated, kinematics=kinematics)
