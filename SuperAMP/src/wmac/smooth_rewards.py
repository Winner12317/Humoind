"""Parameter-free smoothness costs for the existing position-action contract.

HoST contributes the second action difference. HumanUP contributes the joint
velocity, torque-norm and power-norm ideas. This adapter retains this project's
PD, motor curves, action scale and world model. Torque cost uses pre-clipping
PD request; power uses motor-clipped, friction-adjusted commanded effort. Both
are evaluated at transition start and are not simulated actuator_force.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json

import torch
from torch import nn

from .state import joint_positions, joint_velocities


TERM_NAMES = ("action_second_difference", "joint_velocity", "requested_torque", "commanded_power")
CONSTANT_NAMES = ("action_scale", "action_offset", "kp", "kd", "effort_limit",
                  "y1", "y2", "x1", "x2", "friction_static", "friction_dynamic",
                  "activation_velocity")
MOTOR_FORMULA = "g1_unitree_commanded_effort_control_start_v1"


def _static_vector(value, size: int, *, name: str) -> torch.Tensor:
    """Copy a uniform scalar/per-joint/per-env constant into a canonical vector."""
    x = torch.as_tensor(value).detach().to(dtype=torch.float32, device="cpu")
    if x.ndim == 0:
        x = x.expand(size)
    elif x.ndim == 2:
        if x.shape[1] != size or not torch.equal(x, x[:1].expand_as(x)):
            raise ValueError(f"Smooth reward cannot encode varying per-environment {name}")
        x = x[0]
    if x.shape != (size,) or not bool(torch.isfinite(x).all()):
        raise ValueError(f"Smooth reward {name} must be a finite static {size}-vector")
    return x.contiguous().clone()


class SmoothRewardTerms(nn.Module):
    """Differentiable HoST/HumanUP costs, without learned parameters.

    Raw cost units are action², (rad/s)², Nm and W respectively. The caller
    applies each nonnegative coefficient exactly once and scales by task dt.
    """
    def __init__(self, constants: Mapping, *, joint_names: Sequence[str] = (),
                 action_to_joint: Sequence[int] | torch.Tensor | None = None,
                 num_joints: int | None = None):
        super().__init__()
        if set(constants) != set(CONSTANT_NAMES):
            raise ValueError("Smooth reward motor constants are incomplete or contain unknown fields")
        if num_joints is None:
            # Per-joint vectors give the joint count; all-scalar constants keep the G1's 29.
            widths = [int(torch.as_tensor(value).shape[-1]) for value in constants.values()
                      if torch.as_tensor(value).ndim]
            num_joints = widths[0] if widths else 29
        count = self.num_joints = int(num_joints)
        for name in CONSTANT_NAMES:
            self.register_buffer(name, _static_vector(constants[name], count, name=name))
        for name in ("kp", "effort_limit", "x1", "x2", "activation_velocity"):
            if not bool((getattr(self, name) > 0).all()):
                raise ValueError(f"Smooth reward {name} must be strictly positive")
        for name in ("kd", "y1", "y2", "friction_static", "friction_dynamic"):
            if not bool((getattr(self, name) >= 0).all()):
                raise ValueError(f"Smooth reward {name} must be nonnegative")
        if not bool((self.x2 >= self.x1).all()):
            raise ValueError("Smooth reward motor no-load speed must be >= its knee speed")
        order = torch.arange(count) if action_to_joint is None else torch.as_tensor(action_to_joint, dtype=torch.long)
        if order.shape != (count,) or sorted(order.tolist()) != list(range(count)):
            raise ValueError(f"Smooth reward action/joint mapping must be a complete {count}-joint permutation")
        self.register_buffer("action_to_joint", order.clone())
        self._identity_action_order = bool(torch.equal(order, torch.arange(count)))
        self.joint_names = tuple(joint_names)
        if self.joint_names and len(self.joint_names) != count:
            raise ValueError(f"Smooth reward joint name metadata must have {count} entries")

    @classmethod
    def from_env(cls, env):
        """Read actual initialized action/actuator constants; reject unsupported controls."""
        from mjlab.envs.mdp.actions.actions import JointPositionAction
        from src.assets.robots.unitree_g1.unitree_actuators import UnitreeActuator
        robot = env.scene["robot"]
        terms = list(env.action_manager._terms.values())
        if len(terms) != 1 or not isinstance(terms[0], JointPositionAction):
            raise ValueError("Smooth rewards require the existing single G1 JointPositionAction")
        term = terms[0]
        count = len(robot.joint_names)
        if term.action_dim != count or term.cfg.clip is not None:
            raise ValueError(f"Smooth rewards require {count} position actions without a changed action-term clip")
        ids = term._target_ids.detach().cpu().long()
        if sorted(ids.tolist()) != list(range(count)):
            raise ValueError(f"Smooth rewards require all {count} joints")
        inverse = ids.argsort()
        constants = {name: torch.zeros(count) for name in CONSTANT_NAMES}
        constants["action_scale"] = _static_vector(term.scale, count, name="action_scale")[inverse]
        constants["action_offset"] = _static_vector(term.offset, count, name="action_offset")[inverse]
        if bool((robot.data.joint_vel_target != 0).any()) or bool((robot.data.joint_effort_target != 0).any()):
            raise ValueError("Smooth reward proxy requires the original zero velocity/feedforward targets")
        seen = []
        for actuator in robot.actuators:
            if not isinstance(actuator, UnitreeActuator):
                raise ValueError("Smooth rewards require the existing un-delayed Unitree actuator")
            joint_ids = actuator.target_ids.detach().cpu().long()
            seen.extend(joint_ids.tolist())
            values = {
                "kp": actuator.cfg.stiffness, "kd": actuator.cfg.damping,
                "effort_limit": actuator.cfg.effort_limit,
                "y1": actuator._effort_y1, "y2": actuator._effort_y2,
                "x1": actuator._velocity_x1, "x2": actuator._velocity_x2,
                "friction_static": actuator._friction_static,
                "friction_dynamic": actuator._friction_dynamic,
                "activation_velocity": actuator._activation_vel,
            }
            for name, value in values.items():
                constants[name][joint_ids] = _static_vector(value, len(joint_ids), name=name)
        if sorted(seen) != list(range(count)):
            raise ValueError("Smooth reward actuator mapping has missing or duplicated joints")
        return cls(constants, joint_names=robot.joint_names, action_to_joint=inverse).to(env.device)

    @classmethod
    def from_config(cls, env_cfg, joint_names: Sequence[str], default_joint_pos=None):
        """CPU-only equivalent for offline calibration; actual training uses from_env."""
        import re
        from mjlab.envs.mdp.actions.actions import JointPositionActionCfg
        from mjlab.utils.lab_api.string import resolve_matching_names_values
        from src.assets.robots.unitree_g1.unitree_actuators import UnitreeActuatorCfg
        actions = list(env_cfg.actions.values())
        if len(actions) != 1 or not isinstance(actions[0], JointPositionActionCfg):
            raise ValueError("Smooth rewards require one JointPositionActionCfg")
        action = actions[0]
        if action.clip is not None or tuple(action.actuator_names) != (".*",):
            raise ValueError("Offline smooth calibration supports the original all-joint action config")
        names = list(joint_names)
        count = len(names)
        if count < 1:
            raise ValueError("Expected the robot's canonical joint names")
        constants = {name: torch.zeros(count) for name in CONSTANT_NAMES}
        def affine(spec, default):
            if isinstance(spec, (float, int)):
                return torch.full((count,), float(spec))
            result = torch.full((count,), default)
            ids, _, values = resolve_matching_names_values(spec, names)
            result[ids] = torch.tensor(values, dtype=torch.float32)
            return result
        if default_joint_pos is None:
            default_joint_pos = affine(env_cfg.scene.entities["robot"].init_state.joint_pos, 0.)
        constants["action_scale"] = affine(action.scale, 1.)
        constants["action_offset"] = (_static_vector(default_joint_pos, count, name="action_offset")
                                      if action.use_default_offset else affine(action.offset, 0.))
        covered = []
        for cfg in env_cfg.scene.entities["robot"].articulation.actuators:
            if not isinstance(cfg, UnitreeActuatorCfg):
                raise ValueError("Unsupported actuator in offline smooth calibration")
            ids = [i for i, name in enumerate(names) if any(re.fullmatch(pattern, name) for pattern in cfg.target_names_expr)]
            covered.extend(ids)
            values = dict(kp=cfg.stiffness, kd=cfg.damping, effort_limit=cfg.effort_limit,
                          y1=cfg.Y1, y2=cfg.Y1 if cfg.Y2 is None else cfg.Y2,
                          x1=cfg.X1, x2=cfg.X2, friction_static=cfg.Fs,
                          friction_dynamic=cfg.Fd, activation_velocity=cfg.Va)
            for name, value in values.items():
                constants[name][ids] = _static_vector(value, len(ids), name=name)
        if sorted(covered) != list(range(count)):
            raise ValueError("Offline smooth actuator mapping is incomplete or duplicated")
        return cls(constants, joint_names=names)

    def motor_effort(self, state, action, context):
        """PD request, motor-clipped request and friction-adjusted command.

        Evaluated before integration, using action target minus encoder bias.
        Physical substeps and implicit position integration can subsequently
        differ; the returned command is explicitly not actual actuator_force.
        """
        if not self._identity_action_order:
            action = action.index_select(-1, self.action_to_joint)
        velocity = joint_velocities(state)
        # context[..., 1:1+n] is the encoder bias (n joints).
        target = self.action_offset + self.action_scale * action - context[..., 1:1 + self.num_joints]
        request = self.kp * (target - joint_positions(state)) - self.kd * velocity
        same_direction = velocity * request > 0
        bound = torch.where(same_direction, self.y1, self.y2).clamp_max(self.effort_limit)
        taper = (-bound / (self.x2 - self.x1).clamp_min(1e-6)
                 * (velocity.abs() - self.x1) + bound).clamp_min(0)
        bound = torch.where(velocity.abs() < self.x1, bound, taper)
        clipped = torch.maximum(torch.minimum(request, bound), -bound)
        command = clipped - (self.friction_static * torch.tanh(velocity / self.activation_velocity)
                             + self.friction_dynamic * velocity)
        return request, clipped, command

    def forward(self, state, next_state, action, previous, previous_previous, context):
        request, _, effort = self.motor_effort(state, action, context)
        return {
            "action_second_difference": (action - 2 * previous + previous_previous).square().sum(-1),
            "joint_velocity": joint_velocities(next_state).square().sum(-1),
            # A pre-clipping load cost keeps the genuine PD derivative when
            # the physical motor saturates; no straight-through derivative.
            "requested_torque": torch.linalg.vector_norm(request, dim=-1),
            "commanded_power": torch.linalg.vector_norm((effort * joint_velocities(state)).abs(), dim=-1),
        }

    def metadata(self) -> dict:
        result = {
            "formula": MOTOR_FORMULA,
            "joint_names": list(self.joint_names),
            "action_to_joint": self.action_to_joint.detach().cpu().tolist(),
            "constants": {name: getattr(self, name).detach().cpu().tolist() for name in CONSTANT_NAMES},
            "torque_semantics": "control_start_preclip_PD_request_not_actual_actuator_force",
            "power_semantics": "control_start_motor_clipped_friction_adjusted_command_proxy",
            "term_definitions": {
                "action_second_difference": "sum((a_t-2*a_prev+a_prevprev)^2)",
                "joint_velocity": "sum(next_joint_velocity^2)",
                "requested_torque": "L2(preclip_PD_effort_request)",
                "commanded_power": "L2(abs(commanded_effort_proxy*current_joint_velocity))",
            },
        }
        result["sha256"] = hashlib.sha256(json.dumps(result, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        return result
