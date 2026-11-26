"""Official SuperMimic GenCoord dynamics, adapted to WMAC's state layout.

The reference is HybridRobotics/supermimic at
``5efbf114e11f725c31d83e217be987c13c0e4ce2`` (Apache-2.0), specifically
``GenCoordRepresentation.apply_prediction`` and ``WorldModelLearner``'s loss.
Only the storage order changes here: WMAC stores
``[p_w(3), quat_wxyz(4), v_w(3), w_w(3), qj(29), vj(29)]``.
Root quaternions must be unit length, as in the simulator and upstream.

Predictions are physical velocity increments over one control step, not
accelerations and not normalized output coordinates. Gravity, actuator forces
and contacts are learned in those increments; the integrator adds no gravity.
"""

from __future__ import annotations

import math
from numbers import Integral, Real

import torch

from .state import quat_mul, quat_to_matrix

GENCOORD_STATE_DIM = 68
GENCOORD_PREDICTION_DIM = 35
GENCOORD_LOSS_DIM = 70
SUPERMIMIC_REFERENCE_COMMIT = "5efbf114e11f725c31d83e217be987c13c0e4ce2"
GENCOORD_NOMINAL_ERRORS = {
    "root_pos": 0.01,
    "root_ori": 0.02,
    "joint_pos": 0.1,
    "joint_vel": 3.14,
}


def _check_state(state: torch.Tensor, name: str = "state") -> None:
    if not isinstance(state, torch.Tensor) or state.ndim < 1 or state.shape[-1] != 71:
        raise ValueError(f"{name} must have shape (..., 71)")
    if not state.is_floating_point():
        raise ValueError(f"{name} must have a floating-point dtype")


def _check_pair(state: torch.Tensor, other: torch.Tensor, name: str, width: int) -> None:
    if (not isinstance(other, torch.Tensor) or other.ndim < 1
            or other.shape != (*state.shape[:-1], width)):
        raise ValueError(f"{name} must have shape {(*state.shape[:-1], width)}")
    if other.dtype != state.dtype or other.device != state.device:
        raise ValueError(f"{name} must share the state's dtype and device")


def _quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Upstream unit-quaternion rotation, without an extra normalization."""
    xyz = q[..., 1:]
    t = 2 * torch.cross(xyz, v, dim=-1)
    return v + q[..., :1] * t + torch.cross(xyz, t, dim=-1)


def _quat_apply_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    xyz = q[..., 1:]
    t = 2 * torch.cross(xyz, v, dim=-1)
    return v - q[..., :1] * t + torch.cross(xyz, t, dim=-1)


def _quat_integrate(q: torch.Tensor, omega_world: torch.Tensor, dt: float) -> torch.Tensor:
    # Preserve upstream's small-angle regularization and world-frame left
    # multiplication, including its finite derivative at exactly zero omega.
    omega = omega_world * dt
    norm_safe = torch.sqrt(omega.square().sum(-1, keepdim=True) + 1e-8)
    half_angle = norm_safe / 2
    delta = torch.cat((torch.cos(half_angle), torch.sin(half_angle) * omega / norm_safe), -1)
    next_q = quat_mul(delta, q)
    return next_q / next_q.norm(dim=-1, keepdim=True)


def gencoord_features(state: torch.Tensor) -> torch.Tensor:
    """Return ``[height, gravity_b, v_b, w_b, qj, vj]`` in 68 dimensions.

    Velocities use the complete current root frame, including roll and pitch.
    Callers may first replace root Z with height relative to local terrain.
    """
    _check_state(state)
    q = state[..., 3:7]
    world_down = torch.zeros_like(state[..., 7:10])
    world_down[..., 2] = -1
    return torch.cat((state[..., 2:3], _quat_apply_inverse(q, world_down),
                      _quat_apply_inverse(q, state[..., 7:10]),
                      _quat_apply_inverse(q, state[..., 10:13]), state[..., 13:]), -1)


def velocity_increment_targets(state: torch.Tensor, next_state: torch.Tensor) -> torch.Tensor:
    """Express measured velocity deltas in the *current* root frame.

    This is useful for diagnostics and integration checks. Upstream trains on
    the integrated state loss below, rather than a direct increment MSE.
    Subtracting successive body velocities without bringing them into the same
    frame would incorrectly include a coordinate-rotation term when turning.
    """
    _check_state(state)
    _check_pair(state, next_state, "next_state", 71)
    q = state[..., 3:7]
    return torch.cat((_quat_apply_inverse(q, next_state[..., 7:10] - state[..., 7:10]),
                      _quat_apply_inverse(q, next_state[..., 10:13] - state[..., 10:13]),
                      next_state[..., 42:71] - state[..., 42:71]), -1)


def integrate_velocity_increment(
    state: torch.Tensor,
    physical_increment: torch.Tensor,
    dt: float,
    decimation: int,
) -> torch.Tensor:
    """Apply the upstream semi-implicit, constant-acceleration integrator.

    For D simulator substeps, pose integration uses
    ``c=(D+1)/(2D)`` times the velocity increment. Final velocities use its
    complete value. This exactly matches the sum of D semi-implicit substeps
    for translation and joints under constant acceleration; root orientation
    uses the upstream exponential of the weighted world-frame angular twist.
    Arbitrary leading dimensions are supported without broadcasting batches.
    """
    _check_state(state)
    _check_pair(state, physical_increment, "physical_increment", GENCOORD_PREDICTION_DIM)
    if isinstance(dt, bool) or not isinstance(dt, Real) or not math.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be a finite positive control-step duration")
    if isinstance(decimation, bool) or not isinstance(decimation, Integral) or decimation < 1:
        raise ValueError("decimation must be a positive integer")
    c = (decimation + 1) / (2 * decimation)
    q = state[..., 3:7]
    v_b = _quat_apply_inverse(q, state[..., 7:10])
    w_b = _quat_apply_inverse(q, state[..., 10:13])
    dv_b, dw_b, dvj = physical_increment[..., :3], physical_increment[..., 3:6], physical_increment[..., 6:]
    next_v_global = _quat_apply(q, v_b + dv_b)
    next_w_global = _quat_apply(q, w_b + dw_b)
    v_pose = _quat_apply(q, v_b + c * dv_b)
    w_pose = _quat_apply(q, w_b + c * dw_b)
    next_p = state[..., :3] + v_pose * dt
    next_q = _quat_integrate(q, w_pose, dt)
    next_qj = state[..., 13:42] + (state[..., 42:71] + c * dvj) * dt
    next_vj = state[..., 42:71] + dvj
    # The upstream representation rotates these global velocities into the
    # new root frame, then reconstructs them into world coordinates. WMAC
    # stores world velocities directly, so no extra round trip is needed.
    return torch.cat((next_p, next_q, next_v_global, next_w_global, next_qj, next_vj), -1)


def gencoord_loss_terms(state: torch.Tensor) -> dict[str, torch.Tensor]:
    """Integrated state channels scored by the upstream GenCoord learner.

    ``||R1-R2||_F²/2 = 2(1-cos(theta))`` behaves like squared angle error
    near identity and is independent of quaternion sign. There is no direct
    root-velocity loss in this upstream objective.
    """
    _check_state(state)
    return {
        "root_pos": state[..., :3],
        "root_ori": quat_to_matrix(state[..., 3:7]).flatten(-2, -1) * math.sqrt(0.5),
        "joint_pos": state[..., 13:42],
        "joint_vel": state[..., 42:71],
    }


def gencoord_loss(predicted_state: torch.Tensor, target_state: torch.Tensor) -> torch.Tensor:
    """Per-sample Bryson-scaled mean over the 70 upstream loss channels."""
    _check_state(predicted_state, "predicted_state")
    _check_pair(predicted_state, target_state, "target_state", 71)
    predicted, target = gencoord_loss_terms(predicted_state), gencoord_loss_terms(target_state)
    return torch.cat(tuple((predicted[name] - target[name]).square() / nominal ** 2
                           for name, nominal in GENCOORD_NOMINAL_ERRORS.items()), -1).mean(-1)
