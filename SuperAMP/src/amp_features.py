"""Shared AMP features for motion data, simulator observations, and imagined states.

``legacy`` preserves the original 15 features per body, including its rotation
matrix *column* convention. ``grounded_v2`` adds eight world-ground cues without
making the discriminator depend on world yaw or horizontal position. All inputs
use world coordinates and wxyz quaternions, and all operations are differentiable.

Motion NPZ files have no measured contact labels. The foot additions are the
height and vertical velocity of a fixed sole reference point, not contact truth
or contact force. They make support proximity and landing/lift-off visible using
the same kinematics for experts, real policy states, and world-model states.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

LEGACY_SCHEMA = "legacy"
GROUNDED_SCHEMA = "grounded_v2"
AMP_SCHEMAS = (LEGACY_SCHEMA, GROUNDED_SCHEMA)

# Both G1 ``left_foot`` / ``right_foot`` sites in g1.xml use this local offset
# from their ankle-roll body. At flat-foot orientation the point is on the
# capsule soles (capsule center z=-.025, radius=.01). For a tilted foot it is a
# reference-point clearance, rather than the minimum collision-geometry gap.
G1_FOOT_REFERENCE_OFFSET = (0.04, 0.0, -0.035)


def amp_feature_dim(schema: str = LEGACY_SCHEMA, num_bodies: int = 13) -> int:
    """Feature width for one frame; a discriminator transition has twice this."""
    if schema not in AMP_SCHEMAS:
        raise ValueError(f"Unknown AMP feature schema {schema!r}; expected one of {AMP_SCHEMAS}")
    if num_bodies < 1:
        raise ValueError("AMP features require at least one body")
    return 15 * num_bodies + (8 if schema == GROUNDED_SCHEMA else 0)


def amp_feature_metadata(schema: str = LEGACY_SCHEMA, num_bodies: int = 13,
                         foot_reference_offsets: Sequence[Sequence[float]] | None = None) -> dict[str, Any]:
    """Serializable feature layout for checkpoints and experiment descriptions.

    ``foot_reference_offsets`` are the (left, right) sole reference points of a
    robot other than the G1; omitted, the G1 offset is recorded as before.
    """
    dim = amp_feature_dim(schema, num_bodies)
    widths = (
        ("anchor_relative_body_position", 3 * num_bodies),
        ("anchor_relative_rotation_matrix_first_two_columns", 6 * num_bodies),
        ("body_local_linear_velocity", 3 * num_bodies),
        ("body_local_angular_velocity", 3 * num_bodies),
    )
    if schema == GROUNDED_SCHEMA:
        widths += (
            ("pelvis_projected_gravity", 3),
            ("pelvis_ground_relative_height", 1),
            ("foot_reference_ground_relative_height", 2),
            ("foot_reference_world_vertical_velocity", 2),
        )
    start = 0
    fields = []
    for name, width in widths:
        fields.append({"name": name, "start": start, "stop": start + width})
        start += width
    result: dict[str, Any] = {
        "schema": schema,
        "feature_dim": dim,
        "num_bodies": num_bodies,
        "quaternion_convention": "wxyz",
        "fields": fields,
    }
    if schema == GROUNDED_SCHEMA:
        result.update(
            ground_assumption="horizontal plane; ground_height supplied in world coordinates",
            foot_order=["left", "right"],
            foot_reference_local_offset=(list(G1_FOOT_REFERENCE_OFFSET) if foot_reference_offsets is None
                                         else [[float(x) for x in offset] for offset in foot_reference_offsets]),
            support_features="kinematic sole-reference height and vertical velocity; not measured contacts",
        )
    return result


def _quat_normalize(q: torch.Tensor) -> torch.Tensor:
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def _quat_inv(q: torch.Tensor) -> torch.Tensor:
    return torch.cat((q[..., :1], -q[..., 1:]), dim=-1) / q.square().sum(-1, keepdim=True).clamp_min(1e-8)


def _quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    w1, v1 = q1[..., :1], q1[..., 1:]
    w2, v2 = q2[..., :1], q2[..., 1:]
    v1, v2 = torch.broadcast_tensors(v1, v2)
    return torch.cat((w1 * w2 - (v1 * v2).sum(-1, keepdim=True),
                      w1 * v2 + w2 * v1 + torch.cross(v1, v2, dim=-1)), -1)


def _quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    q = _quat_normalize(q)
    xyz, v = torch.broadcast_tensors(q[..., 1:], v)
    uv = torch.cross(xyz, v, dim=-1)
    return v + 2 * (q[..., :1] * uv + torch.cross(xyz, uv, dim=-1))


def _quat_to_matrix(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = _quat_normalize(q).unbind(-1)
    return torch.stack((1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)),
                       -1).reshape(*q.shape[:-1], 3, 3)


def build_amp_features(
    body_pos_w: torch.Tensor,
    body_quat_w: torch.Tensor,
    body_lin_vel_w: torch.Tensor,
    body_ang_vel_w: torch.Tensor,
    *,
    body_ids: Sequence[int] | torch.Tensor,
    anchor_id: int,
    pelvis_id: int | None = None,
    foot_ids: Sequence[int] | torch.Tensor | None = None,
    schema: str = LEGACY_SCHEMA,
    ground_height: float | torch.Tensor = 0.0,
    foot_local_offsets: Sequence[Sequence[float]] | torch.Tensor | None = None,
) -> torch.Tensor:
    """Construct one AMP frame from full-body world kinematics.

    Input shapes are ``(..., bodies, 3/4)``; the output is ``(..., features)``.
    Body IDs index the input body axis. ``grounded_v2`` requires a pelvis ID and
    two foot IDs ordered left/right. ``ground_height`` is a scalar or broadcasts
    to the batch prefix. Optional foot offsets have shape ``(2, 3)``; the default
    matches the G1 XML sole sites. Offsets are rotated with each foot, and their
    world velocity includes the angular-velocity cross-offset term.
    """
    amp_feature_dim(schema, len(body_ids))
    prefix = body_pos_w.shape[:-2]
    if body_pos_w.shape[-1] != 3 or body_quat_w.shape != (*body_pos_w.shape[:-1], 4):
        raise ValueError("AMP body positions/quaternions must have matching (..., bodies, 3/4) shapes")
    if body_lin_vel_w.shape != body_pos_w.shape or body_ang_vel_w.shape != body_pos_w.shape:
        raise ValueError("AMP body velocities must have the same shape as body positions")

    anchor_inv = _quat_inv(body_quat_w[..., anchor_id:anchor_id + 1, :])
    selected_quat = body_quat_w[..., body_ids, :]
    terms = (
        _quat_apply(anchor_inv, body_pos_w[..., body_ids, :] - body_pos_w[..., anchor_id:anchor_id + 1, :]),
        _quat_to_matrix(_quat_mul(anchor_inv, selected_quat))[..., :, :2],
        _quat_apply(_quat_inv(selected_quat), body_lin_vel_w[..., body_ids, :]),
        _quat_apply(_quat_inv(selected_quat), body_ang_vel_w[..., body_ids, :]),
    )
    legacy = torch.cat([term.reshape(*prefix, -1) for term in terms], -1)
    if schema == LEGACY_SCHEMA:
        return legacy
    if pelvis_id is None or foot_ids is None or len(foot_ids) != 2:
        raise ValueError("grounded_v2 AMP features require pelvis_id and two left/right foot_ids")

    ground = torch.as_tensor(ground_height, device=body_pos_w.device, dtype=body_pos_w.dtype)
    ground = torch.broadcast_to(ground, prefix).unsqueeze(-1)
    gravity_w = body_pos_w.new_tensor((0.0, 0.0, -1.0))
    gravity_b = _quat_apply(_quat_inv(body_quat_w[..., pelvis_id, :]), gravity_w)
    pelvis_height = body_pos_w[..., pelvis_id, 2:3] - ground

    if foot_local_offsets is None:
        offsets = body_pos_w.new_tensor((G1_FOOT_REFERENCE_OFFSET, G1_FOOT_REFERENCE_OFFSET))
    else:
        offsets = torch.as_tensor(foot_local_offsets, device=body_pos_w.device, dtype=body_pos_w.dtype)
        if offsets.shape != (2, 3):
            raise ValueError("foot_local_offsets must have shape (2, 3), in left/right order")
    foot_offset_w = _quat_apply(body_quat_w[..., foot_ids, :], offsets)
    foot_height = (body_pos_w[..., foot_ids, :] + foot_offset_w)[..., 2] - ground
    foot_velocity = body_lin_vel_w[..., foot_ids, :] + torch.cross(
        body_ang_vel_w[..., foot_ids, :], foot_offset_w, dim=-1)
    return torch.cat((legacy, gravity_b, pelvis_height, foot_height, foot_velocity[..., 2]), -1)
