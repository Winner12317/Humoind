"""Flat sole geometry for the opt-in imagined recovery foot-slip reward.

The sole of each foot is described by points with a radius: the two end points of
every sole capsule (Unitree G1) or the eight corners of every sole box with radius
zero (EngineAI PM01).  The lowest ``point height - radius`` is the sole clearance.
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import math
import re

import mujoco
import numpy as np
import torch

from .state import quat_apply, quat_normalize


def _local_frame(model, body_id: int, ancestor_id: int) -> tuple[np.ndarray, np.ndarray]:
    """Pose of ``body_id`` in the frame of ``ancestor_id``, through jointless bodies only."""
    position, rotation = np.zeros(3), np.eye(3)
    while body_id != ancestor_id:
        if body_id == 0 or model.body_jntnum[body_id]:
            raise ValueError("Sole geoms must belong to the foot body or to bodies fixed to it")
        local = np.empty(9, dtype=np.float64)
        mujoco.mju_quat2Mat(local, model.body_quat[body_id].copy())
        local = local.reshape(3, 3)
        position, rotation = model.body_pos[body_id] + local @ position, local @ rotation
        body_id = int(model.body_parentid[body_id])
    return position, rotation


def _sole_points(model, geom_id: int, position: np.ndarray, rotation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Support points and radii of one sole geom, in the foot body's frame."""
    local = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(local, model.geom_quat[geom_id].copy())
    local = local.reshape(3, 3)
    center, size, kind = model.geom_pos[geom_id], model.geom_size[geom_id], int(model.geom_type[geom_id])
    if kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
        direction = local[:, 2] * size[1]
        points, radius = np.stack((center - direction, center + direction)), size[0]
    elif kind == mujoco.mjtGeom.mjGEOM_BOX:
        corners = np.asarray(list(itertools.product((-1., 1.), repeat=3))) * size
        points, radius = center + corners @ local.T, 0.
    elif kind == mujoco.mjtGeom.mjGEOM_SPHERE:
        points, radius = center[None], size[0]
    else:
        raise ValueError("Sole geoms must be capsules, boxes or spheres")
    return position + points @ rotation.T, np.full(len(points), radius)


@dataclass(frozen=True)
class FootSlipGeometry:
    foot_ids: tuple[int, int]
    endpoints: torch.Tensor
    radii: torch.Tensor
    site_offsets: torch.Tensor
    contact_height_m: float
    softness_m: float
    command_threshold: float
    differentiate_support: bool = False

    @classmethod
    def from_model(
        cls, model, body_ids, foot_ids, *, contact_height_m, softness_m,
        command_threshold=.1, differentiate_support=False, device="cpu", dtype=torch.float32,
        site_names=("left_foot", "right_foot"),
    ):
        if (not math.isfinite(contact_height_m) or not math.isfinite(softness_m)
                or softness_m <= 0 or not math.isfinite(command_threshold) or command_threshold < 0):
            raise ValueError("Invalid geometry foot-slip height, softness or command threshold")
        if not isinstance(differentiate_support, bool):
            raise ValueError("differentiate_support must be a bool")
        foot_ids = tuple(int(index) for index in foot_ids)
        if len(foot_ids) != 2 or foot_ids[0] == foot_ids[1]:
            raise ValueError("Expected distinct left and right foot FK body indices")
        endpoints, radii, sites = [], [], []
        for slot, side, site_name in zip(foot_ids, ("left", "right"), site_names):
            body_id = int(body_ids[slot])
            # The foot body and the jointless bodies fixed to it (PM01 keeps its sole
            # boxes on LINK_FOOT_*, a fixed child of the ankle-roll link).
            fixed = [body_id]
            for candidate in range(body_id + 1, model.nbody):
                if int(model.body_parentid[candidate]) in fixed and not model.body_jntnum[candidate]:
                    fixed.append(candidate)
            pattern = re.compile(rf"{side}_foot.*_collision")
            geom_indices = [int(index) for index in range(model.ngeom)
                            if int(model.geom_bodyid[index]) in fixed
                            and pattern.fullmatch(model.geom(int(index)).name.rsplit("/", 1)[-1])]
            site_indices = [int(index) for index in np.flatnonzero(model.site_bodyid == body_id)
                            if model.site(int(index)).name.rsplit("/", 1)[-1] == site_name]
            if not geom_indices or len(site_indices) != 1:
                raise ValueError(f"Expected {side} sole collision geoms and the original {site_name} site")
            points, point_radii = zip(*(
                _sole_points(model, index, *_local_frame(model, int(model.geom_bodyid[index]), body_id))
                for index in geom_indices))
            endpoints.append(np.concatenate(points))
            radii.append(np.concatenate(point_radii))
            sites.append(model.site_pos[site_indices[0]].copy())
        if endpoints[0].shape != endpoints[1].shape:
            raise ValueError("Left and right soles must be described by the same number of points")
        return cls(
            foot_ids, torch.as_tensor(np.asarray(endpoints), dtype=dtype, device=device),
            torch.as_tensor(np.asarray(radii), dtype=dtype, device=device),
            torch.as_tensor(np.asarray(sites), dtype=dtype, device=device),
            float(contact_height_m), float(softness_m), float(command_threshold), differentiate_support,
        )

    def kinematic_terms(self, kinematics):
        """Return per-foot sole clearance and original reward-site xy velocity."""
        positions, orientations, linear, angular = (item[..., self.foot_ids, :] for item in kinematics)
        support_positions, support_orientations = (positions, orientations) if self.differentiate_support else (
            positions.detach(), orientations.detach())
        qw, qx, qy, qz = quat_normalize(support_orientations).unbind(-1)
        world_up_local = torch.stack((2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx),
                                      1 - 2 * (qx.square() + qy.square())), -1)
        endpoint_height = (world_up_local.unsqueeze(-2) * self.endpoints).sum(-1)
        clearance = support_positions[..., 2] + (endpoint_height - self.radii).amin(-1)
        arm = quat_apply(orientations, self.site_offsets.to(orientations))
        xy_velocity = (linear + torch.cross(angular, arm, dim=-1))[..., :2]
        return clearance, xy_velocity

    def __call__(self, kinematics, command):
        clearance, xy_velocity = self.kinematic_terms(kinematics)
        support = torch.sigmoid((self.contact_height_m - clearance) / self.softness_m)
        if not self.differentiate_support:
            support = support.detach()
        active = command[..., :2].norm(dim=-1) + command[..., 2].abs() > self.command_threshold
        return (support * xy_velocity.square().sum(-1)).sum(-1) * active


# The class was written for the G1's sole capsules and is kept under its first name.
G1FootSlipGeometry = FootSlipGeometry
