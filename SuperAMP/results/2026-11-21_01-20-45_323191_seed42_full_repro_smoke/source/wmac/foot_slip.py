"""Flat G1 sole geometry for the opt-in imagined recovery foot-slip reward."""

from __future__ import annotations

from dataclasses import dataclass
import math

import mujoco
import numpy as np
import torch

from .state import quat_apply, quat_normalize


@dataclass(frozen=True)
class G1FootSlipGeometry:
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
    ):
        if (not math.isfinite(contact_height_m) or not math.isfinite(softness_m)
                or softness_m <= 0 or not math.isfinite(command_threshold) or command_threshold < 0):
            raise ValueError("Invalid geometry foot-slip height, softness or command threshold")
        if not isinstance(differentiate_support, bool):
            raise ValueError("differentiate_support must be a bool")
        foot_ids = tuple(int(index) for index in foot_ids)
        if len(foot_ids) != 2 or foot_ids[0] == foot_ids[1]:
            raise ValueError("Expected distinct left and right G1 foot FK body indices")
        endpoints, radii, sites = [], [], []
        for slot, side in zip(foot_ids, ("left", "right")):
            body_id = int(body_ids[slot])
            suffixes = {f"{side}_foot{number}_collision" for number in range(1, 8)}
            geom_indices = [int(index) for index in np.flatnonzero(model.geom_bodyid == body_id)
                            if model.geom(int(index)).name.rsplit("/", 1)[-1] in suffixes]
            site_indices = [int(index) for index in np.flatnonzero(model.site_bodyid == body_id)
                            if model.site(int(index)).name.rsplit("/", 1)[-1] == f"{side}_foot"]
            if (len(geom_indices) != 7 or len(site_indices) != 1
                    or not np.all(model.geom_type[geom_indices] == mujoco.mjtGeom.mjGEOM_CAPSULE)):
                raise ValueError(f"Expected seven sole capsules and the original {side}_foot site")
            direction = np.empty((7, 3), dtype=np.float64)
            for row, index in enumerate(geom_indices):
                rotation = np.empty(9, dtype=np.float64)
                mujoco.mju_quat2Mat(rotation, model.geom_quat[index].copy())
                direction[row] = rotation.reshape(3, 3)[:, 2] * model.geom_size[index, 1]
            centers = model.geom_pos[geom_indices]
            endpoints.append(np.stack((centers - direction, centers + direction), axis=1).reshape(14, 3))
            radii.append(np.repeat(model.geom_size[geom_indices, 0], 2))
            sites.append(model.site_pos[site_indices[0]].copy())
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
