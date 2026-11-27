"""Robot dynamics profiles: another G1 model's inertials and joint parameters.

A profile (see ``scripts/extract_robot_dynamics_profile.py``) stores the body
masses, inertias, centres of mass and joint armature/damping/friction loss of a
different G1 model, expressed in this task's body frames. Applying it keeps the
training robot's kinematics, collision geometry, actuators and PD gains; only
the mass distribution and passive joint dynamics change.
"""
from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path

import torch

from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg

_PROFILE_DIR = Path(__file__).resolve().parents[1] / "assets/robots/unitree_g1"
ROBOT_DYNAMICS_PROFILES = {
    "unitree_mujoco_g1_29dof": _PROFILE_DIR / "unitree_mujoco_g1_29dof_dynamics.json",
}
PROFILE_BODY_FIELDS = ("body_mass", "body_ipos", "body_iquat", "body_inertia")
PROFILE_DOF_FIELDS = ("dof_armature", "dof_damping", "dof_frictionloss")


@lru_cache(maxsize=None)
def load_robot_dynamics_profile(name: str) -> dict:
    if name not in ROBOT_DYNAMICS_PROFILES:
        raise ValueError(f"Unknown robot dynamics profile {name!r}; choose from {sorted(ROBOT_DYNAMICS_PROFILES)}")
    profile = json.loads(ROBOT_DYNAMICS_PROFILES[name].read_text())
    if profile.get("format") != "robot-dynamics-profile-v1":
        raise ValueError(f"{name}: unsupported robot dynamics profile format")
    return profile


def profile_tensors(robot, profile: dict, device) -> dict[str, torch.Tensor]:
    """Profile values in the entity's body/joint order (names must match exactly)."""
    body_names, joint_names = tuple(robot.body_names), tuple(robot.joint_names)
    if set(body_names) != set(profile["bodies"]) or set(joint_names) != set(profile["joints"]):
        raise ValueError("Robot dynamics profile does not describe this robot's bodies and joints")
    bodies = [profile["bodies"][name] for name in body_names]
    joints = [profile["joints"][name] for name in joint_names]

    def tensor(rows, key):
        return torch.tensor([row[key] for row in rows], dtype=torch.float32, device=device)

    return {
        "body_ids": robot.indexing.body_ids.to(device=device, dtype=torch.long),
        "dof_ids": robot.indexing.joint_v_adr.to(device=device, dtype=torch.long),
        "body_mass": tensor(bodies, "mass"),
        "body_ipos": tensor(bodies, "ipos"),
        "body_iquat": tensor(bodies, "iquat"),
        "body_inertia": tensor(bodies, "inertia"),
        "dof_armature": tensor(joints, "armature"),
        "dof_damping": tensor(joints, "damping"),
        "dof_frictionloss": tensor(joints, "frictionloss"),
    }


@requires_model_fields(*PROFILE_BODY_FIELDS, *PROFILE_DOF_FIELDS, recompute=RecomputeLevel.set_const)
def apply_robot_dynamics_profile(env, env_ids=None, profile_name="unitree_mujoco_g1_29dof",
                                 asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """Startup event: replace inertials and passive joint dynamics in every listed world."""
    robot = env.scene[asset_cfg.name]
    values = profile_tensors(robot, load_robot_dynamics_profile(profile_name), env.device)
    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
    model = env.sim.model
    bodies, dofs = values["body_ids"], values["dof_ids"]
    for field in PROFILE_BODY_FIELDS:
        getattr(model, field)[ids[:, None], bodies[None, :]] = values[field].to(getattr(model, field).dtype)
    for field in PROFILE_DOF_FIELDS:
        getattr(model, field)[ids[:, None], dofs[None, :]] = values[field].to(getattr(model, field).dtype)


# --- Training-time randomization between the training robot and a profile ---
# Per-environment, startup-only (no per-iteration cost). Four scalars describe
# the whole sample; they are appended (normalized to [-1, 1]) to the world-model
# and critic context so imagined dynamics stay conditioned on the true robot.
ROBOT_DYNAMICS_CONTEXT_WIDTH = 4
ROBOT_DYNAMICS_RANDOMIZATION_SETTINGS = {
    "unitree_g1_v1": {
        "profile": "unitree_mujoco_g1_29dof",
        # 0 = training (Hiking URDF) robot, 1 = profile robot; values beyond
        # [0, 1] extrapolate by 25% past either model.
        "inertial_interpolation": (-.25, 1.25),
        "armature_interpolation": (-.25, 1.25),
        # Multiples of the profile's joint damping / friction loss.
        "damping_scale": (0., 2.),
        "frictionloss_scale": (0., 1.5),
    },
}


def robot_dynamics_metadata(name: str) -> dict:
    settings = dict(ROBOT_DYNAMICS_RANDOMIZATION_SETTINGS[name])
    profile = load_robot_dynamics_profile(settings["profile"])
    return {
        "name": name, **{k: list(v) if isinstance(v, tuple) else v for k, v in settings.items()},
        "profile_source": profile["source"], "profile_sha256": profile["source_file_sha256"],
        "sampling": "uniform per environment at startup; fixed for the run",
        "context": ["inertial_interpolation", "armature_interpolation", "damping_scale",
                    "frictionloss_scale"],
        "context_normalization": "each sampled range mapped linearly onto [-1, 1]",
        "inertia": "body-frame tensors interpolated, then re-diagonalized",
        "unchanged": "kinematics, collision geometry, actuator PD gains and effort limits",
        "actor_input": False,
    }


def _normalize(value, bounds):
    low, high = bounds
    return 2. * (value - low) / (high - low) - 1.


def nominal_robot_dynamics_context(name: str, num_envs: int, device) -> torch.Tensor:
    """Context of the unmodified training robot (used when the event is not installed)."""
    s = ROBOT_DYNAMICS_RANDOMIZATION_SETTINGS[name]
    nominal = (_normalize(0., s["inertial_interpolation"]), _normalize(0., s["armature_interpolation"]),
               _normalize(0., s["damping_scale"]), _normalize(0., s["frictionloss_scale"]))
    return torch.tensor(nominal, device=device, dtype=torch.float32).expand(num_envs, -1).clone()


def _inertia_tensor(iquat, diagonal):
    from mjlab.utils.lab_api.math import matrix_from_quat
    rotation = matrix_from_quat(iquat)
    return rotation @ torch.diag_embed(diagonal) @ rotation.transpose(-1, -2)


@requires_model_fields(*PROFILE_BODY_FIELDS, *PROFILE_DOF_FIELDS, recompute=RecomputeLevel.set_const)
def randomize_robot_dynamics(env, env_ids=None, randomization="unitree_g1_v1",
                             asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """Startup event: sample each world's robot between the training robot and the profile."""
    from mjlab.utils.lab_api.math import quat_from_matrix
    settings = ROBOT_DYNAMICS_RANDOMIZATION_SETTINGS[randomization]
    robot = env.scene[asset_cfg.name]
    target = profile_tensors(robot, load_robot_dynamics_profile(settings["profile"]), env.device)
    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
    bodies, dofs = target["body_ids"], target["dof_ids"]
    base = {field: env.sim.get_default_field(field)[bodies if field.startswith("body") else dofs].float()
            for field in (*PROFILE_BODY_FIELDS, *PROFILE_DOF_FIELDS)}
    count = len(ids)

    def uniform(bounds):
        return torch.empty(count, device=env.device).uniform_(*bounds)

    inertial = uniform(settings["inertial_interpolation"])
    armature = uniform(settings["armature_interpolation"])
    damping = uniform(settings["damping_scale"])
    friction = uniform(settings["frictionloss_scale"])
    lam = inertial[:, None]
    mass = (1 - lam) * base["body_mass"] + lam * target["body_mass"]
    mass = torch.maximum(mass, .25 * torch.minimum(base["body_mass"], target["body_mass"]))
    ipos = (1 - lam[..., None]) * base["body_ipos"] + lam[..., None] * target["body_ipos"]
    tensor = ((1 - lam[..., None, None]) * _inertia_tensor(base["body_iquat"], base["body_inertia"])
              + lam[..., None, None] * _inertia_tensor(target["body_iquat"], target["body_inertia"]))
    eigenvalues, eigenvectors = torch.linalg.eigh(.5 * (tensor + tensor.transpose(-1, -2)))
    # A proper rotation for quat_from_matrix: flip the last axis where needed.
    flip = torch.linalg.det(eigenvectors) < 0
    eigenvectors[..., 2] = torch.where(flip[..., None], -eigenvectors[..., 2], eigenvectors[..., 2])
    iquat = quat_from_matrix(eigenvectors)
    inertia = eigenvalues.clamp_min(1e-7)
    arm = (1 - armature[:, None]) * base["dof_armature"] + armature[:, None] * target["dof_armature"]
    arm = torch.maximum(arm, .25 * torch.minimum(base["dof_armature"], target["dof_armature"]))
    model = env.sim.model
    for field, value in (("body_mass", mass), ("body_ipos", ipos), ("body_iquat", iquat),
                         ("body_inertia", inertia)):
        getattr(model, field)[ids[:, None], bodies[None, :]] = value.to(getattr(model, field).dtype)
    for field, value in (("dof_armature", arm), ("dof_damping", damping[:, None] * target["dof_damping"]),
                         ("dof_frictionloss", friction[:, None] * target["dof_frictionloss"])):
        getattr(model, field)[ids[:, None], dofs[None, :]] = value.to(getattr(model, field).dtype)
    if not hasattr(env, "_robot_dynamics_context"):
        env._robot_dynamics_context = nominal_robot_dynamics_context(randomization, env.num_envs, env.device)
    env._robot_dynamics_context[ids] = torch.stack((
        _normalize(inertial, settings["inertial_interpolation"]),
        _normalize(armature, settings["armature_interpolation"]),
        _normalize(damping, settings["damping_scale"]),
        _normalize(friction, settings["frictionloss_scale"])), -1)
