"""Turn a SuperAMP PM01 checkpoint into what sim2sim.py runs.

    python export_policy.py [--checkpoint ../policy/wamp/v1_78400.pt] [--superamp ../SuperAMP]
    python export_policy.py --checkpoint ../policy/wamp/v2_82400.pt --out policy_wamp_v2
    python export_policy.py --checkpoint ../policy/ppo_amp/model_60000.pt

Both of SuperAMP's trainers are supported: WMAC (SuperMimic-style world model + AMP,
``format: amp-wmac-v1``) and PPO + AMP (the rsl_rl AMP runner). Run it once, in
SuperAMP's environment (it imports SuperAMP to rebuild the actor exactly as training
and evaluation do). It writes, to ``policy/`` for a WMAC checkpoint and to
``policy_ppo_amp/`` for a PPO one (or to ``--out``):

policy.onnx       the deployed actor: observation normaliser + MLP, and for WMAC the
                  ±20 action clamp that its training and evaluation apply (PPO + AMP
                  has no clamp). Input ``obs`` (N, 312), output ``actions`` (N, 23).
policy.json       everything a controller needs to drive it: joint order, default
                  pose, action scale, PD gains and motor limits used in training,
                  observation layout, command ranges, height thresholds, push ranges,
                  all from the SuperAMP task configuration the policy was trained
                  with (and checked against the checkpoint where it records them).
start_poses.npz   fallen poses from the PM01 get-up reference clip, for testing
                  recovery (sim2sim.py key F): the same 8 frames for every policy,
                  read from the clip set the policy was trained with.

The ONNX output is compared with the PyTorch actor on 4096 observations before
anything is written; a PPO actor is also compared with one rebuilt from the
checkpoint with rsl_rl's own normaliser.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve(patterns: dict, names, default=None) -> list:
    """Per-joint values from a {regex: value} mapping (mjlab/IsaacLab convention)."""
    values = []
    for name in names:
        hits = [value for pattern, value in patterns.items() if re.fullmatch(pattern, name)]
        if len(hits) > 1:
            raise ValueError(f"{name} matches several patterns")
        values.append(hits[0] if hits else default)
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, default=HERE.parent / "policy" / "wamp" / "v1_78400.pt")
    parser.add_argument("--superamp", type=Path, default=HERE.parent / "SuperAMP")
    parser.add_argument("--out", type=Path, help="default: policy/ (WMAC) or policy_ppo_amp/ (PPO + AMP)")
    args = parser.parse_args()
    checkpoint, superamp = args.checkpoint.resolve(), args.superamp.resolve()
    sys.path.insert(0, str(superamp))
    sys.path.insert(0, str(superamp / "scripts"))
    os.environ.setdefault("MUJOCO_GL", "egl")

    import onnxruntime as ort
    import torch

    from evaluate_wmac import load_actor  # SuperAMP: rebuilds the deployed actor
    from src.assets.robots.engineai_pm01 import pm01_constants as pm01
    from src.robot_profiles import profile_for_task

    # SuperAMP's PPO checkpoints pickle its own rsl_rl normaliser class.
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    wmac = payload.get("format") == "amp-wmac-v1"
    if not wmac and not ("model_state_dict" in payload and "amp_feature_schema" in payload):
        raise SystemExit(f"{checkpoint} is neither a SuperAMP WMAC nor a SuperAMP PPO + AMP checkpoint")
    config = payload.get("config", {})
    actor, kind = load_actor(checkpoint, "cpu")
    task = actor.amp_task_id
    robot = profile_for_task(task)
    if robot.name != "pm01":
        raise SystemExit(f"checkpoint task {task} is not a PM01 task")
    # WMAC clamps actions in training and evaluation; the PPO runner does not.
    limit = float(getattr(actor, "source_action_limit", None) or config["action_limit"]) if wmac else None
    if args.out is None:
        args.out = HERE / ("policy" if wmac else "policy_ppo_amp")

    joints = list(pm01.PM01_JOINT_NAMES)
    n = len(joints)
    frame_dim = 9 + 3 * n
    obs_dim = actor.norm.mean.numel()
    if obs_dim != 4 * frame_dim:
        raise SystemExit(f"actor input {obs_dim} is not 4 frames of {frame_dim}")

    # The task configuration the policy was trained with (observation order, timing,
    # command ranges after the curriculum, thresholds, pushes).
    import mjlab.tasks  # noqa: F401
    import src.tasks  # noqa: F401
    from mjlab.tasks.registry import load_env_cfg

    env_cfg = load_env_cfg(task)
    actor_group = env_cfg.observations["actor"]
    terms = list(actor_group.terms)
    if terms != ["base_ang_vel", "projected_gravity", "command", "joint_pos", "joint_vel", "actions"]:
        raise SystemExit(f"unexpected actor observation terms {terms}")
    if actor_group.history_length != 4 or getattr(actor_group, "history_ordering", "term") != "time":
        raise SystemExit("expected a 4-frame time-major actor history")
    ranges = env_cfg.commands["twist"].ranges
    command_ranges = {"lin_vel_x": list(ranges.lin_vel_x), "lin_vel_y": list(ranges.lin_vel_y),
                      "ang_vel_z": list(ranges.ang_vel_z)}
    if wmac:
        counter = int(payload.get("env_common_step_counter", 0))
    else:
        from mjlab.tasks.registry import load_rl_cfg
        counter = int(payload["iter"]) * int(load_rl_cfg(task).num_steps_per_env)
    for stage in env_cfg.curriculum["command_vel"].params["velocity_stages"]:
        if counter > stage["step"]:
            for key in command_ranges:
                if stage.get(key) is not None:
                    command_ranges[key] = list(stage[key])
    control_dt = env_cfg.decimation * env_cfg.sim.mujoco.timestep
    # Action scale, PD gains and motor curve of the task's position actuators.
    from src.wmac.smooth_rewards import SmoothRewardTerms
    constants = SmoothRewardTerms.from_config(env_cfg, joints).metadata()["constants"]
    saved = payload.get("smooth_reward_model")
    if saved is not None:
        if list(saved["joint_names"]) != joints or saved["action_to_joint"] != list(range(n)):
            raise SystemExit("unexpected joint order in the checkpoint")
        for key, values in saved["constants"].items():
            if not np.allclose(values, constants[key], rtol=1e-6, atol=1e-9):
                raise SystemExit(f"the task's {key} differs from the value recorded in the checkpoint")
    default_pose = resolve(env_cfg.scene.entities["robot"].init_state.joint_pos, joints, 0.0)
    if not np.allclose(default_pose, constants["action_offset"], atol=1e-6):
        raise SystemExit("default pose in the task differs from the checkpoint's action offset")
    armature = []
    for name in joints:
        owners = [cfg for cfg in pm01.PM01_ARTICULATION.actuators
                  if any(re.fullmatch(p, name) for p in cfg.target_names_expr)]
        armature.append(owners[0].armature)

    # Export: normaliser + MLP + clamp, checked against PyTorch.
    class Deployed(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.actor = actor

        def forward(self, obs):
            action = self.actor(obs)
            return action if limit is None else action.clamp(-limit, limit)

    deployed = Deployed().eval()
    args.out.mkdir(parents=True, exist_ok=True)
    onnx_path = args.out / "policy.onnx"
    with torch.no_grad():
        torch.onnx.export(deployed, torch.zeros(1, obs_dim), str(onnx_path), input_names=["obs"],
                          output_names=["actions"], dynamic_axes={"obs": {0: "batch"}, "actions": {0: "batch"}},
                          opset_version=17, dynamo=False)
    generator = torch.Generator().manual_seed(0)
    std = (actor.norm.var.clamp_min(0).sqrt() + actor.norm.eps)
    samples = actor.norm.mean + std * torch.randn(4096, obs_dim, generator=generator) * 2
    with torch.no_grad():
        expected = deployed(samples).numpy()
    reference_error = None
    if not wmac:
        # Independent rebuild of the PPO inference path: rsl_rl's normaliser, then the actor MLP.
        from rsl_rl.modules.normalizer import EmpiricalNormalization
        weights = {key[len("actor."):]: value for key, value in payload["model_state_dict"].items()
                   if key.startswith("actor.")}
        widths = [weights[key].shape for key in sorted(weights) if key.endswith(".weight")]
        layers = []
        for index, (rows, columns) in enumerate(widths):
            layers.append(torch.nn.Linear(columns, rows))
            if index < len(widths) - 1:
                layers.append(torch.nn.ELU())
        network = torch.nn.Sequential(*layers)
        network.load_state_dict(weights)
        normaliser = EmpiricalNormalization(shape=[obs_dim])
        normaliser.load_state_dict(payload["obs_norm_state_dict"])
        normaliser.eval()
        with torch.no_grad():
            reference_error = float((network(normaliser(samples)) - torch.as_tensor(expected)).abs().max())
        if reference_error > 1e-4:
            raise SystemExit(f"rebuilt PPO actor differs from the exported one by {reference_error}")
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    got = session.run(None, {"obs": samples.numpy().astype(np.float32)})[0]
    error = float(np.abs(got - expected).max())
    if not np.isfinite(got).all() or error > 1e-3:
        raise SystemExit(f"ONNX export differs from the PyTorch actor by {error}")

    # Fallen start poses from the get-up reference clip (23 joints; head yaw is 0).
    # The frames are always chosen on the original clip, so pose N is the same moment of
    # the clip for every policy; the pose itself is read from the clip set the policy
    # was trained with (a WMAC checkpoint records ``reference_motion_root``).
    relative = Path("Recovery") / "fallAndGetUp1_subject1.npz"
    original = superamp / "src" / "assets" / "motions" / "pm01" / "amp" / relative
    with np.load(original) as motion:
        root_pos, root_quat = motion["body_pos_w"][:, 0], motion["body_quat_w"][:, 0]
    w, x, y, z = root_quat.T
    up_z = 1 - 2 * (x * x + y * y)
    lying = np.flatnonzero((root_pos[:, 2] < 0.3) & (np.abs(up_z) < 0.5))
    # Distinct poses: evenly spaced picks over the lying frames.
    picks = lying[np.linspace(0, len(lying) - 1, 8).astype(int)]
    clip = original
    trained_root = config.get("reference_motion_root") if wmac else None
    if trained_root:
        root = Path(trained_root)
        clip = (root if root.is_absolute() else superamp / root) / relative
        if not clip.is_file():
            raise SystemExit(f"the checkpoint was trained with {trained_root}; {clip} is missing")
        recorded = payload.get("reference_motion_dataset")
        if recorded is not None:
            from src.wmac.recovery_style import motion_root_fingerprint
            if motion_root_fingerprint(clip.parent.parent) != recorded:
                raise SystemExit(f"{clip.parent.parent} differs from the clip set recorded in the checkpoint")
    with np.load(clip) as motion:
        root_pos, root_quat, joint_pos = motion["body_pos_w"][:, 0], motion["body_quat_w"][:, 0], motion["joint_pos"]
    poses = np.concatenate((root_pos[picks], root_quat[picks], joint_pos[picks]), axis=1).astype(np.float64)
    poses[:, :2] = 0.0  # start at the origin; heading kept
    np.savez(args.out / "start_poses.npz", qpos=poses, source=str(clip.relative_to(superamp.parent)),
             frames=picks, fps=50.0)

    meta = {
        "format": "pm01-sim2sim-policy-v1",
        "source": {
            "checkpoint": str(checkpoint.relative_to(HERE.parent) if checkpoint.is_relative_to(HERE.parent) else checkpoint),
            "checkpoint_sha256": sha256(checkpoint),
            "checkpoint_format": payload.get("format", "superamp-rsl_rl-amp-runner"), "loaded_as": kind,
            "iteration": payload["iteration"] if wmac else payload["iter"],
            "task": task, "run_name": config.get("run_name"),
            "trained_by": ("SuperAMP WMAC (SuperMimic-style world model + AMP)" if wmac
                           else "SuperAMP PPO + AMP (rsl_rl AMP runner)"),
            "amp_feature_schema": actor.amp_feature_schema,
            "onnx_sha256": sha256(onnx_path), "onnx_max_abs_error_vs_torch": error,
            "rsl_rl_rebuild_max_abs_error": reference_error,
        },
        "joint_names": joints,
        "control": {
            "control_dt": control_dt,
            "training_physics_dt": env_cfg.sim.mujoco.timestep,
            "action_clip": limit,
            "target": "default_joint_pos + action_scale * action" + (
                " (action already clipped by the ONNX graph)" if limit is not None else " (no action clip)"),
            "default_joint_pos": [float(v) for v in constants["action_offset"]],
            "action_scale": [float(v) for v in constants["action_scale"]],
            "kp": [float(v) for v in constants["kp"]],
            "kd": [float(v) for v in constants["kd"]],
            "effort_limit": [float(v) for v in constants["effort_limit"]],
            "velocity_limit": [float(v) for v in constants["x1"]],
            "armature": [float(v) for v in armature],
            "friction_static": [float(v) for v in constants["friction_static"]],
            "friction_viscous": [float(v) for v in constants["friction_dynamic"]],
            "friction_activation_velocity": [float(v) for v in constants["activation_velocity"]],
        },
        "observation": {
            "dim": obs_dim, "history_length": 4, "history_order": "oldest frame first",
            "frame_dim": frame_dim,
            "frame": [
                {"name": "base_ang_vel", "dim": 3, "definition": "gyro at the imu site (LINK_BASE frame), rad/s"},
                {"name": "projected_gravity", "dim": 3, "definition": "unit gravity vector in the LINK_BASE frame"},
                {"name": "command", "dim": 3, "definition": "vx, vy (m/s) in the heading frame, yaw rate (rad/s)"},
                {"name": "joint_pos", "dim": n, "definition": "joint position minus default_joint_pos, rad"},
                {"name": "joint_vel", "dim": n, "definition": "joint velocity, rad/s"},
                {"name": "last_action", "dim": n, "definition": "previous policy output (clipped), before scaling"},
            ],
            "reset": "after a reset every frame of the history equals the first frame, last_action is zero",
        },
        "command_ranges": command_ranges,
        "thresholds_m": {
            "standing_height": float(env_cfg.scene.entities["robot"].init_state.pos[2]),
            "termination_height": float(env_cfg.terminations["bad_base_height"].params["minimum_height"]),
            "termination_tilt_rad": float(env_cfg.terminations["bad_orientation"].params["limit_angle"]),
            "recovery_height": robot.recovery_height,
            "upright_height": robot.upright_height,
        },
        "push": {"velocity_range": env_cfg.events["push_robot"].params["velocity_range"],
                 "interval_s": list(env_cfg.events["push_robot"].interval_range_s)},
    }
    (args.out / "policy.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {onnx_path} (max |ONNX - torch| = {error:.1e}), policy.json, start_poses.npz ({len(picks)} poses)")
    print(f"policy: {meta['source']['trained_by']}, {task}, iteration {meta['source']['iteration']}, "
          f"obs {obs_dim}, actions {n}, control {1 / control_dt:.0f} Hz -> {args.out}")
    print(f"command ranges after the curriculum: {command_ranges}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
