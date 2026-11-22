"""Simulator-level checks of the PM01 tasks (needs the GPU; the unit tests do not).

    python scripts/pm01/check_pm01_env.py                       # all three PM01 tasks
    python scripts/pm01/check_pm01_env.py Unitree-G1-AMP-Flat   # the same checks on a G1 task

For each task:
1. the environment builds and reports its observation widths;
2. reference frames are written into the simulator and the AMP observation the
   discriminator would see is compared with the expert features the runner's
   motion loader builds from the same frames (they must be the same numbers,
   otherwise the discriminator can tell policy from expert by a layout difference);
3. the robot is placed in its default pose and held with zero actions for one
   second, to confirm the soles carry the weight before balance is lost;
4. random actions are applied for 200 steps and the result must stay finite.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch

import mjlab.tasks  # noqa: F401
import src.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from rsl_rl.utils import AMPLoader
from src.robot_profiles import profile_for_bodies

DEFAULT_TASKS = ("EngineAI-PM01-AMP-Flat", "EngineAI-PM01-AMP-Flat-Aligned", "EngineAI-PM01-AMP-Rough")


def write_frames(env, motion, frames) -> None:
    robot = env.scene["robot"]

    def tensor(key, index=...):
        return torch.as_tensor(motion[key][frames][:, index] if index is not ... else motion[key][frames],
                               device=env.device)

    pose = torch.cat((tensor("body_pos_w", 0), tensor("body_quat_w", 0)), -1)
    pose[:, :3] += env.scene.env_origins
    robot.write_root_link_pose_to_sim(pose)
    robot.write_root_link_velocity_to_sim(torch.cat((tensor("body_lin_vel_w", 0), tensor("body_ang_vel_w", 0)), -1))
    robot.write_joint_state_to_sim(tensor("joint_pos"), tensor("joint_vel"))
    env.sim.forward()
    env.scene.update(env.physics_dt)


def amp_observation(env) -> torch.Tensor:
    """The AMP group evaluated now (the manager's own call returns its cached history)."""
    terms = env.observation_manager._group_obs_term_cfgs["amp"]
    return torch.cat([term.func(env, **term.params) for term in terms], -1)


def check(task: str, num_envs: int = 16) -> bool:
    print(f"\n=== {task} ===")
    cfg = load_env_cfg(task)
    cfg.scene.num_envs, cfg.seed = num_envs, 1
    cfg.events.pop("push_robot", None)
    runner_cfg = load_rl_cfg(task)
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
    ok = True
    try:
        observations, _ = env.reset()
        robot = env.scene["robot"]
        profile = profile_for_bodies(robot.body_names)
        action_dim = env.action_manager.total_action_dim
        print(f"robot {profile.name}: {len(robot.joint_names)} joints, {len(robot.body_names)} bodies, "
              f"{action_dim} actions; observations " + ", ".join(f"{k} {v.shape[1]}" for k, v in observations.items()))

        schema = runner_cfg.amp_feature_schema
        loader = AMPLoader(runner_cfg.amp_motion_files, runner_cfg.amp_body_names, runner_cfg.amp_anchor_name,
                           robot.body_names, device="cuda:0", feature_schema=schema,
                           root_name=profile.root_body, foot_names=profile.foot_bodies,
                           foot_reference_offsets=profile.amp_foot_offsets)
        files = sorted(Path(runner_cfg.amp_motion_files).rglob("*.npz"))
        groups = {False: [f for f in files if "Recovery" not in f.parts], True: [f for f in files if "Recovery" in f.parts]}
        worst = 0.0
        for recovery, paths in groups.items():
            path = paths[0]
            with np.load(path) as data:
                motion = {key: data[key] for key in data.files}
            frames = np.linspace(0, len(motion["joint_pos"]) - 2, num_envs).astype(int)
            write_frames(env, motion, frames)
            simulated = amp_observation(env)
            if schema == "legacy":
                index = loader.motion_names.index(path.stem)
                expert = torch.cat([getattr(loader, f"_body_{name}_b_list")[index][frames].flatten(1)
                                    for name in ("pos", "ori", "lin_vel", "ang_vel")], -1)
            else:
                # Pairs are stored per group in file order; the first clip's start frames come first.
                expert = loader._pair_groups[int(recovery)][frames, 0]
            error = float((simulated - expert).abs().max())
            worst = max(worst, error)
            print(f"  AMP features, simulator vs expert loader, {path.parent.name}/{path.stem}: max |diff| {error:.2e}")
        ok &= worst < 1e-4
        print(f"  feature width {simulated.shape[1]}, schema {schema}: {'OK' if worst < 1e-4 else 'MISMATCH'}")

        root = robot.data.default_root_state.clone()
        root[:, :3] += env.scene.env_origins
        robot.write_root_state_to_sim(root)
        robot.write_joint_state_to_sim(robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone())
        env.sim.forward()
        zero = torch.zeros(num_envs, action_dim, device=env.device)
        mass = float(env.sim.mj_model.body_subtreemass[robot.indexing.body_ids[0]])
        height0 = float(robot.data.default_root_state[0, 2])
        forces = []
        for step in range(50):
            env.step(zero)
            if 5 <= step < 15:
                forces.append(-env.scene["feet_ground_contact"].data.force[..., 2].sum(1))
            if step == 14:
                height = float((robot.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]).median())
        support = float(torch.stack(forces).median()) / (mass * 9.81)
        print(f"  default pose, zero action: base {height0:.3f} -> {height:.3f} m after 0.3 s, "
              f"feet carry {100 * support:.0f} % of {mass:.1f} kg")
        if "Rough" not in task:
            # The G1 is spawned 2 cm above its standing height and settles on bent knees.
            ok &= abs(height - height0) < 0.06 and 0.7 < support < 1.3

        finite = True
        for _ in range(200):
            observations, reward, *_ = env.step(torch.randn(num_envs, action_dim, device=env.device))
            finite &= bool(torch.isfinite(reward).all()) and all(bool(torch.isfinite(v).all()) for v in observations.values())
        print(f"  200 random-action steps: {'finite' if finite else 'NON-FINITE'}")
        ok &= finite
    finally:
        env.close()
    print(f"  => {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> int:
    tasks = sys.argv[1:] or DEFAULT_TASKS
    results = {task: check(task) for task in tasks}
    print("\n" + "\n".join(f"{'PASS' if ok else 'FAIL'}  {task}" for task, ok in results.items()))
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
