"""Actor-only, first-episode tests from the exact model default standing pose.

Examples::
    python scripts/evaluate_wmac_zero_start.py --checkpoint RUN/model_138000.pt \
        --device cuda:0 --num-envs 32 --seeds 2032 --cases stand start

The only state reset is the ordinary environment reset before each case.
At startup only the command changes; physics, actions and both histories
continue uninterrupted. Training defaults are not changed by this script.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_wmac_perceptive import (
    PolicyHistory, _assert_no_noise_or_push, load_actor, seed_all,
)


@dataclass(frozen=True)
class SuccessThresholds:
    """Fixed engineering acceptance criteria, not learned task rewards."""

    stand_max_xy_drift_m: float = .20
    stand_mean_xy_speed_m_s: float = .10
    max_base_tilt_degrees: float = 20.
    start_min_forward_displacement_m: float = 1.
    start_final_forward_speed_min_m_s: float = .30
    start_final_forward_speed_max_m_s: float = .70
    start_max_lateral_displacement_m: float = .50
    final_speed_window_seconds: float = 1.
    reach_speed_m_s: float = .30
    reach_speed_sustained_seconds: float = .30


def steps_for(seconds, dt):
    """Avoid a floating point boundary moving the scheduled start by a step."""
    if not (math.isfinite(seconds) and seconds >= 0 and math.isfinite(dt) and dt > 0):
        raise ValueError("Durations must be finite/nonnegative and dt positive")
    steps = round(seconds / dt)
    if not math.isclose(steps * dt, seconds, rel_tol=0., abs_tol=1e-7):
        raise ValueError("Durations must be integer multiples of the control timestep")
    return steps


class ScheduledCommand:
    """Hold explicit body-frame commands on one environment instance only.

    mjlab computes the next observation after command.compute(). Consequently
    this hook advances to k+1 there, preserving the checkpoint's native history
    at the command change. No observation/state/history tensors are rewritten.
    """

    def __init__(self, env, case, wait_seconds, forward_speed):
        import torch
        if case not in ("stand", "start", "immediate_start"):
            raise ValueError("Unknown zero-start case")
        self.env, self.case = env, case
        self.wait_steps = 0 if case == "immediate_start" else steps_for(wait_seconds, env.step_dt)
        self.forward_speed, self.step_index = float(forward_speed), 0
        self.term = env.command_manager.get_term("twist")
        # This is diagnostic metadata only. Auto navigation is bypassed.
        self.term.target_pos_w.copy_(env.scene.env_origins)
        self.term.target_pos_w[:, 0] += 1000.
        self.term.goal_is_traversal.fill_(False)
        self.term.is_in_place_turn.fill_(False)
        self.term.is_heading_env.fill_(False)
        self.term.is_standing_env.fill_(False)
        self.term._resample_command = lambda env_ids: self.apply()
        self.term._update_command = self.apply
        self.term.compute = self.compute
        self.apply()

    def command_at(self, step_index):
        moving = self.case != "stand" and step_index >= self.wait_steps
        return (self.forward_speed if moving else 0., 0., 0.)

    def apply(self):
        import torch
        value = self.term.vel_command_b.new_tensor(self.command_at(self.step_index))
        self.term.vel_command_b.copy_(value.expand_as(self.term.vel_command_b))
        self.term.desired_speed.fill_(float(value[0]))
        self.term.is_standing_env.fill_(self.command_at(self.step_index)==(0.,0.,0.))
        if hasattr(self.term,"random_velocity_mask"):
            self.term.random_velocity_mask.zero_()
            self.term.random_yaw_rate.zero_()

    def compute(self, dt):
        if not math.isclose(float(dt), float(self.env.step_dt), abs_tol=1e-9):
            raise ValueError("Command clock must advance once per physical control step")
        self.step_index += 1
        self.apply()

    def expected_history(self, actor_history):
        import torch
        length=actor_history_length(actor_history)
        commands = [self.command_at(max(0, self.step_index - length+1 + i)) for i in range(length)]
        return actor_history.new_tensor(commands).unsqueeze(0).expand(actor_history.shape[0], -1, -1)

    def assert_observation(self, actor_history, active=None):
        import torch
        observed = actor_history_frames(actor_history)[:, :, 6:9]
        expected = self.expected_history(actor_history)
        if active is not None:
            observed, expected = observed[active], expected[active]
        if not torch.equal(observed, expected):
            raise RuntimeError("Actor command history does not match the physical command schedule")


def actor_history_length(actor_history):
    if actor_history.ndim!=2 or actor_history.shape[1] not in (384,768):
        raise ValueError("Zero-start evaluator requires a versioned 4 x 96 or 8 x 96 actor schema")
    return actor_history.shape[1]//96


def actor_history_frames(actor_history):
    length=actor_history_length(actor_history)
    if length==8:
        from src.wmac_perceptive.hiking_observations import unpack_hiking_proprio_history
        return unpack_hiking_proprio_history(actor_history)
    return actor_history.reshape(actor_history.shape[0],length,96)


def outside_evaluation_patch(env, half_extent=6., margin=.1):
    from src.tasks.amp_perceptive.mdp import root_pose_from_asset
    position = root_pose_from_asset(env.scene["robot"])[:, :2]
    return ((position - env.scene.env_origins[:, :2]).abs() > half_extent-margin).any(-1)


def configure_zero_start(env_cfg, *, terrain="flat", domain_randomization="retained", duration=10.):
    """Transform a new play config, before any managers or randomizers exist."""
    from mjlab.envs import mdp
    from mjlab.managers.event_manager import EventTermCfg
    from mjlab.managers.termination_manager import TerminationTermCfg
    from src.tasks.amp_perceptive.terrain import HikingTerrainCfg
    if terrain not in ("flat", "training_spawn"):
        raise ValueError("Unsupported evaluation terrain")
    if domain_randomization not in ("retained", "nominal"):
        raise ValueError("Unsupported physical randomization mode")
    # These are independent standing resets, never AMP reference states.
    for name in ("init_motion_loader", "reset_from_motion", "randomize_terrain", "push_robot", "reset_hiking_state"):
        env_cfg.events.pop(name, None)
    env_cfg.events["reset_base"] = EventTermCfg(
        func=mdp.reset_root_state_uniform, mode="reset",
        params={"pose_range": {}, "velocity_range": {}},
    )
    env_cfg.events["reset_robot_joints"] = EventTermCfg(
        func=mdp.reset_joints_by_offset, mode="reset",
        params={"position_range": (0., 0.), "velocity_range": (0., 0.)},
    )
    if domain_randomization == "nominal":
        for name in ("foot_friction", "encoder_bias", "base_com", "physics_material"):
            env_cfg.events.pop(name, None)
    for group in env_cfg.observations.values():
        group.enable_corruption = False
    env_cfg.curriculum = {}
    # A test completion is distinct from an environment timeout.
    env_cfg.episode_length_s = max(env_cfg.episode_length_s, duration + 1.)
    if terrain == "flat":
        generator = copy.deepcopy(env_cfg.scene.terrain.terrain_generator)
        template = next(iter(generator.sub_terrains.values()))
        generator.sub_terrains = {"zero_start_flat": HikingTerrainCfg(
            family="recovery_flat", proportion=1., wall_probability=0.,
            flat_patch_sampling=copy.deepcopy(template.flat_patch_sampling),
            collision_horizontal_scale=template.collision_horizontal_scale,
            task_protocol=template.task_protocol,
        )}
        # Retain actual heightfield collision/raycast/oracle code. All tiles
        # and the surrounding plane are flat, with no unseen terrain seams.
        generator.size = (12., 12.)
        generator.num_rows, generator.num_cols = 2, 2
        generator.curriculum = False
        env_cfg.scene.terrain.terrain_generator = generator
        env_cfg.terminations["terrain_boundary"] = TerminationTermCfg(
            func=outside_evaluation_patch, time_out=True, params={"half_extent":6., "margin":.1})
    return env_cfg


def initial_state_audit(env, controller, frame):
    """Fail the experiment unless its physical and history contract holds."""
    import torch
    robot, adapter = env.scene["robot"], env.adapter
    idx, data = robot.indexing, robot.data.data
    root = data.qpos[:, idx.free_joint_q_adr.long()]
    expected_root = robot.data.default_root_state[:, :7].clone()
    expected_root[:, :3] += env.scene.env_origins
    joints = data.qpos[:, idx.joint_q_adr.long()]
    qvel = torch.cat((data.qvel[:, idx.free_joint_v_adr.long()], data.qvel[:, idx.joint_v_adr.long()]), -1)
    history = actor_history_frames(env.obs_buf["actor"])
    errors = {
        "root_qpos_max_abs_error": (root - expected_root).abs().amax(),
        "joint_qpos_max_abs_error": (joints - robot.data.default_joint_pos).abs().amax(),
        "raw_qvel_max_abs": qvel.abs().amax(),
        "previous_action_max_abs": env.action_manager.action.abs().amax(),
        "actor_history_repeat_max_abs_error": (history - history[:, :1]).abs().amax(),
        "actor_history_joint_velocity_max_abs": history[:, :, 38:67].abs().amax(),
        "actor_history_previous_action_max_abs": history[:, :, 67:96].abs().amax(),
        "depth_history_repeat_max_abs_error": (controller.history.buffer - frame[:, None]).abs().amax(),
    }
    result = {name: float(value) for name, value in errors.items()}
    if not all(value == 0. for value in result.values()):
        raise RuntimeError(f"Zero-start initial state/history verification failed: {result}")
    expected_frame=adapter.actor_frame(adapter.state(),env.action_manager.action,
                                       adapter.command(),adapter.context(),noise=False)
    frame_error=(history[:,-1]-expected_frame).abs().amax()
    result["actor_frame_max_abs_error"]=float(frame_error)
    if float(frame_error)>1e-5:
        raise RuntimeError(f"Initial actor frame differs from checkpoint proprioception/scaling: {result}")
    if not bool(torch.isfinite(env.obs_buf["actor"]).all() & torch.isfinite(frame).all()):
        raise RuntimeError("Initial observations contain non-finite values")
    return result, qvel


def score_trace(trace, initial_state, *, case, dt, wait_seconds, thresholds, bad_height, bad_up_z,
                initial_body_pose=None):
    """Score pre-reset physical successors; later auto-reset states never count."""
    import numpy as np
    state = trace["next_state"]
    valid = trace["first_episode_valid"].astype(bool)
    terminated, timeout = trace["terminated"], trace["timeout"]
    steps, n = valid.shape
    initial_pose=initial_state[:,:7] if initial_body_pose is None else initial_body_pose
    pose=trace.get("evaluation_body_pose",state[:,:,:7])
    q = initial_pose[:, 3:7]
    yaw = np.arctan2(2*(q[:,0]*q[:,3] + q[:,1]*q[:,2]), 1-2*(q[:,2]**2+q[:,3]**2))
    axis = np.stack((np.cos(yaw), np.sin(yaw)), -1)
    side = np.stack((-np.sin(yaw), np.cos(yaw)), -1)
    delta = pose[:, :, :2] - initial_pose[None, :, :2]
    displacement = (delta * axis[None]).sum(-1)
    lateral = (delta * side[None]).sum(-1)
    velocity = trace.get("evaluation_body_linear_velocity",state[:,:,7:10])[:,:,:2]
    forward = (velocity * axis[None]).sum(-1)
    speed = np.linalg.norm(velocity, axis=-1)
    current_q = pose[:, :, 3:7]
    heading = np.arctan2(2*(current_q[:,:,0]*current_q[:,:,3] + current_q[:,:,1]*current_q[:,:,2]),
                        1-2*(current_q[:,:,2]**2+current_q[:,:,3]**2))
    yaw_delta = np.arctan2(np.sin(heading-yaw[None]), np.cos(heading-yaw[None]))
    tilt = np.degrees(np.arccos(np.clip(trace["up_z"], -1, 1)))
    is_start = case in ("start", "immediate_start")
    wait_steps = (0 if case == "immediate_start" else steps_for(wait_seconds, dt)) if is_start else steps
    stand_mask = valid & (np.arange(steps)[:, None] < wait_steps)
    walk_mask = valid & (np.arange(steps)[:, None] >= wait_steps)
    sustained_steps = max(1, math.ceil(thresholds.reach_speed_sustained_seconds / dt - 1e-9))
    final_steps = max(1, math.ceil(thresholds.final_speed_window_seconds / dt - 1e-9))
    per_env = []
    for i in range(n):
        mask, smask, wmask = valid[:, i], stand_mask[:, i], walk_mask[:, i]
        ids = np.flatnonzero(mask)
        fallen = bool(np.any(mask & ((trace["relative_height"][:, i] < bad_height) | (trace["up_z"][:, i] < bad_up_z))))
        done = bool(np.any(mask & (terminated[:, i] | timeout[:, i])))
        survived = not done and len(ids) == steps
        stand_drift = float(np.linalg.norm(delta[smask, i], axis=-1).max()) if smask.any() else 0.
        stand_speed = float(speed[smask, i].mean()) if smask.any() else 0.
        stand_yaw_drift = float(np.degrees(np.abs(yaw_delta[smask, i])).max()) if smask.any() else 0.
        angular_velocity=trace.get("evaluation_body_angular_velocity",state[:,:,10:13])
        stand_yaw_rate = float(np.abs(angular_velocity[smask, i, 2]).mean()) if smask.any() else 0.
        max_tilt = float(tilt[mask, i].max()) if mask.any() else None
        stand_tilt = float(tilt[smask, i].max()) if smask.any() else 0.
        standing_complete = bool(smask.sum() == wait_steps)
        standing_survived = standing_complete and not bool(np.any(smask & (terminated[:, i] | timeout[:, i])))
        standing_pass = (standing_survived and stand_drift <= thresholds.stand_max_xy_drift_m and
                         stand_speed <= thresholds.stand_mean_xy_speed_m_s and
                         stand_tilt <= thresholds.max_base_tilt_degrees)
        row = dict(env_id=i, observed_steps=int(mask.sum()), observed_seconds=float(mask.sum()*dt),
            survived=survived, fallen=fallen, terminated=bool(np.any(mask & terminated[:, i])),
            timeout=bool(np.any(mask & timeout[:, i])), stand_max_xy_drift_m=stand_drift,
            stand_mean_xy_speed_m_s=stand_speed, stand_max_base_tilt_degrees=stand_tilt,
            stand_max_yaw_drift_degrees=stand_yaw_drift, stand_mean_abs_yaw_rate_rad_s=stand_yaw_rate,
            standing_window_complete=standing_complete, standing_window_survived=standing_survived,
            max_base_tilt_degrees=max_tilt, standing_window_pass=standing_pass)
        if is_start:
            # Position immediately before the command change, never a reset.
            origin = initial_pose[i, :2] if wait_steps == 0 else pose[min(wait_steps-1, steps-1), i, :2]
            progress = float(((pose[ids[-1], i, :2]-origin)*axis[i]).sum()) if len(ids) and ids[-1] >= wait_steps else 0.
            max_side = float(np.abs(((pose[wmask, i, :2]-origin)*side[i]).sum(-1)).max()) if wmask.any() else 0.
            final_mask = wmask & (np.arange(steps) >= steps-final_steps)
            final_speed = float(forward[final_mask, i].mean()) if final_mask.any() else None
            clock, reach = 0, None
            for j in range(wait_steps, steps):
                clock = clock + 1 if wmask[j] and forward[j, i] >= thresholds.reach_speed_m_s else 0
                if clock >= sustained_steps:
                    reach = float((j-wait_steps+1)*dt)
                    break
            row.update(start_forward_displacement_m=progress, start_max_lateral_displacement_m=max_side,
                start_mean_forward_speed_m_s=float(forward[wmask, i].mean()) if wmask.any() else None,
                final_forward_speed_m_s=final_speed, reach_speed_seconds=reach)
            row["success"] = bool(survived and not fallen and standing_pass and
                max_tilt <= thresholds.max_base_tilt_degrees and
                progress >= thresholds.start_min_forward_displacement_m and
                max_side <= thresholds.start_max_lateral_displacement_m and
                final_speed is not None and thresholds.start_final_forward_speed_min_m_s <= final_speed <= thresholds.start_final_forward_speed_max_m_s)
        else:
            row["success"] = bool(survived and not fallen and standing_pass)
        per_env.append(row)
    def mean(key):
        values = [row[key] for row in per_env if row[key] is not None]
        return float(np.mean(values)) if values else None
    result = dict(case=case, num_envs=n, control_dt_seconds=dt, steps_executed=steps,
        first_episode_survival_fraction=mean("survived"), fall_episode_fraction=mean("fallen"),
        success_fraction=mean("success"), stand_max_xy_drift_m_mean=mean("stand_max_xy_drift_m"),
        stand_mean_xy_speed_m_s=mean("stand_mean_xy_speed_m_s"),
        stand_max_yaw_drift_degrees_mean=mean("stand_max_yaw_drift_degrees"),
        stand_mean_abs_yaw_rate_rad_s=mean("stand_mean_abs_yaw_rate_rad_s"),
        max_base_tilt_degrees_mean=mean("max_base_tilt_degrees"), per_env=per_env)
    if is_start:
        for key in ("start_forward_displacement_m", "start_max_lateral_displacement_m", "start_mean_forward_speed_m_s", "final_forward_speed_m_s", "reach_speed_seconds"):
            result[key+"_mean"] = mean(key)
        result["reach_speed_fraction"] = sum(row["reach_speed_seconds"] is not None for row in per_env)/n
    return result


def evaluate_case(actor, config, args, *, case, seed, output):
    import numpy as np
    import torch
    from scripts.train_wmac_perceptive import make_environment
    from src.tasks.amp_perceptive.sensors import read_depth_frame
    cfg = copy.deepcopy(config)
    cfg.num_envs, cfg.seed, cfg.device = args.num_envs, seed, args.device
    cfg.cuda_graphs, cfg.compile = False, False
    wait_seconds = 0. if case == "immediate_start" else args.wait_seconds
    seconds = args.stand_seconds if case == "stand" else wait_seconds + args.walk_seconds
    transform = lambda env_cfg: configure_zero_start(env_cfg, terrain=args.terrain,
        domain_randomization=args.domain_randomization, duration=seconds)
    seed_all(seed, args.device)
    env = make_environment(cfg, play=True, mode="walk", env_config_transform=transform)
    try:
        _assert_no_noise_or_push(env)
        schedule = ScheduledCommand(env, case, args.wait_seconds, args.forward_speed)
        seed_all(seed, args.device)
        env.reset(seed=seed)
        current_frame = read_depth_frame(env)
        controller = PolicyHistory(actor, env.num_envs, env.device, cfg.action_limit)
        controller.reset(current_frame)
        audit, qvel = initial_state_audit(env, controller, current_frame)
        schedule.assert_observation(env.obs_buf["actor"])
        adapter, dt = env.adapter, float(env.step_dt)
        initial = dict(state=adapter.state().detach().cpu().numpy(),
            raw_qvel=qvel.detach().cpu().numpy(), context=adapter.context().detach().cpu().numpy(),
            actor_history=env.obs_buf["actor"].detach().cpu().numpy(), depth_frame=current_frame.detach().cpu().numpy(),
            terrain_type=env.scene.terrain.terrain_types.detach().cpu().numpy(),
            terrain_level=env.scene.terrain.terrain_levels.detach().cpu().numpy())
        if getattr(adapter,"full_hiking",False):
            p,q,_,_=adapter.kinematics(adapter.state())
            initial["evaluation_body_pose"]=torch.cat((p[:,adapter.anchor_id],q[:,adapter.anchor_id]),-1).detach().cpu().numpy()
        digest = hashlib.sha256()
        for name, value in sorted(initial.items()):
            digest.update(name.encode()); digest.update(np.ascontiguousarray(value).tobytes())
        output.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output / "initial.npz", **initial)
        steps = steps_for(seconds, dt)
        active = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
        rows = []
        began = time.perf_counter()
        with torch.inference_mode():
            for _ in range(steps):
                schedule.assert_observation(env.obs_buf["actor"], active)
                expected = actor_history_frames(env.obs_buf["actor"])[:, -1, 6:9].clone()
                action = controller.action(env.obs_buf["actor"])
                env.step(action)
                transition = env.transition
                if not torch.equal(transition["command"], expected):
                    raise RuntimeError("Captured physical command differs from the command used by Actor")
                state = transition["next_state"]
                if not bool(torch.isfinite(state[active]).all()):
                    raise RuntimeError("Non-finite first-episode physical state")
                quat = state[:, 3:7]
                row = {key: transition[key].detach().clone() for key in
                    ("state", "next_state", "action", "command", "terminated", "timeout")}
                height=adapter.relative_height(state)
                if getattr(adapter,"full_hiking",False):
                    p,q,v,w=adapter.kinematics(state);torso=adapter.anchor_id
                    quat=q[:,torso]
                    height=p[:,torso,2]-transition["context"][:,37].clamp(max=0.)
                    row.update(evaluation_body_pose=torch.cat((p[:,torso],quat),-1).detach().clone(),
                        evaluation_body_linear_velocity=v[:,torso].detach().clone(),
                        evaluation_body_angular_velocity=w[:,torso].detach().clone())
                row.update(first_episode_valid=active.clone(),
                    relative_height=height.detach().clone(),
                    up_z=(1-2*(quat[:, 1].square()+quat[:, 2].square())).detach().clone())
                rows.append(row)
                active &= ~(transition["terminated"] | transition["timeout"])
                controller.after_step(transition, read_depth_frame(env))
        if str(args.device).startswith("cuda"):
            torch.cuda.synchronize()
        trace = {key: torch.stack([row[key] for row in rows]).cpu().numpy() for key in rows[0]}
        trace["time_seconds"] = (np.arange(steps)+1)*dt
        np.savez_compressed(output / "trace.npz", **trace)
        result = score_trace(trace, initial["state"], case=case, dt=dt,
            wait_seconds=wait_seconds, thresholds=SuccessThresholds(),
            bad_height=adapter.height_threshold, bad_up_z=adapter.tilt_cos,
            initial_body_pose=initial.get("evaluation_body_pose"))
        result.update(seed=seed, terrain=args.terrain, domain_randomization=args.domain_randomization,
            duration_seconds=seconds, command_schedule={"wait_seconds":wait_seconds if case != "stand" else seconds,
                "walk_seconds":args.walk_seconds if case != "stand" else 0.,
                "standing_command":[0., 0., 0.], "walking_command":[args.forward_speed, 0., 0.]},
            thresholds=asdict(SuccessThresholds()),
            physical_fall_thresholds={"relative_height_m":adapter.height_threshold, "up_z":adapter.tilt_cos},
            initial_audit=audit, initial_sha256=digest.hexdigest(),
            commands_verified_all_steps=True,
            terrain_geometry={"tile_size_m":[12.,12.] if args.terrain == "flat" else [8.,8.],
                "boundary_half_extent_m":6. if args.terrain == "flat" else 4.},
            depth_timing=("fresh_successor_with_pre_reset_terminal" if getattr(adapter,"full_hiking",False)
                          else "legacy_pre_sense_transition_capture"),
            evaluation_body="torso_link" if getattr(adapter,"full_hiking",False) else "pelvis",
            reset_pose_protocol="local_model_default_standing_pose_with_zero_qvel",
            training_reset_pose_protocol=("source_torso_height_0.9m_with_randomized_pose_joints_and_velocity"
                                         if getattr(adapter,"full_hiking",False) else "checkpoint_training_reset"),
            physical_randomization_events=[name for name in ("foot_friction", "encoder_bias", "base_com", "physics_material") if name in env.cfg.events],
            elapsed_seconds=time.perf_counter()-began,
            initial=str(output / "initial.npz"), trace=str(output / "trace.npz"))
        (output / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
        return result
    finally:
        env.close()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--cases", nargs="+", choices=("stand", "start", "immediate_start"), default=["stand", "start", "immediate_start"])
    p.add_argument("--num-envs", type=int, default=32)
    seeds = p.add_mutually_exclusive_group()
    seeds.add_argument("--seed", type=int)
    seeds.add_argument("--seeds", nargs="+", type=int)
    p.add_argument("--terrain", choices=("flat", "training_spawn"), default="flat")
    p.add_argument("--domain-randomization", choices=("retained", "nominal"), default="retained")
    p.add_argument("--stand-seconds", type=float, default=10.)
    p.add_argument("--wait-seconds", type=float, default=2.)
    p.add_argument("--walk-seconds", type=float, default=8.)
    p.add_argument("--forward-speed", type=float, default=.5)
    p.add_argument("--output", type=Path)
    return p


def main(argv=None):
    import torch
    args = parser().parse_args(argv)
    if args.num_envs < 1 or any(not math.isfinite(x) or x <= 0 for x in (args.stand_seconds, args.walk_seconds, args.forward_speed)) or not math.isfinite(args.wait_seconds) or args.wait_seconds < 0:
        raise ValueError("Use positive environment count, durations and forward speed (wait may be zero)")
    args.seeds = args.seeds or [2032 if args.seed is None else args.seed]
    args.checkpoint = args.checkpoint.expanduser().resolve()
    output = (args.output or args.checkpoint.parent / "evaluations" /
        f"zero_start_{args.checkpoint.stem}_{datetime.now():%Y-%m-%d_%H-%M-%S}").expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    # Publish the criteria/protocol before executing any physical trial.
    manifest = dict(protocol="zero_default_standing_v1", checkpoint=str(args.checkpoint), device=args.device,
        terrain=args.terrain, domain_randomization=args.domain_randomization,
        seeds=args.seeds, cases=args.cases, num_envs=args.num_envs,
        thresholds=asdict(SuccessThresholds()),
        state_assistance="none; ordinary PD actions and gravity; no state writes after initial reset",
        command_history="native time-major four-frame history; schedule supplied before observation computation",
        metrics="first episode only; physical terminal successor included; auto-reset trajectories excluded")
    (output / "protocol.json").write_text(json.dumps(manifest, indent=2)+"\n")
    actor, cfg, payload = load_actor(args.checkpoint, args.device)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    results = []
    for seed in args.seeds:
        for case in args.cases:
            result = evaluate_case(actor, cfg, args, case=case, seed=seed, output=output / f"seed_{seed}" / case)
            results.append(result)
            print(json.dumps({key: result[key] for key in ("case", "seed", "first_episode_survival_fraction", "fall_episode_fraction", "success_fraction")}), flush=True)
            (output / "summary.json").write_text(json.dumps({**manifest, "results":results}, indent=2, allow_nan=False)+"\n")
    print(f"Results: {output / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
