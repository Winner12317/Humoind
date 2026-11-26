"""First-episode physical evaluation of isolated perceptive WMAC checkpoints.

Only the Actor is constructed. The world model never supplies evaluation
images, contact costs, rewards or recovery events: every observation comes
from the physical environment and terrain-relative geometry.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, fields
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def seed_all(seed, device="cpu"):
    import numpy as np
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(seed)


def load_actor(checkpoint, device="cpu"):
    """Validate the sensor schema before constructing/loading only an Actor."""
    import torch
    from src.wmac_perceptive.config import FORMAT, config_from_payload, perception_spec_from_config, LEGACY_SENSOR_PROTOCOL
    from src.wmac_perceptive.checkpoints import architecture_metadata, build_perceptive_actor
    from src.wmac_perceptive.perception import PerceptionSpec
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("format") != FORMAT:
        raise ValueError("Expected amp-wmac-perceptive-v1; flat WMAC/PPO files do not contain a compatible visual policy")
    cfg = config_from_payload(payload)
    visual = payload.get("visual_spec")
    if isinstance(visual,dict) and "include_valid" not in visual and cfg.sensor_protocol==LEGACY_SENSOR_PROTOCOL:
        visual={**visual,"include_valid":True}
    if isinstance(visual, dict) and "depth_protocol" not in visual and cfg.hiking_training_version == 1:
        visual = {**visual, "depth_protocol": "hiking_v1"}
    names = {field.name for field in fields(PerceptionSpec)}
    if not isinstance(visual, dict) or set(visual) != names:
        raise ValueError("Perceptive checkpoint is missing a complete visual_spec")
    spec = PerceptionSpec(**visual)
    # This task version fixes the camera protocol. Width-compatible but
    # different stride/delay/crop must never silently produce a different policy.
    if asdict(spec) != asdict(perception_spec_from_config(cfg)):
        raise ValueError("Saved visual_spec differs from the supported perceptive task protocol")
    if payload.get("architecture") != architecture_metadata(cfg, spec):
        raise ValueError("Checkpoint architecture metadata conflicts with its config/visual_spec")
    if payload.get("amp_grounding") != cfg.ground_height_semantics:
        raise ValueError("Checkpoint terrain grounding metadata conflicts with its configuration")
    actor = build_perceptive_actor(payload, spec=spec)
    try:
        actor.load_state_dict(payload["actor"], strict=True)
    except (KeyError, RuntimeError) as error:
        raise ValueError("Perceptive Actor tensors do not match the saved visual architecture") from error
    if not all(bool(torch.isfinite(value).all()) for value in actor.state_dict().values()):
        raise ValueError("Perceptive Actor contains non-finite weights or statistics")
    actor = actor.to(device).eval()
    actor.precision = "float32"
    return actor, cfg, payload


class PolicyHistory:
    """One history update per physical step, including a separate reset image."""

    def __init__(self, actor, num_envs, device="cpu", action_limit=None):
        from src.wmac_perceptive.perception import DepthHistory
        self.actor, self.action_limit = actor, action_limit
        self.history = DepthHistory(actor.spec, num_envs, device=device)

    def reset(self, frame):
        self.history.reset(frame)

    def action(self, proprio):
        import torch
        from src.wmac_perceptive.perception import pack_actor_input
        action = self.actor(pack_actor_input(proprio, self.history.buffer, self.actor.spec))
        if not bool(torch.isfinite(action).all()):
            raise RuntimeError("Perceptive Actor produced non-finite physical actions")
        return action if self.action_limit is None else action.clamp(-self.action_limit, self.action_limit)

    def after_step(self, transition, post_reset_frame):
        self.history.append(transition["next_depth_frame"])
        done = transition["terminated"] | transition["timeout"]
        self.history.reset(post_reset_frame, done)


class FirstEpisodeMetrics:
    """Sustained terrain-relative recovery and independent tracking criteria."""

    def __init__(self, num_envs, dt, *, device="cpu", height=.65, up_z=.8,
                 stable_seconds=.5, continued_seconds=1., xy_tolerance=.5,
                 yaw_tolerance=.5, bad_height=.5, bad_up_z=.5):
        import torch
        if num_envs < 1 or not dt > 0:
            raise ValueError("Invalid physical evaluation dimensions")
        self.n, self.dt = num_envs, float(dt)
        self.height, self.up_z = height, up_z
        self.bad_height, self.bad_up_z = bad_height, bad_up_z
        self.xy_tolerance, self.yaw_tolerance = xy_tolerance, yaw_tolerance
        self.stable_steps = max(1, math.ceil(stable_seconds/dt-1e-9))
        self.continued_steps = max(1, math.ceil(continued_seconds/dt-1e-9))
        self.active = torch.ones(num_envs, device=device, dtype=torch.bool)
        self.recovered = torch.zeros_like(self.active)
        self.continued = torch.zeros_like(self.active)
        self.fallen = torch.zeros_like(self.active)
        self.terminated = torch.zeros_like(self.active)
        self.timeout = torch.zeros_like(self.active)
        self.stable_clock = torch.zeros(num_envs, device=device, dtype=torch.long)
        self.walk_clock = torch.zeros_like(self.stable_clock)
        self.observed = torch.zeros_like(self.stable_clock)
        self.recovery_time = torch.full((num_envs,), float("nan"), device=device)
        self.xy_sum = torch.zeros(num_envs, device=device)
        self.yaw_sum = torch.zeros_like(self.xy_sum)
        self.valid_sum = torch.zeros_like(self.xy_sum)
        self.progress_sum = torch.zeros_like(self.xy_sum)
        self.distance_sum = torch.zeros_like(self.xy_sum)
        self.step_index = 0

    def update(self, *, relative_height, up_z, xy_error, yaw_error, moving,
               terminated, timeout, sensor_valid, target_progress, target_distance):
        import torch
        valid = self.active.clone()  # Includes a terminal transition's real successor.
        done = terminated | timeout
        upright = (relative_height > self.height) & (up_z > self.up_z)
        self.observed += valid.long()
        self.xy_sum += xy_error*valid; self.yaw_sum += yaw_error*valid
        self.valid_sum += sensor_valid*valid; self.progress_sum += target_progress*valid
        self.distance_sum += target_distance*valid
        self.fallen |= valid & ((relative_height < self.bad_height) | (up_z < self.bad_up_z))
        previous_recovered = self.recovered.clone()
        self.stable_clock = torch.where(valid & upright & ~done, self.stable_clock+1, 0)
        first = valid & ~self.recovered & (self.stable_clock >= self.stable_steps)
        self.recovery_time[first] = (self.step_index+1)*self.dt
        self.recovered |= first
        # Counting starts after the recovery event, not during its preceding
        # upright window; both XY and yaw are independent constraints.
        good = (valid & previous_recovered & upright & ~done & moving &
                (xy_error < self.xy_tolerance) & (yaw_error < self.yaw_tolerance))
        self.walk_clock = torch.where(good, self.walk_clock+1, 0)
        self.continued |= self.walk_clock >= self.continued_steps
        self.terminated |= valid & terminated; self.timeout |= valid & timeout
        self.active &= ~done
        self.step_index += 1
        return {"first_episode_valid": valid, "relative_height": relative_height,
                "up_z": up_z, "xy_error": xy_error, "yaw_error": yaw_error,
                "recovered": self.recovered.clone(), "terminated": terminated,
                "timeout": timeout, "sensor_valid": sensor_valid,
                "target_progress": target_progress, "target_distance": target_distance}

    def summary(self, mode):
        total = max(1, int(self.observed.sum()))
        count = int(self.recovered.sum())
        result = {"mode": mode, "num_envs": self.n, "steps_executed": self.step_index,
            "first_episode_control_steps": int(self.observed.sum()),
            "first_episode_survival_fraction": float(self.active.float().mean()),
            "fall_episode_fraction": float(self.fallen.float().mean()),
            "velocity_xy_error_mean": float(self.xy_sum.sum()/total),
            "velocity_yaw_error_mean": float(self.yaw_sum.sum()/total),
            "sensor_valid_fraction_mean": float(self.valid_sum.sum()/total),
            "target_progress_meters_sum": float(self.progress_sum.sum()),
            "target_progress_meters_per_start": float(self.progress_sum.mean()),
            "target_distance_mean": float(self.distance_sum.sum()/total),
            "per_env": {"observed_steps": self.observed.cpu().tolist(),
                "terminated": self.terminated.cpu().tolist(), "timeout": self.timeout.cpu().tolist(),
                "target_progress_meters": self.progress_sum.cpu().tolist()}}
        if mode == "recovery":
            result.update(recovery_success_count=count, recovery_success_fraction=count/self.n,
                first_stable_recovery_seconds_mean=float(self.recovery_time[self.recovered].mean()) if count else None,
                success_then_walk_count=int(self.continued.sum()),
                success_then_walk_fraction_all_starts=float(self.continued.float().mean()))
            result["per_env"]["recovery_seconds"] = [None if not math.isfinite(x) else x for x in self.recovery_time.cpu().tolist()]
        return result


def restrict_recovery_starts(env):
    """Filter this environment's local reset pool; never mutate a singleton."""
    from src.wmac.state import quat_to_matrix
    manager = env._perceptive_motion_reset_manager
    path = env.cfg.events["init_motion_loader"].params["motion_dir"]
    if path not in manager.recovery_frames:
        raise ValueError("Recovery evaluation requires the original Recovery motion group")
    frames = manager.recovery_frames[path]
    fallen = (frames["root_pos"][:, 2] < .55) | (quat_to_matrix(frames["root_quat"])[:, 2, 2] < .5)
    if not bool(fallen.any()):
        raise ValueError("No genuinely fallen recovery start frames are available")
    manager.recovery_frames[path] = {key:value[fallen] for key,value in frames.items()}
    return int(fallen.sum())


def _quantitative_play_config(cfg):
    # The source PLAY configuration keeps observation noise for interactive
    # playback. The repeatable quantitative benchmark uses nominal readings.
    for group in cfg.observations.values():
        group.enable_corruption = False
    return cfg


def _assert_no_noise_or_push(env):
    if "push_robot" in env.cfg.events:
        raise ValueError("Quantitative perceptive evaluation requires pushes disabled")
    if any(group.enable_corruption for group in env.cfg.observations.values()):
        raise ValueError("Quantitative perceptive evaluation requires observation corruption disabled")


def evaluate_mode(actor, config, args, mode, output):
    import numpy as np
    import torch
    from scripts.train_wmac_perceptive import make_environment
    from src.tasks.amp_perceptive.sensors import read_depth_frame
    from src.wmac.state import heading_quat, quat_apply, quat_to_matrix
    from src.wmac_perceptive.config import G1_HIKING_CALIBRATED_SENSOR_PROTOCOL
    cfg = copy.deepcopy(config)
    cfg.num_envs, cfg.seed, cfg.device = args.num_envs, args.seed, args.device
    cfg.cuda_graphs, cfg.compile = False, False
    seed_all(args.seed, args.device)
    env = make_environment(cfg, play=True, mode=mode, env_config_transform=_quantitative_play_config)
    try:
        _assert_no_noise_or_push(env)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cudnn.allow_tf32 = False
        candidates = restrict_recovery_starts(env) if mode == "recovery" else None
        seed_all(args.seed, args.device)
        observations, _ = env.reset(seed=args.seed)
        adapter = env.adapter
        current_frame = read_depth_frame(env)
        controller = PolicyHistory(actor, env.num_envs, env.device, cfg.action_limit)
        controller.reset(current_frame)
        initial = {"state": adapter.state().cpu().numpy(), "context": adapter.context().cpu().numpy(),
                   "command": adapter.command().cpu().numpy(), "actor_history": observations["actor"].cpu().numpy(),
                   "depth_frame": current_frame.cpu().numpy()}
        term = env.command_manager.get_term("twist")
        initial.update(navigation_target=term.target_pos_w.detach().cpu().numpy(),
                       desired_speed=term.desired_speed.detach().cpu().numpy(),
                       terrain_type=env.scene.terrain.terrain_types.detach().cpu().numpy(),
                       terrain_level=env.scene.terrain.terrain_levels.detach().cpu().numpy())
        digest = hashlib.sha256()
        for name, array in sorted(initial.items()):
            digest.update(name.encode()); digest.update(np.ascontiguousarray(array).tobytes())
        np.savez_compressed(output/f"initial_{mode}.npz", **initial)
        dt = float(env.step_dt)
        metrics = FirstEpisodeMetrics(env.num_envs, dt, device=env.device,
            bad_height=adapter.height_threshold, bad_up_z=adapter.tilt_cos)
        rows = []
        tracking={name:{"steps":0,"xy_error":0.,"yaw_error":0.} for name in ("forward_walk","in_place_turn")}
        if args.device.startswith("cuda"): torch.cuda.synchronize()
        began = time.perf_counter()
        with torch.inference_mode():
            for _ in range(args.steps):
                action = controller.action(env.obs_buf["actor"])
                env.step(action)
                transition = env.transition
                state, previous = transition["next_state"], transition["state"]
                _, rotations, velocities, angular = adapter.kinematics(state)
                torso = adapter.anchor_id
                command = transition["command"]
                desired = quat_apply(heading_quat(rotations[:, torso]),
                                    torch.cat((command[:, :2], torch.zeros_like(command[:, :1])), -1))
                target = transition["navigation_target"][:, :2]
                before = (previous[:, :2]-target).norm(dim=-1)
                distance = (state[:, :2]-target).norm(dim=-1)
                row = metrics.update(relative_height=adapter.relative_height(state),
                    up_z=quat_to_matrix(state[:, 3:7])[:, 2, 2],
                    xy_error=(desired[:, :2]-velocities[:, torso, :2]).norm(dim=-1),
                    yaw_error=(command[:, 2]-angular[:, torso, 2]).abs(),
                    moving=command[:, :2].norm(dim=-1)>.1,
                    terminated=transition["terminated"], timeout=transition["timeout"],
                    sensor_valid=transition["next_depth_frame"][:, 1].mean((-2,-1)),
                    target_progress=torch.where(transition.get("desired_speed",command[:,:2].norm(dim=-1))>0,before-distance,0.), target_distance=distance)
                if cfg.motion_scope=="forward_walk_turn_v1":
                    turning=transition.get("desired_speed",command[:,:2].norm(dim=-1))==0
                    for name,mask in (("forward_walk",~turning),("in_place_turn",turning)):
                        valid=mask&row["first_episode_valid"]
                        tracking[name]["steps"]+=int(valid.sum())
                        tracking[name]["xy_error"]+=float(row["xy_error"][valid].sum())
                        tracking[name]["yaw_error"]+=float(row["yaw_error"][valid].sum())
                if args.save_trace:
                    row["action"] = action
                    rows.append({key:value.detach().cpu().numpy().copy() for key,value in row.items()})
                controller.after_step(transition, read_depth_frame(env))
                if not bool(metrics.active.any()): break
        if args.device.startswith("cuda"): torch.cuda.synchronize()
        result = metrics.summary(mode)
        if cfg.motion_scope=="forward_walk_turn_v1":
            result["motion_tracking"]={name:{"control_steps":item["steps"],
                "velocity_xy_error_mean":item["xy_error"]/item["steps"] if item["steps"] else None,
                "velocity_yaw_error_mean":item["yaw_error"]/item["steps"] if item["steps"] else None}
                for name,item in tracking.items()}
        result.update(steps_requested=args.steps, control_dt=dt, wall_seconds_rollout=time.perf_counter()-began,
                      initial_state_sha256=digest.hexdigest(), recovery_start_candidates=candidates,
                      physical_recovery_window_steps=adapter.max_delay_steps)
        pattern = env.scene["hiking_depth"].cfg.pattern
        result["camera_protocol"] = {"raw_width":pattern.width,"raw_height":pattern.height,
            "horizontal_fov_degrees":pattern.fovx,"vertical_fov_degrees":pattern.fovy,
            "body_relative_position":list(pattern.position),"body_relative_quaternion_wxyz":list(pattern.rotation),
            "alignment":"complete torso orientation including falls", "image_plane_depth_meters":[.1,actor.spec.max_depth]}
        if cfg.sensor_protocol == G1_HIKING_CALIBRATED_SENSOR_PROTOCOL:
            intrinsics = pattern.calibrated_intrinsics
            horizontal, vertical = intrinsics.fov_degrees
            result["camera_protocol"].update(protocol=cfg.sensor_protocol, ray_intrinsics=asdict(intrinsics),
                                            horizontal_fov_degrees=horizontal, vertical_fov_degrees=vertical)
        result["per_terrain_type"] = {}
        types = torch.as_tensor(initial["terrain_type"], device=env.device)
        for terrain_type in types.unique().tolist():
            selected = types == terrain_type
            samples = max(1,int(metrics.observed[selected].sum()))
            result["per_terrain_type"][str(terrain_type)] = {
                "starts":int(selected.sum()), "first_episode_control_steps":int(metrics.observed[selected].sum()),
                "survived_starts":int(metrics.active[selected].sum()),
                "recovery_successes":int(metrics.recovered[selected].sum()) if mode=="recovery" else None,
                "stable_continued_walk":int(metrics.continued[selected].sum()) if mode=="recovery" else None,
                "velocity_xy_error_mean":float(metrics.xy_sum[selected].sum()/samples),
                "target_progress_meters_sum":float(metrics.progress_sum[selected].sum())}
        if rows:
            trace = {key:np.stack([row[key] for row in rows]) for key in rows[0]}
            trace["step_index"] = np.arange(len(rows))
            trace["time_seconds"] = (trace["step_index"]+1)*dt
            path = output/f"trace_{mode}.npz"
            np.savez_compressed(path, **trace)
            result["trace"] = str(path)
        return result
    finally:
        env.close()


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--checkpoint", type=Path, required=True)
    value.add_argument("--mode", choices=("auto", "walk", "recovery", "both"), default="auto")
    value.add_argument("--device", default="cpu", help="Use cuda:0 explicitly for GPU physical evaluation")
    value.add_argument("--num-envs", type=int, default=32)
    value.add_argument("--steps", type=int, default=500)
    value.add_argument("--seed", type=int, default=2032)
    value.add_argument("--save-trace", action="store_true")
    value.add_argument("--output", type=Path)
    return value


def main(argv=None):
    argparser = parser(); args = argparser.parse_args(argv)
    if args.num_envs < 1 or args.steps < 1 or args.seed < 0:
        argparser.error("num-envs/steps must be positive and seed nonnegative")
    args.checkpoint = args.checkpoint.expanduser().resolve()
    if not args.checkpoint.is_file(): argparser.error("Checkpoint does not exist")
    os.environ.setdefault("MUJOCO_GL", "egl"); os.environ.setdefault("OMP_NUM_THREADS", "1")
    output = args.output.expanduser().resolve() if args.output else args.checkpoint.parent/"evaluations"/f"perceptive_{datetime.now():%Y-%m-%d_%H-%M-%S_%f}"
    output.mkdir(parents=True, exist_ok=False)
    actor, cfg, payload = load_actor(args.checkpoint, args.device)
    locomotion=cfg.motion_scope=="forward_walk_turn_v1"
    if args.mode=="auto":args.mode="walk" if locomotion else "both"
    if locomotion and args.mode in ("both","recovery"):
        argparser.error("Recovery was removed from this checkpoint's task; use --mode walk")
    result = {"format":"amp-wmac-perceptive-evaluation-v1", "checkpoint":str(args.checkpoint),
        "checkpoint_sha256":hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "protocol":{"seed":args.seed, "num_envs":args.num_envs, "steps":args.steps,
            "device":args.device, "actor_only":True, "inference_precision":"float32, TF32 disabled",
            "visual_spec":asdict(actor.spec), "observation_noise":False, "pushes":False,
            "history":"physical pre-reset next frame appended; reset environments refill from post-reset frame",
            "first_episode":"terminal transition included, all reset episodes excluded",
            "recovery":"relative pelvis height>.65 and up_z>.8 continuously for0.5s",
            "continued_walking":"after recovery, upright and XY/yaw velocity errors<.5 continuously for1s with translation command>.1",
            "ground":"same compiled terrain queried at pelvis/foot points; not environment origin height",
            "targets":"Hiking-style flat-patch position targets; per-step progress uses the target active for that action",
            "recovery_window":"keeps training max_delay_steps; evaluation does not extend the recovery clock",
            "migration_scope":"MuJoCo heightfield port and body-mounted depth; not a full Hiking IsaacLab/hardware replication",
            "sensor_difference":"separate valid/no-hit channels; no-hit depth1; camera remains body-mounted when fallen",
            "initial_state":"recorded physical arrays, not a complete simulator restart snapshot"},
        "source_ground_height_semantics":cfg.ground_height_semantics, "results":{}, "status":"running"}
    if locomotion:
        for key in ("recovery","continued_walking","recovery_window"):
            result["protocol"].pop(key,None)
        result["protocol"].update(motion_scope=cfg.motion_scope,termination="immediate fall; no get-up window",
            commands="forward walking 0.2-0.7m/s and left/right turns, including explicit in-place turns")
    if cfg.hiking_training_version >= 3:
        result["protocol"].update(robot="Hiking unshod G1",
            sensor_difference="Hiking depth-only at 50 Hz; one control step delay",
            commands="Hiking terrain-dependent walking, yaw and standing commands",
            termination="ordinary Hiking PLAY termination rules",
            targets="Hiking target and random-velocity terrain commands")
    try:
        for mode in (("walk", "recovery") if args.mode == "both" else (args.mode,)):
            result["results"][mode] = evaluate_mode(actor, cfg, args, mode, output)
            (output/"metrics.json").write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
        result["status"] = "completed"
    except BaseException as error:
        result["status"], result["error"] = "failed", {"type":type(error).__name__,"message":str(error)}
        raise
    finally:
        (output/"metrics.json").write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    print(output/"metrics.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
