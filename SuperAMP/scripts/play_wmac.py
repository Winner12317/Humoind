"""View a saved WMAC/AMP Actor in native MuJoCo, a browser, or a video."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime
import json
import math
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from evaluate_wmac import keep_fallen_recovery_frames, load_actor, seed_all


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--mode", choices=("walk", "recovery", "mixed"), default="mixed")
    parser.add_argument("--viewer", choices=("auto", "native", "viser", "none"), default="auto")
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=0, help="Stop after this many control steps; 0 keeps an interactive viewer running")
    parser.add_argument("--command-vx", type=float, default=0.5)
    parser.add_argument("--command-vy", type=float, default=0.0)
    parser.add_argument("--command-yaw", type=float, default=0.0)
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--observation-noise", action="store_true")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--output", type=Path, help="New directory for video and playback settings")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    for name in ("num_envs", "width", "height"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.steps < 0 or not 1 <= args.port <= 65535:
        parser.error("steps must be non-negative and port must be in 1..65535")
    if any(not math.isfinite(getattr(args, name)) for name in ("command_vx", "command_vy", "command_yaw")):
        parser.error("velocity commands must be finite")
    if args.viewer == "auto":
        args.viewer = "native" if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY") else "viser"
    if args.viewer == "none" and not args.steps:
        args.steps = 500
    args.checkpoint = args.checkpoint.expanduser().resolve()
    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint not found: {args.checkpoint}")
    os.environ["MUJOCO_GL"] = "glfw" if args.viewer == "native" else "egl"
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    if args.device.startswith("cuda") and "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device.partition(":")[2] or "0"
        args.device = "cuda:0"
    if args.device.startswith("cuda"):
        os.environ["MUJOCO_EGL_DEVICE_ID"] = args.device.partition(":")[2] or "0"

    import torch
    import mjlab.tasks  # noqa: F401
    import src.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg
    from mjlab.utils.torch import configure_torch_backends

    configure_torch_backends()
    actor, kind = load_actor(args.checkpoint, args.device)
    cfg = load_env_cfg(actor.amp_task_id, play=True)
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    cfg.viewer.width, cfg.viewer.height = args.width, args.height
    cfg.curriculum = {}
    if not args.push:
        cfg.events.pop("push_robot", None)
    else:
        training_cfg = load_env_cfg(actor.amp_task_id)
        cfg.events["push_robot"] = training_cfg.events["push_robot"]
    for group in cfg.observations.values():
        group.enable_corruption = args.observation_noise
    cfg.events["init_motion_loader"].params["delay_reset_env_ratio"] = {
        "walk": 0., "recovery": 1., "mixed": .4,
    }[args.mode]
    command = cfg.commands["twist"]
    command.rel_standing_envs = command.rel_heading_envs = 0.
    command.heading_command = False
    command.ranges.heading = None
    command.ranges.lin_vel_x = (args.command_vx,) * 2
    command.ranges.lin_vel_y = (args.command_vy,) * 2
    command.ranges.ang_vel_z = (args.command_yaw,) * 2
    command.resampling_time_range = (1e9, 1e9)

    output = args.output.expanduser().resolve() if args.output else (
        args.checkpoint.parent / "videos" /
        f"{datetime.now():%Y-%m-%d_%H-%M-%S_%f}_{args.checkpoint.stem}_{args.mode}"
    )
    if args.video or args.output:
        output.mkdir(parents=True, exist_ok=False)
        (output / "playback.json").write_text(json.dumps({
            "checkpoint_format": kind,
            "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "purpose": "visual playback, not the first-episode-only quantitative evaluation protocol",
        }, indent=2) + "\n")

    seed_all(args.seed)
    env = ManagerBasedRlEnv(cfg=cfg, device=args.device, render_mode="rgb_array" if args.video else None)
    try:
        if args.mode == "recovery":
            keep_fallen_recovery_frames(env)
        if args.video:
            from mjlab.utils.wrappers import VideoRecorder
            env = VideoRecorder(env, video_folder=output, step_trigger=lambda step: step == 0,
                video_length=args.steps or 500, name_prefix=f"{args.checkpoint.stem}_{args.mode}")
        seed_all(args.seed)
        env = RslRlVecEnvWrapper(env, clip_actions=getattr(actor, "source_action_limit", None))

        @torch.inference_mode()
        def policy(observations):
            return actor(observations["actor"])

        print(f"[play] {args.checkpoint} ({kind}); mode={args.mode}; viewer={args.viewer}; envs={args.num_envs}", flush=True)
        if args.viewer == "none":
            observations = env.get_observations()
            for _ in range(args.steps):
                action = policy(observations)
                if not torch.isfinite(action).all():
                    raise RuntimeError("Actor produced non-finite actions")
                observations, *_ = env.step(action)
        elif args.viewer == "native":
            from mjlab.viewer import NativeMujocoViewer
            NativeMujocoViewer(env, policy).run(num_steps=args.steps or None)
        else:
            import viser
            from mjlab.viewer import ViserPlayViewer
            # The installed joystick assumes positive range maxima, while our
            # fixed y/yaw commands legitimately have ranges (0, 0). Supply a
            # separate display range only during GUI creation; keep the actual
            # resampling ranges fixed for resets and physical rollout.
            term = env.unwrapped.command_manager.get_term("twist")
            create_gui = term.create_gui

            def create_fixed_command_gui(name, gui_server, get_env_idx):
                actual_ranges = term.cfg.ranges
                display_ranges = copy.copy(actual_ranges)
                for axis in ("lin_vel_x", "lin_vel_y", "ang_vel_z"):
                    extent = min(10., max(1., *(abs(v) for v in getattr(actual_ranges, axis))))
                    setattr(display_ranges, axis, (-extent, extent))
                try:
                    term.cfg.ranges = display_ranges
                    create_gui(name, gui_server, get_env_idx)
                finally:
                    term.cfg.ranges = actual_ranges

            term.create_gui = create_fixed_command_gui
            server = viser.ViserServer(host=args.host, port=args.port, label=f"WMAC {args.mode}")
            try:
                ViserPlayViewer(env, policy, viser_server=server).run(num_steps=args.steps or None)
            finally:
                server.stop()
    finally:
        env.close()
    if args.video:
        print(f"Videos: {output}")


if __name__ == "__main__":
    main()
