"""Play an isolated perceptive WMAC Actor with its real camera history."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_wmac_perceptive import PolicyHistory, load_actor, seed_all, restrict_recovery_starts


def make_history_vec_wrapper(env, actor, action_limit):
    """Wrap reset/step, not render ticks, so GUI redraw never advances history."""
    from mjlab.rl import RslRlVecEnvWrapper
    from tensordict import TensorDict
    from src.tasks.amp_perceptive.sensors import read_depth_frame

    class HistoryVecWrapper(RslRlVecEnvWrapper):
        def __init__(self, raw_env):
            super().__init__(raw_env, clip_actions=action_limit)
            self.policy_history = PolicyHistory(actor, self.num_envs, self.device, action_limit)
            self.policy_history.reset(read_depth_frame(self.unwrapped))

        def reset(self):
            result = super().reset()
            if hasattr(self, "policy_history"):
                self.policy_history.reset(read_depth_frame(self.unwrapped))
            return result

        def step(self, action):
            result = super().step(action)
            self.policy_history.after_step(self.unwrapped.transition, read_depth_frame(self.unwrapped))
            return result

        def get_observations(self):
            # The environment already computed these after the last physical
            # step. A viewer request must not append an extra proprio frame.
            return TensorDict(self.unwrapped.obs_buf, batch_size=[self.num_envs])

    return HistoryVecWrapper(env)


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--checkpoint", type=Path, required=True)
    value.add_argument("--mode", choices=("auto", "walk", "recovery", "mixed"), default="auto")
    value.add_argument("--device", default="cpu")
    value.add_argument("--num-envs", type=int, help="Defaults to the saved training protocol's playback count")
    value.add_argument("--seed", type=int, default=2033)
    value.add_argument("--viewer", choices=("auto", "native", "viser", "none"), default="auto")
    value.add_argument("--steps", type=int, default=0, help="0 runs an interactive viewer until closed")
    value.add_argument("--host", default="127.0.0.1")
    value.add_argument("--port", type=int, default=8081)
    value.add_argument("--output", type=Path, help="New directory for playback protocol; no training state is modified")
    return value


def main(argv=None):
    argparser = parser(); args = argparser.parse_args(argv)
    if (args.num_envs is not None and args.num_envs < 1) or args.steps < 0 or args.seed < 0 or not 1 <= args.port <= 65535:
        argparser.error("Invalid environment count, steps, seed or port")
    args.checkpoint = args.checkpoint.expanduser().resolve()
    if not args.checkpoint.is_file(): argparser.error("Checkpoint does not exist")
    if args.viewer == "auto":
        args.viewer = "native" if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY") else "viser"
    if args.viewer == "none" and args.steps == 0: args.steps = 500
    os.environ["MUJOCO_GL"] = "glfw" if args.viewer == "native" else "egl"
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import torch
    from scripts.train_wmac_perceptive import make_environment
    actor, config, checkpoint = load_actor(args.checkpoint, args.device)
    if args.num_envs is None:
        args.num_envs = 10 if config.hiking_training_version >= 3 else 4
    if args.mode=="auto":args.mode="walk" if config.motion_scope=="forward_walk_turn_v1" else "mixed"
    if config.motion_scope=="forward_walk_turn_v1" and args.mode=="recovery":
        argparser.error("This locomotion checkpoint has no recovery mode")
    robot_variant = checkpoint["architecture"].get("robot_variant", "historical G1 MJCF")
    cfg = copy.deepcopy(config)
    cfg.num_envs, cfg.seed, cfg.device = args.num_envs, args.seed, args.device
    cfg.cuda_graphs, cfg.compile = False, False
    output = None
    if args.output:
        output = args.output.expanduser().resolve(); output.mkdir(parents=True,exist_ok=False)
        (output/"playback.json").write_text(json.dumps({
            "checkpoint":str(args.checkpoint), "mode":args.mode, "seed":args.seed,
            "viewer":args.viewer, "device":args.device, "num_envs":args.num_envs,
            "hiking_training_version":cfg.hiking_training_version, "robot_variant":robot_variant,
            "purpose":"visual playback, not quantitative first-episode evaluation",
            "history":"same physical depth/reset handling as perceptive training",
        },ensure_ascii=False,indent=2)+"\n")
    seed_all(args.seed, args.device)
    raw = make_environment(cfg, play=True, mode=None if args.mode == "mixed" else args.mode)
    env = raw
    try:
        if args.mode == "recovery": restrict_recovery_starts(raw)
        seed_all(args.seed, args.device)
        env = make_history_vec_wrapper(raw, actor, cfg.action_limit)

        @torch.inference_mode()
        def policy(observations):
            return env.policy_history.action(observations["actor"])

        print(f"[play-perceptive] {args.checkpoint}; mode={args.mode}; viewer={args.viewer}; robot={robot_variant}",flush=True)
        if args.viewer == "none":
            observations = env.get_observations()
            for _ in range(args.steps):
                observations, *_ = env.step(policy(observations))
        elif args.viewer == "native":
            from mjlab.viewer import NativeMujocoViewer
            NativeMujocoViewer(env, policy).run(num_steps=args.steps or None)
        else:
            import viser
            from mjlab.viewer import ViserPlayViewer
            # The inherited joystick divides by configured range maxima.
            # Navigation legitimately has vy=(0,0); use a GUI-only display
            # range while preserving the actual position-target commands.
            term = raw.command_manager.get_term("twist")
            create_gui = term.create_gui
            def create_navigation_gui(name, server, get_env_idx):
                actual = term.cfg.ranges
                display = copy.copy(actual)
                for axis in ("lin_vel_x","lin_vel_y","ang_vel_z"):
                    extent=max(1.,*(abs(x) for x in getattr(actual,axis)))
                    setattr(display,axis,(-extent,extent))
                try:
                    term.cfg.ranges=display
                    create_gui(name,server,get_env_idx)
                finally:
                    term.cfg.ranges=actual
            term.create_gui=create_navigation_gui
            server = viser.ViserServer(host=args.host,port=args.port,label="Perceptive WMAC walking")
            try:
                ViserPlayViewer(env,policy,viser_server=server).run(num_steps=args.steps or None)
            finally:
                server.stop()
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
