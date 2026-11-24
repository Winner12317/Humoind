"""Why does a WMAC (SuperMimic + AMP) policy get up faster and less like the reference?

    python scripts/diagnose_recovery_style.py --checkpoint <wmac model_N.pt> [--ppo-checkpoint <ppo model_N.pt>]

Runs the checkpoint in its own training task (half the environments walking from
reference frames, half lying in fallen recovery frames, no pushes, no timeouts),
records the real transitions, and reports, separately for walking and recovery:

1. tempo      get-up duration, peak base vertical speed and joint speeds, against the
              get-up segments of the Recovery reference clip (and the PPO policy, if given);
2. style      the checkpoint's own discriminator score of real policy transitions and
              of expert pairs, and the resulting centered style reward;
3. model      world-model open-loop error along recorded actions (1, 7, 32 steps);
4. optimism   from the same real starts, the actor imagined in the world model versus
              what the same actor actually did: height and uprightness after 0.16 /
              0.32 / 0.64 s;
5. gradient   the actor objective split into task, style and terminal-value parts
              (src/wmac/diagnostics.objective_gradient_probe), gradient norms and cosines.

Nothing is trained. The PPO checkpoint (optional) is only run for tempo and is
scored by the WMAC checkpoint's discriminator, which is one judge, not ground truth.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))


def getup_segments(z, up_z, upright_height, fps, lying_height=0.35, hold=25):
    """(last lying index, first index of `hold` consecutive upright steps) per get-up."""
    import numpy as np

    upright = (z > upright_height) & (up_z > 0.8)
    segments, k = [], 0
    while k < len(z):
        if z[k] >= lying_height:
            k += 1
            continue
        j = k
        while j + hold <= len(z) and not upright[j:j + hold].all():
            j += 1
        if j + hold > len(z):
            break
        last = k + int(np.flatnonzero(z[k:j] < lying_height).max())
        segments.append((last, j))
        k = j + hold
    return segments


def tempo_rows(z, up_z, vz, joint_vel, upright_height, fps):
    import numpy as np

    rows = []
    for a, b in getup_segments(z, up_z, upright_height, fps):
        part = slice(a, b + 1)
        speed = np.abs(joint_vel[part])
        rows.append({"seconds": (b - a) / fps, "peak_vertical_speed": float(vz[part].max()),
                     "joint_speed_p95": float(np.percentile(speed, 95)), "joint_speed_max": float(speed.max())})
    return rows


def summarize(rows):
    import numpy as np

    if not rows:
        return None
    return {key: {"mean": float(np.mean([r[key] for r in rows])), "median": float(np.median([r[key] for r in rows]))}
            for key in rows[0]} | {"count": len(rows)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True, help="WMAC checkpoint (format amp-wmac-v1)")
    parser.add_argument("--ppo-checkpoint", type=Path, help="optional PPO-AMP checkpoint of the same robot, for tempo")
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=500, help="control steps recorded per environment")
    parser.add_argument("--batch-size", type=int, default=128, help="starts per phase for model/gradient probes")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=2031)
    parser.add_argument("--override", action="append", default=[], metavar="FIELD=JSON",
                        help="set a WmacConfig field for this run only, e.g. recovery_value_grad_scale=0.1")
    parser.add_argument("--output", type=Path, help="write the report as JSON")
    args = parser.parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    if args.device.startswith("cuda") and "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device.partition(":")[2] or "0"
        args.device = "cuda:0"

    import tempfile

    import numpy as np
    import torch

    import mjlab.tasks  # noqa: F401
    import src.tasks  # noqa: F401
    from mjlab.tasks.registry import load_env_cfg

    from evaluate_wmac import keep_fallen_recovery_frames, load_actor, seed_all
    from src.recovery_rewards import configure_recovery_height_reward
    from src.robot_profiles import profile_for_task
    from src.wmac.checkpoints import resolve_checkpoint_config
    from src.wmac.diagnostics import objective_gradient_probe
    from src.wmac.env import CaptureEnv
    from src.wmac.recovery_style import configure_recovery_style_env, resolve_motion_root
    from src.wmac.reward_integration import configure_smooth_reward_env
    from src.wmac.runner import WmacRunner
    from src.wmac.state import StateAdapter, joint_velocities, quat_to_matrix

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if payload.get("format") != "amp-wmac-v1":
        raise SystemExit("--checkpoint must be a WMAC checkpoint")
    cfg = resolve_checkpoint_config(payload)
    robot = profile_for_task(cfg.task)
    cfg.num_envs, cfg.replay_steps, cfg.device, cfg.replay_device = args.num_envs, 8, args.device, "cpu"
    cfg.compile, cfg.precision, cfg.evaluation_interval = False, "float32", 0
    cfg.resume, cfg.init_policy = None, None  # mirroring stays as trained: load() checks the protocol
    cfg.validation_env_fraction = 0.0
    cfg.actor_batch_size = cfg.wm_batch_size = cfg.discriminator_batch_size = args.batch_size
    # Try a setting on this checkpoint without training, e.g. the gradient split
    # under --override recovery_value_grad_scale=0.1.
    for item in args.override:
        key, _, value = item.partition("=")
        if not hasattr(cfg, key):
            raise SystemExit(f"--override: unknown configuration field {key!r}")
        setattr(cfg, key, json.loads(value))
    cfg.validate()

    # Half the environments walk from reference frames, half start lying (fallen recovery frames).
    env_cfg = load_env_cfg(cfg.task)
    configure_recovery_height_reward(env_cfg, cfg.recovery_height_reward)
    configure_smooth_reward_env(cfg, env_cfg)
    configure_recovery_style_env(cfg, env_cfg)
    reference_root = None
    if cfg.reference_motion_root:
        reference_root = resolve_motion_root(cfg.reference_motion_root, REPO)
        env_cfg.events["init_motion_loader"].params.update(
            motion_dir=str(reference_root / "WalkandRun"), recovery_dir=str(reference_root / "Recovery"))
        env_cfg.events["reset_from_motion"].params["motion_dir"] = str(reference_root / "WalkandRun")
    env_cfg.scene.num_envs, env_cfg.seed = args.num_envs, args.seed
    env_cfg.episode_length_s = 1e6
    env_cfg.events.pop("push_robot", None)
    env_cfg.curriculum = {}
    loader = env_cfg.events["init_motion_loader"].params
    loader["delay_reset_env_ratio"], loader["max_delay_steps"] = 0.5, args.steps + 1
    seed_all(args.seed)
    env = CaptureEnv(cfg=env_cfg, device=args.device)
    report = {"checkpoint": str(args.checkpoint), "task": cfg.task, "robot": robot.name,
              "iteration": payload.get("iteration")}
    try:
        env.adapter = StateAdapter(env, recovery_counter_semantics=cfg.recovery_counter_semantics)
        if reference_root is not None:
            env._wmac_motion_root = str(reference_root)
        keep_fallen_recovery_frames(env)
        with tempfile.TemporaryDirectory() as log_dir:
            runner = WmacRunner(env, cfg, Path(log_dir))
        # Networks and normalisers only. WmacRunner.load() would also insist on a
        # byte-identical mirrored-motion cache and optimizer state, which a
        # diagnostic on another machine neither has nor needs.
        for name in ("actor", "critic", "world_model", "discriminator", "amp_norm"):
            getattr(runner, name).load_state_dict(payload[name])
        runner._set_amp_norm_settings(payload.get("amp_normalizer_settings", {}))
        if "amp_reward_state" in payload:
            runner.amp_reward_stats.load_state_dict(payload["amp_reward_state"])
        if payload.get("critic_value_upper_bound") is not None:
            runner._critic_vmax = payload["critic_value_upper_bound"]
        actor = runner.actor.eval()
        limit = cfg.action_limit
        dt = env.step_dt
        joints = runner.num_joints

        def record(policy):
            """Roll `policy` in the environment; returns per-step tensors [T, N, ...]."""
            seed_all(args.seed)
            observations, _ = env.reset(seed=args.seed)
            keys = ("state", "context", "command", "previous_action", "actor_history", "action", "next_state", "boundary")
            log = {key: [] for key in keys}
            with torch.no_grad():
                for _ in range(args.steps):
                    action = policy(observations["actor"]).clamp(-limit, limit)
                    observations, *_ = env.step(action)
                    for key in keys:
                        log[key].append(env.transition[key].detach().cpu())
            return {key: torch.stack(value) for key, value in log.items()}

        def phase(data):
            return (data["state"][..., 2] < robot.recovery_height) | (data["context"][..., -1] > 0)

        data = record(actor)
        recovery_mask = phase(data)
        report["recorded"] = {"steps": args.steps, "envs": args.num_envs,
                              "recovery_share": float(recovery_mask.float().mean())}

        # 1. tempo
        def episodes_tempo(states):
            rows = []
            up = quat_to_matrix(states[..., 3:7])[..., 2, 2].numpy()
            for n in range(states.shape[1]):
                s = states[:, n].numpy()
                rows += tempo_rows(s[:, 2], up[:, n], s[:, 9], joint_velocities(states[:, n]).numpy(),
                                   robot.upright_height, 1 / dt)
            return rows
        recovery_dir = Path(loader["recovery_dir"])
        reference_rows = []
        for clip in sorted(recovery_dir.glob("*.npz")):
            with np.load(clip) as motion:
                states = torch.as_tensor(np.concatenate([motion["body_pos_w"][:, 0], motion["body_quat_w"][:, 0],
                                                         motion["body_lin_vel_w"][:, 0], motion["body_ang_vel_w"][:, 0],
                                                         motion["joint_pos"], motion["joint_vel"]], 1))[:, None]
                fps = float(np.asarray(motion["fps"]).reshape(-1)[0])
            up = quat_to_matrix(states[..., 3:7])[..., 2, 2].numpy()[:, 0]
            s = states[:, 0].numpy()
            reference_rows += tempo_rows(s[:, 2], up, s[:, 9], s[:, 13 + joints:], robot.upright_height, fps)
        report["tempo"] = {"reference": summarize(reference_rows), "wmac": summarize(episodes_tempo(data["state"]))}
        ppo_data = None
        if args.ppo_checkpoint:
            ppo, kind = load_actor(args.ppo_checkpoint, args.device)
            ppo_limit = getattr(ppo, "source_action_limit", None)
            ppo_data = record(lambda obs: ppo(obs).clamp(-ppo_limit, ppo_limit) if ppo_limit else ppo(obs))
            report["tempo"]["ppo"] = summarize(episodes_tempo(ppo_data["state"]))

        # 2. style, under this checkpoint's discriminator
        def scores(first, second):
            out = []
            with torch.no_grad():
                for start in range(0, len(first), 8192):
                    reward, score = runner._amp_reward(first[start:start + 8192].to(args.device),
                                                       second[start:start + 8192].to(args.device))
                    out.append(torch.stack((score, reward), -1).cpu())
            return torch.cat(out)

        def real_scores(recorded):
            with torch.no_grad():
                first = runner.adapter.amp_features(recorded["state"].to(args.device)).cpu()
                second = runner.adapter.amp_features(recorded["next_state"].to(args.device)).cpu()
            valid = ~recorded["boundary"]
            mask = phase(recorded)
            return {"walk": scores(first[valid & ~mask], second[valid & ~mask]),
                    "recovery": scores(first[valid & mask], second[valid & mask])}
        groups = {"expert walk": runner.experts.groups[0], "expert recovery": runner.experts.groups[1]}
        rows = {name: scores(pairs[:, 0].cpu(), pairs[:, 1].cpu()) for name, pairs in groups.items() if pairs is not None}
        rows.update({f"wmac {k}": v for k, v in real_scores(data).items()})
        if ppo_data is not None:
            rows.update({f"ppo {k}": v for k, v in real_scores(ppo_data).items()})
        report["style"] = {name: {"score_mean": float(v[:, 0].mean()), "score_p10": float(v[:, 0].quantile(.1)),
                                  "score_p90": float(v[:, 0].quantile(.9)), "style_reward_mean": float(v[:, 1].mean()),
                                  "count": len(v)} for name, v in rows.items() if len(v)}
        report["style"]["reward_center"] = float(runner.amp_reward_stats.robot_mean)

        # Starts whose next 32 steps are recorded without a reset.
        horizon = cfg.imagination_horizon
        generator = torch.Generator().manual_seed(args.seed)
        boundary = data["boundary"]
        valid_start = torch.zeros_like(boundary)
        for t in range(args.steps - horizon):
            valid_start[t] = ~boundary[t:t + horizon].any(0)

        def starts(recovery: bool):
            candidates = torch.nonzero(valid_start & (recovery_mask == recovery))
            pick = candidates[torch.randperm(len(candidates), generator=generator)[:args.batch_size]]
            t, n = pick[:, 0], pick[:, 1]
            batch = {key: data[key][t, n].to(args.device) for key in
                     ("state", "context", "command", "previous_action", "actor_history")}
            future = torch.stack([data["state"][t + k, n] for k in range(horizon)] +
                                 [data["next_state"][t + horizon - 1, n]], 1)
            actions = torch.stack([data["action"][t + k, n] for k in range(horizon)], 1)
            contexts = torch.stack([data["context"][t + k, n] for k in range(horizon)], 1)
            return batch, future.to(args.device), actions.to(args.device), contexts.to(args.device)

        def up_z(states):
            return quat_to_matrix(states[..., 3:7])[..., 2, 2]

        report["model"], report["optimism"], report["gradient"] = {}, {}, {}
        for name, recovery in (("walk", False), ("recovery", True)):
            batch, future, actions, contexts = starts(recovery)
            if len(batch["state"]) < 8:
                continue
            # 3. open-loop error along recorded actions
            with torch.no_grad():
                state, errors = batch["state"], {}
                for k in range(horizon):
                    state = runner.world_model.step(state, actions[:, k], contexts[:, k])
                    if k + 1 in (1, min(7, horizon), horizon):
                        real = future[:, k + 1]
                        errors[k + 1] = {"height_cm": float((state[:, 2] - real[:, 2]).abs().mean() * 100),
                                         "up_z": float((up_z(state) - up_z(real)).abs().mean()),
                                         "joint_rmse_rad": float((state[:, 13:13 + joints] - real[:, 13:13 + joints])
                                                                 .square().mean().sqrt())}
                report["model"][name] = errors
                # 4. imagined vs real under the same actor (no observation noise)
                frame = 9 + 3 * joints
                rollout = runner.imagine(batch, noises=torch.zeros(horizon, len(batch["state"]), frame, device=args.device))
                x = rollout["x"]  # critic features of imagined states 0..H-1: height first, then heading-free wxyz
                imagined_height = x[..., 0]
                q = x[..., 1:5]
                imagined_up = 1 - 2 * (q[..., 1].square() + q[..., 2].square())
                rows = {}
                for k in sorted({min(8, horizon - 1), min(16, horizon - 1), horizon - 1}):
                    rows[round(k * dt, 2)] = {
                        "rise_imagined_cm": float((imagined_height[k] - imagined_height[0]).mean() * 100),
                        "rise_real_cm": float((future[:, k, 2] - future[:, 0, 2]).mean() * 100),
                        "up_z_imagined": float(imagined_up[k].mean()), "up_z_real": float(up_z(future[:, k]).mean())}
                rows["style_score_imagined"] = float(rollout["score"].mean())
                report["optimism"][name] = rows
            # 5. gradient split
            probe = objective_gradient_probe(runner, batch)
            report["gradient"][name] = {"components": {k: {"gradient_norm": v["gradient_norm"],
                                                           "objective_mean": v["objective_mean"]}
                                                       for k, v in probe["components"].items()},
                                        "cosines": probe["cosines"],
                                        "decomposition_relative_residual": probe["decomposition"]["gradient_relative_residual"]}
    finally:
        env.close()

    print_report(report)
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    return 0


def print_report(r: dict) -> None:
    print(f"\n{r['task']} iteration {r['iteration']}: {r['recorded']['envs']} envs x {r['recorded']['steps']} steps, "
          f"{100 * r['recorded']['recovery_share']:.0f} % of steps in recovery")
    print("\n1. get-up tempo (lying -> 0.5 s stably upright)")
    labels = {"seconds": "duration, s", "peak_vertical_speed": "peak base vertical speed, m/s",
              "joint_speed_p95": "joint speed p95, rad/s", "joint_speed_max": "joint speed max, rad/s"}
    columns = [c for c in ("reference", "wmac", "ppo") if r["tempo"].get(c)]
    print("   " + " " * 32 + "".join(f"{c:>18s}" for c in columns))
    for key, label in labels.items():
        print(f"   {label:32s}" + "".join(f"{r['tempo'][c][key]['mean']:>18.2f}" for c in columns))
    print("   " + " " * 32 + "".join(f"{'n=' + str(r['tempo'][c]['count']):>18s}" for c in columns))
    print(f"\n2. discriminator score (expert label +1, policy -1; style reward centred at {r['style']['reward_center']:+.2f})")
    for name, v in r["style"].items():
        if isinstance(v, dict):
            print(f"   {name:18s} score {v['score_mean']:+.2f} (p10 {v['score_p10']:+.2f}, p90 {v['score_p90']:+.2f})"
                  f"  style reward {v['style_reward_mean']:.4f}  n={v['count']}")
    print("\n3. world model open loop along recorded actions")
    for name, errors in r["model"].items():
        print(f"   {name:9s}" + "  ".join(f"H{h}: height {e['height_cm']:.1f} cm, up_z {e['up_z']:.3f}, joints {e['joint_rmse_rad']:.3f} rad"
                                       for h, e in errors.items()))
    print("\n4. imagined vs real, same actor and start")
    for name, rows in r["optimism"].items():
        print(f"   {name:9s}" + "  ".join(f"t+{t}s rise {v['rise_imagined_cm']:+.1f}/{v['rise_real_cm']:+.1f} cm, "
                                       f"up_z {v['up_z_imagined']:.2f}/{v['up_z_real']:.2f}"
                                       for t, v in rows.items() if isinstance(v, dict)) + "   (imagined/real)")
    print("\n5. actor gradient split of the imagined objective")
    for name, g in r["gradient"].items():
        c = g["components"]
        print(f"   {name:9s} |g| task {c['task']['gradient_norm']:.3g}, style {c['style']['gradient_norm']:.3g}, "
              f"terminal value {c['bootstrap']['gradient_norm']:.3g}; cos(total, task/style/value) "
              f"{g['cosines']['total_task']:+.2f}/{g['cosines']['total_style']:+.2f}/{g['cosines']['total_bootstrap']:+.2f}")


if __name__ == "__main__":
    sys.exit(main())
