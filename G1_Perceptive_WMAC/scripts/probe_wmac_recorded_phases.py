"""Probe matched recorded windows by physical phase without training.

Sampling and --plan-only are CPU-only. A full run uses one small environment
for immutable kinematic/reward metadata, not for new physical rollouts. Neither
an optimizer nor production replay is constructed. These are fixed-data model
and gradient sensitivities; they do not establish a historical causal failure.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
from zoneinfo import ZoneInfo

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
DEFAULT_ROOT = REPO / "logs/wmac_diagnostics/2026-11-09_walk_recovery_22000_vs_48000"
WIDTHS = {"state": 71, "next_state": 71, "action": 29, "previous_action": 29,
          "context": 35, "next_context": 35, "command": 3, "actor_history": 384,
          "contact_cost": 2, "reward": None, "terminated": None, "timeout": None,
          "boundary": None, "first_episode_valid": None}
BOOL_FIELDS = {"terminated", "timeout", "boundary", "first_episode_valid"}
PHASES = {"walk": ("initial", "upright"), "recovery": ("initial", "nonupright", "upright")}
OLD_GUARD_LIMIT = 250


def file_sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            result.update(block)
    return result.hexdigest()


def array_sha256(arrays):
    result = hashlib.sha256()
    for key, value in sorted(arrays.items()):
        array = np.ascontiguousarray(value)
        result.update(key.encode())
        result.update(str(array.dtype).encode())
        result.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        result.update(array.tobytes())
    return result.hexdigest()


def atomic_json(path, value):
    temporary = Path(path).with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read_recording(path):
    """Load first-episode transition arrays without accepting pickled data."""
    path = Path(path).expanduser().resolve()
    with np.load(path, allow_pickle=False) as archive:
        missing = set(WIDTHS) - set(archive.files)
        if missing:
            raise ValueError(f"Recording {path} is missing {sorted(missing)}")
        arrays = {key: np.array(archive[key], copy=True) for key in WIDTHS}
        protocol = json.loads(str(archive["recording_protocol_json"].item())) if "recording_protocol_json" in archive else None
    if arrays["state"].ndim != 3 or arrays["state"].shape[-1] != 71:
        raise ValueError("Recorded state must have shape [time,environment,71]")
    steps, environments = arrays["state"].shape[:2]
    if steps < 1 or environments < 1:
        raise ValueError("A recording must contain at least one time step and environment")
    for key, width in WIDTHS.items():
        expected = (steps, environments) if width is None else (steps, environments, width)
        if arrays[key].shape != expected:
            raise ValueError(f"Recording field {key} must have shape {expected}")
        if key in BOOL_FIELDS:
            if arrays[key].dtype.kind not in "bui" or not np.isin(arrays[key], (0, 1)).all():
                raise ValueError(f"Recording flag {key} must contain only booleans")
            arrays[key] = arrays[key].astype(bool, copy=False)
        elif arrays[key].dtype.kind != "f":
            raise ValueError(f"Recording field {key} must be real floating point")
        else:
            arrays[key] = arrays[key].astype(np.float32, copy=False)
    done = arrays["terminated"] | arrays["timeout"]
    if not np.all(arrays["boundary"] | ~done):
        raise ValueError("Recorded true terminals and timeouts must be marked as boundaries")
    active = np.ones(environments, dtype=bool)
    for step in range(steps):
        if not np.array_equal(arrays["first_episode_valid"][step], active):
            raise ValueError("First-episode validity must include the terminal step and exclude every reset episode")
        active &= ~done[step]
    return arrays, {"path": str(path), "sha256": file_sha256(path), "steps": steps,
                    "environments": environments, "protocol": protocol}


def upright_cosine(state):
    quaternion = state[..., 3:7]
    length = np.linalg.norm(quaternion, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        normalized = quaternion / length[..., None]
        up = 1 - 2 * (normalized[..., 1] ** 2 + normalized[..., 2] ** 2)
    return np.where(length > 1e-8, up, np.nan)


def eligible_starts(arrays, phase, length=32, max_start_step=200,
                    height_threshold=.65, upright_threshold=.8):
    """Return canonical [time,env] starts of complete, finite physical windows.

    Phase is defined at the start state, not across the entire window. A
    boundary is permitted only on its final transition, because no successor
    from a reset/push is then crossed. Windows containing any non-finite field
    or recovery counter reaching the original production guard are excluded.
    """
    if phase not in {"initial", "nonupright", "upright"}:
        raise ValueError(f"Unknown physical phase {phase}")
    if length < 1 or max_start_step < 1:
        raise ValueError("Window length and start limit must be positive")
    state = arrays["state"]
    steps, environments = state.shape[:2]
    end = min(max_start_step, steps - length + 1)
    if end < 1:
        return np.empty((0, 2), dtype=np.int64)
    finite = np.ones((steps, environments), dtype=bool)
    for key, value in arrays.items():
        if key not in BOOL_FIELDS:
            finite &= np.isfinite(value).reshape(steps, environments, -1).all(-1)
    finite &= np.isfinite(upright_cosine(state))
    finite &= arrays["context"][..., 34] < OLD_GUARD_LIMIT
    finite &= arrays["next_context"][..., 34] < OLD_GUARD_LIMIT
    valid = arrays["first_episode_valid"] & finite
    upright = (state[..., 2] >= height_threshold) & (upright_cosine(state) >= upright_threshold)
    candidates = []
    for time in range(end):
        if phase == "initial" and time != 0:
            continue
        good = valid[time:time + length].all(0)
        if length > 1:
            good &= ~arrays["boundary"][time:time + length - 1].any(0)
        if phase != "initial":
            good &= upright[time] if phase == "upright" else ~upright[time]
        candidates.extend((time, int(env)) for env in np.flatnonzero(good))
    return np.asarray(candidates, dtype=np.int64).reshape(-1, 2)


def sample_phase(arrays, phase, batch_size=64, length=32, max_start_step=200,
                 seed=9011, scarce_policy="replace"):
    if batch_size < 1 or scarce_policy not in {"replace", "skip"}:
        raise ValueError("Use a positive batch size and scarce policy replace or skip")
    candidates = eligible_starts(arrays, phase, length, max_start_step)
    count = len(candidates)
    metadata = {"phase": phase, "candidate_count": count, "requested_batch_size": batch_size,
                "window_length": length, "exclusive_start_step_limit": max_start_step,
                "seed": seed, "scarce_policy": scarce_policy,
                "phase_definition": "Start-state z>=0.65 and pelvis up_z>=0.8 means upright; initial means time zero."}
    if count == 0 or (count < batch_size and scarce_policy == "skip"):
        return None, {**metadata, "status": "skipped", "reason": "No complete candidates" if not count else "Fewer candidates than requested batch"}
    replace = count < batch_size
    selected = np.random.default_rng(seed).choice(count, size=batch_size, replace=replace)
    indices = candidates[selected]
    times = indices[:, 0, None] + np.arange(length)[None]
    environments = indices[:, 1, None]
    windows = {key: np.array(value[times, environments], copy=True)
               for key, value in arrays.items() if key != "first_episode_valid"}
    windows["mask"] = np.ones((batch_size, length), dtype=np.float32)
    up = upright_cosine(windows["state"])
    nonupright = (windows["state"][..., 2] < .65) | (up < .8)
    return windows, {**metadata, "status": "selected", "sample_with_replacement": replace,
        "unique_selected_starts": int(len(np.unique(indices, axis=0))), "indices_time_env": indices.tolist(),
        "window_sha256": array_sha256(windows), "sampled_time_min": int(indices[:, 0].min()),
        "sampled_time_max": int(indices[:, 0].max()),
        "nonupright_frame_fraction_inside_windows": float(nonupright.mean()),
        "maximum_recorded_recovery_counter": float(max(windows["context"][..., 34].max(), windows["next_context"][..., 34].max()))}


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    result.add_argument("--output", type=Path)
    result.add_argument("--data-seed", type=int, default=2030)
    result.add_argument("--recording-sources", nargs="+", choices=("22000", "48000", "ppo60000"),
                        default=["22000", "48000"])
    result.add_argument("--groups", nargs="+", help="Optional exact group labels, e.g. recording22000_recovery_nonupright")
    result.add_argument("--device", default="cuda:0")
    result.add_argument("--batch-size", type=int, default=64)
    result.add_argument("--gradient-batch-size", type=int, default=8)
    result.add_argument("--window", type=int, default=32)
    result.add_argument("--max-start-step", type=int, default=200)
    result.add_argument("--horizons", nargs="+", type=int, default=[1, 4, 8, 16, 32])
    result.add_argument("--seed", type=int, default=9011)
    result.add_argument("--scarce-policy", choices=("replace", "skip"), default="replace")
    result.add_argument("--skip-gradients", action="store_true")
    result.add_argument("--plan-only", action="store_true")
    return result


def source_snapshot(output):
    paths = [Path("scripts/probe_wmac_recorded_phases.py"), Path("scripts/check_wmac_physics.py"),
             Path("src/amp_features.py"), Path("src/recovery_rewards.py"),
             Path("src/tasks/amp_loco/mdp/rewards.py"),
             *sorted(path.relative_to(REPO) for path in (REPO / "src/wmac").glob("*.py"))]
    hashes = {}
    for relative in paths:
        content = (REPO / relative).read_bytes()
        hashes[str(relative)] = hashlib.sha256(content).hexdigest()
        target = output / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    return {"files_sha256": hashes, "git_commit": subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True, capture_output=True, check=False).stdout.strip()}


def make_runner(adapter, bundles, actor_source, wm_source, d_source, critic_source, device):
    from src.wmac.runner import WmacRunner

    result = object.__new__(WmacRunner)
    result.cfg = copy.deepcopy(bundles[actor_source][0])
    result.cfg.compile, result.cfg.precision, result.cfg.device = False, "tf32", str(device)
    result.device, result.adapter = device, adapter
    result.actor = bundles[actor_source][1]
    result.world_model = bundles[wm_source][2]
    result.critic = bundles[critic_source][3]
    result.discriminator, result.amp_norm = bundles[d_source][4:6]
    result._imag_step_fn = result._imagined_step
    result._reward_fn = result._trajectory_rewards
    result._value_fn = result.critic.forward
    return result


def component_map(actor, wm, discriminator, critic):
    return {"actor": actor, "world_model": wm, "discriminator_and_amp_norm": discriminator, "critic": critic}


def validate_bundles(bundles):
    from src.wmac.actors import actor_architecture
    reference = bundles[22000][0]
    keys = ("task", "amp_feature_schema", "model_hidden", "policy_hidden", "critic_hidden",
            "recovery_counter_semantics", "recovery_height_reward", "actor_type",
            "discriminator_hidden", "action_limit", "imagination_horizon", "task_weight", "style_weight",
            "amp_reward_coef", "gamma", "gae_lambda", "direct_style_grad", "critic_bootstrap", "survival_grad")
    for iteration, values in bundles.items():
        conflicts = [name for name in keys if getattr(values[0], name) != getattr(reference, name)]
        if actor_architecture(values[0]) != actor_architecture(reference):
            conflicts.append("actor_architecture")
        if conflicts:
            raise ValueError(f"Checkpoint {iteration} has incompatible semantics: {conflicts}")
    return reference


def run_forward(group_dir, windows, bundles, adapter, device, horizons):
    from src.wmac.diagnostics import open_loop_probe

    results = {}
    for model_source in (22000, 24000, 48000):
        views, cache = {}, {}
        for view, d_source in (("own_discriminator", model_source), ("fixed_D22000", 22000), ("fixed_D48000", 48000)):
            if d_source not in cache:
                runner = make_runner(adapter, bundles, model_source, model_source, d_source, model_source, device)
                cache[d_source] = open_loop_probe(runner, windows, horizons=tuple(horizons))
            views[view] = {"components": component_map(model_source, model_source, d_source, model_source),
                           "metrics": cache[d_source]}
        results[str(model_source)] = views
        atomic_json(group_dir / "forward.json", results)
        print(f"[recorded-phase] {group_dir.name} forward checkpoint {model_source}", flush=True)
    return {"status": "completed", "path": str(group_dir / "forward.json")}


def run_gradients(group_dir, windows, selection, bundles, adapter, device, seed, batch_size):
    import torch
    from src.wmac.diagnostics import objective_gradient_probe, safe_cosine

    size = min(batch_size, len(windows["state"]))
    fields = ("state", "previous_action", "context", "command", "actor_history")
    batch = {name: windows[name][:size, 0] for name in fields}
    horizon = bundles[22000][0].imagination_horizon
    generator = torch.Generator(device="cpu").manual_seed(seed)
    noise = 2 * torch.rand((horizon, size, 96), generator=generator) - 1
    scale = torch.zeros(96)
    scale[:3], scale[3:6], scale[9:38], scale[38:67] = .2, .05, .01, .5
    noises = (noise * scale).to(device)
    plan = (
        ("actor22000_baseline", 22000, 22000, 22000, 22000),
        ("actor22000_wm24000", 22000, 24000, 22000, 22000),
        ("actor22000_wm48000", 22000, 48000, 22000, 22000),
        ("actor22000_D48000", 22000, 22000, 48000, 22000),
        ("actor22000_critic48000", 22000, 22000, 22000, 48000),
        ("actor22000_all48000_aux", 22000, 48000, 48000, 48000),
        ("actor48000_baseline", 48000, 48000, 48000, 48000),
        ("actor48000_wm22000", 48000, 22000, 48000, 48000),
        ("actor48000_D22000", 48000, 48000, 22000, 48000),
        ("actor48000_critic22000", 48000, 48000, 48000, 22000),
        ("actor48000_all22000_aux", 48000, 22000, 22000, 22000),
    )
    metadata = {"batch_size": size, "indices_time_env": selection["indices_time_env"][:size],
                "starts_sha256": array_sha256({key: value.detach().cpu().numpy() for key, value in batch.items()}),
                "observation_noise_seed": seed,
                "observation_noise_sha256": array_sha256({"noise": noises.detach().cpu().numpy()}),
                "same_starts_and_noise_for_all_cases": True,
                "comparison": "Cosines are within each fixed-Actor family, before clipping/Adam; no optimizer or update occurs."}
    results, baselines = {}, {}
    for label, actor_source, wm_source, d_source, critic_source in plan:
        runner = make_runner(adapter, bundles, actor_source, wm_source, d_source, critic_source, device)
        result = objective_gradient_probe(runner, batch, noises=noises, return_gradients=True)
        gradient = result.pop("_gradients")["total"]
        if actor_source not in baselines:
            baselines[actor_source] = gradient.clone()
        result["components_source"] = component_map(actor_source, wm_source, d_source, critic_source)
        result["cosine_to_actor_family_baseline"] = safe_cosine(baselines[actor_source], gradient)
        results[label] = result
        atomic_json(group_dir / "gradients.json", {"sampling": metadata, "results": results})
        print(f"[recorded-phase] {group_dir.name} {label}", flush=True)
    return {"status": "completed", "path": str(group_dir / "gradients.json")}


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    for key in ("batch_size", "gradient_batch_size", "window", "max_start_step"):
        if getattr(args, key) < 1:
            p.error(f"--{key.replace('_', '-')} must be positive")
    if args.gradient_batch_size > args.batch_size or not args.horizons or any(h < 1 or h > args.window for h in args.horizons):
        p.error("Gradient batch cannot exceed forward batch; all horizons must fit the selected window")
    if args.max_start_step + args.window - 1 > OLD_GUARD_LIMIT:
        p.error("The latest selected successor must precede the original 250-step recovery guard")
    args.root = args.root.expanduser().resolve()
    output = (args.output or args.root / ("recorded_phase_plan" if args.plan_only else "recorded_phase_probes")).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    checkpoints = {iteration: args.root / "checkpoints" / f"model_{iteration}.pt" for iteration in (22000, 24000, 48000)}
    manifest = {"created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                "source": source_snapshot(output), "status": "sampling", "groups": {},
                "checkpoints": {str(key): {"path": str(path), "sha256": file_sha256(path)} for key, path in checkpoints.items()},
                "protocol": {"phase_is_start_state_only": True, "shared_windows_across_checkpoint_views": True,
                    "window_boundary_policy": "Full finite first-episode prefix; boundary allowed only on the final transition.",
                    "original_production_delay_guard": OLD_GUARD_LIMIT,
                    "latest_allowed_start_exclusive": args.max_start_step,
                    "latest_possible_successor_control_step": args.max_start_step + args.window - 1,
                    "recorded_counter_guard": "Both state/next-state recorded counters must remain below 250.",
                    "scoring": "Own D views include D/norm drift; fixed_D22000 and fixed_D48000 hold both scoring components fixed.",
                    "precision": "tf32 eager probe; no compile, replay, optimizer, training or physical rollout"},
                "limitations": ["Sensitivity on recorded, off-policy windows is not causal proof or a policy-quality evaluation.",
                    "WMAC22000 is a task-successful reference, not ground-truth motion style. Old PPO60000 recordings are a style reference and off-policy data for every world model; prediction errors on them do not establish on-policy failure.",
                    "Evaluation recovery windows used a 501-step limit; early starts and counter checks avoid crossing the original 250-step guard but do not make evaluator state a full production snapshot.",
                    "Initial windows are sampled with recorded replacement when fewer than 64 environment starts exist; repeated windows are not independent trials.",
                    "Phase is determined at the start; nonupright windows may include later upright frames, whose fraction is recorded.",
                    "Gradient norms/cosines are before clipping/Adam; a large or reversed gradient alone does not establish damage.",
                    "Grounded foot height/vertical velocity are kinematic cues, not measured contacts."]}
    groups = {}
    env = None
    try:
        possible_groups = {f"recording{source}_{mode}_{phase}" for source in args.recording_sources
                           for mode in ("walk", "recovery") for phase in PHASES[mode]}
        if args.groups and not set(args.groups) <= possible_groups:
            raise ValueError(f"Unknown requested groups: {sorted(set(args.groups) - possible_groups)}")
        for source in dict.fromkeys(args.recording_sources):
            for mode in ("walk", "recovery"):
                requested_phases = [phase for phase in PHASES[mode]
                                    if not args.groups or f"recording{source}_{mode}_{phase}" in args.groups]
                if not requested_phases:
                    continue
                arrays, recording = read_recording(args.root / f"eval_{source}_seed{args.data_seed}" / f"transitions_{mode}.npz")
                recording["policy_type"] = "old_ppo_style_reference" if source == "ppo60000" else "wmac"
                recording["world_model_data_scope"] = "off-policy reference; not evidence of on-policy model failure" if source == "ppo60000" else "recorded checkpoint policy; cross-checkpoint probes are off-policy"
                for phase in requested_phases:
                    label = f"recording{source}_{mode}_{phase}"
                    sample_seed = args.seed + (0 if mode == "walk" else 1000) + {"initial": 0, "nonupright": 100, "upright": 200}[phase]
                    windows, selection = sample_phase(arrays, phase, args.batch_size, args.window,
                        args.max_start_step, sample_seed, args.scarce_policy)
                    manifest["groups"][label] = {"recording": recording, "selection": selection}
                    if windows is not None:
                        groups[label] = windows
                del arrays
        atomic_json(output / "manifest.json", manifest)
        if args.plan_only:
            manifest["status"] = "planned"
            atomic_json(output / "manifest.json", manifest)
            print(f"CPU sampling plan: {output / 'manifest.json'}")
            return

        os.environ.setdefault("MUJOCO_GL", "egl")
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        import torch
        from scripts.check_wmac_physics import load_checkpoint
        from mjlab.tasks.registry import load_env_cfg
        from src.wmac.env import CaptureEnv
        from src.wmac.state import StateAdapter
        from src.recovery_rewards import configure_recovery_height_reward

        device = torch.device(args.device)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        bundles = {key: load_checkpoint(path, device) for key, path in checkpoints.items()}
        source_cfg = validate_bundles(bundles)
        for bundle in bundles.values():
            for module in bundle[1:]:
                if hasattr(module, "precision"):
                    module.precision = "tf32"
        cfg = load_env_cfg(source_cfg.task)
        configure_recovery_height_reward(cfg, source_cfg.recovery_height_reward)
        cfg.scene.num_envs, cfg.seed = 2, args.seed
        env = CaptureEnv(cfg=cfg, device=args.device)
        env.adapter = adapter = StateAdapter(env,
            recovery_counter_semantics=source_cfg.recovery_counter_semantics)
        if adapter.max_delay_steps != OLD_GUARD_LIMIT:
            raise ValueError(f"Production recovery guard changed to {adapter.max_delay_steps}; re-plan the cutoff")
        manifest["runtime"] = {"torch_version": torch.__version__, "device": str(device),
            "allow_tf32": torch.backends.cuda.matmul.allow_tf32, "precision_override": "tf32",
            "source_precision": {str(key): value[0].precision for key, value in bundles.items()},
            "recovery_counter_semantics": source_cfg.recovery_counter_semantics,
            "recovery_height_reward": source_cfg.recovery_height_reward,
            "compile": False, "kinematic_adapter_environments": 2,
            "actual_production_delay_guard": adapter.max_delay_steps}
        manifest["status"] = "running"
        atomic_json(output / "manifest.json", manifest)
        for label, arrays in groups.items():
            group_dir = output / label
            group_dir.mkdir()
            windows = {key: torch.from_numpy(value).to(device) for key, value in arrays.items()}
            record = manifest["groups"][label]
            record["forward"] = run_forward(group_dir, windows, bundles, adapter, device, args.horizons)
            if not args.skip_gradients:
                record["gradients"] = run_gradients(group_dir, windows, record["selection"], bundles,
                    adapter, device, args.seed + 7001, args.gradient_batch_size)
            atomic_json(output / "manifest.json", manifest)
            del windows
        manifest["input_checkpoints_unchanged"] = all(file_sha256(path) == manifest["checkpoints"][str(key)]["sha256"]
                                                      for key, path in checkpoints.items())
        if not manifest["input_checkpoints_unchanged"]:
            raise RuntimeError("An input checkpoint changed during the probe")
        manifest["status"] = "completed"
        atomic_json(output / "manifest.json", manifest)
        print(f"Recorded-phase probes: {output / 'manifest.json'}")
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        atomic_json(output / "manifest.json", manifest)
        raise
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
