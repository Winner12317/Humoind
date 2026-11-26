"""Compare checkpoint components on fixed physical data and run bounded interventions.

This is a diagnostic, not an exact replay of the original training process.
Old checkpoints usually contain neither replay data nor a complete simulator
snapshot. Component swaps measure sensitivity; only reproduced physical decline
in a control arm can make a freeze comparison informative about its cause.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import traceback
from zoneinfo import ZoneInfo

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.wmac.config import WmacConfig
from src.wmac.checkpoints import resolve_checkpoint_config
from src.wmac.actors import actor_architecture
from src.wmac.replay import ReplayBuffer


VARIANTS = ("control", "freeze_world_model", "freeze_discriminator", "freeze_critic",
            "no_direct_style_grad", "no_critic", "short_horizon", "low_actor_lr")
MODULES = ("actor", "world_model", "discriminator", "amp_norm", "critic")
FIELDS = {"state": 71, "next_state": 71, "action": 29, "previous_action": 29,
          "context": 35, "next_context": 35, "command": 3, "actor_history": 384,
          "contact_cost": 2, "reward": None, "terminated": None,
          "timeout": None, "boundary": None}
FLAGS = ("terminated", "timeout", "boundary", "first_episode_valid")
SEMANTIC_KEYS = ("task", "model_hidden", "policy_hidden", "critic_hidden", "discriminator_hidden",
                 "amp_feature_schema", "recovery_counter_semantics", "recovery_height_reward", "actor_type",
                 "action_limit", "task_weight", "style_weight", "amp_reward_coef", "gamma",
                 "gae_lambda", "direct_style_grad", "critic_bootstrap", "survival_grad",
                 "real_critic_coef")
PHASES = {"model": 11, "discriminator": 23, "validation": 37, "actor": 53,
          "probe_starts": 71, "probe_windows": 89, "probe_noise": 101}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def json_clean(value):
    if isinstance(value, dict):
        return {str(k): json_clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_clean(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return json_clean(value.item())
    return value


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_clean(value), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def reserve_output(path):
    """Never merge a new diagnosis into an earlier experiment's evidence."""
    path = Path(path).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=False)
    return path


def snapshot_source(output, copy_files=True):
    """Capture executed diagnostic sources even when the worktree is uncommitted."""
    relative_paths = sorted(path.relative_to(REPO) for path in (REPO / "src/wmac").glob("*.py"))
    relative_paths += [Path("src/amp_features.py"), Path("src/recovery_rewards.py"),
                       Path("src/tasks/amp_loco/mdp/rewards.py")]
    relative_paths += [Path("scripts") / name for name in (
        "diagnose_wmac.py", "evaluate_wmac.py", "train_wmac.py", "check_wmac_physics.py")]
    snapshot = Path(output) / "source"
    hashes = {}
    for relative in relative_paths:
        content = (REPO / relative).read_bytes()
        hashes[str(relative)] = hashlib.sha256(content).hexdigest()
        if copy_files:
            destination = snapshot / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
    def git(*arguments):
        result = subprocess.run(["git", "-C", str(REPO), *arguments],
                                capture_output=True, text=True, check=False)
        return result.stdout if result.returncode == 0 else "unavailable\n"
    status, difference = git("status", "--short"), git("diff", "HEAD")
    if copy_files:
        (snapshot / "git_status.txt").write_text(status)
        (snapshot / "git.diff").write_text(difference)
    return {"files_copied": copy_files, "directory": str(snapshot) if copy_files else None,
            "files_sha256": hashes, "git_commit": git("rev-parse", "HEAD").strip(),
            "git_status": status, "tracked_diff_sha256": hashlib.sha256(difference.encode()).hexdigest(),
            "tracked_diff_path": "source/git.diff" if copy_files else None,
            "note": "Untracked Python sources are included in files_sha256 and copied on execution; git.diff alone excludes them."}


def resolved_config(payload):
    return resolve_checkpoint_config(payload)


def validate_compatible(good, bad):
    if "replay" in good or "replay" in bad:
        raise ValueError("Use checkpoints saved without replay for this diagnostic; the fixed physical dataset must be the only replay source")
    a, b = resolved_config(good), resolved_config(bad)
    conflicts = {key: [getattr(a, key), getattr(b, key)] for key in SEMANTIC_KEYS
                 if getattr(a, key) != getattr(b, key)}
    if actor_architecture(a) != actor_architecture(b):
        conflicts["actor_architecture"] = [actor_architecture(a), actor_architecture(b)]
    if conflicts:
        raise ValueError(f"Component swaps require matching architecture and task semantics: {conflicts}")
    for name in MODULES:
        if name not in good or name not in bad:
            raise ValueError(f"Checkpoint is missing {name}")
        if good[name].keys() != bad[name].keys() or any(
            good[name][key].shape != bad[name][key].shape for key in good[name]
        ):
            raise ValueError(f"Checkpoint {name} tensors have incompatible shapes")
    return a


def read_dataset(paths):
    """Read immutable arrays; reset transitions stay present but cannot be sampled."""
    chunks, metadata = [], []
    environments = None
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        with np.load(path, allow_pickle=False) as archive:
            missing = set(FIELDS).union({"first_episode_valid"}) - set(archive.files)
            if missing:
                raise ValueError(f"Dataset {path} is missing fields: {sorted(missing)}")
            arrays = {key: np.array(archive[key], copy=True) for key in (*FIELDS, "first_episode_valid")}
            protocol = (json.loads(str(archive["recording_protocol_json"].item()))
                        if "recording_protocol_json" in archive else None)
        shape = arrays["state"].shape
        if len(shape) != 3 or shape[2] != 71 or min(shape[:2]) < 1:
            raise ValueError("Recorded state must have shape [positive time, positive environment, 71]")
        steps, count = shape[:2]
        if count < 2 or (environments is not None and count != environments):
            raise ValueError("All chunks must share at least two environment slots for the held-out split")
        environments = count
        for key, width in FIELDS.items():
            expected = (steps, count) if width is None else (steps, count, width)
            if arrays[key].shape != expected:
                raise ValueError(f"Dataset {path} field {key} must have shape {expected}")
            if key not in FLAGS and arrays[key].dtype.kind != "f":
                raise ValueError(f"Dataset {key} must contain floating-point values")
            if key not in FLAGS:
                arrays[key] = arrays[key].astype(np.float32, copy=False)
        for key in FLAGS:
            value = arrays[key]
            if value.shape != (steps, count) or value.dtype.kind not in "bui" or not np.isin(value, (0, 1)).all():
                raise ValueError(f"Dataset flag {key} must be a [time, environment] boolean array")
            arrays[key] = value.astype(bool, copy=False)
        active = np.ones(count, dtype=bool)
        for step in range(steps):
            if not np.array_equal(active, arrays["first_episode_valid"][step]):
                raise ValueError("first_episode_valid must include the terminal transition and exclude every reset episode")
            active &= ~(arrays["terminated"][step] | arrays["timeout"][step])
        chunks.append({key: torch.from_numpy(value) for key, value in arrays.items()})
        metadata.append({"path": str(path), "sha256": sha256(path), "steps": steps,
                         "num_envs": count, "first_episode_valid_count": int(arrays["first_episode_valid"].sum()),
                         "nonfinite_values_by_field": {key: int((~np.isfinite(value)).sum())
                            for key, value in arrays.items() if value.dtype.kind == "f"},
                         "protocol": protocol})
    if not chunks:
        raise ValueError("At least one recorded transition dataset is required")
    return chunks, metadata


def build_replay(chunks, validation_fraction=.1):
    """Quarantine invalid/reset rows in a private replay copy, without changing files."""
    replay = ReplayBuffer(sum(len(chunk["state"]) for chunk in chunks),
                          chunks[0]["state"].shape[1], "cpu", validation_fraction)
    for chunk in chunks:
        replay.mark_boundary()
        for step in range(len(chunk["state"])):
            row = {key: chunk[key][step].clone() for key in FIELDS}
            row["state"][~chunk["first_episode_valid"][step]] = float("nan")
            replay.add(row)
    if not replay.num_valid("train") or not replay.num_valid("validation"):
        raise ValueError("The fixed dataset must contain finite training and held-out transitions")
    return replay


def phase_seed(seed, iteration, phase):
    if phase not in PHASES:
        raise ValueError(f"Unknown diagnostic phase {phase}")
    return (int(seed) * 1000003 + int(iteration) * 104729 + PHASES[phase]) % (2**32)


@contextmanager
def phase_rng(seed, iteration, phase, device="cpu"):
    """Skipping an intervention's update cannot shift another phase's random data."""
    python_state, numpy_state = random.getstate(), np.random.get_state()
    device = torch.device(device)
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        selected = phase_seed(seed, iteration, phase)
        random.seed(selected)
        np.random.seed(selected)
        torch.random.default_generator.manual_seed(selected)
        if device.type == "cuda":
            with torch.cuda.device(device):
                torch.cuda.manual_seed(selected)
        try:
            yield selected
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)


def variant_changes(name, source, short_horizon=8, actor_lr_factor=.25):
    if name not in VARIANTS:
        raise ValueError(f"Unknown intervention {name}")
    changes = {}
    if name == "no_direct_style_grad":
        if not source.direct_style_grad:
            raise ValueError("The source already disables the direct style gradient")
        changes["direct_style_grad"] = False
    elif name == "no_critic":
        if not source.critic_bootstrap:
            raise ValueError("The source already disables critic bootstrapping")
        changes["critic_bootstrap"] = False
    elif name == "short_horizon":
        if not 1 <= short_horizon < source.imagination_horizon:
            raise ValueError("--short-horizon must be strictly shorter than the checkpoint horizon")
        changes["imagination_horizon"] = short_horizon
    elif name == "low_actor_lr":
        if not 0 < actor_lr_factor < 1:
            raise ValueError("--actor-lr-factor must lie strictly between zero and one")
        changes["actor_lr"] = source.actor_lr * actor_lr_factor
    return {"config_overrides": changes,
            "skip_model_update": name == "freeze_world_model",
            "skip_discriminator_update": name == "freeze_discriminator",
            "skip_critic_update": name == "freeze_critic"}


def module_fingerprint(module):
    digest = hashlib.sha256()
    for name, value in module.state_dict().items():
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def state_fingerprint(value):
    """Hash optimizer tensors and metadata without serializing another checkpoint."""
    digest = hashlib.sha256()
    def visit(item):
        digest.update(type(item).__name__.encode())
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(str(tensor.dtype).encode())
            digest.update(str(tuple(tensor.shape)).encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                visit(key)
                visit(item[key])
        elif isinstance(item, (tuple, list)):
            for child in item:
                visit(child)
        else:
            digest.update(repr(item).encode())
    visit(value)
    return digest.hexdigest()


def frozen_model_metrics(runner, replay, batch_size):
    """Use the production gate's overall/walk/recovery maximum without fitting."""
    with torch.no_grad():
        batch = replay.sample_windows(batch_size, runner.cfg.wm_window,
            runner.cfg.recovery_fraction, fields=runner.WM_FIELDS)
        batch = {key: value.to(runner.device) for key, value in batch.items()}
        error, contact = runner._world_model_loss(batch["state"], batch["action"],
            batch["context"], batch["next_state"], batch["contact_cost"], batch["mask"])
        mask = batch["mask"]
        overall = float((error * mask).sum() / mask.sum().clamp_min(1))
        samples = (error * mask).sum(1) / mask.sum(1).clamp_min(1)
        recovery = (batch["state"][:, 0, 2] < .55) | (batch["context"][:, 0, -1] > 0)
        phases = [float(samples[selected].mean()) if selected.any() else float("nan")
                  for selected in (~recovery, recovery)]
        maximum = max(v for v in (overall, *phases) if math.isfinite(v)) if math.isfinite(overall) else float("inf")
    return {"wm_loss": overall, "wm_walk_error": phases[0], "wm_recovery_error": phases[1],
            "wm_gate_error": maximum, "wm_updates_applied_fraction": 1.,
            "model_update_skipped": True,
            "gate_update_fraction_note": "No model update attempted; frozen finite predictions are evaluated by the same gate error formula."}


def evaluate(checkpoint, output, args, save_data=False):
    command = [sys.executable, str(REPO / "scripts/evaluate_wmac.py"), "--checkpoint", str(checkpoint),
               "--mode", "both", "--seed", str(args.evaluation_seed), "--num-envs", str(args.num_envs),
               "--steps", str(args.steps), "--device", args.device, "--save-trace"]
    if save_data:
        command.append("--save-transitions")
    command += ["--output", str(output)]
    with output.with_suffix(".log").open("w") as log:
        subprocess.run(command, cwd=REPO, env={**os.environ, "PYTHONPATH": str(REPO),
                       "OMP_NUM_THREADS": "1", "MUJOCO_GL": "egl"}, stdout=log, stderr=subprocess.STDOUT, check=True)
    return json.loads((output / "metrics.json").read_text())


def ensure_starting_baseline(args, output, manifest):
    """A reused dataset is not a fresh physical baseline for this sweep."""
    if not args.updates or args.no_final_evaluation:
        return None
    references = manifest.setdefault("physical_reference_evaluations", {})
    if "good" not in references:
        directory = Path(output) / "evaluations" / "starting_good"
        directory.parent.mkdir(parents=True, exist_ok=True)
        references["good"] = evaluate(args.good_checkpoint, directory, args)
        manifest["physical_starting_baseline"] = {
            "metrics_path": str(directory / "metrics.json"),
            "source": "fresh evaluation using this sweep's exact physical protocol",
        }
        atomic_json(Path(output) / "manifest.json", manifest)
    else:
        manifest["physical_starting_baseline"] = {
            "metrics_path": str(Path(references["good"]["arguments"]["output"]) / "metrics.json"),
            "source": "good evaluation captured by this diagnostic run",
        }
        atomic_json(Path(output) / "manifest.json", manifest)
    return references["good"]


def audit_evaluation_pair(reference, candidate, tolerance=1e-6):
    """Check actual initial arrays as well as the physical evaluation protocol."""
    keys = ("mode", "seed", "num_envs", "steps", "device", "push", "observation_noise",
            "command_vx", "command_vy", "command_yaw", "height_threshold", "upright_cos",
            "stable_steps", "walk_hold_steps", "walk_velocity_tolerance", "walk_yaw_tolerance")
    protocol_keys = ("first_episode_only", "task", "policy", "discriminator_used",
                     "stable_recovery", "continued_walk")
    reasons, modes = [], {}
    differences = {key: [reference.get("arguments", {}).get(key), candidate.get("arguments", {}).get(key)]
                   for key in keys if key not in reference.get("arguments", {})
                   or key not in candidate.get("arguments", {})
                   or reference["arguments"][key] != candidate["arguments"][key]}
    protocol_differences = {key: [reference.get("protocol", {}).get(key), candidate.get("protocol", {}).get(key)]
                           for key in protocol_keys if key not in reference.get("protocol", {})
                           or key not in candidate.get("protocol", {})
                           or reference["protocol"][key] != candidate["protocol"][key]}
    if differences or protocol_differences:
        reasons.append("Physical evaluation protocols differ")
    for mode in ("walk", "recovery"):
        a, b = reference.get("results", {}).get(mode), candidate.get("results", {}).get(mode)
        record = {"arrays": {}, "all_arrays_exact": False, "all_arrays_within_tolerance": False}
        if not isinstance(a, dict) or not isinstance(b, dict):
            record["error"] = "Missing physical mode result"
            reasons.append(f"Missing {mode} result")
            modes[mode] = record
            continue
        record["recorded_initial_hashes_equal"] = (a["initial_state_sha256"] == b["initial_state_sha256"]
            if a.get("initial_state_sha256") and b.get("initial_state_sha256") else None)
        record["result_protocol_differences"] = {key: [a.get(key), b.get(key)]
            for key in ("num_envs", "steps_requested", "control_dt")
            if key not in a or key not in b or a[key] != b[key]}
        if record["result_protocol_differences"]:
            reasons.append(f"{mode} reported physical dimensions/timing differ or are missing")
        try:
            with np.load(a["initial_state_npz"], allow_pickle=False) as initial_a, np.load(b["initial_state_npz"], allow_pickle=False) as initial_b:
                for name in ("state", "context", "command", "actor_history"):
                    x, y = initial_a[name], initial_b[name]
                    same_shape = x.shape == y.shape
                    finite = bool(np.isfinite(x).all() and np.isfinite(y).all())
                    exact = same_shape and finite and np.array_equal(x, y)
                    close = same_shape and finite and np.allclose(x, y, rtol=0., atol=tolerance)
                    record["arrays"][name] = {"shape_matches": same_shape,
                        "reference_shape": list(x.shape), "candidate_shape": list(y.shape),
                        "dtype_matches": x.dtype == y.dtype, "both_finite": finite,
                        "exact": bool(exact), "within_tolerance": bool(close),
                        "maximum_absolute_difference": float(np.max(np.abs(x.astype(np.float64) - y.astype(np.float64))))
                            if same_shape and finite and x.size else None}
            record["all_arrays_exact"] = all(v["exact"] for v in record["arrays"].values())
            record["all_arrays_within_tolerance"] = all(v["within_tolerance"] for v in record["arrays"].values())
            if not record["all_arrays_within_tolerance"]:
                reasons.append(f"{mode} initial arrays differ beyond tolerance")
        except (OSError, ValueError, KeyError, TypeError) as error:
            record["error"] = str(error)
            reasons.append(f"Cannot validate {mode} initial arrays")
        modes[mode] = record
    allowed = not reasons
    return {"paired_comparison_allowed": allowed, "causal_comparison_eligible": allowed,
            "status": "paired_inputs_verified" if allowed else "paired_comparison_refused",
            "absolute_tolerance": tolerance, "argument_differences": differences,
            "protocol_differences": protocol_differences, "modes": modes, "reasons": reasons,
            "limitation": "Matched initial arrays are necessary for this comparison, but do not guarantee bitwise identical rollouts or establish causality."}


def diagnostic_config(source, replay, args, log_dir):
    cfg = resolved_config({"format": "amp-wmac-v1", "config": asdict(source)})
    cfg.num_envs, cfg.replay_steps = replay.num_envs, replay.capacity_steps
    cfg.actor_batch_size, cfg.wm_batch_size, cfg.wm_updates = args.batch_size, args.wm_batch_size, args.wm_updates
    cfg.device, cfg.replay_device = args.device, "cpu"
    cfg.precision, cfg.compile = args.precision, False
    cfg.validation_env_fraction = replay.validation_env_fraction
    cfg.validation_batch_size = min(args.wm_batch_size, args.batch_size)
    cfg.real_critic_batch_size = min(cfg.real_critic_batch_size, args.batch_size)
    cfg.discriminator_batch_size = args.wm_batch_size
    cfg.log_root, cfg.run_name = str(log_dir.parent), log_dir.name
    cfg.init_policy = cfg.resume = None
    cfg.evaluation_interval = 0
    cfg.validate()
    return cfg


def make_runner(env, source, replay, args, log_dir, checkpoint):
    from src.wmac.runner import WmacRunner

    cfg = diagnostic_config(source, replay, args, log_dir)
    runner = WmacRunner(env, cfg, log_dir)
    runner.load(str(checkpoint))
    runner.replay = replay
    runner._train_env_mask = replay.train_env_mask.to(runner.device)
    # Fused CUDA optimizer state is retained on CUDA; CPU diagnostics need the
    # matching optimizer implementation rather than a loaded fused flag.
    if runner.device.type == "cpu":
        for name in ("opt_actor", "opt_model", "opt_disc", "opt_critic"):
            for group in getattr(runner, name).param_groups:
                group["fused"] = False
    return runner


def close_runner(runner):
    runner.writer.close()
    if runner.csv_file:
        runner.csv_file.close()


def load_components(runner, payload, names=MODULES):
    for name in names:
        getattr(runner, name).load_state_dict(payload[name], strict=True)
    if "amp_norm" in names:
        runner._set_amp_norm_settings(payload.get("amp_normalizer_settings", {}))


def run_probes(runner, good, bad, replay, args, output):
    from src.wmac.diagnostics import objective_gradient_probe, open_loop_probe, safe_cosine

    with phase_rng(args.seed, 0, "probe_starts", args.device):
        batch = replay.sample_starts(args.batch_size, runner.cfg.recovery_fraction,
                                     fields=runner.ACTOR_FIELDS, split="validation")
    with phase_rng(args.seed, 0, "probe_windows", args.device):
        windows = replay.sample_windows(args.batch_size, max(args.horizons),
                                         runner.cfg.recovery_fraction, split="validation")
    batch = {key: value.to(runner.device) for key, value in batch.items()}
    windows = {key: value.to(runner.device) for key, value in windows.items()}
    with phase_rng(args.seed, 0, "probe_noise", args.device):
        noises = runner._observation_noise(args.batch_size)
    cases = {"good_reference": (), "swap_world_model": ("world_model",),
             "swap_discriminator": ("discriminator", "amp_norm"), "swap_critic": ("critic",),
             "swap_all_auxiliaries": ("world_model", "discriminator", "amp_norm", "critic")}
    results, baseline_gradient = {}, None
    for label, donor_names in cases.items():
        load_components(runner, good)
        load_components(runner, bad, donor_names)
        probe = objective_gradient_probe(runner, batch, noises=noises, return_gradients=True)
        gradient = probe.pop("_gradients")["total"]
        if baseline_gradient is None:
            baseline_gradient = gradient.clone()
        probe["total_gradient_cosine_to_good_reference"] = safe_cosine(baseline_gradient, gradient)
        probe["open_loop"] = open_loop_probe(runner, windows, args.horizons)
        probe["bad_checkpoint_components"] = list(donor_names)
        probe["actor_source"] = "good checkpoint (unchanged, including its normalizer)"
        results[label] = probe
        atomic_json(output / "probes.json", {"scope": "Fixed-data component sensitivity; not causal proof",
                    "probe_split": "held-out environments", "same_batch_windows_and_noise": True,
                    "sampler_seed": args.seed, "results": results})
        print(f"[diagnosis] {label}: gradient cosine={probe['total_gradient_cosine_to_good_reference']}", flush=True)
        del gradient
    load_components(runner, good)
    return results


def run_sweep(env, source, replay, args, output, baseline=None):
    results = {}
    for label in args.variants:
        plan = variant_changes(label, source, args.short_horizon, args.actor_lr_factor)
        arm = output / "sweep" / label
        arm.mkdir(parents=True)
        runner = make_runner(env, source, replay, args, arm, args.good_checkpoint)
        source_loaded = {"module_fingerprints": {name: module_fingerprint(getattr(runner, name)) for name in MODULES},
                         "optimizer_fingerprints": {name: state_fingerprint(getattr(runner, name).state_dict())
                             for name in ("opt_actor", "opt_model", "opt_disc", "opt_critic")},
                         "amp_normalizer_settings": {key: getattr(runner.amp_norm, key)
                             for key in ("eps", "variance_epsilon", "clip")}}
        for key, value in plan["config_overrides"].items():
            setattr(runner.cfg, key, value)
        frozen_names = (("world_model",) if plan["skip_model_update"] else
                        ("discriminator", "amp_norm") if plan["skip_discriminator_update"] else
                        ("critic",) if plan["skip_critic_update"] else ())
        before = {name: module_fingerprint(getattr(runner, name)) for name in frozen_names}
        if plan["skip_critic_update"]:
            runner._fit_critic = lambda *unused: {"critic_loss": 0., "real_critic_loss": 0.,
                                                  "critic_grad_norm": 0., "critic_updates_applied_fraction": 0.}
        rows = []
        try:
            for index in range(args.updates):
                runner._set_learning_rates()
                with phase_rng(args.seed, index, "model", args.device):
                    if plan["skip_model_update"]:
                        model = frozen_model_metrics(runner, replay, args.wm_batch_size)
                    else:
                        model = {**runner.fit_model(), "model_update_skipped": False}
                with phase_rng(args.seed, index, "discriminator", args.device):
                    disc = runner.update_discriminator() if not plan["skip_discriminator_update"] else {}
                with phase_rng(args.seed, index, "validation", args.device):
                    validation = runner.validate_model()
                gate = runner._actor_gate(model)
                with phase_rng(args.seed, index, "actor", args.device):
                    updates = runner.update_actor_critic() if gate else {}
                runner.iteration += 1
                row = {"diagnostic_update": index + 1, "source_iteration_plus_updates": runner.iteration,
                       "actor_gate_passed": gate, "actor_update_attempted": gate,
                       "actor_lr": runner.opt_actor.param_groups[0]["lr"],
                       "phase_seeds": {key: phase_seed(args.seed, index, key)
                                       for key in ("model", "discriminator", "validation", "actor")},
                       **model, **disc, **validation, **updates}
                row["nonfinite_fields"] = [key for key, value in row.items()
                    if isinstance(value, float) and not math.isfinite(value)]
                rows.append(row)
                atomic_json(arm / "updates.json", rows)
            after = {name: module_fingerprint(getattr(runner, name)) for name in frozen_names}
            if before != after:
                raise RuntimeError(f"Frozen component changed in {label}: {before} != {after}")
            checkpoint = arm / f"model_{runner.iteration}.pt"
            runner.save(checkpoint, export=False)
            record = {"intervention": plan, "checkpoint": str(checkpoint), "updates": rows,
                      "loaded_source_state": source_loaded,
                      "frozen_component_fingerprints_before": before,
                      "frozen_component_fingerprints_after": after,
                      "actual_config": asdict(runner.cfg), "status": "completed"}
        finally:
            close_runner(runner)
            del runner
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if not args.no_final_evaluation:
            record["physical_evaluation"] = evaluate(checkpoint, arm / "evaluation", args)
            if baseline is None:
                raise RuntimeError("A physical starting baseline is required before interpreting a sweep")
            record["physical_pairing_audit"] = audit_evaluation_pair(baseline, record["physical_evaluation"])
            eligible = record["physical_pairing_audit"]["paired_comparison_allowed"]
            record["paired_recovery_success_change"] = (
                record["physical_evaluation"]["results"]["recovery"]["recovery_success_fraction"]
                - baseline["results"]["recovery"]["recovery_success_fraction"] if eligible else None)
            record["paired_comparison_note"] = (
                "Protocol and initial arrays verified; repeats and reproduced control decline are still required."
                if eligible else "Causal and paired performance comparison refused because protocol or initial arrays do not match.")
        results[label] = record
        atomic_json(output / "sweep.json", {"scope": "Fixed-data short continuation, not on-policy resume",
                    "production_model_gate_enabled": True,
                    "normalization_note": "World-model and critic normalizers remain fixed; AMP normalizer updates with D unless frozen.",
                    "conclusion": "Inconclusive until the control reproduces physical decline; freeze comparisons are hypotheses requiring repeats.",
                    "results": results})
        print(f"[diagnosis] sweep {label}: completed {args.updates} update attempts", flush=True)
    return results


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--good-checkpoint", type=Path, required=True)
    p.add_argument("--bad-checkpoint", type=Path, required=True)
    p.add_argument("--data", type=Path, nargs="+")
    p.add_argument("--output", type=Path)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--precision", choices=("float32", "tf32", "bf16"), default="float32")
    p.add_argument("--seed", type=int, default=4242)
    p.add_argument("--evaluation-seed", type=int, default=2026)
    p.add_argument("--num-envs", type=int, default=32)
    p.add_argument("--steps", type=int, default=250)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--wm-batch-size", type=int, default=32)
    p.add_argument("--wm-updates", type=int, default=1)
    p.add_argument("--horizons", nargs="+", type=int, default=[1, 4, 8, 16, 32])
    p.add_argument("--updates", type=int, default=0)
    p.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    p.add_argument("--short-horizon", type=int, default=8)
    p.add_argument("--actor-lr-factor", type=float, default=.25)
    p.add_argument("--no-final-evaluation", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    for name in ("num_envs", "steps", "batch_size", "wm_batch_size", "wm_updates"):
        if getattr(args, name) < 1:
            p.error(f"--{name.replace('_', '-')} must be positive")
    if args.num_envs < 2 or args.updates < 0 or any(h < 1 for h in args.horizons):
        p.error("Use at least two environments, nonnegative updates and positive horizons")
    args.horizons = sorted(set(args.horizons))
    args.variants = list(dict.fromkeys(args.variants))
    if not math.isfinite(args.actor_lr_factor) or not 0 < args.actor_lr_factor < 1:
        p.error("--actor-lr-factor must lie strictly between zero and one")
    for name in ("good_checkpoint", "bad_checkpoint"):
        path = getattr(args, name).expanduser().resolve()
        if not path.is_file():
            p.error(f"Checkpoint not found: {path}")
        setattr(args, name, path)
    good = torch.load(args.good_checkpoint, map_location="cpu", weights_only=False)
    bad = torch.load(args.bad_checkpoint, map_location="cpu", weights_only=False)
    source = validate_compatible(good, bad)
    plans = {name: variant_changes(name, source, args.short_horizon, args.actor_lr_factor)
             for name in args.variants} if args.updates else {}
    chunks, dataset_metadata = read_dataset(args.data) if args.data else (None, None)
    if chunks and sum(len(chunk["state"]) for chunk in chunks) < max(max(args.horizons), source.wm_window, source.warmup_steps):
        p.error("Recorded dataset is shorter than the requested horizons or checkpoint model window/warmup")
    output = reserve_output(args.output or REPO / "logs/wmac_diagnostics" /
                            datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d_%H-%M-%S_%f"))
    manifest = {"created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                "source_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                    capture_output=True, text=True, check=False).stdout.strip(),
                "script_sha256": sha256(Path(__file__)), "arguments": vars(args),
                "checkpoints": {"good": {"path": args.good_checkpoint, "sha256": sha256(args.good_checkpoint),
                                            "iteration": good.get("iteration")},
                                "bad": {"path": args.bad_checkpoint, "sha256": sha256(args.bad_checkpoint),
                                           "iteration": bad.get("iteration")}},
                "source_config": asdict(source), "data": dataset_metadata, "interventions": plans,
                "source_optimizer_fingerprints": {name: state_fingerprint(good[name])
                    for name in ("opt_actor", "opt_model", "opt_disc", "opt_critic") if name in good},
                "source_amp_normalizer_settings": good.get("amp_normalizer_settings", {}),
                "limitations": ["Checkpoint swap gradients are sensitivity evidence, not causal proof.",
                    "Fixed-data continuation does not reproduce the original on-policy replay or full simulator state.",
                    "Identical initial arrays and seeds do not guarantee bitwise identical physical rollouts.",
                    "A short smoke run does not establish that a component caused forgetting.",
                    "Good/bad labels describe supplied checkpoints, not a new performance claim."],
                "status": "planned"}
    manifest["source_snapshot"] = snapshot_source(output, copy_files=not args.dry_run)
    atomic_json(output / "manifest.json", manifest)
    if args.dry_run:
        print(f"Dry run: {output / 'manifest.json'}")
        return
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu for supported offline diagnostics")
    env = runner = None
    try:
        atomic_json(output / "progress.json", {"phase": "capture" if chunks is None else "load_data"})
        if chunks is None:
            eval_dir = output / "evaluations"
            eval_dir.mkdir()
            manifest["physical_reference_evaluations"] = {}
            for name in ("good", "bad"):
                manifest["physical_reference_evaluations"][name] = evaluate(
                    getattr(args, name + "_checkpoint"), eval_dir / name, args, save_data=True)
            paths = [eval_dir / "good" / f"transitions_{mode}.npz" for mode in ("walk", "recovery")]
            chunks, dataset_metadata = read_dataset(paths)
            manifest["data"] = dataset_metadata
        baseline = ensure_starting_baseline(args, output, manifest)
        replay = build_replay(chunks)
        del chunks
        if replay.size < max(max(args.horizons), source.wm_window, source.warmup_steps):
            raise ValueError("Recorded replay is shorter than the requested windows or checkpoint warmup")
        manifest["replay"] = {"capacity_steps": replay.capacity_steps, "num_envs": replay.num_envs,
                              "train_valid": replay.num_valid("train"),
                              "validation_valid": replay.num_valid("validation"),
                              "quarantined_transitions": replay.invalid_transitions}
        import mjlab.tasks  # noqa: F401
        import src.tasks  # noqa: F401
        from mjlab.tasks.registry import load_env_cfg
        from src.wmac.env import CaptureEnv
        from src.wmac.state import StateAdapter
        from src.recovery_rewards import configure_recovery_height_reward

        torch.backends.cuda.matmul.allow_tf32 = args.precision == "tf32"
        torch.backends.cudnn.allow_tf32 = args.precision == "tf32"
        env_cfg = load_env_cfg(source.task)
        configure_recovery_height_reward(env_cfg, source.recovery_height_reward)
        env_cfg.scene.num_envs, env_cfg.seed = replay.num_envs, args.seed
        env = CaptureEnv(cfg=env_cfg, device=args.device)
        env.adapter = StateAdapter(env, recovery_counter_semantics=source.recovery_counter_semantics)
        runner = make_runner(env, source, replay, args, output / "probe_runner", args.good_checkpoint)
        manifest["diagnostic_config"] = asdict(runner.cfg)
        manifest["config_differences"] = {key: {"source": asdict(source)[key], "diagnostic": value}
                                          for key, value in asdict(runner.cfg).items()
                                          if asdict(source)[key] != value}
        atomic_json(output / "manifest.json", manifest)
        atomic_json(output / "progress.json", {"phase": "component_probes"})
        run_probes(runner, good, bad, replay, args, output)
        close_runner(runner)
        runner = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if args.updates:
            atomic_json(output / "progress.json", {"phase": "fixed_data_sweep"})
            run_sweep(env, source, replay, args, output, baseline=baseline)
        manifest["status"] = "completed"
        atomic_json(output / "manifest.json", manifest)
        atomic_json(output / "summary.json", {"status": "completed", "output": output,
                    "probes": "probes.json", "sweep": "sweep.json" if args.updates else None,
                    "causal_conclusion": "Not established; inspect component sensitivities, then reproduce control decline and repeat intervention comparisons.",
                    "limitations": manifest["limitations"]})
        atomic_json(output / "progress.json", {"phase": "completed"})
        print(f"Diagnosis: {output / 'summary.json'}")
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        atomic_json(output / "manifest.json", manifest)
        atomic_json(output / "progress.json", {"phase": "failed", "error": str(error)})
        raise
    finally:
        if runner is not None:
            close_runner(runner)
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
