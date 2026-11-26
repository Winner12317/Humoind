"""CPU-only style diagnostics on immutable, recorded physical AMP transitions.

Rebuilds one canonical grounded_v2 representation for every policy, including
legacy PPO. This is an evaluation representation, not a checkpoint conversion.
No simulator rollout, optimizer update, or CUDA allocation is performed.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
import hashlib
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.amp_features import amp_feature_metadata, build_amp_features
from src.wmac.networks import RunningNorm
from src.wmac.state import StateAdapter, quat_to_matrix


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def json_clean(value):
    if isinstance(value, dict):
        return {str(key): json_clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_clean(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, np.generic):
        return json_clean(value.item())
    return value


def summary(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = values[np.isfinite(values)]
    result = {"count": int(values.size), "nonfinite_count": int(values.size - finite.size)}
    if finite.size:
        result.update(mean=float(finite.mean()), median=float(np.median(finite)),
                      p05=float(np.quantile(finite, .05)), p95=float(np.quantile(finite, .95)),
                      rms=float(np.sqrt(np.mean(finite ** 2))))
    else:
        result.update(mean=None, median=None, p05=None, p95=None, rms=None)
    return result


def sample_indices(mask, maximum, seed):
    indices = np.flatnonzero(np.asarray(mask, dtype=bool).reshape(-1))
    if len(indices) > maximum:
        indices = np.sort(np.random.default_rng(seed).choice(indices, maximum, replace=False))
    return indices


def nearest_rmse(query, reference, batch_size=256):
    """Exact nearest-neighbor RMS over all pair coordinates, in CPU chunks."""
    if query.ndim != 2 or reference.ndim != 2 or query.shape[1] != reference.shape[1]:
        raise ValueError("Nearest-neighbor arrays must share a nonempty feature axis")
    if not query.shape[1] or not len(reference):
        raise ValueError("Nearest-neighbor reference must be nonempty")
    if not np.isfinite(query).all() or not np.isfinite(reference).all():
        raise ValueError("Nearest-neighbor features must be finite")
    # Double precision avoids cancellation for identical/very close vectors.
    ref = torch.as_tensor(reference, dtype=torch.float64, device="cpu")
    ref_norm = ref.square().sum(-1)
    answer = []
    for start in range(0, len(query), batch_size):
        q = torch.as_tensor(query[start:start + batch_size], dtype=torch.float64, device="cpu")
        squared = (q.square().sum(-1, keepdim=True) + ref_norm[None] - 2 * q @ ref.T).clamp_min(0)
        answer.append((squared.min(-1).values / q.shape[-1]).sqrt().numpy())
    return np.concatenate(answer) if answer else np.empty(0)


def phase_masks(state, valid, dt):
    """Fixed physical phase thresholds; labels describe the transition's start."""
    state = np.asarray(state)
    quat = torch.as_tensor(state[..., 3:7], dtype=torch.float32)
    up_z = quat_to_matrix(quat)[..., 2, 2].numpy()
    low = (state[..., 2] < .55) | (up_z < .5)
    upright = (state[..., 2] > .65) & (up_z > .8)
    early = (np.arange(state.shape[0]) * dt < 2.)[:, None]
    return {
        "all": valid.copy(), "first_2_seconds": valid & early,
        "low_or_tilted": valid & low, "upright": valid & upright,
        "intermediate": valid & ~low & ~upright,
    }


def smoothness(action, previous_action, valid, boundary, dt):
    """Action changes include the recorded first action's actual predecessor.

    The second derivative is an action acceleration / jerk proxy, not a measured
    physical joint jerk. No difference spans a reset, push, or recording boundary.
    """
    first = action - previous_action
    first_values = np.sqrt(np.mean(first[valid].astype(np.float64) ** 2, axis=-1))
    contiguous = valid[1:] & valid[:-1] & ~boundary[:-1]
    if action.shape[0] > 1:
        contiguous &= np.max(np.abs(previous_action[1:] - action[:-1]), axis=-1) <= 1e-5
    second = first[1:] - first[:-1]
    second_values = np.sqrt(np.mean(second[contiguous].astype(np.float64) ** 2, axis=-1))
    return {
        "first_difference_rms": summary(first_values),
        "first_derivative_rms_per_second": summary(first_values / dt),
        "second_difference_rms": summary(second_values),
        "second_derivative_jerk_proxy_rms_per_second_squared": summary(second_values / dt ** 2),
        "definition": "Each row is RMS over 29 normalized actions; second derivative is an action acceleration/jerk proxy, not physical joint jerk.",
    }


def make_cpu_adapter(xml):
    """Compile only G1 kinematic metadata; create no mjlab simulator or device."""
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(xml))
    body_ids = np.arange(1, model.nbody)
    joint_ids = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_HINGE)
    if len(joint_ids) != 29:
        raise ValueError("Expected the canonical 29-hinge G1 model")
    indexing = SimpleNamespace(
        body_ids=torch.tensor(body_ids), joint_ids=torch.tensor(joint_ids),
        geom_ids=torch.arange(model.ngeom),
        geoms=[SimpleNamespace(id=i, name=model.geom(i).name) for i in range(model.ngeom)],
        sites=[SimpleNamespace(id=i, name=model.site(i).name) for i in range(model.nsite)],
    )
    root = torch.zeros(1, 13)
    root[0, :7] = torch.tensor(model.qpos0[:7], dtype=torch.float32)
    robot = SimpleNamespace(
        body_names=tuple(model.body(i).name for i in body_ids), indexing=indexing,
        data=SimpleNamespace(
            default_joint_pos=torch.tensor(model.qpos0[model.jnt_qposadr[joint_ids]], dtype=torch.float32)[None],
            default_joint_vel=torch.zeros(1, 29),
            soft_joint_pos_limits=torch.tensor(model.jnt_range[joint_ids], dtype=torch.float32)[None],
            default_root_state=root),
    )
    env = SimpleNamespace(
        scene={"robot": robot}, device="cpu", step_dt=.02, sim=SimpleNamespace(mj_model=model),
        cfg=SimpleNamespace(
            observations={"amp": SimpleNamespace(terms={"grounded_features": SimpleNamespace(params={"schema": "grounded_v2"})})},
            rewards={}, scale_rewards_by_dt=True,
            terminations={"bad_base_height": SimpleNamespace(params={"minimum_height": .5}),
                          "bad_orientation": SimpleNamespace(params={"limit_angle": np.deg2rad(70.)})}),
        termination_manager=SimpleNamespace(_max_delay_steps=250),
    )
    return StateAdapter(env)


def features(adapter, state, batch_size=2048):
    flat = np.asarray(state, dtype=np.float32).reshape(-1, 71)
    result = []
    with torch.no_grad():
        for start in range(0, len(flat), batch_size):
            result.append(adapter.amp_features(torch.from_numpy(flat[start:start + batch_size])).numpy())
    return np.concatenate(result).reshape(*state.shape[:-1], 203)


def expert_reference(adapter, motion_root, maximum, seed):
    groups, frame_pools, records = {"Recovery": [], "Walk": []}, [], []
    for path in sorted(Path(motion_root).rglob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            fps = float(np.asarray(data["fps"]).reshape(-1)[0])
            if not np.isclose(fps, 50.):
                raise ValueError(f"Expert {path} is {fps}Hz; physical 20ms pairs require matching timing")
            arrays = [torch.tensor(data[key], dtype=torch.float32) for key in
                      ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")]
            if arrays[0].shape[1] != len(adapter.body_names):
                raise ValueError(f"Expert {path} body count differs from canonical G1")
            with torch.no_grad():
                frame = build_amp_features(*arrays, body_ids=adapter.amp_ids, anchor_id=adapter.anchor_id,
                    pelvis_id=adapter.pelvis_id, foot_ids=adapter.foot_ids, schema="grounded_v2").numpy()
            group = "Recovery" if "recovery" in {part.lower() for part in path.parts} else "Walk"
            pairs = np.concatenate((frame[:-1], frame[1:]), axis=-1)
            groups[group].append(pairs)
            frame_pools.append(frame)
            records.append({"path": str(path.resolve()), "sha256": sha256(path), "group": group,
                            "frames": len(frame), "adjacent_pairs": len(pairs), "fps": fps,
                            "contains_action": "action" in data or "actions" in data})
    if not frame_pools or any(not parts for parts in groups.values()):
        raise ValueError("Both Recovery and locomotion expert pools are required")
    all_frames = np.concatenate(frame_pools).astype(np.float64)
    mean, std_raw = all_frames.mean(0), all_frames.std(0)
    std = np.maximum(std_raw, 1e-3)
    mean_pair, std_pair = np.tile(mean, 2), np.tile(std, 2)
    sampled, info = {}, {}
    for i, (name, parts) in enumerate(groups.items()):
        pool = np.concatenate(parts)
        indices = sample_indices(np.ones(len(pool), dtype=bool), maximum, seed + i)
        sampled[name] = ((pool[indices] - mean_pair) / std_pair).astype(np.float32)
        info[name] = {"available_pairs": len(pool), "sampled_pairs": len(indices),
                      "sample_seed": seed + i, "sampled_index_sha256": hashlib.sha256(indices.tobytes()).hexdigest()}
    return mean_pair, std_pair, sampled, {
        "files": records, "pools": info, "normalization": {
            "source": "All expert frames, equally weighted per frame across both groups; frozen for every policy",
            "frame_mean": mean.tolist(), "frame_std": std.tolist(),
            "std_floor": 1e-3, "floored_dimensions": int((std_raw < 1e-3).sum()),
            "clipping": False, "ddof": 0},
        "pair_definition": "Adjacent frames at 50Hz within a single clip; never cross clip boundaries",
        "action_limitation": "Reference motion NPZ files contain pose/velocity, not policy action; no expert-action smoothness or control jerk comparison is available.",
    }


def load_scorer(path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    cfg = payload.get("config", {})
    schema = payload.get("amp_feature_schema", cfg.get("amp_feature_schema", "legacy"))
    if payload.get("format") != "amp-wmac-v1" or schema != "grounded_v2":
        raise ValueError("A common scorer must be a grounded_v2 WMAC checkpoint; legacy D cannot score 203D")
    # Match rsl_rl.modules.discriminator.Discriminator exactly (ReLU, named
    # trunk/amp_linear); the Actor/WM mlp helper uses a different activation.
    trunk, width = [], 406
    for hidden in cfg.get("discriminator_hidden", (1024, 512, 256)):
        trunk.extend((torch.nn.Linear(width, hidden), torch.nn.ReLU()))
        width = hidden
    network = torch.nn.Sequential(OrderedDict(
        trunk=torch.nn.Sequential(*trunk), amp_linear=torch.nn.Linear(width, 1))).eval()
    network.load_state_dict(payload["discriminator"])
    norm = RunningNorm(203).eval()
    norm.load_state_dict(payload["amp_norm"])
    settings = payload.get("amp_normalizer_settings", {})
    for key in ("eps", "variance_epsilon", "clip", "max_count"):
        if key in settings:
            setattr(norm, key, settings[key])
    for module in (network, norm):
        module.requires_grad_(False)
    info = {"checkpoint": str(path), "sha256": sha256(path),
            "iteration": payload.get("iteration"), "training_task": cfg.get("task"),
            "feature_schema": schema, "coefficient": cfg.get("amp_reward_coef", .1),
            "normalizer_settings": settings,
            "interpretation": "One frozen trained D and its frozen normalizer score every policy's real pairs. This is not an independent ground-truth style evaluator."}
    return network, norm, float(info["coefficient"]), info


def score_pairs(pair, scorer):
    network, norm, coefficient, _ = scorer
    scores = []
    with torch.no_grad():
        for start in range(0, len(pair), 1024):
            chunk = torch.as_tensor(pair[start:start + 1024], dtype=torch.float32)
            normalized = torch.cat((norm(chunk[:, :203]), norm(chunk[:, 203:])), -1)
            scores.append(network(normalized).squeeze(-1).numpy())
    raw = np.concatenate(scores) if scores else np.empty(0)
    style = coefficient * np.maximum(1 - .25 * (raw - 1) ** 2, 0)
    return raw, style


def load_recording(directory, mode, metrics):
    path = directory / f"transitions_{mode}.npz"
    with np.load(path, allow_pickle=False) as archive:
        names = ("state", "next_state", "action", "previous_action", "contact_cost", "boundary", "first_episode_valid")
        arrays = {name: np.array(archive[name], copy=True) for name in names}
        protocol = json.loads(str(archive["recording_protocol_json"].item()))
    state = arrays["state"]
    if state.ndim != 3 or state.shape[-1] != 71:
        raise ValueError(f"Invalid recorded state layout: {path}")
    prefix = state.shape[:2]
    widths = {"state": 71, "next_state": 71, "action": 29, "previous_action": 29, "contact_cost": 2}
    for name, value in arrays.items():
        shape = (*prefix, widths[name]) if name in widths else prefix
        if value.shape != shape:
            raise ValueError(f"Recording {path} field {name} must have shape {shape}")
    for name in ("boundary", "first_episode_valid"):
        if arrays[name].dtype.kind not in "bui" or not np.isin(arrays[name], (0, 1)).all():
            raise ValueError(f"Recording {path} field {name} must contain booleans")
        arrays[name] = arrays[name].astype(bool)
    dt = float(protocol["control_dt"])
    if not np.isclose(dt, .02):
        raise ValueError("Expert adjacent pairs are 20ms apart; this recording has another control period")
    original = arrays["first_episode_valid"].astype(bool)
    valid = original.copy()
    for name in widths:
        valid &= np.isfinite(arrays[name]).all(-1)
    valid &= np.linalg.norm(state[..., 3:7], axis=-1) > 1e-8
    valid &= np.linalg.norm(arrays["next_state"][..., 3:7], axis=-1) > 1e-8
    arrays["valid"] = valid
    return arrays, protocol, {"path": str(path), "sha256": sha256(path),
        "shape": list(prefix), "first_episode_count": int(original.sum()),
        "valid_finite_count": int(valid.sum()), "quarantined_nonfinite_or_bad_quaternion_count": int((original & ~valid).sum()),
        "training_feature_schema": metrics.get("protocol", {}).get("amp_feature_schema", "legacy"),
        "training_task": metrics.get("protocol", {}).get("task"),
        "evaluation_representation": "canonical grounded_v2 reconstructed from physical generalized states; no padding or checkpoint/schema conversion"}


def audit_pairing(evaluations):
    """Allow native Flat/Aligned observations, verify actual physical inputs."""
    groups, audits = {}, []
    keys = ("seed", "num_envs", "steps", "push", "observation_noise", "command_vx", "command_vy",
            "command_yaw", "height_threshold", "upright_cos", "stable_steps", "walk_hold_steps",
            "walk_velocity_tolerance", "walk_yaw_tolerance")
    for directory, metrics in evaluations:
        args = metrics["arguments"]
        for mode in metrics["results"]:
            groups.setdefault((args["seed"], mode), []).append((directory, metrics))
    for (seed, mode), members in groups.items():
        base_dir, base_metrics = members[0]
        with np.load(base_dir / f"initial_{mode}.npz", allow_pickle=False) as archive:
            base = {key: np.array(archive[key]) for key in ("state", "context", "command", "actor_history")}
        for directory, metrics in members[1:]:
            differences = {key: [base_metrics["arguments"].get(key), metrics["arguments"].get(key)]
                           for key in keys if base_metrics["arguments"].get(key) != metrics["arguments"].get(key)}
            fields = {}
            with np.load(directory / f"initial_{mode}.npz", allow_pickle=False) as archive:
                for key, first in base.items():
                    second = archive[key]
                    same_layout = first.shape == second.shape and first.dtype == second.dtype
                    finite = bool(np.isfinite(first).all() and np.isfinite(second).all())
                    gap = float(np.max(np.abs(first - second))) if same_layout and first.size else (0. if same_layout else None)
                    fields[key] = {"same_shape_dtype": same_layout, "finite": finite, "max_abs_difference": gap,
                                   "equal_within_1e_6": same_layout and finite and gap <= 1e-6}
            passed = not differences and all(field["equal_within_1e_6"] for field in fields.values())
            audits.append({"seed": seed, "mode": mode, "reference": str(base_dir), "other": str(directory),
                           "protocol_differences": differences, "initial_arrays": fields, "passed": passed,
                           "native_training_tasks": [base_metrics.get("protocol", {}).get("task"), metrics.get("protocol", {}).get("task")]})
            if not passed:
                raise ValueError(f"Physical initial-state pairing failed: {base_dir} vs {directory}: {audits[-1]}")
    return {"comparisons": audits, "all_paired_comparisons_passed": all(row["passed"] for row in audits),
            "note": "Flat vs Flat-Aligned native AMP/critic observations are permitted; matching state/context/command/Actor history is required. Arrays are not full simulator snapshots."}


def analyze_mode(directory, metrics, mode, adapter, reference, scorers, maximum, seed):
    arrays, protocol, provenance = load_recording(directory, mode, metrics)
    valid, dt = arrays["valid"], float(protocol["control_dt"])
    masks = phase_masks(arrays["state"], valid, dt)
    # Quarantine invalid rows before FK; they never enter any aggregate.
    safe = np.where(valid[..., None], arrays["state"], np.zeros_like(arrays["state"]))
    safe_next = np.where(valid[..., None], arrays["next_state"], safe)
    safe[..., 3] = np.where(valid, safe[..., 3], 1.)
    safe_next[..., 3] = np.where(valid, safe_next[..., 3], 1.)
    pair = np.concatenate((features(adapter, safe), features(adapter, safe_next)), -1).reshape(-1, 406)
    mean, std, pools = reference
    normalized_pair = ((pair - mean) / std).astype(np.float32)
    all_indices = np.flatnonzero(valid.reshape(-1))
    d_values = {label: score_pairs(pair[all_indices], scorer) for label, scorer in scorers.items()}
    result = {"provenance": provenance, "physical_task_metrics": {
        key: value for key, value in metrics["results"][mode].items()
        if key not in ("per_env", "transitions_recording", "trace_recording")},
        "action_smoothness": smoothness(arrays["action"], arrays["previous_action"], valid, arrays["boundary"], dt),
        "phases": {}}
    valid_count = int(valid.sum())
    for phase_no, (name, mask) in enumerate(masks.items()):
        selected = sample_indices(mask, maximum, seed + phase_no)
        count = int(mask.sum())
        row = {"transition_count": count, "fraction_of_finite_first_episode": count / valid_count if valid_count else None,
               "sampled_nn_queries": len(selected), "sample_seed": seed + phase_no,
               "contact_cost": {"foot_slip": summary(arrays["contact_cost"][..., 0][mask]),
                                "self_collision": summary(arrays["contact_cost"][..., 1][mask])},
               "expert_nearest_neighbor_rmse": {group: summary(nearest_rmse(normalized_pair[selected], pool))
                                                for group, pool in pools.items()},
               "frozen_discriminator": {}}
        # score arrays use the common valid-row ordering, shared across D22/D48.
        score_mask = mask.reshape(-1)[all_indices]
        for label, (raw, style) in d_values.items():
            chosen = raw[score_mask]
            row["frozen_discriminator"][label] = {
                "raw_score": summary(chosen), "style_reward": summary(style[score_mask]),
                "zero_style_reward_fraction": float((style[score_mask] == 0).mean()) if len(chosen) else None,
                "outside_direct_style_gradient_support_fraction": float(((chosen <= -1) | (chosen >= 3)).mean()) if len(chosen) else None,
            }
        result["phases"][name] = row
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, nargs="*", default=[], help="Optional frozen grounded203 D checkpoints (e.g. 22k and 48k)")
    parser.add_argument("--evaluations", type=Path, nargs="+", required=True, help="Evaluation directories containing metrics.json and transitions_MODE.npz")
    parser.add_argument("--output", type=Path, required=True, help="A new JSON filename or a new directory; never overwrites")
    parser.add_argument("--motion-root", type=Path, default=REPO / "src/assets/motions/g1/amp")
    parser.add_argument("--xml", type=Path, default=REPO / "src/assets/robots/unitree_g1/xmls/g1.xml")
    parser.add_argument("--max-pairs", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20261109)
    parser.add_argument("--threads", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    if not 1 <= args.max_pairs <= 4096:
        parser.error("--max-pairs must be between 1 and 4096")
    args.output = args.output.expanduser().resolve()
    if args.output.exists():
        parser.error("--output must not already exist")
    torch.set_num_threads(args.threads)
    evaluations = []
    for directory in args.evaluations:
        directory = directory.expanduser().resolve()
        metrics = json.loads((directory / "metrics.json").read_text())
        if metrics.get("protocol", {}).get("task") not in ("Unitree-G1-AMP-Flat", "Unitree-G1-AMP-Flat-Aligned"):
            raise ValueError("Only physically equivalent G1 Flat/Flat-Aligned tasks are supported")
        evaluations.append((directory, metrics))
    pairing = audit_pairing(evaluations)
    adapter = make_cpu_adapter(args.xml)
    mean, std, pools, experts = expert_reference(adapter, args.motion_root, args.max_pairs, args.seed)
    scorers = {}
    for index, path in enumerate(args.checkpoint):
        path = path.expanduser().resolve()
        label = f"{index}_{path.stem}"
        scorers[label] = load_scorer(path)
    result = {
        "schema": "amp-recovery-style-analysis-v1", "device": "cpu", "threads": args.threads,
        "arguments": {key: [str(item) for item in value] if isinstance(value, list) else str(value) if isinstance(value, Path) else value
                      for key, value in vars(args).items()},
        "feature_metadata": amp_feature_metadata("grounded_v2"),
        "body_names": list(adapter.body_names), "amp_body_names": list(adapter.amp_body_names),
        "source_sha256": {str(path.relative_to(REPO)) if path.is_relative_to(REPO) else str(path): sha256(path) for path in
                          (Path(__file__).resolve(), REPO / "src/amp_features.py", REPO / "src/wmac/state.py", args.xml.resolve())},
        "initial_state_pairing": pairing, "experts": experts,
        "frozen_discriminators": {label: scorer[3] for label, scorer in scorers.items()},
        "definitions": {
            "nearest_neighbor": "min over fixed sampled expert pairs of sqrt(mean(((physical_pair-expert_pair)/expert_std)**2)), over 406 coordinates; lower is locally closer",
            "phases": "Transition start: low_or_tilted = pelvis height<.55m OR up_z<.5; upright = height>.65m AND up_z>.8; intermediate = neither. first_2_seconds uses start time<2s.",
            "contact_cost": "Captured real reward-manager _step_reward divided by its reward weight. The current manager stores _step_reward=value/scale, so these costs have no extra control-dt or reward-weight factor. Slip is sum of squared contacted-foot XY speeds; self-collision uses the actual configured sensor cost. Neither is a contact force reconstructed from AMP.",
            "style_reward": "checkpoint amp_reward_coef * max(1-.25*(D_score-1)^2,0), with the same frozen D/normalizer for all policies",
        },
        "limitations": [
            "Nearest-neighbor matching is local to one 20ms pair; neighbors may jump between clips/times. It does not establish long-term temporal order, recovery sequence style, or correct contacts.",
            "A frozen training discriminator is not an independent ground-truth style evaluator; differences between D22 and D48 can expose scorer sensitivity, not causal proof.",
            "Motion NPZ files contain no expert actions or measured contact labels; action smoothness and physical contact costs compare policies only.",
            "Failed recovery can change phase occupancy. Read whole-episode metrics together with first-2-second and low/tilted/upright conditional metrics and their counts.",
            "The native legacy195 or grounded203 training schemas remain recorded; common canonical203 evaluation features do not convert a PPO195 checkpoint or its discriminator.",
            "Finite first-episode physical transitions include the pre-reset terminal transition; post-reset episodes are excluded. Pair differences never span reset/push boundaries.",
        ], "evaluations": {}}
    for directory, metrics in evaluations:
        label = directory.name
        if label in result["evaluations"]:
            raise ValueError(f"Duplicate evaluation name {label}")
        entry = {"directory": str(directory), "metrics_sha256": sha256(directory / "metrics.json"),
                 "training_checkpoint": metrics.get("checkpoint"),
                 "training_feature_schema": metrics.get("protocol", {}).get("amp_feature_schema", "legacy"),
                 "native_task": metrics.get("protocol", {}).get("task"), "modes": {}}
        for mode in metrics["results"]:
            print(f"Analyzing {label}/{mode} on CPU", flush=True)
            entry["modes"][mode] = analyze_mode(directory, metrics, mode, adapter,
                (mean, std, pools), scorers, args.max_pairs, args.seed + 10)
        result["evaluations"][label] = entry
    if args.output.suffix.lower() == ".json":
        args.output.parent.mkdir(parents=True, exist_ok=True)
        path = args.output
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        path = args.output / "style_analysis.json"
    # Exclusive creation is checked again after potentially lengthy CPU work.
    with path.open("x") as handle:
        json.dump(json_clean(result), handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"Style analysis: {path}", flush=True)


if __name__ == "__main__":
    main()
