"""Frozen-model diagnosis on recorded physical WMAC trajectories, without training.

Open-loop predictions use recorded actions and contexts. Ground-contact sets are
explicit geometry proxies (collision-primitive floor clearance), not recorded
contact sensors or contact-force ground truth. CPU is the default device.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b""):
            result.update(chunk)
    return result.hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def read_recording(path, source_sha):
    path = Path(path).resolve()
    metadata = json.loads((path.parent/"metrics.json").read_text())
    trial = path.stem.removeprefix("transitions_")
    if trial not in ("b", "b_repeat") or metadata["checkpoints"]["b"]["sha256"] != source_sha:
        raise ValueError(f"Recording is not from the specified frozen WMAC source: {path}")
    if metadata["protocol"]["world_model_used_for_control"] or metadata["protocol"]["network_parameters_updated"]:
        raise ValueError("Use fixed-policy physical recordings")
    fields = ("state", "next_state", "action", "context", "contact_cost", "valid", "boundary",
              "terminated", "timeout", "qacc_cost", "recovery_phase")
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key].copy() for key in fields}
    shape = data["valid"].shape
    for key, width in (("state", 71), ("next_state", 71), ("action", 29), ("context", 35), ("contact_cost", 2)):
        if data[key].shape != (*shape, width):
            raise ValueError(f"Invalid {key} shape: {path}")
    finite = data["valid"].astype(bool).copy()
    for key in ("state", "next_state", "action", "context", "contact_cost", "qacc_cost"):
        finite &= np.isfinite(data[key]).reshape(*shape, -1).all(-1)
    done = data["terminated"] | data["timeout"]
    if np.any(done & ~data["boundary"]):
        raise ValueError("Terminal/timeout transition must be marked as a boundary")
    # Catch unmarked resets/events rather than crossing them in open-loop data.
    continuous = np.max(np.abs(data["next_state"][:-1]-data["state"][1:]), axis=-1) <= 1e-5
    discontinuity = finite[:-1] & finite[1:] & ~data["boundary"][:-1] & ~continuous
    if discontinuity.any():
        raise ValueError(f"Unmarked physical discontinuity: {int(discontinuity.sum())} rows in {path}")
    data["finite_valid"] = finite
    return data, {"path": str(path), "sha256": sha256(path), "metadata_sha256": sha256(path.parent/"metrics.json"),
                  "mode": metadata["protocol"]["mode"], "seed": metadata["protocol"]["seed"], "trial": trial,
                  "valid_rows": int(data["valid"].sum()), "finite_valid_rows": int(finite.sum()),
                  "recorded_ground_contact_sensor_flags": False}


def eligible(data, length):
    valid = data["finite_valid"]
    result = np.zeros_like(valid)
    for step in range(max(0, len(valid)-length+1)):
        good = valid[step:step+length].all(0)
        if length > 1:
            good &= ~data["boundary"][step:step+length-1].any(0)
        result[step] = good
    return result


def up_z(state):
    q = state[..., 3:7]
    q = q/np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-8)
    return 1.-2.*(q[..., 1]**2+q[..., 2]**2)


def clearance(adapter, states):
    """Exact flat-floor z support of XML collision primitives under true FK."""
    import mujoco
    import torch
    from src.wmac.state import quat_apply, quat_mul, quat_to_matrix
    model = adapter.env.sim.mj_model
    names = [i for i in range(model.ngeom) if "collision" in (model.geom(i).name or "")]
    if not names:
        raise ValueError("No named collision primitives in the canonical G1 XML")
    body_local = {int(global_id): index for index, global_id in enumerate(adapter.body_ids.tolist())}
    body = torch.tensor([body_local[int(model.geom_bodyid[i])] for i in names], dtype=torch.long)
    local_pos = torch.tensor(model.geom_pos[names], dtype=torch.float32)
    local_quat = torch.tensor(model.geom_quat[names], dtype=torch.float32)
    size = torch.tensor(model.geom_size[names], dtype=torch.float32)
    types = np.asarray(model.geom_type[names])
    allowed = tuple(int(kind) for kind in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_SPHERE,
               mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_CYLINDER))
    if any(int(kind) not in allowed for kind in types):
        raise ValueError("Collision geometry includes an unsupported support shape")
    flat = np.asarray(states, dtype=np.float32).reshape(-1, 71)
    rows = []
    with torch.no_grad():
        for start in range(0, len(flat), 1024):
            p, q, *_ = adapter.kinematics(torch.from_numpy(flat[start:start+1024]))
            p, q = p[:, body], q[:, body]
            center = p+quat_apply(q, local_pos)
            rotation = quat_to_matrix(quat_mul(q, local_quat))
            z_axis = rotation[..., 2, 2].abs()
            support = torch.zeros_like(center[..., 2])
            for i, kind in enumerate(types):
                if int(kind) == int(mujoco.mjtGeom.mjGEOM_SPHERE):
                    support[:, i] = size[i, 0]
                elif int(kind) == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
                    support[:, i] = size[i, 0]+size[i, 1]*z_axis[:, i]
                elif int(kind) == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
                    support[:, i] = size[i, 1]*z_axis[:, i]+size[i, 0]*(1-z_axis[:, i]**2).clamp_min(0).sqrt()
                else:
                    support[:, i] = (rotation[:, i, 2].abs()*size[i]).sum(-1)
            rows.append((center[..., 2]-support).numpy())
    return np.concatenate(rows).reshape(*states.shape[:-1], len(names)), [model.geom(i).name for i in names]


def groups(data, adapter, threshold):
    before, names = clearance(adapter, data["state"])
    after, _ = clearance(adapter, data["next_state"])
    near_before, near_after = before <= threshold, after <= threshold
    ground_switch = (near_before != near_after).any(-1)
    upright_before = (data["state"][..., 2] >= .65) & (up_z(data["state"]) >= .8)
    upright_after = (data["next_state"][..., 2] >= .65) & (up_z(data["next_state"]) >= .8)
    valid = data["finite_valid"]
    qacc_p95 = np.quantile(data["qacc_cost"][valid], .95)
    recovery = data["recovery_phase"].astype(bool)
    return {"all": np.ones_like(valid), "recovery_phase": recovery,
            "upright": upright_before, "not_upright": ~upright_before,
            "upright_switch": upright_before != upright_after,
            "ground_geometry_switch": ground_switch,
            "ground_geometry_stable": ~ground_switch,
            "recovery_ground_switch": recovery & ground_switch,
            "recovery_ground_stable": recovery & ~ground_switch,
            "upright_ground_switch": upright_before & ground_switch,
            "upright_ground_stable": upright_before & ~ground_switch,
            "physical_qacc_top5pct": data["qacc_cost"] >= qacc_p95}, {
        "ground_proxy_threshold_m": threshold, "ground_proxy_geometry": names,
        "ground_proxy_definition": "A named XML collision primitive changes floor clearance <= threshold during the real transition.",
        "ground_proxy_is_recorded_contact_force": False, "floor_z_m": 0.,
        "qacc_tail_threshold": float(qacc_p95),
        "ground_switch_threshold_sensitivity": {str(distance): int((((before <= distance) != (after <= distance)).any(-1)&valid).sum())
                                                  for distance in (.002, .005, .010)}}


def move_adapter(adapter, device):
    import torch
    adapter.device = device
    for name, value in vars(adapter).items():
        if isinstance(value, torch.Tensor):
            setattr(adapter, name, value.to(device))
    adapter._fk_levels = tuple(replace(level, **{name: getattr(level, name).to(device)
        for name in ("parent_slots", "joint_indices", "joint_mask", "offsets", "orientations", "axes", "pivots", "qrefs")})
        for level in adapter._fk_levels)


def summarize(prediction, target, contacts, true_contacts, predicted_score, real_score, coefficient):
    delta = prediction.astype(np.float64)-target.astype(np.float64)
    finite = np.isfinite(delta).all(-1) & np.isfinite(contacts).all(-1) & np.isfinite(predicted_score)
    result = {"samples": len(delta), "finite_predictions": int(finite.sum()),
              "nonfinite_fraction": float(1.-finite.mean())}
    if not finite.any():
        return result
    delta, pred, true = delta[finite], prediction[finite], target[finite]
    result.update(joint_position_rmse_rad=float(np.sqrt(np.mean(delta[:, 13:42]**2))),
                  joint_velocity_rmse_rad_s=float(np.sqrt(np.mean(delta[:, 42:71]**2))),
                  joint_position_rmse_by_joint_rad=np.sqrt(np.mean(delta[:, 13:42]**2, axis=0)).tolist(),
                  joint_velocity_rmse_by_joint_rad_s=np.sqrt(np.mean(delta[:, 42:71]**2, axis=0)).tolist(),
                  root_height_mae_m=float(np.abs(delta[:, 2]).mean()),
                  root_height_bias_m=float(delta[:, 2].mean()),
                  root_up_z_mae=float(np.abs(up_z(pred)-up_z(true)).mean()))
    for i, name in enumerate(("foot_slip", "self_collisions")):
        p, t = contacts[finite, i].astype(np.float64), true_contacts[finite, i].astype(np.float64)
        result[name] = {"raw_mae": float(np.abs(p-t).mean()), "raw_rmse": float(np.sqrt(np.mean((p-t)**2))),
                        "log1p_mae": float(np.abs(np.log1p(p)-np.log1p(np.maximum(t, 0))).mean()),
                        "predicted_mean": float(p.mean()), "physical_mean": float(t.mean())}
    p, t = predicted_score[finite], real_score[finite]
    reward = lambda x: coefficient*np.maximum(0, 1.-.25*(x-1.)**2)
    result["discriminator"] = {"predicted_score_mean": float(p.mean()), "real_score_mean": float(t.mean()),
        "score_gap_mean": float((p-t).mean()), "score_gap_mae": float(np.abs(p-t).mean()),
        "amp_reward_gap_mean": float((reward(p)-reward(t)).mean()),
        "predicted_zero_style_fraction": float(((p <= -1)|(p >= 3)).mean()),
        "real_zero_style_fraction": float(((t <= -1)|(t >= 3)).mean())}
    return result


def analyze_group(data, starts, model, adapter, scorer, norm, coefficient, args):
    import torch
    results = {str(horizon): [] for horizon in args.horizons}
    with torch.no_grad():
        for first in range(0, len(starts), args.batch_size):
            selected = starts[first:first+args.batch_size]
            step, env = selected[:, 0], selected[:, 1]
            tensor = lambda value: torch.as_tensor(value, dtype=torch.float32, device=args.device)
            state = tensor(data["state"][step, env])
            for horizon in range(1, max(args.horizons)+1):
                time_index = step+horizon-1
                previous = state
                state, contact = model.step_with_contact(previous, tensor(data["action"][time_index, env]),
                                                          tensor(data["context"][time_index, env]))
                if horizon not in args.horizons:
                    continue
                actual_before = tensor(data["state"][time_index, env])
                actual_after = tensor(data["next_state"][time_index, env])
                phi = adapter.amp_features(torch.stack((previous, state, actual_before, actual_after)))
                score = scorer(torch.cat((norm(phi[[0, 2]]), norm(phi[[1, 3]])), -1)).squeeze(-1)
                results[str(horizon)].append((state.cpu().numpy(), actual_after.cpu().numpy(), contact.cpu().numpy(),
                    data["contact_cost"][time_index, env], score[0].cpu().numpy(), score[1].cpu().numpy()))
    return {key: summarize(*(np.concatenate([part[i] for part in chunks]) for i in range(6)), coefficient)
            for key, chunks in results.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--recordings", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 4, 8, 16, 32])
    parser.add_argument("--max-windows-per-group", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--groups", nargs="+", help="Optional explicit phase subset; default analyzes every declared group")
    parser.add_argument("--ground-clearance", type=float, default=.005)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if min(args.horizons) < 1 or min(args.max_windows_per_group, args.batch_size, args.threads) < 1:
        parser.error("Horizons, sample budget, batch size and threads must be positive")
    args.horizons = sorted(set(args.horizons))
    import torch
    from scripts.analyze_amp_recovery_style import make_cpu_adapter, load_scorer
    from src.wmac.checkpoints import resolve_checkpoint_config
    from src.wmac.networks import WorldModel
    from src.wmac.reference import is_official
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    checkpoint = args.checkpoint.expanduser().resolve()
    source_sha, started = sha256(checkpoint), time.perf_counter()
    adapter = make_cpu_adapter(ROOT/"src/assets/robots/unitree_g1/xmls/g1.xml")
    prepared = []
    report = {"schema": "amp-wmac-recovery-gap-model-v1", "checkpoint": str(checkpoint), "checkpoint_sha256": source_sha,
        "analysis_script_sha256": sha256(Path(__file__)),
        "g1_xml_sha256": sha256(ROOT/"src/assets/robots/unitree_g1/xmls/g1.xml"),
        "device": args.device, "precision": "float32", "tf32": False,
        "torch_version": str(torch.__version__), "horizons": args.horizons,
        "joint_order": [adapter.env.sim.mj_model.joint(int(i)).name for i in adapter.robot.indexing.joint_ids],
        "phase_definitions": {"recovery_phase": "Recorded state_z<0.55 OR recovery counter>0",
            "upright": "True state_z>=0.65 AND pelvis up_z>=0.8",
            "upright_switch": "The true transition changes the upright predicate",
            "physical_qacc_top5pct": "True last-physics-substep sum(qacc^2), above this trial's finite-valid P95"},
        "protocol": {"training": False, "normalizer_updates": False, "new_physical_rollouts": False,
            "recorded_actions": True, "recorded_contexts": True, "actor_closed_loop": False,
            "common_complete_window_length": max(args.horizons), "terminal_allowed_only_on_final_step": True,
            "max_windows_per_group": args.max_windows_per_group, "seed": args.seed},
        "limitations": ["Geometry switches are clearance proxies, not recorded contact flags or forces.",
            "State prediction error does not establish inaccurate Actor derivatives.",
            "Original frozen D is a consistency diagnostic, not independent style ground truth.",
            "Repeated trials share initial states and are repeatability checks, not independent seeds.",
            "Groups overlap; their sample counts must not be added."], "recordings": []}
    for index, path in enumerate(args.recordings):
        data, metadata = read_recording(path, source_sha)
        selections, geometry = groups(data, adapter, args.ground_clearance)
        if args.groups:
            if set(args.groups)-set(selections):
                raise ValueError(f"Unknown analysis groups: {sorted(set(args.groups)-set(selections))}")
            selections = {name: selections[name] for name in args.groups}
        allowed = eligible(data, max(args.horizons))
        groups_info = {}
        for group_index, (name, mask) in enumerate(selections.items()):
            candidates = np.argwhere(mask & allowed)
            generator = np.random.default_rng(args.seed+index*101+group_index)
            selected = candidates[generator.choice(len(candidates), min(len(candidates), args.max_windows_per_group), replace=False)] if len(candidates) else candidates
            groups_info[name] = {"eligible_windows": len(candidates), "sampled_windows": len(selected),
                                "starts_sha256": hashlib.sha256(selected.tobytes()).hexdigest()}
            prepared.append((index, name, data, selected))
        report["recordings"].append({**metadata, "geometry": geometry, "groups": groups_info})
    if args.plan_only:
        report["plan_only"] = True
    else:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        cfg = resolve_checkpoint_config(payload)
        model = WorldModel(hidden=cfg.model_hidden, activation="gelu" if is_official(cfg) else "silu",
                           output_zero_init=not is_official(cfg)).to(device).eval()
        model.load_state_dict(payload["world_model"])
        model.precision = "float32"
        scorer, norm, coefficient, info = load_scorer(checkpoint)
        scorer, norm = scorer.to(device), norm.to(device)
        move_adapter(adapter, device)
        for module in (model, scorer, norm):
            module.requires_grad_(False)
        initial_norm = {key: value.clone() for key, value in model.in_norm.state_dict().items()}
        report["discriminator_source"] = info
        for index, name, data, selected in prepared:
            if len(selected):
                report["recordings"][index]["groups"][name]["metrics"] = analyze_group(
                    data, selected, model, adapter, scorer, norm, coefficient, args)
            print(f'[model gap] record={index} group={name} windows={len(selected)}', flush=True)
        if any(not torch.equal(value, initial_norm[key]) for key, value in model.in_norm.state_dict().items()):
            raise RuntimeError("Source model normalizer changed")
        if any(parameter.grad is not None for module in (model, scorer, norm) for parameter in module.parameters()):
            raise RuntimeError("Unexpected parameter gradients")
    if sha256(checkpoint) != source_sha:
        raise RuntimeError("Source checkpoint file changed")
    report["elapsed_seconds"] = time.perf_counter()-started
    write_json(args.output, report)
    print(f'Saved {args.output}', flush=True)


if __name__ == "__main__":
    main()
