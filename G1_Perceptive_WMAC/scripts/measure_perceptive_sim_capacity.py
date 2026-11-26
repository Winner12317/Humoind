"""Bounded, isolated MuJoCo capacity probes; never changes training defaults.

Each candidate runs in a fresh process with the same terrain/seed/action stream.
Only the simulator is constructed. This does not certify full WM training memory.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@dataclass(frozen=True)
class Candidate:
    nconmax: int
    njmax: int
    ccd_iterations: int
    contact_sensor_maxmatch: int | None = None

    @classmethod
    def parse(cls, value):
        try:
            values = tuple(int(item) for item in value.split(":"))
        except ValueError as error:
            raise argparse.ArgumentTypeError("Use nconmax:njmax:ccd_iterations[:maxmatch]") from error
        if len(values) not in (3, 4) or min(values) < 1:
            raise argparse.ArgumentTypeError("Use three or four positive integers separated by ':'")
        return cls(*values)

    def label(self):
        return "_".join(str(value) for value in asdict(self).values() if value is not None)


def warning_counts(text):
    lowered = text.lower()
    return {
        "heightfield_pair_overflow": lowered.count("height field collision overflow"),
        "ccd_workspace_overflow": lowered.count("ccd overflow"),
        "contact_sensor_match_overflow": lowered.count("contact match overflow"),
        "out_of_memory": lowered.count("out of memory"),
    }


def initial_capacity_requirement(message):
    match = re.search(r"\bnconmax overflow \(nconmax must be >=\s*(\d+)\)", message)
    return int(match.group(1)) if match else None


def summarize_counters(nacon, nefc, ncollision, contacts_per_world, naconmax, njmax):
    """nconmax is a shared-pool quota, not a per-world contact cutoff."""
    import numpy as np
    nefc = np.asarray(nefc)
    return {
        "nacon": int(nacon), "ncollision": int(ncollision),
        "nefc_max": int(nefc.max(initial=0)),
        "contacts_per_world_max": int(np.asarray(contacts_per_world).max(initial=0)),
        "contact_pool_overflow": bool(nacon > naconmax),
        "broadphase_pool_overflow": bool(ncollision > naconmax),
        # Some kernels reject an equality's last block already at the boundary.
        "constraint_at_or_over_capacity": bool(np.any(nefc >= njmax)),
        "constraint_overflow": bool(np.any(nefc > njmax)),
    }


def generated_action(step, num_envs, mode, amplitude=.3):
    """A reproducible bounded PD-target perturbation, not a walking policy."""
    import numpy as np
    if mode == "zero": return np.zeros((num_envs, 29), dtype=np.float32)
    phase = (2*np.pi*step*.02/2 + np.arange(29)[None, :]*.7
        + (np.arange(num_envs)[:, None] % 4)*np.pi/2)
    return (amplitude*np.sin(phase)).astype(np.float32)


def grid_assignment(num_envs, rows, cols):
    import numpy as np
    if min(num_envs, rows, cols) < 1: raise ValueError("grid dimensions must be positive")
    indices = np.arange(num_envs, dtype=np.int64)
    return indices % rows, (indices // rows) % cols


def apply_grid_assignment(env):
    """Calibration-only fixed cells, preserving Scene's shared origin tensor."""
    import torch
    terrain = env.scene.terrain
    if terrain.terrain_origins is None: raise ValueError("grid assignment requires generated terrain")
    rows, cols = terrain.terrain_origins.shape[:2]
    levels, kinds = grid_assignment(env.num_envs, rows, cols)
    old_types = terrain.terrain_types.clone()
    old_kind_list = old_types.cpu().tolist()
    first_by_type = {kind: old_kind_list.index(kind) for kind in set(old_kind_list)}
    if not set(kinds.tolist()).issubset(first_by_type):
        raise ValueError("grid assignment cannot remap command caches from the initial terrain columns")
    remap = torch.as_tensor([first_by_type[int(kind)] for kind in kinds], device=env.device)
    command = env.command_manager.get_term("twist")
    for name in ("_hiking_speed_low", "_hiking_speed_high", "random_velocity_mask"):
        value = getattr(command, name, None)
        if value is not None: value.copy_(value[remap].clone())
    terrain.terrain_levels.copy_(torch.as_tensor(levels, device=env.device))
    terrain.terrain_types.copy_(torch.as_tensor(kinds, device=env.device))
    terrain.env_origins.copy_(terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types])
    if env.scene.env_origins.data_ptr() != terrain.env_origins.data_ptr():
        raise RuntimeError("Scene origins do not reference the assigned terrain origins")


def state71_to_qpos_qvel(state):
    """WM's world omega -> MuJoCo free-joint local omega (wxyz quaternion)."""
    import numpy as np
    state = np.asarray(state)
    if state.shape[-1] != 71: raise ValueError("pre-reset snapshots require 71D StateAdapter next_state")
    quaternion = state[..., 3:7]
    norm = np.linalg.norm(quaternion, axis=-1, keepdims=True)
    if np.any(norm < 1e-8): raise ValueError("snapshot contains a zero root quaternion")
    quaternion = quaternion / norm
    inverse_vector = -quaternion[..., 1:]
    omega_world = state[..., 10:13]
    uv = np.cross(inverse_vector, omega_world)
    omega_local = omega_world + 2*(quaternion[..., :1]*uv + np.cross(inverse_vector, uv))
    return (np.concatenate((state[..., :7], state[..., 13:42]), axis=-1),
        np.concatenate((state[..., 7:10], omega_local, state[..., 42:71]), axis=-1))


def select_snapshots(trace, count, num_envs, state="post-control"):
    """Uniform fixed states, including the initial state and final stored frame."""
    import numpy as np
    if state == "pre-reset":
        qpos, qvel = state71_to_qpos_qvel(trace["next_state"])
    elif state == "post-control":
        qpos, qvel = np.asarray(trace["qpos"]), np.asarray(trace["qvel"])
    else: raise ValueError("snapshot-state must be pre-reset or post-control")
    if qpos.ndim != 3 or qvel.ndim != 3 or qpos.shape[:2] != qvel.shape[:2]:
        raise ValueError("snapshot trace needs qpos/qvel[frames,num_envs,state_dimensions]")
    if qpos.shape[1] != num_envs or not len(qpos):
        raise ValueError("snapshot trace num_envs must match the probe, with at least one frame")
    if "initial_qpos" in trace and "initial_qvel" in trace:
        qpos = np.concatenate((np.asarray(trace["initial_qpos"])[None], qpos))
        qvel = np.concatenate((np.asarray(trace["initial_qvel"])[None], qvel))
        source_indices = np.arange(len(qpos)) - 1
    else:
        source_indices = np.arange(len(qpos))
    if count < 1: raise ValueError("snapshot-cases must be positive")
    selected = np.unique(np.linspace(0, len(qpos)-1, min(count, len(qpos)), dtype=int))
    qpos, qvel = qpos[selected].copy(), qvel[selected].copy()
    if not np.isfinite(qpos).all() or not np.isfinite(qvel).all():
        raise ValueError("selected snapshot states must be finite")
    return source_indices[selected], qpos, qvel


def sorted_contact_rows(worldid, geom, position, distance):
    """Canonical stored-contact rows: world, geom0, geom1, x, y, z, dist."""
    import numpy as np
    rows = np.column_stack((worldid, geom, position, distance)).astype(np.float64)
    if not len(rows): return rows.reshape(0, 7)
    order = np.lexsort(tuple(rows[:, column] for column in reversed(range(rows.shape[1]))))
    return rows[order]


def compare_traces(reference, candidate, *, atol, rtol):
    import numpy as np
    result = {}
    keys = ("initial_qpos", "initial_qvel", "qpos", "qvel", "next_state", "reward", "terminated", "timeout",
        "snapshot_qpos", "snapshot_qvel", "qacc", "nefc", "constraint_type_counts", "constraint_force_summary")
    keys = (*keys, *(key for key in reference.keys() if key.startswith(("contact__", "contacts_case__"))))
    for key in keys:
        if key not in reference or key not in candidate: continue
        left, right = reference[key], candidate[key]
        if left.shape != right.shape:
            result[key] = {"same_shape": False, "reference_shape": list(left.shape), "candidate_shape": list(right.shape)}
            continue
        finite = bool(np.isfinite(left).all() and np.isfinite(right).all())
        equal = np.isclose(left, right, atol=atol, rtol=rtol) if left.dtype.kind != "b" else left == right
        difference = np.abs(left.astype(np.float64) - right.astype(np.float64))
        result[key] = {"same_shape": True, "finite": finite,
            "allclose": bool(finite and equal.all()),
            "max_abs_difference": float(difference.max(initial=0)) if finite else None,
            "rmse": (float(np.sqrt(np.mean(difference**2))) if difference.size else 0.) if finite else None,
            "first_differing_step": (int(np.flatnonzero(~equal.reshape(len(equal), -1).all(-1))[0])
                if equal.ndim and not equal.all() else None)}
    return result


def _digest_arrays(values):
    digest = hashlib.sha256()
    for key, value in sorted(values.items()):
        digest.update(key.encode()); digest.update(str(value.shape).encode())
        digest.update(str(value.dtype).encode()); digest.update(value.tobytes())
    return digest.hexdigest()


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def _sample_process_gpu_mib(pid):
    try:
        result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory",
            "--format=csv,noheader,nounits"], text=True, capture_output=True, timeout=5)
        matches = [float(row.split(",")[1].strip()) for row in result.stdout.splitlines()
            if row.split(",")[0].strip() == str(pid)]
        return sum(matches) if matches else None
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None


def measure_snapshot_forward(args, env, observe, contact_rows, result, output):
    """Cold-force forward cases: collision/constraint diagnostics, not rollout."""
    import numpy as np
    import torch
    import mujoco_warp._src.types as warp_types
    data, bridge = env.sim.wp_data, env.sim.data
    with np.load(args.snapshot_trace, allow_pickle=False) as trace:
        indices, qpos, qvel = select_snapshots(trace, args.snapshot_cases, args.num_envs, args.snapshot_state)
    if qpos.shape[-1] != env.sim.mj_model.nq or qvel.shape[-1] != env.sim.mj_model.nv:
        raise ValueError("snapshot dimensions do not match the current physical model")
    if args.snapshot_state == "pre-reset":
        indexing = env.adapter.robot.indexing
        free_q = indexing.free_joint_q_adr.cpu().numpy()
        free_v = indexing.free_joint_v_adr.cpu().numpy()
        joint_q = indexing.joint_q_adr.cpu().numpy()
        joint_v = indexing.joint_v_adr.cpu().numpy()
        if (len(free_q), len(free_v), len(joint_q), len(joint_v)) != (7, 6, 29, 29):
            raise ValueError("pre-reset state71 conversion requires the current single-free-root 29-joint robot")
        mapped_qpos, mapped_qvel = np.empty_like(qpos), np.empty_like(qvel)
        mapped_qpos[..., free_q], mapped_qpos[..., joint_q] = qpos[..., :7], qpos[..., 7:]
        mapped_qvel[..., free_v], mapped_qvel[..., joint_v] = qvel[..., :6], qvel[..., 6:]
        # Initial arrays were recorded directly in MuJoCo layout; only the
        # next_state71-derived rows require the StateAdapter layout mapping.
        mapped_qpos[indices == -1] = qpos[indices == -1]
        mapped_qvel[indices == -1] = qvel[indices == -1]
        qpos, qvel = mapped_qpos, mapped_qvel
    result.update(probe_mode="snapshot_forward", counter_sampling="after each frozen-state forward",
        snapshot_source=str(args.snapshot_trace.resolve()), snapshot_cases_requested=args.snapshot_cases,
        snapshot_source_state=args.snapshot_state,
        snapshot_source_frame_indices=indices.tolist(), snapshot_cases_completed=0,
        snapshot_inputs_sha256=_digest_arrays({"qpos": qpos, "qvel": qvel}),
        steps_requested=0, steps_completed=0,
        frozen_state_conditions={"zero_fields": ["qacc_warmstart", "qfrc_applied", "xfrc_applied", "ctrl", "time"],
            "integration": False, "scope": "collision/constraint diagnostic; not actual actuator or motion trajectory"})
    arrays = {"snapshot_qpos": qpos, "snapshot_qvel": qvel, "snapshot_source_frame_indices": indices}
    qacc_rows, nefc_rows, type_rows, force_rows = [], [], [], []
    num_types = len(warp_types.ConstraintType)
    started = time.perf_counter()
    with torch.no_grad():
        for case, (position, velocity) in enumerate(zip(qpos, qvel)):
            bridge.qpos.copy_(torch.as_tensor(position, device=args.device))
            bridge.qvel.copy_(torch.as_tensor(velocity, device=args.device))
            for name in result["frozen_state_conditions"]["zero_fields"]:
                getattr(bridge, name).zero_()
            env.sim.forward(); observe()
            qacc_rows.append(data.qacc.numpy().copy())
            nefc = data.nefc.numpy().copy(); nefc_rows.append(nefc)
            constraint_type, force = data.efc.type.numpy(), data.efc.force.numpy()
            active = np.arange(data.njmax)[None] < np.minimum(nefc, data.njmax)[:, None]
            type_rows.append(np.stack([((constraint_type == kind) & active).sum(1)
                for kind in range(num_types)], axis=-1))
            force = np.where(active, force, 0.)
            force_rows.append(np.stack((force.sum(1), np.abs(force).sum(1),
                np.sqrt((force.astype(np.float64)**2).sum(1)), np.abs(force).max(1, initial=0.)), axis=-1))
            nacon = min(int(data.nacon.numpy()[0]), data.naconmax)
            arrays[f"contacts_case__{case:04d}"] = sorted_contact_rows(data.contact.worldid.numpy()[:nacon],
                data.contact.geom.numpy()[:nacon], data.contact.pos.numpy()[:nacon], data.contact.dist.numpy()[:nacon])
            result["snapshot_cases_completed"] = case + 1
    arrays.update(qacc=np.stack(qacc_rows), nefc=np.stack(nefc_rows),
        constraint_type_counts=np.stack(type_rows), constraint_force_summary=np.stack(force_rows),
        **{key: np.stack(values) for key, values in contact_rows.items()})
    result["forward_seconds_with_diagnostic_sync"] = time.perf_counter() - started
    result["constraint_type_names"] = [kind.name for kind in warp_types.ConstraintType]
    result["constraint_force_summary_columns"] = ["signed_sum", "absolute_sum", "l2_norm", "max_absolute"]
    result["contact_row_columns"] = ["worldid", "geom0", "geom1", "x", "y", "z", "dist"]
    result["physics_outputs_finite"] = all(np.isfinite(value).all().item() for value in arrays.values())
    np.savez_compressed(output/"trace.npz", **arrays)


def worker(args):
    import numpy as np
    import torch
    import warp as wp
    from scripts.train_wmac_perceptive import make_environment
    from scripts.evaluate_wmac_perceptive import load_actor, PolicyHistory, seed_all
    from src.wmac_perceptive.config import PerceptiveConfig
    candidate = Candidate.parse(args.worker_candidate)
    output = Path(args.worker_output); output.mkdir(parents=True, exist_ok=True)
    result = {"status": "initializing", "candidate": asdict(candidate), "num_envs": args.num_envs,
        "probe_mode": "snapshot_forward" if args.snapshot_trace else "dynamic_rollout",
        "seed": args.seed, "steps_requested": args.steps, "steps_completed": 0,
        "scope": "simulator only; no world model or optimizer constructed",
        "ccd_actual_iterations": None, "ccd_actual_iterations_available": False,
        "ccd_workspace_count_available": False,
        "counter_sampling": "every physics substep after graph launch and synchronization"}
    env = None; counters = []; contact_sensor_max = {}; contact_sensor_unavailable = []; contact_rows = {}
    try:
        seed_all(args.seed, args.device)
        if args.device.startswith("cuda"):
            # manual_seed_all is lazy; peak counters require an initialized
            # CUDA context and selected device before environment creation.
            torch.cuda.set_device(args.device)
            torch.cuda.reset_peak_memory_stats(args.device)
        actor, cfg = None, PerceptiveConfig.defaults("supermimic")
        if args.checkpoint and args.record_actions and not args.snapshot_trace:
            actor, cfg, _ = load_actor(args.checkpoint, args.device)
        elif args.checkpoint:
            # Read the same configuration without constructing an unnecessary Actor.
            from src.wmac_perceptive.config import config_from_payload
            cfg = config_from_payload(torch.load(args.checkpoint, map_location="cpu", weights_only=False))
        cfg.num_envs, cfg.seed, cfg.device = args.num_envs, args.seed, args.device

        def transform(env_cfg):
            env_cfg.sim.nconmax = candidate.nconmax
            env_cfg.sim.njmax = candidate.njmax
            env_cfg.sim.mujoco.ccd_iterations = candidate.ccd_iterations
            maxmatch = candidate.contact_sensor_maxmatch or args.contact_sensor_maxmatch
            if maxmatch is not None: env_cfg.sim.contact_sensor_maxmatch = maxmatch
            if args.max_init_terrain_level is not None:
                if not 0 <= args.max_init_terrain_level < cfg.terrain_rows:
                    raise ValueError("max-init-terrain-level must fit the unchanged terrain row count")
                env_cfg.scene.terrain.max_init_terrain_level = args.max_init_terrain_level
            if args.terrain_assignment == "grid": env_cfg.curriculum = {}
            return env_cfg

        env = make_environment(cfg, play=False, env_config_transform=transform)
        data = env.sim.wp_data
        result["allocated"] = {key: int(getattr(data, key))
            for key in ("nworld", "naconmax", "naccdmax", "njmax", "njmax_pad", "njmax_nnz")
            if hasattr(data, key)}
        result["allocated"]["contact_sensor_maxmatch"] = env.cfg.sim.contact_sensor_maxmatch
        result["allocated"]["ccd_iterations"] = int(env.sim.mj_model.opt.ccd_iterations)
        result["simulator_cuda_graph"] = bool(env.sim.use_cuda_graph)
        result["contact_sensor_layout"] = {name: {"slot_count": len(getattr(sensor, "_slots", ())),
            "primaries": sorted({slot.primary_name for slot in getattr(sensor, "_slots", ())}),
            "fields": list(sensor.cfg.fields), "reduction": sensor.cfg.reduce,
            "num_slots_per_primary": sensor.cfg.num_slots,
            "found_view_shapes": [list(slot.data_view.shape) for slot in getattr(sensor, "_slots", ())
                if slot.field_name == "found" and slot.data_view is not None]}
            for name, sensor in env.scene.sensors.items() if "Contact" in type(sensor).__name__}
        result["versions"] = {name: importlib.metadata.version(name)
            for name in ("mujoco", "mujoco-warp", "mjlab", "warp-lang", "torch")}
        # 3.8.1 EPA workspace allocation formula, not measured occupancy. Other
        # mesh multiccd arrays are additional; counters are internal temporaries.
        import mujoco_warp._src.types as warp_types
        epa_iterations = candidate.ccd_iterations
        result["ccd_workspace_estimate"] = {
            "kind": "EPA arrays lower bound, source allocation formula, not live occupancy",
            "bytes": int(data.naccdmax * (16*(10+2*epa_iterations)
                + 20*(6+warp_types.MJ_MAX_EPAFACES*epa_iterations)
                + 4*warp_types.MJ_MAX_EPAHORIZON)),
            "epa_iterations": epa_iterations,
            "naccdmax": int(data.naccdmax)}
        def observe():
            wp.synchronize_device(env.sim.wp_device)
            nacon, ncollision = int(data.nacon.numpy()[0]), int(data.ncollision.numpy()[0])
            worlds = data.contact.worldid.numpy()[:min(nacon, data.naconmax)]
            worlds = worlds[(worlds >= 0) & (worlds < env.num_envs)]
            counters.append(summarize_counters(nacon, data.nefc.numpy(), ncollision,
                np.bincount(worlds, minlength=env.num_envs), data.naconmax, data.njmax))
            for name, sensor in env.scene.sensors.items():
                if "Contact" not in type(sensor).__name__: continue
                slots = getattr(sensor, "_slots", ())
                fields = {field: [slot.data_view for slot in slots
                    if slot.field_name == field and slot.data_view is not None] for field in ("found", "force")}
                if not fields["found"] and name not in contact_sensor_unavailable:
                    contact_sensor_unavailable.append(name)
                for field, chunks in fields.items():
                    if not chunks: continue
                    # Live sensordata views, not sensor.data: reading does not
                    # update contact history, air-time state or cache timestamps.
                    joined = torch.cat(chunks, dim=-1)
                    if field == "found":
                        contact_sensor_max[name] = max(contact_sensor_max.get(name, 0), int(joined.max().item()))
                    key = f"contact__{name}__{field}"
                    contact_rows.setdefault(key, []).append(joined[:args.trace_contact_envs].cpu().numpy().copy())

        original_step = env.sim.step

        def measured_step():
            original_step(); observe()

        env.sim.step = measured_step
        if args.terrain_assignment == "grid": apply_grid_assignment(env)
        seed_all(args.seed, args.device)
        env.reset(seed=args.seed)
        terrain = env.scene.terrain
        geometry = {key: np.asarray(getattr(env.sim.mj_model, key)).copy()
            for key in ("hfield_data", "hfield_nrow", "hfield_ncol", "hfield_size", "geom_pos", "geom_quat", "geom_size")}
        geometry.update(terrain_types=terrain.terrain_types.cpu().numpy(),
            terrain_levels=terrain.terrain_levels.cpu().numpy(), env_origins=env.scene.env_origins.cpu().numpy())
        result["terrain_and_assignment_sha256"] = _digest_arrays(geometry)
        result["calibration_overrides"] = {"max_init_terrain_level": args.max_init_terrain_level,
            "terrain_assignment": args.terrain_assignment,
            "terrain_curriculum_disabled_for_fixed_grid": args.terrain_assignment == "grid"}
        result["initial_terrain_assignment"] = {key: geometry[key].tolist()
            for key in ("terrain_types", "terrain_levels")}
        result["initial_terrain_distribution"] = {key: {str(int(value)): int(count)
            for value, count in zip(*np.unique(geometry[key], return_counts=True))}
            for key in ("terrain_types", "terrain_levels")}
        result["initial_terrain_cells_observed"] = sorted({(int(level), int(kind))
            for level, kind in zip(geometry["terrain_levels"], geometry["terrain_types"])})
        if args.snapshot_trace:
            measure_snapshot_forward(args, env, observe, contact_rows, result, output)
            result["status"] = "completed"
            return 0
        initial_qpos, initial_qvel = data.qpos.numpy().copy(), data.qvel.numpy().copy()
        actions_path = Path(args.actions_npz)
        recorded = []
        if not args.record_actions:
            with np.load(actions_path, allow_pickle=False) as payload:
                actions = payload["actions"].copy()
            if actions.shape != (args.steps, args.num_envs, 29) or not np.isfinite(actions).all():
                raise ValueError("actions must be finite [steps, num_envs, 29] with matching probe dimensions")
        controller = None
        if actor is not None:
            from src.tasks.amp_perceptive.sensors import read_depth_frame
            controller = PolicyHistory(actor, env.num_envs, env.device, cfg.action_limit)
            controller.reset(read_depth_frame(env))
        rows = {key: [] for key in ("qpos", "qvel", "next_state", "reward", "terminated", "timeout")}
        started = time.perf_counter()
        with torch.inference_mode():
            for step in range(args.steps):
                action = (controller.action(env.obs_buf["actor"]) if controller is not None
                    else torch.as_tensor(generated_action(step, args.num_envs, args.action_mode,
                        args.action_amplitude), device=args.device) if args.record_actions
                    else torch.as_tensor(actions[step], device=args.device))
                if args.record_actions: recorded.append(action.cpu().numpy().copy())
                env.step(action)
                transition = env.transition
                rows["qpos"].append(data.qpos.numpy().copy()); rows["qvel"].append(data.qvel.numpy().copy())
                for key in ("next_state", "reward", "terminated", "timeout"):
                    rows[key].append(transition[key].cpu().numpy().copy())
                if controller is not None: controller.after_step(transition, read_depth_frame(env))
                result["steps_completed"] = step + 1
        wp.synchronize_device(env.sim.wp_device)
        result["rollout_seconds_with_diagnostic_sync"] = time.perf_counter() - started
        if args.record_actions:
            actions = np.stack(recorded)
            np.savez_compressed(actions_path, actions=actions)
        result["actions_sha256"] = _digest_arrays({"actions": actions})
        result["physics_outputs_finite"] = all(np.isfinite(np.stack(values)).all().item()
            for values in rows.values())
        np.savez_compressed(output/"trace.npz", initial_qpos=initial_qpos, initial_qvel=initial_qvel,
            **{key: np.stack(values) for key, values in {**rows, **contact_rows}.items()})
        result["status"] = "completed"
    except Exception as error:
        result.update(status="failed", error_type=type(error).__name__, error=str(error))
        required = initial_capacity_requirement(str(error))
        if required is not None:
            result["required_host_initial_contacts"] = required
            result["capacity_requirement_source"] = "MuJoCoWarp put_data host-MjData initialization guard; no contacts removed or bypassed"
        traceback.print_exc()
    finally:
        result["physics_substeps_observed"] = 0 if args.snapshot_trace else len(counters)
        result["frozen_forward_calls_observed"] = len(counters) if args.snapshot_trace else 0
        result["counter_maxima"] = {key: (any(row[key] for row in counters) if isinstance(counters[0][key], bool)
            else max(row[key] for row in counters)) for key in counters[0]} if counters else {}
        result["contact_sensor_matches_physics_substep_max"] = contact_sensor_max
        result["contact_trace_envs"] = min(args.trace_contact_envs, args.num_envs)
        result["contact_sensor_count_unavailable"] = contact_sensor_unavailable
        result["contact_sensor_match_at_or_over_capacity"] = (any(
            value >= env.cfg.sim.contact_sensor_maxmatch for value in contact_sensor_max.values())
            if env is not None else None)
        _write_json(output/"physics_counters.json", counters)
        if args.device.startswith("cuda"):
            try:
                result["torch_memory"] = {"peak_allocated_bytes": torch.cuda.max_memory_allocated(args.device),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(args.device)}
                free, total = torch.cuda.mem_get_info(args.device)
                result["device_memory_at_end"] = {"free_bytes": free, "total_bytes": total}
            except Exception: pass
        _write_json(output/"worker.json", result)
        if env is not None:
            try: env.close()
            except Exception: traceback.print_exc()
    return 0 if result["status"] == "completed" else 1


def run(args):
    import numpy as np
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=False)
    actions_path = args.actions_npz.resolve() if args.actions_npz else output/"actions.npz"
    reports = []
    reference_path = None; reference_result = None
    for index, candidate in enumerate(args.candidate):
        directory = output/f"{index:02d}_{candidate.label()}"; directory.mkdir()
        command = [sys.executable, str(Path(__file__).resolve()), "--worker-candidate", ":".join(
            str(value) for value in asdict(candidate).values() if value is not None),
            "--worker-output", str(directory), "--num-envs", str(args.num_envs), "--seed", str(args.seed),
            "--steps", str(args.steps), "--device", args.device, "--actions-npz", str(actions_path),
            "--action-mode", args.action_mode, "--action-amplitude", str(args.action_amplitude),
            "--trace-contact-envs", str(args.trace_contact_envs),
            "--terrain-assignment", args.terrain_assignment]
        if args.max_init_terrain_level is not None:
            command.extend(["--max-init-terrain-level", str(args.max_init_terrain_level)])
        if args.contact_sensor_maxmatch is not None:
            command.extend(["--contact-sensor-maxmatch", str(args.contact_sensor_maxmatch)])
        if args.checkpoint: command.extend(["--checkpoint", str(args.checkpoint.resolve())])
        if args.snapshot_trace:
            command.extend(["--snapshot-trace", str(args.snapshot_trace.resolve()),
                "--snapshot-cases", str(args.snapshot_cases), "--snapshot-state", args.snapshot_state])
        elif index == 0 and args.actions_npz is None: command.append("--record-actions")
        with (directory/"stdout.log").open("w") as stdout:
            process = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=subprocess.STDOUT, start_new_session=True)
            began = time.monotonic(); peak = None; timed_out = False
            while process.poll() is None:
                sample = _sample_process_gpu_mib(process.pid)
                if sample is not None: peak = max(peak or 0., sample)
                if time.monotonic() - began > args.timeout:
                    timed_out = True; os.killpg(process.pid, signal.SIGTERM)
                    try: process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL); process.wait()
                    break
                time.sleep(args.poll_seconds)
        worker_path = directory/"worker.json"
        result = json.loads(worker_path.read_text()) if worker_path.exists() else {"status": "failed", "candidate": asdict(candidate)}
        result.update(returncode=process.returncode, timed_out=timed_out,
            sampled_process_gpu_peak_mib=peak, gpu_peak_is_sampled_lower_bound=True,
            warnings=warning_counts((directory/"stdout.log").read_text(errors="replace")))
        pool_flags = ("contact_pool_overflow", "broadphase_pool_overflow", "constraint_at_or_over_capacity")
        result["no_detected_pool_overflow"] = (result["status"] == "completed"
            and not any(result.get("counter_maxima", {}).get(key, False) for key in pool_flags)
            and not result["warnings"]["ccd_workspace_overflow"]
            and not result["warnings"]["contact_sensor_match_overflow"]
            and not result.get("contact_sensor_match_at_or_over_capacity", False))
        result["complete_physics_acceptance"] = False
        result["acceptance_note"] = ("Heightfield pair overflow remains: cannot certify physical correctness."
            if result["warnings"]["heightfield_pair_overflow"] else
            "Bounded workload only; no claim of complete terrain/policy or full-training coverage.")
        trace_path = directory/"trace.npz"
        if trace_path.exists():
            if reference_path is None:
                reference_path = trace_path; reference_result = result
                result["comparison"] = {"reference": True}
            else:
                with np.load(reference_path, allow_pickle=False) as left, np.load(trace_path, allow_pickle=False) as right:
                    result["comparison"] = compare_traces(left, right, atol=args.atol, rtol=args.rtol)
                if args.snapshot_trace:
                    result["same_snapshot_inputs"] = result.get("snapshot_inputs_sha256") == reference_result.get("snapshot_inputs_sha256")
                else:
                    result["same_action_stream"] = result.get("actions_sha256") == reference_result.get("actions_sha256")
                result["same_terrain_and_assignment"] = result.get("terrain_and_assignment_sha256") == reference_result.get("terrain_and_assignment_sha256")
        _write_json(directory/"result.json", result); reports.append(result)
        _write_json(output/"summary.json", {"reports": reports, "atol": args.atol, "rtol": args.rtol,
            "snapshot_trace": str(args.snapshot_trace.resolve()) if args.snapshot_trace else None,
            "actions": None if args.snapshot_trace else str(actions_path),
            "action_source": "frozen-state forward; no actions or integration" if args.snapshot_trace else "supplied stream" if args.actions_npz else
                "checkpoint Actor, recorded once" if args.checkpoint else
                "zero action standing probe" if args.action_mode == "zero" else "bounded scripted PD-target perturbation",
            "limitations": ["No WM/optimizer memory; verify integrated training separately.",
                "Zero actions do not establish walking or difficult-terrain coverage.",
                "CCD iteration saturation and internal CCD count are not exposed in installed MuJoCoWarp.",
                "State divergence alone does not identify capacity truncation; repeated reference probes help bound numerical variability."]})
        print(f"{directory.name}: {result['status']}; GPU sampled peak={peak} MiB; warnings={result['warnings']}", flush=True)
        if not args.snapshot_trace and not actions_path.exists():
            print("Reference did not create a complete action stream; stopping comparisons.", flush=True); break
    return 0 if len(reports) == len(args.candidate) and all(report["status"] == "completed" for report in reports) else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Candidate.parse, action="append", help="Repeat nconmax:njmax:ccd_iterations[:maxmatch]; first is reference")
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--contact-sensor-maxmatch", type=int)
    parser.add_argument("--max-init-terrain-level", type=int,
        help="Calibration-only initial difficulty override; e.g. 9 samples the existing ten rows, without changing terrain geometry")
    parser.add_argument("--terrain-assignment", choices=("random", "grid"), default="random",
        help="Calibration-only fixed cells i%%rows,(i//rows)%%cols; grid disables the probe's terrain curriculum")
    parser.add_argument("--action-mode", choices=("zero", "scripted"), default="zero")
    parser.add_argument("--action-amplitude", type=float, default=.3,
        help="Bounded scripted policy-action amplitude; does not alter actuator scaling")
    parser.add_argument("--trace-contact-envs", type=int, default=16,
        help="Found/force traces for the first N envs at each physics substep; capacity maxima cover every env")
    parser.add_argument("--checkpoint", type=Path, help="First candidate records its Actor actions; later candidates replay those fixed actions")
    parser.add_argument("--actions-npz", type=Path, help="Existing NPZ with finite actions[steps,num_envs,29]")
    parser.add_argument("--snapshot-trace", type=Path,
        help="Frozen-state forward mode: uniformly sample qpos/qvel from an existing trace NPZ, without integration")
    parser.add_argument("--snapshot-cases", type=int, default=16)
    parser.add_argument("--snapshot-state", choices=("pre-reset", "post-control"), default="post-control",
        help="Use terminal next_state71 (convert world omega to local qvel) or stored post-control qpos/qvel")
    parser.add_argument("--output", type=Path, help="New output directory; contains complete worker logs and traces")
    parser.add_argument("--timeout", type=float, default=1200., help="Per-candidate wall limit in seconds")
    parser.add_argument("--poll-seconds", type=float, default=.2)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--worker-candidate", help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", help=argparse.SUPPRESS)
    parser.add_argument("--record-actions", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if min(args.num_envs, args.steps, args.trace_contact_envs, args.snapshot_cases) < 1 or min(args.timeout, args.poll_seconds) <= 0:
        parser.error("num-envs, steps, timeout and poll-seconds must be positive")
    if args.contact_sensor_maxmatch is not None and args.contact_sensor_maxmatch < 1:
        parser.error("contact-sensor-maxmatch must be positive")
    if not 0 <= args.action_amplitude <= 1:
        parser.error("action-amplitude must lie between 0 and 1")
    if args.atol < 0 or args.rtol < 0: parser.error("comparison tolerances must be nonnegative")
    if args.worker_candidate: return worker(args)
    if not args.candidate or args.output is None: parser.error("--candidate and --output are required")
    if args.actions_npz and not args.actions_npz.is_file(): parser.error("actions-npz does not exist")
    if args.snapshot_trace and not args.snapshot_trace.is_file(): parser.error("snapshot-trace does not exist")
    if args.snapshot_trace and args.actions_npz: parser.error("snapshot-trace and actions-npz are alternative probe modes")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
