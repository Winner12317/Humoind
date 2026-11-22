"""Offline physical-reward decomposition of paired PPO/WMAC locomotion traces.

Only captured physical rewards are authoritative. Captured qacc/contact data
and algebraic action/joint-limit terms are separated; endpoint FK is never used
to claim exact reconstruction of lagged height/velocity/body rewards. No policy,
world model, discriminator or additional neural network is loaded or trained.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


COMPONENT_NAMES = (
    "real_task_reward", "qacc_cost", "qacc_reward", "action_rate_cost", "action_rate_reward",
    "foot_slip_cost", "foot_slip_reward", "self_collision_cost", "self_collision_reward",
    "joint_limit_cost", "joint_limit_reward", "termination_reward", "known_penalties_reward",
    "remaining_unseparated_reward", "height_positive_reward", "locomotion_positive_reward_group",
    "height_mask_active_fraction", "physical_substep_qacc_mean_square", "physical_force_mean_square",
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean(value):
    import numpy as np
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(item) for item in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def reward_spec(task, height_semantics):
    # Importing task configuration constructs no simulator or neural network.
    import mujoco
    import mjlab.tasks  # noqa: F401
    import src.tasks  # noqa: F401
    from mjlab.tasks.registry import load_env_cfg
    from src.recovery_rewards import configure_recovery_height_reward
    from src.assets.robots.unitree_g1.g1_constants import G1_XML
    cfg = load_env_cfg(task)
    configure_recovery_height_reward(cfg, height_semantics)
    weights = {name: float(term.weight) for name, term in cfg.rewards.items()}
    known = {"joint_acc_l2", "action_rate_l2", "foot_slip", "self_collisions", "joint_pos_limits", "is_terminated"}
    remainder = set(weights) - known
    if remainder != {"track_anchor_linear_velocity", "track_anchor_angular_velocity", "track_root_height", "body_ang_vel_xy_l2"}:
        raise ValueError(f"Unexpected reward terms cannot be silently included: {remainder}")
    for name in remainder - {"track_root_height"}:
        params = cfg.rewards[name].params
        if params.get("mask_delay") is not True or float(params.get("delay_env_rew_ratio", 1.)) != 0.:
            raise ValueError("Exclusive recorded-context positive-reward decomposition requires the original zero delay ratios")
    if cfg.rewards["track_root_height"].params.get("mask_delay") is not True:
        raise ValueError("Height must be the original delay-mask-only positive reward")
    model = mujoco.MjModel.from_xml_path(str(G1_XML))
    return {"weights": weights, "scale_by_dt": bool(cfg.scale_rewards_by_dt),
            "soft_joint_limit_factor": float(cfg.scene.entities["robot"].articulation.soft_joint_pos_limit_factor),
            "remaining_terms": sorted(remainder), "xml": str(G1_XML), "xml_sha256": sha256(G1_XML),
            "positive_reward_separation": {"method": "Algebraic residual plus verified exclusive real reward masks, without endpoint FK",
                "height_active": "next_context[33]>0 AND next_context[34]>0",
                "height_only_when_active": True, "other3_positive_terms_zero_when_active": True}}, model


def soft_limits(directory, metadata, model, factor):
    import mujoco
    import numpy as np
    with np.load(directory / "physical_model.npz", allow_pickle=False) as physical:
        saved = physical["jnt_range"].copy()
    ids = np.asarray([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                      for name in metadata["joint_names"]])
    if len(ids) != 29 or (ids < 0).any() or len(set(ids.tolist())) != 29:
        raise ValueError("Cannot verify all29 recorded joint names against the G1 XML")
    if saved.shape[1] != model.njnt:
        raise ValueError("Saved global joint ordering is not the G1 XML ordering")
    hard = saved[:, ids].astype(np.float32)
    if not np.allclose(hard, model.jnt_range[ids][None], rtol=1e-6, atol=1e-6):
        raise ValueError("Saved joint ranges differ from the verified XML mapping")
    # Mirror Entity._initialize_data's float32 soft-limit construction.
    middle = (hard[..., 0] + hard[..., 1]) / np.float32(2.)
    width = hard[..., 1] - hard[..., 0]
    half = np.float32(.5) * width * np.float32(factor)
    return np.stack((middle-half, middle+half), -1)


def decompose(path, weights, scale, limits, decimation, joint_names):
    import numpy as np
    with np.load(path, allow_pickle=False) as record:
        r = {name: record[name].copy() for name in ("reward", "qacc_cost", "action", "previous_action",
             "action_delta", "contact_cost", "next_state", "state", "context", "next_context", "valid", "terminated", "qacc", "force")}
    times, environments = r["valid"].shape
    if environments != limits.shape[0] or r["qacc"].shape != (times*decimation, environments, 29):
        raise ValueError("Physical qacc dimensions disagree with the recording protocol")
    last_qacc = r["qacc"][decimation-1::decimation].astype(np.float64)
    if not np.allclose(r["qacc_cost"][r["valid"]], np.square(last_qacc).sum(-1)[r["valid"]], rtol=1e-6, atol=1e-3):
        raise ValueError("Recorded qacc cost does not match the captured physical last substep")
    if not np.allclose(r["action_delta"], r["action"]-r["previous_action"], rtol=1e-6, atol=1e-6):
        raise ValueError("Recorded action_delta differs from the action-manager action difference")
    qacc_cost = r["qacc_cost"].astype(np.float64)
    action_cost = np.square(r["action_delta"].astype(np.float64)).sum(-1)
    positions = r["next_state"][..., 13:42]
    limit_excess = np.maximum(limits[None, ..., 0]-positions, 0.) + np.maximum(positions-limits[None, ..., 1], 0.)
    limit_cost = limit_excess.sum(-1).astype(np.float64)
    slip, collision = [r["contact_cost"][..., i].astype(np.float64) for i in range(2)]
    qacc_reward = scale * weights["joint_acc_l2"] * qacc_cost
    action_reward = scale * weights["action_rate_l2"] * action_cost
    slip_reward = scale * weights["foot_slip"] * slip
    collision_reward = scale * weights["self_collisions"] * collision
    limit_reward = scale * weights["joint_pos_limits"] * limit_cost
    death_reward = scale * weights["is_terminated"] * r["terminated"].astype(np.float64)
    known = qacc_reward + action_reward + slip_reward + collision_reward + limit_reward + death_reward
    total = r["reward"].astype(np.float64)
    positive = total-known
    height_active = (r["next_context"][..., 33] > 0) & (r["next_context"][..., 34] > 0)
    height_reward = np.where(height_active, positive, 0.)
    locomotion_reward = np.where(~height_active, positive, 0.)
    physical_qacc = np.square(r["qacc"].astype(np.float64).reshape(times, decimation, environments, 29)).mean((1, 3))
    physical_force = np.square(r["force"].astype(np.float64).reshape(times, decimation, environments, 29)).mean((1, 3))
    values = (total, qacc_cost, qacc_reward, action_cost, action_reward, slip, slip_reward, collision,
              collision_reward, limit_cost, limit_reward, death_reward, known, positive,
              height_reward, locomotion_reward, height_active.astype(np.float64), physical_qacc, physical_force)
    data = dict(zip(COMPONENT_NAMES, values))
    data.update({f"joint_limit_reward_detail__{name}": scale*weights["joint_pos_limits"]*limit_excess[..., joint].astype(np.float64)
                 for joint, name in enumerate(joint_names)})
    valid = r["valid"].astype(bool)
    for name, value in data.items():
        if not np.isfinite(value[valid]).all():
            raise ValueError(f"Nonfinite valid {name} in {path}")
    return data, valid, (r["state"][..., 2] < .55) | (r["context"][..., 34] > 0)


def per_start_mean(values, mask):
    import numpy as np
    count = mask.sum(0)
    total = np.where(mask, values, 0.).sum(0)
    return np.divide(total, count, out=np.full(total.shape, np.nan), where=count > 0)


def summarize_group(data, masks, dt, bootstrap_count, seed):
    import numpy as np
    per_policy = {}
    output = {}
    component_names = tuple(data["a"])
    for policy in ("a", "b"):
        labels = (policy, policy+"_repeat")
        vectors = {}
        for name in component_names:
            repeats = np.stack([per_start_mean(data[label][name], masks[label]) for label in labels])
            count = np.isfinite(repeats).sum(0)
            vectors[name] = np.divide(np.nansum(repeats, 0), count,
                out=np.full(count.shape, np.nan), where=count > 0)
        per_policy[policy] = vectors
        usable = np.isfinite(vectors["real_task_reward"])
        means = {name: float(values[np.isfinite(values)].mean()) if np.isfinite(values).any() else None
                 for name, values in vectors.items()}
        means["physical_substep_qacc_rms"] = math.sqrt(means["physical_substep_qacc_mean_square"]) if usable.any() else None
        means["physical_force_rms_Nm"] = math.sqrt(means["physical_force_mean_square"]) if usable.any() else None
        output[policy] = {"repeat_average_then_equal_initial_state_mean": means,
                          "initial_states_with_samples": int(usable.sum()),
                          "control_step_exposures_per_repeat": [int(masks[label].sum()) for label in labels],
                          "mean_exposure_seconds_per_initial_state": float(np.mean([masks[label].sum(0).mean()*dt for label in labels]))}
    paired = np.isfinite(per_policy["a"]["real_task_reward"]) & np.isfinite(per_policy["b"]["real_task_reward"])
    paired_count = int(paired.sum())
    differences = {}
    for name in component_names:
        delta = (per_policy["b"][name]-per_policy["a"][name])[paired]
        differences[name] = {"mean_b_minus_a": float(delta.mean()) if paired_count else None,
                             "b_higher_initial_states": int((delta > 0).sum()),
                             "per_initial_state_b_minus_a": delta.tolist()}
        if paired_count and bootstrap_count and name in ("real_task_reward", "qacc_reward", "known_penalties_reward", "remaining_unseparated_reward"):
            # Environment/initial-state is the independent unit; never frames.
            indices = np.random.default_rng(seed).integers(0, paired_count, (bootstrap_count, paired_count))
            differences[name]["exploratory_paired_bootstrap_mean_delta_95"] = np.quantile(delta[indices].mean(1), [.025, .975]).tolist()
    reward_delta = (per_policy["b"]["real_task_reward"]-per_policy["a"]["real_task_reward"])[paired]
    qacc_delta = (per_policy["b"]["qacc_cost"]-per_policy["a"]["qacc_cost"])[paired]
    output["paired"] = {"initial_states": paired_count, "differences": differences,
        "b_higher_reward_and_higher_qacc_initial_states": int(((reward_delta > 0) & (qacc_delta > 0)).sum()),
        "reward_delta_qacc_delta_correlation": float(np.corrcoef(reward_delta, qacc_delta)[0, 1])
            if paired_count > 1 and reward_delta.std() > 0 and qacc_delta.std() > 0 else None}
    return output


def analyze_run(directory, bootstrap_count, seed):
    import numpy as np
    directory = directory.expanduser().resolve()
    metadata_path = directory / "metrics.json"
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("schema") != "amp-wmac-paired-effort-v1":
        raise ValueError(f"Unsupported paired recorder schema in {directory}")
    protocol = metadata["protocol"]
    if protocol["world_model_used_for_control"] or protocol["network_parameters_updated"]:
        raise ValueError("This diagnostic requires physical fixed-policy trajectories")
    spec, model = reward_spec(protocol["task"], protocol["physical_reward"])
    dt = float(protocol["control_dt"])
    scale = dt if spec["scale_by_dt"] else 1.
    limits = soft_limits(directory, metadata, model, spec["soft_joint_limit_factor"])
    data, validity, phase, files = {}, {}, {}, []
    labels = ("a", "a_repeat", "b", "b_repeat")
    for label in labels:
        path = directory / f"transitions_{label}.npz"
        if label not in metadata["trials"]:
            raise ValueError("Both policies require within-initial-state repeat recordings")
        data[label], validity[label], phase[label] = decompose(path, spec["weights"], scale, limits, int(protocol["physics_substeps"]), metadata["joint_names"])
        files.append({"path": str(path), "sha256": sha256(path)})
    common = np.logical_and.reduce([validity[label] for label in labels])
    times = np.arange(common.shape[0])*dt
    groups = {}
    for name, window in (("fixed_0p1_to_2_seconds", (times >= .1) & (times < 2.)),
                         ("fixed_2_to_5_seconds", (times >= 2.) & (times < 5.)),
                         ("fixed_0_to_5_seconds", (times < 5.))):
        groups[name] = summarize_group(data, {label: common & window[:, None] for label in labels}, dt, bootstrap_count, seed)
    for name, recovery in (("own_actual_recovery_phase_secondary", True), ("own_actual_walk_phase_secondary", False)):
        masks = {label: validity[label] & (phase[label] == recovery) for label in labels}
        groups[name] = summarize_group(data, masks, dt, bootstrap_count, seed)
    return {"directory": str(directory), "metadata_sha256": sha256(metadata_path),
            "protocol": protocol, "checkpoints": metadata["checkpoints"], "recordings": files,
            "reward_spec": spec, "reward_scale": scale,
            "measurement_provenance": {
                "real_task_reward": "Captured physical reward at reward time, including true qacc and every configured task term; excludes AMP style.",
                "qacc": "Captured29-joint physical last-substep qacc; recorded double sum-of-squares verified against physical samples; weighted contribution reconstructed within floating-point rounding.",
                "action_rate": "Recorded executed action minus previous action, verified against action_delta; algebraic sum-of-squares.",
                "contact": "Reward-time captured _step_reward/weight; RewardManager _step_reward is unscaled raw*weight, so dt applied once here.",
                "joint_limit": "Reconstructed from captured pre-reset qpos; XML joint-name/range mapping and source float32 soft-limit factor verified. It is a formula reconstruction, not a separately captured reward term.",
                "remaining_unseparated_reward": "Captured total minus known reconstructed penalties; contains4 positive terms together.",
                "height_positive_reward": "Residual assigned to height only when recorded pre-reset next_context delay flag/counter are active; original configuration proves all3 other positive terms are zero then. No endpoint-FK reconstruction.",
                "locomotion_positive_reward_group": "Residual when the recorded delay mask is inactive; height is then exactly zero. Contains linear/angular velocity tracking and body angular stabilization together, not separately attributed.",
            },
            "success": {label: {name: metadata["trials"][label][name] for name in
                ("recovery_success_count", "continued_walk_count", "first_episode_survival_count", "first_stable_recovery_seconds_mean")}
                for label in labels}, "groups": groups}


def joint_position_statistics(position_repeats, mask_repeats, limits, joint_names, scale, weight):
    """Real trajectories weight starts equally; reference is a single clip."""
    import numpy as np
    excess_repeats, pooled_positions, pooled_excess = [], [], []
    for positions, mask in zip(position_repeats, mask_repeats):
        excess = np.maximum(limits[None, ..., 0]-positions, 0.) + np.maximum(positions-limits[None, ..., 1], 0.)
        excess_repeats.append(excess)
        pooled_positions.append(positions[mask])
        pooled_excess.append(excess[mask])
    selected_q, selected_excess = np.concatenate(pooled_positions), np.concatenate(pooled_excess)
    def repeat_initial_mean(values):
        repeats = np.stack([per_start_mean(value, mask) for value, mask in zip(values, mask_repeats)])
        count = np.isfinite(repeats).sum(0)
        average = np.divide(np.nansum(repeats, 0), count, out=np.full(count.shape, np.nan), where=count > 0)
        return float(average[np.isfinite(average)].mean()) if np.isfinite(average).any() else None
    result = {"selected_frames_across_repeats": int(len(selected_q)),
              "any_soft_limit_exceeded_fraction": repeat_initial_mean([(e > 1e-6).any(-1).astype(float) for e in excess_repeats]),
              "mean_sum_soft_excess_rad": repeat_initial_mean([e.sum(-1) for e in excess_repeats]), "per_joint": {}}
    for joint, name in enumerate(joint_names):
        mean_excess = repeat_initial_mean([e[..., joint] for e in excess_repeats])
        result["per_joint"][name] = {
            "soft_lower_upper_rad": [float(limits[0, joint, 0]), float(limits[0, joint, 1])],
            "soft_exceeded_fraction": repeat_initial_mean([(e[..., joint] > 1e-6).astype(float) for e in excess_repeats]),
            "mean_excess_rad": mean_excess,
            "equivalent_joint_limit_reward_per_control_step": mean_excess*scale*weight if mean_excess is not None else None,
            "selected_angle_min_max_rad": [float(selected_q[:, joint].min()), float(selected_q[:, joint].max())] if len(selected_q) else None,
            "selected_pooled_excess_p95_max_rad": np.quantile(selected_excess[:, joint], [.95, 1.]).tolist() if len(selected_q) else None}
    return result


def expert_joint_limit_audit(runs):
    import mujoco
    import numpy as np
    from src.assets.robots.unitree_g1.g1_constants import G1_XML
    model = mujoco.MjModel.from_xml_path(str(G1_XML))
    joint_ids = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_HINGE)
    joint_names = [model.joint(int(joint)).name for joint in joint_ids]
    body_names = [model.body(body).name for body in range(1, model.nbody)]
    source = ROOT / "src/assets/motions/g1/amp/Recovery"
    spec = runs[0]["reward_spec"]
    factor = spec["soft_joint_limit_factor"]
    hard = model.jnt_range[joint_ids].astype(np.float32)
    middle = (hard[:, 0]+hard[:, 1])/np.float32(2.)
    half = np.float32(.5)*(hard[:, 1]-hard[:, 0])*np.float32(factor)
    limits = np.stack((middle-half, middle+half), -1)[None]
    dt = float(runs[0]["protocol"]["control_dt"])
    scale = dt if spec["scale_by_dt"] else 1.
    weight = spec["weights"]["joint_pos_limits"]
    result = {"source_group": "Original Recovery NPZ only; no WalkandRun clips or default/reset poses",
              "joint_names": joint_names, "body_names": body_names,
              "limits_source": {"xml": str(G1_XML), "xml_sha256": sha256(G1_XML), "soft_factor": factor,
                  "hard_limits_by_joint": {name: hard[joint].tolist() for joint, name in enumerate(joint_names)},
                  "joint_limit_reward_weight": weight, "physical_reward_scale": scale},
              "low_or_tilted_phase": "root_z<0.55 OR normalized root up_z<0.5; applied identically to reference and real pre-reset endpoint poses",
              "aggregation": "Within each real initial state, average the two repeat means; then average initial states. Reference is a single time-series clip. Angle extrema/quantiles pool selected frames and are descriptive.",
              "reference": [], "policies": [],
              "limitations": ["Applying the original joint-position penalty to expert kinematics is a counterfactual formula evaluation, not a captured expert reward or proof of a training cause.",
                  "The pose predicate is aligned, but expert and policies still have different phase durations and frame distributions.",
                  "FK is used only to verify the reference joint/body mapping, not to reconstruct physical height/velocity rewards."]}
    for path in sorted(source.glob("*.npz")):
        with np.load(path, allow_pickle=False) as motion:
            q = motion["joint_pos"].copy()
            body_p = motion["body_pos_w"].copy()
            body_q = motion["body_quat_w"].copy()
            fps = float(motion["fps"].reshape(-1)[0])
        if q.shape[1] != len(joint_names) or body_p.shape[1] != len(body_names):
            raise ValueError("Reference motion dimensions disagree with the XML natural order")
        data = mujoco.MjData(model)
        predicted_p, predicted_q = [], []
        for frame in range(len(q)):
            data.qpos[:3], data.qpos[3:7], data.qpos[7:] = body_p[frame, 0], body_q[frame, 0], q[frame]
            mujoco.mj_kinematics(model, data)
            predicted_p.append(data.xpos[1:].copy()); predicted_q.append(data.xquat[1:].copy())
        predicted_p, predicted_q = np.asarray(predicted_p), np.asarray(predicted_q)
        normalized_q = body_q.astype(np.float64)
        normalized_q /= np.linalg.norm(normalized_q, axis=-1, keepdims=True)
        predicted_q /= np.linalg.norm(predicted_q, axis=-1, keepdims=True)
        angle_error = 2*np.arccos(np.clip(np.abs(np.sum(predicted_q*normalized_q, -1)), 0., 1.))
        position_error = np.linalg.norm(predicted_p-body_p, axis=-1)
        if float(position_error.max()) > 1e-4 or float(angle_error.max()) > 1e-4:
            raise ValueError("Reference joint/body mapping was not verified by full-clip XML FK")
        up = 1-2*(normalized_q[:, 0, 1]**2+normalized_q[:, 0, 2]**2)
        low = (body_p[:, 0, 2] < .55) | (up < .5)
        result["reference"].append({"path": str(path), "sha256": sha256(path), "fps": fps,
            "frames": len(q), "low_or_tilted_frames": int(low.sum()),
            "mapping_verified": True, "mapping_method": "All2575 frame FK body positions and sign-invariant normalized quaternions agree with stored motion using XML hinge/body natural order",
            "full_clip_fk_position_rmse_m": float(np.sqrt(np.square(predicted_p-body_p).mean())),
            "full_clip_fk_body_position_max_error_m": float(position_error.max()),
            "full_clip_fk_quaternion_max_angle_error_rad": float(angle_error.max()),
            "all_frames": joint_position_statistics([q[:, None]], [np.ones((len(q), 1), dtype=bool)], limits, joint_names, scale, weight),
            "low_or_tilted_frames_only": joint_position_statistics([q[:, None]], [low[:, None]], limits, joint_names, scale, weight)})
    for run in runs:
        metadata = json.loads((Path(run["directory"])/"metrics.json").read_text())
        if metadata["joint_names"] != joint_names:
            raise ValueError("Recorded natural joint order differs from the verified reference order")
        policy_limits = soft_limits(Path(run["directory"]), metadata, model, factor)
        for policy in ("a", "b"):
            positions, masks = [], []
            for label in (policy, policy+"_repeat"):
                with np.load(Path(run["directory"])/f"transitions_{label}.npz", allow_pickle=False) as record:
                    state, valid = record["next_state"], record["valid"].astype(bool)
                    root_q = state[..., 3:7].astype(np.float64)
                    root_q /= np.linalg.norm(root_q, axis=-1, keepdims=True)
                    up = 1-2*(root_q[..., 1]**2+root_q[..., 2]**2)
                    positions.append(state[..., 13:42].copy())
                    masks.append(valid & ((state[..., 2] < .55) | (up < .5)))
            result["policies"].append({"directory": run["directory"], "policy": policy,
                "checkpoint_sha256": run["checkpoints"][policy]["sha256"],
                "low_or_tilted_frames_only": joint_position_statistics(positions, masks, policy_limits, joint_names, scale, weight)})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", default=[
        ROOT/"logs/ppo60000_vs_wmac25200_recovery_seed2026", ROOT/"logs/ppo60000_vs_wmac25200_walk_seed2026",
        ROOT/"logs/ppo60000_vs_wmac25200_recovery_seed2027",
        ROOT/"logs/ppo60000_vs_wmac40800_recovery_seed2026", ROOT/"logs/ppo60000_vs_wmac40800_walk_seed2026"])
    parser.add_argument("--output", type=Path, default=ROOT/"docs/validation/recovery_gap_reward_20261116.json")
    parser.add_argument("--bootstrap", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20261116)
    args = parser.parse_args()
    if args.bootstrap < 0:
        parser.error("--bootstrap must be nonnegative")
    if len(set(path.expanduser().resolve() for path in args.runs)) != len(args.runs):
        parser.error("--runs must be distinct recording directories")
    output = {"format": "amp-wmac-recovery-gap-physical-reward-v1", "neural_networks_loaded": 0,
              "training_updates": 0, "bootstrap_samples": args.bootstrap, "bootstrap_seed": args.seed,
              "aggregation": "Per initial state: average each repeat's valid-window mean, then average the two repeats; equal initial-state weights. RMS averages squared quantities before sqrt.",
              "limitations": ["Fixed windows use common-valid frames across all4 trajectories; own-phase masks are secondary and have different exposure times.",
                  "Actual phase means the Replay predicate on real recorded inputs (state_z<0.55 OR context[34]>0), not the entire interval until stable recovery; it can end earlier or re-enter.",
                  "Only captured total task reward is authoritative; component reconstruction may differ by floating-point rounding.",
                  "Height versus the other3 positive terms is separated algebraically by verified mutually exclusive recorded delay masks; the3 locomotion terms remain unseparated because endpoint FK can differ from reward-time derived quantities.",
                  "Higher reward on a trajectory with higher qacc is an observed tradeoff, not proof that force itself is rewarded or causes the policy difference.",
                  "Task reward excludes AMP style. No claim about matching reference-motion distribution follows from task reward or success alone.",
                  "Bootstrap intervals describe these fixed-checkpoint initial states, not independent training seeds or universal causal effects."],
              "runs": []}
    for directory in args.runs:
        result = analyze_run(directory, args.bootstrap, args.seed)
        output["runs"].append(result)
        print(f'{result["protocol"]["mode"]} seed{result["protocol"]["seed"]}:', flush=True)
        for group in ("fixed_0p1_to_2_seconds", "fixed_2_to_5_seconds", "own_actual_recovery_phase_secondary"):
            table = result["groups"][group]
            a, b = [table[p]["repeat_average_then_equal_initial_state_mean"] for p in ("a", "b")]
            print(f'  {group}: task={a["real_task_reward"]}/{b["real_task_reward"]}, '
                  f'qacc_contribution={a["qacc_reward"]}/{b["qacc_reward"]}, '
                  f'known_penalties={a["known_penalties_reward"]}/{b["known_penalties_reward"]}, '
                  f'remaining={a["remaining_unseparated_reward"]}/{b["remaining_unseparated_reward"]}', flush=True)
    output["cross_run_shared_start_checks"] = []
    for index, first in enumerate(output["runs"]):
        for second in output["runs"][index+1:]:
            same_scene = all(first["protocol"][name] == second["protocol"][name] for name in ("mode", "seed"))
            if not same_scene:
                continue
            output["cross_run_shared_start_checks"].append({
                "first": first["directory"], "second": second["directory"],
                "same_initial_state_hash": first["protocol"]["initial_state_sha256"] == second["protocol"]["initial_state_sha256"],
                "same_physical_model_hash": first["protocol"]["physical_model_sha256"] == second["protocol"]["physical_model_sha256"],
                "same_ppo_checkpoint_hash": first["checkpoints"]["a"]["sha256"] == second["checkpoints"]["a"]["sha256"],
                "note": "GPU replay results can still differ despite equal starting hashes; each run retains its own paired PPO repeats."})
    output["expert_joint_limit_audit"] = expert_joint_limit_audit(output["runs"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(clean(output), ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    print(f'Offline diagnostic saved: {args.output}', flush=True)


if __name__ == "__main__":
    main()
