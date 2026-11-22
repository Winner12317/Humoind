"""Paired physical reward decomposition and same-trajectory height-v1 counterfactual.

The real task remains upright_v2. No world model, optimizer, or substep effort
recorder is constructed. Snapshot/model hashes must match an existing paired
PPO60000/WMAC40800 reference before any policy rollout is accepted.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from scripts.compare_recovery_effort import clean, content_hash, file_hash, physical_model_arrays


class RewardTermsRecorder:
    """Capture original reward-manager rates after reward and before reset."""
    def __init__(self, env):
        import torch
        from src.tasks.amp_loco.mdp.rewards import track_root_height
        from src.recovery_rewards import recovery_root_up_z
        self.env, self.enabled, self.latest = env, False, None
        self.names = list(env.reward_manager.active_terms)
        self.height_index = self.names.index("track_root_height")
        self.term = env.reward_manager.get_term_cfg("track_root_height")
        if self.term.func is not track_root_height or float(self.term.params.get("upright_power", 0.)) != 2.:
            raise ValueError("The real task must keep its original upright_v2 height function")
        if not env.cfg.scale_rewards_by_dt:
            raise ValueError("This diagnostic requires the original dt-scaled physical rewards")
        self.original_compute = env.metrics_manager.compute
        self.params_before = dict(self.term.params)

        def compute(*args, **kwargs):
            result = self.original_compute(*args, **kwargs)
            if self.enabled:
                with torch.no_grad():
                    terms = env.reward_manager._step_reward.detach().clone()*env.step_dt
                    original_reward = env.reward_buf.detach().clone()
                    height_v2 = terms[:, self.height_index].clone()
                    # Only the local call's upright_power changes; the manager
                    # config and its cached body-state timing remain unchanged.
                    raw_v1 = self.term.func(env, **{**self.term.params, "upright_power": 0.})
                    raw_v2 = self.term.func(env, **self.term.params)
                    height_v1 = raw_v1*float(self.term.weight)*env.step_dt
                    manager = env.termination_manager
                    active = manager._delay_env_mask & (manager._delay_counters > 0)
                    asset_cfg = self.term.params.get("asset_cfg")
                    asset = env.scene[asset_cfg.name if asset_cfg is not None else "robot"]
                    height = asset.data.body_link_pos_w[:, 0, 2]
                    up = recovery_root_up_z(asset.data.body_link_quat_w[:, 0])
                    reconstructed = terms.sum(-1)
                    errors = (reconstructed-original_reward).abs()
                    if not torch.allclose(reconstructed, original_reward, rtol=1e-5, atol=1e-6):
                        raise RuntimeError("Recorded reward-manager terms do not reconstruct the physical task reward")
                    if not torch.allclose(raw_v2*float(self.term.weight)*env.step_dt, height_v2,
                                          rtol=1e-5, atol=1e-6):
                        raise RuntimeError("Height counterfactual does not use the real reward-time state")
                    if bool((height_v1[~active] != 0).any()) or bool((height_v2[~active] != 0).any()):
                        raise RuntimeError("Recovery-height rewards escaped the original active-delay mask")
                    if self.term.params != self.params_before:
                        raise RuntimeError("Counterfactual evaluation changed the real reward configuration")
                    self.latest = {"reward_terms": terms, "height_v1": height_v1.clone(),
                        "height_v2": height_v2, "task_v2": original_reward,
                        "task_v1_counterfactual": original_reward-height_v2+height_v1,
                        "active_delay": active.detach().clone(), "cached_root_height": height.detach().clone(),
                        "cached_root_up_z": up.detach().clone(), "term_sum_abs_error": errors}
            return result
        env.metrics_manager.compute = compute

    def close(self):
        self.enabled = False
        self.env.metrics_manager.compute = self.original_compute


def rollout(env, actor, recorder, steps):
    import numpy as np
    import torch
    from src.recovery_rewards import recovery_root_up_z
    active = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    stable = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    recovered = torch.zeros_like(active)
    recovered_time = torch.full((env.num_envs,), float("nan"), device=env.device)
    observations, rows = env.obs_buf, []
    recorder.enabled = True
    try:
        with torch.no_grad():
            for step in range(steps):
                recorder.latest = None
                raw = actor(observations["actor"])
                limit = getattr(actor, "source_action_limit", None)
                action = raw.clamp(-limit, limit) if limit is not None else raw
                observations, *_ = env.step(action)
                if recorder.latest is None:
                    raise RuntimeError("The pre-reset reward hook did not run")
                transition = env.transition
                row = {key: value.detach().clone() for key, value in transition.items()}
                row.update({key: value.detach().clone() for key, value in recorder.latest.items()})
                row.update(raw_action=raw.detach().clone(), valid=active.clone())
                done = transition["terminated"] | transition["timeout"]
                up = recovery_root_up_z(transition["next_state"][:, 3:7])
                upright = (transition["next_state"][:, 2] > .65) & (up > .8)
                stable = torch.where(active & upright & ~done, stable+1, torch.zeros_like(stable))
                new = active & ~recovered & (stable >= 25)
                recovered_time[new] = (step+1)*env.step_dt
                recovered |= new
                row["upright"] = upright.clone()
                rows.append(row)
                active &= ~done
    finally:
        recorder.enabled = False
    arrays = {key: torch.stack([row[key] for row in rows]).cpu().numpy() for key in rows[0]}
    return arrays, {"recovery_success_count": int(recovered.sum()),
        "first_stable_recovery_seconds_mean": float(recovered_time[recovered].mean()) if recovered.any() else None,
        "first_stable_recovery_seconds": recovered_time.cpu().tolist(),
        "first_episode_survival_count": int(active.sum()),
        "reward_term_reconstruction_max_abs_error": float(arrays["term_sum_abs_error"].max())}


def summarize(arrays, mask, term_names):
    import numpy as np
    selected = int(mask.sum())
    result = {"environment_control_steps": selected, "active_delay_steps": int((mask & arrays["active_delay"]).sum())}
    if not selected:
        return result
    def describe(value):
        returns = np.where(mask, value, 0.).astype(np.float64).sum(0)
        return {"mean_per_selected_control_step": float(value[mask].mean()),
                "return_per_environment": returns.tolist(), "mean_return_per_environment": float(returns.mean())}
    for name in ("task_v2", "task_v1_counterfactual", "height_v1", "height_v2"):
        result[name] = describe(arrays[name])
    result["v2_minus_v1_same_trajectory"] = describe(arrays["task_v2"]-arrays["task_v1_counterfactual"])
    result["terms_v2"] = {name: describe(arrays["reward_terms"][..., i]) for i, name in enumerate(term_names)}
    return result


def run(args):
    import numpy as np
    import torch
    import mjlab.tasks  # noqa: F401
    import src.tasks  # noqa: F401
    from mjlab.tasks.registry import load_env_cfg
    from scripts.check_wmac_physics import PhysicsSnapshot
    from scripts.evaluate_wmac import keep_fallen_recovery_frames, load_actor, seed_all
    from src.recovery_rewards import configure_recovery_height_reward
    from src.wmac.env import CaptureEnv
    from src.wmac.state import StateAdapter
    reference = json.loads(args.reference_metrics.read_text())
    if reference["schema"] != "amp-wmac-paired-effort-v1":
        raise ValueError("Use the existing paired-effort reference metadata")
    protocol = reference["protocol"]
    if (protocol["mode"], protocol["num_envs"], protocol["steps"], protocol["seed"]) != ("recovery", 32, 250, 2026):
        raise ValueError("This probe is pinned to the existing 32-env/250-step/seed2026 recovery pairing")
    actors, provenance = {}, {}
    for label in ("a", "b"):
        info = reference["checkpoints"][label]
        path = Path(info["path"]).expanduser().resolve()
        if file_hash(path) != info["sha256"]:
            raise ValueError(f"Source checkpoint hash changed for {label}")
        actor, kind = load_actor(path, args.device)
        actor.precision = "float32"
        override = info.get("action_limit_explicit_override")
        if override is not None:
            actor.source_action_limit = override
        state_sha = content_hash({name: value.detach().cpu().numpy() for name, value in actor.state_dict().items()})
        if state_sha != info["policy_state_sha256"] or getattr(actor, "source_action_limit", None) != info["outer_action_limit"]:
            raise ValueError("Policy tensors or clipping differ from the reference pairing")
        actors[label], provenance[label] = actor, {**info, "path": str(path), "loaded_kind": kind}
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    cfg = load_env_cfg("Unitree-G1-AMP-Flat-Aligned")
    configure_recovery_height_reward(cfg, "upright_v2")
    cfg.scene.num_envs, cfg.seed, cfg.curriculum = 32, 2026, {}
    cfg.events.pop("push_robot", None)
    for group in cfg.observations.values():
        group.enable_corruption = False
    cfg.events["init_motion_loader"].params.update(delay_reset_env_ratio=1., max_delay_steps=250)
    command = cfg.commands["twist"]
    command.rel_standing_envs = command.rel_heading_envs = 0.
    command.heading_command, command.ranges.heading = False, None
    command.resampling_time_range = (1e9, 1e9)
    command.ranges.lin_vel_x, command.ranges.lin_vel_y, command.ranges.ang_vel_z = (.5, .5), (0., 0.), (0., 0.)
    seed_all(2026)
    env = CaptureEnv(cfg=cfg, device=args.device)
    recorder = None
    try:
        env.adapter = StateAdapter(env)
        candidates = keep_fallen_recovery_frames(env)
        seed_all(2026)
        env.reset(seed=2026)
        initial = {name: value.detach().cpu().numpy().copy() for name, value in {
            "state": env.adapter.state(), "context": env.adapter.context(), "command": env.adapter.command(),
            "actor_history": env.obs_buf["actor"], "previous_action": env.action_manager.action}.items()}
        initial_sha = content_hash(initial)
        model = physical_model_arrays(env)
        model_sha = content_hash(model)
        if initial_sha != protocol["initial_state_sha256"] or model_sha != protocol["physical_model_sha256"]:
            raise RuntimeError(f"Reference pairing mismatch: initial={initial_sha}, model={model_sha}")
        np.savez_compressed(args.output/"initial_state.npz", **initial)
        np.savez_compressed(args.output/"physical_model.npz", **model)
        snapshot = PhysicsSnapshot(env)
        recorder = RewardTermsRecorder(env)
        records, summaries, checks = {}, {}, []
        for trial, policy in (("a", "a"), ("a_repeat", "a"), ("b", "b"), ("b_repeat", "b")):
            snapshot.restore()
            current = {name: value.detach().cpu().numpy().copy() for name, value in {
                "state": env.adapter.state(), "context": env.adapter.context(), "command": env.adapter.command(),
                "actor_history": env.obs_buf["actor"], "previous_action": env.action_manager.action}.items()}
            if content_hash(current) != initial_sha or content_hash(physical_model_arrays(env)) != model_sha:
                raise RuntimeError("AA/BB snapshot restore changed the physical initial condition")
            arrays, summary = rollout(env, actors[policy], recorder, 250)
            if content_hash(physical_model_arrays(env)) != model_sha:
                raise RuntimeError("Physical randomized parameters changed during rollout")
            np.savez_compressed(args.output/f"reward_terms_{trial}.npz", **arrays, term_names=np.asarray(recorder.names))
            records[trial], summaries[trial] = arrays, summary
            checks.append({"trial": trial, "initial_state_sha256": initial_sha, "physical_model_sha256": model_sha})
        times = np.arange(250)*env.step_dt
        windows = {"all_0_to_5": np.ones(250, dtype=bool),
            "startup_0_to_0p1": times < .1, "fixed_0p1_to_2": (times >= .1)&(times < 2.),
            "fixed_2_to_5": (times >= 2.)&(times < 5.)}
        common = records["a"]["valid"] & records["b"]["valid"]
        fixed = {}
        for name, window in windows.items():
            mask = common & window[:, None]
            item = {trial: summarize(arrays, mask & arrays["valid"], recorder.names) for trial, arrays in records.items()}
            item["b_minus_a"] = {key: item["b"][key]["mean_return_per_environment"]-item["a"][key]["mean_return_per_environment"]
                for key in ("task_v2", "task_v1_counterfactual", "height_v1", "height_v2", "v2_minus_v1_same_trajectory")}
            fixed[name] = item
        for trial, arrays in records.items():
            summaries[trial]["own_active_delay_secondary"] = summarize(arrays, arrays["valid"] & arrays["active_delay"], recorder.names)
        for label, info in provenance.items():
            if file_hash(info["path"]) != info["sha256"]:
                raise RuntimeError("A source checkpoint file changed")
        report = {"schema": "amp-wmac-paired-reward-counterfactual-v1", "checkpoints": provenance,
            "reference_metrics": str(args.reference_metrics), "reference_metrics_sha256": file_hash(args.reference_metrics),
            "analysis_script_sha256": file_hash(Path(__file__)),
            "protocol": {**protocol, "real_reward": "upright_v2", "counterfactual_height": "height_only_v1",
                "counterfactual_formula": "task_v1=real_task_v2-height_term_v2+height_term_v1",
                "term_units": "reward_manager._step_reward * control_dt (weight and dt included exactly once)",
                "capture_timing": "after physical reward/metrics and before reset, on the same cached body buffers",
                "network_parameters_updated": False, "world_model_constructed": False, "substep_recorder": False},
            "height_reward_parameters": {key: recorder.term.params[key] for key in
                ("std", "mask_delay", "delay_env_rew_ratio", "upright_power")},
            "height_reward_weight": float(recorder.term.weight), "reward_term_names": recorder.names,
            "snapshot_restore_checks": checks, "trials": summaries, "common_first_episode_windows": fixed,
            "limitations": ["Counterfactual rewards use identical executed trajectories; this is not retraining under v1.",
                "Term sums are task rewards, without AMP; original task/style mixture is not silently applied.",
                "Active-delay summaries have policy-dependent duration; use fixed windows for primary comparisons.",
                "GPU physical rollouts may differ slightly despite exact starting-state/model hashes.",
                "A reward preference on these trajectories is not proof of the historical cause of policy differences."]}
        (args.output/"metrics.json").write_text(json.dumps(clean(report), ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    finally:
        if recorder is not None:
            recorder.close()
        env.close()


def self_test():
    import torch
    from types import SimpleNamespace
    from src.tasks.amp_loco.mdp.rewards import track_root_height
    angle = torch.tensor([torch.pi/2, torch.pi/3, 0.])
    quaternion = torch.stack((torch.cos(angle/2), torch.sin(angle/2), torch.zeros(3), torch.zeros(3)), -1)[:, None]
    position = torch.zeros(3, 1, 3)
    position[:, 0, 2] = torch.tensor([.4, .7, .7])
    data = SimpleNamespace(default_root_state=torch.tensor([[0., 0., .76]]).repeat(3, 1),
                           body_link_pos_w=position, body_link_quat_w=quaternion)
    params = {"std": .3, "mask_delay": True, "delay_env_rew_ratio": 3.5, "upright_power": 2.}
    height = SimpleNamespace(func=track_root_height, params=params, weight=1.)
    env = SimpleNamespace(scene={"robot": SimpleNamespace(data=data)}, step_dt=.02,
        cfg=SimpleNamespace(scale_rewards_by_dt=True), metrics_manager=SimpleNamespace(compute=lambda: "original"),
        termination_manager=SimpleNamespace(_delay_env_mask=torch.ones(3, dtype=torch.bool),
                                             _delay_counters=torch.tensor([1, 2, 0])))
    rate = track_root_height(env, **params)
    env.reward_manager = SimpleNamespace(active_terms=["track_root_height", "other"],
        get_term_cfg=lambda name: height, _step_reward=torch.stack((rate, torch.full_like(rate, -.4)), -1))
    env.reward_buf = env.reward_manager._step_reward.sum(-1)*env.step_dt
    before = env.reward_buf.clone()
    recorder = RewardTermsRecorder(env)
    recorder.enabled = True
    assert env.metrics_manager.compute() == "original"
    row = recorder.latest
    assert row["active_delay"].tolist() == [True, True, False]
    assert row["height_v1"][2] == row["height_v2"][2] == 0
    assert bool((row["height_v1"] >= row["height_v2"]-1e-7).all())
    torch.testing.assert_close(row["task_v1_counterfactual"], before-row["height_v2"]+row["height_v1"])
    torch.testing.assert_close(env.reward_buf, before, rtol=0, atol=0)
    assert params["upright_power"] == 2.
    recorder.close()
    print("CPU self-test passed: exact term rates, original active mask, v1 counterfactual, unchanged real reward/config.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-metrics", type=Path, default=ROOT/"logs/ppo60000_vs_wmac40800_recovery_seed2026/metrics.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test(); return
    if args.output is None:
        parser.error("--output is required outside --self-test")
    args.reference_metrics = args.reference_metrics.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    run(args)
    print(f"Reward counterfactual report: {args.output/'metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
