"""Inspect the aligned PPO/WMAC parameter contract without simulation or training.

The report constructs the real CPU policy/discriminator classes and PPO optimizer.
Matching rows describe parameter choices, not equivalent objectives or updates.
Exit status is 1 if any intended common parameter has drifted.
"""

from __future__ import annotations

import argparse
import contextlib
from copy import deepcopy
from dataclasses import asdict
import io
import inspect
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

PPO_TASK = "Unitree-G1-AMP-Flat-Aligned"


def _plain(value):
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    return value


def _optimizer_settings(optimizer):
    return [
        {key: _plain(group[key]) for key in ("lr", "betas", "eps", "weight_decay")}
        for group in optimizer.param_groups
    ]


def _activation_names(module):
    from torch import nn

    return sorted({type(layer).__name__.lower() for layer in module.modules()
                   if isinstance(layer, (nn.ELU, nn.ReLU, nn.SiLU, nn.Tanh, nn.LeakyReLU))})


def _parameter_count(module):
    return sum(parameter.numel() for parameter in module.parameters())


def build_comparison(num_envs=None, *, wmac_config=None, ppo_config=None):
    """Return the current factory configuration plus actual CPU network counts.

    Optional configurations allow audits/tests to detect drift without modifying
    the registered task. No checkpoints, simulator, CUDA context, or data loader
    are opened. Torch's CPU RNG is restored after network construction.
    """
    import torch

    import src.tasks  # noqa: F401; registers the actual environment/RL factories.
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
    from rsl_rl.algorithms.amp_ppo import AMPPPO
    from rsl_rl.modules import ActorCritic, EmpiricalNormalization
    from rsl_rl.modules.discriminator import Discriminator
    from rsl_rl.utils import Normalizer
    from src.amp_features import amp_feature_dim, amp_feature_metadata
    from src.wmac.config import WmacConfig
    from src.wmac.networks import Critic, RunningNorm
    from src.wmac.actors import actor_architecture, create_actor
    from src.wmac.state import CONTEXT_DIM, STATE_DIM, StateAdapter

    wm = deepcopy(wmac_config) if wmac_config is not None else WmacConfig.for_profile("ppo_aligned")
    ppo = deepcopy(ppo_config) if ppo_config is not None else load_rl_cfg(PPO_TASK)
    wm_env, ppo_env = load_env_cfg(wm.task), load_env_cfg(PPO_TASK)
    if num_envs is not None:
        if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs < 1:
            raise ValueError("num_envs must be a positive integer")
        wm.num_envs = ppo_env.scene.num_envs = num_envs
    wm.validate()

    # Probe the real shared feature functions, using a synthetic valid state.
    # The aligned PPO term delegates to this same adapter, including its context.
    adapter = StateAdapter.__new__(StateAdapter)
    adapter.imu_rotation = torch.tensor([1., 0., 0., 0.])
    adapter.default_joint_pos = adapter.default_joint_vel = torch.zeros(29)
    state = torch.zeros(1, STATE_DIM)
    state[:, 3] = 1
    previous, command, context = torch.zeros(1, 29), torch.zeros(1, 3), torch.zeros(1, CONTEXT_DIM)
    adapter.state = lambda: state
    adapter.command = lambda: command
    adapter.context = lambda: context
    frame_dim = adapter.actor_frame(state, previous, command, context).shape[-1]
    wm_critic_dim = adapter.critic_features(state, previous, command, context).shape[-1]
    synthetic_env = SimpleNamespace(_aligned_critic_state_adapter=adapter,
                                    action_manager=SimpleNamespace(action=previous))
    ppo_critic_group = ppo_env.observations["critic"]
    ppo_critic_dim = sum(term.func(synthetic_env, **term.params).shape[-1]
                         for term in ppo_critic_group.terms.values()) * max(1, ppo_critic_group.history_length)
    actor_term_dims = {"base_ang_vel": 3, "projected_gravity": 3, "command": 3,
                       "joint_pos": 29, "joint_vel": 29, "actions": 29}
    ppo_actor_group = ppo_env.observations["actor"]
    unknown_terms = set(ppo_actor_group.terms) - actor_term_dims.keys()
    if unknown_terms:
        raise ValueError(f"Cannot infer actor observation widths for terms: {sorted(unknown_terms)}")
    ppo_actor_dim = sum(actor_term_dims[name] for name in ppo_actor_group.terms) * max(
        1, ppo_actor_group.history_length)
    wm_actor_dim = frame_dim * 4  # WMAC's established four-frame actor history.
    wm_amp_dim = amp_feature_dim(wm.amp_feature_schema)
    ppo_amp_dim = amp_feature_dim(ppo.amp_feature_schema, len(ppo.amp_body_names))

    with torch.random.fork_rng(devices=[]), contextlib.redirect_stdout(io.StringIO()):
        actor = create_actor(wm, obs_dim=wm_actor_dim, action_dim=29)
        critic = Critic(wm_critic_dim, wm.critic_hidden)
        disc = Discriminator(2 * wm_amp_dim, wm.amp_reward_coef,
                             list(wm.discriminator_hidden), "cpu", wm.task_weight)
        policy = ActorCritic(ppo_actor_dim, ppo_critic_dim, 29,
                             actor_hidden_dims=list(ppo.actor.hidden_dims),
                             critic_hidden_dims=list(ppo.critic.hidden_dims),
                             activation=ppo.actor.activation,
                             init_noise_std=ppo.actor.distribution_cfg["init_std"],
                             noise_std_type=ppo.actor.distribution_cfg["std_type"])
        ppo_disc = Discriminator(2 * ppo_amp_dim, ppo.amp_reward_coef,
                                 ppo.amp_discr_hidden_dims, "cpu", ppo.amp_task_reward_lerp)
        ppo_norm = Normalizer(ppo_amp_dim)
        algorithm = asdict(ppo.algorithm)
        algorithm.pop("class_name", None)
        ppo_algorithm = AMPPPO(policy, ppo_disc, None, ppo_norm, amp_replay_buffer_size=1,
                               amp_normalizer_use_raw=ppo.amp_normalizer_use_raw,
                               amp_gp_normalized=ppo.amp_gp_normalized, **algorithm)
        wm_actor_opt = torch.optim.Adam(actor.parameters(), lr=wm.actor_lr, betas=wm.actor_betas)
        wm_critic_opt = torch.optim.Adam(critic.parameters(), lr=wm.critic_lr)
        disc_optimizer = torch.optim.Adam if wm.discriminator_optimizer == "adam" else torch.optim.AdamW
        wm_disc_opt = disc_optimizer([
            {"params": disc.trunk.parameters(), "weight_decay": wm.discriminator_trunk_weight_decay},
            {"params": disc.amp_linear.parameters(), "weight_decay": wm.discriminator_head_weight_decay},
        ], lr=wm.discriminator_lr)

    actor.norm.max_count = critic.norm.max_count = wm.observation_normalizer_until
    wm_norm = RunningNorm(wm_amp_dim, eps=wm.amp_normalizer_epsilon,
                          max_count=float("inf") if wm.amp_normalizer_until is None else wm.amp_normalizer_until)
    wm_norm.variance_epsilon, wm_norm.clip = wm.amp_normalizer_variance_epsilon, wm.amp_normalizer_clip
    wm_norm.count.fill_(wm.amp_normalizer_initial_count)
    # These are the live runner's EmpiricalNormalization constructor arguments.
    ppo_obs_norm = EmpiricalNormalization([ppo_actor_dim], until=1e8)
    ppo_obs_norm_settings = {"epsilon": ppo_obs_norm.eps, "variance_epsilon": False,
                             "clip": None, "until": ppo_obs_norm.until}
    wm_obs_norm_settings = {"epsilon": actor.norm.eps, "variance_epsilon": actor.norm.variance_epsilon,
                            "clip": actor.norm.clip, "until": actor.norm.max_count}
    wm_amp_norm_settings = {"epsilon": wm_norm.eps, "variance_epsilon": wm_norm.variance_epsilon,
                            "clip": wm_norm.clip, "initial_count": float(wm_norm.count),
                            "until": wm.amp_normalizer_until}
    ppo_amp_norm_settings = {"epsilon": ppo_norm.epsilon, "variance_epsilon": True,
                             "clip": ppo_norm.clip_obs, "initial_count": ppo_norm.count, "until": None}
    ppo_optimizer_groups = _optimizer_settings(ppo_algorithm.optimizer)
    ppo_batch = ppo_env.scene.num_envs * ppo.num_steps_per_env // ppo.algorithm.num_mini_batches
    ppo_updates = ppo.algorithm.num_learning_epochs * ppo.algorithm.num_mini_batches
    wm_collection_mean = float(wm.collect_steps) if wm.collect_steps else 1. / wm.utd
    wm_real_batch_mean = wm.num_envs * wm_collection_mean
    ppo_real_batch = ppo_env.scene.num_envs * ppo.num_steps_per_env
    wm_d_pairs = wm.effective_discriminator_batch_size * wm.discriminator_updates
    ppo_d_pairs = ppo_batch * ppo_updates
    method_budgets = {
        "wmac": {
            "collection_mode": "fixed_steps_override" if wm.collect_steps else "utd_accumulator",
            "configured_collect_steps": wm.collect_steps,
            "configured_utd": wm.utd,
            "effective_outer_updates_per_vectorized_collection_step": 1. / wm_collection_mean,
            "collection_steps_per_outer_iteration_mean": wm_collection_mean,
            "possible_collection_steps_per_outer_iteration": sorted({math.floor(wm_collection_mean), math.ceil(wm_collection_mean)}),
            "real_transitions_per_outer_iteration_mean": wm_real_batch_mean,
            "discriminator_batch_size_per_class": wm.effective_discriminator_batch_size,
            "discriminator_updates_per_outer_iteration": wm.discriminator_updates,
            "discriminator_policy_pairs_per_outer_iteration": wm_d_pairs,
            "discriminator_expert_pairs_per_outer_iteration": wm_d_pairs,
            "discriminator_pairs_per_real_transition_per_class": wm_d_pairs / wm_real_batch_mean,
            "maximum_outer_iterations": wm.iterations,
        },
        "ppo": {
            "collection_mode": "fixed_real_rollout",
            "configured_collect_steps": ppo.num_steps_per_env,
            "configured_utd": None,
            "effective_outer_updates_per_vectorized_collection_step": 1. / ppo.num_steps_per_env,
            "collection_steps_per_outer_iteration_mean": float(ppo.num_steps_per_env),
            "possible_collection_steps_per_outer_iteration": [ppo.num_steps_per_env],
            "real_transitions_per_outer_iteration_mean": ppo_real_batch,
            "discriminator_batch_size_per_class": ppo_batch,
            "discriminator_updates_per_outer_iteration": ppo_updates,
            "discriminator_policy_pairs_per_outer_iteration": ppo_d_pairs,
            "discriminator_expert_pairs_per_outer_iteration": ppo_d_pairs,
            "discriminator_pairs_per_real_transition_per_class": ppo_d_pairs / ppo_real_batch,
            "maximum_outer_iterations": ppo.max_iterations,
        },
    }
    rows = []

    def row(name, wm_value, ppo_value):
        first, second = _plain(wm_value), _plain(ppo_value)
        rows.append({"parameter": name, "wmac": first, "ppo": second, "matched": first == second})

    row("task", wm.task, PPO_TASK)
    row("num_envs", wm.num_envs, ppo_env.scene.num_envs)
    row("actor_observation_dim", wm_actor_dim, ppo_actor_dim)
    row("critic_observation_dim", wm_critic_dim, ppo_critic_dim)
    row("control_step_seconds", wm_env.decimation * wm_env.sim.mujoco.timestep,
        ppo_env.decimation * ppo_env.sim.mujoco.timestep)
    row("actor_hidden", actor_architecture(wm)["policy_hidden"], ppo.actor.hidden_dims)
    row("critic_hidden", wm.critic_hidden, ppo.critic.hidden_dims)
    row("discriminator_hidden", wm.discriminator_hidden, ppo.amp_discr_hidden_dims)
    expert_network = actor.experts[0] if wm.actor_type == "moe" else actor.net
    row("actor_activation", _activation_names(expert_network), _activation_names(policy.actor))
    row("critic_activation", _activation_names(critic.net), _activation_names(policy.critic))
    row("discriminator_activation", _activation_names(disc), _activation_names(ppo_disc))
    hiking = wm.actor_type == "moe" and wm.moe_variant == "hiking"
    row("actor_output_initialization", {"scale": 1. if hiking else wm.policy_output_init_scale,
                                        "zero_bias": False if hiking else wm.policy_output_zero_bias},
        {"scale": 1., "zero_bias": False})
    row("actor_optimizer", _optimizer_settings(wm_actor_opt)[0], ppo_optimizer_groups[0])
    row("critic_optimizer", _optimizer_settings(wm_critic_opt)[0], ppo_optimizer_groups[0])
    row("discriminator_optimizer_class", type(wm_disc_opt).__name__, type(ppo_algorithm.optimizer).__name__)
    row("discriminator_optimizer_groups", _optimizer_settings(wm_disc_opt), ppo_optimizer_groups[1:])
    row("task_weight", wm.task_weight, ppo.amp_task_reward_lerp)
    row("style_weight", wm.style_weight, 1 - ppo.amp_task_reward_lerp)
    row("amp_reward_coef", wm.amp_reward_coef, ppo.amp_reward_coef)
    row("gamma", wm.gamma, ppo_algorithm.gamma)
    row("lambda", wm.gae_lambda, ppo_algorithm.lam)
    row("critic_epochs", wm.critic_epochs, ppo.algorithm.num_learning_epochs)
    row("critic_minibatches", wm.critic_minibatches, ppo.algorithm.num_mini_batches)
    row("observation_normalizer", wm_obs_norm_settings, ppo_obs_norm_settings)
    row("actor_observation_normalization_enabled", getattr(actor, "normalize_observations", True), ppo.actor.obs_normalization)
    row("critic_observation_normalization_enabled", True, ppo.critic.obs_normalization)
    row("amp_feature_schema", wm.amp_feature_schema, ppo.amp_feature_schema)
    row("amp_frame_dim", wm_amp_dim, ppo_amp_dim)
    row("discriminator_input_dim", disc.input_dim, ppo_disc.input_dim)
    row("amp_normalizer", wm_amp_norm_settings, ppo_amp_norm_settings)
    row("amp_raw_moments_after_each_discriminator_update", wm.amp_normalizer_per_update,
        ppo_algorithm.amp_normalizer_use_raw)
    row("expert_recovery_fraction", wm.expert_recovery_fraction, ppo.amp_expert_recovery_fraction)
    row("gradient_penalty_coefficient", wm.gradient_penalty, 10.)
    row("gradient_penalty_uses_normalized_inputs", True, ppo_algorithm.amp_gp_normalized)
    row("initial_action_exploration_std", wm.exploration_noise, ppo.actor.distribution_cfg["init_std"])
    row("max_grad_norm_parameter", wm.max_grad_norm, ppo_algorithm.max_grad_norm)
    counts = {
        "actor": {"wmac": _parameter_count(actor), "ppo": _parameter_count(policy.actor)},
        "critic": {"wmac": _parameter_count(critic), "ppo": _parameter_count(policy.critic)},
        "discriminator": {"wmac": _parameter_count(disc), "ppo": _parameter_count(ppo_disc)},
    }
    for branch, count in counts.items():
        row(f"{branch}_network_parameters", count["wmac"], count["ppo"])
    counts["ppo_distribution_extra"] = _parameter_count(policy) - _parameter_count(policy.actor) - _parameter_count(policy.critic)
    if wm.actor_type == "moe":
        expert_counts = [_parameter_count(expert) for expert in actor.experts]
        router_count = _parameter_count(actor.router)
        active_count = router_count + sum(sorted(expert_counts, reverse=True)[:wm.moe_top_k])
        actor_budget = {"architecture": actor_architecture(wm), "total_parameters": counts["actor"]["wmac"],
            "router_parameters": router_count, "expert_parameters": expert_counts,
            "active_parameters_per_token": active_count,
            "expert_activations": [_activation_names(expert) for expert in actor.experts],
            "router_activations": _activation_names(actor.router),
            "execution": "Hiking dense Softmax, all experts evaluated" if hiking else "sparse selected-token dispatch; ONNX dense fallback is a separate execution mode"}
    else:
        actor_budget = {"architecture": actor_architecture(wm), "total_parameters": counts["actor"]["wmac"],
                        "active_parameters_per_token": counts["actor"]["wmac"], "execution": "dense MLP"}
    actor_budget["parameter_count_scope"] = "network weights and biases; normalization buffers excluded; active counts are not measured latency or FLOPs"

    differences = [
        {"parameter": "recovery_height_objective", "wmac": wm.recovery_height_reward,
         "ppo": "upright_v2" if ppo_env.rewards["track_root_height"].params.get("upright_power", 0.) == 2. else "height_only_v1",
         "reason": "New WMAC experiments explicitly repair the high-but-tilted height objective. The original PPO task default is preserved. A fair objective-controlled training comparison must also configure PPO upright_power=2, or explicitly select height_only_v1 for WMAC."},
        {"parameter": "imagined_recovery_counter", "wmac": wm.recovery_counter_semantics,
         "ppo": "physical DelayedTerminationManager",
         "reason": "WMAC hard_forward_v2 matches the pose-only delayed clock and termination forward values, with a biased straight-through hazard derivative. Timeouts are separately captured; non-delay termination remains a smooth proxy."},
        {"parameter": "collection_schedule", "wmac": {
            key: method_budgets["wmac"][key] for key in (
                "collection_mode", "configured_collect_steps", "configured_utd",
                "collection_steps_per_outer_iteration_mean", "possible_collection_steps_per_outer_iteration",
                "real_transitions_per_outer_iteration_mean")},
         "ppo": {key: method_budgets["ppo"][key] for key in (
                "collection_mode", "configured_collect_steps", "collection_steps_per_outer_iteration_mean",
                "possible_collection_steps_per_outer_iteration", "real_transitions_per_outer_iteration_mean")},
         "reason": "Each method retains its collection cadence; equal outer-iteration counts are not equal real interaction budgets."},
        {"parameter": "discriminator_training_budget", "wmac": {
            key: value for key, value in method_budgets["wmac"].items() if key.startswith("discriminator_")},
         "ppo": {key: value for key, value in method_budgets["ppo"].items() if key.startswith("discriminator_")},
         "reason": "D batch/update counts are independent method budgets. Counts separately include policy pairs and expert pairs; replayed pairs are processed samples, not new physical transitions."},
        {"parameter": "training_stopping_limits", "wmac": {"maximum_outer_iterations": wm.iterations},
         "ppo": {"maximum_outer_iterations": ppo.max_iterations},
         "reason": "Iteration limits are separate safety caps. Compare total time and real interactions to the same predeclared quality target, not equal iteration labels."},
        {"parameter": "actor_objective", "wmac": "deterministic BPTT through learned dynamics, task/style and bootstrap",
         "ppo": "Gaussian likelihood-ratio PPO on physical rollouts",
         "reason": "The requested training methods remain distinct; equal reward coefficients do not equalize actor gradients."},
        {"parameter": "action_exploration", "wmac": {"kind": "fixed additive Gaussian during collection", "std": wm.exploration_noise},
         "ppo": {"kind": "learned per-action Gaussian", "initial_std": ppo.actor.distribution_cfg["init_std"],
                 "entropy_coef": ppo.algorithm.entropy_coef, "minimum_std": list(ppo.min_normalized_std)},
         "reason": "Initial standard deviations match; subsequent exploration and entropy objectives differ."},
        {"parameter": "action_output", "wmac": {"soft_tanh_limit": wm.action_limit, "collection_clip": wm.action_limit},
         "ppo": {"clip_actions": ppo.clip_actions, "mean": "unbounded"},
         "reason": "WMAC retains its differentiable action bound and collection guard."},
        {"parameter": "learning_rate_schedule", "wmac": {"kind": "cosine", "iterations": wm.lr_decay_iterations,
                                                          "final_fraction": wm.lr_final_fraction},
         "ppo": {"kind": ppo.algorithm.schedule, "desired_kl": ppo.algorithm.desired_kl},
         "reason": "Only initial learning rates match; saved or later learning rates need not match."},
        {"parameter": "optimizer_and_gradient_clipping", "wmac": {"groups": "separate Actor/Critic/D optimizers",
                    "norm_clip": "independently clips each module including D", "actor_value_clip": wm.actor_grad_value_clip,
                    "max_action_shift": wm.max_action_shift},
         "ppo": {"groups": "one Adam for Actor/Critic/D", "norm_clip": "joint Actor/Critic; D not clipped",
                 "ratio_clip": ppo.algorithm.clip_param, "clipped_value_loss": ppo.algorithm.use_clipped_value_loss},
         "reason": "The common norm-limit number does not imply the same effective parameter update."},
        {"parameter": "actor_update_budget", "wmac": {"updates": wm.actor_updates, "start_states": wm.actor_batch_size,
                                                       "imagined_horizon": wm.imagination_horizon},
         "ppo": {"updates": ppo_updates, "minibatch_size": ppo_batch, "physical_rollout_steps": ppo.num_steps_per_env},
         "reason": "Imagined states and real rollout transitions are different data and computational budgets."},
        {"parameter": "critic_targets", "wmac": {"source": "imagined lambda returns", "real_td_auxiliary_coef": wm.real_critic_coef},
         "ppo": {"source": "physical-rollout GAE/returns", "value_loss_coef": ppo.algorithm.value_loss_coef},
         "reason": "Shared critic inputs and architecture do not equalize targets or bootstrap input gradients."},
        {"parameter": "replay_and_warmup", "wmac": {"replay_steps": wm.replay_steps, "recovery_fraction": wm.recovery_fraction,
                   "validation_env_fraction": wm.validation_env_fraction, "warmup_steps": wm.warmup_steps,
                   "model_warmup_iterations": wm.model_warmup_iterations, "world_model_lr": wm.world_model_lr},
         "ppo": {"actor_data": "fresh on-policy rollout", "amp_replay_capacity":
                 inspect.signature(AMPPPO).parameters["amp_replay_buffer_size"].default},
         "reason": "Replay ages, model training/gates, validation and evaluation costs remain different."},
    ]
    if wm.actor_type == "moe":
        differences.append({"parameter": "actor_architecture_and_capacity", "wmac": actor_budget,
            "ppo": {"actor_type": "mlp", "total_parameters": counts["actor"]["ppo"],
                    "active_parameters_per_token": counts["actor"]["ppo"]},
            "reason": "Matching one expert's widths does not match total or active Actor capacity. Sparse dispatch does not by itself establish a training or inference speedup."})
    return {
        "format": "amp-parameter-comparison-v1",
        "inspection": "current factories and CPU networks/optimizers; no simulator, checkpoints or training",
        "wmac_profile": wm.profile, "ppo_task": PPO_TASK,
        "all_common_parameters_match": all(item["matched"] for item in rows),
        "common_parameters": rows,
        "method_budgets": method_budgets,
        "comparison_protocol": {
            "primary_endpoint": "total_training_wall_time_and_real_interactions_to_the_same_predeclared_quality_target",
            "evaluation": "matched evaluation protocol and initial conditions; quality target must be specified before comparing checkpoints",
            "cost_accounting": "include collection, model/D/policy/critic training and warmup; report compilation, validation, save and evaluation overhead explicitly",
            "real_interactions": "count all physical transitions including held-out validation environments; report evaluation interactions separately; exclude imagined/replayed/expert pairs",
            "target_not_reached": "report the run as target-not-reached at its measured time/interaction budget rather than equating its last checkpoint to the target",
        },
        "network_parameters": counts,
        "actor_architecture_and_parameters": actor_budget,
        "amp_features": amp_feature_metadata(wm.amp_feature_schema),
        "remaining_differences": differences,
        "limits": [
            "Matching parameter settings are not a claim of equal objectives, gradients, learned weights or training outcomes.",
            "Collection, D training and iteration caps are independent method budgets and are not included in the common-parameter equality contract.",
            "UTD collection steps are long-run means; the accumulator may alternate integer counts. Warmup, gated updates, validation and evaluation still affect total cost.",
            "PPO checkpoint iteration labels are zero-based; WMAC labels count completed iterations. Compare completed updates rather than filename numbers alone.",
            "Normalization formulas/update sources match; different visited states and floating-point implementations produce different moments.",
            "Network counts exclude PPO's separately reported learned Gaussian standard-deviation parameters.",
            "This configuration-only inspection does not establish that either arm reached the chosen quality target or measure a speedup.",
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, help="Override both environment counts; preserve each method's own collection/D budget rules")
    parser.add_argument("--output", type=Path, help="Write JSON to a new file; existing files are never overwritten")
    parser.add_argument("--actor-type", choices=("mlp", "moe"))
    parser.add_argument("--moe-variant", choices=("hiking", "legacy"))
    parser.add_argument("--moe-expert-hidden", nargs="+", type=int)
    parser.add_argument("--policy-hidden", nargs="+", type=int)
    parser.add_argument("--moe-num-experts", type=int)
    parser.add_argument("--moe-top-k", type=int)
    parser.add_argument("--moe-gate-hidden", nargs="+", type=int)
    parser.add_argument("--moe-temperature", type=float)
    args = parser.parse_args(argv)
    if args.num_envs is not None and args.num_envs < 1:
        parser.error("--num-envs must be positive")
    if args.output is not None and args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    overrides = {name: getattr(args, name) for name in (
        "actor_type", "policy_hidden", "moe_variant", "moe_expert_hidden", "moe_num_experts", "moe_top_k", "moe_gate_hidden", "moe_temperature")
        if getattr(args, name) is not None}
    if overrides:
        from src.wmac.config import WmacConfig
        cfg = WmacConfig.for_profile("ppo_aligned")
        for name, value in overrides.items():
            setattr(cfg, name, tuple(value) if name.endswith("hidden") else value)
        try:
            cfg.validate()
        except ValueError as error:
            parser.error(str(error))
        report = build_comparison(args.num_envs, wmac_config=cfg)
    else:
        report = build_comparison(args.num_envs)
    encoded = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output is None:
        print(encoded, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        try:
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(encoded)
        except FileExistsError:
            parser.error(f"Output already exists: {args.output}")
        print(json.dumps({"output": str(args.output.resolve()),
                          "all_common_parameters_match": report["all_common_parameters_match"],
                          "common_parameter_count": len(report["common_parameters"]),
                          "remaining_difference_count": len(report["remaining_differences"])}, ensure_ascii=False))
    return 0 if report["all_common_parameters_match"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
