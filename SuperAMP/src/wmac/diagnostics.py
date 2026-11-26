"""Read-only WMAC sensitivity probes on fixed recorded physical data.

These probes measure model error and changes to the Actor's *proposed* gradient.
They do not establish that a component caused a historical policy regression.
No optimizer is stepped, and state (including RNGs and normalizer buffers) is
restored even if a probe fails. Inputs are never modified.
"""
from __future__ import annotations

from contextlib import contextmanager
from itertools import combinations
import math
import random
from typing import Iterable, Iterator

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.amp_features import amp_feature_metadata

from .networks import frozen_parameters
from .state import joint_count, joint_positions, joint_velocities, quat_inv, quat_mul, quat_normalize


def _recovery_proxy_protocol(runner) -> dict:
    """Label the configured training proxy, including its gradient limitation."""
    semantics = getattr(runner.cfg, "recovery_counter_semantics", "soft_v1")
    height_reward = getattr(runner.cfg, "recovery_height_reward", "height_only_v1")
    if semantics == "hard_forward_v2":
        counter = "hard delayed-recovery count: +1 on bad frames, clear on recovery or deadline; discrete clock has no state gradient"
        survival = "delayed termination has a hard forward value and an explicitly biased straight-through smooth backward surrogate"
    else:
        counter = "legacy soft counter discounts the previous count by the soft bad-pose score"
        survival = "legacy continuous soft bad-pose and sigmoid deadline proxy; no hard-forward straight-through replacement"
    return {
        "actor_type": getattr(runner.cfg, "actor_type", "mlp"),
        "router_auxiliary_loss_included": False,
        "actor_objective_scope": "Task/style/bootstrap objective only; separate real-start MoE balancing and router-z regularizers are excluded from this gradient decomposition.",
        "recovery_counter_semantics": semantics,
        "recovery_height_reward": height_reward,
        "training_height_reward_definition": (
            "original delayed-recovery height reward multiplied by clamp(root up_z,0,1)^2"
            if height_reward == "upright_v2" else "original delayed-recovery height-only reward"),
        "recovery_counter_forward": counter,
        "training_survival_gradient": survival,
        "surrogate_gradient_caveat": "Configured training-proxy derivatives are not verified physical simulator derivatives; this probe does not establish causal regression attribution.",
    }


def _number(value: torch.Tensor | float) -> float | None:
    result = float(value)
    return result if math.isfinite(result) else None


def safe_cosine(first: torch.Tensor, second: torch.Tensor) -> float | None:
    """Cosine similarity; a zero or non-finite gradient has no direction."""
    if first.numel() != second.numel():
        raise ValueError("Gradient vectors must have the same number of elements")
    a, b = first.detach().reshape(-1).double(), second.detach().reshape(-1).double()
    if a.device != b.device:
        b = b.to(a.device)
    if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
        return None
    denominator = a.norm() * b.norm()
    if not bool(denominator > 0):
        return None
    return float((torch.dot(a, b) / denominator).clamp(-1, 1))


def parameter_gradient_vector(
    objective: torch.Tensor, parameters: Iterable[nn.Parameter],
) -> torch.Tensor:
    """Return the gradient of a scalar objective without writing ``.grad``.

    Frozen and unused parameters retain zero slots, so vectors from the same
    architecture have the same indexing. Returned tensors have no autograd graph.
    """
    params = tuple(parameters)
    if not params:
        return torch.empty(0, device=objective.device, dtype=objective.dtype)
    active = tuple(parameter for parameter in params if parameter.requires_grad)
    gradients = (
        torch.autograd.grad(objective, active, allow_unused=True)
        if active and objective.requires_grad else (None,) * len(active)
    )
    lookup = {id(parameter): gradient for parameter, gradient in zip(active, gradients)}
    return torch.cat([
        (lookup[id(parameter)].detach() if lookup.get(id(parameter)) is not None
         else torch.zeros_like(parameter)).reshape(-1)
        for parameter in params
    ])


@contextmanager
def _preserve_state(runner) -> Iterator[None]:
    """Freeze observable probe side effects while leaving input graphs usable."""
    roots = [getattr(runner, name) for name in
             ("actor", "world_model", "discriminator", "critic", "amp_norm")]
    modules = {id(module): module for root in roots for module in root.modules()}
    parameters = {id(parameter): parameter for root in roots for parameter in root.parameters()}
    modes = [(module, module.training) for module in modules.values()]
    flags = [(parameter, parameter.requires_grad) for parameter in parameters.values()]
    buffers = [(buffer, buffer.detach().clone()) for module in modules.values()
               for buffer in module.buffers(recurse=False)]
    grads = [(parameter, parameter.grad,
              parameter.grad.detach().clone() if parameter.grad is not None else None)
             for parameter in parameters.values()]
    py_rng, numpy_rng, cpu_rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        # Actual WMAC MLPs have no dropout or BatchNorm, but eval also makes the
        # helper safe for diagnostic wrappers that contain such layers.
        for root in roots:
            root.eval()
        yield
    finally:
        with torch.no_grad():
            for buffer, saved in buffers:
                buffer.copy_(saved)
            for parameter, original_grad, saved_grad in grads:
                parameter.grad = original_grad
                if original_grad is not None:
                    original_grad.copy_(saved_grad)
        for parameter, flag in flags:
            parameter.requires_grad_(flag)
        for module, mode in modes:
            module.training = mode
        random.setstate(py_rng)
        np.random.set_state(numpy_rng)
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)


@contextmanager
def _objective_terms(cfg, task_weight: float, style_weight: float,
                     bootstrap: bool) -> Iterator[None]:
    saved = cfg.task_weight, cfg.style_weight, cfg.critic_bootstrap
    saved_recovery = getattr(cfg, "recovery_style_weight", None)
    cfg.task_weight, cfg.style_weight, cfg.critic_bootstrap = task_weight, style_weight, bootstrap
    if saved_recovery is not None and style_weight != saved[1]:
        # Components are isolated by zeroing weights; the recovery style weight goes with style.
        if style_weight != 0.:
            raise ValueError("Objective components rescale style only to zero")
        cfg.recovery_style_weight = 0.
    try:
        yield
    finally:
        cfg.task_weight, cfg.style_weight, cfg.critic_bootstrap = saved
        if saved_recovery is not None:
            cfg.recovery_style_weight = saved_recovery


def objective_gradient_probe(runner, batch: dict[str, torch.Tensor],
                             noises: torch.Tensor | None = None,
                             return_gradients: bool = False) -> dict:
    """Decompose the exact current Actor objective on paired starts/noise.

    Task, style and terminal-value terms retain their original configured
    weights, direct-style detachment, and differentiable survival factors. Their
    gradients must sum to the total gradient, including when style input
    gradients or survival gradients have been disabled. Cosines expose agreement
    or conflict; they are not evidence of which update caused past forgetting.

    ``return_gradients`` adds detached CPU tensors under ``_gradients`` for
    comparing component-swap experiments. Remove that key before JSON encoding.
    """
    if not len(batch.get("state", ())):
        raise ValueError("A gradient probe requires at least one replay start")
    # Detach to ensure the diagnostic never differentiates a caller-owned graph.
    values = {key: value.detach() for key, value in batch.items()}
    cfg = runner.cfg
    task, style, bootstrap = cfg.task_weight, cfg.style_weight, cfg.critic_bootstrap
    settings = {
        "total": (task, style, bootstrap),
        "task": (task, 0., False),
        "style": (0., style, False),
        "bootstrap": (0., 0., bootstrap),
    }
    result = {
        "batch_size": len(values["state"]), "horizon": cfg.imagination_horizon,
        "protocol": _recovery_proxy_protocol(runner),
        "gradient_direction": "ascent of mean imagined objective; before clipping/Adam",
        "components": {}, "cosines": {}, "decomposition": {},
    }
    gradients, objectives = {}, {}
    with _preserve_state(runner), torch.enable_grad(), frozen_parameters(
        runner.world_model, runner.discriminator, runner.critic,
    ):
        paired_noise = (runner._observation_noise(len(values["state"])) if noises is None
                        else noises.detach())
        expected = (cfg.imagination_horizon, len(values["state"]), 9 + 3 * joint_count(values["state"]))
        if tuple(paired_noise.shape) != expected:
            raise ValueError(f"Expected paired observation noise with shape {expected}")
        for name, weights in settings.items():
            with _objective_terms(cfg, *weights):
                rollout = runner.imagine(values, noises=paired_noise)
                objective = rollout["objective"].mean()
                gradient = parameter_gradient_vector(objective, runner.actor.parameters())
                objectives[name] = float(objective.detach())
                gradients[name] = gradient.detach().cpu()
                finite_count = int((~torch.isfinite(gradient)).sum())
                result["components"][name] = {
                    "objective_mean": _number(objective.detach()),
                    "gradient_norm": _number(gradient.double().norm()),
                    "gradient_rms": _number(gradient.double().square().mean().sqrt()),
                    "nonfinite_gradient_count": finite_count,
                }
                del rollout, objective, gradient
        for first, second in combinations(("task", "style", "bootstrap"), 2):
            result["cosines"][f"{first}_{second}"] = safe_cosine(gradients[first], gradients[second])
        for name in ("task", "style", "bootstrap"):
            result["cosines"][f"total_{name}"] = safe_cosine(gradients["total"], gradients[name])
        component_sum = sum(gradients[name] for name in ("task", "style", "bootstrap"))
        residual = (gradients["total"].double() - component_sum.double()).norm()
        scale = max(float(gradients["total"].double().norm()),
                    sum(float(gradients[name].double().norm()) for name in ("task", "style", "bootstrap")),
                    1e-12)
        result["decomposition"] = {
            "objective_absolute_residual": _number(abs(objectives["total"] - sum(
                objectives[name] for name in ("task", "style", "bootstrap")))),
            "gradient_l2_residual": _number(residual),
            "gradient_relative_residual": _number(residual / scale),
            "relative_scale": "max(total gradient norm, sum of component norms, 1e-12)",
        }
    if return_gradients:
        result["_gradients"] = gradients
    return result


def _masked_summary(values: torch.Tensor, mask: torch.Tensor) -> dict:
    selected = values[mask]
    finite = selected[torch.isfinite(selected)]
    return {
        "valid_count": int(mask.sum()),
        "nonfinite_count": int(selected.numel() - finite.numel()),
        "mean": _number(finite.mean()) if finite.numel() else None,
        "max": _number(finite.max()) if finite.numel() else None,
    }


def _raw_state_errors(predicted: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
    """Physical-unit errors independent of the model's learned loss scales."""
    predicted_q, target_q = predicted[..., 3:7], target[..., 3:7]
    relative_q = quat_mul(quat_inv(quat_normalize(target_q)), quat_normalize(predicted_q))
    angle = 2 * torch.atan2(relative_q[..., 1:].norm(dim=-1), relative_q[..., 0].abs())
    valid_quaternion = (predicted_q.norm(dim=-1) > 0) & (target_q.norm(dim=-1) > 0)
    angle = torch.where(valid_quaternion, angle, torch.full_like(angle, float("nan")))
    return {
        "root_position_l2_m": (predicted[..., :3] - target[..., :3]).norm(dim=-1),
        "root_orientation_geodesic_rad": angle,
        "root_linear_velocity_l2": (predicted[..., 7:10] - target[..., 7:10]).norm(dim=-1),
        "root_angular_velocity_l2": (predicted[..., 10:13] - target[..., 10:13]).norm(dim=-1),
        "joint_position_rmse_rad": (joint_positions(predicted) - joint_positions(target)).square().mean(-1).sqrt(),
        "joint_velocity_rmse_rad_s": (joint_velocities(predicted) - joint_velocities(target)).square().mean(-1).sqrt(),
    }


def _loss_normalization(world_model) -> dict:
    norm = getattr(world_model, "out_norm", None)
    floor = getattr(world_model, "error_floor", None)
    if norm is None or floor is None:
        return {"available": False, "note": "The supplied model does not expose WMAC loss-normalization buffers."}

    def vector(values):
        return [_number(value) for value in values.detach().cpu().reshape(-1)]

    return {
        "available": True,
        "formula": "mean((WorldModel.targets(target_state,predicted_state) / maximum(out_norm.std(),error_floor)) ** 2)",
        "loss_scale": vector(torch.maximum(norm.std(), floor)),
        "error_floor": vector(floor),
        "output_variance": vector(norm.var),
        "output_count": _number(norm.count),
        "output_epsilon": _number(norm.eps),
        "output_variance_epsilon": bool(norm.variance_epsilon),
        "comparability": "Normalized error can change when these scales change; compare raw_state_errors across models.",
    }


def _paired_scalar_summary(predicted: torch.Tensor, recorded: torch.Tensor) -> dict:
    valid = torch.ones_like(predicted, dtype=torch.bool)
    gap = predicted - recorded
    return {
        "predicted": _masked_summary(predicted, valid),
        "recorded": _masked_summary(recorded, valid),
        "signed_gap": _masked_summary(gap, valid),
        "absolute_gap": _masked_summary(gap.abs(), valid),
    }


def _grounded_feature_errors(schema: str, predicted: torch.Tensor,
                             recorded: torch.Tensor) -> dict:
    """Raw extra-feature errors; sole-reference cues are not contact truth."""
    if schema != "grounded_v2":
        return {"available": False, "reason": "The AMP schema has no grounded extra features."}
    metadata = amp_feature_metadata(schema)
    if predicted.shape[-1] != metadata["feature_dim"] or recorded.shape[-1] != metadata["feature_dim"]:
        raise ValueError("grounded_v2 diagnostics require the shared 203D G1 AMP layout")
    fields = {field["name"]: field for field in metadata["fields"]}
    layout = (
        ("pelvis_projected_gravity", ("x", "y", "z"), "dimensionless"),
        ("pelvis_ground_relative_height", ("pelvis",), "m"),
        ("foot_reference_ground_relative_height", ("left", "right"), "m"),
        ("foot_reference_world_vertical_velocity", ("left", "right"), "m/s"),
    )
    errors = {}
    for name, labels, unit in layout:
        field = fields[name]
        entries = {}
        for index, label in enumerate(labels, start=field["start"]):
            gap = predicted[..., index] - recorded[..., index]
            valid = torch.ones_like(gap, dtype=torch.bool)
            entries[label] = {"signed_gap": _masked_summary(gap, valid),
                              "absolute_gap": _masked_summary(gap.abs(), valid)}
        errors[name] = {"unit": unit, "components": entries}
    return {
        "available": True,
        "support_cues": "Kinematic sole-reference height/vertical velocity; not measured contact or contact force.",
        "fields": errors,
    }


def _amp_transition_probe(runner, predicted_prior: torch.Tensor, predicted_next: torch.Tensor,
                          recorded_prior: torch.Tensor, recorded_next: torch.Tensor,
                          valid: torch.Tensor) -> dict:
    """Score paired predicted/physical transitions under the exact same D/norm."""
    schema = getattr(runner.adapter, "amp_feature_schema",
                     getattr(runner.cfg, "amp_feature_schema", "unspecified"))
    count = int(valid.sum())
    feature_dim = getattr(runner.adapter, "amp_dim", None)
    if count:
        features = runner.adapter.amp_features(torch.stack((
            predicted_prior[valid], predicted_next[valid],
            recorded_prior[valid], recorded_next[valid],
        )))
        if features.ndim != 3 or features.shape[:2] != (4, count):
            raise ValueError("Expected AMP frame features with shape [4, valid_batch, feature_dim]")
        feature_dim = features.shape[-1]
        predicted_features, recorded_features = features[:2], features[2:]
        normalized = runner.amp_norm(features)
        raw_error = predicted_features - recorded_features
        normalized_error = normalized[:2] - normalized[2:]
        predicted_style, predicted_score = runner._amp_reward(features[0], features[1])
        recorded_style, recorded_score = runner._amp_reward(features[2], features[3])
    else:
        # No valid prefix remains at this horizon. Avoid FK/D calls on empty
        # tensors or invalid padding, while keeping the summaries explicit.
        width = feature_dim or (203 if schema == "grounded_v2" else 195 if schema == "legacy" else 0)
        predicted_features = predicted_prior.new_empty((2, 0, width))
        recorded_features = predicted_features.clone()
        raw_error = normalized_error = predicted_features
        predicted_style = predicted_score = recorded_style = recorded_score = predicted_prior.new_empty(0)
    row_mask = torch.ones(count, dtype=torch.bool, device=predicted_prior.device)
    feature_errors = {}
    for label, values in (("raw", raw_error), ("normalized", normalized_error)):
        for frame, index in (("prior", 0), ("next", 1)):
            feature_errors[f"{frame}_{label}_rmse"] = _masked_summary(
                values[index].square().mean(-1).sqrt(), row_mask)
        feature_errors[f"transition_{label}_rmse"] = _masked_summary(
            values.square().mean((0, 2)).sqrt(), row_mask)
    return {
        "feature_schema": schema, "feature_dim": feature_dim, "valid_count": count,
        "pair_definition": "predicted prior/next versus recorded state/next_state at this horizon",
        "scoring": "One fixed discriminator and the same frozen AMP normalizer for both pairs; gap = predicted - recorded.",
        "feature_errors": feature_errors,
        "discriminator_raw_score": _paired_scalar_summary(predicted_score, recorded_score),
        "style_reward": _paired_scalar_summary(predicted_style, recorded_style),
        "grounded_feature_errors": {
            frame: _grounded_feature_errors(schema, predicted_features[index], recorded_features[index])
            for frame, index in (("prior", 0), ("next", 1))
        },
    }


def open_loop_probe(runner, windows: dict[str, torch.Tensor],
                    horizons: tuple[int, ...] = (1, 4, 8, 16, 32)) -> dict:
    """Compare model predictions and one-step TD consistency on physical data.

    Input is ``ReplayBuffer.sample_windows`` layout [batch,time,...], including
    its boundary-aware ``mask``. State predictions are chained, using recorded
    physical actions and recorded context at each step, exactly as in model
    fitting. This is not a closed-loop evaluation of the Actor or of predicted
    context evolution. Errors at horizon h use only windows valid through h.

    The Critic diagnostic is a one-step TD residual on *physical* successors,
    with current D rewards, not a ground-truth value-calibration measurement.
    True terminals disable bootstrap; timeouts alone retain bootstrap.

    At each selected horizon, ``amp_transition`` compares the chained predicted
    transition with its recorded physical transition under one frozen D and AMP
    normalizer. The horizon-1 prior is identical by construction. Raw/normalized
    AMP feature errors, signed D/style gaps and grounded_v2's eight raw extra
    dimensions expose style-score optimism independently of discriminator drift.
    """
    required = {"state", "next_state", "action", "previous_action", "context",
                "next_context", "command", "contact_cost", "reward", "terminated", "mask"}
    missing = required - windows.keys()
    if missing:
        raise ValueError(f"Physical diagnostic windows are missing fields: {sorted(missing)}")
    state = windows["state"]
    # 13 root values plus joint positions and velocities: 71 for the G1, 59 for the PM01.
    if state.ndim != 3 or state.shape[-1] < 15 or state.shape[-1] % 2 == 0 or not state.shape[0]:
        raise ValueError("Expected a nonempty [batch,time,71] state tensor")
    batch, length = state.shape[:2]
    selected_horizons = tuple(sorted(set(horizons)))
    if not selected_horizons or any(not isinstance(h, int) or h < 1 or h > length for h in selected_horizons):
        raise ValueError(f"Horizons must be positive integers no greater than window length {length}")
    for name in required:
        if windows[name].shape[:2] != (batch, length):
            raise ValueError(f"Window field {name!r} has inconsistent batch/time dimensions")
    raw_mask = windows["mask"].bool()
    if raw_mask.ndim != 2:
        raise ValueError("Window mask must have shape [batch,time]")
    mask = raw_mask.long().cumprod(1).bool()
    result = {
        "protocol": {
            **_recovery_proxy_protocol(runner),
            "actions": "recorded physical actions",
            "context": "recorded physical context at each step",
            "state": "chained model prediction from the recorded start",
            "mask": "valid contiguous prefix; no cross-boundary successors",
            "nonfinite": "reported separately; mean/max summarize finite selected rows only",
            "critic": "one-step physical TD residual under the current discriminator; not causal proof or a Monte Carlo calibration error",
            "bootstrap": "true terminals disabled; timeout-only transitions bootstrap",
            "raw_state_error_units": {
                "root_position_l2_m": "m", "root_orientation_geodesic_rad": "rad",
                "root_linear_velocity_l2": "m/s", "root_angular_velocity_l2": "rad/s",
                "joint_position_rmse_rad": "rad", "joint_velocity_rmse_rad_s": "rad/s",
            },
        },
        "batch_size": batch, "window_length": length, "horizons": {},
    }
    with _preserve_state(runner), torch.no_grad():
        result["world_model_loss_normalization"] = _loss_normalization(runner.world_model)
        predicted = state[:, 0].detach()
        for step in range(max(selected_horizons)):
            prior = predicted
            predicted, contact = runner.world_model.step_with_contact(
                prior, windows["action"][:, step], windows["context"][:, step])
            horizon = step + 1
            if horizon not in selected_horizons:
                continue
            next_context, terminated = runner.adapter.advance_context(predicted, windows["context"][:, step])
            reward = runner.adapter.task_reward(prior, predicted, windows["action"][:, step],
                windows["previous_action"][:, step], windows["command"][:, step],
                next_context, terminated, contact)
            valid = mask[:, step]
            contact_error = F.smooth_l1_loss(torch.log1p(contact),
                torch.log1p(windows["contact_cost"][:, step].clamp_min(0)),
                reduction="none").mean(-1)
            result["horizons"][str(horizon)] = {
                "state_error": _masked_summary(runner.world_model.loss(
                    predicted, windows["next_state"][:, step]), valid),
                "contact_log_error": _masked_summary(contact_error, valid),
                "task_reward_absolute_gap": _masked_summary(
                    (reward - windows["reward"][:, step]).abs(), valid),
                "raw_state_errors": {name: _masked_summary(values, valid) for name, values in
                    _raw_state_errors(predicted, windows["next_state"][:, step]).items()},
                "amp_transition": _amp_transition_probe(runner, prior, predicted,
                    windows["state"][:, step], windows["next_state"][:, step], valid),
            }

        # Flatten only valid transitions before evaluating the Critic and D.
        # Invalid padding can contain arbitrary values, including NaN.
        real = {name: windows[name][mask] for name in required if name != "mask"}
        if real["state"].shape[0]:
            features = runner.adapter.amp_features(torch.stack((real["state"], real["next_state"])))
            style, score = runner._amp_reward(features[0], features[1])
            weighted = runner.cfg.style_weight * style
            if getattr(runner.cfg, "recovery_style_weight", None) is not None:
                from .recovery_style import weighted_style
                weighted = weighted_style(runner.cfg, style, real["state"], real["context"], runner._upright_height())
            reward = runner.cfg.task_weight * real["reward"] + weighted
            current = runner.critic(runner.adapter.critic_features(real["state"],
                real["previous_action"], real["command"], real["context"]))
            successor = runner.critic(runner.adapter.critic_features(real["next_state"],
                real["action"], real["command"], real["next_context"]))
            residual = reward + runner.cfg.gamma * (~real["terminated"].bool()).float() * successor - current
            valid_td = torch.ones_like(residual, dtype=torch.bool)
            zero_style_gradient = torch.where(torch.isfinite(score),
                ((score <= -1) | (score >= 3)).to(score.dtype),
                torch.full_like(score, float("nan")))
            result["physical_amp"] = {
                "style_reward": _masked_summary(style, valid_td),
                "raw_score": _masked_summary(score, valid_td),
                "zero_direct_gradient_fraction": _masked_summary(zero_style_gradient, valid_td),
            }
            result["physical_critic_td"] = {
                "signed_residual": _masked_summary(residual, valid_td),
                "absolute_residual": _masked_summary(residual.abs(), valid_td),
                "value": _masked_summary(current, valid_td),
                "target": _masked_summary(reward + runner.cfg.gamma * (
                    ~real["terminated"].bool()).float() * successor, valid_td),
            }
        else:
            empty = state.new_empty(0)
            no_valid = torch.empty(0, dtype=torch.bool, device=state.device)
            result["physical_critic_td"] = {
                key: _masked_summary(empty, no_valid)
                for key in ("signed_residual", "absolute_residual", "value", "target")
            }
            result["physical_amp"] = {
                key: _masked_summary(empty, no_valid)
                for key in ("style_reward", "raw_score", "zero_direct_gradient_fraction")
            }
    return result
