"""HybridRobotics SuperMimic's Muon/Adam partition on PyTorch 2.7.

Muon math is adapted from PyTorch v2.10.0 ``torch/optim/_muon.py`` (BSD-3,
Copyright (c) PyTorch contributors), SHA256
38e95918424bd26604f786195a4aa72afb60674b4ac9ccbde777974013cb1f52.
See THIRD_PARTY_NOTICES.md. The reference learner selects middle Linear
weights within each Sequential for Muon, and fused/capturable Adam otherwise.
This wrapper keeps both algorithms in one strictly identified state_dict.
"""

from __future__ import annotations

from collections.abc import Mapping
import inspect
import math

import torch
from torch import nn
from torch.optim import _functional

REFERENCE_OPTIMIZER_ALGORITHM = "hybrid_supermimic_muon_adam_v1"
PYTORCH_MUON_REFERENCE_SHA256 = "38e95918424bd26604f786195a4aa72afb60674b4ac9ccbde777974013cb1f52"
NS_COEFFICIENTS = (3.4445, -4.7750, 2.0315)
_ADAM_HAS_DECOUPLED_WEIGHT_DECAY = "decoupled_weight_decay" in inspect.signature(_functional.adam).parameters


def set_learning_rate(optimizer, value):
    """Keep tensor LR storage stable while applying the runner's schedule.

    Plain Adam/AdamW groups continue to use their original scalar convention.
    Reference Muon/Adam starts with device tensors, matching official captures.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError("Learning rate must be finite and nonnegative")
    with torch.no_grad():
        for group in optimizer.param_groups:
            if isinstance(group["lr"], torch.Tensor):
                group["lr"].fill_(value)
            else:
                group["lr"] = float(value)


def _zeropower_via_newtonschulz(grad, ns_coefficients=NS_COEFFICIENTS, ns_steps=5, eps=1e-7):
    """The v2.10 quintic Newton-Schulz implementation, including BF16 math."""
    if ns_steps >= 100:
        raise ValueError("Number of steps must be less than 100 for computational efficiency")
    if grad.ndim != 2 or len(ns_coefficients) != 3:
        raise ValueError("Muon requires a matrix and three Newton-Schulz coefficients")
    a, b, c = ns_coefficients
    ortho_grad = grad.bfloat16()
    if grad.size(0) > grad.size(1):
        ortho_grad = ortho_grad.T
    ortho_grad.div_(ortho_grad.norm().clamp(min=eps))
    for _ in range(ns_steps):
        gram_matrix = ortho_grad @ ortho_grad.T
        gram_update = torch.addmm(gram_matrix, gram_matrix, gram_matrix, beta=b, alpha=c)
        ortho_grad = torch.addmm(ortho_grad, gram_update, ortho_grad, beta=a)
    return ortho_grad.T if grad.size(0) > grad.size(1) else ortho_grad


def optimizer_algorithm_name(optimizer):
    if isinstance(optimizer, ReferenceMuonAdam):
        return REFERENCE_OPTIMIZER_ALGORITHM
    if isinstance(optimizer, torch.optim.AdamW):
        return "adamw"
    if isinstance(optimizer, torch.optim.Adam):
        return "adam"
    if isinstance(optimizer, torch.optim.SGD):
        return "sgd"
    raise ValueError("Unknown optimizer algorithm; an explicit checkpoint protocol is required")


class ReferenceMuonAdam(torch.optim.Optimizer):
    algorithm_name = REFERENCE_OPTIMIZER_ALGORITHM
    _supports_device_finite_guard = True

    def __init__(self, adam_params, muon_params, *, lr, betas=(.9, .999), device="cpu",
                 adam_names=None, muon_names=None):
        device = torch.device(device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        if isinstance(lr, bool) or not isinstance(lr, (int, float)) or not math.isfinite(lr) or lr < 0:
            raise ValueError("Reference optimizer lr must be finite and nonnegative")
        if (len(betas) != 2 or any(isinstance(beta, bool) or not isinstance(beta, (int, float))
                                 or not 0 <= beta < 1 for beta in betas)):
            raise ValueError("Adam betas must lie in [0,1)")
        adam_params, muon_params = list(adam_params), list(muon_params)
        parameters = [*adam_params, *muon_params]
        if not parameters or len({id(p) for p in parameters}) != len(parameters):
            raise ValueError("Reference optimizer requires unique nonempty parameters")
        if any(p.device != device or not p.is_floating_point() for p in parameters):
            raise ValueError("Reference optimizer parameters must share the requested floating-point device")
        if any(p.ndim != 2 for p in muon_params):
            raise ValueError("Muon only supports matrix parameters")
        groups = []
        if adam_params:
            groups.append(dict(params=adam_params, kind="adam", lr=torch.tensor(lr, device=device),
                param_names=list(adam_names or (f"adam.{i}" for i in range(len(adam_params)))),
                betas=tuple(betas), eps=1e-8, weight_decay=0., amsgrad=False, maximize=False,
                fused=device.type == "cuda", capturable=device.type == "cuda"))
        if muon_params:
            groups.append(dict(params=muon_params, kind="muon", lr=torch.tensor(lr, device=device),
                param_names=list(muon_names or (f"muon.{i}" for i in range(len(muon_params)))),
                momentum=.95, nesterov=True, ns_coefficients=NS_COEFFICIENTS,
                ns_steps=5, eps=1e-7, adjust_lr_fn="match_rms_adamw", weight_decay=.1))
        super().__init__(groups, defaults={})
        self._zero_found_inf = torch.zeros((), device=device)
        self.initialize_state()

    @torch.no_grad()
    def initialize_state(self):
        """Allocate fixed optimizer state before any CUDA graph captures it."""
        for group in self.param_groups:
            for parameter in group["params"]:
                state = self.state[parameter]
                if group["kind"] == "adam":
                    if "step" not in state:
                        state["step"] = torch.zeros((), dtype=torch.float32, device=parameter.device)
                        state["exp_avg"] = torch.zeros_like(parameter)
                        state["exp_avg_sq"] = torch.zeros_like(parameter)
                elif "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(parameter)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        if getattr(self, "grad_scale", None) is not None:
            raise ValueError("ReferenceMuonAdam accepts finite guards without gradient scaling")
        flag = getattr(self, "found_inf", None)
        flag = self._zero_found_inf if flag is None else flag
        if not isinstance(flag, torch.Tensor) or flag.ndim != 0 or flag.device != self._zero_found_inf.device:
            raise ValueError("found_inf must be a scalar on the optimizer device")
        finite = flag == 0
        # CPU Adam has no fused found_inf entry point. CUDA never reads the
        # finite scalar on the host, including during full-step capture.
        if self._zero_found_inf.device.type == "cpu" and not bool(finite):
            return loss
        found_inf = (~finite).to(torch.float32)
        for group in self.param_groups:
            pairs = [(parameter, parameter.grad) for parameter in group["params"] if parameter.grad is not None]
            if not pairs:
                continue
            if any(grad.is_sparse or grad.is_complex() for _, grad in pairs):
                raise RuntimeError("Reference Muon/Adam requires dense real gradients")
            if group["kind"] == "adam":
                params, grads = zip(*pairs)
                states = [self.state[p] for p in params]
                # PyTorch releases differ in this optional functional kwarg.
                # Omitting it is identical for our fixed zero-decay Adam path.
                extra = {"decoupled_weight_decay": False} if _ADAM_HAS_DECOUPLED_WEIGHT_DECAY else {}
                _functional.adam(list(params), list(grads), [s["exp_avg"] for s in states],
                    [s["exp_avg_sq"] for s in states], [], [s["step"] for s in states],
                    foreach=False, capturable=group["capturable"], differentiable=False,
                    fused=group["fused"], grad_scale=None,
                    found_inf=found_inf if group["fused"] else None, has_complex=False,
                    amsgrad=False, beta1=group["betas"][0],
                    beta2=group["betas"][1], lr=group["lr"], weight_decay=0., eps=group["eps"], maximize=False,
                    **extra)
            else:
                for parameter, grad in pairs:
                    buffer = self.state[parameter]["momentum_buffer"]
                    safe_grad = torch.where(finite, grad, torch.zeros_like(grad))
                    candidate = torch.lerp(buffer, safe_grad, 1-group["momentum"])
                    update = torch.lerp(safe_grad, candidate, group["momentum"]) if group["nesterov"] else candidate
                    update = _zeropower_via_newtonschulz(update, group["ns_coefficients"], group["ns_steps"], group["eps"])
                    learning_rate = torch.as_tensor(group["lr"], device=parameter.device, dtype=parameter.dtype)
                    adjusted_lr = learning_rate * (.2 * math.sqrt(max(parameter.shape)))
                    next_parameter = parameter * (1-learning_rate*group["weight_decay"])
                    next_parameter = next_parameter - update.to(parameter.dtype) * adjusted_lr
                    buffer.copy_(torch.where(finite, candidate, buffer))
                    parameter.copy_(torch.where(finite, next_parameter, parameter))
        return loss

    def state_dict(self):
        result = super().state_dict()
        # A third-party eager scheduler may have replaced a tensor LR by a
        # scalar. Serialize the same strict scalar-tensor schema in either case.
        result["param_groups"] = [dict(group) for group in result["param_groups"]]
        for group, runtime in zip(result["param_groups"], self.param_groups):
            group["lr"] = torch.as_tensor(group["lr"], device=runtime["params"][0].device,
                                           dtype=torch.float32)
        result["algorithm_name"] = self.algorithm_name
        return result

    def load_state_dict(self, state_dict):
        validate_reference_optimizer_state(self, state_dict)
        prepared = {"state": state_dict["state"], "param_groups": [dict(group) for group in state_dict["param_groups"]]}
        runtime = [(group["params"][0].device, group.get("fused"), group.get("capturable")) for group in self.param_groups]
        for saved, (_, fused, capturable) in zip(prepared["param_groups"], runtime):
            if saved["kind"] == "adam":
                saved.update(fused=fused, capturable=capturable)
        super().load_state_dict(prepared)
        for group, (device, _, _) in zip(self.param_groups, runtime):
            group["lr"] = group["lr"].to(device=device, dtype=torch.float32)


def build_reference_optimizer(modules, lr, betas=(.9, .999), device="cpu"):
    """Exact reference Sequential/Linear partition, with shared params deduped."""
    modules = list(modules)
    names, parameters = {}, []
    for index, module in enumerate(modules):
        for name, parameter in module.named_parameters():
            if id(parameter) not in names:
                names[id(parameter)] = f"module_{index}.{name}"
                parameters.append(parameter)
    hidden, seen = [], set()
    for module in modules:
        for seq in module.modules():
            if isinstance(seq, nn.Sequential):
                for linear in [value for value in seq if isinstance(value, nn.Linear)][1:-1]:
                    if id(linear.weight) not in seen:
                        seen.add(id(linear.weight))
                        hidden.append(linear.weight)
    rest = [parameter for parameter in parameters if id(parameter) not in seen]
    return ReferenceMuonAdam(rest, hidden, lr=lr, betas=betas, device=device,
        adam_names=[names[id(p)] for p in rest], muon_names=[names[id(p)] for p in hidden])


def validate_reference_optimizer_state(optimizer, saved, label="optimizer"):
    """Preflight canonical algorithm, named layout, moments and finite values.

    Only fused/capturable runtime booleans adapt to the destination device.
    Scalar LR values may differ after a recorded schedule; their schema is fixed.
    """
    if not isinstance(optimizer, ReferenceMuonAdam):
        raise ValueError(f"{label}: expected ReferenceMuonAdam")
    if not isinstance(saved, Mapping) or saved.get("algorithm_name") != REFERENCE_OPTIMIZER_ALGORITHM:
        raise ValueError(f"{label}: reference Muon/Adam algorithm marker differs")
    groups, states = saved.get("param_groups"), saved.get("state")
    if not isinstance(groups, list) or not isinstance(states, Mapping) or len(groups) != len(optimizer.param_groups):
        raise ValueError(f"{label}: invalid optimizer groups/state")
    used = set()
    canonical_groups = optimizer.state_dict()["param_groups"]
    for actual, expected, canonical in zip(groups, optimizer.param_groups, canonical_groups):
        if not isinstance(actual, Mapping) or set(actual) != set(expected):
            raise ValueError(f"{label}: optimizer group schema differs")
        ids = actual["params"]
        if (not isinstance(ids, list) or len(ids) != len(expected["params"])
                or any(isinstance(key, bool) or not isinstance(key, int) or key < 0 for key in ids)
                or ids != canonical["params"]
                or len(set(ids)) != len(ids) or used.intersection(ids)):
            raise ValueError(f"{label}: optimizer parameter layout differs")
        used.update(ids)
        for key, value in expected.items():
            if key in ("params", "lr"):
                continue
            if key in ("fused", "capturable"):
                if not isinstance(actual[key], bool):
                    raise ValueError(f"{label}: invalid runtime optimizer mode")
                continue
            other = actual[key]
            if isinstance(value, (tuple, list)):
                equal = isinstance(other, (tuple, list)) and tuple(other) == tuple(value)
            else:
                equal = type(other) is type(value) and other == value
            if not equal:
                raise ValueError(f"{label}: optimizer algorithm/layout differs at {key}")
        lr = actual["lr"]
        if (not isinstance(lr, torch.Tensor) or lr.layout != torch.strided or lr.ndim != 0 or lr.dtype != torch.float32
                or not bool(torch.isfinite(lr)) or bool(lr < 0)):
            raise ValueError(f"{label}: invalid scalar learning rate")
        for key, parameter in zip(ids, expected["params"]):
            state = states.get(key)
            required = {"step", "exp_avg", "exp_avg_sq"} if expected["kind"] == "adam" else {"momentum_buffer"}
            if not isinstance(state, Mapping) or set(state) != required:
                raise ValueError(f"{label}: optimizer state fields differ")
            for name, tensor in state.items():
                if (not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided
                        or not bool(torch.isfinite(tensor).all())):
                    raise ValueError(f"{label}: non-finite optimizer state {name}")
                if name == "step":
                    if tensor.ndim != 0 or tensor.dtype != torch.float32 or bool(tensor < 0) or bool(tensor != tensor.floor()):
                        raise ValueError(f"{label}: invalid Adam step")
                elif tensor.shape != parameter.shape or tensor.dtype != parameter.dtype:
                    raise ValueError(f"{label}: optimizer state shape/dtype differs at {name}")
                elif name == "exp_avg_sq" and bool((tensor < 0).any()):
                    raise ValueError(f"{label}: negative Adam second moment")
    if set(states) != used:
        raise ValueError(f"{label}: optimizer contains missing or unexpected state entries")
