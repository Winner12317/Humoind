"""Run the simulator and the learner on different devices.

The environment (MuJoCo-Warp, ray caster, managers and its state adapter) lives
on the simulator device. The learner (networks, replay, imagination) lives on
``cfg.device``. Imagination evaluates the adapter's kinematics, rewards and
terrain queries on learner tensors, so the learner needs its own copy of the
adapter's constants; everything that reads the live simulation stays on the
environment's adapter and is moved across once per control step.

With one device (the default) every function here returns its argument.
"""
from __future__ import annotations

import copy
import dataclasses

import torch

LIVE_ADAPTER_READS = ("state", "command", "context")


def relocate(value, device, *, source=None, shared=(), _memo=None):
    """Copy the project objects reachable from ``value`` with their tensors on ``device``.

    Tensors are moved, containers and instances of classes defined under
    ``src.`` are copied shallowly and descended into, and everything else
    (environment, mjlab entities, configs, NumPy arrays, functions) is shared.
    ``shared`` lists objects that must not be copied even though the rule would;
    ``source`` is the device whose ``torch.device`` attributes are rewritten.
    """
    memo = {} if _memo is None else _memo
    key = id(value)
    if key in memo:
        return memo[key]
    again = lambda item: relocate(item, device, source=source, shared=shared, _memo=memo)
    if any(value is item for item in shared):
        return value
    if isinstance(value, torch.Tensor):
        result = value.to(device)
    elif isinstance(value, torch.device):
        result = torch.device(device) if source is not None and value == torch.device(source) else value
    elif isinstance(value, torch.nn.Module):
        result = copy.deepcopy(value).to(device)
    elif type(value) is dict:
        result = memo[key] = {}
        result.update((name, again(item)) for name, item in value.items())
    elif type(value) is list:
        result = memo[key] = []
        result.extend(again(item) for item in value)
    elif isinstance(value, tuple):
        items = [again(item) for item in value]
        result = type(value)(*items) if hasattr(value, "_fields") else tuple(items) if type(value) is tuple else value
    elif (not isinstance(value, type) and type(value).__module__.startswith("src.")
          and (hasattr(value, "__dict__") or dataclasses.is_dataclass(value))):
        result = memo[key] = copy.copy(value)
        if hasattr(value, "__dict__"):
            for name, item in vars(value).items():
                result.__dict__[name] = again(item)
        else:
            for field in dataclasses.fields(value):
                object.__setattr__(result, field.name, again(getattr(value, field.name)))
    else:
        result = value
    memo[key] = result
    return result


def learner_adapter(adapter, device):
    """The adapter the learner imagines with: ``adapter`` itself, or a copy on ``device``.

    The copy's ``state()``, ``command()`` and ``context()`` still read the live
    simulation through the environment's adapter and return learner tensors.
    """
    device = torch.device(device)
    if torch.device(adapter.device) == device:
        return adapter
    moved = relocate(adapter, device, source=adapter.device, shared=(adapter.env,))
    for name in LIVE_ADAPTER_READS:
        live = getattr(adapter, name)
        setattr(moved, name, lambda live=live: live().to(device))
    return moved


def boundary(runner):
    """``(from_env, to_env)`` for a runner: identities unless it declares two devices."""
    learner, simulator = getattr(runner, "device", None), getattr(runner, "env_device", None)
    if learner is None or simulator is None or torch.device(learner) == torch.device(simulator):
        return (lambda value: value), (lambda value: value)
    learner, simulator = torch.device(learner), torch.device(simulator)
    return (lambda value: to_device(value, learner)), (lambda value: to_device(value, simulator))


def to_device(value, device):
    """Move a tensor, or the tensors of a transition dictionary, to ``device``.

    A value already there is returned unchanged, so single-device runs pay nothing.
    """
    if isinstance(value, torch.Tensor):
        return value if value.device == device else value.to(device, non_blocking=True)
    if isinstance(value, dict):
        if all(not isinstance(item, torch.Tensor) or item.device == device for item in value.values()):
            return value
        return {name: to_device(item, device) for name, item in value.items()}
    return value
