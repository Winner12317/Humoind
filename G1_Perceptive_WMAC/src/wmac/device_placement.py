"""Run the simulator and the learner on different devices.

The environment (MuJoCo-Warp, ray caster, managers and its state adapter) lives
on the simulator device. The learner (networks, replay, imagination) lives on
``cfg.device``. Imagination evaluates the adapter's kinematics, rewards and
terrain queries on learner tensors, so the learner needs its own copy of the
adapter's constants; everything that reads the live simulation stays on the
environment's adapter and is moved across once per control step.

With one device (the default) every function here returns its argument.

Copies between two CUDA devices are staged through pinned CPU memory and
checked. A direct GPU-to-GPU copy uses PCIe peer-to-peer transfer, which
NVIDIA does not support on bare-metal Linux with the IOMMU enabled and which
has been seen to deliver wrong values silently on such machines, while the two
host copies are the ordinary, reliable path. Every crossing is verified with a
checksum computed on both cards. After a mismatch every step is repeated (both
checksums and the copy), and a mismatch that persists is raised, never trained
on. ``configure_transfers`` selects the route and the check.
"""
from __future__ import annotations

import copy
import dataclasses

import torch

LIVE_ADAPTER_READS = ("state", "command", "context")
# "host": GPU -> pinned CPU memory -> GPU. "direct": one peer-to-peer copy.
TRANSFER_ROUTES = ("host", "direct")
TRANSFER_ATTEMPTS = 3
_policy = {"route": "host", "verify": True}
_staging = {}
REPORTED_RECOVERIES = 20
# checksum_mismatches counts failed checks, each followed by a retry; source_readings_differed
# counts the retries in which the source card returned a different checksum for the same tensor.
transfer_statistics = {"cross_gpu_tensors": 0, "checksum_mismatches": 0, "source_readings_differed": 0}


def configure_transfers(route="host", verify=True):
    """Set how tensors cross between two CUDA devices for this process."""
    if route not in TRANSFER_ROUTES:
        raise ValueError(f"device transfer route must be one of {TRANSFER_ROUTES}")
    _policy.update(route=route, verify=bool(verify))


def crosses_gpus(source, target):
    source, target = torch.device(source), torch.device(target)
    return source.type == "cuda" and target.type == "cuda" and source != target


def checksum(tensor):
    """Sum of the tensor's bytes, computed where the tensor lives (a 0-dim int64 tensor)."""
    if tensor.is_complex() or tensor.numel() == 0:
        return torch.zeros((), dtype=torch.int64, device=tensor.device)
    return tensor.contiguous().reshape(-1).view(torch.uint8).sum(dtype=torch.int64)


def _copy(tensor, device, key=None):
    """One unverified copy by the configured route."""
    if _policy["route"] == "direct" or not crosses_gpus(tensor.device, device):
        return tensor.to(device)
    if key is None:
        return tensor.cpu().to(device)
    # Per-step traffic reuses one pinned buffer per stream; both legs block, so
    # the buffer is free again when this returns.
    slot = (key, tuple(tensor.shape), tensor.dtype)
    host = _staging.get(slot)
    if host is None:
        host = _staging[slot] = torch.empty(tensor.shape, dtype=tensor.dtype, pin_memory=torch.cuda.is_available())
    host.copy_(tensor)
    return host.to(device)


def _bytes(tensor):
    tensor = torch.view_as_real(tensor) if tensor.is_complex() else tensor
    return tensor.contiguous().reshape(-1).view(torch.uint8)


def _settled_checksum(tensor):
    """The checksum once every CUDA stream on the tensor's card has finished its work."""
    if tensor.is_cuda:
        torch.cuda.synchronize(tensor.device)
    return int(checksum(tensor))


def _diagnose(tensor, moved):
    """Which step is wrong, by exact comparison through CPU memory (failure path only)."""
    first, second, arrived = tensor.cpu(), tensor.cpu(), moved.cpu()
    facts = {
        "source_stable": bool(torch.equal(_bytes(first), _bytes(second))),
        "copy_equals_source": bool(torch.equal(_bytes(arrived), _bytes(first))),
        "source_checksum_right": int(checksum(tensor)) == int(checksum(first)),
        "target_checksum_right": int(checksum(moved)) == int(checksum(arrived)),
    }
    text = (f"the tensor on {tensor.device} {'is stable' if facts['source_stable'] else 'CHANGES between two readings'}; "
            f"the copy {'equals' if facts['copy_equals_source'] else 'DIFFERS from'} it; "
            f"the checksum computed on {tensor.device} is {'right' if facts['source_checksum_right'] else 'WRONG'}; "
            f"the checksum computed on {moved.device} is {'right' if facts['target_checksum_right'] else 'WRONG'}")
    return facts, text


def _mismatch(names, tensor, moved):
    facts, text = _diagnose(tensor, moved)
    error = RuntimeError(
        f"Tensors {names} arrived on {moved.device} with different contents than they had on {tensor.device}, "
        f"after {TRANSFER_ATTEMPTS + 1} copies and {TRANSFER_ATTEMPTS + 1} readings of each checksum "
        f"(route: {_policy['route']}). Refusing to train on corrupted data. Exact comparison through CPU memory: "
        f"{text}. Run scripts/check_gpu_transfers.py on this machine; see README, 'Training on two GPUs'.")
    error.diagnosis = facts
    return error


def _note(key, what):
    if transfer_statistics["checksum_mismatches"] <= REPORTED_RECOVERIES:
        print(f"[device transfer] checksum mismatch on {key or 'tensor'!r}: {what}. Training continues with "
              "verified data; the count is in runtime.json under device_transfer.", flush=True)


def _recover(tensor, moved, expected, device, key):
    """Slow path after a checksum mismatch: the verified copy, or an error.

    Four steps can have failed: the checksum on the source card, either leg of
    the copy, or the checksum on the target card. Each is repeated, the source
    one after every stream on its card has finished, so that a single bad
    reading of any of them costs a retry and not the run.
    """
    for _ in range(TRANSFER_ATTEMPTS):
        transfer_statistics["checksum_mismatches"] += 1
        source = _settled_checksum(tensor)
        if source != expected:
            transfer_statistics["source_readings_differed"] += 1
            expected = source
        if _settled_checksum(moved) == expected:
            _note(key, "a checksum reading was wrong and the copy was right")
            return moved
        moved = _copy(tensor, device, key)
        if int(checksum(moved)) == expected:
            _note(key, "the copy was wrong and a second copy was right")
            return moved
    raise _mismatch([key or "tensor"], tensor, moved)


def transfer(tensor, device, *, key=None):
    """``tensor`` on ``device``; copies between two GPUs are staged through the host and verified."""
    device = torch.device(device)
    if tensor.device == device:
        return tensor
    if not crosses_gpus(tensor.device, device):
        return tensor.to(device)
    transfer_statistics["cross_gpu_tensors"] += 1
    expected = int(checksum(tensor)) if _policy["verify"] else None
    moved = _copy(tensor, device, key)
    if expected is None or int(checksum(moved)) == expected:
        return moved
    return _recover(tensor, moved, expected, device, key)


def transfer_batch(batch, device):
    """A dictionary's tensors on ``device``, verified with one checksum readback per side."""
    device = torch.device(device)
    names = [name for name, item in batch.items() if isinstance(item, torch.Tensor) and item.device != device]
    if not names:
        return batch
    moved = dict(batch)
    crossing = [name for name in names if crosses_gpus(batch[name].device, device)]
    for name in names:
        if name not in crossing:
            moved[name] = batch[name].to(device)
    if not crossing:
        return moved
    transfer_statistics["cross_gpu_tensors"] += len(crossing)
    expected = (torch.stack([checksum(batch[name]) for name in crossing]).cpu() if _policy["verify"] else None)
    for name in crossing:
        moved[name] = _copy(batch[name], device, name)
    if expected is None:
        return moved
    arrived = torch.stack([checksum(moved[name]) for name in crossing]).cpu()
    for name, good, value in zip(crossing, (arrived == expected).tolist(), expected.tolist()):
        if not good:
            moved[name] = _recover(batch[name], moved[name], value, device, name)
    return moved


def transfer_self_test(first, second, *, trials=10, megabytes=(1, 64)):
    """Exact comparison of copies in both directions by both routes.

    Returns ``{route: {"a->b": [wrong, total], ...}}``. The comparison reads both
    tensors back through the host, which is the path that is known to be sound.
    """
    report, previous = {}, dict(_policy)
    generator = torch.Generator().manual_seed(0)
    try:
        for route in TRANSFER_ROUTES:
            _policy.update(route=route, verify=False)
            report[route] = {}
            for source, target in ((first, second), (second, first)):
                wrong = total = 0
                for trial in range(trials):
                    size = megabytes[trial % len(megabytes)] * 2**20
                    original = torch.randint(0, 256, (size,), dtype=torch.uint8, generator=generator)
                    on_source = original.to(source)
                    arrived = _copy(on_source, torch.device(target))
                    if torch.device(target).type == "cuda":
                        torch.cuda.synchronize(target)
                    wrong += int(not torch.equal(arrived.cpu(), original))
                    total += 1
                report[route][f"{source}->{target}"] = [wrong, total]
    finally:
        _policy.update(previous)
    return report


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
        result = transfer(value, device)
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
        setattr(moved, name, lambda live=live, name=name: transfer(live(), device, key=f"adapter.{name}"))
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

    A value already there is returned unchanged, so single-device runs pay
    nothing. Copies between two GPUs follow ``transfer``'s route and check.
    """
    if isinstance(value, torch.Tensor):
        return transfer(value, device, key=f"stream.{tuple(value.shape)}.{value.dtype}")
    if isinstance(value, dict):
        return transfer_batch(value, device)
    return value
