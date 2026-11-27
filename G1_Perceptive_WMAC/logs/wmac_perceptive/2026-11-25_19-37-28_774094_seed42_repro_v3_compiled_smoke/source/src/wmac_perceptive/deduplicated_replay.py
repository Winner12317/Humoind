"""Lossless temporal storage for Hiking's observed proprioception and context.

The public sampling contract is unchanged. In particular, noisy observations
are stored, never recreated from physical state. Unexpected history/context
values use explicit sparse exceptions rather than an assumed physical label.
"""
from __future__ import annotations

from dataclasses import is_dataclass
import math

import numpy as np
import torch

from src.wmac.replay import ReplayBuffer
from .compact_gpu_replay import CompressedLatentReplay
from .hiking_amp import before_replay_add, after_replay_add
from .hiking_context import advance_hiking_context
from .hiking_observations import pack_hiking_proprio_history, unpack_hiking_proprio_history


def _same_bits(left, right):
    return (left.contiguous().view(torch.int32) == right.contiguous().view(torch.int32))


class _SparseRows:
    """Exact uncommon values; normal collector streams allocate no rows here."""
    def __init__(self):
        self.rows = {}
        self.total_entries = 0

    def put(self, row, mask, values):
        indices = np.flatnonzero(mask)
        self.rows.pop(row, None)
        if len(indices):
            ids = torch.as_tensor(indices, dtype=torch.long, device=values.device)
            self.rows[row] = (ids, values[ids].detach().clone())
            self.total_entries += len(indices)

    def apply(self, values, rows, env):
        for row, (indices, exceptions) in self.rows.items():
            slot = torch.searchsorted(indices, env.contiguous()).clamp_max(len(indices)-1)
            selected = (rows == row) & (indices[slot] == env)
            values = torch.where(selected.reshape(*selected.shape, *((1,)*(values.ndim-selected.ndim))),
                                 exceptions[slot], values)
        return values

    def state_dict(self):
        return dict(rows=self.rows, total_entries=self.total_entries)

    def load_state_dict(self, saved, capacity, num_envs, trailing_shape, device):
        rows = {}
        for row, pair in saved["rows"].items():
            if not isinstance(row, int) or not 0 <= row < capacity:
                raise ValueError("Invalid sparse replay row")
            ids, values = pair
            if (ids.dtype != torch.long or ids.ndim != 1 or not len(ids)
                    or tuple(values.shape) != (len(ids), *trailing_shape)
                    or values.dtype != torch.float32
                    or not bool(((ids >= 0) & (ids < num_envs)).all())
                    or not bool((ids[1:] > ids[:-1]).all())):
                raise ValueError("Invalid sparse replay values")
            rows[row] = (ids.to(device).clone(), values.to(device).clone())
        self.rows = rows
        self.total_entries = int(saved["total_entries"])
        if self.total_entries < sum(len(ids) for ids, _ in rows.values()):
            raise ValueError("Invalid sparse replay exception count")


class _ContextEvents:
    """Episode-stable values stored once per actual bitwise change.

    Refcounts include each live replay row and the current per-env anchor.
    Reused slots therefore cannot invalidate an old sampled transition.
    """
    def __init__(self, capacity, num_envs, device, width=65):
        self.capacity, self.num_envs, self.device = capacity, num_envs, torch.device(device)
        self.width = width
        self.ids = torch.full((capacity, num_envs), -1, dtype=torch.int32, device=device)
        self.ids_cpu = np.full((capacity, num_envs), -1, dtype=np.int32)
        self.current = np.full(num_envs, -1, dtype=np.int64)
        self.references = np.zeros(max(2*num_envs, 1), dtype=np.int64)
        self.values = torch.empty(len(self.references), width, dtype=torch.float32, device=device)
        self.last = torch.empty(num_envs, width, dtype=torch.float32, device=device)
        self.total_events = 0

    def add(self, row, values, changed):
        expired = self.ids_cpu[row]
        expired = expired[expired >= 0]
        np.add.at(self.references, expired, -1)
        envs = np.flatnonzero(changed)
        old = self.current[envs]
        np.add.at(self.references, old[old >= 0], -1)
        free = np.flatnonzero(self.references == 0)
        if len(free) < len(envs):
            old_capacity = len(self.references)
            new_capacity = max(2*old_capacity, old_capacity+len(envs)-len(free))
            new_values = self.values.new_empty(new_capacity, self.width)
            new_values[:old_capacity].copy_(self.values)
            self.values = new_values
            self.references = np.pad(self.references, (0, new_capacity-old_capacity))
            free = np.flatnonzero(self.references == 0)
        slots = free[:len(envs)]
        if len(envs):
            ids = torch.as_tensor(envs, dtype=torch.long, device=self.device)
            self.values[torch.as_tensor(slots, device=self.device)] = values[ids]
            self.current[envs] = slots
            np.add.at(self.references, slots, 1)  # Current anchor.
        np.add.at(self.references, self.current, 1)  # This replay row.
        self.ids_cpu[row] = self.current
        self.ids[row].copy_(torch.as_tensor(self.current, dtype=torch.int32, device=self.device))
        self.last.copy_(values)
        self.total_events += len(envs)

    def get(self, rows, env):
        return self.values[self.ids[rows, env].long()]

    def state_dict(self):
        return dict(ids=self.ids, values=self.values, last=self.last,
            current=torch.from_numpy(self.current.copy()), references=torch.from_numpy(self.references.copy()),
            total_events=self.total_events)

    def load_state_dict(self, saved):
        ids, values, last = saved["ids"], saved["values"], saved["last"]
        current, references = saved["current"], saved["references"]
        if (ids.shape != self.ids.shape or ids.dtype != torch.int32
                or values.ndim != 2 or values.shape[1] != self.width or values.dtype != torch.float32
                or last.shape != self.last.shape or last.dtype != torch.float32
                or current.shape != (self.num_envs,) or current.dtype != torch.long
                or references.shape != (len(values),) or references.dtype != torch.long):
            raise ValueError("Invalid context event storage")
        host_ids, host_current = ids.cpu().numpy(), current.cpu().numpy()
        if (np.any(host_ids < -1) or np.any(host_ids >= len(values))
                or np.any(host_current < -1) or np.any(host_current >= len(values))):
            raise ValueError("Invalid context event references")
        counts = np.bincount(np.concatenate((host_ids[host_ids >= 0], host_current[host_current >= 0])),
                             minlength=len(values))
        if not np.array_equal(counts, references.cpu().numpy()):
            raise ValueError("Context event reference counts differ")
        self.ids.copy_(ids.to(self.device)); self.ids_cpu = host_ids.copy()
        self.values = values.to(self.device).clone(); self.last.copy_(last.to(self.device))
        self.current = host_current.copy(); self.references = counts
        self.total_events = int(saved["total_events"])


class DeduplicatedHikingReplay(CompressedLatentReplay):
    """Exact full-task replay with shared 96D frames and context event values.

    Physical states/actions/contexts/auxiliary labels use the production FP32
    contract. Observations and relative height accept floating inputs and are
    canonicalized to FP32 before finiteness checks, as in the original replay.
    """
    storage_protocol = "hiking_lossless_frame_uint8_v2"
    layout_version = 2
    _FIXED = tuple(range(33)) + tuple(range(35, 38)) + tuple(range(67, 96))
    _DYNAMIC = (33, 34) + tuple(range(38, 67)) + tuple(range(96, 100))
    _AMP_FIELDS = ("_amp_overflow_states", "_amp_episode_starts", "_amp_episode_start", "_amp_pending_reset")

    def __init__(self, *args, camera_capture_steps=0, context_dim=100, **kwargs):
        super().__init__(*args, **kwargs)
        if self.spec.proprio_dim != 768 or self.spec.include_valid:
            raise ValueError("Deduplicated Hiking replay requires 8x96 term-major, depth-only observations")
        self.camera_capture_steps = int(camera_capture_steps)
        if isinstance(context_dim,bool) or not isinstance(context_dim,int) or context_dim<100:
            raise ValueError("Hiking replay context must contain at least the original100 fields")
        self.context_dim=context_dim
        self._fixed_columns=(*self._FIXED,*range(100,context_dim))
        if context_dim>100:
            self.storage_protocol="hiking_lossless_frame_uint8_v3"
            self.layout_version=3
        self._proprio_cache = torch.empty(self.capacity_steps+7, self.num_envs, 96, dtype=torch.float32, device=self.device)
        self._context_events = _ContextEvents(self.capacity_steps, self.num_envs, self.device,len(self._fixed_columns))
        self._history_exceptions = _SparseRows()
        self._next_context_exceptions = _SparseRows()
        self._previous_action_exceptions = _SparseRows()
        self._original_schema = None
        self._incoming_original_finite = None
        self._control_readbacks = 0
        self._fixed_indices = torch.tensor(self._fixed_columns, device=self.device)
        self._dynamic_indices = torch.tensor(self._DYNAMIC, device=self.device)

    def finite_mask(self, batch):
        valid = super().finite_mask(batch)
        if self._incoming_original_finite is not None:
            valid &= self._incoming_original_finite.to(valid.device)
        return valid

    def _index_row(self, row, valid):
        # ReplayBuffer's context[...,34] is stored at dynamic column1 here.
        recovery = self.data["relative_height"][row] < .55
        if self.spec.depth_protocol != "g1_rpl_v1":
            recovery |= self.data["context"][row, :, 1] > 0
        self._valid[row].copy_(valid); self._recovery[row].copy_(recovery)
        category = 2*self._validation_env.long()+recovery.long()
        category = torch.where(valid, category, torch.full_like(category, 4))
        order = torch.argsort(category)
        counts = torch.bincount(category, minlength=5)[:4]
        self._env_pool[row].copy_(order); self._group_counts[row].copy_(counts)
        self._group_offsets[row].copy_(counts.cumsum(0)-counts)
        self._boundaries[row].zero_()
        for name in ("boundary", "terminated", "timeout"):
            if name in self.data:
                self._boundaries[row] |= self.data[name][row].reshape(self.num_envs, -1).bool().any(-1)

    def _schema(self, batch):
        return {k: (tuple(v.shape[1:]), str(torch.float32 if k in ("actor_history", "relative_height") else v.dtype))
                for k, v in batch.items() if k not in self._DEPTH_FIELDS}

    @torch.no_grad()
    def add(self, batch, *, processed_history=None):
        self._validate_perceptive_shapes(batch)
        batch = dict(batch)
        batch["actor_history"] = batch["actor_history"].float()
        batch["relative_height"] = batch["relative_height"].float()
        for name, width in (("state", 71), ("next_state", 71),
                            ("context", self.context_dim), ("next_context", self.context_dim), ("action", 29),
                            ("previous_action", 29), ("contact_cost", 13)):
            value = batch.get(name)
            if (not isinstance(value, torch.Tensor) or tuple(value.shape) != (self.num_envs, width)
                    or value.dtype != torch.float32):
                raise ValueError(f"Deduplicated Hiking replay requires FP32 {name}[N,{width}]")
        if self._RESERVED.intersection(batch):
            raise ValueError("Replay data uses reserved sampling metadata")
        original_finite = ReplayBuffer.finite_mask(batch)
        schema = self._schema(batch)
        if self._original_schema is not None and schema != self._original_schema:
            raise ValueError("Replay fields must stay identical after allocation")
        if processed_history is not None:
            self.bootstrap_processed_history(processed_history)
        if not self._bootstrapped:
            raise ValueError("Shared-frame replay requires bootstrap")
        self._original_schema = schema
        row, time = self.ptr, self.total_steps
        frames = unpack_hiking_proprio_history(batch["actor_history"].float()).to(self.device)
        if not time:
            self._proprio_cache[torch.arange(-7, 0, device=self.device).remainder(len(self._proprio_cache))] = frames[:, :-1].transpose(0, 1)
        self._proprio_cache[time % len(self._proprio_cache)].copy_(frames[:, -1])
        starts = torch.where(self._pending_reset, time, self._episode_start)
        proprio_starts = torch.where(starts < 0, torch.full_like(starts, -7), starts)
        times = torch.maximum(time+torch.arange(-7, 0, device=self.device)[None], proprio_starts[:, None])
        expected = self._proprio_cache[times.remainder(len(self._proprio_cache)), torch.arange(self.num_envs, device=self.device)[:, None]]
        context, action, auxiliary, next_context, previous = (batch[n].to(self.device) for n in
            ("context", "action", "contact_cost", "next_context", "previous_action"))
        fixed = context.index_select(-1, self._fixed_indices)
        predicted_next = advance_hiking_context(context, action, auxiliary,
                                               camera_capture_steps=self.camera_capture_steps)
        # One explicit control readback for all exact-sharing decisions.
        changes = torch.stack((torch.ones(self.num_envs, dtype=torch.bool, device=self.device) if not time else
                ~_same_bits(fixed, self._context_events.last).all(-1),
            ~_same_bits(expected, frames[:, :-1]).all((-1, -2)),
            ~_same_bits(predicted_next, next_context).all(-1),
            ~_same_bits(context[:, 38:67], previous).all(-1)), -1).cpu().numpy()
        self._control_readbacks += int(self.device.type == "cuda")
        self._context_events.add(row, fixed, changes[:, 0])
        self._history_exceptions.put(row, changes[:, 1], frames[:, :-1])
        self._next_context_exceptions.put(row, changes[:, 2], next_context)
        self._previous_action_exceptions.put(row, changes[:, 3], previous)
        stored = {k: v for k, v in batch.items()
                  if k not in (*self._DEPTH_FIELDS, "actor_history", "next_context", "previous_action")}
        stored["context"] = context.index_select(-1, self._dynamic_indices)
        stored["relative_height"] = batch["relative_height"].float()
        visual_valid = self._visual_valid(batch["depth_stack"], False) & self._visual_valid(batch["next_depth_stack"], False)
        self._incoming_original_finite, self._incoming_visual_valid = original_finite, visual_valid
        before_replay_add(self)
        try:
            accepted = ReplayBuffer.add(self, stored)
        finally:
            self._incoming_original_finite = self._incoming_visual_valid = None
        after_replay_add(self, stored, accepted)
        self.frame_cache[time % self.cache_steps].copy_(self._quantize(batch["next_depth_stack"][:, -1:]).to(self.device))
        self._episode_starts[row].copy_(starts); self._episode_start.copy_(starts)
        self._pending_reset.zero_()
        for name in ("terminated", "timeout"):
            if name in batch:
                self._pending_reset |= batch[name].to(self.device).reshape(self.num_envs, -1).bool().any(-1)
        return accepted

    def _fields(self, fields):
        available = tuple(self._original_schema or ()) + self._DEPTH_FIELDS
        selected = available if fields is None else tuple(fields)
        if len(set(selected)) != len(selected):
            raise ValueError("Requested replay fields must be unique")
        missing = set(selected).difference(available)
        if missing:
            raise KeyError(f"Unknown replay fields: {sorted(missing)}")
        return selected

    def _context(self, rows, env):
        fixed = self._context_events.get(rows, env)
        dynamic = self.data["context"][rows, env]
        result = dynamic.new_empty(*dynamic.shape[:-1], self.context_dim)
        result[..., self._fixed_indices] = fixed
        result[..., self._dynamic_indices] = dynamic
        return result

    def _history(self, rows, env):
        starts = self._episode_starts[rows, env]
        starts = torch.where(starts < 0, torch.full_like(starts, -7), starts)
        times = torch.maximum(self._timestamps[rows][..., None]+torch.arange(-7, 1, device=self.device),
                              starts[..., None])
        frames = self._proprio_cache[times.remainder(len(self._proprio_cache)), env[..., None]]
        older = self._history_exceptions.apply(frames[..., :-1, :], rows, env)
        return pack_hiking_proprio_history(torch.cat((older, frames[..., -1:, :]), -2))

    def _gather_fields(self, rows, env, fields):
        result = {}
        context = None
        for name in fields:
            if name in ("context", "next_context", "previous_action") and context is None:
                context = self._context(rows, env)
            if name == "context": result[name] = context
            elif name == "next_context":
                value = advance_hiking_context(context, self.data["action"][rows, env],
                    self.data["contact_cost"][rows, env], camera_capture_steps=self.camera_capture_steps)
                result[name] = self._next_context_exceptions.apply(value, rows, env)
            elif name == "previous_action":
                result[name] = self._previous_action_exceptions.apply(context[..., 38:67], rows, env)
            elif name == "actor_history": result[name] = self._history(rows, env)
            elif name in self._DEPTH_FIELDS: result[name] = self._depth(rows, env, name == "next_depth_stack")
            else: result[name] = self.data[name][rows, env]
        return result

    def state_dict(self):
        result = super().state_dict()
        result.update(layout_version=self.layout_version, camera_capture_steps=self.camera_capture_steps,
            original_schema=self._original_schema, proprio_cache=self._proprio_cache,
            context_events=self._context_events.state_dict(),
            history_exceptions=self._history_exceptions.state_dict(),
            next_context_exceptions=self._next_context_exceptions.state_dict(),
            previous_action_exceptions=self._previous_action_exceptions.state_dict(),
            hiking_amp_state={n: getattr(self, n) for n in self._AMP_FIELDS} if getattr(self, "_hiking_amp_history", False) else None)
        if self.context_dim>100:result["context_dim"]=self.context_dim
        return result

    @torch.no_grad()
    def load_state_dict(self, saved):
        if (saved.get("perceptive_format") != self.storage_protocol or saved.get("layout_version") != self.layout_version
                or saved.get("camera_capture_steps") != self.camera_capture_steps
                or saved.get("context_dim",100)!=self.context_dim):
            raise ValueError("Replay layout differs; old full-history replay requires explicit conversion")
        value = saved.get("proprio_cache")
        if not isinstance(value, torch.Tensor) or value.shape != self._proprio_cache.shape or value.dtype != torch.float32:
            raise ValueError("Invalid proprioceptive frame cache")
        schema = saved.get("original_schema")
        if saved.get("size", 0) and not isinstance(schema, dict):
            raise ValueError("Missing original replay schema")
        schema = schema or {}
        required = dict(state=(71,), next_state=(71,), context=(self.context_dim,), next_context=(self.context_dim,), action=(29,),
            previous_action=(29,), contact_cost=(13,), actor_history=(768,), relative_height=())
        if saved.get("size", 0):
            for name, shape in required.items():
                if (name not in schema or tuple(schema[name][0]) != shape
                        or schema[name][1] != "torch.float32"):
                    raise ValueError("Invalid original Hiking replay schema")
        expected_fields = set(schema).difference(("actor_history", "next_context", "previous_action"))
        if set(saved.get("data", {})) != expected_fields:
            raise ValueError("Compressed replay fields differ from original schema")
        for name, tensor in saved.get("data", {}).items():
            shape, dtype = schema[name]
            shape = (96,) if name == "actor_history" else (35,) if name == "context" else tuple(shape)
            if (not isinstance(tensor, torch.Tensor)
                    or tuple(tensor.shape) != (self.capacity_steps, self.num_envs, *shape)
                    or str(tensor.dtype) != dtype):
                raise ValueError("Invalid compressed Hiking replay field: "+name)
        ptr, size = int(saved["ptr"]), int(saved["size"])
        if not 0 <= ptr < self.capacity_steps or not 0 <= size <= self.capacity_steps:
            raise ValueError("Invalid replay ring indices")
        live_rows = np.arange(ptr-size, ptr) % self.capacity_steps
        events = saved["context_events"]
        if size and (bool((events["ids"][live_rows] < 0).any()) or bool((events["current"] < 0).any())):
            raise ValueError("Live replay row is missing context events")
        amp = saved.get("hiking_amp_state")
        if size and getattr(self, "_hiking_amp_history", False) and amp is None:
            raise ValueError("Missing Hiking AMP replay metadata")
        if amp is not None:
            shapes = {"_amp_overflow_states": ((9, self.num_envs, 71), torch.float32),
                "_amp_episode_starts": ((self.capacity_steps, self.num_envs), torch.long),
                "_amp_episode_start": ((self.num_envs,), torch.long),
                "_amp_pending_reset": ((self.num_envs,), torch.bool)}
            if set(amp) != set(shapes):
                raise ValueError("Hiking AMP replay metadata differs")
            for name, (shape, dtype) in shapes.items():
                if not isinstance(amp[name], torch.Tensor) or tuple(amp[name].shape) != shape or amp[name].dtype != dtype:
                    raise ValueError("Hiking AMP replay metadata differs")
        self._context_events.load_state_dict(saved["context_events"])
        for name, shape in (("history", (7, 96)), ("next_context", (self.context_dim,)), ("previous_action", (29,))):
            getattr(self, "_"+name+"_exceptions").load_state_dict(saved[name+"_exceptions"],
                self.capacity_steps, self.num_envs, shape, self.device)
        super().load_state_dict(saved)
        self._original_schema = schema or None
        self._proprio_cache.copy_(value.to(self.device))
        if amp is not None:
            from .hiking_amp import configure_replay
            configure_replay(self)
            for name in self._AMP_FIELDS:
                target = getattr(self, name); value = amp[name]
                if target.shape != value.shape or target.dtype != value.dtype:
                    raise ValueError("Hiking AMP replay metadata differs")
                target.copy_(value.to(self.device))
            if self.size: self._amp_pending_reset.fill_(True)

    def storage_report(self):
        seen = set(); total = 0
        def visit(value):
            nonlocal total
            if isinstance(value, torch.Tensor):
                storage = value.untyped_storage(); key = (str(value.device), storage.data_ptr())
                if key not in seen: seen.add(key); total += storage.nbytes()
            elif isinstance(value, dict):
                for v in value.values(): visit(v)
            elif isinstance(value, (tuple, list)):
                for v in value: visit(v)
            elif isinstance(value, (_ContextEvents, _SparseRows)) or is_dataclass(value): visit(vars(value))
        visit(vars(self))
        original_native = sum(math.prod(shape)*torch.empty((), dtype=getattr(torch, dtype.split(".")[-1])).element_size()
                              for shape, dtype in (self._original_schema or {}).values())*self.capacity_steps*self.num_envs
        # Count the common original replay allocations once; new codec tensors are excluded.
        original_metadata = sum(getattr(self, n).untyped_storage().nbytes() for n in (
            "frame_cache", "_episode_starts", "_episode_start", "_pending_reset", "_offsets", "_recovery", "_valid",
            "_boundaries", "_cut_rows", "_timestamps", "_env_pool", "_group_counts", "_group_offsets", "_validation_env", "_train_env"))
        original_metadata += sum(getattr(self, n).untyped_storage().nbytes() for n in self._AMP_FIELDS if hasattr(self, n))
        sparse = {}
        for name in ("history", "next_context", "previous_action"):
            codec = getattr(self, "_"+name+"_exceptions")
            sparse[name] = dict(current_entries=sum(len(ids) for ids, _ in codec.rows.values()),
                total_entries=codec.total_entries, bytes=sum(v.numel()*v.element_size()+ids.numel()*ids.element_size()
                                                           for ids, v in codec.rows.values()))
        return dict(layout_version=self.layout_version, storage_protocol=self.storage_protocol,
            logical_dimensions=dict(capacity_steps=self.capacity_steps, num_envs=self.num_envs,
                actor_history=768, context=self.context_dim, next_context=self.context_dim),
            unique_tensor_storage_bytes=total, original_layout_estimate_bytes=original_native+original_metadata,
            schema_initialized=self._original_schema is not None,
            estimate_note="Original-layout estimate includes observed native schema and common fixed replay metadata; excludes transient samples and dynamic prefix caches",
            host_numpy_metadata_bytes=sum(v.nbytes for v in vars(self._context_events).values() if isinstance(v, np.ndarray)),
            proprio_frame_bytes=self._proprio_cache.numel()*self._proprio_cache.element_size(),
            sparse_exceptions=sparse, context_live_events=int(np.count_nonzero(self._context_events.references)),
            context_event_capacity=len(self._context_events.references), context_total_events=self._context_events.total_events,
            additional_cuda_control_readbacks=self._control_readbacks,
            control_readback_bytes_per_add=self.num_envs*4, control_readback_note="One combined sharing-mask CPU readback per CUDA add; no measured GPU timing implied")

    memory_report = storage_report
