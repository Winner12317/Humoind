"""Finite, chronological replay with cached stratified sampling.

A compact environment-index pool is updated once per vectorized add. Sampling
uses prefixes of per-row counts, never a scan of the time-by-environment mask.
A boundary preserves its actual physical successor and masks later transitions;
non-finite transitions are quarantined rather than relabelled as resets.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class _Prefix:
    cumulative: torch.Tensor
    groups: torch.Tensor
    group_count: int
    total: int
    uniform_row_count: int = 0
    uniform_row_offset: int = 0


class ReplayBuffer:
    """A time-by-environment ring with an optional held-out environment split.

    The final fixed subset of environment IDs forms validation; a trajectory
    never changes split as the ring wraps. The default validation fraction zero
    preserves the old all-training behavior. Sampling defaults to train, and an
    empty validation split never falls back to training data.

    recovery_fraction=0 means uniform sampling over all eligible transitions.
    Other fractions stratify by height < .55 or an active recovery counter.
    Non-finite data remains unchanged in storage, but is excluded from starts.
    Window masks stop at an invalid transition; only invalid masked entries are
    padded with the window's first valid transition so losses cannot see NaNs.
    """

    _RESERVED = frozenset(("mask", "time_index", "env_index"))
    _SPLIT_GROUPS = {"train": (0, 1), "validation": (2, 3), "all": (0, 1, 2, 3)}

    def __init__(self, capacity_steps: int, num_envs: int,
                 device: str | torch.device = "cpu",
                 validation_env_fraction: float = 0.0):
        if capacity_steps < 1 or num_envs < 1:
            raise ValueError("Replay capacity and environment count must be positive")
        if not 0 <= validation_env_fraction < 1:
            raise ValueError("validation_env_fraction must lie in [0, 1)")
        if validation_env_fraction > 0 and num_envs < 2:
            raise ValueError("A held-out split requires at least two environments")
        self.capacity_steps = capacity_steps
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.validation_env_fraction = float(validation_env_fraction)
        self.validation_env_count = (
            min(num_envs - 1, max(1, round(num_envs * validation_env_fraction)))
            if validation_env_fraction else 0
        )
        self.train_env_count = num_envs - self.validation_env_count
        self.data: dict[str, torch.Tensor] = {}
        self.ptr = 0
        self.size = 0
        self.total_steps = 0
        self.invalid_transitions = 0
        # True is a proof that every live row is valid. False is conservative
        # (an overwritten historical invalid row need not turn it true again).
        # It is runtime metadata, rebuilt from actual validity on restore.
        self._all_valid = True
        shape = (capacity_steps, num_envs)
        self._recovery = torch.zeros(shape, dtype=torch.bool, device=self.device)
        self._valid = torch.zeros(shape, dtype=torch.bool, device=self.device)
        self._boundaries = torch.zeros(shape, dtype=torch.bool, device=self.device)
        self._cut_rows = torch.zeros(shape, dtype=torch.bool, device=self.device)
        self._timestamps = torch.full((capacity_steps,), -1, dtype=torch.long, device=self.device)
        # Each environment occurs at most once in a row. Invalid IDs occupy an
        # unused suffix. Four segments: train normal/recovery, val normal/recovery.
        self._env_pool = torch.empty(shape, dtype=torch.int32, device=self.device)
        self._group_counts = torch.zeros((capacity_steps, 4), dtype=torch.long, device=self.device)
        self._group_offsets = torch.zeros_like(self._group_counts)
        self._validation_env = (
            torch.arange(num_envs, device=self.device) >= self.train_env_count
        )
        self._train_env = ~self._validation_env
        self._prefix_cache: dict[tuple[int, str, str], _Prefix] = {}
        self.index_cache_builds = 0

    def __len__(self) -> int:
        return self.size * self.num_envs

    @property
    def train_env_mask(self) -> torch.Tensor:
        """Fixed training environment mask on replay.device; treat as read-only."""
        return self._train_env

    @property
    def validation_env_mask(self) -> torch.Tensor:
        """Fixed held-out environment mask on replay.device; treat as read-only."""
        return self._validation_env

    @staticmethod
    @torch.no_grad()
    def finite_mask(batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return per-environment finiteness on the input batch's device."""
        if not batch or "state" not in batch:
            raise ValueError("A replay batch must include its physical state")
        first = batch["state"]
        if not isinstance(first, torch.Tensor) or first.ndim != 2 or first.shape[1] < 3:
            raise ValueError("Replay state must be an environment-by-feature tensor containing height")
        count = first.shape[0]
        valid = torch.ones(count, dtype=torch.bool, device=first.device)
        for key, value in batch.items():
            if not isinstance(value, torch.Tensor) or value.ndim < 1 or value.shape[0] != count:
                raise ValueError(f"Replay field {key!r} has an inconsistent environment dimension")
            if key == "context" and value.ndim != 2:
                raise ValueError("Replay context must be an environment-by-feature tensor")
            if value.is_floating_point() or value.is_complex():
                finite = torch.isfinite(value.detach()).reshape(count, -1).all(-1)
                valid &= finite.to(valid.device)
        return valid

    def _index_row(self, row: int, valid: torch.Tensor) -> None:
        """Update O(num_envs) indices for one row; no full-buffer reclassification."""
        recovery = self.data["state"][row, :, 2] < .55
        if "context" in self.data and self.data["context"].shape[-1] >= 35:
            recovery |= self.data["context"][row, :, 34] > 0
        self._valid[row].copy_(valid)
        self._recovery[row].copy_(recovery)
        category = 2 * self._validation_env.long() + recovery.long()
        category = torch.where(valid, category, torch.full_like(category, 4))
        order = torch.argsort(category)
        counts = torch.bincount(category, minlength=5)[:4]
        self._env_pool[row].copy_(order)
        self._group_counts[row].copy_(counts)
        self._group_offsets[row].copy_(counts.cumsum(0) - counts)
        boundary = self._boundaries[row]
        boundary.zero_()
        for key in ("boundary", "terminated", "timeout"):
            if key in self.data:
                boundary |= self.data[key][row].reshape(self.num_envs, -1).bool().any(-1)

    @torch.no_grad()
    def add(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Store the original transition and return its accepted/finite mask.

        Callers can update normalizers using this returned input-device mask
        after add, so invalid data never reaches running statistics.
        """
        if self._RESERVED.intersection(batch):
            raise ValueError("Replay data uses a reserved sampling metadata field")
        valid_input = self.finite_mask(batch)
        if valid_input.shape[0] != self.num_envs:
            raise ValueError(f"Replay requires {self.num_envs} environment rows")
        if not self.data:
            self.data = {
                key: torch.empty((self.capacity_steps, *value.shape), dtype=value.dtype, device=self.device)
                for key, value in batch.items()
            }
        elif self.data.keys() != batch.keys():
            raise ValueError("Replay fields must stay identical after allocation")
        # Validate every field before any ring mutation.
        for key, value in batch.items():
            if self.data[key].shape[1:] != value.shape or self.data[key].dtype != value.dtype:
                raise ValueError(f"Replay field {key!r} changed shape or dtype")
        for key, value in batch.items():
            self.data[key][self.ptr].copy_(value.detach())
        valid = valid_input.to(self.device)
        self._index_row(self.ptr, valid)
        accepted_count = int(valid.sum())
        self.invalid_transitions += self.num_envs - accepted_count
        self._all_valid = self._all_valid and accepted_count == self.num_envs
        self._cut_rows[self.ptr].zero_()
        self._timestamps[self.ptr] = self.total_steps
        self.ptr = (self.ptr + 1) % self.capacity_steps
        self.size = min(self.size + 1, self.capacity_steps)
        self.total_steps += 1
        self._prefix_cache.clear()
        return valid_input

    @torch.no_grad()
    def mark_boundary(self) -> None:
        """Cut the replay tail before collecting from a newly reset simulator."""
        if self.size:
            self._cut_rows[(self.ptr - 1) % self.capacity_steps].fill_(True)

    def _row(self, age: torch.Tensor) -> torch.Tensor:
        return (self.ptr - self.size + age) % self.capacity_steps

    def _prefix(self, candidate_steps: int, split: str, kind: str) -> _Prefix:
        key = (candidate_steps, split, kind)
        cached = self._prefix_cache.get(key)
        if cached is not None:
            return cached
        if split not in self._SPLIT_GROUPS:
            raise ValueError("split must be train, validation or all")
        split_groups = self._SPLIT_GROUPS[split]
        counts_offsets = {"train": (self.train_env_count, 0),
                          "validation": (self.validation_env_count, self.train_env_count),
                          "all": (self.num_envs, 0)}
        # For a proved-valid replay, every selected row has the same number
        # of train/validation/all environments. Preserve randint's exact high,
        # shape and call order, but avoid a prefix readback and binary search.
        if self._all_valid and kind == "all":
            count, offset = counts_offsets[split]
            groups = torch.tensor(split_groups, dtype=torch.long, device=self.device)
            result = _Prefix(torch.empty(0, dtype=torch.long, device=self.device),
                groups, len(split_groups), candidate_steps*count, count, offset)
            self._prefix_cache[key] = result
            self.index_cache_builds += 1
            return result
        already_built = (candidate_steps, split, "all") in self._prefix_cache
        rows = self._row(torch.arange(candidate_steps, device=self.device))
        row_counts = self._group_counts[rows]
        # Build all three class tables together, with a single small host read
        # for CUDA replay. Later model/discriminator/actor draws reuse them.
        tables = []
        for table_kind in ("all", "recovery", "normal"):
            ids = tuple(g for g in split_groups if table_kind == "all"
                        or (g % 2 == 1) == (table_kind == "recovery"))
            groups = torch.tensor(ids, dtype=torch.long, device=self.device)
            cumulative = row_counts[:, groups].reshape(-1).cumsum(0)
            tables.append((table_kind, groups, cumulative))
        totals = torch.stack([cumulative[-1] if cumulative.numel()
                              else self._group_counts.new_zeros(())
                              for _, _, cumulative in tables]).tolist()
        for (table_kind, groups, cumulative), total in zip(tables, totals):
            count, offset = counts_offsets[split] if self._all_valid and table_kind == "all" else (0, 0)
            self._prefix_cache[(candidate_steps, split, table_kind)] = _Prefix(
                cumulative, groups, len(groups), int(total), count, offset)
        # A lazily completed stratified table belongs to the existing index
        # generation; callers still observe one cached build per split/window.
        self.index_cache_builds += int(not already_built)
        return self._prefix_cache[key]

    def num_valid(self, split: str = "train") -> int:
        """Number of usable transitions in a split; excludes quarantined data."""
        if split not in self._SPLIT_GROUPS:
            raise ValueError("split must be train, validation or all")
        return self._prefix(self.size, split, "all").total if self.size else 0

    def _draw(self, prefix: _Prefix, count: int) -> tuple[torch.Tensor, torch.Tensor]:
        draws = torch.randint(prefix.total, (count,), device=self.device)
        if prefix.uniform_row_count:
            age = draws // prefix.uniform_row_count
            row = self._row(age)
            offset = draws % prefix.uniform_row_count + prefix.uniform_row_offset
            # Keep the original category-sorted pool, including argsort's
            # tie ordering. Replacing it by raw environment IDs would change
            # the same-seed sample sequence even for entirely valid data.
            return age, self._env_pool[row, offset].long()
        slot = torch.searchsorted(prefix.cumulative, draws, right=True)
        age = slot // prefix.group_count
        group = prefix.groups[slot % prefix.group_count]
        before = torch.where(slot > 0, prefix.cumulative[(slot - 1).clamp_min(0)], 0)
        row = self._row(age)
        offset = self._group_offsets[row, group] + draws - before
        env = self._env_pool[row, offset].long()
        return age, env

    def _sample_indices(self, batch_size: int, candidate_steps: int,
                        recovery_fraction: float, split: str = "train") -> tuple[torch.Tensor, torch.Tensor]:
        if batch_size < 1:
            raise ValueError("Replay batch_size must be positive")
        if candidate_steps < 1:
            raise ValueError("Replay has no valid sampling candidates")
        if not 0 <= recovery_fraction <= 1:
            raise ValueError("recovery_fraction must lie in [0, 1]")
        all_samples = self._prefix(candidate_steps, split, "all")
        if not all_samples.total:
            raise ValueError(f"No finite replay transitions in split {split!r}")
        if recovery_fraction == 0:
            return self._draw(all_samples, batch_size)
        recovery = self._prefix(candidate_steps, split, "recovery")
        normal = self._prefix(candidate_steps, split, "normal")
        if not recovery.total or not normal.total:
            return self._draw(all_samples, batch_size)
        nr = round(batch_size * recovery_fraction)
        recovery_age, recovery_env = self._draw(recovery, nr)
        normal_age, normal_env = self._draw(normal, batch_size - nr)
        order = torch.randperm(batch_size, device=self.device)
        return (torch.cat((recovery_age, normal_age))[order],
                torch.cat((recovery_env, normal_env))[order])

    def _fields(self, fields: Sequence[str] | None) -> tuple[str, ...]:
        selected = tuple(self.data) if fields is None else tuple(fields)
        if len(set(selected)) != len(selected):
            raise ValueError("Requested replay fields must be unique")
        missing = set(selected).difference(self.data)
        if missing:
            raise KeyError(f"Unknown replay fields: {sorted(missing)}")
        return selected

    def sample_starts(self, batch_size: int, recovery_fraction: float = .5,
                      fields: Sequence[str] | None = None,
                      split: str = "train") -> dict[str, torch.Tensor]:
        ages, env = self._sample_indices(batch_size, self.size, recovery_fraction, split)
        rows = self._row(ages)
        result = {key: self.data[key][rows, env] for key in self._fields(fields)}
        result["time_index"] = self._timestamps[rows]
        result["env_index"] = env
        return result

    def sample_transitions(self, batch_size: int, recovery_fraction: float = .5,
                           fields: Sequence[str] | None = None,
                           split: str = "train") -> dict[str, torch.Tensor]:
        return self.sample_starts(batch_size, recovery_fraction, fields, split)

    def sample_windows(self, batch_size: int, length: int, recovery_fraction: float = .5,
                       fields: Sequence[str] | None = None,
                       split: str = "train") -> dict[str, torch.Tensor]:
        if length < 1 or length > self.size:
            raise ValueError(f"Cannot sample {length}-step windows from {self.size} replay steps")
        ages, env = self._sample_indices(batch_size, self.size - length + 1, recovery_fraction, split)
        window_ages = ages[:, None] + torch.arange(length, device=self.device)[None, :]
        rows = self._row(window_ages)
        envs = env[:, None].expand_as(rows)
        valid = self._valid[rows, envs]
        boundary = self._boundaries[rows, envs] | self._cut_rows[rows, envs]
        mask = torch.cumprod(torch.cat((
            torch.ones((batch_size, 1), device=self.device),
            (~boundary[:, :-1]).float(),
        ), 1), 1) * valid.float().cumprod(1)
        result = {}
        for key in self._fields(fields):
            values = self.data[key][rows, envs]
            # Initial rows are guaranteed valid by their pool membership.
            # Preserve every finite physical label, including terminal labels.
            if self._all_valid or length == 1:
                result[key] = values
            else:
                keep = valid.reshape(batch_size, length, *((1,) * (values.ndim - 2)))
                result[key] = torch.where(keep, values, values[:, :1].expand_as(values))
        result["mask"] = mask
        result["time_index"] = self._timestamps[rows]
        result["env_index"] = envs
        return result

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": 2, "capacity_steps": self.capacity_steps, "num_envs": self.num_envs,
            "validation_env_fraction": self.validation_env_fraction,
            "ptr": self.ptr, "size": self.size, "total_steps": self.total_steps,
            "invalid_transitions": self.invalid_transitions,
            "data": self.data, "valid": self._valid, "recovery": self._recovery,
            "cut_rows": self._cut_rows, "timestamps": self._timestamps,
        }

    @torch.no_grad()
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if state_dict["capacity_steps"] != self.capacity_steps or state_dict["num_envs"] != self.num_envs:
            raise ValueError("Replay checkpoint capacity/environment dimensions do not match")
        if ("validation_env_fraction" in state_dict
                and state_dict["validation_env_fraction"] != self.validation_env_fraction):
            raise ValueError("Replay checkpoint validation environment split differs")
        ptr, size = int(state_dict["ptr"]), int(state_dict["size"])
        total_steps = int(state_dict["total_steps"])
        if not 0 <= ptr < self.capacity_steps or not 0 <= size <= self.capacity_steps or total_steps < size:
            raise ValueError("Replay checkpoint has invalid ring indices")
        data = {key: value.to(self.device).clone() for key, value in state_dict["data"].items()}
        if size and "state" not in data:
            raise ValueError("Replay checkpoint contains no physical state")
        for key, value in data.items():
            if value.shape[:2] != (self.capacity_steps, self.num_envs):
                raise ValueError(f"Replay checkpoint field {key!r} has invalid dimensions")
        self.ptr, self.size, self.total_steps, self.data = ptr, size, total_steps, data
        self._timestamps.copy_(state_dict["timestamps"])
        self._valid.zero_()
        self._group_counts.zero_()
        self._cut_rows.zero_()
        if "cut_rows" in state_dict:
            self._cut_rows.copy_(state_dict["cut_rows"])
        quarantined = 0
        for age in range(size):
            row = (ptr - size + age) % self.capacity_steps
            batch = {key: value[row] for key, value in data.items()}
            valid = self.finite_mask(batch)
            if "valid" in state_dict:
                valid &= state_dict["valid"][row].to(self.device)
            self._index_row(row, valid)
            quarantined += self.num_envs - int(valid.sum())
        self.invalid_transitions = int(state_dict.get("invalid_transitions", quarantined))
        self._all_valid = quarantined == 0
        self._prefix_cache.clear()
        self.mark_boundary()
