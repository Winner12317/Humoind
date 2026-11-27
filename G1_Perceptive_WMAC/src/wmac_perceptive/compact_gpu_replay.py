"""Explicit uint8 single-frame cache for Instinct visual-latent replay.

The protocol requires delay=1. At transition t, the delayed current stack
reads frames t-36..t-1 and its true successor reads t-35..t (default stride=5).
Only frame t, the last processed frame of the successor stack, is stored. Episode starts
clamp historical indices so reset histories repeat the first new observation.
No learned latent is cached. Native replay sampling, strata, quarantine and
window boundaries retain the existing ReplayBuffer behavior.
"""

from __future__ import annotations

from dataclasses import asdict

import torch

from src.wmac.replay import ReplayBuffer
from .latent_perception import pack_processed_actor_input, processed_stack_shape
from .latent_replay import LatentPerceptiveReplay
from .perception import PerceptionSpec


class NoCompleteReferenceWindow(RuntimeError):
    """Replay has enough rows, but every training sequence crosses a break."""


class CompressedLatentReplay(LatentPerceptiveReplay):
    storage_protocol = "frame_uint8_v1"

    def __init__(self, capacity_steps, num_envs, device="cuda:0", validation_env_fraction=0.,
                 *, spec=PerceptionSpec()):
        if spec.delay != 1:
            raise ValueError("Shared-frame uint8 replay requires the explicit one-frame delay")
        super().__init__(capacity_steps, num_envs, device, validation_env_fraction, spec=spec)
        self.cache_steps = capacity_steps + spec.ring_length
        _, height, width = processed_stack_shape(spec)
        self.frame_cache = torch.empty(self.cache_steps, num_envs, spec.visual_channels, height, width,
                                       dtype=torch.uint8, device=self.device)
        self._episode_starts = torch.empty(capacity_steps, num_envs, dtype=torch.long, device=self.device)
        self._episode_start = torch.full((num_envs,), 1-spec.ring_length, dtype=torch.long, device=self.device)
        self._pending_reset = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self._bootstrapped = False
        self._reference_window_index = None
        self._incoming_visual_valid = None
        self._offsets = torch.tensor(tuple(index-spec.ring_length+1 for index in spec.selected_indices),
                                     dtype=torch.long, device=self.device)

    @staticmethod
    def estimated_frame_cache_bytes(capacity_steps, num_envs, spec=PerceptionSpec()):
        _, height, width = processed_stack_shape(spec)
        return (capacity_steps + spec.ring_length) * num_envs * spec.visual_channels * height * width

    @staticmethod
    def _visual_valid(image, include_valid=True):
        # Depth is normalized and validity is a binary channel. Non-finite or
        # unrepresentable inputs are quarantined, never accepted after clipping.
        depth = image[..., 0::2, :, :] if include_valid else image
        prefix = image.shape[0]
        # Gaussian accumulation can put an all-one image a few ULPs above one.
        # Limit this solely to depth roundoff, and cap low-precision tolerance
        # so a real out-of-range observation (e.g. 1.001) is still quarantined.
        tolerance = min(8 * torch.finfo(depth.dtype).eps, 1e-6)
        accepted = (torch.isfinite(image).reshape(prefix, -1).all(-1)
                    & ((depth >= -tolerance) & (depth <= 1 + tolerance)).reshape(prefix, -1).all(-1))
        if include_valid:
            valid = image[..., 1::2, :, :]
            accepted = accepted & ((valid == 0) | (valid == 1)).reshape(prefix, -1).all(-1)
        return accepted

    @staticmethod
    def _quantize(frame):
        # Invalid input rows retain their quarantine flag; benign cache bytes
        # keep padding/gather finite without inventing valid physical samples.
        safe = torch.nan_to_num(frame.detach(), nan=1., posinf=1., neginf=0.).clamp(0, 1)
        return (safe * 255).round().to(torch.uint8)

    @torch.no_grad()
    def bootstrap_processed_history(self, history):
        """Initialize from reset-safe processed history [N,L,C,Hc,Wc].

        The newest history frame is physical frame t=0. Initialization is
        explicit and allowed only before the first add. Use bootstrap_reset
        when every environment has just reset and all L frames are identical.
        """
        _, height, width = processed_stack_shape(self.spec)
        channels = self.spec.visual_channels
        expected = (self.num_envs, self.spec.ring_length, channels, height, width)
        if self.total_steps or self._bootstrapped:
            raise ValueError("Frame-cache bootstrap is only valid before the first transition")
        if (not isinstance(history, torch.Tensor) or tuple(history.shape) != expected
                or not history.is_floating_point()):
            raise ValueError(f"Processed bootstrap history must have shape {expected}")
        images = history.reshape(self.num_envs, channels*self.spec.ring_length, height, width)
        if not bool(self._visual_valid(images, self.spec.include_valid).all()):
            raise ValueError("Bootstrap history must contain normalized depth and, when present, binary validity")
        frames = self._quantize(history).to(self.device).transpose(0, 1)
        times = torch.arange(1-self.spec.ring_length, 1, device=self.device)
        self.frame_cache[times.remainder(self.cache_steps)] = frames
        self._bootstrapped = True

    bootstrap = bootstrap_processed_history

    @torch.no_grad()
    def bootstrap_reset(self, frame):
        _, height, width = processed_stack_shape(self.spec)
        if tuple(frame.shape) != (self.num_envs, self.spec.visual_channels, height, width):
            raise ValueError("Reset frame must be [N,visual_channels,cropped_height,cropped_width]")
        self.bootstrap_processed_history(frame[:, None].expand(-1, self.spec.ring_length, -1, -1, -1))

    def finite_mask(self, batch):
        valid = ReplayBuffer.finite_mask(batch)
        if self._incoming_visual_valid is not None:
            valid = valid & self._incoming_visual_valid.to(valid.device)
        return valid

    @torch.no_grad()
    def add(self, batch, *, processed_history=None):
        self._validate_perceptive_shapes(batch)
        if processed_history is not None:
            self.bootstrap_processed_history(processed_history)
        if not self._bootstrapped:
            raise ValueError("Shared-frame replay requires an explicit processed-history bootstrap")
        visual_valid = (self._visual_valid(batch["depth_stack"], self.spec.include_valid)
                       & self._visual_valid(batch["next_depth_stack"], self.spec.include_valid))
        latest = self._quantize(batch["next_depth_stack"][:, -self.spec.visual_channels:]).to(self.device)
        stored = {name: value for name, value in batch.items() if name not in self._DEPTH_FIELDS}
        stored["actor_history"] = stored["actor_history"].float()
        stored["relative_height"] = stored["relative_height"].float()
        row, time = self.ptr, self.total_steps
        starts = torch.where(self._pending_reset, time, self._episode_start)
        self._incoming_visual_valid = visual_valid
        from .hiking_amp import before_replay_add, after_replay_add
        before_replay_add(self)
        try:
            accepted = ReplayBuffer.add(self, stored)
        finally:
            self._incoming_visual_valid = None
        after_replay_add(self, stored, accepted)
        self.frame_cache[time % self.cache_steps].copy_(latest)
        self._episode_starts[row].copy_(starts)
        self._episode_start.copy_(starts)
        self._pending_reset.zero_()
        for name in ("terminated", "timeout"):
            if name in batch:
                self._pending_reset |= batch[name].to(self.device).reshape(self.num_envs, -1).bool().any(-1)
        return accepted

    @torch.no_grad()
    def mark_boundary(self):
        ReplayBuffer.mark_boundary(self)
        self._pending_reset.fill_(True)
        if getattr(self, "_hiking_amp_history", False):
            self._amp_pending_reset.fill_(True)

    def _fields(self, fields):
        available = (*self.data, *self._DEPTH_FIELDS)
        selected = available if fields is None else tuple(fields)
        if len(set(selected)) != len(selected):
            raise ValueError("Requested replay fields must be unique")
        missing = set(selected).difference(available)
        if missing:
            raise KeyError(f"Unknown replay fields: {sorted(missing)}")
        return selected

    def _depth(self, rows, env, successor=False):
        time = self._timestamps[rows]
        starts = self._episode_starts[rows, env]
        frames = torch.maximum(time[..., None] + self._offsets + int(successor), starts[..., None])
        values = self.frame_cache[frames.remainder(self.cache_steps), env[..., None]].float() / 255.
        return values.flatten(-4, -3)

    def _gather_fields(self, rows, env, fields):
        result = {}
        for name in fields:
            if name in self._DEPTH_FIELDS:
                result[name] = self._depth(rows, env, name == "next_depth_stack")
            else:
                result[name] = self.data[name][rows, env]
        return result

    def sample_starts(self, batch_size, recovery_fraction=0., fields=None, split="train"):
        selected = self._fields(fields)
        pack = "actor_history" in selected
        requested = tuple(dict.fromkeys((*selected, "depth_stack"))) if pack else selected
        ages, env = self._sample_indices(batch_size, self.size, recovery_fraction, split)
        rows = self._row(ages)
        result = self._gather_fields(rows, env, requested)
        if pack:
            result["actor_history"] = pack_processed_actor_input(result["actor_history"].float(), result["depth_stack"], self.spec)
            if "depth_stack" not in selected:
                result.pop("depth_stack")
        result["time_index"], result["env_index"] = self._timestamps[rows], env
        from .hiking_amp import add_sample_history
        return add_sample_history(self, result)

    def sample_windows(self, batch_size, length, recovery_fraction=0., fields=None, split="train"):
        return self._sample_windows(batch_size,length,recovery_fraction,fields,split)

    def sample_model_windows(self,batch_size,length,recovery_fraction=0.,fields=None,split="train"):
        """Same draw/masks, only the initial current stack, original cache codes.

        Online targets are still encoded afresh after each optimizer update.
        Keeping cache bytes until encoding avoids float copies in a WM macro.
        The ordinary replay API and its complete window shapes stay unchanged.
        """
        return self._sample_windows(batch_size,length,recovery_fraction,fields,split,model=True)

    def sample_reference_model_windows(self, batch_size, length, recovery_fraction=0., fields=None, split="train"):
        """Sample eight real state frames for seven supervised transitions.

        SuperMimic samples the environment uniformly among those with a complete
        episode window, then the start uniformly within that environment. A new
        navigation command does not reset physics. Invalid transitions and
        simulator restarts still cut the window.
        """
        if batch_size < 1 or not 0 < length < self.size or split != "train" or recovery_fraction != 0.:
            raise ValueError("Reference WM needs a full train window of length + 1 state frames")
        key = (self.total_steps, length)
        if self._reference_window_index is None or self._reference_window_index[0] != key:
            self._reference_window_index = None
            chronological = self._row(torch.arange(self.size, device=self.device))
            invalid = (~self._valid[chronological] | self._cut_rows[chronological]
                       | self.data["terminated"][chronological] | self.data["timeout"][chronological])
            # Ages [start, start + length) contain the seven actions. The
            # eighth pre-action state must already have been collected.
            c = torch.nn.functional.pad(invalid[:-1].T.cumsum(1, dtype=torch.int32), (1, 0))
            valid = (c[:, length:] == c[:, :-length]) & self._train_env[:, None]
            counts = valid.cumsum(1, dtype=torch.int32)
            total = counts[:, -1].contiguous()
            if not (total > 0).any():
                # Only the first size-1 rows can supply supervised transitions.
                # Report overlapping causes on training environments, on failure only.
                rows = chronological[:-1]
                causes = {
                    "invalid": ~self._valid[rows],
                    "terminated": self.data["terminated"][rows],
                    "timeout": self.data["timeout"][rows],
                    "explicit_cut": self._cut_rows[rows],
                }
                numbers = torch.stack([(mask & self._train_env).sum() for mask in causes.values()]).tolist()
                details = ", ".join(f"{name}={n}" for name, n in zip(causes, numbers))
                raise NoCompleteReferenceWindow(
                    f"No complete in-episode state window available for the reference WM "
                    f"(need {length} consecutive valid transitions; {self.size} replay rows, "
                    f"{self.train_env_count} training environments; {details})"
                )
            self._reference_window_index = (key, counts, total)
        _, counts, total = self._reference_window_index
        env = torch.multinomial((total > 0).float(), batch_size, replacement=True)
        rank = (torch.rand(batch_size, device=self.device) * total[env]).to(torch.int32)
        age = torch.searchsorted(counts[env], rank[:, None], right=True).squeeze(1)
        return self._sample_windows(batch_size, length, 0., fields, split, model=True,
                                    complete_indices=(age, env))

    def _model_depth(self,rows,env,successor,*,time=None,starts=None):
        cache=getattr(self,"frame_cache",None)
        if cache is None or cache.dtype!=torch.uint8:
            return self._depth(rows,env,successor)
        time=self._timestamps[rows] if time is None else time
        starts=self._episode_starts[rows,env] if starts is None else starts
        frames=torch.maximum(time[...,None]+self._offsets+int(successor),starts[...,None])
        return cache[frames.remainder(self.cache_steps),env[...,None]].flatten(-4,-3)

    def _sample_windows(self,batch_size,length,recovery_fraction,fields,split,*,model=False,
                        complete_indices=None):
        if length < 1 or length > self.size:
            raise ValueError(f"Cannot sample {length}-step windows from {self.size} replay steps")
        ages, env = (complete_indices if complete_indices is not None else
                     self._sample_indices(batch_size, self.size-length+1, recovery_fraction, split))
        window_ages = ages[:, None] + torch.arange(length, device=self.device)[None, :]
        rows = self._row(window_ages)
        envs = env[:, None].expand_as(rows)
        valid = self._valid[rows, envs]
        if complete_indices is not None:
            mask = torch.ones_like(valid, dtype=torch.float32)
        else:
            boundary = self._boundaries[rows, envs] | self._cut_rows[rows, envs]
            mask = torch.cumprod(torch.cat((torch.ones(batch_size, 1, device=self.device),
                (~boundary[:, :-1]).float()), 1), 1) * valid.float().cumprod(1)
        selected=self._fields(fields)
        time_index=self._timestamps[rows]
        if model:
            result=self._gather_fields(rows,envs,tuple(name for name in selected if name not in self._DEPTH_FIELDS))
            cache=getattr(self,"frame_cache",None)
            starts=(self._episode_starts[rows,envs] if cache is not None and cache.dtype==torch.uint8
                    and any(name in selected for name in self._DEPTH_FIELDS) else None)
            if "depth_stack" in selected:
                result["depth_stack"]=self._model_depth(rows[:,:1],envs[:,:1],False,
                    time=time_index[:,:1],starts=starts[:,:1] if starts is not None else None)
            if "next_depth_stack" in selected:
                result["next_depth_stack"]=self._model_depth(rows,envs,True,time=time_index,starts=starts)
            result={name:result[name] for name in selected}
        else:
            result = self._gather_fields(rows, envs, selected)
        if not self._all_valid and length != 1:
            for key, values in result.items():
                if model and key=="depth_stack":
                    # Its time dimension is one: both original where branches
                    # are the exact same initial stack, even after quarantine.
                    continue
                keep = valid.reshape(batch_size, length, *((1,) * (values.ndim-2)))
                result[key] = torch.where(keep, values, values[:, :1].expand_as(values))
        elif "previous_action" in result and hasattr(self,"_previous_action_exceptions"):
            # Deduplicated reconstruction can return a slice of context.
            # The old padding made window fields independent/contiguous even
            # when padding was an identity. Preserve that public contract;
            # model fitting does not request this small action-only field.
            result["previous_action"]=result["previous_action"].clone(memory_format=torch.contiguous_format)
        result["mask"], result["time_index"], result["env_index"] = mask, time_index, envs
        return result

    def state_dict(self):
        result = ReplayBuffer.state_dict(self)
        result.update(perceptive_format=self.storage_protocol, perception_spec=asdict(self.spec),
            frame_cache=self.frame_cache, episode_starts=self._episode_starts,
            episode_start=self._episode_start, pending_reset=self._pending_reset,
            bootstrapped=self._bootstrapped)
        return result

    @torch.no_grad()
    def load_state_dict(self, saved):
        if (saved.get("perceptive_format") != self.storage_protocol
                or {**{"depth_protocol":"hiking_v1"}, **saved.get("perception_spec",{})} != asdict(self.spec)):
            raise ValueError("Compressed frame-cache replay protocol/spec differs")
        required = {"frame_cache": self.frame_cache, "episode_starts": self._episode_starts,
                    "episode_start": self._episode_start, "pending_reset": self._pending_reset}
        for name, expected in required.items():
            value = saved.get(name)
            if not isinstance(value, torch.Tensor) or value.shape != expected.shape or value.dtype != expected.dtype:
                raise ValueError(f"Invalid compressed replay field {name}")
        if not isinstance(saved.get("bootstrapped"), bool) or (saved.get("size", 0) and not saved["bootstrapped"]):
            raise ValueError("Compressed replay bootstrap metadata missing")
        if set(saved.get("data", {})).intersection(self._DEPTH_FIELDS):
            raise ValueError("Compressed replay must not silently load full-stack storage")
        self._reference_window_index = None
        ReplayBuffer.load_state_dict(self, saved)
        for name, target in required.items():
            target.copy_(saved[name].to(self.device))
        self._bootstrapped = saved["bootstrapped"]
        if self.size:
            # Base restore cuts the tail because the simulator starts a new
            # episode. Its first new image must not borrow saved camera frames.
            self._pending_reset.fill_(True)
