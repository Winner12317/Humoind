"""Hiking's fresh 24-step AMP rollout and five complete discriminator epochs.

A cursor groups fresh vector steps into the original Hiking 24-step AMP rollouts,
including all environments independently of the WM validation split. Each
finite transition is visited once per epoch, rather than sampled repeatedly
from the long-lived world-model replay.

Single-frame AMP features are extracted in independent no-grad chunks. Policy
histories are flattened once per rollout. Expert histories are read from the
environment-specific playback timeline (static legal windows for legacy runs). A logical minibatch keeps
Hiking's optimizer-step size; its input-gradient penalty uses weighted
microbatches and optional fixed-shape CUDA replay.
"""
from __future__ import annotations

from dataclasses import dataclass
from contextlib import nullcontext

import torch

from .hiking_amp import FRAME_DIM, HISTORY_LENGTH, flatten_history
from .amp_execution import prediction_and_penalty


ROLLOUT_STEPS = 24
LEARNING_EPOCHS = 5
MINI_BATCHES = 4


@dataclass(frozen=True)
class HikingAmpRollout:
    start: int
    end: int
    time: torch.Tensor
    env: torch.Tensor
    invalid_windows: int

    @property
    def size(self):
        return self.time.numel()

    def minibatches(self, *, generator=None):
        """One shuffle, reused for all five epochs, as in AmpStorage.

        Production rollout sizes divide by four. Balanced tail partitions also
        preserve every sample for interrupted/tiny final rollouts; the source
        storage's floor division would discard a non-divisible remainder.
        """
        order = torch.randperm(self.size, device=self.time.device, generator=generator)
        pieces = tuple(part for part in order.tensor_split(MINI_BATCHES) if part.numel())
        for _ in range(LEARNING_EPOCHS):
            yield from pieces


class HikingAmpBudget:
    """A fresh replay cursor; never persist it without the corresponding replay.

    Perceptive checkpoints recreate the simulator and replay on resume, so a
    new budget starts at that new replay's time zero. Warmup transitions count
    toward the first rollout. Historical v2 runs also flush the final short tail.
    """
    def __init__(self):
        self.next_time = 0
        self.windows = 0
        self.invalid_windows = 0
        self.rollouts = 0

    def claim(self, replay, *, flush=False):
        """Return complete fresh rollouts, plus the last tail when requested."""
        oldest = replay.total_steps-replay.size
        if self.next_time < oldest:
            raise RuntimeError("Untrained Hiking AMP transitions were overwritten by replay")
        if self.next_time > replay.total_steps:
            raise RuntimeError("Hiking AMP cursor belongs to a different replay")
        result = []
        while replay.total_steps-self.next_time >= ROLLOUT_STEPS or (
                flush and self.next_time < replay.total_steps):
            start = self.next_time
            end = min(start+ROLLOUT_STEPS, replay.total_steps)
            times = torch.arange(start, end, device=replay.device)
            rows = times.remainder(replay.capacity_steps)
            if not torch.equal(replay._timestamps[rows], times):
                raise RuntimeError("Hiking AMP rollout replay timestamps are stale")
            valid = replay._valid[rows]
            flat = valid.reshape(-1).nonzero(as_tuple=False).flatten()
            time = start+flat.div(replay.num_envs, rounding_mode="floor")
            env = flat.remainder(replay.num_envs)
            invalid = valid.numel()-flat.numel()
            result.append(HikingAmpRollout(start, end, time, env, invalid))
            self.next_time = end
            self.windows += flat.numel()
            self.invalid_windows += invalid
            self.rollouts += 1
        return result


class HikingAmpFeatureCache:
    """Single-frame features, with the existing episode-safe history contract.

    A successor AMP history contains nine observed states and its true physical
    successor. Cache each needed frame once, then gather histories for every
    epoch. At 4096 environments a complete rollout uses about 59 MiB, rather
    than copying five epochs of full 670-dimensional windows or depth images.
    """
    @torch.no_grad()
    def __init__(self, adapter, replay, rollout, device, microbatch_size, *,
                 feature_chunk_size=None, feature_fn=None, post_step_observations=False):
        if not getattr(replay, "_hiking_amp_history", False):
            raise ValueError("Hiking AMP requires configured replay episode histories")
        if microbatch_size < 1:
            raise ValueError("AMP microbatch size must be positive")
        self.start, self.end = rollout.start, rollout.end
        self.begin = max(0, self.start-(HISTORY_LENGTH-2))
        self.device = torch.device(device)
        self.num_envs = replay.num_envs
        feature_chunk_size = feature_chunk_size or microbatch_size
        if feature_chunk_size < 1:
            raise ValueError("AMP feature chunk size must be positive")
        feature_fn = feature_fn or adapter.amp_features
        self.episode_starts = replay._amp_episode_starts[
            torch.arange(self.start, self.end, device=replay.device).remainder(replay.capacity_steps)
        ].to(self.device)
        oldest = replay.total_steps-replay.size
        if self.begin < max(0, oldest-(HISTORY_LENGTH-1)):
            raise RuntimeError("Hiking AMP predecessor states were overwritten")

        def encode(begin, end, *, successor):
            count = (end-begin)*replay.num_envs
            output = None
            for offset in range(0, count, feature_chunk_size):
                flat = torch.arange(offset, min(offset+feature_chunk_size, count), device=replay.device)
                time = begin+flat.div(replay.num_envs, rounding_mode="floor")
                env = flat.remainder(replay.num_envs)
                states = replay.data["next_state" if successor else "state"][
                    time.remainder(replay.capacity_steps), env]
                if not successor:
                    historical = replay._amp_overflow_states[
                        time.remainder(HISTORY_LENGTH-1), env]
                    states = torch.where((time < oldest)[:, None], historical, states)
                # Invalid rows never become examples. Keep their unused cache
                # entries harmless while retaining their explicit drop count.
                features = feature_fn(torch.nan_to_num(states).to(self.device))
                if features.shape != (len(flat), FRAME_DIM):
                    raise ValueError("Hiking AMP cache requires 67-dimensional frame features")
                if output is None:
                    output = features.new_empty(count, FRAME_DIM)
                output[offset:offset+len(flat)].copy_(features)
            return output.reshape(end-begin, replay.num_envs, FRAME_DIM)

        self.observed = encode(self.begin, self.end, successor=False)
        self.post_step_resets = None
        if post_step_observations:
            # Wasabi reads observations returned by env.step, which include
            # interval events and resets. Each is the following collection
            # state, except at the live replay head. Encode that one row once.
            last = (adapter.state() if self.end == replay.total_steps else
                    replay.data["state"][self.end % replay.capacity_steps])
            last = torch.cat([feature_fn(torch.nan_to_num(part).to(self.device))
                              for part in last.split(feature_chunk_size)])
            self.successors = torch.cat((self.observed[self.start+1-self.begin:], last[None]))
            rows = torch.arange(self.start, self.end, device=replay.device) % replay.capacity_steps
            self.post_step_resets = (replay.data["terminated"][rows] | replay.data["timeout"][rows]).to(self.device)
        else:
            self.successors = encode(self.start, self.end, successor=True)

    def policy(self, time, env):
        time, env = time.to(self.device), env.to(self.device)
        starts = self.episode_starts[time-self.start, env]
        history_time = torch.maximum(time[:, None]+torch.arange(
            2-HISTORY_LENGTH, 1, device=self.device), starts[:, None])
        observed = self.observed[history_time-self.begin, env[:, None]]
        successor = self.successors[time-self.start, env]
        if self.post_step_resets is not None:
            reset = self.post_step_resets[time-self.start, env]
            observed = torch.where(reset[:, None, None], successor[:, None], observed)
        return flatten_history(torch.cat((observed, successor[:, None]), 1))


class HikingAmpPolicyCache:
    """Build each fresh flattened policy window once, then reuse five times."""
    @torch.no_grad()
    def __init__(self, features, rollout, chunk_size=4096):
        self.rollout = rollout
        self.values = features.observed.new_empty(rollout.size, HISTORY_LENGTH*FRAME_DIM)
        for begin in range(0, rollout.size, chunk_size):
            end = min(begin+chunk_size, rollout.size)
            self.values[begin:end].copy_(features.policy(rollout.time[begin:end],rollout.env[begin:end]))

    def select(self, selection):
        return self.values[selection.to(self.values.device)]



class HikingEpisodeReference:
    """One original or mirrored reference timeline per physical environment.

    The reference sensor samples a continuous start uniformly in [0, .9)
    of clip duration and a Bernoulli mirror on reset. Its current frame uses
    round((start_seconds + elapsed_seconds) * fps). Observation history repeats
    the first frame after an environment reset.
    """
    def __init__(self, experts, num_envs, dt):
        self.frames = experts.groups[0]
        self.length = experts.official_reference_frames
        self.dt = dt
        self.fps = 1. / dt
        self.device = self.frames.device
        self.num_envs = num_envs
        self.start, self.offset = self._draw((num_envs,))
        self.age = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        frame = self._frame(self.age)
        self.history = frame[:, None].expand(-1, HISTORY_LENGTH).clone()
        self.next_time = 0

    def _draw(self, shape):
        start = torch.rand(shape, device=self.device) * .9 * self.length / self.fps
        offset = torch.randint(2, shape, device=self.device) * self.length
        return start, offset

    def _frame(self, age):
        index = ((self.start + age * self.dt) * self.fps).round().long()
        return index.clamp(0, self.length - 1) + self.offset

    @torch.no_grad()
    def advance(self, rollout, replay):
        if rollout.start != self.next_time:
            raise ValueError("Hiking reference observations require chronological physical rollouts")
        rows = torch.arange(rollout.start, rollout.end, device=replay.device).remainder(replay.capacity_steps)
        resets = (replay.data["terminated"][rows] | replay.data["timeout"][rows]).to(self.device)
        starts, offsets = self._draw(resets.shape)
        frame_indices = []
        for step, reset in enumerate(resets.unbind()):
            self.age = self.age + 1
            # Hiking's dataset_exhausted checks the next aiming frame and
            # resets only the reference sensor when its data runs out.
            next_index = ((self.start + (self.age+1)*self.dt) * self.fps).round()
            restart = reset | (next_index >= self.length)
            self.start = torch.where(restart, starts[step], self.start)
            self.offset = torch.where(restart, offsets[step], self.offset)
            self.age = torch.where(restart, 0, self.age)
            frame = self._frame(self.age)[:, None]
            history = torch.cat((self.history[:, 1:], frame), dim=1)
            self.history = torch.where(reset[:, None], frame, history)
            frame_indices.append(self.history)
        self.next_time = rollout.end
        self.indices = torch.stack(frame_indices)[
            (rollout.time - rollout.start).to(self.device), rollout.env.to(self.device)]

    def select(self, indices):
        return flatten_history(self.frames[self.indices[indices.to(self.device)]])


class HikingAmpReferenceCache:
    """Static legal expert windows, indexed by the unchanged sampled end IDs.

    The existing dataset has 40,328 legal windows. Caching those once uses
    about 103 MiB, regardless of the number of environments or rollout draws.
    Unused clip-prefix rows are never sampled and need no valid window.
    """
    @torch.no_grad()
    def __init__(self, experts, device, chunk_size=4096):
        self.experts = experts
        frames = experts.groups[0]
        self.values = frames.new_empty(len(frames), HISTORY_LENGTH*FRAME_DIM, device=device)
        for ends in experts._hiking_history_ends.split(chunk_size):
            self.values[ends.to(device)] = reference_features(experts, ends, device)

    def select(self, ends):
        return self.values[ends.to(self.values.device)]


@torch.no_grad()
def sample_reference_ends(experts, size):
    """Keep one expert pairing per real window throughout all five epochs."""
    ends = experts._hiking_history_ends
    return ends[torch.randint(len(ends), (size,), device=ends.device)]


@torch.no_grad()
def reference_features(experts, ends, device):
    frames = experts.groups[0]
    times = ends.to(frames.device)[:, None]+torch.arange(
        1-HISTORY_LENGTH, 1, device=frames.device)
    return flatten_history(frames[times]).to(device)


def discriminator_step(discriminator, optimizer, pairs, size, *, gradient_penalty,
                       trunk_weight_decay, head_weight_decay, execution=None):
    """One exact logical Hiking optimizer step, using weighted microbatches.

    The discriminator is Hiking's independent-example ReLU MLP (no batch
    normalization or dropout). Therefore sums of per-example prediction and
    input-gradient losses give the same gradients as a full logical batch,
    up to floating-point reduction order. Explicit L2 terms are added once.
    """
    if size < 1:
        raise ValueError("AMP logical minibatch size must be positive")
    with execution.context() if execution is not None else nullcontext():
        optimizer.zero_grad(set_to_none=True)
        if execution is not None:
            execution.begin_step()
        prediction_total = penalty_total = None
        seen = 0
        for policy, expert in pairs:
            count = len(policy)
            if count < 1 or len(expert) != count:
                raise ValueError("AMP microbatch policy/reference sizes differ")
            result = execution.accumulate(policy,expert,size) if execution is not None else None
            if result is None:
                examples = torch.cat((policy, expert), 0).detach().requires_grad_(True)
                prediction, penalty = prediction_and_penalty(discriminator, examples, count, gradient_penalty)
                ((prediction+penalty)*(count/size)).backward()
            else:
                prediction, penalty = result
            prediction = prediction.detach()*(count/size)
            penalty = penalty.detach()*(count/size)
            prediction_total = prediction if prediction_total is None else prediction_total+prediction
            penalty_total = penalty if penalty_total is None else penalty_total+penalty
            seen += count
        if seen != size:
            raise ValueError("AMP microbatches do not cover the logical minibatch")
        regularization = trunk_weight_decay*sum(p.square().sum() for p in discriminator.parameters())
        regularization = regularization+head_weight_decay*discriminator.amp_linear.weight.square().sum()
        regularization.backward()
        norm = torch.linalg.vector_norm(torch.stack(tuple(
            p.grad.detach().norm() for p in discriminator.parameters() if p.grad is not None)))
        optimizer.step()
        return torch.stack((prediction_total+penalty_total+regularization.detach(), penalty_total, norm))
