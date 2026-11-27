"""Full captured Actor/Critic updates with the selected training schedule.

Host decisions stay in the runner. Old validated schedules call invalidate()
after optimizer-state rollback. The original schedule keeps a fixed horizon,
warms three full macros and shares its ordered graph resources with the WM.
"""
from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from src.wmac.networks import frozen_parameters
from src.wmac.routing import router_losses
from .latent_execution import WMExecution, graph_warmup_steps, safe_optimizer_step


def _snapshot(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    return {name: _snapshot(item) for name, item in value.items()}


class CapturedTensorUpdate(WMExecution):
    """Reuse fixed gradient ownership, device guard and pointer validation."""

    def __init__(self, kernel, optimizer, *, max_grad_norm=1., clip_value=0.,
                 mode="graph_full", warmup=3, compile_loss=False, compile_mode="default", guard_nonfinite=True,
                 graph_resources=None):
        holder = SimpleNamespace(opt_model=optimizer,
            cfg=SimpleNamespace(max_grad_norm=max_grad_norm, compile_mode=compile_mode))
        super().__init__(holder, mode=mode, warmup=warmup, _kernel=kernel,
                         compile_loss=compile_loss, clip_value=clip_value, guard_nonfinite=guard_nonfinite,
                         graph_resources=graph_resources)

    def step(self, batch):
        prepared = self.kernel.prepare(batch)
        with self._update_stream():
            loss, outputs = self._forward_backward(prepared)
            if self.mode == "graph_full" and self._graph is not None:
                norm, accepted = self._static_step_outputs
            else:
                norm = self._clip_gradients()
                accepted = safe_optimizer_step(loss, norm, self.optimizer, guard=self.guard_nonfinite)
            if self._graph is None:
                self.optimizer.zero_grad(set_to_none=True)
            self._calls += 1
            self.last_loss = loss.detach().clone()
            result = _snapshot(outputs), norm.detach().clone(), accepted.clone()
        return result

    __call__ = step


class _ActorLoss(nn.Module):
    metric_names = ()

    def __init__(self, runner):
        super().__init__()
        object.__setattr__(self, "runner", runner)
        self.models = nn.ModuleList((runner.actor, runner.world_model, runner.critic,
                                     runner.discriminator, runner.amp_norm))
        self.balance_coef = runner.cfg.moe_load_balance_coef
        self.z_coef = runner.cfg.moe_router_z_coef

    @staticmethod
    def prepare(batch, **unused):
        return batch

    def forward(self, batch):
        chunk = {name: value for name, value in batch.items() if name not in ("_noise", "_router_load")}
        rollout = self.runner.imagine(chunk, noises=batch["_noise"])
        task_loss = -rollout["objective"].mean()
        auxiliary = task_loss.new_zeros(())
        if self.balance_coef or self.z_coef:
            balance, z, _ = router_losses(self.runner.actor, chunk["actor_history"], batch["_router_load"])
            if self.balance_coef:
                auxiliary = auxiliary+self.balance_coef*balance
            if self.z_coef:
                auxiliary = auxiliary+self.z_coef*z
        loss = task_loss+auxiliary
        return loss, {**{name: value.detach() for name, value in rollout.items()},
                      "loss": loss.detach(), "task_loss": task_loss.detach(), "auxiliary_loss": auxiliary.detach()}


class ActorExecution:
    """One full-batch Actor update; host rollback remains the caller's job."""

    def __init__(self, runner, graph_resources=None):
        self.runner = runner
        self.graph_resources = graph_resources
        self.execution = None
        self._shape_key = None

    def invalidate(self):
        if self.execution is not None:
            self.execution.invalidate()
        self.execution, self._shape_key = None, None

    def step(self, chunk, noise, full_router_load=None):
        if full_router_load is None:
            full_router_load = noise.new_ones(getattr(self.runner.actor, "num_experts", 1))
        batch = {**chunk, "_noise": noise, "_router_load": full_router_load}
        key = (self.runner._effective_horizon(), tuple((name, tuple(value.shape), value.dtype, value.device)
               for name, value in sorted(batch.items())))
        if key != self._shape_key:
            self.invalidate()
            self.execution = CapturedTensorUpdate(_ActorLoss(self.runner), self.runner.opt_actor,
                max_grad_norm=self.runner.cfg.max_grad_norm, clip_value=self.runner.cfg.actor_grad_value_clip,
                warmup=graph_warmup_steps(self.runner.cfg, "actor"), compile_loss=False,
                guard_nonfinite=getattr(self.runner.cfg, "training_schedule", "validated_v2") != "supermimic_v1",
                graph_resources=self.graph_resources)
            self._shape_key = key
        with frozen_parameters(self.runner.world_model, self.runner.discriminator, self.runner.critic):
            return self.execution.step(batch)

    __call__ = step


class _CriticLoss(nn.Module):
    metric_names = ()

    def __init__(self, critic):
        super().__init__()
        self.critic = critic

    @staticmethod
    def prepare(batch, **unused):
        return {name: value.detach() for name, value in batch.items()}

    def forward(self, batch):
        loss = (self.critic(batch["features"])-batch["target"]).square().mean()
        return loss, {"loss": loss.detach()}


def make_critic_execution(runner, graph_resources=None):
    """Build one fixed-shape critic executor; callers cache by minibatch shape."""
    return CapturedTensorUpdate(_CriticLoss(runner.critic), runner.opt_critic,
        max_grad_norm=runner.cfg.max_grad_norm, warmup=graph_warmup_steps(runner.cfg, "critic"),
        compile_loss=True, compile_mode=runner.cfg.compile_mode,
        guard_nonfinite=getattr(runner.cfg, "training_schedule", "validated_v2") != "supermimic_v1",
        graph_resources=graph_resources)
