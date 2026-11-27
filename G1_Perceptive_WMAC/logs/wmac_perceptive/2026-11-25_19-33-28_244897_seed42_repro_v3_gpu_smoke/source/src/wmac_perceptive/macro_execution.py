"""Official-sized captured learner macros, with persistent gradient ownership.

Reference: SuperMimic abc124c1 ``CapturedUpdate`` and policy learner. One
policy macro contains the Actor step followed by all Critic minibatches.
Warmup performs three real macros; recording itself performs no extra update.
The small CPU/eager path executes the same callback for numerical comparisons.
"""
from __future__ import annotations

from contextlib import contextmanager
from collections.abc import Mapping

import torch
from torch import nn

from src.wmac.networks import frozen_parameters
from src.wmac.routing import router_losses
from .latent_execution import graph_warmup_steps


def _snapshot(outputs):
    if not isinstance(outputs, Mapping) or any(not isinstance(value, torch.Tensor)
                                               for value in outputs.values()):
        raise ValueError("A captured macro must return a dictionary of tensors")
    return {name: value.detach().clone() for name, value in outputs.items()}


class CapturedMacroUpdate:
    """Capture a whole learner macro, preserving input and optimizer addresses.

    ``fn(batch)`` performs its own forward/backward/clip/optimizer calls and
    uses ``zero_grad(optimizer)`` here. Before capture that method has ordinary
    set-to-none semantics. Capture owns buffers only for parameters which
    actually acquired gradients during warmup, preserving unused parameters.
    Other training phases may clear ``.grad``; replay rebinds these buffers.
    ``invalidate`` is required after restoring optimizer state or rebinding
    models. State identity is checked before replay, so omission fails closed.
    """

    def __init__(self, fn, modules, optimizers, *, warmup=3, cuda_graphs=True,
                 graph_resources=None):
        if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 1:
            raise ValueError("Macro warmup must be a positive number of real iterations")
        self.fn = fn
        self.modules, self.optimizers = tuple(modules), tuple(optimizers)
        self._parameters = tuple(dict.fromkeys(parameter for optimizer in self.optimizers
            for group in optimizer.param_groups for parameter in group["params"]))
        if not self._parameters:
            raise ValueError("A learner macro requires optimizer parameters")
        self.device = self._parameters[0].device
        if any(parameter.device != self.device for parameter in self._parameters):
            raise ValueError("A learner macro requires parameters on one device")
        if cuda_graphs and self.device.type != "cuda":
            raise ValueError("CUDA macro capture requires CUDA parameters")
        self.mode = "graph_full" if cuda_graphs else "eager"
        self.warmup, self._calls = warmup, 0
        self._graph = self._static_inputs = self._static_outputs = None
        self._graph_identity = None
        self._gradient_buffers = {}
        self._active_parameters = set()
        self._capture_stream = self._capture_pool = None
        if graph_resources is not None:
            if not cuda_graphs or not isinstance(graph_resources, tuple) or len(graph_resources) != 2:
                raise ValueError("Macro graph resources require a CUDA stream/pool pair")
            stream, pool = graph_resources
            if not isinstance(stream, torch.cuda.Stream) or stream.device != self.device:
                raise ValueError("Macro capture stream must match the parameter device")
            self._capture_stream, self._capture_pool = stream, pool
        elif cuda_graphs:
            self._capture_stream = torch.cuda.Stream(device=self.device)
            self._capture_pool = torch.cuda.graph_pool_handle()

    @contextmanager
    def _update_stream(self):
        if self._capture_stream is None:
            yield
            return
        caller = torch.cuda.current_stream(self.device)
        self._capture_stream.wait_stream(caller)
        try:
            with torch.cuda.stream(self._capture_stream):
                yield
        finally:
            caller.wait_stream(self._capture_stream)

    def zero_grad(self, optimizer):
        if optimizer not in self.optimizers:
            raise ValueError("Macro tried to clear an optimizer it does not own")
        if not self._gradient_buffers:
            optimizer.zero_grad(set_to_none=True)
            return
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                gradient = self._gradient_buffers.get(parameter)
                parameter.grad = gradient
                if gradient is not None:
                    gradient.zero_()

    def _bind_gradients(self):
        for parameter in self._parameters:
            parameter.grad = self._gradient_buffers.get(parameter)

    def _identity(self):
        result = []
        def append(name, value):
            if isinstance(value, torch.Tensor):
                result.append((name, id(value), value.data_ptr(), tuple(value.shape), value.dtype,
                               value.device, value.requires_grad))
            elif isinstance(value, (tuple, list)):
                result.append((name, tuple(value)))
            else:
                result.append((name, value))
        for index, module in enumerate(self.modules):
            for name, parameter in module.named_parameters():
                append(f"module:{index}:parameter:{name}", parameter)
            for name, buffer in module.named_buffers():
                append(f"module:{index}:buffer:{name}", buffer)
        for index, parameter in enumerate(self._parameters):
            append(f"gradient:{index}", self._gradient_buffers.get(parameter))
        for index, optimizer in enumerate(self.optimizers):
            for name in ("_zero_found_inf",):
                if hasattr(optimizer, name):
                    append(f"optimizer:{index}:{name}", getattr(optimizer, name))
            for group_index, group in enumerate(optimizer.param_groups):
                for name, value in group.items():
                    if name != "params":
                        append(f"optimizer:{index}:group:{group_index}:{name}", value)
                for parameter_index, parameter in enumerate(group["params"]):
                    append(f"optimizer:{index}:parameter:{group_index}:{parameter_index}", parameter)
                    for name, value in sorted(optimizer.state[parameter].items()):
                        append(f"optimizer:{index}:state:{group_index}:{parameter_index}:{name}", value)
        return result

    def _check_identity(self):
        current = self._identity()
        if current != self._graph_identity:
            mismatch = next((left[0] for left, right in zip(current, self._graph_identity)
                             if left != right), "state layout")
            raise RuntimeError(f"Captured macro state moved or was rebound: {mismatch}")

    def _copy_inputs(self, batch):
        if not isinstance(batch, Mapping) or any(not isinstance(value, torch.Tensor)
                                                 for value in batch.values()):
            raise ValueError("Macro inputs must be a dictionary of tensors")
        if self._static_inputs is None:
            self._static_inputs = {name: value.clone() for name, value in batch.items()}
        else:
            if batch.keys() != self._static_inputs.keys():
                raise ValueError("Captured macro input fields changed; invalidate first")
            for name, value in batch.items():
                target = self._static_inputs[name]
                if value.shape != target.shape or value.dtype != target.dtype or value.device != target.device:
                    raise ValueError(f"Captured macro input schema changed: {name}; invalidate first")
            for name, value in batch.items():
                self._static_inputs[name].copy_(value)
        return self._static_inputs

    def _stack_inputs(self, batches):
        """Pack detached WM batches straight into the persistent graph inputs.

        Ordinary stack followed by _copy_inputs writes the large image-code
        fields twice. Validate the complete schema before overwriting any
        captured storage, then use stack's out parameter for one write.
        """
        if not batches or any(not isinstance(batch, Mapping) for batch in batches):
            raise ValueError("Stacked macro inputs require nonempty tensor dictionaries")
        names=batches[0].keys()
        if not names or any(batch.keys()!=names for batch in batches):
            raise ValueError("Stacked macro input fields differ")
        schema={}
        for name in names:
            first=batches[0][name]
            if not isinstance(first,torch.Tensor):
                raise ValueError("Stacked macro inputs must contain detached tensors")
            for batch in batches:
                value=batch[name]
                if (not isinstance(value,torch.Tensor) or value.requires_grad
                        or value.shape!=first.shape or value.dtype!=first.dtype or value.device!=first.device):
                    raise ValueError(f"Stacked macro input schema differs: {name}")
            schema[name]=((len(batches),*first.shape),first.dtype,first.device)
        if self._static_inputs is not None:
            if names!=self._static_inputs.keys():
                raise ValueError("Captured macro input fields changed; invalidate first")
            for name,(shape,dtype,device) in schema.items():
                target=self._static_inputs[name]
                if target.shape!=shape or target.dtype!=dtype or target.device!=device:
                    raise ValueError(f"Captured macro input schema changed: {name}; invalidate first")
        else:
            self._static_inputs={name:batches[0][name].new_empty(shape)
                                 for name,(shape,_,_) in schema.items()}
        with torch.no_grad():
            for name in names:
                torch.stack([batch[name] for batch in batches],out=self._static_inputs[name])
        return self._static_inputs

    def step(self, batch):
        return self._step(batch,stacked=False)

    def step_stacked(self, batches):
        """Run one macro from detached, uniformly shaped raw replay batches."""
        return self._step(batches,stacked=True)

    def _step(self, batch, *, stacked):
        with self._update_stream():
            inputs = (self._stack_inputs(batch) if stacked else
                      self._copy_inputs(batch) if self.mode == "graph_full" else batch)
            if self.mode != "graph_full" or self._calls < self.warmup:
                outputs = self.fn(inputs)
                self._active_parameters.update(parameter for parameter in self._parameters
                                               if parameter.grad is not None)
            else:
                if self._graph is None:
                    for optimizer in self.optimizers:
                        initialize = getattr(optimizer, "initialize_state", None)
                        if initialize is not None:
                            initialize()
                        for group in optimizer.param_groups:
                            lr = group.get("lr")
                            if not isinstance(lr, torch.Tensor) or lr.ndim or lr.device != self.device:
                                raise ValueError("Captured macro optimizers require persistent CUDA scalar LR tensors")
                    self._gradient_buffers = {parameter: torch.zeros_like(parameter)
                        for parameter in self._parameters if parameter in self._active_parameters}
                    self._bind_gradients()
                    identity = self._identity()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=self._capture_stream, pool=self._capture_pool):
                        outputs = self.fn(inputs)
                    self._graph, self._static_outputs = graph, outputs
                    self._graph_identity = identity
                    self._check_identity()
                else:
                    self._check_identity()
                self._bind_gradients()
                self._graph.replay()
                outputs = self._static_outputs
            self._calls += 1
            return _snapshot(outputs)

    __call__ = step

    def invalidate(self):
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=True)
        if self._graph is not None:
            self._graph.reset()
        self._graph = self._static_inputs = self._static_outputs = self._graph_identity = None
        self._gradient_buffers = {}
        self._active_parameters = set()
        self._calls = 0


@contextmanager
def _trainable(module):
    parameters = tuple(module.parameters())
    flags = tuple(parameter.requires_grad for parameter in parameters)
    for parameter in parameters:
        parameter.requires_grad_(True)
    try:
        yield
    finally:
        for parameter, flag in zip(parameters, flags):
            parameter.requires_grad_(flag)


class ActorCriticMacroExecution:
    """One Actor update followed immediately by the whole Critic fitting loop.

    Sampling the Critic permutation is inside the macro, as in the reference.
    Replay advances the CUDA generator on each real macro. Input policy noise
    remains supplied by the caller. Historical/microbatched schedules use the
    existing executors; this adapter has no policy rollback or finite veto.
    """

    def __init__(self, runner, graph_resources=None, *, cuda_graphs=True):
        self.runner = runner
        self.execution = CapturedMacroUpdate(self._update_from,
            (runner.actor, runner.world_model, runner.critic, runner.discriminator, runner.amp_norm),
            (runner.opt_actor, runner.opt_critic), warmup=graph_warmup_steps(runner.cfg, "actor"),
            cuda_graphs=cuda_graphs, graph_resources=graph_resources)

    def invalidate(self):
        self.execution.invalidate()

    def _update_from(self, batch):
        runner, cfg = self.runner, self.runner.cfg
        chunk = {name: value for name, value in batch.items() if name not in ("_noise", "_router_load")}
        self.execution.zero_grad(runner.opt_actor)
        with frozen_parameters(runner.world_model, runner.discriminator, runner.critic):
            rollout = runner.imagine(chunk, noises=batch["_noise"])
            task_loss = -rollout["objective"].mean()
            auxiliary = task_loss.new_zeros(())
            if cfg.moe_load_balance_coef or cfg.moe_router_z_coef:
                balance, z, _ = router_losses(runner.actor, chunk["actor_history"], batch["_router_load"])
                if cfg.moe_load_balance_coef:
                    auxiliary = auxiliary + cfg.moe_load_balance_coef * balance
                if cfg.moe_router_z_coef:
                    auxiliary = auxiliary + cfg.moe_router_z_coef * z
            loss = task_loss + auxiliary
            loss.backward()
            parameters = tuple(parameter for group in runner.opt_actor.param_groups for parameter in group["params"])
            norm = nn.utils.clip_grad_norm_(parameters, cfg.max_grad_norm, foreach=True)
            if cfg.actor_grad_value_clip:
                nn.utils.clip_grad_value_(parameters, cfg.actor_grad_value_clip, foreach=True)
            runner.opt_actor.step()

        records = []
        if cfg.critic_bootstrap:
            features = rollout["x"].flatten(0, 1).detach()
            target = rollout["targets"].flatten().detach()
            count = len(features) // cfg.critic_minibatches * cfg.critic_minibatches
            order = torch.randperm(count, device=features.device)
            # The reference reuses one permutation for all five epochs.
            # Features/targets are detached and constant while the Critic's
            # weights change, so gather this table once rather than 20 times.
            minibatches=tuple(zip(features[order].tensor_split(cfg.critic_minibatches),
                                  target[order].tensor_split(cfg.critic_minibatches)))
            with _trainable(runner.critic):
                for _ in range(cfg.critic_epochs):
                    for batch_features,batch_target in minibatches:
                        if not len(batch_features):
                            continue
                        self.execution.zero_grad(runner.opt_critic)
                        critic_loss = runner._latent_critic_loss_fn(batch_features,batch_target)
                        critic_loss.backward()
                        critic_norm = nn.utils.clip_grad_norm_(runner.critic.parameters(), cfg.max_grad_norm, foreach=True)
                        runner.opt_critic.step()
                        records.append(torch.stack((critic_loss.detach(), critic_loss.new_zeros(()),
                                                     critic_norm.detach(), critic_loss.new_ones(()))))
        summary = torch.stack(records).mean(0) if records else loss.new_zeros(4)
        return {**{name: value.detach() for name, value in rollout.items()},
            "loss": loss.detach(), "task_loss": task_loss.detach(), "auxiliary_loss": auxiliary.detach(),
            "_macro_actor_norm": norm.detach(), "_macro_actor_accepted": loss.new_ones((), dtype=torch.bool),
            "_macro_critic_summary": summary}

    def step(self, chunk, noise, full_router_load=None):
        if full_router_load is None:
            full_router_load = noise.new_ones(getattr(self.runner.actor, "num_experts", 1))
        outputs = self.execution.step({**chunk, "_noise": noise, "_router_load": full_router_load})
        norm = outputs.pop("_macro_actor_norm")
        accepted = outputs.pop("_macro_actor_accepted")
        critic_summary = outputs.pop("_macro_critic_summary")
        # The captured loop owns the buffers; the physical runner's next
        # Actor phase may still require every frozen network's binding None.
        self.runner.opt_critic.zero_grad(set_to_none=True)
        return outputs, norm, accepted, critic_summary

    __call__ = step
