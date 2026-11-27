"""Tensor-only Instinct latent WM fitting and SuperMimic update execution.

Statistics are updated before the differentiable rollout. The official
schedule applies every optimizer step; old validated schedules retain their
finite guard. Meter/image/FK diagnostics stay in validation.
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext

import torch
from torch import nn
from torch.nn import functional as F

from .instinct_latent import normalized_latent_errors

WM_BATCH_FIELDS = ("state", "next_state", "action", "context", "contact_cost",
                   "depth_stack", "next_depth_stack", "mask")


def graph_warmup_steps(cfg, kind, warmup=None):
    """Count minibatch calls in the official first three training macros."""
    warmup = cfg.graph_warmup if warmup is None else warmup
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 1:
        raise ValueError("Graph warmup must be a positive integer")
    multiplier = 1
    if getattr(cfg, "training_schedule", "validated_v2") == "supermimic_v1":
        factors = {"wm": ("wm_updates",), "actor": ("actor_updates",),
                   "critic": ("actor_updates", "critic_epochs", "critic_minibatches")}
        if kind not in factors:
            raise ValueError("Unknown captured update kind")
        for name in factors[kind]:
            count = getattr(cfg, name, 1)
            if isinstance(count, bool) or not isinstance(count, int) or count < 1:
                raise ValueError(f"{name} must be a positive integer")
            multiplier *= count
    return warmup*multiplier


def safe_optimizer_step(loss, norm, optimizer, *, guard=True):
    """Return whether an update was applied, preserving the chosen schedule.

    PyTorch fused Adam's found_inf protocol skips parameters and moments and
    rolls back its already-incremented step counters. No scaling is applied.
    ReferenceMuonAdam supplies the same device guard for its complete state.
    Other optimizers/devices retain one real host guard per update. With
    guard=False, match the official unconditional optimizer step.
    """
    if not guard:
        # Official SuperMimic _optimize_step always applies the optimizer.
        optimizer.step()
        return torch.ones((), dtype=torch.bool, device=loss.device)
    finite = torch.isfinite(loss.detach()) & torch.isfinite(norm.detach())
    cuda_parameters = all(all(parameter.is_cuda for parameter in group["params"])
                          for group in optimizer.param_groups)
    fused_cuda = (cuda_parameters and (
        getattr(optimizer, "_supports_device_finite_guard", False) or
        (isinstance(optimizer, (torch.optim.Adam, torch.optim.AdamW))
         and getattr(optimizer, "_step_supports_amp_scaling", False)
         and all(group.get("fused", False) for group in optimizer.param_groups))))
    if fused_cuda:
        absent = object()
        previous = {name: getattr(optimizer, name, absent) for name in ("found_inf", "grad_scale")}
        if any(value is not absent and value is not None for value in previous.values()):
            raise ValueError("Optimizer already has an external AMP scaling/finite-check context")
        optimizer.found_inf = (~finite).float()
        optimizer.grad_scale = None
        try:
            optimizer.step()
        finally:
            for name, value in previous.items():
                if value is absent:
                    delattr(optimizer, name)
                else:
                    setattr(optimizer, name, value)
    elif bool(finite):
        optimizer.step()
    return finite


def _masked_mean(value, valid):
    count = valid.sum()
    total = torch.where(valid, value, torch.zeros_like(value)).sum()
    return torch.where(count > 0, total/count.clamp_min(1), value.new_full((), float("inf")))


class LatentWorldModelKernel(nn.Module):
    """Pure fixed-shape loss kernel over the actual model/encoder/decoder."""

    def __init__(self, runner):
        super().__init__()
        self.world_model = runner.world_model
        self.vision = runner.actor.vision
        self.decoder = runner.latent_decoder
        self.window = runner.cfg.wm_window
        self.latent_dim = runner.cfg.latent_dim
        self.recon_images = runner.cfg.latent_recon_images
        self.latent_weight = runner.cfg.latent_loss_coef
        self.recon_weight = runner.cfg.latent_recon_coef
        self.stat_rate = runner.cfg.latent_stat_rate
        self.precision = runner.cfg.precision
        self.camera_capture_steps=5 if getattr(runner.cfg,"sensor_protocol",None)=="g1_rpl_depth_v1" else 0
        self.metric_names = ("wm_loss", "wm_contact_loss", "wm_last_error", "wm_walk_error",
            "wm_latent_loss", "wm_latent_recon_l1", "wm_latent_persistence_error", "wm_latent_target_std",
            *(f"wm_h{h}_model_error" for h in range(1, self.window+1)),
            *(f"wm_h{h}_latent_error" for h in range(1, self.window+1)), "wm_prediction_finite")

    def _encode(self, stack):
        prefix = stack.shape[:-3]
        image = stack.reshape(-1, *stack.shape[-3:]).to(self.vision.conv[0].weight.dtype)
        context = (torch.autocast("cuda", dtype=torch.bfloat16)
                   if self.precision == "bf16" and image.device.type == "cuda" else nullcontext())
        with context:
            latent = self.vision.projection(self.vision.conv(image).flatten(1))
        if latent.dtype == torch.bfloat16:
            latent = latent.float()
        return latent.reshape(*prefix, self.latent_dim)

    @torch.no_grad()
    def prepare(self, batch, *, update_statistics=True):
        """Encode current targets without gradients and optionally advance EMA.

        All returned values are tensors; the fixed dictionary is suitable for
        fullgraph compilation. No replay codes or encoder outputs are cached
        across optimizer updates.
        """
        # Replay may supply its original uint8 cache codes. Decode with the
        # exact same FP32 division as regular sampling, without re-quantizing.
        def image(value):
            return value.float()/255. if value.dtype == torch.uint8 else value.float()
        prepared = {name: batch[name] for name in WM_BATCH_FIELDS
                    if name not in ("depth_stack", "next_depth_stack")}
        start = self._encode(image(batch["depth_stack"][:, 0])).detach()
        target = self._encode(image(batch["next_depth_stack"])).detach()
        if update_statistics:
            previous = torch.cat((start[:, None], target[:, :-1]), 1)
            self.world_model.update_z_stats(target, target-previous, batch["mask"], self.stat_rate)
        flat_mask=batch["mask"].bool().flatten()
        order=torch.argsort(flat_mask.to(torch.int32),descending=True,stable=True)
        selected=order[:min(self.recon_images,flat_mask.shape[0])]
        stacks=batch["next_depth_stack"].flatten(0,1).index_select(0,selected)
        prepared.update(initial_latent=start,target_latent=target,
            reconstruction_stacks=image(stacks),
            reconstruction_mask=flat_mask.index_select(0,selected))
        return prepared

    def prepare_shared(self,batch,*,update_statistics=True):
        """Encode reconstruction rows once while keeping WM targets stopped.

        E has no cross-example normalization/dropout. Selected rows can share
        their current encoding with the reconstruction objective; only their
        detached values enter target statistics and WM supervision. Every
        optimizer update invokes this again with that update's online E.
        """
        if not torch.is_grad_enabled():
            return self.prepare(batch,update_statistics=update_statistics)
        def image(value):
            return value.float()/255. if value.dtype==torch.uint8 else value.float()
        with torch.no_grad():
            prepared={name:batch[name] for name in WM_BATCH_FIELDS
                      if name not in ("depth_stack","next_depth_stack")}
            start=self._encode(image(batch["depth_stack"][:,0])).detach()
            flat=batch["next_depth_stack"].flatten(0,1)
            mask=batch["mask"].bool().flatten()
            order=torch.argsort(mask.to(torch.int32),descending=True,stable=True)
            count=min(self.recon_images,len(order))
            selected,remaining=order[:count],order[count:]
            reconstruction=image(flat.index_select(0,selected))
        reconstruction_latent=self._encode(reconstruction)
        with torch.no_grad():
            target=reconstruction_latent.new_empty(len(order),self.latent_dim)
            target.index_copy_(0,selected,reconstruction_latent.detach())
            if remaining.numel():
                other=self._encode(image(flat.index_select(0,remaining))).detach()
                target.index_copy_(0,remaining,other)
            target=target.reshape(*batch["mask"].shape,self.latent_dim)
            if update_statistics:
                previous=torch.cat((start[:,None],target[:,:-1]),1)
                self.world_model.update_z_stats(target,target-previous,batch["mask"],self.stat_rate)
            prepared.update(initial_latent=start,target_latent=target,
                reconstruction_stacks=reconstruction,reconstruction_mask=mask.index_select(0,selected))
        prepared["reconstruction_latent"]=reconstruction_latent
        return prepared

    def forward(self, batch):
        state = batch["state"][:, 0]
        context=batch["context"][:,0]
        latent, target = batch["initial_latent"], batch["target_latent"]
        mask = batch["mask"].bool()
        errors, contacts, latents, finite = [], [], [], []
        for step in range(self.window):
            active = mask[:, step]
            state = torch.where(active[:, None], state, batch["state"][:, step])
            if self.world_model.num_contact==13:
                context=torch.where(active[:,None],context,batch["context"][:,step])
            else:
                context=batch["context"][:,step]
            state, contact, latent = self.world_model.step_with_latent(
                state, batch["action"][:, step], context, latent)
            if self.world_model.num_contact==13:
                from .hiking_context import advance_hiking_context
                context=advance_hiking_context(context,batch["action"][:,step],contact,
                    camera_capture_steps=self.camera_capture_steps)
            errors.append(self.world_model.loss(state, batch["next_state"][:, step]))
            contacts.append(F.smooth_l1_loss(torch.log1p(contact),
                torch.log1p(batch["contact_cost"][:, step].clamp_min(0)), reduction="none").mean(-1))
            latents.append(latent)
            finite.append(torch.isfinite(state).all(-1) & torch.isfinite(contact).all(-1)
                          & torch.isfinite(latent).all(-1))
        physical = torch.stack(errors, 1)
        contact_errors = torch.stack(contacts, 1)
        predictions = torch.stack(latents, 1)
        latent_errors = normalized_latent_errors(predictions, target)
        persistence = normalized_latent_errors(batch["initial_latent"][:, None].expand_as(target), target)
        dyn, contact = _masked_mean(physical, mask), _masked_mean(contact_errors, mask)
        latent_loss = (latent_errors*batch["mask"]).sum()/mask.shape[0]

        # A fixed-size gather replaces boolean indexing/nonzero. Valid rows
        # sort first in their original order; padded decoder rows get zero
        # weight, preserving the old first-Br-valid reconstruction mean.
        stacks = batch["reconstruction_stacks"]
        encoded=(batch["reconstruction_latent"] if "reconstruction_latent" in batch else self._encode(stacks))
        reconstruction = (self.decoder(encoded)-stacks).abs().flatten(1).mean(-1)
        reconstruction = _masked_mean(reconstruction, batch["reconstruction_mask"])
        total = dyn+contact+self.latent_weight*latent_loss+self.recon_weight*reconstruction

        weight = mask[..., None].to(target.dtype)
        n = weight.sum().clamp_min(1)
        center = (target*weight).sum((0, 1))/n
        std = (((target-center).square()*weight).sum((0, 1))/n).clamp_min(0).sqrt().mean()
        prediction_finite = (torch.stack(finite, 1) | ~mask).all()
        metrics = torch.stack((dyn, contact, _masked_mean(physical[:, -1], mask[:, -1]), dyn,
            _masked_mean(latent_errors, mask), reconstruction, _masked_mean(persistence, mask), std,
            *(_masked_mean(physical[:, :h], mask[:, :h]) for h in range(1, self.window+1)),
            *(_masked_mean(latent_errors[:, :h], mask[:, :h]) for h in range(1, self.window+1)),
            prediction_finite.to(dyn.dtype))).detach()
        return total, metrics


class WMExecution:
    """One guarded update with eager, compiled or captured execution.

    ``graph`` captures forward/backward only. ``graph_full`` additionally
    captures global clipping and the device-guarded optimizer step, with
    persistent CUDA scalar learning rates. Target encoding/EMA stay outside
    both graphs so every update uses the current online encoder. The original
    schedule warms up three complete macros; optional shared (stream, pool)
    resources preserve WM/Actor/Critic capture and replay ordering.
    """

    def __init__(self, runner, mode="compile", *, backend="inductor", fullgraph=True, warmup=3,
                 _kernel=None, compile_loss=True, clip_value=0., guard_nonfinite=None, graph_resources=None):
        if mode not in ("eager", "compile", "graph", "graph_full"):
            raise ValueError("WMExecution mode must be eager, compile, graph or graph_full")
        if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 1:
            raise ValueError("Graph warmup must be a positive integer")
        self.kernel = LatentWorldModelKernel(runner) if _kernel is None else _kernel
        self.shared_reconstruction=(mode not in ("graph","graph_full")
            and getattr(runner.cfg,"hiking_training_version",1)>=2 and hasattr(self.kernel,"prepare_shared"))
        self.prepare_function=self.kernel.prepare_shared if self.shared_reconstruction else self.kernel.prepare
        self.optimizer = runner.opt_model
        self.max_grad_norm = runner.cfg.max_grad_norm
        self.clip_value = clip_value
        self.guard_nonfinite = (getattr(runner.cfg, "training_schedule", "validated_v2") != "supermimic_v1"
                                if guard_nonfinite is None else guard_nonfinite)
        if not isinstance(self.guard_nonfinite, bool):
            raise ValueError("guard_nonfinite must be boolean")
        self.metric_names = self.kernel.metric_names
        self.mode = mode
        self.warmup = graph_warmup_steps(runner.cfg, "wm", warmup) if _kernel is None else warmup
        self._calls, self._graph, self._static_inputs = 0, None, None
        self._static_outputs = None
        self._static_step_outputs = None
        self._graph_identity = None
        self._parameters = tuple(parameter for group in self.optimizer.param_groups for parameter in group["params"])
        self._captured_gradients = None
        self.last_loss = None
        if mode in ("graph", "graph_full") and not next(self.kernel.parameters()).is_cuda:
            raise ValueError("WMExecution graph mode requires CUDA parameters")
        # Autograd retains forward-stream information across mixed compiled
        # and eager subgraphs. Warmup and capture must use this same stream.
        self._capture_stream, self._capture_pool = None, None
        if graph_resources is not None:
            if mode not in ("graph", "graph_full"):
                raise ValueError("Shared graph resources require graph execution")
            if not isinstance(graph_resources, tuple) or len(graph_resources) != 2:
                raise ValueError("Graph resources must be a (CUDA stream, pool) tuple")
            stream, pool = graph_resources
            if not isinstance(stream, torch.cuda.Stream) or stream.device != next(self.kernel.parameters()).device:
                raise ValueError("Graph resource stream must match the model CUDA device")
            self._capture_stream, self._capture_pool = stream, pool
        elif mode in ("graph", "graph_full"):
            self._capture_stream = torch.cuda.Stream(device=next(self.kernel.parameters()).device)
        if mode == "graph_full":
            self._validate_full_graph_optimizer()
        compile_options = dict(backend=backend, fullgraph=fullgraph, dynamic=False)
        if backend == "inductor":
            compile_options["mode"] = getattr(runner.cfg, "compile_mode", "default")
        self.loss_function = self.kernel if mode == "eager" or not compile_loss else torch.compile(self.kernel, **compile_options)

    def _clip_gradients(self):
        norm = nn.utils.clip_grad_norm_(self._parameters, self.max_grad_norm, foreach=True)
        if self.clip_value:
            nn.utils.clip_grad_value_(self._parameters, self.clip_value, foreach=True)
        return norm

    @contextmanager
    def _update_stream(self):
        if self._capture_stream is None:
            yield
            return
        caller = torch.cuda.current_stream(self._capture_stream.device)
        self._capture_stream.wait_stream(caller)
        try:
            with torch.cuda.stream(self._capture_stream):
                yield
        finally:
            caller.wait_stream(self._capture_stream)

    def _validate_full_graph_optimizer(self):
        reference_guard = getattr(self.optimizer, "_supports_device_finite_guard", False)
        native_guard = (isinstance(self.optimizer, (torch.optim.Adam, torch.optim.AdamW))
                        and getattr(self.optimizer, "_step_supports_amp_scaling", False))
        if not (reference_guard or native_guard):
            raise ValueError("graph_full requires a device-finite-guard optimizer")
        for group in self.optimizer.param_groups:
            lr = group.get("lr")
            if (not isinstance(lr, torch.Tensor) or lr.ndim != 0 or not lr.is_cuda
                    or lr.dtype != torch.float32 or lr.requires_grad):
                raise ValueError("graph_full requires persistent scalar float32 CUDA learning-rate tensors")
            if not reference_guard and not (group.get("fused", False) and group.get("capturable", False)):
                raise ValueError("graph_full native Adam requires fused=True and capturable=True")

    def _execution_identity(self):
        """Storage identity, never numerical values: updated tensor LR is legal."""
        result = []
        def append(label, value):
            if isinstance(value, torch.Tensor):
                result.append((label, id(value), value.data_ptr(), tuple(value.shape), value.dtype,
                               value.device, value.requires_grad))
            else:
                result.append((label, value))
        for name, parameter in self.kernel.named_parameters():
            append("parameter:"+name, parameter)
        for name, buffer in self.kernel.named_buffers():
            append("buffer:"+name, buffer)
        for index, gradient in enumerate(self._captured_gradients or ()):
            append(f"gradient:{index}", gradient)
        for group_index, group in enumerate(self.optimizer.param_groups):
            for key, value in group.items():
                if key != "params":
                    append(f"optimizer_group:{group_index}:{key}", value)
            for parameter_index, parameter in enumerate(group["params"]):
                append(f"optimizer_parameter:{group_index}:{parameter_index}", parameter)
                for key, value in sorted(self.optimizer.state[parameter].items()):
                    append(f"optimizer_state:{group_index}:{parameter_index}:{key}", value)
        append("clip_norm", self.max_grad_norm)
        append("clip_value", self.clip_value)
        append("guard_nonfinite", self.guard_nonfinite)
        append("capture_stream", None if self._capture_stream is None else self._capture_stream.cuda_stream)
        append("capture_pool", self._capture_pool)
        return result

    def _check_graph_identity(self):
        self._validate_full_graph_optimizer()
        current = self._execution_identity()
        if current != self._graph_identity:
            mismatch = next((left[0] for left, right in zip(current, self._graph_identity)
                             if left != right), "state layout")
            raise RuntimeError(f"Captured WM state moved or was rebound: {mismatch}")

    def _forward_backward(self, prepared):
        if self.mode in ("graph", "graph_full"):
            # Match SuperMimic's CapturedUpdate: warmup, capture and replay
            # all read the same persistent input buffers. Replay can return
            # views whose clone has a different stride; cloning only at
            # capture would trigger Dynamo recompilation inside the graph.
            if self._static_inputs is None:
                self._static_inputs = {name: value.clone() for name, value in prepared.items()}
            else:
                if prepared.keys() != self._static_inputs.keys():
                    raise ValueError("Captured WM batch fields changed")
                for name, source in prepared.items():
                    target = self._static_inputs[name]
                    if source.shape != target.shape or source.dtype != target.dtype or source.device != target.device:
                        raise ValueError(f"Captured WM batch schema changed: {name}")
                for name, source in prepared.items():
                    self._static_inputs[name].copy_(source)
            prepared = self._static_inputs
        if self.mode not in ("graph", "graph_full") or self._calls < self.warmup:
            self.optimizer.zero_grad(set_to_none=True)
            loss, metrics = self.loss_function(prepared)
            loss.backward()
            return loss, metrics
        if self._graph is None:
            if self.mode == "graph_full":
                self._validate_full_graph_optimizer()
                initialize = getattr(self.optimizer, "initialize_state", None)
                if initialize is not None:
                    initialize()
                if any(not self.optimizer.state[parameter] for parameter in self._parameters):
                    raise RuntimeError("graph_full optimizer states must be initialized before capture")
            stream = self._capture_stream
            # Other training phases legitimately clear the optimizer's grad
            # bindings. Own fixed buffers and record their clear+accumulation
            # in the graph rather than relying on backward's Python bindings.
            self._captured_gradients = tuple(torch.zeros_like(parameter)
                                             for parameter in self._parameters)
            for parameter, gradient in zip(self._parameters, self._captured_gradients):
                parameter.grad = gradient
            if self.mode == "graph_full":
                self._graph_identity = self._execution_identity()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=self._capture_pool, stream=stream):
                for gradient in self._captured_gradients:
                    gradient.zero_()
                loss, metrics = self.loss_function(self._static_inputs)
                loss.backward()
                if self.mode == "graph_full":
                    norm = self._clip_gradients()
                    accepted = safe_optimizer_step(loss, norm, self.optimizer, guard=self.guard_nonfinite)
            self._graph, self._static_outputs = graph, (loss, metrics)
            if self.mode == "graph_full":
                self._static_step_outputs = norm, accepted
                self._check_graph_identity()
        else:
            if self.mode == "graph_full":
                self._check_graph_identity()
        # zero_grad(set_to_none=True) in Actor/critic phases must not orphan
        # the buffers which captured CUDA kernels write on replay.
        for parameter, gradient in zip(self._parameters, self._captured_gradients):
            parameter.grad = gradient
        self._graph.replay()
        return self._static_outputs

    def step(self, batch):
        prepared = self.prepare_function(batch)
        with self._update_stream():
            loss, metrics = self._forward_backward(prepared)
            if self.mode == "graph_full" and self._graph is not None:
                norm, applied = self._static_step_outputs
            else:
                norm = self._clip_gradients()
                applied = safe_optimizer_step(loss, norm, self.optimizer, guard=self.guard_nonfinite)
            if self._graph is None:
                self.optimizer.zero_grad(set_to_none=True)
            self._calls += 1
            self.last_loss = loss.detach().clone()
            # Captured outputs alias replay-owned buffers. Each update's small
            # metrics snapshot must survive the following replay for aggregation.
            result = metrics.clone(), norm.detach().clone(), applied.clone()
        return result

    __call__ = step

    def invalidate(self):
        """Discard a graph after optimizer restore or a shape/horizon change."""
        self.optimizer.zero_grad(set_to_none=True)
        if self._graph is not None:
            self._graph.reset()
        self._graph = self._static_inputs = self._static_outputs = None
        self._static_step_outputs = self._graph_identity = self._captured_gradients = None
        self._calls = 0
