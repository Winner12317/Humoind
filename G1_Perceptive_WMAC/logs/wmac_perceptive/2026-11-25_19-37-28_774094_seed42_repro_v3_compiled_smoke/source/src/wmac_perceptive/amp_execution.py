"""Execution-only acceleration for Hiking's unchanged AMP objective.

The GP needs differentiating an input gradient, so it deliberately uses eager
autograd inside CUDA Graphs rather than compiling the discriminator with
AOTAutograd. Only actual training microbatches warm up the capture stream.
"""
from __future__ import annotations

from contextlib import contextmanager

import torch


def prediction_and_penalty(discriminator, examples, count, coefficient):
    score = discriminator(examples).squeeze(-1)
    prediction = .5*((score[:count]+1).square().mean()+(score[count:]-1).square().mean())
    grad = torch.autograd.grad(score.sum(), examples, create_graph=True)[0]
    penalty = coefficient*grad.square().sum(-1).mean()
    return prediction, penalty


class AmpMicrobatchGraph:
    """Capture one fixed-size GP/backward, accumulating into stable gradients.

    The first three real microbatches execute eagerly on the capture stream.
    The fourth records the graph and replays it once; every later real
    microbatch also replays once. No warmup/capture calls optimizer.step.
    Irregular final microbatches retain the same eager weighted objective.
    """
    def __init__(self, discriminator, microbatch_size, coefficient):
        parameter = next(discriminator.parameters())
        if parameter.device.type != "cuda":
            raise ValueError("AMP CUDA Graph execution requires CUDA parameters")
        self.discriminator = discriminator
        self.microbatch_size = microbatch_size
        self.coefficient = coefficient
        self.device = parameter.device
        self.stream = torch.cuda.Stream(device=self.device)
        self.examples = parameter.new_empty(2*microbatch_size, discriminator.input_dim,
            requires_grad=True)
        self.examples.grad = torch.zeros_like(self.examples)
        self.weight = parameter.new_zeros(())
        self.gradients = tuple(torch.zeros_like(p) for p in discriminator.parameters())
        self.graph = None
        self.outputs = None
        self.real_warmup_microbatches = 0
        self.capture_count = 0
        self.replay_count = 0
        self._weight_value = None

    @contextmanager
    def context(self):
        current = torch.cuda.current_stream(self.device)
        self.stream.wait_stream(current)
        try:
            with torch.cuda.stream(self.stream):
                yield
        finally:
            current.wait_stream(self.stream)

    def begin_step(self):
        # The optimizer and runner may have set grad=None since last use.
        # Rebind the exact storage referenced by the captured accumulation.
        for parameter, gradient in zip(self.discriminator.parameters(), self.gradients):
            parameter.grad = gradient
            gradient.zero_()

    def _compute(self):
        self.examples.grad.zero_()
        prediction, penalty = prediction_and_penalty(self.discriminator, self.examples,
            self.microbatch_size, self.coefficient)
        ((prediction+penalty)*self.weight).backward()
        return prediction.detach(), penalty.detach()

    def accumulate(self, policy, expert, size):
        if len(policy) != self.microbatch_size:
            return None
        with torch.no_grad():
            self.examples[:self.microbatch_size].copy_(policy)
            self.examples[self.microbatch_size:].copy_(expert)
            value = self.microbatch_size/size
            if value != self._weight_value:
                self.weight.fill_(value)
                self._weight_value = value
        if self.real_warmup_microbatches < 3:
            self.real_warmup_microbatches += 1
            return self._compute()
        if self.graph is None:
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, stream=self.stream):
                self.outputs = self._compute()
            self.capture_count += 1
        # Capture records kernels; this replay executes the current real
        # microbatch exactly once, including the first captured microbatch.
        self.graph.replay()
        self.replay_count += 1
        return self.outputs


class AmpLogicalBatchExecution:
    """One fixed logical AMP backward, retaining the original microbatch order.

    Persistent interleaved policy/expert inputs are labels, never full-batch
    autograd leaves. Every microbatch has its own detached storage view, so GP
    backward cannot scatter into a full [2*logical_batch,670] input gradient.
    Three real logical updates warm up the capture stream. Recording the
    fourth performs no optimizer update; its first replay performs that real
    backward once. AdamW remains outside the graph with its original options.
    """
    def __init__(self,discriminator,optimizer,logical_batch_size,microbatch_size,*,
                 gradient_penalty,trunk_weight_decay,head_weight_decay,warmup=3,cuda_graphs=True):
        if logical_batch_size<1 or microbatch_size<1 or logical_batch_size%microbatch_size:
            raise ValueError("AMP logical capture requires complete execution microbatches")
        if isinstance(warmup,bool) or not isinstance(warmup,int) or warmup<1:
            raise ValueError("AMP logical warmup must count positive real updates")
        self.discriminator,self.optimizer=discriminator,optimizer
        self.logical_batch_size,self.microbatch_size=logical_batch_size,microbatch_size
        self.coefficient=gradient_penalty
        self.trunk_weight_decay,self.head_weight_decay=trunk_weight_decay,head_weight_decay
        self.parameters=tuple(discriminator.parameters())
        self.device=self.parameters[0].device
        if cuda_graphs and self.device.type!='cuda':
            raise ValueError("AMP logical CUDA capture requires CUDA parameters")
        self.mode='graph_logical' if cuda_graphs else 'eager'
        self.warmup=warmup
        self._calls=0
        self._graph=self._inputs=self._outputs=self._identity_at_capture=None
        self._gradients=()
        self._leaves=()
        self._input_gradient=None
        self.capture_count=self.replay_count=0
        self.stream=torch.cuda.Stream(device=self.device) if cuda_graphs else None
        self.pool=torch.cuda.graph_pool_handle() if cuda_graphs else None

    @property
    def graph(self):return self._graph

    @property
    def real_warmup_updates(self):return min(self._calls,self.warmup)

    @property
    def real_warmup_logical_updates(self):return self.real_warmup_updates

    @contextmanager
    def context(self):
        if self.stream is None:
            yield
            return
        caller=torch.cuda.current_stream(self.device)
        self.stream.wait_stream(caller)
        try:
            with torch.cuda.stream(self.stream):yield
        finally:
            caller.wait_stream(self.stream)

    def _bind_gradients(self):
        for parameter,gradient in zip(self.parameters,self._gradients):
            parameter.grad=gradient

    def _zero_grad(self):
        if self._gradients:
            self._bind_gradients()
            for gradient in self._gradients:gradient.zero_()
        else:
            self.optimizer.zero_grad(set_to_none=True)

    def _identity(self):
        values=[]
        for name,parameter in self.discriminator.named_parameters():
            values.append((name,id(parameter),parameter.data_ptr(),tuple(parameter.shape),
                           parameter.dtype,parameter.device,parameter.requires_grad))
        for name,buffer in self.discriminator.named_buffers():
            values.append((name,id(buffer),buffer.data_ptr(),tuple(buffer.shape),buffer.dtype,buffer.device))
        values.extend((id(gradient),gradient.data_ptr()) for gradient in self._gradients)
        values.append(('interleaved_labels',self._inputs[0].data_ptr()))
        values.append(('microbatch_input_gradient',self._input_gradient.data_ptr()))
        return tuple(values)

    def _backward(self):
        self._zero_grad()
        prediction_total=penalty_total=None
        weight=self.microbatch_size/self.logical_batch_size
        for examples in self._leaves:
            # The independent leaves are only 2*microbatch rows. Their data
            # were filled directly before replay, avoiding a second GPU copy.
            # Input gradients share one scratch buffer and are never used as
            # training labels or propagated into the full static input.
            self._input_gradient.zero_()
            prediction,penalty=prediction_and_penalty(self.discriminator,examples,
                self.microbatch_size,self.coefficient)
            ((prediction+penalty)*weight).backward()
            prediction=prediction.detach()*weight
            penalty=penalty.detach()*weight
            prediction_total=prediction if prediction_total is None else prediction_total+prediction
            penalty_total=penalty if penalty_total is None else penalty_total+penalty
        regularization=self.trunk_weight_decay*sum(p.square().sum() for p in self.parameters)
        regularization=regularization+self.head_weight_decay*self.discriminator.amp_linear.weight.square().sum()
        regularization.backward()
        norm=torch.linalg.vector_norm(torch.stack(tuple(p.grad.detach().norm() for p in self.parameters
                                                      if p.grad is not None)))
        return torch.stack((prediction_total+penalty_total+regularization.detach(),penalty_total,norm))

    def step(self,policy,expert):
        parameter=self.parameters[0]
        expected=(self.logical_batch_size,self.discriminator.input_dim)
        for value in (policy,expert):
            if (tuple(value.shape)!=expected or value.dtype!=parameter.dtype or value.device!=self.device):
                raise ValueError("AMP logical input shape/dtype/device changed; use the variable-size path")
        with self.context():
            count=self.logical_batch_size//self.microbatch_size
            if self._inputs is None:
                storage=parameter.new_empty(count,2*self.microbatch_size,self.discriminator.input_dim)
                self._inputs=(storage,)
                self._input_gradient=parameter.new_zeros(2*self.microbatch_size,self.discriminator.input_dim)
                self._leaves=tuple(storage[index].detach().requires_grad_(True) for index in range(count))
                for examples in self._leaves:examples.grad=self._input_gradient
            with torch.no_grad():
                self._inputs[0][:,:self.microbatch_size].copy_(policy.reshape(count,self.microbatch_size,-1))
                self._inputs[0][:,self.microbatch_size:].copy_(expert.reshape(count,self.microbatch_size,-1))
            if self.mode!='graph_logical' or self._calls<self.warmup:
                values=self._backward()
            else:
                if self._graph is None:
                    self._gradients=tuple(torch.zeros_like(p) for p in self.parameters)
                    self._bind_gradients()
                    identity=self._identity()
                    graph=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph,stream=self.stream,pool=self.pool):
                        values=self._backward()
                    self._graph,self._outputs=graph,values
                    self._identity_at_capture=identity
                    self.capture_count+=1
                if self._identity()!=self._identity_at_capture:
                    raise RuntimeError("AMP logical capture parameter/gradient storage changed; invalidate first")
                self._bind_gradients()
                self._graph.replay()
                self.replay_count+=1
                values=self._outputs
            # AdamW options, scalar LR, moments and unconditional update stay
            # exactly on the existing path. There is no clipping or new gate.
            self.optimizer.step()
            self._calls+=1
            return values.detach().clone()

    __call__=step

    def invalidate(self):
        self.optimizer.zero_grad(set_to_none=True)
        if self._graph is not None:self._graph.reset()
        self._graph=self._inputs=self._outputs=self._identity_at_capture=None
        self._gradients=()
        self._leaves=()
        self._input_gradient=None
        self._calls=self.capture_count=self.replay_count=0
