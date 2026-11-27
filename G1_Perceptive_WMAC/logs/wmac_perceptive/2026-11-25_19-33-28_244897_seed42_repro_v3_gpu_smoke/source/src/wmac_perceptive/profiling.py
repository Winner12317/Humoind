"""Explicit, bounded profiler window around the unmodified training loop.

Normal training never imports this module. This observer adds labels and
profiler steps only; configuration, sampling, warmup, checkpoint cadence and
learner calls stay owned by the ordinary runner.
"""
from __future__ import annotations

from contextlib import contextmanager,ExitStack
from dataclasses import dataclass,asdict
from functools import wraps
from pathlib import Path
from unittest.mock import patch
import json

import torch


@dataclass(frozen=True)
class ProfileWindow:
    wait_iterations: int=100
    active_iterations: int=24
    memory: bool=False

    def __post_init__(self):
        for name,minimum in (("wait_iterations",0),("active_iterations",1)):
            value=getattr(self,name)
            if isinstance(value,bool) or not isinstance(value,int) or value<minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")


@contextmanager
def observe_training(runner_type,window: ProfileWindow):
    """Observe one real learn call, restoring class methods even after errors.

    Trace timings include instrumentation and must not replace the normal
    CSV throughput baseline. The process continues ordinary training after
    one wait/warmup/active trace cycle; no synthetic learner steps are added.
    """
    state={"profiler":None,"runner":None,"completed_rows":0}
    original_learn,original_log=runner_type.learn,runner_type._log

    def ready(profiler):
        runner=state["runner"]
        output=Path(runner.log_dir)/"profile"
        output.mkdir(parents=True,exist_ok=True)
        profiler.export_chrome_trace(str(output/"trace.json"))
        averages=profiler.key_averages()
        cuda=str(runner.cfg.device).startswith("cuda")
        (output/"operators.txt").write_text(averages.table(
            sort_by="self_cuda_time_total" if cuda else "self_cpu_time_total",row_limit=80)+'\n')
        (output/"summary.json").write_text(json.dumps(dict(
            window=asdict(window),iteration_at_trace_end=runner.iteration,
            recorded_complete_iterations=max(0,min(window.active_iterations,
                state["completed_rows"]-window.wait_iterations-1)),
            full_window_completed=state["completed_rows"]>=window.wait_iterations+1+window.active_iterations,
            num_envs=runner.cfg.num_envs,training_configuration_modified=False,
            synthetic_training_updates=0,
            scope="One instrumented real-training window; training continues afterwards.",
            notes=["Profiler timing includes observer overhead; compare ordinary metrics.csv windows for throughput.",
                   "CPU phase labels describe launch scopes; CUDA activity and overlap must be inspected in trace.json.",
                   "Default window skips100 real learner iterations, warms profiler for1 and records24.",
                   "External simulator allocations are additional to the PyTorch allocator."]),indent=2)+'\n')

    @wraps(original_learn)
    def learn(runner,*args,**kwargs):
        if state["profiler"] is not None:
            raise RuntimeError("Nested profiled learn calls are unsupported")
        activities=[torch.profiler.ProfilerActivity.CPU]
        if str(runner.cfg.device).startswith("cuda"):
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        profiler=torch.profiler.profile(activities=activities,
            schedule=torch.profiler.schedule(wait=window.wait_iterations,warmup=1,
                                             active=window.active_iterations,repeat=1),
            on_trace_ready=ready,record_shapes=False,profile_memory=window.memory,with_stack=False)
        state.update(profiler=profiler,runner=runner,completed_rows=0)
        try:
            with profiler:return original_learn(runner,*args,**kwargs)
        finally:
            state.update(profiler=None,runner=None)

    @wraps(original_log)
    def log(runner,*args,**kwargs):
        result=original_log(runner,*args,**kwargs)
        if state["profiler"] is not None:
            state["completed_rows"]+=1
            state["profiler"].step()
        return result

    def labelled(name,method):
        @wraps(method)
        def call(runner,*args,**kwargs):
            profiler=state["profiler"]
            if profiler is not None and profiler.current_action!=torch.profiler.ProfilerAction.NONE:
                with torch.profiler.record_function(f"WMAC::{name}"):
                    return method(runner,*args,**kwargs)
            return method(runner,*args,**kwargs)
        return call

    with ExitStack() as stack:
        stack.enter_context(patch.object(runner_type,"learn",learn))
        stack.enter_context(patch.object(runner_type,"_log",log))
        for name in ("collect","fit_model","update_discriminator","validate_model",
                     "update_actor_critic","save","evaluate_checkpoint"):
            method=getattr(runner_type,name,None)
            if method is not None:stack.enter_context(patch.object(runner_type,name,labelled(name,method)))
        yield
