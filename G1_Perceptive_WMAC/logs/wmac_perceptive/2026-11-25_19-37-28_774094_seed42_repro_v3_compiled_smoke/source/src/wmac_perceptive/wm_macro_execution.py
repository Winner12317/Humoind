"""One captured SuperMimic WM macro, with online Instinct target encoding."""
from __future__ import annotations

import torch

from .latent_execution import LatentWorldModelKernel, WM_BATCH_FIELDS
from .macro_execution import CapturedMacroUpdate


class WMMacroExecution:
    """Ten sequential full updates in one graph, not ten cached latent targets.

    Raw replay codes/labels are the only persistent inputs. Every sub-update
    encodes its targets and updates latent statistics with the then-current E.
    Reconstruction keeps its ordinary encoder/decoder gradient. Three warmup
    calls are three real macros, exactly as in the reference learner.
    """
    def __init__(self,runner,graph_resources=None,*,cuda_graphs=True,compile_loss=True,backend="inductor"):
        self.runner=runner
        self.kernel=LatentWorldModelKernel(runner)
        self.metric_names=self.kernel.metric_names
        self.updates=runner.cfg.wm_updates
        options=dict(backend=backend,fullgraph=runner.cfg.compile_fullgraph,dynamic=False)
        if backend=="inductor":options["mode"]=runner.cfg.compile_mode
        compiled=compile_loss and runner.cfg.compile
        self.shared_reconstruction=getattr(runner.cfg,"hiking_training_version",1)>=2
        prepare=(self.kernel.prepare_shared if self.shared_reconstruction
                 else self.kernel.prepare)
        self.prepare_function=torch.compile(prepare,**options) if compiled else prepare
        self.loss_function=torch.compile(self.kernel,**options) if compiled else self.kernel
        self.execution=CapturedMacroUpdate(self._update_from,(self.kernel,),(runner.opt_model,),
            warmup=runner.cfg.graph_warmup,cuda_graphs=cuda_graphs,graph_resources=graph_resources)
        self.mode=self.execution.mode
        self.last_loss=None

    @property
    def _graph(self):return self.execution._graph

    @property
    def _calls(self):return self.execution._calls*self.updates

    @property
    def macro_calls(self):return self.execution._calls

    @property
    def warmup(self):return self.execution.warmup*self.updates

    def _update_from(self,batches):
        records=[]
        for index in range(self.updates):
            raw={name:values[index] for name,values in batches.items()}
            prepared=self.prepare_function(raw)
            self.execution.zero_grad(self.runner.opt_model)
            loss,metrics=self.loss_function(prepared)
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(self.execution._parameters,
                self.runner.cfg.max_grad_norm,foreach=True)
            self.runner.opt_model.step()
            records.append(torch.cat((metrics,norm.detach().reshape(1),metrics.new_ones(1))))
        return {"records":torch.stack(records),"last_loss":loss.detach()}

    def step_macro(self,batches):
        if len(batches)!=self.updates:
            raise ValueError("WM macro must contain the configured number of complete updates")
        # Sampling draws labels/codes only. Encoding is inside _update_from,
        # after the preceding update has changed both E and the dynamics.
        inputs=[{name:batch[name] for name in WM_BATCH_FIELDS} for batch in batches]
        outputs=self.execution.step_stacked(inputs)
        self.last_loss=outputs["last_loss"]
        return outputs["records"]

    def invalidate(self):
        self.execution.invalidate()
        self.last_loss=None
