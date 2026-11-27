"""Shared FP16 frame replay for zero-centred, already processed RPL images.

The two caches deliberately preserve both the current observation and its
pre-reset terminal successor. With delay=0 these can differ from the next
transition's reset image, so a single cache cannot represent both. Histories
are reconstructed at sample time and latents are never cached.
"""
from __future__ import annotations

from dataclasses import asdict

import torch

from src.wmac.replay import ReplayBuffer
from .compact_gpu_replay import CompressedLatentReplay
from .latent_perception import processed_stack_shape
from .latent_replay import LatentPerceptiveReplay
from .perception import PerceptionSpec


def _default_spec():
    values = dict(height=27, width=48, ring_length=36, frames=8, stride=5, delay=0,
                  proprio_dim=768, crop_top=0, crop_bottom=0, crop_left=0, crop_right=0,
                  blur_kernel=1, include_valid=False, max_depth=2., depth_protocol="g1_rpl_v1")
    return PerceptionSpec(**values)


class CenteredDepthReplay(CompressedLatentReplay):
    """Replay's native masks/splits with two single-frame FP16 image caches."""

    storage_protocol = "centered_frame_fp16_v1"
    _AMP_FIELDS = ("_amp_overflow_states", "_amp_episode_starts", "_amp_episode_start", "_amp_pending_reset")

    def __init__(self, capacity_steps, num_envs, device="cuda:0", validation_env_fraction=0., *, spec=None):
        spec = _default_spec() if spec is None else spec
        if spec.delay != 0 or spec.include_valid:
            raise ValueError("Centered frame replay requires delay0 and depth-only observations")
        LatentPerceptiveReplay.__init__(self,capacity_steps,num_envs,device,validation_env_fraction,spec=spec)
        self.cache_steps = capacity_steps+spec.ring_length
        _, height, width = processed_stack_shape(spec)
        shape = (num_envs, spec.visual_channels, height, width)
        self.current_frame_cache = torch.empty((self.cache_steps,*shape),dtype=torch.float16,device=self.device)
        self.end_frame_cache = torch.empty((capacity_steps,*shape),dtype=torch.float16,device=self.device)
        self._episode_starts = torch.empty(capacity_steps,num_envs,dtype=torch.long,device=self.device)
        self._episode_start = torch.full((num_envs,),1-spec.ring_length,dtype=torch.long,device=self.device)
        self._pending_reset = torch.zeros(num_envs,dtype=torch.bool,device=self.device)
        self._bootstrapped = False
        self._incoming_visual_valid = None
        self._offsets = torch.tensor(tuple(i-spec.ring_length+1 for i in spec.selected_indices),
                                     dtype=torch.long,device=self.device)

    @staticmethod
    def estimated_frame_cache_bytes(capacity_steps,num_envs,spec=None):
        spec = _default_spec() if spec is None else spec
        _,height,width = processed_stack_shape(spec)
        return (2*capacity_steps+spec.ring_length)*num_envs*spec.visual_channels*height*width*2

    @staticmethod
    def _visual_valid(image, include_valid=False):
        if include_valid:
            raise ValueError("Centred replay has no encoded validity channel")
        values = image.detach().reshape(image.shape[0],-1)
        return (torch.isfinite(values) & (values.abs()<=torch.finfo(torch.float16).max)).all(-1)

    @staticmethod
    def _quantize(frame):
        # The original invalid-row flag remains quarantined. Padding/cache
        # values stay finite without clipping valid negative values/overshoot.
        bound = torch.finfo(torch.float16).max
        return torch.nan_to_num(frame.detach(),nan=0.,posinf=0.,neginf=0.).clamp(-bound,bound).to(torch.float16)

    @torch.no_grad()
    def bootstrap_processed_history(self,history):
        _,height,width = processed_stack_shape(self.spec)
        expected = (self.num_envs,self.spec.ring_length,self.spec.visual_channels,height,width)
        if self.total_steps or self._bootstrapped:
            raise ValueError("Frame-cache bootstrap is only valid before the first transition")
        if (not isinstance(history,torch.Tensor) or tuple(history.shape)!=expected
                or not history.is_floating_point()):
            raise ValueError(f"Processed bootstrap history must have shape {expected}")
        if not bool(self._visual_valid(history).all()):
            raise ValueError("Bootstrap history must contain finite FP16-representable centred depth")
        times = torch.arange(1-self.spec.ring_length,1,device=self.device)
        self.current_frame_cache[times.remainder(self.cache_steps)] = self._quantize(history).to(self.device).transpose(0,1)
        self._bootstrapped = True

    bootstrap = bootstrap_processed_history

    @torch.no_grad()
    def add(self,batch,*,processed_history=None):
        self._validate_perceptive_shapes(batch)
        if processed_history is not None:
            self.bootstrap_processed_history(processed_history)
        if not self._bootstrapped:
            raise ValueError("Centred frame replay requires an explicit processed-history bootstrap")
        visual_valid = self._visual_valid(batch["depth_stack"]) & self._visual_valid(batch["next_depth_stack"])
        stored = {k:v for k,v in batch.items() if k not in self._DEPTH_FIELDS}
        stored["actor_history"] = stored["actor_history"].float()
        stored["relative_height"] = stored["relative_height"].float()
        # Validate before touching either visual or Hiking AMP metadata.
        if self._RESERVED.intersection(stored):
            raise ValueError("Replay data uses a reserved sampling metadata field")
        ReplayBuffer.finite_mask(stored)
        if self.data:
            if self.data.keys()!=stored.keys():
                raise ValueError("Replay fields must stay identical after allocation")
            for name,value in stored.items():
                if self.data[name].shape[1:]!=value.shape or self.data[name].dtype!=value.dtype:
                    raise ValueError(f"Replay field {name!r} changed shape or dtype")
        row,time = self.ptr,self.total_steps
        starts = torch.where(self._pending_reset,time,self._episode_start)
        self._incoming_visual_valid = visual_valid
        from .hiking_amp import before_replay_add,after_replay_add
        before_replay_add(self)
        try:
            accepted = ReplayBuffer.add(self,stored)
        finally:
            self._incoming_visual_valid = None
        after_replay_add(self,stored,accepted)
        channels = self.spec.visual_channels
        self.current_frame_cache[time%self.cache_steps].copy_(self._quantize(batch["depth_stack"][:,-channels:]).to(self.device))
        self.end_frame_cache[row].copy_(self._quantize(batch["next_depth_stack"][:,-channels:]).to(self.device))
        self._episode_starts[row].copy_(starts)
        self._episode_start.copy_(starts)
        self._pending_reset.zero_()
        for name in ("terminated","timeout"):
            if name in batch:
                self._pending_reset |= batch[name].to(self.device).reshape(self.num_envs,-1).bool().any(-1)
        return accepted

    def _depth(self,rows,env,successor=False):
        time = self._timestamps[rows]
        starts = self._episode_starts[rows,env]
        times = torch.maximum(time[...,None]+self._offsets+int(successor),starts[...,None])
        values = self.current_frame_cache[times.remainder(self.cache_steps),env[...,None]].float()
        if successor:
            # Only offset0's successor is t+1; preserve this transition's
            # terminal image instead of reading a later transition's reset.
            future = times==time[...,None]+1
            end = self.end_frame_cache[rows,env].float().unsqueeze(-4)
            values = torch.where(future[...,None,None,None],end,values)
        return values.flatten(-4,-3)

    def state_dict(self):
        result = ReplayBuffer.state_dict(self)
        result.update(perceptive_format=self.storage_protocol,perception_spec=asdict(self.spec),
            current_frame_cache=self.current_frame_cache,end_frame_cache=self.end_frame_cache,
            episode_starts=self._episode_starts,episode_start=self._episode_start,
            pending_reset=self._pending_reset,bootstrapped=self._bootstrapped,
            hiking_amp_history=getattr(self,"_hiking_amp_history",False))
        if result["hiking_amp_history"]:
            result["hiking_amp_state"] = {name:getattr(self,name) for name in self._AMP_FIELDS}
        return result

    @torch.no_grad()
    def load_state_dict(self,saved):
        if saved.get("perceptive_format")!=self.storage_protocol or saved.get("perception_spec")!=asdict(self.spec):
            raise ValueError("Centred FP16 replay protocol/spec differs")
        required = {"current_frame_cache":self.current_frame_cache,"end_frame_cache":self.end_frame_cache,
            "episode_starts":self._episode_starts,"episode_start":self._episode_start,"pending_reset":self._pending_reset}
        for name,expected in required.items():
            value = saved.get(name)
            if not isinstance(value,torch.Tensor) or value.shape!=expected.shape or value.dtype!=expected.dtype:
                raise ValueError(f"Invalid centred replay field {name}")
        if not isinstance(saved.get("bootstrapped"),bool) or (saved.get("size",0) and not saved["bootstrapped"]):
            raise ValueError("Centred replay bootstrap metadata missing")
        if set(saved.get("data",{})).intersection(self._DEPTH_FIELDS):
            raise ValueError("Centred replay must not silently load full-stack storage")
        amp_saved = saved.get("hiking_amp_history",False)
        if not isinstance(amp_saved,bool):
            raise ValueError("Invalid Hiking AMP replay marker")
        if amp_saved:
            from .hiking_amp import configure_replay
            configure_replay(self)
            for name in self._AMP_FIELDS:
                value = saved.get("hiking_amp_state",{}).get(name)
                expected = getattr(self,name)
                if not isinstance(value,torch.Tensor) or value.shape!=expected.shape or value.dtype!=expected.dtype:
                    raise ValueError(f"Invalid Hiking AMP replay field {name}")
        elif getattr(self,"_hiking_amp_history",False) and saved.get("size",0):
            raise ValueError("Centred replay is missing configured Hiking AMP histories")
        ReplayBuffer.load_state_dict(self,saved)
        for name,target in required.items():
            target.copy_(saved[name].to(self.device))
        if amp_saved:
            for name in self._AMP_FIELDS:
                getattr(self,name).copy_(saved["hiking_amp_state"][name].to(self.device))
        self._bootstrapped = saved["bootstrapped"]
        if self.size:
            self._pending_reset.fill_(True)
            if getattr(self,"_hiking_amp_history",False):
                self._amp_pending_reset.fill_(True)
