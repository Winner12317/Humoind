"""Terrain/depth-conditioned forward walking and turning with model BPTT.

Real transitions train physical dynamics, contact proxies and future images.
Actor BPTT advances predicted views at every step; no privileged map is passed
to the Actor. Pixel prediction and geometry interpolation remain learned/model
proxies. Held-out walking and recovery diagnostics select the usable rollout
length; full-image reconstruction error is not a policy-update veto.
"""
from __future__ import annotations

from dataclasses import asdict,replace
from collections.abc import Mapping
from pathlib import Path
import copy
import hashlib
import json
import math
import os
import random
import warnings

import numpy as np
import torch
from torch.nn import functional as F

from src.wmac.runner import WmacRunner, _device_batch, lambda_returns
from src.wmac.networks import Critic, frozen_parameters
from src.wmac.routing import is_moe,load_fraction,router_losses,routing_metrics,parameter_grad_norm
from src.tasks.amp_perceptive.sensors import read_depth_frame
from src.tasks.amp_perceptive.commands import command_from_state
from .config import (
    FORMAT, G1_HIKING_CALIBRATED_SENSOR_PROTOCOL, G1_HIKING_DR_SENSOR_PROTOCOL, LEGACY_WALKING_EXPERT,
    TRANSFORMER_ACTOR_TYPE, perception_spec_from_config,
)
from .perception import PerceptionSpec,DepthHistory,pack_actor_input,unpack_actor_input,advance_depth_ring
from .models import PerceptiveActor,TerrainPerceptiveWorldModel
from .expert import PerceptiveExpertPairs
from .replay import PerceptiveReplay
from .augmentation import augment_depth_frame
from .visual_metrics import depth_metrics, policy_view_loss_terms
from src.training_cost import WALL_TIME_SCOPE


MODULES=("actor","critic","world_model","discriminator","amp_norm")
OPTIMIZERS=("opt_actor","opt_critic","opt_model","opt_disc")
PHYSICAL_METRICS=("root_position_rmse_m","root_orientation_error_rad","root_velocity_rmse_m_s",
                  "root_angular_velocity_rmse_rad_s","joint_position_rmse_rad","joint_velocity_rmse_rad_s")


class PerceptiveRunner(WmacRunner):
    WM_FIELDS=(*WmacRunner.WM_FIELDS,"depth_frame","next_depth_frame","relative_height")
    VALIDATION_FIELDS=(*WM_FIELDS,"previous_action","command","reward")
    ACTOR_FIELDS=(*WmacRunner.ACTOR_FIELDS,"navigation_target","desired_speed","depth_frame","relative_height")

    @property
    def uses_latent(self):
        return getattr(self.cfg,"visual_mode","pixel_v2")=="instinct_latent_v1"

    @property
    def uses_fast_execution(self):
        return self.uses_latent and getattr(self.cfg,"execution_profile","legacy")=="supermimic"

    @property
    def uses_reference_schedule(self):
        return getattr(self.cfg,"training_schedule","validated_v2")=="supermimic_v1"

    @property
    def uses_hiking_amp(self):
        return getattr(self.cfg, "amp_feature_schema", None) == "hiking_v1"

    @property
    def uses_standing_objective(self):
        return getattr(self.cfg, "locomotion_objective", "legacy_walk_turn_v1") in ("hiking_standing_v1", "hiking_full_v1")

    @property
    def uses_full_hiking_task(self):
        return getattr(self.cfg,"locomotion_objective",None)=="hiking_full_v1"

    @property
    def uses_reference_policy_step(self):
        return self.uses_fast_execution and self.uses_full_hiking_task and self.uses_reference_schedule

    def _observation_noise(self,batch):
        if not self.uses_full_hiking_task:
            return super()._observation_noise(batch)
        scale=self.adapter.observation_noise_scale()
        return (2*torch.rand(self.cfg.imagination_horizon,batch,96,device=self.device)-1)*scale

    @property
    def validation_steps(self):
        return getattr(self.cfg,"validation_horizon",0) or self.cfg.wm_window

    def __init__(self,env,cfg,log_dir):
        self.spec=perception_spec_from_config(cfg)
        source_policy=cfg.init_policy
        cfg.init_policy=None
        try:
            super().__init__(env,cfg,log_dir)
        finally:
            cfg.init_policy=source_policy
        if cfg.dynamics_parameterization=="supermimic_gencoord_v1":
            if not math.isclose(float(env.step_dt),cfg.dynamics_dt,rel_tol=0.,abs_tol=1e-10):
                raise ValueError("GenCoord control dt differs from the physical environment")
            if env.cfg.decimation!=cfg.dynamics_decimation:
                raise ValueError("GenCoord decimation differs from the physical environment")
        if self.uses_latent:
            from .latent_models import LatentPerceptiveActor,LatentPerceptiveWorldModel
            from .instinct_latent import InstinctLatentDecoder
            if cfg.actor_type != TRANSFORMER_ACTOR_TYPE:
                self.actor=LatentPerceptiveActor(self.spec,cfg.latent_dim,gate_hidden=cfg.moe_gate_hidden).to(self.device)
            action_transform = None
            if cfg.hiking_training_version >= 3:
                action_term = env.action_manager.get_term("joint_pos")
                action_transform = tuple(self._from_env(value) for value in (action_term.scale, action_term.offset))
            self.world_model=LatentPerceptiveWorldModel(self.adapter.terrain,self.spec,
                hidden=cfg.model_hidden,latent_dim=cfg.latent_dim,
                context_dim=getattr(self.adapter,"context_dim",35),
                contact_dim=getattr(self.adapter,"contact_dim",2),
                dt=cfg.dynamics_dt,decimation=cfg.dynamics_decimation,
                continuous_normalization=cfg.hiking_training_version>=2,
                action_transform=action_transform,
                robot_dynamics_columns=getattr(self.adapter,"robot_dynamics_columns",0)).to(self.device)
            shape=(self.spec.height-self.spec.crop_top-self.spec.crop_bottom,
                   self.spec.width-self.spec.crop_left-self.spec.crop_right)
            self.latent_decoder=InstinctLatentDecoder(cfg.latent_dim,self.spec.visual_channels*self.spec.frames,shape,
                bounded_output=self.spec.depth_protocol!="g1_rpl_v1").to(self.device)
            self._model_training_modules=torch.nn.ModuleList((self.world_model,self.actor.vision,self.latent_decoder))
            self.WM_FIELDS=(*WmacRunner.WM_FIELDS,"depth_stack","next_depth_stack","relative_height")
            self.VALIDATION_FIELDS=(*self.WM_FIELDS,"previous_action","command","reward")
            self.ACTOR_FIELDS=(*WmacRunner.ACTOR_FIELDS,"navigation_target","desired_speed","depth_stack","relative_height")
        else:
            self.actor=PerceptiveActor(self.spec,gate_hidden=cfg.moe_gate_hidden).to(self.device)
            self.world_model=TerrainPerceptiveWorldModel(self.adapter.terrain,self.spec,
                hidden=cfg.model_hidden,dynamics_parameterization=cfg.dynamics_parameterization,
                dt=cfg.dynamics_dt,decimation=cfg.dynamics_decimation).to(self.device)
        if self.uses_standing_objective:
            self.ACTOR_FIELDS=(*self.ACTOR_FIELDS,"command_standing")
        if self.uses_full_hiking_task:
            self.ACTOR_FIELDS=(*self.ACTOR_FIELDS,"random_yaw_rate","random_velocity_mask")
        self.actor.precision=self.world_model.precision=cfg.precision
        self._critic_value_ceiling=None
        if self.uses_reference_schedule:
            from .reference_training import build_reference_critic,reference_value_ceiling
            self.critic=build_reference_critic(self.critic.norm.mean.numel(),cfg.critic_hidden,self.device)
            self.critic.precision=cfg.precision
            self._value_fn=self._compile(self.critic.forward,"reference critic")
            self._critic_value_ceiling=reference_value_ceiling(self.adapter,cfg)
        fused=self.device.type=="cuda"
        policy_parameters=self.actor.policy.parameters() if self.uses_latent else self.actor.parameters()
        model_parameters=self._model_training_modules.parameters() if self.uses_latent else self.world_model.parameters()
        if self.uses_fast_execution and cfg.execution_optimizer=="muon_adam":
            from .optimizers import build_reference_optimizer
            self.opt_actor=build_reference_optimizer([self.actor.policy],lr=cfg.actor_lr,
                betas=cfg.actor_betas,device=self.device)
            self.opt_model=build_reference_optimizer(list(self._model_training_modules),
                lr=cfg.world_model_lr,device=self.device)
            self.opt_critic=build_reference_optimizer([self.critic],lr=cfg.critic_lr,device=self.device)
        else:
            self.opt_actor=torch.optim.Adam(policy_parameters,lr=cfg.actor_lr,betas=cfg.actor_betas,fused=fused)
            self.opt_model=torch.optim.Adam(model_parameters,lr=cfg.world_model_lr,fused=fused)
        if self.uses_latent:
            if self.uses_fast_execution and cfg.replay_storage=="frame_fp16":
                from .centered_depth_replay import CenteredDepthReplay
                self.replay=CenteredDepthReplay(cfg.replay_steps,env.num_envs,cfg.replay_device,
                    cfg.validation_env_fraction,spec=self.spec)
            elif self.uses_fast_execution and cfg.replay_storage=="frame_uint8":
                from .compact_gpu_replay import CompressedLatentReplay
                replay_class=CompressedLatentReplay
                if self.uses_full_hiking_task and not self.spec.include_valid:
                    from .deduplicated_replay import DeduplicatedHikingReplay
                    replay_class=DeduplicatedHikingReplay
                replay_options={"spec":self.spec}
                if replay_class is not CompressedLatentReplay:
                    replay_options["context_dim"]=self.adapter.context_dim
                self.replay=replay_class(cfg.replay_steps,env.num_envs,cfg.replay_device,
                    cfg.validation_env_fraction,**replay_options)
            else:
                from .latent_replay import LatentPerceptiveReplay
                self.replay=LatentPerceptiveReplay(cfg.replay_steps,env.num_envs,cfg.replay_device,cfg.validation_env_fraction,spec=self.spec)
        else:
            self.replay=PerceptiveReplay(cfg.replay_steps,env.num_envs,"cpu",cfg.validation_env_fraction,spec=self.spec)
        if self.uses_hiking_amp:
            from rsl_rl.modules.discriminator import Discriminator
            from .hiking_amp import configure_replay
            self.discriminator = Discriminator(670, cfg.amp_reward_coef,
                list(cfg.discriminator_hidden), str(self.device), cfg.task_weight).to(self.device)
            self.amp_norm = torch.nn.Identity()
            # Hiking uses explicit L2 terms in its loss. AdamW's own decay
            # stays at its upstream default .01, with a single parameter group.
            self.opt_disc = torch.optim.AdamW(self.discriminator.parameters(), lr=cfg.discriminator_lr,
                betas=(.9,.999), weight_decay=.01, fused=fused)
            configure_replay(self.replay)
        self._train_env_mask=self.replay.train_env_mask.to(self.device)
        self._processed_history=self.uses_fast_execution
        history_spec=self.spec
        if self._processed_history:
            history_spec=replace(self.spec,height=self.spec.height-self.spec.crop_top-self.spec.crop_bottom,
                width=self.spec.width-self.spec.crop_left-self.spec.crop_right,
                crop_top=0,crop_bottom=0,crop_left=0,crop_right=0,blur_kernel=1)
        self.depth_history=DepthHistory(history_spec,env.num_envs,self.device)
        self.current_depth=self._from_env(read_depth_frame(env)).detach()
        self.depth_history.reset(self._policy_depth_frame(self.current_depth))
        if hasattr(self.replay,"bootstrap_reset") and not cfg.resume:
            frame=self.depth_history.ring[:,0]
            self.replay.bootstrap_reset(frame if self.spec.include_valid else frame[:,:1])
        if self.uses_fast_execution:
            self._setup_fast_execution()
        self.validation=self._empty_validation()
        self._ensure_training_progress()
        if source_policy:
            self.initialize_from_policy(source_policy)

    def _make_actor(self, cfg, obs_dim):
        if cfg.actor_type != TRANSFORMER_ACTOR_TYPE:
            return super()._make_actor(cfg, obs_dim)
        from .latent_models import LatentPerceptiveActor
        return LatentPerceptiveActor(self.spec, cfg.latent_dim, actor_type=cfg.actor_type)

    def _make_experts(self,cfg,env):
        # Avoid constructing the flat runner's mixed walk/recovery pool first.
        return PerceptiveExpertPairs(cfg,env,str(self.device),cfg.motion_root(Path(__file__).resolve().parents[2]))

    def update_discriminator(self):
        if not self.uses_hiking_amp:
            return super().update_discriminator()
        if self.uses_full_hiking_task and getattr(self.cfg,"hiking_training_version",1)>=2:
            return self._update_hiking_discriminator_rollout()
        from .hiking_amp import flatten_history
        size = self.cfg.effective_discriminator_batch_size
        records = []
        for _ in range(self.cfg.discriminator_updates):
            batch = _device_batch(self.replay.sample_starts(size, 0.,
                fields=("state", "next_state")), self.device)
            with torch.no_grad():
                state_history = torch.cat((batch["amp_state_history"][:,1:],
                                           batch["next_state"][:,None]),1)
                policy = flatten_history(self.adapter.amp_features(state_history))
                expert = flatten_history(self.experts.sample_histories(size))
            examples = torch.cat((policy, expert),0).detach().requires_grad_(True)
            score = self.discriminator(examples).squeeze(-1)
            policy_score, expert_score = score[:size], score[size:]
            prediction = .5*((policy_score+1).square().mean()+(expert_score-1).square().mean())
            grad = torch.autograd.grad(score.sum(), examples, create_graph=True)[0]
            penalty = self.cfg.gradient_penalty*grad.square().sum(-1).mean()
            regularization = self.cfg.discriminator_trunk_weight_decay*sum(
                p.square().sum() for p in self.discriminator.parameters())
            regularization = regularization+self.cfg.discriminator_head_weight_decay*self.discriminator.amp_linear.weight.square().sum()
            loss = prediction+penalty+regularization
            self.opt_disc.zero_grad(set_to_none=True)
            loss.backward()
            # Upstream Wasabi updates its discriminator without gradient clipping.
            norm = torch.linalg.vector_norm(torch.stack(tuple(
                p.grad.detach().norm() for p in self.discriminator.parameters() if p.grad is not None)))
            self.opt_disc.step()
            records.append(torch.stack((loss.detach(), penalty.detach(), norm)))
        self.opt_disc.zero_grad(set_to_none=True)
        result = torch.stack(records).mean(0).cpu().tolist()
        return dict(zip(("discriminator_loss", "discriminator_grad_penalty", "discriminator_grad_norm"), result))

    def _update_hiking_discriminator_rollout(self):
        from .hiking_amp_budget import (HikingAmpBudget,HikingAmpFeatureCache,HikingAmpPolicyCache,HikingEpisodeReference,
            HikingAmpReferenceCache,LEARNING_EPOCHS,MINI_BATCHES,ROLLOUT_STEPS,
            discriminator_step,sample_reference_ends)
        post_step_observations = getattr(self.cfg, "hiking_training_version", 1) >= 3
        shared_expert_pool = getattr(self.cfg, "walking_expert_source", "none") == LEGACY_WALKING_EXPERT
        if getattr(self,"_hiking_amp_budget_replay",None) is not self.replay:
            self._hiking_amp_budget_replay=self.replay
            self._hiking_amp_budget=HikingAmpBudget()
            self._amp_episode_reference = (HikingEpisodeReference(
                self.experts,self.replay.num_envs,self.env.step_dt)
                if post_step_observations and not shared_expert_pool else None)
        budget=self._hiking_amp_budget
        final=self.iteration==self.cfg.iterations-1
        rollouts=budget.claim(self.replay,flush=final and not post_step_observations)
        microbatch_size=self.cfg.effective_discriminator_batch_size
        records=[]
        windows=invalid=0
        for rollout in rollouts:
            windows+=rollout.size;invalid+=rollout.invalid_windows
            playback=self._amp_episode_reference
            if playback is not None:
                playback.advance(rollout,self.replay)
            if not rollout.size:
                continue
            if not hasattr(self,"_hiking_amp_feature_encoder"):
                encoder=self.adapter.amp_features
                if self.device.type=="cuda" and getattr(self.cfg,"compile",False):
                    encoder=torch.compile(encoder,mode=getattr(self.cfg,"compile_mode","default"),
                        fullgraph=getattr(self.cfg,"compile_fullgraph",False),dynamic=False)
                self._hiking_amp_feature_encoder=encoder
            execution=getattr(self,"_hiking_amp_gp_execution",None)
            graph_enabled=self.device.type=="cuda" and getattr(self.cfg,"cuda_graphs",False)
            features=HikingAmpFeatureCache(self.adapter,self.replay,rollout,self.device,microbatch_size,
                feature_chunk_size=4096,feature_fn=self._hiking_amp_feature_encoder,
                post_step_observations=post_step_observations)
            policy_cache=HikingAmpPolicyCache(features,rollout)
            del features
            if playback is not None:
                reference_cache=playback
                expert_ends=torch.arange(rollout.size,device=self.device)
            else:
                reference_cache=getattr(self,"_hiking_amp_reference_cache",None)
                if reference_cache is None or reference_cache.experts is not self.experts:
                    reference_cache=HikingAmpReferenceCache(self.experts,self.device)
                    self._hiking_amp_reference_cache=reference_cache
                expert_ends=sample_reference_ends(self.experts,rollout.size)
            logical_size=ROLLOUT_STEPS*self.replay.num_envs//MINI_BATCHES
            full_fixed=(graph_enabled and rollout.end-rollout.start==ROLLOUT_STEPS
                and not rollout.invalid_windows and logical_size%microbatch_size==0)
            logical=getattr(self,"_hiking_amp_logical_execution",None)
            if full_fixed and logical is None:
                from .amp_execution import AmpLogicalBatchExecution
                logical=AmpLogicalBatchExecution(self.discriminator,self.opt_disc,logical_size,microbatch_size,
                    gradient_penalty=self.cfg.gradient_penalty,
                    trunk_weight_decay=self.cfg.discriminator_trunk_weight_decay,
                    head_weight_decay=self.cfg.discriminator_head_weight_decay,
                    warmup=getattr(self.cfg,"graph_warmup",3))
                self._hiking_amp_logical_execution=logical
            for selection in rollout.minibatches():
                if full_fixed:
                    records.append(logical.step(policy_cache.select(selection),
                        reference_cache.select(expert_ends[selection.to(expert_ends.device)])))
                    continue
                if execution is None and graph_enabled:
                    from .amp_execution import AmpMicrobatchGraph
                    execution=AmpMicrobatchGraph(self.discriminator,microbatch_size,self.cfg.gradient_penalty)
                    self._hiking_amp_gp_execution=execution
                def pairs():
                    for selected in selection.split(microbatch_size):
                        yield (policy_cache.select(selected),
                            reference_cache.select(expert_ends[selected.to(expert_ends.device)]))
                records.append(discriminator_step(self.discriminator,self.opt_disc,pairs(),len(selection),
                    gradient_penalty=self.cfg.gradient_penalty,
                    trunk_weight_decay=self.cfg.discriminator_trunk_weight_decay,
                    head_weight_decay=self.cfg.discriminator_head_weight_decay,execution=execution))
        self.opt_disc.zero_grad(set_to_none=True)
        result=(torch.stack(records).mean(0).cpu().tolist() if records else [0.,0.,0.])
        metrics=dict(zip(("discriminator_loss","discriminator_grad_penalty","discriminator_grad_norm"),result))
        metrics.update(discriminator_updates_applied=len(records),amp_rollout_blocks=len(rollouts),
            amp_rollout_pending_steps=self.replay.total_steps-budget.next_time,
            amp_policy_windows=windows,amp_policy_exposures=LEARNING_EPOCHS*windows,
            amp_invalid_windows_dropped=invalid,amp_policy_windows_total=budget.windows,
            amp_policy_exposures_total=LEARNING_EPOCHS*budget.windows,
            amp_invalid_windows_dropped_total=budget.invalid_windows)
        return metrics

    def _motion_phases(self):
        return ("walk",) if getattr(self.cfg,"motion_scope","joint_recovery_v1")=="forward_walk_turn_v1" else ("walk","recovery")

    def _recovery_mask(self,batch):
        if self.cfg.motion_scope=="forward_walk_turn_v1":
            return torch.zeros_like(batch["relative_height"],dtype=torch.bool)
        return (batch["relative_height"]<.55)|(batch["context"][...,34]>0)

    def _augment_depth(self,frame):
        return augment_depth_frame(frame,std_m=self.cfg.depth_noise_std_m,
            dropout_probability=self.cfg.depth_dropout_probability,max_depth=self.spec.max_depth)

    def _policy_depth_frame(self,frame):
        if self.spec.depth_protocol=="g1_rpl_v1":
            # The physical stream has already sampled corruption, resized and
            # applied its camera-tick delay. Actor and WM share this exact draw.
            return frame
        value=self._augment_depth(frame)
        if getattr(self,"_processed_history",False):
            from .visual_metrics import preprocess_depth_channels
            value=preprocess_depth_channels(value,self.spec,channels=2)
            if not self.spec.include_valid:
                # Official Hiking clips only after crop and Gaussian blur.
                value=torch.cat((value[:,:1].clamp(0,1),value[:,1:]),dim=1)
            return value
        return value

    def _setup_fast_execution(self):
        from .compiled_imagination import LatentImaginedStep,LatentTrajectoryTerms,CriticLoss
        from .latent_execution import WMExecution
        self._latent_critic_loss_module=CriticLoss(self.critic)
        if self.uses_reference_policy_step:
            from .reference_policy_step import HikingWorldModelStep
            self._reference_step_module=HikingWorldModelStep(self)
            self._reference_step_fn=self._compile(self._reference_step_module,"reference WM environment step")
        else:
            self._latent_step_module=LatentImaginedStep(self)
            self._latent_trajectory_module=LatentTrajectoryTerms(self)
            self._latent_step_fn=self._compile(self._latent_step_module,"latent imagination")
            self._latent_trajectory_fn=self._compile(self._latent_trajectory_module,"trajectory FK/reward")
        self._latent_critic_loss_fn=self._compile(self._latent_critic_loss_module,"critic loss")
        graph_mode="graph_full" if self.cfg.execution_optimizer=="muon_adam" else "graph"
        resources=None
        if self.uses_reference_schedule and self.cfg.cuda_graphs:
            resources=(torch.cuda.Stream(device=self.device),torch.cuda.graph_pool_handle())
        self._reference_graph_resources=resources
        self._latent_wm_execution=WMExecution(self,mode=(graph_mode if self.cfg.cuda_graphs
            else "compile" if self.cfg.compile else "eager"),warmup=self.cfg.graph_warmup,
            graph_resources=resources)
        self._latent_actor_execution=self._latent_critic_execution=None
        self._latent_actor_critic_macro_execution=None
        if (self.cfg.cuda_graphs and self.cfg.execution_optimizer=="muon_adam"
                and self.uses_reference_schedule and self.cfg.hiking_training_version>=2):
            from .wm_macro_execution import WMMacroExecution
            from .macro_execution import ActorCriticMacroExecution
            self._latent_wm_execution=WMMacroExecution(self,graph_resources=resources)
            if not self.cfg.actor_microbatch_size or self.cfg.actor_microbatch_size>=self.cfg.actor_batch_size:
                self._latent_actor_critic_macro_execution=ActorCriticMacroExecution(self,graph_resources=resources)
                return
        if self.cfg.cuda_graphs and self.cfg.execution_optimizer=="muon_adam":
            from .captured_updates import ActorExecution,make_critic_execution
            self._latent_actor_execution=ActorExecution(self,graph_resources=resources)
            self._latent_critic_execution=make_critic_execution(self,graph_resources=resources)

    def _fit_critic(self,x,targets):
        if self.uses_fast_execution:
            from .compiled_imagination import fit_critic_fast
            return fit_critic_fast(self,x,targets)
        return super()._fit_critic(x,targets)

    def learn(self):
        if self.cfg.hiking_training_version < 3:
            return super().learn()
        from .reference_loop import learn_reference
        try:
            learn_reference(self)
        finally:
            self.writer.close()
            if self.csv_file:
                self.csv_file.close()

    def _next_collect_count(self):
        if self.uses_reference_schedule and not self.cfg.collect_steps and self.cfg.utd<=1.:
            # Source Ratio schedules updates after vector steps 9,11,14,...
            # at its default ratio .4 (earlier events lack replay frames).
            current=self.sim_steps//self.env.num_envs
            completed=math.floor(max(0,current-1)*self.cfg.utd+1e-9)
            next_step=math.ceil((completed+1)/self.cfg.utd-1e-9)+1
            return max(1,next_step-current)
        return super()._next_collect_count()

    def _optimizer_step(self,loss,optimizer,module,actor=False,backward=True):
        if not self.uses_reference_schedule or optimizer is self.opt_disc:
            return super()._optimizer_step(loss,optimizer,module,actor=actor,backward=backward)
        if backward:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(module.parameters(),self.cfg.max_grad_norm,foreach=True)
        if actor and self.cfg.actor_grad_value_clip:
            torch.nn.utils.clip_grad_value_(module.parameters(),self.cfg.actor_grad_value_clip,foreach=True)
        optimizer.step()
        return norm.detach(),True

    def _fingerprints(self):
        terrain=self.adapter.terrain
        h=hashlib.sha256()
        for name in ("heights","centers","half_sizes"):
            value=getattr(terrain,name).detach().cpu().contiguous()
            h.update(name.encode());h.update(str(tuple(value.shape)).encode());h.update(value.numpy().tobytes())
        # Support heights alone omit the physical walls that change pixels and
        # contacts. Include their numerical geometry, excluding random names.
        model=self.env.sim.mj_model
        import mujoco
        body_id=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,"terrain")
        if body_id<0:raise ValueError("Terrain body missing from the compiled collision model")
        ids=np.flatnonzero(model.geom_bodyid==body_id)
        for name in ("geom_type","geom_pos","geom_quat","geom_size","geom_contype","geom_conaffinity"):
            value=np.ascontiguousarray(getattr(model,name)[ids])
            h.update(name.encode());h.update(str(value.shape).encode());h.update(value.tobytes())
        if self.cfg.hiking_training_version >= 3:
            # The verifier at environment/expert setup checks both actual files.
            # Only their content defines this protocol across machines.
            from .locomotion_data import HIKING_SOURCE_SHA256
            source = {name: digest for name, digest in HIKING_SOURCE_SHA256.items()
                      if name.endswith((".yaml", ".npz"))}
            if self.cfg.walking_expert_source == LEGACY_WALKING_EXPERT:
                source[LEGACY_WALKING_EXPERT] = self.experts.legacy_walking_sha256
            return h.hexdigest(), hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()
        manifest=self.cfg.motion_root(Path(__file__).resolve().parents[2])/"mirror_manifest.json"
        return h.hexdigest(),hashlib.sha256(manifest.read_bytes()).hexdigest()

    @torch.no_grad()
    def collect(self,count):
        if self.uses_latent:
            from .latent_training import collect_latent
            return collect_latent(self,count)
        totals=dict(real_task_reward=0.,terminated_fraction=0.,timeout_fraction=0.,recovery_fraction=0.,
                    boundary_fraction=0.,invalid_transition_fraction=0.,sensor_valid_fraction=0.,standing_command_fraction=0.)
        for _ in range(count):
            proprio=self._from_env(self.env.obs_buf["actor"])
            ring=self.depth_history.snapshot()
            packed=pack_actor_input(torch.nan_to_num(proprio),ring,self.spec)
            action=(self.actor(packed)+self.cfg.exploration_noise*torch.randn(self.env.num_envs,29,device=self.device))
            action=action.clamp(-self.cfg.action_limit,self.cfg.action_limit)
            if not bool(torch.isfinite(action).all()):raise RuntimeError("Non-finite perceptive policy action")
            current=self.current_depth.detach().clone()
            self.env.step(self._to_env(action))
            batch=self._from_env(self.env.transition)
            if "command_standing" in batch:
                totals["standing_command_fraction"]+=float(batch["command_standing"].float().mean())
            batch.update(depth_ring=ring.to(torch.float16),depth_frame=current.to(torch.float16))
            batch["next_depth_frame"]=batch["next_depth_frame"].to(torch.float16)
            accepted=self.replay.add(batch)
            train=accepted&self._train_env_mask
            self.world_model.update_normalizers(batch["state"][train],batch["action"][train],batch["context"][train],
                batch["next_state"][train],depth_frame=current[train])
            self.critic.norm.update(self.adapter.critic_features(batch["state"][train],batch["previous_action"][train],
                batch["command"][train],batch["context"][train]))
            # next_depth_frame is BEFORE reset, never replace that model label.
            # Only the next policy's ring is reset with its new-episode image.
            self.current_depth=self._from_env(read_depth_frame(self.env)).detach()
            done=batch["terminated"]|batch["timeout"]
            self.depth_history.append(self._augment_depth(self.current_depth),reset_mask=done)
            self.sim_steps+=self.env.num_envs
            totals["real_task_reward"]+=float(batch["reward"][accepted].mean()) if accepted.any() else 0.
            for name,key in (("terminated_fraction","terminated"),("timeout_fraction","timeout"),("boundary_fraction","boundary")):
                totals[name]+=float(batch[key].float().mean())
            totals["recovery_fraction"]+=float(self._recovery_mask(batch).float().mean())
            totals["invalid_transition_fraction"]+=float((~accepted).float().mean())
            totals["sensor_valid_fraction"]+=float(current[:,1].mean())
        return {key:value/max(count,1) for key,value in totals.items()}

    def _rollout_model_metrics(self,batch):
        if self.uses_latent:
            from .latent_training import rollout_latent_model_metrics
            return rollout_latent_model_metrics(self,batch)
        state=batch["state"][:,0]
        frame=batch["depth_frame"][:,0].float()
        records=[]
        for step in range(self.cfg.wm_window):
            active=batch["mask"][:,step].bool()
            state=torch.where(active[:,None],state,batch["state"][:,step])
            frame=torch.where(active[:,None,None,None],frame,batch["depth_frame"][:,step].float())
            state,contact,frame=self.world_model.step_with_perception(state,batch["action"][:,step],batch["context"][:,step],frame)
            target=batch["next_depth_frame"][:,step].float()
            terms=(policy_view_loss_terms(frame,target,self.spec)
                   if self.cfg.perception_training_version>=2 else self.world_model.pixel_loss_terms(frame,target))
            with torch.no_grad():
                visual=depth_metrics(frame.detach(),target,self.spec)
                truth=batch["next_state"][:,step]
                from src.wmac.networks import _quat_to_rotvec
                from src.wmac.state import quat_inv,quat_mul
                errors=[(state[...,section]-truth[...,section]).square().mean(-1).sqrt()
                        for section in (slice(0,3),slice(7,10),slice(10,13),slice(13,42),slice(42,71))]
                orientation=_quat_to_rotvec(quat_mul(quat_inv(truth[...,3:7]),state[...,3:7])).norm(dim=-1)
                physical=dict(zip(PHYSICAL_METRICS,(errors[0],orientation,*errors[1:])))
                finite=(torch.isfinite(state).all(-1)&torch.isfinite(contact).all(-1)&
                        torch.isfinite(frame).flatten(1).all(-1))
            records.append(dict(
                model_error=self.world_model.loss(state,batch["next_state"][:,step]),
                contact_error=F.smooth_l1_loss(torch.log1p(contact),torch.log1p(batch["contact_cost"][:,step].clamp_min(0)),reduction="none").mean(-1),
                depth_loss=terms["depth"],valid_loss=terms["valid"],finite=finite.float(),**visual,**physical))
        return {key:torch.stack([row[key] for row in records],1) for key in records[0]}

    def _perceptive_model_errors(self,batch):
        """Compatibility interface for the original full-image diagnostics."""
        values=self._rollout_model_metrics(batch)
        return (values["model_error"],values["contact_error"],values["depth_loss"]+values["valid_loss"],
                values["full_rmse"],values["full_mask_error"])

    @staticmethod
    def _masked_mean(values,mask,*,empty=float("inf")):
        selected=values[mask.bool()]
        return selected.mean() if selected.numel() else values.new_tensor(empty)

    def fit_model(self):
        if self.uses_latent:
            from .latent_training import fit_latent_model
            return fit_latent_model(self)
        records=[]
        for _ in range(self.cfg.wm_updates):
            batch=_device_batch(self.replay.sample_windows(self.cfg.wm_batch_size,self.cfg.wm_window,
                self.cfg.recovery_fraction,fields=self.WM_FIELDS),self.device)
            values=self._rollout_model_metrics(batch)
            errors,contacts=values["model_error"],values["contact_error"]
            mask=batch["mask"];denom=mask.sum().clamp_min(1)
            dyn=self._masked_mean(errors,mask);contact=self._masked_mean(contacts,mask)
            depth_loss=self._masked_mean(values["depth_loss"],mask)
            valid_loss=self._masked_mean(values["valid_loss"],mask)
            mask_coef=(self.cfg.depth_mask_loss_coef if self.cfg.perception_training_version>=2 else self.cfg.depth_loss_coef)
            visual_loss=self.cfg.depth_loss_coef*depth_loss+mask_coef*valid_loss
            norm,ok=self._optimizer_step(dyn+contact+visual_loss,self.opt_model,self.world_model)
            recovery=self._recovery_mask(batch)[:,0]
            per=torch.stack([self._masked_mean(value,valid) for value,valid in zip(errors.detach(),mask)])
            selections={"walk":~recovery,"recovery":recovery}
            phases=[per[selections[name]].mean() if selections[name].any() else dyn.new_tensor(float("nan"))
                    for name in self._motion_phases()]
            last=self._masked_mean(errors[:,-1].detach(),mask[:,-1])
            prefix=[self._masked_mean(errors.detach()[:,:h],mask[:,:h]) for h in range(1,self.cfg.wm_window+1)]
            records.append(torch.stack((dyn.detach(),contact.detach(),last,norm,*phases,dyn.new_tensor(float(ok)),
                (depth_loss+valid_loss).detach(),self._masked_mean(values["full_rmse"],mask),
                self._masked_mean(values["full_mask_error"],mask),depth_loss.detach(),valid_loss.detach(),
                visual_loss.detach(),self._masked_mean(values["actor_rmse"],mask),*prefix,
                *(self._masked_mean(values[name],mask) for name in PHYSICAL_METRICS))))
        values=torch.stack(records).mean(0).detach().cpu().tolist()
        result=dict(zip(("wm_loss","wm_contact_loss","wm_last_error","wm_grad_norm",
            *(f"wm_{phase}_error" for phase in self._motion_phases()),
            "wm_updates_applied_fraction","wm_depth_loss","wm_depth_rmse","wm_depth_mask_error",
            "wm_depth_regression_loss","wm_valid_loss","wm_visual_loss","wm_actor_depth_rmse",
            *(f"wm_h{h}_model_error" for h in range(1,self.cfg.wm_window+1)),
            *(f"wm_{name}" for name in PHYSICAL_METRICS)),values))
        self.last_model_loss=result["wm_loss"]
        finite=[result[k] for k in ("wm_loss",*(f"wm_{phase}_error" for phase in self._motion_phases())) if math.isfinite(result[k])]
        result["wm_gate_error"]=max(finite) if math.isfinite(result["wm_loss"]) else float("inf")
        return result

    @torch.no_grad()
    def validate_model(self):
        if self.uses_reference_schedule and self.replay.size<self.validation_steps:
            return self.validation
        batch=_device_batch(self.replay.sample_windows(self.cfg.validation_batch_size,self.validation_steps,
            self.cfg.recovery_fraction,fields=self.VALIDATION_FIELDS,split="validation"),self.device)
        values=self._rollout_model_metrics(batch)
        errors,contacts,depth,mask_error=(values[key] for key in ("model_error","contact_error","full_rmse","full_mask_error"))
        mask=batch["mask"];denom=mask.sum().clamp_min(1)
        recovery=self._recovery_mask(batch)[:,0]
        fast_diagnostics=getattr(self.cfg,"hiking_training_version",1)>=2
        if fast_diagnostics:
            from .diagnostics import masked_mean,validation_tensors,read_metrics
            per=masked_mean(errors,mask,dim=1)
            per_depth=masked_mean(depth,mask,dim=1)
        else:
            per=torch.stack([self._masked_mean(value,valid) for value,valid in zip(errors,mask)])
            per_depth=torch.stack([self._masked_mean(value,valid) for value,valid in zip(depth,mask)])
        # Separate one-step reward-head agreement from the free-running model
        # error. A missing measurement must never be reported as a perfect0.
        starts=batch["state"].flatten(0,1);actions=batch["action"].flatten(0,1)
        contexts=batch["context"].flatten(0,1)
        if self.uses_latent:
            latent=self.actor.encode_processed(batch["depth_stack"].flatten(0,1).float())
            predicted,pcontact,_=self.world_model.step_with_latent(starts,actions,contexts,latent)
        else:
            predicted,pcontact,_=self.world_model.step_with_perception(starts,actions,contexts,batch["depth_frame"].flatten(0,1).float())
        nc,terminated=(self.adapter.advance_context(predicted,contexts,action=actions,contact=pcontact)
            if self.uses_full_hiking_task else self.adapter.advance_context(predicted,contexts))
        task=self.adapter.task_reward(starts,predicted,actions,batch["previous_action"].flatten(0,1),
            batch["command"].flatten(0,1),nc,terminated,pcontact).reshape_as(mask)
        gap=(masked_mean((task-batch["reward"]).abs(),mask) if fast_diagnostics
             else self._masked_mean((task-batch["reward"]).abs(),mask))
        if fast_diagnostics:
            result=dict(validation_model_error=masked_mean(errors,mask),
                validation_contact_error=masked_mean(contacts,mask),validation_reward_gap=gap,
                validation_depth_rmse=masked_mean(depth,mask),validation_depth_mask_error=masked_mean(mask_error,mask))
            selections={"walk":~recovery,"recovery":recovery}
            for name in self._motion_phases():
                selected=selections[name]
                result[f"validation_{name}_error"]=masked_mean(per,selected)
                result[f"validation_{name}_depth_rmse"]=masked_mean(per_depth,selected)
                result[f"validation_{name}_samples"]=selected.sum()
            result.update(validation_tensors(self,values,mask,recovery,PHYSICAL_METRICS))
            self.validation=read_metrics(result)
            return self.validation
        result=dict(validation_model_error=float(self._masked_mean(errors,mask)),
            validation_contact_error=float(self._masked_mean(contacts,mask)),validation_reward_gap=float(gap),
            validation_depth_rmse=float(self._masked_mean(depth,mask)),validation_depth_mask_error=float(self._masked_mean(mask_error,mask)))
        selections={"walk":~recovery,"recovery":recovery}
        for name in self._motion_phases():
            selection=selections[name]
            result[f"validation_{name}_error"]=float(per[selection].mean()) if selection.any() else float("inf")
            result[f"validation_{name}_depth_rmse"]=float(per_depth[selection].mean()) if selection.any() else float("inf")
            result[f"validation_{name}_samples"]=float(selection.sum())
        result.update(self._validation_diagnostics(values,mask,recovery))
        self.validation=result
        return result

    def _empty_validation(self):
        shape=(2,self.validation_steps)
        names=("model_error","contact_error","full_rmse","actor_rmse","outside_rmse", "full_mask_error",
               "actor_mask_error","actor_valid_pixels","full_valid_pixels","outside_valid_pixels","finite",*PHYSICAL_METRICS)
        if self.uses_latent:names=(*names,"latent_error","latent_persistence_error")
        values={key:torch.full(shape,float("inf")) for key in names}
        result=dict(validation_model_error=float("inf"),validation_contact_error=float("inf"),
                    validation_reward_gap=float("inf"),validation_depth_rmse=float("inf"),
                    validation_depth_mask_error=float("inf"),validation_walk_error=float("inf"),
                    validation_recovery_error=float("inf"),validation_walk_depth_rmse=float("inf"),
                    validation_recovery_depth_rmse=float("inf"),validation_walk_samples=0.,validation_recovery_samples=0.)
        result.update(self._validation_diagnostics(values,torch.zeros(shape),torch.tensor([False,True])))
        if self.cfg.motion_scope=="forward_walk_turn_v1":
            result={key:value for key,value in result.items() if "recovery" not in key}
        return result

    def _validation_diagnostics(self,values,mask,recovery):
        """Every key exists from iteration one, including unsupported horizons.

        Step metrics describe free-running predictions, not teacher-forced
        one-step predictions. Horizon metrics take the worst step mean in the
        prefix so an accurate first image cannot hide a bad final prediction.
        Empty depth regions report their sample counts and cannot promote H.
        """
        from .diagnostics import validation_tensors,read_metrics
        if getattr(self.cfg,"hiking_training_version",1)<2:
            return self._legacy_validation_diagnostics(values,mask,recovery)
        return read_metrics(validation_tensors(self,values,mask,recovery,PHYSICAL_METRICS))

    def _legacy_validation_diagnostics(self,values,mask,recovery):
        """Historical scalar oracle retained for diagnostics parity checks."""
        result={}
        for name in PHYSICAL_METRICS:
            if name in values:
                result[f"validation_{name}"]=float(self._masked_mean(values[name],mask))
        if self.uses_latent:
            for name in ("latent_error","latent_persistence_error"):
                result[f"validation_{name}"]=float(self._masked_mean(values[name],mask))
        for region in ("full","actor","outside"):
            valid=mask.bool()&(values[f"{region}_valid_pixels"]>0)
            missing=float("nan") if self.uses_latent and region!="actor" else 0.
            result[f"validation_{region}_depth_rmse"]=float(self._masked_mean(values[f"{region}_rmse"],valid,empty=missing))
            result[f"validation_{region}_depth_samples"]=float(valid.sum())
        selections={"walk":~recovery,"recovery":recovery}
        for phase in self._motion_phases():
            selection=selections[phase]
            metric_names=("model_error","contact_error","actor_depth_rmse","actor_mask_error")
            if self.uses_latent:metric_names=(*metric_names,"latent_error","latent_persistence_error")
            prefix_metrics={name:[] for name in metric_names}
            for index in range(values["model_error"].shape[1]):
                valid=selection&mask[:,index].bool()
                depth_valid=valid&(values["actor_valid_pixels"][:,index]>0)
                step_values={
                    "model_error":float(self._masked_mean(values["model_error"][:,index],valid)),
                    "contact_error":float(self._masked_mean(values["contact_error"][:,index],valid)),
                    "actor_depth_rmse":float(self._masked_mean(values["actor_rmse"][:,index],depth_valid,empty=0.)),
                    "actor_mask_error":float(self._masked_mean(values["actor_mask_error"][:,index],valid))}
                if self.uses_latent:
                    for name in ("latent_error","latent_persistence_error"):
                        step_values[name]=float(self._masked_mean(values[name][:,index],valid))
                for key,value in step_values.items():prefix_metrics[key].append(value)
                step=index+1
                for region in ("full","actor","outside"):
                    region_valid=valid&(values[f"{region}_valid_pixels"][:,index]>0)
                    missing=float("nan") if self.uses_latent and region!="actor" else 0.
                    result[f"validation_{phase}_step{step}_{region}_depth_rmse"]=float(
                        self._masked_mean(values[f"{region}_rmse"][:,index],region_valid,empty=missing))
                    result[f"validation_{phase}_step{step}_{region}_depth_samples"]=float(region_valid.sum())
                prefix=f"validation_{phase}_h{step}_"
                result[prefix+"samples"]=float(valid.sum())
                result[prefix+"depth_samples"]=float(depth_valid.sum())
                prefix_valid=selection[:,None]&mask[:,:step].bool()
                # A CUDA float32 mean of all-one flags can round below 1.
                # Count exactly so only actual invalid predictions shorten H.
                prefix_count=int(prefix_valid.sum())
                finite_count=int((prefix_valid&(values["finite"][:,:step]==1)).sum())
                result[prefix+"finite_fraction"]=finite_count/prefix_count if prefix_count else 0.
                for key,sequence in prefix_metrics.items():
                    result[prefix+key]=max(sequence) if all(math.isfinite(x) for x in sequence) else float("inf")
        return result

    def _ensure_training_progress(self):
        defaults=dict(actor_updates_accepted=0,actor_updates_rejected=0,
                      _last_actor_update_iteration=None,_progress_start_iteration=getattr(self,"iteration",0),
                      _last_stall_warning_iteration=None,_active_horizon=self.cfg.imagination_horizon_start,
                      perception_training_history=[])
        for key,value in defaults.items():
            if not hasattr(self,key):setattr(self,key,value)

    def _effective_horizon(self):
        if self.uses_reference_schedule:return self.cfg.imagination_horizon
        if self.cfg.perception_training_version==1:return self.cfg.imagination_horizon
        self._ensure_training_progress()
        return self._active_horizon

    def _actor_gate(self,row):
        self._ensure_training_progress()
        if self.uses_reference_schedule:
            self._active_horizon=self.cfg.imagination_horizon
            row.update(active_imagination_horizon=self._active_horizon,gate_reason="supermimic_schedule")
            return True
        until=getattr(self,"world_model_warmup_until",None)
        if until is not None and self.iteration<until:
            calibrating=(self.cfg.critic_bootstrap and self.critic_calibration_updates<self.cfg.critic_warmup_iterations
                         and self.iteration>=until-self.cfg.critic_warmup_iterations)
            if not calibrating:
                row.update(active_imagination_horizon=self._effective_horizon(),gate_reason="world_model_warmup",
                           gate_training_finite=1.,gate_model_updates=1.,gate_samples=1.,gate_dynamics=1.,
                           gate_prediction_finite=1.,gate_depth_progression_limited=0.)
                return False
        if self.cfg.perception_training_version>=2:return self._adaptive_actor_gate(row)
        if not super()._actor_gate(row):return False
        for phase in ("walk","recovery"):
            minimum=getattr(self.cfg,f"validation_min_{phase}_samples")
            count=self.validation.get(f"validation_{phase}_samples",0)
            if not math.isfinite(count) or count<minimum:return False
            physical=self.validation.get(f"validation_{phase}_error",float("inf"))
            if not math.isfinite(physical) or physical>self.cfg.max_validation_model_error:return False
            depth=self.validation.get(f"validation_{phase}_depth_rmse",float("inf"))
            if not math.isfinite(depth) or depth>self.cfg.max_validation_depth_error:return False
        mask=self.validation["validation_depth_mask_error"]
        return math.isfinite(mask) and mask<=self.cfg.max_validation_mask_error

    def _adaptive_actor_gate(self,row):
        def check(horizon):
            training_horizon=min(horizon,self.cfg.wm_window)
            train=row.get(f"wm_h{training_horizon}_model_error",float("inf"))
            if not math.isfinite(train) or train>self.cfg.max_model_loss_for_actor:return "training_dynamics"
            if row.get("wm_updates_applied_fraction",0.)!=1.:return "model_optimizer"
            for phase in self._motion_phases():
                prefix=f"validation_{phase}_h{horizon}_"
                samples=self.validation.get(prefix+"samples",0.)
                if not math.isfinite(samples) or samples<getattr(self.cfg,f"validation_min_{phase}_samples"):return "validation_samples"
                finite=self.validation.get(prefix+"finite_fraction",0.)
                if not math.isfinite(finite) or finite!=1.:return "nonfinite_prediction"
                for metric,limit in (("model_error",self.cfg.max_validation_model_error),
                                     ("contact_error",self.cfg.max_validation_contact_error)):
                    value=self.validation.get(prefix+metric,float("inf"))
                    if not math.isfinite(value) or value>limit:return "validation_dynamics"
                visual_metrics=("latent_error",) if self.uses_latent else ("actor_depth_rmse","actor_mask_error")
                for metric in visual_metrics:
                    if not math.isfinite(self.validation.get(prefix+metric,float("inf"))):return "nonfinite_visual"
                if self.uses_latent and self.cfg.max_validation_latent_error is not None:
                    if self.validation[prefix+"latent_error"]>self.cfg.max_validation_latent_error:return "validation_latent"
            return "ready"

        first=self.cfg.imagination_horizon_start
        candidates=[first]
        while candidates[-1]<self.cfg.imagination_horizon:
            candidates.append(min(self.cfg.imagination_horizon,candidates[-1]*2))
        allowed=min(len(candidates)-1,self.actor_updates_accepted//self.cfg.horizon_growth_updates)
        selected=first;reason=check(first);limited=False
        if reason=="ready":
            for horizon in candidates[1:allowed+1]:
                if check(horizon)!="ready":limited=True;break
                trusted=True
                for phase in self._motion_phases():
                    metric="latent_error" if self.uses_latent else "actor_depth_rmse"
                    sample_metric="samples" if self.uses_latent else "depth_samples"
                    baseline=self.validation.get(f"validation_{phase}_h1_{metric}",float("inf"))
                    count=self.validation.get(f"validation_{phase}_h1_{sample_metric}",0.)
                    count_h=self.validation.get(f"validation_{phase}_h{horizon}_{sample_metric}",0.)
                    error=self.validation[f"validation_{phase}_h{horizon}_{metric}"]
                    ratio=self.cfg.latent_rollout_error_ratio if self.uses_latent else self.cfg.depth_rollout_error_ratio
                    # No-hit observations are legitimate at H1, but do not
                    # establish evidence that future visible images are good.
                    minimum=getattr(self.cfg,f"validation_min_{phase}_samples")
                    if (not all(math.isfinite(x) for x in (baseline,count,count_h))
                            or min(count,count_h)<minimum or error>baseline*ratio):
                        trusted=False;break
                if not trusted:limited=True;break
                selected=horizon
        self._active_horizon=selected
        row.update(active_imagination_horizon=selected,gate_reason=reason,
                   gate_training_finite=float(reason!="training_dynamics"),
                   gate_model_updates=float(reason!="model_optimizer"),
                   gate_samples=float(reason!="validation_samples"),
                   gate_dynamics=float(reason not in ("training_dynamics","validation_dynamics")),
                   gate_prediction_finite=float(reason not in ("nonfinite_prediction","nonfinite_visual")),
                   gate_depth_progression_limited=float(limited))
        return reason=="ready"

    def _check_actor_stall(self):
        self._ensure_training_progress()
        anchor=max(self.cfg.model_warmup_iterations,self._progress_start_iteration,
                   self._last_actor_update_iteration or 0,getattr(self,"world_model_warmup_until",None) or 0)
        stalled=max(0,self.iteration-anchor)
        interval=self.cfg.actor_stall_warning_iterations
        last=self._last_stall_warning_iteration
        if self.cfg.perception_training_version>=2 and stalled>=interval and (last is None or self.iteration-last>=interval):
            warnings.warn(f"No accepted Actor update for {stalled} iterations; inspect gate_reason, "
                          "active_imagination_horizon and actor_update_rejected before extending training.",RuntimeWarning,stacklevel=2)
            self._last_stall_warning_iteration=self.iteration
        return stalled

    def _terrain_curriculum_metrics(self):
        """Mean curriculum level per terrain family (Hiking logs these; one bincount per row)."""
        terrain=getattr(self.env.scene,"terrain",None)
        levels=getattr(terrain,"terrain_levels",None)
        types=getattr(terrain,"terrain_types",None)
        generator=getattr(getattr(terrain,"cfg",None),"terrain_generator",None)
        if levels is None or types is None or generator is None or not getattr(generator,"curriculum",False):
            return {}
        if getattr(self,"_terrain_family_of_column",None) is None:
            names=list(generator.sub_terrains)
            weights=np.array([generator.sub_terrains[name].proportion for name in names],dtype=float)
            cumulative=np.cumsum(weights/weights.sum())
            # mjlab TerrainGenerator._generate_curriculum_terrains column allocation.
            columns=[int(np.min(np.where(column/generator.num_cols+.001<cumulative)[0]))
                     for column in range(generator.num_cols)]
            self._terrain_family_names=names
            self._terrain_family_of_column=torch.tensor(columns,device=types.device,dtype=torch.long)
        family=self._terrain_family_of_column[types.long()]
        count=len(self._terrain_family_names)
        totals=torch.bincount(family,weights=levels.float(),minlength=count)
        envs=torch.bincount(family,minlength=count).clamp_min(1)
        return {f"terrain_level_{name}":value for name,value in
                zip(self._terrain_family_names,(totals/envs).tolist())}

    def _log(self,row):
        row.update(self._terrain_curriculum_metrics())
        stalled=self._check_actor_stall()
        row.update(actor_updates_accepted=self.actor_updates_accepted,actor_updates_rejected=self.actor_updates_rejected,
                   actor_update_stall_iterations=stalled,active_imagination_horizon=self._effective_horizon())
        wm=getattr(self,"_latent_wm_execution",None)
        policy=getattr(self,"_latent_actor_critic_macro_execution",None)
        actor=getattr(getattr(self,"_latent_actor_execution",None),"execution",None)
        critic=getattr(self,"_latent_critic_execution",None)
        if policy is not None:
            actor=policy.execution
            critic=policy.execution if self.cfg.critic_bootstrap else None
        amp=getattr(self,"_hiking_amp_gp_execution",None)
        amp_logical=getattr(self,"_hiking_amp_logical_execution",None)
        logical_captured=getattr(amp_logical,"graph",None) is not None
        micro_captured=getattr(amp,"graph",None) is not None
        row.update(wm_graph_captured=float(getattr(wm,"_graph",None) is not None),
            actor_graph_captured=float(getattr(actor,"_graph",None) is not None),
            critic_graph_captured=float(getattr(critic,"_graph",None) is not None),
            amp_gp_graph_captured=float(logical_captured or micro_captured),
            amp_logical_graph_captured=float(logical_captured),amp_micro_graph_captured=float(micro_captured))
        if self.cfg.perception_training_version>=2:
            if not row.get("actor_active"):
                if self.iteration<=self.cfg.model_warmup_iterations:row["gate_reason"]="model_warmup"
                elif row.get("critic_calibration"):row["gate_reason"]="critic_calibration"
            row["imagined_steps_per_second"]*=self._effective_horizon()/self.cfg.imagination_horizon
        super()._log(row)
        if "vector_step" not in row and self.iteration%self.cfg.log_interval==0:
            print(f"[Perceptive] H={self._effective_horizon()} gate={row.get('gate_reason','legacy')} "
                  f"accepted={self.actor_updates_accepted} rejected={self.actor_updates_rejected} stall={stalled}",flush=True)

    def _imagine_reference_policy(self,batch,noises=None):
        """Official policy_learner loop over a compiled WM environment step.

        SuperMimic abc124c1 policy_learner.py accumulates each step's return
        and saves detached Critic observations. Only the last observation
        supplies a differentiable bootstrap. The visual stop-gradients stay
        in LatentImaginedStep/LatentPerceptiveActor, as in InstinctMJ03563ee.
        Hiking's AMP history is the task-specific reward adapter.
        """
        from .latent_perception import unpack_processed_actor_input
        from .hiking_observations import unpack_hiking_proprio_history
        from .hiking_amp import flatten_history
        horizon=self._effective_horizon()
        state,context=batch["state"],batch["context"]
        proprio,stack=unpack_processed_actor_input(batch["actor_history"],self.spec)
        with torch.no_grad():
            latent=self.actor.encode_processed(stack).detach()
        history=unpack_hiking_proprio_history(proprio)
        noises=self._observation_noise(len(state)) if noises is None else noises
        standing=batch["command_standing"]
        options=dict(standing_mask=standing,random_yaw_rate=batch["random_yaw_rate"],
                     random_velocity_mask=batch["random_velocity_mask"])
        previous=batch["previous_action"]
        command=self.adapter.navigation_command(state,batch["navigation_target"],batch["desired_speed"],**options)
        critic_obs=self.adapter.critic_features(state,previous,command,context)
        if self.uses_hiking_amp:
            amp_history=self.adapter.amp_features(batch["amp_state_history"])
        else:
            previous_amp=self.adapter.amp_features(state)
        objective=state.new_zeros(len(state));survival=torch.ones_like(objective)
        critic_rollout=[];tasks=[];styles=[];scores=[];rewards=[];continuations=[]
        first_action=None
        for step in range(horizon):
            # Official rollout.append(TensorDict(...).detach()). No gradient
            # is cut from the live last observation used by _bootstrap.
            critic_rollout.append(critic_obs.detach())
            (state,context,history,latent,action,command,contact,terminated,
             task,amp_next,critic_obs)=self._reference_step_fn(
                state,context,history,latent,batch["navigation_target"],batch["desired_speed"],
                noises[step],standing,random_yaw_rate=batch["random_yaw_rate"],
                random_velocity_mask=batch["random_velocity_mask"],previous_action=previous)
            if first_action is None:first_action=action
            if self.uses_hiking_amp:
                amp_history=torch.cat((amp_history[:,1:],amp_next[:,None]),1)
                window=flatten_history(amp_history)
                if self.cfg.precision=="bf16" and window.device.type=="cuda":
                    with torch.autocast("cuda",dtype=torch.bfloat16):
                        score=self.discriminator(window).float().squeeze(-1)
                else:score=self.discriminator(window).squeeze(-1)
                style=self.cfg.amp_reward_coef*(1-.25*(score-1).square()).clamp_min(0)
            else:
                style,score=self._amp_reward(previous_amp,amp_next)
                previous_amp=amp_next
            reward=self.cfg.task_weight*task+self.cfg.style_weight*(style if self.cfg.direct_style_grad else style.detach())
            objective=objective+self.cfg.gamma**step*survival*reward
            continuation=1-terminated
            if not self.cfg.survival_grad:continuation=continuation.detach()
            survival=survival*continuation
            tasks.append(task.detach());styles.append(style.detach());scores.append(score.detach())
            rewards.append(reward.detach());continuations.append(continuation.detach())
            previous=action
        if self.cfg.critic_bootstrap:
            bootstrap=self._value(critic_obs).clamp(max=self._critic_value_ceiling)
            objective=objective+self.cfg.gamma**horizon*survival*bootstrap
        critic_x=torch.stack(critic_rollout)
        with torch.no_grad():
            values=torch.cat((self._value(critic_x),self._value(critic_obs)[None]),0)
            targets=lambda_returns(torch.stack(rewards),torch.stack(continuations),values,
                self.cfg.gamma,self.cfg.gae_lambda).clamp(max=self._critic_value_ceiling)
        return dict(objective=objective,x=critic_x,targets=targets.detach(),task=torch.stack(tasks),
            style=torch.stack(styles),score=torch.stack(scores),survival=survival.detach(),first_action=first_action)

    def imagine(self,batch,noises=None):
        if self.uses_reference_policy_step:
            return self._imagine_reference_policy(batch,noises)
        horizon=self._effective_horizon()
        state,context=batch["state"],batch["context"]
        # A zero desired speed also represents turning. Require the explicit
        # sampled command mode for the standing objective, including bootstrap.
        standing=batch["command_standing"] if self.uses_standing_objective else None
        command_function=self.adapter.navigation_command if self.uses_full_hiking_task else command_from_state
        command_options={"standing_mask":standing}
        if self.uses_full_hiking_task:
            command_options.update(random_yaw_rate=batch["random_yaw_rate"],random_velocity_mask=batch["random_velocity_mask"])
        previous=batch["previous_action"]
        if self.uses_latent:
            from .latent_perception import unpack_processed_actor_input
            proprio,stack=unpack_processed_actor_input(batch["actor_history"],self.spec)
            with torch.no_grad():latent=self.actor.encode_processed(stack).detach()
        else:
            proprio,ring=unpack_actor_input(batch["actor_history"],self.spec)
        if self.uses_full_hiking_task:
            from .hiking_observations import unpack_hiking_proprio_history
            history=unpack_hiking_proprio_history(proprio)
        else:
            history=proprio.reshape(len(state),self.spec.proprio_dim//96,96)
        if not self.uses_latent:frame=batch["depth_frame"].float()
        noises=self._observation_noise(len(state)) if noises is None else noises
        states=[state];contexts=[context];actions=[];previous_actions=[];commands=[];terminations=[];contacts=[]
        for step in range(horizon):
            previous_actions.append(previous)
            if self.uses_fast_execution:
                state,context,history,latent,action,command,contact,terminated=self._latent_step_fn(
                    state,context,history,latent,batch["navigation_target"],batch["desired_speed"],noises[step],standing,
                    **({"random_yaw_rate":batch["random_yaw_rate"],"random_velocity_mask":batch["random_velocity_mask"]}
                       if self.uses_full_hiking_task else {}))
            elif self.uses_latent:
                command=command_function(state,batch["navigation_target"],batch["desired_speed"],**command_options)
                if self.uses_full_hiking_task:
                    from .hiking_observations import pack_hiking_proprio_history
                    policy_history=pack_hiking_proprio_history(history)
                else:policy_history=history.flatten(1)
                action=self.actor.forward_latent(policy_history,latent).clamp(
                    -self.cfg.action_limit,self.cfg.action_limit)
                state,contact,latent=self.world_model.step_with_latent(state,action,context,latent)
            else:
                command=command_function(state,batch["navigation_target"],batch["desired_speed"],**command_options)
                action=self.actor(pack_actor_input(history.flatten(1),ring,self.spec)).clamp(
                    -self.cfg.action_limit,self.cfg.action_limit)
                state,contact,frame=self.world_model.step_with_perception(state,action,context,frame)
            commands.append(command)
            if not self.uses_fast_execution:
                context,terminated=(self.adapter.advance_context(state,context,action=action,contact=contact)
                    if self.uses_full_hiking_task else self.adapter.advance_context(state,context))
                next_command=command_function(state,batch["navigation_target"],batch["desired_speed"],**command_options)
                observation=self.adapter.actor_frame(state,action,next_command,context)+noises[step]
                history=torch.cat((history[:,1:],observation[:,None]),1)
            if not self.uses_latent:ring=advance_depth_ring(ring,self._augment_depth(frame),self.spec)
            states.append(state);contexts.append(context);actions.append(action);contacts.append(contact);terminations.append(terminated)
            previous=action
        states=torch.stack(states);contexts=torch.stack(contexts);actions=torch.stack(actions)
        previous_actions=torch.stack(previous_actions);commands=torch.stack(commands);terminated=torch.stack(terminations);contacts=torch.stack(contacts)
        if self.uses_fast_execution:
            task,amp=self._latent_trajectory_fn(states,actions,previous_actions,commands,contexts,terminated,contacts)
        else:
            fk=self.adapter.analyze_state(states)
            amp=self.adapter.amp_features(states,kinematics=fk)
            task=self.adapter.task_reward(states[:-1],states[1:],actions,previous_actions,commands,contexts[1:],terminated,
                contacts,kinematics=tuple(value[1:] for value in fk))
        styles=[];scores=[]
        if self.uses_hiking_amp:
            from .hiking_amp import flatten_history, sliding_histories
            initial = self.adapter.amp_features(batch["amp_state_history"])
            windows = flatten_history(sliding_histories(initial, amp[1:])).flatten(0,1)
            for start in range(0,len(windows),self.cfg.reward_chunk_size):
                chunk = windows[start:start+self.cfg.reward_chunk_size]
                if self.cfg.precision == "bf16" and chunk.device.type == "cuda":
                    with torch.autocast("cuda",dtype=torch.bfloat16):
                        score=self.discriminator(chunk).float().squeeze(-1)
                else:
                    score=self.discriminator(chunk).squeeze(-1)
                style=self.cfg.amp_reward_coef*(1-.25*(score-1).square()).clamp_min(0)
                styles.append(style);scores.append(score)
        else:
            first,second=amp[:-1].flatten(0,1),amp[1:].flatten(0,1)
            for start in range(0,len(first),self.cfg.reward_chunk_size):
                style,score=self._amp_reward(first[start:start+self.cfg.reward_chunk_size],second[start:start+self.cfg.reward_chunk_size])
                styles.append(style);scores.append(score)
        style=torch.cat(styles).reshape_as(task);score=torch.cat(scores).reshape_as(task)
        rewards=self.cfg.task_weight*task+self.cfg.style_weight*(style if self.cfg.direct_style_grad else style.detach())
        continuation=1-terminated
        if not self.cfg.survival_grad:continuation=continuation.detach()
        if self.uses_fast_execution:
            # Official policy_learner.py accumulates survival with one
            # multiplication per imagined step. Eager cumprod's PyTorch 2.7
            # backward reads a CUDA scalar on the host and cannot be captured.
            survival=torch.ones_like(continuation[0])
            alive_steps=[]
            for step in range(horizon):
                survival=survival*continuation[step]
                alive_steps.append(survival)
            alive=torch.stack(alive_steps)
        else:
            alive=torch.cumprod(continuation,0)
        c_t=torch.cat((torch.ones_like(alive[:1]),alive[:-1]),0)
        discount=self.cfg.gamma**torch.arange(horizon,device=self.device,dtype=states.dtype)
        objective=(discount[:,None]*c_t*rewards).sum(0)
        last_command=command_function(states[-1],batch["navigation_target"],batch["desired_speed"],**command_options)
        all_commands=torch.cat((commands,last_command[None]),0)
        all_previous=torch.cat((batch["previous_action"][None],actions),0)
        critic_x=self.adapter.critic_features(states,all_previous,all_commands,contexts)
        if self.cfg.critic_bootstrap:
            bootstrap=self._value(critic_x[-1])
            if self.uses_reference_schedule:bootstrap=bootstrap.clamp(max=self._critic_value_ceiling)
            objective=objective+self.cfg.gamma**horizon*alive[-1]*bootstrap
        with torch.no_grad():
            old_values=self._value(critic_x.detach())
            targets=lambda_returns((self.cfg.task_weight*task+self.cfg.style_weight*style).detach(),continuation.detach(),old_values,self.cfg.gamma,self.cfg.gae_lambda)
            if self.uses_reference_schedule:targets=targets.clamp(max=self._critic_value_ceiling)
        return dict(objective=objective,x=critic_x[:-1].detach(),targets=targets.detach(),task=task.detach(),style=style.detach(),
                    score=score.detach(),survival=alive[-1].detach(),first_action=actions[0])

    @torch.no_grad()
    def initialize_from_policy(self,path):
        if self.uses_latent:
            raise ValueError("Instinct latent vision requires a fresh experiment; pixel Actor warmstarts change encoder training semantics")
        payload=torch.load(path,map_location=self.device,weights_only=False)
        if payload.get("format")=="amp-wmac-v1":
            from src.wmac.checkpoints import checkpoint_actor_architecture
            arch=checkpoint_actor_architecture(payload)
            if arch.get("moe_variant")!="hiking" or arch.get("moe_gate_hidden")!=[256]:
                raise ValueError("Use a paper-gate flat Hiking policy for function-preserving perceptive initialization")
            self.actor.copy_from_flat_hiking(payload["actor"])
        elif payload.get("format")==FORMAT:
            from .checkpoints import architecture_metadata
            if payload.get("architecture")!=architecture_metadata(self.cfg,self.spec):raise ValueError("Perceptive policy architecture mismatch")
            self.actor.load_state_dict(payload["actor"])
        else:raise ValueError("Unsupported perceptive warmstart; ordinary flat MLP/PPO requires explicit distillation")
        self.actor_initialization_history.append(dict(kind="perceptive_policy_initialization",source=str(Path(path).resolve()),
            source_sha256=self._checkpoint_sha256(path),other_networks_loaded=False,actor_optimizer_loaded=False,
            visual_columns_zero_initialized=payload.get("format")=="amp-wmac-v1"))

    def save(self,path,export=True):
        from .checkpoints import architecture_metadata
        self._ensure_training_progress()
        terrain,motion=self._fingerprints()
        payload=dict(format=FORMAT,implementation_version=self.cfg.perception_training_version,config=asdict(self.cfg),architecture=architecture_metadata(self.cfg,self.spec),
            visual_spec=asdict(self.spec),terrain_fingerprint=terrain,motion_manifest_sha256=motion,
            amp_grounding=self.cfg.ground_height_semantics,iteration=self.iteration,sim_steps=self.sim_steps,
            imagination_steps=self.imagination_steps,wall_seconds=self.wall_time,collect_accumulator=self._collect_accumulator,
            critic_calibration_updates=self.critic_calibration_updates,env_common_step_counter=self.env.common_step_counter,
            actor_initialization_history=self.actor_initialization_history,validation=self.validation,
            torch_rng=torch.get_rng_state(),numpy_rng=np.random.get_state(),python_rng=random.getstate(),
            amp_normalizer_settings=({} if self.uses_hiking_amp else
                {key:getattr(self.amp_norm,key) for key in ("eps","variance_epsilon","clip","max_count")}))
        payload["perception_training_history"]=self.perception_training_history
        payload["world_model_warmup_until"]=getattr(self,"world_model_warmup_until",None)
        payload["perception_training_progress"]={key:getattr(self,key) for key in (
            "actor_updates_accepted","actor_updates_rejected","_active_horizon","_last_actor_update_iteration",
            "_progress_start_iteration","_last_stall_warning_iteration")}
        payload["training_cost"]=dict(physical_transitions=self.sim_steps,wall_seconds=payload["wall_seconds"],
            trajectory_id=self._cost_trajectory_id,gpu_count=int(self.device.type=="cuda"),
            wall_time_scope=WALL_TIME_SCOPE,history_complete=self._cost_history_complete)
        modules=(*MODULES,"latent_decoder") if self.uses_latent else MODULES
        for name in (*modules,*OPTIMIZERS):payload[name]=getattr(self,name).state_dict()
        if torch.cuda.is_available():payload["cuda_rng"]=torch.cuda.get_rng_state_all()
        path=Path(path);temporary=path.with_suffix(".pt.tmp");torch.save(payload,temporary);os.replace(temporary,path)
        if export:self.export_policy(path.parent/"policy.onnx")

    def load(self,path,**unused):
        from .checkpoints import validate_checkpoint,validate_module_states,validate_optimizer_states
        payload=torch.load(path,map_location=self.device,weights_only=False)
        if self.cfg.hiking_training_version>=2 and self.replay.total_steps:
            raise ValueError("Hiking v2 resume requires a fresh runner with empty Replay; recreate the runner before loading")
        terrain,motion=self._fingerprints()
        migration=validate_checkpoint(payload,self.cfg,terrain_fingerprint=terrain,motion_fingerprint=motion)
        physical_migration=migration.get("world_model_migration")
        until=payload.get("world_model_warmup_until")
        if until is not None and (isinstance(until,bool) or not isinstance(until,int)
                or not 0<=until<=payload["iteration"]+self.cfg.model_warmup_iterations):
            raise ValueError("Invalid saved world-model warmup anchor")
        history=payload.get("perception_training_history",[])
        if not isinstance(history,list) or not all(isinstance(record,Mapping) for record in history):
            raise ValueError("Invalid perception training history")
        last_physical_migration=next((record for record in reversed(history)
            if record.get("kind")=="world_model_parameterization_migration"),None)
        if last_physical_migration is not None:
            source_iteration=last_physical_migration.get("source_iteration")
            recorded_until=last_physical_migration.get("world_model_warmup_until")
            if (isinstance(source_iteration,bool) or not isinstance(source_iteration,int) or source_iteration<0
                    or isinstance(recorded_until,bool) or not isinstance(recorded_until,int)
                    or recorded_until<source_iteration):
                raise ValueError("Invalid world-model migration warmup history")
            # The anchor remains part of the resume contract after it expires.
            # Missing or changed top-level metadata must not silently bypass a
            # migrated model's warmup, even when its source iteration is high.
            if until!=recorded_until:
                raise ValueError("Saved world-model warmup anchor differs from migration history")
        progress=payload.get("perception_training_progress")
        if progress is not None:
            required={"actor_updates_accepted","actor_updates_rejected","_active_horizon","_last_actor_update_iteration",
                      "_progress_start_iteration","_last_stall_warning_iteration"}
            if not isinstance(progress,Mapping) or set(progress)!=required:
                raise ValueError("Incomplete perception training progress")
            iteration=payload.get("iteration")
            if isinstance(iteration,bool) or not isinstance(iteration,int) or iteration<0:
                raise ValueError("Invalid saved iteration")
            for key in ("actor_updates_accepted","actor_updates_rejected","_active_horizon","_progress_start_iteration"):
                value=progress.get(key)
                if isinstance(value,bool) or not isinstance(value,int) or value<0:
                    raise ValueError(f"Invalid perception training progress: {key}")
            if not 1<=progress["_active_horizon"]<=payload["config"]["imagination_horizon"]:
                raise ValueError("Invalid saved imagination horizon")
            if progress["_progress_start_iteration"]>iteration:
                raise ValueError("Invalid perception training progress start iteration")
            for key in ("_last_actor_update_iteration","_last_stall_warning_iteration"):
                value=progress.get(key)
                if value is not None and (isinstance(value,bool) or not isinstance(value,int) or not 0<=value<=payload["iteration"]):
                    raise ValueError(f"Invalid perception training progress: {key}")
        elif self.cfg.perception_training_version>=2 and not migration:
            raise ValueError("Version 2 resume requires saved perception training progress")
        all_modules=(*MODULES,"latent_decoder") if self.uses_latent else MODULES
        modules=tuple(name for name in all_modules if not physical_migration or name not in ("world_model","critic"))
        optimizers=tuple(name for name in OPTIMIZERS if not physical_migration or name not in ("opt_model","opt_critic"))
        validate_module_states(payload,{name:getattr(self,name) for name in modules})
        validate_optimizer_states(payload,{name:getattr(self,name) for name in optimizers})
        for name in optimizers:
            saved=payload[name];current=getattr(self,name).state_dict()
            if len(saved["param_groups"])!=len(current["param_groups"]) or any(len(a["params"])!=len(b["params"]) for a,b in zip(saved["param_groups"],current["param_groups"])):
                raise ValueError(f"Perceptive optimizer layout mismatch: {name}")
        if physical_migration:
            # Rebuild after every preflight check, rather than relying on the
            # receiver accidentally being a fresh, never-trained instance.
            dtype=next(self.world_model.parameters()).dtype
            self.world_model=TerrainPerceptiveWorldModel(self.adapter.terrain,self.spec,
                hidden=self.cfg.model_hidden,dynamics_parameterization=self.cfg.dynamics_parameterization,
                dt=self.cfg.dynamics_dt,decimation=self.cfg.dynamics_decimation).to(device=self.device,dtype=dtype)
            self.world_model.precision=self.cfg.precision
            width=self.critic.norm.mean.numel()
            self.critic=Critic(width,hidden=self.cfg.critic_hidden).to(device=self.device,dtype=dtype)
            self.critic.precision=self.cfg.precision
            self.critic.norm.max_count=self.cfg.observation_normalizer_until
            fused=self.device.type=="cuda"
            self.opt_model=torch.optim.Adam(self.world_model.parameters(),lr=self.cfg.world_model_lr,fused=fused)
            self.opt_critic=torch.optim.Adam(self.critic.parameters(),lr=self.cfg.critic_lr,fused=fused)
            self._value_fn=self._compile(self.critic.forward,"critic")
        for name in (*modules,*optimizers):getattr(self,name).load_state_dict(payload[name])
        if self.device.type=="cpu":
            for name in OPTIMIZERS:
                for group in getattr(self,name).param_groups:
                    if "fused" in group:group["fused"]=False
        if not self.uses_hiking_amp:
            self._set_amp_norm_settings(payload.get("amp_normalizer_settings",{}))
        self.iteration=payload["iteration"];self.sim_steps=payload["sim_steps"];self.imagination_steps=payload["imagination_steps"]
        self.wall_offset=payload["wall_seconds"];self._collect_accumulator=payload["collect_accumulator"]
        self.critic_calibration_updates=payload["critic_calibration_updates"]
        self.env.common_step_counter=payload.get("env_common_step_counter",0)
        self.actor_initialization_history=payload.get("actor_initialization_history",[])
        self.perception_training_history=list(history)
        self.world_model_warmup_until=until
        if migration:
            self.actor_updates_accepted=self.actor_updates_rejected=0
            self._active_horizon=self.cfg.imagination_horizon_start
            self._progress_start_iteration=self.iteration
            self._last_actor_update_iteration=self._last_stall_warning_iteration=None
            self.critic_calibration_updates=0
            record=dict(migration,source=str(Path(path).resolve()),source_sha256=self._checkpoint_sha256(path),
                        source_iteration=self.iteration,critic_calibration_reset=True,replay_and_validation_reset=True)
            if physical_migration:
                self.world_model_warmup_until=self.iteration+self.cfg.model_warmup_iterations
                record.update(kind="world_model_parameterization_migration",preserved_modules=list(modules),
                              preserved_optimizers=list(optimizers),reset_modules=["world_model","critic"],
                              reset_optimizers=["opt_model","opt_critic"],world_model_warmup_until=self.world_model_warmup_until)
            else:record["network_and_optimizer_states_preserved"]=True
            self.perception_training_history.append(record)
        elif progress:
            for key,value in progress.items():
                if key in ("actor_updates_accepted","actor_updates_rejected","_active_horizon","_last_actor_update_iteration",
                            "_progress_start_iteration","_last_stall_warning_iteration"):
                    setattr(self,key,value)
        else:
            self._ensure_training_progress()
            self._progress_start_iteration=self.iteration
        self.validation=self._empty_validation()
        self._cost_history_complete=False
        # Checkpoints do not carry replay frames. AMP's fresh-window cursor
        # must never retain a rollout position from the receiving runner.
        for name in ("_hiking_amp_budget", "_hiking_amp_budget_replay"):
            if hasattr(self,name):delattr(self,name)
        # Simulator resets require new camera history/replay and fresh domain
        # validation. Loaded values never certify a recreated scene.
        self.current_depth=self._from_env(read_depth_frame(self.env)).detach();self.depth_history.reset(self._policy_depth_frame(self.current_depth))
        if hasattr(self.replay,"bootstrap_reset") and not self.replay._bootstrapped:
            frame=self.depth_history.ring[:,0]
            self.replay.bootstrap_reset(frame if self.spec.include_valid else frame[:,:1])
        if self.uses_fast_execution:self._setup_fast_execution()
        torch.set_rng_state(payload["torch_rng"].cpu());np.random.set_state(payload["numpy_rng"]);random.setstate(payload["python_rng"])
        if "cuda_rng" in payload and torch.cuda.is_available():torch.cuda.set_rng_state_all([x.cpu() for x in payload["cuda_rng"]])

    def export_policy(self,path):
        from .onnx_export import export_raw_packed_actor
        export_raw_packed_actor(self.actor,path)
        from mjlab.rl.exporter_utils import attach_metadata_to_onnx,get_base_metadata
        from .checkpoints import architecture_metadata
        architecture=architecture_metadata(self.cfg,self.spec)
        metadata=get_base_metadata(self.env,self.log_dir.name)
        metadata.update(dict(algorithm=FORMAT,visual_spec=json.dumps(asdict(self.spec)),
            camera_protocol=json.dumps(architecture["camera"]),
            observation_width=str(self.spec.packed_dim),camera_history_order="oldest_to_newest",depth_channels="normalized_depth,valid",
            privilege_terrain_in_actor="false",recovery_height_reward=self.cfg.recovery_height_reward,
            physical_action_clip=str(self.cfg.action_limit)))
        metadata["motion_scope"]=self.cfg.motion_scope
        metadata["visual_mode"]=getattr(self.cfg,"visual_mode","pixel_v2")
        if self.cfg.actor_type == TRANSFORMER_ACTOR_TYPE:
            metadata["actor_architecture"] = json.dumps(architecture["actor"], sort_keys=True)
        metadata["sensor_protocol"]=self.cfg.sensor_protocol
        metadata["visual_encoder_channels"]=str(self.spec.visual_channels*self.spec.frames)
        if self.uses_full_hiking_task:
            from .hiking_actuators import read_hiking_actuator_parameters
            physical=read_hiking_actuator_parameters(self.env.scene["robot"])
            # Generic mjlab export reads MuJoCo motor gain=1/bias=0. Ideal
            # PD is computed in our actuator wrapper, so export its real gains
            # rather than those motor transmission coefficients.
            for key,name in (("joint_stiffness","kp"),("joint_damping","kd"),
                             ("joint_effort_limits","effort_limit"),("joint_velocity_limits","velocity_limit")):
                metadata[key]=physical[name][0].detach().cpu().tolist()
            metadata.update(observation_history_order="term",observation_history_length=8,
                proprioception_width=768,proprioception_frame="torso_link",
                proprioception_term_widths="3,3,3,29,29,29",
                angular_velocity_scale=.25,joint_velocity_scale=.05,
                locomotion_objective="hiking_full_v1")
        if not self.spec.include_valid:
            metadata["depth_channels"]="raw_depth_divided_by_2.5,diagnostic_valid_unused_by_policy"
            metadata["depth_preprocessing"]="crop_then_gaussian_blur_then_clip_0_1;do_not_preclip_raw_depth"
        if self.spec.depth_protocol=="g1_rpl_v1":
            metadata["depth_channels"]="RPL_centered_depth,diagnostic_valid_unused_by_policy"
            metadata["depth_preprocessing"]="G1_Z16_scale_to_metres;resize_240x135;clip_0.1_2;antialiased_bicubic_48x27;(depth-0.1)/1.9-0.5;no_reclip"
            metadata["camera_capture_hz"]="10"
            metadata["camera_delay_capture_ticks"]="2"
        if self.cfg.sensor_protocol in (G1_HIKING_CALIBRATED_SENSOR_PROTOCOL, G1_HIKING_DR_SENSOR_PROTOCOL):
            metadata["depth_preprocessing"]="measured_G1_Z16_to_metres;nearest_exact_resize_64x36;Hiking_crop_top18_left16_right16;gaussian3_sigma1;clip_0_2.5;divide_2.5"
            metadata["camera_capture_hz"]="50"
            metadata["camera_delay_policy_steps"]="1"
            metadata["camera_hardware_stream_hz"]="30"
            metadata["camera_randomization_in_deployment"]="false"
        metadata["future_depth_model_in_deployment"]="false"
        if self.cfg.motion_scope=="forward_walk_turn_v1":
            metadata.pop("recovery_height_reward",None)
            metadata["command_bounds"]=json.dumps(architecture_metadata(self.cfg,self.spec)["commands"])
        attach_metadata_to_onnx(str(path),metadata)
        if self.uses_full_hiking_task and self.cfg.hiking_training_version>=3:
            # Joint sign conventions, torso-frame IMU and depth rules for Unitree-convention harnesses.
            from .unitree_interface import deployment_contract
            contract=deployment_contract(camera=architecture.get("camera"))
            contract["training_robot_dynamics_randomization"]=self.cfg.robot_dynamics_randomization
            (Path(path).parent/"deployment_contract.json").write_text(json.dumps(contract,indent=1)+"\n")

    def update_actor_critic(self, update_actor=True):
        self._ensure_training_progress()
        batch = _device_batch(self.replay.sample_starts(self.cfg.actor_batch_size,
            self.cfg.recovery_fraction,fields=self.ACTOR_FIELDS),self.device)
        self.opt_model.zero_grad(set_to_none=True); self.opt_disc.zero_grad(set_to_none=True)
        self.opt_critic.zero_grad(set_to_none=True); self.opt_actor.zero_grad(set_to_none=True)
        macro=getattr(self,"_latent_actor_critic_macro_execution",None)
        if update_actor and macro is not None:
            return self._update_actor_critic_macro(batch,macro)
        with torch.no_grad():
            old_action = None if self.uses_reference_schedule else self.actor(batch['actor_history']).clone()
            recovery = self._recovery_mask(batch)
            router_stats = routing_metrics(self.actor, batch['actor_history'], recovery)
            if is_moe(self.actor):
                real_route = self.actor.route(batch['actor_history'])
                full_router_load = load_fraction(real_route.indices, self.actor.num_experts,
                                                  real_route.probabilities.dtype)
            else:
                full_router_load = None
        rollback_enabled=update_actor and not self.uses_reference_schedule
        parameter_snapshot = [parameter.detach().clone() for parameter in self.actor.parameters()] if rollback_enabled else None
        optimizer_snapshot = copy.deepcopy(self.opt_actor.state_dict()) if rollback_enabled else None
        size = self.cfg.actor_microbatch_size or self.cfg.actor_batch_size
        records, xs, ys, router_records = [], [], [], []
        finite_loss = True
        actor_execution=getattr(self,"_latent_actor_execution",None)
        use_actor_graph=update_actor and actor_execution is not None and size>=len(batch['state'])
        captured_step=None
        with frozen_parameters(self.world_model,self.discriminator,self.critic):
            for start in range(0,len(batch['state']),size):
                chunk = {key:value[start:start+size] for key,value in batch.items()}
                with torch.set_grad_enabled(update_actor):
                    if use_actor_graph:
                        rollout,graph_norm,graph_ok=actor_execution.step(
                            chunk,self._observation_noise(len(chunk['state'])),full_router_load)
                        captured_step=(graph_norm,graph_ok)
                        task_loss,auxiliary,loss=(rollout['task_loss'],rollout['auxiliary_loss'],rollout['loss'])
                        finite_loss=bool(graph_ok)
                    else:
                        rollout = self.imagine(chunk)
                        task_loss = -rollout['objective'].mean()
                        auxiliary = task_loss.new_zeros(())
                    if not use_actor_graph and is_moe(self.actor) and update_actor:
                        balance, z, _ = router_losses(self.actor, chunk['actor_history'], full_router_load)
                        # A disabled regularizer must not poison a finite task
                        # objective through 0*inf at extreme router logit scales.
                        if self.cfg.moe_load_balance_coef:
                            auxiliary = auxiliary + self.cfg.moe_load_balance_coef*balance
                        if self.cfg.moe_router_z_coef:
                            auxiliary = auxiliary + self.cfg.moe_router_z_coef*z
                    # Only the router on real starts receives the auxiliary
                    # signal. Imagined rewards/continuation/Critic targets are
                    # unchanged; routing losses do not steer predicted states.
                    loss = task_loss + auxiliary
                    if update_actor and not use_actor_graph:
                        finite = True if self.uses_reference_schedule else bool(torch.isfinite(loss.detach()))
                        finite_loss = finite_loss and finite
                        if finite:
                            (loss * (len(chunk['state'])/len(batch['state']))).backward()
                xs.append(rollout['x']); ys.append(rollout['targets'])
                record = torch.stack((loss.detach(),rollout['task'].mean(),rollout['style'].mean(),
                    ((rollout['score']<=-1)|(rollout['score']>=3)).float().mean(),rollout['survival'].mean()))
                records.append((len(chunk['state']),record))
                router_records.append((len(chunk['state']), torch.stack((task_loss.detach(), auxiliary.detach()))))
                del rollout
        if is_moe(self.actor):
            expert_norms = [parameter_grad_norm(expert) for expert in self.actor.experts]
            router_stats.update(moe_router_grad_norm=parameter_grad_norm(self.actor.router),
                                moe_expert_grad_norm_mean=sum(expert_norms)/len(expert_norms),
                                moe_expert_grad_norm_max=max(expert_norms))
        norm, stepped = batch['state'].new_zeros(()), False
        if captured_step is not None:
            norm,accepted=captured_step
            stepped=bool(accepted)
        elif update_actor and finite_loss:
            norm, stepped = self._optimizer_step(None,self.opt_actor,self.actor,actor=True,backward=False)
        if self.uses_reference_schedule:
            shift=batch['state'].new_full((),float('nan'))
            rejected=False
        else:
            with torch.no_grad():
                shift = (self.actor(batch['actor_history'])-old_action).square().mean().sqrt()
            rejected = update_actor and (not stepped or not bool(torch.isfinite(shift)) or float(shift)>self.cfg.max_action_shift)
        if rejected:
            with torch.no_grad():
                for parameter, saved in zip(self.actor.parameters(),parameter_snapshot): parameter.copy_(saved)
            self.opt_actor.load_state_dict(optimizer_snapshot)
            if actor_execution is not None:actor_execution.invalidate()
        if update_actor:
            if rejected:self.actor_updates_rejected+=1
            else:
                self.actor_updates_accepted+=1
                self._last_actor_update_iteration=self.iteration+1
        for module in (self.world_model,self.discriminator,self.critic):
            if any(parameter.grad is not None for parameter in module.parameters()):
                raise RuntimeError('Actor BPTT leaked parameter gradients into a frozen network')
        aggregate = sum(n*value for n,value in records)/len(batch['state'])
        values = aggregate.cpu().tolist()
        result = dict(zip(('actor_loss','imagined_task_reward','imagined_style_reward',
                           'style_zero_gradient_fraction','imagined_survival'),values))
        result.update(actor_grad_norm=float(norm),actor_action_shift=float(shift),actor_update_rejected=float(rejected),
                      critic_loss=0.,real_critic_loss=0.,critic_grad_norm=0.,critic_updates_applied_fraction=0.)
        task_loss, auxiliary_loss = (sum(n*v for n,v in router_records)/len(batch['state'])).cpu().tolist()
        router_stats.update(actor_task_loss=task_loss, actor_router_aux_loss=auxiliary_loss)
        result.update(router_stats)
        if self.cfg.critic_bootstrap:
            result.update(self._fit_critic(torch.cat(xs,1),torch.cat(ys,1)))
        self.imagination_steps += self._effective_horizon()*len(batch['state'])
        return result

    def _update_actor_critic_macro(self,batch,macro):
        from .diagnostics import routing_tensors,grad_norm,read_metrics
        with torch.no_grad():
            router_stats,route=routing_tensors(self.actor,batch['actor_history'],self._recovery_mask(batch))
            full_load=(load_fraction(route.indices,self.actor.num_experts,route.probabilities.dtype)
                       if route is not None else None)
        rollout,norm,accepted,critic=macro.step(batch,
            self._observation_noise(len(batch['state'])),full_load)
        if is_moe(self.actor):
            experts=torch.stack([grad_norm(expert).double() for expert in self.actor.experts])
            router_stats.update(moe_router_grad_norm=grad_norm(self.actor.router),
                moe_expert_grad_norm_mean=experts.mean(),moe_expert_grad_norm_max=experts.max())
        for module in (self.world_model,self.discriminator,self.critic):
            if any(parameter.grad is not None for parameter in module.parameters()):
                raise RuntimeError('Actor BPTT leaked parameter gradients into a frozen network')
        result=dict(actor_loss=rollout['loss'],imagined_task_reward=rollout['task'].mean(),
            imagined_style_reward=rollout['style'].mean(),
            style_zero_gradient_fraction=((rollout['score']<=-1)|(rollout['score']>=3)).float().mean(),
            imagined_survival=rollout['survival'].mean(),actor_grad_norm=norm,
            actor_action_shift=float('nan'),actor_update_rejected=0.,
            critic_loss=critic[0],real_critic_loss=critic[1],critic_grad_norm=critic[2],
            critic_updates_applied_fraction=critic[3])
        router_stats.update(actor_task_loss=rollout['task_loss'],actor_router_aux_loss=rollout['auxiliary_loss'])
        result.update(router_stats)
        # Official updates are unconditional; accepted is the constant true
        # device marker. Avoid a host read solely to rediscover that constant.
        self.actor_updates_accepted+=1
        self._last_actor_update_iteration=self.iteration+1
        self.imagination_steps+=self._effective_horizon()*len(batch['state'])
        return read_metrics(result)
