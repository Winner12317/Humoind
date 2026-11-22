"""Model-dominated AMP actor-critic, isolated from the PPO baseline.

The optimizer follows SuperMimic's collect / fit dynamics / BPTT / value-fit
loop. AMP and recovery-aware rewards are extensions for this repository.
No PPO surrogate or likelihood-ratio policy update is used here.
"""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import os
import random
import time
import uuid
import subprocess
import sys
import shutil
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

from mjlab.rl.exporter_utils import attach_metadata_to_onnx, get_base_metadata
from mjlab.tasks.registry import load_rl_cfg
from rsl_rl.modules.discriminator import Discriminator
from src.amp_features import amp_feature_dim, amp_feature_metadata, build_amp_features
from src.robot_profiles import G1, profile_for_bodies
from src.training_cost import WALL_TIME_SCOPE

from .networks import Actor, Critic, RunningNorm, WorldModel, frozen_parameters
from .amp_rewards import CenteredAmpReward, clipped_lsgan_reward
from .smooth_rewards import SmoothRewardTerms
from .reward_integration import (smooth_rewards_enabled, previous_previous_action, smooth_cost_rate)
from .actors import actor_architecture, create_actor
from .checkpoints import (checkpoint_recovery_semantics, checkpoint_height_reward_semantics,
                          checkpoint_actor_architecture)
from .replay import ReplayBuffer
from .routing import (is_moe, load_fraction, router_losses, routing_metrics,
                      routing_metric_keys, parameter_grad_norm)
from .state import quat_apply, quat_inv, quat_mul, quat_to_matrix
from .reference import (is_official, config_protocol, checkpoint_protocol, amp_value_upper_bound,
                        amp_reward_protocol, validate_training_protocol_resume,
                        smooth_reward_protocol, validate_smooth_reward_resume,
                        validate_smooth_reward_retune, checkpoint_smooth_reward_protocol,
                        SMOOTH_REWARD_COEFFICIENT_KEYS, IMAGINED_FOOT_SLIP_GEOMETRIC_MODES,
                        IMAGINED_FOOT_SLIP_INSTANTAMP,
                        imagined_foot_slip_protocol, validate_imagined_foot_slip_resume)
from .optimizers import (build_reference_optimizer, optimizer_algorithm_name,
                         validate_reference_optimizer_state, set_learning_rate)


def lambda_returns(rewards, continuation, values, gamma=0.99, lam=0.95):
    """Fixed lambda targets; time limits are represented separately from death.

    Shapes are (H,B), (H,B), (H+1,B). This function deliberately detaches
    every input: fitting a value must never train an actor or its dynamics.
    """
    with torch.no_grad():
        rewards, continuation, values = rewards.detach(), continuation.detach(), values.detach()
        ret = values[-1]
        targets = []
        for t in range(rewards.shape[0] - 1, -1, -1):
            ret = rewards[t] + gamma * continuation[t] * ((1 - lam) * values[t + 1] + lam * ret)
            targets.append(ret)
        return torch.stack(targets[::-1])


def differentiable_amp_reward(discriminator, normalizer, first, second, coefficient, *, reward_mapper=None):
    """Keep the input graph and optionally use a restored checkpoint mapping.

    The default retains the historical PPO-compatible clipped reward. Callers
    probing centered checkpoints must pass their saved scalar EMA mapper.
    """
    score = discriminator(torch.cat((normalizer(first), normalizer(second)), -1)).squeeze(-1)
    reward = (coefficient * (1 - 0.25 * (score - 1).square()).clamp_min(0)
              if reward_mapper is None else reward_mapper(score))
    return reward, score


def _device_batch(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def _masked_stat_mean(values, selected, *, empty):
    count = selected.sum(-1)
    mean = torch.where(selected, values, 0.).sum(-1) / count.clamp_min(1)
    return torch.where(count > 0, mean, values.new_full((), empty))


def _collect_tick_statistics(batch, accepted, recovery_height=.55):
    recovery = (batch['state'][:, 2] < recovery_height) | (batch['context'][:, -1] > 0)
    return torch.stack((
        _masked_stat_mean(batch['reward'].reshape_as(accepted), accepted, empty=0.),
        batch['terminated'].float().mean(), batch['timeout'].float().mean(),
        recovery.float().mean(), batch['boundary'].float().mean(), (~accepted).float().mean(),
    ))


class ExpertPairs:
    """Adjacent pairs within clips, balanced across recovery and locomotion.

    Features use the baseline AMP convention, vectorized over whole clips.
    No expert pair crosses a clip boundary or duplicates the final frame.
    """
    def __init__(self, cfg, env, device):
        runner_cfg = load_rl_cfg(cfg.task)
        all_names = list(env.scene['robot'].body_names)
        body_ids = [all_names.index(name) for name in runner_cfg.amp_body_names]
        anchor_id = all_names.index(runner_cfg.amp_anchor_name)
        schema = getattr(cfg, 'amp_feature_schema', 'legacy')
        profile = profile_for_bodies(all_names)
        pelvis_id = all_names.index(profile.root_body)
        foot_ids = tuple(all_names.index(name) for name in profile.foot_bodies)
        motion_root = Path(getattr(env, '_wmac_motion_root', runner_cfg.amp_motion_files))
        groups = [[], []]
        for path in sorted(motion_root.rglob('*.npz')):
            relative = path.relative_to(motion_root)
            if relative.parts[0] not in ('WalkandRun', 'Recovery'):
                raise ValueError(f'Unexpected expert motion group: {relative}')
            with np.load(path) as data:
                p, q, v, w = [torch.as_tensor(data[key], device=device, dtype=torch.float32)
                               for key in ('body_pos_w', 'body_quat_w', 'body_lin_vel_w', 'body_ang_vel_w')]
            if schema == 'legacy':
                anchor_inv = quat_inv(q[:, anchor_id:anchor_id+1])
                terms = [quat_apply(anchor_inv, p[:, body_ids] - p[:, anchor_id:anchor_id+1]),
                         quat_to_matrix(quat_mul(anchor_inv, q[:, body_ids]))[..., :2],
                         quat_apply(quat_inv(q[:, body_ids]), v[:, body_ids]),
                         quat_apply(quat_inv(q[:, body_ids]), w[:, body_ids])]
                features = torch.cat([term.flatten(1) for term in terms], -1)
            else:
                features = build_amp_features(p, q, v, w, body_ids=body_ids,
                    anchor_id=anchor_id, pelvis_id=pelvis_id, foot_ids=foot_ids, schema=schema,
                    foot_local_offsets=profile.amp_foot_offsets)
            if len(features) > 1:
                groups[int(relative.parts[0] == 'Recovery')].append(torch.stack((features[:-1], features[1:]), 1))
        self.groups = [torch.cat(group) if group else None for group in groups]
        if not any(group is not None for group in self.groups):
            raise ValueError('No adjacent expert motion pairs were found')
        self.device = device
        self.fraction = cfg.expert_recovery_fraction

    def sample(self, size):
        if self.fraction == 0:
            available = [g for g in self.groups if g is not None]
            if len(available) == 1:
                pool = available[0]
            else:
                # Keep one persistent combined array rather than rebuilding it per update.
                if not hasattr(self, '_uniform'):
                    self._uniform = torch.cat(available)
                pool = self._uniform
            pairs = pool[torch.randint(len(pool), (size,), device=self.device)]
        else:
            nr = int(round(size * self.fraction))
            counts = (size - nr, nr)
            parts = []
            for i, count in enumerate(counts):
                if count:
                    pool = self.groups[i] if self.groups[i] is not None else self.groups[1-i]
                    parts.append(pool[torch.randint(len(pool), (count,), device=self.device)])
            pairs = torch.cat(parts)
            pairs = pairs[torch.randperm(size, device=self.device)]
        return pairs[:, 0], pairs[:, 1]


class WmacRunner:
    """SuperMimic-style BPTT with real-data calibration and batched reward heads."""
    WM_FIELDS = ('state', 'next_state', 'action', 'context', 'contact_cost')
    ACTOR_FIELDS = ('state', 'previous_action', 'context', 'command', 'actor_history')
    REAL_FIELDS = ('state', 'next_state', 'previous_action', 'action', 'context',
                   'next_context', 'command', 'reward', 'terminated')
    # G1 sizes; __init__ replaces them from the state adapter of the robot in use.
    num_joints = G1.num_joints
    actor_frame_dim = 9 + 3 * G1.num_joints
    actor_obs_dim = 4 * (9 + 3 * G1.num_joints)
    recovery_height = G1.recovery_height

    def __init__(self, env, cfg, log_dir):
        self.started, self.wall_offset = time.perf_counter(), 0.0
        self._cost_history_complete = True
        self._cost_trajectory_id = str(uuid.uuid4())
        self.env, self.cfg, self.adapter = env, cfg, env.adapter
        # Robot-dependent sizes; an adapter without them is the 29-joint G1.
        self.num_joints = getattr(self.adapter, 'num_joints', G1.num_joints)
        self.actor_frame_dim = 9 + 3 * self.num_joints
        self.recovery_height = getattr(self.adapter, 'recovery_height', G1.recovery_height)
        self.adapter.recovery_counter_semantics = cfg.recovery_counter_semantics
        expected_power = 2. if cfg.recovery_height_reward == 'upright_v2' else 0.
        # Reward managers prepare their own term config at construction time.
        # Check both copies rather than silently changing only imagined rewards.
        for term in (self.adapter.reward_cfg['track_root_height'],
                     env.reward_manager.get_term_cfg('track_root_height')):
            if float(term.params.get('upright_power', 0.)) != expected_power:
                raise ValueError('Configure recovery height reward before creating the environment; '
                                 'real and imagined objectives must match recovery_height_reward')
        self.recovery_semantics_history = []
        self.imagined_foot_slip_migration_history = []
        self.device = torch.device(cfg.device)
        if cfg.imagined_recovery_foot_slip in IMAGINED_FOOT_SLIP_GEOMETRIC_MODES:
            foot_slip = imagined_foot_slip_protocol(cfg)
            self.adapter.configure_recovery_foot_slip(
                contact_height_m=foot_slip['recovery_foot_slip_contact_height_m'],
                softness_m=foot_slip['recovery_foot_slip_softness_m'],
                differentiate_support=cfg.imagined_recovery_foot_slip == IMAGINED_FOOT_SLIP_INSTANTAMP)
        self.smooth_reward_terms = None
        if smooth_rewards_enabled(cfg):
            if 'wmac_smooth_penalty' not in env.reward_manager.active_terms:
                raise ValueError('Configure smooth reward term before creating the environment')
            self.smooth_reward_terms = SmoothRewardTerms.from_env(env).to(self.device)
            env._wmac_smooth_reward_terms = self.smooth_reward_terms
            env._wmac_smooth_reward_cfg = cfg
        if cfg.amp_feature_schema != self.adapter.amp_feature_schema:
            raise ValueError('WMAC AMP schema does not match the environment; select the matching task/profile')
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        obs, _ = env.reset()
        self.actor_obs_dim = obs['actor'].shape[-1]
        self.actor = create_actor(cfg, obs_dim=self.actor_obs_dim, action_dim=self.num_joints).to(self.device)
        self.actor_initialization_history = []
        features = self.adapter.critic_features(self.adapter.state(), env.action_manager.action,
                                                self.adapter.command(), self.adapter.context())
        official = is_official(cfg)
        self.critic = Critic(features.shape[-1], hidden=cfg.critic_hidden,
            activation='gelu' if official else 'elu', layer_norm=official).to(self.device)
        self.actor.norm.max_count = self.critic.norm.max_count = cfg.observation_normalizer_until
        self.world_model = WorldModel(context_dim=self.num_joints + 6, hidden=cfg.model_hidden,
            activation='gelu' if official else 'silu', output_zero_init=not official,
            num_joints=self.num_joints).to(self.device)
        self._critic_vmax = amp_value_upper_bound(self.adapter, cfg) if official else None
        for module in (self.actor, self.critic, self.world_model):
            module.precision = cfg.precision
        amp_dim = amp_feature_dim(cfg.amp_feature_schema)
        self.discriminator = Discriminator(2*amp_dim, cfg.amp_reward_coef,
            list(cfg.discriminator_hidden), str(self.device), cfg.task_weight).to(self.device)
        self.amp_norm = RunningNorm(amp_dim, eps=cfg.amp_normalizer_epsilon,
            max_count=cfg.amp_normalizer_until if cfg.amp_normalizer_until is not None else float('inf')).to(self.device)
        self.amp_norm.variance_epsilon = cfg.amp_normalizer_variance_epsilon
        self.amp_norm.clip = cfg.amp_normalizer_clip
        self.amp_norm.count.fill_(cfg.amp_normalizer_initial_count)
        self.amp_reward_stats = CenteredAmpReward(cfg.amp_reward_coef, cfg.amp_score_scale,
                                                cfg.amp_score_ema_decay).to(self.device)
        fused = self.device.type == 'cuda'
        if cfg.optimizer_algorithm == 'muon_adam':
            self.opt_actor = build_reference_optimizer([self.actor], cfg.actor_lr, cfg.actor_betas, self.device)
            self.opt_critic = build_reference_optimizer([self.critic], cfg.critic_lr, device=self.device)
            self.opt_model = build_reference_optimizer([self.world_model], cfg.world_model_lr, device=self.device)
        else:
            self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr,
                                              betas=cfg.actor_betas, fused=fused)
            self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr, fused=fused)
            self.opt_model = torch.optim.Adam(self.world_model.parameters(), lr=cfg.world_model_lr, fused=fused)
        if cfg.discriminator_optimizer == 'adamw' and cfg.discriminator_trunk_weight_decay == cfg.discriminator_head_weight_decay:
            # Keep the legacy one-group optimizer layout loadable.
            self.opt_disc = torch.optim.AdamW(self.discriminator.parameters(), lr=cfg.discriminator_lr,
                weight_decay=cfg.discriminator_trunk_weight_decay, fused=fused)
        else:
            disc_groups = [
                {'params': self.discriminator.trunk.parameters(), 'weight_decay': cfg.discriminator_trunk_weight_decay, 'name': 'amp_trunk'},
                {'params': self.discriminator.amp_linear.parameters(), 'weight_decay': cfg.discriminator_head_weight_decay, 'name': 'amp_head'},
            ]
            optimizer = torch.optim.Adam if cfg.discriminator_optimizer == 'adam' else torch.optim.AdamW
            self.opt_disc = optimizer(disc_groups, lr=cfg.discriminator_lr, fused=fused)
        replay_device = cfg.replay_device
        if self.device.type == 'cpu' and replay_device.startswith('cuda'):
            replay_device = 'cpu'
        self.replay = ReplayBuffer(cfg.replay_steps, env.num_envs, replay_device,
                                   validation_env_fraction=cfg.validation_env_fraction,
                                   recovery_height=self.recovery_height)
        self._train_env_mask = self.replay.train_env_mask.to(self.device)
        self.experts = ExpertPairs(cfg, env, str(self.device))
        self.writer = SummaryWriter(str(self.log_dir))
        self.iteration = self.sim_steps = self.imagination_steps = 0
        self._collect_accumulator = 0.0
        self.csv_file = self.csv_writer = None
        self.last_model_loss = float('inf')
        self.validation = {'validation_model_error': float('inf'),
            'validation_contact_error': float('inf'), 'validation_reward_gap': float('inf'),
            'validation_walk_error': float('inf'), 'validation_recovery_error': float('inf')}
        self.best_eval_score = None
        self.critic_calibration_updates = 0
        if cfg.init_policy:
            self.initialize_from_policy(cfg.init_policy)
        self._wm_loss_fn = self._compile(self._world_model_loss, 'world model')
        self._imag_step_fn = self._compile(self._imagined_step, 'imagination step')
        self._reward_fn = self._compile(self._trajectory_rewards, 'batched reward/FK')
        self._value_fn = self._compile(self.critic.forward, 'critic')
        print(f'[WMAC] K={cfg.wm_window}, H={cfg.imagination_horizon}, actor_batch={cfg.actor_batch_size}, '
              f'precision={cfg.precision}, compile={cfg.compile}; first calls include compilation', flush=True)
        if cfg.actor_type == 'moe':
            execution = ('Hiking dense Softmax fusion; no Actor normalization or output tanh.'
                         if cfg.moe_variant == 'hiking' else
                         'Legacy dynamic expert dispatch runs eagerly with autograd.')
            print(f'[WMAC] MoE Actor: E={cfg.moe_num_experts}, top_k={cfg.moe_top_k}; '
                  f'{execution} Other heads retain their compilation settings. '
                  f'Router auxiliary coefficients: balance={cfg.moe_load_balance_coef}, z={cfg.moe_router_z_coef}', flush=True)
            print(f'[WMAC] Actor architecture: {json.dumps(actor_architecture(cfg), sort_keys=True)}', flush=True)

    def _compile(self, function, label):
        if not self.cfg.compile:
            return function
        if not hasattr(torch, 'compile'):
            raise RuntimeError('torch.compile is unavailable; select --no-compile')
        return torch.compile(function, mode=self.cfg.compile_mode,
                             fullgraph=self.cfg.compile_fullgraph, dynamic=False)

    def _sync(self):
        if self.cfg.benchmark_sync and self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

    @property
    def wall_time(self):
        return self.wall_offset + time.perf_counter() - self.started

    def _lr_factor(self, iteration):
        progress = min(iteration / max(self.cfg.lr_decay_iterations, 1), 1.0)
        return self.cfg.lr_final_fraction + (1 - self.cfg.lr_final_fraction) * .5 * (1 + math.cos(math.pi * progress))

    def _set_learning_rates(self):
        factor = self._lr_factor(self.iteration)
        for name, base in [('opt_actor', self.cfg.actor_lr), ('opt_model', self.cfg.world_model_lr),
                           ('opt_critic', self.cfg.critic_lr), ('opt_disc', self.cfg.discriminator_lr)]:
            # Official model/actor/critic rates are constant. AMP's separate
            # discriminator retains its existing schedule and optimizer.
            selected_factor = 1. if is_official(self.cfg) and name != 'opt_disc' else factor
            set_learning_rate(getattr(self, name), base * selected_factor)

    def _next_collect_count(self):
        if self.cfg.collect_steps:
            return self.cfg.collect_steps
        self._collect_accumulator += 1.0 / self.cfg.utd
        count = int(self._collect_accumulator + 1e-9)
        self._collect_accumulator -= count
        return count

    def _optimizer_step(self, loss, optimizer, module, actor=False, backward=True):
        if backward:
            optimizer.zero_grad(set_to_none=True)
            if not bool(torch.isfinite(loss.detach())):
                return loss.new_tensor(float('nan')), False
            loss.backward()
        norm = nn.utils.clip_grad_norm_(module.parameters(), self.cfg.max_grad_norm, foreach=True)
        finite = bool(torch.isfinite(norm))
        if finite:
            if actor and self.cfg.actor_grad_value_clip:
                nn.utils.clip_grad_value_(module.parameters(), self.cfg.actor_grad_value_clip, foreach=True)
            optimizer.step()
        else:
            optimizer.zero_grad(set_to_none=True)
        return norm.detach(), finite

    @torch.no_grad()
    def initialize_from_policy(self, path):
        payload = torch.load(path, map_location=self.device, weights_only=False)
        is_wmac = payload.get('format') == 'amp-wmac-v1'
        source_counter = checkpoint_recovery_semantics(payload) if is_wmac else 'physical_termination'
        source_height = checkpoint_height_reward_semantics(payload) if is_wmac else 'unknown_ppo_reward'
        source_architecture = checkpoint_actor_architecture(payload) if is_wmac else {'actor_type': 'ppo_mlp'}
        if is_wmac and not getattr(self.cfg, 'actor_type', 'mlp') == 'moe':
            if (is_official(self.cfg) or is_official(payload.get('config', {}))) and source_architecture != actor_architecture(self.cfg):
                raise ValueError('Official network blocks require matching Actor architecture for initialization; old MLP/PPO weights need a separate distillation experiment')
        target_moe = getattr(self.cfg, 'actor_type', 'mlp') == 'moe'
        if target_moe and (getattr(self.cfg, 'moe_variant', 'legacy') == 'hiking' or
                           source_architecture.get('moe_variant') == 'hiking'):
            if not is_wmac or source_architecture != actor_architecture(self.cfg):
                raise ValueError('Hiking warmstart requires an identical Hiking Actor architecture; '
                                 'MLP/legacy MoE conversion would change widths, normalization or output transform. '
                                 'Start a fresh Hiking run without --init-policy.')
        if is_wmac and source_architecture['actor_type'] == 'moe' and not target_moe:
            raise ValueError('MoE to MLP warmstart needs an explicit distillation experiment; implicit expert averaging is not supported')
        saved_schema = payload.get('amp_feature_schema', payload.get('config', {}).get('amp_feature_schema', 'legacy'))
        if self.cfg.init_discriminator and saved_schema != self.cfg.amp_feature_schema:
            raise ValueError('AMP schema mismatch for discriminator warmstart; use --no-init-discriminator for actor-only initialization')
        if payload.get('format') == 'amp-wmac-v1':
            if target_moe and source_architecture['actor_type'] == 'mlp':
                self.actor.load_mlp_state_dict(payload['actor'], noise_std=self.cfg.moe_init_noise_std)
            else:
                self.actor.load_state_dict(payload['actor'])
            if self.cfg.init_discriminator:
                self.discriminator.load_state_dict(payload['discriminator'])
                self.amp_norm.load_state_dict(payload['amp_norm'])
                self._set_amp_norm_settings(payload.get('amp_normalizer_settings', {}))
        else:
            try:
                if target_moe:
                    self.actor.load_baseline(payload, noise_std=self.cfg.moe_init_noise_std)
                else:
                    self.actor.load_baseline(payload)
            except RuntimeError as error:
                raise ValueError('Warmstart Actor widths differ; use the balanced profile or explicitly match --policy-hidden') from error
            if self.cfg.init_discriminator:
                if 'discriminator_state_dict' not in payload or 'amp_normalizer' not in payload:
                    raise ValueError('Warmstart needs discriminator and AMP statistics; use --no-init-discriminator for an actor-only initialization')
                self.discriminator.load_state_dict(payload['discriminator_state_dict'])
                old = payload['amp_normalizer']
                self.amp_norm.mean.copy_(torch.as_tensor(old.mean, device=self.device, dtype=torch.float32))
                self.amp_norm.var.copy_(torch.as_tensor(old.var, device=self.device, dtype=torch.float32))
                self.amp_norm.count.fill_(float(old.count))
                self._set_amp_norm_settings({'eps': float(old.epsilon), 'variance_epsilon': True,
                                             'clip': float(old.clip_obs)})
        for module in (self.actor, self.discriminator, self.amp_norm):
            if not all(bool(torch.isfinite(value).all()) for value in module.state_dict().values()):
                raise ValueError('Warmstart checkpoint has non-finite parameters/statistics')
        if not hasattr(self, 'recovery_semantics_history'):
            self.recovery_semantics_history = []
        self.recovery_semantics_history.append({
            'kind': 'policy_initialization', 'source_checkpoint': str(Path(path).resolve()),
            'source_sha256': self._checkpoint_sha256(path), 'source_iteration': payload.get('iteration', payload.get('iter')),
            'source_counter': source_counter, 'source_height_reward': source_height,
            'target_counter': self.cfg.recovery_counter_semantics,
            'target_height_reward': self.cfg.recovery_height_reward,
            'scope': 'actor_and_discriminator' if self.cfg.init_discriminator else 'actor_only',
            'critic_model_optimizer_states_loaded': False,
        })
        if not hasattr(self, 'actor_initialization_history'):
            self.actor_initialization_history = []
        cloned_experts = target_moe and (not is_wmac or source_architecture['actor_type'] == 'mlp')
        restored_router = target_moe and is_wmac and source_architecture['actor_type'] == 'moe'
        self.actor_initialization_history.append({
            'kind': 'policy_initialization', 'source_checkpoint': str(Path(path).resolve()),
            'source_sha256': self._checkpoint_sha256(path), 'source_architecture': source_architecture,
            'target_architecture': actor_architecture(self.cfg),
            'expert_head_noise_std': self.cfg.moe_init_noise_std if cloned_experts else 0.,
            'router_initialization': ('checkpoint' if restored_router else 'new_random' if target_moe else 'not_applicable'),
            'actor_optimizer_loaded': False,
            'note': ('MLP experts are cloned before optional head perturbation; router is newly initialized. '
                     'Exact MLP action preservation requires zero perturbation and the same action bound.'
                     if cloned_experts else 'MoE experts and router are restored from the source policy; no clone perturbation is applied.'
                     if restored_router else 'Ordinary MLP policy weights are restored; no router is present.'),
        })
        print(f'[WMAC] Warmstart Actor and D={self.cfg.init_discriminator}; Critic will be recalibrated before Actor updates', flush=True)

    def _set_amp_norm_settings(self, settings):
        for key in ('eps', 'variance_epsilon', 'clip', 'max_count'):
            if key in settings:
                setattr(self.amp_norm, key, settings[key])

    @torch.no_grad()
    def collect(self, count):
        totals = dict(real_task_reward=0., terminated_fraction=0., timeout_fraction=0.,
                      recovery_fraction=0., boundary_fraction=0., invalid_transition_fraction=0.)
        ticks = [] if self.device.type == 'cuda' and is_official(self.cfg) else None
        for _ in range(count):
            obs = self.env.obs_buf['actor']
            finite_obs = torch.isfinite(obs).all(-1)
            if getattr(self.actor, 'normalize_observations', True):
                self.actor.norm.update(obs[finite_obs & self._train_env_mask])
            safe_obs = torch.nan_to_num(obs, nan=0., posinf=0., neginf=0.)
            action = (self.actor(safe_obs) + self.cfg.exploration_noise * torch.randn(
                self.env.num_envs, self.num_joints, device=self.device)).clamp(-self.cfg.action_limit, self.cfg.action_limit)
            if not bool(torch.isfinite(action).all()):
                raise RuntimeError('Non-finite policy action: refusing to contaminate the physical simulation')
            self.env.step(action)
            batch = self.env.transition
            accepted = self.replay.add(batch)
            train = accepted & self._train_env_mask
            self.world_model.update_normalizers(batch['state'][train], batch['action'][train],
                                                batch['context'][train], batch['next_state'][train])
            self.critic.norm.update(self.adapter.critic_features(batch['state'][train],
                batch['previous_action'][train], batch['command'][train], batch['context'][train]))
            self.sim_steps += self.env.num_envs
            if ticks is not None:
                ticks.append(_collect_tick_statistics(batch, accepted, self.recovery_height))
            else:
                totals['real_task_reward'] += float(batch['reward'][accepted].mean()) if accepted.any() else 0.
                totals['terminated_fraction'] += float(batch['terminated'].float().mean())
                totals['timeout_fraction'] += float(batch['timeout'].float().mean())
                totals['boundary_fraction'] += float(batch['boundary'].float().mean())
                totals['recovery_fraction'] += float(((batch['state'][:, 2] < self.recovery_height) | (batch['context'][:, -1] > 0)).float().mean())
                totals['invalid_transition_fraction'] += float((~accepted).float().mean())
        if ticks:
            average = (torch.stack(ticks).sum(0, dtype=torch.float64) / count).cpu().tolist()
            return dict(zip(totals, average))
        return {key: value / max(count, 1) for key, value in totals.items()}

    def _world_model_loss(self, states, actions, contexts, targets, contacts, mask):
        state = states[:, 0]
        errors, contact_errors = [], []
        for step in range(self.cfg.wm_window):
            state = torch.where(mask[:, step, None].bool(), state, states[:, step])
            predicted, contact = self.world_model.step_with_contact(state, actions[:, step], contexts[:, step])
            errors.append(self.world_model.loss(predicted, targets[:, step]))
            contact_errors.append(F.smooth_l1_loss(torch.log1p(contact), torch.log1p(contacts[:, step].clamp_min(0)), reduction='none').mean(-1))
            state = predicted
        return torch.stack(errors, 1), torch.stack(contact_errors, 1)

    def fit_model(self):
        records = []
        batch_statistics = self.device.type == 'cuda' and is_official(self.cfg)
        for _ in range(self.cfg.wm_updates):
            batch = _device_batch(self.replay.sample_windows(self.cfg.wm_batch_size, self.cfg.wm_window,
                self.cfg.recovery_fraction, fields=self.WM_FIELDS), self.device)
            errors, contact_errors = self._wm_loss_fn(batch['state'], batch['action'], batch['context'],
                batch['next_state'], batch['contact_cost'], batch['mask'])
            mask = batch['mask']
            denom = mask.sum().clamp_min(1)
            dyn = (errors * mask).sum() / denom
            contact = (contact_errors * mask).sum() / denom
            norm, ok = self._optimizer_step(dyn + contact, self.opt_model, self.world_model)
            recovery = (batch['state'][:, 0, 2] < self.recovery_height) | (batch['context'][:, 0, -1] > 0)
            sample_error = (errors.detach() * mask).sum(1) / mask.sum(1).clamp_min(1)
            if batch_statistics:
                phases = _masked_stat_mean(sample_error, torch.stack((~recovery, recovery)), empty=float('nan')).unbind()
            else:
                phases = [sample_error[selection].mean() if selection.any() else dyn.new_tensor(float('nan'))
                          for selection in (~recovery, recovery)]
            end_mask = mask[:, -1]
            last = (errors[:, -1].detach() * end_mask).sum() / end_mask.sum().clamp_min(1)
            records.append(torch.stack((dyn.detach(), contact.detach(), last, norm, *phases,
                                        dyn.new_tensor(float(ok)))))
        average = torch.stack(records).mean(0).detach().cpu().tolist()
        result = dict(zip(('wm_loss', 'wm_contact_loss', 'wm_last_error', 'wm_grad_norm',
                           'wm_walk_error', 'wm_recovery_error', 'wm_updates_applied_fraction'), average))
        self.last_model_loss = result['wm_loss']
        valid = [result[k] for k in ('wm_loss','wm_walk_error','wm_recovery_error') if math.isfinite(result[k])]
        result['wm_gate_error'] = max(valid) if math.isfinite(result['wm_loss']) else float('inf')
        return result

    def _amp_reward(self, first, second):
        if self.cfg.precision == 'bf16' and first.device.type == 'cuda':
            with torch.autocast('cuda', dtype=torch.bfloat16):
                score = self.discriminator(torch.cat((self.amp_norm(first), self.amp_norm(second)), -1)).float().squeeze(-1)
        else:
            score = self.discriminator(torch.cat((self.amp_norm(first), self.amp_norm(second)), -1)).squeeze(-1)
        if self.cfg.amp_reward_mapping == 'instantamp_centered_tanh_v1':
            return self.amp_reward_stats(score), score
        return clipped_lsgan_reward(score, self.cfg.amp_reward_coef), score

    def update_discriminator(self):
        records = []
        size = self.cfg.effective_discriminator_batch_size
        if not self.cfg.amp_normalizer_per_update:
            with torch.no_grad():
                batch = _device_batch(self.replay.sample_starts(size,
                    self.cfg.recovery_fraction, fields=('state','next_state')), self.device)
                first, second = self.adapter.amp_features(batch['state']), self.adapter.amp_features(batch['next_state'])
                ef, es = self.experts.sample(len(first))
                self.amp_norm.update(torch.cat((first, second, ef, es)))
        for _ in range(self.cfg.discriminator_updates):
            batch = _device_batch(self.replay.sample_starts(size,
                self.cfg.recovery_fraction, fields=('state','next_state')), self.device)
            with torch.no_grad():
                first, second = self.adapter.amp_features(batch['state']), self.adapter.amp_features(batch['next_state'])
                policy = torch.cat((self.amp_norm(first), self.amp_norm(second)), -1)
                ef, es = self.experts.sample(size)
                expert = torch.cat((self.amp_norm(ef), self.amp_norm(es)), -1)
            # Keep the zero-centered input-gradient penalty in fp32.
            expert = expert.detach().requires_grad_(True)
            policy_score, expert_score = self.discriminator(policy), self.discriminator(expert)
            input_grad = torch.autograd.grad(expert_score.sum(), expert, create_graph=True)[0]
            gp = self.cfg.gradient_penalty * input_grad.square().sum(-1).mean()
            loss = .5 * ((expert_score - 1).square().mean() + (policy_score + 1).square().mean()) + gp
            norm, ok = self._optimizer_step(loss, self.opt_disc, self.discriminator)
            if ok and self.cfg.amp_reward_mapping == 'instantamp_centered_tanh_v1':
                self.amp_reward_stats.update(policy_score, expert_score)
            if self.cfg.amp_normalizer_per_update and ok:
                self.amp_norm.update(torch.cat((first, second, ef, es)))
            records.append(torch.stack((loss.detach(), gp.detach(), norm)))
        self.opt_disc.zero_grad(set_to_none=True)
        metrics = torch.stack(records).mean(0)
        keys = ('discriminator_loss','discriminator_grad_penalty','discriminator_grad_norm')
        if self.cfg.amp_reward_mapping == 'instantamp_centered_tanh_v1':
            stats = self.amp_reward_stats
            metrics = torch.cat((metrics, torch.stack((stats.robot_mean, stats.expert_mean,
                stats.initialized.float(), stats.updates.float()))))
            keys += ('amp_robot_score_ema','amp_expert_score_ema','amp_score_initialized','amp_score_ema_updates')
        return dict(zip(keys, metrics.cpu().tolist()))

    def _imagined_step(self, state, context, history, command, noise):
        action = self.actor(history.flatten(1))
        if is_official(self.cfg):
            action = action.clamp(-self.cfg.action_limit, self.cfg.action_limit)
        predicted, contact = self.world_model.step_with_contact(state, action, context)
        next_context, terminated = self.adapter.advance_context(predicted, context)
        observation = self.adapter.actor_frame(predicted, action, command, next_context, noise=False) + noise
        history = torch.cat((history[:, 1:], observation.unsqueeze(1)), 1)
        return predicted, action, next_context, terminated, contact, history

    def _value(self, features):
        # Calibration/target inference runs infrequently and needs no backward
        # graph. Avoid separate Inductor specializations for these cold shapes.
        return self._value_fn(features) if torch.is_grad_enabled() else self.critic(features)

    def _cap_values(self, values):
        if not is_official(self.cfg):
            return values
        cap = getattr(self, '_critic_vmax', None)
        if cap is None:
            cap = amp_value_upper_bound(self.adapter, self.cfg)
        return values.clamp(max=cap)

    def _observation_noise(self, batch):
        frame, joints = self.actor_frame_dim, self.num_joints
        scale = torch.zeros(frame, device=self.device)
        scale[:3], scale[3:6], scale[9:9+joints], scale[9+joints:9+2*joints] = .2, .05, .01, .5
        return (2 * torch.rand(self.cfg.imagination_horizon, batch, frame, device=self.device) - 1) * scale

    def _task_reward(self, state, next_state, action, previous_action, command,
                     next_context, terminated, contact_cost, kinematics=None, *,
                     previous_previous=None, current_context=None):
        if getattr(self.adapter, 'recovery_foot_slip', None) is not None:
            task = self.adapter.task_reward(state, next_state, action, previous_action, command,
                next_context, terminated, contact_cost, kinematics=kinematics, current_context=current_context)
        else:
            task = self.adapter.task_reward(state, next_state, action, previous_action, command,
                next_context, terminated, contact_cost, kinematics=kinematics)
        if smooth_rewards_enabled(self.cfg):
            if previous_previous is None or current_context is None:
                raise ValueError('Smooth task reward requires true previous-previous action and current context')
            raw = self.smooth_reward_terms(state, next_state, action, previous_action,
                                          previous_previous, current_context)
            task = task - self.adapter.reward_scale * smooth_cost_rate(raw, self.cfg)
        return task

    def _trajectory_rewards(self, states, contexts, actions, previous, command, terminated, contact,
                            previous_previous=None):
        # One batched FK over H+1 states; shared by task and both AMP pair ends.
        fk = self.adapter.analyze_state(states)
        features = self.adapter.amp_features(states, kinematics=fk)
        task = self._task_reward(states[:-1], states[1:], actions, previous,
            command.unsqueeze(0).expand(actions.shape[0], -1, -1), contexts[1:], terminated,
            contact, kinematics=tuple(value[1:] for value in fk),
            previous_previous=previous_previous, current_context=contexts[:-1])
        first, second = features[:-1].flatten(0,1), features[1:].flatten(0,1)
        styles, scores = [], []
        for start in range(0, first.shape[0], self.cfg.reward_chunk_size):
            style, score = self._amp_reward(first[start:start+self.cfg.reward_chunk_size],
                                            second[start:start+self.cfg.reward_chunk_size])
            styles.append(style); scores.append(score)
        return task, torch.cat(styles).reshape_as(task), torch.cat(scores).reshape_as(task)

    def imagine(self, batch, noises=None):
        state, context = batch['state'], batch['context']
        previous, command = batch['previous_action'], batch['command']
        history = batch['actor_history'].reshape(len(state),4,self.actor_frame_dim)
        states, contexts, actions, previous_actions, terminated, contacts = [state], [context], [], [], [], []
        noises = self._observation_noise(len(state)) if noises is None else noises
        step_fn = self._imag_step_fn if torch.is_grad_enabled() else self._imagined_step
        reward_fn = self._reward_fn if torch.is_grad_enabled() else self._trajectory_rewards
        for step in range(self.cfg.imagination_horizon):
            previous_actions.append(previous)
            state, action, context, done, contact, history = step_fn(state, context, history, command, noises[step])
            states.append(state); contexts.append(context); actions.append(action)
            terminated.append(done); contacts.append(contact)
            previous = action
        states, contexts = torch.stack(states), torch.stack(contexts)
        actions, previous_actions = torch.stack(actions), torch.stack(previous_actions)
        terminated, contacts = torch.stack(terminated), torch.stack(contacts)
        if smooth_rewards_enabled(self.cfg):
            initial_older = previous_previous_action(batch['actor_history'])
            older_actions = torch.cat((initial_older.unsqueeze(0), previous_actions[:-1]), 0)
            task, style, score = reward_fn(states, contexts, actions, previous_actions,
                                          command, terminated, contacts, older_actions)
        else:
            task, style, score = reward_fn(states, contexts, actions, previous_actions,
                                          command, terminated, contacts)
        direct_style = style if self.cfg.direct_style_grad else style.detach()
        rewards = self.cfg.task_weight * task + self.cfg.style_weight * direct_style
        continuation = 1 - terminated
        if not self.cfg.survival_grad:
            continuation = continuation.detach()
        alive = torch.cumprod(continuation, 0)
        c_t = torch.cat((torch.ones_like(alive[:1]), alive[:-1]),0)
        discounts = self.cfg.gamma ** torch.arange(self.cfg.imagination_horizon,device=self.device,dtype=torch.float32)
        objective = (discounts[:,None] * c_t * rewards).sum(0)
        previous_states = torch.cat((batch['previous_action'].unsqueeze(0), actions),0)
        critic_x = self.adapter.critic_features(states, previous_states,
            command.unsqueeze(0).expand(states.shape[0],-1,-1), contexts)
        if self.cfg.critic_bootstrap:
            objective = objective + self.cfg.gamma**self.cfg.imagination_horizon * alive[-1] * self._cap_values(self._value(critic_x[-1]))
        # Targets are cached before any optimizer step and never train other nets.
        with torch.no_grad():
            old_values = self._value(critic_x.detach())
            targets = lambda_returns((self.cfg.task_weight*task+self.cfg.style_weight*style).detach(),
                continuation.detach(), old_values, self.cfg.gamma, self.cfg.gae_lambda)
            targets = self._cap_values(targets)
        return {'objective':objective, 'x':critic_x[:-1].detach(), 'targets':targets.detach(),
                'task':task.detach(), 'style':style.detach(), 'score':score.detach(),
                'survival':alive[-1].detach(), 'first_action':actions[0]}

    def _real_critic_batch(self, size):
        # This auxiliary TD anchor is an engineering extension, not the paper's
        # imagined lambda-return. Keep its batch bounded and refresh D rewards.
        batch = _device_batch(self.replay.sample_starts(size, self.cfg.recovery_fraction,
            fields=self.REAL_FIELDS), self.device)
        with torch.no_grad():
            states = torch.stack((batch['state'],batch['next_state']))
            phi = self.adapter.amp_features(states)
            style, _ = self._amp_reward(phi[0],phi[1])
            x = self.adapter.critic_features(batch['state'],batch['previous_action'],batch['command'],batch['context'])
            nx = self.adapter.critic_features(batch['next_state'],batch['action'],batch['command'],batch['next_context'])
            reward = self.cfg.task_weight*batch['reward']+self.cfg.style_weight*style
            target = reward+self.cfg.gamma*(~batch['terminated']).float()*self._value(nx)
        return x.detach(), target.detach()

    def _fit_critic(self, x, targets):
        xf, yf = x.flatten(0,1).detach(), targets.flatten().detach()
        real_x, real_y = (self._real_critic_batch(min(len(xf),self.cfg.real_critic_batch_size))
                          if self.cfg.real_critic_coef else (None,None))
        records = []
        minibatch_count = self.cfg.critic_minibatches
        reference_minibatches = None
        if is_official(self.cfg):
            indices = torch.randperm(len(xf)//minibatch_count*minibatch_count, device=self.device)
            reference_minibatches = tuple(zip(xf[indices].tensor_split(minibatch_count),
                                              yf[indices].tensor_split(minibatch_count)))
        for _ in range(self.cfg.critic_epochs):
            if reference_minibatches is None:
                indices = torch.randperm(len(xf), device=self.device).tensor_split(minibatch_count)
                minibatches = ((xf[idx], yf[idx]) for idx in indices if len(idx))
            else:
                minibatches = reference_minibatches
            for features, target in minibatches:
                if not len(features):
                    continue
                loss = F.mse_loss(self._value_fn(features), target)
                real_loss = loss.new_zeros(())
                if real_x is not None:
                    ri = torch.randint(len(real_x),(min(len(features),len(real_x)),),device=self.device)
                    real_loss = F.mse_loss(self._value_fn(real_x[ri]),real_y[ri])
                norm, ok = self._optimizer_step(loss+self.cfg.real_critic_coef*real_loss,self.opt_critic,self.critic)
                records.append(torch.stack((loss.detach(),real_loss.detach(),norm,loss.new_tensor(float(ok)))))
        values = torch.stack(records).mean(0).cpu().tolist()
        return dict(zip(('critic_loss','real_critic_loss','critic_grad_norm','critic_updates_applied_fraction'),values))

    def update_actor_critic(self, update_actor=True):
        batch = _device_batch(self.replay.sample_starts(self.cfg.actor_batch_size,
            self.cfg.recovery_fraction,fields=self.ACTOR_FIELDS),self.device)
        self.opt_model.zero_grad(set_to_none=True); self.opt_disc.zero_grad(set_to_none=True)
        self.opt_critic.zero_grad(set_to_none=True); self.opt_actor.zero_grad(set_to_none=True)
        with torch.no_grad():
            old_action = self.actor(batch['actor_history']).clone()
            recovery = (batch['state'][:, 2] < self.recovery_height) | (batch['context'][:, -1] > 0)
            router_stats = routing_metrics(self.actor, batch['actor_history'], recovery)
            if is_moe(self.actor):
                real_route = self.actor.route(batch['actor_history'])
                full_router_load = load_fraction(real_route.indices, self.actor.num_experts,
                                                  real_route.probabilities.dtype)
            else:
                full_router_load = None
        parameter_snapshot = [parameter.detach().clone() for parameter in self.actor.parameters()] if update_actor else None
        optimizer_snapshot = copy.deepcopy(self.opt_actor.state_dict()) if update_actor else None
        size = self.cfg.actor_microbatch_size or self.cfg.actor_batch_size
        records, xs, ys, router_records = [], [], [], []
        finite_loss = True
        with frozen_parameters(self.world_model,self.discriminator,self.critic):
            for start in range(0,len(batch['state']),size):
                chunk = {key:value[start:start+size] for key,value in batch.items()}
                with torch.set_grad_enabled(update_actor):
                    rollout = self.imagine(chunk)
                    task_loss = -rollout['objective'].mean()
                    auxiliary = task_loss.new_zeros(())
                    if is_moe(self.actor) and update_actor:
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
                    if update_actor:
                        finite = bool(torch.isfinite(loss.detach()))
                        finite_loss = finite_loss and finite
                        if finite:
                            (loss * (len(chunk['state'])/len(batch['state']))).backward()
                xs.append(rollout['x']); ys.append(rollout['targets'])
                if self.cfg.amp_reward_mapping == 'instantamp_centered_tanh_v1':
                    # Measures the actual score-to-reward derivative, including
                    # floating-point tanh saturation; not the whole actor Jacobian.
                    zero_slope = self.amp_reward_stats.score_slope(rollout['score']) == 0
                else:
                    zero_slope = (rollout['score']<=-1)|(rollout['score']>=3)
                record = torch.stack((loss.detach(),rollout['task'].mean(),rollout['style'].mean(),
                    zero_slope.float().mean(),rollout['survival'].mean()))
                records.append((len(chunk['state']),record))
                router_records.append((len(chunk['state']), torch.stack((task_loss.detach(), auxiliary.detach()))))
                del rollout
        if is_moe(self.actor):
            expert_norms = [parameter_grad_norm(expert) for expert in self.actor.experts]
            router_stats.update(moe_router_grad_norm=parameter_grad_norm(self.actor.router),
                                moe_expert_grad_norm_mean=sum(expert_norms)/len(expert_norms),
                                moe_expert_grad_norm_max=max(expert_norms))
        norm, stepped = old_action.new_zeros(()), False
        if update_actor and finite_loss:
            norm, stepped = self._optimizer_step(None,self.opt_actor,self.actor,actor=True,backward=False)
        with torch.no_grad():
            shift = (self.actor(batch['actor_history'])-old_action).square().mean().sqrt()
        rejected = update_actor and (not stepped or not bool(torch.isfinite(shift)) or
            (not is_official(self.cfg) and float(shift)>self.cfg.max_action_shift))
        if rejected:
            with torch.no_grad():
                for parameter, saved in zip(self.actor.parameters(),parameter_snapshot): parameter.copy_(saved)
            self.opt_actor.load_state_dict(optimizer_snapshot)
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
        self.imagination_steps += self.cfg.imagination_horizon*len(batch['state'])
        return result

    @torch.no_grad()
    def validate_model(self):
        split = 'validation' if self.cfg.validation_env_fraction else 'train'
        reward_fields = ('actor_history',) if smooth_rewards_enabled(self.cfg) else ()
        batch = _device_batch(self.replay.sample_windows(self.cfg.validation_batch_size,self.cfg.wm_window,
            self.cfg.recovery_fraction,fields=(*self.WM_FIELDS,'previous_action','command','next_context','reward','terminated',*reward_fields),
            split=split),self.device)
        errors, contact_error = self._world_model_loss(batch['state'],batch['action'],batch['context'],
            batch['next_state'],batch['contact_cost'],batch['mask'])
        mask=batch['mask']; denominator=mask.sum().clamp_min(1)
        sample=(errors*mask).sum(1)/mask.sum(1).clamp_min(1)
        recovery=(batch['state'][:,0,2]<self.recovery_height)|(batch['context'][:,0,-1]>0)
        phase=[]
        for selection in (~recovery,recovery):
            phase.append(float(sample[selection].mean()) if selection.any() else float('nan'))
        state=batch['state'][:,0];action=batch['action'][:,0];context=batch['context'][:,0]
        predicted, contact=self.world_model.step_with_contact(state,action,context)
        ctx,term=self.adapter.advance_context(predicted,context)
        older = (previous_previous_action(batch['actor_history'][:,0])
                 if smooth_rewards_enabled(self.cfg) else None)
        task=self._task_reward(state,predicted,action,batch['previous_action'][:,0],
            batch['command'][:,0],ctx,term,contact,previous_previous=older,current_context=context)
        values={'validation_model_error':float((errors*mask).sum()/denominator),
            'validation_contact_error':float((contact_error*mask).sum()/denominator),
            'validation_reward_gap':float((task-batch['reward'][:,0]).abs().mean()),
            'validation_walk_error':phase[0],'validation_recovery_error':phase[1]}
        self.validation=values
        return values

    def _actor_gate(self,row):
        if (self.cfg.amp_reward_mapping == 'instantamp_centered_tanh_v1'
                and row.get('amp_score_initialized', 0.) != 1.):
            # Reuse the already transferred D metrics, with no new GPU sync.
            # A failed first D step cannot train the actor against center=0.
            return False
        if is_official(self.cfg):
            # Upstream has no validation-error or action-shift training gate.
            # Keep non-finite/failed-optimizer protection and log validation.
            return math.isfinite(row['wm_gate_error']) and row['wm_updates_applied_fraction'] == 1.
        if not math.isfinite(row['wm_gate_error']) or row['wm_gate_error']>self.cfg.max_model_loss_for_actor:
            return False
        if row['wm_updates_applied_fraction']<1.:
            return False
        if self.cfg.validation_env_fraction:
            if not all(math.isfinite(self.validation[k]) for k in ('validation_model_error','validation_contact_error')):
                return False
            errors=[self.validation[k] for k in ('validation_model_error','validation_walk_error','validation_recovery_error')
                    if math.isfinite(self.validation[k])]
            return bool(errors) and max(errors)<=self.cfg.max_validation_model_error and self.validation['validation_contact_error']<=self.cfg.max_validation_contact_error
        return True

    def _log(self,row):
        row={'iteration':self.iteration,'wall_seconds':self.wall_time,'sim_steps':self.sim_steps,
             'imagination_steps':self.imagination_steps,**row}
        if self.csv_writer is None:
            self.csv_file=(self.log_dir/'metrics.csv').open('a',newline='')
            self.csv_writer=csv.DictWriter(self.csv_file,fieldnames=list(row))
            if self.csv_file.tell()==0:self.csv_writer.writeheader()
        self.csv_writer.writerow(row);self.csv_file.flush()
        for key,value in row.items():
            if isinstance(value,(int,float)):self.writer.add_scalar(key,value,self.sim_steps)
        if self.iteration%self.cfg.log_interval==0:
            self.writer.flush()
            print(f'[WMAC] iter={self.iteration} sim={self.sim_steps} imag={self.imagination_steps} '
                  f'wall={self.wall_time:.1f}s wm={row["wm_loss"]:.3g} val={row["validation_model_error"]:.3g} '
                  f'actor={row["actor_loss"]:.3g} active={int(row["actor_active"])}',flush=True)

    def learn(self):
        try:
            if self.replay.size<self.cfg.wm_window:
                self.collect(max(self.cfg.warmup_steps,self.cfg.wm_window))
            for iteration in range(self.iteration,self.cfg.iterations):
                self.iteration=iteration;self._set_learning_rates();self._sync();started=time.perf_counter()
                count=self._next_collect_count();row=self.collect(count);self._sync();t1=time.perf_counter()
                row.update(self.fit_model());self._sync();t2=time.perf_counter()
                row.update(self.update_discriminator());self._sync();t3=time.perf_counter()
                if iteration%self.cfg.validation_interval==0 or not math.isfinite(self.validation['validation_model_error']):
                    self.validate_model()
                gate=self._actor_gate(row)
                needs_calibration=(self.cfg.critic_bootstrap and
                    self.critic_calibration_updates<self.cfg.critic_warmup_iterations)
                active=iteration>=self.cfg.model_warmup_iterations and gate and not needs_calibration
                calibrate=(needs_calibration and gate and not active and
                    iteration>=max(0,self.cfg.model_warmup_iterations-self.cfg.critic_warmup_iterations))
                actor_keys=('actor_loss','actor_grad_norm','actor_update_rejected','actor_action_shift','critic_loss',
                    'real_critic_loss','critic_grad_norm','critic_updates_applied_fraction','imagined_task_reward',
                    'imagined_style_reward','style_zero_gradient_fraction','imagined_survival')
                actor_keys = (*actor_keys, *routing_metric_keys(getattr(self, 'actor', None)))
                row.update({key:0. for key in actor_keys})
                if active or calibrate:
                    samples=[self.update_actor_critic(update_actor=active) for _ in range(self.cfg.actor_updates)]
                    row.update({key:float(np.mean([sample[key] for sample in samples])) for key in actor_keys})
                    if calibrate and row['critic_updates_applied_fraction']==1. and math.isfinite(row['critic_loss']):
                        self.critic_calibration_updates += 1
                self._sync();t4=time.perf_counter();self.iteration=iteration+1
                row.update(self.validation)
                row.update(actor_active=float(active),critic_calibration=float(calibrate),collect_steps=count,
                    critic_calibration_updates=self.critic_calibration_updates,
                    learning_rate=float(self.opt_actor.param_groups[0]['lr']),collect_seconds=t1-started,
                    model_seconds=t2-t1,disc_seconds=t3-t2,actor_critic_seconds=t4-t3,
                    save_seconds=0.,evaluation_seconds=0.)
                if self.iteration%self.cfg.save_interval==0 or self.iteration==self.cfg.iterations:
                    begin=time.perf_counter();self.save(self.log_dir/f'model_{self.iteration}.pt');row['save_seconds']=time.perf_counter()-begin
                if self.cfg.evaluation_interval and self.iteration%self.cfg.evaluation_interval==0:
                    begin=time.perf_counter();self.evaluate_checkpoint();row['evaluation_seconds']=time.perf_counter()-begin
                duration=time.perf_counter()-started
                row['physical_steps_per_second']=count*self.env.num_envs/max(duration,1e-8)
                row['imagined_steps_per_second']=self.cfg.actor_updates*self.cfg.actor_batch_size*self.cfg.imagination_horizon/max(t4-t3,1e-8) if active else 0.
                self._log(row)
        finally:
            self.writer.close()
            if self.csv_file:self.csv_file.close()

    @staticmethod
    def _checkpoint_sha256(path):
        digest = hashlib.sha256()
        with Path(path).open('rb') as stream:
            for block in iter(lambda: stream.read(4 << 20), b''):
                digest.update(block)
        return digest.hexdigest()

    def _motion_dataset_metadata(self):
        if not self.cfg.mirror_walk_motions:
            return None
        root = getattr(self.env, '_wmac_motion_root', None)
        if root is None:
            raise ValueError('Mirrored training requires the configured reset/expert motion dataset')
        manifest = json.loads((Path(root)/'mirror_manifest.json').read_text())
        content = {key: manifest[key] for key in ('format', 'motion_scope', 'source_sha256', 'output_sha256')}
        # G1 manifests name the model hash 'g1_xml_sha256'; other robots 'robot_xml_sha256'.
        key = 'g1_xml_sha256' if 'g1_xml_sha256' in manifest['robot_mapping'] else 'robot_xml_sha256'
        content[key] = manifest['robot_mapping'][key]
        fingerprint = hashlib.sha256(json.dumps(content, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        return dict(format=content['format'], content_sha256=fingerprint,
                    scope=content['motion_scope'], walking_clips=manifest['walking_total_clips'],
                    recovery_clips=manifest['recovery_original_clips'])

    def save(self,path,export=True):
        path=Path(path)
        elapsed = self.wall_time
        actor_version = (6 if self.cfg.moe_variant == 'hiking' else 5) if self.cfg.actor_type == 'moe' else 4
        if config_protocol(self.cfg) != dict(reference_alignment='legacy_v1', optimizer_algorithm='adam', mirror_walk_motions=False):
            actor_version = 7
        if self.cfg.amp_reward_mapping == 'instantamp_centered_tanh_v1':
            actor_version = 11
            CenteredAmpReward.validate_state(self.amp_reward_stats.state_dict())
        if smooth_rewards_enabled(self.cfg):
            actor_version = 12
        foot_slip = imagined_foot_slip_protocol(self.cfg)
        if foot_slip['imagined_recovery_foot_slip'] in IMAGINED_FOOT_SLIP_GEOMETRIC_MODES:
            actor_version = 13
        payload={'format':'amp-wmac-v1','implementation_version':actor_version,'config':asdict(self.cfg),
            'training_protocol':config_protocol(self.cfg),
            'amp_reward_protocol':amp_reward_protocol(self.cfg),
            'smooth_reward_protocol':smooth_reward_protocol(self.cfg),
            'smooth_reward_migration_history':getattr(self,'smooth_reward_migration_history',[]),
            'smooth_reward_retune_history':getattr(self,'smooth_reward_retune_history',[]),
            'amp_reward_migration_history':getattr(self,'amp_reward_migration_history',[]),
            'optimizer_algorithms':{name:optimizer_algorithm_name(getattr(self,name)) for name in ('opt_actor','opt_critic','opt_model','opt_disc')},
            'motion_dataset':self._motion_dataset_metadata(),
            'critic_value_upper_bound':amp_value_upper_bound(self.adapter,self.cfg) if is_official(self.cfg) else None,
            'actor_type':self.cfg.actor_type,'actor_architecture':actor_architecture(self.cfg),
            'actor_initialization_history':getattr(self,'actor_initialization_history',[]),
            'recovery_counter_semantics':self.cfg.recovery_counter_semantics,
            'recovery_height_reward':self.cfg.recovery_height_reward,
            'recovery_semantics_history':getattr(self,'recovery_semantics_history',[]),
            'recovery_derivative': 'biased_straight_through_delayed_hazard_detached_clock' if self.cfg.recovery_counter_semantics == 'hard_forward_v2' else 'legacy_smooth_counter',
            'amp_feature_schema':self.cfg.amp_feature_schema,
            'amp_feature_dim':amp_feature_dim(self.cfg.amp_feature_schema),
            'amp_feature_metadata':amp_feature_metadata(self.cfg.amp_feature_schema,
                foot_reference_offsets=getattr(getattr(self.adapter, 'profile', G1), 'amp_foot_offsets', None)),
            'iteration':self.iteration,'sim_steps':self.sim_steps,'imagination_steps':self.imagination_steps,
            'wall_seconds':elapsed,'collect_accumulator':self._collect_accumulator,
            'training_cost': {'physical_transitions': self.sim_steps, 'wall_seconds': elapsed,
                'trajectory_id': getattr(self, '_cost_trajectory_id', None),
                'gpu_count': 1 if self.device.type == 'cuda' else 0,
                'wall_time_scope': WALL_TIME_SCOPE,
                'history_complete': getattr(self, '_cost_history_complete', True)},
            'env_common_step_counter':self.env.common_step_counter,'validation':self.validation,
            'best_eval_score':self.best_eval_score,
            'critic_calibration_updates':self.critic_calibration_updates,
            'amp_normalizer_settings':{key:getattr(self.amp_norm,key) for key in ('eps','variance_epsilon','clip','max_count')},
            'torch_rng':torch.get_rng_state(),'numpy_rng':np.random.get_state(),'python_rng':random.getstate()}
        if torch.cuda.is_available():payload['cuda_rng']=torch.cuda.get_rng_state_all()
        for name in ('actor','critic','world_model','discriminator','amp_norm','opt_actor','opt_critic','opt_model','opt_disc'):
            payload[name]=getattr(self,name).state_dict()
        if self.cfg.amp_reward_mapping == 'instantamp_centered_tanh_v1':
            payload['amp_reward_state'] = self.amp_reward_stats.state_dict()
        if smooth_rewards_enabled(self.cfg):
            payload['smooth_reward_model'] = self.smooth_reward_terms.metadata()
        if actor_version == 13:
            payload['imagined_foot_slip_protocol'] = foot_slip
            payload['imagined_foot_slip_migration_history'] = getattr(self, 'imagined_foot_slip_migration_history', [])
        if self.cfg.checkpoint_replay:payload['replay']=self.replay.state_dict()
        temporary=path.with_suffix('.pt.tmp');torch.save(payload,temporary);os.replace(temporary,path)
        if export:self.export_policy(path.parent/'policy.onnx')

    def load(self,path,*,allow_recovery_semantics_migration=False):
        payload=torch.load(path,map_location=self.device,weights_only=False)
        if payload.get('format')!='amp-wmac-v1':raise ValueError('Use --init-policy for a PPO checkpoint')
        saved_actor_architecture = checkpoint_actor_architecture(payload)
        if saved_actor_architecture != actor_architecture(self.cfg):
            raise ValueError(f'Actor architecture mismatch: saved={saved_actor_architecture}, '
                             f'target={actor_architecture(self.cfg)}. Use --init-policy for a separate Actor experiment; '
                             'resume cannot change MLP/MoE structure, routing or temperature.')
        amp_migrated = validate_training_protocol_resume(payload, self.cfg)
        smooth_migrated = validate_smooth_reward_resume(payload, self.cfg)
        smooth_retuned = validate_smooth_reward_retune(payload, self.cfg)
        foot_slip_migrated = validate_imagined_foot_slip_resume(payload, self.cfg)
        if smooth_rewards_enabled(self.cfg):
            expected = self.smooth_reward_terms.metadata()
            if not smooth_migrated and payload.get('smooth_reward_model') != expected:
                raise ValueError('Smooth reward motor constants/mapping differ from the saved protocol')
            if (smooth_migrated or smooth_retuned) and getattr(self.replay, 'size', 0):
                raise ValueError('Smooth reward migration/retune requires an empty target replay')
        if self.cfg.amp_reward_mapping == 'instantamp_centered_tanh_v1' and not amp_migrated:
            CenteredAmpReward.validate_state(payload.get('amp_reward_state'))
        if payload.get('implementation_version', 1) >= 7:
            expected_optimizers = {name:optimizer_algorithm_name(getattr(self,name)) for name in ('opt_actor','opt_critic','opt_model','opt_disc')}
            if payload.get('optimizer_algorithms') != expected_optimizers:
                raise ValueError('Checkpoint optimizer algorithms conflict with the requested runtime')
            if payload.get('motion_dataset') != self._motion_dataset_metadata():
                raise ValueError('Checkpoint mirrored motion content differs from the configured dataset')
            expected_cap = amp_value_upper_bound(self.adapter,self.cfg) if is_official(self.cfg) else None
            if payload.get('critic_value_upper_bound') != expected_cap:
                raise ValueError('Checkpoint critic reward upper bound differs from the configured AMP objective')
        for name in ('opt_actor','opt_critic','opt_model'):
            optimizer = getattr(self,name)
            if optimizer_algorithm_name(optimizer) == 'hybrid_supermimic_muon_adam_v1':
                validate_reference_optimizer_state(optimizer, payload[name], label=name)
        if self.cfg.actor_type == 'moe':
            for key in ('moe_load_balance_coef', 'moe_router_z_coef'):
                if payload['config'].get(key) != getattr(self.cfg, key):
                    raise ValueError(f'Resume mismatch for {key}; start a separate controlled Actor experiment')
        saved_schema = payload.get('amp_feature_schema', payload['config'].get('amp_feature_schema', 'legacy'))
        if saved_schema != self.cfg.amp_feature_schema or payload.get('amp_feature_dim', amp_feature_dim(saved_schema)) != amp_feature_dim(self.cfg.amp_feature_schema):
            raise ValueError('AMP feature schema mismatch; legacy 195D and grounded_v2 203D checkpoints cannot be resumed interchangeably')
        saved_counter = checkpoint_recovery_semantics(payload)
        saved_height = checkpoint_height_reward_semantics(payload)
        semantic_changes = {}
        for key, source, target in (
            ('recovery_counter_semantics', saved_counter, self.cfg.recovery_counter_semantics),
            ('recovery_height_reward', saved_height, self.cfg.recovery_height_reward),
        ):
            if source != target:
                semantic_changes[key] = {'source': source, 'target': target}
        authorized = allow_recovery_semantics_migration or self.cfg.allow_recovery_semantics_migration
        if semantic_changes and not authorized:
            raise ValueError(f'Recovery semantics mismatch: {semantic_changes}. '
                             'For a separate controlled experiment, explicitly select the target modes and '
                             '--allow-recovery-semantics-migration; ordinary resume must retain the saved modes.')
        if 'recovery_height_reward' in semantic_changes and getattr(self.replay, 'size', 0):
            raise ValueError('Height-reward migration requires a fresh empty target replay; '
                             'existing transitions may contain reward labels for the previous objective')
        for key in ('task','model_hidden','policy_hidden','critic_hidden','discriminator_hidden','action_limit',
                    'task_weight','style_weight','amp_reward_coef','direct_style_grad','critic_bootstrap','survival_grad',
                    'real_critic_coef','gamma','gae_lambda'):
            if payload['config'].get(key)!=asdict(self.cfg).get(key):
                raise ValueError(f'Resume mismatch for {key}; restore its config or start a separate warmstart experiment')
        if smooth_migrated and 'recovery_height_reward' in semantic_changes:
            raise ValueError('Migrate smooth rewards and recovery height in separate experiments')
        # All semantic checks precede mutation of parameters, Adam, RNG or clocks.
        self.adapter.recovery_counter_semantics = self.cfg.recovery_counter_semantics
        for name in ('actor','critic','world_model','discriminator','amp_norm','opt_actor','opt_critic','opt_model','opt_disc'):
            getattr(self,name).load_state_dict(payload[name])
        self._set_amp_norm_settings(payload.get('amp_normalizer_settings',{}))
        if self.cfg.amp_reward_mapping == 'instantamp_centered_tanh_v1':
            if amp_migrated:
                # First ordinary successful D step initializes from real replay.
                # No extra calibration, sampling, optimizer step or WM reset.
                self.amp_reward_stats = CenteredAmpReward(self.cfg.amp_reward_coef,
                    self.cfg.amp_score_scale, self.cfg.amp_score_ema_decay).to(self.device)
            else:
                self.amp_reward_stats.load_state_dict(payload['amp_reward_state'])
        self.iteration,self.sim_steps=payload['iteration'],payload['sim_steps']
        self.imagination_steps=payload.get('imagination_steps',0)
        self.wall_offset=payload['wall_seconds'];self._collect_accumulator=payload.get('collect_accumulator',0.)
        saved_cost = payload.get('training_cost')
        self._cost_history_complete = bool(isinstance(saved_cost, dict)
            and saved_cost.get('history_complete') is True
            and saved_cost.get('wall_time_scope') == WALL_TIME_SCOPE
            and saved_cost.get('physical_transitions') == self.sim_steps
            and saved_cost.get('wall_seconds') == self.wall_offset
            and saved_cost.get('gpu_count') == (1 if self.device.type == 'cuda' else 0))
        if isinstance(saved_cost, dict) and saved_cost.get('trajectory_id'):
            self._cost_trajectory_id = saved_cost['trajectory_id']
        self.env.common_step_counter=payload.get('env_common_step_counter',0)
        self.best_eval_score=payload.get('best_eval_score')
        self.critic_calibration_updates=payload.get('critic_calibration_updates',0)
        discard_replay = (smooth_migrated or smooth_retuned or 'recovery_height_reward' in semantic_changes) and 'replay' in payload
        if 'replay' in payload and not discard_replay:
            self.replay.load_state_dict(payload['replay']);self.replay.mark_boundary()
        self.recovery_semantics_history = list(payload.get('recovery_semantics_history', []))
        self.actor_initialization_history = list(payload.get('actor_initialization_history', []))
        self.amp_reward_migration_history = list(payload.get('amp_reward_migration_history', []))
        self.smooth_reward_migration_history = list(payload.get('smooth_reward_migration_history', []))
        self.smooth_reward_retune_history = list(payload.get('smooth_reward_retune_history', []))
        self.imagined_foot_slip_migration_history = list(payload.get('imagined_foot_slip_migration_history', []))
        if foot_slip_migrated:
            record = {
                'kind': 'explicit_imagined_foot_slip_migration', 'source_checkpoint': str(Path(path).resolve()),
                'source_sha256': self._checkpoint_sha256(path), 'source_iteration': self.iteration,
                'target_protocol': imagined_foot_slip_protocol(self.cfg),
                'physical_reward_unchanged': True, 'saved_replay_preserved': 'replay' in payload,
                'networks_optimizers_and_ema_preserved': True, 'extra_updates_or_sampling': 0,
            }
            self.imagined_foot_slip_migration_history.append(record)
            self._cost_trajectory_id = str(uuid.uuid4())
            self.best_eval_score = None
            if hasattr(self, 'log_dir'):
                (self.log_dir / 'imagined_foot_slip_migration.json').write_text(json.dumps(record, indent=2) + '\n')
            print('[WMAC] Explicit imagined foot slip fork; real reward/replay and all model weights preserved.', flush=True)
        if smooth_migrated:
            record = {
                'kind':'explicit_smooth_reward_migration', 'source_checkpoint':str(Path(path).resolve()),
                'source_sha256':self._checkpoint_sha256(path), 'source_iteration':self.iteration,
                'target_protocol':smooth_reward_protocol(self.cfg),
                'networks_and_optimizers_preserved':True, 'saved_replay_discarded':discard_replay,
                'replay_note':'Old task reward labels do not contain the new smooth penalties.',
                'extra_warmup_iterations':0, 'extra_networks_or_optimizer_steps':0,
                'critic_transition':'Existing critic fits new imagined lambda returns; no added supervision.',
            }
            self.smooth_reward_migration_history.append(record)
            self._cost_trajectory_id = str(uuid.uuid4())
            self.best_eval_score = None
            if hasattr(self,'log_dir'):
                (self.log_dir/'smooth_reward_migration.json').write_text(json.dumps(record,indent=2)+'\n')
            print('[WMAC] Explicit smooth reward migration; network/optimizer states preserved, '
                  f'saved replay discarded={discard_replay}; no extra calibration.',flush=True)
        if smooth_retuned:
            source_protocol = checkpoint_smooth_reward_protocol(payload)
            target_protocol = smooth_reward_protocol(self.cfg)
            record = {
                'kind':'explicit_smooth_reward_retune', 'source_checkpoint':str(Path(path).resolve()),
                'source_sha256':self._checkpoint_sha256(path), 'source_iteration':self.iteration,
                'source_protocol':source_protocol, 'target_protocol':target_protocol,
                'coefficient_changes':{
                    key:{'source':source_protocol[key], 'target':target_protocol[key]}
                    for key in SMOOTH_REWARD_COEFFICIENT_KEYS
                    if source_protocol[key] != target_protocol[key]},
                'authorization':'explicit_allow_smooth_reward_retune_switch',
                'networks_and_optimizers_preserved':True, 'amp_ema_preserved':True,
                'motor_constants_and_formula_preserved':True,
                'saved_replay_discarded':discard_replay,
                'replay_note':'Saved task reward labels use the previous smooth coefficients.',
                'best_and_validation_reset':True,
                'extra_warmup_iterations':0, 'extra_networks_or_optimizer_steps':0,
                'critic_transition':'Existing critic fits new imagined lambda returns; no added supervision.',
            }
            self.smooth_reward_retune_history.append(record)
            self._cost_trajectory_id = str(uuid.uuid4())
            self.best_eval_score = None
            self.last_model_loss = float('inf')
            self.validation = {key:float('inf') for key in (
                'validation_model_error','validation_contact_error','validation_reward_gap',
                'validation_walk_error','validation_recovery_error')}
            if hasattr(self,'log_dir'):
                (self.log_dir/'smooth_reward_retune.json').write_text(json.dumps(record,indent=2)+'\n')
            print('[WMAC] Explicit smooth coefficient retune; network/optimizer/AMP EMA states preserved, '
                  f'saved replay discarded={discard_replay}; no extra calibration.',flush=True)
        if amp_migrated:
            record = {
                'kind': 'explicit_amp_reward_migration', 'source_checkpoint': str(Path(path).resolve()),
                'source_sha256': self._checkpoint_sha256(path), 'source_iteration': self.iteration,
                'source_protocol': payload.get('amp_reward_protocol', {'amp_reward_mapping':'clipped_lsgan_v1'}),
                'target_protocol': amp_reward_protocol(self.cfg),
                'networks_and_optimizers_preserved': True, 'replay_preserved_if_saved': not discard_replay,
                'extra_updates_or_sampling': 0,
                'ema_initialization': 'first_successful_existing_discriminator_update',
                'critic_transition': 'existing critic fits new imagined lambda returns; no added supervision',
            }
            self.amp_reward_migration_history.append(record)
            self._cost_trajectory_id = str(uuid.uuid4())
            self.best_eval_score = None
            if hasattr(self, 'log_dir'):
                (self.log_dir/'amp_reward_migration.json').write_text(json.dumps(record, indent=2)+'\n')
            print('[WMAC] Explicit centered AMP migration; networks/optimizers/counters preserved; '
                  'EMA initializes in the next existing D update.', flush=True)
        if semantic_changes:
            self.recovery_semantics_history.append({
                'kind': 'explicit_resume_migration', 'source_checkpoint': str(Path(path).resolve()),
                'source_sha256': self._checkpoint_sha256(path), 'source_iteration': self.iteration,
                'changes': semantic_changes, 'authorization': 'explicit_migration_switch',
                'networks_and_optimizers_preserved': True,
                'critic_calibration_updates_preserved': True,
                'saved_replay_discarded': discard_replay,
                'replay_note': 'Old physical reward labels cannot be reused after a height-objective change.' if discard_replay else None,
            })
            self._cost_trajectory_id = str(uuid.uuid4())
            self.best_eval_score = None
            if hasattr(self, 'log_dir'):
                (self.log_dir/'recovery_semantics_migration.json').write_text(
                    json.dumps(self.recovery_semantics_history[-1], indent=2)+'\n')
            print(f'[WMAC] Explicit recovery migration: {semantic_changes}; network/Adam states preserved, '
                  f'saved replay discarded={discard_replay}', flush=True)
        # Fresh physical data is needed after recreating a simulator; old model
        # validation values cannot certify the new run's starting distribution.
        torch.set_rng_state(payload['torch_rng'].cpu());np.random.set_state(payload['numpy_rng']);random.setstate(payload['python_rng'])
        if 'cuda_rng' in payload and torch.cuda.is_available():torch.cuda.set_rng_state_all([x.cpu() for x in payload['cuda_rng']])
        print(f'[WMAC] Resumed iteration {self.iteration}; fresh held-out validation required',flush=True)

    def export_policy(self,path):
        policy=copy.deepcopy(self.actor).cpu().eval();policy.precision='float32'
        if hasattr(policy, 'export_dense'):
            # Trace all branches, keeping TopK/Gather live for arbitrary inputs.
            # The exported graph computes every expert; it is not sparse runtime.
            policy.export_dense = True
        width=self.actor_obs_dim
        torch.onnx.export(policy,torch.zeros(1,width),str(path),input_names=['obs'],output_names=['actions'],
            dynamic_axes={'obs':{0:'batch'},'actions':{0:'batch'}},opset_version=18,dynamo=False)
        metadata=get_base_metadata(self.env,self.log_dir.name)
        metadata.update(algorithm='amp-wmac-v2',observation_history_order='time',observation_history_length=4,observation_width=width)
        metadata.update(training_protocol=json.dumps(config_protocol(self.cfg), sort_keys=True))
        metadata.update(amp_reward_protocol=json.dumps(amp_reward_protocol(self.cfg), sort_keys=True))
        metadata.update(smooth_reward_protocol=json.dumps(smooth_reward_protocol(self.cfg), sort_keys=True))
        metadata.update(physical_action_clip=str(self.cfg.action_limit))
        metadata.update(recovery_counter_semantics=self.cfg.recovery_counter_semantics,
                        recovery_height_reward=self.cfg.recovery_height_reward)
        metadata.update(actor_type=self.cfg.actor_type,
                        actor_architecture=json.dumps(actor_architecture(self.cfg), sort_keys=True))
        if self.cfg.actor_type == 'moe':
            metadata.update(moe_runtime='dense_softmax_hiking' if self.cfg.moe_variant == 'hiking' else 'dense_masked_topk_export',
                            moe_variant=self.cfg.moe_variant,
                            moe_num_experts=str(self.cfg.moe_num_experts),moe_top_k=str(self.cfg.moe_top_k))
        attach_metadata_to_onnx(str(path),metadata)

    def evaluate_checkpoint(self):
        checkpoint=self.log_dir/f'model_{self.iteration}.pt'
        if not checkpoint.exists():self.save(checkpoint)
        output=self.log_dir/'evaluations'/f'iteration_{self.iteration}'
        command=[sys.executable,str(Path(__file__).resolve().parents[2]/'scripts/evaluate_wmac.py'),
                 '--checkpoint',str(checkpoint),'--mode','both','--seed','2026','--num-envs',str(self.cfg.evaluation_envs),
                 '--steps',str(self.cfg.evaluation_steps),'--device',str(self.device),
                 '--recovery-height-reward',self.cfg.recovery_height_reward,'--output',str(output)]
        if self.device.type=='cuda':torch.cuda.empty_cache()
        with (self.log_dir/f'evaluation_{self.iteration}.log').open('w') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
        result=json.loads((output/'metrics.json').read_text())['results']
        recovery,walking=result['recovery'],result['walk']
        score=(recovery['recovery_success_fraction'],recovery['final_upright_fraction'],walking['first_episode_survival_fraction'],
               -walking['velocity_xy_error_mean'],-(recovery['first_stable_recovery_seconds_mean'] or self.cfg.evaluation_steps*self.env.step_dt))
        if self.best_eval_score is None or tuple(score)>tuple(self.best_eval_score):
            self.best_eval_score=tuple(score)
            self.save(checkpoint,export=False)
            shutil.copy2(checkpoint,self.log_dir/'best_model.pt')
            (self.log_dir/'best_selection.json').write_text(json.dumps({'iteration':self.iteration,'score':score,
                'physical_evaluation':str(output/'metrics.json')},indent=2)+'\n')
