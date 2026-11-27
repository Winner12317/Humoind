"""Fixed-shape differentiable imagination kernels for the latent runner.

The simulator/replay/validation and host decisions stay outside compilation.
Tensor kernels and complete captured updates follow official SuperMimic;
the latent observation and local terrain/AMP objectives remain task adapters.
"""
from __future__ import annotations

import torch
from torch import nn

from src.tasks.amp_perceptive.commands import command_from_state


class LatentImaginedStep(nn.Module):
    def __init__(self, runner):
        super().__init__()
        self.actor = runner.actor
        self.world_model = runner.world_model
        object.__setattr__(self, "adapter", runner.adapter)
        self.action_limit = runner.cfg.action_limit
        self.full_hiking = getattr(runner.cfg,"locomotion_objective",None)=="hiking_full_v1"
        self.share_hiking_kinematics = self.full_hiking and getattr(runner.cfg,"hiking_training_version",1)>=2

    def forward(self, state, context, history, latent, target, desired_speed, noise, standing_mask=None,
                random_yaw_rate=None,random_velocity_mask=None):
        result,_,_=self._step_with_geometry(state,context,history,latent,target,desired_speed,noise,
            standing_mask,random_yaw_rate,random_velocity_mask)
        return result

    def _step_with_geometry(self,state,context,history,latent,target,desired_speed,noise,
                            standing_mask=None,random_yaw_rate=None,random_velocity_mask=None):
        """Return geometry owned by this differentiable step, never cached.

        Full Hiking v2 consumers use the same successor FK and command for
        termination, observations, rewards and AMP. No tensor survives on the
        module/adapter or crosses an optimizer update/detach boundary.
        """
        command_fn=self.adapter.navigation_command if self.full_hiking else command_from_state
        options={"standing_mask":standing_mask}
        if self.full_hiking:
            options.update(random_yaw_rate=random_yaw_rate,random_velocity_mask=random_velocity_mask)
        command = command_fn(state, target, desired_speed, **options)
        if self.full_hiking:
            from .hiking_observations import pack_hiking_proprio_history
            policy_history=pack_hiking_proprio_history(history)
        else:policy_history=history.flatten(1)
        action = self.actor.forward_latent(policy_history, latent)
        if self.action_limit is not None:
            action = action.clamp(-self.action_limit, self.action_limit)
        state, contact, latent = self.world_model.step_with_latent(state, action, context, latent)
        kinematics=self.adapter.analyze_state(state) if self.share_hiking_kinematics else None
        if kinematics is not None:
            context,terminated=self.adapter.advance_context(state,context,action=action,contact=contact,
                kinematics=kinematics)
            next_command=command_fn(state,target,desired_speed,**options,kinematics=kinematics)
            observation=self.adapter.actor_frame(state,action,next_command,context,kinematics=kinematics)+noise
        else:
            context, terminated = (self.adapter.advance_context(state,context,action=action,contact=contact)
                if self.full_hiking else self.adapter.advance_context(state,context))
            next_command = command_fn(state, target, desired_speed, **options)
            observation = self.adapter.actor_frame(state, action, next_command, context) + noise
        history = torch.cat((history[:, 1:], observation[:, None]), 1)
        return (state,context,history,latent,action,command,contact,terminated),next_command,kinematics


class LatentTrajectoryTerms(nn.Module):
    """One FK pass over the trajectory, shared by task reward and AMP inputs."""

    def __init__(self, runner):
        super().__init__()
        object.__setattr__(self, "adapter", runner.adapter)

    def forward(self, states, actions, previous_actions, commands, contexts, terminated, contacts):
        fk = self.adapter.analyze_state(states)
        amp = self.adapter.amp_features(states, kinematics=fk)
        task = self.adapter.task_reward(states[:-1], states[1:], actions, previous_actions,
            commands, contexts[1:], terminated, contacts,
            kinematics=tuple(value[1:] for value in fk))
        return task, amp


class CriticLoss(nn.Module):
    def __init__(self, critic):
        super().__init__()
        self.critic = critic

    def forward(self, features, target):
        return (self.critic(features)-target).square().mean()


def fit_critic_fast(runner, features, target):
    """Preserve the critic objective while reading diagnostics once per batch."""
    from .latent_execution import safe_optimizer_step
    x, y = features.flatten(0, 1).detach(), target.flatten().detach()
    records = []
    sample_count=(len(x)//runner.cfg.critic_minibatches*runner.cfg.critic_minibatches
                  if runner.uses_reference_schedule else len(x))
    shared_order=(torch.randperm(sample_count,device=x.device)
        if runner.cfg.execution_optimizer=="muon_adam" else None)
    for _ in range(runner.cfg.critic_epochs):
        order = shared_order if shared_order is not None else torch.randperm(len(x), device=x.device)
        for indices in order.tensor_split(runner.cfg.critic_minibatches):
            if not len(indices):
                continue
            execution=getattr(runner,"_latent_critic_execution",None)
            if execution is not None:
                features,targets=x[indices],y[indices]
                schema=(tuple(features.shape),tuple(targets.shape),x.dtype,y.dtype,x.device)
                previous=getattr(runner,"_latent_critic_batch_schema",None)
                if previous is not None and schema!=previous:
                    execution.invalidate()
                runner._latent_critic_batch_schema=schema
                outputs,norm,accepted=execution.step({"features":features,"target":targets})
                loss=outputs["loss"]
            else:
                runner.opt_critic.zero_grad(set_to_none=True)
                loss = runner._latent_critic_loss_fn(x[indices], y[indices])
                loss.backward()
                norm = nn.utils.clip_grad_norm_(runner.critic.parameters(), runner.cfg.max_grad_norm, foreach=True)
                accepted = safe_optimizer_step(loss,norm,runner.opt_critic,
                    guard=not runner.uses_reference_schedule)
            records.append(torch.stack((loss.detach(), loss.new_zeros(()), norm.detach(), accepted.float())))
    result = torch.stack(records).mean(0).detach().cpu().tolist()
    return dict(zip(("critic_loss", "real_critic_loss", "critic_grad_norm", "critic_updates_applied_fraction"), result))
