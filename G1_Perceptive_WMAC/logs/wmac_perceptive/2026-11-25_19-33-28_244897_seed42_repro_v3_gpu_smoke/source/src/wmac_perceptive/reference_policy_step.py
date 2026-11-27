"""Single-step Hiking reward and observations at the SuperMimic step boundary.

Upstream reference: HybridRobotics/supermimic at
abc124c1bd5976341034668af3d12c087f2f1f55:
https://github.com/HybridRobotics/supermimic/blob/abc124c1bd5976341034668af3d12c087f2f1f55/src/supermimic/envs/world_model_env.py#L114

Its ``step`` bundles dynamics, termination and reward before returning the next
observations. Hiking task/AMP terms and latent perception remain local task
adapters. Critic snapshots are detached by the caller when stored, as in the
upstream policy learner; the next observation returned here keeps its gradient.
"""

from __future__ import annotations

from .compiled_imagination import LatentImaginedStep


class HikingWorldModelStep(LatentImaginedStep):
    """Advance one imagined state and evaluate its reward/next observations."""

    def forward(self, state, context, history, latent, target, desired_speed, noise,
                standing_mask=None, random_yaw_rate=None, random_velocity_mask=None,
                *, previous_action):
        result,next_command,kinematics = self._step_with_geometry(state,context,history,latent,target,
            desired_speed,noise,standing_mask,random_yaw_rate,random_velocity_mask)
        next_state, next_context, _, _, action, command, contact, terminated = result

        # Official _wm_step scores the next state with the current command and
        # previous action; the reference clip/command advances after the reward.
        # https://github.com/HybridRobotics/supermimic/blob/abc124c1bd5976341034668af3d12c087f2f1f55/src/supermimic/envs/world_model_env.py#L179
        if kinematics is None:
            kinematics = self.adapter.analyze_state(next_state)
        task = self.adapter.task_reward(state, next_state, action, previous_action,
            command, next_context, terminated, contact, kinematics=kinematics)
        amp_next = self.adapter.amp_features(next_state, kinematics=kinematics)

        if not self.share_hiking_kinematics:
            next_command = self.adapter.navigation_command(next_state, target, desired_speed,
                standing_mask=standing_mask, random_yaw_rate=random_yaw_rate,
                random_velocity_mask=random_velocity_mask)
        critic_next = self.adapter.critic_features(next_state, action, next_command, next_context)
        return (*result, task, amp_next, critic_next)
