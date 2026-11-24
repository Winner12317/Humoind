"""Capture actual pre-reset transitions without changing mjlab step semantics."""

from __future__ import annotations

import torch

from mjlab.envs import ManagerBasedRlEnv

from .state import StateAdapter


class CaptureEnv(ManagerBasedRlEnv):
    """An instance-local capture hook runs after rewards and before auto-reset.

    Attach ``adapter = StateAdapter(env)`` after construction. No simulator code
    is patched and no extra forward pass changes the baseline reward timing.
    """

    adapter: StateAdapter

    def __init__(self, *args, **kwargs):
        self.transition = None
        self._capture_enabled = False
        self._physical_end = None
        super().__init__(*args, **kwargs)
        original_compute = self.metrics_manager.compute

        def capture_after_metrics(*args, **kwargs):
            result = original_compute(*args, **kwargs)
            if self._capture_enabled and hasattr(self, "adapter"):
                with torch.no_grad():
                    costs = []
                    for name in ("foot_slip", "self_collisions"):
                        manager = self.reward_manager
                        if name in manager.active_terms:
                            i = manager.active_terms.index(name)
                            weight = float(manager.get_term_cfg(name).weight)
                            costs.append(manager._step_reward[:, i]/weight if weight else torch.zeros_like(self.reward_buf))
                        else:
                            costs.append(torch.zeros_like(self.reward_buf))
                    self._physical_end = {
                        "next_state": self.adapter.state().detach().clone(),
                        "next_context": self.adapter.context().detach().clone(),
                        "terminated": self.reset_terminated.detach().clone(),
                        "timeout": self.reset_time_outs.detach().clone(),
                        "reward": self.reward_buf.detach().clone(),
                        "contact_cost": torch.stack(costs, -1).detach().clone(),
                        "action": self.action_manager.action.detach().clone(),
                    }
            return result

        self.metrics_manager.compute = capture_after_metrics

    def step(self, action):
        if not hasattr(self, "adapter"):
            raise RuntimeError("Attach env.adapter = StateAdapter(env) before stepping CaptureEnv")
        with torch.no_grad():
            start = {
                "state": self.adapter.state().detach().clone(),
                "context": self.adapter.context().detach().clone(),
                "command": self.adapter.command().detach().clone(),
                "previous_action": self.action_manager.action.detach().clone(),
                "actor_history": self.observation_manager.compute()["actor"].detach().clone(),
            }
        if getattr(self, '_wmac_smooth_reward_terms', None) is not None:
            # Reuse captured physical state/history for the reward manager.
            self._wmac_smooth_start = start
        if getattr(self, '_wmac_recovery_tempo', None) is not None:
            self._wmac_step_start = start
        self._physical_end = None
        self._capture_enabled = True
        try:
            result = super().step(action)
        finally:
            self._capture_enabled = False
            self._wmac_smooth_start = None
            self._wmac_step_start = None
        if self._physical_end is None:
            raise RuntimeError("Pre-reset transition capture did not run")
        with torch.no_grad():
            end = self._physical_end
            done = end["terminated"] | end["timeout"]
            # Interval pushes are applied after our physical capture. They do
            # not invalidate this transition, but break the next rollout window.
            changed = (self.adapter.state()-end["next_state"]).abs().amax(-1) > 1e-6
            end["boundary"] = done | (changed & ~done)
            self.transition = {**start, **end}
        return result
