"""Capture pre-reset physical images separately from reset policy history."""
from __future__ import annotations

import torch
from src.wmac.env import CaptureEnv


class PerceptiveCaptureEnv(CaptureEnv):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self._fresh_depth_capture=bool(getattr(self.cfg,"_hiking_full_protocol",False))
        original=self.metrics_manager.compute
        def capture_image(*args,**kwargs):
            result=original(*args,**kwargs)
            if self._fresh_depth_capture and self._capture_enabled and self._physical_end is not None:
                from src.tasks.amp_perceptive.hiking_rewards import hiking_auxiliary_targets
                self._physical_end["contact_cost"]=hiking_auxiliary_targets(self).detach().clone()
            if self._capture_enabled and self._physical_end is not None and not self._fresh_depth_capture:
                from src.tasks.amp_perceptive.sensors import read_depth_frame
                self._physical_end["next_depth_frame"]=read_depth_frame(self).detach().clone()
            return result
        self.metrics_manager.compute=capture_image
        original_reset=self._reset_idx
        def capture_before_reset(env_ids):
            # mjlab has already tested that reset_env_ids is nonempty. Hooking
            # here avoids a second GPU->CPU termination check on every step.
            if self._fresh_depth_capture and self._capture_enabled and self._physical_end is not None:
                from src.tasks.amp_perceptive.sensors import read_depth_frame
                # mj_step's body transforms precede its final integration.
                # Refresh geometry before measuring the terminal camera pose.
                self.sim.forward()
                self.sim.sense()
                self._physical_end["terminal_depth_frame"]=read_depth_frame(self).detach().clone()
            if hasattr(self,"_rpl_camera_stream"):
                mask=torch.zeros(self.num_envs,dtype=torch.bool,device=self.device)
                mask[env_ids]=True
                self._rpl_camera_stream.reset(mask)
            if hasattr(self,"_hiking_depth_capture"):
                mask=torch.zeros(self.num_envs,dtype=torch.bool,device=self.device)
                mask[env_ids]=True
                self._hiking_depth_capture.reset(mask)
            return original_reset(env_ids)
        self._reset_idx=capture_before_reset

    def _actor_history_for_capture(self):
        if getattr(self,"_fresh_depth_capture",False):
            return self.obs_buf["actor"]
        return super()._actor_history_for_capture()

    def step(self,action):
        # Keep precisely the proprioceptive observation used to select action;
        # another manager.compute() could draw a different corruption sample.
        proprio=self.obs_buf["actor"].detach().clone()
        command=self.command_manager.get_term("twist")
        target=command.target_pos_w.detach().clone()
        speed=command.desired_speed.detach().clone()
        standing=command.is_standing_env.detach().clone()
        random_yaw=(command.random_yaw_rate.detach().clone() if hasattr(command,"random_yaw_rate") else None)
        random_mask=(command.random_velocity_mask.detach().clone() if hasattr(command,"random_velocity_mask") else None)
        result=super().step(action)
        if getattr(self,"_fresh_depth_capture",False):
            from src.tasks.amp_perceptive.sensors import read_depth_frame
            # The ordinary mjlab sense occurs after reset and forward. It is
            # both the physical successor for surviving envs and fresh input
            # for reset policy histories. Observation history is updated only
            # by mjlab, never by the terminal camera measurement above.
            depth=read_depth_frame(self).detach().clone()
            terminal=self.transition.pop("terminal_depth_frame",None)
            if terminal is not None:
                done=self.transition["terminated"]|self.transition["timeout"]
                depth=torch.where(done[:,None,None,None],terminal,depth)
            self.transition["next_depth_frame"]=depth
        target_changed=(command.target_pos_w-target).abs().amax(-1)>1e-6
        speed_changed=(command.desired_speed-speed).abs()>1e-6
        command_changed=target_changed|speed_changed|(command.is_standing_env!=standing)
        if random_yaw is not None:
            command_changed|=(command.random_yaw_rate-random_yaw).abs()>1e-6
            command_changed|=command.random_velocity_mask!=random_mask
            self.transition.update(random_yaw_rate=random_yaw,random_velocity_mask=random_mask)
        self.transition["boundary"]|=command_changed
        self.transition.update(actor_history=proprio,navigation_target=target,desired_speed=speed,
            command_standing=standing,command_boundary=command_changed,
            relative_height=self.adapter.relative_height(self.transition["state"]).detach())
        return result
