"""Terrain-relative geometry and matching real/imagined walking objectives."""
from __future__ import annotations

import torch

from src.amp_features import build_amp_features, G1_FOOT_REFERENCE_OFFSET
from src.wmac.state import StateAdapter, quat_apply, quat_inv


class PerceptiveStateAdapter(StateAdapter):
    def __init__(self, env, recovery_counter_semantics="hard_forward_v2",motion_scope="joint_recovery_v1",
                 amp_feature_schema=None):
        super().__init__(env,recovery_counter_semantics=recovery_counter_semantics)
        if amp_feature_schema is not None:
            from src.amp_features import amp_feature_dim
            self.amp_feature_schema = amp_feature_schema
            self.amp_dim = amp_feature_dim(amp_feature_schema)
        if self.amp_feature_schema == "hiking_v1":
            model = env.sim.mj_model
            torso_id = int(self.robot.indexing.body_ids[self.anchor_id])
            self._hiking_torso_com_offset = torch.as_tensor(
                model.body_ipos[torso_id].copy(), dtype=torch.float32, device=self.device)
        self.motion_scope=motion_scope
        self.full_hiking=bool(getattr(env.cfg,"_hiking_full_protocol",False))
        self.hiking_training_version=getattr(env.cfg,"_hiking_training_version",1)
        self.material_geom_ids=None
        if self.full_hiking and self.hiking_training_version >= 2:
            from .hiking_materials import hiking_material_geom_ids
            self.material_geom_ids=hiking_material_geom_ids(
                self.robot, env.sim.mj_model, self.device, self.hiking_training_version)
        self.context_dim=(100+(len(self.material_geom_ids) if self.material_geom_ids is not None else 0)
                          if self.full_hiking else 35)
        # Optional robot-dynamics randomization sample, appended after the materials.
        self.robot_dynamics_randomization=getattr(env.cfg,"_robot_dynamics_randomization","none")
        self.robot_dynamics_columns=0
        if self.full_hiking and self.robot_dynamics_randomization!="none":
            from .robot_dynamics import ROBOT_DYNAMICS_CONTEXT_WIDTH
            self.robot_dynamics_columns=ROBOT_DYNAMICS_CONTEXT_WIDTH
            self.context_dim+=self.robot_dynamics_columns
        self.contact_dim=13 if self.full_hiking else 2
        self.camera_capture_steps=(5 if getattr(env.cfg,"_perceptive_sensor_protocol",None)
            =="g1_rpl_depth_v1" else 0)
        if motion_scope=="forward_walk_turn_v1":self.max_delay_steps=0
        from src.tasks.amp_perceptive.terrain import TerrainHeightField
        self.terrain = getattr(env,"_perceptive_terrain_oracle",None)
        if self.terrain is None:
            self.terrain = TerrainHeightField.from_env(env)
            env._perceptive_terrain_oracle = self.terrain
        if self.full_hiking:
            from src.tasks.amp_perceptive.hiking_rewards import initialize_hiking_reward_adapter
            initialize_hiking_reward_adapter(self)

    def context(self):
        base=super().context()
        if not getattr(self,"full_hiking",False):
            return base
        from .hiking_actuators import read_hiking_delay_per_joint,read_hiking_delay_initialized
        from src.tasks.amp_perceptive.hiking_rewards import hiking_airtime_state
        initialized=read_hiking_delay_initialized(self.robot).to(base.dtype)
        period=getattr(self,"camera_capture_steps",0)
        clock=torch.full_like(initialized[...,None],self.env.common_step_counter%period if period else 0)
        base=torch.cat((base[...,:33],initialized[...,None],clock),-1)
        parts=(base,self.env.scene.env_origins,self.env.action_manager.action,
               read_hiking_delay_per_joint(self.robot),hiking_airtime_state(self.env))
        ids=getattr(self,"material_geom_ids",None)
        if ids is not None:
            friction=self.env.sim.model.geom_friction
            if friction.ndim==2:friction=friction.unsqueeze(0)
            applied=friction.expand(self.env.num_envs,-1,-1).index_select(1,ids)[...,0]
            parts=(*parts,applied)
        if getattr(self,"robot_dynamics_columns",0):
            sample=getattr(self.env,"_robot_dynamics_context",None)
            if sample is None:
                # Event not installed (e.g. PLAY): the unmodified training robot.
                from .robot_dynamics import nominal_robot_dynamics_context
                sample=nominal_robot_dynamics_context(self.robot_dynamics_randomization,
                                                      self.env.num_envs,base.device)
            parts=(*parts,sample.to(base.dtype))
        return torch.cat(parts,-1)

    def actor_frame(self,state,previous_action,command,context,noise=False,kinematics=None):
        if not getattr(self,"full_hiking",False):
            return super().actor_frame(state,previous_action,command,context,noise=noise)
        from .hiking_observations import hiking_actor_frame
        return hiking_actor_frame(self,state,previous_action,command,context,noise=noise,kinematics=kinematics)

    def observation_noise_scale(self):
        from .hiking_observations import hiking_proprio_noise_amplitudes
        return hiking_proprio_noise_amplitudes(device=self.device)

    def navigation_command(self,state,target,desired_speed,standing_mask=None,
                           random_yaw_rate=None,random_velocity_mask=None,kinematics=None):
        from src.tasks.amp_perceptive.commands import hiking_command_from_state
        position,quaternion,_,_=(self.kinematics(state) if kinematics is None else kinematics)
        torso_pose=torch.cat((position[...,self.anchor_id,:],quaternion[...,self.anchor_id,:]),-1)
        return hiking_command_from_state(torso_pose,target,desired_speed,standing_mask=standing_mask,
            random_yaw_rate=random_yaw_rate,random_velocity_mask=random_velocity_mask)

    def ground_relative_state(self,state):
        height = state[...,2:3]-self.terrain.height_at(state[...,:2]).unsqueeze(-1)
        return torch.cat((state[...,:2],height,state[...,3:]),-1)

    def relative_height(self,state):
        return state[...,2]-self.terrain.height_at(state[...,:2])

    def amp_features(self,state,kinematics=None):
        p,q,v,w = self.kinematics(state) if kinematics is None else kinematics
        if self.amp_feature_schema == "hiking_v1":
            from .hiking_amp import frame
            torso = self.anchor_id
            # IsaacLab root_lin_vel_b refers to torso COM, while reference
            # poses are differentiated at the torso origin.
            offset = quat_apply(q[...,torso,:],self._hiking_torso_com_offset)
            velocity = v[...,torso,:]+torch.cross(w[...,torso,:],offset,dim=-1)
            return frame(state[...,13:42],state[...,42:71],q[...,torso,:],velocity,w[...,torso,:],
                         self.default_joint_pos,self.default_joint_vel)
        features=build_amp_features(p,q,v,w,body_ids=self.amp_ids,anchor_id=self.anchor_id,
            pelvis_id=self.pelvis_id,foot_ids=self.foot_ids,schema="grounded_v2")
        pelvis_ground=self.terrain.height_at(p[...,self.pelvis_id,:2])
        offset=p.new_tensor((G1_FOOT_REFERENCE_OFFSET,G1_FOOT_REFERENCE_OFFSET))
        sole=p[...,self.foot_ids,:]+quat_apply(q[...,self.foot_ids,:],offset)
        foot_ground=self.terrain.height_at(sole[...,:2])
        # Keep the203D layout; the separate perceptive checkpoint metadata
        # identifies the changed per-point terrain grounding convention.
        return torch.cat((features[...,:198],features[...,198:199]-pelvis_ground[...,None],
                          features[...,199:201]-foot_ground,features[...,201:]),-1)

    def critic_features(self,state,previous_action,command,context):
        if getattr(self, "hiking_training_version", 1) >= 3:
            from src.wmac.gencoord import gencoord_features
            from .hiking_context import normalize_hiking_sim_context
            # The true previous action is already carried in context[38:67].
            return torch.cat((gencoord_features(state), command, normalize_hiking_sim_context(context,getattr(self,"robot_dynamics_columns",0)),
                              self.terrain.local_patch(state)), -1)
        physical=super().critic_features(self.ground_relative_state(state),previous_action,command,context)
        return torch.cat((physical,self.terrain.local_patch(state)),-1)

    def advance_context(self,next_state,context,action=None,contact=None,kinematics=None):
        if getattr(self,"full_hiking",False):
            from .hiking_context import advance_hiking_context
            if action is None or contact is None:
                raise ValueError("Full Hiking imagination requires action and thirteen auxiliary predictions")
            p,q,_,_=(self.kinematics(next_state) if kinematics is None else kinematics)
            # The auxiliary predictor carries the same four substep timers
            # as collection; do not replace them with a control-step clock.
            next_context=advance_hiking_context(context,action,contact,
                camera_capture_steps=getattr(self,"camera_capture_steps",0))
            force=contact[...,5]
            gravity=quat_apply(quat_inv(q[...,self.anchor_id,:]),
                next_state.new_tensor((0.,0.,-1.)))
            height=p[...,self.anchor_id,2]-context[...,37].clamp(max=0.)
            if getattr(self, "hiking_training_version", 1) >= 3:
                # Apply SuperMimic's sigmoid to the worst normalized physical
                # failure criterion. Each individual Hiking threshold scores 1.
                minimum_height = .5
                height_violation = 1. + (minimum_height - height) / minimum_height
                tilt_violation = (1. + gravity[..., 2]) / (1. - next_state.new_tensor(1.).cos())
                normalized = torch.maximum(torch.maximum(height_violation, tilt_violation), force)
                return next_context, torch.sigmoid((normalized - 1.) / .05)
            soft_height=torch.sigmoid((.5-height)/.025)
            soft_tilt=torch.sigmoid((gravity[...,2]+next_state.new_tensor(1.).cos())/.025)
            soft_contact=torch.sigmoid((force-1.)/.1)
            soft=1-(1-soft_height)*(1-soft_tilt)*(1-soft_contact)
            if getattr(self,"hiking_training_version",1)<2:
                # Explicit legacy checkpoints retain their original objective.
                from src.tasks.amp_perceptive.hiking_rewards import hiking_termination_terms
                death=hiking_termination_terms(p[...,self.anchor_id,:],gravity,context[...,35:38],force)
                hard=(death["base_contact"]|death["bad_orientation"]|death["root_height"]).to(next_state.dtype)
                return next_context,hard+(soft-soft.detach())
            # SuperMimic imagination uses the same soft probability in the
            # forward return and its gradients. Real Hiking failures retain
            # their hard thresholds in the physical environment.
            return next_context,soft
        if getattr(self,"motion_scope","joint_recovery_v1")=="forward_walk_turn_v1":
            relative=self.ground_relative_state(next_state)
            # The two historical recovery slots are padding in this task.
            clean=torch.cat((context[...,:33],torch.zeros_like(context[...,33:])),dim=-1)
            _,soft=super().advance_context(relative,clean)
            gravity=quat_apply(quat_inv(relative[...,3:7]),relative.new_tensor((0.,0.,-1.)))
            hard=((relative[...,2]<self.height_threshold)|
                  (torch.acos(-gravity[...,2]).abs()>self.tilt_angle)).to(soft.dtype)
            return clean,hard+(soft-soft.detach())
        return super().advance_context(self.ground_relative_state(next_state),context)

    def task_reward(self,state,next_state,action,previous_action,command,next_context,terminated,
                    contact_cost,kinematics=None):
        fk=self.kinematics(next_state) if kinematics is None else kinematics
        if getattr(self,"full_hiking",False):
            from src.tasks.amp_perceptive.hiking_rewards import hiking_reward_inputs,hiking_weighted_reward
            inputs=hiking_reward_inputs(self,state,next_state,action,previous_action,command,
                next_context,terminated,kinematics=fk)
            return hiking_weighted_reward(inputs,contact_cost,reward_scale=self.reward_scale)
        value=super().task_reward(self.ground_relative_state(state),self.ground_relative_state(next_state),
            action,previous_action,command,next_context,terminated,contact_cost,kinematics=fk)
        if "stand_still" in self.reward_cfg:
            from src.tasks.amp_perceptive.mdp import hiking_stand_still_term
            cfg=self.reward_cfg["stand_still"]
            value=value+float(cfg.weight)*self.reward_scale*hiking_stand_still_term(
                next_state[...,13:42],self.default_joint_pos,command,
                threshold=float(cfg.params.get("threshold",.15)),offset=float(cfg.params.get("offset",4.)))
        if "foot_edge_safety" in self.reward_cfg:
            from .safety import foot_volume_edge_cost
            cfg=self.reward_cfg["foot_edge_safety"]
            value=value+float(cfg.weight)*self.reward_scale*foot_volume_edge_cost(
                fk,self.foot_ids,self.terrain,margin=float(cfg.params.get("margin",.04)))
        return value
