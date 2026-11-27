"""Hiking-style speed-controlled commands towards valid flat terrain patches."""
from __future__ import annotations

from dataclasses import dataclass
import math
import torch
from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommand,UniformVelocityCommandCfg


def heading_error_to_target(state,target_pos_w,source_wrap=False):
    """Differentiable heading error shared by navigation and turn completion."""
    delta=target_pos_w[...,:2]-state[...,:2]
    distance=delta.norm(dim=-1)
    q=state[...,3:7];q=q/q.norm(dim=-1,keepdim=True).clamp_min(1e-8)
    w,x,y,z=q.unbind(-1)
    sy,cy=2*(w*z+x*y),1-2*(y*y+z*z)
    singular=sy.square()+cy.square()<1e-8
    yaw=torch.atan2(torch.where(singular,torch.zeros_like(sy),sy),
                    torch.where(singular,torch.ones_like(cy),cy))
    safe_x=torch.where(distance<1e-6,torch.ones_like(distance)*1e-6,delta[...,0])
    target_yaw=torch.atan2(delta[...,1],safe_x)
    difference=target_yaw-yaw
    if source_wrap:
        # IsaacLab wrap_to_pi preserves +pi for positive odd multiples.
        # atan2(sin(pi),cos(pi)) can instead produce -pi in float32.
        wrapped=(difference+torch.pi)%(2*torch.pi)
        angle=torch.where((wrapped==0)&(difference>0),torch.pi,wrapped-torch.pi)
    else:
        angle=torch.atan2(torch.sin(difference),torch.cos(difference))
    return distance,angle


def command_from_state(state,target_pos_w,desired_speed,k_v=2.,k_yaw=2.,yaw_limit=1.,standing_mask=None):
    """Forward-only vx, zero vy, and signed yaw velocity.

    A zero desired speed can denote an in-place turn, so standing has its own
    explicit mask. The same mask and target are used during imagination;
    only the resulting three velocity commands are policy inputs.
    """
    distance,angle=heading_error_to_target(state,target_pos_w)
    # Speed control and turn-to-target logic from the position-command idea;
    # both yaw directions remain represented, unlike a one-sided clip pool.
    speed=torch.minimum(desired_speed.clamp_min(0.),k_v*distance)
    forward=speed*torch.cos(angle).clamp_min(0.)
    yaw_rate=torch.where(distance<1e-6,torch.zeros_like(angle),(k_yaw*angle).clamp(-yaw_limit,yaw_limit))
    command=torch.stack((forward,torch.zeros_like(forward),yaw_rate),-1)
    if standing_mask is not None:
        command=torch.where(standing_mask[...,None],torch.zeros_like(command),command)
    return command


def hiking_command_from_state(state,target_pos_w,desired_speed,standing_mask=None,
                              random_yaw_rate=None,random_velocity_mask=None,
                              k_v=2.,k_yaw=2.,yaw_limit=1.,target_distance_threshold=.4):
    """Hiking PoseVelocityCommand, specialized to its forward-only configuration.

    ``state`` is expressed at Hiking's torso root, including its heading. The
    independent body-x clamp is significant: clipping speed before projecting
    onto the heading gives a different command. The source random-velocity
    terrain overrides standing after target navigation, and supplies in-place
    turns with zero translation.
    """
    distance,angle=heading_error_to_target(state,target_pos_w,source_wrap=True)
    forward=torch.minimum((k_v*distance*torch.cos(angle)).clamp_min(0.),desired_speed.clamp_min(0.))
    yaw_rate=(k_yaw*angle).clamp(-yaw_limit,yaw_limit)
    command=torch.stack((forward,torch.zeros_like(forward),yaw_rate),-1)
    command=command*(distance>target_distance_threshold)[...,None]
    if standing_mask is not None:
        command=torch.where(standing_mask[...,None],torch.zeros_like(command),command)
    if random_velocity_mask is not None:
        if random_yaw_rate is None:
            raise ValueError("The Hiking random-velocity mode requires its sampled yaw rate")
        random_command=torch.stack((torch.zeros_like(forward),torch.zeros_like(forward),random_yaw_rate),-1)
        command=torch.where(random_velocity_mask[...,None],random_command,command)
    return command


class FlatPatchVelocityCommand(UniformVelocityCommand):
    def __init__(self,cfg,env):
        if not math.isfinite(cfg.in_place_turn_fraction) or not 0<=cfg.in_place_turn_fraction<=1:
            raise ValueError("in_place_turn_fraction must lie in [0,1]")
        if not math.isfinite(cfg.rel_standing_envs) or not 0<=cfg.rel_standing_envs<=1:
            raise ValueError("rel_standing_envs must lie in [0,1]")
        if not (0<cfg.turn_tolerance<cfg.minimum_turn_angle<=cfg.maximum_turn_angle<=math.pi):
            raise ValueError("Turn tolerances and sampled angles must lie in (0, pi]")
        super().__init__(cfg,env)
        self.target_pos_w=torch.zeros(self.num_envs,3,device=self.device)
        self.desired_speed=torch.zeros(self.num_envs,device=self.device)
        self.goals_reached=torch.zeros(self.num_envs,device=self.device)
        self.goal_is_traversal=torch.zeros(self.num_envs,device=self.device,dtype=torch.bool)
        self.is_in_place_turn=torch.zeros_like(self.goal_is_traversal)
        self.turn_goals_reached=torch.zeros(self.num_envs,device=self.device)
        # The base reset clears metrics after curriculum.compute has consumed
        # the final episode counts. Resampling a target does not clear it.
        self.metrics["navigation_goals_reached"]=self.goals_reached
        self.metrics["turn_goals_reached"]=self.turn_goals_reached
        if cfg.task_protocol=="hiking_full_v1":
            import numpy as np
            generator=env.scene.terrain.cfg.terrain_generator
            names=list(generator.sub_terrains)
            proportions=np.array([c.proportion for c in generator.sub_terrains.values()])
            cumulative=np.cumsum(proportions/proportions.sum())
            source_sampling = getattr(getattr(env, "cfg", None), "_hiking_training_version", 1) >= 3
            column_family=np.searchsorted(cumulative,np.arange(generator.num_cols)/generator.num_cols+.001,
                                          side="right" if source_sampling else "left")
            family_id=torch.as_tensor(column_family,device=self.device)[env.scene.terrain.terrain_types]
            self._hiking_speed_low=torch.full((self.num_envs,),.45,device=self.device)
            self._hiking_speed_high=torch.full_like(self._hiking_speed_low,.8)
            if "perlin_rough" in names:self._hiking_speed_high[family_id==names.index("perlin_rough")]=1.
            self.random_velocity_mask=(family_id==names.index("perlin_rough_stand") if "perlin_rough_stand" in names
                                       else torch.zeros(self.num_envs,dtype=torch.bool,device=self.device))
            self._hiking_speed_low[self.random_velocity_mask]=0.
            self._hiking_speed_high[self.random_velocity_mask]=0.
            self.random_yaw_rate=torch.zeros_like(self.desired_speed)
            self.metrics["tracking_exp_vel_xy"]=torch.zeros_like(self.desired_speed)
            self.metrics["tracking_exp_vel_yaw"]=torch.zeros_like(self.desired_speed)

    def _root_pose(self):
        # EntityData has pose/velocity accessors, not root_link_state_w.
        # Generalized coordinates also avoid stale xpos during reset events
        # (command resampling precedes the following sim.forward()).
        if self.cfg.task_protocol=="hiking_full_v1":
            p,q,_,_=self._hiking_kinematics()
            return torch.cat((p[:,self._hiking_adapter.anchor_id],q[:,self._hiking_adapter.anchor_id]),-1)
        from .mdp import root_pose_from_asset
        return root_pose_from_asset(self.robot)

    def _hiking_kinematics(self):
        if not hasattr(self,"_hiking_adapter"):
            from src.wmac.state import StateAdapter
            self._hiking_adapter=StateAdapter(self._env)
        return self._hiking_adapter.kinematics(self._hiking_adapter.state())

    def _update_metrics(self):
        if self.cfg.task_protocol!="hiking_full_v1":
            return super()._update_metrics()
        from src.wmac.state import quat_apply,quat_inv
        _,q,v,w=self._hiking_kinematics()
        anchor=self._hiking_adapter.anchor_id
        com=quat_apply(q[:,anchor],self._hiking_adapter.default_torso_com)
        torso_com_v=v[:,anchor]+torch.cross(w[:,anchor],com,dim=-1)
        vb=quat_apply(quat_inv(q[:,anchor]),torso_com_v)
        wb=quat_apply(quat_inv(q[:,anchor]),w[:,anchor])
        difference=self.vel_command_b[:,:2]-vb[:,:2]
        yaw_difference=self.vel_command_b[:,2]-wb[:,2]
        max_command_step=self.cfg.resampling_time_range[1]/self._env.step_dt
        self.metrics["error_vel_xy"].add_(difference.norm(dim=-1)/max_command_step)
        self.metrics["error_vel_yaw"].add_(yaw_difference.abs()/max_command_step)
        maximum_steps=self._env.max_episode_length
        self.metrics["tracking_exp_vel_xy"].add_(torch.exp(-difference.square().sum(-1)/.5**2)/maximum_steps)
        self.metrics["tracking_exp_vel_yaw"].add_(torch.exp(-yaw_difference.square()/.5**2)/maximum_steps)

    def _resample_command(self,env_ids):
        terrain=self._env.scene.terrain
        patches=terrain.flat_patches["target"][terrain.terrain_levels[env_ids],terrain.terrain_types[env_ids]]
        if self.cfg.task_protocol=="hiking_full_v1":
            chosen=torch.randint(patches.shape[1],(len(env_ids),),device=self.device)
            self.target_pos_w[env_ids]=patches[torch.arange(len(env_ids),device=self.device),chosen]
            low,high=self._hiking_speed_low[env_ids],self._hiking_speed_high[env_ids]
            self.desired_speed[env_ids]=low+torch.rand(len(env_ids),device=self.device)*(high-low)
            source_sampling = getattr(getattr(self._env, "cfg", None), "_hiking_training_version", 1) >= 3
            if source_sampling:
                # PoseVelocityCommand draws all three speed limits even when
                # two ranges are constant, then independently chooses standing.
                r = torch.empty_like(low)
                r.uniform_(0., 1.)
                r.uniform_(0., 1.)
                self.is_standing_env[env_ids] = r.uniform_(0., 1.) <= self.cfg.rel_standing_envs
                selected = torch.zeros_like(self.random_velocity_mask)
                selected[env_ids] = True
                turn_ids = (selected & self.random_velocity_mask).nonzero(as_tuple=True)[0]
                if turn_ids.numel():
                    # The two constant zero linear commands also consume draws.
                    r = torch.empty(len(turn_ids), device=self.device)
                    r.uniform_(0., 1.)
                    r.uniform_(0., 1.)
                    lo, hi = self.cfg.ranges.ang_vel_z
                    yaw = lo + r.uniform_(0., 1.) * (hi - lo)
                    self.random_yaw_rate[turn_ids] = yaw * (yaw.abs() > .5)
            else:
                self.is_standing_env[env_ids]=torch.rand(len(env_ids),device=self.device)<=self.cfg.rel_standing_envs
                yaw=torch.empty(len(env_ids),device=self.device).uniform_(*self.cfg.ranges.ang_vel_z)
                self.random_yaw_rate[env_ids]=yaw*(yaw.abs()>.5)
            self.is_in_place_turn[env_ids]=self.random_velocity_mask[env_ids]&(self.random_yaw_rate[env_ids]!=0)
            self.goal_is_traversal[env_ids]=~self.random_velocity_mask[env_ids]&~self.is_standing_env[env_ids]
            self.is_heading_env[env_ids]=False
            if source_sampling:
                # The source publishes the new command in _update_command.
                return
            self.vel_command_b[env_ids]=hiking_command_from_state(
                self._root_pose()[env_ids],self.target_pos_w[env_ids],self.desired_speed[env_ids],
                self.is_standing_env[env_ids],self.random_yaw_rate[env_ids],self.random_velocity_mask[env_ids])
            return
        # All candidates are in the same terrain tile and certified flat by
        # the generator. Prefer targets 1m away, preserving a traversal signal.
        root=self._root_pose()[env_ids,:3]
        distance=(patches[...,:2]-root[:,None,:2]).norm(dim=-1)
        eligible=(distance>=self.cfg.minimum_target_distance)&(distance<=self.cfg.maximum_target_distance)
        scores=torch.rand_like(distance)
        scores=torch.where(eligible,scores,torch.full_like(scores,-1.))
        chosen=scores.argmax(-1)
        fallback=(distance-(self.cfg.minimum_target_distance+self.cfg.maximum_target_distance)/2).abs().argmin(-1)
        chosen=torch.where(eligible.any(-1),chosen,fallback)
        self.target_pos_w[env_ids]=patches[torch.arange(len(env_ids),device=self.device),chosen]
        self.goal_is_traversal[env_ids]=distance[torch.arange(len(env_ids),device=self.device),chosen]>=self.cfg.minimum_target_distance
        self.desired_speed[env_ids]=torch.empty(len(env_ids),device=self.device).uniform_(*self.cfg.ranges.lin_vel_x)
        # In-place turns are sampled explicitly, rather than relying on an
        # occasional navigation target happening to be behind the robot.
        turn=(torch.rand(len(env_ids),device=self.device)<self.cfg.in_place_turn_fraction
              if self.cfg.in_place_turn_fraction else torch.zeros(len(env_ids),device=self.device,dtype=torch.bool))
        self.is_in_place_turn[env_ids]=turn
        if turn.any():
            ids=env_ids[turn]
            pose=self._root_pose()[ids]
            q=pose[:,3:7];q=q/q.norm(dim=-1,keepdim=True).clamp_min(1e-8)
            w,x,y,z=q.unbind(-1)
            yaw=torch.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
            angle=torch.empty(len(ids),device=self.device).uniform_(self.cfg.minimum_turn_angle,self.cfg.maximum_turn_angle)
            sign=torch.where(torch.rand(len(ids),device=self.device)<.5,-torch.ones_like(angle),torch.ones_like(angle))
            target_yaw=yaw+sign*angle
            # This is a heading landmark only: it is never a traversal goal.
            radius=max(2.,self.cfg.minimum_target_distance)
            self.target_pos_w[ids]=pose[:,:3]
            self.target_pos_w[ids,:2]+=radius*torch.stack((target_yaw.cos(),target_yaw.sin()),-1)
            self.desired_speed[ids]=0.
            self.goal_is_traversal[ids]=False
        # Hiking samples a standing mode independently of the moving command.
        # The existing turn fraction therefore applies to non-standing draws:
        # .05 standing and .2 turning yield .05/.19/.76 stand/turn/walk draws.
        # Do not consume another random draw for historical no-standing runs.
        standing=(torch.rand(len(env_ids),device=self.device)<self.cfg.rel_standing_envs
                  if self.cfg.rel_standing_envs else torch.zeros(len(env_ids),device=self.device,dtype=torch.bool))
        self.is_standing_env[env_ids]=standing
        if standing.any():
            ids=env_ids[standing]
            self.desired_speed[ids]=0.
            self.is_in_place_turn[ids]=False
            self.goal_is_traversal[ids]=False
        self.is_heading_env[env_ids]=False
        if self.cfg.rel_standing_envs:
            # CommandTerm.reset only resamples; it does not call update. Publish
            # the command now so reset observations and replay mode agree.
            # Historical checkpoints preserve their original reset timing.
            self.vel_command_b[env_ids]=command_from_state(
                self._root_pose()[env_ids],self.target_pos_w[env_ids],self.desired_speed[env_ids],
                self.cfg.k_v,self.cfg.k_yaw,self.cfg.ranges.ang_vel_z[1],
                standing_mask=self.is_standing_env[env_ids])

    def _update_command(self):
        root=self._root_pose()
        if self.cfg.task_protocol=="hiking_full_v1":
            self.vel_command_b[:]=hiking_command_from_state(root,self.target_pos_w,self.desired_speed,
                self.is_standing_env,self.random_yaw_rate,self.random_velocity_mask)
            return
        self.vel_command_b[:]=command_from_state(root,self.target_pos_w,self.desired_speed,
                                                self.cfg.k_v,self.cfg.k_yaw,self.cfg.ranges.ang_vel_z[1],
                                                standing_mask=self.is_standing_env)
        distance,heading_error=heading_error_to_target(root,self.target_pos_w)
        reached=(distance<self.cfg.goal_tolerance)&~self.is_in_place_turn&~self.is_standing_env
        turn_reached=self.is_in_place_turn&(heading_error.abs()<self.cfg.turn_tolerance)&~self.is_standing_env
        self.goals_reached.add_((reached&self.goal_is_traversal).to(self.goals_reached.dtype))
        self.turn_goals_reached.add_(turn_reached.to(self.turn_goals_reached.dtype))
        finished=reached|turn_reached
        if finished.any():
            ids=finished.nonzero(as_tuple=False).flatten()
            if self.cfg.rel_standing_envs:
                # A newly selected standing command receives its full sampled
                # duration, including when the preceding goal finished early.
                self._resample(ids)
            else:
                self._resample_command(ids)


@dataclass(kw_only=True)
class FlatPatchVelocityCommandCfg(UniformVelocityCommandCfg):
    task_protocol: str="legacy_walk_turn_v1"
    k_v: float=2.
    k_yaw: float=2.
    minimum_target_distance: float=1.
    maximum_target_distance: float=5.5
    goal_tolerance: float=.25
    in_place_turn_fraction: float=0.
    minimum_turn_angle: float=math.pi/4
    maximum_turn_angle: float=math.pi
    turn_tolerance: float=.12

    def build(self,env):return FlatPatchVelocityCommand(self,env)


def hiking_tracking_terrain_curriculum(env,env_ids,command_name="twist",
                                       lin_vel_threshold=(.3,.6),ang_vel_threshold=(0.,0.)):
    """Original Hiking tracking-score curriculum, evaluated before metric reset."""
    if env_ids is None:env_ids=torch.arange(env.num_envs,device=env.device)
    command=env.command_manager.get_term(command_name)
    xy=command.metrics["tracking_exp_vel_xy"][env_ids]
    yaw=command.metrics["tracking_exp_vel_yaw"][env_ids]
    up=(xy>lin_vel_threshold[1])&(yaw>ang_vel_threshold[1])
    down=(xy<lin_vel_threshold[0])&~up
    # mjlab calls curricula during the explicit initial reset as well. Up to
    # training version 3 that call is ignored. IsaacLab's first env.reset() does
    # run the curriculum, with tracking scores still zero, so Hiking starts every
    # robot one level below its sampled one; version 4 reproduces that once.
    active=env.episode_length_buf[env_ids]>0
    if (getattr(getattr(env,"cfg",None),"_hiking_training_version",1)>=4
            and not getattr(env,"_hiking_initial_curriculum_applied",False)
            and int(env.common_step_counter)==0):
        env._hiking_initial_curriculum_applied=True
        active=torch.ones_like(active)
    env.scene.terrain.update_env_origins(env_ids,up&active,down&active)
    return env.scene.terrain.terrain_levels.float().mean()


def navigation_terrain_curriculum(env,env_ids,command_name="twist"):
    """Advance actual terrain rows after valid traversal with a stable finish.

    Called before the reset events and command metrics reset. Initial/reset-
    before-first-step calls cannot demote. A timeout with no reached goal is
    judged by actual travel; failed recovery cannot promote a terrain level.
    """
    if env_ids is None:env_ids=torch.arange(env.num_envs,device=env.device)
    # ManagerBasedRlEnv.reset() runs curriculum once before its first step;
    # reset_terminated does not exist yet. That call is initialization, not a
    # failed episode, and must neither move terrain rows nor consume goals.
    length=env.episode_length_buf[env_ids]
    failed_buffer=getattr(env,"reset_terminated",None)
    if not len(env_ids) or failed_buffer is None or not bool((length>0).any()):
        return {"terrain_level":env.scene.terrain.terrain_levels[env_ids].float().mean()
                                if len(env_ids) else torch.zeros((),device=env.device),
                "promoted_fraction":torch.zeros((),device=env.device),
                "demoted_fraction":torch.zeros((),device=env.device)}
    command=env.command_manager.get_term(command_name)
    robot=env.scene["robot"]
    from .mdp import get_terrain_oracle,root_pose_from_asset
    pose=root_pose_from_asset(robot)[env_ids]
    position,quat=pose[:,:3],pose[:,3:7]
    height=position[:,2]-get_terrain_oracle(env).height_at(position[:,:2])
    upright=1-2*(quat[:,1].square()+quat[:,2].square())
    stable=(height>.65)&(upright>.8)
    goals=command.goals_reached[env_ids]
    failed=failed_buffer[env_ids]
    distance=(position[:,:2]-env.scene.env_origins[env_ids,:2]).norm(dim=-1)
    expected=command.desired_speed[env_ids]*length*env.step_dt
    up=(length>0)&(goals>=1)&stable&~failed
    down=(length>0)&(goals<1)&(failed|(distance<expected*.25))&~up
    env.scene.terrain.update_env_origins(env_ids,up,down)
    return {"terrain_level":env.scene.terrain.terrain_levels[env_ids].float().mean(),
            "promoted_fraction":up.float().mean(),"demoted_fraction":down.float().mean()}
