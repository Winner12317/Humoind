"""Hiking's five delayed ideal-PD actuator groups for the local G1 MJCF.

Parameters follow instinctlab/assets/unitree_g1.py at de6b92f. Delay is sampled
once at reset, independently per environment and actuator group, in physical
substeps (0..2). This is not mjlab's default per-substep stochastic delay.
MuJoCo does not expose PhysX's joint velocity solver limit; its source values
are retained as soft limits for rewards, without inventing a torque-speed law.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import re

import torch

from mjlab.actuator.actuator import Actuator,ActuatorCfg,ActuatorCmd
from mjlab.utils.spec import create_motor_actuator

HIKING_MAX_DELAY_SUBSTEPS=2
HIKING_ACTUATOR_GROUPS=("legs","feet","waist","waist_yaw","arms")
_NATURAL_FREQ=10*2.*3.1415926535
_ARMATURE={"5020":.003609725,"7520_14":.010177520,"7520_22":.025101925,"4010":.00425}
_GROUP_EXPRESSIONS={
    "legs":(".*_hip_yaw_joint",".*_hip_roll_joint",".*_hip_pitch_joint",".*_knee_joint"),
    "feet":(".*_ankle_pitch_joint",".*_ankle_roll_joint"),
    "waist":("waist_roll_joint","waist_pitch_joint"),
    "waist_yaw":("waist_yaw_joint",),
    "arms":(".*_shoulder_pitch_joint",".*_shoulder_roll_joint",".*_shoulder_yaw_joint",
            ".*_elbow_joint",".*_wrist_roll_joint",".*_wrist_pitch_joint",".*_wrist_yaw_joint"),
}


def _joint_parameters(name):
    name=name.rsplit("/",1)[-1]
    group=next((group for group,patterns in _GROUP_EXPRESSIONS.items()
                if any(re.fullmatch(pattern,name) for pattern in patterns)),None)
    if group is None:raise ValueError(f"Not a Hiking G1 joint: {name}")
    factor=1.
    if group in ("feet","waist"):
        motor,effort,velocity,factor="5020",50.,37.,2.
    elif group=="waist_yaw" or "hip_yaw" in name or "hip_pitch" in name:
        motor,effort,velocity="7520_14",88.,32.
    elif group=="legs":
        motor,effort,velocity="7520_22",139.,20.
    elif "wrist_pitch" in name or "wrist_yaw" in name:
        motor,effort,velocity="4010",5.,22.
    else:
        motor,effort,velocity="5020",25.,37.
    armature=factor*_ARMATURE[motor]
    return dict(kp=armature*_NATURAL_FREQ**2,kd=4.*armature*_NATURAL_FREQ,
                effort_limit=effort,velocity_limit=velocity,armature=armature,
                delay_group=HIKING_ACTUATOR_GROUPS.index(group))


def get_hiking_joint_parameters(joint_names,*,device=None,dtype=torch.float32):
    """Named-order source parameters; tensors have shape [number of joints]."""
    rows=[_joint_parameters(name) for name in joint_names]
    return {key:torch.tensor([row[key] for row in rows],device=device,
            dtype=torch.long if key=="delay_group" else dtype) for key in rows[0]}


class ResetFixedDelay:
    """Fixed-lag history with IsaacLab's first-sample backfill semantics."""
    def __init__(self,num_envs,width,device,max_lag=HIKING_MAX_DELAY_SUBSTEPS):
        if max_lag not in range(HIKING_MAX_DELAY_SUBSTEPS+1):
            raise ValueError("Actuator lag must be 0..2 physics substeps")
        self.max_lag=max_lag
        self.history=torch.zeros(num_envs,3,width,device=device)
        self.lags=torch.zeros(num_envs,dtype=torch.long,device=device)
        self.length=torch.zeros(num_envs,dtype=torch.long,device=device)

    def reset(self,env_ids=None):
        ids=slice(None) if env_ids is None else env_ids
        self.history[ids]=0
        self.length[ids]=0
        self.lags[ids]=torch.randint(0,self.max_lag+1,self.lags[ids].shape,device=self.lags.device)

    def append_and_get(self,value):
        self.history[:,1:].copy_(self.history[:,:-1].clone())
        self.history[:,0].copy_(value)
        effective=torch.minimum(self.lags,self.length)
        self.length.add_(1).clamp_(max=2)
        return self.history.gather(1,effective[:,None,None].expand(-1,1,value.shape[-1])).squeeze(1)


@dataclass(kw_only=True)
class HikingDelayedPdActuatorCfg(ActuatorCfg):
    group: str
    # Largest reset-sampled lag in physics substeps. Hiking's delayed actuators
    # use 2; its shod task runs the undelayed ``beyondmimic_g1_29dof_actuators``.
    max_delay_substeps: int = HIKING_MAX_DELAY_SUBSTEPS

    def build(self,entity,target_ids,target_names):
        return HikingDelayedPdActuator(self,entity,target_ids,target_names)


class HikingDelayedPdActuator(Actuator):
    def edit_spec(self,spec,target_names):
        for name in target_names:
            p=_joint_parameters(name)
            self._mjs_actuators.append(create_motor_actuator(spec,name,
                effort_limit=p["effort_limit"],armature=p["armature"],frictionloss=0.,
                transmission_type=self.transmission_type))

    def initialize(self,mj_model,model,data,device):
        super().initialize(mj_model,model,data,device)
        parameters=get_hiking_joint_parameters(self._target_names,device=device)
        for attribute,key in (("stiffness","kp"),("damping","kd"),("force_limit","effort_limit"),
                              ("velocity_limit","velocity_limit")):
            setattr(self,attribute,parameters[key].expand(data.nworld,-1).clone())
        self.default_stiffness=self.stiffness.clone()
        self.default_damping=self.damping.clone()
        self.default_force_limit=self.force_limit.clone()
        self.delay=ResetFixedDelay(data.nworld,3*len(self._target_names),device,
                                   max_lag=self.cfg.max_delay_substeps)

    def compute(self,cmd:ActuatorCmd):
        targets=self.delay.append_and_get(torch.cat((cmd.position_target,cmd.velocity_target,cmd.effort_target),-1))
        position,velocity,effort=targets.chunk(3,-1)
        raw=self.stiffness*(position-cmd.pos)+self.damping*(velocity-cmd.vel)+effort
        return torch.clamp(raw,-self.force_limit,self.force_limit)

    def reset(self,env_ids=None):
        self.delay.reset(env_ids)


def configure_hiking_actuators(cfg,*,max_delay_substeps=HIKING_MAX_DELAY_SUBSTEPS):
    robot=cfg.scene.entities["robot"]
    # Legacy G1 presets share their articulation object: detach before edits.
    robot.articulation=deepcopy(robot.articulation)
    robot.articulation.actuators=tuple(HikingDelayedPdActuatorCfg(group=group,target_names_expr=patterns,
                                        max_delay_substeps=max_delay_substeps)
                                    for group,patterns in _GROUP_EXPRESSIONS.items())
    robot.articulation.soft_joint_pos_limit_factor=.9
    action=cfg.actions["joint_pos"]
    action.scale={pattern:.25*_joint_parameters(re.sub(r"^\.\*_","left_",pattern))["effort_limit"]/
                    _joint_parameters(re.sub(r"^\.\*_","left_",pattern))["kp"]
                  for patterns in _GROUP_EXPRESSIONS.values() for pattern in patterns}
    return cfg


def read_hiking_delay_per_joint(robot):
    """Actual reset-sampled lag, [env, joint], independent of history fill."""
    out=torch.zeros_like(robot.data.joint_pos)
    for actuator in robot.actuators:
        if isinstance(actuator,HikingDelayedPdActuator):
            out[:,actuator.target_ids]=actuator.delay.lags[:,None].to(out.dtype)
    return out


def read_hiking_delay_initialized(robot):
    """After one policy action all histories are filled (decimation 4 > lag 2)."""
    for actuator in robot.actuators:
        if isinstance(actuator,HikingDelayedPdActuator):return actuator.delay.length>0
    return torch.zeros(robot.data.joint_pos.shape[0],dtype=torch.bool,device=robot.data.joint_pos.device)


def read_hiking_actuator_parameters(robot):
    """Runtime per-environment tensors in robot.joint_names order."""
    out={name:torch.zeros_like(robot.data.joint_pos) for name in ("kp","kd","effort_limit","velocity_limit")}
    for actuator in robot.actuators:
        if not isinstance(actuator,HikingDelayedPdActuator):
            raise ValueError("Hiking parameters requested for a non-Hiking actuator")
        for key,attribute in (("kp","stiffness"),("kd","damping"),("effort_limit","force_limit"),
                              ("velocity_limit","velocity_limit")):
            out[key][:,actuator.target_ids]=getattr(actuator,attribute)
    return out
