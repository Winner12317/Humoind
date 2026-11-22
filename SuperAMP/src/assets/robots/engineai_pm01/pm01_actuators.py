"""EngineAI PM01 joint actuators.

PM01 uses two joint modules.  EngineAI publishes their peak torque, maximum speed
and reflected rotor inertia, but no torque-speed curve and no friction
identification, so the model below is the PD + limit + friction actuator that the
G1 uses (``UnitreeActuator``) with PM01 numbers:

* Peak torque is available at any speed up to the published maximum speed and no
  motor torque is produced above it (``X1 = X2 = velocity limit``).
* Joint friction is the pair EngineAI's own MuJoCo model declares for every joint
  (``<joint damping="0.6" frictionloss="0.8"/>`` in serial_pm01_edu.xml): 0.8 N m
  Coulomb friction and 0.6 N m s/rad viscous friction.  They live here, in the
  actuator, exactly like the Unitree ``Fs`` / ``Fd`` values, so pm01.xml carries no
  passive joint damping of its own.

Sources:
  engineai_rl_lab  source/engineai_rl_lab/engineai_rl_lab/tasks/tracking/robots/pm01.py
                   (commit c53f7ec): armature, effort and velocity limits
  engineai_robotics_native_sdk  assets/resource/robot/pm01_edu/xml/serial_pm01_edu.xml
                   (commit 6065bb2): joint friction
"""

from __future__ import annotations

from dataclasses import dataclass

from src.assets.robots.unitree_g1.unitree_actuators import UnitreeActuatorCfg

PM01_JOINT_FRICTION_STATIC = 0.8  # N m
PM01_JOINT_FRICTION_VISCOUS = 0.6  # N m s/rad

# Reflected rotor inertia (kg m^2), peak torque (N m) and maximum speed (rad/s).
ARMATURE_Q90, EFFORT_LIMIT_Q90, VELOCITY_LIMIT_Q90 = 0.0453, 164.0, 26.3
ARMATURE_Q25, EFFORT_LIMIT_Q25, VELOCITY_LIMIT_Q25 = 0.0067, 52.0, 35.2


@dataclass(kw_only=True)
class Pm01ActuatorCfg(UnitreeActuatorCfg):
  """Shared PM01 friction; the torque-speed fields are set per module below."""

  Fs: float = PM01_JOINT_FRICTION_STATIC
  Fd: float = PM01_JOINT_FRICTION_VISCOUS


@dataclass(kw_only=True)
class Pm01ActuatorCfg_Q90(Pm01ActuatorCfg):
  """High-torque module: hip pitch, hip roll, knee."""

  X1: float = VELOCITY_LIMIT_Q90
  X2: float = VELOCITY_LIMIT_Q90
  Y1: float = EFFORT_LIMIT_Q90
  Y2: float = EFFORT_LIMIT_Q90
  armature: float = ARMATURE_Q90


@dataclass(kw_only=True)
class Pm01ActuatorCfg_Q25(Pm01ActuatorCfg):
  """Low-torque module: hip yaw, ankles, waist, shoulders, elbows."""

  X1: float = VELOCITY_LIMIT_Q25
  X2: float = VELOCITY_LIMIT_Q25
  Y1: float = EFFORT_LIMIT_Q25
  Y2: float = EFFORT_LIMIT_Q25
  armature: float = ARMATURE_Q25
