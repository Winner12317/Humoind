"""EngineAI PM01 constants (23 actuated joints, J00..J22; the head yaw is welded).

The robot model is ``xmls/pm01.xml``, generated from EngineAI's official MJCF by
``scripts/pm01/build_pm01_mjcf.py``.  Gains follow the recipe this repository uses
for the G1 (10 Hz natural frequency on the reflected rotor inertia, damping ratio
2), which is also how EngineAI's own tracking configuration for PM01 is written;
the per-module numbers are in ``pm01_actuators.py``.
"""

from pathlib import Path

import mujoco

from src import SRC_PATH
from src.assets.robots.engineai_pm01.pm01_actuators import (
  ARMATURE_Q25,
  ARMATURE_Q90,
  EFFORT_LIMIT_Q25,
  EFFORT_LIMIT_Q90,
  Pm01ActuatorCfg_Q25,
  Pm01ActuatorCfg_Q90,
)
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

PM01_XML: Path = SRC_PATH / "assets" / "robots" / "engineai_pm01" / "xmls" / "pm01.xml"
assert PM01_XML.exists()


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, PM01_XML.parent / "assets", meshdir)
  return assets


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(PM01_XML))
  spec.assets = get_assets(spec.meshdir)
  return spec


##
# Names. The policy, the motion files and the deployment side all use this order.
##

PM01_JOINT_NAMES: tuple[str, ...] = (
  "J00_HIP_PITCH_L",
  "J01_HIP_ROLL_L",
  "J02_HIP_YAW_L",
  "J03_KNEE_PITCH_L",
  "J04_ANKLE_PITCH_L",
  "J05_ANKLE_ROLL_L",
  "J06_HIP_PITCH_R",
  "J07_HIP_ROLL_R",
  "J08_HIP_YAW_R",
  "J09_KNEE_PITCH_R",
  "J10_ANKLE_PITCH_R",
  "J11_ANKLE_ROLL_R",
  "J12_WAIST_YAW",
  "J13_SHOULDER_PITCH_L",
  "J14_SHOULDER_ROLL_L",
  "J15_SHOULDER_YAW_L",
  "J16_ELBOW_PITCH_L",
  "J17_ELBOW_YAW_L",
  "J18_SHOULDER_PITCH_R",
  "J19_SHOULDER_ROLL_R",
  "J20_SHOULDER_YAW_R",
  "J21_ELBOW_PITCH_R",
  "J22_ELBOW_YAW_R",
)
PM01_WELDED_JOINT_NAMES: tuple[str, ...] = ("J23_HEAD_YAW",)

PM01_ROOT_BODY = "LINK_BASE"
PM01_TORSO_BODY = "LINK_TORSO_YAW"
PM01_FOOT_BODIES: tuple[str, str] = ("LINK_ANKLE_ROLL_L", "LINK_ANKLE_ROLL_R")
PM01_FOOT_SITES: tuple[str, str] = ("left_foot", "right_foot")
PM01_FOOT_GEOMS: tuple[str, ...] = (
  "left_foot_sole_collision",
  "left_foot_toe_collision",
  "right_foot_sole_collision",
  "right_foot_toe_collision",
)
PM01_IMU_SITE = "imu"

# The 13 bodies the AMP discriminator sees, in model order; one-to-one with the G1
# list (pelvis, hip roll, knee, ankle roll, shoulder roll, elbow, hand end).
PM01_AMP_BODY_NAMES: tuple[str, ...] = (
  "LINK_BASE",
  "LINK_HIP_ROLL_L",
  "LINK_KNEE_PITCH_L",
  "LINK_ANKLE_ROLL_L",
  "LINK_HIP_ROLL_R",
  "LINK_KNEE_PITCH_R",
  "LINK_ANKLE_ROLL_R",
  "LINK_SHOULDER_ROLL_L",
  "LINK_ELBOW_PITCH_L",
  "LINK_ELBOW_YAW_L",
  "LINK_SHOULDER_ROLL_R",
  "LINK_ELBOW_PITCH_R",
  "LINK_ELBOW_YAW_R",
)

##
# Heights (m), all for LINK_BASE above a flat floor.
##

# FK of the default pose below with the soles on the floor gives 0.8193.
PM01_STANDING_HEIGHT = 0.82
# G1 terminates below 0.50 m; its walk/run references never drop under 0.664 m.
# The retargeted PM01 references run 3.5 % higher (lowest walk/run frame 0.689 m).
PM01_MIN_BASE_HEIGHT = 0.52
# Height below which a state counts as "on the ground / getting up" in the WMAC
# replay and expert stratification (0.55 m for the G1), scaled the same way.
PM01_RECOVERY_HEIGHT = 0.57

##
# Actuator config.
##

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10 Hz
DAMPING_RATIO = 2.0

STIFFNESS_Q90 = ARMATURE_Q90 * NATURAL_FREQ**2  # 178.8 N m/rad
STIFFNESS_Q25 = ARMATURE_Q25 * NATURAL_FREQ**2  # 26.45 N m/rad
DAMPING_Q90 = 2.0 * DAMPING_RATIO * ARMATURE_Q90 * NATURAL_FREQ  # 11.39 N m s/rad
DAMPING_Q25 = 2.0 * DAMPING_RATIO * ARMATURE_Q25 * NATURAL_FREQ  # 1.684 N m s/rad
# EngineAI's tracking configuration gives the ankles this damping instead of
# DAMPING_Q25; kept as published.
DAMPING_ANKLE = 0.5

PM01_ACTUATOR_Q90 = Pm01ActuatorCfg_Q90(
  target_names_expr=(".*_HIP_PITCH_.*", ".*_HIP_ROLL_.*", ".*_KNEE_PITCH_.*"),
  stiffness=STIFFNESS_Q90,
  damping=DAMPING_Q90,
  effort_limit=EFFORT_LIMIT_Q90,
  armature=ARMATURE_Q90,
)
PM01_ACTUATOR_Q25 = Pm01ActuatorCfg_Q25(
  target_names_expr=(
    ".*_HIP_YAW_.*",
    ".*_WAIST_YAW",
    ".*_SHOULDER_PITCH_.*",
    ".*_SHOULDER_ROLL_.*",
    ".*_SHOULDER_YAW_.*",
    ".*_ELBOW_PITCH_.*",
    ".*_ELBOW_YAW_.*",
  ),
  stiffness=STIFFNESS_Q25,
  damping=DAMPING_Q25,
  effort_limit=EFFORT_LIMIT_Q25,
  armature=ARMATURE_Q25,
)
PM01_ACTUATOR_ANKLE = Pm01ActuatorCfg_Q25(
  target_names_expr=(".*_ANKLE_PITCH_.*", ".*_ANKLE_ROLL_.*"),
  stiffness=STIFFNESS_Q25,
  damping=DAMPING_ANKLE,
  effort_limit=EFFORT_LIMIT_Q25,
  armature=ARMATURE_Q25,
)

##
# Keyframe config.
##

# EngineAI's default pose for PM01 (the offsets its own policies are trained and
# deployed with): legs almost straight, arms slightly out and bent.
HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, PM01_STANDING_HEIGHT),
  joint_pos={
    ".*_HIP_PITCH_.*": -0.06,
    ".*_KNEE_PITCH_.*": 0.12,
    ".*_ANKLE_PITCH_.*": -0.06,
    ".*_ELBOW_PITCH_.*": -0.25,
    "J14_SHOULDER_ROLL_L": 0.15,
    "J19_SHOULDER_ROLL_R": -0.15,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

_FOOT_GEOMS = r"^(left|right)_foot_(sole|toe)_collision$"

# All of EngineAI's collision primitives, self-collisions included (pm01.xml excludes
# the hand/thigh pairs that overlap by construction). Self-collisions get condim=1,
# foot collisions condim=3.
FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  condim={_FOOT_GEOMS: 3, ".*_collision": 1},
  priority={_FOOT_GEOMS: 1},
  friction={_FOOT_GEOMS: (0.6,)},
)

FULL_COLLISION_WITHOUT_SELF = CollisionCfg(
  geom_names_expr=(".*_collision",),
  contype=0,
  conaffinity=1,
  condim={_FOOT_GEOMS: 3, ".*_collision": 1},
  priority={_FOOT_GEOMS: 1},
  friction={_FOOT_GEOMS: (0.6,)},
)

FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(_FOOT_GEOMS,),
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
)

##
# Final config.
##

PM01_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    PM01_ACTUATOR_Q90,
    PM01_ACTUATOR_Q25,
    PM01_ACTUATOR_ANKLE,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_pm01_robot_cfg() -> EntityCfg:
  """Get a fresh PM01 robot configuration instance."""
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=PM01_ARTICULATION,
  )


# Same rule as the G1: a unit action moves the target by a quarter of the position
# error that saturates the motor. 0.229 rad on the Q90 joints, 0.491 rad on the rest.
PM01_ACTION_SCALE: dict[str, float] = {}
for a in PM01_ARTICULATION.actuators:
  assert isinstance(a, BuiltinPositionActuatorCfg)
  e = a.effort_limit
  s = a.stiffness
  names = a.target_names_expr
  assert e is not None
  for n in names:
    PM01_ACTION_SCALE[n] = 0.25 * e / s


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_pm01_robot_cfg())

  viewer.launch(robot.spec.compile())
