"""EngineAI PM01 AMP Locomotion environment configurations.

Same tasks as config/g1, with everything that names or measures the robot taken
from src/assets/robots/engineai_pm01/pm01_constants.py.
"""

import os

from src.assets.robots import (
  PM01_ACTION_SCALE,
  get_pm01_robot_cfg,
)
from src.assets.robots.engineai_pm01.pm01_constants import (
  PM01_AMP_BODY_NAMES,
  PM01_FOOT_BODIES,
  PM01_FOOT_GEOMS,
  PM01_FOOT_SITES,
  PM01_MIN_BASE_HEIGHT,
  PM01_ROOT_BODY,
  PM01_TORSO_BODY,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, RayCastSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from src.tasks.amp_loco.amp_env_cfg import make_amp_env_cfg
from src.tasks.amp_loco.mdp.observations import (
  aligned_critic_features,
  robot_grounded_amp_features,
)

def pm01_amp_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create EngineAI PM01 rough terrain velocity configuration."""
  cfg = make_amp_env_cfg()

  # Keep CCD high enough for stability but avoid Warp OOM from excessive EPA buffers.
  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 500
  cfg.sim.nconmax = 48

  cfg.scene.entities = {"robot": get_pm01_robot_cfg()}

  # Set raycast sensor frame to the PM01 base link.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      sensor.frame.name = PM01_ROOT_BODY

  site_names = PM01_FOOT_SITES
  geom_names = PM01_FOOT_GEOMS
  body_names = PM01_AMP_BODY_NAMES
  anchor_name = PM01_TORSO_BODY
  root_name = PM01_ROOT_BODY

  # The sole and toe boxes sit on LINK_FOOT_*, a child of the ankle-roll link.
  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=rf"^({'|'.join(PM01_FOOT_BODIES)})$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )

  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern=root_name, entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern=root_name, entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )

  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    self_collision_cfg,
  )

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = PM01_ACTION_SCALE

  cfg.viewer.body_name = anchor_name

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.viz.z_offset = 1.25  # PM01 is 1.38 m tall

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  cfg.events["base_com"].params["asset_cfg"].body_names = (anchor_name,)

  # Configure motion reset to sample from the entire motion with a delay.
  cfg.events["init_motion_loader"].params["delay_reset_env_ratio"] = 0.4
  cfg.events["init_motion_loader"].params["max_delay_steps"] = 250

  # Set motion data path for startup loader and reset.
  _motion_base = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "..", "assets", "motions", "pm01", "amp"
  )
  _motion_dir = os.path.abspath(os.path.join(_motion_base, "WalkandRun"))
  _recovery_dir = os.path.abspath(os.path.join(_motion_base, "Recovery"))

  cfg.events["init_motion_loader"].params["motion_dir"] = _motion_dir
  cfg.events["init_motion_loader"].params["recovery_dir"] = _recovery_dir
  cfg.events["reset_from_motion"].params["motion_dir"] = _motion_dir

  cfg.rewards["track_anchor_linear_velocity"].params["anchor_cfg"].body_names = (anchor_name,)
  cfg.rewards["track_anchor_angular_velocity"].params["anchor_cfg"].body_names = (anchor_name,)
  cfg.rewards["foot_slip"].params["asset_cfg"].site_names = site_names
  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-0.1,
    params={"sensor_name": self_collision_cfg.name, "force_threshold": 10.0},
  )
  cfg.rewards["body_ang_vel_xy_l2"].params["body_cfg"].body_names = (root_name,)

  # The base link of a walking PM01 is 3.5 % higher than the G1 pelvis.
  cfg.terminations["bad_base_height"].params["minimum_height"] = PM01_MIN_BASE_HEIGHT

  cfg.observations["critic"].terms["body_pos_b"].params["anchor_cfg"].body_names = (anchor_name,)
  cfg.observations["critic"].terms["body_pos_b"].params["body_cfg"].body_names = body_names

  cfg.observations["critic"].terms["body_ori_b"].params["anchor_cfg"].body_names = (anchor_name,)
  cfg.observations["critic"].terms["body_ori_b"].params["body_cfg"].body_names = body_names

  cfg.observations["amp"].terms["body_pos_b"].params["anchor_cfg"].body_names = (anchor_name,)
  cfg.observations["amp"].terms["body_pos_b"].params["body_cfg"].body_names = body_names

  cfg.observations["amp"].terms["body_ori_b"].params["anchor_cfg"].body_names = (anchor_name,)
  cfg.observations["amp"].terms["body_ori_b"].params["body_cfg"].body_names = body_names

  cfg.observations["amp"].terms["body_lin_vel_b"].params["anchor_cfg"].body_names = (anchor_name,)
  cfg.observations["amp"].terms["body_lin_vel_b"].params["body_cfg"].body_names = body_names

  cfg.observations["amp"].terms["body_ang_vel_b"].params["anchor_cfg"].body_names = (anchor_name,)
  cfg.observations["amp"].terms["body_ang_vel_b"].params["body_cfg"].body_names = body_names

  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.curriculum = {}
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )

    cfg.events["init_motion_loader"].params["delay_reset_env_ratio"] = 1.0

    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def pm01_amp_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create EngineAI PM01 flat terrain velocity configuration."""
  cfg = pm01_amp_rough_env_cfg(play=play)

  cfg.sim.njmax = 640
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 256
  cfg.sim.nconmax = None

  # Switch to flat terrain.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # Remove raycast sensor and height scan (no terrain to scan).
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )
  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]

  # Disable terrain curriculum (not present in play mode since rough clears all).
  cfg.curriculum.pop("terrain_levels", None)

  if play:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-1.5, 3.0)
    twist_cmd.ranges.lin_vel_y = (-1.0, 1.0)
    twist_cmd.ranges.ang_vel_z = (-3.14 / 2, 3.14 / 2)

  return cfg


def pm01_amp_flat_aligned_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Flat PPO/WM comparison with identical grounded AMP and critic inputs."""
  cfg = pm01_amp_flat_env_cfg(play=play)
  if not play:
    cfg.scene.num_envs = 4096
  old_terms = cfg.observations["amp"].terms
  body_cfg = old_terms["body_pos_b"].params["body_cfg"]
  body_cfg.preserve_order = True
  anchor_cfg = old_terms["body_pos_b"].params["anchor_cfg"]
  cfg.observations["amp"] = ObservationGroupCfg(
    terms={
      "grounded_features": ObservationTermCfg(
        func=robot_grounded_amp_features,
        params={
          "anchor_cfg": anchor_cfg,
          "body_cfg": body_cfg,
          "pelvis_cfg": SceneEntityCfg("robot", body_names=(PM01_ROOT_BODY,)),
          "foot_cfg": SceneEntityCfg(
            "robot", body_names=PM01_FOOT_BODIES,
            preserve_order=True,
          ),
          "schema": "grounded_v2",
        },
      ),
    },
    concatenate_terms=True,
    enable_corruption=False,
    history_length=1,
  )
  cfg.observations["critic"] = ObservationGroupCfg(
    terms={"physical_context": ObservationTermCfg(func=aligned_critic_features)},
    concatenate_terms=True,
    enable_corruption=False,
    history_length=1,
  )
  return cfg
