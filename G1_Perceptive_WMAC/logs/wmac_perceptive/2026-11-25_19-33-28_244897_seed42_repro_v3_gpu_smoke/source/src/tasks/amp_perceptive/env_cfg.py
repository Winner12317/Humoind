"""Perceptive forward walking and turning over Hiking terrain families."""
from __future__ import annotations

from dataclasses import fields
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg,ObservationTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ObjRef,RayCastSensorCfg
from src.tasks.amp_loco.config.g1.env_cfgs import g1_amp_flat_aligned_env_cfg
from src.wmac_perceptive.config import (
    DEPTH_ONLY_SENSOR_PROTOCOLS, G1_HIKING_DR_SENSOR_PROTOCOL, G1_RPL_SENSOR_PROTOCOL,
    HIKING_FIXED_CAMERA_PROTOCOLS, LEGACY_SENSOR_PROTOCOL,
)
from .commands import FlatPatchVelocityCommandCfg,navigation_terrain_curriculum
from .terrain import hiking_terrain_generator
from .sensors import HikingDepthPatternCfg, HikingRayCastSensorCfg, depth_observation, HIKING_CAMERA_MAX_DISTANCE
from . import mdp


def g1_amp_perceptive_hiking_env_cfg(play=False,motion_scope="forward_walk_turn_v1",
                                   sensor_protocol=LEGACY_SENSOR_PROTOCOL,
                                   motion_data_protocol="filtered_walk_turn_v2",
                                   locomotion_objective="legacy_walk_turn_v1",
                                   hiking_training_version=1):
    """Build the current walking task or an explicit historical joint task.

    The scope is explicit so loading an old checkpoint for comparison cannot
    silently give it different resets, commands, rewards or terrain weights.
    """
    if motion_scope not in ("forward_walk_turn_v1","joint_recovery_v1"):
        raise ValueError(f"Unsupported perceptive motion scope: {motion_scope}")
    if sensor_protocol != LEGACY_SENSOR_PROTOCOL and sensor_protocol not in DEPTH_ONLY_SENSOR_PROTOCOLS:
        raise ValueError(f"Unsupported perceptive sensor protocol: {sensor_protocol}")
    if motion_data_protocol not in ("filtered_walk_turn_v2","hiking_walk_v1"):
        raise ValueError(f"Unsupported perceptive motion-data protocol: {motion_data_protocol}")
    if locomotion_objective not in ("legacy_walk_turn_v1","hiking_standing_v1","hiking_full_v1"):
        raise ValueError(f"Unsupported perceptive locomotion objective: {locomotion_objective}")
    walking_only=motion_scope=="forward_walk_turn_v1"
    official_motion=motion_data_protocol=="hiking_walk_v1"
    full_objective=locomotion_objective=="hiking_full_v1"
    standing_objective=locomotion_objective in ("hiking_standing_v1","hiking_full_v1")
    if official_motion and not walking_only:
        raise ValueError("Hiking walking references require the forward-walk/turn task")
    if standing_objective and not (walking_only and official_motion):
        raise ValueError("Hiking standing objective requires forward-walk/turn and hiking_walk_v1 data")
    cfg=g1_amp_flat_aligned_env_cfg(play=play)
    cfg._hiking_full_protocol=full_objective
    cfg._hiking_training_version=hiking_training_version
    cfg._perceptive_sensor_protocol=sensor_protocol
    cfg.scene.terrain.terrain_type="generator"
    cfg.scene.terrain.terrain_generator=hiking_terrain_generator(
        recovery_flat_fraction=0. if walking_only else .25,walking_only=walking_only,
        collision_horizontal_scale=.05 if hiking_training_version >= 3 else .1,
        task_protocol=locomotion_objective)
    cfg.scene.terrain.max_init_terrain_level=1
    # The physical camera follows full torso orientation. Falling terminates
    # the walking-only task; the historical joint scope preserves old behavior.
    official_camera=sensor_protocol in HIKING_FIXED_CAMERA_PROTOCOLS
    sensor_type=HikingRayCastSensorCfg if official_camera else RayCastSensorCfg
    depth=sensor_type(name="hiking_depth",frame=ObjRef(type="body",name="torso_link",entity="robot"),
                     pattern=HikingDepthPatternCfg(sensor_protocol=sensor_protocol),ray_alignment="base",
                     max_distance=2*HIKING_CAMERA_MAX_DISTANCE if official_camera else 4.,
                     include_geom_groups=(0,2),exclude_parent_body=not official_camera,debug_vis=False)
    if sensor_protocol==G1_RPL_SENSOR_PROTOCOL:
        from .g1_rpl_sensor import G1RPLDepthPatternCfg,G1RPLRayCastSensorCfg
        depth=G1RPLRayCastSensorCfg(name="hiking_depth",
            frame=ObjRef(type="body",name="torso_link",entity="robot"),
            pattern=G1RPLDepthPatternCfg(),randomize_camera=not play,ray_alignment="base",
            max_distance=2.5,include_geom_groups=(0,2),exclude_parent_body=False,debug_vis=False)
        cfg._rpl_depth_randomization=not play
    if sensor_protocol==G1_HIKING_DR_SENSOR_PROTOCOL:
        from .g1_hiking_dr_sensor import G1HikingDRDepthPatternCfg,G1HikingDRRayCastSensorCfg
        depth=G1HikingDRRayCastSensorCfg(name="hiking_depth",
            frame=ObjRef(type="body",name="torso_link",entity="robot"),
            pattern=G1HikingDRDepthPatternCfg(),randomize_camera=not play,ray_alignment="base",
            max_distance=2*HIKING_CAMERA_MAX_DISTANCE,include_geom_groups=(0,2),exclude_parent_body=False,debug_vis=False)
        cfg._hiking_depth_randomization=not play
    cfg.scene.sensors=tuple(cfg.scene.sensors or ())+(depth,)
    cfg.observations["perception"]=ObservationGroupCfg(
        terms={"depth_and_valid":ObservationTermCfg(func=depth_observation)},
        concatenate_terms=True,enable_corruption=False,history_length=1)
    if official_motion:
        # This WMAC runner builds Hiking AMP histories from replay states.
        # The historical grounded observation is neither an Actor input nor
        # a discriminator input in this protocol.
        cfg.observations.pop("amp", None)
    else:
        cfg.observations["amp"].terms["grounded_features"].func=mdp.terrain_relative_amp_observation
    cfg.observations["critic"].terms["physical_context"].func=mdp.terrain_relative_critic_observation
    if walking_only:
        # This was a recovery-only bonus, not a normal walking height reward.
        cfg.rewards.pop("track_root_height",None)
        cfg.metrics.pop("mean_delay_steps",None)
        for term in cfg.rewards.values():
            if "mask_delay" in term.params:term.params["mask_delay"]=False
            if "delay_env_rew_ratio" in term.params:term.params["delay_env_rew_ratio"]=1.
    else:
        cfg.rewards["track_root_height"].func=mdp.terrain_recovery_height_reward
    if standing_objective:
        cfg.rewards["stand_still"]=RewardTermCfg(func=mdp.hiking_stand_still,weight=-.3,
            params={"command_name":"twist","threshold":.15,"offset":4.})
    from src.wmac_perceptive.safety import terrain_foot_edge_reward
    cfg.rewards["foot_edge_safety"]=RewardTermCfg(func=terrain_foot_edge_reward,weight=-.5,params={"margin":.04})
    cfg.terminations["bad_base_height"].func=mdp.terrain_root_height_below_minimum
    cfg.terminations["bad_orientation"].func=mdp.terrain_bad_orientation
    cfg.terminations["terrain_boundary"]=TerminationTermCfg(func=mdp.outside_training_patch,time_out=True)
    if official_motion:
        # Hiking's AMP reference manager does not provide physical reset
        # states. Keep the full source trajectory's terrain elevations out of
        # spawn poses; use each local model's standing defaults on the existing
        # terrain-origin pad with the exact official randomization ranges.
        cfg.events.pop("init_motion_loader")
        cfg.events.pop("reset_from_motion")
        reset=mdp.hiking_reset_protocol_metadata()
        cfg.events["reset_base"]=EventTermCfg(
            func=envs_mdp.reset_root_state_uniform,mode="reset",
            params={"pose_range":reset["pose_range"],"velocity_range":reset["velocity_range"]})
        cfg.events["reset_robot_joints"]=EventTermCfg(
            func=envs_mdp.reset_joints_by_offset,mode="reset",
            params={"position_range":reset["joint_position_offset_range"],
                    "velocity_range":reset["joint_velocity_offset_range"]})
    else:
        cfg.events["init_motion_loader"].func=mdp.init_perceptive_motion_loader
        cfg.events["init_motion_loader"].params.update(
            motion_scope=motion_scope,delay_reset_env_ratio=0. if walking_only else .4,
            max_delay_steps=0 if walking_only else 250)
        if walking_only:cfg.events["init_motion_loader"].params["recovery_dir"]=None
        cfg.events["reset_from_motion"].func=mdp.reset_on_flat_patch
    old=cfg.commands["twist"]
    values={f.name:getattr(old,f.name) for f in fields(old)}
    values.update(heading_command=False,rel_standing_envs=.05 if standing_objective else 0.,
                  rel_heading_envs=0.,init_velocity_prob=0.)
    if standing_objective:values["resampling_time_range"]=(8.,12.)
    new=FlatPatchVelocityCommandCfg(**values,in_place_turn_fraction=.2 if walking_only else 0.)
    new.ranges.lin_vel_x=(.2,.7) if walking_only else (.45,1.)
    new.ranges.lin_vel_y=(0.,0.)
    new.ranges.ang_vel_z=(-1.,1.)
    new.ranges.heading=None
    cfg.commands["twist"]=new
    # Only reached traversal goals can advance terrain difficulty. In-place
    # turning has its own targets and never counts as terrain traversal.
    cfg.curriculum={"navigation_terrain":CurriculumTermCfg(func=navigation_terrain_curriculum,
                                                           params={"command_name":"twist"})}
    cfg.sim.mujoco.ccd_iterations=100
    cfg.sim.contact_sensor_maxmatch=500
    cfg.sim.nconmax=512
    cfg.sim.njmax=4096
    if full_objective:
        # Hiking uses PhysX, so these capacities have no numerical PhysX
        # equivalent. Start from InstinctMJ Parkour 03563ee's MuJoCo settings
        # (128/700/128 CCD/128 sensor matches). This compiled G1 scene needs
        # at least 172 host contacts at initialization: retain 256 instead of
        # bypassing that check. Independent dynamic/frozen-state probes are
        # recorded in docs/validation/replay_capacity_optimization.json.
        cfg.sim.nconmax=256
        cfg.sim.njmax=700
        cfg.sim.contact_sensor_maxmatch=128
        cfg.sim.mujoco.ccd_iterations=128
    if play:
        cfg.curriculum={}
        # The old play callback is ordered after motion reset. Keeping it would
        # move origins to another tile after placing the robot on the old pad.
        cfg.events.pop("randomize_terrain",None)
        cfg.scene.terrain.terrain_generator.num_rows=2
        cfg.scene.terrain.terrain_generator.num_cols=20
    if full_objective:
        from .hiking_task import configure_hiking_task
        configure_hiking_task(cfg,play=play)
    return cfg
