"""Versioned Hiking task configuration; learning remains SuperMimic WMAC."""
from __future__ import annotations

from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch

from .hiking_rewards import (HIKING_REWARD_WEIGHTS,HikingContactSensorCfg,
    hiking_real_reward,hiking_real_termination)


def _height_failure(env,minimum_height=.5):
    if minimum_height!=.5:raise ValueError("Hiking height threshold is .5m")
    return hiking_real_termination(env,"root_height")


def _orientation_failure(env,limit_angle=1.):
    if limit_angle!=1.:raise ValueError("Hiking tilt threshold is1rad")
    return hiking_real_termination(env,"bad_orientation")


def configure_hiking_task(cfg,*,play=False):
    from src.wmac_perceptive.hiking_actuators import configure_hiking_actuators
    from src.wmac_perceptive.hiking_observations import configure_hiking_observations
    from .commands import hiking_tracking_terrain_curriculum
    source_robot = getattr(cfg, "_hiking_training_version", 1) >= 3
    if source_robot:
        from .hiking_robot import get_hiking_robot_spec
        cfg.scene.entities["robot"].spec_fn = get_hiking_robot_spec
        cfg.scene.env_spacing = 2.5
        # The single all-body Hiking contact sensor supplies the measured
        # forces and timers used by every contact reward, label and reset.
        cfg.scene.sensors = tuple(sensor for sensor in cfg.scene.sensors
                                  if sensor.name not in ("feet_ground_contact", "self_collision"))
    configure_hiking_actuators(cfg)
    configure_hiking_observations(cfg)
    from .hiking_resets import configure_hiking_resets
    configure_hiking_resets(cfg)
    cfg.scene.sensors=tuple(s for s in cfg.scene.sensors if s.name!="hiking_contact")+(
        HikingContactSensorCfg(name="hiking_contact",
            primary=ContactMatch(mode="body",pattern=".*",entity="robot"),
            fields=("found","force"),reduce="netforce",num_slots=1,
            history_length=3,track_air_time=True),)
    cfg.rewards={name:RewardTermCfg(func=hiking_real_reward,weight=weight,
                                  params={"term_name":name})
                 for name,weight in HIKING_REWARD_WEIGHTS.items()}
    cfg.terminations={
        "time_out":cfg.terminations["time_out"],
        "bad_base_height":TerminationTermCfg(func=_height_failure,params={"minimum_height":.5}),
        "bad_orientation":TerminationTermCfg(func=_orientation_failure,params={"limit_angle":1.}),
        "base_contact":TerminationTermCfg(func=hiking_real_termination,params={"term_name":"base_contact"}),
        "terrain_boundary":TerminationTermCfg(func=hiking_real_termination,
            params={"term_name":"terrain_out_bound"},time_out=True),
    }
    cfg.episode_length_s=20.
    cfg.scene.terrain.max_init_terrain_level=5
    command=cfg.commands["twist"]
    command.task_protocol="hiking_full_v1"
    command.in_place_turn_fraction=0.
    command.rel_standing_envs=.05
    command.resampling_time_range=(8.,12.)
    command.ranges.lin_vel_x=(0.,0.)
    command.ranges.lin_vel_y=(0.,0.)
    command.ranges.ang_vel_z=(-1.,1.)
    cfg.curriculum={} if play and not source_robot else {"terrain_levels":CurriculumTermCfg(
        func=hiking_tracking_terrain_curriculum,params={"command_name":"twist"})}
    for name in ("push_robot","encoder_bias","base_com","foot_friction"):
        cfg.events.pop(name,None)
    # Material randomization is installed below by the explicit MuJoCo adapter;
    # no velocity pushes, encoder offsets or COM changes are inherited.
    from src.wmac_perceptive.hiking_materials import configure_hiking_materials
    configure_hiking_materials(cfg)
    if source_robot and play:
        # G1ParkourRoughEnvCfg_PLAY uses the ordinary G1 and keeps root twist DR.
        cfg.scene.num_envs = 10
        cfg.episode_length_s = 10.
        cfg.terminations.pop("bad_base_height")
        cfg.events.pop("physics_material")
        cfg.events["reset_hiking_state"].params["joint_position_range"] = (0., 0.)
        cfg.observations["actor"].enable_corruption = True
        configure_hiking_play_terrain(cfg.scene.terrain.terrain_generator)
    return cfg


def configure_hiking_play_terrain(generator):
    """Apply source PLAY overrides after selecting the terrain seed."""
    generator.num_rows = 4
    generator.num_cols = 10
    for sub_terrain in generator.sub_terrains.values():
        sub_terrain.wall_probability = 0.
