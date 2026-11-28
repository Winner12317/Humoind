"""Train depth-perceptive forward walking and turning on Hiking terrain.

Use ``source scripts/activate_env.sh`` from this checkout. The original
``train_wmac.py`` and flat-task checkpoints retain their existing contract.
"""
from __future__ import annotations

import argparse
import hashlib
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import json
import os
import random
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import tyro
from src.wmac_perceptive.config import (
    DYNAMICS_FIELDS, LEGACY_DYNAMICS, PERCEPTION_TRAINING_FIELDS, SUPERMIMIC_TRAINING_SCHEDULE,
    PerceptiveConfig, config_from_payload, HIKING_FIXED_CAMERA_PROTOCOLS, SUPERMIMIC_REFERENCE_COMMIT,
    HIKING_MOTION_PROTOCOL,G1_RPL_SENSOR_PROTOCOL,G1_HIKING_DR_SENSOR_PROTOCOL,
    G1_HIKING_CALIBRATED_SENSOR_PROTOCOL,
    LEGACY_WALKING_EXPERT, TRANSFORMER_ACTOR_TYPE,
)


def parse_config(argv=None):
    arguments=list(sys.argv[1:] if argv is None else argv)
    preliminary=argparse.ArgumentParser(add_help=False)
    preliminary.add_argument("--resume")
    preliminary.add_argument("--execution-profile")
    preliminary.add_argument("--hiking-training-version", type=int)
    preliminary.add_argument("--migrate-perception-training", action="store_true")
    preliminary.add_argument("--migrate-world-model", action="store_true")
    selected,_=preliminary.parse_known_args(arguments)
    if selected.migrate_world_model and selected.migrate_perception_training:
        raise ValueError("Migrate perception training and the world model in separate explicit resumes")
    defaults=PerceptiveConfig.defaults(selected.execution_profile or "supermimic",
        hiking_training_version=selected.hiking_training_version)
    if selected.resume and selected.resume!="None":
        import torch
        payload=torch.load(Path(selected.resume).expanduser(),map_location="cpu",weights_only=False)
        defaults=config_from_payload(payload)
        if selected.execution_profile and selected.execution_profile != defaults.execution_profile:
            raise ValueError("Ordinary resume cannot change execution_profile; start a fresh experiment")
        defaults.resume=selected.resume
        if selected.migrate_perception_training:
            if defaults.perception_training_version != 1:
                raise ValueError("Perception training migration requires a version 1 checkpoint")
            repaired=PerceptiveConfig.defaults()
            for name in PERCEPTION_TRAINING_FIELDS:
                setattr(defaults,name,getattr(repaired,name))
            defaults.imagination_horizon=min(defaults.imagination_horizon,defaults.wm_window)
        if selected.migrate_world_model:
            if defaults.dynamics_parameterization != LEGACY_DYNAMICS:
                raise ValueError("World-model migration requires an independent_residual_v1 checkpoint")
            repaired=PerceptiveConfig.defaults()
            for name in DYNAMICS_FIELDS:
                setattr(defaults,name,getattr(repaired,name))
    elif selected.migrate_perception_training:
        raise ValueError("--migrate-perception-training requires --resume from a version 1 checkpoint")
    elif selected.migrate_world_model:
        raise ValueError("--migrate-world-model requires --resume from an independent_residual_v1 checkpoint")
    resume_visual_mode=defaults.visual_mode
    resume_actor_type=defaults.actor_type
    resume_execution_profile=defaults.execution_profile
    resume_execution_optimizer=defaults.execution_optimizer
    resume_training_schedule=defaults.training_schedule
    resume_sensor_protocol=defaults.sensor_protocol
    resume_motion_data_protocol=defaults.motion_data_protocol
    resume_locomotion_objective=defaults.locomotion_objective
    resume_standing_expert_source=defaults.standing_expert_source
    resume_walking_expert_source=defaults.walking_expert_source
    resume_hiking_training_version=defaults.hiking_training_version
    resume_robot_dynamics_randomization=defaults.robot_dynamics_randomization
    resume_robot_variant=defaults.robot_variant
    cfg=tyro.cli(PerceptiveConfig,default=defaults,args=arguments)
    cfg.apply_preset()
    if selected.resume and selected.resume!="None" and cfg.hiking_training_version != resume_hiking_training_version:
        raise ValueError("Ordinary resume cannot change hiking_training_version; start a fresh experiment for the new training protocol")
    if (selected.resume and selected.resume!="None"
            and cfg.robot_dynamics_randomization != resume_robot_dynamics_randomization):
        raise ValueError("Ordinary resume cannot change robot_dynamics_randomization (it widens the world-model "
                         "and critic context); start a fresh experiment")
    if selected.resume and selected.resume!="None" and cfg.robot_variant != resume_robot_variant:
        raise ValueError("Ordinary resume cannot change robot_variant; start a fresh experiment")
    if selected.resume and selected.resume!="None" and cfg.visual_mode != resume_visual_mode:
        raise ValueError("Ordinary resume cannot change visual_mode; Instinct-latent training starts a fresh experiment")
    if selected.resume and selected.resume!="None" and cfg.actor_type != resume_actor_type:
        raise ValueError("Ordinary resume cannot change actor_type; start a fresh experiment")
    if selected.resume and selected.resume!="None" and cfg.execution_profile != resume_execution_profile:
        raise ValueError("Ordinary resume cannot change execution_profile; start a fresh experiment")
    if selected.resume and selected.resume!="None" and cfg.execution_optimizer != resume_execution_optimizer:
        raise ValueError("Ordinary resume cannot change execution_optimizer; start a fresh experiment")
    if selected.resume and selected.resume!="None" and cfg.training_schedule != resume_training_schedule:
        raise ValueError("Ordinary resume cannot change training_schedule; start a fresh experiment")
    if selected.resume and selected.resume!="None" and cfg.sensor_protocol != resume_sensor_protocol:
        raise ValueError("Ordinary resume cannot change sensor_protocol; start a fresh experiment for the new camera")
    if selected.resume and selected.resume!="None" and cfg.motion_data_protocol != resume_motion_data_protocol:
        raise ValueError("Ordinary resume cannot change motion_data_protocol; official Hiking data/AMP requires a fresh experiment")
    if selected.resume and selected.resume!="None" and cfg.locomotion_objective != resume_locomotion_objective:
        raise ValueError("Ordinary resume cannot change locomotion_objective; start a fresh Hiking standing experiment")
    if selected.resume and selected.resume!="None" and cfg.standing_expert_source != resume_standing_expert_source:
        raise ValueError("Ordinary resume cannot change standing_expert_source; start a fresh experiment")
    if selected.resume and selected.resume!="None" and cfg.walking_expert_source != resume_walking_expert_source:
        raise ValueError("Ordinary resume cannot change walking_expert_source; start a fresh experiment")
    if (not (selected.resume and selected.resume!="None")
            and cfg.training_schedule == SUPERMIMIC_TRAINING_SCHEDULE
            and not any(argument.split("=",1)[0] == "--imagination-horizon-start" for argument in arguments)):
        cfg.imagination_horizon_start=cfg.imagination_horizon
    if (not (selected.resume and selected.resume!="None")
            and cfg.training_schedule == SUPERMIMIC_TRAINING_SCHEDULE
            and not any(argument.split("=",1)[0] == "--utd" for argument in arguments)):
        # Official replay_ratio counts state frames (K+1), not transitions.
        # Preserve ratio16 when fresh engineering overrides change N or B.
        cfg.utd=cfg.num_envs/(cfg.wm_updates*cfg.wm_batch_size*(cfg.wm_window+1))*16.
    cfg.validate()
    return cfg


def make_environment(cfg,*,play=False,mode=None,env_config_transform=None):
    import mjlab.tasks  # noqa:F401
    import src.tasks  # noqa:F401
    from src.recovery_rewards import configure_recovery_height_reward
    from src.tasks.amp_perceptive.env_cfg import g1_amp_perceptive_hiking_env_cfg
    from src.tasks.amp_perceptive.terrain import hiking_terrain_generator
    from src.wmac_perceptive.motions import prepare_mirrored_dataset,prepare_locomotion_dataset
    from src.wmac_perceptive.hiking_data import prepare_hiking_dataset
    from src.wmac_perceptive.env import PerceptiveCaptureEnv
    from src.wmac_perceptive.state import PerceptiveStateAdapter
    cfg.validate()
    if cfg.motion_root(ROOT).is_relative_to(Path('/home/autel/AMP_mjlab_wmac')):
        raise ValueError("Perceptive motion cache must not modify the original training checkout")
    # Generated mirrors remain in this isolated checkout, never overwrite the
    # source assets or the running flat experiment's loader/cache.
    locomotion=cfg.motion_scope=="forward_walk_turn_v1"
    if locomotion and mode=="recovery":raise ValueError("The locomotion-only task has no recovery mode")
    official_data=cfg.motion_data_protocol==HIKING_MOTION_PROTOCOL
    if cfg.hiking_training_version >= 3:
        from src.wmac_perceptive.hiking_data import official_hiking_reference
        official_hiking_reference(cfg.motion_root(ROOT))
    else:
        prepare=prepare_locomotion_dataset if locomotion else prepare_mirrored_dataset
        if official_data:prepare=prepare_hiking_dataset
        if cfg.standing_expert_source != "none":
            from src.wmac_perceptive.standing_data import prepare_hiking_standing_dataset
            prepare=prepare_hiking_standing_dataset
        paths=prepare(ROOT/"src/assets/motions/g1/amp",cfg.motion_root(ROOT))
    env_cfg=g1_amp_perceptive_hiking_env_cfg(play=play,motion_scope=cfg.motion_scope,
        sensor_protocol=cfg.sensor_protocol,motion_data_protocol=cfg.motion_data_protocol,
        locomotion_objective=cfg.locomotion_objective, hiking_training_version=cfg.hiking_training_version,
        robot_variant=getattr(cfg,"robot_variant","unshod"),
        collision_horizontal_scale=cfg.collision_horizontal_scale,
        depth_geometry=getattr(cfg,"depth_geometry","collision"))
    env_cfg.seed=cfg.seed;env_cfg.scene.num_envs=cfg.num_envs
    env_cfg.scene.terrain.terrain_generator=hiking_terrain_generator(
        seed=cfg.seed,rows=cfg.terrain_rows,cols=cfg.terrain_cols,recovery_flat_fraction=cfg.recovery_flat_fraction,
        collision_horizontal_scale=cfg.collision_horizontal_scale,walking_only=locomotion,
        task_protocol=cfg.locomotion_objective,
        depth_source_mesh=getattr(cfg,"depth_geometry","collision")=="source_mesh")
    if play and cfg.hiking_training_version >= 3:
        from src.tasks.amp_perceptive.hiking_task import configure_hiking_play_terrain
        configure_hiking_play_terrain(env_cfg.scene.terrain.terrain_generator)
    if not official_data:
        env_cfg.events["init_motion_loader"].params.update(motion_dir=str(paths.walk),recovery_dir=None if locomotion else str(paths.recovery))
        env_cfg.events["reset_from_motion"].params["motion_dir"]=str(paths.walk)
    if mode is not None:
        if mode not in ("walk","recovery"):raise ValueError("Unknown initial-state mode")
        if not official_data:
            env_cfg.events["init_motion_loader"].params["delay_reset_env_ratio"]=1. if mode=="recovery" else 0.
    if not locomotion:configure_recovery_height_reward(env_cfg,cfg.recovery_height_reward)
    if play:
        if cfg.hiking_training_version < 3:
            env_cfg.observations["actor"].enable_corruption=False
        env_cfg.events.pop("push_robot",None)
        if cfg.hiking_training_version < 3:
            env_cfg.curriculum={}
    env_cfg._robot_dynamics_randomization=getattr(cfg,"robot_dynamics_randomization","none")
    if env_cfg._robot_dynamics_randomization!="none" and not play:
        from mjlab.managers.event_manager import EventTermCfg
        from src.wmac_perceptive.robot_dynamics import randomize_robot_dynamics
        env_cfg.events["robot_dynamics"]=EventTermCfg(func=randomize_robot_dynamics,mode="startup",
            params={"randomization":env_cfg._robot_dynamics_randomization})
    if env_config_transform is not None:
        env_cfg=env_config_transform(env_cfg)
    env=PerceptiveCaptureEnv(cfg=env_cfg,device=getattr(cfg,"sim_device","") or cfg.device)
    env.adapter=PerceptiveStateAdapter(env,recovery_counter_semantics=cfg.recovery_counter_semantics,
        motion_scope=cfg.motion_scope,amp_feature_schema=cfg.amp_feature_schema)
    return env


def assign_devices(cfg,environment):
    """Expose only the cards this run uses and renumber them from zero.

    The learner (with replay) becomes cuda:0 and a separate simulator card
    cuda:1. A caller who already set CUDA_VISIBLE_DEVICES keeps its numbering.
    """
    if cfg.device.startswith("cuda") and "CUDA_VISIBLE_DEVICES" not in environment:
        learner=cfg.device.partition(":")[2] or "0"
        visible=[learner]
        if cfg.sim_device.startswith("cuda"):
            simulator=cfg.sim_device.partition(":")[2] or "0"
            if simulator!=learner:visible.append(simulator)
            cfg.sim_device=f"cuda:{visible.index(simulator)}"
        environment["CUDA_VISIBLE_DEVICES"]=",".join(visible);cfg.device="cuda:0"
    if cfg.sim_device==cfg.device:cfg.sim_device=""


def check_device_transfers(cfg):
    """Select the cross-GPU copy route and test both routes before anything is built.

    Returns the test report, or None when the simulator is not on a second GPU.
    A machine whose direct copies are wrong still trains correctly on the host
    route; a wrong copy on the selected route stops the run here.
    """
    import torch
    from src.wmac.device_placement import configure_transfers,crosses_gpus,transfer_self_test
    configure_transfers(cfg.device_transfer,cfg.verify_device_transfer)
    if not crosses_gpus(cfg.sim_device,cfg.device):return None
    report=transfer_self_test(cfg.device,cfg.sim_device)
    for route,directions in report.items():
        summary=", ".join(f"{direction} {wrong}/{total} wrong" for direction,(wrong,total) in directions.items())
        print(f"[Perceptive WMAC] GPU-to-GPU copy test, {route} route: {summary}",flush=True)
    if any(wrong for wrong,_ in report[cfg.device_transfer].values()):
        raise RuntimeError(f"Copies between {cfg.device} and {cfg.sim_device} arrive with wrong values on the "
            f"selected '{cfg.device_transfer}' route. "+("Use --device-transfer host." if cfg.device_transfer=="direct"
            else "Even copies through CPU memory are wrong: suspect a GPU, its slot or its memory, and do not train on this pair."))
    if any(wrong for wrong,_ in report["direct"].values()):
        print("[Perceptive WMAC] Direct GPU-to-GPU copies are unreliable on this pair; "
              "this run stages every copy through CPU memory and checksums it.",flush=True)
    print(f"[Perceptive WMAC] cross-GPU copies: {cfg.device_transfer} route, "
          f"{'checksummed on both cards' if cfg.verify_device_transfer else 'NOT verified'}",flush=True)
    return report


# Whole-card simulator use on the 2.5 cm collision grid, fitted to 4.1, 5.7, 8.9 and 12.2 GB
# measured at 256, 512, 1,024 and 1,536 robots (RTX 4080). Larger counts are extrapolated.
SIMULATOR_GPU_BYTES=(2.5*2**30,6.4*2**20)


def report_gpu_memory(cfg,environment,memory):
    """Print free memory on each card this run uses, before anything large is allocated.

    ``memory(device)`` returns (free, total) bytes for the whole card, so memory held by
    other processes shows up here. Returns {role: {...}}; warns when the simulator card
    has less free memory than the simulator is expected to need.
    """
    visible=[item for item in environment.get("CUDA_VISIBLE_DEVICES","").split(",") if item]
    simulator=cfg.sim_device or cfg.device
    roles=({"learner and replay":cfg.device,"simulator":simulator} if simulator!=cfg.device
           else {"simulator, learner and replay":cfg.device})
    need=SIMULATOR_GPU_BYTES[0]+SIMULATOR_GPU_BYTES[1]*cfg.num_envs if cfg.hiking_training_version>=3 else None
    report={}
    for role,device in roles.items():
        if not device.startswith("cuda"):continue
        index=int(device.partition(":")[2] or 0);free,total=memory(device)
        card=f"GPU {visible[index]}" if index<len(visible) else device
        report[role]={"device":device,"card":card,"free_bytes":int(free),"total_bytes":int(total)}
        line=f"[Perceptive WMAC] {role} on {device} ({card}): {free/2**30:.1f} of {total/2**30:.1f} GB free"
        if "simulator" in role and need is not None:
            report[role]["estimated_simulator_bytes"]=int(need)
            line+=f"; the simulator needs about {need/2**30:.1f} GB for {cfg.num_envs} robots"
        print(line,flush=True)
        if total-free>2*2**30:  # this process's own CUDA context stays well below
            print(f"[Perceptive WMAC] WARNING: {(total-free)/2**30:.1f} GB of {card} is already in use by another "
                  "process. `nvidia-smi` lists it.",flush=True)
        if "simulator" in role and need is not None and free<need:
            print(f"[Perceptive WMAC] WARNING: {card} has too little free memory for the simulator. Expect an "
                  "out-of-memory error while the environment is built or first reset. Free the card, choose "
                  "another one, or lower --num-envs.",flush=True)
    return report


def snapshot_sources(log_dir,cfg=None):
    for folder in ("wmac","wmac_perceptive"):
        shutil.copytree(ROOT/"src"/folder,log_dir/"source"/"src"/folder,ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(ROOT/"src/tasks/amp_perceptive",log_dir/"source/src/tasks/amp_perceptive",ignore=shutil.ignore_patterns("__pycache__"))
    if getattr(cfg,"sensor_protocol",None) in (
        G1_HIKING_CALIBRATED_SENSOR_PROTOCOL, G1_RPL_SENSOR_PROTOCOL, G1_HIKING_DR_SENSOR_PROTOCOL
    ):
        shutil.copytree(ROOT/"src/assets/cameras",log_dir/"source/src/assets/cameras")
    if getattr(cfg, "hiking_training_version", 1) >= 3:
        for asset in ("robots/unitree_g1/g1_29dof_torsobase_popsicle.urdf",
                      "robots/unitree_g1/g1_29dof_torsobase_popsicle_with_shoe.urdf",
                      "robots/unitree_g1/hiking_head_link.STL",
                      "motions/g1/hiking_official/SOURCE_MANIFEST.json"):
            target = log_dir / "source/src/assets" / asset
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / "src/assets" / asset, target)
    if getattr(cfg,"robot_dynamics_randomization","none")!="none":
        from src.wmac_perceptive.robot_dynamics import ROBOT_DYNAMICS_PROFILES,ROBOT_DYNAMICS_RANDOMIZATION_SETTINGS
        profile=ROBOT_DYNAMICS_PROFILES[ROBOT_DYNAMICS_RANDOMIZATION_SETTINGS[cfg.robot_dynamics_randomization]["profile"]]
        target=log_dir/"source/src/assets/robots/unitree_g1"/profile.name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(profile,target)
    scripts=log_dir/"source/scripts";scripts.mkdir(parents=True,exist_ok=True)
    for name in ("train_wmac_perceptive.py","prepare_mirrored_motions.py","evaluate_wmac_perceptive.py",
                 "play_wmac_perceptive.py","check_wmac_perceptive_training.py","evaluate_wmac_zero_start.py"):
        if (ROOT/"scripts"/name).is_file():shutil.copy2(ROOT/"scripts"/name,scripts/name)
    def git(*args):
        r=subprocess.run(["git","-C",str(ROOT),*args],capture_output=True,text=True)
        return r.stdout.strip() if r.returncode==0 else "unavailable"
    metadata=dict(repository=str(ROOT),commit=git("rev-parse","HEAD"),branch=git("branch","--show-current"),status=git("status","--short"),
        paper="https://arxiv.org/abs/2601.07718",reference_commit="de6b92f01eb155d51f30e8ed1ea7aecd2eb1f057",
        isolated_from="/home/autel/AMP_mjlab_wmac",scope="Perceptive WMAC; config.motion_scope distinguishes forward walking/turning from historical joint training",
        explicit_differences=["Hiking5cm generated terrain downsampled to10cm collision/raycast heightfields by default; configurable integer grid; not Isaac triangle slope-threshold meshes",
            "Locomotion scope uses a conservative walking terrain profile and no recovery allocation; legacy checkpoints retain their historical geometry",
            "Body-mounted full-orientation camera; upstream camera also follows the full body pose",
            "Separate valid pixel channels; model predicts future images instead of using unknown true future views",
            "Fixed one-step image latency; AMP has per-point terrain-relative height",
            "Training protocol and maximum/runtime imagination horizon are recorded in config, checkpoints and metrics"])
    metadata["visual_mode"]=getattr(cfg,"visual_mode","pixel_v2")
    if getattr(cfg, "actor_type", None) == TRANSFORMER_ACTOR_TYPE:
        from src.wmac_perceptive.checkpoints import architecture_metadata
        metadata["actor_architecture"] = architecture_metadata(cfg)["actor"]
    metadata["execution_profile"]=getattr(cfg,"execution_profile","legacy")
    metadata["motion_data_protocol"]=getattr(cfg,"motion_data_protocol","filtered_walk_turn_v2")
    metadata["locomotion_objective"]=getattr(cfg,"locomotion_objective","legacy_walk_turn_v1")
    metadata["standing_expert_source"]=getattr(cfg,"standing_expert_source","none")
    if metadata["motion_data_protocol"]==HIKING_MOTION_PROTOCOL:
        metadata["motion_reference"]={
            "source":"official_complete_walking_only_npz_and_yaml",
            "source_frames":18982,"loaded_frames":18981,
            "frame_content_filter":None,"height_normalization":None,
            "symmetry":"sagittal_mirror", "real_reset":"independent_default_standing_pose",
            "amp":"67_features_x_10_frames_term_major_no_running_normalizer",
            "upstream_instinct_rl_reference":"574c8d50",
            "discriminator_update_budget":"WMAC_2x512_per_learner_update;not_PPO_rollout_schedule"}
        if cfg.hiking_training_version>=2:
            metadata["motion_reference"]["discriminator_update_budget"]=(
                "Hiking_24_fresh_vector_steps_x_5_epochs_x_4_logical_minibatches;"
                "weighted_execution_microbatches;all_environments")
        metadata["explicit_differences"]=[value.replace(
            "AMP has per-point terrain-relative height",
            "AMP uses the official terrain-independent terms") for value in metadata["explicit_differences"]]
    metadata["hiking_training_version"]=cfg.hiking_training_version
    if metadata["execution_profile"]=="supermimic":
        metadata["execution_reference"]={"repository":"https://github.com/HybridRobotics/supermimic",
            "commit":SUPERMIMIC_REFERENCE_COMMIT,
            "scope":"training scale, compiled tensor kernels and optional complete CUDA updates",
            "optimizer":cfg.execution_optimizer,
            "optimizer_implementation":"PyTorch 2.10 Muon math adapted to installed PyTorch 2.7" if cfg.execution_optimizer=="muon_adam" else "saved Adam protocol",
            "replay_storage":cfg.replay_storage}
    if metadata["visual_mode"]=="instinct_latent_v1":
        metadata["visual_reference"]={"repository":"https://github.com/project-instinct/InstinctMJ",
            "branch":"supermimic","commit":"03563ee02665f327425106372264070315673748",
            "scope":"joint-stack latent residual prediction, online reconstruction and visual stop-gradients"}
        metadata["explicit_differences"]=[text.replace("model predicts future images instead of using unknown true future views",
            "model predicts the joint-stack latent; decoder is training/validation-only")
            for text in metadata["explicit_differences"]]
        metadata["explicit_differences"]=[text.replace("Fixed one-step image latency",
            "One configured history-delay step plus the current pre-sense capture lag")
            for text in metadata["explicit_differences"]]
    metadata["sensor_protocol"]=getattr(cfg,"sensor_protocol","local_depth_valid_v1")
    if metadata["sensor_protocol"] in (*HIKING_FIXED_CAMERA_PROTOCOLS, G1_HIKING_DR_SENSOR_PROTOCOL):
        from src.tasks.amp_perceptive.sensors import camera_protocol_metadata
        metadata["camera_reference"]=camera_protocol_metadata(cfg.sensor_protocol)
        metadata["explicit_differences"]=[text for text in metadata["explicit_differences"]
            if not text.startswith("Separate valid pixel channels")]
        metadata["explicit_differences"].append(
            "Instinct depth-only joint-stack latent; raw validity is diagnostic only; shared uint8 single-frame Replay")
    if getattr(cfg,"locomotion_objective",None)=="hiking_full_v1":
        metadata["explicit_differences"]=[
            "SuperMimic GenCoord BPTT and InstinctMJ latent objective retained instead of Hiking PPO",
            "User task restricts commands to forward walking and turning; legacy standing expert supplement retained",
            "MuJoCo heightfield/box collision solver differs from PhysX slope-threshold triangle meshes",
            "MuJoCo has a single sliding-friction coefficient; source dynamic friction is applied, static/restitution draws are diagnostic only",
            "PhysX joint solver velocity limit has no direct MuJoCo equivalent; source soft velocity-limit reward retained",
            "Local G1 MJCF pelvis free root retained; task quantities transformed to the Hiking torso frame",
            "13 supervised physics auxiliary outputs cover contacts, torque costs, acceleration and substep timers; GenCoord physical branch remains35 velocity deltas",
        ]
        metadata["depth_capture"]="Fresh post-sense successor; pre-reset terminal view captured separately; configured one-step delay only"
        metadata["proprioception"]="8 samples per term in source term-major layout, 768 total, torso frame, gyro*.25 and joint velocity*.05"
    if metadata["sensor_protocol"]==G1_RPL_SENSOR_PROTOCOL:
        from src.tasks.amp_perceptive.sensors import camera_protocol_metadata
        metadata["camera_reference"]=camera_protocol_metadata(cfg.sensor_protocol)
        metadata["depth_capture"]="RPL10Hz capture and two-capture delay; current/terminal packets use one corruption draw"
        metadata["explicit_differences"] += [
            "Measured single G1 depth camera K with unchanged, not newly calibrated, torso mount",
            "RPL source image randomization and preprocessing, not RPL PPO/distillation or a new world-model architecture",
            "Signed FP16 current/successor single-frame caches preserve bicubic overshoot and separate reset/terminal images"]
    if metadata["sensor_protocol"]==G1_HIKING_DR_SENSOR_PROTOCOL:
        from src.tasks.amp_perceptive.sensors import camera_protocol_metadata
        metadata["camera_reference"]=camera_protocol_metadata(cfg.sensor_protocol)
        metadata["depth_capture"]="Hiking50Hz capture; original one-control-step history delay; terminal/reset packets remain separate"
        metadata["explicit_differences"] += [
            "Measured G1 intrinsics with Hiking torso-local nominal mount",
            "RPL-inspired pixel/camera randomization adapted to Hiking64x36 and18x32 ROI; no RPL resolution, clipping, normalization or timing",
            "Camera DR ranges are explicit RPL-derived starting ranges, not original Hiking parameters or measured mounting uncertainty"]
    if cfg.hiking_training_version >= 3:
        shod = getattr(cfg, "robot_variant", "unshod") == "shoe"
        metadata["hiking_task"] = "G1ParkourEnvCfg" if shod else "G1ParkourRoughEnvCfg"
        metadata["robot_variant"] = "upstream_shod_G1_undelayed_actuators" if shod else "upstream_unshod_G1"
        from src.wmac_perceptive.locomotion_data import HIKING_SOURCE_SHA256
        robot_assets = (("g1_29dof_torsobase_popsicle_with_shoe.urdf" if shod else "g1_29dof_torsobase_popsicle.urdf"),
                        "hiking_head_link.STL")
        if cfg.hiking_training_version >= 4:
            metadata["hiking_alignment"] = {
                "camera": cfg.sensor_protocol,
                "terrain_edges": "GreedyconcatEdgeCylinder on Hiking's slope_threshold=1 terrain mesh",
                "collision_geometry": (f"MuJoCo heightfield on a {cfg.collision_horizontal_scale} m grid; "
                                       "a vertical riser is a ramp one grid cell wide"),
                "depth_geometry": ("Hiking's vertical-riser terrain mesh (non-colliding)"
                                   if cfg.depth_geometry == "source_mesh" else "the collision heightfield"),
                "discriminator_logit_weight_l2": cfg.discriminator_head_weight_decay,
                "initial_terrain_level": "sampled level minus one, as IsaacLab's first reset",
            }
        metadata["robot_asset_sha256"] = {name: hashlib.sha256(
            (log_dir / "source/src/assets/robots/unitree_g1" / name).read_bytes()).hexdigest()
            for name in robot_assets}
        metadata["motion_reference"]["source_files_sha256"] = HIKING_SOURCE_SHA256
        metadata["motion_reference"]["expert_sampling"] = "per-environment reference clock and reset symmetry"
        if cfg.walking_expert_source == LEGACY_WALKING_EXPERT:
            metadata["motion_reference"]["source"] = "complete_official_hiking_plus_historical_local_walk_turn"
            metadata["motion_reference"]["legacy_walking_reference"] = {
                "source": cfg.walking_expert_source,
                "historical_local_protocol_commit": "44bd98ba",
            }
            metadata["motion_reference"]["expert_sampling"] = (
                "uniform valid ten-frame windows across all official and legacy walking clips; shared across terrains"
            )
        metadata["execution_reference"]["iteration_unit"] = "one vector environment control step"
        metadata["execution_reference"]["checkpoint_interval_unit"] = "learner updates"
        metadata["explicit_differences"] = [
            description for description in metadata["explicit_differences"]
            if "legacy standing expert supplement" not in description]
    if metadata["sensor_protocol"] == G1_HIKING_CALIBRATED_SENSOR_PROTOCOL:
        metadata["explicit_differences"].append(
            "Measured G1 depth intrinsics at the nominal Hiking torso mount")
    (log_dir/"source.json").write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n')
    (log_dir/"source.diff").write_text(git("diff","HEAD")+'\n')


def main(cfg):
    import numpy as np
    import torch
    from mjlab.utils.torch import configure_torch_backends
    from src.wmac_perceptive.runner import PerceptiveRunner
    from src.wmac_perceptive.physics_checks import require_fixed_hfield_collision
    cfg.validate()
    # Before any log directory exists: refuse physics that drops terrain contacts.
    hfield=require_fixed_hfield_collision(cfg.allow_unpatched_hfield)
    os.environ.setdefault("MUJOCO_GL","egl");os.environ.setdefault("OMP_NUM_THREADS","1")
    assign_devices(cfg,os.environ)
    random.seed(cfg.seed);np.random.seed(cfg.seed);torch.manual_seed(cfg.seed)
    if cfg.device.startswith("cuda"):
        if not torch.cuda.is_available():raise RuntimeError("CUDA unavailable")
        torch.cuda.manual_seed_all(cfg.seed)
    configure_torch_backends(allow_tf32=cfg.precision=="tf32")
    # Outputs may live on shared storage, directly or via a checkout-local symlink.
    base=Path(cfg.log_root).expanduser();base=base if base.is_absolute() else ROOT/base
    if cfg.motion_root(ROOT).is_relative_to(Path('/home/autel/AMP_mjlab_wmac')):
        raise ValueError("Perceptive motion cache must not modify the original training checkout")
    if Path(cfg.run_name).name!=cfg.run_name:raise ValueError("run_name must be one path component")
    log_dir=base/f"{datetime.now():%Y-%m-%d_%H-%M-%S_%f}_seed{cfg.seed}_{cfg.run_name}"
    log_dir.mkdir(parents=True,exist_ok=False)
    (log_dir/"config.json").write_text(json.dumps(asdict(cfg),ensure_ascii=False,indent=2)+'\n')
    snapshot_sources(log_dir,cfg)
    print(f"[Perceptive WMAC] log_dir={log_dir}",flush=True)
    print(f"[Perceptive WMAC] config={json.dumps(asdict(cfg),sort_keys=True)}",flush=True)
    if cfg.hiking_training_version >= 3:
        reference = ("Hiking and historical walking references" if cfg.walking_expert_source == LEGACY_WALKING_EXPERT
                     else "official Hiking walking reference")
        robot = "shod G1 (no actuator delay)" if cfg.robot_variant == "shoe" else "unshod G1"
        terrain = (f"collision grid {cfg.collision_horizontal_scale} m, depth sees "
                   f"{'Hiking riser mesh' if cfg.depth_geometry == 'source_mesh' else 'collision heightfield'}")
        print(f"[Perceptive WMAC] Hiking v{cfg.hiking_training_version} task, {robot}, camera={cfg.sensor_protocol}, "
              f"{terrain}, {reference}. Iterations count vector control steps.", flush=True)
    elif cfg.hiking_training_version>=2:
        print("[Perceptive WMAC] Hiking training v2: per-geom friction conditioning, coherent soft imagined termination, continuous WM statistics, AMP 24-step rollouts x 5 epochs.",flush=True)
    elif cfg.resume:
        print("[Perceptive WMAC] Resuming legacy Hiking training v1: the four v2 corrections are inactive. Start a fresh experiment without --resume to use v2.",flush=True)
    if cfg.visual_mode=="instinct_latent_v1":
        print("[Perceptive WMAC] InstinctMJ joint-stack latent vision: online reconstruction trains E; Actor uses stopped visual gradients; GenCoord physics retained.",flush=True)
    else:
        print("[Perceptive WMAC] Learned future-depth prediction is an experimental extension; official SuperMimic predicts physical state, not depth images.",flush=True)
    if cfg.actor_type == TRANSFORMER_ACTOR_TYPE:
        print("[Perceptive WMAC] Transformer Actor uses state-conditioned readout over proprioception, depth and command tokens.",
              flush=True)
    print(f"[Perceptive WMAC] heightfield collisions: {hfield['status']} (mujoco-warp {hfield['mujoco_warp_version']})",
          flush=True)
    if cfg.robot_dynamics_randomization!="none":
        print(f"[Perceptive WMAC] robot dynamics randomization: {cfg.robot_dynamics_randomization} "
              "(startup-only; sampled values condition the world model and critic, not the Actor)",flush=True)
    transfer_check=None
    if cfg.sim_device:
        print(f"[Perceptive WMAC] placement: simulator on {cfg.sim_device}; learner and replay on {cfg.device} "
              f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','unset')})",flush=True)
        transfer_check=check_device_transfers(cfg)
    env=runner=None;started=time.perf_counter();complete=False;failure=None
    memory_at_start=report_gpu_memory(cfg,os.environ,torch.cuda.mem_get_info)
    if cfg.device.startswith("cuda"):torch.cuda.reset_peak_memory_stats()
    try:
        env=make_environment(cfg)
        runner=PerceptiveRunner(env,cfg,log_dir)
        print(f"[Perceptive WMAC] replay_layout={getattr(runner.replay,'storage_protocol',type(runner.replay).__name__)}; "
              f"sim nconmax={env.cfg.sim.nconmax}, njmax={env.cfg.sim.njmax}, "
              f"ccd_iterations={env.cfg.sim.mujoco.ccd_iterations}, "
              f"contact_sensor_maxmatch={env.cfg.sim.contact_sensor_maxmatch}",flush=True)
        if cfg.resume:runner.load(Path(cfg.resume).expanduser())
        if cfg.perception_training_version>=2 and cfg.hiking_training_version<3:
            # Keep the exact starting policy for a matched physical comparison.
            runner.save(log_dir/f"model_{runner.iteration}.pt",export=False)
        runner.learn();complete=True
    except BaseException as error:
        failure={"type":type(error).__name__,"message":str(error)}
        raise
    finally:
        runtime=dict(completed=complete,wall_seconds=time.perf_counter()-started,hfield_collision=hfield)
        if failure is not None:runtime["error"]=failure
        if cfg.device.startswith("cuda"):
            runtime["torch_cuda_peak_allocated_bytes"]=torch.cuda.max_memory_allocated(cfg.device)
            runtime["torch_cuda_peak_reserved_bytes"]=torch.cuda.max_memory_reserved(cfg.device)
            runtime["devices"]={"learner_and_replay":cfg.device,"simulator":cfg.sim_device or cfg.device}
            if transfer_check is not None:
                from src.wmac.device_placement import transfer_statistics
                runtime["device_transfer"]={"route":cfg.device_transfer,"verified":cfg.verify_device_transfer,
                    "startup_test_wrong_of_total":transfer_check,**transfer_statistics}
            # Whole-card use as the driver sees it: PyTorch, the simulator and CUDA graphs together.
            runtime["gpu_memory_used_bytes"]={name:(lambda free,total:total-free)(*torch.cuda.mem_get_info(name))
                for name in dict.fromkeys((cfg.device,cfg.sim_device or cfg.device)) if name.startswith("cuda")}
            runtime["gpu_memory_at_start"]=memory_at_start
            runtime["torch_memory_scope"]="PyTorch allocator; external simulator/driver allocations are additional"
        if runner is not None and runner.uses_fast_execution:
            execution=runner._latent_wm_execution
            runtime["wm_execution_mode"]=execution.mode
            runtime["wm_forward_backward_graph_captured"]=execution._graph is not None
            runtime["wm_full_update_graph_captured"]=(execution.mode=="graph_full" and execution._graph is not None)
            runtime["wm_execution_updates"]=execution._calls
            actor_execution=getattr(runner,"_latent_actor_execution",None)
            actor_update=getattr(actor_execution,"execution",None)
            critic_update=getattr(runner,"_latent_critic_execution",None)
            policy_macro=getattr(runner,"_latent_actor_critic_macro_execution",None)
            if policy_macro is not None:
                actor_update=policy_macro.execution
                critic_update=policy_macro.execution if cfg.critic_bootstrap else None
            runtime["wm_update_granularity"]=("complete_macro" if hasattr(execution,"step_macro") else "minibatch")
            wm_inputs=getattr(getattr(execution,"execution",execution),"_static_inputs",None)
            runtime["wm_graph_persistent_input_bytes"]=(sum(value.numel()*value.element_size()
                for value in wm_inputs.values()) if wm_inputs is not None else 0)
            runtime["policy_update_granularity"]=("Actor_and_all_Critic_updates" if policy_macro is not None
                                                   else "separate_Actor_and_Critic")
            amp_execution=getattr(runner,"_hiking_amp_gp_execution",None)
            amp_logical=getattr(runner,"_hiking_amp_logical_execution",None)
            logical_captured=getattr(amp_logical,"graph",None) is not None
            micro_captured=getattr(amp_execution,"graph",None) is not None
            runtime["amp_gp_graph_captured"]=logical_captured or micro_captured
            runtime["amp_execution_granularity"]=("logical_minibatch" if logical_captured
                else "microbatch" if micro_captured else "not_captured")
            runtime["wm_shared_reconstruction"]=getattr(execution,"shared_reconstruction",False)
            if amp_logical is not None:
                runtime["amp_logical_graph_real_warmup_updates"]=amp_logical.real_warmup_logical_updates
                runtime["amp_logical_graph_replays"]=amp_logical.replay_count
            if amp_execution is not None:
                runtime["amp_gp_real_warmup_microbatches"]=amp_execution.real_warmup_microbatches
                runtime["amp_gp_graph_replays"]=amp_execution.replay_count
            runtime["actor_full_update_graph_captured"]=(actor_update is not None and actor_update._graph is not None)
            runtime["critic_full_update_graph_captured"]=(critic_update is not None and critic_update._graph is not None)
        if runner is not None and hasattr(runner.replay,"storage_report"):
            runtime["replay_storage"]=runner.replay.storage_report()
        if env is not None:
            runtime["simulation_capacity"]={
                "nconmax":env.cfg.sim.nconmax,"njmax":env.cfg.sim.njmax,
                "ccd_iterations":env.cfg.sim.mujoco.ccd_iterations,
                "contact_sensor_maxmatch":env.cfg.sim.contact_sensor_maxmatch,
                "naconmax":env.sim.wp_data.naconmax,"naccdmax":env.sim.wp_data.naccdmax}
        # Keep allocator/layout evidence even if releasing a failed CUDA graph
        # raises during simulator shutdown.
        (log_dir/"runtime.json").write_text(json.dumps(runtime,indent=2)+'\n')
        if env is not None:env.close()


if __name__=="__main__":main(parse_config())
