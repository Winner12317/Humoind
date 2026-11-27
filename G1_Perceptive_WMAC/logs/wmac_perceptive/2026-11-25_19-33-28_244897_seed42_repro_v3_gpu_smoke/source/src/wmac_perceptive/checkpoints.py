"""Preflight guards for the separate perceptive-WMAC checkpoint format.

Nothing in this module loads parameters, optimizer moments, random states or
simulation counters. A runner must call both guards before mutating itself.
An old flat/PPO checkpoint is never inferred into a terrain/sensor policy.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, is_dataclass, replace
import json
import math
from numbers import Real

import torch

from .config import (
    DYNAMICS_FIELDS, FORMAT, GENCOORD_DYNAMICS, LEGACY_DYNAMICS,
    LATENT_VISUAL_MODE, TRANSFORMER_ACTOR_TYPE, PERCEPTION_TRAINING_FIELDS, config_from_payload,
    SUPERMIMIC_EXECUTION_PROFILE, SUPERMIMIC_TRAINING_SCHEDULE, saved_training_config,
    DEPTH_ONLY_SENSOR_PROTOCOLS, LEGACY_SENSOR_PROTOCOL, G1_RPL_SENSOR_PROTOCOL,G1_HIKING_DR_SENSOR_PROTOCOL, perception_spec_from_config,
    HIKING_MOTION_PROTOCOL, HIKING_STANDING_OBJECTIVE,HIKING_FULL_OBJECTIVE, LEGACY_WALKING_EXPERT,
)
from .perception import PerceptionSpec


# These change output placement, logging, or the fresh process environment;
# they do not relabel the saved training objective, terrain or sensor history.
MUTABLE_RESUME_FIELDS = frozenset((
    "iterations", "log_root", "run_name", "resume", "init_policy",
    "device", "replay_device", "save_interval", "log_interval",
    "evaluation_interval", "evaluation_envs", "evaluation_steps",
    "mirrored_motion_root",
    "migrate_perception_training",
    "migrate_world_model",
))

# Changes are allowed only as a recorded v1 -> v2 migration. Learning rate and
# action-shift bounds remain identical unless explicitly requested on that
# migration command; ordinary resumes may not change either.
PERCEPTION_MIGRATION_FIELDS = PERCEPTION_TRAINING_FIELDS | frozenset((
    "depth_loss_coef", "actor_lr", "max_action_shift", "imagination_horizon",
))


def _json_value(value, label):
    """Canonical tuple/list equality while rejecting non-JSON/nonfinite values."""
    try:
        return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must contain finite JSON-compatible values") from error


def _config_values(cfg):
    if not is_dataclass(cfg):
        raise ValueError("Expected the explicit perceptive dataclass configuration")
    return asdict(cfg)


def architecture_metadata(cfg, spec: PerceptionSpec | None = None) -> dict:
    """Full behavior schema: camera history, preprocessing and model interfaces.

    The default sensor spec is explicit and serialized with all dataclass
    fields, including future crop/blur settings. A caller using a nondefault
    sensor must supply it here; current training config fixes the default spec.
    """
    cfg.validate()
    spec = perception_spec_from_config(cfg) if spec is None else spec
    if not isinstance(spec, PerceptionSpec):
        raise ValueError("Expected a PerceptionSpec")
    if spec.include_valid != perception_spec_from_config(cfg).include_valid:
        raise ValueError("Perception spec include_valid conflicts with sensor_protocol")
    if spec.depth_protocol != perception_spec_from_config(cfg).depth_protocol:
        raise ValueError("Perception depth protocol conflicts with sensor_protocol")
    from src.tasks.amp_perceptive.sensors import HikingDepthPatternCfg
    camera=HikingDepthPatternCfg()
    spec_values = asdict(spec)
    if spec.depth_protocol=="hiking_v1":
        spec_values.pop("depth_protocol",None)
    if cfg.sensor_protocol == LEGACY_SENSOR_PROTOCOL:
        spec_values.pop("include_valid", None)
    metadata = {
        "format": FORMAT,
        "architecture_version": 1,
        "task": cfg.task,
        "state_dim": 71,
        "context_dim": 35,
        "action_dim": 29,
        "perception": _json_value(spec_values, "Perception spec"),
        "camera": {"frame":"torso_link","alignment":"base_full_orientation",
                   "fov_x_degrees":camera.fovx,"fov_y_degrees":camera.fovy,
                   "position":list(camera.position),"rotation_wxyz":list(camera.rotation),
                   "projection":"image_plane_X_forward_Y_left_Z_up",
                   "pixel_coordinates":"inclusive_linspace_minus1_to_plus1",
                   "minimum_depth_m":.1,"maximum_depth_m":spec.max_depth},
        "ground_height_semantics": cfg.ground_height_semantics,
        "terrain": {"source_horizontal_scale_m":.05,
                    "collision_horizontal_scale_m":cfg.collision_horizontal_scale,
                    "physical_depth":"rays against the same compiled collision heightfields",
                    "support_height":"decoded from those compiled heightfields"},
        "amp": {"schema": cfg.amp_feature_schema, "frame_dim": 203, "pair_dim": 406,
                "expert_reference": "original-project flat-ground motions; mirrored walking, unchanged Recovery",
                "policy_height_reference": "per-point terrain height at pelvis and each sole"},
        "actor": {
            "proprio_history_dim": spec.proprio_dim,
            "packed_input_dim": spec.packed_dim,
            "visual_embedding_dim": 128,
            "depth_encoder": {"input_channels": spec.visual_channels*spec.frames, "conv_channels": [4],
                              "kernel_sizes": [3], "padding": [1], "max_pool": [2, 2],
                              "projection_hidden": [256, 256], "activation": "ReLU"},
            "observation_normalization": False,
            "output_transform": "linear",
        },
        "world_model": {"kind": "terrain_and_current_depth_conditioned_physical_residual",
                        "hidden": list(cfg.model_hidden), "depth_embedding_dim": 64,
                        "next_view": "predicted normalized depth and validity, advanced every imagined step",
                        "contact_outputs": 2},
        "critic": {"hidden": list(cfg.critic_hidden),
                   "features": "terrain-relative physical features plus privileged terrain patch"},
        "discriminator": {"hidden": list(cfg.discriminator_hidden), "input_dim": 406},
    }
    if cfg.actor_type == TRANSFORMER_ACTOR_TYPE:
        from .attention_policy import PerceptiveAttentionPolicy
        metadata["actor"].update({
            "kind": "instinct_joint_stack_latent_transformer",
            "embedding_dim": PerceptiveAttentionPolicy.EMBED_DIM,
            "attention_heads": PerceptiveAttentionPolicy.NUM_HEADS,
            "attention_blocks": 1,
            "feedforward_dim": PerceptiveAttentionPolicy.FFN_DIM,
            "token_count": PerceptiveAttentionPolicy.NUM_TOKENS,
            "token_layout": ["chronological_proprioception", "depth_latent", "latest_command", "readout"],
            "position_encoding": "learned",
            "fusion": "self_attention_readout_conditioned_on_latest_proprioception_and_command",
            "activation": "GELU",
            "dropout": 0.,
        })
    else:
        metadata["actor"].update({
            "kind": ("instinct_joint_stack_latent_hiking_moe" if cfg.visual_mode == LATENT_VISUAL_MODE
                     else "depth_cnn_hiking_moe"),
            "moe_input_dim": spec.proprio_dim+128,
            "moe_variant": cfg.moe_variant,
            "num_experts": cfg.moe_num_experts,
            "experts_per_sample": cfg.moe_top_k,
            "expert_hidden": list(cfg.moe_expert_hidden),
            "gate_hidden": list(cfg.moe_gate_hidden),
            "activation": "ELU",
            "fusion": "dense_full_softmax",
            "temperature": cfg.moe_temperature,
        })
    if cfg.dynamics_parameterization == GENCOORD_DYNAMICS:
        metadata["world_model"].update({
            "kind": "terrain_and_current_depth_conditioned_gencoord",
            "parameterization": GENCOORD_DYNAMICS,
            "input_physical_dim": 68,
            "output_physical_dim": 35,
            "activation": "GELU",
            "output_normalization": "none",
            "supervision": "integrated_pose_joint_bryson_v1",
            "integration": {"dt": cfg.dynamics_dt, "decimation": cfg.dynamics_decimation,
                            "substep_coeff": (cfg.dynamics_decimation+1)/(2*cfg.dynamics_decimation)},
        })
    if cfg.visual_mode == LATENT_VISUAL_MODE:
        metadata["visual_mode"] = LATENT_VISUAL_MODE
        metadata["visual_protocol"] = {
            "reference_repository": "https://github.com/project-instinct/InstinctMJ",
            "reference_commit": "03563ee02665f327425106372264070315673748",
            "encoding": "crop_then_depth_blur_then_joint_depth_valid_stack_cnn",
            "encoder_training": "online_stack_reconstruction_only",
            "actor_latent_input": "stop_gradient_at_every_imagined_step",
            "latent_targets": "detached_reencoding_of_real_stacks_each_model_update",
            "history_order": "oldest_to_newest",
            "ring_length": spec.ring_length,
            "selected_frames": spec.frames,
            "stride": spec.stride,
            "delay": spec.delay,
            "selected_indices": list(spec.selected_indices),
            "imagination": "propagate_predicted_joint_stack_latent_each_control_step",
            "pixel_decoder_in_rollout": False,
            "validation_error": "dimensionless_mean_squared_latent_error_divided_by_target_variance",
            "absolute_validation_veto": "disabled_by_default;explicit_normalized_mse_only",
        }
        metadata["actor"].update({
            "latent_dim": cfg.latent_dim,
            "compact_input_dim": spec.proprio_dim+cfg.latent_dim,
        })
        metadata["world_model"].pop("depth_embedding_dim")
        metadata["world_model"].update({
            "kind": "instinct_joint_stack_latent_conditioned_gencoord",
            "latent_input_dim": cfg.latent_dim,
            "input_normalizer_dim": 213,
            "input_dim": 213+cfg.latent_dim,
            "output_dim": 35+2+cfg.latent_dim,
            "shared_mlp_outputs": {"physical_velocity_increment": 35, "contact": 2,
                                   "latent_delta": cfg.latent_dim},
            "latent_transition": "z_next=z+predicted_delta*dz_std",
            "latent_statistics": {"kind": "ema", "rate": cfg.latent_stat_rate,
                                  "buffers": ["z_mean", "z_std", "dz_std"]},
            "next_view": "predicted joint-stack latent, advanced every imagined step",
            "pixel_decoder_in_rollout": False,
        })
        metadata["latent_decoder"] = {
            "kind": "InstinctLatentDecoder",
            "input_dim": cfg.latent_dim,
            "input_projection": [1280],
            "transpose_conv_channels": [32, 16],
            "output_channels": spec.visual_channels*spec.frames,
            "output_shape": [spec.height-spec.crop_top-spec.crop_bottom,
                             spec.width-spec.crop_left-spec.crop_right],
            "objective": "processed_depth_valid_stack_l1",
            "use": "training_and_validation_only",
        }
        metadata["optimizer_partition"] = {
            "opt_actor": ["actor.policy"],
            "opt_model": ["world_model", "actor.vision", "latent_decoder"],
        }
    if cfg.execution_profile == SUPERMIMIC_EXECUTION_PROFILE:
        replay = {"storage_protocol": "legacy_processed_stack_fp16",
                  "latent_cache": False}
        if cfg.replay_storage == "frame_uint8":
            replay = {
                "storage_protocol": "frame_uint8_v1",
                "representation": "time_shared_processed_single_frames_not_complete_stacks",
                "cache_frames": cfg.replay_steps+spec.ring_length,
                "depth": "round(255*normalized_depth), decoded_float32_div255",
                "valid": "binary_0_or_255, decoded_exact_0_or_1",
                "history": "gather_saved_time_indices_with_original_stride_delay_and_reset_boundaries",
                "latent_cache": False,
            }
        # Placement is deliberately excluded. Moving the same saved tensors
        # from cuda:0 to cuda:1 is not a new observation/training protocol.
        metadata["execution_protocol"] = {
            "profile": SUPERMIMIC_EXECUTION_PROFILE,
            "compiled_kernels": cfg.compile,
            "precision": cfg.precision,
            "cuda_graphs": cfg.cuda_graphs,
            "graph_warmup": cfg.graph_warmup,
            "validation_horizon": cfg.validation_horizon,
            "world_model_optimizer": "Adam",
            "replay": replay,
        }
        if cfg.execution_optimizer == "muon_adam":
            metadata["execution_protocol"].update({
                "world_model_optimizer": "ReferenceMuonAdam",
                "optimizer_protocol": {
                    "algorithm": "hybrid_supermimic_muon_adam_v1",
                    "reference": "PyTorch_2.10_Muon_compatible_on_installed_2.7",
                    "grouping": "each_Sequential_Linear_layers[1:-1].weight_Muon;all_remaining_parameters_Adam",
                    "muon": {"momentum": .95, "nesterov": True, "ns_steps": 5,
                             "ns_coefficients": [3.4445, -4.775, 2.0315], "eps": 1e-7,
                             "adjust_lr_fn": "match_rms_adamw", "weight_decay": .1},
                    "adam": {"model_and_critic_betas": [.9, .999], "actor_betas": list(cfg.actor_betas),
                             "eps": 1e-8, "weight_decay": 0.,
                             "capturable": True, "fused_cuda": True},
                    "nonfinite_guard": "device_finite_flag_skips_parameters_moments_and_steps",
                },
                "cuda_graph_scope": "world_model_critic_and_unmicrobatched_actor_forward_backward_gradient_clip_and_optimizer_step",
                "cuda_graph_components": {
                    "world_model": "full_update", "critic": "full_update",
                    "actor": "full_update" if (not cfg.actor_microbatch_size
                             or cfg.actor_microbatch_size >= cfg.actor_batch_size) else "microbatch_uncaptured_update",
                    "host": "horizon_gating_acceptance_and_actor_drift_rollback",
                },
                "critic_protocol": {"hidden": list(cfg.critic_hidden), "normalization": "RunningNorm",
                                    "activation": "ELU", "official_value_ceiling_clamp": False},
                "learning_rate_schedule": "constant" if cfg.lr_final_fraction == 1. else "configured_cosine_decay",
                "learning_rates": {"actor": cfg.actor_lr, "world_model": cfg.world_model_lr,
                                   "critic": cfg.critic_lr, "amp_discriminator": cfg.discriminator_lr},
                "lr_final_fraction": cfg.lr_final_fraction,
                "world_model_window": {"reference_default_state_frames": 8,
                                       "predicted_transitions": cfg.wm_window},
            })
            if cfg.training_schedule == SUPERMIMIC_TRAINING_SCHEDULE:
                protocol = metadata["execution_protocol"]
                protocol["training_schedule"] = SUPERMIMIC_TRAINING_SCHEDULE
                protocol["graph_warmup_unit"] = "macro_training_iterations"
                protocol["graph_warmup_updates"] = {
                    "world_model": cfg.graph_warmup*cfg.wm_updates,
                    "actor": cfg.graph_warmup*cfg.actor_updates,
                    "critic": cfg.graph_warmup*cfg.actor_updates*cfg.critic_epochs*cfg.critic_minibatches,
                }
                protocol["cuda_graph_pool"] = "per_runner_shared_stream_pool_ordered_world_model_actor_critic"
                protocol["optimizer_protocol"]["nonfinite_guard"] = "disabled_official_unconditional_step"
                protocol["cuda_graph_components"]["host"] = "fixed_h_unconditional_updates_without_quality_gate_or_actor_rollback"
                protocol["critic_protocol"] = {
                    "hidden": list(cfg.critic_hidden), "normalization": "LayerNorm_per_hidden",
                    "input_normalization": False, "activation": "GELU",
                    "layer_norm_eps": 1e-5, "layer_norm_affine": True,
                    "gelu_approximate": "none",
                    "official_value_ceiling_clamp": True,
                    "value_ceiling": "(task_weight*reward_scale*sum_positive_reward_weights+style_weight*amp_reward_coef)/(1-gamma)",
                    "clamp_sites": ["actor_bootstrap_value", "completed_gae_returns"],
                    "lower_value_clamp": False,
                }
                protocol["update_schedule"] = {
                    "model_warmup_iterations": 0, "critic_warmup_iterations": 0,
                    "fixed_imagination_horizon": cfg.imagination_horizon,
                    "validation_role": "diagnostics_only_not_training_gate",
                    "actor_quality_gate": False, "actor_drift_rollback": False,
                    "reference_replay_ratio": 16.,
                    "realized_replay_ratio": cfg.utd*cfg.wm_updates*cfg.wm_batch_size*(cfg.wm_window+1)/cfg.num_envs,
                    "replay_ratio_state_frames": cfg.wm_window+1,
                    "collection": "advance_real_step_counter_to_next_reference_replay_ratio_event",
                }
    if cfg.motion_scope=="forward_walk_turn_v1":
        metadata["motion_scope"]=cfg.motion_scope
        metadata["termination"]="immediate_fall_no_recovery"
        metadata["terrain"]["motion_profile"]="walking_only_v1"
        metadata["commands"]={"forward_speed_m_s":[.2,.7],"lateral_speed_m_s":[0.,0.],
                              "yaw_rate_rad_s":[-1.,1.],"in_place_turn_fraction":.2}
        metadata["amp"]["expert_reference"]="Filtered Hiking and labeled GRAIL terrain motions plus forward-walk/turn supplements"
    if cfg.sensor_protocol in DEPTH_ONLY_SENSOR_PROTOCOLS:
        from src.tasks.amp_perceptive.sensors import camera_protocol_metadata
        metadata["architecture_version"] = 2
        metadata["sensor_protocol"] = cfg.sensor_protocol
        metadata["camera"] = _json_value(camera_protocol_metadata(cfg.sensor_protocol), "depth camera protocol")
        metadata["visual_protocol"]["encoding"] = "crop_then_depth_blur_then_clip_normalize_then_joint_depth_only_stack_cnn"
        metadata["visual_protocol"]["validity"] = "sensor_diagnostics_only_never_actor_encoder_decoder_or_world_model_input"
        metadata["latent_decoder"]["objective"] = "processed_depth_only_stack_l1"
        if "execution_protocol" in metadata and cfg.replay_storage == "frame_uint8":
            metadata["execution_protocol"]["replay"].pop("valid", None)
            metadata["execution_protocol"]["replay"]["channels_per_frame"] = 1
    if cfg.motion_data_protocol == HIKING_MOTION_PROTOCOL:
        from src.tasks.amp_perceptive.mdp import hiking_reset_protocol_metadata
        metadata["architecture_version"] = 3
        metadata["motion_data_protocol"] = cfg.motion_data_protocol
        metadata["reset"] = _json_value(hiking_reset_protocol_metadata(), "Hiking reset protocol")
        metadata["amp"] = {
            "schema": "hiking_v1", "frame_dim": 67, "history_length": 10,
            "input_dim": 670, "root_frame": "torso_link",
            "fields": ["projected_gravity", "joint_pos_relative_default", "joint_vel_times_0.05",
                       "base_lin_vel_body", "base_ang_vel_body"],
            "flatten_order": "term_then_oldest_to_newest_then_component",
            "expert_reference": "Complete official Hiking walking YAML-selected NPZ plus sagittal symmetry",
            "frame_filter": None, "height_normalization": None,
            "source_frames": 18982, "raw_reference_seconds": 379.62,
            "loaded_frames": 18981, "loaded_reference_seconds": 379.60,
            "interpolation": "author_bilinear_exclusive_timeline_50Hz",
            "normalization": "none",
            "style_reward": "task_reward + amp_reward_coef * max(0,1-0.25*(D-1)^2)",
        }
        metadata["discriminator"]["input_dim"] = 670
        metadata["discriminator"].update({
            "activation": "ReLU", "normalization": "none",
            "optimizer": "AdamW", "optimizer_weight_decay": .01,
            "parameter_l2_coef": cfg.discriminator_trunk_weight_decay,
            "logit_weight_l2_coef": cfg.discriminator_head_weight_decay,
            "gradient_penalty_samples": "concatenated_policy_and_expert",
            "gradient_clip": None,
            "update_budget": "WMAC discriminator_updates x discriminator_batch_size; not the PPO rollout budget",
        })
        if "execution_protocol" in metadata:
            metadata["execution_protocol"]["replay"]["amp_history"] = "reset_safe_time_shared_single_frames"
    if cfg.locomotion_objective in (HIKING_STANDING_OBJECTIVE,HIKING_FULL_OBJECTIVE):
        metadata["architecture_version"] = 4
        metadata["locomotion_objective"] = HIKING_STANDING_OBJECTIVE
        metadata["commands"].update({
            "standing_probability_per_resample": .05,
            "in_place_turn_fraction_conditional_on_not_standing": .2,
            "resampling_time_range_seconds": [8., 12.],
            "standing_command": [0., 0., 0.],
            "standing_state": "explicit_command_standing_bool_in_replay_and_all_imagination_steps",
            "resampling": "publish_new_command_before_observation_and_keep_standing_until_timer_expires",
        })
        metadata["standing_reward"] = {
            "source": "Hiking InstinctLab de6b92f parkour/mdp/rewards.py stand_still",
            "weight": -.3, "offset": 4., "threshold": .15,
            "raw": "(sum(abs(q-q_default))-4)*I(norm(command_xy)<.15)*I(abs(command_yaw)<.15)",
            "reward_scale": "control_dt_once_in_real_and_imagined_rewards",
            "maximum_unscaled_contribution": 1.2,
        }
        if cfg.standing_expert_source != "none":
            metadata["standing_expert_source"] = cfg.standing_expert_source
            metadata["amp"]["expert_reference"] += "; user-requested legacy G1 locked standing reference plus mirror"
            metadata["amp"]["standing_supplement"] = {
                "protocol": cfg.standing_expert_source,
                "source": "stand_still_locked_001.csv",
                "source_frame_count": 1201,
                "source_frame_rate": "unrecorded",
                "representation": "all source rows re-expressed at 50Hz; static posture, not measured 24-second motion",
                "sampling": "uniform complete ten-frame windows across all separate clips; no extra standing weight",
            }
    if cfg.locomotion_objective==HIKING_FULL_OBJECTIVE:
        from src.tasks.amp_perceptive.hiking_rewards import HIKING_AUX_NAMES,HIKING_REWARD_WEIGHTS
        metadata.update(architecture_version=5,locomotion_objective=HIKING_FULL_OBJECTIVE,context_dim=100)
        metadata["proprioception"]={"history_length":8,"frame_width":96,"history_order":"term",
            "angular_velocity_scale":.25,"joint_velocity_scale":.05,"base_frame":"torso_link"}
        metadata["task_rewards"]={"reference":"InstinctLab de6b92f plain G1", "weights":HIKING_REWARD_WEIGHTS,
            "auxiliary_labels":list(HIKING_AUX_NAMES),"episode_seconds":20.,"predicted_timer_clipping":False,
            "same_real_and_imagined_formulas":True}
        metadata["world_model"].update(contact_outputs=13,auxiliary_outputs=list(HIKING_AUX_NAMES),
            input_normalizer_dim=278,input_dim=278+cfg.latent_dim,output_dim=35+13+cfg.latent_dim,
            shared_mlp_outputs={"physical_velocity_increment":35,"contact":13,"latent_delta":cfg.latent_dim})
        metadata["context_layout"]={"legacy_physics": [0,33],"delay_initialized":[33,34],
            "padding":[34,35],"env_origin":[35,38],"previous_action":[38,67],
            "joint_delay_substeps":[67,96],"foot_air_contact_timers":[96,100]}
        metadata["commands"].update(standing_probability_per_resample=.05,
            in_place_turn_fraction_conditional_on_not_standing=None,
            in_place_turn_source="perlin_rough_stand random-velocity terrain",
            resampling="source timer only; no early target-completion resampling",
            allocation="terrain-specific source speeds; user-requested forward-only walking")
        metadata["commands"].update(forward_speed_m_s=[0.,1.],in_place_turn_fraction=None,
            traversal_speed_by_terrain={"perlin_rough":[.45,1.],"other_traversal_terrain":[.45,.8]},
            random_terrain_yaw_deadband=.5,random_terrain_overrides_standing_mask=True)
        metadata["visual_protocol"]["capture_timing"]="post-step fresh sense; pre-reset terminal frame captured separately"
        metadata["terrain"].update(profile="full_Hiking_10_families",source_horizontal_scale_m=.05,
            motion_profile="hiking_full_v1",
            physical_depth="MuJoCo heightfields and rotated box primitives",
            edge_penalty="100 points per foot; source sharp mesh-edge cylinders",
            curriculum="source exponential velocity tracking thresholds")
        metadata["actuators"]={"type":"ideal PD", "delay_substeps":[0,2],
            "sampling":"independent at reset for five source actuator groups"}
        metadata["reset"].update(initial_state="source_torso_pose_and_COM_twist_converted_to_pelvis_by_FK_after_joint_sampling",
            torso_height_above_origin_m=.9,local_root_height="derived, not copied from torso height",
            sampling="joint offsets and torso pose/twist retain source ranges")
        metadata["termination"]={"root":"torso_link", "height_origin":"env_origin_z_clamped_max_zero",
            "minimum_height_m":.5,"maximum_tilt_rad":1.,"illegal_torso_contact_force_N":1.,
            "terrain_timeout":"whole terrain map boundary minus2m", "episode_seconds":20.}
        if "critic_protocol" in metadata.get("execution_protocol",{}):
            metadata["execution_protocol"]["critic_protocol"]["value_ceiling"]=(
                "(task_weight*reward_scale*18.2+style_weight*amp_reward_coef)/(1-gamma); source20s airtime bound, no predicted timer clipping")
    if cfg.sensor_protocol==G1_RPL_SENSOR_PROTOCOL:
        from src.tasks.amp_perceptive.sensors import camera_protocol_metadata
        metadata.update(architecture_version=6,sensor_protocol=cfg.sensor_protocol)
        metadata["camera"]=_json_value(camera_protocol_metadata(cfg.sensor_protocol),"G1 camera protocol")
        metadata["visual_protocol"].update(encoding="physical_RPL_packet_then_joint_8_frame_depth_CNN",
            capture_timing="10Hz software sample; two capture ticks synthetic delay; no additional policy step delay",
            value_range="centered_depth_with_bicubic_overshoot; not_clipped_to_0_1")
        metadata["latent_decoder"]["objective"]="centered_processed_depth_stack_l1"
        metadata["latent_decoder"]["output_activation"]="linear"
        metadata["context_layout"].pop("padding",None)
        metadata["context_layout"]["camera_control_tick_mod5"]=[34,35]
        metadata["camera"]["capture_clock_input"]="existing WM context slot34; advanced modulo5; not a gait phase or Actor input"
        if "execution_protocol" in metadata:
            metadata["execution_protocol"]["replay"].update(storage="frame_fp16",
                storage_protocol="centered_frame_fp16_v1",
                channels_per_frame=1,frames_per_transition=2,
                terminal_reset_semantics="distinct_pre_reset_successor_and_post_reset_current_frames")
    if cfg.sensor_protocol==G1_HIKING_DR_SENSOR_PROTOCOL:
        from src.tasks.amp_perceptive.sensors import camera_protocol_metadata
        metadata.update(architecture_version=7,sensor_protocol=cfg.sensor_protocol)
        metadata["camera"]=_json_value(camera_protocol_metadata(cfg.sensor_protocol),"Hiking calibrated camera DR")
        metadata["visual_protocol"].update(
            encoding="Hiking_crop_blur_clip_normalize_with_adapted_ROI_pixel_randomization",
            capture_timing="50Hz_physical_capture_then_original_one_control_step_history_delay",
            value_range="0_to_1",randomization="one_physical_draw_shared_by_Actor_and_WM")
        metadata["latent_decoder"].update(output_activation="sigmoid",objective="processed_depth_only_stack_l1")
    if cfg.hiking_training_version>=2:
        from .hiking_materials import hiking_material_geom_names
        names=(hiking_material_geom_names(cfg.hiking_training_version)
               if cfg.hiking_training_version >= 3 else hiking_material_geom_names())
        width=100+len(names)
        metadata.update(architecture_version=8,hiking_training_version=cfg.hiking_training_version,
                        context_dim=width)
        metadata["context_layout"]["applied_collision_sliding_friction"]=[100,width]
        metadata["material_conditioning"]={
            "coefficient":"actual_MuJoCo_geom_friction[:,geom,0]",
            "geom_names":list(names),"ordering":"lexicographic_collision_name",
            "scope":"all_active_robot_collision_geoms", "actor_input":False,
            "legacy_mean_friction_retained":True}
        physical=68+29+width+81
        metadata["world_model"].update(input_normalizer_dim=physical,input_dim=physical+cfg.latent_dim,
            normalization="continuous_empirical_statistics_no_sample_limit")
        metadata["critic"]["input_dim"]=136+(width-35)+81
        metadata["termination"]["imagination"]="continuous_soft_OR_of_Hiking_threshold_probabilities;no_STE"
        metadata["amp"]["training_budget"]={"real_rollout_steps":24,"epochs":5,"minibatches":4,
            "policy_window_coverage":"each_accepted_collected_window_once_per_epoch_all_environments",
            "reference_pairing":"sampled_once_per_collected_window_reused_across_epochs",
            "optimizer_step":"one_per_logical_minibatch;weighted_microbatch_accumulation",
            "resume":"pending_rollout_restarts_with_fresh_unsaved_Replay"}
        if cfg.replay_storage=="frame_uint8":
            metadata["execution_protocol"]["replay"].update(
                storage_protocol="hiking_lossless_frame_uint8_v3",context_dim=width,
                fixed_context_columns=65+len(names),dynamic_context_columns=35)
    if cfg.hiking_training_version >= 3:
        metadata["architecture_version"] = 9
        metadata["commands"]["allocation"] = "Hiking terrain-specific command ranges, standing mask and yaw deadband"
        metadata["robot_variant"] = "InstinctLab G1ParkourRoughEnvCfg (unshod)"
        metadata["amp"].update(expert_reference="complete official Hiking walking sequence",
            expert_sampling="per-environment reference cursor, start in [0,.9] of duration, episode symmetry",
            update_budget="24 fresh vector steps, five epochs, four minibatches")
        if cfg.walking_expert_source == LEGACY_WALKING_EXPERT:
            metadata["walking_expert_source"] = cfg.walking_expert_source
            metadata["amp"].update(
                expert_reference="complete official Hiking sequence plus pre-Hiking local walking and turns",
                expert_sampling="uniform contiguous ten-frame windows across the shared mirrored expert pool",
                frame_filter="historical segment selection for local clips; complete official sequence",
            )
        metadata["world_model"].update(input_normalizer_dim=97,
            normalization="empirical represented state and processed joint-position action; friction and actuator-delay context range-scaled",
            root_height="world_z", action_input="joint_position_target")
        metadata["critic"]["input_dim"] = 68 + 3 + width + 81
        metadata["termination"]["imagination"] = "sigmoid of worst normalized Hiking threshold violation, softness 0.05"
        metadata["execution_protocol"]["clock"] = "vector control steps, replay_ratio=16 by default"
        metadata["execution_protocol"]["critic_protocol"]["value_ceiling"] = (
            "(task_weight*dt*sum_positive_task_weights+style_weight*amp_reward_coef)/(1-gamma)")
        metadata["latent_decoder"]["output_activation"] = "sigmoid"
    return metadata


def build_perceptive_actor(payload, *, spec: PerceptionSpec | None = None, device="cpu"):
    """Construct the saved visual Actor; callers load its full state next.

    The online depth encoder is part of the Actor checkpoint. Evaluation and
    playback require neither the world model nor its training-only decoder.
    """
    cfg = config_from_payload(payload)
    spec = perception_spec_from_config(cfg) if spec is None else spec
    if cfg.visual_mode == LATENT_VISUAL_MODE:
        from .latent_models import LatentPerceptiveActor
        actor = LatentPerceptiveActor(spec=spec, latent_dim=cfg.latent_dim,
                                     gate_hidden=cfg.moe_gate_hidden, actor_type=cfg.actor_type)
    else:
        from .models import PerceptiveActor
        actor = PerceptiveActor(spec,gate_hidden=cfg.moe_gate_hidden)
    return actor.to(device)


def validate_checkpoint(payload, cfg, *, terrain_fingerprint=None, motion_fingerprint=None) -> dict:
    """Reject incompatible behavior or data before any runtime state changes.

    UTD, seed, update batches, rewards, AMP, terrain settings and recovery
    semantics are deliberately strict. Logging paths and iteration caps can
    change. Moving a dataset is permitted only if its supplied manifest
    fingerprint still matches. Pass the current physical/data fingerprints
    when restoring a runner; omitting them only validates saved provenance.
    """
    if not isinstance(payload, Mapping) or payload.get("format") != FORMAT:
        raise ValueError("Perceptive resume requires amp-wmac-perceptive-v1; flat WMAC/PPO checkpoints are incompatible")
    cfg.validate()
    saved = saved_training_config(payload.get("config"))
    expected_cfg = cfg
    if cfg.migrate_world_model:
        if (saved["dynamics_parameterization"] != LEGACY_DYNAMICS
                or cfg.dynamics_parameterization != GENCOORD_DYNAMICS):
            raise ValueError("Explicit world-model migration only supports independent_residual_v1 -> supermimic_gencoord_v1")
        # Validate the complete source architecture. Only the physical model
        # is replaced; this flag must not authorize changing the Actor camera,
        # motion task, terrain, expert data or other network dimensions.
        expected_cfg = replace(cfg, migrate_world_model=False,
            **{name: saved[name] for name in DYNAMICS_FIELDS})
    expected = architecture_metadata(expected_cfg)
    if "architecture" not in payload or _json_value(payload["architecture"], "Checkpoint architecture") != expected:
        raise ValueError("Perceptive checkpoint architecture, visual spec or terrain-height semantics differ")
    if "visual_spec" in payload:
        visual = payload["visual_spec"]
        if not isinstance(visual, Mapping):
            raise ValueError("Checkpoint visual_spec must be a complete sensor schema")
        visual = dict(visual)
        visual.setdefault("depth_protocol","hiking_v1")
        if expected_cfg.sensor_protocol == LEGACY_SENSOR_PROTOCOL:
            visual.setdefault("include_valid", True)
        if _json_value(visual, "Checkpoint visual_spec") != _json_value(
                asdict(perception_spec_from_config(expected_cfg)), "Expected visual_spec"):
            raise ValueError("Checkpoint visual_spec differs from sensor_protocol and architecture")
    requested = _config_values(cfg)
    immutable = set(requested)-MUTABLE_RESUME_FIELDS
    missing = immutable-set(saved)
    if missing:
        raise ValueError(f"Checkpoint is missing strict training configuration fields: {sorted(missing)}")
    conflicts = {name for name in immutable
                 if _json_value(saved[name], f"Saved config {name}") != _json_value(requested[name], f"Current config {name}")}
    migration = {}
    if cfg.migrate_world_model:
        unsupported = conflicts-DYNAMICS_FIELDS
        if unsupported:
            raise ValueError(f"World-model migration changes unsupported strict training configuration: {sorted(unsupported)}")
        migration = {"world_model_migration": {
            "from_parameterization": LEGACY_DYNAMICS,
            "to_parameterization": GENCOORD_DYNAMICS,
            "changes": {name: {"before": _json_value(saved[name], f"Saved config {name}"),
                               "after": _json_value(requested[name], f"Current config {name}")}
                        for name in sorted(conflicts)},
        }}
    elif cfg.migrate_perception_training:
        if saved["perception_training_version"] != 1 or cfg.perception_training_version != 2:
            raise ValueError("Explicit perception training migration only supports version 1 -> 2")
        unsupported = conflicts-PERCEPTION_MIGRATION_FIELDS
        if unsupported:
            raise ValueError(f"Perception migration changes unsupported strict training configuration: {sorted(unsupported)}")
        migration = {
            "from_version": 1, "to_version": 2,
            "changes": {name: {"before": _json_value(saved[name], f"Saved config {name}"),
                               "after": _json_value(requested[name], f"Current config {name}")}
                        for name in sorted(conflicts)},
        }
    elif conflicts:
        raise ValueError(f"Perceptive resume changes strict training configuration: {sorted(conflicts)}")
    for key, actual in (("terrain_fingerprint", terrain_fingerprint),
                        ("motion_manifest_sha256", motion_fingerprint)):
        recorded = payload.get(key)
        if not recorded:
            raise ValueError(f"Perceptive checkpoint is missing {key}")
        normalized = _json_value(recorded, f"Checkpoint {key}")
        if actual is not None and normalized != _json_value(actual, f"Current {key}"):
            raise ValueError(f"Perceptive checkpoint {key} differs from the current environment/data")
    return migration


def validate_module_states(payload, modules: Mapping[str, torch.nn.Module]) -> None:
    """Validate every named module's full tensor tree without loading any one.

    Device placement may change; dtype and tensor layout may not. This avoids
    PyTorch's usual partial mutation when a late load_state_dict shape fails.
    Optimizer/RNG loading belongs to the caller after all preflight guards.
    """
    if not isinstance(payload, Mapping) or not isinstance(modules, Mapping):
        raise ValueError("Expected checkpoint and module mappings")
    for name, module in modules.items():
        saved = payload.get(name)
        if not isinstance(saved, Mapping):
            raise ValueError(f"Checkpoint has no state mapping for module {name!r}")
        expected = module.state_dict()
        if set(saved) != set(expected):
            raise ValueError(f"Checkpoint module {name!r} tensor keys differ")
        for key, target in expected.items():
            value = saved[key]
            label = f"{name}.{key}"
            if not isinstance(value, torch.Tensor) or not isinstance(target, torch.Tensor):
                raise ValueError(f"Checkpoint tensor {label!r} is not a tensor")
            if value.shape != target.shape:
                raise ValueError(f"Checkpoint tensor {label!r} shape differs: {tuple(value.shape)} != {tuple(target.shape)}")
            if value.dtype != target.dtype:
                raise ValueError(f"Checkpoint tensor {label!r} dtype differs: {value.dtype} != {target.dtype}")
            if value.layout != target.layout:
                raise ValueError(f"Checkpoint tensor {label!r} layout differs")
            if (value.is_floating_point() or value.is_complex()) and not bool(torch.isfinite(value).all()):
                raise ValueError(f"Checkpoint tensor {label!r} is non-finite")


def _finite_scalar(value, label, *, minimum=None, maximum=None):
    if isinstance(value, torch.Tensor):
        if value.layout != torch.strided or value.numel() != 1 or value.is_complex() or value.dtype == torch.bool:
            raise ValueError(f"{label} must be one real scalar")
        number = float(value.detach().item())
    elif isinstance(value, Real) and not isinstance(value, bool):
        number = float(value)
    else:
        raise ValueError(f"{label} must be one real scalar")
    if not math.isfinite(number) or (minimum is not None and number < minimum) or (maximum is not None and number >= maximum):
        raise ValueError(f"{label} is outside its finite legal range")
    return number


def validate_optimizer_states(payload, optimizers: Mapping[str, torch.optim.Optimizer]) -> None:
    """Preflight optimizer moments and ordered parameter mappings, without load.

    New-format checkpoints use PyTorch's ordered integer parameter IDs. Their
    ordered group lists must match the freshly constructed optimizer exactly;
    silently swapping moments between equal-shaped parameters is not allowed.
    State tensor device placement may change, while shapes/dtypes stay exact.
    The caller can disable ``fused`` after loading for a CUDA-to-CPU restore.
    """
    if not isinstance(payload, Mapping) or not isinstance(optimizers, Mapping):
        raise ValueError("Expected checkpoint and optimizer mappings")
    for name, optimizer in optimizers.items():
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise ValueError(f"Expected a PyTorch optimizer for {name!r}")
        saved = payload.get(name)
        if not isinstance(saved, Mapping) or not isinstance(saved.get("state"), Mapping):
            raise ValueError(f"Checkpoint has no optimizer state mapping for {name!r}")
        if not isinstance(optimizer, (torch.optim.Adam, torch.optim.AdamW, torch.optim.SGD)):
            from .optimizers import ReferenceMuonAdam, validate_reference_optimizer_state
            if isinstance(optimizer, ReferenceMuonAdam):
                validate_reference_optimizer_state(optimizer, saved, label=f"Optimizer {name!r}")
                continue
            raise ValueError(f"Optimizer {name!r} has no supported strict state protocol")
        if "algorithm_name" in saved:
            raise ValueError(f"Optimizer {name!r} algorithm marker differs from the native optimizer protocol")
        groups = saved.get("param_groups")
        if not isinstance(groups, (list, tuple)) or len(groups) != len(optimizer.param_groups):
            raise ValueError(f"Optimizer {name!r} parameter group count differs")
        current = optimizer.state_dict()
        parameters = {}
        parameter_objects = set()
        is_adam = isinstance(optimizer, (torch.optim.Adam, torch.optim.AdamW))
        is_sgd = isinstance(optimizer, torch.optim.SGD)
        if not (is_adam or is_sgd):
            raise ValueError(f"Optimizer {name!r} has no supported strict state protocol")
        for group_index, (group, current_group, runtime_group) in enumerate(zip(groups, current["param_groups"], optimizer.param_groups)):
            label = f"Optimizer {name!r} group {group_index}"
            if not isinstance(group, Mapping) or not isinstance(group.get("params"), (list, tuple)):
                raise ValueError(f"{label} has no ordered parameter IDs")
            identifiers = group["params"]
            if (any(isinstance(identifier, bool) or not isinstance(identifier, int) or identifier < 0
                    for identifier in identifiers) or list(identifiers) != list(current_group["params"])):
                raise ValueError(f"{label} ordered parameter mapping differs")
            if len(identifiers) != len(runtime_group["params"]):
                raise ValueError(f"{label} parameter count differs")
            for key in ("name", "param_names"):
                if key in current_group and _json_value(group.get(key), f"{label} {key}") != _json_value(current_group[key], f"Current {label} {key}"):
                    raise ValueError(f"{label} named parameter mapping differs")
            required = {"lr"}
            if is_adam:
                required |= {"betas", "eps", "weight_decay"}
            if is_sgd:
                required |= {"momentum", "dampening", "weight_decay", "nesterov"}
            if required-set(group):
                raise ValueError(f"{label} is missing optimizer hyperparameters: {sorted(required-set(group))}")
            for key in ("lr", "initial_lr", "eps", "weight_decay", "momentum", "dampening"):
                if key in group:
                    _finite_scalar(group[key], f"{label} {key}", minimum=0.)
            if "betas" in group:
                betas = group["betas"]
                if not isinstance(betas, (list, tuple)) or len(betas) != 2:
                    raise ValueError(f"{label} betas must contain two scalars")
                for index, beta in enumerate(betas):
                    _finite_scalar(beta, f"{label} beta{index}", minimum=0., maximum=1.)
            for key in ("maximize", "amsgrad", "nesterov", "capturable", "differentiable", "decoupled_weight_decay"):
                if key in group and not isinstance(group[key], bool):
                    raise ValueError(f"{label} {key} must be boolean")
            for key in ("foreach", "fused"):
                if key in group and group[key] is not None and not isinstance(group[key], bool):
                    raise ValueError(f"{label} {key} must be boolean or None")
            if group.get("nesterov", False) and (float(group.get("momentum", 0.)) <= 0 or float(group.get("dampening", 0.)) != 0):
                raise ValueError(f"{label} Nesterov requires positive momentum and zero dampening")
            for identifier, parameter in zip(identifiers, runtime_group["params"]):
                if identifier in parameters or id(parameter) in parameter_objects:
                    raise ValueError(f"{label} repeats an optimizer parameter")
                if not isinstance(parameter, torch.Tensor):
                    raise ValueError(f"{label} runtime parameter is not a tensor")
                parameters[identifier] = (parameter, group)
                parameter_objects.add(id(parameter))
        for identifier, state in saved["state"].items():
            if isinstance(identifier, bool) or not isinstance(identifier, int) or identifier not in parameters:
                raise ValueError(f"Optimizer {name!r} state refers to an unknown parameter ID")
            if not isinstance(state, Mapping):
                raise ValueError(f"Optimizer {name!r} parameter state is not a mapping")
            parameter, group = parameters[identifier]
            if is_adam and state:
                required = {"step", "exp_avg", "exp_avg_sq"}
                if group.get("amsgrad", False): required.add("max_exp_avg_sq")
                if required-set(state):
                    raise ValueError(f"Optimizer {name!r} Adam state is incomplete: {sorted(required-set(state))}")
            for key, value in state.items():
                label = f"Optimizer {name!r} parameter {identifier} state {key!r}"
                if key == "step":
                    _finite_scalar(value, label, minimum=0.)
                    continue
                if not isinstance(value, torch.Tensor):
                    raise ValueError(f"{label} must be a parameter-shaped tensor")
                if value.shape != parameter.shape or value.dtype != parameter.dtype or value.layout != parameter.layout:
                    raise ValueError(f"{label} shape, dtype or layout differs from its parameter")
                if (value.is_floating_point() or value.is_complex()) and not bool(torch.isfinite(value).all()):
                    raise ValueError(f"{label} is non-finite")
                if key in {"exp_avg_sq", "max_exp_avg_sq"} and not value.is_complex() and bool((value < 0).any()):
                    raise ValueError(f"{label} second moment must be nonnegative")
