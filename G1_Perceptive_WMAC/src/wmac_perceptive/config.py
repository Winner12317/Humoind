"""Explicit configuration for the isolated joint terrain/recovery experiment."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from collections.abc import Mapping
from pathlib import Path
import math
from numbers import Real

from src.wmac.config import WmacConfig

TASK = "Unitree-G1-AMP-Perceptive-Hiking"
FORMAT = "amp-wmac-perceptive-v1"
LOCOMOTION_SCOPE = "forward_walk_turn_v1"
LEGACY_SCOPE = "joint_recovery_v1"
GENCOORD_DYNAMICS = "supermimic_gencoord_v1"
LEGACY_DYNAMICS = "independent_residual_v1"
DYNAMICS_FIELDS = frozenset((
    "dynamics_parameterization", "dynamics_dt", "dynamics_decimation",
))
PIXEL_VISUAL_MODE = "pixel_v2"
LATENT_VISUAL_MODE = "instinct_latent_v1"
TRANSFORMER_ACTOR_TYPE = "transformer"
VISUAL_FIELDS = frozenset((
    "visual_mode", "latent_dim", "latent_loss_coef", "latent_recon_coef",
    "latent_recon_images", "latent_stat_rate", "max_validation_latent_error",
    "latent_rollout_error_ratio",
))
LEGACY_EXECUTION_PROFILE = "legacy"
SUPERMIMIC_EXECUTION_PROFILE = "supermimic"
VALIDATED_TRAINING_SCHEDULE = "validated_v2"
SUPERMIMIC_TRAINING_SCHEDULE = "supermimic_v1"
SUPERMIMIC_REFERENCE_COMMIT = "abc124c1bd5976341034668af3d12c087f2f1f55"
LEGACY_SENSOR_PROTOCOL = "local_depth_valid_v1"
HIKING_SENSOR_PROTOCOL = "hiking_depth_only_v1"
G1_HIKING_CALIBRATED_SENSOR_PROTOCOL = "g1_hiking_calibrated_v1"
G1_RPL_SENSOR_PROTOCOL = "g1_rpl_depth_v1"
G1_HIKING_DR_SENSOR_PROTOCOL = "g1_hiking_dr_v1"
HIKING_FIXED_CAMERA_PROTOCOLS = (HIKING_SENSOR_PROTOCOL, G1_HIKING_CALIBRATED_SENSOR_PROTOCOL)
DEPTH_ONLY_SENSOR_PROTOCOLS = (*HIKING_FIXED_CAMERA_PROTOCOLS, G1_RPL_SENSOR_PROTOCOL, G1_HIKING_DR_SENSOR_PROTOCOL)
FILTERED_MOTION_PROTOCOL = "filtered_walk_turn_v2"
HIKING_MOTION_PROTOCOL = "hiking_walk_v1"
LEGACY_LOCOMOTION_OBJECTIVE = "legacy_walk_turn_v1"
HIKING_STANDING_OBJECTIVE = "hiking_standing_v1"
HIKING_FULL_OBJECTIVE = "hiking_full_v1"
# Version 4 returns the task to the original Hiking implementation wherever the
# simulator allows it (see ``HIKING_ALIGNED_VERSION`` uses and README): Hiking's
# nominal pinhole camera, its single reference motion with per-environment
# playback, terrain edges detected on Hiking's vertical-riser terrain mesh, and
# the discriminator regularisation its trainer actually applies. Version 3 and
# earlier checkpoints keep the protocol they were trained with.
# Version 5 is version 4 with the defaults for training a policy for the
# physical G1 (see ``HIKING_REAL_ROBOT_VERSION`` and README, "Real-robot
# deployment"): the depth camera sees Hiking's vertical-riser mesh, collision
# uses the 2.5 cm grid, and each robot's mass distribution and joint dynamics
# are sampled around Unitree's G1 model. Each stays a separate flag, and
# ``--hiking-training-version 4`` selects the plain Hiking-aligned task.
HIKING_TRAINING_VERSION = 5
HIKING_ALIGNED_VERSION = 4
HIKING_REAL_ROBOT_VERSION = 5
# Hiking ships two G1 task variants. "unshod" is G1ParkourRoughEnvCfg: bare
# feet and reset-sampled actuator delay. "shoe" is G1ParkourEnvCfg, the variant
# behind the registered task Instinct-Parkour-Target-Amp-G1-v0: soles 23 mm
# lower (collision, foot volume points, foot-plane offset) and, because that
# config swaps the robot after assigning actuators, no actuator delay.
ROBOT_VARIANTS = ("unshod", "shoe")
# Terrain the depth camera sees. "collision" is the MuJoCo heightfield the feet
# touch, where every Hiking riser is a ramp one collision cell wide.
# "source_mesh" is Hiking's own terrain mesh with vertical risers, added as
# non-colliding geometry; pair it with --collision-horizontal-scale 0.025 so
# what the feet touch stays within 2.5 cm of what the camera sees.
DEPTH_GEOMETRIES = ("collision", "source_mesh")
# "unitree_g1_v1": interpolate/extrapolate between the Hiking robot and unitree_mujoco's G1.
ROBOT_DYNAMICS_RANDOMIZATIONS = ("none", "unitree_g1_v1")
LEGACY_G1_STANDING_EXPERT = "legacy_g1_locked_v1"
LEGACY_WALKING_EXPERT = "pre_hiking_local_walk_turn_v2"
EXECUTION_FIELDS = frozenset((
    "execution_profile", "cuda_graphs", "graph_warmup", "replay_storage", "validation_horizon",
    "execution_optimizer",
    "training_schedule",
))

# Network/sensor architecture is unchanged. This version names the training
# objective and update protocol, which must not change during an ordinary
# checkpoint resume.
PERCEPTION_TRAINING_FIELDS = frozenset((
    "perception_training_version", "depth_mask_loss_coef",
    "imagination_horizon_start", "horizon_growth_updates",
    "depth_rollout_error_ratio", "actor_stall_warning_iterations",
))


def perception_spec_from_config(cfg):
    """Explicit sensor schema for a fresh run or its saved legacy protocol."""
    from .perception import PerceptionSpec
    protocol = getattr(cfg, "sensor_protocol", LEGACY_SENSOR_PROTOCOL)
    if protocol != LEGACY_SENSOR_PROTOCOL and protocol not in DEPTH_ONLY_SENSOR_PROTOCOLS:
        raise ValueError("Unsupported sensor_protocol")
    if protocol == G1_RPL_SENSOR_PROTOCOL:
        return PerceptionSpec(height=27,width=48,ring_length=36,delay=0,max_depth=2.,
            proprio_dim=768 if getattr(cfg,"locomotion_objective",None)==HIKING_FULL_OBJECTIVE else 384,
            crop_top=0,crop_bottom=0,crop_left=0,crop_right=0,blur_kernel=1,
            include_valid=False,depth_protocol="g1_rpl_v1")
    return PerceptionSpec(include_valid=protocol == LEGACY_SENSOR_PROTOCOL,
        proprio_dim=768 if getattr(cfg, "locomotion_objective", None) == HIKING_FULL_OBJECTIVE else 384)


def saved_training_config(saved):
    """Fill only fields introduced after the original v1 training protocol.

    Historical files do not record these fields. They retain the original
    full-image objective, hard depth gate and fixed imagination horizon rather
    than silently receiving new-run defaults. Incomplete v2 records are not
    repaired here; strict resume preflight must reject them.
    """
    if not isinstance(saved, Mapping):
        raise ValueError("Perceptive checkpoint requires its complete saved configuration")
    values = dict(saved)
    # Old weights consume depth+valid and the original endpoint camera rays.
    # Never reinterpret them as the new Hiking depth-only observation.
    values.setdefault("sensor_protocol", LEGACY_SENSOR_PROTOCOL)
    # Old checkpoints keep their filtered expert distribution and motion resets.
    values.setdefault("motion_data_protocol", FILTERED_MOTION_PROTOCOL)
    # Old policies were never trained with sustained zero commands or the
    # Hiking stand_still term. Preserve that objective on ordinary resume.
    values.setdefault("locomotion_objective", LEGACY_LOCOMOTION_OBJECTIVE)
    # Version 2 adds complete material conditioning, soft imagined termination,
    # continuous WM normalization and the original Hiking AMP sample budget.
    # Missing markers describe the historical 100D context and update rules.
    values.setdefault("hiking_training_version", 1)
    # Checkpoints written before the Hiking depth mesh existed saw the collision heightfields.
    values.setdefault("depth_geometry", "collision")
    # Robot-dynamics randomization widens the world-model and critic context;
    # a checkpoint that does not record it was trained without it.
    values.setdefault("robot_dynamics_randomization", "none")
    values.setdefault("standing_expert_source", "none")
    values.setdefault("walking_expert_source", "none")
    if "execution_profile" not in values:
        if (EXECUTION_FIELDS-{"execution_profile"}) & set(values):
            raise ValueError("Perceptive checkpoint is missing its execution_profile marker")
        values.update(execution_profile=LEGACY_EXECUTION_PROFILE, cuda_graphs=False,
                      graph_warmup=3, replay_storage="legacy", validation_horizon=0,
                      execution_optimizer="adam", training_schedule=VALIDATED_TRAINING_SCHEDULE)
    else:
        # The first optimized profile used Adam before Muon was added. Missing
        # this one later field must preserve that algorithm, never adopt the
        # new-run Muon default or relabel existing optimizer moments.
        values.setdefault("execution_optimizer", "adam")
        values.setdefault("training_schedule", VALIDATED_TRAINING_SCHEDULE)
        if EXECUTION_FIELDS-set(values):
            raise ValueError("Perceptive checkpoint is missing its complete execution configuration")
    # Historical policies consume pixel histories. A recorded visual marker
    # must include the complete protocol, even when packed widths coincide.
    if "visual_mode" not in values:
        if (VISUAL_FIELDS-{"visual_mode"}) & set(values):
            raise ValueError("Perceptive checkpoint is missing its visual_mode marker")
        values.update({"visual_mode": PIXEL_VISUAL_MODE, "latent_dim": 128,
                       "latent_loss_coef": 1., "latent_recon_coef": 1.,
                       "latent_recon_images": 512, "latent_stat_rate": .01,
                       "max_validation_latent_error": None, "latent_rollout_error_ratio": 1.5})
    elif VISUAL_FIELDS-set(values):
        raise ValueError("Perceptive checkpoint is missing its complete visual training configuration")
    # A checkpoint predating the locomotion-only task retains its original
    # objective; changing the motion pool is a new experiment, not a resume.
    values.setdefault("motion_scope", LEGACY_SCOPE)
    # Earlier physical models predicted unrelated pose/velocity residuals.
    # The new network has different input/output dimensions and cannot be
    # substituted merely by filling a missing marker with new-run defaults.
    if "dynamics_parameterization" not in values:
        if (DYNAMICS_FIELDS-{"dynamics_parameterization"}) & set(values):
            raise ValueError("Perceptive checkpoint is missing its dynamics_parameterization marker")
        values["dynamics_parameterization"] = LEGACY_DYNAMICS
    if values["dynamics_parameterization"] == LEGACY_DYNAMICS:
        values.setdefault("dynamics_dt", .02)
        values.setdefault("dynamics_decimation", 4)
    elif DYNAMICS_FIELDS-set(values):
        raise ValueError("Perceptive checkpoint is missing its GenCoord integration settings")
    if ("perception_training_version" not in values
            and (PERCEPTION_TRAINING_FIELDS-{"perception_training_version"}) & set(values)):
        raise ValueError("Perceptive checkpoint is missing its perception_training_version marker")
    if values.get("perception_training_version", 1) == 1:
        legacy = {
            "perception_training_version": 1,
            "depth_mask_loss_coef": 1.,
            "imagination_horizon_start": values.get("imagination_horizon", 8),
            "horizon_growth_updates": 100,
            "depth_rollout_error_ratio": 1.5,
            "actor_stall_warning_iterations": 200,
        }
        for name, value in legacy.items():
            values.setdefault(name, value)
    return values


@dataclass
class PerceptiveConfig(WmacConfig):
    task: str = TASK
    profile: str = "ppo_aligned"
    action_limit: float | None = 20.
    num_envs: int = 128
    replay_steps: int = 128
    replay_device: str = "cpu"
    # Select "transformer" for the state-conditioned latent Hiking Actor.
    actor_type: str = "moe"
    moe_variant: str = "hiking"
    amp_feature_schema: str = "grounded_v2"
    actor_batch_size: int = 64
    actor_microbatch_size: int = 16
    wm_batch_size: int = 64
    imagination_horizon: int = 8
    compile: bool = False
    log_root: str = "logs/wmac_perceptive"
    run_name: str = "hiking_forward_walk_turn"
    motion_scope: str = LOCOMOTION_SCOPE
    motion_data_protocol: str = FILTERED_MOTION_PROTOCOL
    locomotion_objective: str = LEGACY_LOCOMOTION_OBJECTIVE
    hiking_training_version: int = 1
    standing_expert_source: str = "none"
    walking_expert_source: str = "none"
    init_discriminator: bool = False
    real_critic_coef: float = 0.
    evaluation_interval: int = 0
    mirrored_motion_root: str = ".perceptive_cache/motions/g1_forward_walk_turn_v2"
    terrain_rows: int = 10
    terrain_cols: int = 20
    collision_horizontal_scale: float = .1
    recovery_flat_fraction: float = 0.
    recovery_fraction: float = 0.
    expert_recovery_fraction: float = 0.
    depth_loss_coef: float = 1.
    perception_training_version: int = 2
    depth_mask_loss_coef: float = 1.
    imagination_horizon_start: int = 1
    horizon_growth_updates: int = 100
    depth_rollout_error_ratio: float = 1.5
    actor_stall_warning_iterations: int = 200
    migrate_perception_training: bool = False
    dynamics_parameterization: str = GENCOORD_DYNAMICS
    # Control-step seconds and physics substeps within that control step.
    # These must match the simulator, not a camera frame/history interval.
    dynamics_dt: float = .02
    dynamics_decimation: int = 4
    migrate_world_model: bool = False
    # Explicit opt-in experiment: historical/new pixel training is unchanged.
    visual_mode: str = PIXEL_VISUAL_MODE
    sensor_protocol: str = LEGACY_SENSOR_PROTOCOL
    latent_dim: int = 128
    latent_loss_coef: float = 1.
    latent_recon_coef: float = 1.
    latent_recon_images: int = 512
    latent_stat_rate: float = .01
    # No uncalibrated absolute veto by default. An explicit value is measured
    # as latent MSE divided by latent variance, never metres.
    max_validation_latent_error: float | None = None
    latent_rollout_error_ratio: float = 1.5
    # Explicit new-run execution defaults; ordinary resume never reapplies
    # them over saved batch sizes, replay representation or rollout semantics.
    execution_profile: str = LEGACY_EXECUTION_PROFILE
    execution_optimizer: str = "adam"
    training_schedule: str = VALIDATED_TRAINING_SCHEDULE
    cuda_graphs: bool = False
    graph_warmup: int = 3
    replay_storage: str = "legacy"
    # Zero retains the historical validation window=wm_window contract.
    validation_horizon: int = 0
    # Historical v1 hard-gate thresholds. V2 retains them as diagnostics only;
    # full-image reconstruction does not veto policy learning.
    max_validation_depth_error: float = .15
    max_validation_mask_error: float = .15
    depth_noise_std_m: float = .005
    depth_dropout_probability: float = .01
    validation_min_recovery_samples: int = 0
    validation_min_walk_samples: int = 8
    ground_height_semantics: str = "per_point_terrain_v1"
    # Opt-in robot dynamics randomization between the Hiking training robot and
    # another G1 model (see src/wmac_perceptive/robot_dynamics.py). The sampled
    # values are appended to the world-model/critic context; "none" reproduces
    # every earlier run and checkpoint.
    robot_dynamics_randomization: str = "none"
    # Which of Hiking's two G1 task variants to simulate (see ROBOT_VARIANTS).
    robot_variant: str = "unshod"
    # What the depth camera's terrain rays hit (see DEPTH_GEOMETRIES): the
    # collision heightfields, or Hiking's own vertical-riser terrain mesh.
    depth_geometry: str = "collision"
    # Device of the simulator (physics, depth rays, environment managers).
    # Empty: the learner's ``device``. Another GPU, e.g. "cuda:1", keeps the
    # simulator's memory off the learner's card (src/wmac/device_placement.py).
    # Placement belongs to the machine, not the experiment: it is never
    # restored from a checkpoint.
    sim_device: str = ""
    # MuJoCo-Warp 3.8.1 drops heightfield contacts unless patched. Training
    # refuses that physics unless this is set; resumes never inherit it.
    allow_unpatched_hfield: bool = False

    @classmethod
    def defaults(cls, execution_profile=LEGACY_EXECUTION_PROFILE, *, hiking_training_version=None):
        # Carry the repaired common AMP/recovery/UTD settings into a new,
        # explicitly smaller sensor-memory budget. No old profile is changed.
        base = asdict(WmacConfig.for_profile("ppo_aligned"))
        specific=cls()
        for field in fields(cls):
            if field.name not in {f.name for f in fields(WmacConfig)}:
                continue
            if field.name in cls.__annotations__:
                base[field.name] = getattr(specific, field.name)
        cfg = cls(**base)
        if execution_profile == SUPERMIMIC_EXECUTION_PROFILE:
            cfg.execution_profile = execution_profile
            cfg.visual_mode = LATENT_VISUAL_MODE
            cfg.motion_data_protocol = HIKING_MOTION_PROTOCOL
            cfg.locomotion_objective = HIKING_FULL_OBJECTIVE
            cfg.hiking_training_version = (HIKING_TRAINING_VERSION if hiking_training_version is None
                                          else hiking_training_version)
            cfg.moe_gate_hidden = ()
            cfg.collision_horizontal_scale = .05
            cfg.standing_expert_source = "none"
            cfg.mirrored_motion_root = ".perceptive_cache/sources/hiking"
            if cfg.hiking_training_version >= 3:
                aligned = cfg.hiking_training_version >= HIKING_ALIGNED_VERSION
                # Version 3 used the measured G1 RealSense intrinsics and added 13
                # local clips to the reference pool; both remain selectable.
                cfg.sensor_protocol = HIKING_SENSOR_PROTOCOL if aligned else G1_HIKING_CALIBRATED_SENSOR_PROTOCOL
                cfg.walking_expert_source = "none" if aligned else LEGACY_WALKING_EXPERT
                if cfg.hiking_training_version >= HIKING_REAL_ROBOT_VERSION:
                    cfg.collision_horizontal_scale = .025
                    cfg.depth_geometry = "source_mesh"
                    cfg.robot_dynamics_randomization = "unitree_g1_v1"
                cfg.action_limit = None
                cfg.validation_env_fraction = 0.
                cfg.iterations = 1_000_000  # Vector control steps.
                cfg.log_interval, cfg.save_interval = 50, 1000
            else:
                cfg.sensor_protocol = G1_HIKING_DR_SENSOR_PROTOCOL
                cfg.standing_expert_source = LEGACY_G1_STANDING_EXPERT
                cfg.mirrored_motion_root = ".perceptive_cache/motions/g1_hiking_walk_stand_v1"
            cfg.amp_feature_schema = "hiking_v1"
            cfg.discriminator_hidden = (1024, 512)
            cfg.discriminator_lr = 1e-4
            cfg.discriminator_optimizer = "adamw"
            cfg.gradient_penalty = 5.
            cfg.amp_reward_coef = .25
            cfg.task_weight = cfg.style_weight = 1.
            cfg.discriminator_trunk_weight_decay = 3e-4
            # Hiking configures 0.04 for the logit layer, but instinct_rl 574c8d5
            # looks the coefficient up under a name that does not exist
            # ("logit_weight_decay_coef") and so applies 1.0; 0.04 only switches
            # the term on. Version 4 reproduces what the trainer executes.
            cfg.discriminator_head_weight_decay = (
                1. if cfg.hiking_training_version >= HIKING_ALIGNED_VERSION else .04)
            cfg.depth_noise_std_m = cfg.depth_dropout_probability = 0.
            cfg.num_envs, cfg.wm_batch_size, cfg.wm_updates = 4096, 2048, 10
            # Official horizon=8 counts state frames, i.e. seven transitions.
            # This task's wm_window counts transitions, so use seven here.
            cfg.wm_window, cfg.actor_batch_size, cfg.actor_microbatch_size = 7, 4096, 0
            cfg.imagination_horizon, cfg.imagination_horizon_start = 32, 1
            cfg.replay_steps, cfg.replay_device = 1000, "cuda:0"
            cfg.replay_storage, cfg.validation_horizon = "frame_uint8", 32
            cfg.compile, cfg.precision = True, "tf32"
            cfg.execution_optimizer = "muon_adam"
            cfg.training_schedule = SUPERMIMIC_TRAINING_SCHEDULE
            cfg.actor_lr, cfg.lr_final_fraction = 2e-4, 1.0
            cfg.critic_lr, cfg.actor_betas = 2e-4, (.9, .95)
            cfg.critic_hidden = (256, 128)
            cfg.cuda_graphs = True
            cfg.imagination_horizon_start = cfg.imagination_horizon
            cfg.model_warmup_iterations = cfg.critic_warmup_iterations = 0
            cfg.warmup_steps, cfg.exploration_noise = 8, .1
        elif execution_profile != LEGACY_EXECUTION_PROFILE:
            raise ValueError("Unknown execution_profile; choose legacy or supermimic")
        return cfg

    def validate(self):
        if (not isinstance(self.hiking_training_version,int) or isinstance(self.hiking_training_version, bool)
                or self.hiking_training_version not in (1, 2, 3, 4, HIKING_TRAINING_VERSION)):
            raise ValueError("Unsupported Hiking training version")
        if self.robot_variant not in ROBOT_VARIANTS:
            raise ValueError(f"robot_variant must be one of {ROBOT_VARIANTS}")
        if self.robot_variant != "unshod" and self.hiking_training_version < HIKING_ALIGNED_VERSION:
            raise ValueError("The shoe robot variant requires Hiking training version 4")
        if self.sim_device and self.sim_device != "cpu" and not self.sim_device.startswith("cuda"):
            raise ValueError("sim_device must be empty, cpu or a CUDA device")
        if self.sim_device and self.sim_device != self.device and self.hiking_training_version < 3:
            raise ValueError("A separate simulator device requires the Hiking task (training version 3 or later)")
        if self.depth_geometry not in DEPTH_GEOMETRIES:
            raise ValueError(f"depth_geometry must be one of {DEPTH_GEOMETRIES}")
        if self.depth_geometry != "collision" and self.hiking_training_version < 3:
            raise ValueError("The Hiking depth mesh requires the Hiking task (training version 3 or later)")
        if self.robot_dynamics_randomization not in ROBOT_DYNAMICS_RANDOMIZATIONS:
            raise ValueError(f"robot_dynamics_randomization must be one of {ROBOT_DYNAMICS_RANDOMIZATIONS}")
        if self.robot_dynamics_randomization != "none" and self.hiking_training_version < 3:
            raise ValueError("Robot dynamics randomization requires the Hiking v3 task")
        if self.hiking_training_version >= 2 and (
                self.execution_profile != "supermimic"
                or self.locomotion_objective != HIKING_FULL_OBJECTIVE
                or self.visual_mode != LATENT_VISUAL_MODE
                or self.dynamics_parameterization != GENCOORD_DYNAMICS
                or self.training_schedule != SUPERMIMIC_TRAINING_SCHEDULE):
            raise ValueError("Hiking training v2 requires the full Hiking latent GenCoord reference schedule")
        if self.hiking_training_version >= 3 and (
                self.sensor_protocol not in HIKING_FIXED_CAMERA_PROTOCOLS or self.standing_expert_source != "none"
                or self.replay_storage != "frame_uint8"):
            raise ValueError("Hiking v3 architecture uses a fixed Hiking depth camera, official walking expert and shared uint8 Replay")
        # Flat WMAC's prohibition remains intact. Validate the common settings
        # on a distinct base-config instance, then enforce this task's contract.
        values = {f.name: getattr(self, f.name) for f in fields(WmacConfig)}
        values["task"] = "Unitree-G1-AMP-Flat-Aligned"
        if self.actor_type == TRANSFORMER_ACTOR_TYPE:
            if self.hiking_training_version < 3 or self.visual_mode != LATENT_VISUAL_MODE:
                raise ValueError("Transformer Actor requires the current Hiking latent training protocol")
            if self.moe_load_balance_coef != 0. or self.moe_router_z_coef != 0.:
                raise ValueError("Transformer Actor has no MoE router auxiliary losses")
            values["actor_type"] = "mlp"
        if self.amp_feature_schema == "hiking_v1":
            # The shared flat validator knows its own schemas; the distinct
            # perceptive Hiking observation/history contract is checked below.
            values["amp_feature_schema"] = "grounded_v2"
        if self.action_limit is None and self.hiking_training_version >= 3:
            # The flat configuration requires a finite clip. Hiking's joint
            # position action does not clip the raw policy output.
            values["action_limit"] = WmacConfig().action_limit
        if self.action_limit is None and self.hiking_training_version < 3:
            raise ValueError("Historical action_limit must be positive")
        WmacConfig(**values).validate()
        if self.task != TASK or self.profile != "ppo_aligned":
            raise ValueError("Use the dedicated perceptive task with the ppo_aligned common settings")
        if self.actor_type != TRANSFORMER_ACTOR_TYPE:
            expected_gate=() if self.locomotion_objective==HIKING_FULL_OBJECTIVE else (256,)
            if self.actor_type != "moe" or self.moe_variant != "hiking" or self.moe_gate_hidden != expected_gate:
                raise ValueError("Perceptive Hiking gate must match its saved task protocol (full source [], historical [256])")
        if self.recovery_counter_semantics != "hard_forward_v2" or self.recovery_height_reward != "upright_v2":
            raise ValueError("Perceptive recovery retains hard_forward_v2/upright_v2")
        if self.ground_height_semantics != "per_point_terrain_v1":
            raise ValueError("Unsupported terrain-relative recovery/AMP semantics")
        if self.motion_data_protocol not in (FILTERED_MOTION_PROTOCOL, HIKING_MOTION_PROTOCOL):
            raise ValueError("Unsupported motion_data_protocol")
        if self.locomotion_objective not in (LEGACY_LOCOMOTION_OBJECTIVE, HIKING_STANDING_OBJECTIVE, HIKING_FULL_OBJECTIVE):
            raise ValueError("Unsupported locomotion_objective")
        if self.locomotion_objective in (HIKING_STANDING_OBJECTIVE, HIKING_FULL_OBJECTIVE) and (
                self.motion_scope != LOCOMOTION_SCOPE or self.motion_data_protocol != HIKING_MOTION_PROTOCOL):
            raise ValueError("Hiking standing objective requires forward walking with official Hiking data")
        if self.locomotion_objective==HIKING_FULL_OBJECTIVE and (
                self.visual_mode!=LATENT_VISUAL_MODE or self.dynamics_parameterization!=GENCOORD_DYNAMICS):
            raise ValueError("The full Hiking task requires Instinct latent vision and GenCoord dynamics")
        if self.standing_expert_source not in ("none", LEGACY_G1_STANDING_EXPERT):
            raise ValueError("Unsupported standing_expert_source")
        if self.standing_expert_source != "none" and self.locomotion_objective not in (HIKING_STANDING_OBJECTIVE, HIKING_FULL_OBJECTIVE):
            raise ValueError("Standing expert supplement requires the Hiking standing objective")
        if self.walking_expert_source not in ("none", LEGACY_WALKING_EXPERT):
            raise ValueError("Unsupported walking_expert_source")
        if self.walking_expert_source != "none" and self.hiking_training_version < 3:
            raise ValueError("The historical walking reference pool requires Hiking v3")
        if self.motion_data_protocol == HIKING_MOTION_PROTOCOL:
            if self.motion_scope != LOCOMOTION_SCOPE or self.amp_feature_schema != "hiking_v1":
                raise ValueError("Official Hiking walking data requires locomotion-only Hiking AMP")
            if self.discriminator_hidden != (1024, 512):
                raise ValueError("Hiking AMP uses discriminator hidden layers (1024, 512)")
        elif self.amp_feature_schema != "grounded_v2":
            raise ValueError("Historical filtered data requires its grounded_v2 AMP schema")
        if self.real_critic_coef != 0 or self.init_discriminator:
            raise ValueError("This isolated version uses imagined value fitting and fresh terrain AMP statistics")
        if self.checkpoint_replay:
            raise ValueError("Perceptive replay is intentionally not saved; collect fresh sensor histories on resume")
        optimized = (self.execution_profile == SUPERMIMIC_EXECUTION_PROFILE
                     and self.visual_mode == LATENT_VISUAL_MODE)
        if self.compile and not optimized:
            raise ValueError("Use --no-compile for the initial terrain/sensor migration; baseline compilation is unchanged")
        if self.evaluation_interval:
            raise ValueError("Use evaluate_wmac_perceptive.py; flat automatic checkpoint evaluation is incompatible")
        if self.terrain_rows < 1 or self.terrain_cols < 1:
            raise ValueError("Terrain dimensions must be positive")
        scale = self.collision_horizontal_scale
        refined=(self.locomotion_objective==HIKING_FULL_OBJECTIVE and math.isclose(scale,.025,abs_tol=1e-10)
                 and self.hiking_training_version >= 3)
        if self.locomotion_objective==HIKING_FULL_OBJECTIVE and not refined and not math.isclose(scale,.05,abs_tol=1e-10):
            raise ValueError("Full Hiking terrain collides on the original .05m grid, or from training version 3 "
                             "on its .025m refinement; choose a historical task for coarse terrain")
        minimum_scale=(.025 if refined else .05) if self.locomotion_objective==HIKING_FULL_OBJECTIVE else .1
        if not refined and (not math.isfinite(scale) or scale < minimum_scale or not math.isclose(scale/.05,round(scale/.05),abs_tol=1e-8) or not math.isclose(8./scale,round(8./scale),abs_tol=1e-8)):
            raise ValueError("Collision grid must be at least 10cm, a finite multiple of 5cm dividing the 8m tile; 5cm overflows G1 fallen-body contacts")
        if self.motion_scope not in (LOCOMOTION_SCOPE,LEGACY_SCOPE):
            raise ValueError("Unknown perceptive motion_scope")
        if self.motion_scope==LOCOMOTION_SCOPE:
            if self.perception_training_version!=2:
                raise ValueError("Locomotion-only training requires the current training protocol")
            if any(value!=0 for value in (self.recovery_fraction,self.expert_recovery_fraction,
                                          self.recovery_flat_fraction,self.validation_min_recovery_samples)):
                raise ValueError("Locomotion-only training does not allocate recovery data, environments or validation")
        elif not 0 < self.recovery_flat_fraction < 1:
            raise ValueError("Historical joint training requires its flat-recovery allocation")
        for name in ("depth_loss_coef", "depth_mask_loss_coef", "max_validation_depth_error", "max_validation_mask_error"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if self.validation_min_walk_samples<1 or (self.motion_scope==LEGACY_SCOPE and self.validation_min_recovery_samples<1):
            raise ValueError("Every active motion domain requires held-out model validation samples")
        if self.replay_device != "cpu" and not optimized:
            raise ValueError("Store raw depth rings on CPU to bound GPU memory")
        if self.hiking_training_version < 3 and not 0<self.validation_env_fraction<1:
            raise ValueError("Perceptive WMAC requires a held-out environment split for both motion domains")
        if not math.isfinite(self.depth_noise_std_m) or self.depth_noise_std_m<0:
            raise ValueError("Depth noise standard deviation must be finite and nonnegative")
        if not 0<=self.depth_dropout_probability<1:
            raise ValueError("Depth dropout probability must lie in [0,1)")
        if (isinstance(self.perception_training_version, bool)
                or not isinstance(self.perception_training_version, int)
                or self.perception_training_version not in (1, 2)):
            raise ValueError("perception_training_version must be 1 (historical) or 2")
        for name in ("imagination_horizon_start", "horizon_growth_updates", "actor_stall_warning_iterations"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (isinstance(self.depth_rollout_error_ratio, bool)
                or not math.isfinite(self.depth_rollout_error_ratio) or self.depth_rollout_error_ratio < 1):
            raise ValueError("depth_rollout_error_ratio must be finite and at least 1")
        if self.perception_training_version == 2:
            if not self.imagination_horizon_start <= self.imagination_horizon:
                raise ValueError("V2 requires imagination_horizon_start <= imagination_horizon")
            if not optimized and self.imagination_horizon > self.wm_window:
                raise ValueError("V2 requires imagination_horizon_start <= imagination_horizon <= wm_window")
        if not isinstance(self.migrate_perception_training, bool):
            raise ValueError("migrate_perception_training must be a boolean")
        if self.migrate_perception_training and self.perception_training_version != 2:
            raise ValueError("Perception training migration targets version 2")
        if self.dynamics_parameterization not in (GENCOORD_DYNAMICS, LEGACY_DYNAMICS):
            raise ValueError("Unsupported physical dynamics_parameterization")
        if (isinstance(self.dynamics_dt, bool) or not isinstance(self.dynamics_dt, Real)
                or not math.isfinite(self.dynamics_dt)
                or self.dynamics_dt <= 0):
            raise ValueError("dynamics_dt must be positive finite control-step seconds")
        if (isinstance(self.dynamics_decimation, bool)
                or not isinstance(self.dynamics_decimation, int) or self.dynamics_decimation < 1):
            raise ValueError("dynamics_decimation must be a positive integer")
        if not isinstance(self.migrate_world_model, bool):
            raise ValueError("migrate_world_model must be a boolean")
        if self.migrate_world_model and self.dynamics_parameterization != GENCOORD_DYNAMICS:
            raise ValueError("World-model migration targets supermimic_gencoord_v1")
        if self.migrate_world_model and self.migrate_perception_training:
            raise ValueError("Migrate perception training and the world model in separate explicit resumes")
        if self.visual_mode not in (PIXEL_VISUAL_MODE, LATENT_VISUAL_MODE):
            raise ValueError("Unsupported visual_mode")
        if self.sensor_protocol != LEGACY_SENSOR_PROTOCOL and self.sensor_protocol not in DEPTH_ONLY_SENSOR_PROTOCOLS:
            raise ValueError("Unsupported sensor_protocol")
        if self.sensor_protocol in DEPTH_ONLY_SENSOR_PROTOCOLS:
            if self.visual_mode != LATENT_VISUAL_MODE:
                raise ValueError("Hiking depth-only sensor_protocol requires Instinct latent vision")
            if self.depth_noise_std_m != 0 or self.depth_dropout_probability != 0:
                raise ValueError("Hiking sensor_protocol does not add local depth noise or dropout")
        if self.sensor_protocol == G1_RPL_SENSOR_PROTOCOL:
            if self.execution_profile != SUPERMIMIC_EXECUTION_PROFILE or self.replay_storage != "frame_fp16":
                raise ValueError("G1 RPL depth requires SuperMimic execution and signed FP16 frame replay")
            if self.locomotion_objective!=HIKING_FULL_OBJECTIVE:
                raise ValueError("G1 RPL depth uses the full Hiking task's existing 100D context")
        if self.sensor_protocol==G1_HIKING_DR_SENSOR_PROTOCOL:
            if self.execution_profile!=SUPERMIMIC_EXECUTION_PROFILE or self.replay_storage!="frame_uint8":
                raise ValueError("Hiking camera randomization preserves SuperMimic and uint8 frame replay")
            if self.locomotion_objective!=HIKING_FULL_OBJECTIVE:
                raise ValueError("Hiking camera randomization uses the full Hiking task")
        for name in ("latent_dim", "latent_recon_images"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name, minimum in (("latent_loss_coef", 0.), ("latent_recon_coef", 0.),
                              ("latent_rollout_error_ratio", 1.)):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value)
                    or (value <= minimum if minimum == 0. else value < minimum)):
                raise ValueError(f"{name} must be finite and {'positive' if minimum == 0. else 'at least 1'}")
        if (isinstance(self.latent_stat_rate, bool) or not isinstance(self.latent_stat_rate, Real)
                or not math.isfinite(self.latent_stat_rate) or not 0 < self.latent_stat_rate <= 1):
            raise ValueError("latent_stat_rate must be finite and in (0, 1]")
        if self.max_validation_latent_error is not None:
            value = self.max_validation_latent_error
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
                raise ValueError("max_validation_latent_error must be None or positive finite normalized MSE")
        if self.visual_mode == LATENT_VISUAL_MODE:
            if (self.motion_scope != LOCOMOTION_SCOPE or self.dynamics_parameterization != GENCOORD_DYNAMICS
                    or self.perception_training_version != 2):
                raise ValueError("Instinct-latent training requires forward_walk_turn_v1, GenCoord and perception training v2")
            if self.latent_dim != 128:
                raise ValueError("Instinct-latent training requires the existing 128-dimensional joint-stack encoder")
            if self.init_policy:
                raise ValueError("Instinct-latent training requires a fresh Actor; pixel/flat Actor warmstart is unsupported")
            if self.migrate_world_model or self.migrate_perception_training:
                raise ValueError("Instinct-latent training cannot combine legacy migration flags")
        if self.execution_profile not in (LEGACY_EXECUTION_PROFILE, SUPERMIMIC_EXECUTION_PROFILE):
            raise ValueError("Unsupported execution_profile")
        if self.execution_optimizer not in ("adam", "muon_adam"):
            raise ValueError("Unsupported execution_optimizer")
        if self.training_schedule not in (VALIDATED_TRAINING_SCHEDULE, SUPERMIMIC_TRAINING_SCHEDULE):
            raise ValueError("Unsupported training_schedule")
        if self.training_schedule == SUPERMIMIC_TRAINING_SCHEDULE:
            if not optimized or self.execution_optimizer != "muon_adam":
                raise ValueError("SuperMimic training schedule requires its latent execution and Muon/Adam optimizer")
            if (self.imagination_horizon_start != self.imagination_horizon
                    or self.model_warmup_iterations or self.critic_warmup_iterations):
                raise ValueError("SuperMimic training schedule uses fixed H and no model/critic warmup")
        if not isinstance(self.cuda_graphs, bool):
            raise ValueError("cuda_graphs must be a boolean")
        if isinstance(self.graph_warmup, bool) or not isinstance(self.graph_warmup, int) or self.graph_warmup < 1:
            raise ValueError("graph_warmup must be a positive integer")
        if (isinstance(self.validation_horizon, bool) or not isinstance(self.validation_horizon, int)
                or self.validation_horizon < 0):
            raise ValueError("validation_horizon must be a nonnegative integer")
        if self.replay_storage not in ("legacy", "frame_uint8", "frame_fp16"):
            raise ValueError("Unsupported replay_storage")
        if self.replay_storage=="frame_fp16" and self.sensor_protocol!=G1_RPL_SENSOR_PROTOCOL:
            raise ValueError("Signed FP16 frame replay requires the G1 RPL depth sensor protocol")
        if self.execution_profile == LEGACY_EXECUTION_PROFILE:
            if self.execution_optimizer != "adam":
                raise ValueError("Legacy execution retains Adam; Muon requires a new SuperMimic experiment")
            if self.cuda_graphs or self.replay_storage != "legacy" or self.validation_horizon:
                raise ValueError("Legacy execution retains its replay representation and wm_window validation")
        else:
            if not optimized:
                raise ValueError("SuperMimic execution requires instinct_latent_v1")
            if self.migrate_world_model or self.migrate_perception_training:
                raise ValueError("Execution profiles start fresh; legacy migration flags are unsupported")
            if not self.validation_horizon >= self.imagination_horizon >= self.wm_window:
                raise ValueError("SuperMimic execution requires validation_horizon >= imagination_horizon >= wm_window")
            if self.replay_steps < self.validation_horizon:
                raise ValueError("Replay capacity must cover validation_horizon")
            if (self.training_schedule == VALIDATED_TRAINING_SCHEDULE
                    and self.warmup_steps < self.validation_horizon):
                raise ValueError("Initial warmup_steps must cover validation_horizon for the validated schedule")
            if self.replay_storage in ("frame_uint8","frame_fp16") and not self.replay_device.startswith("cuda"):
                raise ValueError("Shared frame_uint8 replay requires a CUDA replay_device")
        if self.cuda_graphs and (not optimized or not self.compile or not self.device.startswith("cuda")):
            raise ValueError("CUDA graphs require SuperMimic latent execution, CUDA and compiled world-model kernels")

    def motion_root(self, project_root: Path) -> Path:
        value = Path(self.mirrored_motion_root).expanduser()
        return (value if value.is_absolute() else project_root/value).resolve()


def config_from_payload(payload):
    if payload.get("format") != FORMAT:
        raise ValueError("Use the perceptive checkpoint format; flat/PPO resumes cannot infer sensor/terrain semantics")
    saved = saved_training_config(payload.get("config"))
    allowed = {f.name for f in fields(PerceptiveConfig)}
    values = {key:value for key,value in saved.items() if key in allowed}
    defaults = PerceptiveConfig.defaults()
    for key,value in values.items():
        if isinstance(getattr(defaults,key),tuple): value=tuple(value)
        setattr(defaults,key,value)
    defaults.resume = None
    defaults.init_policy = None
    defaults.migrate_perception_training = False
    defaults.migrate_world_model = False
    # Physics checks are rerun for every process; a resume never inherits an override.
    defaults.allow_unpatched_hfield = False
    # Neither does it inherit where the previous process placed its simulator.
    defaults.sim_device = ""
    defaults.validate()
    return defaults
