"""Resolved configuration for model-dominant AMP actor-critic experiments."""

from dataclasses import dataclass
import math


PRESETS = (
    "full",
    "no_direct_style_grad",
    "task_only",
    "no_critic",
    "no_survival_grad",
    "uniform_replay",
    "one_step_wm",
)

PROFILES = ("balanced", "supermimic", "legacy", "ppo_aligned", "supermimic_official")
RECOVERY_COUNTER_SEMANTICS = ("soft_v1", "hard_forward_v2")
RECOVERY_HEIGHT_REWARDS = ("height_only_v1", "upright_v2")


@dataclass
class WmacConfig:
    profile: str = "balanced"
    # Version the reference details independently of the validated AMP method.
    reference_alignment: str = "legacy_v1"
    optimizer_algorithm: str = "adam"
    mirror_walk_motions: bool = False
    motion_cache_root: str = ".wmac_cache/motions/g1_mirrored"
    preset: str = "full"
    seed: int = 42
    task: str = "Unitree-G1-AMP-Flat"
    amp_feature_schema: str = "legacy"
    num_envs: int = 1024
    iterations: int = 320000
    collect_steps: int = 0
    utd: float = 0.4
    warmup_steps: int = 64
    model_warmup_iterations: int = 500
    critic_warmup_iterations: int = 20
    replay_steps: int = 256
    wm_window: int = 8
    wm_updates: int = 4
    wm_batch_size: int = 512
    imagination_horizon: int = 32
    actor_batch_size: int = 512
    actor_updates: int = 1
    critic_epochs: int = 5
    critic_minibatches: int = 4
    discriminator_updates: int = 2
    discriminator_batch_size: int = 512
    actor_lr: float = 2e-4
    actor_betas: tuple[float, float] = (0.9, 0.95)
    policy_output_init_scale: float = 0.01
    policy_output_zero_bias: bool = True
    actor_grad_value_clip: float = 0.1
    critic_lr: float = 2e-4
    world_model_lr: float = 2e-4
    discriminator_lr: float = 1e-4
    discriminator_optimizer: str = "adamw"
    discriminator_trunk_weight_decay: float = 1e-4
    discriminator_head_weight_decay: float = 1e-4
    observation_normalizer_until: float = 2e8
    amp_normalizer_epsilon: float = 1e-2
    amp_normalizer_variance_epsilon: bool = False
    amp_normalizer_clip: float | None = None
    amp_normalizer_initial_count: float = 0.0
    amp_normalizer_until: float | None = 2e8
    amp_normalizer_per_update: bool = False
    lr_decay_iterations: int = 250000
    lr_final_fraction: float = 0.03
    gamma: float = 0.99
    gae_lambda: float = 0.95
    exploration_noise: float = 0.1
    action_limit: float = 20.0
    max_grad_norm: float = 1.0
    task_weight: float = 0.75
    style_weight: float = 0.25
    amp_reward_coef: float = 0.1
    # Selective InstantAMP reward adaptation; the LSGAN discriminator is retained.
    # A labelled mode keeps historical checkpoints on the clipped quadratic.
    amp_reward_mapping: str = "clipped_lsgan_v1"
    amp_score_ema_decay: float = 0.99
    amp_score_scale: float = 1.0
    allow_amp_reward_migration: bool = False
    # Reward-only adaptation: no action rescaling curriculum or new model head.
    # Coefficients are nonnegative costs, subtracted before the existing task
    # dt scaling. They do nothing unless the explicit versioned mode is enabled.
    smooth_reward_mode: str = "none"
    smooth_action_second_coef: float = 0.01
    smooth_joint_velocity_coef: float = 0.00025
    smooth_torque_coef: float = 0.00025
    smooth_power_coef: float = 0.000025
    allow_smooth_reward_migration: bool = False
    # Authorize a separate active-to-active coefficient experiment at resume.
    # This one-shot authorization is never inherited from a checkpoint.
    allow_smooth_reward_retune: bool = False
    # Optional flat-recovery geometry modes use detached or InstantAMP full support gradients.
    imagined_recovery_foot_slip: str = "learned_v1"
    # The InstantAMP mode supplies both geometry parameters at 0.001 m.
    recovery_foot_slip_contact_height_m: float | None = None
    recovery_foot_slip_softness_m: float | None = None
    allow_imagined_foot_slip_migration: bool = False
    gradient_penalty: float = 10.0
    direct_style_grad: bool = True
    critic_bootstrap: bool = True
    survival_grad: bool = True
    # Version the two recovery interventions independently for controlled tests.
    # Checkpoints without these fields must resolve to the historical v1 rules.
    recovery_counter_semantics: str = "hard_forward_v2"
    recovery_height_reward: str = "upright_v2"
    allow_recovery_semantics_migration: bool = False
    recovery_fraction: float = 0.5
    expert_recovery_fraction: float = 0.5
    # Opt-in fixes for abrupt, off-reference get-ups; see src/wmac/recovery_style.py.
    # The defaults are the historical behaviour.
    recovery_value_grad_scale: float = 1.0
    recovery_height_reward_ratio: float | None = None
    recovery_tempo_mode: str = "none"
    recovery_tempo_base_rise_limit: float = 0.9
    recovery_tempo_base_rise_coef: float = 10.0
    recovery_tempo_joint_speed_percentile: float = 99.0
    recovery_tempo_joint_speed_coef: float = 0.02
    discriminator_imagined_fraction: float = 0.0
    discriminator_penalty_points: str = "expert"
    recovery_refresh_seconds: float | None = None
    wm_recovery_fraction: float | None = None
    recovery_model_error_gate: float | None = None
    recovery_max_action_shift: float | None = None
    reference_motion_root: str | None = None
    allow_recovery_style_migration: bool = False
    max_model_loss_for_actor: float = 20.0
    actor_trust_coef: float = 0.0
    real_critic_coef: float = 0.5
    real_critic_batch_size: int = 2048
    max_action_shift: float = 0.25
    replay_device: str = "cuda:0"
    device: str = "cuda:0"
    precision: str = "bf16"
    compile: bool = True
    compile_mode: str = "default"
    compile_fullgraph: bool = False
    reward_chunk_size: int = 32768
    actor_microbatch_size: int = 0
    benchmark_sync: bool = False
    validation_env_fraction: float = 0.1
    validation_interval: int = 10
    validation_batch_size: int = 256
    max_validation_model_error: float = 20.0
    max_validation_contact_error: float = 2.0
    save_interval: int = 2000
    log_interval: int = 50
    evaluation_interval: int = 2000
    evaluation_steps: int = 250
    evaluation_envs: int = 32
    run_name: str = ""
    log_root: str | None = None
    resume: str | None = None
    init_policy: str | None = None
    init_discriminator: bool = True
    checkpoint_replay: bool = False
    model_hidden: tuple[int, ...] = (512, 512, 512)
    # Actor-only intervention: retain the original MLP unless explicitly enabled.
    actor_type: str = "mlp"
    policy_hidden: tuple[int, ...] = (512, 256, 128)
    # Hiking MoE uses the paper's [256] gate and the author's dense expert
    # implementation. Historical checkpoints retain their saved architecture.
    moe_variant: str = "hiking"
    moe_expert_hidden: tuple[int, ...] = (256, 128, 64)
    moe_num_experts: int = 4
    moe_top_k: int = 4
    moe_gate_hidden: tuple[int, ...] = (256,)
    moe_temperature: float = 1.0
    moe_load_balance_coef: float = 0.0
    moe_router_z_coef: float = 0.0
    moe_init_noise_std: float = 0.0
    critic_hidden: tuple[int, ...] = (256, 128)
    discriminator_hidden: tuple[int, ...] = (1024, 512, 256)

    @classmethod
    def for_profile(cls, name: str) -> "WmacConfig":
        """Return defaults before CLI parsing so explicit flags take precedence."""
        if name not in PROFILES:
            raise ValueError(f"Unknown profile {name!r}; choose from {PROFILES}.")
        cfg = cls(profile=name)
        if name == "supermimic":
            cfg.num_envs = 4096
            cfg.replay_steps = 1000
            cfg.model_hidden = (2048, 2048, 2048)
            cfg.wm_updates = 10
            cfg.wm_batch_size = 2048
            cfg.actor_batch_size = 4096
            cfg.policy_hidden = (2048, 1024, 512, 256)
            cfg.precision = "tf32"
            cfg.real_critic_coef = 0.0
        elif name == "supermimic_official":
            # AMP/recovery retain the existing aligned flat task. Only the
            # compatible SuperMimic training details adopt the official code.
            cfg = cls.for_profile("ppo_aligned")
            cfg.profile = name
            cfg.reference_alignment = "official_details_v1"
            cfg.optimizer_algorithm = "muon_adam"
            cfg.mirror_walk_motions = True
            cfg.iterations = 400000  # 1M collection ticks / 2.5 ticks per update.
            cfg.model_hidden = (2048, 2048, 2048)
            cfg.policy_hidden = (2048, 1024, 512, 256)
            cfg.critic_hidden = (256, 128)
            # Official horizon=8 counts stored state frames: seven scored
            # transitions in this repository's state/next_state pair buffer.
            cfg.wm_window = 7
            cfg.wm_updates, cfg.wm_batch_size = 10, 2048
            cfg.actor_batch_size, cfg.actor_updates = 4096, 1
            cfg.imagination_horizon = 32
            cfg.actor_lr = cfg.critic_lr = cfg.world_model_lr = 2e-4
            cfg.actor_betas = (.9, .95)
            cfg.actor_grad_value_clip, cfg.max_grad_norm = .1, 1.
            cfg.policy_output_init_scale, cfg.policy_output_zero_bias = 1., False
            cfg.exploration_noise = .1
            cfg.model_warmup_iterations = cfg.critic_warmup_iterations = 0
            cfg.warmup_steps = 8
            cfg.save_interval, cfg.log_interval = 400, 4
            cfg.real_critic_coef = 0.
        elif name == "legacy":
            cfg.iterations = 100000
            cfg.collect_steps = 8
            cfg.model_warmup_iterations = 100
            cfg.imagination_horizon = 16
            cfg.critic_epochs = 4
            cfg.action_limit = 5.0
            cfg.precision = "float32"
            cfg.compile = False
            cfg.validation_env_fraction = 0.0
            cfg.replay_device = "cpu"
            cfg.save_interval = 100
            cfg.log_interval = 1
        elif name == "ppo_aligned":
            cfg.task = "Unitree-G1-AMP-Flat-Aligned"
            cfg.amp_feature_schema = "grounded_v2"
            cfg.iterations = 320000
            cfg.num_envs = 4096
            # Keep WMAC's collect/update schedule; PPO retains its own 24-step
            # rollout. Outer iterations are not a shared data-budget unit.
            cfg.collect_steps = 0
            cfg.utd = 0.4
            cfg.warmup_steps = 64
            cfg.replay_steps = 1000
            # Keep the world-model supervision and imagination method/scale.
            cfg.model_hidden = (2048, 2048, 2048)
            cfg.wm_updates, cfg.wm_batch_size = 10, 2048
            cfg.actor_batch_size = 4096
            cfg.policy_hidden = cfg.critic_hidden = (512, 256, 128)
            cfg.actor_lr = cfg.critic_lr = cfg.discriminator_lr = 1e-3
            cfg.actor_betas = (0.9, 0.999)
            cfg.policy_output_init_scale, cfg.policy_output_zero_bias = 1.0, False
            cfg.exploration_noise = 0.25
            cfg.discriminator_optimizer = "adam"
            cfg.discriminator_trunk_weight_decay = 1e-3
            cfg.discriminator_head_weight_decay = 1e-1
            # AMP is an extension to SuperMimic. Restore the WMAC replay-based
            # budget instead of copying PPO's epochs/minibatch loop.
            cfg.discriminator_updates = 2
            cfg.discriminator_batch_size = 512
            cfg.observation_normalizer_until = 1e8
            cfg.amp_normalizer_epsilon = 1e-4
            cfg.amp_normalizer_variance_epsilon = True
            cfg.amp_normalizer_clip = 10.0
            cfg.amp_normalizer_initial_count = 1e-4
            cfg.amp_normalizer_until = None
            cfg.amp_normalizer_per_update = True
            cfg.real_critic_coef = 0.0
            cfg.precision = "tf32"
        return cfg

    def resolve_robot_defaults(self) -> None:
        """Point robot-specific defaults that still hold the G1 value at the task's robot.

        Profiles are written for the G1. Selecting another robot's task (for example
        ``--task EngineAI-PM01-AMP-Flat-Aligned``) must not reuse the G1's mirrored
        motion cache; an explicitly chosen cache directory is left alone.
        """
        from src.robot_profiles import G1, profile_for_task
        robot = profile_for_task(self.task)
        if robot is not G1 and self.motion_cache_root == G1.mirrored_motion_cache:
            self.motion_cache_root = robot.mirrored_motion_cache
        if self.reference_motion_root and self.motion_cache_root == robot.mirrored_motion_cache:
            # A mirrored cache belongs to one source directory.
            from pathlib import Path
            self.motion_cache_root += "__" + Path(self.reference_motion_root).expanduser().resolve().name

    @property
    def effective_discriminator_batch_size(self) -> int:
        """Zero requests the PPO minibatch size for the actual physical batch."""
        return self.discriminator_batch_size or max(
            1, self.num_envs * self.collect_steps // self.critic_minibatches)

    def apply_preset(self) -> "WmacConfig":
        """Apply the named intervention after parsing, retaining every other option.

        A preset takes precedence over a conflicting explicit switch. The full
        preset makes no changes, so individual switches also work independently.
        """
        if self.preset not in PRESETS:
            raise ValueError(f"Unknown preset {self.preset!r}; choose from {PRESETS}.")
        if self.preset == "no_direct_style_grad":
            self.direct_style_grad = False
        elif self.preset == "task_only":
            self.task_weight, self.style_weight = 1.0, 0.0
            self.direct_style_grad = False
        elif self.preset == "no_critic":
            self.critic_bootstrap = False
        elif self.preset == "no_survival_grad":
            self.survival_grad = False
        elif self.preset == "uniform_replay":
            self.recovery_fraction = 0.0
        elif self.preset == "one_step_wm":
            self.wm_window = 1
        return self

    def validate(self) -> None:
        if self.profile not in PROFILES:
            raise ValueError(f"Unknown profile: {self.profile!r}")
        if self.preset not in PRESETS:
            raise ValueError(f"Unknown preset: {self.preset!r}")
        if self.reference_alignment not in ("legacy_v1", "official_details_v1"):
            raise ValueError("Unknown SuperMimic reference alignment")
        if self.optimizer_algorithm not in ("adam", "muon_adam"):
            raise ValueError("optimizer_algorithm must be adam or muon_adam")
        from .reference import amp_reward_protocol, imagined_foot_slip_protocol, smooth_reward_protocol
        amp_reward_protocol(self)
        smooth_reward_protocol(self)
        imagined_foot_slip_protocol(self)
        from .recovery_style import recovery_style_protocol
        recovery_style_protocol(self)
        if self.reference_motion_root is not None and (
                not isinstance(self.reference_motion_root, str) or not self.reference_motion_root):
            raise ValueError("reference_motion_root must be None or a nonempty path")
        if not isinstance(self.allow_recovery_style_migration, bool):
            raise ValueError("allow_recovery_style_migration must be a bool")
        if self.allow_recovery_style_migration and not self.resume:
            raise ValueError("Recovery-style migration requires --resume")
        if not isinstance(self.allow_imagined_foot_slip_migration, bool):
            raise ValueError("allow_imagined_foot_slip_migration must be a bool")
        if self.allow_imagined_foot_slip_migration and not self.resume:
            raise ValueError("Imagined foot slip migration requires --resume")
        if not isinstance(self.allow_amp_reward_migration, bool):
            raise ValueError("allow_amp_reward_migration must be a bool")
        if not isinstance(self.allow_smooth_reward_migration, bool):
            raise ValueError("allow_smooth_reward_migration must be a bool")
        if not isinstance(self.allow_smooth_reward_retune, bool):
            raise ValueError("allow_smooth_reward_retune must be a bool")
        if not isinstance(self.mirror_walk_motions, bool) or not isinstance(self.motion_cache_root, str) or not self.motion_cache_root:
            raise ValueError("Motion mirroring requires an explicit bool and nonempty cache path")
        if self.recovery_counter_semantics not in RECOVERY_COUNTER_SEMANTICS:
            raise ValueError(f"recovery_counter_semantics must be one of {RECOVERY_COUNTER_SEMANTICS}.")
        if self.recovery_height_reward not in RECOVERY_HEIGHT_REWARDS:
            raise ValueError(f"recovery_height_reward must be one of {RECOVERY_HEIGHT_REWARDS}.")
        if self.actor_type not in ("mlp", "moe"):
            raise ValueError("actor_type must be mlp or moe.")
        if self.moe_variant not in ("hiking", "legacy"):
            raise ValueError("moe_variant must be hiking or legacy.")
        if not self.moe_expert_hidden or any(isinstance(n, bool) or not isinstance(n, int) or n < 1
                                            for n in self.moe_expert_hidden):
            raise ValueError("moe_expert_hidden must contain positive integer widths.")
        if (isinstance(self.moe_num_experts, bool) or not isinstance(self.moe_num_experts, int)
                or self.moe_num_experts < 2):
            raise ValueError("moe_num_experts must be an integer >= 2.")
        if (isinstance(self.moe_top_k, bool) or not isinstance(self.moe_top_k, int)
                or not 2 <= self.moe_top_k <= self.moe_num_experts):
            raise ValueError("moe_top_k must lie in [2, moe_num_experts]; renormalized Top-1 has no task gradient for the router.")
        if any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in self.moe_gate_hidden):
            raise ValueError("moe_gate_hidden must contain positive integer widths.")
        if not math.isfinite(self.moe_temperature) or self.moe_temperature <= 0:
            raise ValueError("moe_temperature must be finite and positive.")
        for name in ("moe_load_balance_coef", "moe_router_z_coef", "moe_init_noise_std"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and non-negative.")
        if self.actor_type == "moe" and self.moe_variant == "hiking":
            required = {"moe_num_experts": 4, "moe_top_k": 4,
                        "moe_expert_hidden": (256, 128, 64),
                        "moe_temperature": 1.0, "moe_load_balance_coef": 0.0,
                        "moe_router_z_coef": 0.0, "moe_init_noise_std": 0.0}
            for name, expected in required.items():
                if getattr(self, name) != expected:
                    raise ValueError(f"Hiking alignment requires {name}={expected!r}; "
                                     "use moe_variant=legacy for a separate configurable MoE experiment.")
            if tuple(self.moe_gate_hidden) not in ((256,), ()):
                raise ValueError("Hiking gate hidden units must be [256] (paper default) or [] "
                                 "(historical author-code checkpoint).")
        if self.actor_type == "moe" and self.moe_variant == "legacy" and self.compile and self.compile_fullgraph:
            raise ValueError("Sparse MoE uses dynamic expert dispatch; use compile_fullgraph=False or --no-compile. Other WMAC heads can still compile.")
        positive_ints = (
            "num_envs", "iterations", "replay_steps", "wm_window",
            "wm_updates", "wm_batch_size", "imagination_horizon", "actor_batch_size",
            "actor_updates", "critic_epochs", "critic_minibatches",
            "discriminator_updates", "save_interval", "log_interval",
            "validation_interval", "validation_batch_size", "real_critic_batch_size",
            "lr_decay_iterations", "reward_chunk_size", "evaluation_steps", "evaluation_envs",
        )
        for name in positive_ints:
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1.")
        if self.discriminator_batch_size < 0 or (self.discriminator_batch_size == 0 and self.collect_steps < 1):
            raise ValueError("discriminator_batch_size=0 requires a positive fixed collect_steps.")
        if self.model_warmup_iterations < 0:
            raise ValueError("model_warmup_iterations must be non-negative.")
        for name in ("collect_steps", "critic_warmup_iterations", "actor_microbatch_size", "evaluation_interval"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative.")
        if self.warmup_steps < self.wm_window:
            raise ValueError("warmup_steps must cover at least one wm_window.")
        if self.replay_steps < self.wm_window:
            raise ValueError("replay_steps must cover at least one wm_window.")
        if self.critic_minibatches > self.actor_batch_size * self.imagination_horizon:
            raise ValueError("critic_minibatches exceeds the number of imagined samples.")
        for name in ("actor_lr", "critic_lr", "world_model_lr", "discriminator_lr",
                     "action_limit", "max_grad_norm", "max_model_loss_for_actor", "max_action_shift",
                     "utd", "actor_grad_value_clip", "max_validation_model_error", "max_validation_contact_error"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        for name in ("exploration_noise", "task_weight", "style_weight", "amp_reward_coef",
                     "gradient_penalty", "actor_trust_coef", "real_critic_coef",
                     "policy_output_init_scale", "discriminator_trunk_weight_decay",
                     "discriminator_head_weight_decay", "amp_normalizer_initial_count"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative.")
        if self.discriminator_optimizer not in ("adam", "adamw"):
            raise ValueError("discriminator_optimizer must be adam or adamw.")
        for name in ("observation_normalizer_until", "amp_normalizer_epsilon"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        for name in ("amp_normalizer_clip", "amp_normalizer_until"):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be None or finite and positive.")
        if self.task_weight + self.style_weight <= 0:
            raise ValueError("At least one reward weight must be positive.")
        if not 0 <= self.gamma < 1:
            raise ValueError("gamma must be in [0, 1).")
        if not 0 <= self.gae_lambda <= 1:
            raise ValueError("gae_lambda must be in [0, 1].")
        if not 0 <= self.recovery_fraction <= 1:
            raise ValueError("recovery_fraction must be in [0, 1].")
        if not 0 <= self.expert_recovery_fraction <= 1:
            raise ValueError("expert_recovery_fraction must be in [0, 1].")
        if not 0 <= self.validation_env_fraction < 1:
            raise ValueError("validation_env_fraction must be in [0, 1).")
        if len(self.actor_betas) != 2 or any(not 0 <= x < 1 for x in self.actor_betas):
            raise ValueError("actor_betas must contain two values in [0, 1).")
        if not 0 < self.lr_final_fraction <= 1:
            raise ValueError("lr_final_fraction must be in (0, 1].")
        if self.precision not in ("float32", "tf32", "bf16"):
            raise ValueError("precision must be float32, tf32 or bf16.")
        if self.compile_mode not in ("default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"):
            raise ValueError("Unsupported torch.compile mode.")
        if self.replay_device != "cpu" and not self.replay_device.startswith("cuda"):
            raise ValueError("replay_device must be cpu or a CUDA device.")
        if self.resume and self.init_policy:
            raise ValueError("Use resume or init_policy, not both.")
        from src.amp_features import AMP_SCHEMAS
        if self.amp_feature_schema not in AMP_SCHEMAS:
            raise ValueError(f"Unknown AMP feature schema: {self.amp_feature_schema}")
        from src.robot_profiles import PROFILES as ROBOT_PROFILES
        flat_tasks = tuple(task for robot in ROBOT_PROFILES for task in robot.flat_tasks)
        if self.task not in flat_tasks:
            raise ValueError(
                "This version models flat ground. Use Unitree-G1-AMP-Flat; "
                "rough-terrain imagination needs terrain observations and contact geometry. "
                f"Flat tasks: {', '.join(flat_tasks)}."
            )
        if (self.task.endswith("-Aligned")) != (self.amp_feature_schema == "grounded_v2"):
            raise ValueError("Use grounded_v2 with Unitree-G1-AMP-Flat-Aligned; legacy task uses legacy AMP features. "
                             "The same pairing holds for the EngineAI-PM01 flat tasks.")
        for name in ("model_hidden", "policy_hidden", "critic_hidden", "discriminator_hidden"):
            if not getattr(self, name) or any(x < 1 for x in getattr(self, name)):
                raise ValueError(f"{name} must contain positive layer widths.")
