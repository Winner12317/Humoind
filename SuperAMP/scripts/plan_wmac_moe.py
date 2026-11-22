"""Write the historical legacy-MoE six-arm plan; never launch training.

This is not the Hiking author-code architecture. To start that architecture,
use train_wmac.py --actor-type moe --moe-variant hiking.

In a new terminal, first ``source scripts/activate_env.sh``. Example:
  python scripts/plan_wmac_moe.py --seeds 42 43 44 --iterations 60000 \
      --num-envs 4096 --output /tmp/wmac_moe_plan.json
Only CPU Actor modules are constructed to count their actual parameters.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import shlex
import sys

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.wmac.actors import actor_architecture, create_actor
from src.wmac.config import WmacConfig

ACTOR_FIELDS = (
    "actor_type", "policy_hidden", "moe_variant", "moe_expert_hidden", "moe_num_experts", "moe_top_k",
    "moe_gate_hidden", "moe_temperature", "moe_load_balance_coef",
    "moe_router_z_coef", "moe_init_noise_std",
)
EXPERIMENT_FIELDS = (*ACTOR_FIELDS, "seed", "run_name")
PINNED_COMMON_FIELDS = (
    "collect_steps", "utd", "warmup_steps", "imagination_horizon",
    "recovery_counter_semantics", "recovery_height_reward",
)


def _json_value(value):
    return json.loads(json.dumps(value))


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def actor_parameter_budget(cfg):
    """Actual factory counts, with CPU placement and the caller's RNG restored."""
    import torch

    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        actor = create_actor(cfg)
        total = sum(parameter.numel() for parameter in actor.parameters())
        if cfg.actor_type == "moe":
            router = sum(parameter.numel() for parameter in actor.router.parameters())
            experts = [sum(parameter.numel() for parameter in expert.parameters())
                       for expert in actor.experts]
            if len(set(experts)) != 1 or total != router + sum(experts):
                raise ValueError("Parameter matching requires equal independent experts and a router")
            active = router + cfg.moe_top_k * experts[0]
        else:
            router, experts, active = 0, [], total
        if any(parameter.device.type != "cpu" for parameter in actor.parameters()):
            raise RuntimeError("Plan construction must remain on CPU")
    return {"total_parameters": total, "active_parameters_per_sample": active,
            "router_parameters": router, "expert_parameters": experts,
            "count_definition": "sum(actual create_actor(...).parameters()); shared parameters counted once; normalization buffers excluded",
            "active_count_limitation": "Router plus K full expert networks for one sample; parameter capacity is not FLOPs or elapsed time."}


def _nearest_mlp_widths(target, first, last=128):
    """Search the integer middle width; validate the selected count with factory."""
    def count(middle):
        widths = (384, first, middle, last, 29)
        return sum((left + 1) * right for left, right in zip(widths, widths[1:]))
    middle = min(range(1, 4097), key=lambda width: (abs(count(width)-target), width))
    return first, middle, last


def training_budget(cfg):
    """Fresh-run physical training interactions, separate from evaluation data."""
    if cfg.collect_steps != 0:
        raise ValueError("This plan requires WMAC's UTD-controlled collection")
    loop_steps = Fraction(cfg.iterations, 1) / Fraction(str(cfg.utd))
    loop_steps = loop_steps.numerator // loop_steps.denominator
    warmup = max(cfg.warmup_steps, cfg.wm_window)
    return {"max_iterations": cfg.iterations, "utd": cfg.utd, "num_envs": cfg.num_envs,
            "initial_collection_steps_per_env": warmup,
            "loop_collection_steps_per_env": loop_steps,
            "physical_training_steps_per_env": warmup + loop_steps,
            "physical_training_transitions": (warmup + loop_steps) * cfg.num_envs,
            "formula": "(max(warmup_steps, wm_window) + floor(iterations / utd)) * num_envs",
            "scope": "Fresh run with empty replay; includes initial collection and model/Critic warmup iterations; excludes physical evaluation, exports and retries.",
            "warmup_note": "model_warmup_iterations and critic_warmup_iterations gate updates within the same loop; they are not extra iterations added to this budget."}


def command_for(cfg):
    argv = ["python", "scripts/train_wmac.py", "--profile", "ppo_aligned",
            "--iterations", str(cfg.iterations), "--num-envs", str(cfg.num_envs),
            "--seed", str(cfg.seed), "--run-name", cfg.run_name]
    for name in (*PINNED_COMMON_FIELDS, *ACTOR_FIELDS):
        value = getattr(cfg, name)
        argv.append("--" + name.replace("_", "-"))
        if isinstance(value, (list, tuple)):
            argv.extend(str(item) for item in value)
        else:
            argv.append(str(value))
    return argv


def build_plan(iterations=60000, seeds=(42, 43, 44), num_envs=4096):
    _positive_integer(iterations, "iterations")
    _positive_integer(num_envs, "num_envs")
    seeds = tuple(seeds)
    if (not seeds or len(set(seeds)) != len(seeds)
            or any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds)):
        raise ValueError("seeds must be a nonempty list of distinct nonnegative integers")
    base = replace(WmacConfig.for_profile("ppo_aligned"), iterations=iterations,
                   num_envs=num_envs, actor_type="mlp", moe_variant="legacy", moe_num_experts=4,
                   moe_top_k=2, moe_gate_hidden=(256,), moe_temperature=1.,
                   moe_load_balance_coef=0., moe_router_z_coef=0., moe_init_noise_std=0.)
    base.validate()
    if (base.recovery_counter_semantics, base.recovery_height_reward) != ("hard_forward_v2", "upright_v2"):
        raise ValueError("This plan requires the current recovery v2 defaults")
    configs = {
        "mlp": base,
        "moe_dense": replace(base, actor_type="moe", moe_top_k=4),
        "moe_top2": replace(base, actor_type="moe"),
        "moe_top2_regularized": replace(base, actor_type="moe", moe_load_balance_coef=.01, moe_router_z_coef=.001),
    }
    budgets = {label: actor_parameter_budget(cfg) for label, cfg in configs.items()}
    targets = {"mlp_total_matched": (budgets["moe_top2"]["total_parameters"], 1024, "total_parameters"),
               "mlp_active_matched": (budgets["moe_top2"]["active_parameters_per_sample"], 768, "active_parameters_per_sample")}
    matches = {}
    for label, (target, first, criterion) in targets.items():
        configs[label] = replace(base, policy_hidden=_nearest_mlp_widths(target, first))
        budgets[label] = actor_parameter_budget(configs[label])
        actual = budgets[label]["total_parameters"]
        matches[label] = {"target_arm": "moe_top2", "target_measure": criterion,
                          "target_parameters": target, "actual_parameters": actual,
                          "signed_relative_error": (actual-target)/target,
                          "absolute_relative_error": abs(actual-target)/target,
                          "exact_match": actual == target,
                          "search": "Integer middle width in [1,4096], fixed first/last hidden widths; actual chosen network counted using create_actor."}
    common = {key: value for key, value in asdict(base).items() if key not in EXPERIMENT_FIELDS}
    arms, runs = [], []
    for label, cfg in configs.items():
        cfg.validate()
        current_common = {key: value for key, value in asdict(cfg).items() if key not in EXPERIMENT_FIELDS}
        if current_common != common:
            raise RuntimeError(f"Non-Actor training settings differ in arm {label}")
        arms.append({"name": label, "actor_architecture": actor_architecture(cfg),
                     "actor_intervention": {key: getattr(cfg, key) for key in ACTOR_FIELDS},
                     "parameters": budgets[label], "parameter_match": matches.get(label),
                     "objective": "original WMAC plus router regularizers on real replay starts" if cfg.moe_load_balance_coef or cfg.moe_router_z_coef else "original WMAC objective; no router auxiliary loss",
                     "compilation": {"compile": cfg.compile, "compile_fullgraph": cfg.compile_fullgraph,
                         "actor_dispatch": "local eager compiler-disable fence for dynamic expert dispatch; surrounding WMAC remains compiled" if cfg.actor_type == "moe" else "original MLP compilation path"}})
        for seed in seeds:
            run_cfg = replace(cfg, seed=seed, run_name=f"moe_plan_{label}_seed{seed}_iter{iterations}_envs{num_envs}")
            argv = command_for(run_cfg)
            runs.append({"arm": label, "seed": seed, "run_name": run_cfg.run_name,
                         "config": asdict(run_cfg), "argv": argv, "shell_command": shlex.join(argv)})
    source_files = tuple(sorted(path.relative_to(REPO).as_posix() for path in (REPO/"src/wmac").glob("*.py"))) + (
        "src/amp_features.py", "src/recovery_rewards.py", "src/training_cost.py",
        "src/tasks/amp_loco/mdp/rewards.py", "src/tasks/amp_loco/mdp/terminations.py",
        "scripts/train_wmac.py", "scripts/evaluate_wmac.py", "scripts/plan_wmac_moe.py")
    return _json_value({"format": "wmac-moe-long-comparison-plan-v1", "status": "planned_only_not_started",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "repository": str(REPO),
        "architecture_scope": "Historical legacy-MoE comparison, not the Hiking author-code Actor; use train_wmac.py --actor-type moe --moe-variant hiking for Hiking.",
        "environment_setup": {"cwd": str(REPO), "new_terminal_first": "source scripts/activate_env.sh",
                              "execution": "Review JSON, then run selected argv in the activated environment. This tool never executes or queues these commands."},
        "profile": "ppo_aligned", "seeds": seeds, "arm_count": len(arms), "run_count": len(runs),
        "initialization": "Independent from-scratch runs; no init_policy, resume, shared weights or optimizer migration. Same seed labels do not guarantee identical on-policy trajectories.",
        "source_sha256": {name: hashlib.sha256((REPO/name).read_bytes()).hexdigest() for name in source_files},
        "common_training_config": common, "training_budget_per_run": training_budget(base),
        "arms": arms, "runs": runs,
        "endpoints": ["Recovery success rate with failed trials retained", "Recovery time, reported alongside success and failures",
                      "Walking velocity error and falls", "Style distance against one fixed expert-reference protocol; moving discriminator reward alone is insufficient",
                      "End-to-end wall time including collection, updates, evaluation and saving", "Real physical interaction count; report training/evaluation separately",
                      "Expert utilization by motion phase, routing switches, action jumps and rejected Actor updates"],
        "evaluation": {"physical_protocol": "All arms use the same seeds, commands, initial-state audit, recovery v2 semantics and fixed evaluation horizons; training Actor architecture belongs to source provenance.",
                       "selection": "Compare all predeclared checkpoints or one common selection rule; do not select each arm's best test seed.",
                       "save_interval": base.save_interval, "evaluation_interval": base.evaluation_interval,
                       "evaluation_steps": base.evaluation_steps, "evaluation_envs": base.evaluation_envs,
                       "defaults": "Keep the current profile's saving and evaluation defaults; they are included in every resolved config."},
        "limits": ["A single seed or one fixed-budget result cannot establish that an architecture is generally better. Use multiple seeds and report variation.",
                   "From-scratch arms use their own independent initialization. The same seed label does not guarantee bitwise shared WM/Critic/D weights because Actor construction consumes different random numbers; use the fixed-data tool for a shared-state intervention.",
                   "Primary comparison is time and real interactions to the same task quality; also report the equal-budget endpoints.",
                   "Parameter matches are approximate, not identical: signed/absolute relative errors are recorded. Active parameters are not FLOPs or a speed guarantee.",
                   "MoE dense/top2 share one total capacity; changing routing or regularization is not an additional bank of parameters.",
                   "Nonzero router coefficients add losses to the optimization objective; zero coefficients give the pure-structure comparison.",
                   "Model, discriminator, Critic, rewards, H32 and UTD stay common. This plan does not copy a PPO rollout schedule or add perception inputs.",
                   "Commands resolve against the checked source defaults. Re-audit the saved config/source hashes if code changes before execution."]})


def write_plan(plan, output):
    path = Path(output).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(plan, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=60000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--output", type=Path, required=True, help="New JSON file; existing files are never overwritten")
    args = parser.parse_args(argv)
    if args.output.expanduser().exists():
        parser.error("--output must be a new file; refusing to overwrite")
    try:
        plan = build_plan(args.iterations, args.seeds, args.num_envs)
        path = write_plan(plan, args.output)
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))
    print(f"Plan only: {plan['arm_count']} arms x {len(plan['seeds'])} seeds = {plan['run_count']} runs; no training launched.")
    print(f"Output: {path}")
    print(f"Training transitions per run: {plan['training_budget_per_run']['physical_training_transitions']:,}")
    print("In a new terminal first: source scripts/activate_env.sh")
    print("Example argv rendered for review: " + plan["runs"][0]["shell_command"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
