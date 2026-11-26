"""Benchmark the archived and optimized WMAC runners at matched sample budgets.

The coordinator prepares physical data once, then runs each case in a separate
process. It reports first-call compilation separately from steady measurements.
No horizon, batch or model width is silently reduced after an OOM. Use --dry-run
to inspect the resolved cases without initializing CUDA.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, fields
from datetime import datetime
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import threading
import time
import traceback

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
DEFAULT_BASELINE = REPO/"logs/optimization_2026-11-06/baseline/wmac_baseline"
DEFAULT_CHECKPOINT = REPO/"logs/wmac/2026-11-06_09-30-21_254674_seed42_full_wmac_scratch/model_200.pt"
MODULE_NAMES = ("actor", "critic", "world_model", "discriminator", "amp_norm")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(clean(value), indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


@contextmanager
def heartbeat(label):
    stop = threading.Event()
    started = time.perf_counter()
    def beat():
        while not stop.wait(15):
            print(f"[BENCH heartbeat] {label}: {time.perf_counter()-started:.1f}s elapsed", flush=True)
    thread = threading.Thread(target=beat, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=1)


def config_types(baseline):
    baseline = Path(baseline)
    if not (baseline/"runner.py").is_file():
        raise FileNotFoundError(f"Archived baseline package not found: {baseline}")
    sys.path.insert(0, str(baseline.parent))
    old = importlib.import_module(baseline.name+".config").WmacConfig
    from src.wmac.config import WmacConfig
    return old, WmacConfig


def make_plan(args):
    old_type, new_type = config_types(args.baseline)
    production = new_type.for_profile("supermimic" if args.reference_scale else "balanced")
    # This benchmark compares implementations under the historical objective.
    # Algorithm/reward changes require their own controlled experiment.
    production.recovery_counter_semantics = "soft_v1"
    production.recovery_height_reward = "height_only_v1"
    production.actor_type = "mlp"
    production.num_envs = args.num_envs
    production.actor_batch_size = args.batch_size
    production.wm_window = args.wm_window
    production.wm_updates = args.wm_updates
    production.wm_batch_size = args.wm_batch_size
    production.precision = args.precision
    production.compile = args.compile
    production.device = args.device
    production.actor_microbatch_size = args.actor_microbatch_size
    # Preserve enough replay rows for the shared dataset plus every measured
    # collection call. This is a benchmark capacity, not a reduced sample budget.
    capacity = max(128, args.data_steps+(args.warmup+args.repetitions+2)*args.collect_steps)
    production.replay_steps = capacity
    production.warmup_steps = args.data_steps
    production.evaluation_interval = 0
    production.save_interval = 10**9
    production.log_interval = 10**9
    production.init_policy = production.resume = None
    production.benchmark_sync = True
    cases = []
    horizons = list(dict.fromkeys((16, args.horizon)))
    old_fields = {field.name for field in fields(old_type)}
    for horizon in horizons:
        equal = new_type.for_profile("legacy")
        equal.recovery_counter_semantics = "soft_v1"
        equal.recovery_height_reward = "height_only_v1"
        equal.actor_type = "mlp"
        equal.num_envs = args.num_envs
        equal.actor_batch_size = args.batch_size
        equal.imagination_horizon = horizon
        equal.model_hidden = production.model_hidden
        equal.policy_hidden = production.policy_hidden
        equal.wm_window, equal.wm_updates, equal.wm_batch_size = args.wm_window, args.wm_updates, args.wm_batch_size
        equal.collect_steps = args.collect_steps
        equal.warmup_steps, equal.replay_steps = args.data_steps, capacity
        equal.validation_env_fraction = 0.
        equal.real_critic_batch_size = horizon*args.batch_size
        equal.actor_trust_coef = 0.
        equal.actor_microbatch_size = args.actor_microbatch_size
        equal.precision, equal.compile, equal.device = args.precision, args.compile, args.device
        equal.replay_device = args.device if args.device.startswith("cuda") else "cpu"
        equal.evaluation_interval, equal.save_interval, equal.log_interval = 0, 10**9, 10**9
        equal.init_policy = equal.resume = None
        equal.benchmark_sync = True
        old_values = {key: value for key, value in asdict(equal).items() if key in old_fields}
        old_values["replay_device"] = "cpu"
        old_cfg = old_type(**old_values)
        cases.extend((
            {"name": f"legacy_h{horizon}", "implementation": "legacy", "matched_budget": True,
             "config": asdict(old_cfg), "effective_precision": "tf32" if args.precision == "tf32" else "float32"},
            {"name": f"optimized_h{horizon}", "implementation": "optimized", "matched_budget": True,
             "config": asdict(equal), "effective_precision": args.precision},
        ))
    # Keep the production default's H32, 5 critic epochs, bounded auxiliary real
    # batch and UTD collection cadence visible as a distinct, unmatched case.
    cases.append({"name": "optimized_default_h32", "implementation": "optimized", "matched_budget": False,
                  "config": asdict(production), "effective_precision": args.precision})
    if args.cases:
        selected = set(args.cases.split(","))
        missing = selected-{case["name"] for case in cases}
        if missing:
            raise ValueError(f"Unknown cases: {sorted(missing)}")
        cases = [case for case in cases if case["name"] in selected]
    source_files = [Path(__file__).resolve(), REPO/"src/recovery_rewards.py",
                    REPO/"src/tasks/amp_loco/mdp/rewards.py", *sorted((REPO/"src/wmac").glob("*.py")),
                    *sorted(Path(args.baseline).glob("*.py"))]
    return {"arguments": vars(args), "cases": cases,
            "source_sha256": {str(path): sha256(path) for path in source_files},
            "methodology": {
                "shared_data": "One physical dataset and one network/normalizer initialization for every case",
                "recovery_modes": "Historical soft_v1 counters and height_only_v1 reward are retained in every implementation case; new behavior is tested separately",
                "actor_architecture": "This archived implementation suite measures MLP Actors only; sparse MoE requires a separate benchmark",
                "matched_cases": "Same N, actor B/H, model widths, K, WM batches/updates, D batches/updates, critic epochs and auxiliary real batch",
                "precision": "Legacy remains fp32 (or global TF32); requested bf16 is an explicitly reported optimized arithmetic change",
                "replay": "Legacy CPU replay vs optimized device replay are implementation choices, with identical initial data",
                "samples": "Replay starts have the same phase mixture, not necessarily identical RNG draw indices across replay implementations",
                "measurement": "CUDA synchronized wall time; cold first full iteration separated from warmup and steady median/range",
                "full_iteration": "Collection + WM fitting + discriminator + full Actor BPTT/Critic fitting; saves, evaluation, periodic validation and logging excluded",
                "production_case": "Actual new defaults are separately labeled; no pure optimization speedup is inferred from this unmatched workload",
            }}


def imports_for_worker(plan):
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import numpy as np
    import torch
    import mjlab.tasks  # noqa: F401
    import src.tasks  # noqa: F401
    old_type, new_type = config_types(plan["arguments"]["baseline"])
    return torch, np, old_type, new_type


def seed_everything(torch, np, seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_precision(torch, precision):
    torch.backends.cuda.matmul.allow_tf32 = precision == "tf32"
    torch.backends.cudnn.allow_tf32 = precision == "tf32"
    torch.set_float32_matmul_precision("high" if precision == "tf32" else "highest")


def make_environment(cfg, implementation, baseline):
    from mjlab.tasks.registry import load_env_cfg
    from src.recovery_rewards import configure_recovery_height_reward
    package = Path(baseline).name if implementation == "legacy" else "src.wmac"
    env_module = importlib.import_module(package+".env")
    state_module = importlib.import_module(package+".state")
    env_cfg = load_env_cfg(cfg.task)
    configure_recovery_height_reward(env_cfg, getattr(cfg, "recovery_height_reward", "height_only_v1"))
    env_cfg.scene.num_envs, env_cfg.seed = cfg.num_envs, cfg.seed
    mode = getattr(cfg, "recovery_counter_semantics", "soft_v1")
    if implementation == "legacy" and mode != "soft_v1":
        raise ValueError("The archived benchmark adapter only implements recovery_counter_semantics='soft_v1'")
    env = env_module.CaptureEnv(cfg=env_cfg, device=cfg.device)
    if implementation == "legacy":
        # Archived adapters predate the constructor keyword; their forward
        # implementation remains the historical soft rule.
        env.adapter = state_module.StateAdapter(env)
        env.adapter.recovery_counter_semantics = mode
    else:
        env.adapter = state_module.StateAdapter(env, recovery_counter_semantics=mode)
    return env


def load_shared_networks(torch, runner, payload):
    from src.wmac.actors import actor_architecture
    if actor_architecture(runner.cfg) != actor_architecture(payload.get("config", {})):
        raise ValueError("Shared benchmark Actor architecture differs from the target runner")
    for name in MODULE_NAMES:
        getattr(runner, name).load_state_dict(payload[name], strict=True)
    settings = payload.get("amp_normalizer_settings", {})
    for key, value in settings.items():
        setattr(runner.amp_norm, key, value)


def module_hashes(runner):
    result = {}
    for name in MODULE_NAMES:
        digest = hashlib.sha256()
        for key, value in sorted(getattr(runner, name).state_dict().items()):
            tensor = value.detach().cpu().contiguous()
            digest.update(key.encode());digest.update(str(tensor.dtype).encode())
            digest.update(str(tuple(tensor.shape)).encode());digest.update(tensor.numpy().tobytes())
        result[name] = digest.hexdigest()
    return result


def prepare_shared(plan, artifacts):
    torch, np, _, new_type = imports_for_worker(plan)
    args = argparse.Namespace(**plan["arguments"])
    seed_everything(torch, np, args.seed)
    cfg_values = next((case["config"] for case in plan["cases"] if case["implementation"] == "optimized"),
                      asdict(new_type.for_profile("supermimic" if args.reference_scale else "balanced")))
    cfg = new_type(**{key: value for key, value in cfg_values.items() if key in {f.name for f in fields(new_type)}})
    cfg.num_envs, cfg.seed = args.num_envs, args.seed
    cfg.compile, cfg.precision, cfg.replay_device, cfg.validation_env_fraction = False, "float32", "cpu", 0.
    cfg.actor_batch_size, cfg.imagination_horizon = args.batch_size, args.horizon
    cfg.model_hidden = (2048, 2048, 2048) if args.reference_scale else (512, 512, 512)
    cfg.policy_hidden = (2048, 1024, 512, 256) if args.reference_scale else (512, 256, 128)
    cfg.device, cfg.wm_window = args.device, args.wm_window
    cfg.wm_updates, cfg.wm_batch_size = args.prepare_wm_updates, args.wm_batch_size
    cfg.init_policy = cfg.resume = None
    from src.wmac.runner import WmacRunner
    env = make_environment(cfg, "optimized", args.baseline)
    runner = None
    started = time.perf_counter()
    try:
        runner = WmacRunner(env, cfg, artifacts/"prepare")
        checkpoint_loaded = False
        source_checkpoint = args.checkpoint
        source_modes = None
        source_actor_architecture = None
        skipped_reason = None
        if source_checkpoint:
            source = torch.load(source_checkpoint, map_location=args.device, weights_only=False)
            if source.get("format") == "amp-wmac-v1":
                from src.wmac.checkpoints import resolve_checkpoint_config
                from src.wmac.actors import actor_architecture
                source_cfg = resolve_checkpoint_config(source)
                source_actor_architecture = actor_architecture(source_cfg)
                source_modes = {key: getattr(source_cfg, key) for key in
                                ("recovery_counter_semantics", "recovery_height_reward")}
            architecture = source.get("config", {})
            compatible = all(tuple(architecture.get(key, ())) == tuple(getattr(cfg, key))
                             for key in ("model_hidden", "policy_hidden", "critic_hidden", "discriminator_hidden"))
            if source_actor_architecture is not None:
                compatible = compatible and source_actor_architecture == actor_architecture(cfg)
            if compatible and all(name in source for name in MODULE_NAMES):
                load_shared_networks(torch, runner, source)
                checkpoint_loaded = True
            else:
                skipped_reason = ("Sparse MoE checkpoints cannot initialize this archived MLP benchmark suite; every case instead shares the same seeded MLP initialization and fitted dynamics"
                    if source_actor_architecture is not None and source_actor_architecture["actor_type"] == "moe" else
                    "Checkpoint architecture differs; every case instead shares the same seeded initialization and fitted dynamics")
        rows = []
        with heartbeat("prepare shared physical data"):
            for step in range(args.data_steps):
                runner.collect(1)
                rows.append({key: value.detach().cpu().clone() for key, value in env.transition.items()})
                if (step+1) % 16 == 0:
                    print(f"[BENCH prepare] collected {step+1}/{args.data_steps} physical steps", flush=True)
        for row in rows:
            if any(value.is_floating_point() and not bool(torch.isfinite(value).all()) for value in row.values()):
                raise RuntimeError("Shared physical data contains non-finite values; refusing to benchmark different filtering workloads")
        model_fit = None
        if not checkpoint_loaded and args.prepare_wm_updates:
            with heartbeat("prepare shared world-model calibration"):
                model_fit = runner.fit_model()
        payload = {name: {key: value.detach().cpu() for key, value in getattr(runner, name).state_dict().items()}
                   for name in MODULE_NAMES}
        payload["amp_normalizer_settings"] = {key: getattr(runner.amp_norm, key)
                                              for key in ("eps", "variance_epsilon", "clip")}
        payload["rows"] = rows
        payload["config"] = asdict(cfg)
        path = artifacts/"shared_data_and_networks.pt"
        torch.save(payload, path)
        result = {"path": str(path), "sha256": sha256(path), "source_checkpoint": source_checkpoint,
                  "source_checkpoint_sha256": sha256(source_checkpoint) if source_checkpoint else None,
                  "source_checkpoint_recovery_modes": source_modes,
                  "source_checkpoint_actor_architecture": source_actor_architecture,
                  "measurement_recovery_modes": {key: getattr(cfg, key) for key in
                      ("recovery_counter_semantics", "recovery_height_reward")},
                  "checkpoint_scope": "shared network/normalizer initialization only; optimizer and simulator state are not resumed",
                  "checkpoint_loaded": checkpoint_loaded, "checkpoint_skip_reason": skipped_reason,
                  "physical_rows": args.data_steps, "physical_transitions": args.data_steps*args.num_envs,
                  "network_sha256": module_hashes(runner), "world_model_calibration": model_fit,
                  "seconds": time.perf_counter()-started}
        atomic_json(artifacts/"prepare.json", result)
    finally:
        if runner is not None:
            runner.writer.close()
        env.close()


def summarize(records):
    values = [record["seconds"] for record in records]
    return {"median_seconds": statistics.median(values), "min_seconds": min(values), "max_seconds": max(values),
            "peak_allocated_gib": max(record["peak_allocated_gib"] for record in records),
            "peak_reserved_gib": max(record["peak_reserved_gib"] for record in records), "samples": len(values)}


def benchmark_case(plan, artifacts, case_name):
    torch, np, old_type, new_type = imports_for_worker(plan)
    args = argparse.Namespace(**plan["arguments"])
    case = next(item for item in plan["cases"] if item["name"] == case_name)
    directory = artifacts/case_name
    directory.mkdir(parents=True, exist_ok=True)
    result = {"case": case, "status": "initializing", "source_sha256": plan["source_sha256"]}
    runner = env = None
    try:
        for path, expected in plan["source_sha256"].items():
            if sha256(path) != expected:
                raise RuntimeError(f"Source changed after benchmark plan was written: {path}")
        seed_everything(torch, np, args.seed)
        set_precision(torch, case["effective_precision"])
        cfg = (old_type if case["implementation"] == "legacy" else new_type)(**case["config"])
        package = Path(args.baseline).name if case["implementation"] == "legacy" else "src.wmac"
        runner_type = importlib.import_module(package+".runner").WmacRunner
        env = make_environment(cfg, case["implementation"], args.baseline)
        with heartbeat(case_name+" runner initialization/expert loading"):
            runner = runner_type(env, cfg, directory)
        shared_path = artifacts/"shared_data_and_networks.pt"
        payload = torch.load(shared_path, map_location="cpu", weights_only=False)
        load_shared_networks(torch, runner, payload)
        initial_hashes = module_hashes(runner)
        preparation = json.loads((artifacts/"prepare.json").read_text())
        if initial_hashes != preparation["network_sha256"]:
            raise RuntimeError("The runner did not receive the identical network/normalizer initialization")
        for row in payload["rows"]:
            runner.replay.add(row)
        runner.replay.mark_boundary()
        del payload
        if torch.cuda.is_available() and args.device.startswith("cuda"):
            torch.cuda.synchronize(args.device)
            hardware = {"name": torch.cuda.get_device_name(args.device),
                        "total_memory_gib": torch.cuda.get_device_properties(args.device).total_memory/2**30}
        else:
            hardware = {"name": "cpu"}
        result.update(config=asdict(cfg), initial_network_sha256=initial_hashes,
                      shared_data_sha256=preparation["sha256"], hardware=hardware,
                      torch_version=torch.__version__, cuda_version=torch.version.cuda,
                      compile_cache=str(os.environ.get("TORCHINDUCTOR_CACHE_DIR", "default")))
        is_cuda = args.device.startswith("cuda")
        def sync():
            if is_cuda:
                torch.cuda.synchronize(args.device)
        def measure(label, function):
            sync()
            if is_cuda:
                torch.cuda.reset_peak_memory_stats(args.device)
            start = time.perf_counter()
            with heartbeat(case_name+" "+label):
                output = function()
                sync()
            record = {"seconds": time.perf_counter()-start,
                      "peak_allocated_gib": torch.cuda.max_memory_allocated(args.device)/2**30 if is_cuda else 0.,
                      "peak_reserved_gib": torch.cuda.max_memory_reserved(args.device)/2**30 if is_cuda else 0.,
                      "metrics": output}
            if label.endswith("actor_critic"):
                if not all(math.isfinite(float(output[key])) for key in ("actor_loss", "actor_grad_norm", "critic_loss")):
                    raise RuntimeError("Non-finite Actor/Critic computation; no speedup may be inferred from skipped backward")
            print(f"[BENCH] {case_name} {label}: {record['seconds']:.4f}s, peak={record['peak_allocated_gib']:.3f}GiB", flush=True)
            return record
        def full_iteration(label):
            sync();start = time.perf_counter()
            count = runner._next_collect_count() if hasattr(runner, "_next_collect_count") else cfg.collect_steps
            phases = {
                "collect": measure(label+" collect", lambda: runner.collect(count)),
                "world_model": measure(label+" world_model", runner.fit_model),
                "discriminator": measure(label+" discriminator", runner.update_discriminator),
                "actor_critic": measure(label+" actor_critic", runner.update_actor_critic),
            }
            sync()
            record = {"seconds": time.perf_counter()-start,
                      "peak_allocated_gib": max(value["peak_allocated_gib"] for value in phases.values()),
                      "peak_reserved_gib": max(value["peak_reserved_gib"] for value in phases.values()),
                      "collect_steps": count, "phases": phases}
            runner.iteration += 1
            return record
        result["cold_first_iteration"] = full_iteration("cold_first_call_includes_compile")
        atomic_json(directory/"result.json", result)
        result["full_iteration_warmup"] = [full_iteration(f"warmup_{i+1}") for i in range(args.warmup)]
        full_records = [full_iteration(f"steady_{i+1}") for i in range(args.repetitions)]
        for i in range(args.warmup):
            measure(f"actor_only_warmup_{i+1} actor_critic", runner.update_actor_critic)
        actor_records = [measure(f"actor_only_steady_{i+1} actor_critic", runner.update_actor_critic)
                         for i in range(args.repetitions)]
        result.update(status="complete", full_iteration_records=full_records, actor_critic_records=actor_records,
                      full_iteration=summarize(full_records), actor_critic=summarize(actor_records),
                      phase_summaries={name: summarize([record["phases"][name] for record in full_records])
                                       for name in ("collect", "world_model", "discriminator", "actor_critic")})
    except Exception as error:
        result.update(status="out_of_memory" if "out of memory" in str(error).lower() else "error",
                      error=str(error), traceback=traceback.format_exc())
        print(result["traceback"], flush=True)
    finally:
        atomic_json(directory/"result.json", result)
        if runner is not None:
            runner.writer.close()
        if env is not None:
            env.close()
    if result["status"] != "complete":
        raise SystemExit(1)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--num-envs", type=int)
    result.add_argument("--batch-size", type=int)
    result.add_argument("--horizon", type=int, default=32)
    result.add_argument("--wm-window", type=int, default=8)
    result.add_argument("--wm-updates", type=int)
    result.add_argument("--wm-batch-size", type=int)
    result.add_argument("--repetitions", type=int, default=5)
    result.add_argument("--warmup", type=int, default=2)
    result.add_argument("--precision", choices=("float32", "bf16", "tf32"), default="bf16")
    result.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    result.add_argument("--reference-scale", action="store_true")
    result.add_argument("--actor-microbatch-size", type=int, default=0,
                        help="Full actor batch is retained; only accumulation chunks change (0 means full batch)")
    result.add_argument("--collect-steps", type=int, default=8)
    result.add_argument("--data-steps", type=int, default=64)
    result.add_argument("--prepare-wm-updates", type=int, default=4)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--device", default="cuda:0")
    result.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT) if DEFAULT_CHECKPOINT.is_file() else None)
    result.add_argument("--compile-cache", type=Path,
                        help="Reuse an existing TorchInductor disk cache; the path is recorded with first-call timing")
    result.add_argument("--baseline", default=str(DEFAULT_BASELINE))
    result.add_argument("--cases", help="Comma-separated case names; default runs the full comparison matrix")
    result.add_argument("--output", type=Path)
    result.add_argument("--dry-run", action="store_true")
    result.add_argument("--worker", choices=("prepare", "case"), help=argparse.SUPPRESS)
    result.add_argument("--plan-file", type=Path, help=argparse.SUPPRESS)
    result.add_argument("--case-name", help=argparse.SUPPRESS)
    return result


def main():
    args = parser().parse_args()
    if args.worker:
        plan = json.loads(args.plan_file.read_text())
        artifacts = args.plan_file.parent
        if args.worker == "prepare":
            with heartbeat("shared dataset worker"):
                prepare_shared(plan, artifacts)
        else:
            benchmark_case(plan, artifacts, args.case_name)
        return
    args.num_envs = args.num_envs if args.num_envs is not None else (4096 if args.reference_scale else 1024)
    args.batch_size = args.batch_size if args.batch_size is not None else (4096 if args.reference_scale else 512)
    args.wm_updates = args.wm_updates if args.wm_updates is not None else (10 if args.reference_scale else 4)
    args.wm_batch_size = args.wm_batch_size if args.wm_batch_size is not None else (2048 if args.reference_scale else 512)
    for name in ("num_envs", "batch_size", "horizon", "wm_window", "wm_updates", "wm_batch_size", "repetitions", "collect_steps"):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    if args.warmup < 0 or args.actor_microbatch_size < 0 or args.prepare_wm_updates < 0 or args.data_steps < args.wm_window:
        raise ValueError("Warmups/chunks must be nonnegative; shared data must cover the WM window")
    output = args.output or REPO/"logs/optimization_2026-11-06"/f"benchmark_{datetime.now():%Y%m%d_%H%M%S}"/"report.json"
    args.output = str(output.resolve())
    plan = make_plan(args)
    if args.dry_run:
        print(json.dumps(clean(plan), indent=2))
        return
    artifacts = output.parent/(output.stem+"_artifacts")
    artifacts.mkdir(parents=True, exist_ok=False)
    plan_file = artifacts/"plan.json"
    atomic_json(plan_file, plan)
    environment = os.environ.copy()
    # Shared cache is disclosed in the report. The first case can be cold;
    # later shapes may reuse on-disk kernels. Per-case first calls remain timed.
    environment["TORCHINDUCTOR_CACHE_DIR"] = str((args.compile_cache or artifacts/"inductor_cache").resolve())
    command = [sys.executable, str(Path(__file__).resolve()), "--plan-file", str(plan_file)]
    print(f"[BENCH] preparing shared data; artifacts={artifacts}", flush=True)
    subprocess.run([*command, "--worker", "prepare"], env=environment, check=True)
    results = []
    for case in plan["cases"]:
        print(f"[BENCH] starting {case['name']}", flush=True)
        subprocess.run([*command, "--worker", "case", "--case-name", case["name"]], env=environment, check=False)
        path = artifacts/case["name"]/"result.json"
        results.append(json.loads(path.read_text()) if path.is_file() else {"case": case, "status": "worker_failed_without_report"})
        atomic_json(output, {"plan": plan, "results": results, "in_progress": True})
    complete = {item["case"]["name"]: item for item in results if item["status"] == "complete"}
    comparisons = []
    for horizon in dict.fromkeys((16, args.horizon)):
        old, new = complete.get(f"legacy_h{horizon}"), complete.get(f"optimized_h{horizon}")
        if old is not None and new is not None:
            comparisons.append({"horizon": horizon, "same_sample_budget": True,
                "same_numerical_precision": old["case"]["effective_precision"] == new["case"]["effective_precision"],
                "actor_critic_speedup": old["actor_critic"]["median_seconds"]/new["actor_critic"]["median_seconds"],
                "full_iteration_speedup": old["full_iteration"]["median_seconds"]/new["full_iteration"]["median_seconds"]})
    atomic_json(output, {"plan": plan, "preparation": json.loads((artifacts/"prepare.json").read_text()),
                         "results": results, "matched_comparisons": comparisons, "in_progress": False})
    print(f"[BENCH] report={output.resolve()}", flush=True)
    print(json.dumps(comparisons, indent=2), flush=True)
    if any(item["status"] != "complete" for item in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
