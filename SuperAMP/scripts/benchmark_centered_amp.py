"""Bounded benchmark of centered AMP arithmetic and the existing D input gradient.

No simulator, replay, policy, model, optimizer or training checkpoint is used.
The official-profile discriminator architecture and reward chunk size are kept.
This measures a component; it cannot estimate whole-iteration training latency.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import torch
from rsl_rl.modules.discriminator import Discriminator
from src.wmac.amp_rewards import CenteredAmpReward, clipped_lsgan_reward


def percentile(values, fraction):
    values = sorted(values)
    location = (len(values) - 1) * fraction
    lo, hi = int(location), min(int(location) + 1, len(values) - 1)
    return values[lo] * (hi - location) + values[hi] * (location - lo) if lo != hi else values[lo]


def summarize(values):
    return {"median_ms": statistics.median(values), "p10_ms": percentile(values, .1),
            "p90_ms": percentile(values, .9), "samples_ms": values}


def measured(fn, iterations, device):
    if device.type == "cuda":
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(device)
        begin.record()
        for _ in range(iterations):
            fn()
        end.record()
        end.synchronize()
        return begin.elapsed_time(end) / iterations
    started = time.perf_counter()
    for _ in range(iterations):
        fn()
    return 1000 * (time.perf_counter() - started) / iterations


def paired(functions, *, device, repeats, iterations, warmup):
    compile_and_warmup = {}
    for name, fn in functions.items():
        started = time.perf_counter()
        for _ in range(warmup):
            fn()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        compile_and_warmup[name] = time.perf_counter() - started
    samples = {name: [] for name in functions}
    names = list(functions)
    for index in range(repeats):
        for name in (names if index % 2 == 0 else names[::-1]):
            samples[name].append(measured(functions[name], iterations, device))
    result = {name: summarize(values) for name, values in samples.items()}
    result["first_calls_excluded_seconds"] = compile_and_warmup
    if "clipped" in samples and "centered" in samples:
        differences = [new - old for old, new in zip(samples["clipped"], samples["centered"])]
        result["paired_centered_minus_clipped"] = summarize(differences)
        result["ratio_of_medians"] = result["centered"]["median_ms"] / result["clipped"]["median_ms"]
    return result


def gpu_processes():
    try:
        command = ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"]
        return subprocess.run(command, text=True, capture_output=True, check=False).stdout.strip()
    except FileNotFoundError:
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32768)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--output", type=Path, default=REPO / "docs/validation/centered_amp_component_benchmark.json")
    args = parser.parse_args()
    if min(args.batch_size, args.repeats, args.warmup) <= 0:
        parser.error("batch-size, repeats and warmup must be positive")
    torch.set_num_threads(1)
    torch.manual_seed(4726)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True
    discriminator = Discriminator(406, .1, [1024, 512, 256], device, .75).eval()
    discriminator.requires_grad_(False)
    mapping = CenteredAmpReward(.1, 1., .99).to(device)
    policy_scores, expert_scores = torch.randn(512, device=device) - .5, torch.randn(512, device=device) + .5
    mapping.update(policy_scores, expert_scores)
    first_processes = gpu_processes() if device.type == "cuda" else None
    mappings = {"clipped": lambda score: clipped_lsgan_reward(score, .1), "centered": mapping}
    x = torch.randn(args.batch_size, 406, device=device, requires_grad=True)
    scores = torch.randn(args.batch_size, device=device, requires_grad=True)
    full_functions, score_functions = {}, {}
    for name, reward in mappings.items():
        def full_forward(value, reward=reward):
            if device.type == "cuda":
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    raw_score = discriminator(value).float().squeeze(-1)
            else:
                raw_score = discriminator(value).squeeze(-1)
            return reward(raw_score)
        full = torch.compile(full_forward, fullgraph=True) if args.compile else full_forward
        score_forward = torch.compile(reward, fullgraph=True) if args.compile else reward
        def full_backward(full=full):
            x.grad = None
            full(x).sum().backward()
        def score_backward(score_forward=score_forward):
            scores.grad = None
            score_forward(scores).sum().backward()
        full_functions[name] = full_backward
        score_functions[name] = score_backward
    result = {
        "scope": "Component benchmark only; not end-to-end WMAC iteration time or learning performance",
        "device": str(device), "torch": torch.__version__, "compiled": args.compile,
        "batch_size": args.batch_size, "discriminator": [406, 1024, 512, 256, 1],
        "discriminator_precision": "bf16 autocast, score/mapping fp32" if device.type == "cuda" else "fp32",
        "existing_discriminator_update_batch_size": 512,
        "new_networks": 0, "extra_discriminator_forward_passes": 0,
        "extra_optimizer_steps": 0, "physical_sampling": 0,
        "concurrency_at_start": first_processes,
        "limitations": [
            "Concurrent GPU training can perturb both old and new timings and inflate paired differences.",
            "Synthetic normalized AMP inputs, random fixed discriminator weights; numerical correctness is tested separately.",
            "No actor/world model/FK/replay/critic is timed; ratios do not imply whole-training slowdown.",
            "First calls, CUDA initialization and optional torch.compile costs are excluded from steady measurements.",
        ],
    }
    print("Benchmarking frozen D forward + reward + input backward", flush=True)
    result["discriminator_reward_input_backward"] = paired(full_functions, device=device,
        repeats=args.repeats, iterations=3, warmup=args.warmup)
    print("Benchmarking score mapping forward/backward", flush=True)
    result["score_mapping_forward_backward"] = paired(score_functions, device=device,
        repeats=args.repeats, iterations=20, warmup=args.warmup)
    print("Benchmarking scalar EMA reuse of two existing 512-score batches", flush=True)
    result["ema_update"] = paired({"centered": lambda: mapping.update(policy_scores, expert_scores)},
        device=device, repeats=args.repeats, iterations=20, warmup=args.warmup)
    result["finite_input_gradient"] = bool(torch.isfinite(x.grad).all())
    result["discriminator_parameter_gradients_absent"] = all(p.grad is None for p in discriminator.parameters())
    result["concurrency_at_end"] = gpu_processes() if device.type == "cuda" else None
    if device.type == "cuda":
        result["device_name"] = torch.cuda.get_device_name(device)
        result["peak_process_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "full_path": result["discriminator_reward_input_backward"],
                      "mapping": result["score_mapping_forward_backward"], "ema": result["ema_update"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
