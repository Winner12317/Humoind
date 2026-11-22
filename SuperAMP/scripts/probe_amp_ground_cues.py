"""Frozen-D CPU ablation of existing normalized AMP ground cues on real traces.

This diagnostic changes only temporary discriminator inputs. Zero is the
normalizer mean, may be out of distribution, and is not a training proposal.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from scripts.analyze_amp_recovery_style import (
    features, json_clean, load_scorer, make_cpu_adapter, sha256,
)
from scripts.analyze_recovery_gap_style import masks
from src.wmac.state import quat_to_matrix


GROUND = np.r_[195:203, 398:406]
LEGACY = np.r_[0:195, 203:398]


def module_hash(module):
    result = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        array = value.detach().cpu().numpy()
        result.update(name.encode())
        result.update(str(array.dtype).encode())
        result.update(str(array.shape).encode())
        result.update(array.tobytes())
    return result.hexdigest()


def score_summary(score, coefficient):
    if not len(score):
        return {"samples": 0}
    reward = coefficient * np.maximum(1. - .25 * (score - 1.) ** 2, 0.)
    return {
        "samples": len(score), "score_mean": float(score.mean()),
        "score_std": float(score.std()),
        "score_below_minus1_fraction": float((score < -1.).mean()),
        "score_above3_fraction": float((score > 3.).mean()),
        "score_exact_boundary_fraction": float(((score == -1.) | (score == 3.)).mean()),
        "amp_clipped_zero_fraction": float(((score <= -1.) | (score >= 3.)).mean()),
        "style_reward_mean": float(reward.mean()),
    }


def gradient_summary(gradient):
    if not len(gradient):
        return {"samples": 0}
    total_l1 = np.abs(gradient).sum(dtype=np.float64)
    answer = {"samples": len(gradient), "total_dimensions": 406}
    for name, indices in (("ground", GROUND), ("legacy", LEGACY)):
        selected = gradient[:, indices].astype(np.float64)
        l1 = np.abs(selected).sum()
        answer[name] = {
            "dimensions": len(indices),
            "rms_per_coordinate": float(np.sqrt(np.square(selected).mean())),
            "mean_abs_per_coordinate": float(np.abs(selected).mean()),
            "total_l1_fraction": float(l1 / total_l1) if total_l1 else None,
            "l1_per_transition_mean": float(l1 / len(selected)),
        }
    return answer


def clipping_summary(clipped):
    if not len(clipped):
        return {"samples": 0}
    answer = {"samples": len(clipped), "any_coordinate_fraction": float(clipped.any(-1).mean()),
              "coordinate_fraction": float(clipped.mean())}
    for name, indices in (("ground", GROUND), ("legacy", LEGACY)):
        selected = clipped[:, indices]
        answer[name] = {"dimensions": len(indices), "coordinate_fraction": float(selected.mean()),
                        "any_coordinate_fraction": float(selected.any(-1).mean())}
    return answer


def probe(pair, scorer, include_legacy, batch_size):
    network, norm, _, _ = scorer
    names = ["full", "ground_at_normalizer_mean"]
    if include_legacy:
        names.append("legacy_at_normalizer_mean")
    scores, gradients, clipping = {name: [] for name in names}, [], []
    for start in range(0, len(pair), batch_size):
        chunk = torch.from_numpy(pair[start:start + batch_size])
        with torch.no_grad():
            unbounded = torch.cat(((chunk[:, :203] - norm.mean) / norm.std(),
                                   (chunk[:, 203:] - norm.mean) / norm.std()), -1)
            clipped = (unbounded.abs() > norm.clip) if norm.clip is not None else torch.zeros_like(unbounded, dtype=torch.bool)
            normalized = torch.cat((norm(chunk[:, :203]), norm(chunk[:, 203:])), -1)
        x = normalized.detach().requires_grad_(True)
        raw = network(x).squeeze(-1)
        gradient = torch.autograd.grad(raw.sum(), x, create_graph=False)[0]
        if not bool(torch.isfinite(raw).all() and torch.isfinite(gradient).all()):
            raise ValueError("Nonfinite frozen-D scores or normalized-input gradients")
        scores["full"].append(raw.detach().numpy())
        gradients.append(gradient.numpy())
        clipping.append(clipped.numpy())
        with torch.no_grad():
            for name, indices in (("ground_at_normalizer_mean", GROUND),
                                  ("legacy_at_normalizer_mean", LEGACY)):
                if name in scores:
                    ablated = normalized.clone()
                    ablated[:, indices] = 0.
                    scores[name].append(network(ablated).squeeze(-1).numpy())
    return ({name: np.concatenate(parts) for name, parts in scores.items()},
            np.concatenate(gradients), np.concatenate(clipping))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT/"logs/wmac/2026-11-16_11-31-06_022422_seed42_full_official_flat_mirrored/model_40800.pt")
    parser.add_argument("--paired", type=Path, nargs="+", default=[ROOT/"logs/ppo60000_vs_wmac40800_recovery_seed2026", ROOT/"logs/ppo60000_vs_wmac40800_walk_seed2026"])
    parser.add_argument("--output", type=Path, default=ROOT/"docs/validation/recovery_gap_ground_cues_20261116.json")
    parser.add_argument("--include-legacy-ablation", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1024)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.output.exists():
        parser.error("Output already exists; choose a new file to preserve the audit")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    started = time.perf_counter()
    scorer = load_scorer(args.checkpoint.resolve())
    network, norm, coefficient, info = scorer
    before = {"discriminator": module_hash(network), "normalizer": module_hash(norm)}
    adapter = make_cpu_adapter(ROOT/"src/assets/robots/unitree_g1/xmls/g1.xml")
    report = {
        "schema": "amp-frozen-ground-cue-ablation-v1", "device": "cpu", "torch_threads": 2,
        "training_updates": 0, "scorer": info, "script_sha256": sha256(Path(__file__)),
        "ground_indices": GROUND.tolist(), "legacy_indices": LEGACY.tolist(),
        "gradient_definition": "Per-pair derivative of raw D score with respect to the full normalized-and-clipped406 input; not AMP reward, state, action or simulator derivatives.",
        "groups": "Shared first-episode mask across all trials in each recording. Groups overlap. Stable walking additionally requires no replay recovery, pelvis height>=0.65m, root up_z>=0.9.",
        "limitations": [
            "Ablation sets selected normalized coordinates to zero, corresponding to their frozen normalizer means. The resulting combination can be out of distribution.",
            "Ablation is a sensitivity diagnostic, not evidence that ground cues should be removed or a training/model change is beneficial.",
            "Ground contains16 coordinates; legacy contains390. Compare per-coordinate RMS alongside total L1 fractions; total fractions alone depend on dimension count.",
            "Raw-score gradients can be nonzero where clipped AMP reward and its direct gradient are zero.",
            "A frozen trained discriminator is not an independent reference-style ground truth.",
        ], "recordings": {},
    }
    for directory in args.paired:
        metadata_path = directory/"metrics.json"
        metadata = json.loads(metadata_path.read_text())
        protocol = metadata["protocol"]
        if protocol.get("world_model_used_for_control") or protocol.get("network_parameters_updated"):
            raise ValueError("Use fixed-policy physical recordings")
        data = {}
        for trial in metadata["trials"]:
            with np.load(directory/f"transitions_{trial}.npz", allow_pickle=False) as archive:
                data[trial] = {key: archive[key].copy() for key in ("state", "next_state", "context", "valid")}
        common = np.logical_and.reduce([value["valid"].astype(bool) for value in data.values()])
        for arrays in data.values():
            for key in ("state", "next_state", "context"):
                common &= np.isfinite(arrays[key]).all(-1)
        entries = {}
        for trial, arrays in data.items():
            print(f"Ground-cue probe {directory.name}/{trial}: {int(common.sum())} common real pairs", flush=True)
            pair = np.concatenate((features(adapter, arrays["state"][common]),
                                   features(adapter, arrays["next_state"][common])), -1)
            if not len(pair):
                raise ValueError("No common finite physical transitions")
            scores, gradient, clipped = probe(pair, scorer, args.include_legacy_ablation, args.batch_size)
            selected_groups = masks(arrays, common)
            up = quat_to_matrix(torch.from_numpy(arrays["state"][..., 3:7])).numpy()[..., 2, 2]
            selected_groups["stable_walking"] = selected_groups["replay_walk_phase"] & (arrays["state"][..., 2] >= .65) & (up >= .9)
            phases = {}
            for name, group in selected_groups.items():
                selection = group[common]
                row = {"transitions": int(selection.sum()),
                       "normalizer_input_clipping": clipping_summary(clipped[selection]),
                       "full_score_input_gradient": gradient_summary(gradient[selection]),
                       "variants": {variant: score_summary(raw[selection], coefficient)
                                    for variant, raw in scores.items()}}
                if selection.any():
                    row["ablation_score_delta_mean"] = {variant: float((raw[selection]-scores["full"][selection]).mean())
                                                          for variant, raw in scores.items() if variant != "full"}
                phases[name] = row
            entries[trial] = {"policy": trial.removesuffix("_repeat"),
                              "policy_sha256": metadata["checkpoints"][trial.removesuffix("_repeat")]["sha256"],
                              "recording_sha256": sha256(directory/f"transitions_{trial}.npz"), "phases": phases}
        report["recordings"][directory.name] = {"protocol": protocol, "metrics_sha256": sha256(metadata_path), "trials": entries}
    after = {"discriminator": module_hash(network), "normalizer": module_hash(norm)}
    if before != after or any(p.grad is not None for p in network.parameters()):
        raise RuntimeError("Frozen parameters, normalizer buffers or parameter gradients changed")
    report.update(parameters_and_buffers_unchanged=True, parameter_gradients_absent=True,
                  module_hashes_before=before, module_hashes_after=after,
                  elapsed_seconds=time.perf_counter()-started)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(json_clean(report), ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    temporary.replace(args.output)
    print(args.output, flush=True)


if __name__ == "__main__":
    main()
