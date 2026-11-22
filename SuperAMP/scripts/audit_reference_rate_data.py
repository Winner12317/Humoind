"""Read-only data/loader audit of the hypothesis that reference playback sped up.

CPU only. Does not change fps, assets, cached mirrors, networks or training.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.wmac.mirroring import build_g1_mirror_map
from src.wmac.runner import ExpertPairs
from rsl_rl.utils.motion_loader import AMPLoader


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: np.array(data[key], copy=True) for key in data.files}


def error(expected, actual):
    a, b = np.asarray(expected, dtype=np.float64), np.asarray(actual, dtype=np.float64)
    difference = a - b
    return dict(max_abs=float(np.abs(difference).max()), rms=float(np.sqrt(np.square(difference).mean())),
                exact_array_equal=bool(np.array_equal(a, b)))


def velocity_fit(fd, velocity):
    x, y = np.asarray(fd, dtype=np.float64).reshape(-1), np.asarray(velocity, dtype=np.float64).reshape(-1)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    denominator = np.dot(x, x)
    scale = np.dot(x, y) / denominator if denominator else np.nan
    return dict(velocity_over_fd_least_squares_scale=float(scale),
                velocity_rms=float(np.sqrt(np.square(y).mean())),
                fd_rms=float(np.sqrt(np.square(x).mean())),
                rmse=float(np.sqrt(np.square(y - x).mean())),
                correlation=float(np.corrcoef(x, y)[0, 1]))


def quat_mul(a, b):
    aw, av, bw, bv = a[..., :1], a[..., 1:], b[..., :1], b[..., 1:]
    return np.concatenate((aw * bw - (av * bv).sum(-1, keepdims=True),
                           aw * bv + bw * av + np.cross(av, bv)), -1)


def angular_fd(q, dt):
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    previous_inv = q[:-2].copy()
    previous_inv[..., 1:] *= -1
    relative = quat_mul(q[2:], previous_inv)
    relative = np.where(relative[..., :1] < 0, -relative, relative)
    vnorm = np.linalg.norm(relative[..., 1:], axis=-1, keepdims=True)
    angle = 2 * np.arctan2(vnorm, relative[..., :1])
    return relative[..., 1:] * np.divide(angle, vnorm, out=np.full_like(angle, 2.), where=vnorm > 1e-12) / (2 * dt)


def data_audit(source, cache, old_assets, mapping):
    files, mirrors = {}, {}
    for path in sorted(source.rglob("*.npz")):
        rel = str(path.relative_to(source)); data = read(path)
        n = len(data["joint_pos"]); fps = float(data["fps"].reshape(-1)[0]); dt = 1 / fps
        copied = cache / rel; old = old_assets / rel
        files[rel] = dict(frames=n, fps=fps, frame_dt_seconds=dt,
                          N_over_fps_seconds=n / fps, first_to_last_span_seconds=(n - 1) / fps,
                          metadata_keys=sorted(data), source_sha256=sha(path),
                          cached_original_sha256=sha(copied), cached_original_byte_equal=sha(path) == sha(copied),
                          old_project_sha256=sha(old) if old.is_file() else None,
                          old_project_byte_equal=sha(path) == sha(old) if old.is_file() else None,
                          joint_velocity_consistency=velocity_fit((data["joint_pos"][2:] - data["joint_pos"][:-2]) / (2 * dt), data["joint_vel"][1:-1]),
                          world_linear_velocity_consistency=velocity_fit((data["body_pos_w"][2:] - data["body_pos_w"][:-2]) / (2 * dt), data["body_lin_vel_w"][1:-1]),
                          world_angular_velocity_consistency=velocity_fit(angular_fd(data["body_quat_w"], dt), data["body_ang_vel_w"][1:-1]))
        if path.parent.name != "WalkandRun":
            continue
        mirrored_path = copied.with_name(copied.stem + "__mirror_y.npz"); mirror = read(mirrored_path)
        jp, bp = mapping.joint_permutation, mapping.body_permutation
        expected = {
            "fps": data["fps"],
            "joint_pos": ((data["joint_pos"][:, jp] - mapping.joint_reference[jp]) * mapping.joint_sign + mapping.joint_reference),
            "joint_vel": data["joint_vel"][:, jp] * mapping.joint_sign,
            "body_lin_vel_w": data["body_lin_vel_w"][:, bp] * np.array([1., -1., 1.]),
            "body_ang_vel_w": data["body_ang_vel_w"][:, bp] * np.array([-1., 1., -1.]),
            "body_quat_w": data["body_quat_w"][:, bp] * np.array([1., -1., 1., -1.]),
        }
        p = data["body_pos_w"][:, bp] * np.array([1., -1., 1.])
        p[..., 1] += 2 * data["body_pos_w"][0, 0, 1]
        expected["body_pos_w"] = p
        mirrors[rel] = dict(mirrored_file=str(mirrored_path.relative_to(cache)), frames_preserved=n == len(mirror["joint_pos"]),
                            fps_preserved=np.array_equal(data["fps"], mirror["fps"]),
                            independent_reflection_errors={key: error(values, mirror[key]) for key, values in expected.items()},
                            joint_step_rms_original=float(np.sqrt(np.square(np.diff(data["joint_pos"], axis=0).astype(np.float64)).mean())),
                            joint_step_rms_mirrored=float(np.sqrt(np.square(np.diff(mirror["joint_pos"], axis=0).astype(np.float64)).mean())))
    cached_rates = {str(p.relative_to(cache)): float(read(p)["fps"].reshape(-1)[0]) for p in sorted(cache.rglob("*.npz"))}
    return dict(original_clips=files, mirrored_walk_checks=mirrors, cached_fps=cached_rates,
                all_original_copies_byte_equal=all(x["cached_original_byte_equal"] for x in files.values()),
                all_old_project_assets_byte_equal=all(x["old_project_byte_equal"] for x in files.values()),
                all_cached_fps_values=sorted(set(cached_rates.values())),
                counts=dict(original_clips=len(files), cached_clips=len(cached_rates), mirrored_walk_clips=len(mirrors)),
                note="N/fps is loader duration convention; first-to-last timestamps span (N-1)/fps. Neither means playback acceleration.")


def loader_audit(source, cache, old_loader_path, mapping, samples):
    body_names = ("pelvis", "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link", "right_hip_roll_link",
                  "right_knee_link", "right_ankle_roll_link", "left_shoulder_roll_link", "left_elbow_link",
                  "left_wrist_yaw_link", "right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link")
    cfg = SimpleNamespace(task="Unitree-G1-AMP-Flat-Aligned", amp_feature_schema="grounded_v2", expert_recovery_fraction=.5)
    rl = SimpleNamespace(amp_body_names=body_names, amp_anchor_name="torso_link", amp_motion_files=str(source))
    env = SimpleNamespace(scene={"robot": SimpleNamespace(body_names=mapping.body_names)}, _wmac_motion_root=str(cache))
    with patch("src.wmac.runner.load_rl_cfg", return_value=rl):
        wmac_cached = ExpertPairs(cfg, env, "cpu")
        env._wmac_motion_root = str(source)
        wmac_original = ExpertPairs(cfg, env, "cpu")
    ppo_grounded = AMPLoader(str(cache), body_names, "torso_link", mapping.body_names,
                            device="cpu", feature_schema="grounded_v2", expert_recovery_fraction=.5)
    pools_equal = [torch.equal(a, b) for a, b in zip(ppo_grounded._pair_groups, wmac_cached.groups)]
    torch.manual_seed(20261117); first = ppo_grounded._sample_grounded_pairs(samples)
    torch.manual_seed(20261117); second = wmac_cached.sample(samples)
    seeded_equal = [torch.equal(a, b) for a, b in zip(first, second)]
    spec = importlib.util.spec_from_file_location("read_only_original_ppo_loader", old_loader_path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        old = module.AMPLoader(str(source), body_names, "torso_link", mapping.body_names, device="cpu")
    offsets = [0, 0]; per_clip = []
    paths = sorted(source.rglob("*.npz"))
    for index, path in enumerate(paths):
        components = [getattr(old, name)[index].flatten(1) for name in
                      ("_body_pos_b_list", "_body_ori_b_list", "_body_lin_vel_b_list", "_body_ang_vel_b_list")]
        old_features = torch.cat(components, -1)
        pairs = torch.stack((old_features[:-1], old_features[1:]), 1)
        group = int(path.parent.name == "Recovery"); count = len(pairs)
        new = wmac_original.groups[group][offsets[group]:offsets[group]+count, :, :old_features.shape[-1]]
        difference = (pairs - new).abs()
        per_clip.append(dict(clip=str(path.relative_to(source)), valid_nonterminal_pairs=count,
                             shared_195_feature_max_abs_difference=float(difference.max()),
                             shared_195_feature_rms_difference=float((pairs - new).square().mean().sqrt()),
                             old_terminal_self_pair_probability_if_clip_uniform=1 / len(old_features),
                             nonterminal_pair_spacing_seconds=1 / float(read(path)["fps"].reshape(-1)[0])))
        offsets[group] += count
    # Run the original generator with deterministic indices that include its final frame.
    def indices(low, high, shape, **kwargs):
        return torch.tensor([0, min(1, high - 1), high - 2, high - 1], device=kwargs.get("device", "cpu"))
    sampled_checks = []
    with patch("torch.randint", side_effect=indices):
        for index, (first, second) in enumerate(old.feed_forward_generator(len(paths), 4)):
            sampled_checks.append(dict(clip=str(paths[index].relative_to(source)),
                                       terminal_pair_is_same_frame=bool(torch.equal(first[-1], second[-1])),
                                       first_pair_has_nonzero_feature_delta=bool((first[0] != second[0]).any())))
    return dict(grounded_ppo_wmac_full_pools_exact_equal=pools_equal,
                grounded_ppo_wmac_seeded_sample_count=samples,
                grounded_ppo_wmac_seeded_first_second_exact_equal=seeded_equal,
                old_legacy_loader_source=str(old_loader_path), old_legacy_loader_sha256=sha(old_loader_path),
                nonterminal_old195_vs_current203_shared_prefix_per_clip=per_clip,
                original_generator_actual_index_checks=sampled_checks,
                current_pool_pair_counts={"walk": len(wmac_cached.groups[0]), "recovery": len(wmac_cached.groups[1])},
                distinction="Sampling more pairs or choosing a different recovery fraction changes dataset weighting; t→t+1 still spans 20 ms on these 50 fps files.")


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "src/assets/motions/g1/amp")
    parser.add_argument("--cache", type=Path, default=ROOT / ".wmac_cache/motions/g1_mirrored")
    parser.add_argument("--old-project", type=Path, default=Path("/home/autel/AMP_mjlab"))
    parser.add_argument("--samples", type=int, default=16384)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "logs/wmac/2026-11-16_19-08-55_304755_seed42_full_official_flat_centered_amp")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/validation/reference_rate_data_20261117.json")
    args = parser.parse_args()
    mapping = build_g1_mirror_map()
    result = dict(schema="reference-rate-read-only-data-audit-v1", device="cpu", network_updates=0,
                  changed_data_files=0,
                  assets=data_audit(args.source, args.cache, args.old_project / "src/assets/motions/g1/amp", mapping),
                  loaders=loader_audit(args.source, args.cache, args.old_project / "rsl_rl/utils/motion_loader.py", mapping, args.samples))
    run_manifest = args.run_dir / "motion_dataset.json"
    if run_manifest.is_file():
        saved = json.loads(run_manifest.read_text())
        current = json.loads((args.cache / "mirror_manifest.json").read_text())
        result["current_training_run_dataset_manifest"] = {
            "path": str(run_manifest), "sha256": sha(run_manifest),
            "exact_matches_current_mirror_manifest": saved == current,
            "source_sha256_matches": saved.get("source_sha256") == current.get("source_sha256"),
            "output_sha256_matches": saved.get("output_sha256") == current.get("output_sha256"),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "counts": result["assets"]["counts"],
                      "fps_values": result["assets"]["all_cached_fps_values"],
                      "cache_copy_exact": result["assets"]["all_original_copies_byte_equal"],
                      "old_original_assets_exact": result["assets"]["all_old_project_assets_byte_equal"],
                      "grounded_full_pools_equal": result["loaders"]["grounded_ppo_wmac_full_pools_exact_equal"],
                      "seeded_samples_equal": result["loaders"]["grounded_ppo_wmac_seeded_first_second_exact_equal"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
