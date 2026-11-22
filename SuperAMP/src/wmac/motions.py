"""Build an immutable, isolated original+mirrored walking expert dataset."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import shutil
import tempfile

import numpy as np

from src.robot_profiles import G1, PROFILES, RobotProfile

from .mirroring import build_mirror_map, mirror_motion, turn_statistics, validate_motion


MIRROR_SUFFIX = "__mirror_y"


@dataclass(frozen=True)
class MotionDatasetPaths:
    root: Path
    walk: Path
    recovery: Path
    manifest: Path


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_motion(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as values:
        return {key: np.array(values[key], copy=True) for key in values.files}


def dataset_paths(root: Path | str) -> MotionDatasetPaths:
    path = Path(root).expanduser().resolve()
    return MotionDatasetPaths(path, path/"WalkandRun", path/"Recovery", path/"mirror_manifest.json")


def verify_mirrored_dataset(output_root: Path | str, *, source_root: Path | str | None = None) -> dict:
    paths = dataset_paths(output_root)
    manifest = json.loads(paths.manifest.read_text())
    formats = {profile.mirrored_motion_format for profile in PROFILES}
    if manifest.get("format") not in formats or manifest.get("status") != "complete":
        raise ValueError("Output is not a completed mirrored-motion dataset")
    expected = manifest["output_sha256"]
    actual = {str(p.relative_to(paths.root)) for p in paths.root.rglob("*.npz")}
    if actual != set(expected):
        raise ValueError("Mirrored output file set changed; use a fresh cache directory")
    for name, digest in expected.items():
        if file_sha256(paths.root/name) != digest:
            raise ValueError(f"Mirrored output was modified: {name}")
    if source_root is not None:
        source = Path(source_root).expanduser().resolve()
        source_files = {str(p.relative_to(source)) for group in ("WalkandRun", "Recovery") for p in (source/group).glob("*.npz")}
        if source_files != set(manifest["source_sha256"]):
            raise ValueError("Source motion file set changed; use a fresh cache directory")
        for name, digest in manifest["source_sha256"].items():
            if file_sha256(source/name) != digest:
                raise ValueError(f"Source motion was modified: {name}; use a fresh cache directory")
    return manifest


def _asymmetry_summary(mapping) -> str:
    """One sentence naming the measured left/right differences of a robot model."""
    measured = mapping.measured_asymmetry()
    angles = measured["hinge_axis_asymmetry_deg"]
    worst = max(angles, key=angles.get)
    partner = mapping.joint_names[mapping.joint_permutation[mapping.joint_names.index(worst)]]
    parts = [f"hinge axes of {worst} and {partner} differ by {angles[worst]:.2f} deg after reflection",
             f"rest offsets by {mapping.maximum_rest_position_asymmetry:.1e} m"]
    for name, entry in measured["asymmetric_joint_ranges"].items():
        low, high = entry["range_rad"]
        parts.append(f"the range of {name} is one-sided ({low:g} to {high:g} rad)")
    return ", ".join(parts)


def prepare_mirrored_dataset(source_root: Path | str, output_root: Path | str,
                             *, xml_path: Path | str | None = None,
                             profile: RobotProfile = G1) -> MotionDatasetPaths:
    """Copy original clips and append one mirrored walking clip per original.

    Recovery files are copied byte-for-byte and never mirrored or dropped.
    Keeping each partner in a separate NPZ makes original/mirror samples 50/50
    under both uniform-clip reset sampling and duration-based expert pairs.
    Existing caches are verified and reused, never overwritten or re-augmented.
    ``profile`` selects the robot (default G1); ``xml_path`` overrides its model.
    """
    source = Path(source_root).expanduser().resolve()
    output = dataset_paths(output_root)
    if output.root == source or output.root.is_relative_to(source) or source.is_relative_to(output.root):
        raise ValueError("Mirrored output must be separate from the original motion directory")
    mapping = build_mirror_map(profile, xml_path)
    xml_key = "g1_xml_sha256" if profile is G1 else "robot_xml_sha256"
    originals = sorted((source/"WalkandRun").glob("*.npz"))
    recovery = sorted((source/"Recovery").glob("*.npz"))
    if not originals or not recovery:
        raise ValueError("Both original WalkandRun and Recovery data are required")
    if any(MIRROR_SUFFIX in path.stem for path in originals):
        raise ValueError("Source already contains mirrored clips; do not mirror twice")
    inputs = {str(path.relative_to(source)): file_sha256(path) for path in (*originals, *recovery)}
    if output.root.exists():
        existing = verify_mirrored_dataset(output.root, source_root=source)
        if existing.get("format") != profile.mirrored_motion_format:
            raise ValueError("The mirrored-motion directory belongs to another robot; use a fresh one")
        if existing["robot_mapping"][xml_key] != mapping.xml_sha256:
            raise ValueError("Robot XML changed; use a fresh mirrored-motion directory")
        return output
    output.root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.root.name}.build-", dir=output.root.parent))
    try:
        (staging/"WalkandRun").mkdir(); (staging/"Recovery").mkdir()
        before, after, clips = [], [], []
        for path in originals:
            original = read_motion(path)
            validate_motion(original, mapping)
            mirrored = mirror_motion(original, mapping)
            target = staging/"WalkandRun"/path.name
            mirrored_name = path.stem+MIRROR_SUFFIX+".npz"
            shutil.copy2(path, target)
            np.savez_compressed(staging/"WalkandRun"/mirrored_name, **mirrored)
            before.append(original); after.extend((original, mirrored))
            clips.append({"source": str(path.relative_to(source)), "original_output": str(target.relative_to(staging)),
                          "mirrored_output": "WalkandRun/"+mirrored_name, "frames": len(original["joint_pos"]),
                          "fps": float(np.asarray(original["fps"]).reshape(-1)[0]),
                          "reflection_plane_y": float(original["body_pos_w"][0, 0, 1]),
                          "clip_boundary_preserved": True})
        for path in recovery:
            validate_motion(read_motion(path), mapping)
            shutil.copy2(path, staging/"Recovery"/path.name)
        for name, digest in inputs.items():
            if file_sha256(source/name) != digest:
                raise RuntimeError("An original motion changed during preparation")
        output_hashes = {str(p.relative_to(staging)): file_sha256(p)
                         for group in ("WalkandRun", "Recovery") for p in sorted((staging/group).glob("*.npz"))}
        stats = {"before": turn_statistics(before), "after": turn_statistics(after)}
        if stats["after"]["positive_yaw_frames"] != stats["after"]["negative_yaw_frames"]:
            raise RuntimeError("Original+mirrored yaw distribution is not balanced")
        manifest = {"format": profile.mirrored_motion_format, "status": "complete",
                    "source_root": str(source), "output_root": str(output.root),
                    "motion_scope": "original_WalkandRun_plus_unmodified_Recovery",
                    "reference_commit": "92ca7bb8a3d8ea959d934fc13b5aae7e30317757",
                    "reference_utility": "src/wmac_perceptive/motions.py:prepare_mirrored_dataset",
                    "source_sha256": inputs, "output_sha256": output_hashes,
                    "robot_mapping": mapping.metadata(), "walking_clip_pairs": clips,
                    "walking_original_clips": len(originals), "walking_total_clips": 2*len(originals),
                    "recovery_original_clips": len(recovery), "recovery_copied_byte_for_byte": True,
                    "turn_statistics": stats,
                    "sampling": "Equal original/mirror frame counts and clip partners; recovery remains a separate unmodified group.",
                    "pairing": "Each NPZ remains one contiguous clip. Expert adjacent pairs must stay within the same NPZ.",
                    "limits": ["Mirroring fixes the signed walking expert distribution, not policy/task success.",
                               ("G1 XML has a measured 10 micrometre shoulder offset asymmetry; the robot itself is not edited."
                                if profile is G1 else
                                f"{profile.name.upper()} XML is not exactly mirror-symmetric: {_asymmetry_summary(mapping)}; "
                                "the measured values are in robot_mapping, mirrored clips are checked against "
                                "the real joint ranges, and the robot itself is not edited."),
                               "Existing caches and source data are immutable; changed inputs require a new output directory."]}
        (staging/"mirror_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
        staging.rename(output.root)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    verify_mirrored_dataset(output.root, source_root=source)
    return output
