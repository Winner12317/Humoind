"""Complete official Hiking walking references, without local motion filtering.

InstinctLab's ``filtered_motion_selection_filepath`` selects NPZ filenames from
YAML; it does not impose the old local frame-level walk/support heuristics.
The public non-running sequence is kept intact, including its source joins.
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile

from .locomotion_data import (
    HIKING_PROJECT_URL, HIKING_DOWNLOAD_URL, HIKING_ARCHIVE_SHA256,
    HIKING_MOTION_SHA256, HIKING_REFERENCE_COMMIT, convert_hiking_motion,
    ensure_hiking_source,
)
from .mirroring import DEFAULT_G1_XML, build_g1_mirror_map, mirror_motion, validate_motion, turn_statistics
from .motions import MIRROR_SUFFIX, dataset_paths, file_sha256, read_motion

HIKING_DATASET_FORMAT = "g1-hiking-official-walking-v1"
HIKING_DATA_PROTOCOL = "hiking_walk_v1"


def official_hiking_reference(directory=None, *, prepare=True) -> Path:
    """Verify the pinned release and return its YAML-selected raw torso motion."""
    import yaml
    from .locomotion_data import DEFAULT_HIKING_SOURCE, verify_hiking_source_files

    directory = DEFAULT_HIKING_SOURCE if directory is None else Path(directory)
    source = ensure_hiking_source(directory) if prepare else verify_hiking_source_files(directory)
    selection = source.with_name("parkour_motion_without_run.yaml")
    if yaml.safe_load(selection.read_text()) != {"selected_files": [source.name]}:
        raise ValueError("Hiking YAML does not select the official complete walking reference")
    return source


def verify_hiking_dataset(output_root, *, source_root=None):
    """Verify immutable converted files; no filtering, downloads or repairs."""
    del source_root  # Native local motion assets are not used by this protocol.
    paths = dataset_paths(output_root)
    manifest = json.loads(paths.manifest.read_text())
    if (manifest.get("format") != HIKING_DATASET_FORMAT
            or manifest.get("status") != "complete"
            or manifest.get("motion_data_protocol") != HIKING_DATA_PROTOCOL):
        raise ValueError("Expected the complete official Hiking walking dataset")
    if manifest.get("interpolation_protocol") != "hiking_bilinear_v1":
        raise ValueError("Official Hiking cache lacks its exact resampling protocol")
    if paths.recovery.exists():
        raise ValueError("Official Hiking walking dataset cannot contain Recovery")
    expected = manifest["output_sha256"]
    actual = {str(p.relative_to(paths.root)) for p in paths.root.rglob("*.npz")}
    if actual != set(expected) or len(actual) != 2:
        raise ValueError("Official Hiking output file set changed")
    for name, digest in expected.items():
        if file_sha256(paths.root/name) != digest:
            raise ValueError(f"Official Hiking output was modified: {name}")
    if manifest["hiking_source_sha256"] != HIKING_MOTION_SHA256:
        raise ValueError("Official Hiking source digest differs from the pinned release")
    return manifest


def prepare_hiking_dataset(source_root, output_root, *, xml_path=DEFAULT_G1_XML):
    """Port the official walking file selection, full timeline and symmetry.

    ``source_root`` is accepted for the preparation interface shared with old
    checkpoints. Only the pinned Hiking release is used for new experiments.
    """
    del source_root
    import numpy as np

    output = dataset_paths(output_root)
    mapping = build_g1_mirror_map(xml_path)
    if output.root.exists():
        manifest = verify_hiking_dataset(output.root)
        if manifest["robot_mapping"] != mapping.metadata():
            raise ValueError("Official Hiking robot mapping changed; use a fresh cache")
        return output
    source = ensure_hiking_source()
    if file_sha256(source) != HIKING_MOTION_SHA256:
        raise ValueError("Hiking motion differs from the pinned public release")
    original, conversion = convert_hiking_motion(
        source, xml_path=xml_path, velocity_protocol="hiking_frontward_v1")
    with np.load(source, allow_pickle=False) as raw:
        source_frames = len(raw["joint_pos"])
    mirrored = mirror_motion(original, mapping)
    frames = validate_motion(original, mapping)
    validate_motion(mirrored, mapping)
    fps = float(original["fps"].reshape(-1)[0])
    # Match the author's actual YAML selection, without interpreting the word
    # "filtered" as a request for our former frame-level heuristics.
    selection = source.with_name("parkour_motion_without_run.yaml")
    import yaml
    if yaml.safe_load(selection.read_text()) != {"selected_files": [source.name]}:
        raise ValueError("Hiking YAML file selection differs from the pinned walking release")
    output.root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.root.name}.build-", dir=output.root.parent))
    try:
        walk = staging/"WalkandRun"
        walk.mkdir()
        name = "hiking_official_walk.npz"
        mirror_name = Path(name).stem+MIRROR_SUFFIX+".npz"
        np.savez_compressed(walk/name, **original)
        np.savez_compressed(walk/mirror_name, **mirrored)
        hashes = {"WalkandRun/"+p.name: file_sha256(p) for p in sorted(walk.glob("*.npz"))}
        manifest = dict(
            format=HIKING_DATASET_FORMAT, status="complete", motion_scope="forward_walk_turn_v1",
            motion_data_protocol=HIKING_DATA_PROTOCOL, output_root=str(output.root),
            robot_mapping=mapping.metadata(), output_sha256=hashes,
            hiking_source_sha256=HIKING_MOTION_SHA256,
            hiking_motion_data_used=True, grail_motion_data_used=False,
            hiking_retained_frames=frames, grail_retained_frames=0,
            local_supplement_retained_frames=0,
            walking_original_clips=1, walking_total_clips=2,
            walking_original_frames=frames, walking_total_frames=2*frames,
            source_frames=source_frames, source_transition_seconds=(source_frames-1)/fps,
            walking_original_transition_seconds=(frames-1)/fps,
            walking_original_frame_count_seconds=frames/fps,
            recovery_original_clips=0, expert_recovery_fraction=0.,
            filters=None, frame_selection="Full official walking-only YAML-selected NPZ through the author's resampling convention",
            height_normalization="None; preserve the author's full world trajectory and vertical velocities",
            velocity_protocol="hiking_frontward_v1",
            interpolation_protocol="hiking_bilinear_v1",
            real_reset="Default robot state with the official random pose/velocity offsets; independent of AMP references",
            amp_protocol="Hiking 67 features x 10 frames; no inferred floor/contact/foot-height features",
            source_metadata=[dict(source="official_hiking_public_data", project_url=HIKING_PROJECT_URL,
                download_url=HIKING_DOWNLOAD_URL, archive_sha256=HIKING_ARCHIVE_SHA256,
                npz_sha256=HIKING_MOTION_SHA256, reference_commit=HIKING_REFERENCE_COMMIT,
                path=str(source), selection_yaml=str(selection), selection_sha256=file_sha256(selection),
                frames=frames, source_frames=source_frames, fps=fps, conversion=conversion)],
            clips=[dict(source="hiking_public", file=source.name, source_start_frame=0,
                source_end_frame_exclusive=frames, frames=frames, fps=fps,
                original_output="WalkandRun/"+name, mirrored_output="WalkandRun/"+mirror_name)],
            turn_statistics=turn_statistics([original, mirrored]),
            terrain_reference_categories={},
            limitations=[
                "The author labels this release walking-only; local instantaneous velocity, support and pose heuristics do not relabel or remove its frames.",
                "Terrain traversal is learned through task rewards and online interaction; there is no per-terrain demonstration quota.",
                "The public reference has no internal clip boundaries or foot-contact truth. The original file timeline, including internal joins, is preserved as in the upstream loader.",
                "Joint names and torso-root poses are converted to the local pelvis-root MJCF; local robot assets differ from the author's URDF.",
            ])
        (staging/"mirror_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
        staging.rename(output.root)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    verify_hiking_dataset(output.root)
    print(f"[hiking-data] Full official walking data: {frames} frames, {(frames-1)/fps:.2f}s + mirror; no local frame filter", flush=True)
    return output
