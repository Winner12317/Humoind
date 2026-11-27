"""Build an immutable, isolated original+mirrored walking expert dataset."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import json
import shutil
import tempfile
import math

import numpy as np

from .mirroring import DEFAULT_G1_XML, build_g1_mirror_map, mirror_motion, turn_statistics, validate_motion


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
    if manifest.get("format") != "g1-sagittal-mirrored-motions-v1" or manifest.get("status") != "complete":
        raise ValueError("Output is not a completed mirrored-motion dataset")
    expected = manifest["output_sha256"]
    actual = {str(p.relative_to(paths.root)) for group in (paths.walk, paths.recovery) for p in group.glob("*.npz")}
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


def prepare_mirrored_dataset(source_root: Path | str, output_root: Path | str,
                             *, xml_path: Path | str = DEFAULT_G1_XML) -> MotionDatasetPaths:
    """Copy original clips and append one mirrored walking clip per original.

    Recovery files are copied byte-for-byte and never mirrored or dropped.
    Keeping each partner in a separate NPZ makes original/mirror samples 50/50
    under both uniform-clip reset sampling and duration-based expert pairs.
    Existing caches are verified and reused, never overwritten or re-augmented.
    """
    source = Path(source_root).expanduser().resolve()
    output = dataset_paths(output_root)
    if output.root == source or output.root.is_relative_to(source) or source.is_relative_to(output.root):
        raise ValueError("Mirrored output must be separate from the original motion directory")
    mapping = build_g1_mirror_map(xml_path)
    originals = sorted((source/"WalkandRun").glob("*.npz"))
    recovery = sorted((source/"Recovery").glob("*.npz"))
    if not originals or not recovery:
        raise ValueError("Both original WalkandRun and Recovery data are required")
    if any(MIRROR_SUFFIX in path.stem for path in originals):
        raise ValueError("Source already contains mirrored clips; do not mirror twice")
    inputs = {str(path.relative_to(source)): file_sha256(path) for path in (*originals, *recovery)}
    if output.root.exists():
        existing = verify_mirrored_dataset(output.root, source_root=source)
        if existing["robot_mapping"]["g1_xml_sha256"] != mapping.xml_sha256:
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
        manifest = {"format": "g1-sagittal-mirrored-motions-v1", "status": "complete",
                    "source_root": str(source), "output_root": str(output.root),
                    "source_sha256": inputs, "output_sha256": output_hashes,
                    "robot_mapping": mapping.metadata(), "walking_clip_pairs": clips,
                    "walking_original_clips": len(originals), "walking_total_clips": 2*len(originals),
                    "recovery_original_clips": len(recovery), "recovery_copied_byte_for_byte": True,
                    "turn_statistics": stats,
                    "sampling": "Equal original/mirror frame counts and clip partners; recovery remains a separate unmodified group.",
                    "pairing": "Each NPZ remains one contiguous clip. Expert adjacent pairs must stay within the same NPZ.",
                    "limits": ["Mirroring fixes the signed walking expert distribution, not policy/task success.",
                               "G1 XML has a measured 10 micrometre shoulder offset asymmetry; the robot itself is not edited.",
                               "Existing caches and source data are immutable; changed inputs require a new output directory."]}
        (staging/"mirror_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
        staging.rename(output.root)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    verify_mirrored_dataset(output.root, source_root=source)
    return output


LOCOMOTION_FORMAT="g1-forward-walk-turn-motions-v2"


def verify_locomotion_dataset(output_root: Path | str, *, source_root: Path | str | None=None) -> dict:
    """Verify the single-group locomotion cache without touching legacy data."""
    paths=dataset_paths(output_root)
    manifest=json.loads(paths.manifest.read_text())
    if (manifest.get("format")!=LOCOMOTION_FORMAT or manifest.get("status")!="complete"
            or manifest.get("motion_scope")!="forward_walk_turn_v1"):
        raise ValueError("Expected the forward-walk/turn-only motion dataset")
    if paths.recovery.exists():raise ValueError("Locomotion cache must not contain a Recovery directory")
    expected=manifest["output_sha256"]
    actual={str(p.relative_to(paths.root)) for p in paths.root.rglob('*.npz')}
    if actual!=set(expected) or any(not name.startswith('WalkandRun/') for name in actual):
        raise ValueError("Locomotion output file set changed; use a fresh cache directory")
    for name,digest in expected.items():
        if file_sha256(paths.root/name)!=digest:raise ValueError(f"Locomotion output was modified: {name}")
    if source_root is not None:
        source=Path(source_root).expanduser().resolve()
        files={p.name:p for p in (source/'WalkandRun').glob('*.npz')}
        if set(files)!=set(manifest['local_source_sha256']):raise ValueError("Local motion source file set changed")
        for name,digest in manifest['local_source_sha256'].items():
            if file_sha256(files[name])!=digest:raise ValueError(f"Local motion source was modified: {name}")
    return manifest


def prepare_locomotion_dataset(source_root: Path | str, output_root: Path | str, *,
                               xml_path: Path | str=DEFAULT_G1_XML, include_hiking: bool=True,
                               include_grail: bool=True) -> MotionDatasetPaths:
    """Build walking/turning experts from pinned Hiking data and local clips.

    Every retained interval is contiguous in its original sequence. No pair
    crosses a rejected frame, source join, interval or mirror boundary.
    Original assets and the joint-recovery cache are never edited.
    """
    from .locomotion_data import (LocomotionFilter,ensure_hiking_source,convert_hiking_motion,
        frame_selection,contiguous_segments,slice_motion,HIKING_PROJECT_URL,HIKING_DOWNLOAD_URL,
        HIKING_ARCHIVE_SHA256,HIKING_MOTION_SHA256,HIKING_REFERENCE_COMMIT)
    source=Path(source_root).expanduser().resolve();output=dataset_paths(output_root)
    if output.root==source or output.root.is_relative_to(source) or source.is_relative_to(output.root):
        raise ValueError("Locomotion output must be separate from original motion assets")
    mapping=build_g1_mirror_map(xml_path);filters=LocomotionFilter()
    originals=sorted((source/'WalkandRun').glob('*.npz'))
    inputs={p.name:file_sha256(p) for p in originals}
    if output.root.exists():
        manifest=verify_locomotion_dataset(output.root,source_root=source)
        if manifest['robot_mapping']['g1_xml_sha256']!=mapping.xml_sha256 or manifest['filters']!=asdict(filters):
            raise ValueError("Locomotion robot/filter protocol changed; use a fresh cache")
        if bool(manifest['hiking_motion_data_used'])!=bool(include_hiking):
            raise ValueError("Locomotion source protocol changed; use a fresh cache")
        if bool(manifest.get('grail_motion_data_used'))!=bool(include_grail):
            raise ValueError("GRAIL source protocol changed; use a fresh cache")
        return output
    hiking_path=ensure_hiking_source() if include_hiking else None
    if hiking_path is not None and file_sha256(hiking_path)!=HIKING_MOTION_SHA256:
        raise ValueError("Hiking motion differs from the pinned public source")
    candidates=[];source_metadata=[];rejected=[]
    if hiking_path is not None:
        print('[locomotion-data] Converting official Hiking torso-root reference to native G1 pelvis FK',flush=True)
        hiking,conversion=convert_hiking_motion(hiking_path,xml_path=xml_path)
        candidates.append(('hiking_public',hiking_path,hiking,True))
        source_metadata.append(dict(source='official_hiking_public_data',project_url=HIKING_PROJECT_URL,
            download_url=HIKING_DOWNLOAD_URL,archive_sha256=HIKING_ARCHIVE_SHA256,npz_sha256=HIKING_MOTION_SHA256,
            reference_commit=HIKING_REFERENCE_COMMIT,path=str(hiking_path),frames=len(hiking['joint_pos']),conversion=conversion,
            license='InstinctLab code is CC BY-NC 4.0; public Data&Model package has no separate data license text. No claim of permissive data redistribution.',
            labels_available=['joint_names','joint_pos','base_pos_w','base_quat_w','framerate'],
            labels_not_available=['per-clip action names','terrain geometry','contact forces','depth images','complex-terrain task labels']))
    if include_grail:
        from .grail_data import ensure_grail_sources,convert_grail_motion
        for reference in ensure_grail_sources():
            motion,metadata=convert_grail_motion(reference,xml_path=xml_path)
            candidates.append((f'grail_{reference["category"]}',reference['files']['robot'],motion,True))
            source_metadata.append(metadata)
    forbidden=('jog','run','backward','sideway','recovery','fall','getup')
    for path in originals:
        if any(word in path.stem.lower() for word in forbidden):
            rejected.append(dict(source='local',file=path.name,reason='Excluded behavior in source clip name'));continue
        if not any(word in path.stem.lower() for word in ('walk','turn','rotate')):
            rejected.append(dict(source='local',file=path.name,reason='Unclassified local behavior'));continue
        candidates.append(('local_supplement',path,read_motion(path),False))
    output.root.parent.mkdir(parents=True,exist_ok=True)
    staging=Path(tempfile.mkdtemp(prefix=f'.{output.root.name}.build-',dir=output.root.parent))
    clips=[];source_audits=[];selected_motions=[]
    try:
        (staging/'WalkandRun').mkdir()
        for origin,path,motion,normalize in candidates:
            validate_motion(motion,mapping)
            valid,diagnostics=frame_selection(motion,mapping,filters=filters)
            fps=float(np.asarray(motion['fps']).reshape(-1)[0])
            intervals=contiguous_segments(valid,max(2,math.ceil(fps*filters.minimum_segment_seconds)))
            audit=dict(source=origin,file=path.name,input_frames=len(valid),selected_frames=int(valid.sum()),
                retained_frames=sum(end-start for start,end in intervals),retained_intervals=len(intervals),
                rejected_by_condition={key:int((~diagnostics[key]).sum()) for key in ('support','upright','continuous')})
            source_audits.append(audit)
            for index,(start,end) in enumerate(intervals):
                selected=slice_motion(motion,start,end,support_height=diagnostics['support_height'] if normalize else None)
                validate_motion(selected,mapping)
                # Both partners retain the same source frame interval.
                source_stem=(hashlib.sha256(path.name.encode()).hexdigest()[:12] if origin.startswith('grail_') else path.stem)
                stem=f'{origin}_{source_stem}__segment{index:03d}'
                first=staging/'WalkandRun'/f'{stem}.npz';second=first.with_name(stem+MIRROR_SUFFIX+'.npz')
                reflected=mirror_motion(selected,mapping)
                np.savez_compressed(first,**selected);np.savez_compressed(second,**reflected)
                selected_motions.extend((selected,reflected))
                local=diagnostics['smoothed_forward_velocity'][start:end];side=diagnostics['smoothed_lateral_velocity'][start:end]
                height=diagnostics['support_height'][start:end]
                clips.append(dict(source=origin,file=path.name,source_start_frame=start,source_end_frame_exclusive=end,
                    frames=end-start,original_output=str(first.relative_to(staging)),mirrored_output=str(second.relative_to(staging)),
                    forward_frames=int(diagnostics['forward'][start:end].sum()),turn_frames=int(diagnostics['turn'][start:end].sum()),
                    maximum_smoothed_forward_m_s=float(local.max()),minimum_smoothed_forward_m_s=float(local.min()),
                    maximum_absolute_smoothed_lateral_m_s=float(np.abs(side).max()),
                    source_support_height_range_m=float(np.ptp(height)),height_normalized=normalize,
                    adjacent_pair_boundary_preserved=True))
        if not clips:raise ValueError("No continuous non-running forward-walk/turn intervals passed the motion filter")
        if not any(row['forward_frames'] for row in clips) or not any(row['turn_frames'] for row in clips):
            raise ValueError("Selected motion data must contain both forward walking and in-place turning")
        for path in originals:
            if file_sha256(path)!=inputs[path.name]:raise RuntimeError("A source motion changed during preparation")
        hashes={str(p.relative_to(staging)):file_sha256(p) for p in sorted((staging/'WalkandRun').glob('*.npz'))}
        stats=turn_statistics(selected_motions)
        manifest=dict(format=LOCOMOTION_FORMAT,status='complete',motion_scope='forward_walk_turn_v1',
            source_root=str(source),output_root=str(output.root),local_source_sha256=inputs,output_sha256=hashes,
            robot_mapping=mapping.metadata(),filters=asdict(filters),source_metadata=source_metadata,
            hiking_motion_data_used=include_hiking,hiking_retained_frames=sum(row['frames'] for row in clips if row['source']=='hiking_public'),
            grail_motion_data_used=include_grail,grail_retained_frames=sum(row['frames'] for row in clips if row['source'].startswith('grail_')),
            terrain_reference_categories={category:dict(retained_intervals=sum(row['source']==f'grail_{category}' for row in clips),
                retained_frames=sum(row['frames'] for row in clips if row['source']==f'grail_{category}')) for category in ('stair_p1','slope','curb')} if include_grail else {},
            local_supplement_retained_frames=sum(row['frames'] for row in clips if row['source']=='local_supplement'),
            walking_original_clips=len(clips),walking_total_clips=len(hashes),walking_total_frames=stats['frames'],
            recovery_original_clips=0,expert_recovery_fraction=0.,excluded_behaviors=['recovery','running','lateral locomotion','backward locomotion'],
            clips=clips,source_audits=source_audits,rejected_source_clips=rejected,turn_statistics=stats,
            pair_protocol='Adjacent frames only inside each retained interval; mirror partners remain separate NPZ files.',
            height_normalization='Hiking/GRAIL: subtract per-frame lowest-sole inferred support height from every body position and recompute all velocities. Local flat-ground clips retain original coordinates.',
            limitations=['Selected pose references have no foot-contact force labels. Any-foot low-world-speed/own-height-envelope support filtering is a conservative kinematic proxy.',
                'Support-relative normalization preserves articulated step style but removes absolute terrain elevation and inferred vertical support-frame movement.',
                'Grounded AMP heights share an inferred lowest-sole reference, not a true per-foot terrain map; an upper planted foot on stairs can be represented as apparent clearance.',
                'Sustained backward/lateral behavior is excluded using smoothed body-heading velocities; small instantaneous gait sway is not a sideways/backward command.',
                'Hiking package has no per-motion terrain labels. The three GRAIL supplements have stairs/slope/curb category provenance and paired USD, but no verified per-frame foot contacts.',
                'Complex-terrain interactions and depth observations are generated by the actual training simulator, not supplied by this pose-only reference package.'])
        (staging/'mirror_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
        staging.rename(output.root)
    except BaseException:
        shutil.rmtree(staging,ignore_errors=True);raise
    verify_locomotion_dataset(output.root,source_root=source)
    print(f'[locomotion-data] Built {len(hashes)} walk/turn clips, {stats["frames"]} frames; Recovery=0',flush=True)
    return output
