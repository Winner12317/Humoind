"""Full Hiking references plus an explicitly sourced legacy G1 standing clip.

The supplemental CSV is an existing locked pose from the user's old G1
framework. It is never extracted by filtering the author's walking timeline.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
import tempfile

import mujoco
import numpy as np

from .hiking_data import HIKING_DATA_PROTOCOL, prepare_hiking_dataset
from .locomotion_data import HIKING_MOTION_SHA256
from .mirroring import DEFAULT_G1_XML, build_g1_mirror_map, mirror_motion, validate_motion, turn_statistics
from .motions import MIRROR_SUFFIX, dataset_paths, file_sha256, read_motion


STANDING_EXPERT_SOURCE = "legacy_g1_locked_v1"
STANDING_DATASET_FORMAT = "g1-hiking-walking-plus-legacy-standing-v1"
STANDING_BUNDLE = Path(__file__).resolve().parents[1]/"assets/motions/g1/standing_legacy"
STANDING_CSV_NAME = "stand_still_locked_001.csv"
STANDING_CSV_SHA256 = "a9f812567fb714ac0296d2611b3a5745d727ca5971225e833a0ddeac453aa3d6"
STANDING_OUTPUT_NAME = "legacy_g1_stand_still_locked_001.npz"
STANDING_FRAMES = 1201
STANDING_FPS = 50.
# Exact column order in the archived G1 scripts/csv_to_npz.py:480-510.
CSV_JOINT_NAMES = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)


def verify_standing_source(source_directory=None):
    directory=STANDING_BUNDLE if source_directory is None else Path(source_directory)
    csv=directory/STANDING_CSV_NAME
    if file_sha256(csv)!=STANDING_CSV_SHA256:
        raise ValueError("Legacy G1 standing source was modified")
    metadata=json.loads((directory/"SOURCE_MANIFEST.json").read_text())
    if (metadata.get("sha256")!=STANDING_CSV_SHA256 or metadata.get("file")!=STANDING_CSV_NAME
            or metadata.get("source_fps") is not None
            or metadata.get("output_representation_fps")!=STANDING_FPS):
        raise ValueError("Legacy standing source provenance differs from its pinned representation")
    return metadata


def convert_legacy_g1_standing(path=None, *, xml_path=DEFAULT_G1_XML):
    """Map the unchanged CSV by joint name and apply current G1 FK once.

    All 1201 source rows are identical. Their unknown original sample rate is
    not inferred: output is explicitly a 50 Hz representation of that pose.
    """
    path=STANDING_BUNDLE/STANDING_CSV_NAME if path is None else Path(path)
    if file_sha256(path)!=STANDING_CSV_SHA256:
        raise ValueError("Legacy G1 standing source was modified")
    raw=np.loadtxt(path,delimiter=",")
    if raw.shape!=(STANDING_FRAMES,36) or not np.isfinite(raw).all():
        raise ValueError("Unexpected legacy G1 standing CSV layout")
    if not np.array_equal(raw,np.broadcast_to(raw[:1],raw.shape)):
        raise ValueError("Legacy standing source must be the complete locked pose, not filtered moving data")
    mapping=build_g1_mirror_map(xml_path)
    if set(mapping.joint_names)!=set(CSV_JOINT_NAMES):
        raise ValueError("Standing source joint names differ from the current G1")
    joints=raw[0,7:][[CSV_JOINT_NAMES.index(name) for name in mapping.joint_names]]
    model=mujoco.MjModel.from_xml_path(str(xml_path))
    data=mujoco.MjData(model)
    hinge=np.flatnonzero(model.jnt_type==mujoco.mjtJoint.mjJNT_HINGE)
    if np.any((joints<model.jnt_range[hinge,0])|(joints>model.jnt_range[hinge,1])):
        raise ValueError("Legacy G1 standing pose exceeds current joint limits")
    data.qpos[:3]=raw[0,:3]
    data.qpos[3:7]=raw[0,[6,3,4,5]]  # CSV xyzw -> MuJoCo wxyz.
    if not np.isclose(np.linalg.norm(data.qpos[3:7]),1.,atol=1e-7):
        raise ValueError("Legacy G1 standing root quaternion is not normalized")
    data.qpos[model.jnt_qposadr[hinge]]=joints
    data.qvel[:]=0.
    mujoco.mj_forward(model,data)
    repeat=lambda value:np.repeat(np.asarray(value,dtype=np.float32)[None],STANDING_FRAMES,axis=0)
    motion=dict(fps=np.array([STANDING_FPS],dtype=np.float32),joint_pos=repeat(joints),
        joint_vel=np.zeros((STANDING_FRAMES,29),dtype=np.float32),
        body_pos_w=repeat(data.xpos[1:]),body_quat_w=repeat(data.xquat[1:]),
        body_lin_vel_w=np.zeros((STANDING_FRAMES,30,3),dtype=np.float32),
        body_ang_vel_w=np.zeros((STANDING_FRAMES,30,3),dtype=np.float32))
    validate_motion(motion,mapping)
    return motion


def verify_hiking_standing_dataset(output_root, *, source_root=None):
    """Read-only verification of the four-clip standing supplement protocol."""
    del source_root
    paths=dataset_paths(output_root)
    manifest=json.loads(paths.manifest.read_text())
    if (manifest.get("format")!=STANDING_DATASET_FORMAT or manifest.get("status")!="complete"
            or manifest.get("motion_data_protocol")!=HIKING_DATA_PROTOCOL
            or manifest.get("standing_expert_source")!=STANDING_EXPERT_SOURCE):
        raise ValueError("Expected complete Hiking walking plus legacy G1 standing dataset")
    if (manifest.get("hiking_source_sha256")!=HIKING_MOTION_SHA256
            or manifest.get("standing_source_sha256")!=STANDING_CSV_SHA256
            or manifest.get("interpolation_protocol")!="hiking_bilinear_v1"):
        raise ValueError("Hiking/standing sources or interpolation differ from the pinned protocol")
    if paths.recovery.exists():
        raise ValueError("Hiking standing supplement cannot contain Recovery")
    originals=("hiking_official_walk.npz",STANDING_OUTPUT_NAME)
    names={"WalkandRun/"+name for original in originals
           for name in (original,Path(original).stem+MIRROR_SUFFIX+".npz")}
    actual={str(path.relative_to(paths.root)) for path in paths.root.rglob("*.npz")}
    if actual!=names or set(manifest["output_sha256"])!=names:
        raise ValueError("Hiking standing output file set changed")
    for name,digest in manifest["output_sha256"].items():
        if file_sha256(paths.root/name)!=digest:
            raise ValueError(f"Hiking standing output was modified: {name}")
    official_names={"WalkandRun/hiking_official_walk.npz",
                    "WalkandRun/hiking_official_walk"+MIRROR_SUFFIX+".npz"}
    if set(manifest["official_walking_output_sha256"])!=official_names:
        raise ValueError("Official Hiking walking provenance must contain the exact original/mirror pair")
    for name,digest in manifest["official_walking_output_sha256"].items():
        if name not in names or manifest["output_sha256"][name]!=digest:
            raise ValueError("Official Hiking walking files were changed by standing supplementation")
    return manifest


def prepare_hiking_standing_dataset(source_root, output_root, *, xml_path=DEFAULT_G1_XML):
    """Keep the complete official walking pair and append legacy stand+mirror.

    No custom expert resampling probability is introduced. Existing uniform
    sampling over valid ten-frame windows determines the supplemental share.
    """
    output=dataset_paths(output_root)
    mapping=build_g1_mirror_map(xml_path)
    source_metadata=verify_standing_source()
    if output.root.exists():
        manifest=verify_hiking_standing_dataset(output.root)
        if manifest["robot_mapping"]!=mapping.metadata():
            raise ValueError("Standing dataset robot mapping changed; use a fresh cache")
        return output
    standing=convert_legacy_g1_standing(xml_path=xml_path)
    mirrored=mirror_motion(standing,mapping)
    validate_motion(mirrored,mapping)
    output.root.parent.mkdir(parents=True,exist_ok=True)
    staging=Path(tempfile.mkdtemp(prefix=f".{output.root.name}.build-",dir=output.root.parent))
    try:
        # Build within temporary storage, leaving every existing official-only
        # cache untouched; copy its two NPZ bytes without re-encoding them.
        official=prepare_hiking_dataset(source_root,staging/"official",xml_path=xml_path)
        manifest=json.loads(official.manifest.read_text())
        official_hashes=copy.deepcopy(manifest["output_sha256"])
        walk=staging/"WalkandRun"
        walk.mkdir()
        for name in official_hashes:
            shutil.copy2(official.root/name,staging/name)
        np.savez_compressed(walk/STANDING_OUTPUT_NAME,**standing)
        mirror_name=Path(STANDING_OUTPUT_NAME).stem+MIRROR_SUFFIX+".npz"
        np.savez_compressed(walk/mirror_name,**mirrored)
        hiking_frames=manifest["walking_original_frames"]
        history_length=10
        stand_windows=2*(STANDING_FRAMES-history_length+1)
        hiking_windows=2*(hiking_frames-history_length+1)
        manifest.update(format=STANDING_DATASET_FORMAT,output_root=str(output.root),
            standing_expert_source=STANDING_EXPERT_SOURCE,standing_source_sha256=STANDING_CSV_SHA256,
            official_walking_output_sha256=official_hashes,
            output_sha256={"WalkandRun/"+p.name:file_sha256(p) for p in sorted(walk.glob("*.npz"))},
            walking_original_clips=2,walking_total_clips=4,
            walking_original_frames=hiking_frames+STANDING_FRAMES,
            walking_total_frames=2*(hiking_frames+STANDING_FRAMES),
            official_walking_original_frames=hiking_frames,
            official_walking_original_transition_seconds=manifest["walking_original_transition_seconds"],
            official_walking_original_frame_count_seconds=manifest["walking_original_frame_count_seconds"],
            walking_original_transition_seconds=manifest["walking_original_transition_seconds"]+(STANDING_FRAMES-1)/STANDING_FPS,
            walking_original_frame_count_seconds=manifest["walking_original_frame_count_seconds"]+STANDING_FRAMES/STANDING_FPS,
            local_supplement_retained_frames=STANDING_FRAMES,
            standing_original_frames=STANDING_FRAMES,standing_total_frames=2*STANDING_FRAMES,
            standing_representation_fps=STANDING_FPS,standing_source_fps=None,
            standing_representation_transition_seconds=(STANDING_FRAMES-1)/STANDING_FPS,
            standing_history_windows=stand_windows,hiking_history_windows=hiking_windows,
            total_history_windows=stand_windows+hiking_windows,
            natural_standing_expert_fraction=stand_windows/(stand_windows+hiking_windows),
            expert_sampling="Unchanged uniform valid ten-frame windows within individual NPZ files; no standing-specific weight or conditional discriminator",
            standing_provenance=source_metadata,
            standing_joint_mapping=dict(csv_joint_names=list(CSV_JOINT_NAMES),
                native_joint_names=list(mapping.joint_names),
                native_from_csv=[CSV_JOINT_NAMES.index(name) for name in mapping.joint_names],
                root_link="pelvis",source_quaternion="xyzw",output_quaternion="wxyz"))
        manifest["turn_statistics"]=turn_statistics([read_motion(path) for path in sorted(walk.glob("*.npz"))])
        manifest["source_metadata"].append(dict(source=STANDING_EXPERT_SOURCE,
            path=str(STANDING_BUNDLE/STANDING_CSV_NAME),sha256=STANDING_CSV_SHA256,
            frames=STANDING_FRAMES,fps=STANDING_FPS,source_fps=None,
            temporal_semantics="Re-expression of an existing identical static pose at 50 Hz; original recording rate/duration unknown",
            provenance=source_metadata))
        manifest["clips"].append(dict(source=STANDING_EXPERT_SOURCE,file=STANDING_CSV_NAME,
            source_start_frame=0,source_end_frame_exclusive=STANDING_FRAMES,
            frames=STANDING_FRAMES,fps=STANDING_FPS,
            original_output="WalkandRun/"+STANDING_OUTPUT_NAME,
            mirrored_output="WalkandRun/"+mirror_name))
        manifest["limitations"].extend([
            "The supplemental standing reference is user-requested legacy G1 project data, not part of the original Hiking YAML dataset.",
            "The source CSV has no frame-rate metadata. Its unchanged static rows are represented at 50 Hz; this does not establish their original recorded duration.",
            "The standing source is a locked pose, not a claim of a dynamic balance demonstration or proof of stable closed-loop standing.",
        ])
        shutil.rmtree(official.root)
        (staging/"mirror_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
        staging.rename(output.root)
    except BaseException:
        shutil.rmtree(staging,ignore_errors=True)
        raise
    verify_hiking_standing_dataset(output.root)
    print(f"[hiking-data] Complete Hiking walking pair + legacy G1 standing pair; {stand_windows/(stand_windows+hiking_windows):.2%} of expert history windows are standing",flush=True)
    return output
