"""Pinned Hiking reference conversion and conservative walk/turn selection.

The public Hiking NPZ provides joint/root poses, not camera, terrain or
contact labels. Foot support and height normalization below are explicitly
kinematic proxies, not recovered ground-truth contacts or terrain geometry.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import shutil
import urllib.request
import zipfile

import mujoco
import numpy as np

from src.amp_features import G1_FOOT_REFERENCE_OFFSET
from .mirroring import DEFAULT_G1_XML, build_g1_mirror_map, validate_motion


HIKING_PROJECT_URL = "https://project-instinct.github.io/hiking-in-the-wild/"
HIKING_DOWNLOAD_URL = "https://drive.usercontent.google.com/download?id=10tQylYHdKLVDVnmVHrygLB70neHnPdoF&export=download"
HIKING_ARCHIVE_SHA256 = "bbba90bdb77bbab19e74cbe9f556ea0b9330ea9d125ad8271d431aa0e37a6609"
HIKING_MOTION_SHA256 = "7cfb7c1dcaa6f2a55a13c4849be9e17b4c960ce4015c500ac0ddfb9d77f4ba5b"
HIKING_MOTION_MEMBER = "data&model/parkour_motion_reference/parkour_motion_without_run_retargetted.npz"
HIKING_REFERENCE_COMMIT = "de6b92f01eb155d51f30e8ed1ea7aecd2eb1f057"
DEFAULT_HIKING_SOURCE = Path(__file__).resolve().parents[2]/".perceptive_cache/sources/hiking"
HIKING_BUNDLED_SOURCE = Path(__file__).resolve().parents[1]/"assets/motions/g1/hiking_official"
HIKING_SOURCE_SHA256 = {
    "parkour_motion_without_run_retargetted.npz": HIKING_MOTION_SHA256,
    "parkour_motion_without_run.yaml": "f79e5bbc9207976e1610459ab3727a9e1da6d5c0c6cc75793dcec34b81cb7679",
    "README.md": "ef5f61f763d5fa16f4a6f9598086b5ed624863a8f6ee151ef2439af13efb51ce",
}


@dataclass(frozen=True)
class LocomotionFilter:
    max_forward_speed: float = 1.1
    min_forward_speed: float = .08
    max_lateral_speed: float = .12
    turn_translation_speed: float = .12
    min_turn_rate: float = .10
    minimum_up_z: float = .90
    minimum_pelvis_height: float = .55
    maximum_pelvis_height: float = .95
    support_world_speed: float = .25
    support_height_tolerance: float = .035
    smoothing_seconds: float = .20
    minimum_segment_seconds: float = .40


def verify_hiking_source_files(directory: Path | str) -> Path:
    """Check all training-source bytes against fixed upstream file digests."""
    directory = Path(directory).expanduser().resolve()
    for name, expected in HIKING_SOURCE_SHA256.items():
        path = directory/name
        if not path.is_file():
            raise FileNotFoundError(f"Missing Hiking source file: {path}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Hiking source SHA256 differs from the pinned release (modified): {path}")
    return directory/Path(HIKING_MOTION_MEMBER).name


def ensure_hiking_source(directory: Path | str = DEFAULT_HIKING_SOURCE) -> Path:
    """Prepare the exact source files from cache, repository bundle or download.

    A normal checkout contains the three training files and needs no external
    dataset network request. Download remains a fallback for source-only
    installations lacking the bundle. Existing files are never overwritten.
    """
    directory = Path(directory).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    archive = directory/"hiking-in-the-wild_Data_Model.zip"
    if not archive.exists():
        if all((directory/name).is_file() for name in HIKING_SOURCE_SHA256):
            return verify_hiking_source_files(directory)
        bundle = Path(HIKING_BUNDLED_SOURCE)
        if bundle.is_dir():
            verify_hiking_source_files(bundle)
            for name in HIKING_SOURCE_SHA256:
                target = directory/name
                if target.exists() and target.read_bytes() != (bundle/name).read_bytes():
                    raise ValueError(f"Cached Hiking source was modified: {target.name}")
                if not target.exists():
                    shutil.copy2(bundle/name, target)
            print("[hiking-data] Using verified official walking data bundled with the repository", flush=True)
            return verify_hiking_source_files(directory)
    if not archive.exists():
        print(f"[locomotion-data] Downloading Hiking source: {HIKING_PROJECT_URL}", flush=True)
        with urllib.request.urlopen(HIKING_DOWNLOAD_URL, timeout=60) as response:
            data = response.read(20*1024*1024+1)
        if len(data)>20*1024*1024:
            raise ValueError("Hiking archive exceeds the pinned package size limit")
        if hashlib.sha256(data).hexdigest()!=HIKING_ARCHIVE_SHA256:
            raise ValueError("Hiking archive SHA256 differs from the pinned public release")
        temporary=archive.with_suffix('.download');temporary.write_bytes(data);temporary.replace(archive)
    if hashlib.sha256(archive.read_bytes()).hexdigest()!=HIKING_ARCHIVE_SHA256:
        raise ValueError("Cached Hiking archive was modified; use a fresh source directory")
    names=(HIKING_MOTION_MEMBER,"data&model/parkour_motion_reference/parkour_motion_without_run.yaml","data&model/README.md")
    with zipfile.ZipFile(archive) as package:
        for name in names:
            content=package.read(name);target=directory/Path(name).name
            if target.exists() and target.read_bytes()!=content:
                raise ValueError(f"Cached Hiking source was modified: {target.name}")
            if not target.exists():target.write_bytes(content)
    return verify_hiking_source_files(directory)


def quaternion_multiply(first,second):
    w1,v1=first[...,:1],first[...,1:];w2,v2=second[...,:1],second[...,1:]
    return np.concatenate((w1*w2-(v1*v2).sum(-1,keepdims=True),w1*v2+w2*v1+np.cross(v1,v2)),-1)


def rotate_vector(q,v):
    return v+2*np.cross(q[...,1:],np.cross(q[...,1:],v)+q[...,:1]*v)


def angular_velocity(quaternions,dt):
    inverse=quaternions[:-1].copy();inverse[...,1:]*=-1
    delta=quaternion_multiply(quaternions[1:],inverse)
    delta=np.where(delta[...,:1]<0,-delta,delta)
    size=np.linalg.norm(delta[...,1:],axis=-1,keepdims=True)
    value=2*np.arctan2(size,delta[...,:1])*delta[...,1:]/np.maximum(size,1e-12)/dt
    return np.concatenate((value,value[-1:]),axis=0)


def motion_velocities(motion):
    dt=1/float(np.asarray(motion["fps"]).reshape(-1)[0])
    def difference(value):
        velocity=np.diff(value,axis=0)/dt
        return np.concatenate((velocity,velocity[-1:]),axis=0).astype(np.float32)
    motion["joint_vel"]=difference(motion["joint_pos"])
    motion["body_lin_vel_w"]=difference(motion["body_pos_w"])
    motion["body_ang_vel_w"]=angular_velocity(motion["body_quat_w"],dt).astype(np.float32)
    return motion


def interpolate_hiking_motion(positions, quaternions, joints, fps=50.):
    """Port InstinctLab motion_interpolate_bilinear and quat_slerp_batch.

    The reference uses an exclusive target timeline even at 50→50 Hz. It
    therefore loads N−1 frames from the public N-frame file; this is the
    author's resampling convention, with no motion-content selection.
    """
    import torch
    p, q, j = (torch.as_tensor(value, dtype=torch.float32) for value in (positions, quaternions, joints))
    index = torch.arange(np.ceil((len(p)-1)/fps*50.)).float()/50.*fps
    index = index[index <= len(p)-1]
    front, back = index.floor().long(), index.ceil().long()
    ratio = index-front.float()
    def linear(value):
        return ratio[:, None]*value[back]+(1-ratio[:, None])*value[front]
    first, second = q[front], q[back]
    dot = (first*second).sum(-1, keepdim=True).clamp(-1, 1)
    second = torch.where(dot < 0, -second, second)
    dot = dot.abs()
    theta = torch.acos(dot)
    sine = torch.sin(theta)
    similar = (dot > (1-1e-9)) | (theta.abs() < 1e-9)
    sine = torch.where(sine == 0, torch.ones_like(sine), sine)
    s1, s2 = torch.sin((1-ratio[:, None])*theta)/sine, torch.sin(ratio[:, None]*theta)/sine
    quaternion = (s1*first+s2*second)*(~similar)+similar*first
    quaternion = torch.nn.functional.normalize(quaternion, p=2, dim=-1, eps=1e-9)
    return linear(p).numpy(), quaternion.numpy(), linear(j).numpy()


def convert_hiking_motion(path: Path | str, *, xml_path=DEFAULT_G1_XML,
                          velocity_protocol="legacy_repeat_last"):
    """Map named joints and the author's torso root to native pelvis-root FK.

    Official G1ParkourRoughEnvCfg uses a torso-root URDF and torso_link motion
    reference. The inverse torso transform must be applied before current-MJCF
    FK; copying base_pos_w into pelvis would add a systematic pose error.
    """
    with np.load(path,allow_pickle=False) as data:
        required={"framerate","joint_names","joint_pos","base_pos_w","base_quat_w"}
        if set(data.files)!=required:raise ValueError("Unexpected Hiking motion schema")
        values={key:np.array(data[key],copy=True) for key in data.files}
    mapping=build_g1_mirror_map(xml_path)
    names=values["joint_names"].tolist()
    if len(set(names))!=29 or set(names)!=set(mapping.joint_names):raise ValueError("Hiking joint-name mapping differs from G1")
    joints=values["joint_pos"][:,[names.index(name) for name in mapping.joint_names]]
    positions,quaternions=values["base_pos_w"],values["base_quat_w"]
    fps=float(values["framerate"])
    if fps!=50. or joints.shape!=(len(positions),29) or quaternions.shape!=(len(positions),4):
        raise ValueError("Expected the pinned 50Hz G1 Hiking pose array")
    if not all(np.isfinite(v).all() for v in (joints,positions,quaternions)) or not np.allclose(np.linalg.norm(quaternions,axis=-1),1.,atol=2e-4):
        raise ValueError("Invalid Hiking poses")
    if velocity_protocol == "hiking_frontward_v1":
        positions, quaternions, joints = interpolate_hiking_motion(positions, quaternions, joints, fps)
    model=mujoco.MjModel.from_xml_path(str(xml_path));data=mujoco.MjData(model)
    torso=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,"torso_link")
    all_positions=[];all_quaternions=[]
    maximum_torso_position_error=maximum_torso_quaternion_error=0.
    for joint,position,quaternion in zip(joints,positions,quaternions):
        data.qpos[:3]=0.;data.qpos[3:7]=(1.,0.,0.,0.);data.qpos[7:]=joint;mujoco.mj_forward(model,data)
        local_position=data.xpos[torso].copy();inverse=data.xquat[torso].copy();inverse[1:]*=-1
        root_q=quaternion_multiply(quaternion,inverse)
        data.qpos[:3]=position-rotate_vector(root_q,local_position)
        data.qpos[3:7]=root_q;mujoco.mj_forward(model,data)
        maximum_torso_position_error=max(maximum_torso_position_error,float(np.abs(data.xpos[torso]-position).max()))
        maximum_torso_quaternion_error=max(maximum_torso_quaternion_error,1-abs(float(data.xquat[torso]@quaternion)))
        all_positions.append(data.xpos[1:].copy());all_quaternions.append(data.xquat[1:].copy())
    motion={"fps":np.asarray([fps],dtype=np.float32),"joint_pos":joints.astype(np.float32),
            "body_pos_w":np.asarray(all_positions,dtype=np.float32),"body_quat_w":np.asarray(all_quaternions,dtype=np.float32)}
    motion_velocities(motion)
    if velocity_protocol == "hiking_frontward_v1":
        # InstinctLab estimate_velocity/estimate_angular_velocity("frontward")
        # clamp the successor index at the sequence end, yielding zero there.
        for key in ("joint_vel", "body_lin_vel_w", "body_ang_vel_w"):
            motion[key][-1] = 0.
    elif velocity_protocol != "legacy_repeat_last":
        raise ValueError("Unsupported Hiking velocity protocol")
    validate_motion(motion,mapping)
    return motion,{"source_base_link":"torso_link","destination_base_link":"pelvis",
                   "joint_mapping":"Exact joint-name permutation; current G1 MJCF forward kinematics",
                   "maximum_torso_position_reconstruction_error_m":maximum_torso_position_error,
                   "maximum_torso_quaternion_dot_error":maximum_torso_quaternion_error,
                   "velocity_method":("Forward finite difference; world-frame quaternion log; final velocity zero (Hiking frontward)"
                       if velocity_protocol == "hiking_frontward_v1" else
                       "Forward finite difference; world-frame quaternion log; final velocity repeats preceding estimate")}


def _smooth(values,width):
    width=max(1,int(width));width+=1-width%2
    padded=np.pad(values,((width//2,width//2),)+((0,0),)*(values.ndim-1),mode='edge')
    return np.stack([padded[i:i+len(values)] for i in range(width)]).mean(0)


def frame_selection(motion,mapping,*,filters=LocomotionFilter()):
    """Classify sustained walking/turning, not instantaneous gait sway.

    Support requires ANY near-stationary sole close to that foot's own local
    lower envelope. A higher planted foot on a stair must not be rejected
    because the other, lower foot is swinging. This is a flight-rejection
    proxy, not contact-force truth. Commands enforce exactly vy=0 and vx>=0.
    """
    fps=float(np.asarray(motion['fps']).reshape(-1)[0]);q=motion['body_quat_w'][:,0]
    w,x,y,z=q.T;yaw=np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))
    velocity=motion['body_lin_vel_w'][:,0]
    local=np.stack((np.cos(yaw)*velocity[:,0]+np.sin(yaw)*velocity[:,1],
                    -np.sin(yaw)*velocity[:,0]+np.cos(yaw)*velocity[:,1]),-1)
    smooth=_smooth(local,round(fps*filters.smoothing_seconds));speed=np.linalg.norm(smooth,axis=-1)
    yaw_rate=_smooth(motion['body_ang_vel_w'][:,0,2],round(fps*filters.smoothing_seconds))
    feet=[mapping.body_names.index(name) for name in ('left_ankle_roll_link','right_ankle_roll_link')]
    sole=motion['body_pos_w'][:,feet]+rotate_vector(motion['body_quat_w'][:,feet],np.asarray(G1_FOOT_REFERENCE_OFFSET))
    height=sole[...,2];lower=height.min(-1)
    span=max(1,round(fps*.12));padded=np.pad(height,((span,span),(0,0)),mode='edge')
    envelope=np.stack([padded[i:i+len(lower)] for i in range(2*span+1)]).min(0)
    sole_velocity=np.diff(sole,axis=0,append=sole[-1:])*fps
    support=((height<=envelope+filters.support_height_tolerance)&
             (np.linalg.norm(sole_velocity,axis=-1)<=filters.support_world_speed)).any(-1)
    relative_height=motion['body_pos_w'][:,0,2]-lower
    upright=(1-2*(x*x+y*y)>=filters.minimum_up_z)&(relative_height>=filters.minimum_pelvis_height)&(relative_height<=filters.maximum_pelvis_height)
    forward=(smooth[:,0]>=filters.min_forward_speed)&(smooth[:,0]<=filters.max_forward_speed)&(np.abs(smooth[:,1])<=filters.max_lateral_speed)
    turn=(speed<=filters.turn_translation_speed)&(np.abs(yaw_rate)>=filters.min_turn_rate)
    # Joins in a concatenated public sequence are not legitimate transitions.
    jump=(np.linalg.norm(np.diff(motion['body_pos_w'][:,0],axis=0),axis=-1)>.12)
    jump|=np.max(np.abs(np.diff(motion['joint_pos'],axis=0)),axis=-1)>.6
    relative=quaternion_multiply(q[1:],q[:-1]*np.asarray([1.,-1.,-1.,-1.]))
    jump|=2*np.arccos(np.minimum(1.,np.abs(relative[:,0])))>.3
    continuous=np.ones(len(q),dtype=bool);continuous[:-1]&=~jump;continuous[1:]&=~jump
    valid=upright&support&continuous&(forward|turn)
    return valid,{"forward":forward,"turn":turn,"support":support,"upright":upright,"continuous":continuous,
                  "support_height":lower,"smoothed_forward_velocity":smooth[:,0],"smoothed_lateral_velocity":smooth[:,1],
                  "smoothed_yaw_rate":yaw_rate,"instantaneous_forward_velocity":local[:,0]}


def contiguous_segments(valid,minimum_frames):
    padded=np.r_[False,np.asarray(valid,dtype=bool),False]
    edges=np.flatnonzero(padded[1:]!=padded[:-1])
    return [(int(start),int(end)) for start,end in zip(edges[::2],edges[1::2]) if end-start>=minimum_frames]


def slice_motion(motion,start,end,*,support_height=None):
    result={key:value.copy() if key=='fps' else value[start:end].copy() for key,value in motion.items()}
    if support_height is not None:
        result['body_pos_w'][...,2]-=support_height[start:end,None]
        # This is a support-relative reference trajectory; recompute velocities
        # rather than pairing shifted positions with inconsistent old speeds.
        motion_velocities(result)
    return result
