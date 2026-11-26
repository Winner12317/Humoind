"""Three pinned terrain-only GRAIL references; no full-dataset download.

The selected files carry terrain-category provenance and paired static USD
assets. Their metadata has no foot-contact labels: stance remains explicitly
inferred from kinematics, never promoted to a ground-truth contact annotation.
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import urllib.parse
import urllib.request

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation,Slerp

from .locomotion_data import motion_velocities
from .mirroring import DEFAULT_G1_XML,build_g1_mirror_map,validate_motion


GRAIL_REPOSITORY="nvidia/PhysicalAI-Robotics-Locomanipulation-GRAIL"
GRAIL_REVISION="1ef0e1c0319f79c927941a5c1828f635837a2f30"
GRAIL_CODE_COMMIT="dcef95d2033a1224bdafa57204ce90bb33a85e96"
DEFAULT_GRAIL_SOURCE=Path(__file__).resolve().parents[2]/'.perceptive_cache/sources/grail'
_STAIR="terrain_stairs__aged_terracotta_tile_straight_single_flight_staircase_with_12_warm_orange_steps_in_one_straight_run_no_handrails_no_curves_no_turns__0000"
# robot, metadata, object transforms, and corresponding immutable terrain USD.
GRAIL_SAMPLES=(
 ('stair_p1',_STAIR,('769d1e123e3b09a8266d98e5c08d88da7c3b50da312d6760720cf5ce892997e5','7d88be608419afd2be127e4ac4c5a8aa1b4863fdf84e86c5ec93356881ee861d','963e966e220f915b06a5ee15af20c7a2b19a94bb5d72c3225459b5466522718e','b4299498a2ae103a5a100d7389854f1cd4748c547d5b9e096a45ba443e820b25')),
 ('slope','terrain_slopes__slope_000__000',('b77480d5f8f3339a3064276d6f9d443ac3a3456f20eb9195d48add176e561ee1','7d88be608419afd2be127e4ac4c5a8aa1b4863fdf84e86c5ec93356881ee861d','efbcdbbdd178642703e36889147fb3335fb07ade0c0d68f6dcaae48a5b6d19ab','8d1e696fb5bd2aecfa17797549bddd001093b060773cb313185a6a46db7eb5a5')),
 ('curb','terrain_curbs__curb_000__000',('72d78a31b58323a2b7dabea9ff2f07e5f285019ed3ad506461773c5ecd5b8c0a','7d88be608419afd2be127e4ac4c5a8aa1b4863fdf84e86c5ec93356881ee861d','459c733c1f9f3dcecc19e940d86f4a3a4685ce783c0e1c69c386550ed23f728d','72397ff1fc5423bd96c6784e2f30aeacd8e5f7f7984337e134674ef80f045d5e')))
# Official GRAIL G1 MJCF depth-first hinge order (code commit above), not
# IsaacLab's reordered articulation columns. pose_aa axis checks verify it.
GRAIL_JOINT_NAMES=(
 'left_hip_pitch_joint','left_hip_roll_joint','left_hip_yaw_joint','left_knee_joint','left_ankle_pitch_joint','left_ankle_roll_joint',
 'right_hip_pitch_joint','right_hip_roll_joint','right_hip_yaw_joint','right_knee_joint','right_ankle_pitch_joint','right_ankle_roll_joint',
 'waist_yaw_joint','waist_roll_joint','waist_pitch_joint',
 'left_shoulder_pitch_joint','left_shoulder_roll_joint','left_shoulder_yaw_joint','left_elbow_joint','left_wrist_roll_joint','left_wrist_pitch_joint','left_wrist_yaw_joint',
 'right_shoulder_pitch_joint','right_shoulder_roll_joint','right_shoulder_yaw_joint','right_elbow_joint','right_wrist_roll_joint','right_wrist_pitch_joint','right_wrist_yaw_joint')


def ensure_grail_sources(directory=DEFAULT_GRAIL_SOURCE):
    directory=Path(directory).expanduser().resolve();result=[]
    for category,stem,digests in GRAIL_SAMPLES:
        files={};records=[]
        for group,digest in zip(('robot','meta','objects','object_usd'),digests):
            suffix='.usd' if group=='object_usd' else '.pkl'
            relative=f'data/{category}/{group}/{stem}{suffix}'
            target=directory/relative
            url=f'https://huggingface.co/datasets/{GRAIL_REPOSITORY}/resolve/{GRAIL_REVISION}/'+urllib.parse.quote(relative,safe='/')
            if not target.exists():
                print(f'[locomotion-data] Downloading one GRAIL {category} {group} reference',flush=True)
                target.parent.mkdir(parents=True,exist_ok=True)
                with urllib.request.urlopen(url,timeout=60) as response:content=response.read(4*1024*1024+1)
                if len(content)>4*1024*1024 or hashlib.sha256(content).hexdigest()!=digest:
                    raise ValueError(f'GRAIL source size/SHA256 mismatch: {relative}')
                temporary=target.with_suffix(target.suffix+'.download');temporary.write_bytes(content);temporary.replace(target)
            if hashlib.sha256(target.read_bytes()).hexdigest()!=digest:raise ValueError(f'Cached GRAIL source modified: {relative}')
            files[group]=target;records.append(dict(path=relative,sha256=digest,url=url,bytes=target.stat().st_size))
        result.append(dict(category=category,files=files,source_records=records))
    return result


def load_numeric_joblib(path):
    """Decode only numeric joblib arrays and primitive containers, no globals.

    The pinned source hashes are verified before this function is used by the
    dataset builder. Object arrays are refused because joblib otherwise uses
    an unrestricted nested pickle loader for their payloads.
    """
    from joblib.numpy_pickle import NumpyArrayWrapper,NumpyUnpickler
    class NumericArrayWrapper(NumpyArrayWrapper):
        def read(self,unpickler,ensure_native_byte_order):
            if self.dtype.hasobject or self.subclass is not np.ndarray:
                raise ValueError('GRAIL source must contain only plain numeric arrays')
            if np.prod(self.shape)>10_000_000:raise ValueError('GRAIL numeric array exceeds the small-reference budget')
            return super().read(unpickler,ensure_native_byte_order)
    class NumericUnpickler(NumpyUnpickler):
        def find_class(self,module,name):
            if (module,name)==('joblib.numpy_pickle','NumpyArrayWrapper'):return NumericArrayWrapper
            if (module,name) in {('numpy','ndarray'),('numpy','dtype'),('numpy.core.multiarray','scalar'),('numpy._core.multiarray','scalar')}:
                return super().find_class(module,name)
            raise ValueError(f'Unsupported GRAIL pickle global: {module}.{name}')
    path=Path(path)
    with path.open('rb') as stream:return NumericUnpickler(str(path),stream,ensure_native_byte_order=True).load()


def resample_grail_pose(record,target_fps=50.):
    source_fps=float(record['fps']);position=np.asarray(record['root_trans_offset']);joint=np.asarray(record['dof']);quaternion=np.asarray(record['root_rot'])
    if source_fps!=25. or target_fps!=50. or joint.shape!=(len(position),29) or quaternion.shape!=(len(position),4) or len(position)<2:
        raise ValueError('Expected the pinned 25Hz GRAIL G1 trajectory')
    if not all(np.isfinite(x).all() for x in (position,joint,quaternion)):raise ValueError('Nonfinite GRAIL pose')
    if not np.allclose(np.linalg.norm(quaternion,axis=-1),1.,atol=2e-4):raise ValueError('Invalid GRAIL XYZW quaternion')
    source_time=np.arange(len(position))/source_fps
    target_time=np.arange(2*(len(position)-1)+1)/target_fps
    interpolate=lambda values:np.stack([np.interp(target_time,source_time,values[:,i]) for i in range(values.shape[-1])],-1)
    root_quat_xyzw=Slerp(source_time,Rotation.from_quat(quaternion))(target_time).as_quat()
    return interpolate(position),root_quat_xyzw[:,[3,0,1,2]],interpolate(joint),dict(
        source_fps=source_fps,target_fps=target_fps,source_frames=len(source_time),target_frames=len(target_time),
        first_to_last_seconds=float(source_time[-1]),translation_and_joints='Linear interpolation',orientation='Quaternion SLERP XYZW -> WXYZ')


def convert_grail_motion(source,*,xml_path=DEFAULT_G1_XML):
    """Use GRAIL's native pelvis root and verify dof/axis-angle conventions."""
    files=source['files'];robot=load_numeric_joblib(files['robot']);objects=load_numeric_joblib(files['objects']);meta=load_numeric_joblib(files['meta'])
    if not isinstance(robot,dict) or len(robot)!=1 or set(robot)!=set(objects):raise ValueError('GRAIL robot/object sequence identity mismatch')
    key=next(iter(robot));record=robot[key];obj=objects[key]
    for hand in ('contact_points_left_hand','contact_points_right_hand'):
        if any(np.asarray(points).size for points in obj.get(hand,{}).values()):raise ValueError('Reject manipulation/contact-hand GRAIL reference')
    model=mujoco.MjModel.from_xml_path(str(xml_path));mapping=build_g1_mirror_map(xml_path)
    if set(GRAIL_JOINT_NAMES)!=set(mapping.joint_names):raise ValueError('GRAIL joint set differs from native G1')
    source_to_native=[GRAIL_JOINT_NAMES.index(name) for name in mapping.joint_names]
    axes=model.jnt_axis[np.flatnonzero(model.jnt_type==mujoco.mjtJoint.mjJNT_HINGE)]
    source_axes=axes[[mapping.joint_names.index(name) for name in GRAIL_JOINT_NAMES]]
    pose_aa=np.asarray(record['pose_aa'])
    if pose_aa.shape!=(len(record['dof']),30,3) or not np.allclose(pose_aa[:,1:],record['dof'][...,None]*source_axes,atol=1e-6):
        raise ValueError('GRAIL DOF order/hinge axes disagree with source pose_aa')
    quaternion_check=Rotation.from_rotvec(pose_aa[:,0]).as_quat()
    if not np.allclose(np.abs((quaternion_check*record['root_rot']).sum(-1)),1.,atol=2e-5):
        raise ValueError('GRAIL root quaternion convention disagrees with pose_aa')
    position,quaternion,joint,resampling=resample_grail_pose(record)
    data=mujoco.MjData(model);positions=[];quaternions=[]
    for p,q,j in zip(position,quaternion,joint):
        data.qpos[:3]=p;data.qpos[3:7]=q;data.qpos[7:]=j[source_to_native];mujoco.mj_forward(model,data)
        positions.append(data.xpos[1:].copy());quaternions.append(data.xquat[1:].copy())
    motion=motion_velocities(dict(fps=np.asarray([50.],dtype=np.float32),joint_pos=joint[:,source_to_native].astype(np.float32),
        body_pos_w=np.asarray(positions,dtype=np.float32),body_quat_w=np.asarray(quaternions,dtype=np.float32)))
    validate_motion(motion,mapping)
    object_position=np.asarray(obj['root_pos']);object_quaternion=np.asarray(obj['root_quat'])
    metadata=dict(source='GRAIL_terrain',terrain_category=source['category'],dataset=f'https://huggingface.co/datasets/{GRAIL_REPOSITORY}',
        revision=GRAIL_REVISION,code_commit=GRAIL_CODE_COMMIT,source_records=source['source_records'],license='CC BY-NC 4.0',
        source_sequence_key=key,source_base_link='pelvis',source_quaternion_order='XYZW',destination_quaternion_order='WXYZ',
        source_joint_names=list(GRAIL_JOINT_NAMES),joint_mapping='Named native-MJCF permutation, independently checked against pose_aa hinge axes',
        resampling=resampling,terrain_asset=str(files['object_usd']),terrain_object_position_first=object_position[0].tolist(),
        terrain_object_quaternion_first_wxyz=object_quaternion[0].tolist(),terrain_object_scale=np.asarray(obj['scale']).tolist(),
        terrain_object_translation_range_m=np.ptp(object_position,axis=0).tolist(),metadata=meta,hand_contact_points=0,
        foot_contact_labels_available=False,terrain_mesh_used_for_expert_ground_heights=False,
        terrain_pose_protocol='Paired object transform/scale and USD retained for provenance; raw pelvis poses stay in source world before explicit support-relative normalization.',
        limitations=['Selected source meta has no foot-contact truth. Support remains a world-velocity and per-foot temporal-height proxy.',
                     'Paired USD is not imported as a per-foot AMP ground oracle. Lowest-sole normalization is an approximation, not exact mesh-relative grounding.'])
    return motion,metadata
