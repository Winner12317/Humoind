"""Motion-scope, source conversion, immutable cache and segment regressions."""
from __future__ import annotations

import hashlib
import io
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import zipfile
import pickle

import mujoco
import numpy as np

from src.wmac_perceptive.locomotion_data import (HIKING_MOTION_MEMBER,convert_hiking_motion,
    ensure_hiking_source,frame_selection,contiguous_segments,motion_velocities)
from src.wmac_perceptive.mirroring import DEFAULT_G1_XML,build_g1_mirror_map
from src.wmac_perceptive.motions import prepare_locomotion_dataset,verify_locomotion_dataset,read_motion
from src.wmac_perceptive.grail_data import load_numeric_joblib,resample_grail_pose
from src.wmac_perceptive.expert import PerceptiveExpertPairs


def fixture_motion(*,velocity=(.2,0.,0.),yaw_rate=0.,frames=60):
    model=mujoco.MjModel.from_xml_path(str(DEFAULT_G1_XML));data=mujoco.MjData(model)
    positions=[];quaternions=[];joints=[]
    for index in range(frames):
        angle=yaw_rate*index*.02
        data.qpos[:3]=np.asarray(velocity)*index*.02+(0.,0.,.79)
        data.qpos[3:7]=(np.cos(angle/2),0.,0.,np.sin(angle/2))
        mujoco.mj_forward(model,data)
        positions.append(data.xpos[1:].copy());quaternions.append(data.xquat[1:].copy());joints.append(data.qpos[7:].copy())
    return motion_velocities(dict(fps=np.asarray([50.],dtype=np.float32),joint_pos=np.asarray(joints,dtype=np.float32),
        body_pos_w=np.asarray(positions,dtype=np.float32),body_quat_w=np.asarray(quaternions,dtype=np.float32)))


class LocomotionDataTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.mapping=build_g1_mirror_map()

    def test_sustained_forward_and_in_place_turn_retained_other_motion_rejected(self):
        for motion in (fixture_motion(),fixture_motion(velocity=(0.,0.,0.),yaw_rate=.5)):
            valid,_=frame_selection(motion,self.mapping)
            self.assertGreater(int(valid.sum()),40)
        for kind,velocity in (('backward',(-.4,0.,0.)),('lateral',(0.,.4,0.)),('fast',(2.,0.,0.)),('flight',(.4,0.,.6))):
            with self.subTest(kind=kind):
                valid,_=frame_selection(fixture_motion(velocity=velocity),self.mapping)
                self.assertEqual(contiguous_segments(valid,20),[])

    def test_invalid_frame_and_teleport_never_join_expert_intervals(self):
        valid=np.r_[np.ones(25,dtype=bool),False,np.ones(30,dtype=bool)]
        self.assertEqual(contiguous_segments(valid,20),[(0,25),(26,56)])
        motion=fixture_motion()
        motion['body_pos_w'][30:,:,0]+=10.
        selected,diagnostic=frame_selection(motion,self.mapping)
        self.assertFalse(bool(diagnostic['continuous'][29]))
        self.assertFalse(bool(diagnostic['continuous'][30]))
        self.assertFalse(bool(selected[29]))

    def test_higher_planted_foot_can_support_when_lower_foot_is_swinging(self):
        motion=fixture_motion(frames=40)
        feet=[self.mapping.body_names.index(name) for name in ('left_ankle_roll_link','right_ankle_roll_link')]
        time=np.arange(40)*.02
        motion['body_pos_w'][:,feet[0],2]+=.03*np.sin(time*20.)
        motion['body_pos_w'][:,feet[1]]=motion['body_pos_w'][0,feet[1]]
        motion['body_pos_w'][:,feet[1],2]+=.2
        valid,diagnostic=frame_selection(motion,self.mapping)
        self.assertTrue(bool(diagnostic['support'].all()))
        self.assertGreater(int(valid.sum()),30)

    def test_grail_resampling_preserves_endpoints_and_uses_quaternion_slerp(self):
        from scipy.spatial.transform import Rotation
        source=dict(fps=25.,root_trans_offset=np.asarray([[0.,0.,.8],[.04,0.,.9]]),dof=np.zeros((2,29)),
                    root_rot=Rotation.from_euler('z',np.asarray([[0.],[np.pi/2]])).as_quat())
        position,quaternion,joints,metadata=resample_grail_pose(source)
        self.assertEqual((len(position),len(joints)),(3,3))
        np.testing.assert_allclose(position[[0,-1]],source['root_trans_offset'])
        np.testing.assert_allclose(position[1],[.02,0.,.85])
        np.testing.assert_allclose(Rotation.from_quat(quaternion[:,[1,2,3,0]]).as_euler('xyz')[:,2],[0.,np.pi/4,np.pi/2],atol=1e-8)
        self.assertEqual(metadata['target_fps'],50.)

    def test_grail_numeric_loader_refuses_globals_and_object_arrays(self):
        import joblib
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'numeric.pkl';joblib.dump({'x':np.arange(6,dtype=np.float32).reshape(2,3)},path)
            np.testing.assert_array_equal(load_numeric_joblib(path)['x'],np.arange(6).reshape(2,3))
            joblib.dump({'x':np.asarray([{'not':'numeric'}],dtype=object)},path)
            with self.assertRaisesRegex(ValueError,'numeric'):load_numeric_joblib(path)
            class Unsupported:
                def __reduce__(self):return (eval,('1+1',))
            path.write_bytes(pickle.dumps(Unsupported()))
            with self.assertRaisesRegex(ValueError,'global'):load_numeric_joblib(path)

    def test_official_torso_root_and_named_joint_permutation_reconstruct_native_fk(self):
        expected=fixture_motion(velocity=(.3,0.,0.),yaw_rate=.2,frames=8)
        torso=self.mapping.body_names.index('torso_link')
        order=np.arange(29)[::-1]
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'official_schema.npz'
            np.savez_compressed(path,framerate=np.asarray(50.),joint_names=np.asarray(self.mapping.joint_names)[order],
                joint_pos=expected['joint_pos'][:,order],base_pos_w=expected['body_pos_w'][:,torso],
                base_quat_w=expected['body_quat_w'][:,torso])
            actual,audit=convert_hiking_motion(path)
        np.testing.assert_allclose(actual['joint_pos'],expected['joint_pos'],atol=0.,rtol=0.)
        np.testing.assert_allclose(actual['body_pos_w'],expected['body_pos_w'],atol=2e-7,rtol=0.)
        self.assertLess(audit['maximum_torso_position_reconstruction_error_m'],1e-7)
        self.assertEqual(audit['source_base_link'],'torso_link')
        # Directly copying torso z into pelvis is observably wrong.
        self.assertGreater(float(np.abs(expected['body_pos_w'][:,torso,2]-expected['body_pos_w'][:,0,2]).max()),.01)

    def test_local_supplement_cache_excludes_recovery_names_and_preserves_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'source';(source/'WalkandRun').mkdir(parents=True);(source/'Recovery').mkdir()
            for name,motion in [('walk_forward.npz',fixture_motion()),('idle_turn.npz',fixture_motion(velocity=(0.,0.,0.),yaw_rate=.5)),
                                ('jog_forward.npz',fixture_motion())]:
                np.savez_compressed(source/'WalkandRun'/name,**motion)
            recovery=source/'Recovery/fall_and_get_up.npz';recovery.write_bytes(b'not read by locomotion preparation')
            original={str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source.rglob('*.npz')}
            output=Path(tmp)/'output';paths=prepare_locomotion_dataset(source,output,include_hiking=False,include_grail=False)
            manifest=verify_locomotion_dataset(output,source_root=source)
            self.assertEqual(manifest['motion_scope'],'forward_walk_turn_v1')
            self.assertEqual(manifest['recovery_original_clips'],0)
            self.assertFalse(paths.recovery.exists())
            self.assertEqual(manifest['walking_total_clips'],4)
            self.assertTrue(any(row['turn_frames'] for row in manifest['clips']))
            self.assertEqual(manifest['rejected_source_clips'][0]['file'],'jog_forward.npz')
            cfg=SimpleNamespace(task='fixture',motion_scope='forward_walk_turn_v1',amp_feature_schema='grounded_v2',expert_recovery_fraction=0.)
            env=SimpleNamespace(scene={'robot':SimpleNamespace(body_names=self.mapping.body_names)},step_dt=.02)
            rl=SimpleNamespace(amp_body_names=self.mapping.body_names[:13],amp_anchor_name='torso_link')
            with patch('src.wmac_perceptive.expert.load_rl_cfg',return_value=rl):
                expert=PerceptiveExpertPairs(cfg,env,'cpu',output)
            self.assertIsNone(expert.groups[1])
            self.assertEqual(expert.fraction,0.)
            self.assertEqual(expert.sample(11)[0].shape,(11,203))
            self.assertEqual(expert.metadata['motion_scope'],'forward_walk_turn_v1')
            for name,digest in original.items():self.assertEqual(hashlib.sha256((source/name).read_bytes()).hexdigest(),digest)
            self.assertEqual(prepare_locomotion_dataset(source,output,include_hiking=False,include_grail=False),paths)
            target=next(paths.walk.glob('*.npz'));target.write_bytes(target.read_bytes()+b'changed')
            with self.assertRaisesRegex(ValueError,'modified'):verify_locomotion_dataset(output)

    def test_source_download_checks_pinned_digest_and_extracts_only_named_members(self):
        stream=io.BytesIO()
        members={HIKING_MOTION_MEMBER:b'source bytes','data&model/parkour_motion_reference/parkour_motion_without_run.yaml':b'files: []',
                 'data&model/README.md':b'data reference','../../unexpected.py':b'not extracted'}
        with zipfile.ZipFile(stream,'w') as archive:
            for name,value in members.items():archive.writestr(name,value)
        content=stream.getvalue();digest=hashlib.sha256(content).hexdigest()
        source_hashes={Path(name).name:hashlib.sha256(value).hexdigest() for name,value in members.items() if name!='../../unexpected.py'}
        with tempfile.TemporaryDirectory() as tmp,patch('src.wmac_perceptive.locomotion_data.HIKING_ARCHIVE_SHA256',digest),\
             patch('src.wmac_perceptive.locomotion_data.HIKING_BUNDLED_SOURCE',Path(tmp)/'no-bundle'),\
             patch('src.wmac_perceptive.locomotion_data.HIKING_SOURCE_SHA256',source_hashes),\
             patch('urllib.request.urlopen',return_value=io.BytesIO(content)) as request:
            path=ensure_hiking_source(tmp)
            self.assertEqual(path.read_bytes(),b'source bytes')
            self.assertEqual(len(list(Path(tmp).iterdir())),4)
            self.assertEqual(ensure_hiking_source(tmp),path)
            request.assert_called_once()
            path.write_bytes(b'mutated')
            with self.assertRaisesRegex(ValueError,'modified'):ensure_hiking_source(tmp)
        with tempfile.TemporaryDirectory() as tmp,\
             patch('src.wmac_perceptive.locomotion_data.HIKING_BUNDLED_SOURCE',Path(tmp)/'no-bundle'),\
             patch('urllib.request.urlopen',return_value=io.BytesIO(content)):
            with self.assertRaisesRegex(ValueError,'SHA256'):ensure_hiking_source(tmp)


if __name__=='__main__':unittest.main()
