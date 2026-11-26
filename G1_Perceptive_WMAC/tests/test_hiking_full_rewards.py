"""Numerical oracle tests against the pinned original reward functions."""
from __future__ import annotations

import ast
import math
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

import torch

from src.wmac.state import quat_apply, quat_inv
from src.tasks.amp_perceptive.hiking_rewards import (
    HIKING_REWARD_WEIGHTS, HIKING_AUX_NAMES, HikingContactSensor,
    hiking_auxiliary_from_measurements, hiking_reward_terms,
    hiking_weighted_reward, hiking_volume_penetration,
    hiking_termination_terms, hiking_terrain_out_of_bounds,
)


def fixture(batch=8):
    generator = torch.Generator().manual_seed(147)
    def random(*shape):
        return torch.randn(*shape, dtype=torch.float64, generator=generator)
    q = random(batch, 29)
    joints = NS(joint_pos=q, default_joint_pos=random(batch,29), joint_vel=random(batch,29)*20.,
        joint_acc=random(batch,29)*20., applied_torque=random(batch,29)*100.,
        soft_joint_pos_limits=torch.tensor([[-.5,.5]],dtype=q.dtype).expand(batch,29,2),
        soft_joint_vel_limits=torch.full_like(q,20.), joint_effort_limits=torch.full_like(q,50.),
        root_lin_vel_b=random(batch,3),root_ang_vel_b=random(batch,3),
        projected_gravity_b=random(batch,3), body_pos_w=random(batch,5,3),
        body_lin_vel_w=random(batch,5,3),body_ang_vel_w=random(batch,5,3),
        heading_w=random(batch),GRAVITY_VEC_W=q.new_tensor([0.,0.,-1.]).expand(batch,3))
    quaternion = random(batch,5,4)
    joints.body_quat_w = quaternion / quaternion.norm(dim=-1,keepdim=True)
    joints.projected_gravity_b = quat_apply(quat_inv(joints.body_quat_w[:,0]),joints.GRAVITY_VEC_W)
    contacts = random(batch,3,5,3)
    contacts[0]=0.;contacts[1,:,2]=0.;contacts[2,:,3]=0.
    air = torch.zeros(batch,5,dtype=q.dtype);contact=air.clone()
    air[:,2]=.1;contact[:,3]=.2
    air[1,2]=0.;contact[1,2]=.3
    air[2,3]=.4;contact[2,3]=0.
    command=random(batch,3);command[0]=0.;command[1]=torch.tensor([0.,0.,.5])
    action=random(batch,29);previous=random(batch,29)
    stiffness=torch.full_like(q,40.)
    leg_ids=tuple(range(12));hip_ids=(1,2,7,8);upper_ids=tuple(range(12,29));feet=(2,3)
    penetration=random(batch,2,100,3)*.01;points_vel=random(batch,2,100,3)
    heights=random(batch,2,2)*.1;heights[0,0,0]=float("inf")
    auxiliary=hiking_auxiliary_from_measurements(applied_torque=joints.applied_torque,
        joint_velocity=joints.joint_vel,joint_acceleration=joints.joint_acc,
        stiffness=stiffness,effort_limits=joints.joint_effort_limits,leg_joint_ids=leg_ids,
        contact_force_max=contacts.norm(dim=-1).max(1).values,foot_body_ids=feet,torso_body_id=0,
        current_air_time=air,current_contact_time=contact)
    terminated=torch.arange(batch)%3==0
    inputs=dict(command=command,torso_linear_velocity_b=joints.root_lin_vel_b,
        torso_angular_velocity_b=joints.root_ang_vel_b,torso_projected_gravity=joints.projected_gravity_b,
        pelvis_projected_gravity=quat_apply(quat_inv(joints.body_quat_w[:,1]),joints.GRAVITY_VEC_W),
        foot_projected_gravity=quat_apply(quat_inv(joints.body_quat_w[:,feet]),joints.GRAVITY_VEC_W[:,None]),
        torso_heading=joints.heading_w,foot_position_w=joints.body_pos_w[:,feet],
        foot_linear_velocity_w=joints.body_lin_vel_w[:,feet],foot_ground_height=heights,
        joint_position=q,default_joint_position=joints.default_joint_pos,joint_velocity=joints.joint_vel,
        joint_position_limits=joints.soft_joint_pos_limits,joint_velocity_limits=joints.soft_joint_vel_limits,
        hip_joint_ids=hip_ids,upper_body_joint_ids=upper_ids,action=action,previous_action=previous,
        terminated=terminated,volume_points_penetration=hiking_volume_penetration(penetration,points_vel))
    class Scene(dict):
        @property
        def sensors(self):return self
    scene=Scene(robot=NS(data=joints,actuators={"all":NS(joint_indices=slice(None),stiffness=stiffness)}),
        contact_forces=NS(data=NS(net_forces_w_history=contacts,current_air_time=air,current_contact_time=contact)),
        leg_volume_points=NS(data=NS(penetration_offset=penetration,points_vel_w=points_vel)))
    for i,side in enumerate(("left","right")):
        rays=torch.zeros(batch,2,3,dtype=q.dtype);rays[...,2]=heights[:,i]
        scene[side+"_height_scanner"]=NS(data=NS(ray_hits_w=rays))
    env=NS(scene=scene,command_manager=NS(get_command=lambda _:command),
        termination_manager=NS(terminated=terminated),action_manager=NS(action=action,prev_action=previous))
    return inputs,auxiliary,env,leg_ids


def source_functions():
    root=Path("/tmp/hiking_locomotion_source/source/instinctlab/instinctlab")
    paths=[Path("/tmp/hiking-isaaclab-camera-reference/rewards.py"),
        root/"envs/mdp/rewards/regularizations.py",root/"envs/mdp/rewards/volume_points.py",
        root/"tasks/parkour/mdp/rewards.py"]
    if not all(path.is_file() for path in paths):
        raise unittest.SkipTest("Pinned Hiking and IsaacLab source checkouts not present")
    class EntityCfg:
        def __init__(self,name,**kwargs):
            self.name=name;self.joint_ids=slice(None);self.body_ids=slice(None)
            self.__dict__.update(kwargs)
    namespace=dict(torch=torch,SceneEntityCfg=EntityCfg,
        quat_apply_inverse=lambda q,v:quat_apply(quat_inv(q),v))
    for path in paths:
        tree=ast.parse(path.read_text())
        nodes=[node for node in tree.body if isinstance(node,ast.FunctionDef)]
        module=ast.Module(body=[ast.ImportFrom(module="__future__",names=[ast.alias(name="annotations")],level=0),*nodes],type_ignores=[])
        exec(compile(ast.fix_missing_locations(module),str(path),"exec"),namespace)
    return namespace,EntityCfg


class HikingFullRewardsTest(unittest.TestCase):
    def test_volume_points_velocity_matches_original_sensor_with_com_velocity(self):
        path=Path("/tmp/hiking_locomotion_source/source/instinctlab/instinctlab/sensors/volume_points/volume_points.py")
        if not path.exists():self.skipTest("Pinned Hiking sensor source not present")
        method=next(node for node in ast.walk(ast.parse(path.read_text()))
            if isinstance(node,ast.FunctionDef) and node.name=="_refresh_volume_points")
        module=ast.Module(body=[ast.ImportFrom(module="__future__",names=[ast.alias(name="annotations")],level=0),method],type_ignores=[])
        namespace=dict(torch=torch,math_utils=NS(
            convert_quat=lambda q,to:torch.cat((q[...,-1:],q[...,:3]),-1),
            transform_points=lambda points,pos,quat:pos[:,None]+quat_apply(quat[:,None],points)))
        exec(compile(ast.fix_missing_locations(module),str(path),"exec"),namespace)
        from src.wmac_perceptive.safety import hiking_foot_volume_points
        inputs,_,env,_=fixture();robot=env.scene["robot"].data;feet=(2,3)
        q=robot.body_quat_w[:,feet];p=robot.body_pos_w[:,feet]
        com_velocity=robot.body_lin_vel_w[:,feet]+1.
        omega=robot.body_ang_vel_w[:,feet]
        pattern=torch.linspace(-.03,.12,300,dtype=p.dtype).reshape(100,3)
        shape=(*p.shape[:2],100,3)
        sensor=NS(num_bodies=2,_volume_points_pattern=pattern,
            body_physx_view=NS(get_transforms=lambda:torch.cat((p,q[...,1:],q[...,:1]),-1),
                get_velocities=lambda:torch.cat((com_velocity,omega),-1)),
            _data=NS(pos_w=torch.zeros_like(p),quat_w=torch.zeros_like(q),vel_w=torch.zeros_like(p),
                ang_vel_w=torch.zeros_like(p),point_num_each_body=100,
                points_pos_w=torch.zeros(shape,dtype=p.dtype),points_vel_w=torch.zeros(shape,dtype=p.dtype)))
        namespace["_refresh_volume_points"](sensor,slice(None))
        fk=(robot.body_pos_w,robot.body_quat_w,robot.body_lin_vel_w,robot.body_ang_vel_w)
        positions,velocities=hiking_foot_volume_points(fk,feet,pattern,foot_linear_velocity=com_velocity)
        torch.testing.assert_close(positions,sensor._data.points_pos_w,rtol=1e-12,atol=1e-12)
        torch.testing.assert_close(velocities,sensor._data.points_vel_w,rtol=1e-12,atol=1e-12)
        _,link_velocities=hiking_foot_volume_points(fk,feet,pattern)
        self.assertGreater(float((velocities-link_velocities).abs().max()),.9)

    def test_all_26_terms_match_executed_pinned_original(self):
        source,Cfg=source_functions()
        config=Path("/tmp/hiking_locomotion_source/source/instinctlab/instinctlab/tasks/parkour/config/parkour_env_cfg.py")
        reward_class=next(node for node in ast.parse(config.read_text()).body
            if isinstance(node,ast.ClassDef) and node.name=="G1Rewards")
        original_weights={node.targets[0].id:ast.literal_eval(next(keyword.value
            for keyword in node.value.keywords if keyword.arg=="weight"))
            for node in reward_class.body if isinstance(node,ast.Assign)
            and isinstance(node.value,ast.Call)}
        self.assertEqual(HIKING_REWARD_WEIGHTS,original_weights)
        inputs,aux,env,legs=fixture()
        feet=Cfg("robot",body_ids=(2,3));contacts=Cfg("contact_forces",body_ids=(2,3))
        original={
            "track_lin_vel_xy_exp":("track_lin_vel_xy_exp",dict(std=.5,command_name="twist")),
            "track_ang_vel_z_exp":("track_ang_vel_z_exp",dict(std=.5,command_name="twist")),
            "heading_error":("heading_error",dict(command_name="twist")),
            "dont_wait":("dont_wait",dict(command_name="twist")),"is_alive":("is_alive",{}),
            "stand_still":("stand_still",dict(command_name="twist",offset=4.)),
            "volume_points_penetration":("volume_points_penetration",dict(sensor_cfg=Cfg("leg_volume_points"))),
            "feet_air_time":("feet_air_time",dict(command_name="twist",vel_threshold=.15,sensor_cfg=contacts)),
            "feet_slide":("contact_slide",dict(sensor_cfg=contacts,asset_cfg=feet,threshold=1.)),
            "joint_deviation_hip":("joint_deviation_square",dict(asset_cfg=Cfg("robot",joint_ids=inputs["hip_joint_ids"]))),
            "ang_vel_xy_l2":("ang_vel_xy_l2",{}),
            "dof_torques_l2":("joint_torques_l2",dict(asset_cfg=Cfg("robot",joint_ids=legs))),
            "dof_acc_l2":("joint_acc_l2",{}),"dof_vel_l2":("joint_vel_l2",{}),
            "action_rate_l2":("action_rate_l2",{}),"flat_orientation_l2":("flat_orientation_l2",{}),
            "pelvis_orientation_l2":("link_orientation",dict(asset_cfg=Cfg("robot",body_ids=(1,)))),
            "feet_flat_ori":("feet_orientation_contact",dict(sensor_cfg=contacts,asset_cfg=feet)),
            "feet_at_plane":("feet_at_plane",dict(contact_sensor_cfg=contacts,
                left_height_scanner_cfg=Cfg("left_height_scanner"),right_height_scanner_cfg=Cfg("right_height_scanner"),asset_cfg=feet,height_offset=.035)),
            "feet_close_xy":("feet_close_xy_gauss",dict(threshold=.12,std=math.sqrt(.05),asset_cfg=feet)),
            "energy":("motors_power_square",dict(asset_cfg=Cfg("robot",joint_ids=legs))),
            "freeze_upper_body":("joint_deviation_l1",dict(asset_cfg=Cfg("robot",joint_ids=inputs["upper_body_joint_ids"]))),
            "dof_pos_limits":("joint_pos_limits",{}),"dof_vel_limits":("joint_vel_limits",dict(soft_ratio=.9)),
            "torque_limits":("applied_torque_limits_by_ratio",dict(limit_ratio=.8)),
            "undesired_contacts":("undesired_contacts",dict(threshold=1.,sensor_cfg=Cfg("contact_forces",body_ids=(0,1,4)))),
        }
        actual=hiking_reward_terms(inputs,aux)
        self.assertEqual(set(actual),set(original));self.assertEqual(len(actual),26)
        for name,(function,kwargs) in original.items():
            with self.subTest(name=name):
                expected=source[function](env,**kwargs).to(actual[name])
                torch.testing.assert_close(actual[name],expected,rtol=1e-12,atol=1e-12)
        expected=sum(source[fn](env,**kwargs)*HIKING_REWARD_WEIGHTS[name] for name,(fn,kwargs) in original.items())*.02
        torch.testing.assert_close(hiking_weighted_reward(inputs,aux,.02),expected,rtol=1e-12,atol=1e-12)

    def test_exact_command_contact_gates_and_limit_clipping(self):
        inputs,aux,_,_=fixture(8)
        inputs["command"]=torch.tensor([[0.,0.,0.],[.15,0.,0.],[0.,0.,.15],[0.,0.,.151],
            [.3,0.,0.],[.301,0.,0.],[.301,0.,0.],[.301,0.,0.]],dtype=torch.float64)
        inputs["torso_linear_velocity_b"][:,0]=torch.tensor([0,0,0,0,-1,.15,0.,-.15],dtype=torch.float64)
        inputs["joint_position"]=inputs["default_joint_position"].clone()
        aux[:,8]=.2;aux[:,6]=1.;aux[:,7]=1.
        raw=hiking_reward_terms(inputs,aux)
        torch.testing.assert_close(raw["stand_still"],raw["stand_still"].new_tensor([-4,0,0,0,0,0,0,0]))
        torch.testing.assert_close(raw["feet_air_time"],raw["feet_air_time"].new_tensor([0,0,0,.2,.2,.2,.2,.2]))
        torch.testing.assert_close(raw["dont_wait"],raw["dont_wait"].new_tensor([0,0,0,0,0,0,1,2]))
        for name in ("feet_slide","feet_flat_ori","feet_at_plane"):
            torch.testing.assert_close(raw[name],torch.zeros_like(raw[name]))
        inputs["joint_velocity"]=torch.full_like(inputs["joint_velocity"],100.)
        torch.testing.assert_close(hiking_reward_terms(inputs,aux)["dof_vel_limits"],torch.full((8,),29.,dtype=torch.float64))

    def test_timers_use_force_threshold_not_found_and_preserve_substeps(self):
        sensor=HikingContactSensor.__new__(HikingContactSensor)
        zero=torch.zeros(1,2)
        sensor._air_time_state=NS(current_air_time=zero.clone(),current_contact_time=zero.clone(),
            last_air_time=zero.clone(),last_contact_time=zero.clone(),last_time=torch.zeros(1))
        sensor._data=NS(time=torch.zeros(1))
        forces=torch.zeros(1,2,3);forces[:,:,2]=torch.tensor([[.9,1.1]])
        sensor._extract_sensor_data=lambda:NS(force=forces,found=torch.ones(1,2))
        for step in range(1,5):
            sensor._data.time[:]=step*.005;sensor._update_air_time_tracking()
        torch.testing.assert_close(sensor._air_time_state.current_air_time,torch.tensor([[.02,0.]]))
        torch.testing.assert_close(sensor._air_time_state.current_contact_time,torch.tensor([[0.,.02]]))
        forces[:,:,2]=torch.tensor([[1.01,1.]])
        sensor._data.time[:]=.025;sensor._update_air_time_tracking()
        torch.testing.assert_close(sensor._air_time_state.current_air_time,torch.tensor([[0.,.005]]))
        torch.testing.assert_close(sensor._air_time_state.last_air_time,torch.tensor([[.025,0.]]))
        torch.testing.assert_close(sensor._air_time_state.last_contact_time,torch.tensor([[0.,.025]]))

    def test_termination_root_reference_and_strict_boundaries(self):
        position=torch.tensor([[0.,0.,.5],[0.,0.,.499],[0.,0.,.1],[0.,0.,.6]])
        gravity=torch.tensor([[0.,0.,-1.]]).expand(4,-1)
        origin=torch.tensor([[0.,0.,2.],[0.,0.,0.],[0.,0.,-.5],[0.,0.,0.]])
        force=torch.tensor([1.,1.,0.,1.001])
        terms=hiking_termination_terms(position,gravity,origin,force)
        self.assertEqual(terms["root_height"].tolist(),[False,True,False,False])
        self.assertEqual(terms["base_contact"].tolist(),[False,False,False,True])
        self.assertFalse(terms["bad_orientation"].any())
        pos=torch.tensor([[8.,0.,0.],[8.01,0.,0.],[0.,18.01,0.]])
        self.assertEqual(hiking_terrain_out_of_bounds(pos,torch.tensor([10.,20.])).tolist(),[False,True,True])

    def test_fullgraph_backward_matches_eager_with_auxiliary_predictions(self):
        inputs,aux,_,_=fixture()
        inputs["joint_position"]=inputs["joint_position"].requires_grad_()
        aux=aux.requires_grad_()
        eager=hiking_weighted_reward(inputs,aux,.02)
        gradient=torch.autograd.grad(eager.sum(),(inputs["joint_position"],aux),retain_graph=True)
        compiled=torch.compile(hiking_weighted_reward,backend="aot_eager",fullgraph=True)
        actual=compiled(inputs,aux,.02)
        actual_gradient=torch.autograd.grad(actual.sum(),(inputs["joint_position"],aux))
        torch.testing.assert_close(actual,eager)
        for expected,observed in zip(gradient,actual_gradient):torch.testing.assert_close(observed,expected)
        self.assertEqual(len(HIKING_AUX_NAMES),13)
        self.assertTrue(torch.isfinite(actual_gradient[0]).all())


if __name__ == "__main__":unittest.main()
