"""CPU checks for migrated geometry, perception and retained recovery rules."""
from __future__ import annotations

from dataclasses import replace
import math
from types import SimpleNamespace
import unittest

import mujoco
import numpy as np
import torch

from mjlab.terrains import TerrainGenerator
from mjlab.managers.termination_manager import TerminationManager,TerminationTermCfg
from src.tasks.amp_loco.mdp.terminations import DelayedTerminationManager
from src.tasks.amp_perceptive.terrain import (
    HIKING_PROPORTIONS,HikingTerrainCfg,TerrainHeightField,
    hiking_height_array,hiking_terrain_generator,
)
from src.tasks.amp_perceptive.sensors import (
    HikingDepthPatternCfg,depth_packet_from_ranges,
)
from src.tasks.amp_perceptive.commands import (
    command_from_state,navigation_terrain_curriculum,FlatPatchVelocityCommand,
)
from src.tasks.amp_perceptive.env_cfg import g1_amp_perceptive_hiking_env_cfg
from src.tasks.amp_perceptive.mdp import (
    terrain_recovery_height_reward,terrain_root_height_below_minimum,
    terrain_bad_orientation,root_pose_from_asset,
    PerceptiveDelayedTerminationManager,
)
from src.wmac.state import quat_apply,quat_inv
from src.tasks.amp_perceptive.entrypoint import PerceptiveTrainingEntrypointRequired


class MigratedTerrainTest(unittest.TestCase):
    def test_all_author_families_are_deterministic_and_recovery_pad_is_clear(self):
        self.assertAlmostEqual(sum(HIKING_PROPORTIONS.values()),1.)
        for name in (*HIKING_PROPORTIONS,"recovery_flat"):
            with self.subTest(name=name):
                a=hiking_height_array(name,.7,np.random.default_rng(12))
                b=hiking_height_array(name,.7,np.random.default_rng(12))
                np.testing.assert_array_equal(a,b)
                self.assertEqual(a.shape,(161,161))
                self.assertTrue(np.isfinite(a).all())
                self.assertEqual(float(np.ptp(a[65:96,65:96])),0.)
        cfg=hiking_terrain_generator()
        self.assertEqual((cfg.num_rows,cfg.num_cols),(10,20))
        self.assertEqual(cfg.sub_terrains["recovery_flat"].proportion,.25)
        for name,proportion in HIKING_PROPORTIONS.items():
            self.assertAlmostEqual(cfg.sub_terrains[name].proportion,.75*proportion)
        self.assertTrue(all(t.horizontal_scale == .05 and t.collision_horizontal_scale == .1
                            for t in cfg.sub_terrains.values()))

    def test_collision_resolution_only_accepts_exact_source_grid_decimation(self):
        for scale in (.05,.1,.2,.25,.4):
            cfg=hiking_terrain_generator(collision_horizontal_scale=scale)
            self.assertTrue(all(t.collision_horizontal_scale == scale for t in cfg.sub_terrains.values()))
        for scale in (0.,-.1,.04,.11,.125,.15,float("nan"),float("inf")):
            with self.subTest(scale=scale),self.assertRaises(ValueError):
                hiking_terrain_generator(collision_horizontal_scale=scale)
        cfg=HikingTerrainCfg(collision_horizontal_scale=.15)
        with self.assertRaises(ValueError):
            cfg.function(.5,mujoco.MjSpec(),np.random.default_rng(3))

    def test_coarse_collision_preserves_source_and_certifies_its_own_flat_patches(self):
        patches=hiking_terrain_generator().sub_terrains["pyramid_stairs"].flat_patch_sampling
        cfg=HikingTerrainCfg(family="pyramid_stairs",size=(8.,8.),wall_probability=0.,flat_patch_sampling=patches)
        spec=mujoco.MjSpec();spec.worldbody.add_body(name="terrain")
        output=cfg.function(.9,spec,np.random.default_rng(9))
        model=spec.compile()
        oracle=TerrainHeightField.from_model(model)
        source=hiking_height_array("pyramid_stairs",.9,np.random.default_rng(9))
        self.assertEqual(source.shape,(161,161))
        self.assertEqual(oracle.heights.shape,(1,81,81))
        expected=source[::2,::2].copy()
        # Physical pad is rounded outwards: 1.6m at 10cm, never the smaller
        # 1.4m plateau produced by bare decimation of the 1.5m source pad.
        expected[32:49,32:49]=source[80,80]
        np.testing.assert_allclose(oracle.heights[0].numpy(),expected,atol=3e-7)
        self.assertEqual(float(np.ptp(oracle.heights[0,32:49,32:49].numpy())),0.)
        np.testing.assert_allclose(output.origin,[4.,4.,source[80,80]],atol=1e-7)
        for patch in output.flat_patches["target"]:
            ix,iy=np.rint(patch[:2]/.1).astype(int)
            actual=oracle.heights[0,iy-2:iy+3,ix-2:ix+3].numpy()
            self.assertLessEqual(float(np.ptp(actual)),.05+3e-7)
            self.assertAlmostEqual(float(patch[2]),float(oracle.height_at(torch.tensor(patch[:2],dtype=torch.float32))),places=5)

    def test_collision_model_compiles_and_oracle_reads_its_actual_heights(self):
        cfg=hiking_terrain_generator(rows=2,cols=20)
        spec=mujoco.MjSpec()
        generator=TerrainGenerator(cfg)
        generator.compile(spec)
        model=spec.compile()
        self.assertEqual(model.nhfield,40)
        self.assertTrue(np.all(model.hfield_nrow==81))
        self.assertTrue(np.all(model.hfield_ncol==81))
        oracle=TerrainHeightField.from_model(model)
        centers=oracle.centers
        height=oracle.height_at(centers)
        origins=generator.terrain_origins.reshape(-1,3)
        expected=np.array([origins[np.argmin(np.linalg.norm(origins[:,:2]-center.numpy(),axis=1)),2] for center in centers])
        np.testing.assert_allclose(height.numpy(),expected,atol=2e-6)
        self.assertEqual(oracle.local_patch(torch.cat((centers[:3],height[:3,None],torch.tensor([[1.,0,0,0]]).repeat(3,1),torch.zeros(3,64)),1)).shape,(3,81))
        self.assertTrue(oracle.contains(centers).all())
        self.assertFalse(bool(oracle.contains(torch.tensor([[10000.,0.]]))[0]))
        self.assertTrue(torch.isfinite(oracle.edge_distance_at(centers)).all())

    def test_bilinear_height_and_coordinate_gradients_match_a_known_plane(self):
        line=torch.linspace(-1,1,11,dtype=torch.float64)
        x,y=torch.meshgrid(line,line,indexing="xy")
        h=(.2*x+.3*y+.7).unsqueeze(0)
        oracle=TerrainHeightField(h,torch.zeros(1,2),torch.ones(1,2))
        points=torch.tensor([[[.23,-.14],[-.27,.18]]],dtype=torch.float64,requires_grad=True)
        value=oracle.height_at(points)
        torch.testing.assert_close(value,.2*points[...,0]+.3*points[...,1]+.7,rtol=0,atol=1e-12)
        grad=torch.autograd.grad(value.sum(),points)[0]
        torch.testing.assert_close(grad,points.new_tensor([.2,.3]).expand_as(points),rtol=0,atol=1e-12)
        self.assertEqual(oracle.edge_distance_at(points).shape,(1,2))


class DepthAndNavigationTest(unittest.TestCase):
    def test_body_camera_center_faces_forward_down_and_uses_range_to_plane_depth(self):
        pattern=HikingDepthPatternCfg(width=3,height=3)
        origin,rays=pattern.generate_rays(None,"cpu")
        self.assertEqual(origin.shape,(9,3))
        self.assertGreater(float(rays[4,0]),.6)
        self.assertLess(float(rays[4,2]),-.7)
        self.assertTrue(torch.allclose(rays.norm(dim=-1),torch.ones(9)))
        _,cosine=pattern.optical_rays()
        ranges=torch.full((1,9),1.)/cosine
        packet=depth_packet_from_ranges(ranges,pattern)
        torch.testing.assert_close(packet[:,0],torch.full((1,3,3),.4))
        self.assertTrue((packet[:,1]==1).all())
        ranges[0,0]=-1;ranges[0,1]=float("nan");ranges[0,2]=100
        packet=depth_packet_from_ranges(ranges,pattern)
        self.assertTrue(torch.isfinite(packet).all())
        self.assertTrue((packet[0,0,0]==1).all())
        self.assertTrue((packet[0,1,0]==0).all())

    def test_navigation_turns_both_directions_and_has_finite_goal_gradients(self):
        state=torch.zeros(3,71,dtype=torch.float64);state[:,3]=1
        state.requires_grad_(True)
        targets=torch.tensor([[1.,1.,0.],[1.,-1.,0.],[0.,0.,0.]],dtype=torch.float64)
        cmd=command_from_state(state,targets,torch.full((3,),.5))
        self.assertGreater(float(cmd[0,2]),0.)
        self.assertLess(float(cmd[1,2]),0.)
        self.assertEqual(cmd[2].tolist(),[0.,0.,0.])
        cmd.sum().backward()
        self.assertTrue(torch.isfinite(state.grad).all())


class RecoveryRetentionTest(unittest.TestCase):
    def test_registry_rejects_legacy_blind_ppo_runner_but_retains_env_and_expert_metadata(self):
        from mjlab.tasks.registry import load_env_cfg,load_rl_cfg,load_runner_cls
        name="Unitree-G1-AMP-Perceptive-Hiking"
        runner=load_runner_cls(name)
        self.assertIs(runner,PerceptiveTrainingEntrypointRequired)
        with self.assertRaisesRegex(ValueError,"scripts/train_wmac_perceptive.py"):
            runner(object(),{},log_dir="unused",device="cpu")
        self.assertIn("perception",load_env_cfg(name).observations)
        self.assertTrue(load_rl_cfg(name).amp_body_names)

    def test_task_preserves_recovery_data_window_and_384_proprioception_configuration(self):
        cfg=g1_amp_perceptive_hiking_env_cfg(motion_scope="joint_recovery_v1")
        self.assertEqual(cfg.events["init_motion_loader"].params["max_delay_steps"],250)
        self.assertEqual(cfg.events["init_motion_loader"].params["delay_reset_env_ratio"],.4)
        self.assertTrue(cfg.events["init_motion_loader"].params["recovery_dir"].endswith("Recovery"))
        self.assertEqual(cfg.observations["actor"].history_length,4)
        self.assertNotIn("height_scan",cfg.observations["actor"].terms)
        self.assertEqual(cfg.observations["perception"].history_length,1)
        self.assertNotIn("torso_contact",cfg.terminations)
        self.assertFalse(cfg.terminations["bad_orientation"].time_out)
        self.assertIn("navigation_terrain",cfg.curriculum)
        self.assertNotIn("randomize_terrain",g1_amp_perceptive_hiking_env_cfg(play=True).events)

    def test_real_height_reward_and_bad_height_are_invariant_to_platform_elevation(self):
        maps=torch.stack((torch.zeros(5,5),torch.full((5,5),2.)))
        oracle=TerrainHeightField(maps,torch.tensor([[0.,0.],[2.,0.]]),torch.ones(2,2))
        position=torch.tensor([[0.,0.,.79],[2.,0.,2.79]])
        data=SimpleNamespace(body_link_pos_w=position[:,None],root_link_pos_w=position,
                              body_link_quat_w=torch.tensor([[[1.,0,0,0]],[[1.,0,0,0]]]),
                              default_root_state=torch.tensor([[0.,0.,.79],[0.,0.,.79]]))
        env=SimpleNamespace(scene={"robot":SimpleNamespace(data=data)},
                             _perceptive_terrain_oracle=oracle,
                             termination_manager=SimpleNamespace(_delay_env_mask=torch.ones(2,dtype=torch.bool),_delay_counters=torch.ones(2)))
        reward=terrain_recovery_height_reward(env,.2,mask_delay=True,upright_power=2.)
        torch.testing.assert_close(reward,torch.ones(2))
        self.assertFalse(terrain_root_height_below_minimum(env,.5).any())
        position[:,2]-=.5
        self.assertTrue(terrain_root_height_below_minimum(env,.5).all())

    def test_qpos_root_pose_overrides_stale_derived_height_and_orientation(self):
        qpos=torch.tensor([[0.,0.,.79,1.,0.,0.,0.],[0.,0.,.49,0.,1.,0.,0.]],dtype=torch.float64)
        data=SimpleNamespace(data=SimpleNamespace(qpos=qpos),
            root_link_pos_w=torch.tensor([[0.,0.,.1],[0.,0.,.9]],dtype=torch.float64),
            root_link_quat_w=torch.tensor([[0.,1.,0.,0.],[1.,0.,0.,0.]],dtype=torch.float64),
            body_link_pos_w=torch.zeros(2,1,3,dtype=torch.float64),
            body_link_quat_w=torch.tensor([[[0.,1.,0.,0.]],[[1.,0.,0.,0.]]],dtype=torch.float64),
            default_root_state=torch.tensor([[0.,0.,.79],[0.,0.,.79]],dtype=torch.float64))
        asset=SimpleNamespace(data=data,indexing=SimpleNamespace(free_joint_q_adr=torch.arange(7)))
        oracle=TerrainHeightField(torch.zeros(1,5,5,dtype=torch.float64),torch.zeros(1,2),torch.ones(1,2))
        env=SimpleNamespace(scene={"robot":asset},_perceptive_terrain_oracle=oracle,
            termination_manager=SimpleNamespace(_delay_env_mask=torch.ones(2,dtype=torch.bool),_delay_counters=torch.ones(2)))
        torch.testing.assert_close(root_pose_from_asset(asset),qpos,rtol=0,atol=0)
        self.assertEqual(terrain_root_height_below_minimum(env,.5).tolist(),[False,True])
        self.assertEqual(terrain_bad_orientation(env,math.radians(70)).tolist(),[False,True])
        reward=terrain_recovery_height_reward(env,.2,mask_delay=True,upright_power=2.)
        torch.testing.assert_close(reward,qpos.new_tensor([1.,0.]),rtol=0,atol=0)

    def test_qpos_height_and_acos_orientation_boundaries_match_imagined_predicates(self):
        angle=torch.tensor([70.-1e-6,70.,70.+1e-6],dtype=torch.float64)*math.pi/180
        quat=torch.stack((torch.cos(angle/2),torch.sin(angle/2),torch.zeros_like(angle),torch.zeros_like(angle)),-1)
        qpos=torch.cat((torch.zeros(3,2,dtype=torch.float64),torch.tensor([[.5-1e-8],[.5],[.5+1e-8]],dtype=torch.float64),quat),-1)
        asset=SimpleNamespace(data=SimpleNamespace(data=SimpleNamespace(qpos=qpos)),
                              indexing=SimpleNamespace(free_joint_q_adr=torch.arange(7)))
        oracle=TerrainHeightField(torch.zeros(1,5,5,dtype=torch.float64),torch.zeros(1,2),torch.ones(1,2))
        env=SimpleNamespace(scene={"robot":asset},_perceptive_terrain_oracle=oracle)
        gravity=quat_apply(quat_inv(quat),quat.new_tensor([0.,0.,-1.]))
        expected=torch.acos(-gravity[:,2]).abs()>math.radians(70.)
        torch.testing.assert_close(terrain_bad_orientation(env,math.radians(70.)),expected,rtol=0,atol=0)
        self.assertEqual(terrain_root_height_below_minimum(env,.5).tolist(),[True,False,False])

    def test_real_delayed_manager_retains_250_step_window_on_both_platform_heights(self):
        qpos=torch.tensor([[0.,0.,.2,1.,0.,0.,0.],[2.,0.,2.2,1.,0.,0.,0.]],dtype=torch.float64)
        asset=SimpleNamespace(data=SimpleNamespace(data=SimpleNamespace(qpos=qpos)),
                              indexing=SimpleNamespace(free_joint_q_adr=torch.arange(7)))
        maps=torch.stack((torch.zeros(5,5,dtype=torch.float64),torch.full((5,5),2.,dtype=torch.float64)))
        oracle=TerrainHeightField(maps,torch.tensor([[0.,0.],[2.,0.]]),torch.ones(2,2))
        env=SimpleNamespace(num_envs=2,device="cpu",scene={"robot":asset},_perceptive_terrain_oracle=oracle)
        base=TerminationManager({
            "height":TerminationTermCfg(func=terrain_root_height_below_minimum,params={"minimum_height":.5}),
            "orientation":TerminationTermCfg(func=terrain_bad_orientation,params={"limit_angle":math.radians(70.)}),
        },env)
        manager=PerceptiveDelayedTerminationManager(base,torch.ones(2,dtype=torch.bool),250)
        for step in range(1,251):
            done=manager.compute()
            self.assertEqual(done.tolist(),[step==250,step==250])
            self.assertEqual(manager._delay_counters.tolist(),[step if step<250 else 0]*2)
        manager._delay_counters.fill_(125)
        qpos[:,2]=qpos.new_tensor([.79,2.79])
        self.assertFalse(manager.compute().any())
        self.assertEqual(manager._delay_counters.tolist(),[0,0])

    def test_timeout_reset_clears_only_new_episode_counter_and_preserves_other_environment(self):
        qpos=torch.tensor([[0.,0.,.2,1.,0.,0.,0.],[0.,0.,.2,1.,0.,0.,0.]])
        asset=SimpleNamespace(data=SimpleNamespace(data=SimpleNamespace(qpos=qpos)),
                              indexing=SimpleNamespace(free_joint_q_adr=torch.arange(7)))
        oracle=TerrainHeightField(torch.zeros(1,5,5),torch.zeros(1,2),torch.ones(1,2))
        env=SimpleNamespace(num_envs=2,device="cpu",scene={"robot":asset},_perceptive_terrain_oracle=oracle,
                            timeout=torch.tensor([False,False]))
        def timeout(environment):return environment.timeout
        base=TerminationManager({
            "height":TerminationTermCfg(func=terrain_root_height_below_minimum,params={"minimum_height":.5}),
            "timeout":TerminationTermCfg(func=timeout,time_out=True),
        },env)
        manager=PerceptiveDelayedTerminationManager(base,torch.ones(2,dtype=torch.bool),250)
        for _ in range(40):self.assertFalse(manager.compute().any())
        env.timeout[0]=True
        self.assertEqual(manager.compute().tolist(),[True,False])
        self.assertEqual(manager._delay_counters.tolist(),[41,41])
        manager.reset(torch.tensor([0]))
        self.assertEqual(manager._delay_counters.tolist(),[0,41])
        env.timeout[0]=False
        self.assertFalse(manager.compute().any())
        self.assertEqual(manager._delay_counters.tolist(),[1,42])
        manager.reset()
        self.assertEqual(manager._delay_counters.tolist(),[0,0])


class NavigationCurriculumTest(unittest.TestCase):
    def fixture(self):
        class Scene(dict):pass
        position=torch.tensor([[0.,0.,.79],[.1,0.,.79],[.2,0.,.3]])
        data=SimpleNamespace(root_link_pos_w=position,
                             root_link_quat_w=torch.tensor([[1.,0.,0.,0.]]).repeat(3,1))
        scene=Scene(robot=SimpleNamespace(data=data));scene.env_origins=torch.zeros(3,3)
        class Terrain:
            terrain_levels=torch.ones(3,dtype=torch.long)
            update_calls=0
            def update_env_origins(self,ids,up,down):
                self.update_calls+=1
                self.up,self.down=up.clone(),down.clone()
                self.terrain_levels[ids]+=up.long()-down.long()
        scene.terrain=Terrain()
        command=SimpleNamespace(goals_reached=torch.tensor([1.,0.,1.]),desired_speed=torch.full((3,),.5))
        oracle=TerrainHeightField(torch.zeros(1,5,5),torch.zeros(1,2),torch.full((1,2),4.))
        env=SimpleNamespace(num_envs=3,device=torch.device("cpu"),step_dt=.02,scene=scene,
                            command_manager=SimpleNamespace(get_term=lambda name:command),
                            episode_length_buf=torch.full((3,),500),reset_terminated=torch.tensor([False,True,False]),
                            _perceptive_terrain_oracle=oracle)
        return env

    def test_real_goal_and_stable_finish_promote_but_failed_recovery_cannot(self):
        env=self.fixture()
        navigation_terrain_curriculum(env,torch.arange(3))
        self.assertEqual(env.scene.terrain.up.tolist(),[True,False,False])
        self.assertEqual(env.scene.terrain.down.tolist(),[False,True,False])
        self.assertEqual(env.scene.terrain.terrain_levels.tolist(),[2,0,1])
        env.episode_length_buf.zero_()
        navigation_terrain_curriculum(env,torch.arange(3))
        self.assertEqual(env.scene.terrain.update_calls,1)
        self.assertEqual(env.scene.terrain.terrain_levels.tolist(),[2,0,1])

    def test_target_resampling_preserves_goals_until_episode_command_reset(self):
        class Scene(dict):pass
        cfg=g1_amp_perceptive_hiking_env_cfg().commands["twist"]
        data=SimpleNamespace(root_link_pos_w=torch.zeros(2,3),
                             root_link_quat_w=torch.tensor([[1.,0.,0.,0.]]).repeat(2,1))
        scene=Scene(robot=SimpleNamespace(data=data))
        scene.terrain=SimpleNamespace(flat_patches={"target":torch.tensor([[[[1.,0.,0.],[-1.,0.,0.],[0.,1.,0.],[0.,-1.,0.]]]])},
                                        terrain_levels=torch.zeros(2,dtype=torch.long),terrain_types=torch.zeros(2,dtype=torch.long))
        env=SimpleNamespace(num_envs=2,device=torch.device("cpu"),scene=scene,step_dt=.02)
        command=FlatPatchVelocityCommand(cfg,env)
        ids=torch.arange(2)
        command.goals_reached.fill_(3)
        command._resample_command(ids)
        self.assertEqual(command.goals_reached.tolist(),[3.,3.])
        command._update_command()  # Uses the installed EntityData pose API.
        self.assertTrue(torch.isfinite(command.command).all())
        command.reset(torch.tensor([1]))
        self.assertEqual(command.goals_reached.tolist(),[3.,0.])

    def test_first_reset_without_termination_buffer_leaves_rows_and_goals_untouched(self):
        env=self.fixture()
        del env.reset_terminated
        env.episode_length_buf.zero_()
        command=env.command_manager.get_term("twist")
        goals=command.goals_reached.clone()
        levels=env.scene.terrain.terrain_levels.clone()
        result=navigation_terrain_curriculum(env,torch.arange(3))
        torch.testing.assert_close(command.goals_reached,goals,rtol=0,atol=0)
        torch.testing.assert_close(env.scene.terrain.terrain_levels,levels,rtol=0,atol=0)
        self.assertFalse(hasattr(env.scene.terrain,"up"))
        self.assertEqual(float(result["promoted_fraction"]),0.)
        self.assertEqual(float(result["demoted_fraction"]),0.)


if __name__=="__main__":unittest.main()
