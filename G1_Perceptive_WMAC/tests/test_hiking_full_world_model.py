"""Full-objective recurrent context/auxiliary-head integration on CPU.

Small fixtures isolate the production model and rollout kernels; they do not
claim to instantiate a complete Hiking data/terrain training environment.
The analytic plane has no edge cylinders and has exact vertical ray heights.
"""
from __future__ import annotations

from types import SimpleNamespace as NS
from dataclasses import replace
import unittest
from unittest.mock import patch

import torch

from src.wmac.networks import Critic, frozen_parameters
from src.wmac_perceptive.hiking_context import advance_hiking_context
from src.wmac_perceptive.hiking_rewards import initialize_hiking_reward_adapter
from src.wmac_perceptive.latent_models import LatentPerceptiveActor,LatentPerceptiveWorldModel
from src.wmac_perceptive.hiking_observations import pack_hiking_proprio_history
from src.wmac_perceptive.latent_execution import LatentWorldModelKernel
from src.wmac_perceptive.latent_training import rollout_latent_model_metrics
from src.wmac_perceptive.compiled_imagination import LatentImaginedStep,LatentTrajectoryTerms
from src.wmac_perceptive.latent_perception import pack_processed_actor_input
from test_perceptive_latent_runner import latent_runner_fixture
from test_perceptive_runner import PlaneOracle


class NoEdges:
    def penetration_depth(self,points):return torch.zeros_like(points[...,0])


class ExactPlane(PlaneOracle):
    hiking_edges=NoEdges()
    def hiking_ray_height_at(self,starts):return self.height_at(starts[...,:2])


def full_fixture():
    runner,batch,noises=latent_runner_fixture()
    runner.cfg.locomotion_objective="hiking_full_v1"
    runner.cfg.moe_gate_hidden=()
    runner.spec=replace(runner.spec,proprio_dim=768)
    runner.actor=LatentPerceptiveActor(runner.spec,gate_hidden=runner.cfg.moe_gate_hidden).double()
    adapter=runner.adapter
    adapter.full_hiking=True;adapter.context_dim=100;adapter.contact_dim=13
    adapter.terrain=ExactPlane()
    adapter.robot.joint_names=tuple(adapter.env.sim.mj_model.joint(int(i)).name for i in adapter.robot.indexing.joint_ids)
    initialize_hiking_reward_adapter(adapter)
    runner.world_model=LatentPerceptiveWorldModel(adapter.terrain,runner.spec,
        hidden=(16,),context_dim=100,contact_dim=13).double()
    with torch.no_grad():runner.world_model.net[-1].weight.normal_(0.,.002)
    runner.critic=Critic(282,hidden=(16,8)).double();runner._value_fn=runner.critic.forward
    batch["context"]=torch.cat((batch["context"][:,:33],torch.zeros(len(batch["state"]),67,dtype=torch.float64)),-1)
    batch["context"][:,38:67]=batch["previous_action"]
    batch["context"][:,67:96]=1.
    batch["command_standing"]=torch.tensor([True,False,False])
    batch["random_yaw_rate"]=torch.tensor([0.,.4,-.4],dtype=torch.float64)
    batch["random_velocity_mask"]=torch.tensor([False,True,False])
    command=adapter.navigation_command(batch["state"],batch["navigation_target"],batch["desired_speed"],
        standing_mask=batch["command_standing"],random_yaw_rate=batch["random_yaw_rate"],
        random_velocity_mask=batch["random_velocity_mask"])
    frame=adapter.actor_frame(batch["state"],batch["previous_action"],command,batch["context"])
    history=pack_hiking_proprio_history(frame[:,None].expand(-1,8,-1))
    batch["actor_history"]=pack_processed_actor_input(history,batch["depth_stack"],runner.spec)
    return runner,batch,noises


class HikingFullWorldModelTest(unittest.TestCase):
    def test_context_carries_action_delay_origin_and_predicted_timers(self):
        context=torch.arange(300,dtype=torch.float64).reshape(3,100)/100
        context[:,33:35]=0.
        action=torch.randn(3,29,dtype=torch.float64,requires_grad=True)
        auxiliary=torch.randn(3,13,dtype=torch.float64,requires_grad=True).abs()
        result=advance_hiking_context(context,action,auxiliary)
        self.assertEqual(result.shape,(3,100))
        torch.testing.assert_close(result[:,:33],context[:,:33])
        torch.testing.assert_close(result[:,33],torch.ones(3,dtype=torch.float64))
        torch.testing.assert_close(result[:,34:38],context[:,34:38])
        torch.testing.assert_close(result[:,38:67],action)
        torch.testing.assert_close(result[:,67:96],context[:,67:96])
        torch.testing.assert_close(result[:,96:100],auxiliary[:,9:13])
        gradients=torch.autograd.grad(result.sum(),(action,auxiliary))
        torch.testing.assert_close(gradients[0],torch.ones_like(action))
        torch.testing.assert_close(gradients[1][:,:9],torch.zeros_like(auxiliary[:,:9]))
        torch.testing.assert_close(gradients[1][:,9:],torch.ones_like(auxiliary[:,9:]))
        with self.assertRaisesRegex(ValueError,"100/29/13"):
            advance_hiking_context(context,action,auxiliary[:,:2])

    def test_head_keeps_35_gencoord_and_128_latent_with_13_auxiliary(self):
        runner,batch,_=full_fixture();model=runner.world_model
        with torch.no_grad():
            model.net[-1].weight.zero_()
            model.net[-1].bias.copy_(torch.arange(176,dtype=torch.float64)*.001)
        latent=torch.zeros(3,128,dtype=torch.float64)
        state,aux,next_latent=model.step_with_latent(batch["state"],batch["previous_action"],batch["context"],latent)
        torch.testing.assert_close(state,model.integrate(batch["state"],model.net[-1].bias[:35].expand(3,-1)))
        torch.testing.assert_close(aux,torch.nn.functional.softplus(model.net[-1].bias[35:48]).expand(3,-1))
        torch.testing.assert_close(next_latent,model.net[-1].bias[48:].expand(3,-1))
        self.assertEqual((state.shape[-1],aux.shape[-1],next_latent.shape[-1]),(71,13,128))
        with self.assertRaisesRegex(ValueError,"requires action"):
            runner.adapter.advance_context(state,batch["context"])

    def test_wm_training_and_validation_never_read_future_true_timers(self):
        runner,starts,_=full_fixture();count,steps=3,runner.cfg.wm_window
        expand=lambda tensor:tensor[:,None].expand(count,steps,*tensor.shape[1:]).clone()
        batch=dict(state=expand(starts["state"]),next_state=expand(starts["state"]),
            action=expand(starts["previous_action"]),context=expand(starts["context"]),
            contact_cost=torch.zeros(count,steps,13,dtype=torch.float64),
            depth_stack=expand(starts["depth_stack"]),next_depth_stack=expand(starts["depth_stack"]),
            mask=torch.ones(count,steps,dtype=torch.bool))
        changed={key:value.clone() for key,value in batch.items()}
        changed["context"][:,1:,96:100]=123.
        for kernel in (False,True):
            with self.subTest(compiled_loss_kernel=kernel):
                engine=LatentWorldModelKernel(runner) if kernel else None
                def run(values):
                    if engine is not None:return engine(engine.prepare(values,update_statistics=False))[0]
                    result=rollout_latent_model_metrics(runner,values)
                    return torch.stack((result["model_error"],result["contact_error"],result["latent_error"]))
                expected=run(batch);actual=run(changed)
                torch.testing.assert_close(actual,expected,rtol=0,atol=0)
                seen=[];method=runner.world_model.step_with_latent
                def record(state,action,context,latent):
                    result=method(state,action,context,latent)
                    seen.append((context.detach().clone(),result[1].detach().clone()))
                    return result
                with patch.object(runner.world_model,"step_with_latent",side_effect=record):run(changed)
                for i in range(1,steps):
                    torch.testing.assert_close(seen[i][0][:,96:100],seen[i-1][1][:,9:13])
                    torch.testing.assert_close(seen[i][0][:,38:67],batch["action"][:,i-1])

    def test_full_trajectory_aot_values_and_actor_gradients_match_eager(self):
        runner,batch,noises=full_fixture()
        with frozen_parameters(runner.world_model,runner.discriminator,runner.critic):
            eager=runner.imagine(batch,noises=noises)
            gradient=torch.autograd.grad(eager["objective"].sum(),tuple(runner.actor.parameters()),allow_unused=True)
            runner.cfg.execution_profile="supermimic"
            runner._latent_step_fn=torch.compile(LatentImaginedStep(runner),backend="aot_eager",fullgraph=True)
            runner._latent_trajectory_fn=torch.compile(LatentTrajectoryTerms(runner),backend="aot_eager",fullgraph=True)
            actual=runner.imagine(batch,noises=noises)
            actual_gradient=torch.autograd.grad(actual["objective"].sum(),tuple(runner.actor.parameters()),allow_unused=True)
        for name in eager:torch.testing.assert_close(actual[name],eager[name],rtol=1e-8,atol=1e-10)
        for expected,observed in zip(gradient,actual_gradient):
            if expected is None:self.assertIsNone(observed)
            else:
                self.assertTrue(torch.isfinite(observed).all())
                torch.testing.assert_close(observed,expected,rtol=1e-7,atol=1e-9)


if __name__=="__main__":unittest.main()
