"""Standing command mode through replay, BPTT, compilation and saved protocols.

The small G1 runner fixtures isolate the new command/reward wiring. Setting a
standing-objective flag on those historical tiny fixtures is intentional:
they are not full official-data environment or training-config fixtures.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, replace
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from scripts.train_wmac_perceptive import parse_config
from src.tasks.amp_perceptive.commands import command_from_state
from src.tasks.amp_perceptive.env_cfg import g1_amp_perceptive_hiking_env_cfg
from src.wmac.networks import frozen_parameters
from src.wmac_perceptive.checkpoints import architecture_metadata, validate_checkpoint
from src.wmac_perceptive.compact_gpu_replay import CompressedLatentReplay
from src.wmac_perceptive.compiled_imagination import LatentImaginedStep, LatentTrajectoryTerms
from src.wmac_perceptive.config import (
    FORMAT, HIKING_STANDING_OBJECTIVE, HIKING_FULL_OBJECTIVE, LEGACY_LOCOMOTION_OBJECTIVE, HIKING_SENSOR_PROTOCOL,
    PerceptiveConfig, config_from_payload,
)
from src.wmac_perceptive.latent_perception import pack_processed_actor_input, unpack_processed_actor_input
from src.wmac_perceptive.latent_replay import LatentPerceptiveReplay
from src.wmac_perceptive.perception import advance_depth_ring, pack_actor_input, unpack_actor_input
from src.wmac_perceptive.reference_training import reference_value_ceiling
from src.wmac_perceptive.replay import PerceptiveReplay
from src.wmac_perceptive.runner import MODULES
from test_perceptive_compact_gpu_replay import SPEC, frame, native_batch
from test_perceptive_latent_runner import latent_runner_fixture
from test_perceptive_runner import runner_fixture


def standing_runner_fixture(*, latent, compiled=False):
    runner,batch,noises=(latent_runner_fixture() if latent else runner_fixture())
    runner.cfg.locomotion_objective=HIKING_STANDING_OBJECTIVE
    runner.adapter.reward_cfg["stand_still"]=g1_amp_perceptive_hiking_env_cfg(
        motion_data_protocol="hiking_walk_v1",locomotion_objective=HIKING_STANDING_OBJECTIVE).rewards["stand_still"]
    batch["command_standing"]=torch.tensor([True,False,True])
    # Both standing and turning have zero forward speed; only the explicit
    # Boolean mode distinguishes which must suppress heading correction.
    batch["desired_speed"].zero_()
    batch["context"][:,33:]=0.
    batch["command"]=command_from_state(batch["state"],batch["navigation_target"],
        batch["desired_speed"],standing_mask=batch["command_standing"])
    unpack=unpack_processed_actor_input if latent else unpack_actor_input
    pack=pack_processed_actor_input if latent else pack_actor_input
    proprio,visual=unpack(batch["actor_history"],runner.spec)
    history=proprio.reshape(-1,4,96).clone()
    history[:,:,6:9]=batch["command"][:,None]
    batch["actor_history"]=pack(history.flatten(1),visual,runner.spec)
    # Commands have no observation noise in the actual task.
    noises[:,:,6:9]=0.
    if compiled:
        runner.cfg.execution_profile="supermimic"
        runner._latent_step_fn=torch.compile(LatentImaginedStep(runner),
            backend="aot_eager",fullgraph=True,dynamic=False)
        runner._latent_trajectory_fn=torch.compile(LatentTrajectoryTerms(runner),
            backend="aot_eager",fullgraph=True,dynamic=False)
    return runner,batch,noises


def run_with_command_capture(runner,batch,noises):
    """Record only at Python boundaries, outside compiled tensor kernels."""
    seen={"actor_histories":[],"new_frames":[],"reward_commands":[],"critic_commands":[]}
    original_critic=runner.adapter.critic_features
    def critic(states,previous,commands,contexts):
        seen["critic_commands"].append(commands.detach().clone())
        return original_critic(states,previous,commands,contexts)
    if runner.uses_fast_execution:
        step_fn,trajectory_fn=runner._latent_step_fn,runner._latent_trajectory_fn
        def step(*args):
            seen["actor_histories"].append(args[2].detach().clone())
            result=step_fn(*args)
            seen["new_frames"].append(result[2][:,-1].detach().clone())
            return result
        def terms(*args):
            seen["reward_commands"].append(args[3].detach().clone())
            return trajectory_fn(*args)
        with patch.object(runner,"_latent_step_fn",side_effect=step), \
                patch.object(runner,"_latent_trajectory_fn",side_effect=terms), \
                patch.object(runner.adapter,"critic_features",side_effect=critic):
            output=runner.imagine(batch,noises=noises)
    else:
        original_frame,original_reward=runner.adapter.actor_frame,runner.adapter.task_reward
        method="forward_latent" if runner.uses_latent else "forward"
        original_actor=getattr(runner.actor,method)
        def actor(observation,*args):
            proprio=observation if runner.uses_latent else unpack_actor_input(observation,runner.spec)[0]
            seen["actor_histories"].append(proprio.reshape(-1,4,96).detach().clone())
            return original_actor(observation,*args)
        def frame_fn(*args,**kwargs):
            result=original_frame(*args,**kwargs)
            seen["new_frames"].append(result.detach().clone())
            return result
        def reward(*args,**kwargs):
            seen["reward_commands"].append(args[4].detach().clone())
            return original_reward(*args,**kwargs)
        with patch.object(runner.actor,method,side_effect=actor), \
                patch.object(runner.adapter,"actor_frame",side_effect=frame_fn), \
                patch.object(runner.adapter,"task_reward",side_effect=reward), \
                patch.object(runner.adapter,"critic_features",side_effect=critic):
            output=runner.imagine(batch,noises=noises)
    return output,seen


class HikingStandingIntegrationTest(unittest.TestCase):
    def assert_commands(self,runner,batch,seen):
        mask=batch["command_standing"]
        self.assertEqual(len(seen["actor_histories"]),runner.cfg.imagination_horizon)
        self.assertEqual(len(seen["new_frames"]),runner.cfg.imagination_horizon)
        for history in seen["actor_histories"]:
            torch.testing.assert_close(history[mask,:,6:9],torch.zeros_like(history[mask,:,6:9]),rtol=0,atol=0)
            self.assertTrue(bool((history[~mask,:,8].abs()>.1).all()))
        for observation in seen["new_frames"]:
            torch.testing.assert_close(observation[mask,6:9],torch.zeros_like(observation[mask,6:9]),rtol=0,atol=0)
            self.assertTrue(bool((observation[~mask,8].abs()>.1).all()))
        self.assertEqual(len(seen["reward_commands"]),1)
        self.assertEqual(len(seen["critic_commands"]),1)
        for command in (*seen["reward_commands"],*seen["critic_commands"]):
            torch.testing.assert_close(command[:,mask],torch.zeros_like(command[:,mask]),rtol=0,atol=0)
            torch.testing.assert_close(command[:,~mask,:2],torch.zeros_like(command[:,~mask,:2]),rtol=0,atol=0)
            self.assertTrue(bool((command[:,~mask,2].abs()>.1).all()))
        # Includes the final bootstrap command, not only reward transitions.
        self.assertEqual(seen["critic_commands"][0].shape[0],runner.cfg.imagination_horizon+1)

    def test_pixel_and_latent_eager_keep_standing_through_all_imagined_consumers(self):
        for latent in (False,True):
            with self.subTest(latent=latent):
                runner,batch,noises=standing_runner_fixture(latent=latent)
                with frozen_parameters(runner.world_model,runner.discriminator,runner.critic):
                    output,seen=run_with_command_capture(runner,batch,noises)
                self.assertTrue(bool(torch.isfinite(output["objective"]).all()))
                self.assert_commands(runner,batch,seen)
                missing={key:value for key,value in batch.items() if key!="command_standing"}
                with self.assertRaisesRegex(KeyError,"command_standing"):
                    runner.imagine(missing,noises=noises)

    def test_fullgraph_aot_latent_matches_eager_values_actor_gradients_and_commands(self):
        eager,starts,noises=standing_runner_fixture(latent=True)
        compiled,compiled_starts,compiled_noises=standing_runner_fixture(latent=True,compiled=True)
        for name in (*MODULES,"latent_decoder"):
            getattr(compiled,name).load_state_dict(getattr(eager,name).state_dict())
        results=[];gradients=[]
        for runner,batch,noise in ((eager,starts,noises),(compiled,compiled_starts,compiled_noises)):
            with frozen_parameters(runner.world_model,runner.discriminator,runner.critic):
                output,seen=run_with_command_capture(runner,batch,noise)
                gradient=torch.autograd.grad(output["objective"].sum(),tuple(runner.actor.parameters()),allow_unused=True)
            self.assert_commands(runner,batch,seen)
            results.append(output);gradients.append(gradient)
        for key in results[0]:
            torch.testing.assert_close(results[1][key],results[0][key],rtol=1e-7,atol=1e-9)
        for expected,actual in zip(*gradients):
            if expected is None:self.assertIsNone(actual)
            else:torch.testing.assert_close(actual,expected,rtol=2e-6,atol=2e-9)

    def test_all_replay_representations_preserve_boolean_mode_at_sampled_rows(self):
        generator=torch.Generator().manual_seed(188)
        history=torch.stack([frame(generator) for _ in range(SPEC.ring_length)],1)
        compressed=CompressedLatentReplay(8,4,"cpu",spec=SPEC)
        compressed.bootstrap_processed_history(history)
        replay_types=(compressed,LatentPerceptiveReplay(8,4,"cpu",spec=SPEC),
                      PerceptiveReplay(8,4,"cpu",spec=SPEC))
        for step in range(5):
            following=advance_depth_ring(history,frame(generator),SPEC)
            batch=native_batch(step,history,following)
            batch["command_standing"]=(batch["state"][:,0].long()+batch["state"][:,1].long())%2==0
            for replay in replay_types:
                row=batch
                if type(replay) is PerceptiveReplay:
                    row={key:value for key,value in batch.items() if key not in ("depth_stack","next_depth_stack")}
                    row.update(depth_ring=history,depth_frame=history[:,-1],next_depth_frame=following[:,-1])
                replay.add(row)
            history=following
        for replay in replay_types:
            with self.subTest(replay=type(replay).__name__):
                self.assertEqual(replay.data["command_standing"].dtype,torch.bool)
                for windows in (False,True):
                    fields=("state","command_standing")
                    sample=(replay.sample_windows(128,3,0.,fields) if windows else replay.sample_starts(128,0.,fields))
                    self.assertEqual(sample["command_standing"].dtype,torch.bool)
                    expected=(sample["state"][...,0].long()+sample["state"][...,1].long())%2==0
                    valid=sample["mask"].bool() if windows else torch.ones_like(expected)
                    torch.testing.assert_close(sample["command_standing"][valid],expected[valid],rtol=0,atol=0)

    def test_reference_value_ceiling_includes_positive_offset_with_negative_weight(self):
        adapter=SimpleNamespace(reward_scale=.02,reward_cfg={"tracking":SimpleNamespace(weight=2.)})
        cfg=SimpleNamespace(gamma=.99,task_weight=1.,style_weight=1.,amp_reward_coef=.25)
        previous=reference_value_ceiling(adapter,cfg)
        adapter.reward_cfg["stand_still"]=SimpleNamespace(weight=-.3,params={"offset":4.})
        updated=reference_value_ceiling(adapter,cfg)
        self.assertAlmostEqual(updated-previous,1.2*.02/(1-.99),places=12)

    def test_fresh_default_enables_standing_while_old_payload_and_resume_stay_legacy(self):
        fresh=parse_config([])
        self.assertEqual(fresh.locomotion_objective,HIKING_FULL_OBJECTIVE)
        fresh.validate()
        old=replace(parse_config(["--hiking-training-version", "1"]),locomotion_objective=LEGACY_LOCOMOTION_OBJECTIVE,
                    sensor_protocol=HIKING_SENSOR_PROTOCOL,replay_storage="frame_uint8",
                    collision_horizontal_scale=.1,
                    moe_gate_hidden=(256,),
                    standing_expert_source="none",mirrored_motion_root=".perceptive_cache/motions/g1_hiking_walk_v1")
        metadata=architecture_metadata(old)
        self.assertNotIn("locomotion_objective",metadata)
        self.assertNotIn("standing_reward",metadata)
        self.assertNotIn("standing_probability_per_resample",metadata["commands"])
        saved=asdict(old);saved.pop("locomotion_objective");saved.pop("standing_expert_source")
        payload={"format":FORMAT,"config":saved,"architecture":copy.deepcopy(metadata),
                 "terrain_fingerprint":"fixture-terrain","motion_manifest_sha256":"fixture-motion"}
        restored=config_from_payload(payload)
        self.assertEqual(restored.locomotion_objective,LEGACY_LOCOMOTION_OBJECTIVE)
        self.assertEqual(architecture_metadata(restored),metadata)
        self.assertEqual(validate_checkpoint(payload,restored),{})
        self.assertNotIn("locomotion_objective",payload["config"])
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"old.pt";torch.save(payload,path)
            self.assertEqual(parse_config(["--resume",str(path)]).locomotion_objective,LEGACY_LOCOMOTION_OBJECTIVE)
            with self.assertRaisesRegex(ValueError,"Ordinary resume cannot change locomotion_objective"):
                parse_config(["--resume",str(path),"--locomotion-objective",HIKING_STANDING_OBJECTIVE])
        updated=architecture_metadata(fresh)
        self.assertEqual(updated["commands"]["standing_probability_per_resample"],.05)
        self.assertEqual(updated["standing_reward"]["maximum_unscaled_contribution"],1.2)


if __name__=="__main__":unittest.main()
