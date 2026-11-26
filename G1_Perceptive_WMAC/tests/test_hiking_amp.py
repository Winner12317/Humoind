"""Hiking AMP source parity, reset-safe replay histories and imagined gradients."""
from __future__ import annotations

from dataclasses import replace
import copy
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch
from torch import nn

from src.wmac.replay import ReplayBuffer
from src.wmac.networks import frozen_parameters
from src.wmac_perceptive.hiking_amp import (
    FRAME_DIM, HISTORY_LENGTH, frame, flatten_history, sliding_histories,
    configure_replay, before_replay_add, after_replay_add, add_sample_history,
)


class HikingAmpFeaturesTest(unittest.TestCase):
    @unittest.skipUnless(os.environ.get('WMAC_RUN_CUDA_TESTS')=='1', 'Optional CUDA capture test')
    def test_frame_can_be_captured_without_cpu_to_gpu_constant_copy(self):
        if not torch.cuda.is_available():
            self.skipTest('CUDA unavailable')
        q=torch.randn(3,4,device='cuda');q/=q.norm(dim=-1,keepdim=True)
        values=(torch.randn(3,29,device='cuda'),torch.randn(3,29,device='cuda'),q,
                torch.randn(3,3,device='cuda'),torch.randn(3,3,device='cuda'),
                torch.zeros(29,device='cuda'))
        stream=torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):frame(*values)
        torch.cuda.current_stream().wait_stream(stream)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual=frame(*values)
        values[0].add_(.1)
        graph.replay()
        expected=frame(*values)
        torch.testing.assert_close(actual,expected,rtol=1e-6,atol=1e-6)

    def test_features_and_history_match_executed_upstream_functions(self):
        path = Path('/tmp/hiking_amp_reference_oracle.py')
        if not path.exists():
            self.skipTest('Pinned Hiking/IsaacLab source oracle unavailable')
        spec = importlib.util.spec_from_file_location('hiking_source_oracle',path)
        reference = importlib.util.module_from_spec(spec);spec.loader.exec_module(reference)
        torch.manual_seed(123)
        q = torch.randn(7,4);q /= q.norm(dim=-1,keepdim=True)
        values = (torch.randn(7,29),torch.randn(7,29),q,
                  torch.randn(7,3),torch.randn(7,3),torch.randn(7,29),torch.randn(7,29))
        expected = reference.build_frame(*values)
        actual = frame(*values)
        torch.testing.assert_close(actual, expected, rtol=1e-6,atol=1e-6)
        history = torch.randn(7,10,67)
        torch.testing.assert_close(flatten_history(history),reference.flatten_history(history),rtol=0,atol=0)

    def test_imagined_history_retains_real_predecessors_and_live_gradients(self):
        initial = torch.arange(10.).view(1,10,1).expand(2,10,67)
        future = torch.arange(10.,13.).view(3,1,1).expand(3,2,67).clone().requires_grad_(True)
        windows = sliding_histories(initial,future)
        self.assertEqual(tuple(windows.shape),(3,2,10,67))
        for step in range(3):
            expected = torch.arange(step+1.,step+11.)
            torch.testing.assert_close(windows[step,0,:,0],expected)
        flatten_history(windows).sum().backward()
        torch.testing.assert_close(future.grad[:,0,0],torch.tensor([3.,2.,1.]))

    def test_absolute_translation_does_not_enter_amp(self):
        from test_perceptive_latent_runner import latent_runner_fixture
        runner,batch,_ = latent_runner_fixture()
        adapter=runner.adapter
        adapter.amp_feature_schema='hiking_v1'
        adapter._hiking_torso_com_offset=batch['state'].new_tensor([.01,0.,.02])
        states=batch['state'].clone()
        expected=adapter.amp_features(states)
        states[:,:3]+=states.new_tensor([5.,-3.,.8])
        actual=adapter.amp_features(states)
        torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)
        self.assertEqual(actual.shape[-1],67)


class HikingReplayHistoryTest(unittest.TestCase):
    def test_overwritten_states_reset_padding_and_navigation_boundaries(self):
        for capacity in (3,8,17):
            replay=ReplayBuffer(capacity,3)
            configure_replay(replay)
            reference=[];hist=None;pending=torch.zeros(3,dtype=torch.bool)
            for step in range(32):
                states=torch.arange(3.).view(3,1).expand(3,71).clone()*100+step
                hist=(states[:,None].expand(-1,10,-1).clone() if hist is None else
                      torch.cat((hist[:,1:],states[:,None]),1))
                hist=torch.where(pending[:,None,None],states[:,None].expand_as(hist),hist)
                reference.append(hist.clone())
                done=torch.tensor([step in (3,19),step in (8,24),step in (13,26)])
                navigation=torch.full((3,),step in (5,7,21),dtype=torch.bool)
                batch={'state':states,'next_state':states+1,'terminated':done,
                       'timeout':torch.zeros(3,dtype=torch.bool),'boundary':navigation}
                before_replay_add(replay);accepted=replay.add(batch);after_replay_add(replay,batch,accepted)
                pending=done
                for time in range(max(0,step-capacity+1),step+1):
                    sample={'time_index':torch.full((3,),time,dtype=torch.long),'env_index':torch.arange(3)}
                    actual=add_sample_history(replay,sample)['amp_state_history']
                    torch.testing.assert_close(actual,reference[time],rtol=0,atol=0)
            self.assertEqual(replay._amp_overflow_states.shape,(9,3,71))
            self.assertNotIn('amp_state_history',replay.data)

    def test_native_latent_and_compressed_add_hooks_keep_same_histories(self):
        from test_perceptive_compact_gpu_replay import SPEC,frame as depth_frame,native_batch
        from src.wmac_perceptive.compact_gpu_replay import CompressedLatentReplay
        from src.wmac_perceptive.latent_replay import LatentPerceptiveReplay
        generator=torch.Generator().manual_seed(7)
        history=torch.stack([depth_frame(generator) for _ in range(SPEC.ring_length)],1)
        instances=[LatentPerceptiveReplay(4,4,spec=SPEC),CompressedLatentReplay(4,4,device='cpu',spec=SPEC)]
        for replay in instances:configure_replay(replay)
        instances[1].bootstrap_processed_history(history)
        for step in range(16):
            from src.wmac_perceptive.perception import advance_depth_ring
            following=advance_depth_ring(history,depth_frame(generator),SPEC)
            batch=native_batch(step,history,following,invalid=False,bad_image=False)
            batch['state'][:,0]=step+100*torch.arange(4)
            for replay in instances:replay.add(batch)
            history=following
        samples=[]
        for replay in instances:
            torch.manual_seed(31)
            samples.append(replay.sample_starts(100,fields=('state','next_state')))
        self.assertIn('amp_state_history',samples[0])
        torch.testing.assert_close(samples[0]['amp_state_history'],samples[1]['amp_state_history'],rtol=0,atol=0)


class HikingAmpBpttTest(unittest.TestCase):
    def test_discriminator_update_matches_original_wasabi_loss_and_adamw(self):
        from rsl_rl.modules.discriminator import Discriminator
        from src.wmac_perceptive.runner import PerceptiveRunner
        oracle_path=Path('/tmp/hiking_amp_reference_oracle.py')
        source_path=Path('/tmp/instinct-rl-hiking-reference/instinct_rl/algorithms/wasabi.py')
        if not (oracle_path.exists() and source_path.exists()):
            self.skipTest('Pinned Hiking Wasabi source oracle unavailable')
        spec=importlib.util.spec_from_file_location('hiking_loss_source_oracle',oracle_path)
        oracle=importlib.util.module_from_spec(spec);spec.loader.exec_module(oracle)
        methods=oracle.extract(source_path,{'compute_amp_losses','compute_discriminator_gradient'},
            scope='WasabiAlgoMixin',namespace={'buffer_func':lambda obj,func,*args:func(obj,*args)})
        class Proxy(nn.Module):
            def __init__(self,model):super().__init__();self.inner=model
            def forward(self,x,masks=None):return self.inner(x)
            def logit_layer_weights(self):return self.inner.amp_linear.weight
        torch.manual_seed(40)
        runner=object.__new__(PerceptiveRunner)
        runner.cfg=SimpleNamespace(amp_feature_schema='hiking_v1',effective_discriminator_batch_size=4,
            discriminator_updates=1,gradient_penalty=5.,discriminator_trunk_weight_decay=3e-4,
            discriminator_head_weight_decay=.04)
        runner.device=torch.device('cpu')
        runner.discriminator=Discriminator(670,.25,[12,6],'cpu')
        runner.opt_disc=torch.optim.AdamW(runner.discriminator.parameters(),lr=1e-4,betas=(.9,.999),weight_decay=.01)
        runner.adapter=SimpleNamespace(amp_features=lambda x:x[...,:67])
        history=torch.randn(4,10,71);next_state=torch.randn(4,71);expert=torch.randn(4,10,67)
        batch={'amp_state_history':history,'state':history[:,-1],'next_state':next_state}
        runner.replay=SimpleNamespace(sample_starts=lambda *a,**kw:batch)
        runner.experts=SimpleNamespace(sample_histories=lambda size:expert)
        reference_model=copy.deepcopy(runner.discriminator)
        reference=SimpleNamespace(discriminator=Proxy(reference_model),device='cpu',
            discriminator_loss_func='MSELoss',discriminator_gradient_penalty_coef=5.,
            discriminator_backbone_gradient_only=False,discriminator_gradient_torlerance=0.,
            discriminator_logit_weight_decay_coef=.04)
        reference.compute_discriminator_gradient=lambda minibatch:methods['compute_discriminator_gradient'](reference,minibatch)
        policy_history=torch.cat((history[:,1:,:67],next_state[:,None,:67]),1)
        minibatch=SimpleNamespace(actor_states=flatten_history(policy_history),reference_states=flatten_history(expert),masks=None)
        losses,_,_=methods['compute_amp_losses'](reference,minibatch)
        total=losses['discriminator_loss']+5.*losses['discriminator_gradient_penalty']+3e-4*losses['discriminator_weight_decay']+.04*losses['logit_weight_decay']
        optimizer=torch.optim.AdamW(reference_model.parameters(),lr=1e-4,betas=(.9,.999),weight_decay=.01)
        total.backward();optimizer.step()
        actual=runner.update_discriminator()
        self.assertAlmostEqual(actual['discriminator_loss'],float(total.detach()),places=6)
        self.assertAlmostEqual(actual['discriminator_grad_penalty'],float(5.*losses['discriminator_gradient_penalty'].detach()),places=6)
        for parameter,expected in zip(runner.discriminator.parameters(),reference_model.parameters()):
            torch.testing.assert_close(parameter,expected,rtol=1e-6,atol=1e-7)

    def test_policy_bptt_uses_ten_frames_and_keeps_discriminator_frozen(self):
        from rsl_rl.modules.discriminator import Discriminator
        from test_perceptive_latent_runner import latent_runner_fixture
        runner,batch,noise=latent_runner_fixture(horizon=3)
        runner.cfg=replace(runner.cfg,amp_feature_schema='hiking_v1',motion_data_protocol='hiking_walk_v1')
        runner.adapter.amp_feature_schema='hiking_v1'
        runner.adapter._hiking_torso_com_offset=batch['state'].new_tensor([.01,0.,.02])
        runner.discriminator=Discriminator(670,.25,[12,6],'cpu').double()
        runner.amp_norm=nn.Identity()
        batch['amp_state_history']=batch['state'][:,None].expand(-1,10,-1).clone()
        inputs=[]
        hook=runner.discriminator.register_forward_pre_hook(lambda module,args:inputs.append(args[0].shape))
        with frozen_parameters(runner.world_model,runner.discriminator,runner.critic):
            rollout=runner.imagine(batch,noises=noise)
            (-rollout['objective'].mean()).backward()
        hook.remove()
        self.assertTrue(inputs)
        self.assertTrue(all(shape[-1]==670 for shape in inputs))
        self.assertTrue(all(p.grad is None for p in runner.discriminator.parameters()))
        gradients=[p.grad for p in runner.actor.policy.parameters() if p.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(bool(torch.isfinite(g).all()) for g in gradients))
        self.assertGreater(sum(float(g.square().sum()) for g in gradients),0.)


if __name__=='__main__':unittest.main()
