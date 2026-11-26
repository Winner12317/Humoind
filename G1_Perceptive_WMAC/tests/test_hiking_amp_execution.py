"""Opt-in CUDA comparisons of real AMP warmup, GP capture and optimizer steps."""
from __future__ import annotations

import copy
from dataclasses import fields,is_dataclass,replace
import os
from types import SimpleNamespace
import unittest

import torch

from rsl_rl.modules.discriminator import Discriminator
from src.wmac_perceptive.amp_execution import AmpMicrobatchGraph,prediction_and_penalty
from src.wmac_perceptive.hiking_amp_budget import discriminator_step


@unittest.skipUnless(os.environ.get('WMAC_RUN_CUDA_TESTS')=='1','Optional small AMP CUDA execution tests')
class HikingAmpCudaExecutionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest('CUDA unavailable')
        if os.environ.get('WMAC_AMP_CUDA_MEMORY_FRACTION'):
            torch.cuda.set_per_process_memory_fraction(float(os.environ['WMAC_AMP_CUDA_MEMORY_FRACTION']))
        cls.original_tf32=torch.backends.cuda.matmul.allow_tf32

    @classmethod
    def tearDownClass(cls):
        torch.backends.cuda.matmul.allow_tf32=cls.original_tf32

    def test_real_microbatches_before_during_after_capture_accumulate_exactly_once(self):
        torch.backends.cuda.matmul.allow_tf32=False
        torch.manual_seed(186)
        model=Discriminator(670,.25,[32,16],'cuda').double()
        reference=copy.deepcopy(model)
        execution=AmpMicrobatchGraph(model,4,5.)
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,betas=(.9,.999),weight_decay=.01,fused=True)
        with execution.context():
            execution.begin_step()
            for index in range(10):
                policy=torch.randn(4,670,device='cuda',dtype=torch.float64)
                expert=torch.randn_like(policy)
                examples=torch.cat((policy,expert)).requires_grad_(True)
                prediction,penalty=prediction_and_penalty(reference,examples,4,5.)
                ((prediction+penalty)/10).backward()
                actual=execution.accumulate(policy,expert,40)
                torch.testing.assert_close(torch.stack(actual),torch.stack((prediction.detach(),penalty.detach())),
                    rtol=1e-10,atol=1e-11)
                for parameter,expected in zip(model.parameters(),reference.parameters()):
                    torch.testing.assert_close(parameter.grad,expected.grad,rtol=1e-9,atol=1e-11)
                self.assertEqual(execution.real_warmup_microbatches,min(index+1,3))
                self.assertEqual(execution.capture_count,int(index>=3))
                self.assertEqual(execution.replay_count,max(0,index-2))
                # Warmup and capture do not silently take optimizer updates.
                self.assertFalse(optimizer.state)

    def _logical_steps(self,dtype,tf32,hidden,sizes,microbatch):
        torch.backends.cuda.matmul.allow_tf32=tf32
        torch.manual_seed(471)
        model=Discriminator(670,.25,hidden,'cuda').to(dtype=dtype)
        reference=copy.deepcopy(model)
        execution=AmpMicrobatchGraph(model,microbatch,5.)
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,betas=(.9,.999),weight_decay=.01,fused=True)
        reference_optimizer=torch.optim.AdamW(reference.parameters(),lr=1e-4,betas=(.9,.999),weight_decay=.01,fused=True)
        rtol,atol=(1e-9,1e-10) if dtype==torch.float64 else (3e-4,3e-6)
        for step,size in enumerate(sizes,1):
            policy=torch.randn(size,670,device='cuda',dtype=dtype)
            expert=torch.randn_like(policy)
            expected=discriminator_step(reference,reference_optimizer,zip(policy.split(microbatch),expert.split(microbatch)),size,
                gradient_penalty=5.,trunk_weight_decay=3e-4,head_weight_decay=.04)
            actual=discriminator_step(model,optimizer,zip(policy.split(microbatch),expert.split(microbatch)),size,
                gradient_penalty=5.,trunk_weight_decay=3e-4,head_weight_decay=.04,execution=execution)
            torch.testing.assert_close(actual,expected,rtol=rtol,atol=atol)
            for parameter,target in zip(model.parameters(),reference.parameters()):
                torch.testing.assert_close(parameter.grad,target.grad,rtol=rtol,atol=atol)
                torch.testing.assert_close(parameter,target,rtol=rtol,atol=atol)
                self.assertEqual(int(optimizer.state[parameter]['step']),step)
                for key in ('exp_avg','exp_avg_sq'):
                    torch.testing.assert_close(optimizer.state[parameter][key],reference_optimizer.state[target][key],rtol=rtol,atol=atol)
            # Simulate the runner clearing .grad between rollout calls; a
            # graph replay must rebind its captured buffers on the next step.
            optimizer.zero_grad(set_to_none=True)
        self.assertEqual(execution.real_warmup_microbatches,3)
        self.assertEqual(execution.capture_count,1)
        return execution

    def test_logical_steps_regularization_norm_optimizer_and_irregular_tails(self):
        execution=self._logical_steps(torch.float64,False,[32,16],(16,20,9,16),4)
        self.assertEqual(execution.replay_count,12)

    def test_real_1024_512_discriminator_tf32_fixed_512_microbatch(self):
        execution=self._logical_steps(torch.float32,True,[1024,512],(2048,2048,2048),512)
        self.assertEqual(execution.replay_count,9)

    def test_compiled_real_g1_fk_features_match_eager_without_training_updates(self):
        from test_reference_policy_step import reference_loop_fixture
        runner,batch,_=reference_loop_fixture()
        def move(value):
            if torch.is_tensor(value):
                return value.to(device='cuda',dtype=torch.float32 if value.is_floating_point() else value.dtype)
            if is_dataclass(value):
                return replace(value,**{field.name:move(getattr(value,field.name)) for field in fields(value)})
            if isinstance(value,list):return [move(item) for item in value]
            if isinstance(value,tuple):return tuple(move(item) for item in value)
            if isinstance(value,dict):return {key:move(item) for key,item in value.items()}
            return value
        for name,value in tuple(vars(runner.adapter).items()):
            setattr(runner.adapter,name,move(value))
        runner.adapter.device=torch.device('cuda')
        encoder=torch.compile(runner.adapter.amp_features,fullgraph=True,dynamic=False)
        state=batch['state'].float().cuda().repeat(43,1)
        with torch.no_grad():
            for index in range(2):
                inputs=state.clone()
                inputs[:,13:42]+=.01*index
                actual=encoder(inputs)
                expected=runner.adapter.amp_features(inputs)
                torch.testing.assert_close(actual,expected,rtol=3e-5,atol=3e-6)
                self.assertFalse(actual.requires_grad)

    def test_runner_graph_keeps_twenty_steps_and_five_exposures_including_final_tail(self):
        from src.wmac.replay import ReplayBuffer
        from src.wmac_perceptive.hiking_amp import configure_replay
        from src.wmac_perceptive.runner import PerceptiveRunner
        from test_hiking_amp_budget import add_steps
        runner=object.__new__(PerceptiveRunner)
        runner.cfg=SimpleNamespace(amp_feature_schema='hiking_v1',locomotion_objective='hiking_full_v1',
            hiking_training_version=2,iterations=4,effective_discriminator_batch_size=8,
            gradient_penalty=5.,discriminator_trunk_weight_decay=3e-4,discriminator_head_weight_decay=.04,
            compile=False,cuda_graphs=True)
        runner.device=torch.device('cuda')
        runner.replay=ReplayBuffer(48,8,runner.device,.25)
        configure_replay(runner.replay)
        runner.adapter=SimpleNamespace(amp_features=lambda states:states[...,:67])
        runner.experts=SimpleNamespace(groups=[torch.randn(60,67,device='cuda'),None],
            _hiking_history_ends=torch.arange(9,60,device='cuda'))
        runner.discriminator=Discriminator(670,.25,[32,16],'cuda')
        runner.opt_disc=torch.optim.AdamW(runner.discriminator.parameters(),lr=1e-4,betas=(.9,.999),weight_decay=.01,fused=True)
        records=[]
        for iteration,count in enumerate((10,14,3,2)):
            runner.iteration=iteration
            add_steps(runner.replay,count)
            records.append(runner.update_discriminator())
            if iteration==1:
                self.assertFalse(hasattr(runner,'_hiking_amp_gp_execution'))
        self.assertEqual([row['discriminator_updates_applied'] for row in records],[0,20,0,20])
        self.assertEqual(records[-1]['amp_policy_exposures_total'],29*8*5)
        self.assertEqual(len({tuple(row) for row in records}),1)
        execution=runner._hiking_amp_gp_execution
        self.assertEqual((execution.real_warmup_microbatches,execution.capture_count,execution.replay_count),(3,1,17))
        logical=runner._hiking_amp_logical_execution
        self.assertEqual((logical.real_warmup_logical_updates,logical.capture_count,logical.replay_count),(3,1,17))
        for parameter in runner.discriminator.parameters():
            self.assertEqual(int(runner.opt_disc.state[parameter]['step']),40)
            self.assertIsNone(parameter.grad)

    def test_invalid_rollout_uses_micro_then_complete_rollout_rebinds_logical_gradients(self):
        from src.wmac.replay import ReplayBuffer
        from src.wmac_perceptive.hiking_amp import configure_replay
        from src.wmac_perceptive.runner import PerceptiveRunner
        from test_hiking_amp_budget import add_steps
        runner=object.__new__(PerceptiveRunner)
        runner.cfg=SimpleNamespace(amp_feature_schema='hiking_v1',locomotion_objective='hiking_full_v1',
            hiking_training_version=2,iterations=3,effective_discriminator_batch_size=8,
            gradient_penalty=5.,discriminator_trunk_weight_decay=3e-4,discriminator_head_weight_decay=.04,
            compile=False,cuda_graphs=True)
        runner.device=torch.device('cuda')
        runner.replay=ReplayBuffer(80,8,runner.device,.25)
        configure_replay(runner.replay)
        runner.adapter=SimpleNamespace(amp_features=lambda states:states[...,:67])
        runner.experts=SimpleNamespace(groups=[torch.randn(60,67,device='cuda'),None],
            _hiking_history_ends=torch.arange(9,60,device='cuda'))
        runner.discriminator=Discriminator(670,.25,[32,16],'cuda')
        runner.opt_disc=torch.optim.AdamW(runner.discriminator.parameters(),lr=1e-4,betas=(.9,.999),weight_decay=.01,fused=True)
        rows=[]
        for iteration in range(3):
            runner.iteration=iteration
            add_steps(runner.replay,24,invalid={(27,2)})
            rows.append(runner.update_discriminator())
            if iteration==1:self.assertEqual(runner._hiking_amp_logical_execution._calls,20)
        self.assertEqual([row['discriminator_updates_applied'] for row in rows],[20,20,20])
        self.assertEqual([row['amp_invalid_windows_dropped'] for row in rows],[0,1,0])
        self.assertEqual(runner._hiking_amp_logical_execution._calls,40)
        self.assertEqual(runner._hiking_amp_logical_execution.capture_count,1)
        self.assertEqual(rows[-1]['amp_policy_exposures_total'],(72*8-1)*5)
        for parameter in runner.discriminator.parameters():
            self.assertEqual(int(runner.opt_disc.state[parameter]['step']),60)


if __name__=='__main__':
    unittest.main()
