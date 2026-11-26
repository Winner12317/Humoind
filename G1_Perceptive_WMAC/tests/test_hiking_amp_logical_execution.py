"""AMP logical capture preserves all 512-sized GP steps and native AdamW."""
from __future__ import annotations

import copy
import os
import statistics
import time
import unittest

import torch

from rsl_rl.modules.discriminator import Discriminator
from src.wmac_perceptive.amp_execution import AmpLogicalBatchExecution,AmpMicrobatchGraph
from src.wmac_perceptive.hiking_amp_budget import discriminator_step


def options(optimizer):
    return [{key:value for key,value in group.items() if key!='params'} for group in optimizer.param_groups]


class HikingAmpLogicalCpuTest(unittest.TestCase):
    def test_eager_logical_path_matches_loss_grad_norm_native_adamw_and_unchanged_options(self):
        torch.manual_seed(641)
        model=Discriminator(670,.25,[12,6],'cpu').double()
        reference=copy.deepcopy(model)
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,betas=(.9,.999),weight_decay=.01)
        reference_optimizer=torch.optim.AdamW(reference.parameters(),lr=1e-4,betas=(.9,.999),weight_decay=.01)
        before=copy.deepcopy(options(optimizer))
        execution=AmpLogicalBatchExecution(model,optimizer,16,4,gradient_penalty=5.,
            trunk_weight_decay=3e-4,head_weight_decay=.04,cuda_graphs=False)
        for step in range(1,6):
            policy=torch.randn(16,1340,dtype=torch.float64)[:,::2].requires_grad_(True)
            expert=torch.randn_like(policy).requires_grad_(True)
            actual=execution.step(policy,expert)
            expected=discriminator_step(reference,reference_optimizer,zip(policy.split(4),expert.split(4)),16,
                gradient_penalty=5.,trunk_weight_decay=3e-4,head_weight_decay=.04)
            torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)
            for parameter,target in zip(model.parameters(),reference.parameters()):
                torch.testing.assert_close(parameter.grad,target.grad,rtol=1e-12,atol=1e-12)
                torch.testing.assert_close(parameter,target,rtol=1e-12,atol=1e-12)
                self.assertEqual(int(optimizer.state[parameter]['step']),step)
            self.assertIsNone(policy.grad)
            self.assertIsNone(expert.grad)
        self.assertEqual(options(optimizer),before)
        self.assertIsInstance(optimizer.param_groups[0]['lr'],float)
        self.assertFalse(optimizer.param_groups[0]['capturable'])
        self.assertEqual(execution._calls,5)

    def test_logical_shape_is_bounded_and_requires_full_microbatches(self):
        model=Discriminator(670,.25,[12,6],'cpu')
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4)
        with self.assertRaisesRegex(ValueError,'complete'):
            AmpLogicalBatchExecution(model,optimizer,17,4,gradient_penalty=5.,trunk_weight_decay=3e-4,
                head_weight_decay=.04,cuda_graphs=False)
        execution=AmpLogicalBatchExecution(model,optimizer,16,4,gradient_penalty=5.,
            trunk_weight_decay=3e-4,head_weight_decay=.04,cuda_graphs=False)
        with self.assertRaisesRegex(ValueError,'variable-size'):
            execution.step(torch.randn(12,670),torch.randn(12,670))
        self.assertFalse(optimizer.state)
        self.assertEqual(execution._calls,0)


@unittest.skipUnless(os.environ.get('WMAC_RUN_CUDA_TESTS')=='1','Optional AMP logical CUDA capture tests')
class HikingAmpLogicalCudaTest(unittest.TestCase):
    reports=[]

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():raise unittest.SkipTest('CUDA unavailable')
        if os.environ.get('WMAC_AMP_CUDA_MEMORY_FRACTION'):
            torch.cuda.set_per_process_memory_fraction(float(os.environ['WMAC_AMP_CUDA_MEMORY_FRACTION']))
        cls.previous_tf32=torch.backends.cuda.matmul.allow_tf32

    @classmethod
    def tearDownClass(cls):torch.backends.cuda.matmul.allow_tf32=cls.previous_tf32

    def compare(self,logical,micro,hidden,dtype,steps):
        torch.backends.cuda.matmul.allow_tf32=dtype==torch.float32
        torch.manual_seed(20261118)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model=Discriminator(670,.25,hidden,'cuda').to(dtype=dtype)
        reference=copy.deepcopy(model)
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,betas=(.9,.999),weight_decay=.01,fused=True)
        reference_optimizer=torch.optim.AdamW(reference.parameters(),lr=1e-4,betas=(.9,.999),weight_decay=.01,fused=True)
        before=copy.deepcopy(options(optimizer))
        execution=AmpLogicalBatchExecution(model,optimizer,logical,micro,gradient_penalty=5.,
            trunk_weight_decay=3e-4,head_weight_decay=.04)
        baseline=AmpMicrobatchGraph(reference,micro,5.)
        observed_inputs=[]
        hook=model.register_forward_pre_hook(lambda module,args:observed_inputs.append(
            (tuple(args[0].shape),args[0].is_leaf,args[0].grad_fn is None)))
        rtol,atol=(1e-9,1e-10) if dtype==torch.float64 else (3e-4,3e-6)
        rows=[]
        for step in range(1,steps+1):
            source=torch.randn(logical,1340 if step%2==0 else 670,device='cuda',dtype=dtype)
            policy=source[:,::2] if step%2==0 else source
            expert=torch.randn_like(policy)
            def run_reference():
                return discriminator_step(reference,reference_optimizer,zip(policy.split(micro),expert.split(micro)),logical,
                    gradient_penalty=5.,trunk_weight_decay=3e-4,head_weight_decay=.04,execution=baseline)
            def timed(call):
                torch.cuda.synchronize()
                started=time.perf_counter()
                value=call()
                torch.cuda.synchronize()
                return value,time.perf_counter()-started
            if step%2:
                expected,micro_seconds=timed(run_reference)
                actual,logical_seconds=timed(lambda:execution.step(policy,expert))
            else:
                actual,logical_seconds=timed(lambda:execution.step(policy,expert))
                expected,micro_seconds=timed(run_reference)
            torch.testing.assert_close(actual,expected,rtol=rtol,atol=atol)
            parameter_error=gradient_error=moment_error=0.
            for parameter,target in zip(model.parameters(),reference.parameters()):
                torch.testing.assert_close(parameter.grad,target.grad,rtol=rtol,atol=atol)
                torch.testing.assert_close(parameter,target,rtol=rtol,atol=atol)
                self.assertEqual(int(optimizer.state[parameter]['step']),step)
                parameter_error=max(parameter_error,float((parameter-target).abs().max()))
                gradient_error=max(gradient_error,float((parameter.grad-target.grad).abs().max()))
                for key in ('exp_avg','exp_avg_sq'):
                    torch.testing.assert_close(optimizer.state[parameter][key],reference_optimizer.state[target][key],rtol=rtol,atol=atol)
                    moment_error=max(moment_error,float((optimizer.state[parameter][key]-reference_optimizer.state[target][key]).abs().max()))
            self.assertEqual(execution.real_warmup_logical_updates,min(step,3))
            self.assertEqual(execution.capture_count,int(step>=4))
            self.assertEqual(execution.replay_count,max(0,step-3))
            self.assertEqual(options(optimizer),before)
            self.assertFalse(any(value.requires_grad for value in execution._inputs))
            rows.append(dict(step=step,micro_graph_seconds=micro_seconds,logical_graph_seconds=logical_seconds,
                parameter_max_abs_difference=parameter_error,gradient_max_abs_difference=gradient_error,
                moment_max_abs_difference=moment_error,captured=execution.graph is not None,
                logical_replays=execution.replay_count,loss_gp_norm=actual.tolist()))
            # Other runner phases are allowed to clear all discriminator .grad.
            optimizer.zero_grad(set_to_none=True)
            reference_optimizer.zero_grad(set_to_none=True)
        hook.remove()
        self.assertEqual(set(observed_inputs),{((2*micro,670),True,True)})
        self.assertEqual(execution._calls,steps)
        steady=rows[4:]
        report=dict(logical_batch=logical,microbatch=micro,hidden=hidden,dtype=str(dtype),
            optimizer_options_unchanged=True,real_optimizer_updates=steps,
            logical_warmup_updates=execution.real_warmup_logical_updates,
            logical_captures=execution.capture_count,logical_replays=execution.replay_count,
            baseline_micro_replays=baseline.replay_count,rows=rows,
            peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
            peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
            static_label_inputs_mib=sum(value.numel()*value.element_size() for value in execution._inputs)/2**20,
            notes=['No simulator or full training is started; the existing flat training remains active.',
                   'Timing alternates paths on the same inputs; TF32 comparisons use stated numerical tolerances.',
                   'Each GP leaf has 2*microbatch rows; full static label inputs never require gradients.'])
        if steady:
            report.update(steady_median_micro_graph_seconds=statistics.median(row['micro_graph_seconds'] for row in steady),
                steady_median_logical_graph_seconds=statistics.median(row['logical_graph_seconds'] for row in steady))
        self.reports.append(report)
        return execution

    def test_small_double_logical_capture_real_updates_and_rebinding(self):
        self.compare(16,4,[32,16],torch.float64,6)

    def test_actual_4096_environment_budget_tf32_discriminator_and_optimizer(self):
        self.compare(24576,512,[1024,512],torch.float32,8)


if __name__=='__main__':unittest.main()
