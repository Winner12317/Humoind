"""Reduced replay/graph image inputs preserve loss, gradients and online E."""
from __future__ import annotations

import copy
import unittest

import torch

from src.wmac_perceptive.latent_execution import LatentWorldModelKernel,WMExecution
from src.wmac_perceptive.wm_macro_execution import WMMacroExecution
import test_hiking_material_conditioning as material_checks
from test_perceptive_latent_execution import fixture


class WMInputTransportTest(unittest.TestCase):
    def test_replay_draws_masks_and_uint8_values_match_full_window_after_wrap(self):
        replay,_,stream=material_checks.MaterialReplayTest().make_pair()
        for index in range(20):
            if index==11:replay.mark_boundary();stream.reset_all()
            replay.add(stream.batch())
        for split in ('train','validation','all'):
            torch.manual_seed(123)
            original=replay.sample_windows(17,3,fields=('state','next_state','action','context',
                'contact_cost','depth_stack','next_depth_stack'),split=split)
            torch.manual_seed(123)
            compact=replay.sample_model_windows(17,3,fields=('state','next_state','action','context',
                'contact_cost','depth_stack','next_depth_stack'),split=split)
            self.assertEqual(original.keys(),compact.keys())
            for name,value in original.items():
                if name=='depth_stack':
                    self.assertEqual(compact[name].shape[1],1)
                    self.assertEqual(compact[name].dtype,torch.uint8)
                    torch.testing.assert_close(compact[name].float()/255.,value[:,:1],rtol=0,atol=0)
                elif name=='next_depth_stack':
                    torch.testing.assert_close(compact[name].float()/255.,value,rtol=0,atol=0)
                else:torch.testing.assert_close(compact[name],value,rtol=0,atol=0)

    def test_only_needed_graph_inputs_preserve_loss_all_gradients_and_statistics(self):
        source,batch=fixture(recon_images=5)
        target,_=fixture(recon_images=5)
        target._model_training_modules.load_state_dict(source._model_training_modules.state_dict())
        raw=dict(batch)
        for name in ('depth_stack','next_depth_stack'):
            raw[name]=(batch[name].float()*255).round().to(torch.uint8)
            batch[name]=raw[name].float()/255.
        raw['depth_stack']=raw['depth_stack'][:,:1]
        first,second=LatentWorldModelKernel(source),LatentWorldModelKernel(target)
        a,b=first.prepare(batch),second.prepare(raw)
        self.assertNotIn('depth_stack',b)
        self.assertNotIn('next_depth_stack',b)
        self.assertEqual(len(b['reconstruction_stacks']),5)
        for name in a:torch.testing.assert_close(a[name],b[name],rtol=0,atol=0)
        expected,expected_metrics=first(a)
        actual,actual_metrics=second(b)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        torch.testing.assert_close(actual_metrics,expected_metrics,rtol=0,atol=0)
        expected.backward();actual.backward()
        for left,right in zip(first.parameters(),second.parameters()):
            if left.grad is None:self.assertIsNone(right.grad)
            else:torch.testing.assert_close(left.grad,right.grad,rtol=0,atol=0)
        for name in ('z_mean','z_std','dz_std'):
            torch.testing.assert_close(getattr(source.world_model,name),getattr(target.world_model,name),rtol=0,atol=0)

    def test_whole_macro_matches_sequential_updates_with_fresh_encoder_targets(self):
        source,batch=fixture(recon_images=5)
        target,_=fixture(recon_images=5)
        for runner in (source,target):
            runner.cfg.training_schedule='supermimic_v1'
            runner.cfg.wm_updates=3
        target._model_training_modules.load_state_dict(source._model_training_modules.state_dict())
        sequential=WMExecution(source,mode='eager')
        macro=WMMacroExecution(target,cuda_graphs=False,compile_loss=False)
        seen=[]
        prepare=macro.prepare_function
        def record(values):
            seen.append(next(target.actor.vision.parameters()).detach().clone())
            return prepare(values)
        macro.prepare_function=record
        for _ in range(4):
            records=[]
            for _ in range(3):
                metrics,norm,accepted=sequential.step(batch)
                records.append(torch.cat((metrics,norm.reshape(1),accepted.float().reshape(1))))
            actual=macro.step_macro([batch]*3)
            torch.testing.assert_close(actual,torch.stack(records),rtol=2e-6,atol=2e-8)
            for name,value in source._model_training_modules.state_dict().items():
                torch.testing.assert_close(target._model_training_modules.state_dict()[name],value,rtol=2e-6,atol=2e-8)
        self.assertEqual(len(seen),12)
        self.assertTrue(all(not torch.equal(left,right) for left,right in zip(seen,seen[1:])))
        for value in target.opt_model.state.values():self.assertEqual(float(value['step']),12.)
        self.assertEqual(macro._calls,12)
        self.assertEqual(macro.macro_calls,4)

    def test_macro_rejects_missing_update_before_mutation(self):
        runner,batch=fixture()
        runner.cfg.wm_updates=2
        execution=WMMacroExecution(runner,cuda_graphs=False,compile_loss=False)
        original=copy.deepcopy(runner._model_training_modules.state_dict())
        with self.assertRaisesRegex(ValueError,'configured number'):
            execution.step_macro([batch])
        for key,value in original.items():
            torch.testing.assert_close(runner._model_training_modules.state_dict()[key],value,rtol=0,atol=0)


if __name__=='__main__':unittest.main()
