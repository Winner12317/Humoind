"""Raw WM batches pack directly into stable graph storage without a recopy."""
from __future__ import annotations

import copy
import os
import unittest
from unittest.mock import patch

import torch
from torch import nn

from src.wmac_perceptive.macro_execution import CapturedMacroUpdate
from src.wmac_perceptive.optimizers import build_reference_optimizer


def batches(device="cpu"):
    return [dict(x=torch.arange(18,device=device,dtype=torch.float64).reshape(3,6)[:,::2]+i,
                 target=torch.full((3,2),i/10,device=device,dtype=torch.float64),
                 codes=torch.arange(12,device=device,dtype=torch.uint8).reshape(3,4)+i,
                 mask=torch.tensor([True,False,True],device=device)) for i in range(3)]


def make(module, *, device="cpu", capture=False):
    optimizer=build_reference_optimizer((module,),1e-3,device=device)
    execution=None
    def update(batch):
        losses=[]
        for index in range(len(batch["x"])):
            execution.zero_grad(optimizer)
            loss=(module(batch["x"][index])-batch["target"][index]).square().mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(module.parameters(),1.,foreach=True)
            optimizer.step()
            losses.append(loss.detach())
        return dict(loss=torch.stack(losses),codes=batch["codes"],mask=batch["mask"])
    execution=CapturedMacroUpdate(update,(module,),(optimizer,),cuda_graphs=capture)
    return execution,optimizer


class DirectStagingTest(unittest.TestCase):
    def test_stack_out_reuses_every_address_and_preserves_raw_codes_masks_and_order(self):
        model=nn.Linear(3,2).double()
        execution,_=make(model)
        values=batches()
        source=copy.deepcopy(values)
        packed=execution._stack_inputs(values)
        addresses={name:value.data_ptr() for name,value in packed.items()}
        for name in packed:
            torch.testing.assert_close(packed[name],torch.stack([item[name] for item in values]),rtol=0,atol=0)
        with patch("torch.stack",wraps=torch.stack) as stack:
            execution._stack_inputs(list(reversed(values)))
        self.assertEqual(stack.call_count,len(packed))
        for call in stack.call_args_list:
            self.assertIn(call.kwargs["out"].data_ptr(),addresses.values())
        self.assertEqual(addresses,{name:value.data_ptr() for name,value in packed.items()})
        for name in packed:
            torch.testing.assert_close(packed[name],torch.stack([item[name] for item in reversed(values)]),rtol=0,atol=0)
        for before,after in zip(source,values):
            for name in before:torch.testing.assert_close(before[name],after[name],rtol=0,atol=0)

    def test_all_schema_errors_are_checked_before_existing_buffers_are_overwritten(self):
        execution,_=make(nn.Linear(3,2).double())
        values=batches()
        packed=execution._stack_inputs(values)
        original={name:value.clone() for name,value in packed.items()}
        for change in (lambda x:x[1].update(mask=torch.ones(4,dtype=torch.bool)),
                       lambda x:x[1].update(codes=x[1]["codes"].long()),
                       lambda x:x[1].pop("target"),
                       lambda x:x[1].update(x=x[1]["x"].requires_grad_())):
            altered=copy.deepcopy(values)
            altered[0]["x"].add_(99)
            change(altered)
            with self.assertRaises(ValueError):execution._stack_inputs(altered)
            for name in original:torch.testing.assert_close(packed[name],original[name],rtol=0,atol=0)
        with self.assertRaises(ValueError):execution._stack_inputs([])

    def test_sequential_online_updates_and_optimizer_state_match_previous_stack_path(self):
        torch.manual_seed(7701)
        module=nn.Linear(3,2).double()
        target=copy.deepcopy(module)
        direct,opt=make(module)
        reference,ref_opt=make(target)
        original=batches()
        snapshots=[]
        for _ in range(6):
            values=[{name:value.clone() for name,value in item.items()} for item in original]
            expected=reference.step({name:torch.stack([item[name] for item in values]) for name in values[0]})
            with patch.object(direct,"_copy_inputs",side_effect=AssertionError("redundant packed-input copy")):
                actual=direct.step_stacked(values)
            snapshots.append(actual["codes"])
            for name in expected:torch.testing.assert_close(actual[name],expected[name],rtol=0,atol=0)
            for left,right in zip(module.parameters(),target.parameters()):
                torch.testing.assert_close(left,right,rtol=0,atol=0)
                for key in opt.state[left]:torch.testing.assert_close(opt.state[left][key],ref_opt.state[right][key],rtol=0,atol=0)
        self.assertEqual(direct._calls,6)
        for value in opt.state.values():self.assertEqual(float(value["step"]),18.)
        for item in snapshots:torch.testing.assert_close(item,snapshots[0],rtol=0,atol=0)
        direct.invalidate()
        self.assertIsNone(direct._static_inputs)
        self.assertEqual(direct._calls,0)


@unittest.skipUnless(os.environ.get("PERCEPTIVE_TEST_DIRECT_STAGING_CUDA")=="1" and torch.cuda.is_available(),
                     "set PERCEPTIVE_TEST_DIRECT_STAGING_CUDA=1 for bounded CUDA staging/capture")
class DirectStagingCudaTest(unittest.TestCase):
    def test_side_stream_warmup_capture_replay_and_reused_storage_match_eager(self):
        torch.manual_seed(7701)
        module=nn.Linear(3,2).double().cuda()
        reference=copy.deepcopy(module)
        direct,opt=make(module,device="cuda",capture=True)
        eager,ref_opt=make(reference,device="cuda")
        addresses=None
        for iteration in range(6):
            values=batches("cuda")
            for value in values:value["target"].add_(iteration/20)
            actual=direct.step_stacked(values)
            expected=eager.step({name:torch.stack([item[name] for item in values]) for name in values[0]})
            torch.cuda.synchronize()
            now={name:value.data_ptr() for name,value in direct._static_inputs.items()}
            if addresses is None:addresses=now
            self.assertEqual(addresses,now)
            self.assertEqual(direct._graph is not None,iteration>=3)
            for name in expected:torch.testing.assert_close(actual[name],expected[name],rtol=1e-10,atol=1e-12)
            for left,right in zip(module.parameters(),reference.parameters()):
                torch.testing.assert_close(left,right,rtol=1e-10,atol=1e-12)
                for key in opt.state[left]:torch.testing.assert_close(opt.state[left][key],ref_opt.state[right][key],rtol=1e-10,atol=1e-12)
        for value in opt.state.values():self.assertEqual(float(value["step"]),18.)


if __name__=="__main__":unittest.main()
