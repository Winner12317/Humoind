"""Batched metrics keep masks, NaN/empty behavior and exact finite counts."""
from __future__ import annotations

import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from src.wmac.routing import routing_metrics
from src.wmac_perceptive.diagnostics import read_metrics,routing_tensors
from src.wmac_perceptive.runner import PerceptiveRunner,PHYSICAL_METRICS
from test_hiking_training_v2 import v2_reference_fixture


class BatchedDiagnosticsTest(unittest.TestCase):
    def test_validation_matches_scalar_oracle_with_padding_missing_regions_and_nonfinite(self):
        torch.manual_seed(1711)
        runner=object.__new__(PerceptiveRunner)
        runner.cfg=SimpleNamespace(hiking_training_version=2,visual_mode='instinct_latent_v1',
            motion_scope='forward_walk_turn_v1')
        shape=(13,32)
        names=(*PHYSICAL_METRICS,'model_error','contact_error','actor_mask_error','actor_rmse',
            'full_rmse','outside_rmse','latent_error','latent_persistence_error')
        values={name:torch.rand(shape,dtype=torch.float64) for name in names}
        values.update({name:torch.randint(0,3,shape) for name in
            ('actor_valid_pixels','full_valid_pixels','outside_valid_pixels')})
        values['finite']=torch.ones(shape,dtype=torch.float64)
        values['finite'][2,19]=float('nan')
        mask=torch.rand(shape)>.3
        mask[:,7]=False
        recovery=torch.arange(shape[0])%3==0
        values['full_rmse'].fill_(float('nan'))
        values['full_valid_pixels'].zero_()
        values['outside_valid_pixels'].zero_()
        values['actor_rmse'][~mask]=float('nan')
        expected=runner._legacy_validation_diagnostics(values,mask,recovery)
        original_cpu=torch.Tensor.cpu
        reads=[]
        def read(value,*args,**kwargs):
            reads.append(1)
            return original_cpu(value,*args,**kwargs)
        with patch.object(torch.Tensor,'cpu',read):actual=runner._validation_diagnostics(values,mask,recovery)
        self.assertEqual(len(reads),1)
        self.assertEqual(list(actual),list(expected))
        for key,value in expected.items():
            if math.isnan(value):self.assertTrue(math.isnan(actual[key]),key)
            elif math.isinf(value):self.assertEqual(actual[key],value,key)
            else:self.assertAlmostEqual(actual[key],value,places=12,msg=key)

    def test_routing_uses_same_values_without_boolean_index_host_decisions(self):
        runner,batch,_=v2_reference_fixture()
        for recovery in (torch.zeros(3,dtype=torch.bool),torch.ones(3,dtype=torch.bool),
                         torch.tensor([False,True,False])):
            expected=routing_metrics(runner.actor,batch['actor_history'],recovery)
            tensors,route=routing_tensors(runner.actor,batch['actor_history'],recovery)
            self.assertIsNotNone(route)
            actual=read_metrics(tensors)
            self.assertEqual(actual.keys(),expected.keys())
            for name,value in expected.items():self.assertAlmostEqual(actual[name],value,places=12,msg=name)


if __name__=='__main__':unittest.main()
