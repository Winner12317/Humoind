"""Geometry safety proxy must not reward below-ground model predictions."""
import unittest
import torch
from src.wmac_perceptive.safety import foot_volume_edge_cost


class EdgeOracle:
    def height_at(self,xy):return xy[...,0]*0.
    def edge_distance_at(self,xy):return xy[...,0]*0.


class FootSafetyTest(unittest.TestCase):
    def test_deep_penetration_does_not_evade_near_edge_cost(self):
        p=torch.tensor([[[0.,0.,.04],[0.,.2,.04]]])
        q=torch.zeros(1,2,4);q[...,0]=1.
        v=torch.zeros_like(p);v[...,0]=1.
        w=torch.zeros_like(p)
        standing=foot_volume_edge_cost((p,q,v,w),(0,1),EdgeOracle())
        deep=p.clone();deep[...,2]-=.5
        penetrating=foot_volume_edge_cost((deep,q,v,w),(0,1),EdgeOracle())
        airborne=p.clone();airborne[...,2]+=1.
        flying=foot_volume_edge_cost((airborne,q,v,w),(0,1),EdgeOracle())
        self.assertGreaterEqual(float(penetrating),float(standing))
        self.assertLess(float(flying),.001)


if __name__=='__main__':unittest.main()
