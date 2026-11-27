"""Versioned foot-volume costs shared by physics and imagined trajectories.

Full Hiking uses 100 points per foot and actual mesh-edge virtual cylinders;
historical checkpoints retain their 18-point raster-distance proxy.
"""
from __future__ import annotations

import torch

from src.wmac.state import quat_apply


def foot_volume_edge_cost(kinematics,foot_ids,terrain,margin=.04):
    p,q,v,w=kinematics
    axes=(p.new_tensor([-.04,.04,.12]),p.new_tensor([-.025,0.,.025]),p.new_tensor([-.035,-.015]))
    local=torch.stack(torch.meshgrid(*axes,indexing="ij"),-1).reshape(18,3)
    offset=quat_apply(q[...,foot_ids,None,:],local)
    points=p[...,foot_ids,None,:]+offset
    speeds=v[...,foot_ids,None,:]+torch.cross(w[...,foot_ids,None,:].expand_as(offset),offset,dim=-1)
    distance=terrain.edge_distance_at(points[...,:2])
    clearance=points[...,2]-terrain.height_at(points[...,:2])
    # Below-ground predictions must not evade the edge cost by becoming
    # "far from the surface". Airborne points fade out; penetrating points do
    # not. This remains a heightfield proxy, not a physical contact derivative.
    near_surface=torch.sigmoid((.035-clearance)/.01)
    penetration=(margin-distance).clamp_min(0)*near_surface
    return (penetration*(speeds.norm(dim=-1)+1e-3)).sum((-1,-2))


def hiking_foot_volume_points(kinematics,foot_ids,foot_points=None,foot_linear_velocity=None):
    """Original ordinary G1 task grid: 10 x 5 x 2 = 100 points per foot.

    This model has no shoe collision geometry; ShoeConfigMixin's lowered z
    interval belongs to the distinct shoe task and is not mixed into this one.
    """
    p,q,v,w=kinematics
    if foot_points is None:
        axes=(torch.linspace(-.025,.12,10,device=p.device,dtype=p.dtype),
              torch.linspace(-.03,.03,5,device=p.device,dtype=p.dtype),
              torch.linspace(-.04,0.,2,device=p.device,dtype=p.dtype))
        foot_points=torch.stack(torch.meshgrid(*axes,indexing="ij"),-1).reshape(100,3)
    else:foot_points=foot_points.to(p)
    offset=quat_apply(q[...,foot_ids,None,:],foot_points)
    points=p[...,foot_ids,None,:]+offset
    base_velocity=v[...,foot_ids,:] if foot_linear_velocity is None else foot_linear_velocity
    # Hiking's VolumePoints sensor combines PhysX COM linear velocity with
    # cross(angular_velocity, point-link_origin), as its original code does.
    speeds=base_velocity[...,None,:]+torch.cross(w[...,foot_ids,None,:].expand_as(offset),offset,dim=-1)
    return points,speeds


def hiking_foot_edge_cost(kinematics,foot_ids,terrain,foot_points=None,foot_linear_velocity=None):
    """Source penetration reward on actual mesh sharp-edge cylinders.

    The original selects the largest cylinder penetration per point, then
    sums penetration depth times (point speed + 1e-6). No extra height gate,
    near-surface sigmoid or raster distance is applied to the full task.
    """
    if terrain.hiking_edges is None:
        raise RuntimeError("Full Hiking edge reward requires hiking_full_v1 terrain")
    points,speeds=hiking_foot_volume_points(kinematics,foot_ids,foot_points,
                                         foot_linear_velocity=foot_linear_velocity)
    penetration=terrain.hiking_edges.penetration_depth(points)
    return (penetration*(speeds.norm(dim=-1)+1e-6)).sum((-1,-2))


def terrain_foot_edge_reward(env,margin=.04):
    from src.tasks.amp_perceptive.mdp import get_terrain_oracle
    from .state import PerceptiveStateAdapter
    adapter=getattr(env,"_perceptive_reward_adapter",None)
    if adapter is None:
        adapter=PerceptiveStateAdapter(env)
        env._perceptive_reward_adapter=adapter
    return foot_volume_edge_cost(adapter.kinematics(adapter.state()),adapter.foot_ids,
                                 get_terrain_oracle(env),margin=margin)
