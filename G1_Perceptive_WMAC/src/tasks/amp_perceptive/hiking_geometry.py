"""Exact MuJoCo terrain ray queries and Hiking mesh-edge virtual cylinders.

Greedy concatenation is adapted from project-instinct/InstinctLab
 de6b92f01eb155d51f30e8ed1ea7aecd2eb1f057, edge_cylinder.py (CC-BY-NC-4.0).
The 70-degree mesh adjacency threshold, 30-degree continuation, original
0.05m line-fit tolerance and min_points=2 are retained. Unlike a raster edge
proxy, detection uses actual heightfield triangles and rotated box meshes.
MuJoCo cannot reproduce PhysX steep-slope vertex displacement; these queries
therefore follow the physical MuJoCo geometry without shifting invisible edges.
"""
from __future__ import annotations
from types import SimpleNamespace
import math
import random
import numpy as np
import torch
import trimesh


def heightfield_mesh(height,center,half_size):
    ny,nx=height.shape
    x=np.linspace(center[0]-half_size[0],center[0]+half_size[0],nx)
    y=np.linspace(center[1]-half_size[1],center[1]+half_size[1],ny)
    xx,yy=np.meshgrid(x,y)
    vertices=np.stack((xx,yy,height),-1).reshape(-1,3)
    i=(np.arange(ny-1)[:,None]*nx+np.arange(nx-1)[None]).reshape(-1)
    triangles=np.stack((np.stack((i,i+1,i+nx+1),-1),np.stack((i,i+nx+1,i+nx),-1)),1).reshape(-1,3)
    return trimesh.Trimesh(vertices=vertices,faces=triangles,process=False)


def sharp_edges(mesh,seed=0):
    angles=mesh.face_adjacency_angles
    selected=mesh.face_adjacency_edges[angles>np.deg2rad(70.)]
    if not len(selected):return np.empty((0,6),dtype=np.float32)
    points=mesh.vertices[selected].reshape(-1,6)
    settings=SimpleNamespace(cfg=SimpleNamespace(adjacent_angle_threshold=30.,min_points=2),rng=random.Random(seed))
    return merge_sharp_edges(settings,points).reshape(-1,6)


class SpatialCandidates2D:
    """Conservative fixed-shape acceleration; selected values stay differentiable.

    AABB insertion includes every intersected cell. Unlike nearest-edge lookup,
    this retains all candidate cylinders/boxes, so the max penetration and
    topmost vertical ray hit are exact for the supplied physical geometry.
    """
    def __init__(self,bounds,cell_size,device):
        bounds=np.asarray(bounds,dtype=np.float64).reshape(-1,2,2)
        if len(bounds):
            lower=np.floor(bounds[:,0].min(0)/cell_size)*cell_size-cell_size
            upper=np.ceil(bounds[:,1].max(0)/cell_size)*cell_size+cell_size
        else:lower=np.zeros(2);upper=np.ones(2)*cell_size
        shape=np.maximum(np.ceil((upper-lower)/cell_size).astype(int),1)
        cells={}
        for index,b in enumerate(bounds):
            first=np.floor((b[0]-lower)/cell_size).astype(int)
            last=np.floor((b[1]-lower)/cell_size).astype(int)
            for x in range(first[0],last[0]+1):
                for y in range(first[1],last[1]+1):
                    cells.setdefault(x*int(shape[1])+y,[]).append(index)
        count=max((len(ids) for ids in cells.values()),default=1)
        table=np.full((int(np.prod(shape)),count),-1,dtype=np.int32)
        for key,ids in cells.items():table[key,:len(ids)]=ids
        self.table=torch.as_tensor(table,device=device)
        self.lower=torch.as_tensor(lower,dtype=torch.float32,device=device)
        self.cell_size=float(cell_size)
        self.shape=(int(shape[0]),int(shape[1]))

    def query(self,xy):
        ij=torch.floor((xy-self.lower)/self.cell_size).long()
        inside=(ij[...,0]>=0)&(ij[...,0]<self.shape[0])&(ij[...,1]>=0)&(ij[...,1]<self.shape[1])
        idx=ij[...,0].clamp(0,self.shape[0]-1)*self.shape[1]+ij[...,1].clamp(0,self.shape[1]-1)
        ids=self.table[idx].long()
        return ids.clamp_min(0),(ids>=0)&inside[...,None]


class HikingEdgeCylinders:
    def __init__(self,edges,device="cpu",radius=.05):
        edges=np.asarray(edges,dtype=np.float32).reshape(-1,2,3)
        self.edge_count=len(edges);self.radius=float(radius)
        bounds=np.stack((edges.min(1)[:,:2]-radius,edges.max(1)[:,:2]+radius),1)
        self.grid=SpatialCandidates2D(bounds,.1,device)
        if not len(edges):edges=np.zeros((1,2,3),dtype=np.float32);edges[0,1,0]=1.
        self.edges=torch.as_tensor(edges,device=device)

    def penetration_depth(self,points):
        ids,valid=self.grid.query(points[...,:2])
        segment=self.edges[ids]
        start,end=segment[...,0,:],segment[...,1,:]
        delta=end-start
        length=delta.norm(dim=-1).clamp_min(1e-8)
        direction=delta/length[...,None]
        offset=points[...,None,:]-start
        distance_along=(offset*direction).sum(-1)
        radial=offset-distance_along[...,None]*direction
        distance=radial.norm(dim=-1)
        # Flat cylinder caps: source explicitly excludes projection beyond the
        # segment ends (these are cylinders, not rounded capsules).
        valid=valid&(distance_along>=0)&(distance_along<=length)
        depth=torch.where(valid,(self.radius-distance).clamp_min(0.),torch.zeros_like(distance))
        return depth.amax(-1)


class HikingBoxRays:
    """Vertical rays against yaw-rotated collision boxes, including tile walls."""
    def __init__(self,boxes,device="cpu"):
        # rows: center xyz, half-size xyz, yaw.
        # Preserve compiled model precision for inclusion tests at tile-wall
        # faces. Rounding a 4.025m centre to float32 can move a .025m half-face
        # across x=4 and turn a boundary ray's 5m wall hit into a floor hit.
        # Height outputs retain the caller's float32 dtype; these comparisons
        # have no continuous derivative and do not enlarge the BPTT graph.
        boxes=np.asarray(boxes,dtype=np.float64).reshape(-1,7)
        self.box_count=len(boxes)
        extent=np.abs(np.cos(boxes[:,6,None]))*boxes[:,3:5]+np.abs(np.sin(boxes[:,6,None]))*boxes[:,[4,3]]
        bounds=np.stack((boxes[:,:2]-extent,boxes[:,:2]+extent),1)
        self.grid=SpatialCandidates2D(bounds,.5,device)
        if not len(boxes):boxes=np.zeros((1,7),dtype=np.float64)
        self.boxes=torch.as_tensor(boxes,device=device)

    def height_at(self,xy,ray_z=None):
        ids,valid=self.grid.query(xy)
        boxes=self.boxes[ids]
        delta=xy.double()[...,None,:]-boxes[...,:2]
        c,s=boxes[...,6].cos(),boxes[...,6].sin()
        x=c*delta[...,0]+s*delta[...,1]
        y=-s*delta[...,0]+c*delta[...,1]
        hit=valid&(x.abs()<=boxes[...,3])&(y.abs()<=boxes[...,4])
        top=boxes[...,2]+boxes[...,5]
        if ray_z is not None:
            bottom=boxes[...,2]-boxes[...,5]
            top=torch.where(ray_z[...,None]>=top,top,bottom)
            hit=hit&(ray_z[...,None]>=bottom)
        return torch.where(hit,top,torch.full_like(top,-torch.inf)).amax(-1).to(xy.dtype)

def merge_sharp_edges(self, edge_coords: np.ndarray) -> np.ndarray:
    line_pts = edge_coords.reshape(-1, 3)
    V, inv_idx = np.unique(line_pts, axis=0, return_inverse=True)
    E_pairs = inv_idx.reshape(-1, 2)
    adj_list = {i: set() for i in range(V.shape[0])}
    for u, v in E_pairs:
        if u != v:
            adj_list[u].add(v)
            adj_list[v].add(u)
    num_edges_V = np.array([len(adj_list[i]) for i in range(V.shape[0])], dtype=int)
    available_edges = set(np.where(num_edges_V > 0)[0])
    cos_threshold = np.cos(np.deg2rad(self.cfg.adjacent_angle_threshold))
    processed_edge_coords = []

    def compute_max_distance_to_line_vec(V, v_set):
        A = V[v_set[0]]
        B = V[v_set[-1]]
        AB = B - A
        norm_AB = np.linalg.norm(AB)
        if norm_AB == 0:
            return (v_set[0], 0.0)
        pts = V[v_set]
        dists = np.linalg.norm(np.cross(pts - A, pts - B), axis=1) / norm_AB
        max_idx = np.argmax(dists)
        return (v_set[max_idx], dists[max_idx])
    while available_edges:
        selected_vertex = self.rng.choice(sorted(available_edges))
        v_set = [selected_vertex]
        if adj_list[selected_vertex]:
            neighbor = next(iter(adj_list[selected_vertex]))
            v_set.append(neighbor)
            adj_list[selected_vertex].remove(neighbor)
            adj_list[neighbor].remove(selected_vertex)
            for vid in [selected_vertex, neighbor]:
                num_edges_V[vid] -= 1
                if num_edges_V[vid] == 0:
                    available_edges.discard(vid)
        while True:
            find_neighbor = False
            start, end = (v_set[0], v_set[-1])
            neighbors = list(adj_list[start])
            if neighbors:
                dirs = V[start] - V[neighbors]
                dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
                start_dir = V[v_set[1]] - V[start]
                start_dir /= np.linalg.norm(start_dir)
                dots = dirs @ start_dir
                idx = np.where(dots > cos_threshold)[0]
                if idx.size > 0:
                    n = neighbors[idx[0]]
                    v_set.insert(0, n)
                    adj_list[start].remove(n)
                    adj_list[n].remove(start)
                    for vid in [start, n]:
                        num_edges_V[vid] -= 1
                        if num_edges_V[vid] == 0:
                            available_edges.discard(vid)
                    find_neighbor = True
            neighbors = list(adj_list[end])
            if neighbors:
                dirs = V[neighbors] - V[end]
                dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
                end_dir = V[end] - V[v_set[-2]]
                end_dir /= np.linalg.norm(end_dir)
                dots = dirs @ end_dir
                idx = np.where(dots > cos_threshold)[0]
                if idx.size > 0:
                    n = neighbors[idx[0]]
                    v_set.append(n)
                    adj_list[end].remove(n)
                    adj_list[n].remove(end)
                    for vid in [end, n]:
                        num_edges_V[vid] -= 1
                        if num_edges_V[vid] == 0:
                            available_edges.discard(vid)
                    find_neighbor = True
            if not find_neighbor:
                break
        while len(v_set) >= self.cfg.min_points:
            for i in range(len(v_set) - 1):
                max_vi, max_dist = compute_max_distance_to_line_vec(V, v_set[i:])
                if max_dist < 0.05:
                    break
            if len(v_set) - i >= self.cfg.min_points:
                processed_edge_coords.append(np.concatenate([V[v_set[i]], V[v_set[-1]]]))
            v_set = v_set[:i + 1]
    return np.array(processed_edge_coords, dtype=np.float32)
