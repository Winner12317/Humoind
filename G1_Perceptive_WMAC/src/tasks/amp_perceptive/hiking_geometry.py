"""Exact MuJoCo terrain ray queries and Hiking mesh-edge virtual cylinders.

Greedy concatenation is adapted from project-instinct/InstinctLab
 de6b92f01eb155d51f30e8ed1ea7aecd2eb1f057, edge_cylinder.py (CC-BY-NC-4.0).
The 70-degree mesh adjacency threshold, 30-degree continuation, original
0.05m line-fit tolerance and min_points=2 are retained.

Two edge sources exist. Training version 4 (``hiking_source_mesh_edges``)
detects edges on Hiking's own terrain mesh: IsaacLab's height-field mesh with
``slope_threshold=1`` (every rise steeper than 45 degrees becomes a vertical
face), walls, rotated boxes and the flat mesh-box ground, concatenated and
merged exactly as InstinctLab's TerrainImporter does before detection. Versions
up to 3 (``sharp_edges`` on ``heightfield_mesh``) detect on the MuJoCo
heightfield triangles, whose one-cell ramps fall below the 70-degree threshold
for rises under about 0.14 m. Collision and ray queries follow the MuJoCo
geometry in every version: a heightfield cannot hold a vertical face.
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


def isaaclab_height_field_mesh(height_field, horizontal_scale=.05, vertical_scale=.005, slope_threshold=1.):
    """IsaacLab f73c331 ``convert_height_field_to_mesh`` (BSD-3-Clause).

    ``height_field`` is the integer array indexed [x, y]. Vertices keep the
    source float32 precision and tile-local coordinates; triangle order and
    the 00-to-11 diagonal are the source's.
    """
    hf = np.asarray(height_field)
    num_rows, num_cols = hf.shape
    y = np.linspace(0, (num_cols - 1) * horizontal_scale, num_cols)
    x = np.linspace(0, (num_rows - 1) * horizontal_scale, num_rows)
    yy, xx = np.meshgrid(y, x)
    if slope_threshold is not None:
        threshold = slope_threshold * horizontal_scale / vertical_scale
        move_x = np.zeros((num_rows, num_cols))
        move_y = np.zeros((num_rows, num_cols))
        move_corners = np.zeros((num_rows, num_cols))
        move_x[: num_rows - 1, :] += hf[1:num_rows, :] - hf[: num_rows - 1, :] > threshold
        move_x[1:num_rows, :] -= hf[: num_rows - 1, :] - hf[1:num_rows, :] > threshold
        move_y[:, : num_cols - 1] += hf[:, 1:num_cols] - hf[:, : num_cols - 1] > threshold
        move_y[:, 1:num_cols] -= hf[:, : num_cols - 1] - hf[:, 1:num_cols] > threshold
        move_corners[: num_rows - 1, : num_cols - 1] += (
            hf[1:num_rows, 1:num_cols] - hf[: num_rows - 1, : num_cols - 1] > threshold)
        move_corners[1:num_rows, 1:num_cols] -= (
            hf[: num_rows - 1, : num_cols - 1] - hf[1:num_rows, 1:num_cols] > threshold)
        xx = xx + (move_x + move_corners * (move_x == 0)) * horizontal_scale
        yy = yy + (move_y + move_corners * (move_y == 0)) * horizontal_scale
    vertices = np.zeros((num_rows * num_cols, 3), dtype=np.float32)
    vertices[:, 0] = xx.flatten()
    vertices[:, 1] = yy.flatten()
    vertices[:, 2] = hf.flatten() * vertical_scale
    ind0 = (np.arange(num_rows - 1)[:, None] * num_cols + np.arange(num_cols - 1)[None]).reshape(-1)
    ind1, ind2, ind3 = ind0 + 1, ind0 + num_cols, ind0 + num_cols + 1
    triangles = np.stack((np.stack((ind0, ind3, ind1), -1), np.stack((ind0, ind2, ind3), -1)), 1).reshape(-1, 3)
    return vertices, triangles


def source_mesh_height_grid(vertices, triangles, shape, spacing, fallback=None):
    """Upper surface of a height-field mesh at the points (i*spacing, j*spacing).

    Returns heights indexed [x, y]. A sample that lies on a vertical riser takes
    the riser's top, so a plateau keeps its full source extent and the drop
    happens within the next ``spacing``. Every triangle is rasterized with its
    edges included; vertical and collapsed faces have no area and are covered by
    their neighbours. ``fallback`` [x, y] supplies samples no triangle covers
    (a tile-border vertex displaced inwards leaves such a strip upstream).
    """
    nx, ny = shape
    corners = np.asarray(vertices, dtype=np.float64)[np.asarray(triangles)]
    xy, z = corners[..., :2] / spacing, corners[..., 2]
    # Source vertices are float32 multiples of the grid spacing; snap them so a
    # sample exactly on a triangle's edge or corner is not lost to rounding.
    snapped = np.rint(xy)
    xy = np.where(np.abs(xy - snapped) < 1e-3, snapped, xy)
    (x0, y0), (x1, y1), (x2, y2) = (xy[:, k].T for k in range(3))
    area = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    keep = np.abs(area) > 1e-9
    xy, z, area = xy[keep], z[keep], area[keep]
    (x0, y0), (x1, y1), (x2, y2) = (xy[:, k].T for k in range(3))
    tolerance = 1e-6
    low = np.ceil(xy.min(1) - tolerance).astype(np.int64)
    span = np.floor(xy.max(1) + tolerance).astype(np.int64) - low + 1
    heights = np.full((nx, ny), -np.inf)
    for dx in range(int(span[:, 0].max())):
        for dy in range(int(span[:, 1].max())):
            ix, iy = low[:, 0] + dx, low[:, 1] + dy
            px, py = ix.astype(np.float64), iy.astype(np.float64)
            w0 = ((y1 - y2) * (px - x2) + (x2 - x1) * (py - y2)) / area
            w1 = ((y2 - y0) * (px - x2) + (x0 - x2) * (py - y2)) / area
            w2 = 1. - w0 - w1
            inside = ((dx < span[:, 0]) & (dy < span[:, 1]) & (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
                      & (w0 >= -tolerance) & (w1 >= -tolerance) & (w2 >= -tolerance))
            value = w0 * z[:, 0] + w1 * z[:, 1] + w2 * z[:, 2]
            np.maximum.at(heights, (ix[inside], iy[inside]), value[inside])
    missing = np.isneginf(heights)
    if missing.any():
        if fallback is None:
            raise ValueError("The mesh does not cover every grid sample")
        heights[missing] = np.asarray(fallback, dtype=np.float64)[missing]
    return heights


def hiking_source_terrain_mesh(tiles, boxes):
    """Hiking's merged terrain mesh, as InstinctLab's TerrainImporter sees it.

    ``tiles`` holds (integer heights [x, y] or None for the flat mesh-box
    ground, tile corner xy, tile size xy); ``boxes`` holds world-frame rows
    (centre xyz, half-size xyz, yaw) for walls and rotated boxes. The outer
    border is omitted: only its top faces survive upstream and they add no
    edge. Each part is built with the source trimesh call, so duplicate
    vertices, zero-thickness faces and shared wall corners merge as upstream.
    """
    parts = []
    for heights, corner, size in tiles:
        if heights is None:
            # random_multi_box_terrain without Perlin ground: a zero-height box.
            mesh = trimesh.creation.box((size[0], size[1], 0.), trimesh.transformations.translation_matrix(
                (.5 * size[0], .5 * size[1], 0.)))
        else:
            vertices, triangles = isaaclab_height_field_mesh(heights)
            mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
        mesh.apply_translation((corner[0], corner[1], 0.))
        parts.append(mesh)
    for x, y, z, hx, hy, hz, yaw in boxes:
        transform = trimesh.transformations.translation_matrix((x, y, z)) @ trimesh.transformations.rotation_matrix(
            yaw, (0, 0, 1))
        parts.append(trimesh.creation.box(extents=(2 * hx, 2 * hy, 2 * hz), transform=transform))
    mesh = trimesh.util.concatenate(parts)
    # TerrainImporter.import_mesh, InstinctLab de6b92f.
    mesh.merge_vertices()
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    return mesh


def _sharp_edge_coordinates(mesh, angle_threshold=70., prune=True):
    """Vertex coordinates [n, 6] of mesh edges whose face angle exceeds the threshold.

    Two faces both tilted at most 35 degrees from +z cannot meet at more than
    70 degrees, so only faces steeper than that (or degenerate, whose zero
    normal trimesh scores as 90 degrees) and the faces touching them need
    adjacency. Every edge of a steep face keeps all its incident faces, so the
    pruned result equals the full-mesh one, including edges trimesh drops for
    having more than two faces.
    """
    if prune:
        steep = ~(mesh.face_normals[:, 2] >= math.cos(math.radians(angle_threshold / 2.)) + 1e-9)
        if not steep.any():
            return np.empty((0, 6))
        touched = np.zeros(len(mesh.vertices), dtype=bool)
        touched[mesh.faces[steep].reshape(-1)] = True
        mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces[touched[mesh.faces].any(1)], process=False)
    selected = mesh.face_adjacency_edges[mesh.face_adjacency_angles > np.deg2rad(angle_threshold)]
    if not len(selected):
        return np.empty((0, 6))
    return np.asarray(mesh.vertices)[selected].reshape(-1, 6)


def _merge_by_component(edge_coords, seed=0):
    """Greedy concatenation, one connected group of sharp edges at a time.

    Chains never leave a connected group, so this equals the source's single
    pass over all edges up to its unseeded random start vertices, without
    rebuilding the list of every remaining vertex for each chain.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    if not len(edge_coords):
        return np.empty((0, 6), dtype=np.float32)
    points, inverse = np.unique(edge_coords.reshape(-1, 3), axis=0, return_inverse=True)
    pairs = inverse.reshape(-1, 2)
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(points),) * 2)
    _, labels = connected_components(graph, directed=False)
    edge_labels = labels[pairs[:, 0]]
    order = np.argsort(edge_labels, kind="stable")
    bounds = np.flatnonzero(np.diff(edge_labels[order])) + 1
    settings = SimpleNamespace(cfg=SimpleNamespace(adjacent_angle_threshold=30., min_points=2), rng=random.Random(seed))
    merged = [merge_sharp_edges(settings, edge_coords[group]).reshape(-1, 6) for group in np.split(order, bounds)]
    return np.concatenate(merged, axis=0).astype(np.float32)


def hiking_source_mesh_edges(tiles, boxes, seed=0):
    """Hiking's edge cylinders [n, 6]: GreedyconcatEdgeCylinder on the source terrain mesh."""
    return _merge_by_component(_sharp_edge_coordinates(hiking_source_terrain_mesh(tiles, boxes)), seed=seed)


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
