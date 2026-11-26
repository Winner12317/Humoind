"""Versioned MuJoCo ports of Hiking's procedural terrain families.

The full protocol copies source arrays, ten-family proportions, complete
difficulty, family borders/platforms, origins and target sampling. Its physical
geometry uses 5cm MuJoCo heightfields plus actual yaw-rotated boxes; full WM
queries follow those triangles/boxes. PhysX steep-slope vertex displacement
cannot be represented by a MuJoCo heightfield and remains documented.

Historical protocols retain their independent 10cm collision arrays, added
spawn pad, rasterized boxes and optional conservative seven-family walking
profile. They are separate checkpoint semantics, not Hiking author defaults.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal
import uuid

import mujoco
import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter, distance_transform_edt
import torch

from mjlab.terrains import FlatPatchSamplingCfg, SubTerrainCfg, TerrainGeneratorCfg
from mjlab.terrains.terrain_generator import TerrainGeometry, TerrainOutput

HIKING_COMMIT = "de6b92f01eb155d51f30e8ed1ea7aecd2eb1f057"
HIKING_PROPORTIONS = {
    "perlin_rough": .05, "perlin_rough_stand": .05, "square_gaps": .10,
    "pyramid_stairs": .15, "pyramid_stairs_high": .10,
    "pyramid_stairs_inv": .15, "pyramid_stairs_inv_high": .10,
    "boxes": .10, "mesh_boxes": .10, "hf_pyramid_slope_inv": .10,
}
WALKING_EXCLUDED_FAMILIES = frozenset(("square_gaps","pyramid_stairs_high","pyramid_stairs_inv_high"))
_WALKING_WEIGHT_SUM = sum(weight for name,weight in HIKING_PROPORTIONS.items() if name not in WALKING_EXCLUDED_FAMILIES)
WALKING_TERRAIN_PROPORTIONS = {
    name:weight/_WALKING_WEIGHT_SUM for name,weight in HIKING_PROPORTIONS.items()
    if name not in WALKING_EXCLUDED_FAMILIES
}


def _perlin(shape: tuple[int, int], rng: np.random.Generator,
            amplitude: float, frequency: int = 20) -> np.ndarray:
    """Gradient-noise spectrum/centering follows Hiking's two-octave setup.

    RNG is explicit and instance-local instead of upstream global np.random.
    The author oversamples a 4x grid before cropping; reproduce that spectrum
    without allocating the unused region, using the same 160/320 lattices.
    """
    nx, ny = shape
    result = np.zeros(shape, dtype=np.float64)
    for octave, gain in ((0, 1.), (1, .25)):
        # 8m * frequency=160 lattice cells across the author's 640 samples.
        cells = int(frequency * 8 * (2 ** octave))
        gx, gy = np.meshgrid(np.arange(nx) * cells / (4 * nx),
                             np.arange(ny) * cells / (4 * ny), indexing="ij")
        ix, iy = gx.astype(int), gy.astype(int)
        tx, ty = gx - ix, gy - iy
        angles = rng.uniform(0, 2 * np.pi, (cells + 1, cells + 1))
        cx, sy = np.cos(angles), np.sin(angles)
        values = []
        for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
            values.append(cx[ix + dx, iy + dy] * (tx - dx) +
                          sy[ix + dx, iy + dy] * (ty - dy))
        fx = tx ** 3 * (10 + tx * (-15 + 6 * tx))
        fy = ty ** 3 * (10 + ty * (-15 + 6 * ty))
        noise = (1-fy) * ((1-fx)*values[0] + fx*values[1]) + fy*((1-fx)*values[2] + fx*values[3])
        result += gain * amplitude * math.sqrt(2) * .5 * noise
    return result - result.mean()


def hiking_height_array(family: str, difficulty: float, rng: np.random.Generator,
                        size=(8., 8.), horizontal_scale=.05, vertical_scale=.005,
                        recovery_pad_width=1.5) -> np.ndarray:
    """Return physical heights in (y,x) order, including both tile endpoints."""
    if family not in (*HIKING_PROPORTIONS, "recovery_flat"):
        raise ValueError(f"Unknown Hiking terrain family {family!r}")
    if not 0 <= difficulty <= 1:
        raise ValueError("difficulty must lie in [0,1]")
    nx, ny = (int(round(v / horizontal_scale)) for v in size)
    height = np.zeros((nx, ny), dtype=np.float64)  # author x,y convention
    if family.startswith("perlin"):
        height += _perlin((nx, ny), rng, difficulty * .1)
    elif family.startswith("pyramid_stairs"):
        high = "high" in family
        step = round((.05 + difficulty * ((.45 if high else .23) - .05)) / vertical_scale) * vertical_scale
        if "inv" in family:
            step = -step
        run = int(round((1.5 if high else .3) / horizontal_scale))
        platform = int(round((4. if high else 2.5) / horizontal_scale))
        start, stop = run, min(nx, ny) - run
        level = step
        while stop - start > 0:
            height[start:stop, start:stop] = level
            if stop - start <= platform:
                break
            start += run; stop -= run; level += step
        height += _perlin((nx, ny), rng, .05)
    elif family == "square_gaps":
        width = int(round((.1 + .6*difficulty) / horizontal_scale))
        platform = int(round(2.5 / horizontal_scale))
        x0, x1 = (nx-platform)//2, (nx+platform)//2
        y0, y1 = (ny-platform)//2, (ny+platform)//2
        depth = -round(rng.uniform(.4,.6)/vertical_scale)*vertical_scale
        height[x0-width:x1+width, y0-width:y1+width] = depth
        height[x0:x1,y0:y1] = 0.
    elif family == "boxes":
        value = .05 + .4*difficulty
        for _ in range(20):
            wx, wy = (int(rng.choice(np.arange(16,30,4))) for _ in range(2))
            x, y = int(rng.integers(1,nx-wx)), int(rng.integers(1,ny-wy))
            height[x:x+wx,y:y+wy] = value
        height += _perlin((nx, ny), rng, .05)
    elif family == "mesh_boxes":
        xx, yy = np.meshgrid(np.arange(nx)*horizontal_scale,
                             np.arange(ny)*horizontal_scale, indexing="ij")
        for _ in range(int(.3 * size[0] * size[1] / (.4*.4))):
            wx, wy = rng.uniform(.3,.5,2)
            x, y = rng.uniform(wx/2,size[0]-wx/2), rng.uniform(wy/2,size[1]-wy/2)
            if abs(x-size[0]/2) < .75+wx/2 and abs(y-size[1]/2) < .75+wy/2:
                continue
            angle = rng.uniform(0,2*np.pi)
            dx, dy = xx-x, yy-y
            bx, by = np.cos(angle)*dx+np.sin(angle)*dy, -np.sin(angle)*dx+np.cos(angle)*dy
            mask = (abs(bx)<=wx/2)&(abs(by)<=wy/2)
            height[mask] = np.maximum(height[mask], .1+.3*difficulty+rng.uniform(-.05,.05))
    elif family == "hf_pyramid_slope_inv":
        xx = (nx//2-abs(nx//2-np.arange(nx)))/(nx//2)
        yy = (ny//2-abs(ny//2-np.arange(ny)))/(ny//2)
        height = -.7*difficulty*size[0]/2 * xx[:,None] * yy[None,:]
        at_platform = -(.7*difficulty*size[0]/2)*((size[0]/2-.75)/(size[0]/2))**2
        height = np.maximum(height, at_platform)
    # Motion resets need a clear starting patch rather than assuming every
    # narrow stair tread is legal. The optional legacy joint task reuses the
    # same patch for fallen G1 poses; the current walking scope does not.
    half = int(round(recovery_pad_width/(2*horizontal_scale)))
    center_height = float(np.median(height[nx//2-2:nx//2+3, ny//2-2:ny//2+3]))
    height[nx//2-half:nx//2+half+1,ny//2-half:ny//2+half+1] = center_height
    height = np.rint(height/vertical_scale)*vertical_scale
    return np.pad(height.T, ((0,1),(0,1)), mode="edge").astype(np.float32)


@dataclass(kw_only=True)
class HikingTerrainCfg(SubTerrainCfg):
    family: str = "perlin_rough"
    horizontal_scale: float = .05
    collision_horizontal_scale: float = .1
    vertical_scale: float = .005
    wall_probability: float = .3
    wall_height: float = 5.
    wall_thickness: float = .05
    recovery_pad_width: float = 1.5
    maximum_difficulty: float = 1.
    task_protocol: str = "legacy_walk_turn_v1"

    def function(self, difficulty, spec, rng):
        if self.task_protocol=="hiking_full_v1":
            return _hiking_full_terrain(self,difficulty,spec,rng)
        if not math.isfinite(difficulty) or not 0<=difficulty<=1:
            raise ValueError("difficulty must lie in [0,1]")
        if not math.isfinite(self.maximum_difficulty) or not 0<self.maximum_difficulty<=1:
            raise ValueError("maximum_difficulty must lie in (0,1]")
        stride = _collision_stride(self.size, self.horizontal_scale,
                                   self.collision_horizontal_scale)
        source = hiking_height_array(self.family, difficulty*self.maximum_difficulty, rng, self.size,
                                    self.horizontal_scale, self.vertical_scale,
                                    self.recovery_pad_width)
        height = source[::stride, ::stride].copy()
        # Downsampling alone could shrink a 1.5m pad to 1.4m. Round its physical
        # extent outwards on the collision grid and certify targets on this
        # same array, rather than using the finer source as privileged truth.
        pad_half = math.ceil(self.recovery_pad_width /
                             (2 * self.collision_horizontal_scale)) * self.collision_horizontal_scale
        xx = np.arange(height.shape[1]) * self.collision_horizontal_scale
        yy = np.arange(height.shape[0]) * self.collision_horizontal_scale
        pad = ((abs(yy[:, None] - self.size[1]/2) <= pad_half + 1e-8) &
               (abs(xx[None, :] - self.size[0]/2) <= pad_half + 1e-8))
        center_height = source[source.shape[0]//2, source.shape[1]//2]
        height[pad] = center_height
        low, high = float(height.min()), float(height.max())
        span = max(high-low, .005)
        name = "hiking_" + uuid.uuid4().hex
        field = spec.add_hfield(name=name, size=[self.size[0]/2,self.size[1]/2,span,1.],
                               nrow=height.shape[0],ncol=height.shape[1],
                               userdata=((height-low)/span).flatten().tolist())
        body=spec.body("terrain")
        geom=body.add_geom(type=mujoco.mjtGeom.mjGEOM_HFIELD,hfieldname=name,
                           pos=[self.size[0]/2,self.size[1]/2,low],group=0)
        geometries=[TerrainGeometry(geom=geom,hfield=field,color=(.45,.5,.48,1.))]
        # Walls are real collision boxes, not mistaken for traversable ground.
        if self.family != "recovery_flat":
            for axis, direction in ((0,-1),(0,1),(1,-1),(1,1)):
                if rng.random() >= self.wall_probability: continue
                size=[self.size[0]/2,self.size[1]/2,self.wall_height/2]
                size[axis]=self.wall_thickness/2
                position=[self.size[0]/2,self.size[1]/2,self.wall_height/2]
                position[axis]=(0. if direction<0 else self.size[axis])+direction*self.wall_thickness/2
                wall=body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX,size=size,pos=position,group=0)
                geometries.append(TerrainGeometry(geom=wall,color=(.3,.3,.3,1.)))
        origin=np.array([self.size[0]/2,self.size[1]/2,float(center_height)])
        patches={}
        for key, cfg in (self.flat_patch_sampling or {}).items():
            if key=="spawn":
                patches[key]=np.repeat(origin[None],cfg.num_patches,axis=0)
                continue
            radius=int(math.ceil(cfg.patch_radius/self.collision_horizontal_scale))
            kernel=2*radius+1
            safe=(maximum_filter(height,size=kernel)-minimum_filter(height,size=kernel))<=cfg.max_height_diff
            safe[:radius+1]=False;safe[-radius-1:]=False;safe[:,:radius+1]=False;safe[:,-radius-1:]=False
            iy,ix=np.where(safe)
            if not len(ix): patches[key]=np.repeat(origin[None],cfg.num_patches,axis=0)
            else:
                chosen=rng.choice(len(ix),cfg.num_patches,replace=len(ix)<cfg.num_patches)
                patches[key]=np.stack((ix[chosen]*self.collision_horizontal_scale,iy[chosen]*self.collision_horizontal_scale,height[iy[chosen],ix[chosen]]),-1)
        return TerrainOutput(origin=origin,geometries=geometries,flat_patches=patches)


def _collision_stride(size, horizontal_scale, collision_horizontal_scale):
    """Only exact decimation is allowed; terrain extents cannot change."""
    if not (math.isfinite(horizontal_scale) and horizontal_scale > 0 and
            math.isfinite(collision_horizontal_scale) and collision_horizontal_scale > 0):
        raise ValueError("Terrain horizontal scales must be finite and positive")
    ratio = collision_horizontal_scale / horizontal_scale
    stride = int(round(ratio))
    if stride < 1 or not math.isclose(ratio, stride, rel_tol=0., abs_tol=1e-8):
        raise ValueError("collision_horizontal_scale must be an integer multiple of the source horizontal_scale")
    for extent in size:
        cells = extent / collision_horizontal_scale
        source_cells = extent / horizontal_scale
        if (cells < 2 or not math.isclose(cells, round(cells), rel_tol=0., abs_tol=1e-8) or
                not math.isclose(source_cells, round(source_cells), rel_tol=0., abs_tol=1e-8)):
            raise ValueError("Terrain size must divide exactly into source and collision grids with at least two collision cells")
    return stride


def _triangle_height_numpy(height,xy,size):
    """MuJoCo's 00-to-11 cell diagonal, as verified against mj_ray."""
    ny,nx=height.shape
    uv=xy/np.asarray(size)*np.array([nx-1,ny-1])
    ix=np.floor(uv[...,0]).astype(np.int64).clip(0,nx-2)
    iy=np.floor(uv[...,1]).astype(np.int64).clip(0,ny-2)
    fx,fy=uv[...,0]-ix,uv[...,1]-iy
    z00,z10=height[iy,ix],height[iy,ix+1]
    z01,z11=height[iy+1,ix],height[iy+1,ix+1]
    z=np.where(fx>=fy,z00+(z10-z00)*fx+(z11-z10)*fy,z00+(z11-z01)*fx+(z01-z00)*fy)
    return np.where(((xy>=0)&(xy<=np.asarray(size))).all(-1),z,np.inf)


def _hiking_patch_height_numpy(height,xy,size,boxes):
    z=_triangle_height_numpy(height,xy,size)
    for x,y,zc,hx,hy,hz,yaw in boxes:
        dx,dy=xy[...,0]-x,xy[...,1]-y
        bx,by=np.cos(yaw)*dx+np.sin(yaw)*dy,-np.sin(yaw)*dx+np.cos(yaw)*dy
        hit=(abs(bx)<=hx)&(abs(by)<=hy)
        z=np.where(hit,np.maximum(z,zc+hz),z)
    return z


def _hiking_full_terrain(cfg,difficulty,spec,rng):
    from .hiking_terrain_source import source_height_array,source_rotated_boxes
    height,origin=source_height_array(cfg.family,difficulty,rng,cfg.size,cfg.horizontal_scale,cfg.vertical_scale)
    low,high=float(height.min()),float(height.max());span=max(high-low,.005)
    name="hiking_full_"+uuid.uuid4().hex
    field=spec.add_hfield(name=name,size=[cfg.size[0]/2,cfg.size[1]/2,span,1.],
        nrow=height.shape[0],ncol=height.shape[1],userdata=((height-low)/span).flatten().tolist())
    body=spec.body("terrain")
    geom=body.add_geom(type=mujoco.mjtGeom.mjGEOM_HFIELD,hfieldname=name,
        pos=[cfg.size[0]/2,cfg.size[1]/2,low],group=0)
    geometries=[TerrainGeometry(geom=geom,hfield=field,color=(.45,.5,.48,1.))]
    boxes=source_rotated_boxes(difficulty,rng,cfg.size) if cfg.family=="mesh_boxes" else []
    for axis,direction in ((0,-1),(0,1),(1,-1),(1,1)):
        if rng.random()>=cfg.wall_probability:continue
        half=[cfg.size[0]/2,cfg.size[1]/2,cfg.wall_height/2];half[axis]=cfg.wall_thickness/2
        center=[cfg.size[0]/2,cfg.size[1]/2,cfg.wall_height/2]
        center[axis]=(0. if direction<0 else cfg.size[axis])+direction*cfg.wall_thickness/2
        boxes.append((*center,*half,0.))
    for x,y,z,hx,hy,hz,yaw in boxes:
        box=body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX,size=[hx,hy,hz],pos=[x,y,z],
            quat=[math.cos(yaw/2),0.,0.,math.sin(yaw/2)],group=0)
        geometries.append(TerrainGeometry(geom=box,color=(.35,.4,.38,1.)))
    patches={}
    for key,patch_cfg in (cfg.flat_patch_sampling or {}).items():
        if key=="spawn" and cfg.family=="recovery_flat":
            patches[key]=np.repeat(origin[None],patch_cfg.num_patches,axis=0)
            continue
        if key!="target":raise ValueError("Full Hiking samples target patches only; spawn uses the source origin")
        radii=(.05,.10,.15) if cfg.family=="mesh_boxes" else (.05,.10,.15,.20)
        angle=np.linspace(0.,2*np.pi,10)
        offsets=np.concatenate([r*np.stack((np.cos(angle),np.sin(angle)),-1) for r in radii])
        selected=np.zeros((patch_cfg.num_patches,3));pending=np.arange(patch_cfg.num_patches)
        fixed=cfg.family.startswith("pyramid_stairs") or cfg.family=="square_gaps"
        for _ in range(10000):
            if len(pending)==0:break
            candidates=(np.repeat((origin[:2]+np.array([3.7,0.]))[None],len(pending),0) if fixed else
                        rng.uniform(np.zeros(2),np.asarray(cfg.size),(len(pending),2)))
            heights=_hiking_patch_height_numpy(height,candidates[:,None]+offsets[None],cfg.size,boxes)
            valid=np.isfinite(heights).all(-1)&((heights.max(-1)-heights.min(-1))<=patch_cfg.max_height_diff)
            ids=pending[valid]
            selected[ids,:2]=candidates[valid];selected[ids,2]=heights[valid,-1]
            pending=pending[~valid]
        if len(pending):raise RuntimeError(f"Hiking {cfg.family}: could not sample {len(pending)} flat target patches")
        patches[key]=selected
    return TerrainOutput(origin=origin,geometries=geometries,flat_patches=patches)


def hiking_terrain_generator(seed=0, rows=10, cols=20, recovery_flat_fraction=None,
                            collision_horizontal_scale=.1,walking_only=False,task_protocol="legacy_walk_turn_v1"):
    """Generate original joint-task terrain or a conservative walking profile.

    The walking profile retains roughness, ordinary stairs, boxes and slopes,
    with original relative weights renormalized after excluding jumps/gaps.
    A curriculum value of one maps to source difficulty .4 (.2 for rotated
    boxes, whose independent height noise otherwise permits larger steps).
    This limits ordinary stair rise to about .12m, boxes to .21m and source
    slope parameter to .28. Collision discretization and transitions still
    require empirical traversal validation; no walking success is implied.
    """
    if task_protocol=="hiking_full_v1":
        # The full source task includes gaps/high stairs and its complete
        # difficulty range. Forward-only commands do not imply deleting these
        # terrains. MuJoCo uses the source .05m height grid plus convex boxes;
        # PhysX's steep-slope vertex displacement is an explicit engine limit.
        patches={"target":FlatPatchSamplingCfg(num_patches=50,patch_radius=.2,max_height_diff=.05)}
        return TerrainGeneratorCfg(seed=seed,size=(8.,8.),border_width=3.,num_rows=rows,num_cols=cols,
            curriculum=True,color_scheme="none",sub_terrains={
                name:HikingTerrainCfg(family=name,size=(8.,8.),proportion=proportion,flat_patch_sampling=patches,
                    collision_horizontal_scale=.05,recovery_pad_width=0.,maximum_difficulty=1.,task_protocol=task_protocol)
                for name,proportion in HIKING_PROPORTIONS.items()})
    if recovery_flat_fraction is None:recovery_flat_fraction=0. if walking_only else .25
    if not 0 <= recovery_flat_fraction < 1:
        raise ValueError("recovery_flat_fraction must lie in [0,1)")
    if walking_only and recovery_flat_fraction:
        raise ValueError("Walking-only terrain cannot allocate a recovery-flat family")
    _collision_stride((8., 8.), .05, collision_horizontal_scale)
    patches={"spawn":FlatPatchSamplingCfg(num_patches=1,patch_radius=.65,max_height_diff=.05),
             "target":FlatPatchSamplingCfg(num_patches=50,patch_radius=.2,max_height_diff=.05)}
    sub={}
    if recovery_flat_fraction:
        sub["recovery_flat"]=HikingTerrainCfg(family="recovery_flat",proportion=recovery_flat_fraction,flat_patch_sampling=patches,
                                             collision_horizontal_scale=collision_horizontal_scale)
    proportions=WALKING_TERRAIN_PROPORTIONS if walking_only else HIKING_PROPORTIONS
    sub.update({name:HikingTerrainCfg(family=name,proportion=proportion*(1-recovery_flat_fraction),flat_patch_sampling=patches,
                                    collision_horizontal_scale=collision_horizontal_scale,
                                    maximum_difficulty=(.2 if name=="mesh_boxes" else .4) if walking_only else 1.)
                for name,proportion in proportions.items()})
    return TerrainGeneratorCfg(seed=seed,size=(8.,8.),border_width=3.,num_rows=rows,num_cols=cols,
                               curriculum=True,sub_terrains=sub,color_scheme="none")


class TerrainHeightField:
    """Differentiable global terrain queries decoded from the collision model.

    Historical queries retain bilinear interpolation and raster edge distance.
    Full queries use the actual piecewise-triangular hfield and analytic box
    rays, including rotated obstacles, walls and the outer floor border. Sharp
    mesh edges use separate original-rule virtual cylinders.
    """
    patch_dim=81

    def __init__(self, heights, centers, half_sizes):
        self.exact_triangles=False
        self.hiking_edges=None
        self.hiking_boxes=None
        self.heights=heights
        self.centers=centers.to(heights)
        self.half_sizes=half_sizes.to(heights)
        xs=torch.unique(self.centers[:,0],sorted=True);ys=torch.unique(self.centers[:,1],sorted=True)
        self.grid_min=torch.stack((xs[0],ys[0]))-self.half_sizes[0]
        self.tile_size=2*self.half_sizes[0]
        self.grid_shape=(len(xs),len(ys))
        # Critics query this path outside torch.compile during CUDA graph
        # capture. Materialize constants once instead of uploading a new CPU
        # tensor inside contains() on every captured call.
        self.grid_max=self.grid_min+self.tile_size*torch.tensor(
            self.grid_shape,device=heights.device,dtype=heights.dtype)
        self.lookup=torch.full(self.grid_shape,-1,dtype=torch.long,device=heights.device)
        for i,center in enumerate(self.centers):
            row=int(torch.argmin((xs-center[0]).abs()));col=int(torch.argmin((ys-center[1]).abs()))
            self.lookup[row,col]=i
        if (self.lookup<0).any(): raise ValueError("Terrain query requires a full rectangular tile grid")
        if not torch.allclose(self.half_sizes,self.half_sizes[:1].expand_as(self.half_sizes)):
            raise ValueError("Terrain query requires equal tile sizes")
        distance=[]
        scale=float(self.tile_size[0])/(heights.shape[-1]-1)
        for array in heights.detach().cpu().numpy():
            edge=np.zeros_like(array,dtype=bool)
            dx=np.abs(np.diff(array,axis=1))>.03
            dy=np.abs(np.diff(array,axis=0))>.03
            edge[:,:-1]|=dx;edge[:,1:]|=dx
            edge[:-1,:]|=dy;edge[1:,:]|=dy
            distance.append(distance_transform_edt(~edge)*scale if edge.any() else np.full_like(array,1e3))
        self.edge_distances=torch.as_tensor(np.stack(distance),device=heights.device,dtype=heights.dtype)

    @classmethod
    def from_model(cls,model,device="cpu"):
        values=[];centers=[];sizes=[];base_boxes=[];terrain_body_ids=set();full=False
        for gid in range(model.ngeom):
            if int(model.geom_type[gid])!=int(mujoco.mjtGeom.mjGEOM_HFIELD):continue
            fid=int(model.geom_dataid[gid]);start=int(model.hfield_adr[fid]);ny=int(model.hfield_nrow[fid]);nx=int(model.hfield_ncol[fid])
            full|="hiking_full_" in (mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_HFIELD,fid) or "")
            terrain_body_ids.add(int(model.geom_bodyid[gid]))
            array=np.array(model.hfield_data[start:start+nx*ny]).reshape(ny,nx)
            # Generated terrain body has identity pose; local positions already
            # carry the terrain generator's global grid translations.
            values.append(array*model.hfield_size[fid,2]+model.geom_pos[gid,2])
            centers.append(model.geom_pos[gid,:2].copy());sizes.append(model.hfield_size[fid,:2].copy())
            base_half=float(model.hfield_size[fid,3])/2
            base_boxes.append((*model.geom_pos[gid,:2],model.geom_pos[gid,2]-base_half,
                               *model.hfield_size[fid,:2],base_half,0.))
        if not values:raise ValueError("No heightfield terrain geometry found")
        result=cls(torch.tensor(np.stack(values),device=device,dtype=torch.float32),
                   torch.tensor(np.stack(centers),device=device,dtype=torch.float32),
                   torch.tensor(np.stack(sizes),device=device,dtype=torch.float32))
        if full:
            from .hiking_geometry import heightfield_mesh,sharp_edges,HikingEdgeCylinders,HikingBoxRays
            import trimesh
            edges=[];boxes=[]
            for index,(height,center,size) in enumerate(zip(values,centers,sizes)):
                edges.append(sharp_edges(heightfield_mesh(height,center,size),seed=index))
            for gid in range(model.ngeom):
                if (int(model.geom_bodyid[gid]) not in terrain_body_ids or
                    int(model.geom_type[gid])!=int(mujoco.mjtGeom.mjGEOM_BOX)):
                    continue
                q=model.geom_quat[gid]
                if abs(q[1])+abs(q[2])>1e-6:raise ValueError("Hiking terrain boxes must rotate around world z only")
                yaw=2*np.arctan2(q[3],q[0]);position=model.geom_pos[gid].copy();half=model.geom_size[gid].copy()
                boxes.append((*position,*half,yaw))
                transform=trimesh.transformations.quaternion_matrix(q)
                transform[:3,3]=position
                mesh=trimesh.creation.box(extents=half*2,transform=transform)
                edges.append(sharp_edges(mesh,seed=gid))
            result.exact_triangles=True
            # mj_rayHfield also intersects its base box. Retain it for rays
            # starting below a surface, not just the usual +20m foot rays.
            result.hiking_boxes=HikingBoxRays(boxes+base_boxes,device)
            result.hiking_edges=HikingEdgeCylinders(np.concatenate(edges,axis=0),device)
        return result

    @classmethod
    def from_env(cls,env):
        cached=getattr(env,"_perceptive_terrain_oracle",None)
        if cached is None:
            cached=cls.from_model(env.sim.mj_model,device=env.device)
            env._perceptive_terrain_oracle=cached
        return cached

    def contains(self,xy):
        return ((xy>=self.grid_min)&(xy<=self.grid_max)).all(-1)

    def height_at(self,xy):
        if self.exact_triangles:
            height=self._sample(self.heights,xy,triangular=True)
            height=torch.where(self.contains(xy),height,torch.zeros_like(height))
            return torch.maximum(height,self.hiking_boxes.height_at(xy))
        return self._sample(self.heights,xy)

    def hiking_ray_height_at(self,ray_start_w):
        """Exact downward rays on the full protocol's physical terrain surface.

        Hfield triangles share MuJoCo's 00-to-11 diagonal, and rotated boxes
        are analytic. Selection is piecewise differentiable in ray xy. The
        foot-scanner starts 20m above the feet, matching the source sensor.
        """
        if not self.exact_triangles:
            raise RuntimeError("Hiking foot rays require the hiking_full_v1 terrain geometry")
        xy=ray_start_w[...,:2];start_z=ray_start_w[...,2]
        ground=self._sample(self.heights,xy,triangular=True)
        ground=torch.where(self.contains(xy)&(ground<=start_z),ground,torch.full_like(ground,-torch.inf))
        height=torch.maximum(ground,self.hiking_boxes.height_at(xy,start_z))
        return torch.where(torch.isneginf(height),torch.full_like(height,torch.inf),height)

    def edge_distance_at(self,xy):
        """Distance to >3cm height jumps, a raster proxy rather than mesh dihedrals."""
        return self._sample(self.edge_distances,xy)

    def _sample(self,maps,xy,triangular=False):
        prefix=xy.shape[:-1];flat=xy.reshape(-1,2)
        indices=torch.floor((flat-self.grid_min)/self.tile_size).long()
        row=indices[:,0].clamp(0,self.grid_shape[0]-1);col=indices[:,1].clamp(0,self.grid_shape[1]-1)
        tile=self.lookup[row,col]
        uv=((flat-self.centers[tile])/self.tile_size+.5).clamp(0,1)
        ny,nx=maps.shape[-2:]
        x,y=uv[:,0]*(nx-1),uv[:,1]*(ny-1)
        ix=x.floor().long().clamp(0,nx-2);iy=y.floor().long().clamp(0,ny-2)
        fx,fy=x-ix,y-iy
        z00=maps[tile,iy,ix];z10=maps[tile,iy,ix+1]
        z01=maps[tile,iy+1,ix];z11=maps[tile,iy+1,ix+1]
        if triangular:
            result=torch.where(fx>=fy,z00+(z10-z00)*fx+(z11-z10)*fy,
                               z00+(z11-z01)*fx+(z01-z00)*fy)
        else:
            result=(1-fy)*((1-fx)*z00+fx*z10)+fy*((1-fx)*z01+fx*z11)
        return result.reshape(prefix)

    def local_patch(self,state):
        line=torch.linspace(-.8,.8,9,device=state.device,dtype=state.dtype)
        x,y=torch.meshgrid(line,line,indexing="xy");offset=torch.stack((x.flatten(),y.flatten()),-1)
        q=state[...,3:7];q=q/q.norm(dim=-1,keepdim=True).clamp_min(1e-8)
        w,xq,yq,zq=q.unbind(-1)
        yaw=torch.atan2(2*(w*zq+xq*yq),1-2*(yq*yq+zq*zq))
        c,s=torch.cos(yaw)[...,None],torch.sin(yaw)[...,None]
        dx=c*offset[:,0]-s*offset[:,1];dy=s*offset[:,0]+c*offset[:,1]
        points=state[...,:2].unsqueeze(-2)+torch.stack((dx,dy),-1)
        return self.height_at(points)-self.height_at(state[...,:2]).unsqueeze(-1)
