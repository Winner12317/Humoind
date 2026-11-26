"""Hiking source terrain arrays, adapted from InstinctLab de6b92f01eb155d51.

The original functions, noise construction, dimensions and quantization are
retained; NumPy RNG is supplied per terrain instead of process-global state.
IsaacLab mesh conversion is deliberately not used for MuJoCo collision: that
engine requires heightfields or convex primitives. See NOTICE.txt for the
CC-BY-NC-4.0 source attribution and licenses/instinctlab-CC-BY-NC-4.0.txt.
"""
from __future__ import annotations
from types import SimpleNamespace
from typing import Sequence
import numpy as np
import scipy.interpolate as interpolate


def source_height_array(family, difficulty, rng, size=(8.,8.), horizontal_scale=.05, vertical_scale=.005):
    """Original height-field decorator's border, array and origin conventions.

    Returns physical heights in MuJoCo (y,x) order and the official origin.
    Unlike the historical port there is no additional flattened spawn pad.
    """
    border_width=1. if family.startswith("pyramid_stairs") or family in ("square_gaps","hf_pyramid_slope_inv") else 0.
    shape=tuple(int(v/horizontal_scale)+1 for v in size)
    border=int(border_width/horizontal_scale)+1
    inner=tuple((v-2*border)*horizontal_scale for v in shape)
    shared=dict(size=inner,horizontal_scale=horizontal_scale,vertical_scale=vertical_scale,
                slope_threshold=1.,rng=rng)
    noise=SimpleNamespace(**shared,noise_scale=.05,noise_frequency=20,fractal_octaves=2,
                          fractal_lacunarity=2.,fractal_gain=.25,centering=True)
    cfg=SimpleNamespace(**shared,perlin_cfg=None)
    if family.startswith("perlin"):
        cfg=noise;cfg.noise_scale=(0.,.1)
        values=perlin_plane_terrain(difficulty,cfg)
    elif family.startswith("pyramid_stairs"):
        high="high" in family
        cfg.step_height_range=(.05,.45 if high else .23)
        cfg.step_width=1.5 if high else .3
        cfg.platform_width=4. if high else 2.5
        cfg.inverted="inv" in family;cfg.perlin_cfg=noise
        values=perlin_pyramid_stairs_terrain(difficulty,cfg)
    elif family=="square_gaps":
        cfg.gap_distance_range=(.1,.7);cfg.gap_depth=(.4,.6);cfg.platform_width=2.5
        values=perlin_square_gap_terrain(difficulty,cfg)
    elif family=="boxes":
        cfg.obstacle_height_range=(.05,.45);cfg.obstacle_width_range=(.8,1.5)
        cfg.obstacle_height_mode="fixed";cfg.num_obstacles=20;cfg.platform_width=1.5;cfg.perlin_cfg=noise
        values=perlin_discrete_obstacles_terrain(difficulty,cfg)
    elif family=="hf_pyramid_slope_inv":
        cfg.slope_range=(0.,.7);cfg.inverted=True;cfg.platform_width=1.5
        cfg.perlin_cfg=noise;noise.noise_scale=0.
        values=perlin_pyramid_sloped_terrain(difficulty,cfg)
    elif family in ("mesh_boxes","recovery_flat"):
        values=np.zeros(tuple(v-2*border for v in shape),dtype=np.int16)
    else:
        raise ValueError(f"Unknown Hiking terrain family: {family}")
    heights=np.zeros(shape,dtype=np.int16)
    heights[border:-border,border:-border]=values
    x1,x2=int((size[0]/2-1)/horizontal_scale),int((size[0]/2+1)/horizontal_scale)
    y1,y2=int((size[1]/2-1)/horizontal_scale),int((size[1]/2+1)/horizontal_scale)
    origin=np.array([size[0]/2,size[1]/2,heights[x1:x2,y1:y2].max()*vertical_scale])
    return (heights.T*vertical_scale).astype(np.float32),origin


def source_rotated_boxes(difficulty,rng,size=(8.,8.)):
    """Original random_multi_box_terrain boxes, without rasterizing rotation."""
    result=[]
    for _ in range(max(1,int(.3*(size[0]*size[1])/(.4*.4)))):
        width=.4+rng.uniform(-1,1)*.1
        length=.4+rng.uniform(-1,1)*.1
        height=.1+difficulty*.3+rng.uniform(-1,1)*.05
        x=rng.uniform(width/2,size[0]-width/2)
        y=rng.uniform(length/2,size[1]-length/2)
        if (size[0]/2-.75-width/2<x<size[0]/2+.75+width/2 and
            size[1]/2-.75-length/2<y<size[1]/2+.75+length/2):
            continue
        yaw=rng.uniform(0,2*np.pi)
        result.append((x,y,height/2,width/2,length/2,height/2,yaw))
    return result

def generate_perlin_noise_2d(shape: Sequence[int], res: Sequence[int], rng) -> np.ndarray:

    def f(t):
        return 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]].transpose(1, 2, 0) % 1
    angles = 2 * np.pi * rng.random((res[0] + 1, res[1] + 1))
    gradients = np.dstack((np.cos(angles), np.sin(angles)))
    g00 = gradients[0:-1, 0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g10 = gradients[1:, 0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g01 = gradients[0:-1, 1:].repeat(d[0], 0).repeat(d[1], 1)
    g11 = gradients[1:, 1:].repeat(d[0], 0).repeat(d[1], 1)
    n00 = np.sum(grid * g00, 2)
    n10 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1])) * g10, 2)
    n01 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1] - 1)) * g01, 2)
    n11 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1] - 1)) * g11, 2)
    t = f(grid)
    n0 = n00 * (1 - t[:, :, 0]) + t[:, :, 0] * n10
    n1 = n01 * (1 - t[:, :, 0]) + t[:, :, 0] * n11
    return np.sqrt(2) * ((1 - t[:, :, 1]) * n0 + t[:, :, 1] * n1) * 0.5 + 0.5

def generate_fractal_noise_2d(xSize=20, ySize=20, xSamples=1600, ySamples=1600, frequency=10, fractalOctaves=2, fractalLacunarity=2.0, fractalGain=0.25, zScale=0.23, centering=False, rng=None) -> np.ndarray:
    xScale = int(frequency * xSize)
    yScale = int(frequency * ySize)
    amplitude = 1
    expected_xSamples = int(xScale * fractalLacunarity ** fractalOctaves)
    expected_ySamples = int(yScale * fractalLacunarity ** fractalOctaves)
    if xSamples > expected_xSamples or ySamples > expected_ySamples:
        raise RuntimeError('Situation not checked, using expected_*Samples is in case the *Samples is not the multiple of *Size')
    noise = np.zeros((expected_xSamples, expected_ySamples))
    for _ in range(fractalOctaves):
        noise += amplitude * generate_perlin_noise_2d(noise.shape, (xScale, yScale), rng) * zScale
        amplitude *= fractalGain
        xScale, yScale = (int(fractalLacunarity * xScale), int(fractalLacunarity * yScale))
    if centering:
        noise -= np.mean(noise)
    return noise[:xSamples, :ySamples].copy()

def generate_perlin_noise(difficulty: float, cfg: hf_terrains_cfg.PerlinPlaneTerrainCfg) -> np.ndarray:
    noise_scale = cfg.noise_scale if not isinstance(cfg.noise_scale, (list, tuple)) else interpolate.interp1d([0, 1], cfg.noise_scale, kind='linear')(difficulty)
    return (generate_fractal_noise_2d(xSize=cfg.size[0], ySize=cfg.size[1], xSamples=int(cfg.size[0] / cfg.horizontal_scale), ySamples=int(cfg.size[1] / cfg.horizontal_scale), frequency=cfg.noise_frequency, fractalOctaves=cfg.fractal_octaves, fractalLacunarity=cfg.fractal_lacunarity, fractalGain=cfg.fractal_gain, zScale=noise_scale, centering=cfg.centering, rng=cfg.rng) / cfg.vertical_scale).astype(np.int16)

def perlin_plane_terrain(difficulty: float, cfg: hf_terrains_cfg.PerlinPlaneTerrainCfg) -> np.ndarray:
    return generate_perlin_noise(difficulty=difficulty, cfg=cfg)

def perlin_pyramid_sloped_terrain(difficulty: float, cfg: hf_terrains_cfg.PerlinPyramidSlopedTerrainCfg) -> np.ndarray:
    if cfg.inverted:
        slope = -cfg.slope_range[0] - difficulty * (cfg.slope_range[1] - cfg.slope_range[0])
    else:
        slope = cfg.slope_range[0] + difficulty * (cfg.slope_range[1] - cfg.slope_range[0])
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    height_max = int(slope * cfg.size[0] / 2 / cfg.vertical_scale)
    center_x = int(width_pixels / 2)
    center_y = int(length_pixels / 2)
    x = np.arange(0, width_pixels)
    y = np.arange(0, length_pixels)
    xx, yy = np.meshgrid(x, y, sparse=True)
    xx = (center_x - np.abs(center_x - xx)) / center_x
    yy = (center_y - np.abs(center_y - yy)) / center_y
    xx = xx.reshape(width_pixels, 1)
    yy = yy.reshape(1, length_pixels)
    hf_raw = np.zeros((width_pixels, length_pixels))
    hf_raw = height_max * xx * yy
    platform_width = int(cfg.platform_width / cfg.horizontal_scale / 2)
    x_pf = width_pixels // 2 - platform_width
    y_pf = length_pixels // 2 - platform_width
    z_pf = hf_raw[x_pf, y_pf]
    hf_raw = np.clip(hf_raw, min(0, z_pf), max(0, z_pf))
    if cfg.perlin_cfg is not None:
        perlin_cfg = cfg.perlin_cfg
        perlin_cfg.size = cfg.size
        perlin_cfg.horizontal_scale = cfg.horizontal_scale
        perlin_cfg.vertical_scale = cfg.vertical_scale
        perlin_cfg.slope_threshold = cfg.slope_threshold
        perlin_noise = generate_perlin_noise(difficulty, perlin_cfg)
        hf_raw += perlin_noise
    return np.rint(hf_raw).astype(np.int16)

def perlin_pyramid_stairs_terrain(difficulty: float, cfg: hf_terrains_cfg.PerlinPyramidStairsTerrainCfg) -> np.ndarray:
    step_height = cfg.step_height_range[0] + difficulty * (cfg.step_height_range[1] - cfg.step_height_range[0])
    if cfg.inverted:
        step_height *= -1
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    step_width = round(cfg.step_width / cfg.horizontal_scale)
    step_height = round(step_height / cfg.vertical_scale)
    platform_width = round(cfg.platform_width / cfg.horizontal_scale)
    hf_raw = np.zeros((width_pixels, length_pixels))
    current_step_height = 0
    start_x, start_y = (0, 0)
    stop_x, stop_y = (width_pixels, length_pixels)
    while stop_x - start_x > platform_width and stop_y - start_y > platform_width:
        start_x += step_width
        stop_x -= step_width
        start_y += step_width
        stop_y -= step_width
        current_step_height += step_height
        hf_raw[start_x:stop_x, start_y:stop_y] = current_step_height
    if cfg.perlin_cfg is not None:
        perlin_cfg = cfg.perlin_cfg
        perlin_cfg.size = cfg.size
        perlin_cfg.horizontal_scale = cfg.horizontal_scale
        perlin_cfg.vertical_scale = cfg.vertical_scale
        perlin_cfg.slope_threshold = cfg.slope_threshold
        perlin_noise = generate_perlin_noise(difficulty, perlin_cfg)
        hf_raw += perlin_noise
    return np.rint(hf_raw).astype(np.int16)

def perlin_discrete_obstacles_terrain(difficulty: float, cfg: hf_terrains_cfg.PerlinDiscreteObstaclesTerrainCfg) -> np.ndarray:
    obs_height = cfg.obstacle_height_range[0] + difficulty * (cfg.obstacle_height_range[1] - cfg.obstacle_height_range[0])
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    obs_height = round(obs_height / cfg.vertical_scale)
    obs_width_min = round(cfg.obstacle_width_range[0] / cfg.horizontal_scale)
    obs_width_max = round(cfg.obstacle_width_range[1] / cfg.horizontal_scale)
    platform_width = round(cfg.platform_width / cfg.horizontal_scale)
    obs_width_range = np.arange(obs_width_min, obs_width_max, 4)
    obs_length_range = np.arange(obs_width_min, obs_width_max, 4)
    obs_x_range = np.arange(1, width_pixels, 4)
    obs_y_range = np.arange(1, length_pixels, 4)
    hf_raw = np.zeros((width_pixels, length_pixels))
    for _ in range(cfg.num_obstacles):
        if cfg.obstacle_height_mode == 'choice':
            height = cfg.rng.choice([-obs_height, -obs_height // 2, obs_height // 2, obs_height])
        elif cfg.obstacle_height_mode == 'fixed':
            height = obs_height
        else:
            raise ValueError(f"Unknown obstacle height mode '{cfg.obstacle_height_mode}'. Must be 'choice' or 'fixed'.")
        width = round(cfg.rng.choice(obs_width_range))
        length = round(cfg.rng.choice(obs_length_range))
        x_start = int(cfg.rng.choice(obs_x_range))
        y_start = int(cfg.rng.choice(obs_y_range))
        if x_start + width > width_pixels:
            x_start = width_pixels - width - 1
        if y_start + length > length_pixels:
            y_start = length_pixels - length - 1
        hf_raw[x_start:x_start + width, y_start:y_start + length] = height
    x1 = (width_pixels - platform_width) // 2
    x2 = (width_pixels + platform_width) // 2
    y1 = (length_pixels - platform_width) // 2
    y2 = (length_pixels + platform_width) // 2
    hf_raw[x1:x2, y1:y2] = 0
    if cfg.perlin_cfg is not None:
        perlin_cfg = cfg.perlin_cfg
        perlin_cfg.size = cfg.size
        perlin_cfg.horizontal_scale = cfg.horizontal_scale
        perlin_cfg.vertical_scale = cfg.vertical_scale
        perlin_cfg.slope_threshold = cfg.slope_threshold
        perlin_noise = generate_perlin_noise(difficulty, perlin_cfg)
        hf_raw += perlin_noise
    return np.rint(hf_raw).astype(np.int16)

def perlin_square_gap_terrain(difficulty: float, cfg: hf_terrains_cfg.PerlinSquareGapTerrainCfg) -> np.ndarray:
    gap_distance = cfg.gap_distance_range[0] + difficulty * (cfg.gap_distance_range[1] - cfg.gap_distance_range[0])
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    platform_width = round(cfg.platform_width / cfg.horizontal_scale)
    hf_raw = np.zeros((width_pixels, length_pixels))
    gap_distance_px = round(gap_distance / cfg.horizontal_scale)
    platform_start_x = (width_pixels - platform_width) // 2
    platform_end_x = platform_start_x + platform_width
    platform_start_y = (length_pixels - platform_width) // 2
    platform_end_y = platform_start_y + platform_width
    platform_start_x = max(0, platform_start_x)
    platform_start_y = max(0, platform_start_y)
    platform_end_x = min(width_pixels, platform_end_x)
    platform_end_y = min(length_pixels, platform_end_y)
    hf_raw[platform_start_x:platform_end_x, platform_start_y:platform_end_y] = 0
    gap_depth = cfg.rng.uniform(cfg.gap_depth[0], cfg.gap_depth[1])
    gap_value = -round(gap_depth / cfg.vertical_scale)
    sx = platform_start_x - gap_distance_px
    ex = platform_end_x + gap_distance_px
    sy = platform_start_y - gap_distance_px
    ey = platform_end_y + gap_distance_px
    sx_clamped = max(0, sx)
    ex_clamped = min(width_pixels, ex)
    sy_clamped = max(0, sy)
    ey_clamped = min(length_pixels, ey)
    t = gap_distance_px
    hf_raw[sx_clamped:ex_clamped, sy_clamped:sy_clamped + t] = gap_value
    hf_raw[sx_clamped:ex_clamped, max(sy_clamped, ey_clamped - t):ey_clamped] = gap_value
    hf_raw[sx_clamped:sx_clamped + t, sy_clamped:ey_clamped] = gap_value
    hf_raw[max(sx_clamped, ex_clamped - t):ex_clamped, sy_clamped:ey_clamped] = gap_value
    ey_clamped = min(length_pixels, ey)
    t = gap_distance_px
    if cfg.perlin_cfg is not None:
        perlin_cfg = cfg.perlin_cfg
        perlin_cfg.size = cfg.size
        perlin_cfg.horizontal_scale = cfg.horizontal_scale
        perlin_cfg.vertical_scale = cfg.vertical_scale
        perlin_cfg.slope_threshold = cfg.slope_threshold
        perlin_noise = generate_perlin_noise(difficulty, perlin_cfg)
        hf_raw += perlin_noise
    return np.rint(hf_raw).astype(np.int16)
