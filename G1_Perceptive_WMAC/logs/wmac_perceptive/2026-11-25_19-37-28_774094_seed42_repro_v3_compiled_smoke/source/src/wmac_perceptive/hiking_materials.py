"""Source material buckets with an explicit MuJoCo contact-model boundary.

Hiking samples static/dynamic friction and restitution per collision shape.
MuJoCo has one sliding-friction coefficient rather than a static/dynamic pair;
we apply the sampled dynamic coefficient. The source static and restitution
samples are recorded for audit, not silently reinterpreted as solref values.
Restitution parity needs contact calibration or the original PhysX backend.
"""
from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import torch

from mjlab.managers.event_manager import requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg


def sample_hiking_material_buckets(num_buckets=64,*,device="cpu",hiking_training_version=1):
    if num_buckets<1:raise ValueError("Material bucket count must be positive")
    if hiking_training_version >= 3:
        # IsaacLab randomize_rigid_body_material samples row-major on CPU.
        ranges = torch.tensor(((.3, 1.6), (.3, 1.6), (.05, .5)), device="cpu")
        buckets = torch.rand((num_buckets, 3), device="cpu") * (ranges[:, 1] - ranges[:, 0]) + ranges[:, 0]
        buckets = buckets.to(device)
    else:
        buckets=torch.empty(num_buckets,3,device=device)
        buckets[:,0].uniform_(.3,1.6)
        buckets[:,1].uniform_(.3,1.6)
        buckets[:,2].uniform_(.05,.5)
    buckets[:,1]=torch.minimum(buckets[:,0],buckets[:,1])
    return buckets


def hiking_material_protocol_metadata():
    return dict(num_buckets=64,static_friction_range=[.3,1.6],dynamic_friction_range=[.3,1.6],
                restitution_range=[.05,.5],make_consistent=True,scope="all robot collision shapes",
                applied_sliding_friction="sampled dynamic coefficient",
                separate_static_friction_applied=False,restitution_applied=False,
                robot_ground_combine="robot priority 1 over terrain priority 0; matches source multiply-by-1",
                robot_robot_combine="MuJoCo max differs from source PhysX multiply",
                solref_mapping=None,backend_difference="MuJoCo contact law differs from PhysX")


@lru_cache(maxsize=3)
def hiking_material_geom_names(hiking_training_version=2):
    """Canonical collision names from the actual configured G1 asset.

    Only coefficients applied by the MuJoCo material adapter are conditioned
    on: sliding friction for each active collision geom. Sorting names makes
    checkpoint order independent of terrain/body insertion in the scene.
    """
    from mjlab.entity import Entity
    from src.tasks.amp_loco.config.g1.env_cfgs import g1_amp_flat_aligned_env_cfg
    cfg = g1_amp_flat_aligned_env_cfg()
    cfg._hiking_training_version = hiking_training_version
    if hiking_training_version >= 3:
        from src.tasks.amp_perceptive.hiking_robot import get_hiking_robot_spec
        cfg.scene.entities["robot"].spec_fn = get_hiking_robot_spec
    configure_hiking_materials(cfg)
    model = Entity(cfg.scene.entities["robot"]).spec.compile()
    names = tuple(sorted(model.geom(i).name.split("/")[-1] for i in range(model.ngeom)
                         if model.geom_contype[i] | model.geom_conaffinity[i]))
    if not names or len(set(names)) != len(names) or any(not name for name in names):
        raise ValueError("Hiking material conditioning requires unique named collision geoms")
    return names


def hiking_material_geom_ids(robot, model, device, hiking_training_version=2):
    """Map the canonical material vector to the compiled scene, once."""
    available = {model.geom(int(i)).name.split("/")[-1]: int(i)
                 for i in robot.indexing.geom_ids
                 if model.geom_contype[int(i)] | model.geom_conaffinity[int(i)]}
    names = hiking_material_geom_names(hiking_training_version)
    if set(available) != set(names):
        raise ValueError("Compiled Hiking collision/material schema differs from the checkpoint asset")
    return torch.tensor([available[name] for name in names], dtype=torch.long, device=device)


def configure_hiking_materials(cfg):
    from mjlab.managers.event_manager import EventTermCfg
    cfg.events["physics_material"]=EventTermCfg(func=randomize_hiking_material,mode="startup",
        params={"asset_cfg":SceneEntityCfg("robot")})
    robot=cfg.scene.entities["robot"]
    robot.collisions=deepcopy(robot.collisions)
    for collision in robot.collisions:
        # Source ground friction is 1 with multiply combination. MuJoCo's
        # equal-priority max would otherwise erase all samples below 1.
        collision.priority=1
        collision.condim=3
        if getattr(cfg, "_hiking_training_version", 1) >= 3:
            # ParkourEnvCfg installs its terrain material as the simulation
            # default. PLAY disables per-shape randomization and retains 1.0.
            collision.friction = (1.,)
    return cfg


@requires_model_fields("geom_friction")
def randomize_hiking_material(env,env_ids=None,asset_cfg=SceneEntityCfg("robot")):
    """Startup-only randomization; preserve torsional/rolling friction/solref."""
    robot=env.scene[asset_cfg.name]
    ids=torch.arange(env.num_envs,device=env.device) if env_ids is None else env_ids
    model=env.sim.mj_model
    geom_ids=robot.indexing.geom_ids
    # Visual-only geometries have no contact material in the source PhysX
    # randomizer. Local collision geoms are selected by their compiled masks.
    colliding=torch.as_tensor((model.geom_contype|model.geom_conaffinity)!=0,device=env.device)
    geom_ids=geom_ids[colliding[geom_ids]]
    version = getattr(getattr(env, "cfg", None), "_hiking_training_version", 1)
    if not hasattr(env,"_hiking_material_buckets"):
        env._hiking_material_buckets=sample_hiking_material_buckets(
            device=env.device, hiking_training_version=version)
        env._hiking_material_assignments=torch.full((env.num_envs,len(geom_ids)),-1,
                                                   dtype=torch.long,device=env.device)
        env._hiking_material_geom_ids=geom_ids
    sampling_device = "cpu" if version >= 3 else env.device
    choices=torch.randint(0,64,(len(ids),len(geom_ids)),device=sampling_device).to(env.device)
    env._hiking_material_assignments[ids]=choices
    samples=env._hiking_material_buckets[choices]
    env.sim.model.geom_friction[ids[:,None],geom_ids[None,:],0]=samples[...,1]
