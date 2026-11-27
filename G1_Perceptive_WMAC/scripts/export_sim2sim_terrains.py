"""Write Hiking training terrains as heightfield files for a MuJoCo sim-to-sim.

Each ``<name>.npz`` holds one 8 x 8 m tile from the same generator training uses
(``source_height_array``): ``heights`` in metres in MuJoCo (row = y, column = x)
order, ``size``, the spawn ``origin`` (tile centre) and the walking ``target``
(3.7 m along +x, as Hiking's stair/gap targets). Stair families: pyramid_stairs
descends from the spawn platform, pyramid_stairs_inv ascends.

    python scripts/export_sim2sim_terrains.py --output-dir <dir>
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_SET = {
    # name: (family, difficulty). Stair rise = 0.05 + difficulty * 0.18 m (0.3 m treads).
    "rough_0.5": ("perlin_rough", .5),
    "rough_1.0": ("perlin_rough", 1.),
    "stairs_up_0.3": ("pyramid_stairs_inv", .3),
    "stairs_up_0.6": ("pyramid_stairs_inv", .6),
    "stairs_down_0.3": ("pyramid_stairs", .3),
    "stairs_down_0.6": ("pyramid_stairs", .6),
    "slope_0.5": ("hf_pyramid_slope_inv", .5),
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    import numpy as np
    from src.tasks.amp_perceptive.hiking_terrain_source import source_height_array
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, (family, difficulty) in DEFAULT_SET.items():
        rng = np.random.default_rng(args.seed)
        heights, origin = source_height_array(family, difficulty, rng)
        target = origin[:2] + np.array([3.7, 0.])
        column, row = (int(round(v / .05)) for v in target)
        np.savez(args.output_dir / f"{name}.npz", heights=heights, size=np.array([8., 8.]),
                 horizontal_scale=.05, origin=origin, target=np.array([*target, heights[row, column]]),
                 family=family, difficulty=difficulty, seed=args.seed)
        print(f"{name:16s} {family:22s} difficulty {difficulty:.2f}: heights {heights.min():+.3f}..{heights.max():+.3f} m, "
              f"spawn z {origin[2]:+.3f}, target z {heights[row, column]:+.3f}")


if __name__ == "__main__":
    main()
