# Third-party notices

The Muon Newton–Schulz operations and update equations in
`src/wmac_perceptive/optimizers.py` adapt PyTorch v2.10.0
[`torch/optim/_muon.py`](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/optim/_muon.py),
SHA256 `38e95918424bd26604f786195a4aa72afb60674b4ac9ccbde777974013cb1f52`.
Copyright (c) PyTorch contributors. The original BSD-3-Clause license and
PyTorch's bundled notices are reproduced in
[licenses/pytorch-v2.10-BSD-3-Clause.txt](licenses/pytorch-v2.10-BSD-3-Clause.txt).
The adaptation supplies the official SuperMimic parameter partition, unified
checkpoint state and device-side finite guard on the installed PyTorch 2.7.

The legacy local-increment dynamics parameterization and training-loop design reference
[SuperMimic's unofficial reproduction](https://github.com/zita-ch/supermimic_vibed),
commit `0525a23a2dcfa61f8543f98b84c075fe66d9d286`. Its MIT license follows.
Existing mjlab and RSL-RL components retain their own license headers.

The visual-latent decoder, EMA statistics and variance-normalized prediction
objective in `src/wmac_perceptive/instinct_latent.py` adapt the files
`src/instinct_mj/rl/supermimic/networks.py` and `runner.py` from
[project-instinct/InstinctMJ, supermimic branch](https://github.com/project-instinct/InstinctMJ/tree/03563ee02665f327425106372264070315673748),
commit `03563ee02665f327425106372264070315673748`. Those two upstream files
explicitly carry the supermimic_vibed MIT attribution, reproduced in
[licenses/instinct-supermimic-vibed-MIT.txt](licenses/instinct-supermimic-vibed-MIT.txt).
This attribution applies to those identified files; the rest of the upstream
repository has its separate root license. New `hiking_depth_only_v1` runs use
the upstream eight depth-only encoder/decoder channels. The former additional
validity channels remain only in the explicit `local_depth_valid_v1` checkpoint
protocol; raw validity can still be used for diagnostics. The local integration
retains GenCoord physical integration and this task's AMP and observation
conventions. Fixed-source tests compare the adapted numeric
operations and gradients against the attributed upstream functions.

The perceptive GenCoord implementation in `src/wmac/gencoord.py` adapts
`GenCoordRepresentation.apply_prediction` and the integrated-state loss from
[HybridRobotics/SuperMimic](https://github.com/HybridRobotics/supermimic), commit
`5efbf114e11f725c31d83e217be987c13c0e4ce2`, licensed under Apache-2.0.
The local adaptation changes state storage order and supplies a PyTorch-only
interface for the terrain/depth-conditioned model. Source-parity tests extract
only the corresponding functions from an optional, hash-checked upstream
checkout. The upstream license is reproduced in
[licenses/supermimic-Apache-2.0.txt](licenses/supermimic-Apache-2.0.txt).
The optimizer parameter partition, persistent-stream CUDA update capture,
survival recurrence and reference critic also adapt this same commit's
`learners/learner_utils.py`, `learners/policy_learner.py` and `models.py`.
Local wrappers retain the visual latent, terrain and AMP task interfaces.
On 2026-11-11, the training reference was checked against current official
`main@abc124c1bd5976341034668af3d12c087f2f1f55`: its 35 source files under
`src/` are byte-identical to the originally adapted `5efbf114` snapshot.
Current training therefore follows the published code's Muon/Adam split and
constant learning rate, not the paper's Adam-only cosine schedule.

The `hiking_depth_only_v1` camera and depth preprocessing adapt the Hiking
task and camera/noise configuration from
[project-instinct/InstinctLab](https://github.com/project-instinct/InstinctLab/tree/de6b92f01eb155d51f30e8ed1ea7aecd2eb1f057),
commit `de6b92f01eb155d51f30e8ed1ea7aecd2eb1f057`, the code linked by the
[Hiking in the Wild project](https://project-instinct.github.io/hiking-in-the-wild/).
The source is the G1 parkour task and `sensors/noisy_camera`,
`sensors/grouped_ray_caster`, and depth observation/preprocessing code beneath
`source/instinctlab/instinctlab/`. The `hiking_walk_v1` data and AMP integration
also adapts this commit's G1 parkour motion configuration,
`motion_reference/utils.py` and `utils/math.py` for the official file selection,
bilinear interpolation and forward-difference convention, and
`tasks/parkour/config/parkour_env_cfg.py` for the 67-dimensional AMP observation
terms, ten-frame history and independent standing reset configuration.
These components are covered by that
repository's Creative Commons Attribution-NonCommercial 4.0 International
license, reproduced in
[licenses/instinctlab-CC-BY-NC-4.0.txt](licenses/instinctlab-CC-BY-NC-4.0.txt).
The local adaptation implements the camera and image-processing conventions
in the MuJoCo/PyTorch task, retains an internal validity packet for diagnostics,
and stores shared replay frames. The older `local_depth_valid_v1` protocol
remains available only with its recorded observation semantics. This camera
attribution is separate from the InstinctMJ latent files' MIT attribution above.
The source interpolation is ported to PyTorch with the author's exclusive
timeline, while torso-root reference poses are converted to the local
pelvis-root robot model. Fresh walking experiments retain the full official
reference rather than the historical local frame filter. The published source
now includes three byte-for-byte original files from the official public
Data & Model package under `src/assets/motions/g1/hiking_official/`: the complete
walking reference NPZ, its selection YAML and the upstream README. Their source
archive digest, original member paths and file hashes are recorded in that
directory's `SOURCE_MANIFEST.json`; `NOTICE.txt` describes the bundle. Author
policy checkpoints from the archive and generated caches are not bundled.
The public package contains no separate data-license text. The InstinctLab
code license above is not asserted as the license of these data files; this
attribution preserves provenance and does not assign additional data rights.

The `hiking_walk_v1` AMP discriminator loss, regularization, auxiliary reward
and AdamW update in `src/wmac_perceptive/runner.py` adapt
`instinct_rl/algorithms/wasabi.py` and `instinct_rl/algorithms/ppo.py` from
[project-instinct/instinct_rl](https://github.com/project-instinct/instinct_rl/tree/574c8d50cdd16618cf8f4e0df553f84759c1d837),
commit `574c8d50cdd16618cf8f4e0df553f84759c1d837`.
Copyright (c) 2026 Ziwen Zhuang. The complete upstream Modified MIT License,
including its additional attribution condition, is reproduced unchanged in
[licenses/instinct-rl-Modified-MIT.txt](licenses/instinct-rl-Modified-MIT.txt),
SHA256 `b586fa163d3daaee86f2e77a4e68db158ed336a1fe920a6634d211332134ce4b`.
The local adaptation trains the discriminator from WMAC replay and samples
complete ten-frame reference windows; its update budget remains the WMAC
budget rather than the upstream PPO rollout schedule. Historical checkpoint
protocols retain their earlier AMP behavior.

The unshod body model used by Hiking training version 3 includes the unchanged
`g1_29dof_torsobase_popsicle.urdf` and `head_link.STL` from the same
[InstinctLab G1 assets](https://github.com/project-instinct/InstinctLab/tree/de6b92f01eb155d51f30e8ed1ea7aecd2eb1f057/source/instinctlab/instinctlab/assets/resources/unitree_g1).
The local head mesh is named `hiking_head_link.STL`. The MuJoCo adapter
expresses the upstream rigid bodies in the pelvis-root GenCoord coordinates.

The optional shod robot of training version 4 (`--robot-variant shoe`) uses the
unchanged `g1_29dof_torsoBase_popsicle_with_shoe.urdf` from
`source/instinctlab/instinctlab/tasks/parkour/urdf/` of the same InstinctLab
commit, stored as `g1_29dof_torsobase_popsicle_with_shoe.urdf` under the same
CC-BY-NC-4.0 license.

Training version 4 detects terrain edges on Hiking's own terrain mesh.
`isaaclab_height_field_mesh` in `src/tasks/amp_perceptive/hiking_geometry.py` is
a port of `convert_height_field_to_mesh` from
[isaac-sim/IsaacLab](https://github.com/isaac-sim/IsaacLab/tree/f73c33173801f5f8afea4142482e47b7710c2b75),
`source/isaaclab/isaaclab/terrains/height_field/utils.py`, BSD-3-Clause
(`licenses/isaaclab-BSD-3-Clause.txt`). The unchanged source file is kept in
`reproduce/references/IsaacLab/` for the regression test. The mesh assembly
and merge steps follow InstinctLab's `terrains/terrain_importer.py` under the
InstinctLab license above.

The `hiking_full_v1` task port also adapts InstinctLab `de6b92f`'s complete
ordinary G1 task reward formulas, observation history/scales, delayed ideal-PD
parameters, command allocation, curriculum, terrain generation and reset
distribution. `src/tasks/amp_perceptive/hiking_terrain_source.py` and
`hiking_geometry.py` adapt its procedural terrain and virtual edge-cylinder
construction, including greedy edge merging. These adaptations retain the
InstinctLab attribution and CC-BY-NC-4.0 license above. The code is expressed
through MuJoCo and Torch, with physical-engine differences recorded in the
task audit. The unchanged GenCoord learner and InstinctMJ latent interfaces
retain their separate source attributions. Ordinary resume preserves the
historical task protocol rather than relabeling old policy weights.

The `g1_rpl_depth_v1` depth-camera adaptation uses the camera preset and
depth observation normalization from
[amazon-far/RPL](https://github.com/amazon-far/RPL/tree/1ca0250ba3745381cfbfb2b72ebcf55e995076da),
commit `1ca0250ba3745381cfbfb2b72ebcf55e995076da`, specifically
`rpl_training/config_values/camera.py` and `rpl_training/terms/observation.py`.
Its noise, terrain-biased random side masking, capture-delay ring, pose and
field-of-view randomization adapt the pinned Holosoma submodule,
[amazon-far/holosoma](https://github.com/amazon-far/holosoma/tree/91647eed1ba82db3b242cd1057035ab76b63c5c0),
commit `91647eed1ba82db3b242cd1057035ab76b63c5c0`, specifically
`src/holosoma/holosoma/managers/camera/terms/warp_depth.py`.
Both repositories are licensed under Apache-2.0; the complete standard license
is already included at
[licenses/supermimic-Apache-2.0.txt](licenses/supermimic-Apache-2.0.txt).
Both upstream `NOTICE` files state:

> Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

Local adapted components are `src/wmac_perceptive/rpl_depth.py`,
`rpl_camera_stream.py`, and the calibrated sensor integration. They retain the
original noise/dropout/occlusion order and numeric ranges while using a single
measured G1 body-mounted camera rather than the external front/back camera rig.
The measured intrinsic profile and raw-Z16 adapter are local data/integration;
they do not imply that RPL supplied G1 mounting calibration. The local GenCoord
world model and joint-stack latent prediction remain independently attributed
above; RPL's PPO/DAgger training architecture is not imported. The paper and
released code differences, including FOV amplitude and projection terminology,
are recorded in `docs/G1_RPL_CAMERA.html` and
`docs/validation/g1_rpl_camera_sources.json`.

The randomized Hiking camera protocol is implemented in
`src/wmac_perceptive/hiking_depth_randomization.py` and
`src/tasks/amp_perceptive/g1_hiking_dr_sensor.py`. It borrows the randomization
ideas and initial uncertainty ranges, while retaining Hiking's image geometry,
preprocessing and timing. ROI-relative occlusion widths and synthetic dropout
filled at the 2.5 m policy far endpoint are local adaptations; physical ray
misses retain Hiking's original fill. See `docs/HIKING_CAMERA_RANDOMIZATION.html`.

MIT License

Copyright (c) 2026 ETH Zurich, Chong Zhang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

The robot dynamics profile
`src/assets/robots/unitree_g1/unitree_mujoco_g1_29dof_dynamics.json` contains
numerical body masses, inertias, centres of mass and joint armature, damping and
friction loss extracted by `scripts/extract_robot_dynamics_profile.py` from
[unitreerobotics/unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco/tree/1eb6642e3f3fdfb7fb13a9794fd6a2dd93ea0e7d)
`unitree_robots/g1/g1_29dof.xml`, commit `1eb6642e3f3fdfb7fb13a9794fd6a2dd93ea0e7d`,
SHA256 `423e28bd718b19f7a65cda539b6f794ddbb268b4b9bdbd85f4bd982b30729617`.
No meshes are copied. Copyright (c) 2016-2024 HangZhou YuShu TECHNOLOGY CO.,LTD.
("Unitree Robotics"); its BSD-3-Clause license is reproduced in
[licenses/unitree_mujoco-BSD-3-Clause.txt](licenses/unitree_mujoco-BSD-3-Clause.txt).
