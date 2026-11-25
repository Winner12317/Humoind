# EngineAI PM01 in this repository

The PM01 trains with the same two pipelines as the G1: PPO + AMP (`scripts/train.py`)
and the world-model actor-critic (`scripts/train_wmac.py`). Its reference motions are
the 18 G1 clips retargeted to PM01 (`<workspace>/pm01_retarget`, 9,992 frames at
50 Hz, frame for frame the G1 set).

## Train

On a Linux GPU server, run `bash scripts/install_env.sh` from the SuperAMP directory.
Run the activation commands printed by the installer before starting training.

```bash
# PPO + AMP, flat ground (or EngineAI-PM01-AMP-Rough)
python scripts/train.py EngineAI-PM01-AMP-Flat --env.scene.num-envs=4096

# WMAC: the G1 launch command with the PM01 task added
bash scripts/train_smooth_weights_v3.sh \
  --task EngineAI-PM01-AMP-Flat-Aligned \
  --num-envs 4096 \
  --imagined-recovery-foot-slip learned_v1 \
  --discriminator-updates 2 --discriminator-batch-size 25600 \
  --run-name pm01_local_4096
```

`--task` is the only PM01-specific argument. The mirrored motion set is built on the
first WMAC run in `.wmac_cache/motions/pm01_mirrored` (the G1 cache is not touched).
`scripts/evaluate_wmac.py`, `scripts/play_wmac.py` and `scripts/play.py` take PM01
checkpoints as they take G1 ones; evaluation thresholds follow the robot.

### WMAC with the recovery-style fixes

`scripts/train_recovery_style_v1.sh` is the launch command above plus the opt-in fixes
for abrupt, off-reference get-ups (`src/wmac/recovery_style.py` describes each option),
and `src/assets/motions/pm01/amp_clean` is the PM01 clip set with the get-up clip
repaired by `scripts/clean_recovery_reference.py --robot pm01`.

```bash
# from scratch
bash scripts/train_recovery_style_v1.sh \
  --task EngineAI-PM01-AMP-Flat-Aligned \
  --reference-motion-root src/assets/motions/pm01/amp_clean \
  --num-envs 1024 \
  --iterations 400000 \
  --imagined-recovery-foot-slip learned_v1 \
  --discriminator-updates 2 --discriminator-batch-size 25600 \
  --run-name pm01_recovery_style

# continue a checkpoint that was trained without the fixes (its saved settings are
# restored; --num-envs etc. are not repeated)
bash scripts/train_recovery_style_v1.sh \
  --resume "$HOME/checkpoints/model_78400.pt" --allow-recovery-style-migration \
  --reference-motion-root src/assets/motions/pm01/amp_clean \
  --iterations 79600 --run-name pm01_recovery_style_fork
```

Both commands save under `logs/rsl_rl/pm01_amp_locomotion_aligned/wmac_<timestamp>_...`
when launched from the SuperAMP directory. To use a custom root, set `AMP_LOG_ROOT`
or pass `--log-root` on the command line. The resume example adds 1,200 iterations to `model_78400.pt` and restores its
environment count. The 1,024-environment setup was used for the PM01 pilot on an RTX 4080.

The mirrored set for the cleaned clips is built on the first run in
`.wmac_cache/motions/pm01_mirrored__amp_clean`. Training evaluations start from the
cleaned clips too; for a manual one pass
`scripts/evaluate_wmac.py --reference-motion-root src/assets/motions/pm01/amp_clean`.
A checkpoint trained this way resumes without any of these flags. PPO + AMP
(`scripts/train.py`) still reads `src/assets/motions/pm01/amp`.

What the cleaning changed in `Recovery/fallAndGetUp1_subject1.npz` (the walking clips
are byte-identical copies), measured on the training collision model:

| | `amp` | `amp_clean` |
|---|---|---|
| On-the-ground frames more than 1 cm in the floor | 2.4 % (max 7.4 cm) | 0 |
| Frames with links overlapping by more than 1 cm | 4.0 % (max 6.8 cm) | 0 |
| Fastest joint 20 ms after a reset from rest, worst frame | 6.8 rad/s | 3.0 rad/s |
| Joint speed p99 / max over the clip, rad/s | 6.51 / 23.9 | 6.51 / 24.0 |

26 % of the frames change; the largest joint change is 0.37 rad and the largest root
lift 6.4 cm.

Validation so far is a fine-tuning pilot, not a full run: `policy/wamp/model_78400.pt`
continued for 1,200 iterations (1,024 environments, one seed, RTX 4080), evaluated
with `evaluate_wmac.py` (128 starts) and `diagnose_recovery_style.py`.

| | Reference clip | Checkpoint | + unchanged | + fixes | + fixes and `amp_clean` |
|---|---|---|---|---|---|
| Get-up success (of 128) | — | 128 | 30 | 127 | 128 |
| Peak base vertical speed, m/s | 0.87 | 1.30 | 1.08 | 0.90 | 0.85 |
| Joint speed p95 / max, rad/s | 3.6 / 12.0 | 5.3 / 16.9 | 6.1 / 20.8 | 4.4 / 11.9 | 4.2 / 11.8 |
| Height gained in 0.62 s, imagined vs real, cm | — | 10.8 / 7.5 | 14.7 / 4.2 | 11.1 / 11.1 | 14.1 / 13.8 |
| Walking velocity error at 0.5 m/s, m/s | — | 0.26 | 0.27 | 0.32 | 0.24 |
| Seconds per iteration, quiet machine | — | — | 0.744 | — | 0.759 |

Every continued run starts from an empty replay buffer, which by itself cost the
unchanged run most of its get-up. The discriminator still tells the get-up from the
reference (score −0.63 against +0.46), so the tempo matches the reference and the
style does not yet. Of the options, the reset refresh keeps the get-up reliable and the
others set its tempo: refresh alone gave 128 / 128 at 1.64 m/s, the others without it
93 / 128 at 0.84 m/s.

| Task | Use | Actor input / actions | AMP features |
|---|---|---|---|
| `EngineAI-PM01-AMP-Flat` | PPO, flat | 312 (4 × 78) / 23 | 195, legacy |
| `EngineAI-PM01-AMP-Rough` | PPO, rough terrain | 1060 / 23 | 195, legacy |
| `EngineAI-PM01-AMP-Flat-Aligned` | WMAC and its PPO comparison arm | 312 / 23 | 203, grounded_v2 |

## Where things are

| Path | Content |
|---|---|
| `src/assets/robots/engineai_pm01/xmls/pm01.xml` | The robot model: EngineAI's official MJCF with the changes listed below. Generated; edit `scripts/pm01/build_pm01_mjcf.py`, not the file. |
| `src/assets/robots/engineai_pm01/pm01_constants.py`, `pm01_actuators.py` | Joint and body names, gains, limits, friction, default pose, heights, collision rules, action scale. |
| `src/assets/motions/pm01/amp/{WalkandRun,Recovery}/` | The reference clips in this repository's NPZ layout (23 joints, 29 bodies). |
| `src/tasks/amp_loco/config/pm01/` | The three tasks and their runner configuration. |
| `src/robot_profiles.py` | What the shared code needs to know per robot (names, sole point, heights, mirroring rules). |
| `scripts/pm01/` | `build_pm01_mjcf.py` (model), `convert_pm01_motions.py` (motions), `audit_pm01_collisions.py`, `check_pm01_env.py` (simulator checks). |
| `tests/test_pm01_*.py` | 51 CPU tests of the model, actuators, tasks, motions, mirroring and the WMAC adapter on PM01. |

Regenerating, should the retargeted clips or the official model change:

```bash
python scripts/pm01/build_pm01_mjcf.py <engineai_robotics_native_sdk>/assets/resource/robot/pm01_edu [--meshes]
python scripts/pm01/convert_pm01_motions.py            # reads ../pm01_retarget/pm01_npz_50hz
python -m unittest discover -s tests -p "test_pm01_*.py"
python scripts/pm01/check_pm01_env.py                  # needs the GPU
```

## Robot-specific choices

Sources: **SDK** = `engineai_robotics_native_sdk` (commit 6065bb2), file
`assets/resource/robot/pm01_edu/xml/serial_pm01_edu.xml`; **rl_lab** = `engineai_rl_lab`
(commit c53f7ec), `tasks/tracking/robots/pm01.py` unless another file is named.

| | G1 here | PM01 | Basis |
|---|---|---|---|
| Model | `g1.xml` | Official MJCF: same bodies, inertias, hinge axes, ranges and collision primitives; total 40.92 kg | SDK; the build script compares the result with the official model |
| Policy joints | 29 | 23, `J00`…`J22` in SDK order; `J23_HEAD_YAW` is welded | EngineAI's own AMP task drives these 23 (rl_lab `tasks/locomotion/amp/robots/pm01.py`); the head joint is zero in every clip |
| Gains | 10 Hz natural frequency on the rotor inertia, damping ratio 2 | Same rule: 178.8 / 11.39 on hip pitch, hip roll, knee (armature 0.0453); 26.45 / 1.684 elsewhere (0.0067); ankle damping 0.5 | rl_lab, which applies this rule to PM01 |
| Torque and speed limits | Unitree torque-speed curves | 164 N m, 26.3 rad/s (hip pitch, hip roll, knee); 52 N m, 35.2 rad/s (others). Full torque up to the speed limit, none above | rl_lab. EngineAI publishes no torque-speed curve |
| Joint friction | Unitree values per motor | 0.8 N m Coulomb and 0.6 N m s/rad viscous on every joint, in the actuator model | SDK (`<joint damping="0.6" frictionloss="0.8"/>`) |
| Action scale | 0.25 × effort / stiffness | Same rule: 0.229 rad (hip pitch, hip roll, knee), 0.491 rad (others) | rl_lab's `PM01_ACTION_SCALE` |
| Default pose | knees bent | hip pitch −0.06, knee 0.12, ankle pitch −0.06, shoulder roll ±0.15, elbow pitch −0.25 | rl_lab |
| Spawn / target standing height | 0.78 m | 0.82 m | Forward kinematics of the default pose: soles on the floor at 0.8193 m |
| Termination height | 0.50 m | 0.52 m | PM01's base walks 3.5 % higher than G1's pelvis in the references (lowest walk/run frame 0.689 m vs 0.664 m) |
| "Fallen" height (WMAC replay and expert split) | 0.55 m | 0.57 m | Same ratio |
| "Standing again" height (evaluation) | 0.65 m | 0.67 m | Same ratio |
| AMP bodies | pelvis; hip roll, knee, ankle roll; shoulder roll, elbow, wrist yaw (13) | `LINK_BASE`; `HIP_ROLL`, `KNEE_PITCH`, `ANKLE_ROLL`; `SHOULDER_ROLL`, `ELBOW_PITCH`, `ELBOW_YAW` (13) | One-to-one, so the feature widths (195 / 203) are unchanged |
| AMP anchor and velocity-tracking body | `torso_link` | `LINK_TORSO_YAW` | |
| Feet | 7 capsules per foot on the ankle-roll link | Official sole box and toe box per foot, on `LINK_FOOT_*` | SDK |
| Sole reference point (foot sites, grounded AMP features, slip reward) | (0.04, 0, −0.035) in the ankle-roll frame | (0.017, ±0.00166, −0.04631): centre of the underside of the sole box | Read from `pm01.xml` |
| IMU | site `imu_in_pelvis` | site `imu` at (0.02311, 0, −0.09631) in `LINK_BASE` | SDK |
| Self-collision | on, 4 body pairs excluded | on, 4 body pairs excluded: each hand against the thigh and hip-roll cylinders of its own side | With all joints at zero the official hand sphere sits 2.37 cm inside the thigh cylinder and 1.01 cm inside the hip-roll cylinder. Nothing else overlaps at zero or in the default pose (`audit_pm01_collisions.py`) |
| Left/right mirroring (WMAC) | exact to 10 µm | pitch joints keep their sign, roll and yaw joints flip. The official model is not exactly symmetric: the elbow-pitch axes differ from a mirror pair by 0.31°, and the waist yaw range is one-sided (−4.014 … 1.57 rad) | Measured and written to `mirror_manifest.json` (`robot_mapping.hinge_axis_asymmetry_deg` for every joint, `robot_mapping.asymmetric_joint_ranges`). Mirrored joints reproduce the mirrored bodies to 0.8 mm; each mirrored clip is checked against the real joint limits |

Everything else in the tasks (rewards and their weights, command ranges, pushes,
noise, delayed-reset recovery) is the G1 task unchanged.

## Checks

- `python -m unittest discover -s tests`: 420 tests pass, the 369 that existed before
  the port and 51 for PM01.
- `python scripts/pm01/check_pm01_env.py`: on all three tasks the AMP features computed
  in the simulator equal the expert features to 2e-5 (195 and 203 wide), the default
  pose carries the robot's weight on its soles, and random actions stay finite.
- G1 is unchanged: its mirrored motion set is byte-identical to the one built before
  the port, and its WMAC smoke run reproduces the earlier one to within the run-to-run
  spread of two identical runs.
- PPO, `EngineAI-PM01-AMP-Flat`, 4096 environments, 1,500 iterations, against the same
  run on `Unitree-G1-AMP-Flat` (run kept in
  `logs/rsl_rl/pm01_amp_locomotion/2026-11-21_12-58-41_first_check`):

  | | PM01 | G1 |
  |---|---|---|
  | Mean reward at iteration 250 / 500 / 1,500 | 3.1 / 18.3 / 27.2 | −1.6 / 14.0 / 28.2 |
  | Mean episode length at iteration 1,500 (limit 1,000) | 760 | 804 |
  | Walking at 0.5 m/s for 10 s, 128 starts: still up | 97.7 % | 100 % |
  | Same, planar velocity error | 0.31 m/s | 0.23 m/s |
  | Getting up from the floor, 128 starts | 3.9 % | 3.1 % |

  Neither robot gets up yet at this budget; the upstream README places that at around
  20,000 iterations. A full-length PM01 run has not been done.
- WMAC, `EngineAI-PM01-AMP-Flat-Aligned`, through `train_smooth_weights_v3.sh` with the
  production networks and `torch.compile`, 1024 environments, 400 iterations (272 s on
  an RTX 4080): world-model loss 11.9 → 0.096, held-out model error 8.7 → 0.31, no
  non-finite value; the evaluation that training starts every `--evaluation-interval`
  iterations ran at 200 and 400 (still up after 2 s of walking: 38 % → 97 % of 32
  starts). The geometric foot-slip mode (`geometry_instantamp_full_support_v1`) was run
  for three iterations only. The 4096-environment formal configuration was not run; it
  was validated for the G1 on a 48 GB card.

## What changed in shared code

The WMAC code named G1 bodies inline and assumed 29 joints in tensor slices. It now
takes names from `src/robot_profiles.py` and sizes from the joint count; with a G1
task every value is as before. Touched: `src/wmac/{state,networks,runner,replay,
smooth_rewards,reward_integration,foot_slip,mirroring,motions,diagnostics,config,
reference}.py`, `src/amp_features.py`, `rsl_rl/utils/motion_loader.py`,
`rsl_rl/runners/amp_on_policy_runner.py`, `scripts/train_wmac.py`,
`scripts/evaluate_wmac.py`, `src/assets/robots/__init__.py`.

These 18 files are hash-pinned in `docs/reproduction/source_manifest.json`, so
`docs/reproduction/verify_reproduction.py` now reports them as differing from the G1
reproduction snapshot.

## Not ported, and open points

- **G1-only tools.** `scripts/csv_to_npz.py`, `scripts/diagnose_wmac.py` and the
  analysis scripts (`analyze_*`, `audit_*`, `compare_*`, `probe_*`, `benchmark_*`,
  `calibrate_*`, `validate_*`) still assume the G1.
- **Rotor inertia of the low-torque module.** rl_lab gives 0.0067 kg m², the SDK MJCF
  0.039175 kg m² for the same joints (they agree on the high-torque module, 0.0453).
  The gains are derived from this number, so the choice matters; 0.0067 is used.
- **Effort limit of the low-torque module.** 52 N m (rl_lab tracking, used here),
  54.9 / 61 N m in rl_lab's AMP task, 61 N m in the SDK.
- **Actuator delay and control rate.** EngineAI's AMP task delays actions by 2 to 8
  physics steps and steps its policy at 100 Hz. The tasks here keep this repository's
  50 Hz policy on a 200 Hz simulation and have no delay model, as for the G1.
- **Smooth-cost coefficients** (`train_smooth_weights_v3.sh`) were tuned on G1
  magnitudes. PM01 is 23 % heavier with stronger motors, so its torque and power costs
  are larger for the same motion. They are passed unchanged.
- **Reference motion limits** carried over from the retargeting: leg-against-leg
  overlap in 24 frames of the right sidestep and 18 frames of the get-up clip.
