# PM01 sim-to-sim: SuperAMP policies in MuJoCo

Runs a PM01 policy in MuJoCo under EngineAI's own sim-to-sim conditions, with
keyboard control, plus a scripted test that reports how well it walks, gets up and
takes pushes. Five policies are exported and ready to run:

| Policy | Checkpoint | Run with |
|---|---|---|
| SuperMimic + AMP v1 (default) | `policy/wamp/v1_78400.pt` | `./run.sh` |
| SuperMimic + AMP v2 (recovery-style fixes) | `policy/wamp/v2_82400.pt` | `./run.sh --policy-dir policy_wamp_v2` |
| SuperMimic + AMP v3 (recovery style weight 1.0, 50,400 iterations) | `policy/wamp/v3_50400.pt` | `./run.sh --policy-dir policy_wamp_v3` |
| SuperMimic + AMP v3 (recovery style weight 1.0, 70,400 iterations) | `policy/wamp/v3_70400.pt` | `./run.sh --policy-dir policy_wamp_v3_70400` |
| PPO + AMP | `policy/ppo_amp/model_60000.pt` | `./run.sh --policy-dir policy_ppo_amp` |

```bash
cd "pm01_sim2sim"
./run.sh                        # viewer + keyboard, EngineAI SDK physics
./run.sh --start fallen         # start lying on the floor
./run.sh --dynamics training    # the training environment's joint/actuator model instead
./run.sh --headless             # scripted test, prints a report (about 10 s)
./run.sh --headless --record run.mp4 --json report.json
```

Every option combines with `--policy-dir`, for example
`./run.sh --policy-dir policy_wamp_v3 --start fallen`.

`run.sh` uses SuperAMP's conda environment (`~/miniconda3/envs/superamp`; override with
`AMP_CONDA_ROOT` / `AMP_CONDA_ENV`) and clears the ROS paths this machine's shell
exports. The runtime needs only `mujoco`, `onnxruntime` and `numpy` (`imageio` for
`--record`); it does not import SuperAMP.

For the selected 50,400-iteration PM01 policy, the native MNN export, download commands and the hardware controller interface are in the [PM01 deployment guide](PM01_DEPLOYMENT.md).

### Recovery starts

`--recovery-poses auto` uses an exported `start_poses.npz` if present, then the eight frames named by the policy's `start_poses.reference.json` in your local PM01 reference clip. The default motion directory is `SuperAMP/src/assets/motions/pm01` in this repository. Set `PM01_MOTION_ROOT` to use another directory with the same `amp/` and `amp_clean/` layout. Standing, walking and push-only tests can run without either source.

When the reference is unavailable, the runner generates eight simulator-only falls from the policy's default joint angles. It tilts the base by 80° around eight evenly spaced horizontal axes, disables actuation, and simulates 1.5 seconds per fall. The reset uses those resulting poses at rest. Select `--recovery-poses generic` to repeat this test even with reference data installed. Select `--recovery-poses reference` to require the historical reference starts. The terminal and JSON recovery reports identify the set as `generic_simulated` or `reference_motion`. All historical recovery tables and archived reports below use `reference_motion`.

```bash
./run.sh --policy-dir policy_wamp_v3 --headless --scenario recovery --recovery-poses generic
./run.sh --policy-dir policy_wamp_v3 --headless --scenario walk push
```

## Keys

Type in the terminal that runs `./run.sh`. The arrow keys also work in the viewer
window; letters do not, because the viewer uses every letter as a display shortcut.

| Key | Action |
|---|---|
| W / S, ↑ / ↓ | forward / backward speed ±0.1 m/s |
| A / D | sideways speed, left / right ±0.1 m/s |
| Q / E, ← / → | turn left / right ±0.1 rad/s |
| Space or X | stop (zero command) |
| R | reset standing in the default pose |
| F | reset lying on the floor (cycles through 8 poses from the get-up clip) |
| P | push: random root-velocity kick with the training push ranges |
| − / = | slower / faster playback (1/8× … 1×) |
| H | help |
| Ctrl+C, or close the viewer | quit |

Commands are limited to what the policy was trained on: vx −1 … 2 m/s, vy −1 … 1 m/s,
yaw rate −1 … 1 rad/s. The viewer shows the command and the measured torso velocity
(top left) and draws them as arrows at the feet: green commanded, blue measured.
Ctrl + right-drag on a body pulls it (MuJoCo's perturbation).

## What the checkpoints are

None of the checkpoints is an engineai_rl_lab checkpoint; all were written by SuperAMP,
with the PM01 port in `SuperAMP/`:

- `v1_78400.pt` (delivered as `model_78400.pt`): **SuperAMP's WMAC trainer**
  (SuperMimic-style world model + AMP; `format: amp-wmac-v1`), task
  `EngineAI-PM01-AMP-Flat-Aligned`, 78,400 iterations with 1,024 environments;
  `run_name: pm01_supermimic_amp_gpu1`.
- `v2_82400.pt`: the same trainer and task, trained from scratch with the
  recovery-style fixes (`SuperAMP/scripts/train_recovery_style_v1.sh`) on the cleaned
  clip set `src/assets/motions/pm01/amp_clean`: 82,400 iterations with 4,096
  environments; `run_name: pm01_recovery_style`. The clip set recorded in the
  checkpoint is the one in this workspace (same content hash).
- `v3_50400.pt`: the same trainer, task and clip set as v2, trained from scratch with
  v2's fixes plus `recovery_style_weight: 1.0` (the opt-in extra style weight on get-up
  transitions; `recovery_patience_reward: 2.0` is recorded but inactive, because
  `recovery_patience_seconds` is unset): 50,400 iterations with 4,096 environments;
  `run_name: pm01_recovery_style1_scratch`. Everything the controller uses (joint order,
  default pose, action scale, gains, observation layout, command ranges, lying poses) is
  identical to v2's.
- `model_60000.pt`: **SuperAMP's PPO + AMP runner** (its rsl_rl AMP runner: the file
  holds `model_state_dict`, `discriminator_state_dict`, `amp_feature_schema: legacy`
  and a pickled `rsl_rl.utils.utils.Normalizer` that only SuperAMP's rsl_rl defines),
  task `EngineAI-PM01-AMP-Flat`, 60,000 iterations.

The two tasks differ only in the AMP features the discriminator sees. What the
policy observes and outputs is identical, so all five run on the same controller. The
differences are inside the exported graph: the PPO actor is a 512-256-128 ELU network
behind rsl_rl's observation normaliser and has no action clamp; the WMAC actors are
2048-1024-512-256 with a ±20 clamp. The shared contract is SuperAMP's:

| | These policies (SuperAMP) | engineai_rl_lab AMP policy (SDK `rl_lab` controller) |
|---|---|---|
| Control rate | 50 Hz | 100 Hz |
| History | 4 frames, time-major: each frame `[gyro, gravity, command, q − q₀, q̇, last action]` (78), oldest first | 15 frames, term-major: all joint-position history, then velocities, actions, gyro, gravity; one command at the end |
| Gains | 178.8 / 11.4 (hip pitch, hip roll, knee), 26.5 / 1.68 (others), ankles 26.5 / 0.5 | 110 / 5 (hip pitch, knee), 70 / 3 (hip roll, yaw), 30 / 0.3 (ankles), 50 / 0.3 (arms), 50 / 3 (waist) |
| Action scale | 0.229 / 0.491 rad | 0.5 / 0.2 rad |

So EngineAI's C++ `rl_lab` runner in engineai_robotics_native_sdk cannot run these
policies without code changes. This package therefore reproduces that runner's
simulation conditions in Python, as described next.

## How it follows EngineAI's sim-to-sim setup

engineai_rl_lab's README delegates sim-to-sim to engineai_robotics_native_sdk
(`scripts/run_mujoco.sh` + the `rl_lab` controller + a virtual gamepad). Taken from
there (commit 6065bb2):

- **Robot model:** the official MJCF. `model/pm01_edu.xml`, `serial_pm01_edu.xml`,
  `serial_links.xml`, `serial_actuators.xml`, `serial_sensors.xml` and
  `environment/ground.xml` are byte-identical copies. Only `assets.xml` is new: the
  visual meshes are 9 MB decimated copies instead of the 300 MB textured originals.
  Nothing in it collides or has mass.
- **Physics:** 2 ms steps, implicitfast, ground friction 1.0, no self-collision
  exclusions.
- **Motors:** `τ = kp (q* − q) + kd (0 − q̇)` evaluated every physics step (as the
  SDK's `main.cc` does) and written to the official torque motors, which clamp it at
  ±164 / ±61 N·m. The official joint armature (0.045 / 0.039), damping (0.6) and
  friction loss (0.8) apply.
- **Head:** the head yaw joint, which the policy does not drive, is held by the SDK
  controller's head gains (kp 50, kd 0.3).
- **Observations:** read from the IMU sensors (`imu_quaternion`, `imu_angular_velocity`)
  after each physics step, as the SDK publishes them; joint state from the encoders.
- **Commands:** like the gamepad's sticks, except that the full trained range is
  available (the SDK scales vy by 0.1).

`--dynamics training` swaps in the training environment's dynamics on the same
geometry:
- 5 ms steps;
- armature 0.0453 / 0.0067;
- PD as an implicit position actuator;
- torque limits 164 / 52 N·m, with no torque above the motor speed limit;
- actuator friction `0.8 tanh(q̇/0.01) + 0.6 q̇`;
- head welded, foot friction 0.6, and the four hand–thigh exclusions.

## Validation: SuperMimic + AMP v1 (`v1_78400.pt`)

**Observation pipeline.** The runner and the training environment (mjlab) were put in
the same jogging state. Every observation term agrees to within 6e-8:
- angular velocity 6.0e-8;
- gravity 2.8e-8;
- command 0;
- joint positions 7.5e-9;
- joint velocities 0;
- last action 2.7e-8.

Run closed-loop from the same standing start (training physics, vx 0.6, yaw rate
0.3), the two walk 2.49 m and 2.44 m in 5 s along the same curve.

**Scripted test** (`results/`; `./run.sh --headless`), compared with the training
simulator:

| | SDK physics | training physics | training simulator (mjlab) |
|---|---|---|---|
| Walking script (58 s: forward 0.5–1.5, backward, sideways, turning, arc) | no fall | no fall | — |
| Forward 0.5 / 1.0 / 1.5 m/s → measured | 0.26 / 0.87 / 1.30 | 0.29 / 0.88 / 1.33 | 0.5 m/s: error 0.26; 1.0 m/s: error 0.21 |
| Backward 0.5 m/s → measured | 0.50 | 0.51 | — |
| Sideways ±0.4 m/s → measured | ≤ 0.02 | ≤ 0.08 | error 0.39 m/s |
| Turn in place ±0.8 rad/s → measured | 0.07 | 0.08–0.13 | error 0.64 rad/s |
| Get up from 8 lying poses (from rest, 8 s) | 7 / 8, 1.1 s on average | 7 / 8, 1.0 s | 7 poses 8/8, pose 3 3/8 |
| 10 training-strength pushes while walking at 0.5 m/s | 0 falls | 0 falls | — |
| Torque at its limit (share of control steps) | 0 % | ≤ 1 % | — |

The training-simulator column uses SuperAMP's `evaluate_wmac.py` with 64 environments
per command, and 8 environments per lying pose started from rest. With its own
protocol (128 starts, from moving reference frames) mjlab gets up 128 / 128 times, in
1.6 s on average.

The SDK physics barely changes the behaviour; the policy transfers. Several
weaknesses are present in all three columns, so they belong to the checkpoint, not
to the sim-to-sim gap:
- **Sideways and turning:** it hardly moves sideways. Commanded to turn in place at
  0.8 rad/s it turns at 0.07–0.13 rad/s (about 0.16 in the training simulator).
  Turning while walking works partly: the arc reached 0.24 of 0.5 rad/s.
- **Slow walking:** at 0.5 m/s it walks about half as fast as commanded; it tracks
  better at 1.0 m/s and above.
- **Lying pose 3:** a mid-roll frame with the legs in the air. From rest, the robot
  rolls onto its back and stays there with one arm raised. The training simulator
  does the same in 5 of 8 runs.

## Validation: SuperMimic + AMP v2 (`v2_82400.pt`)

**Export.** The ONNX graph matches the PyTorch actor to 5.2e-6 on 4,096 observations.
In the same jogging state the runner's observation agrees with the training
environment's to within 6e-8 per term; from the same standing start (training
physics, vx 0.6, yaw rate 0.3) the two walk 2.22 m and 2.04 m in 5 s.

**Scripted test** (`results/wamp_v2_*`; `./run.sh --headless --policy-dir policy_wamp_v2`):

| | SDK physics | training physics | training simulator (mjlab) |
|---|---|---|---|
| Walking script (58 s) | no fall | no fall | — |
| Forward 0.5 / 1.0 / 1.5 m/s → measured | 0.18 / 0.90 / 1.25 | 0.20 / 0.90 / 1.23 | error 0.31 / 0.15 / 0.31 m/s |
| Backward 0.5 m/s → measured | 0.46 | 0.47 | error 0.14 m/s |
| Sideways ±0.4 m/s → measured | ≤ 0.02 | ≤ 0.01 | error 0.40 m/s |
| Turn in place ±0.8 rad/s → measured | ≤ 0.03 | ≤ 0.02 | error 0.76 rad/s |
| Arc 0.6 m/s, 0.5 rad/s → measured | 0.51 m/s, 0.27 rad/s | 0.54 m/s, 0.29 rad/s | — |
| Get up from 8 lying poses (from rest, 8 s) | 8 / 8, 2.2 s on average | 8 / 8, 2.0 s | all 8 poses 8/8 |
| 10 training-strength pushes while walking at 0.5 m/s | 0 falls | 0 falls | — |
| Torque at its limit (share of control steps) | 0 % | ≤ 5.6 % (at 1.5 m/s) | — |

The training-simulator column is measured as for the other policies (64 environments
per command, 8 per lying pose). With its own protocol (128 starts from moving frames
of the cleaned clip) mjlab gets up 128 / 128 times, in 2.0 s on average.

The policy transfers: SDK physics and training physics give the same behaviour.

**Get-up tempo.** This is what the recovery-style fixes were for, and it carries over
to SDK physics. Measured from the lying start until the robot first stands, over the
poses it gets up from (`tempo until standing` in the report):

| SDK physics | Reference clip | v1 | v2 | v3 | PPO + AMP |
|---|---|---|---|---|---|
| Time to stand, average | — | 1.1 s (7 of 8 poses) | 2.2 s | 2.6 s | 1.1 s |
| Peak upward base speed, average over poses (highest) | 0.87 m/s | 1.41 (1.94) | 0.91 (1.12) | 0.95 (1.18) | 1.97 (2.35) |
| Joint speed, 95th percentile / highest | 3.6 / 12.0 rad/s | 7.3 / 21.0 | 3.2 / 12.5 | 4.0 / 14.9 | 7.5 / 22.5 |

The reference column is the get-up part of the PM01 clip, measured by
`SuperAMP/scripts/diagnose_recovery_style.py`.

Compared with v1:
- **Getting up:** all 8 lying poses now work, including pose 3 (the mid-roll frame v1
  fails from), at about the reference clip's speeds and in twice the time.
- **Walking:** no falls, as before. Backward and 1.0 m/s tracking are about the same;
  at 0.5 m/s it is slower than v1 (0.18–0.20 against 0.26–0.29 m/s).
- **Sideways and turning in place:** still absent, slightly more so than v1 (turning in
  place 0.01–0.03 rad/s against 0.07–0.13). Turning while walking works partly (the
  arc reached 0.27–0.29 of 0.5 rad/s).
- **Poses 1 and 8** are the cleaned clip's versions of the same frames (joints moved by
  up to 0.11 rad so that nothing starts inside the floor); poses 2–7 are identical.

## Validation: SuperMimic + AMP v3 (`v3_50400.pt`)

**Export.** The ONNX graph matches the PyTorch actor to 5.2e-6 on 4,096 observations.
The exported `policy.json` equals v2's except for the provenance fields, and the runner
is unchanged, so v2's observation-pipeline check (6e-8 per term against mjlab) applies.

**Scripted test** (`results/wamp_v3_*`; `./run.sh --headless --policy-dir policy_wamp_v3`).
The training-simulator (mjlab) column was not measured for v3.

| | SDK physics | training physics |
|---|---|---|
| Walking script (58 s) | no fall | no fall |
| Forward 0.5 / 1.0 / 1.5 m/s → measured | 0.25 / 0.86 / 1.21 | 0.29 / 0.88 / 1.22 |
| Yaw drift while walking forward 0.5 / 1.0 m/s | −0.13 / −0.12 rad/s | +0.04 / −0.11 rad/s |
| Backward 0.5 m/s → measured | 0.49 | 0.48 |
| Sideways ±0.4 m/s → measured | ≤ 0.01 | ≤ 0.00 |
| Turn in place ±0.8 rad/s → measured | ≤ 0.04 | 0.08–0.11 |
| Arc 0.6 m/s, 0.5 rad/s → measured | 0.32 m/s, 0.16 rad/s | 0.34 m/s, 0.31 rad/s |
| Get up from 8 lying poses (from rest, 8 s) | 8 / 8, 2.6 s on average (max 3.4) | 8 / 8, 2.6 s (max 4.4) |
| 10 training-strength pushes while walking at 0.5 m/s | 0 falls | 0 falls |
| Torque at its limit (share of control steps) | 0 % | ≤ 8.0 % (at 1.5 m/s) |

The policy transfers: SDK physics and training physics give the same behaviour.

Compared with v2 (one scripted run each, so differences of a few hundredths are noise):
- **Getting up:** still 8 / 8, about 0.4 s slower on average, with slightly faster
  peaks (see the tempo table above).
- **Walking:** no falls. Better at 0.5 m/s (0.25–0.29 against 0.18–0.20 m/s); backward
  tracking is about the same.
- **New:** at 1.0 m/s forward it veers right at about 0.12 rad/s under both physics
  models, and on the arc it walks at 0.32–0.34 instead of 0.51–0.54 m/s.
- **Sideways and turning in place:** still essentially absent.

## Validation: PPO + AMP (`model_60000.pt`)

**Export.** The ONNX graph matches the PyTorch actor to 3.1e-5 on 4,096 observations,
and the actor matches one rebuilt from the checkpoint with rsl_rl's own normaliser
exactly. In the same jogging state the runner's observation agrees with the training
environment's to within 6e-8 per term; from the same standing start (training
physics, vx 0.6, yaw rate 0.3) the two walk 2.04 m and 1.88 m in 5 s.

**Scripted test** (`results/ppo_amp_*`; `./run.sh --headless --policy-dir policy_ppo_amp`):

| | SDK physics | training physics | training simulator (mjlab) |
|---|---|---|---|
| Walking script (58 s) | no fall | no fall | — |
| Forward 0.5 / 1.0 / 1.5 m/s → measured | 0.06 / 0.93 / 1.32 | 0.13 / 0.95 / 1.32 | error 0.35 / 0.20 / 0.30 m/s |
| Backward 0.5 m/s → measured | 0.24 | 0.34 | error 0.30 m/s |
| Sideways ±0.4 m/s → measured | 0.00 | 0.01 | error 0.40 m/s |
| Turn in place ±0.8 rad/s → measured | 0.02 | ≤ 0.04 | error 0.76 rad/s |
| Arc 0.6 m/s, 0.5 rad/s → measured | 0.51 m/s, 0.21 rad/s | 0.52 m/s, 0.24 rad/s | — |
| Get up from 8 lying poses (from rest, 8 s) | 8 / 8, 1.1 s on average | 8 / 8, 1.1 s | 7 poses 8/8, pose 8 5/8 |
| 10 training-strength pushes while walking at 0.5 m/s | 0 falls | 0 falls | — |
| Torque at its limit (share of control steps) | 0 % | ≤ 1.6 % | — |

The training-simulator column is measured as for the other policy (64 environments
per command, 8 per lying pose). With its own protocol (128 starts from moving
reference frames) mjlab gets up 127 / 128 times, in 1.4 s on average.

The policy transfers: SDK physics and training physics give the same behaviour, and
its weaknesses also show in the training simulator, so they belong to the checkpoint:
- **Slow walking:** commanded 0.5 m/s forward it almost stands still (0.06–0.13 m/s);
  from 1.0 m/s up it tracks well.
- **Sideways and turning in place:** it does not move. Turning while walking works
  partly (the arc reached 0.21–0.24 of 0.5 rad/s).
- Unlike the SuperMimic + AMP policy it gets up from lying pose 3 (the mid-roll frame).

## Files

| Path | Content |
|---|---|
| `sim2sim.py` | The runner: model build, observation history, ONNX inference, PD loop, viewer, keyboard, scripted test. |
| `export_policy.py` | Checkpoint → `policy/` (WMAC) or `policy_ppo_amp/` (PPO + AMP), or `--out <dir>`; run with SuperAMP present: `./run.sh export [--checkpoint …] [--out …]`. Checks the ONNX output against PyTorch (max error 4.9e-6 for v1, 5.2e-6 for v2 and v3, 3.1e-5 for PPO + AMP). |
| `policy/policy.onnx` | SuperMimic + AMP v1: observation normaliser + MLP + the ±20 action clamp; input `obs` (N, 312), output `actions` (N, 23). |
| `policy_wamp_v2/` | The same three files for SuperMimic + AMP v2. |
| `policy_wamp_v3/` | The same three files for SuperMimic + AMP v3. |
| `policy_ppo_amp/` | The same three files for the PPO + AMP policy (no action clamp). |
| `policy/policy.json` | Joint order, default pose, action scale, PD gains, motor limits, observation layout, command ranges, height thresholds, push ranges; source checkpoint and hashes. |
| `policy/start_poses.npz` | The 8 lying poses (frames 57 … 2381 of the PM01 get-up clip). Every policy gets the same frames, read from the clip set it was trained with (v2: the cleaned set). |
| `model/` | Official PM01 MJCF and ground (see above), decimated meshes, EngineAI's BSD-3 licence. |
| `results/` | Reports of `--headless` under both physics settings, and a video of the SDK-physics run, for each policy (`wamp_v2_*` for v2, `wamp_v3_*` for v3, `ppo_amp_*` for PPO + AMP). |

For another SuperAMP PM01 checkpoint of either trainer: `./run.sh export --checkpoint <path> [--out <dir>]`,
then `./run.sh --policy-dir <dir>`.
