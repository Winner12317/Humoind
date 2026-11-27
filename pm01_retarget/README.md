# SuperAMP G1 reference motions retargeted to EngineAI PM01

The 18 Unitree G1 reference motions used by `SuperAMP` (17 walk/run clips and one
fall-and-get-up recovery clip), retargeted onto EngineAI PM01 with the
robot-to-robot pipeline of [Human-Humanoid-Tools](../Human-Humanoid-Tools) (hhtools).

## What is here

| Path | Content |
|------|---------|
| `pm01_csv/<group>/<clip>.csv` | **Final clips.** Headerless rows of `x y z qx qy qz qw` + 24 joint angles. 120 Hz for `WalkandRun`, 50 Hz for `Recovery` (the rates of the G1 sources). |
| `pm01_npz_50hz/<group>/<clip>.npz` | The same clips resampled to 50 Hz: `fps`, `root_pos` (T, 3), `root_rot` (T, 4, xyzw), `dof_pos` (T, 24), `dof_names`. Frame count and timing match the G1 NPZ of the same name in `SuperAMP/src/assets/motions/g1/amp/`. |
| `pm01_hhtools/<group>/<clip>.csv` | Unmodified hhtools export (`# key: value` comments, then `time,root_*,dof_<joint>` columns). Kept for comparison: it is what the final clips looked like before the contact refinement. |
| `pm01_refined/<group>/<clip>.csv` | Every clip after `scripts/refine_motions.py` (see Contact refinement), same layout as `pm01_hhtools`. `pm01_csv` and `pm01_npz_50hz` are made from these. |
| `models/pm01_sdk_collision.xml` | PM01's official collision model: the collision primitives of EngineAI's MJCF, without meshes. All contact checks and the refinement use it. |
| `g1_source/<group>/<clip>.csv` | The G1 inputs, rewritten in hhtools' trajectory format. |
| `qa/videos/*.mp4` | G1 source (top) and PM01 result (bottom), side and front view, for every clip. |
| `qa/validation.json`, `qa/validation_50hz.json` | Per-clip comparison of the final clips against the G1 source (`scripts/validate.py`), at the source rate and for the 50 Hz NPZ set. |
| `qa/refine/<clip>.json` | What the refinement changed in each clip, before and after. |
| `qa/hhtools_diagnostics/` | hhtools' own tracking and contact diagnostics for every clip. |
| `qa/ground_shift.json` | Which input each final clip was made from and the vertical shift applied to it. |
| `qa/calibration_*.png`, `qa/calibration_report.json` | The G1/PM01 calibration pose: hhtools' proposal and the one used. |
| `scripts/` | Everything needed to regenerate the above, and `review_motions.py`, a browser viewer for stepping through the clips. |

Conventions in every PM01 file: metres, radians, Z up; the root is `LINK_BASE`;
the root quaternion is `x y z w`; joints are in EngineAI's SDK order
`J00_HIP_PITCH_L … J22_ELBOW_YAW_R, J23_HEAD_YAW` (`joint_sdk_names` in
`engineai_rl_lab`). `J23_HEAD_YAW` is 0 throughout, because G1 has no head joint.

**SuperAMP** trains PM01 on these clips. Its copy, in its own NPZ layout (body poses
and velocities of the PM01 model in `SuperAMP/src/assets/robots/engineai_pm01`, 23
joints with the welded head dropped), is `SuperAMP/src/assets/motions/pm01/amp/` and is
made from `pm01_npz_50hz` by `SuperAMP/scripts/pm01/convert_pm01_motions.py`; see
`SuperAMP/docs/pm01/README.md`. SuperAMP's own `csv_to_npz.py` builds a G1 scene and
does not read PM01 files.

For **engineai_rl_lab**: the headerless CSVs have the layout of
`engineai_rl_lab/datasets/tracking/pm01/*.csv`, so its `csv_to_npz.py` reads them when
given the input rate (120 or 50). The NPZs have the layout its AMP loader reads
(`datasets/amp/pm01/data`). That loader takes consecutive frames as its history and
the AMP task there steps at 100 Hz, so for that task regenerate the set with
`python export_formats.py --npz-fps 100`.

## Robot model and calibration

The PM01 preset is `Human-Humanoid-Tools/configs/robots/pm01/`. Its URDF is
`serial_pm01_edu.urdf` from `engineai_robotics_native_sdk`, the file the PM01 USD in
`engineai_rl_lab` was converted from. Provenance and every edit made to the
auto-generated preset are listed at the top of its `robot.yaml`.

`r2r_calibration_g1_29dof.yaml` in the same folder is the PM01 pose that corresponds
to G1's zero configuration. hhtools applies G1's motion relative to that pose, so a
misalignment there appears in every frame. hhtools' own proposal leaves the arms
16–19° off; the saved pose (`scripts/calibrate_pair.py`) aligns the upper-arm and
forearm axes of the two robots exactly.

## Differences from a stock hhtools run

1. **Driver.** `Human-Humanoid-Tools/scripts/batch_r2r_retarget.py` fails at export at
   the checked-out commit (13b6dd3). `scripts/run_retarget.py` calls the same library
   functions in the order hhtools' Web/Agent runtime does. The clone is not patched.
2. **Stride scale.** hhtools moves the pelvis by the stature ratio of the two robots
   (1.059) but places the feet with the leg ratio (1.025), so planted feet drift.
   The pelvis travel is scaled by the leg ratio instead: PM01 covers 1.025 × the
   distance G1 does.
3. **Joint speed cap.** Raised from hhtools' 8 rad/s to 26.3 rad/s, PM01's slowest
   actuator limit. The G1 jog clips move the knee at up to 18 rad/s.
4. **Grounding.** hhtools puts the single lowest point a sole reaches in a clip on
   the floor, which leaves the feet hovering 1–3 cm during flat stance. The contact
   refinement below replaces that. (`export_formats.py` still has the earlier
   remedy, one constant vertical shift per `WalkandRun` clip, for a clip that has
   no refined version; no delivered clip uses it.)
5. **Contact refinement.** Every clip is repaired in a second pass, described next.

## Contact refinement

`scripts/refine_motions.py` post-processes the hhtools result of every clip against
PM01's official collision model. That model's feet are a 20 cm × 11 cm sole
box plus a toe box, longer and wider than G1's feet and than the URDF foot shell, so
a pose copied from G1 can put a toe under the floor or one foot inside the other.
The pass does, in order:

1. maps the elbow pitch smoothly into the soft joint range (centre ± 0.9 × half
   range, `engineai_rl_lab`'s soft limit) instead of leaving it on the hard stop;
2. rebuilds the pelvis height from G1's (G1's pelvis height × the leg ratio + one
   constant per clip), because hhtools lifts the whole robot when a source toe is
   below the floor; while G1 lies or kneels, the height is the one that puts PM01's
   lowest point where G1's is;
3. gives each PM01 sole the height G1's sole has at that instant, never below the
   floor; where the tilted sole would reach lower, the foot's pitch and roll are
   reduced first (which leaves the leg alone) and the foot is raised for the rest;
4. solves one horizontal correction per foot and frame for the whole clip (a QP):
   while G1's foot is planted the point of PM01's sole on the floor must not move,
   corrections blend smoothly through the swing, and the two feet's collision boxes
   stay 1 cm apart;
5. moves the feet there with leg inverse kinematics, dropping the pelvis where a
   planted foot would be out of reach;
6. moves the arm joints the least amount that takes forearms and hands out of the
   torso, head and each other.

The pelvis path and orientation, the waist and the head are untouched.

Over the 17 walk/run clips (7,417 frames of the 50 Hz NPZ set, official collision
model), from the plain hhtools result with one height shift per clip to the refined
clips, next to the G1 source:

| | hhtools + shift | refined | G1 |
|---|---|---|---|
| Sole slide in stance, mean / p95 | 2.3 / 8.8 cm/s | 0.3 / 0.6 cm/s | 1.4 / 5.0 cm/s |
| Planted-foot samples > 2 cm above the floor | 4.5 % | 0.0 % | 0.0 % |
| Frames with a sole > 1 cm under the floor | 17.8 % | 0.0 % | 22.2 % |
| Frames with the feet > 1 cm into each other | 63 | 0 | 1 |
| Frames with any self-intersection > 1 cm | 18.2 % | 17.1 % | 43.5 % |
| Thigh / shank / foot rotation error vs G1, mean | 2.4° / 1.8° / 0.2° | 3.3° / 3.5° / 2.2° | – |

The sole slide is the motion of the sole point that is on the floor, counted while
G1's foot is planted. (The speed of the foot frame, which also counts the foot
rolling over its toe or heel, goes from 2.8 to 1.9 cm/s; G1: 2.0.) The G1 column of
the first row is from the motion review that prompted the refinement
(`qa/pm01_vs_g1_50hz.csv`).

The four clips that review singled out, before → after (G1 in brackets):

| | left-turn jog | right sidestep | left-arc walk | fall and get up |
|---|---|---|---|---|
| Frames with a sole > 1 cm under the floor | 20.6 % → 0 % (23.8 %) | 6.5 % → 0 % (8.1 %) | 45.9 % → 0 % (33.5 %) | 0.2 % → 0 % (5.4 %) |
| Deepest sole penetration | 3.15 → 0.28 cm | 1.36 → 0.43 cm | 3.42 → 0.03 cm | 1.28 → 1.00 cm |
| Frames with the feet > 1 cm into each other | 0 → 0 | 54 → 0 | 0 → 0 | 0 → 0 |
| Sole slide in stance | 7.9 → 0.9 cm/s (3.3) | 3.3 → 0.2 cm/s (1.7) | 4.5 → 0.2 cm/s (2.2) | 6.5 → 4.1 cm/s (5.8) |
| Planted-foot samples > 2 cm above the floor | 0 % → 0 % | 0.3 % → 0 % | 12.0 % → 0 % | upright frames with both feet > 2 cm up: 93 % → 0.3 % |
| Left elbow within 0.01 rad of its stop | 31 % → 0 % | – | – | 5 % → 0 % |
| Frames with forearm or hand > 1 cm inside torso, head or the other arm | 6 → 0 | 0 → 0 | 0 → 0 | 145 → 0 |
| Thigh / shank / foot rotation error vs G1, mean | 3.7° / 4.1° / 0.5° → 4.2° / 5.2° / 2.5° | 2.2° / 1.5° / 0.3° → 3.2° / 3.0° / 2.3° | 2.9° / 2.2° / 0.2° → 5.0° / 4.4° / 4.6° | 4.1° / 4.4° / 0.9° → 4.1° / 4.8° / 1.7° |

What the repair costs:

- The legs follow G1 less closely (last rows). G1's own soles are more than 1 cm
  below the floor in 22 % of the walk/run frames, and where G1 sinks PM01 has to
  flatten its foot or bend its leg. `--keep-foot-tilt` keeps G1's foot angle and
  bends the knee instead (larger thigh and shank error).
- The pelvis drops where a planted foot would otherwise be out of reach: up to
  4.4 cm in `walk_sideway_right_loop_001__A022` (the cross-behind step, which PM01's
  longer feet have to take wider), 3.1 cm in `jog_forward_loop_003__A021`, 2.6 cm or
  less elsewhere.
- Joint accelerations rise: the 99th percentile is 1.25 times hhtools' in the median
  clip and 1.9 times in the worst (`walk_forward_loop_002__A022`). Joint speeds stay
  below 0.75 of the actuator limits.

`python refine_motions.py --clip NAME` refines single clips; without it every clip
is done (about 75 s).

## Known limitations

- PM01's waist only yaws. G1 leans its torso forward about 10° at the waist in most
  clips; PM01 follows G1's pelvis and keeps its torso upright. The arms keep G1's
  joint motion relative to the torso.
- PM01's ankle roll is limited to ±15° and so is G1's, and the G1 data sits on that
  limit in many clips; so does the result (about 4 % of all frames per ankle).
- PM01's elbow cannot flex as far as G1's. In the refined clips it stops 9.6° short
  of its hard stop (the soft range), so in `arc_jog_left_loop_002__A029` the forearm
  is up to 9.6° more open than hhtools had it, and more than that against G1.
- The refinement keeps the two feet apart, not the rest of the legs. In
  `walk_sideway_right_loop_001__A022` the shin collision boxes overlap by more than
  1 cm in 19 of 630 frames (50 Hz) during the cross-over steps, and a foot and the
  other shin in 5 (deepest 5.1 cm). `fallAndGetUp1_subject1` has 18 such frames
  while the robot is on the floor. No other clip has any.
- In `fallAndGetUp1_subject1` the feet are grounded while the robot stands on them.
  While it lies or kneels only the pelvis height is corrected; contacts of knees,
  hands and feet in those phases are as hhtools left them.
- The hand collision spheres of the official model overlap the hip cylinders by up
  to 2.4 cm with the arms hanging straight down, so "self-intersection" counts
  include hand-hip pairs that are part of the model, not of the motion.

## Reviewing the motions

`scripts/review_motions.py` opens a viewer in the browser (default
`http://127.0.0.1:8765`, this machine only):

```bash
cd "pm01_retarget/scripts"
env -u PYTHONPATH -u LD_LIBRARY_PATH -u VIRTUAL_ENV -u CONDA_PREFIX PYTHONNOUSERSITE=1 \
    ../../Human-Humanoid-Tools/.venv/bin/python review_motions.py      # --port N, --no-browser
```

Pick a motion set (`pm01_csv`, `pm01_npz_50hz`, `pm01_hhtools`, `pm01_refined`) and a clip, then
step with <kbd>←</kbd> <kbd>→</kbd> (<kbd>Shift</kbd>: 10 frames), <kbd>Home</kbd> /
<kbd>End</kbd>, <kbd>Space</kbd> to play, <kbd>[</kbd> <kbd>]</kbd> for the previous /
next clip. PM01 (blue) is drawn with the G1 source frame it was retargeted from
(orange): beside it, as a translucent overlay, at G1's own positions, or hidden.
The panel on the right gives the numbers of `validate.py` for the frame on screen:
segment rotation error, sole heights and foot speed, ground and self penetration,
and every joint against its limits. The chart under the view plots one of these
over the whole clip; click it to jump to a frame, or use *Go to worst frame*.
The address bar keeps the clip, frame and view, so a frame can be bookmarked.

Poses and numbers are computed by MuJoCo on the models `validate.py` uses; PM01's
collision shapes are those of the official model, its visual shapes the URDF meshes.
To see what the refinement changed, compare a clip in `pm01_hhtools` and `pm01_csv`. In *Beside* and
*Overlay* G1 is moved horizontally to stay with PM01, which travels 2.5 % further;
*Own path* shows G1's stored positions. three.js (MIT) is vendored in
`scripts/review_viewer/vendor/`, so the viewer needs no network access.

## Regenerating

All scripts run with the hhtools environment. This machine's shell exports ROS
paths that break it, so clear them:

```bash
cd "pm01_retarget/scripts"
PY="env -u PYTHONPATH -u LD_LIBRARY_PATH -u VIRTUAL_ENV -u CONDA_PREFIX PYTHONNOUSERSITE=1 ../../Human-Humanoid-Tools/.venv/bin/python"

$PY check_g1_source.py     # G1 data conventions and hhtools' G1 model vs SuperAMP's
$PY prepare_g1_sources.py  # SuperAMP G1 clips -> g1_source/
$PY calibrate_pair.py      # G1 <-> PM01 calibration pose
$PY run_retarget.py        # hhtools retarget -> pm01_hhtools/   (about 4 minutes)
$PY refine_motions.py      # contact refinement of every clip -> pm01_refined/, qa/refine/   (about 75 s)
$PY export_formats.py      # -> pm01_csv/, pm01_npz_50hz/
$PY validate.py            # -> qa/validation.json
$PY validate.py --pm01 ../pm01_npz_50hz --tag validation_50hz
```

After `export_formats.py`, refresh SuperAMP's copy of the clips with
`python scripts/pm01/convert_pm01_motions.py` in `SuperAMP` (its own environment).

`models/pm01_sdk_collision.xml` is written by
`make_collision_model.py <engineai_robotics_native_sdk>/assets/resource/robot/pm01_edu/xml/serial_pm01_edu.xml`.

The PM01 preset must exist first; it was created with
`hhtools robot add <dir with serial_pm01_edu.urdf and meshes/> --name pm01` and then
edited as described in its `robot.yaml`. The G1 preset comes from
`python scripts/install_builtin_robots.py --only g1_29dof` in the hhtools clone.

## Motion credits and restoration

Motion Data by [Bones Studio](https://bones.studio/). The published walk/run videos show the G1 source and PM01 retarget side by side. The recovery motion comes from Ubisoft La Forge's LAFAN1, retargeted for G1 by Unitree Robotics. The [motion source guide](../SuperAMP/docs/pm01/MOTION_SOURCES.md) lists the licences, original input layout and checksums for restoring or regenerating the trajectory files locally.
