# Motion sources and local restoration

The 17 walk and run clips are BONES-SEED motions. Training data includes Motion Data by [Bones Studio](https://bones.studio/). Use of the underlying dataset is subject to the [BONES Motion Capture Dataset License Agreement](https://bones.studio/info/seed-license). The [publisher's dataset](https://huggingface.co/datasets/bones-studio/seed) provides the G1 motion files. The local clip identifiers and file hashes are in [LOCAL_MOTION_SHA256SUMS](LOCAL_MOTION_SHA256SUMS).

The recovery clip `fallAndGetUp1_subject1` is from Ubisoft La Forge's LAFAN1. The local excerpt begins at frame 407 (zero-based) of Unitree's 30 Hz G1 retarget and is resampled to 50 Hz. The source retarget is credited to Unitree Robotics. Ubisoft publishes the underlying motion under CC BY-NC-ND 4.0, which restricts public sharing of adaptations. [Ubisoft LAFAN1 and data licence](https://github.com/ubisoft/ubisoft-laforge-animation-dataset/blob/master/license.txt) · [Unitree's retarget and credits](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset).

## Restore the training inputs locally

An existing authorized copy of the prepared motion inputs gives byte-for-byte reproducibility. Place the following directories under `SuperAMP/`, retaining the filenames, then check the hashes. The directory and filenames also appear in the checksum list.

- `motion_data_csv/amp/` contains the 17 headerless G1 walk/run CSV files at 120 Hz.
- `src/assets/motions/g1/amp/` contains `WalkandRun/` and `Recovery/` with the 50 Hz G1 motion NPZ files.

The following checks the original inputs. Run it from the Humoind repository root on Linux.

```bash
grep -E '  SuperAMP/(motion_data_csv/amp/|src/assets/motions/g1/amp/)' \
  SuperAMP/docs/pm01/LOCAL_MOTION_SHA256SUMS | sha256sum -c -
```

To reproduce the optional cleaned G1 clip set, run `python scripts/clean_recovery_reference.py --robot g1` in the SuperAMP environment from the `SuperAMP/` directory. The cleaning tool expects a new output directory. The [cleaning report](g1_clean_manifest.json) records the inputs, outputs and settings of the original run.

For the intermediate G1 and PM01 trajectories, follow [Regenerating](../../../pm01_retarget/README.md#regenerating) in the retarget guide. It covers preparation of the G1 source, calibration, PM01 retargeting, contact refinement and CSV/NPZ export. The checksum list also gives the exact names and hashes for restoring these intermediates from your own copy. The public QA directory contains calibration results, error statistics and rendered walk/run comparisons.

## Rebuild locally from the original publishers

Use the G1 export of the 17 named BONES-SEED clips from your authorized download. SuperAMP expects a headerless CSV with 36 columns per frame: base XYZ in metres, quaternion XYZW, then 29 joint positions in radians in the order declared in `scripts/csv_to_npz.py`. Use a prepared CSV directly if its SHA-256 matches. Verify the source format, joint order and quaternion convention before converting a different dataset release. Obtain `g1/fallAndGetUp1_subject1.csv` from Unitree's LAFAN1 retarget linked above. Keep the licensed source and converted trajectories in your local workspace.

The publisher's native G1 CSVs use a header, centimetres and degrees. For those files, save the [pinned CSV conversion script](https://github.com/ccrpRepo/robot_retargeter/blob/f1418972319287c1b93af0f7a3b445f613cff5e4/scripts/convert_bones_to_lafan1.py) and run it on each of the 17 unmirrored clip IDs listed in the manifest. From the SuperAMP directory, an example invocation for one clip is:

```bash
python /path/to/convert_bones_to_lafan1.py \
  --input-csv /path/to/licensed/g1/csv/SESSION/arc_jog_left_loop_002__A029.csv \
  --output-root motion_data_csv/amp
```

In the prepared SuperAMP environment, run the following from `SuperAMP/`. Replace the example input path with the local G1 recovery CSV. The recovery source range is inclusive, numbered from 1.

```bash
python scripts/csv_to_npz.py --input-dir motion_data_csv/amp \
  --output-dir src/assets/motions/g1/amp/WalkandRun --input-fps 120 --output-fps 50
python scripts/csv_to_npz.py --input-file /path/to/licensed/g1/fallAndGetUp1_subject1.csv \
  --output-dir src/assets/motions/g1/amp/Recovery --input-fps 30 --output-fps 50 \
  --line-range '(408,1953)'
```

To use these inputs in the legacy perceptive project, copy its two original input directories locally. From the Humoind repository root, run:

```bash
mkdir -p G1_Perceptive_WMAC/motion_data_csv G1_Perceptive_WMAC/src/assets/motions/g1
cp -a SuperAMP/motion_data_csv/amp G1_Perceptive_WMAC/motion_data_csv/
cp -a SuperAMP/src/assets/motions/g1/amp G1_Perceptive_WMAC/src/assets/motions/g1/
```

For PM01, run the retarget guide's preparation, calibration, retarget, refinement and final export steps in the specified hhtools environment. After `pm01_retarget/pm01_npz_50hz/{WalkandRun,Recovery}/` exists, return to the SuperAMP environment and run the following from `SuperAMP/`. Use a fresh output directory for cleaning.

```bash
python scripts/pm01/convert_pm01_motions.py
python scripts/clean_recovery_reference.py --robot pm01
```

These commands produce `src/assets/motions/pm01/amp/` and `src/assets/motions/pm01/amp_clean/`. Each has 17 walk/run clips and one recovery clip. The [PM01 cleaning report](pm01_clean_manifest.json) records the reference run. Numerical conversions can differ between simulator builds or devices. The checksum list identifies the exact training copies.
