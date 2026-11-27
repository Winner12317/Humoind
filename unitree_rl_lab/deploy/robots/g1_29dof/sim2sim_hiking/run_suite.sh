#!/usr/bin/env bash
# Repeated unattended tests over terrains and seeds; prints a summary table.
#   bash run_suite.sh [out_dir] [seeds] [terrains...]   (extra hiking_mujoco args via SUITE_ARGS)
set -u
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
out="${1:-$here/results/suite}"; seeds="${2:-5}"; shift $(( $# > 2 ? 2 : $# ))
terrains=("$@"); [[ ${#terrains[@]} -gt 0 ]] || terrains=(flat rough_0.5 rough_1.0 slope_0.5 stairs_down_0.3 stairs_down_0.6 stairs_up_0.3 stairs_up_0.6)
mkdir -p "$out"
for terrain in "${terrains[@]}"; do
  for seed in $(seq 0 $((seeds - 1))); do
    bash "$here/run_auto.sh" "$terrain" "$out/${terrain}_seed${seed}.json" --seed "$seed" --walk-seconds 10 ${SUITE_ARGS:-} > /dev/null 2>&1
  done
done
python3 - "$out" <<'PY'
import json, sys, glob, collections
rows = collections.defaultdict(list)
for path in sorted(glob.glob(f"{sys.argv[1]}/*_seed*.json")):
    r = json.load(open(path))
    rows[r["terrain"]].append(r)
print(f"{'terrain':18s} {'runs':>4s} {'falls':>6s} {'reached 3.4 m':>14s} {'mean progress':>14s}")
for terrain, runs in rows.items():
    falls = sum(r["fell_at"] is not None for r in runs)
    reached = sum(r["reached_target_at"] is not None for r in runs)
    progress = sum(r.get("progress_x_m", 0.) for r in runs) / len(runs)
    print(f"{terrain:18s} {len(runs):4d} {falls:6d} {reached:14d} {progress:13.2f}m")
PY
