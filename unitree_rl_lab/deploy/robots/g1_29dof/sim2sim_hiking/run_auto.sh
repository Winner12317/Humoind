#!/usr/bin/env bash
# One unattended sim-to-sim test: hiking_mujoco.py (simulator + depth camera +
# scripted remote) and the unchanged g1_ctrl binary, over loopback DDS.
#   bash run_auto.sh <terrain> [results.json] [extra hiking_mujoco.py args...]
# Terrains: flat, rough_0.5, rough_1.0, stairs_up_0.3, stairs_up_0.6, stairs_down_0.3, stairs_down_0.6, slope_0.5
set -u
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
terrain="${1:-flat}"; results="${2:-$here/results/$terrain.json}"; shift $(( $# > 2 ? 2 : $# ))
python="${HIKING_SIM_PYTHON:-$HOME/.local/share/g1-hiking-sim2sim/venv/bin/python}"
ctrl="$here/../build/g1_ctrl"
[[ -x "$ctrl" ]] || { echo "Build g1_ctrl first (deploy/robots/g1_29dof/build)"; exit 2; }
mkdir -p "$(dirname -- "$results")"
results="$(cd -- "$(dirname -- "$results")" && pwd)/$(basename -- "$results")"
clean_env=(env -u PYTHONPATH -u LD_LIBRARY_PATH -u VIRTUAL_ENV -u CONDA_PREFIX PYTHONNOUSERSITE=1)
if pgrep -x g1_ctrl >/dev/null; then echo "Another g1_ctrl is running (two controllers would both command the robot)"; exit 2; fi
"${clean_env[@]}" "$python" "$here/hiking_mujoco.py" --terrain "$terrain" --auto --results "$results" "$@" &
sim=$!
sleep 1
(cd "$here/../build" && exec "${clean_env[@]}" ./g1_ctrl --network lo > "${results%.json}.g1_ctrl.log" 2>&1) &
ctrl_pid=$!
trap 'kill "$sim" "$ctrl_pid" 2>/dev/null' INT TERM
wait "$sim"; status=$?
kill "$ctrl_pid" 2>/dev/null; wait "$ctrl_pid" 2>/dev/null
exit $status
