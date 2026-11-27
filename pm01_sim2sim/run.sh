#!/usr/bin/env bash
# Run the PM01 sim-to-sim tools in SuperAMP's conda environment, with the ROS paths this
# machine's shell exports cleared (they break MuJoCo and PyTorch imports).
#
#   ./run.sh [sim2sim.py options]          interactive viewer (default) or --headless test
#   ./run.sh export [export_policy.py options]   re-export a checkpoint to policy/
set -e
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python="${AMP_CONDA_ROOT:-$HOME/miniconda3}/envs/${AMP_CONDA_ENV:-superamp}/bin/python"
script="$here/sim2sim.py"
if [[ "${1:-}" == "export" ]]; then
  shift
  script="$here/export_policy.py"
fi
exec env -u PYTHONPATH -u LD_LIBRARY_PATH -u VIRTUAL_ENV PYTHONNOUSERSITE=1 OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}" \
  "$python" "$script" "$@"
