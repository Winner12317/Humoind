#!/usr/bin/env bash
# Source this file from Bash to activate SuperAMP.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  printf 'Source this file to activate SuperAMP: source activate_local.sh\n' >&2
  exit 1
fi

AMP_PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)" || return 1
export AMP_PROJECT_DIR
export AMP_CONDA_ROOT=/home/yao/miniconda3
export AMP_CONDA_ENV=superamp
export AMP_LOG_ROOT="$AMP_PROJECT_DIR/results"
export XDG_CACHE_HOME="$AMP_PROJECT_DIR/.wmac_cache"
export TORCH_HOME="$XDG_CACHE_HOME/torch"
# Keep host ROS and CUDA paths out of the pinned training runtime.
export PYTHONPATH="$AMP_PROJECT_DIR"
export PYTHONNOUSERSITE=1
unset LD_LIBRARY_PATH
cd -- "$AMP_PROJECT_DIR" || return 1
source scripts/activate_env.sh
