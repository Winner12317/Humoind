#!/usr/bin/env bash
# Usage: source scripts/activate_env.sh
# Set AMP_CONDA_ENV first to use another existing Conda environment.
# Set AMP_CONDA_ROOT if Conda is installed outside the usual locations.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  printf 'Source this script to activate the training environment: source scripts/activate_env.sh\n' >&2
  exit 1
fi

_amp_activate_training_env() {
  local amp_project_root amp_conda_init amp_conda_root amp_env
  amp_project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)" || return 1
  amp_env="${AMP_CONDA_ENV:-yao-rl}"
  amp_conda_init=""

  if [[ -n "${AMP_CONDA_ROOT:-}" ]]; then
    amp_conda_init="$AMP_CONDA_ROOT/etc/profile.d/conda.sh"
  elif [[ -n "${CONDA_EXE:-}" ]]; then
    amp_conda_init="$(dirname -- "$(dirname -- "$CONDA_EXE")")/etc/profile.d/conda.sh"
  elif command -v conda >/dev/null 2>&1; then
    amp_conda_root="$(conda info --base)" || return 1
    amp_conda_init="$amp_conda_root/etc/profile.d/conda.sh"
  else
    for amp_conda_root in "$HOME/miniforge3" "$HOME/miniconda3" "$HOME/anaconda3"; do
      if [[ -r "$amp_conda_root/etc/profile.d/conda.sh" ]]; then
        amp_conda_init="$amp_conda_root/etc/profile.d/conda.sh"
        break
      fi
    done
  fi

  if [[ ! -r "$amp_conda_init" ]]; then
    printf 'Cannot find Conda initialization script. Set AMP_CONDA_ROOT to your Conda installation directory.\n' >&2
    return 1
  fi
  source "$amp_conda_init" || return 1
  conda activate "$amp_env" || return 1

  # mjlab depends on the upstream distribution, while AMP uses this local fork.
  case "${PYTHONPATH-}" in
    "$amp_project_root"|"$amp_project_root":*) ;;
    *) export PYTHONPATH="$amp_project_root${PYTHONPATH:+:$PYTHONPATH}" ;;
  esac
  export MUJOCO_GL=egl
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
  printf 'AMP training environment: %s\n' "$CONDA_PREFIX"
}

if _amp_activate_training_env; then
  unset -f _amp_activate_training_env
else
  unset -f _amp_activate_training_env
  return 1
fi
