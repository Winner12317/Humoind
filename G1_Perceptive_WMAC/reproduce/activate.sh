# Source this file in Bash from any working directory.
_amp_repro_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
_amp_repro_env="$(cd -- "$_amp_repro_root/.conda-reproduce" && pwd -P)"
source "$_amp_repro_root/.bootstrap/miniforge/etc/profile.d/conda.sh"
conda activate "$_amp_repro_env"
export PYTHONPATH="$_amp_repro_root"
export PYTHONNOUSERSITE=1
export MUJOCO_GL=egl
export OMP_NUM_THREADS=1
cd -- "$_amp_repro_root"
unset _amp_repro_root _amp_repro_env
