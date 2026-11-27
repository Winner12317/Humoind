#!/usr/bin/env bash
set -euo pipefail
# MuJoCo-Warp 3.8.1 drops heightfield foot-terrain contacts unless patched, and
# training refuses that physics. Apply the fix by default; --keep-unpatched-hfield
# reproduces the original audited (incorrect) environment.
amp_hfield_action=--apply
case "${1:-}" in
  ""|--apply-hfield-fix) ;;
  --keep-unpatched-hfield) amp_hfield_action=--check ;;
  *) echo 'Usage: bash reproduce/install.sh [--keep-unpatched-hfield]' >&2; exit 2 ;;
esac
amp_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$amp_root"
if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then
  echo 'This recipe targets Linux x86_64 and an NVIDIA CUDA GPU.' >&2; exit 2
fi
command -v nvidia-smi >/dev/null || { echo 'Install the NVIDIA driver first; nvidia-smi must work.' >&2; exit 2; }
command -v curl >/dev/null || { echo 'Install curl first.' >&2; exit 2; }
# A dedicated, local environment; no inherited Conda environment is reused.
amp_bootstrap="$amp_root/.bootstrap"
amp_conda="$amp_bootstrap/miniforge"
amp_environment="$amp_root/.conda-reproduce"
if [[ ! -x "$amp_conda/bin/conda" ]]; then
  mkdir -p "$amp_bootstrap"
  amp_installer="$amp_bootstrap/Miniforge3-24.11.3-2-Linux-x86_64.sh"
  curl --fail --location --retry 3 'https://github.com/conda-forge/miniforge/releases/download/24.11.3-2/Miniforge3-24.11.3-2-Linux-x86_64.sh' --output "$amp_installer"
  printf '%s  %s\n' '65af53dad30b3fcbd1cb1d4ad62fd3a86221464754844544558aae3a28795189' "$amp_installer" | sha256sum --check -
  bash "$amp_installer" -b -p "$amp_conda"
fi
if [[ ! -x "$amp_environment/bin/python" ]]; then
  "$amp_conda/bin/conda" create --yes --prefix "$amp_environment" --file reproduce/conda-linux-64.lock
fi
amp_python="$amp_environment/bin/python"
"$amp_python" -m pip install --index-url https://pypi.org/simple 'pip==26.2.1' 'setuptools==84.0.0' 'wheel==0.48.0'
"$amp_python" -m pip install --no-deps 'torch==2.7.0+cu126' 'torchvision==0.22.0+cu126' --index-url https://download.pytorch.org/whl/cu126
"$amp_python" -m pip install --index-url https://pypi.org/simple --requirement reproduce/requirements-lock.txt
# mjlab requires upstream rsl-rl-lib==5.0.1. Keep its distribution installed;
# PYTHONPATH below gives this repository's AMP fork import priority.
"$amp_python" -m pip install --no-deps --no-build-isolation --editable .
export PYTHONPATH="$amp_root${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL=egl
export OMP_NUM_THREADS=1
"$amp_python" - <<'PY'
from pathlib import Path
import shutil,hashlib,mjlab
source=Path('mjlab_patch/mjlab/managers/observation_manager.py')
assert hashlib.sha256(source.read_bytes()).hexdigest()=='7d0060915b3b8ccef48a0f272117e775789a1eae336f90c013ba50c3eca24162'
target=Path(mjlab.__file__).parent/'managers/observation_manager.py'
shutil.copyfile(source,target)
print('Installed term/time history-ordering patch:',target)
PY
# Applied by default (see the top of this file). The audited environment was
# unpatched; --keep-unpatched-hfield leaves it that way and only checks it.
"$amp_python" scripts/fix_mujoco_warp_hfield.py "$amp_hfield_action"
"$amp_python" reproduce/fix_mjlab_cuda_stream.py --apply
"$amp_python" -m pip check
"$amp_python" - <<'PY'
import torch,rsl_rl,sys
from pathlib import Path
assert torch.cuda.is_available(),'CUDA unavailable: check NVIDIA driver and GPU visibility'
assert Path(rsl_rl.__file__).resolve().is_relative_to(Path.cwd()),'Wrong rsl_rl imported'
print('Python:',sys.version)
print('Torch:',torch.__version__,'GPU:',torch.cuda.get_device_name())
print('Local AMP rsl_rl:',rsl_rl.__file__)
PY
# Acquire the pinned Hiking reference from its author or use the local source
# described in QUICKSTART.md. All subsequent motion checks run offline.
"$amp_python" scripts/prepare_mirrored_motions.py
"$amp_python" reproduce/verify-offline-inputs.py
# CPU contract checks include the current URDF/head assets, attention readout,
# vector-step clock, sensor capture and execution reuse. No long training.
CUDA_VISIBLE_DEVICES="" PYTHONPATH="$amp_root:$amp_root/tests" "$amp_python" -m unittest \
  test_hiking_bundled_source test_hiking_amp test_hiking_observations \
  test_hiking_standing_expert \
  test_hiking_full_world_model test_hiking_depth_randomization \
  test_perceptive_reference_schedule test_mujoco_warp_hfield_fix test_mjlab_cuda_stream_fix \
  test_hiking_reference_robot test_hiking_alignment_v4 test_hiking_real_robot_options test_device_placement test_perceptive_attention_policy \
  test_hiking_depth_capture_execution test_hiking_vector_clock \
  test_replay_sampling_execution test_wm_macro_direct_staging \
  test_imagination_execution_reuse test_perceptive_profiler
cat > reproduce/activate.sh <<'ACTIVATE'
_amp_repro_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$_amp_repro_root/.bootstrap/miniforge/etc/profile.d/conda.sh"
conda activate "$_amp_repro_root/.conda-reproduce"
export PYTHONPATH="$_amp_repro_root${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL=egl
export OMP_NUM_THREADS=1
cd "$_amp_repro_root"
unset _amp_repro_root
ACTIVATE
printf '\nInstallation and motion preparation complete.\nRun: source reproduce/activate.sh\nThen run the small validation commands in the document before full training.\n'
