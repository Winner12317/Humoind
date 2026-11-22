#!/usr/bin/env bash
set -euo pipefail

die() {
  printf '[SuperAMP] %s\n' "$*" >&2
  exit 1
}

case "${1:-}" in
  -h|--help)
    cat <<'HELP'
Usage: bash scripts/install_env.sh [--check]
Create or reuse the superamp Conda environment on Linux x86_64.
--check verifies an existing environment. It skips installation.
AMP_CONDA_ROOT selects a Conda installation, AMP_CONDA_ENV selects the environment name.
HELP
    exit 0
    ;;
  ""|--check) ;;
  *) die "Unknown option: $1 (see --help)" ;;
esac
(( $# <= 1 )) || die 'Expected at most one argument (see --help).'
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] ||
  die 'This setup uses the Linux x86_64 dependency lock.'
unset PYTHONHOME PYTHONPATH
export PYTHONNOUSERSITE=1

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$project_root"
env_name="${AMP_CONDA_ENV:-superamp}"
check_only="${1:-}"
[[ "$env_name" =~ ^[[:alnum:]_][[:alnum:]_.-]*$ && "$env_name" != base && "$env_name" != root ]] ||
  die 'AMP_CONDA_ENV must name a dedicated Conda environment.'

conda_lock="$project_root/docs/reproduction/conda-linux-64.txt"
requirements="$project_root/docs/reproduction/requirements-linux-py311-cu126.txt"
[[ -f "$conda_lock" && -f "$requirements" && -f setup.py ]] ||
  die 'Run this script from a complete SuperAMP checkout.'

if [[ -n "${AMP_CONDA_ROOT:-}" ]]; then
  conda_root="$AMP_CONDA_ROOT"
elif [[ -x "${CONDA_EXE:-}" ]]; then
  conda_root="$("$CONDA_EXE" info --base)"
elif conda_exe="$(type -P conda)"; then
  conda_root="$("$conda_exe" info --base)"
else
  conda_root="$HOME/miniconda3"
  for candidate in "$HOME/miniconda3" "$HOME/miniforge3" "$HOME/anaconda3"; do
    if [[ -x "$candidate/bin/conda" ]]; then
      conda_root="$candidate"
      break
    fi
  done
fi

if [[ ! -x "$conda_root/bin/conda" ]]; then
  [[ -z "$check_only" ]] || die "Conda was not found at $conda_root."
  [[ ! -e "$conda_root" && ! -L "$conda_root" ]] ||
    die "Conda is missing from the existing directory $conda_root. Set AMP_CONDA_ROOT to another location."
  command -v curl >/dev/null || die 'Install curl to download Miniconda.'
  miniconda_version=Miniconda3-py311_26.9.4-1-Linux-x86_64.sh
  miniconda_sha256=6dcf907532813e1007605cbcbd3012becb348de2657f9faaf03494b864faa240
  installer_dir="$(mktemp -d)"
  installer="$installer_dir/$miniconda_version"
  trap 'rm -f -- "$installer"; rmdir -- "$installer_dir"' EXIT
  printf '[SuperAMP] Installing Miniconda in %s\n' "$conda_root"
  curl --fail --location --retry 3 --connect-timeout 20 \
    "https://repo.anaconda.com/miniconda/$miniconda_version" -o "$installer"
  printf '%s  %s\n' "$miniconda_sha256" "$installer" | sha256sum --check --status ||
    die 'Miniconda installer SHA-256 mismatch.'
  mkdir -p -- "$(dirname -- "$conda_root")"
  bash "$installer" -b -p "$conda_root"
fi

conda_exe="$conda_root/bin/conda"
run_env() {
  "$conda_exe" run --no-capture-output --name "$env_name" "$@"
}

env_status="$("$conda_exe" info --json | "$conda_root/bin/python" -c '
import json
from pathlib import Path
import sys

name = sys.argv[1]
for directory in json.load(sys.stdin)["envs_dirs"]:
    prefix = Path(directory) / name
    if (prefix / "conda-meta" / "history").is_file():
        print("existing")
        break
    if prefix.exists() or prefix.is_symlink():
        sys.exit(f"Conda environment path already exists without Conda metadata: {prefix}")
else:
    print("missing")
' "$env_name")"
if [[ "$env_status" == missing ]]; then
  [[ -z "$check_only" ]] || die "Conda environment $env_name is missing."
  printf '[SuperAMP] Creating %s with Python 3.11.17\n' "$env_name"
  "$conda_exe" create --name "$env_name" --yes --no-default-packages \
    --override-channels --channel https://conda.anaconda.org/conda-forge \
    --file "$conda_lock"
fi

export PYTHONPATH="$project_root"
export MUJOCO_GL=egl

run_env python - "$conda_root" <<'PY'
from ctypes.util import find_library
from pathlib import Path
import sys

if Path(sys.prefix).resolve() == Path(sys.argv[1]).resolve():
    sys.exit("Use a dedicated environment outside the Conda base prefix.")
if sys.version_info[:3] != (3, 11, 17):
    sys.exit(f"SuperAMP requires Python 3.11.17, found {sys.version.split()[0]}. "
             "Set AMP_CONDA_ENV to a new environment name.")
missing = [name for name in ("EGL", "GL") if not find_library(name)]
if missing:
    sys.exit(f"Missing system libraries: {', '.join(missing)}. "
             "On Ubuntu install them with sudo apt-get install libegl1 libgl1.")
PY

if [[ -z "$check_only" ]]; then
  export PIP_CONFIG_FILE=/dev/null
  export PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.org/simple}"
  unset PIP_USER PIP_TARGET PIP_PREFIX PIP_EXTRA_INDEX_URL PIP_CONSTRAINT PIP_BUILD_CONSTRAINT PIP_REQUIREMENT
  run_env python -m pip install --no-input 'pip==26.2.1' 'setuptools==84.0.0' 'wheel==0.48.0'
  run_env python -m pip install --no-input --index-url https://download.pytorch.org/whl/cu126 \
    'torch==2.7.0+cu126' 'torchvision==0.22.0+cu126'
  run_env python -m pip install --no-input -r "$requirements"
  run_env python -m pip install --no-input --no-build-isolation --no-deps -e .
  run_env python - <<'PY'
from importlib.metadata import distribution
from pathlib import Path
import shutil
import sys
from tempfile import NamedTemporaryFile

relative = "mjlab/managers/observation_manager.py"
package = distribution("mjlab")
patch = Path("mjlab_patch", relative).read_bytes()
target = Path(package.locate_file(relative)).resolve()
if package.version != "1.2.0" or not target.is_relative_to(Path(sys.prefix).resolve()):
    sys.exit(f"Unexpected mjlab installation: {package.version}, {target}")
if target.read_bytes() != patch:
    backup = target.with_suffix(".py.before_amp_patch")
    if not backup.exists():
        shutil.copy2(target, backup)
    with NamedTemporaryFile(dir=target.parent, delete=False) as temp:
        replacement = Path(temp.name)
        temp.write(patch)
    try:
        shutil.copymode(target, replacement)
        replacement.replace(target)
    finally:
        replacement.unlink(missing_ok=True)
print("SuperAMP mjlab history patch installed.")
PY
fi

run_env python -m pip check
run_env python - "$requirements" <<'PY'
from importlib.metadata import distribution, version
from pathlib import Path
import sys

from packaging.requirements import Requirement

for line in Path(sys.argv[1]).read_text().splitlines():
    line = line.strip()
    if not line or line.startswith(("#", "--")):
        continue
    requirement = Requirement(line)
    installed = version(requirement.name)
    if installed not in requirement.specifier:
        sys.exit(f"Version mismatch: {requirement.name} {installed} ({requirement.specifier})")
if version("pip") != "26.2.1":
    sys.exit("Expected pip 26.2.1.")

import torch
import warp as wp
import rsl_rl
import src.tasks  # registers the G1 and PM01 tasks
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.tasks.registry import list_tasks

if not Path(rsl_rl.__file__).resolve().is_relative_to(Path.cwd() / "rsl_rl"):
    sys.exit(f"Wrong RSL-RL source: {rsl_rl.__file__}")
relative = "mjlab/managers/observation_manager.py"
target = Path(distribution("mjlab").locate_file(relative))
if ("history_ordering" not in ObservationGroupCfg.__dataclass_fields__
        or target.read_bytes() != Path("mjlab_patch", relative).read_bytes()):
    sys.exit("The installed mjlab history patch differs from the project.")
required_tasks = {"Unitree-G1-AMP-Flat", "EngineAI-PM01-AMP-Flat", "EngineAI-PM01-AMP-Flat-Aligned"}
missing_tasks = required_tasks.difference(list_tasks())
if missing_tasks:
    sys.exit(f"Missing SuperAMP tasks: {sorted(missing_tasks)}")

print(f"Python {sys.version.split()[0]}, PyTorch {torch.__version__}, mjlab {version('mjlab')}")
if not torch.cuda.is_available():
    print("CUDA device unavailable. Dependency checks passed; GPU training remains unverified.")
else:
    result = torch.ones((2, 2), device="cuda")
    if (result @ result)[0, 0].item() != 2.0:
        sys.exit("PyTorch CUDA calculation failed.")
    if wp.zeros(1, dtype=wp.float32, device="cuda:0").numpy()[0] != 0.0:
        sys.exit("Warp CUDA allocation failed.")
    print(f"PyTorch and Warp CUDA checks passed on {torch.cuda.get_device_name(0)}.")

for robot in ("g1", "pm01"):
    motions = Path("src/assets/motions") / robot / "amp"
    if any(not any((motions / group).glob("*.npz")) for group in ("WalkandRun", "Recovery")):
        print(f"Restore {robot.upper()} reference motions in {motions} before training it.")
PY

printf '[SuperAMP] Activate and enter the training directory with\n'
printf '  cd %q\n' "$project_root"
printf '  export AMP_CONDA_ROOT=%q AMP_CONDA_ENV=%q\n' "$conda_root" "$env_name"
printf '  source scripts/activate_env.sh\n'
