#!/usr/bin/env bash
# Run from a dedicated, already activated Python 3.11.17 environment.
set -euo pipefail
amp_repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$amp_repo_root"
python - <<'PY'
import platform
import sys
import os
from pathlib import Path
assert sys.version_info[:3] == (3, 11, 17), sys.version
assert platform.system() == 'Linux' and platform.machine() == 'x86_64'
if os.environ.get('CONDA_PREFIX'):
    assert Path(os.environ['CONDA_PREFIX']).resolve() == Path(sys.prefix).resolve()
    assert os.environ.get('CONDA_DEFAULT_ENV') != 'base', 'Use a dedicated environment, not base'
else:
    assert sys.prefix != sys.base_prefix, 'Activate a dedicated environment first'
PY
python -m pip install 'pip==26.2.1'
python -m pip install 'setuptools==84.0.0' 'wheel==0.48.0'
python -m pip install 'torch==2.7.0' 'torchvision==0.22.0' --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r docs/reproduction/requirements-linux-py311-cu126.txt
python -m pip install --no-deps --no-build-isolation -e .
python - <<'PY'
from importlib.metadata import distribution
from pathlib import Path
import hashlib
import shutil
package = distribution('mjlab')
assert package.version == '1.2.0', package.version
source = Path('mjlab_patch/mjlab/managers/observation_manager.py').resolve()
target = Path(package.locate_file('mjlab/managers/observation_manager.py')).resolve()
assert source.is_file() and target.is_file()
assert hashlib.sha256(source.read_bytes()).hexdigest() == '7d0060915b3b8ccef48a0f272117e775789a1eae336f90c013ba50c3eca24162'
if target.read_bytes() != source.read_bytes():
    backup = target.with_suffix('.py.before_amp_patch')
    if not backup.exists():
        shutil.copy2(target, backup)
    shutil.copy2(source, target)
assert target.read_bytes() == source.read_bytes()
print('Installed required mjlab history patch:', target)
PY
export PYTHONPATH="$amp_repo_root${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL=egl
python -m pip check
python docs/reproduction/verify_reproduction.py --environment
