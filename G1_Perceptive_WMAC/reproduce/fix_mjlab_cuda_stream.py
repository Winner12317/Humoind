"""Keep the owning CUDA device when mjlab 1.2.0 wraps a Warp stream."""

import argparse
import hashlib
import importlib.metadata
import os
from pathlib import Path
import stat
import tempfile


MODULE = "mjlab/sim/sim_data.py"
VERSION = "1.2.0"
ORIGINAL = "bc73da4811fc44336f9016d826ef1b2044b8cb0830f2ad01a06b9b84ffa8b4c6"
PATCHED = "1b4e6ef0f8a20d7242d969efcc31bd83a6b43f5f902307e5c3114210141fe47b"
OLD_RETURN = b"return torch.cuda.ExternalStream(warp_stream.cuda_stream)"
NEW_RETURN = b"return wp.stream_to_torch(warp_stream)"


def check_installation(distribution, *, apply=False):
    if distribution.version != VERSION:
        raise ValueError(f"Expected mjlab {VERSION}, found {distribution.version}")
    if MODULE not in {str(entry) for entry in distribution.files or ()}:
        raise ValueError(f"The installed mjlab distribution does not own {MODULE}")
    root = Path(distribution.locate_file("")).resolve()
    target = Path(distribution.locate_file(MODULE))
    if target.resolve() != root / MODULE:
        raise ValueError(f"Unexpected mjlab source path: {target}")
    source = target.read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    if digest == PATCHED:
        return "already patched"
    if digest != ORIGINAL:
        raise ValueError(f"Unrecognized mjlab {VERSION} source SHA256 for {target}: {digest}")
    if not apply:
        return "patch available"
    updated = source.replace(OLD_RETURN, NEW_RETURN, 1)
    if hashlib.sha256(updated).hexdigest() != PATCHED:
        raise ValueError("The CUDA stream patch did not produce the expected source")
    # Keep readers on a complete file even when installation is interrupted.
    descriptor, name = tempfile.mkstemp(dir=target.parent)
    temp = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(updated)
            os.fchmod(handle.fileno(), stat.S_IMODE(target.stat().st_mode))
        os.replace(temp, target)
    finally:
        temp.unlink(missing_ok=True)
    return "patched"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--check", action="store_true", help="report patch status (default)")
    action.add_argument("--apply", action="store_true", help="apply the verified change")
    args = parser.parse_args()
    try:
        status = check_installation(importlib.metadata.distribution("mjlab"), apply=args.apply)
    except (importlib.metadata.PackageNotFoundError, OSError, ValueError) as error:
        parser.exit(1, f"mjlab CUDA stream check failed: {error}\n")
    print(f"mjlab {VERSION} CUDA stream device: {status}")


if __name__ == "__main__":
    main()
