"""Explicit, offline backport of MuJoCo-Warp's heightfield contact fix.

Default invocation only checks the installed source. --apply writes a verified
backup before replacing one Python source file; --restore reverses only this
exact patch. Restart training afterwards; running processes are not modified.

Upstream: google-deepmind/mujoco_warp commit
0571f6721752f4eb3f5c2da3d4ecce8307a733cd (PR #1677).
The adapted kernel changes retain Copyright 2025 The Newton Developers and
the original Apache-2.0 license in collision_convex.py. Unlike the later
adhesion API, version 3.8.1 write_contact accepts dist < margin, independent
of gap. Its backport must use that same threshold, not margin + gap.
"""
from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import stat
import sys
import tempfile


UPSTREAM_COMMIT = "0571f6721752f4eb3f5c2da3d4ecce8307a733cd"
SUPPORTED_VERSION = "3.8.1"
ORIGINAL_SHA256 = "1932c52be6867bed6ae3265ae683102f54099ae278ea93c3dcd25ff1f5e9c8bf"
CORE_SHA256 = "3c12b154998af28e257ff80437ca4250546947a6b3bf2af7fa3f1210fdff7d74"


class PatchError(RuntimeError):
    pass


def sha256(source: bytes) -> str:
    return hashlib.sha256(source).hexdigest()


def _builder(source: str):
    tree = ast.parse(source)
    builders = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                and node.name == "ccd_hfield_kernel_builder"]
    if len(builders) != 1:
        raise PatchError("Expected exactly one heightfield kernel builder")
    return builders[0]


def _replace_once(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise PatchError("Pinned source patch context did not match exactly once")
    return source.replace(old, new, 1)


def transformed_source(source: bytes, core_source: bytes) -> bytes:
    """Return patched bytes, guarded by both old CCD and contact semantics."""
    if sha256(source) != ORIGINAL_SHA256 or sha256(core_source) != CORE_SHA256:
        raise PatchError("Source SHA256 differs from the audited MuJoCo-Warp 3.8.1 files")
    text = source.decode("utf-8")
    node = _builder(text)
    lines = text.splitlines(keepends=True)
    original = "".join(lines[node.lineno-1:node.end_lineno])
    result = _replace_once(original,
        "    count = int(0)\n    for r in range(rmin, rmax):\n",
        "    count = int(0)\n    noverflow = int(0)\n    for r in range(rmin, rmax):\n"
        "      if noverflow > 0:\n        break\n\n")
    result = _replace_once(result,
        "      for c in range(cmin + 1, cmax + 1):\n",
        "      for c in range(cmin + 1, cmax + 1):\n"
        "        if noverflow > 0:\n          break\n\n")
    warning = (
        '            wp.printf(\n'
        '              "height field collision overflow, number of collisions >= %u - please adjust resolution: \\n decrease the number of hfield rows/cols or modify size of colliding geom\\n",\n'
        '              MJ_MAXCONPAIR,\n'
        '            )\n')
    result = _replace_once(result,
        "          if count >= MJ_MAXCONPAIR:\n"+warning+"            continue\n",
        "          if count >= MJ_MAXCONPAIR:\n            noverflow = 1\n            break\n")
    result = _replace_once(result,
        "          if ncontact == 0:\n            continue\n",
        "          # Skip separated prisms, matching 3.8.1 write_contact (dist < margin).\n"
        "          # Upstream's later adhesion API instead uses margin + gap.\n"
        "          if ncontact == 0 or dist >= margin:\n            continue\n")
    result = _replace_once(result,
        "    # contact 0: minimum distance\n",
        "    # Warn once per pair; real overflow remains visible.\n"
        "    if noverflow > 0:\n"+
        "\n".join(line[6:] for line in warning.rstrip("\n").split("\n"))+"\n\n"
        "    # Skip write when no prisms collide.\n    if count == 0:\n      return\n\n"
        "    # contact 0: minimum distance\n")
    updated = "".join(lines[:node.lineno-1])+result+"".join(lines[node.end_lineno:])
    ast.parse(updated)
    return updated.encode("utf-8")


def has_upstream_fix(source: bytes) -> bool:
    """Recognize the complete upstream control flow without modifying it."""
    try:
        builder = _builder(source.decode("utf-8"))
        kernel = next(node for node in builder.body
                      if isinstance(node, ast.FunctionDef) and node.name == "ccd_hfield_kernel")
        tests = {ast.unparse(node.test): node for node in ast.walk(kernel) if isinstance(node, ast.If)}
        if not any(key in tests for key in ("ncontact == 0 or dist >= margin + gap", "ncontact == 0 or dist >= margin")):
            return False
        zero = tests.get("count == 0")
        cap = tests.get("count >= MJ_MAXCONPAIR")
        if zero is None or not any(isinstance(node, ast.Return) for node in zero.body):
            return False
        if cap is None or not any(isinstance(node, ast.Break) for node in cap.body):
            return False
        if not any(isinstance(node, ast.Assign) and ast.unparse(node) == "noverflow = 1" for node in cap.body):
            return False
        loops = [node for node in ast.walk(kernel) if isinstance(node, ast.For)
                 and ast.unparse(node.target) in ("r", "c")]
        if len(loops) != 2 or any(not isinstance(node.body[0], ast.If)
                or ast.unparse(node.body[0].test) != "noverflow > 0"
                or not any(isinstance(item, ast.Break) for item in node.body[0].body) for node in loops):
            return False
        # The warning must be outside all grid loops, not merely suppressed.
        outside = [node for node in kernel.body if isinstance(node, ast.If)
                   and ast.unparse(node.test) == "noverflow > 0"]
        return any(any(isinstance(item, ast.Call) and ast.unparse(item.func) == "wp.printf"
                       for item in ast.walk(node)) for node in outside)
    except (SyntaxError, UnicodeDecodeError, PatchError, StopIteration):
        return False


def installation_status(source: bytes, core_source: bytes, version: str) -> str:
    digest = sha256(source)
    if digest in (ORIGINAL_SHA256,PATCHED_SHA256):
        if version != SUPPORTED_VERSION or sha256(core_source) != CORE_SHA256:
            raise PatchError("Pinned collision file has a mismatched version or contact helper SHA256")
        if digest == ORIGINAL_SHA256:
            return "needs_patch"
        # This also validates every changed byte in our adaptation.
        # Reverse-transforming is unnecessary: compare the precomputed hash.
        return "patched"
    if has_upstream_fix(source):
        return "upstream_fixed"
    raise PatchError("Unsupported version/source: refusing to patch an unaudited collision implementation")


def installed_files():
    distribution = importlib.metadata.distribution("mujoco-warp")
    path = Path(distribution.locate_file("mujoco_warp/_src/collision_convex.py"))
    return path, distribution.version


def _atomic_write(path: Path, data: bytes, mode: int):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix=path.name+".", suffix=".tmp", dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            os.fchmod(handle.fileno(), mode)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _create_backup(path: Path, data: bytes, mode: int):
    """Publish a complete backup atomically without replacing another backup."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix=path.name+".",suffix=".tmp",dir=path.parent,delete=False) as handle:
            temporary = Path(handle.name)
            os.fchmod(handle.fileno(),mode)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary,path)
        except FileExistsError:
            if path.is_symlink() or not path.is_file() or path.read_bytes() != data:
                raise PatchError("Backup path appeared with unexpected contents") from None
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def process_installation(path: Path, version: str, action="check") -> dict:
    """Inspect, install or restore a pinned file; used with temp copies in tests."""
    if action not in ("check", "apply", "restore"):
        raise ValueError("Unknown patch action")
    path = Path(path)
    if path.is_symlink():
        raise PatchError("Refusing to replace a symlinked collision source")
    source = path.read_bytes()
    core = path.with_name("collision_core.py").read_bytes()
    status = installation_status(source, core, version)
    backup = path.with_name(path.name+".before_hfield_"+UPSTREAM_COMMIT[:8]+"_"+ORIGINAL_SHA256[:12]+".bak")
    report = dict(status=status,action=action,version=version,path=str(path),backup=str(backup),
                  source_sha256=sha256(source),write_contact_module_sha256=sha256(core),changed=False,
                  upstream_commit=UPSTREAM_COMMIT)
    if action == "check" or (action == "apply" and status != "needs_patch"):
        return report
    if action == "restore":
        if status not in ("patched", "needs_patch"):
            raise PatchError("Restore only supports this exact patch and audited original")
        if not backup.is_file() or backup.is_symlink() or sha256(backup.read_bytes()) != ORIGINAL_SHA256:
            raise PatchError("Original backup is missing or its SHA256 changed")
        replacement = backup.read_bytes()
        if status == "needs_patch":
            return report
    else:
        replacement = transformed_source(source, core)
        if sha256(replacement) != PATCHED_SHA256:
            raise PatchError("Generated patch SHA256 differs from the audited result")
        if backup.exists():
            if backup.is_symlink() or not backup.is_file() or sha256(backup.read_bytes()) != ORIGINAL_SHA256:
                raise PatchError("Backup path exists with unexpected contents")
        else:
            _create_backup(backup,source,stat.S_IMODE(path.stat().st_mode))
    if path.read_bytes() != source:
        raise PatchError("Collision source changed during preparation; no replacement made")
    _atomic_write(path,replacement,stat.S_IMODE(path.stat().st_mode))
    report.update(changed=True,status="patched" if action == "apply" else "needs_patch",
                  source_sha256=sha256(replacement),restart_required=True)
    return report


# Filled from the exact guarded transformation, not from the installed file.
PATCHED_SHA256 = "5e223fb221cbb34c62394345462c785b29665e0c42728d3601c06d307a017727"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--check",action="store_true",help="Inspect only (default)")
    group.add_argument("--apply",action="store_true",help="Back up and patch the audited dependency")
    group.add_argument("--restore",action="store_true",help="Restore only this patch's verified backup")
    parser.add_argument("--diff",action="store_true",help="Print the proposed source diff (read-only unless --apply)")
    args = parser.parse_args(argv)
    if args.restore and args.diff:
        parser.error("--diff describes the forward patch; use it with --check or --apply")
    try:
        path,version = installed_files()
        if args.diff:
            original = path.read_bytes()
            core = path.with_name("collision_core.py").read_bytes()
            status = installation_status(original,core,version)
            if status == "needs_patch":
                sys.stdout.writelines(difflib.unified_diff(original.decode().splitlines(keepends=True),
                    transformed_source(original,core).decode().splitlines(keepends=True),
                    fromfile=str(path),tofile=str(path)+" (upstream-compatible backport)"))
        action = "apply" if args.apply else "restore" if args.restore else "check"
        print(json.dumps(process_installation(path,version,action),ensure_ascii=False,indent=2))
        return 0
    except (PatchError,OSError,importlib.metadata.PackageNotFoundError) as error:
        print(f"Heightfield patch refused: {error}",file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
