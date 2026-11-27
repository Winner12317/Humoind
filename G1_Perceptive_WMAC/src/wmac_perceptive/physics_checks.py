"""Refuse to train on MuJoCo-Warp 3.8.1's unpatched heightfield contacts.

Unpatched 3.8.1 counts non-contact heightfield prisms toward its 50-candidate
limit and silently drops later foot-terrain contacts (hundreds of overflow
warnings per 10 s evaluation on the Hiking terrain). A policy trained there
learns to rely on missing stair-riser contacts: with correct contacts the
`sensor_transform.pt` checkpoint's stair-ascent failures rose from 61% to 82%.
"""
from __future__ import annotations

from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[2]
FIXED_STATUSES = ("patched", "upstream_fixed")
_INSTRUCTIONS = ("Apply the heightfield contact fix first:\n"
                 "    python scripts/fix_mujoco_warp_hfield.py --apply\n"
                 "or pass --allow-unpatched-hfield to reproduce the original (incorrect) audited physics.")


def hfield_collision_status() -> dict:
    """Inspect the installed MuJoCo-Warp collision source without importing Warp kernels."""
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    from scripts import fix_mujoco_warp_hfield as fix
    path, version = fix.installed_files()
    try:
        status = fix.installation_status(path.read_bytes(), path.with_name("collision_core.py").read_bytes(), version)
    except fix.PatchError as error:
        status = f"unrecognized: {error}"
    return {"status": status, "mujoco_warp_version": version, "path": str(path)}


def require_fixed_hfield_collision(allow_unpatched: bool = False) -> dict:
    report = hfield_collision_status()
    if report["status"] in FIXED_STATUSES:
        return report
    if report["status"] == "needs_patch":
        if allow_unpatched:
            print("[Perceptive WMAC] WARNING: training on unpatched MuJoCo-Warp heightfield contacts "
                  "(foot-terrain contacts are dropped). " + _INSTRUCTIONS.splitlines()[0], flush=True)
            return report
        raise RuntimeError("MuJoCo-Warp 3.8.1 heightfield collisions are unpatched: foot-terrain contacts "
                           "are silently dropped on stairs and rough terrain.\n" + _INSTRUCTIONS)
    # A newer or different MuJoCo-Warp: the check cannot vouch for it either way.
    print(f"[Perceptive WMAC] Heightfield collision check: {report['status']}", flush=True)
    return report
