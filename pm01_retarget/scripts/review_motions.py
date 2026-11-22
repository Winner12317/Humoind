"""Browser viewer for stepping through the retargeted PM01 motions frame by frame.

Starts a local web server and opens the viewer.  Pick a clip, then step with the
arrow keys; the PM01 result is drawn next to (or on top of) the G1 source frame
it was retargeted from, with the per-frame numbers of validate.py beside it:
segment rotation error, sole heights, joint angles against their limits,
ground and self penetration.

    python review_motions.py                 # http://127.0.0.1:8765
    python review_motions.py --port 9000 --no-browser

Every folder ``pm01_retarget/pm01_*`` is offered as a motion set (the final
``pm01_csv``, the 50 Hz ``pm01_npz_50hz``, the raw ``pm01_hhtools`` export, the
``pm01_refined`` clips); a clip is shown when ``g1_source/<group>/<clip>.csv``
exists for it.  Poses and numbers come from MuJoCo forward kinematics on the
models validate.py uses, so what is drawn is exactly what the files contain.
PM01's "collision" shapes are those of EngineAI's official MJCF; its "visual"
shapes are the URDF meshes.

Run with the hhtools environment (see ../README.md).
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import sys
import threading
import webbrowser
from collections import OrderedDict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import mujoco
import numpy as np

from common import G1_JOINT_NAMES, G1_MJCF, PM01_JOINT_NAMES, PM01_SDK_COLLISION, WORK, resample_like_csv_to_npz
from validate import (
    FEET,
    PM01_SPEED_LIMITS,
    PM01_URDF,
    SEGMENTS,
    Robot,
    calibration_rotations,
    frame_metrics,
    load_hhtools_csv,
    load_pm01_motion,
    stance_frames,
)

VIEWER_DIR = Path(__file__).resolve().parent / "review_viewer"
SET_NOTES = {
    "pm01_csv": "final clips, source frame rate",
    "pm01_hhtools": "raw hhtools export, before grounding and refinement",
    "pm01_refined": "output of refine_motions.py (also in the final sets)",
}
GEOM_TYPES = {
    mujoco.mjtGeom.mjGEOM_SPHERE: "sphere",
    mujoco.mjtGeom.mjGEOM_CAPSULE: "capsule",
    mujoco.mjtGeom.mjGEOM_CYLINDER: "cylinder",
    mujoco.mjtGeom.mjGEOM_BOX: "box",
    mujoco.mjtGeom.mjGEOM_MESH: "mesh",
}
CLIP_CACHE_SIZE = 6


def pack(header: dict, arrays: dict[str, np.ndarray]) -> bytes:
    """``[uint32 header length][JSON header, space padded to 4 bytes][array bytes...]``.

    The header gains ``arrays: {name: {dtype, shape, offset}}``; offsets count from
    the end of the header and are multiples of 4.
    """
    index, blobs, offset = {}, [], 0
    for name, array in arrays.items():
        array = np.ascontiguousarray(array)
        if array.dtype not in (np.float32, np.uint32):
            raise TypeError(f"{name}: {array.dtype}")
        index[name] = {"dtype": array.dtype.name, "shape": list(array.shape), "offset": offset}
        blobs.append(array.tobytes())
        offset += array.nbytes
    text = json.dumps({**header, "arrays": index}, separators=(",", ":")).encode("utf-8")
    text += b" " * (-len(text) % 4)
    return np.uint32(len(text)).tobytes() + text + b"".join(blobs)


def wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    return q[..., [1, 2, 3, 0]]


def quat_mul_wxyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=-1,
    )


def model_payload(
    robot: Robot, joint_names: tuple[str, ...], speed_limits: np.ndarray | None, visual: Robot | None = None
) -> tuple[dict, dict]:
    """Geometry of one robot: per-body geoms (visual and collision) and the meshes they use.

    Bodies, joints and collision geoms are ``robot``'s.  With ``visual``, the visual geoms are
    taken from that model instead and attached to ``robot``'s bodies of the same name (the
    URDF import fuses the base link into "world"; it is LINK_BASE in the floating-base model).
    """
    meshes: list[dict] = []
    mesh_index: dict[tuple[int, int], int] = {}
    verts, faces = [], []
    vert_count = face_count = 0
    geoms = []
    sources = [(robot, True, lambda b: b)]
    if visual is not None:
        names = [robot.model.body(b).name for b in range(robot.model.nbody)]
        base = names.index("LINK_BASE")
        sources.append((visual, False, lambda b: base if visual.model.body(b).name == "world" else names.index(visual.model.body(b).name)))
    for source_id, (source, want_collision, body_of) in enumerate(sources):
        m = source.model
        for g in range(m.ngeom):
            kind = GEOM_TYPES.get(mujoco.mjtGeom(m.geom_type[g]))
            collision = bool(m.geom_contype[g] or m.geom_conaffinity[g])
            if kind is None or (visual is not None and collision != want_collision):
                continue
            entry = {
                "body": int(body_of(int(m.geom_bodyid[g]))),
                "type": kind,
                "size": [float(v) for v in m.geom_size[g]],
                "pos": [float(v) for v in m.geom_pos[g]],
                "quat": [float(v) for v in wxyz_to_xyzw(m.geom_quat[g])],
                "collision": collision,
            }
            if kind == "mesh":
                key = (source_id, int(m.geom_dataid[g]))
                if key not in mesh_index:
                    mesh = key[1]
                    v0, vn = int(m.mesh_vertadr[mesh]), int(m.mesh_vertnum[mesh])
                    f0, fn = int(m.mesh_faceadr[mesh]), int(m.mesh_facenum[mesh])
                    mesh_index[key] = len(meshes)
                    meshes.append({"vert": [vert_count, vn], "face": [face_count, fn]})
                    verts.append(m.mesh_vert[v0 : v0 + vn].astype(np.float32))
                    faces.append(m.mesh_face[f0 : f0 + fn].astype(np.uint32))
                    vert_count += vn
                    face_count += fn
                entry["mesh"] = mesh_index[key]
            geoms.append(entry)
    m = robot.model
    header = {
        "bodies": [m.body(b).name for b in range(m.nbody)],
        "geoms": geoms,
        "meshes": meshes,
        "joints": list(joint_names),
        "limits": [[float(lo), float(hi)] for lo, hi in robot.limits],
        "speed_limits": None if speed_limits is None else [float(v) for v in speed_limits],
    }
    arrays = {
        "verts": np.concatenate(verts) if verts else np.zeros((0, 3), np.float32),
        "faces": np.concatenate(faces) if faces else np.zeros((0, 3), np.uint32),
    }
    return header, arrays


def body_poses(robot: Robot, joint_q: np.ndarray) -> np.ndarray:
    """World pose of every body per frame: (F, nbody, 7) = position + xyzw quaternion."""
    d = robot.data
    out = np.empty((joint_q.shape[0], robot.model.nbody, 7), dtype=np.float32)
    for f, row in enumerate(joint_q):
        root_r, root_p = robot.set_pose(row)
        quat = d.xquat
        if not robot.floating:
            q = row[3:7] / np.linalg.norm(row[3:7])
            quat = quat_mul_wxyz(np.array([q[3], q[0], q[1], q[2]]), quat)
        out[f, :, :3] = d.xpos @ root_r.T + root_p
        out[f, :, 3:] = wxyz_to_xyzw(quat)
    return out


class Reviewer:
    def __init__(self, work: Path) -> None:
        self.work = work
        self.g1 = Robot(G1_MJCF, G1_JOINT_NAMES)
        self.pm = Robot(PM01_SDK_COLLISION, PM01_JOINT_NAMES)  # poses, numbers, collision shapes
        self.pm_visual = Robot(PM01_URDF, PM01_JOINT_NAMES)  # meshes to draw
        self.c_g1, self.c_pm = calibration_rotations(self.g1, self.pm)
        self.lock = threading.Lock()  # one MjData per robot: FK is not re-entrant
        self._models: bytes | None = None
        self._clips: OrderedDict[tuple, bytes] = OrderedDict()

    # --- catalogue ---------------------------------------------------------
    def source_path(self, group: str, clip: str) -> Path:
        return self.work / "g1_source" / group / f"{clip}.csv"

    def catalogue(self) -> dict:
        sets = []
        for folder in sorted(p for p in self.work.glob("pm01_*") if p.is_dir()):
            groups: dict[str, list[str]] = {}
            for path in sorted(folder.glob("*/*")):
                if path.suffix not in (".csv", ".npz"):
                    continue
                if self.source_path(path.parent.name, path.stem).is_file():
                    groups.setdefault(path.parent.name, []).append(path.stem)
            if groups:
                note = SET_NOTES.get(folder.name) or ("resampled NPZ set" if "npz" in folder.name else "")
                sets.append({"name": folder.name, "note": note, "groups": groups})
        return {"sets": sets}

    def clip_path(self, set_name: str, group: str, clip: str) -> Path:
        """Resolve a request against the catalogue, so only listed files are ever opened."""
        for entry in self.catalogue()["sets"]:
            if entry["name"] == set_name and clip in entry["groups"].get(group, []):
                for suffix in (".csv", ".npz"):
                    path = self.work / set_name / group / f"{clip}{suffix}"
                    if path.is_file():
                        return path
        raise KeyError(f"{set_name}/{group}/{clip}")

    # --- payloads ----------------------------------------------------------
    def models(self) -> bytes:
        with self.lock:
            if self._models is None:
                header, arrays = {}, {}
                for key, robot, names, speed, visual in (
                    ("pm01", self.pm, PM01_JOINT_NAMES, PM01_SPEED_LIMITS, self.pm_visual),
                    ("g1", self.g1, G1_JOINT_NAMES, None, None),
                ):
                    h, a = model_payload(robot, names, speed, visual)
                    header[key] = h
                    arrays.update({f"{key}_{name}": value for name, value in a.items()})
                self._models = pack(header, arrays)
            return self._models

    def clip(self, set_name: str, group: str, clip: str) -> bytes:
        path = self.clip_path(set_name, group, clip)
        src_path = self.source_path(group, clip)
        key = (str(path), path.stat().st_mtime_ns, src_path.stat().st_mtime_ns)
        with self.lock:
            if key in self._clips:
                self._clips.move_to_end(key)
                return self._clips[key]
            payload = self._build_clip(path, src_path, set_name, group, clip)
            self._clips[key] = payload
            while len(self._clips) > CLIP_CACHE_SIZE:
                self._clips.popitem(last=False)
            return payload

    def _build_clip(self, path: Path, src_path: Path, set_name: str, group: str, clip: str) -> bytes:
        src, src_fps, src_names, _ = load_hhtools_csv(src_path)
        if src_names != G1_JOINT_NAMES:
            raise ValueError(f"{src_path}: unexpected joint order")
        if path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as data:
                fps = float(data["fps"])
                out = np.concatenate([data["root_pos"], data["root_rot"], data["dof_pos"]], axis=1).astype(np.float64)
            if abs(fps - src_fps) > 1e-9:  # bring the source to the NPZ rate the way the NPZ was made
                src = resample_like_csv_to_npz(src, src_fps, fps)
        else:
            out, fps, out_names, _ = load_pm01_motion(path, src_fps)
            if out_names != PM01_JOINT_NAMES:
                raise ValueError(f"{path}: unexpected joint order")

        notes = []
        if abs(fps - src_fps) > 1e-9 and path.suffix != ".npz":
            notes.append(f"PM01 clip is {fps:g} Hz but the G1 source is {src_fps:g} Hz: frames are not aligned")
        if out.shape[0] != src.shape[0]:
            notes.append(f"PM01 clip has {out.shape[0]} frames, G1 source {src.shape[0]}: showing the common part")

        metrics = frame_metrics(self.g1, self.pm, self.c_g1, self.c_pm, src, out)
        n = metrics["frames"]
        src, out = src[:n], out[:n]
        _, foot_speed, planted = stance_frames(metrics, 1.0 / fps)

        pairs: dict[str, list[str]] = {}
        pair_index: dict[str, list[int]] = {}
        for robot in ("pm", "g1"):
            names: list[str] = []
            index = []
            for pair in metrics["self_pair"][robot]:
                if pair is None:
                    index.append(-1)
                    continue
                label = " × ".join(pair)
                if label not in names:
                    names.append(label)
                index.append(names.index(label))
            pairs[robot], pair_index[robot] = names, index

        feet = list(FEET)  # L, R
        header = {
            "set": set_name,
            "group": group,
            "clip": clip,
            "file": str(path.relative_to(self.work)),
            "fps": fps,
            "frames": n,
            "segments": list(SEGMENTS),
            "limb_segments": list(metrics["dir_err"]),
            "feet": feet,
            "self_pairs": pairs,
            "self_pair_index": pair_index,
            "notes": notes,
        }
        f32 = np.float32
        arrays = {
            "pm_pose": body_poses(self.pm, out),
            "g1_pose": body_poses(self.g1, src),
            "pm_q": out.astype(f32),
            "g1_q": src.astype(f32),
            "seg_err": np.stack([metrics["seg_err"][s] for s in SEGMENTS], axis=1).astype(f32),
            "dir_err": np.stack(list(metrics["dir_err"].values()), axis=1).astype(f32),
            # columns: PM01 L, PM01 R, G1 L, G1 R
            "sole": np.stack([metrics["sole"][r][k] for r in ("pm", "g1") for k in feet], axis=1).astype(f32),
            "foot_speed": np.stack([foot_speed[r][k] for r in ("pm", "g1") for k in feet], axis=1).astype(f32),
            "planted": np.stack([planted[k] for k in feet], axis=1).astype(f32),
            # columns: PM01, G1
            "body_low": np.stack([metrics["body_low"][r] for r in ("pm", "g1")], axis=1).astype(f32),
            "self_pen": np.stack([metrics["self_pen"][r] for r in ("pm", "g1")], axis=1).astype(f32),
        }
        return pack(header, arrays)


def make_handler(reviewer: Reviewer) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args) -> None:  # quiet: only errors reach the terminal
            if args and str(args[1])[:1] in ("4", "5"):
                sys.stderr.write(f"{self.address_string()} {fmt % args}\n")

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _error(self, status: int, message: str) -> None:
            self._send(status, json.dumps({"error": message}).encode("utf-8"), "application/json")

        def do_GET(self) -> None:  # noqa: N802 - http.server API
            url = urlparse(self.path)
            try:
                if url.path == "/api/catalogue":
                    body = json.dumps(reviewer.catalogue()).encode("utf-8")
                    return self._send(HTTPStatus.OK, body, "application/json")
                if url.path == "/api/models":
                    return self._send(HTTPStatus.OK, reviewer.models(), "application/octet-stream")
                if url.path == "/api/clip":
                    query = parse_qs(url.query)
                    args = [query.get(k, [""])[0] for k in ("set", "group", "clip")]
                    return self._send(HTTPStatus.OK, reviewer.clip(*args), "application/octet-stream")
                return self._static(url.path)
            except KeyError as exc:
                self._error(HTTPStatus.NOT_FOUND, f"not found: {exc}")
            except BrokenPipeError:
                pass
            except Exception as exc:  # noqa: BLE001 - report to the page instead of dropping the connection
                self._error(HTTPStatus.INTERNAL_SERVER_ERROR, f"{type(exc).__name__}: {exc}")

        def _static(self, url_path: str) -> None:
            relative = "index.html" if url_path in ("", "/") else url_path.lstrip("/")
            path = (VIEWER_DIR / relative).resolve()
            if VIEWER_DIR not in path.parents or not path.is_file():
                raise KeyError(url_path)
            content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            if path.suffix in (".js", ".mjs"):
                content_type = "text/javascript"
            if content_type.startswith("text/"):
                content_type += "; charset=utf-8"
            self._send(HTTPStatus.OK, path.read_bytes(), content_type)

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1", help="Address to listen on (default: this machine only).")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true", help="Do not open the viewer in the default browser.")
    args = parser.parse_args()

    reviewer = Reviewer(WORK)
    sets = reviewer.catalogue()["sets"]
    if not sets:
        raise SystemExit(f"no PM01 motion sets with matching g1_source clips under {WORK}")
    try:
        server = ThreadingHTTPServer((args.host, args.port), make_handler(reviewer))
    except OSError as exc:
        raise SystemExit(f"cannot listen on {args.host}:{args.port} ({exc.strerror}); try --port <other>") from exc
    server.daemon_threads = True

    url = f"http://{'127.0.0.1' if args.host in ('0.0.0.0', '') else args.host}:{args.port}/"
    for entry in sets:
        print(f"{entry['name']:16s} {sum(len(v) for v in entry['groups'].values()):3d} clips  {entry['note']}")
    print(f"\nPM01 motion review: {url}   (Ctrl+C to stop)", flush=True)
    if not args.no_browser:
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
