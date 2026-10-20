# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""Reader, writer and checker for the 4DCodeBench world-package layout.

Layout (units are metres, Z-up; every frame-indexed array shares one timeline of F frames)::

    camera.json          {"intrinsics": K (3x3), "extrinsic": E (4x4, camera-to-world)}
    render.mp4           optional here; F frames when present
    meshes/<id>.npz      vertices_TTTT (V, 3) float32 and faces_TTTT (T, 3) int32 per frame of one
                         contiguous interval; TTTT is the zero-padded frame number
    dynamics/<name>.npz  pos (F, N, 3) float32 (each column finite over one interval, NaN
                         elsewhere), ids (M,) uint16 naming the meshes that show it, optional
                         faces (T, 3) or tets (K, 4) int32
    solver/manifest.json optional, for identity-less liquids

A mesh that no dynamics archive lists is static and must be identical in every frame of its
interval. This module was written from the published description of the layout and checks
the rules above; it reads arrays with NumPy (``allow_pickle=False``) and writes byte-for-byte
reproducible archives (fixed timestamps, sorted members).
"""

from __future__ import annotations

import io
import json
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

__all__ = [
    "FormatError",
    "Violation",
    "MeshSequence",
    "MotionArchive",
    "Package",
    "read_package",
    "write_package",
    "check_package",
    "read_npz",
    "write_npz",
    "frame_tag",
]

_MESH_FILE = re.compile(r"^([1-9][0-9]{0,9})\.npz$")
_FRAME_MEMBER = re.compile(r"^(vertices|faces)_([0-9]{4,10})$")
MAX_MESSAGES_PER_FILE = 20


class FormatError(ValueError):
    """A file cannot be read as part of a world package."""


@dataclass(frozen=True)
class Violation:
    where: str
    message: str

    def __str__(self) -> str:
        return f"{self.where}: {self.message}"


def frame_tag(frame: int) -> str:
    """Member suffix of a frame: at least four digits, zero-padded."""
    return f"{int(frame):04d}"


@dataclass
class MeshSequence:
    """``meshes/<id>.npz``: one vertex/face pair per frame of ``[first_frame, first_frame + n)``."""

    mesh_id: int
    first_frame: int
    vertices: list[np.ndarray]
    faces: list[np.ndarray]

    @property
    def last_frame(self) -> int:
        return self.first_frame + len(self.vertices) - 1

    @classmethod
    def static(cls, mesh_id: int, vertices: Any, faces: Any, frame_count: int, first_frame: int = 0) -> "MeshSequence":
        v = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
        f = np.asarray(faces, dtype=np.int32).reshape(-1, 3)
        n = frame_count - first_frame
        return cls(mesh_id, first_frame, [v] * n, [f] * n)

    def at(self, frame: int) -> tuple[np.ndarray, np.ndarray] | None:
        i = frame - self.first_frame
        if 0 <= i < len(self.vertices):
            return self.vertices[i], self.faces[i]
        return None


@dataclass
class MotionArchive:
    """``dynamics/<name>.npz``: material points over time, shown by the meshes in ``ids``."""

    name: str
    pos: np.ndarray                       # (F, N, 3) float32, NaN outside each column's lifetime
    ids: np.ndarray                       # (M,) uint16
    faces: np.ndarray | None = None       # (T, 3) int32
    tets: np.ndarray | None = None        # (K, 4) int32


@dataclass
class Package:
    intrinsics: np.ndarray                # (3, 3)
    extrinsic: np.ndarray                 # (4, 4) camera-to-world
    meshes: dict[int, MeshSequence] = field(default_factory=dict)
    dynamics: dict[str, MotionArchive] = field(default_factory=dict)
    solver_manifest: Mapping[str, Any] | None = None
    has_render: bool = False

    def frame_count(self) -> int:
        """F: the longest timeline any dynamics archive or mesh archive implies."""
        counts = [int(np.shape(d.pos)[0]) for d in self.dynamics.values() if np.ndim(d.pos) == 3]
        counts += [m.last_frame + 1 for m in self.meshes.values()]
        return max(counts, default=0)

    def dynamic_mesh_ids(self) -> set[int]:
        return {int(i) for d in self.dynamics.values() for i in d.ids}


# ----------------------------------------------------------------------------- NPZ

def read_npz(path: str | Path) -> dict[str, np.ndarray]:
    """Members of an ``.npz`` archive by name (without ``.npy``); never unpickles."""
    try:
        with zipfile.ZipFile(path) as zf:
            names = zf.namelist()
        bad = [n for n in names if not n.endswith(".npy")]
        if bad:
            raise FormatError(f"member {bad[0]!r} is not a .npy array")
        if len(set(names)) != len(names):
            raise FormatError("an archive member appears twice")
        with np.load(path, allow_pickle=False) as data:
            return {name: np.asarray(data[name]) for name in data.files}
    except (zipfile.BadZipFile, OSError, ValueError) as exc:
        if isinstance(exc, FormatError):
            raise
        raise FormatError(f"cannot read {Path(path).name}: {exc}") from None


def write_npz(path: str | Path, members: Mapping[str, np.ndarray], compressed: bool = True) -> Path:
    """Write a reproducible ``.npz``: members sorted, fixed timestamps and attributes."""
    buffer = io.BytesIO()
    method = zipfile.ZIP_DEFLATED if compressed else zipfile.ZIP_STORED
    with zipfile.ZipFile(buffer, "w", compression=method) as zf:
        for name in sorted(members):
            array_bytes = io.BytesIO()
            np.lib.format.write_array(array_bytes, np.ascontiguousarray(members[name]), allow_pickle=False)
            info = zipfile.ZipInfo(name + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = method
            info.create_system = 3
            info.external_attr = 0o644 << 16
            zf.writestr(info, array_bytes.getvalue())
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(buffer.getvalue())
    return target


# ----------------------------------------------------------------------------- read / write

def _read_camera(path: Path) -> tuple[np.ndarray, np.ndarray]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        K = np.asarray(raw["intrinsics"], dtype=float)
        E = np.asarray(raw["extrinsic"], dtype=float)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise FormatError(f"camera.json: {exc}") from None
    if K.shape != (3, 3) or E.shape != (4, 4):
        raise FormatError("camera.json: intrinsics must be 3x3 and extrinsic 4x4")
    return K, E


def read_package(directory: str | Path) -> Package:
    """Read a world package. Raises :class:`FormatError` on files that cannot be read;
    use :func:`check_package` for a full rule check."""
    root = Path(directory)
    if not root.is_dir():
        raise FormatError(f"{root} is not a directory")
    K, E = _read_camera(root / "camera.json")
    pkg = Package(K, E, has_render=(root / "render.mp4").is_file())
    for path in sorted((root / "meshes").glob("*.npz")):
        match = _MESH_FILE.match(path.name)
        if not match:
            raise FormatError(f"meshes/{path.name}: file name must be <positive integer>.npz")
        frames: dict[int, dict[str, np.ndarray]] = {}
        for name, array in read_npz(path).items():
            m = _FRAME_MEMBER.match(name)
            if not m or m.group(2) != frame_tag(int(m.group(2))):
                raise FormatError(f"meshes/{path.name}: unexpected member {name!r}")
            frames.setdefault(int(m.group(2)), {})[m.group(1)] = array
        if not frames:
            raise FormatError(f"meshes/{path.name}: holds no frames")
        order = sorted(frames)
        if order != list(range(order[0], order[-1] + 1)):
            raise FormatError(f"meshes/{path.name}: frames are not contiguous")
        try:
            pkg.meshes[int(match.group(1))] = MeshSequence(
                int(match.group(1)), order[0], [frames[t]["vertices"] for t in order], [frames[t]["faces"] for t in order])
        except KeyError:
            raise FormatError(f"meshes/{path.name}: a frame lacks vertices or faces") from None
    for path in sorted((root / "dynamics").glob("*.npz")):
        members = read_npz(path)
        if "pos" not in members or "ids" not in members:
            raise FormatError(f"dynamics/{path.name}: needs 'pos' and 'ids'")
        pkg.dynamics[path.stem] = MotionArchive(path.stem, members["pos"], members["ids"], members.get("faces"),
                                                  members.get("tets"))
    manifest = root / "solver" / "manifest.json"
    if manifest.is_file():
        try:
            pkg.solver_manifest = json.loads(manifest.read_text(encoding="utf-8"))
        except ValueError as exc:
            raise FormatError(f"solver/manifest.json: {exc}") from None
    return pkg


def _matrix_json(M: np.ndarray) -> list[list[float]]:
    return [[float(x) for x in row] for row in np.asarray(M, dtype=float)]


def write_package(pkg: Package, directory: str | Path, compressed: bool = True, check: bool = True) -> Path:
    """Write ``pkg`` into ``directory`` (existing ``meshes/`` and ``dynamics/`` archives are replaced).

    With ``check`` the package is validated first and nothing is written when it breaks a rule.
    """
    root = Path(directory)
    if check:
        problems = check_model(pkg)
        if problems:
            raise FormatError(f"refusing to write an invalid package: {problems[0]}")
    for sub in ("meshes", "dynamics"):
        (root / sub).mkdir(parents=True, exist_ok=True)
        for stale in (root / sub).glob("*.npz"):
            stale.unlink()
    camera = {"intrinsics": _matrix_json(pkg.intrinsics), "extrinsic": _matrix_json(pkg.extrinsic)}
    (root / "camera.json").write_text(json.dumps(camera, indent=2) + "\n", encoding="utf-8")
    for mesh_id, seq in sorted(pkg.meshes.items()):
        members: dict[str, np.ndarray] = {}
        for i, (v, f) in enumerate(zip(seq.vertices, seq.faces)):
            tag = frame_tag(seq.first_frame + i)
            members[f"vertices_{tag}"] = np.asarray(v, dtype="<f4").reshape(-1, 3)
            members[f"faces_{tag}"] = np.asarray(f, dtype="<i4").reshape(-1, 3)
        write_npz(root / "meshes" / f"{mesh_id}.npz", members, compressed)
    for name, d in sorted(pkg.dynamics.items()):
        members = {"pos": np.asarray(d.pos, dtype="<f4"), "ids": np.asarray(d.ids, dtype="<u2")}
        if d.faces is not None:
            members["faces"] = np.asarray(d.faces, dtype="<i4").reshape(-1, 3)
        if d.tets is not None:
            members["tets"] = np.asarray(d.tets, dtype="<i4").reshape(-1, 4)
        write_npz(root / "dynamics" / f"{name}.npz", members, compressed)
    if pkg.solver_manifest is not None:
        (root / "solver").mkdir(exist_ok=True)
        (root / "solver" / "manifest.json").write_text(json.dumps(pkg.solver_manifest, indent=2) + "\n", encoding="utf-8")
    return root


# ----------------------------------------------------------------------------- checks

class _Report:
    def __init__(self) -> None:
        self.items: list[Violation] = []
        self.counts: dict[str, int] = {}

    def add(self, where: str, message: str) -> None:
        key = where.split(":")[0]
        self.counts[key] = self.counts.get(key, 0) + 1
        if self.counts[key] <= MAX_MESSAGES_PER_FILE:
            self.items.append(Violation(where, message))

    def finish(self) -> list[Violation]:
        for key, n in sorted(self.counts.items()):
            if n > MAX_MESSAGES_PER_FILE:
                self.items.append(Violation(key, f"... {n - MAX_MESSAGES_PER_FILE} more problems"))
        return self.items


def _check_camera(K: np.ndarray, E: np.ndarray, report: _Report, tol: float = 1e-4) -> None:
    if K.shape != (3, 3) or not np.all(np.isfinite(K)):
        report.add("camera.json:intrinsics", "must be a 3x3 matrix of finite numbers")
    if E.shape != (4, 4) or not np.all(np.isfinite(E)):
        report.add("camera.json:extrinsic", "must be a 4x4 matrix of finite numbers")
        return
    R = E[:3, :3]
    if np.max(np.abs(R.T @ R - np.eye(3))) > tol:
        report.add("camera.json:extrinsic", "rotation block is not orthonormal")
    elif abs(np.linalg.det(R) - 1.0) > tol:
        report.add("camera.json:extrinsic", "rotation block is a reflection")
    if np.max(np.abs(E[3] - np.array([0.0, 0.0, 0.0, 1.0]))) > tol:
        report.add("camera.json:extrinsic", "bottom row must be [0, 0, 0, 1]")


def _column_lifetimes(pos: np.ndarray, where: str, report: _Report) -> None:
    finite = np.isfinite(pos)
    all_finite = finite.all(axis=2)
    all_nan = np.isnan(pos).all(axis=2)
    mixed = ~(all_finite | all_nan)
    if np.any(mixed):
        t, j = np.argwhere(mixed)[0]
        report.add(f"{where}:pos", f"column {j} frame {t} mixes finite and NaN components or holds an infinity")
    alive = all_finite.astype(np.int8)
    for j in np.flatnonzero(alive.sum(axis=0) == 0):
        report.add(f"{where}:pos", f"column {j} is never finite")
    starts = np.diff(np.vstack([np.zeros((1, alive.shape[1]), np.int8), alive]), axis=0) == 1
    for j in np.flatnonzero(starts.sum(axis=0) > 1):
        report.add(f"{where}:pos", f"column {j} is finite over more than one interval")


def check_model(pkg: Package, frame_count: int | None = None) -> list[Violation]:
    """Rule check of an in-memory package (everything except file naming and render presence)."""
    report = _Report()
    _check_camera(np.asarray(pkg.intrinsics, dtype=float), np.asarray(pkg.extrinsic, dtype=float), report)
    F = int(frame_count) if frame_count is not None else pkg.frame_count()
    dynamic_ids: set[int] = set()
    for name, d in sorted(pkg.dynamics.items()):
        where = f"dynamics/{name}.npz"
        pos = np.asarray(d.pos)
        if pos.dtype != np.float32 or pos.ndim != 3 or pos.shape[2] != 3:
            report.add(f"{where}:pos", f"must be (F, N, 3) float32, found {pos.shape} {pos.dtype}")
        else:
            if pos.shape[0] == 0 or pos.shape[1] == 0:
                report.add(f"{where}:pos", "needs at least one frame and one column")
            elif pos.shape[0] != F:
                report.add(f"{where}:pos", f"has {pos.shape[0]} frames, the sequence has {F}")
            _column_lifetimes(pos, where, report)
        ids = np.asarray(d.ids)
        if ids.dtype != np.uint16 or ids.ndim != 1 or len(ids) == 0:
            report.add(f"{where}:ids", f"must be (M,) uint16 with M >= 1, found {ids.shape} {ids.dtype}")
        for i in ids.reshape(-1).tolist():
            dynamic_ids.add(int(i))
            if int(i) not in pkg.meshes:
                report.add(f"{where}:ids", f"id {i} has no meshes/{i}.npz")
        if d.faces is not None and d.tets is not None:
            report.add(where, "holds both 'faces' and 'tets'")
        for member, arity in (("faces", 3), ("tets", 4)):
            conn = getattr(d, member)
            if conn is None:
                continue
            conn = np.asarray(conn)
            if conn.dtype != np.int32 or conn.ndim != 2 or conn.shape[1] != arity:
                report.add(f"{where}:{member}", f"must be (K, {arity}) int32, found {conn.shape} {conn.dtype}")
            elif pos.ndim == 3 and conn.size and (conn.min() < 0 or conn.max() >= pos.shape[1]):
                report.add(f"{where}:{member}", "indexes a column that does not exist")
    liquid = set()
    if pkg.solver_manifest is not None:
        liquid = {int(i) for i in pkg.solver_manifest.get("liquid_ids", []) if isinstance(i, (int, float))}
        for key in ("cache", "fps", "time_scale", "metres_per_unit", "domain_min", "frames", "liquid_ids"):
            if key not in pkg.solver_manifest:
                report.add("solver/manifest.json", f"missing {key!r}")
    for mesh_id, seq in sorted(pkg.meshes.items()):
        where = f"meshes/{mesh_id}.npz"
        if seq.last_frame >= F:
            report.add(where, f"frame {seq.last_frame} is not below F = {F}")
        for i, (v, f) in enumerate(zip(seq.vertices, seq.faces)):
            fw = f"{where}:{frame_tag(seq.first_frame + i)}"
            v, f = np.asarray(v), np.asarray(f)
            if v.dtype != np.float32 or v.ndim != 2 or v.shape[1] != 3:
                report.add(fw, f"vertices must be (V, 3) float32, found {v.shape} {v.dtype}")
                continue
            if f.dtype != np.int32 or f.ndim != 2 or f.shape[1] != 3:
                report.add(fw, f"faces must be (T, 3) int32, found {f.shape} {f.dtype}")
                continue
            if not np.all(np.isfinite(v)):
                report.add(fw, "vertices contain a non-finite value")
            if f.size and (f.min() < 0 or f.max() >= len(v)):
                report.add(fw, "a face indexes a vertex that does not exist")
        if mesh_id not in dynamic_ids and mesh_id not in liquid and len(seq.vertices) > 1:
            v0, f0 = np.asarray(seq.vertices[0]), np.asarray(seq.faces[0])
            for i in range(1, len(seq.vertices)):
                vi, fi = np.asarray(seq.vertices[i]), np.asarray(seq.faces[i])
                if vi.shape != v0.shape or fi.shape != f0.shape or not (np.array_equal(vi, v0) and np.array_equal(fi, f0)):
                    report.add(where, f"is static (no dynamics archive lists it) but changes at frame {seq.first_frame + i}")
                    break
    return report.finish()


def check_package(directory: str | Path, frame_count: int | None = None, require_render: bool = False) -> list[Violation]:
    """Every rule violation of the package in ``directory`` (empty when it is valid)."""
    root = Path(directory)
    problems: list[Violation] = []
    if require_render and not (root / "render.mp4").is_file():
        problems.append(Violation("render.mp4", "missing"))
    for sub in ("meshes", "dynamics"):
        if not (root / sub).is_dir():
            problems.append(Violation(f"{sub}/", "missing"))
        else:
            for entry in sorted((root / sub).iterdir()):
                if not entry.is_file() or entry.suffix != ".npz":
                    problems.append(Violation(f"{sub}/{entry.name}", "is not an .npz archive"))
    try:
        pkg = read_package(root)
    except FormatError as exc:
        return problems + [Violation("package", str(exc))]
    return problems + check_model(pkg, frame_count)
