# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""The in-memory V2E world: camera track, static geometry, rigid object tracks and events.

The same model describes a reconstruction, a run exported from an engine and synthetic
truth, so the scorer can compare any two of them. Arrays are indexed by video frame
``0 .. frame_count - 1``; time is ``frame / fps``. Unknown poses are NaN rows.

Each object has a hull in its own body frame (x forward, y left, z up) and a pose per frame.
Position metrics use the hull's bounding-box centre (:meth:`ObjectTrack.centre_track`), so they
do not depend on where a producer put the body origin.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
from scipy.spatial.transform import Rotation

from openbeam_v2e.schema.common import Measured, default_symmetry
from openbeam_v2e.world.camera import CameraTrack
from openbeam_v2e.world.geometry import mesh_bounds, normalize_quats, quat_to_matrix

__all__ = ["StaticMesh", "ObjectTrack", "EventInterval", "World", "intervals_from_mask"]


def intervals_from_mask(mask: Any) -> list[tuple[int, int]]:
    """Inclusive ``(first, last)`` runs of True in a boolean array."""
    m = np.asarray(mask, dtype=bool).astype(np.int8)
    edges = np.diff(np.concatenate([[0], m, [0]]))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1) - 1
    return [(int(a), int(b)) for a, b in zip(starts, ends)]


@dataclass
class StaticMesh:
    """Static geometry in the world frame (ground, ramps, walls, barriers)."""

    mesh_id: int
    vertices: np.ndarray
    faces: np.ndarray
    name: str = ""
    role: str = "collision"
    sigma_m: float | None = None
    observed: bool = True

    def __post_init__(self) -> None:
        self.vertices = np.asarray(self.vertices, dtype=float).reshape(-1, 3)
        self.faces = np.asarray(self.faces, dtype=np.int32).reshape(-1, 3)

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return mesh_bounds(self.vertices)


@dataclass
class ObjectTrack:
    """One rigid dynamic object: hull in its body frame and a pose per frame."""

    id: str
    cls: str
    hull_vertices: np.ndarray                     # (V, 3) body frame
    hull_faces: np.ndarray                        # (T, 3)
    position: np.ndarray                          # (F, 3) world, NaN when unknown
    quat_xyzw: np.ndarray                         # (F, 4) body-to-world, NaN when unknown
    visible: np.ndarray | None = None             # (F,) observed in the video (defaults to known)
    sigma_position_m: np.ndarray | None = None    # (F,) one sigma, NaN when unknown
    sigma_attitude_rad: np.ndarray | None = None  # (F,)
    dims_m: dict[str, Measured] = field(default_factory=dict)
    symmetry: str | None = None
    mass_kg: Measured | None = None
    mesh_id: int | None = None
    annotation: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.hull_vertices = np.asarray(self.hull_vertices, dtype=float).reshape(-1, 3)
        self.hull_faces = np.asarray(self.hull_faces, dtype=np.int32).reshape(-1, 3)
        self.position = np.asarray(self.position, dtype=float).reshape(-1, 3)
        self.quat_xyzw = normalize_quats(np.asarray(self.quat_xyzw, dtype=float).reshape(-1, 4))
        F = len(self.position)
        if len(self.quat_xyzw) != F:
            raise ValueError(f"object {self.id!r}: {F} positions but {len(self.quat_xyzw)} attitudes")
        known = self.known
        self.position[~known] = np.nan
        self.quat_xyzw[~known] = np.nan
        self.visible = known.copy() if self.visible is None else np.asarray(self.visible, dtype=bool).reshape(F)
        for name in ("sigma_position_m", "sigma_attitude_rad"):
            value = getattr(self, name)
            setattr(self, name, np.full(F, np.nan) if value is None else np.asarray(value, dtype=float).reshape(F))

    # ------------------------------------------------------------------ basic properties
    @property
    def frame_count(self) -> int:
        return len(self.position)

    @property
    def known(self) -> np.ndarray:
        return np.all(np.isfinite(self.position), axis=1) & np.all(np.isfinite(self.quat_xyzw), axis=1)

    @property
    def symmetry_class(self) -> str:
        return self.symmetry or default_symmetry(self.cls)

    def known_interval(self) -> tuple[int, int] | None:
        idx = np.flatnonzero(self.known)
        return (int(idx[0]), int(idx[-1])) if len(idx) else None

    def rotations(self) -> np.ndarray:
        """(F, 3, 3) body-to-world rotations, NaN where unknown."""
        return quat_to_matrix(self.quat_xyzw)

    def hull_centre_body(self) -> np.ndarray:
        lo, hi = mesh_bounds(self.hull_vertices)
        return 0.5 * (lo + hi)

    def hull_extents(self) -> np.ndarray:
        lo, hi = mesh_bounds(self.hull_vertices)
        return hi - lo

    def centre_track(self) -> np.ndarray:
        """(F, 3) world position of the hull's bounding-box centre."""
        return self.position + np.einsum("fij,j->fi", self.rotations(), self.hull_centre_body())

    def hull_world(self, frame: int) -> np.ndarray | None:
        if not self.known[frame]:
            return None
        R = quat_to_matrix(self.quat_xyzw[frame])
        return self.hull_vertices @ R.T + self.position[frame]

    def lowest_point(self) -> np.ndarray:
        """(F,) world height of the lowest hull vertex, NaN where unknown."""
        out = np.full(self.frame_count, np.nan)
        R = self.rotations()
        known = self.known
        if np.any(known):
            z = np.einsum("fj,vj->fv", R[known][:, 2, :], self.hull_vertices) + self.position[known][:, 2:3]
            out[known] = z.min(axis=1)
        return out

    def velocity(self, fps: float, centre: bool = True) -> np.ndarray:
        """(F, 3) velocity by central differences over known frames (one-sided at the ends)."""
        p = self.centre_track() if centre else self.position
        return _differentiate(p, fps)

    def angular_velocity(self, fps: float) -> np.ndarray:
        """(F, 3) world-frame angular velocity (rad/s) from consecutive attitudes."""
        R = self.rotations()
        F = self.frame_count
        out = np.full((F, 3), np.nan)
        known = self.known
        for i in range(F):
            a, b = i - 1, i + 1
            if a < 0 or not known[a]:
                a = i
            if b >= F or not known[b]:
                b = i
            if a == b or not (known[a] and known[b]):
                continue
            delta = R[b] @ R[a].T
            out[i] = Rotation.from_matrix(delta).as_rotvec() * fps / (b - a)
        return out

    def speed(self, fps: float) -> np.ndarray:
        return np.linalg.norm(self.velocity(fps), axis=1)


def _differentiate(p: np.ndarray, fps: float) -> np.ndarray:
    F = len(p)
    out = np.full_like(p, np.nan)
    known = np.all(np.isfinite(p), axis=1)
    for a, b in intervals_from_mask(known):
        if b > a:
            out[a:b + 1] = np.gradient(p[a:b + 1], axis=0) * fps
        else:
            out[a] = 0.0 if F == 1 else np.nan
    return out


@dataclass(frozen=True)
class EventInterval:
    name: str
    frame: tuple[int, int]
    kind: str = "custom"
    actors: tuple[str, ...] = ()
    evidence: str = ""

    @property
    def centre(self) -> float:
        return 0.5 * (self.frame[0] + self.frame[1])

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {"name": self.name, "frame": [int(self.frame[0]), int(self.frame[1])], "kind": self.kind}
        if self.actors:
            out["actors"] = list(self.actors)
        if self.evidence:
            out["evidence"] = self.evidence
        return out

    @classmethod
    def from_json(cls, raw: Mapping[str, Any]) -> "EventInterval":
        return cls(raw["name"], (int(raw["frame"][0]), int(raw["frame"][1])), raw.get("kind", "custom"),
                   tuple(raw.get("actors", ())), raw.get("evidence", ""))


@dataclass
class World:
    """A reconstruction, a run export or synthetic truth on one video timeline."""

    fps: float
    frame_count: int
    camera_track: CameraTrack
    statics: list[StaticMesh] = field(default_factory=list)
    objects: list[ObjectTrack] = field(default_factory=list)
    events: list[EventInterval] = field(default_factory=list)
    kind: str = "reconstruction"
    models: list[dict[str, Any]] = field(default_factory=list)
    residuals: dict[str, dict[str, float]] = field(default_factory=lambda: {"fitted": {}, "held_out": {}})
    camera_sigma_px: np.ndarray | float | None = None
    outlines: dict[str, dict[int, list[np.ndarray]]] = field(default_factory=dict)
    source_video: dict[str, str] = field(default_factory=dict)
    notes: str = ""

    def __post_init__(self) -> None:
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        if self.camera_track.frame_count != self.frame_count:
            raise ValueError(f"camera track has {self.camera_track.frame_count} frames, world has {self.frame_count}")
        for obj in self.objects:
            if obj.frame_count != self.frame_count:
                raise ValueError(f"object {obj.id!r} has {obj.frame_count} frames, world has {self.frame_count}")
        ids = [o.id for o in self.objects]
        if len(set(ids)) != len(ids):
            raise ValueError("object ids must be unique")

    @property
    def image_size(self) -> tuple[int, int]:
        return self.camera_track.width, self.camera_track.height

    def times(self) -> np.ndarray:
        return np.arange(self.frame_count) / self.fps

    def object(self, object_id: str) -> ObjectTrack:
        for obj in self.objects:
            if obj.id == object_id:
                return obj
        raise KeyError(f"world has no object {object_id!r}")

    def has_object(self, object_id: str) -> bool:
        return any(o.id == object_id for o in self.objects)

    def event(self, name: str) -> EventInterval:
        for ev in self.events:
            if ev.name == name:
                return ev
        raise KeyError(f"world has no event {name!r}")

    def static_geometry(self, roles: tuple[str, ...] = ("collision", "drivable")) -> tuple[np.ndarray, np.ndarray]:
        """All static meshes with one of ``roles`` merged into one ``(vertices, faces)``."""
        verts, faces, offset = [], [], 0
        for s in self.statics:
            if s.role not in roles:
                continue
            verts.append(s.vertices)
            faces.append(s.faces + offset)
            offset += len(s.vertices)
        if not verts:
            return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int32)
        return np.vstack(verts), np.vstack(faces).astype(np.int32)
