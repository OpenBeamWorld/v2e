# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""Static geometry of benchmark cases, in two consistent forms.

Each level primitive of :mod:`openbeam_v2e.bench.families` becomes:

* **visual parts**: convex polytopes (vertices in world coordinates) that the renderer draws, the
  truth stores as static meshes, and event detection measures contacts against;
* **collision boxes**: oriented boxes that box-based connectors (the ``fake`` test connector)
  collide with.

The two agree on every surface an object can touch. A sloped segment is drawn as a closed prism
standing on the ground but collides as a thick plank whose top face is the slope; the plank's
underside is never reachable from above.

Ramps are a run of straight segments whose slope steepens toward the top edge (a polygonal
transition, as on real launch ramps), so a vehicle with ordinary ground clearance rolls onto them
instead of striking the foot. The take-off angle is the last segment's angle.

Conventions: Z-up, metres, degrees in parameters. Ramps and trip ramps rise toward +X; an incline
rises toward -X (its top is on the -X side).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from openbeam_v2e.bench.families import Static

__all__ = ["CollisionBox", "StaticSolid", "solid_for", "solids_for", "ramp_profile", "PLANK_THICKNESS_M"]

PLANK_THICKNESS_M = 0.4


@dataclass(frozen=True)
class CollisionBox:
    center: np.ndarray
    quat_xyzw: np.ndarray
    half_extents: np.ndarray


@dataclass(frozen=True)
class StaticSolid:
    """A static primitive as visual convex parts and collision boxes (no boxes for the ground plane)."""

    static_id: str
    kind: str
    parts: tuple[np.ndarray, ...]                 # each (N, 3), world coordinates, convex
    boxes: tuple[CollisionBox, ...] = ()
    observed: bool = True
    profile: tuple[tuple[float, float], ...] = field(default=())  # (x, z) top-surface profile in the local frame

    @property
    def vertices(self) -> np.ndarray:
        return np.concatenate(self.parts)

    @property
    def top_z(self) -> float:
        return float(self.vertices[:, 2].max())

    def fake_primitives(self, mu: float | None = None) -> list[dict[str, Any]]:
        """The ``fake`` connector's static primitives (empty for the ground plane)."""
        out = []
        for k, box in enumerate(self.boxes):
            prim: dict[str, Any] = {
                "name": self.static_id,
                "kind": "box",
                "center": [float(v) for v in box.center],
                "quat_xyzw": [float(v) for v in box.quat_xyzw],
                "half_extents": [float(v) for v in box.half_extents],
            }
            if mu is not None:
                prim["mu"] = float(mu)
            out.append(prim)
        return out

    def meshes(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """One outward-wound triangle mesh (vertices, faces) per convex part."""
        out = []
        for verts in self.parts:
            hull = ConvexHull(verts)
            tris = []
            for simplex, eq in zip(hull.simplices, hull.equations):
                a, b, c = verts[simplex]
                if np.dot(np.cross(b - a, c - a), eq[:3]) < 0:
                    simplex = simplex[[0, 2, 1]]
                tris.append(simplex)
            out.append((verts.astype(np.float32), np.asarray(tris, dtype=np.int32)))
        return out


def _box_vertices(center: np.ndarray, R: np.ndarray, half: np.ndarray) -> np.ndarray:
    corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float) * half
    return corners @ R.T + center


def _yaw(deg: float) -> np.ndarray:
    return Rotation.from_euler("z", deg, degrees=True).as_matrix()


def ramp_profile(segments: Sequence[Sequence[float]]) -> list[tuple[float, float]]:
    """Top-surface profile ``[(x, z), ...]`` of a ramp from ``[(slope_length_m, angle_deg), ...]``.

    The profile starts at ``(0, 0)`` and rises toward +X.
    """
    pts = [(0.0, 0.0)]
    for length, angle in segments:
        x, z = pts[-1]
        a = math.radians(float(angle))
        pts.append((x + float(length) * math.cos(a), z + float(length) * math.sin(a)))
    return pts


def _profile_solid(static: Static, profile: Sequence[tuple[float, float]], width: float,
                   x_offset: float, mirror: bool = False) -> StaticSolid:
    """Prisms under each profile segment, plus one collision plank per sloped segment.

    ``profile`` x values are shifted by ``x_offset`` into the primitive's local frame; with
    ``mirror`` the profile is reflected (x -> -x) so the surface rises toward -X.
    """
    yaw = _yaw(static.yaw_deg)
    origin = np.asarray(static.position_m, float)
    hw = width / 2.0
    sign = -1.0 if mirror else 1.0
    parts, boxes = [], []
    local_profile = [(sign * (x + x_offset), z) for x, z in profile]
    for (xa, za), (xb, zb) in zip(local_profile[:-1], local_profile[1:]):
        local = np.array([[x, y, z] for x, ztop in ((xa, za), (xb, zb)) for y in (-hw, hw) for z in (0.0, ztop)])
        parts.append(local @ yaw.T + origin)
        dx, dz = xb - xa, zb - za
        length = math.hypot(dx, dz)
        if length <= 1e-9:
            continue
        pitch = -math.atan2(dz, dx)  # rotation about +Y that maps +X onto the segment direction
        R_local = Rotation.from_euler("y", pitch).as_matrix()
        normal = R_local @ np.array([0.0, 0.0, 1.0])
        if normal[2] < 0:  # segment runs toward -X: flip so the plank's top faces up
            R_local = Rotation.from_euler("y", pitch + math.pi).as_matrix()
            normal = R_local @ np.array([0.0, 0.0, 1.0])
        mid = np.array([(xa + xb) / 2.0, 0.0, (za + zb) / 2.0])
        centre_local = mid - normal * PLANK_THICKNESS_M / 2.0
        R = yaw @ R_local
        boxes.append(CollisionBox(centre_local @ yaw.T + origin, Rotation.from_matrix(R).as_quat(),
                                  np.array([length / 2.0, hw, PLANK_THICKNESS_M / 2.0])))
    return StaticSolid(static.id, static.kind, tuple(parts), tuple(boxes), static.observed,
                       tuple((float(x), float(z)) for x, z in local_profile))


def _upright_box(static: Static, size: Sequence[float]) -> StaticSolid:
    yaw = _yaw(static.yaw_deg)
    half = np.asarray(size, float) / 2.0
    centre = np.asarray(static.position_m, float) + yaw @ np.array([0.0, 0.0, half[2]])
    return StaticSolid(static.id, static.kind, (_box_vertices(centre, yaw, half),),
                       (CollisionBox(centre, Rotation.from_matrix(yaw).as_quat(), half),), static.observed)


def solid_for(static: Static) -> StaticSolid:
    """Visual parts and collision boxes of one level primitive."""
    p = static.params
    if static.kind == "ground":
        h = float(p.get("size_m", 200.0)) / 2.0
        origin = np.asarray(static.position_m, float)
        verts = np.array([[sx * h, sy * h, z] for sx in (-1, 1) for sy in (-1, 1) for z in (-0.01, 0.0)]) + origin
        return StaticSolid(static.id, "ground", (verts,), (), static.observed)
    if static.kind == "barrier":
        return _upright_box(static, (p["depth_m"], p["width_m"], p["height_m"]))
    if static.kind == "slab":
        return _upright_box(static, (p["size_x_m"], p["size_y_m"], p["thickness_m"]))
    if static.kind == "post":
        return _upright_box(static, (p["size_m"], p["size_m"], p["height_m"]))
    if static.kind in ("wedge_ramp", "trip_ramp"):
        profile = ramp_profile(p["segments"])
        run = profile[-1][0]
        # The primitive's position is the centre of its footprint at ground level.
        return _profile_solid(static, profile, float(p["width_m"]), -run / 2.0)
    if static.kind == "incline":
        a = math.radians(float(p["angle_deg"]))
        length = float(p["length_m"])
        profile = [(0.0, 0.0), (length * math.cos(a), length * math.sin(a))]
        run = profile[-1][0]
        return _profile_solid(static, profile, float(p["width_m"]), -run / 2.0, mirror=True)
    raise ValueError(f"unknown static primitive {static.kind!r}")


def solids_for(statics: Sequence[Static]) -> list[StaticSolid]:
    return [solid_for(s) for s in statics]


def describe(solids: Sequence[StaticSolid]) -> list[Mapping[str, Any]]:
    """JSON-ready summary of solids (used in provenance records)."""
    return [{"id": s.static_id, "kind": s.kind, "parts": [np.round(v, 6).tolist() for v in s.parts]} for s in solids]
