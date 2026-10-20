# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""Run-up synthesis for actors that enter the footage already moving (``drive_in``).

Such an actor spawns at rest at the start of an approach and its controller accelerates it so
that it passes its frame-0 pose at the observed speed. The approach extends the path backwards
from the frame-0 pose: a straight line, or a constant-curvature arc when the first half second
of the track bends. Its length is ``L = v^2 / (2 a_ref) + settle`` (``a_ref`` = 2.0 m/s^2 and a
10 m settle stretch by default).

Where the reconstructed drivable surface does not reach that far, a flat slab of synthesised
ground (``observed: false``) carries the approach at the frame-0 surface height. A clearance
corridor (actor width plus 2 m on each side, 3 m high) along the approach must be free of
static geometry and of other objects' frame-0 hulls; otherwise the scene is refused, since the
actor could not reach its frame-0 state without hitting something.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
from scipy.spatial import cKDTree

from openbeam_v2e.compile.content import Refusal
from openbeam_v2e.compile.level import CollisionMesh
from openbeam_v2e.schema.common import Pose
from openbeam_v2e.world.geometry import ground_height, matrix_to_quat
from openbeam_v2e.world.model import ObjectTrack

__all__ = ["A_REF_MPS2", "SETTLE_M", "CORRIDOR_MARGIN_M", "CORRIDOR_HEIGHT_M", "Runup", "synthesize_runup",
           "fit_entry_shape", "corridor_hits"]

A_REF_MPS2 = 2.0
SETTLE_M = 10.0
CORRIDOR_MARGIN_M = 2.0
CORRIDOR_HEIGHT_M = 3.0
GROUND_CLEARANCE_M = 0.10      # geometry this close to the surface counts as the surface itself
ENTRY_WINDOW_S = 0.5
STRAIGHT_RADIUS_M = 300.0      # flatter arcs are treated as straight


@dataclass
class Runup:
    actor: str
    shape: str                     # "straight" | "arc"
    length_m: float
    speed_mps: float
    accel_mps2: float
    surface_z_m: float
    start_pose: Pose
    path_m: np.ndarray             # (N, 3) reference-point path from the run-up start to the frame-0 pose
    s_m: np.ndarray                # (N,) arc length from the run-up start
    t_s: np.ndarray                # (N,) time from the start of motion
    v_mps: np.ndarray              # (N,) speed profile
    ground: CollisionMesh | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def natural_t0_s(self) -> float:
        """Time from the start of motion to the frame-0 pose."""
        return float(self.t_s[-1])


def _kasa_circle(xy: np.ndarray) -> tuple[np.ndarray, float, float] | None:
    """Algebraic circle fit; returns (centre, radius, rms residual) or None when degenerate."""
    A = np.column_stack([xy[:, 0], xy[:, 1], np.ones(len(xy))])
    b = -(xy[:, 0] ** 2 + xy[:, 1] ** 2)
    try:
        (D, E, F), *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    centre = np.array([-D / 2.0, -E / 2.0])
    r2 = centre @ centre - F
    if not np.isfinite(r2) or r2 <= 0:
        return None
    r = math.sqrt(r2)
    rms = float(np.sqrt(np.mean((np.linalg.norm(xy - centre, axis=1) - r) ** 2)))
    return centre, r, rms


def fit_entry_shape(xy: np.ndarray, heading_xy: np.ndarray, mode: str = "auto") -> dict[str, Any]:
    """Shape of the entry path from the first observed positions (travel order)."""
    xy = np.asarray(xy, dtype=float)
    d = xy[-1] - xy[0] if len(xy) >= 2 else np.zeros(2)
    if np.linalg.norm(d) < 1e-6:
        d = np.asarray(heading_xy, dtype=float)
    d = d / np.linalg.norm(d)
    straight = {"shape": "straight", "direction": d}
    if mode == "straight" or len(xy) < 3:
        return straight
    spread = float(np.linalg.norm(xy[-1] - xy[0]))
    fit = _kasa_circle(xy) if spread > 1.0 else None
    if fit is None:
        return straight
    centre, r, rms = fit
    if (r > STRAIGHT_RADIUS_M or rms > 0.05 * spread) and mode == "auto":
        return straight
    # travel direction at frame 0: tangent of the circle at the first point, oriented like d
    radial = xy[0] - centre
    tangent = np.array([-radial[1], radial[0]]) / np.linalg.norm(radial)
    sense = 1.0 if tangent @ d >= 0 else -1.0
    return {"shape": "arc", "centre": centre, "radius": r, "sense": sense, "direction": sense * tangent}


def _backward_path(p0: np.ndarray, shape: dict[str, Any], length: float,
                   step: float = 0.5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Points (xy) at distances ``s_back`` behind ``p0``, the unit travel tangents there, and ``s_back``."""
    s_back = np.linspace(0.0, length, max(2, int(math.ceil(length / step)) + 1))
    if shape["shape"] == "straight":
        d = shape["direction"]
        pts = p0[None, :2] - s_back[:, None] * d[None, :]
        tangents = np.repeat(d[None, :], len(s_back), axis=0)
    else:
        c, r, sense = shape["centre"], shape["radius"], shape["sense"]
        theta0 = math.atan2(p0[1] - c[1], p0[0] - c[0])
        theta = theta0 - sense * s_back / r
        pts = np.column_stack([c[0] + r * np.cos(theta), c[1] + r * np.sin(theta)])
        tangents = sense * np.column_stack([-np.sin(theta), np.cos(theta)])
    return pts, tangents, s_back


def _slab(points_xy: np.ndarray, tangents: np.ndarray, half_width: float, z_top: float,
          thickness: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    normals = np.column_stack([-tangents[:, 1], tangents[:, 0]])
    left = points_xy + half_width * normals
    right = points_xy - half_width * normals
    n = len(points_xy)
    verts = []
    for i in range(n):
        verts += [[*left[i], z_top], [*right[i], z_top], [*left[i], z_top - thickness], [*right[i], z_top - thickness]]
    faces = []
    for i in range(n - 1):
        a, b = 4 * i, 4 * (i + 1)
        faces += [[a, a + 1, b + 1], [a, b + 1, b],                    # top
                  [a + 2, b + 3, a + 3], [a + 2, b + 2, b + 3],        # bottom
                  [a, b, b + 2], [a, b + 2, a + 2],                    # left side
                  [a + 1, a + 3, b + 3], [a + 1, b + 3, b + 1]]        # right side
    last = 4 * (n - 1)
    faces += [[0, 2, 3], [0, 3, 1], [last, last + 1, last + 3], [last, last + 3, last + 2]]
    return np.array(verts, dtype=float), np.array(faces, dtype=np.int32)


def _triangle_samples(vertices: np.ndarray, faces: np.ndarray, spacing: float = 0.5, max_per_edge: int = 40) -> np.ndarray:
    out = [vertices]
    for tri in faces:
        a, b, c = vertices[tri]
        edge = max(np.linalg.norm(b - a), np.linalg.norm(c - b), np.linalg.norm(a - c))
        n = int(min(max_per_edge, max(1, math.ceil(edge / spacing))))
        u, v = np.meshgrid(np.arange(n + 1), np.arange(n + 1))
        keep = (u + v) <= n
        u, v = u[keep] / n, v[keep] / n
        out.append(a + u[:, None] * (b - a) + v[:, None] * (c - a))
    return np.vstack(out)


def corridor_hits(path_xy: np.ndarray, tangents: np.ndarray, surface_z: float, half_width: float,
                  points: np.ndarray) -> np.ndarray:
    """Mask of ``points`` inside the clearance corridor along ``path_xy`` (dense samples)."""
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    if len(pts) == 0:
        return np.zeros(0, dtype=bool)
    tree = cKDTree(path_xy)
    dist, idx = tree.query(pts[:, :2])
    height = pts[:, 2] - surface_z
    inside = (dist <= half_width) & (height > GROUND_CLEARANCE_M) & (height < CORRIDOR_HEIGHT_M)
    # points past either end of the path are not in the corridor
    along = np.einsum("ij,ij->i", pts[:, :2] - path_xy[idx], tangents[idx])
    first, last = idx == 0, idx == len(path_xy) - 1
    inside &= ~(first & (along > 0.25)) & ~(last & (along < -0.25))
    return inside


def synthesize_runup(obj: ObjectTrack, fps: float, speed_mps: float, frame0: int,
                     static_vertices: np.ndarray, static_faces: np.ndarray,
                     others: Sequence[tuple[str, np.ndarray]] = (), width_m: float | None = None,
                     path_mode: str = "auto", accel_mps2: float = A_REF_MPS2, settle_m: float = SETTLE_M,
                     actor: str | None = None) -> Runup | Refusal:
    """Plan the approach of a ``drive_in`` actor from its reconstructed track."""
    name = actor or obj.id
    if speed_mps < 0.5:
        return Refusal("drive_in_too_slow", "an actor slower than 0.5 m/s at frame 0 should start 'at_rest'", name)
    if not obj.known[frame0]:
        return Refusal("pose_unknown", f"the reconstruction has no pose for frame {frame0}", name)
    p0 = obj.position[frame0]
    R0 = obj.rotations()[frame0]
    window = [k for k in range(frame0, min(obj.frame_count, frame0 + int(round(ENTRY_WINDOW_S * fps)) + 1))
              if obj.known[k]]
    xy = obj.position[window, :2]
    heading = R0[:2, 0] / (np.linalg.norm(R0[:2, 0]) or 1.0)
    shape = fit_entry_shape(xy, heading, path_mode)
    length = speed_mps ** 2 / (2.0 * accel_mps2) + settle_m
    notes: list[str] = []
    if shape["shape"] == "arc":
        if length > math.pi * shape["radius"]:
            return Refusal("runup_too_curved",
                           f"the entry arc (radius {shape['radius']:.1f} m) cannot hold a {length:.1f} m approach; "
                           "start the scene earlier", name)
        lateral = speed_mps ** 2 / shape["radius"]
        if lateral > 6.0:
            notes.append(f"entry arc needs {lateral:.1f} m/s^2 lateral acceleration at the observed speed")

    # frame-0 surface: highest static surface under the reference point, else the hull's lowest point
    lowest = float(obj.lowest_point()[frame0])
    gz = ground_height(p0[None, :2], static_vertices, static_faces, below=[lowest + 0.3])[0] if len(static_faces) else np.nan
    surface = float(gz) if np.isfinite(gz) else lowest
    h0 = float(p0[2] - surface)
    width = float(width_m) if width_m is not None else float(obj.hull_extents()[1])
    half = 0.5 * width + CORRIDOR_MARGIN_M

    pts_xy, tangents, s_back = _backward_path(p0, shape, length + 5.0)
    in_runup = s_back <= length + 1e-9
    path_xy, path_t = pts_xy[in_runup], tangents[in_runup]

    # clearance corridor: static geometry and other objects' frame-0 hulls
    blocked = []
    if len(static_faces):
        samples = _triangle_samples(np.asarray(static_vertices, float), np.asarray(static_faces, np.int64))
        hits = corridor_hits(path_xy, path_t, surface, half, samples)
        if np.any(hits):
            blocked.append(f"static geometry ({int(np.count_nonzero(hits))} sample points)")
    for other, points in others:
        if np.any(corridor_hits(path_xy, path_t, surface, half, points)):
            blocked.append(f"object {other!r} at frame 0")
    if blocked:
        return Refusal("corridor_blocked", "the run-up clearance corridor is blocked by " + ", ".join(blocked), name)

    # synthesised ground where the observed surface stops
    cover = ground_height(pts_xy, static_vertices, static_faces, below=np.full(len(pts_xy), surface + 0.3)) \
        if len(static_faces) else np.full(len(pts_xy), np.nan)
    covered = np.isfinite(cover) & (np.abs(cover - surface) <= 0.05)
    first_gap = int(np.argmin(covered)) if not covered.all() else len(covered)
    ground = None
    if first_gap < len(covered):
        start = max(0, first_gap - 2)
        verts, faces = _slab(pts_xy[start:], tangents[start:], half, surface)
        ground = CollisionMesh(f"runup_{name}", verts, faces, source="runup", role="drivable", observed=False,
                               actor=name)
        notes.append(f"synthesised {s_back[-1] - s_back[start]:.1f} m of run-up ground beyond the observed surface")

    # reference path in travel order (start -> frame-0 pose) and the speed profile
    order = np.arange(len(path_xy))[::-1]
    path = np.column_stack([path_xy[order], np.full(len(order), surface + h0)])
    s = length - s_back[in_runup][order]
    s_accel = speed_mps ** 2 / (2.0 * accel_mps2)
    t_accel = speed_mps / accel_mps2
    v = np.where(s < s_accel, np.sqrt(2.0 * accel_mps2 * np.maximum(s, 0.0)), speed_mps)
    t = np.where(s < s_accel, np.sqrt(2.0 * np.maximum(s, 0.0) / accel_mps2), t_accel + (s - s_accel) / speed_mps)
    tangent0 = path_t[order][0]
    yaw = math.atan2(tangent0[1], tangent0[0])
    Rs = np.array([[math.cos(yaw), -math.sin(yaw), 0.0], [math.sin(yaw), math.cos(yaw), 0.0], [0.0, 0.0, 1.0]])
    start_pose = Pose(tuple(path[0]), tuple(matrix_to_quat(Rs)))
    return Runup(name, shape["shape"], float(length), float(speed_mps), float(accel_mps2), surface, start_pose, path, s,
                 t, v, ground, notes)
