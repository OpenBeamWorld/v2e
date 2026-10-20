# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""Rigid-body geometry: rotations, rigid fits, simple meshes and distance queries.

Conventions: world frame Z-up, metres; quaternions ``[x, y, z, w]`` (the order SciPy uses);
a pose maps body coordinates to world coordinates as ``p_world = R @ p_body + t``. The V2E body
frame of an object is x forward, y left, z up. Arrays of poses are NaN where a pose is unknown.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

__all__ = [
    "quat_to_matrix",
    "matrix_to_quat",
    "normalize_quats",
    "geodesic_angle",
    "yaw_of",
    "heading_vector",
    "slerp_quats",
    "kabsch",
    "principal_frame",
    "transform_points",
    "box_mesh",
    "cylinder_mesh",
    "sphere_mesh",
    "mesh_bounds",
    "ground_height",
    "points_to_triangles_distance",
    "faces_in_box",
    "triangle_bounds",
]


def normalize_quats(q: Any) -> np.ndarray:
    """Unit quaternions with ``w >= 0`` (the sign is irrelevant for rotations); NaN rows stay NaN."""
    q = np.array(q, dtype=float, copy=True)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        q = q / norm
    flip = q[..., 3:4] < 0
    return np.where(flip, -q, q)


def quat_to_matrix(q: Any) -> np.ndarray:
    """``[x, y, z, w]`` -> rotation matrix; works on (4,) or (N, 4); NaN rows give NaN matrices."""
    q = np.asarray(q, dtype=float)
    single = q.ndim == 1
    q2 = np.atleast_2d(q)
    out = np.full((len(q2), 3, 3), np.nan)
    ok = np.all(np.isfinite(q2), axis=1)
    if np.any(ok):
        out[ok] = Rotation.from_quat(q2[ok]).as_matrix()
    return out[0] if single else out


def matrix_to_quat(R: Any) -> np.ndarray:
    """Rotation matrix (3, 3) or (N, 3, 3) -> ``[x, y, z, w]`` with ``w >= 0``."""
    R = np.asarray(R, dtype=float)
    single = R.ndim == 2
    R2 = R[None] if single else R
    out = np.full((len(R2), 4), np.nan)
    ok = np.all(np.isfinite(R2.reshape(len(R2), -1)), axis=1)
    if np.any(ok):
        out[ok] = normalize_quats(Rotation.from_matrix(R2[ok]).as_quat())
    return out[0] if single else out


def geodesic_angle(Ra: Any, Rb: Any) -> np.ndarray | float:
    """Angle (radians) of the rotation taking ``Ra`` to ``Rb``; broadcasts over leading axes."""
    Ra = np.asarray(Ra, dtype=float)
    Rb = np.asarray(Rb, dtype=float)
    M = np.swapaxes(Ra, -1, -2) @ Rb
    cos_angle = (np.trace(M, axis1=-2, axis2=-1) - 1.0) / 2.0
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    return float(angle) if np.ndim(angle) == 0 else angle


def heading_vector(R: Any) -> np.ndarray:
    """World direction of the body +x axis (forward)."""
    return np.asarray(R, dtype=float)[..., :, 0]


def yaw_of(R: Any) -> np.ndarray | float:
    """Heading of the body +x axis about world +Z, radians in (-pi, pi]."""
    f = heading_vector(R)
    yaw = np.arctan2(f[..., 1], f[..., 0])
    return float(yaw) if np.ndim(yaw) == 0 else yaw


def slerp_quats(times: Any, quats: Any, query: Any) -> np.ndarray:
    """Spherical interpolation of a quaternion track; NaN outside ``[times[0], times[-1]]``."""
    times = np.asarray(times, dtype=float)
    quats = np.asarray(quats, dtype=float)
    query = np.asarray(query, dtype=float)
    out = np.full((len(query), 4), np.nan)
    if len(times) == 0:
        return out
    inside = (query >= times[0]) & (query <= times[-1])
    if len(times) == 1:
        out[query == times[0]] = normalize_quats(quats[0])
        return out
    if np.any(inside):
        slerp = Slerp(times, Rotation.from_quat(quats))
        out[inside] = normalize_quats(slerp(query[inside]).as_quat())
    return out


def kabsch(src: Any, dst: Any, weights: Any | None = None) -> tuple[np.ndarray, np.ndarray, float]:
    """Best rigid fit ``dst ~ R @ src + t`` (rows are points). Returns ``(R, t, rms)``."""
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)
    w = np.ones(len(src)) if weights is None else np.asarray(weights, dtype=float)
    w = w / np.sum(w)
    cs = w @ src
    cd = w @ dst
    H = (src - cs).T @ ((dst - cd) * w[:, None])
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T)) or 1.0
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    t = cd - R @ cs
    residual = dst - (src @ R.T + t)
    rms = float(math.sqrt(np.sum(w * np.sum(residual * residual, axis=1))))
    return R, t, rms


def principal_frame(points: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A body frame for an unlabelled point set.

    Returns ``(R, centre, extents)``: the columns of ``R`` are the principal axes ordered from
    the longest to the shortest extent (a right-handed frame), ``centre`` is the centre of the
    bounding box in that frame and ``extents`` its side lengths.
    """
    pts = np.asarray(points, dtype=float)
    mean = pts.mean(axis=0)
    _, _, Vt = np.linalg.svd(pts - mean, full_matrices=False)
    R = Vt.T.copy()
    if R.shape[1] < 3:  # degenerate input: complete the basis
        R = np.linalg.qr(np.hstack([R, np.eye(3)]))[0][:, :3]
    if np.linalg.det(R) < 0:
        R[:, 2] *= -1.0
    local = (pts - mean) @ R
    lo, hi = local.min(axis=0), local.max(axis=0)
    order = np.argsort(-(hi - lo), kind="stable")
    R = R[:, order]
    if np.linalg.det(R) < 0:
        R[:, 2] *= -1.0
    local = (pts - mean) @ R
    lo, hi = local.min(axis=0), local.max(axis=0)
    centre = mean + R @ (0.5 * (lo + hi))
    return R, centre, hi - lo


def transform_points(points: Any, R: Any, t: Any) -> np.ndarray:
    """``points @ R.T + t`` for (N, 3) points."""
    return np.asarray(points, dtype=float) @ np.asarray(R, dtype=float).T + np.asarray(t, dtype=float)


def box_mesh(length: float, width: float, height: float, centre: Any = (0.0, 0.0, 0.0)) -> tuple[np.ndarray, np.ndarray]:
    """A closed box (8 vertices, 12 outward-facing triangles) centred at ``centre``."""
    hx, hy, hz = 0.5 * length, 0.5 * width, 0.5 * height
    v = np.array([[x, y, z] for x in (-hx, hx) for y in (-hy, hy) for z in (-hz, hz)], dtype=float)
    f = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                  [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]], dtype=np.int32)
    return v + np.asarray(centre, dtype=float), f


def cylinder_mesh(radius: float, length: float, segments: int = 24, axis: str = "z") -> tuple[np.ndarray, np.ndarray]:
    """A closed cylinder centred at the origin with its axis along ``axis``."""
    a = np.linspace(0.0, 2.0 * math.pi, segments, endpoint=False)
    ring = np.stack([radius * np.cos(a), radius * np.sin(a)], axis=1)
    h = 0.5 * length
    bottom = np.hstack([ring, np.full((segments, 1), -h)])
    top = np.hstack([ring, np.full((segments, 1), h)])
    v = np.vstack([bottom, top, [[0.0, 0.0, -h], [0.0, 0.0, h]]])
    faces = []
    cb, ct = 2 * segments, 2 * segments + 1
    for i in range(segments):
        j = (i + 1) % segments
        faces += [[i, j, segments + j], [i, segments + j, segments + i], [cb, j, i], [ct, segments + i, segments + j]]
    order = {"x": [2, 0, 1], "y": [1, 2, 0], "z": [0, 1, 2]}[axis]
    return v[:, order], np.array(faces, dtype=np.int32)


def sphere_mesh(radius: float, rings: int = 10, segments: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """A UV sphere centred at the origin."""
    verts = [[0.0, 0.0, radius]]
    for i in range(1, rings):
        theta = math.pi * i / rings
        for j in range(segments):
            phi = 2.0 * math.pi * j / segments
            verts.append([radius * math.sin(theta) * math.cos(phi), radius * math.sin(theta) * math.sin(phi),
                          radius * math.cos(theta)])
    verts.append([0.0, 0.0, -radius])
    faces = []
    south = len(verts) - 1
    for j in range(segments):
        faces.append([0, 1 + j, 1 + (j + 1) % segments])
    for i in range(rings - 2):
        a0, b0 = 1 + i * segments, 1 + (i + 1) * segments
        for j in range(segments):
            j1 = (j + 1) % segments
            faces += [[a0 + j, b0 + j, b0 + j1], [a0 + j, b0 + j1, a0 + j1]]
    last = 1 + (rings - 2) * segments
    for j in range(segments):
        faces.append([south, last + (j + 1) % segments, last + j])
    return np.array(verts, dtype=float), np.array(faces, dtype=np.int32)


def mesh_bounds(vertices: Any) -> tuple[np.ndarray, np.ndarray]:
    v = np.asarray(vertices, dtype=float).reshape(-1, 3)
    return v.min(axis=0), v.max(axis=0)


def triangle_bounds(vertices: Any, faces: Any) -> tuple[np.ndarray, np.ndarray]:
    """Per-face bounding boxes ``(lo (T, 3), hi (T, 3))``; compute once for meshes queried often."""
    v = np.asarray(vertices, dtype=float)
    f = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    tri = v[f]
    return tri.min(axis=1), tri.max(axis=1)


def faces_in_box(vertices: Any, faces: Any, lo: Any, hi: Any, margin: float = 0.0, dims: int = 3,
                 bounds: tuple[np.ndarray, np.ndarray] | None = None) -> np.ndarray:
    """The faces whose bounding box overlaps the box ``[lo - margin, hi + margin]`` (first ``dims`` axes).

    Pass ``bounds`` from :func:`triangle_bounds` to avoid recomputing them for a large static mesh.
    """
    f = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if len(f) == 0:
        return f
    t_lo, t_hi = bounds if bounds is not None else triangle_bounds(vertices, f)
    lo = np.asarray(lo, dtype=float)[:dims] - margin
    hi = np.asarray(hi, dtype=float)[:dims] + margin
    keep = np.all((t_hi[:, :dims] >= lo) & (t_lo[:, :dims] <= hi), axis=1)
    return f[keep]


def ground_height(points_xy: Any, vertices: Any, faces: Any, below: Any | None = None,
                  chunk: int = 256) -> np.ndarray:
    """Height of the highest triangle under each (x, y) point, NaN where none lies under it.

    With ``below`` (heights per point), only surfaces at or below that height count, so a
    query from just above a surface finds that surface rather than a roof overhead.
    """
    pts = np.atleast_2d(np.asarray(points_xy, dtype=float))[:, :2]
    v = np.asarray(vertices, dtype=float)
    f = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    out = np.full(len(pts), np.nan)
    if len(f) == 0 or len(pts) == 0:
        return out
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    d = (b[:, 1] - c[:, 1]) * (a[:, 0] - c[:, 0]) + (c[:, 0] - b[:, 0]) * (a[:, 1] - c[:, 1])
    usable = np.abs(d) > 1e-12          # triangles seen edge-on from above have no height
    a, b, c, d = a[usable], b[usable], c[usable], d[usable]
    if len(d) == 0:
        return out
    limit = None if below is None else np.asarray(below, dtype=float).reshape(-1)
    eps = 1e-9
    for lo in range(0, len(pts), chunk):
        x = pts[lo:lo + chunk, 0:1]
        y = pts[lo:lo + chunk, 1:2]
        l1 = ((b[:, 1] - c[:, 1]) * (x - c[:, 0]) + (c[:, 0] - b[:, 0]) * (y - c[:, 1])) / d
        l2 = ((c[:, 1] - a[:, 1]) * (x - c[:, 0]) + (a[:, 0] - c[:, 0]) * (y - c[:, 1])) / d
        l3 = 1.0 - l1 - l2
        inside = (l1 >= -eps) & (l2 >= -eps) & (l3 >= -eps)
        z = l1 * a[:, 2] + l2 * b[:, 2] + l3 * c[:, 2]
        if limit is not None:
            inside &= z <= limit[lo:lo + chunk, None] + 1e-6
        z = np.where(inside, z, -np.inf)
        best = z.max(axis=1)
        out[lo:lo + chunk] = np.where(np.isfinite(best), best, np.nan)
    return out


def points_to_triangles_distance(points: Any, vertices: Any, faces: Any) -> np.ndarray:
    """Unsigned distance from each point to the nearest triangle of a mesh (exact, vectorised)."""
    p = np.atleast_2d(np.asarray(points, dtype=float))
    v = np.asarray(vertices, dtype=float)
    f = np.asarray(faces, dtype=np.int64)
    if len(f) == 0:
        return np.full(len(p), np.inf)
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    ab, ac = b - a, c - a
    out = np.empty(len(p))
    for i, q in enumerate(p):
        out[i] = float(np.sqrt(np.min(_sq_dist_point_triangles(q, a, b, c, ab, ac))))
    return out


def _sq_dist_point_triangles(q: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray,
                             ab: np.ndarray, ac: np.ndarray) -> np.ndarray:
    """Squared distances from one point to many triangles (closest-point regions)."""
    ap = q - a
    d1 = np.einsum("ij,ij->i", ab, ap)
    d2 = np.einsum("ij,ij->i", ac, ap)
    bp = q - b
    d3 = np.einsum("ij,ij->i", ab, bp)
    d4 = np.einsum("ij,ij->i", ac, bp)
    cp = q - c
    d5 = np.einsum("ij,ij->i", ab, cp)
    d6 = np.einsum("ij,ij->i", ac, cp)
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2
    closest = np.empty_like(a)
    done = np.zeros(len(a), dtype=bool)

    def take(mask: np.ndarray, value: np.ndarray) -> None:
        sel = mask & ~done
        closest[sel] = value[sel]
        done[sel] = True

    take((d1 <= 0) & (d2 <= 0), a)
    take((d3 >= 0) & (d4 <= d3), b)
    with np.errstate(invalid="ignore", divide="ignore"):
        v_ab = d1 / (d1 - d3)
        take((vc <= 0) & (d1 >= 0) & (d3 <= 0), a + v_ab[:, None] * ab)
        take((d6 >= 0) & (d5 <= d6), c)
        w_ac = d2 / (d2 - d6)
        take((vb <= 0) & (d2 >= 0) & (d6 <= 0), a + w_ac[:, None] * ac)
        w_bc = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        take((va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0), b + w_bc[:, None] * (c - b))
        denom = 1.0 / (va + vb + vc)
        v_in = vb * denom
        w_in = vc * denom
    interior = a + v_in[:, None] * ab + w_in[:, None] * ac
    take(np.ones(len(a), dtype=bool), interior)
    diff = closest - q
    return np.einsum("ij,ij->i", diff, diff)
