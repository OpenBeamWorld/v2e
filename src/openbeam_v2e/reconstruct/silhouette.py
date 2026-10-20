# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""Silhouette machinery: outlines of template hulls, masks, signed distances and overlap.

Masks are boolean arrays indexed ``[row, col]``; the pixel centre of ``[row, col]`` is at image
coordinates ``(col + 0.5, row + 0.5)`` (see :mod:`openbeam_v2e.reconstruct.geom`).

The projected outline of a hull is a list of convex polygons (one per convex part). Signed
distances are positive outside a shape and negative inside, in pixels, so a fitted outline makes
both of these residuals small:

* the mask's signed distance sampled at points along the hull outline;
* the outline's signed distance evaluated at points on the mask boundary.

Both are continuous in the pose, which is what lets a least-squares solver fit poses to masks
and report a covariance.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from scipy import ndimage

from openbeam_v2e.reconstruct.geom import Camera
from openbeam_v2e.reconstruct.hulls import Hull


# --------------------------------------------------------------------------- polygons

def convex_hull_2d(points: Any) -> np.ndarray:
    """Counter-clockwise convex hull (in image coordinates, i.e. y down) of 2-D points."""
    P = np.unique(np.asarray(points, float).reshape(-1, 2), axis=0)
    if len(P) < 3:
        return P
    P = P[np.lexsort((P[:, 1], P[:, 0]))]

    def cross(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))

    lower: list[np.ndarray] = []
    for p in P:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list[np.ndarray] = []
    for p in P[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


def polygon_area(poly: Any) -> float:
    P = np.asarray(poly, float)
    if len(P) < 3:
        return 0.0
    x, y = P[:, 0], P[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def polygon_mask(polygons: Sequence[Any], width: int, height: int) -> np.ndarray:
    """Union of filled polygons (even-odd rule per polygon) sampled at pixel centres."""
    mask = np.zeros((height, width), bool)
    ys = np.arange(height) + 0.5
    xs = np.arange(width) + 0.5
    for poly in polygons:
        P = np.asarray(poly, float)
        if len(P) < 3 or not np.all(np.isfinite(P)):
            continue
        m = np.zeros((height, width), bool)
        r0 = max(0, int(np.floor(P[:, 1].min())))
        r1 = min(height, int(np.ceil(P[:, 1].max())) + 1)
        if r1 <= r0:
            continue
        n = len(P)
        for i in range(n):
            (xa, ya), (xb, yb) = P[i], P[(i + 1) % n]
            if ya == yb:
                continue
            lo, hi = min(ya, yb), max(ya, yb)
            rows = np.nonzero((ys[r0:r1] >= lo) & (ys[r0:r1] < hi))[0] + r0
            if len(rows) == 0:
                continue
            xc = xa + (ys[rows] - ya) * (xb - xa) / (yb - ya)
            m[rows] ^= xs[None, :] >= xc[:, None]
        mask |= m
    return mask


def clip_polygons(polygons: Sequence[np.ndarray], width: int, height: int) -> list[np.ndarray]:
    """Intersect convex outlines with the image rectangle, preserving their winding.

    A silhouette mask says nothing about the hull beyond the image border. Fits must compare
    the visible projection, just as :func:`polygon_mask` does, not pull offscreen edges inside.
    Empty and zero-area intersections have no outline.
    """
    out = []
    for poly in polygons:
        P = np.asarray(poly, float)
        for axis, bound, sign in ((0, 0.0, 1), (0, float(width), -1),
                                  (1, 0.0, 1), (1, float(height), -1)):
            if len(P) < 3:
                break
            d = sign * (P[:, axis] - bound)
            if np.all(d >= 0):
                continue
            if np.all(d < 0):
                P = np.empty((0, 2))
                break
            points = []
            for a, b, da, db in zip(np.roll(P, 1, axis=0), P, np.roll(d, 1), d):
                if (da >= 0) != (db >= 0):
                    crossing = a + (da / (da - db)) * (b - a)
                    crossing[axis] = bound
                    points.append(crossing)
                if db >= 0:
                    points.append(b)
            P = np.asarray(points, float).reshape(-1, 2)
        if len(P) >= 3 and abs(polygon_area(P)) > 1e-12:
            out.append(P)
    return out


def convex_polygon_sdf(points: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Signed distance (positive outside) from points (N, 2) to a convex polygon (K, 2)."""
    P = np.asarray(points, float)
    if len(poly) < 3:
        return np.full(len(P), np.inf)
    A = poly
    B = np.roll(poly, -1, axis=0)
    E = B - A                                    # (K, 2)
    L2 = np.maximum(np.sum(E * E, axis=1), 1e-18)
    D = P[:, None, :] - A[None, :, :]            # (N, K, 2)
    t = np.clip(np.sum(D * E[None], axis=2) / L2[None], 0.0, 1.0)
    closest = A[None] + t[..., None] * E[None]
    dist = np.sqrt(np.min(np.sum((P[:, None, :] - closest) ** 2, axis=2), axis=1))
    # inside test: same side of every edge (orientation from the signed area)
    orient = 1.0 if polygon_area(poly) >= 0 else -1.0
    crossv = E[None, :, 0] * D[..., 1] - E[None, :, 1] * D[..., 0]
    inside = np.all(orient * crossv >= 0.0, axis=1)
    return np.where(inside, -dist, dist)


def union_sdf(points: Any, polygons: Sequence[np.ndarray]) -> np.ndarray:
    """Signed distance to a union of convex polygons (exact outside, conservative inside)."""
    P = np.atleast_2d(np.asarray(points, float))
    out = np.full(len(P), np.inf)
    for poly in polygons:
        out = np.minimum(out, convex_polygon_sdf(P, poly))
    return out


def sample_outline(polygons: Sequence[np.ndarray], n: int) -> tuple[np.ndarray, np.ndarray]:
    """``n`` points spread by arc length over the edges of all polygons, plus a weight per point.

    The weight fades from 1 to 0 for points that lie inside another polygon of the union (those
    edges are not part of the visible outline). The point count is fixed, so the residual vector
    of a fit keeps its length while the pose changes.
    """
    segs = []
    owner = []
    for k, poly in enumerate(polygons):
        if len(poly) < 2:
            continue
        A = np.asarray(poly, float)
        B = np.roll(A, -1, axis=0)
        segs.append(np.concatenate([A, B], axis=1))
        owner.extend([k] * len(A))
    if not segs:
        return np.zeros((n, 2)), np.zeros(n)
    S = np.concatenate(segs)
    owner_arr = np.array(owner)
    lengths = np.linalg.norm(S[:, 2:] - S[:, :2], axis=1)
    total = float(lengths.sum())
    if total <= 0:
        return np.repeat(S[:1, :2], n, axis=0), np.ones(n)
    cum = np.concatenate([[0.0], np.cumsum(lengths)])
    s = (np.arange(n) + 0.5) * total / n
    idx = np.clip(np.searchsorted(cum, s, side="right") - 1, 0, len(S) - 1)
    u = (s - cum[idx]) / np.maximum(lengths[idx], 1e-12)
    pts = S[idx, :2] + u[:, None] * (S[idx, 2:] - S[idx, :2])
    w = np.ones(n)
    if len(polygons) > 1:
        for k, poly in enumerate(polygons):
            sel = owner_arr[idx] != k
            if np.any(sel) and len(poly) >= 3:
                d = convex_polygon_sdf(pts[sel], np.asarray(poly, float))
                w[sel] = np.minimum(w[sel], np.clip(d / 1.5, 0.0, 1.0))
    return pts, w


# --------------------------------------------------------------------------- hull outlines

def hull_outline(hull: Hull, R_wb: Any, position: Any, camera: Camera,
                 min_depth: float = 0.05) -> list[np.ndarray] | None:
    """Projected outline of the hull at a pose: one convex polygon (pixels) per convex part.

    Returns ``None`` when any vertex is behind (or too close to) the camera. Lens distortion is
    applied to the vertices; edges between them are drawn straight, which is accurate for objects
    that cover a small part of the image.
    """
    polys = []
    for verts in hull.world_parts(R_wb, position):
        uv, z = camera.project(verts)
        if np.any(z < min_depth) or not np.all(np.isfinite(uv)):
            return None
        polys.append(convex_hull_2d(uv))
    return polys


def hull_mask(hull: Hull, R_wb: Any, position: Any, camera: Camera) -> np.ndarray:
    polys = hull_outline(hull, R_wb, position, camera)
    if polys is None:
        return np.zeros((camera.height, camera.width), bool)
    return polygon_mask(polys, camera.width, camera.height)


# --------------------------------------------------------------------------- masks

def iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.count_nonzero(a & b)
    union = np.count_nonzero(a | b)
    return float(inter) / union if union else 0.0


def signed_distance(mask: np.ndarray) -> np.ndarray:
    """Signed distance field of a mask in pixels: positive outside, negative inside.

    The boundary is taken half-way between an inside and an outside pixel centre.
    """
    m = np.asarray(mask, bool)
    if not m.any():
        return np.full(m.shape, 1e6)
    if m.all():
        return np.full(m.shape, -1e6)
    outside = ndimage.distance_transform_edt(~m)
    inside = ndimage.distance_transform_edt(m)
    return np.where(m, -(inside - 0.5), outside - 0.5)


def sample_field(field: np.ndarray, uv: Any, outside_value: float | None = None) -> np.ndarray:
    """Bilinear samples of a per-pixel field at image coordinates ``uv`` (N, 2).

    Points outside the image get the nearest edge value plus the distance to the image, which
    keeps signed-distance residuals growing outside the frame.
    """
    uv = np.atleast_2d(np.asarray(uv, float))
    h, w = field.shape
    col = uv[:, 0] - 0.5
    row = uv[:, 1] - 0.5
    cc = np.clip(col, 0, w - 1)
    rr = np.clip(row, 0, h - 1)
    vals = ndimage.map_coordinates(field, [rr, cc], order=1, mode="nearest")
    off = np.hypot(col - cc, row - rr)
    if outside_value is not None:
        return np.where(off > 0, outside_value, vals)
    return vals + off


def boundary_pixels(mask: np.ndarray) -> np.ndarray:
    """Image coordinates (N, 2) of the centres of the mask's boundary pixels."""
    m = np.asarray(mask, bool)
    er = ndimage.binary_erosion(m, structure=np.ones((3, 3), bool), border_value=0)
    rows, cols = np.nonzero(m & ~er)
    return np.stack([cols + 0.5, rows + 0.5], axis=1).astype(float)


def boundary_edge_points(mask: np.ndarray) -> np.ndarray:
    """Midpoints (N, 2) of the pixel edges between mask and non-mask pixels.

    These lie on the mask's staircase outline, so for a straight object edge they scatter evenly
    about it; boundary pixel centres would sit half a pixel inside and bias fits towards
    smaller objects.
    """
    m = np.pad(np.asarray(mask, bool), 1)
    hx = m[:, 1:] != m[:, :-1]          # edge between column c-1 and c (padded coordinates)
    rows, cols = np.nonzero(hx)
    ph = np.stack([cols.astype(float) + 1.0 - 1.0, rows.astype(float) + 0.5 - 1.0], axis=1)
    vx = m[1:, :] != m[:-1, :]
    rows, cols = np.nonzero(vx)
    pv = np.stack([cols.astype(float) + 0.5 - 1.0, rows.astype(float) + 1.0 - 1.0], axis=1)
    return np.concatenate([ph, pv])


def boundary_samples(mask: np.ndarray, n: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """At most ``n`` points spread along the mask outline (deterministic by default)."""
    pts = boundary_edge_points(mask)
    if len(pts) <= n:
        return pts
    if rng is None:
        # order by angle around the centroid, then take evenly spaced points
        c = pts.mean(axis=0)
        order = np.argsort(np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0]), kind="stable")
        sel = order[np.linspace(0, len(pts) - 1, n).astype(int)]
        return pts[sel]
    return pts[rng.choice(len(pts), n, replace=False)]


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """(x0, y0, x1, y1) pixel bounds of a mask, end exclusive; ``None`` when empty."""
    rows = np.nonzero(mask.any(axis=1))[0]
    if len(rows) == 0:
        return None
    cols = np.nonzero(mask.any(axis=0))[0]
    return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1


def bbox_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


# --------------------------------------------------------------------------- mask outlines

def _simplify(points: np.ndarray, tol: float) -> np.ndarray:
    """Douglas-Peucker simplification of a closed polyline (first point not repeated)."""
    if len(points) <= 4 or tol <= 0:
        return points
    # split the ring at the two points farthest apart
    d2 = np.sum((points - points[0]) ** 2, axis=1)
    far = int(np.argmax(d2))

    def dp(seg: np.ndarray) -> list[np.ndarray]:
        if len(seg) <= 2:
            return [seg[0]]
        a, b = seg[0], seg[-1]
        ab = b - a
        L = float(np.hypot(*ab))
        if L < 1e-12:
            dist = np.hypot(*(seg - a).T)
        else:
            dist = np.abs(ab[0] * (seg[:, 1] - a[1]) - ab[1] * (seg[:, 0] - a[0])) / L
        k = int(np.argmax(dist))
        if dist[k] > tol:
            return dp(seg[: k + 1]) + dp(seg[k:])
        return [seg[0]]

    ring = np.concatenate([points, points[:1]])
    out = dp(ring[: far + 1]) + dp(ring[far:])
    return np.array(out)


def mask_to_polygons(mask: np.ndarray, tolerance_px: float = 0.7, min_area_px: float = 4.0,
                     max_polygons: int = 8) -> list[np.ndarray]:
    """Outer outlines of a mask's regions as polygons in image coordinates.

    The outline follows the pixel edges between the mask and the background exactly (so filling
    the polygon at pixel centres gives the mask back, holes aside), then Douglas-Peucker removes
    vertices within ``tolerance_px`` of the outline. Largest regions first.
    """
    m = np.asarray(mask, bool)
    filled = ndimage.binary_fill_holes(m)
    lab, n = ndimage.label(filled)
    polys: list[tuple[float, np.ndarray]] = []
    for k, sl in enumerate(ndimage.find_objects(lab), start=1):
        if sl is None:
            continue
        reg = np.pad(lab[sl] == k, 1)
        y_off, x_off = sl[0].start - 1, sl[1].start - 1
        # directed boundary edges with the region on the left (counter-clockwise in y-down image)
        nxt: dict[tuple[int, int], tuple[int, int]] = {}
        rows, cols = np.nonzero(reg[1:-1, 1:-1])
        rows = rows + 1
        cols = cols + 1
        for r, c in zip(rows.tolist(), cols.tolist()):
            if not reg[r - 1, c]:
                nxt[(c + 1, r)] = (c, r)          # top edge, right to left
            if not reg[r + 1, c]:
                nxt[(c, r + 1)] = (c + 1, r + 1)  # bottom edge, left to right
            if not reg[r, c - 1]:
                nxt[(c, r)] = (c, r + 1)          # left edge, top to bottom
            if not reg[r, c + 1]:
                nxt[(c + 1, r + 1)] = (c + 1, r)  # right edge, bottom to top
        if not nxt:
            continue
        # follow the longest loop (the outer boundary after filling holes)
        start = min(nxt)
        loop = [start]
        cur = nxt[start]
        guard = 0
        while cur != start and guard < 4 * len(nxt) + 4:
            loop.append(cur)
            cur = nxt.get(cur, start)
            guard += 1
        P = np.array(loop, float) + np.array([x_off, y_off], float)
        # drop collinear staircase vertices before simplifying
        keep = np.ones(len(P), bool)
        for i in range(len(P)):
            a, b, c2 = P[i - 1], P[i], P[(i + 1) % len(P)]
            if abs((b[0] - a[0]) * (c2[1] - b[1]) - (b[1] - a[1]) * (c2[0] - b[0])) < 1e-12:
                keep[i] = False
        P = P[keep] if keep.sum() >= 3 else P
        P = _simplify(P, tolerance_px)
        area = abs(polygon_area(P))
        if area >= min_area_px and len(P) >= 3:
            polys.append((area, P))
    polys.sort(key=lambda t: -t[0])
    return [p for _, p in polys[:max_polygons]]
