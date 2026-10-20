# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""Binary silhouettes: rasterise projected meshes and pixel polygons into masks.

Masks are ``(height, width)`` boolean arrays; a pixel is set when its centre
``(col + 0.5, row + 0.5)`` lies inside a triangle or polygon (edges included). Triangles with a
vertex behind the camera's near plane are skipped and counted, so callers can report how much
of a silhouette was not drawn.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from openbeam_v2e.world.camera import Camera

__all__ = ["rasterize_triangles", "rasterize_polygons", "mesh_silhouette", "mask_iou", "mask_bbox"]


def rasterize_triangles(uv: Any, faces: Any, width: int, height: int, valid: Any | None = None,
                        mask: np.ndarray | None = None) -> tuple[np.ndarray, int]:
    """Fill the triangles ``faces`` of the projected points ``uv`` (pixels).

    Returns ``(mask, skipped)`` where ``skipped`` counts triangles left out because a vertex
    was invalid (behind the camera or not finite).
    """
    uv = np.asarray(uv, dtype=float)
    faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    out = np.zeros((height, width), dtype=bool) if mask is None else mask
    ok = np.all(np.isfinite(uv), axis=1)
    if valid is not None:
        ok &= np.asarray(valid, dtype=bool)
    skipped = 0
    for tri in faces:
        if not ok[tri].all():
            skipped += 1
            continue
        p = uv[tri]
        x0 = max(int(np.floor(p[:, 0].min() - 0.5)), 0)
        x1 = min(int(np.ceil(p[:, 0].max() - 0.5)), width - 1)
        y0 = max(int(np.floor(p[:, 1].min() - 0.5)), 0)
        y1 = min(int(np.ceil(p[:, 1].max() - 0.5)), height - 1)
        if x1 < x0 or y1 < y0:
            continue
        xs = np.arange(x0, x1 + 1) + 0.5
        ys = np.arange(y0, y1 + 1) + 0.5
        X, Y = np.meshgrid(xs, ys)
        (ax, ay), (bx, by), (cx, cy) = p
        area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
        if area == 0.0:
            continue
        e0 = (bx - ax) * (Y - ay) - (by - ay) * (X - ax)
        e1 = (cx - bx) * (Y - by) - (cy - by) * (X - bx)
        e2 = (ax - cx) * (Y - cy) - (ay - cy) * (X - cx)
        if area > 0:
            inside = (e0 >= 0) & (e1 >= 0) & (e2 >= 0)
        else:
            inside = (e0 <= 0) & (e1 <= 0) & (e2 <= 0)
        out[y0:y1 + 1, x0:x1 + 1] |= inside
    return out, skipped


def rasterize_polygons(polygons: Iterable[Any], width: int, height: int) -> np.ndarray:
    """Fill pixel polygons (even-odd rule); each polygon is a sequence of ``(x, y)`` points."""
    out = np.zeros((height, width), dtype=bool)
    xs = np.arange(width) + 0.5
    for poly in polygons:
        p = np.asarray(poly, dtype=float).reshape(-1, 2)
        if len(p) < 3:
            continue
        q = np.roll(p, -1, axis=0)
        y_lo = max(int(np.floor(p[:, 1].min() - 0.5)), 0)
        y_hi = min(int(np.ceil(p[:, 1].max() - 0.5)), height - 1)
        for row in range(y_lo, y_hi + 1):
            y = row + 0.5
            crosses = (p[:, 1] <= y) != (q[:, 1] <= y)
            if not np.any(crosses):
                continue
            a, b = p[crosses], q[crosses]
            x_hit = a[:, 0] + (y - a[:, 1]) * (b[:, 0] - a[:, 0]) / (b[:, 1] - a[:, 1])
            count = np.sum(xs[:, None] > x_hit[None, :], axis=1)
            out[row] ^= (count % 2 == 1)
    return out


def mesh_silhouette(camera: Camera, vertices: Any, faces: Any, mask: np.ndarray | None = None) -> tuple[np.ndarray, int]:
    """Silhouette of a world-space mesh seen through ``camera``; returns ``(mask, skipped)``."""
    uv, _, valid = camera.project(vertices)
    return rasterize_triangles(uv, faces, camera.width, camera.height, valid, mask)


def mask_iou(a: np.ndarray, b: np.ndarray) -> float | None:
    """Intersection over union, or None when both masks are empty."""
    union = np.count_nonzero(a | b)
    if union == 0:
        return None
    return float(np.count_nonzero(a & b) / union)


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """``(x0, y0, x1, y1)`` inclusive pixel bounds of a mask, or None when empty."""
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if len(rows) == 0:
        return None
    return int(cols[0]), int(rows[0]), int(cols[-1]), int(rows[-1])
