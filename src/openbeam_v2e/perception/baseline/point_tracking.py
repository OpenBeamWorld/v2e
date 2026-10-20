# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""Corner detection and pyramidal Lucas-Kanade point tracking in NumPy/SciPy.

* :func:`good_features`: Shi-Tomasi corners (smallest eigenvalue of the structure tensor) with
  non-maximum suppression, a minimum spacing and an optional exclusion mask;
* :func:`lucas_kanade`: translation-only pyramidal Lucas-Kanade for many points at once
  (inverse-compositional updates, bilinear sampling), with a per-point status;
* :class:`LucasKanadeTracker`: the ``baseline.lk`` point-tracking model built from both, with a
  forward-backward consistency check.

Image coordinates follow :mod:`openbeam_v2e.reconstruct.geom` (pixel centres at +0.5).
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from scipy import ndimage

from openbeam_v2e.perception.interfaces import PointTracker
from openbeam_v2e.perception.types import PointTracks
from openbeam_v2e.reconstruct.video import to_gray


def good_features(gray: np.ndarray, max_corners: int = 400, quality: float = 0.01,
                  min_distance: int = 7, exclude: np.ndarray | None = None, border: int = 8,
                  sigma: float = 1.5) -> np.ndarray:
    """Shi-Tomasi corners as image coordinates (N, 2), strongest first."""
    g = np.asarray(gray, np.float64)
    gx = ndimage.sobel(g, axis=1) / 8.0
    gy = ndimage.sobel(g, axis=0) / 8.0
    a = ndimage.gaussian_filter(gx * gx, sigma)
    b = ndimage.gaussian_filter(gx * gy, sigma)
    c = ndimage.gaussian_filter(gy * gy, sigma)
    lam = 0.5 * (a + c - np.sqrt((a - c) ** 2 + 4.0 * b * b))
    if exclude is not None:
        lam = np.where(exclude, 0.0, lam)
    if border > 0:
        lam[:border] = lam[-border:] = 0.0
        lam[:, :border] = lam[:, -border:] = 0.0
    peak = lam >= ndimage.maximum_filter(lam, size=2 * min_distance + 1)
    thr = quality * float(lam.max()) if lam.max() > 0 else np.inf
    rows, cols = np.nonzero(peak & (lam > thr))
    order = np.argsort(-lam[rows, cols])[:max_corners]
    return np.stack([cols[order] + 0.5, rows[order] + 0.5], axis=1).astype(float)


def _pyramid(gray: np.ndarray, levels: int) -> list[np.ndarray]:
    pyr = [np.asarray(gray, np.float64)]
    for _ in range(1, levels):
        g = ndimage.gaussian_filter(pyr[-1], 1.0)
        pyr.append(g[::2, ::2])
    return pyr


def _sample(img: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Bilinear samples at image coordinates (pixel centres at +0.5)."""
    return ndimage.map_coordinates(img, [y - 0.5, x - 0.5], order=1, mode="nearest")


def lucas_kanade(prev: np.ndarray, nxt: np.ndarray, points: np.ndarray,
                 guess: np.ndarray | None = None, win: int = 7, levels: int = 3, iterations: int = 15,
                 min_eig: float = 1e-5, max_residual: float = 0.08,
                 pyramids: tuple[list[np.ndarray], list[np.ndarray]] | None = None
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Track ``points`` (N, 2) from grey image ``prev`` to ``nxt``.

    Returns the new positions (N, 2), a status (N,) bool and the RMS intensity residual per point.
    ``guess`` (N, 2) optionally gives starting positions in ``nxt`` (e.g. from a motion model).
    """
    P0 = np.asarray(points, float).reshape(-1, 2)
    n = len(P0)
    if n == 0:
        return np.zeros((0, 2)), np.zeros(0, bool), np.zeros(0)
    pyr0, pyr1 = pyramids if pyramids is not None else (_pyramid(prev, levels), _pyramid(nxt, levels))
    levels = min(levels, len(pyr0))
    off = np.arange(-win, win + 1, dtype=float)
    ox, oy = np.meshgrid(off, off)
    ox, oy = ox.ravel(), oy.ravel()
    P1 = np.asarray(guess, float).reshape(-1, 2).copy() if guess is not None else P0.copy()
    ok = np.ones(n, bool)
    resid = np.full(n, np.inf)
    for lev in range(levels - 1, -1, -1):
        s = 2.0 ** lev
        I0, I1 = pyr0[lev], pyr1[lev]
        gx0 = ndimage.sobel(I0, axis=1) / 8.0
        gy0 = ndimage.sobel(I0, axis=0) / 8.0
        p0 = P0 / s
        p1 = P1 / s
        X0 = p0[:, :1] + ox[None]
        Y0 = p0[:, 1:] + oy[None]
        T = _sample(I0, X0.ravel(), Y0.ravel()).reshape(n, -1)
        Tx = _sample(gx0, X0.ravel(), Y0.ravel()).reshape(n, -1)
        Ty = _sample(gy0, X0.ravel(), Y0.ravel()).reshape(n, -1)
        a = np.sum(Tx * Tx, axis=1)
        b = np.sum(Tx * Ty, axis=1)
        c = np.sum(Ty * Ty, axis=1)
        det = a * c - b * b
        eig = 0.5 * (a + c - np.sqrt((a - c) ** 2 + 4 * b * b)) / len(ox)
        good = (eig > min_eig) & (np.abs(det) > 1e-18)
        ia = np.where(good, c / np.where(good, det, 1.0), 0.0)
        ib = np.where(good, -b / np.where(good, det, 1.0), 0.0)
        ic = np.where(good, a / np.where(good, det, 1.0), 0.0)
        for _ in range(iterations):
            X1 = p1[:, :1] + ox[None]
            Y1 = p1[:, 1:] + oy[None]
            I = _sample(I1, X1.ravel(), Y1.ravel()).reshape(n, -1)
            e = I - T
            # brightness offset compensation: remove the mean difference per patch
            e = e - e.mean(axis=1, keepdims=True)
            bx = np.sum(Tx * e, axis=1)
            by = np.sum(Ty * e, axis=1)
            dx = ia * bx + ib * by
            dy = ib * bx + ic * by
            p1 = p1 - np.stack([dx, dy], axis=1)
            if np.all(np.abs(dx) + np.abs(dy) < 0.01):
                break
        if lev == 0:
            X1 = p1[:, :1] + ox[None]
            Y1 = p1[:, 1:] + oy[None]
            I = _sample(I1, X1.ravel(), Y1.ravel()).reshape(n, -1)
            e = I - T
            e = e - e.mean(axis=1, keepdims=True)
            resid = np.sqrt(np.mean(e * e, axis=1))
            ok &= good
        P1 = p1 * s
    h, w = np.asarray(nxt).shape[:2]
    inside = (P1[:, 0] >= win) & (P1[:, 0] <= w - win) & (P1[:, 1] >= win) & (P1[:, 1] <= h - win)
    ok &= inside & np.all(np.isfinite(P1), axis=1) & (resid < max_residual)
    return P1, ok, resid


class LucasKanadeTracker(PointTracker):
    """``baseline.lk``: pyramidal Lucas-Kanade from frame to frame with a forward-backward check."""

    name = "baseline.lk"
    summary = "Pyramidal Lucas-Kanade point tracking (NumPy/SciPy, CPU)."

    def __init__(self, win: int = 7, levels: int = 3, fb_threshold_px: float = 0.7, **params: Any) -> None:
        super().__init__(win=win, levels=levels, fb_threshold_px=fb_threshold_px, **params)

    def track(self, video: Any, queries: np.ndarray, frames: Sequence[int]) -> PointTracks:
        frames = list(frames)
        Q = np.asarray(queries, float).reshape(-1, 3)
        n, T = len(Q), len(frames)
        pts = np.full((n, T, 2), np.nan)
        vis = np.zeros((n, T), bool)
        pos = {f: i for i, f in enumerate(frames)}
        win, levels, fb = self.params["win"], self.params["levels"], self.params["fb_threshold_px"]
        prev_gray = None
        for ti, f in enumerate(frames):
            gray = to_gray(video.read(f))
            # start queries that begin at this frame
            start = np.nonzero(Q[:, 0].astype(int) == f)[0]
            pts[start, ti] = Q[start, 1:]
            vis[start, ti] = True
            if prev_gray is not None and ti > 0:
                alive = vis[:, ti - 1] & ~vis[:, ti]
                idx = np.nonzero(alive)[0]
                if len(idx):
                    p_prev = pts[idx, ti - 1]
                    pyr_a, pyr_b = _pyramid(prev_gray, levels), _pyramid(gray, levels)
                    p_new, ok, _ = lucas_kanade(prev_gray, gray, p_prev, win=win, levels=levels,
                                                pyramids=(pyr_a, pyr_b))
                    p_back, ok_b, _ = lucas_kanade(gray, prev_gray, p_new, win=win, levels=levels,
                                                   pyramids=(pyr_b, pyr_a))
                    good = ok & ok_b & (np.linalg.norm(p_back - p_prev, axis=1) < fb)
                    pts[idx[good], ti] = p_new[good]
                    vis[idx[good], ti] = True
            prev_gray = gray
        del pos
        return PointTracks(np.asarray(frames), pts, vis, [self.provenance()])
