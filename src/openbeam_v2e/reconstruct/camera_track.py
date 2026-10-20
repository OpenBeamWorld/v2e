# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""Camera tracking math for a camera that stays at one position (pan, tilt, roll, zoom).

Views from one position are related by a pure rotation plus a change of focal length, so a pixel
seen in one frame maps to another frame exactly, whatever its depth. The functions here solve a
frame's (pan, tilt, roll, focal) by bounded robust least squares:

* :func:`solve_from_rays`: from matched pixels whose viewing rays are known (rays come from a
  frame whose camera is already solved, typically a keyframe);
* :func:`refine_with_landmarks`: from known world points (surveyed landmarks or static scene
  points) seen in the frame, with the camera position held;
* :func:`smooth_track`: Whittaker smoothing of chosen parameters (for example the roll, which few
  features pin down) weighted by their sigmas.

The baseline tracker in :mod:`openbeam_v2e.perception.baseline.camera_tracking` chains these
over a video.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from openbeam_v2e.reconstruct.geom import Camera, Intrinsics, ptz_angles, ptz_rotation
from openbeam_v2e.reconstruct.solve import bounded_lsq

STATE = ("pan", "tilt", "roll", "f")


@dataclass
class FrameSolve:
    state: np.ndarray        # pan, tilt, roll (rad), f (px)
    sigma: np.ndarray        # one-sigma of the four
    rms_px: float
    inliers: np.ndarray      # bool per match
    success: bool


def camera_from_state(intrinsics: Intrinsics, centre: Any, state: Sequence[float]) -> Camera:
    pan, tilt, roll, f = state
    return Camera(intrinsics.with_focal(float(f)), ptz_rotation(pan, tilt, roll), centre)


def state_from_camera(camera: Camera) -> np.ndarray:
    return np.array([*camera.ptz(), camera.intrinsics.f], float)


def _project_dirs(intr: Intrinsics, state: np.ndarray, dirs: np.ndarray) -> np.ndarray:
    R = ptz_rotation(state[0], state[1], state[2])
    c = dirs @ R.T
    z = np.where(c[:, 2] > 1e-9, c[:, 2], np.nan)
    xn = c[:, :2] / z[:, None]
    d = intr.distort(xn)
    return np.stack([state[3] * d[:, 0] + intr.cx, state[3] * intr.aspect * d[:, 1] + intr.cy], axis=1)


def solve_from_rays(intrinsics: Intrinsics, rays: np.ndarray, observed: np.ndarray,
                    guess: Sequence[float], free: Sequence[str] = STATE, huber_px: float = 1.5,
                    sigma_px: float = 0.5, reject_px: float = 3.0, rounds: int = 2,
                    focal_bounds: tuple[float, float] | None = None) -> FrameSolve:
    """Solve one frame's state from world ray directions (N, 3) and their pixels (N, 2)."""
    rays = np.asarray(rays, float)
    obs = np.asarray(observed, float)
    inl = np.all(np.isfinite(obs), axis=1)
    x = np.asarray(guess, float).copy()
    fixed = [s for s in STATE if s not in free]
    lo = {"tilt": -math.pi / 2, "roll": -math.pi / 4, "f": 0.05 * intrinsics.width}
    hi = {"tilt": math.pi / 2, "roll": math.pi / 4, "f": 100.0 * intrinsics.width}
    if focal_bounds is not None:
        lo["f"], hi["f"] = focal_bounds
    res = None
    for _ in range(max(1, rounds)):
        if inl.sum() < 4:
            return FrameSolve(x, np.full(4, np.inf), math.inf, inl, False)
        d, o = rays[inl], obs[inl]

        def resid(s: np.ndarray) -> np.ndarray:
            r = (_project_dirs(intrinsics, s, d) - o) / sigma_px
            return np.where(np.isfinite(r), r, 1e3).ravel()

        res = bounded_lsq(resid, x, STATE, lo, hi, fixed=fixed, loss="huber", f_scale=huber_px / sigma_px,
                          x_scale=[1e-3, 1e-3, 1e-3, max(1.0, 1e-3 * x[3])])
        x = res.x
        err = np.linalg.norm(_project_dirs(intrinsics, x, rays) - obs, axis=1)
        new_inl = np.isfinite(err) & (err < reject_px)
        if np.array_equal(new_inl, inl):
            break
        inl = new_inl
    assert res is not None
    err = np.linalg.norm(_project_dirs(intrinsics, x, rays[inl]) - obs[inl], axis=1)
    sigma = np.array([res.sigma(s) for s in STATE])
    return FrameSolve(x, sigma, float(np.sqrt(np.mean(err ** 2))) if len(err) else math.inf, inl,
                      bool(res.success))


def rays_from_pixels(camera: Camera, uv: np.ndarray) -> np.ndarray:
    return camera.rays(np.asarray(uv, float))


def refine_with_landmarks(intrinsics: Intrinsics, centre: Any, world: np.ndarray, observed: np.ndarray,
                          guess: Sequence[float], free: Sequence[str] = STATE, sigma_px: float = 1.0,
                          huber_px: float = 2.0, reject_px: float = 6.0) -> FrameSolve:
    """Solve one frame's state from known world points; the camera position stays fixed."""
    dirs = np.asarray(world, float) - np.asarray(centre, float)
    dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
    return solve_from_rays(intrinsics, dirs, observed, guess, free, huber_px, sigma_px, reject_px)


def whittaker(y: np.ndarray, w: np.ndarray, lam: float) -> np.ndarray:
    """Whittaker smoother: minimise sum w (z - y)^2 + lam sum (second difference of z)^2."""
    y = np.asarray(y, float)
    w = np.asarray(w, float)
    n = len(y)
    if n < 3:
        return y.copy()
    D = sparse.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(n - 2, n))
    A = sparse.diags(w) + lam * (D.T @ D)
    return spsolve(A.tocsc(), w * y)


def smooth_track(values: np.ndarray, sigma: np.ndarray, lam: float) -> np.ndarray:
    """Smooth one parameter series with weights 1 / sigma^2 (infinite sigma = no data)."""
    s = np.where(np.isfinite(sigma) & (sigma > 0), sigma, np.inf)
    w = np.where(np.isfinite(s), 1.0 / np.maximum(s, 1e-12) ** 2, 0.0)
    if w.max() <= 0:
        return np.asarray(values, float).copy()
    w = w / w.max()
    w = np.maximum(w, 1e-9)
    return whittaker(np.where(np.isfinite(values), values, 0.0), w, lam)


def predict_state(history: Sequence[np.ndarray]) -> np.ndarray:
    """Constant-rate prediction of the next state from the last two."""
    if len(history) == 0:
        raise ValueError("empty history")
    if len(history) == 1:
        return np.asarray(history[-1], float).copy()
    a, b = np.asarray(history[-2], float), np.asarray(history[-1], float)
    nxt = b + (b - a)
    nxt[3] = b[3] * (b[3] / a[3]) if a[3] > 0 else b[3]
    return nxt


def state_angles_deg(state: Sequence[float]) -> tuple[float, float, float, float]:
    return (math.degrees(state[0]), math.degrees(state[1]), math.degrees(state[2]), float(state[3]))


def ptz_of(camera: Camera) -> tuple[float, float, float]:
    return ptz_angles(camera.R_cw)
