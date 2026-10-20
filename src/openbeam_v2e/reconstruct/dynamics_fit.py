# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""Physics-consistent fits of track segments.

* :func:`fit_ballistic`: weighted least squares of a ballistic (airborne) segment with known gravity:
  ``p(t) = p0 + v0 t - g t^2 / 2 z``; returns the initial state with its covariance and a
  chi-square consistency check. Replacing the smoothed positions of an airborne segment by this
  fit is optional (:func:`apply_ballistic`), and the residuals of the original measurements stay
  available for the record.
* :func:`acceleration_bounds`: frames where a track's acceleration exceeds what the object class
  can produce under its own power (a flag for review, not a correction).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from openbeam_v2e.perception.types import PoseTrack
from openbeam_v2e.reconstruct.geom import quat_to_matrix
from openbeam_v2e.reconstruct.hulls import make_hull

G = 9.80665


@dataclass
class BallisticFit:
    p0: np.ndarray
    v0: np.ndarray
    cov: np.ndarray            # (6, 6) over (p0, v0)
    t0: float
    rms_m: float
    chi2_per_dof: float
    applied: bool = False

    def position(self, t: np.ndarray, g: float = G) -> np.ndarray:
        dt = np.asarray(t, float) - self.t0
        p = self.p0[None] + self.v0[None] * dt[:, None]
        p[:, 2] -= 0.5 * g * dt * dt
        return p

    def velocity(self, t: np.ndarray, g: float = G) -> np.ndarray:
        dt = np.asarray(t, float) - self.t0
        v = np.repeat(self.v0[None], len(dt), axis=0)
        v[:, 2] -= g * dt
        return v


def fit_ballistic(times: np.ndarray, positions: np.ndarray, sigmas: np.ndarray, g: float = G) -> BallisticFit:
    """Fit a ballistic (airborne) segment; ``sigmas`` (N, 3) are per-coordinate one-sigma errors."""
    t = np.asarray(times, float)
    P = np.asarray(positions, float)
    S = np.asarray(sigmas, float)
    if t.ndim != 1 or len(t) < 3 or P.shape != (len(t), 3) or S.shape != P.shape:
        raise ValueError("ballistic fit requires at least three times and matching (N, 3) positions/sigmas")
    if not (np.isfinite(t).all() and np.isfinite(P).all() and np.isfinite(S).all() and np.isfinite(g)):
        raise ValueError("ballistic inputs must be finite")
    if np.any(np.diff(t) <= 0) or np.any(S < 0):
        raise ValueError("ballistic times must increase and sigmas must be >= 0")
    S = np.maximum(S, 1e-6)
    t0 = float(t[0])
    dt = t - t0
    p0 = np.zeros(3)
    v0 = np.zeros(3)
    cov = np.zeros((6, 6))
    chi2 = 0.0
    resid = []
    for k in range(3):
        y = P[:, k] + (0.5 * g * dt * dt if k == 2 else 0.0)
        A = np.c_[np.ones_like(dt), dt] / S[:, k:k + 1]
        b = y / S[:, k]
        coef, *_ = np.linalg.lstsq(A, b, rcond=None)
        p0[k], v0[k] = coef
        c = np.linalg.pinv(A.T @ A)
        cov[np.ix_([k, k + 3], [k, k + 3])] = c
        r = b - A @ coef
        chi2 += float(r @ r)
        resid.append((y - (coef[0] + coef[1] * dt)))
    dof = max(1, 3 * len(t) - 6)
    rms = float(np.sqrt(np.mean(np.sum(np.stack(resid, axis=1) ** 2, axis=1))))
    return BallisticFit(p0, v0, cov, t0, rms, chi2 / dof)


def apply_ballistic(track: PoseTrack, start: int, end: int, fps: float, g: float = G,
                    max_chi2_per_dof: float | None = None, ground_z: float | None = None) -> BallisticFit:
    """Refine ``start..end`` from fitted poses only, optionally rejecting an inconsistent model.

    Held-out/interpolated poses are prediction times, never additional measurements. The
    chi-square ratio is a screening heuristic for correlated smoothed poses, not a calibrated
    p-value. This model acts on the tracked body origin, not a measured centre of mass.
    When ``ground_z`` is supplied, a proposal that penetrates that plane is rejected too.
    """
    if track.motion != "free":
        raise ValueError("ballistic refinement requires free motion")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("ballistic fps must be positive and finite")
    if max_chi2_per_dof is not None and (not np.isfinite(max_chi2_per_dof) or max_chi2_per_dof <= 0):
        raise ValueError("ballistic chi-square limit must be positive and finite, or None")
    if ground_z is not None and not np.isfinite(ground_z):
        raise ValueError("ballistic ground height must be finite, or None")
    sel = (track.frames >= start) & (track.frames <= end)
    variance = np.diagonal(track.cov[:, :3, :3], axis1=1, axis2=2)
    measured = sel & track.fitted & track.observed & ~track.heldout \
        & np.all(np.isfinite(track.position), axis=1) \
        & np.all(np.isfinite(variance) & (variance >= 0), axis=1)
    if measured.sum() < 3:
        raise ValueError("ballistic refinement requires at least three fitted observations")
    t = track.frames[measured] / fps
    sig = np.sqrt(np.maximum(variance[measured], 1e-8))
    fit = fit_ballistic(t, track.position[measured], sig, g)
    predicted_t = track.frames[sel] / fps
    predicted_position = fit.position(predicted_t, g)
    reason = None
    if not np.isfinite(fit.chi2_per_dof):
        reason = "nonfinite ballistic consistency statistic"
    elif max_chi2_per_dof is not None and fit.chi2_per_dof > max_chi2_per_dof:
        reason = "ballistic residual exceeds consistency limit"
    min_height = None
    if ground_z is not None:
        hull = make_hull(track.template, track.dims, **track.template_params)
        min_height = float(np.min([hull.lowest_point(quat_to_matrix(q), p) - ground_z
                                   for q, p in zip(track.quat[sel], predicted_position)]))
        if not np.isfinite(min_height) or min_height < 0:
            reason = "ballistic prediction penetrates ground or has nonfinite clearance"
    fit.applied = reason is None
    if fit.applied:
        track.position[sel] = predicted_position
        track.velocity[sel] = fit.velocity(predicted_t, g)
    track.diagnostics.setdefault("ballistic_segments", []).append(
        {"frames": [int(start), int(end)], "fit_frames": track.frames[measured].tolist(),
         "rms_m": fit.rms_m, "chi2_per_dof": fit.chi2_per_dof, "max_chi2_per_dof": max_chi2_per_dof,
         "applied": fit.applied, "reason": reason, "min_predicted_height_m": min_height,
         "reference_point": "tracked body origin; not measured centre of mass",
         "covariance": "input pose covariance retained; correlated refinement uncertainty not calibrated"})
    return fit


def acceleration_bounds(track: PoseTrack, fps: float, max_accel_mps2: float) -> list[int]:
    acc = np.gradient(track.velocity, 1.0 / fps, axis=0)
    a = np.linalg.norm(acc[:, :2], axis=1) if track.motion == "ground" else np.linalg.norm(acc, axis=1)
    return [int(f) for f in track.frames[a > max_accel_mps2]]
