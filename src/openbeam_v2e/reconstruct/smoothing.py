# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""Physics-consistent smoothing of per-frame measurements.

:func:`rts_smooth` runs a Kalman filter and Rauch-Tung-Striebel smoother over one coordinate
with a constant-acceleration motion model driven by white jerk. It

* weighs every measurement by its own variance (from the per-frame fit covariance),
* bridges frames without a measurement (occlusion, held-out frames) with the model,
* returns smoothed value, velocity and acceleration with their variances for every frame.

The jerk density ``q`` states how quickly acceleration may change, which is the physical prior:
a car's acceleration changes over fractions of a second, a free-flying object's barely at all.
:func:`jerk_density` turns an intuitive "typical jerk" in units per second cubed into ``q``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Smoothed:
    value: np.ndarray       # (T,)
    rate: np.ndarray        # (T,) first derivative
    accel: np.ndarray       # (T,) second derivative
    var: np.ndarray         # (T,) variance of value
    rate_var: np.ndarray    # (T,)


def jerk_density(typical_jerk: float, dt: float) -> float:
    """White-jerk spectral density giving a jerk of about ``typical_jerk`` per step."""
    return float(typical_jerk) ** 2 * dt


def rts_smooth(times: np.ndarray, z: np.ndarray, r: np.ndarray, q: float,
               x0_var: tuple[float, float, float] = (1e4, 1e4, 1e4)) -> Smoothed:
    """Smooth measurements ``z`` (NaN = missing) with variances ``r`` at ``times``.

    ``q`` is the white-jerk spectral density (units^2 / s^5).
    """
    t = np.asarray(times, float)
    z = np.asarray(z, float)
    r = np.asarray(r, float)
    n = len(t)
    have = np.isfinite(z) & np.isfinite(r) & (r > 0)
    if not have.any():
        nan = np.full(n, np.nan)
        return Smoothed(nan, nan, nan, np.full(n, np.inf), np.full(n, np.inf))
    xs = np.zeros((n, 3))
    Ps = np.zeros((n, 3, 3))
    xp = np.zeros((n, 3))
    Pp = np.zeros((n, 3, 3))
    F_all = np.zeros((n, 3, 3))
    first = int(np.argmax(have))
    x = np.array([z[first], 0.0, 0.0])
    P = np.diag(x0_var).astype(float)
    H = np.array([1.0, 0.0, 0.0])
    for k in range(n):
        if k > 0:
            dt = t[k] - t[k - 1]
            F = np.array([[1.0, dt, 0.5 * dt * dt], [0.0, 1.0, dt], [0.0, 0.0, 1.0]])
            Q = q * np.array([[dt ** 5 / 20, dt ** 4 / 8, dt ** 3 / 6],
                              [dt ** 4 / 8, dt ** 3 / 3, dt ** 2 / 2],
                              [dt ** 3 / 6, dt ** 2 / 2, dt]])
            x = F @ x
            P = F @ P @ F.T + Q
            F_all[k] = F
        xp[k], Pp[k] = x, P
        if have[k]:
            S = P[0, 0] + r[k]
            K = P[:, 0] / S
            x = x + K * (z[k] - x[0])
            P = P - np.outer(K, H @ P)
        xs[k], Ps[k] = x, P
    # backward pass
    for k in range(n - 2, -1, -1):
        F = F_all[k + 1]
        try:
            C = Ps[k] @ F.T @ np.linalg.inv(Pp[k + 1])
        except np.linalg.LinAlgError:
            C = Ps[k] @ F.T @ np.linalg.pinv(Pp[k + 1])
        xs[k] = xs[k] + C @ (xs[k + 1] - xp[k + 1])
        Ps[k] = Ps[k] + C @ (Ps[k + 1] - Pp[k + 1]) @ C.T
    return Smoothed(xs[:, 0], xs[:, 1], xs[:, 2], np.maximum(Ps[:, 0, 0], 0.0), np.maximum(Ps[:, 1, 1], 0.0))
