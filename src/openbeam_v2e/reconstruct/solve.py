# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""Bounded least squares with pinned parameters and a covariance estimate.

Every fit in reconstruction (camera calibration, camera tracking, pose fits, dimension fits)
goes through :func:`bounded_lsq`, which adds three things to ``scipy.optimize.least_squares``:

* **pinned parameters:** a parameter whose lower and upper bounds are equal (or that is named in
  ``fixed``) is held at that value and removed from the solve, instead of being rejected;
* **named parameters,** so results and covariances can be read by name;
* **a covariance** from the Jacobian at the solution, ``s^2 (J^T J)^-1`` with ``s^2`` the
  residual variance per degree of freedom (floored at 1 when residuals are already in sigma
  units), optionally inflated for residuals that are correlated (dense samples along an outline
  are not independent measurements).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from scipy.optimize import least_squares


@dataclass
class LsqResult:
    names: tuple[str, ...]
    x: np.ndarray                 # all parameters, pinned ones included
    free: np.ndarray              # bool mask of solved parameters
    cov: np.ndarray               # (n, n) covariance; zero rows/columns for pinned parameters
    residuals: np.ndarray
    cost: float
    rms: float
    dof: int
    success: bool
    message: str
    nfev: int

    def value(self, name: str) -> float:
        return float(self.x[self.names.index(name)])

    def sigma(self, name: str) -> float:
        i = self.names.index(name)
        return float(np.sqrt(max(self.cov[i, i], 0.0)))

    def as_dict(self) -> dict[str, float]:
        return {n: float(v) for n, v in zip(self.names, self.x)}

    def sigmas(self) -> dict[str, float]:
        return {n: float(np.sqrt(max(self.cov[i, i], 0.0))) for i, n in enumerate(self.names)}


def covariance_from_jacobian(J: np.ndarray, residuals: np.ndarray, sigma_units: bool = True,
                             inflation: float = 1.0, robust: bool = False) -> tuple[np.ndarray, int]:
    """``s^2 (J^T J)^-1`` and the degrees of freedom.

    With ``robust`` the residual scale comes from the median absolute deviation, so outliers that
    a robust loss already down-weighted do not inflate the covariance.
    """
    m, n = J.shape
    dof = max(1, m - n)
    if robust:
        s2 = robust_sigma(residuals) ** 2 * m / dof
        if not np.isfinite(s2):
            s2 = float(residuals @ residuals) / dof
    else:
        s2 = float(residuals @ residuals) / dof
    if sigma_units:
        s2 = max(s2, 1.0)
    JTJ = J.T @ J
    try:
        cov = np.linalg.inv(JTJ)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(JTJ)
    if not np.all(np.isfinite(cov)):
        cov = np.linalg.pinv(JTJ)
    return cov * s2 * inflation ** 2, dof


def bounded_lsq(fun: Callable[[np.ndarray], np.ndarray], x0: Sequence[float],
                names: Sequence[str] | None = None,
                lower: Sequence[float] | Mapping[str, float] | None = None,
                upper: Sequence[float] | Mapping[str, float] | None = None,
                fixed: Sequence[str] = (), loss: str = "linear", f_scale: float = 1.0,
                x_scale: Sequence[float] | str = "jac", sigma_units: bool = True,
                inflation: float = 1.0, max_nfev: int | None = None, diff_step: float | None = None,
                **kwargs: Any) -> LsqResult:
    """Minimise ``sum(rho(fun(x)^2))`` within bounds, holding pinned parameters.

    ``fun`` receives the full parameter vector (pinned values included). Bounds may be given as
    sequences or as ``{name: value}`` maps; missing entries are unbounded.
    """
    x0 = np.asarray(x0, float).copy()
    n = len(x0)
    names = tuple(names) if names is not None else tuple(f"p{i}" for i in range(n))
    if len(names) != n:
        raise ValueError("names and x0 differ in length")

    def as_bounds(b: Any, default: float) -> np.ndarray:
        if b is None:
            return np.full(n, default)
        if isinstance(b, Mapping):
            out = np.full(n, default)
            for k, v in b.items():
                out[names.index(k)] = v
            return out
        return np.asarray(b, float)

    lo = as_bounds(lower, -np.inf)
    hi = as_bounds(upper, np.inf)
    if np.any(lo > hi):
        raise ValueError("lower bound above upper bound")
    pinned = (hi - lo) <= 1e-12
    for name in fixed:
        pinned[names.index(name)] = True
    x0 = np.where(pinned & np.isfinite(lo) & ((hi - lo) <= 1e-12), lo, x0)
    x0 = np.clip(x0, lo, hi)
    free = ~pinned
    idx = np.nonzero(free)[0]

    def full(z: np.ndarray) -> np.ndarray:
        x = x0.copy()
        x[idx] = z
        return x

    if len(idx) == 0:
        r = np.asarray(fun(x0), float)
        cost = 0.5 * float(r @ r)
        return LsqResult(names, x0, free, np.zeros((n, n)), r, cost,
                         float(np.sqrt(np.mean(r * r))) if len(r) else 0.0, len(r), True,
                         "all parameters pinned", 1)

    z0 = x0[idx]
    lo_f, hi_f = lo[idx], hi[idx]
    # keep the start strictly inside open bounds (least_squares requires it)
    span = np.where(np.isfinite(hi_f - lo_f), hi_f - lo_f, 1.0)
    z0 = np.clip(z0, lo_f + 1e-9 * span, hi_f - 1e-9 * span)
    xs = x_scale if isinstance(x_scale, str) else np.asarray(x_scale, float)[idx]
    opts: dict[str, Any] = dict(loss=loss, f_scale=f_scale, x_scale=xs, method="trf")
    if max_nfev is not None:
        opts["max_nfev"] = max_nfev
    if diff_step is not None:
        opts["diff_step"] = diff_step
    opts.update(kwargs)
    res = least_squares(lambda z: np.asarray(fun(full(z)), float), z0, bounds=(lo_f, hi_f), **opts)
    x = full(res.x)
    r = np.asarray(fun(x), float)
    cov_free, dof = covariance_from_jacobian(res.jac, r, sigma_units, inflation, robust=loss != "linear")
    cov = np.zeros((n, n))
    cov[np.ix_(idx, idx)] = cov_free
    return LsqResult(names, x, free, cov, r, 0.5 * float(r @ r),
                     float(np.sqrt(np.mean(r * r))) if len(r) else 0.0, dof,
                     bool(res.success), str(res.message), int(res.nfev))


def huber_weights(r: np.ndarray, k: float) -> np.ndarray:
    a = np.abs(r)
    return np.where(a <= k, 1.0, k / np.maximum(a, 1e-12))


def robust_sigma(x: Any) -> float:
    """1.4826 x median absolute deviation (a robust standard deviation)."""
    a = np.asarray(x, float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return float("nan")
    return float(1.4826 * np.median(np.abs(a - np.median(a))))
