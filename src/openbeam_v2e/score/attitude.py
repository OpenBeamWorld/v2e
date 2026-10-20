# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""Attitude error modulo the rotations an object cannot show.

A symmetric object looks the same after some rotations of its body frame, so footage cannot
tell those attitudes apart. The error between a reference attitude ``R_ref`` and a run attitude
``R_run`` is therefore the smallest geodesic angle of ``R_ref^T R_run S`` over the object's
symmetry group ``S``:

* ``none`` (vehicles): only the identity;
* ``box``: the rotations that map the measured box onto itself (half turns about each axis;
  quarter turns about an axis whose two cross dimensions are equal; the full cube group when
  all three are equal), decided from the hull extents with a relative tolerance;
* ``cylinder``: only the direction of the axis counts (spin about it is invisible); the axis is
  the hull's odd-one-out extent;
* ``cylinder_symmetric``: as ``cylinder``, and the two ends look alike, so the axis sign is
  invisible too;
* ``sphere``: attitude is not scored.
"""

from __future__ import annotations

import itertools
import math
from typing import Any

import numpy as np

from openbeam_v2e.world.geometry import geodesic_angle

__all__ = ["symmetry_group", "attitude_error", "cylinder_axis"]

EQUAL_TOL = 0.05


def _equal(a: float, b: float, tol: float = EQUAL_TOL) -> bool:
    return abs(a - b) <= tol * max(a, b, 1e-9)


def _proper_signed_permutations(axes_free: list[list[int]]) -> list[np.ndarray]:
    """Rotation matrices permuting axes within each group of equal extents (with sign changes)."""
    out = []
    for perm in itertools.permutations(range(3)):
        if any(sorted(perm[i] for i in group) != sorted(group) for group in axes_free):
            continue
        for signs in itertools.product((1.0, -1.0), repeat=3):
            M = np.zeros((3, 3))
            for i, j in enumerate(perm):
                M[j, i] = signs[i]
            if np.linalg.det(M) > 0.5:
                out.append(M)
    return out


def symmetry_group(symmetry: str, extents: Any) -> list[np.ndarray]:
    """Body-frame rotations the object cannot show (``box`` and ``none``; others use axis tests)."""
    if symmetry != "box":
        return [np.eye(3)]
    e = [float(x) for x in extents]
    groups: list[list[int]] = []
    for i in range(3):
        for g in groups:
            if _equal(e[g[0]], e[i]):
                g.append(i)
                break
        else:
            groups.append([i])
    return _proper_signed_permutations(groups)


def cylinder_axis(extents: Any) -> np.ndarray:
    """Body axis of a cylinder: the extent that differs from the other two."""
    e = np.asarray(extents, dtype=float)
    pairs = [(0, 1, 2), (0, 2, 1), (1, 2, 0)]
    best = min(pairs, key=lambda p: abs(e[p[0]] - e[p[1]]) / max(e[p[0]], e[p[1]], 1e-9))
    axis = np.zeros(3)
    axis[best[2]] = 1.0
    return axis


def attitude_error(R_ref: Any, R_run: Any, symmetry: str = "none", extents: Any = (1.0, 1.0, 1.0)) -> np.ndarray:
    """Per-frame attitude error in radians (NaN where either attitude is unknown or not scored).

    ``R_ref`` and ``R_run`` are (F, 3, 3) body-to-world rotations.
    """
    A = np.asarray(R_ref, dtype=float)
    B = np.asarray(R_run, dtype=float)
    F = len(A)
    out = np.full(F, np.nan)
    ok = np.all(np.isfinite(A.reshape(F, -1)), axis=1) & np.all(np.isfinite(B.reshape(F, -1)), axis=1)
    if symmetry == "sphere" or not np.any(ok):
        return out
    if symmetry in ("cylinder", "cylinder_symmetric"):
        axis = cylinder_axis(extents)
        a = A[ok] @ axis
        b = B[ok] @ axis
        dots = np.einsum("ij,ij->i", a, b)
        if symmetry == "cylinder_symmetric":
            dots = np.abs(dots)
        out[ok] = np.arccos(np.clip(dots, -1.0, 1.0))
        return out
    group = symmetry_group(symmetry, extents)
    best = np.full(int(np.count_nonzero(ok)), math.inf)
    for S in group:
        best = np.minimum(best, geodesic_angle(A[ok], B[ok] @ S))
    out[ok] = best
    return out
