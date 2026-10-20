# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""Object dimensions from silhouettes over many frames, with sigma.

Dimensions and poses are estimated by alternation. Ground dimension fits jointly optimize the
selected frames' poses so a wrong starting size does not pin the measuring geometry. Free-body
dimension fits still hold poses fixed and report explicitly conditional uncertainty: they do not
resolve monocular scale/depth ambiguity. Different views can constrain more dimensions than one
view, but local covariance does not cover wrong motion models, templates or ambiguous modes.

The pose of a hull is its bottom-centre origin, so changing a dimension grows the hull about that
origin; for ``ground`` objects this keeps the hull on the ground.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from openbeam_v2e.reconstruct.geom import rot_z
from openbeam_v2e.reconstruct.hulls import Hull, make_hull
from openbeam_v2e.reconstruct.objects import DIM_NAMES
from openbeam_v2e.reconstruct.pose_fit import MaskObservation, silhouette_residuals
from openbeam_v2e.reconstruct.solve import bounded_lsq


@dataclass
class DimensionFit:
    dims: np.ndarray
    sigma: np.ndarray
    rms_px: float
    frames: list[int]
    success: bool
    poses: list[tuple[np.ndarray, np.ndarray]] | None = None
    covariance_kind: str = "conditional_fixed_poses"


def select_frames(scores: Mapping[int, float], k: int) -> list[int]:
    """Up to ``k`` frames spread evenly over time, preferring well-fitted ones in each slot."""
    frames = sorted(scores)
    if len(frames) <= k:
        return frames
    slots = np.array_split(np.array(frames), k)
    return [int(max(s, key=lambda f: scores[int(f)])) for s in slots if len(s)]


def fit_dimensions(template: str, template_params: Mapping[str, float], dims0: Sequence[float],
                   observations: Sequence[MaskObservation], poses: Sequence[tuple[np.ndarray, np.ndarray]],
                   prior: Mapping[str, tuple[float, float]], ground_z: float = 0.0, n_outline: int = 80,
                   min_dim: float = 0.05, corr_px: float = 4.0,
                   motion: str | None = None, sigma_px: float = 1.0) -> DimensionFit:
    """Fit dimensions, jointly with per-frame (x, y, yaw) for ``motion='ground'``.

    Ground poses are nuisance parameters, not known geometry. The joint covariance accounts
    locally for their coupling with dimensions; it is not calibrated multimodal uncertainty.
    Other motion models retain the fixed-pose solve and its explicitly conditional covariance.
    """
    obs = list(observations)
    pz = list(poses)
    if not obs or len(obs) != len(pz):
        raise ValueError("dimension fitting requires matching nonempty observations and poses")
    if not np.isfinite(sigma_px) or sigma_px <= 0:
        raise ValueError("dimension boundary sigma must be positive and finite")
    mu = np.array([prior.get(n, (dims0[i], 10.0 * dims0[i]))[0] for i, n in enumerate(DIM_NAMES)], float)
    sd = np.array([prior.get(n, (dims0[i], 10.0 * dims0[i]))[1] for i, n in enumerate(DIM_NAMES)], float)
    sd = np.maximum(sd, 1e-3)
    joint = motion == "ground"
    yaw0 = [float(np.arctan2(R[1, 0], R[0, 0])) for R, _ in pz]
    tilt = [rot_z(-yaw) @ R for yaw, (R, _) in zip(yaw0, pz)]

    def poses_of(x: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        if not joint:
            return pz
        return [(rot_z(yaw) @ T, np.array([px, py, ground_z]))
                for (px, py, yaw), T in zip(x[3:].reshape(-1, 3), tilt)]

    def hull_of(d: np.ndarray) -> Hull:
        return make_hull(template, d, **dict(template_params))

    n_res = sum(n_outline + len(o.boundary) for o in obs)

    def resid(x: np.ndarray) -> np.ndarray:
        d = x[:3]
        try:
            hull = hull_of(d)
        except ValueError:
            return np.full(n_res + 3, 50.0 / sigma_px)
        parts = []
        for o, (R, p) in zip(obs, poses_of(x)):
            parts.append(silhouette_residuals(np.r_[p, 0.0, 0.0, 0.0], hull, o, "free", R, ground_z,
                                             n_outline, sigma_px))
        parts.append((d - mu) / sd)
        return np.concatenate(parts)

    perim = sum(2.0 * ((o.bbox[2] - o.bbox[0]) + (o.bbox[3] - o.bbox[1])) for o in obs)
    infl = float(np.sqrt(max(1.0, n_res / max(4.0, 2.0 * perim / corr_px))))
    x0 = np.asarray(dims0, float)
    names = list(DIM_NAMES)
    scale = [0.05] * 3
    if joint:
        x0 = np.r_[x0, np.array([[p[0], p[1], yaw] for (_, p), yaw in zip(pz, yaw0)]).ravel()]
        names += [f"frame{i}.{name}" for i in range(len(obs)) for name in ("x", "y", "yaw")]
        scale += [0.05, 0.05, 0.01] * len(obs)
    res = bounded_lsq(resid, x0, names, lower=[min_dim] * 3 + [-np.inf] * (len(x0) - 3),
                      upper=[100.0] * 3 + [np.inf] * (len(x0) - 3), loss="soft_l1", f_scale=2.0, x_scale=scale,
                      inflation=infl, max_nfev=120 if joint else 60)
    sig = np.array([res.sigma(n) for n in DIM_NAMES])
    rms = float(np.sqrt(np.mean(res.residuals[:-3] ** 2))) * sigma_px if len(res.residuals) > 3 else float("nan")
    return DimensionFit(res.x[:3].copy(), sig, rms, [o.frame for o in obs], res.success,
                        poses_of(res.x) if joint else None,
                        "local_joint_ground_poses" if joint else "conditional_fixed_poses")


def scaled_hull(template: str, dims: Sequence[float], params: Mapping[str, Any]) -> Hull:
    return make_hull(template, dims, **dict(params))
