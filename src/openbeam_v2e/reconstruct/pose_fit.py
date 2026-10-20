# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""Single-frame pose fits of a template hull to an object mask.

A pose is fitted by bounded robust least squares on two sets of residuals (see
:mod:`openbeam_v2e.reconstruct.silhouette`): the mask's signed distance at points along the
hull's projected outline, and the outline's signed distance at points on the mask boundary.
The projected polygons are intersected with the image rectangle before comparing outlines:
pixels outside the view are unknown, not evidence that the physical hull ends at the border.
Two motion models are supported:

* ``ground``: the object rests on the ground plane ``z = ground_z``; parameters (x, y, yaw);
* ``free``: six degrees of freedom; parameters (x, y, z) and a small world-frame rotation vector
  applied to a reference attitude, so the solve never meets an Euler singularity.

Every fit returns a covariance from the Jacobian, inflated for the correlation between nearby
outline samples, and the silhouette IoU. Rotation hypotheses (:func:`search_rotations`) start
fits from several attitudes and keep the best, which handles objects whose initial attitude is
unknown.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from openbeam_v2e.reconstruct.geom import Camera, from_yaw_pitch_roll, rotvec_to_matrix, rot_z
from openbeam_v2e.reconstruct.hulls import Hull
from openbeam_v2e.reconstruct.silhouette import (
    boundary_samples,
    clip_polygons,
    hull_outline,
    iou,
    mask_bbox,
    polygon_mask,
    sample_field,
    sample_outline,
    signed_distance,
    union_sdf,
)
from openbeam_v2e.reconstruct.solve import bounded_lsq

GROUND = "ground"
FREE = "free"


@dataclass
class MaskObservation:
    """One frame's mask prepared for fitting (signed distance over a padded crop)."""

    frame: int
    camera: Camera
    mask: np.ndarray            # full-frame bool mask
    sdf: np.ndarray             # signed distance over the crop
    x0: int
    y0: int
    boundary: np.ndarray        # (M, 2) boundary samples, image coordinates
    bbox: tuple[int, int, int, int]
    area: int

    def sdf_at(self, uv: np.ndarray) -> np.ndarray:
        return sample_field(self.sdf, np.asarray(uv, float) - np.array([self.x0, self.y0]))


def make_observation(frame: int, camera: Camera, mask: np.ndarray, n_boundary: int = 160,
                     margin: int = 48) -> MaskObservation | None:
    m = np.asarray(mask, bool)
    box = mask_bbox(m)
    if box is None:
        return None
    x0, y0 = max(0, box[0] - margin), max(0, box[1] - margin)
    x1, y1 = min(m.shape[1], box[2] + margin), min(m.shape[0], box[3] + margin)
    # The observed mask ends at the image border even when the physical hull does not.
    # Pad before the distance transform so that this clipped boundary has distance zero.
    sdf = signed_distance(np.pad(m[y0:y1, x0:x1], 1))[1:-1, 1:-1]
    return MaskObservation(frame, camera, m, sdf, x0, y0, boundary_samples(m, n_boundary), box,
                           int(m.sum()))


@dataclass
class PoseFit:
    R: np.ndarray               # body-to-world rotation
    position: np.ndarray        # body origin, world frame
    cov: np.ndarray             # (6, 6): position, world rotation vector
    params: np.ndarray
    rms_px: float
    iou: float
    success: bool
    cost: float


def pose_from_params(x: np.ndarray, motion: str, R_ref: np.ndarray, ground_z: float) -> tuple[np.ndarray, np.ndarray]:
    if motion == GROUND:
        return rot_z(x[2]) @ R_ref, np.array([x[0], x[1], ground_z])
    return rotvec_to_matrix(x[3:6]) @ R_ref, np.asarray(x[:3], float)


def silhouette_residuals(x: np.ndarray, hull: Hull, obs: MaskObservation, motion: str, R_ref: np.ndarray,
                         ground_z: float, n_outline: int = 120, sigma_px: float = 1.0) -> np.ndarray:
    R, p = pose_from_params(x, motion, R_ref, ground_z)
    polys = hull_outline(hull, R, p, obs.camera)
    m = len(obs.boundary)
    if polys is None:
        return np.full(n_outline + m, 50.0 / sigma_px)
    polys = clip_polygons(polys, obs.camera.width, obs.camera.height)
    if not polys:
        return np.full(n_outline + m, 50.0 / sigma_px)
    pts, w = sample_outline(polys, n_outline)
    r1 = w * obs.sdf_at(pts)
    r2 = union_sdf(obs.boundary, polys) if m else np.zeros(0)
    r = np.r_[r1, r2] / sigma_px
    return np.where(np.isfinite(r), r, 50.0 / sigma_px)


def outline_iou(hull: Hull, R: np.ndarray, p: np.ndarray, obs: MaskObservation) -> float:
    polys = hull_outline(hull, R, p, obs.camera)
    if polys is None:
        return 0.0
    cam = obs.camera
    box = obs.bbox
    pad = 32
    x0, y0 = max(0, box[0] - pad), max(0, box[1] - pad)
    x1, y1 = min(cam.width, box[2] + pad), min(cam.height, box[3] + pad)
    # evaluate on the union of the mask box and the outline box
    allp = np.concatenate(polys)
    x0 = int(max(0, min(x0, np.floor(allp[:, 0].min()))))
    y0 = int(max(0, min(y0, np.floor(allp[:, 1].min()))))
    x1 = int(min(cam.width, max(x1, np.ceil(allp[:, 0].max()) + 1)))
    y1 = int(min(cam.height, max(y1, np.ceil(allp[:, 1].max()) + 1)))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    shifted = [q - np.array([x0, y0]) for q in polys]
    pm = polygon_mask(shifted, x1 - x0, y1 - y0)
    return iou(pm, obs.mask[y0:y1, x0:x1])


def _correlation_inflation(obs: MaskObservation, n_outline: int, corr_px: float) -> float:
    """sqrt(samples / independent samples): neighbouring outline samples share their errors."""
    perim = max(1.0, 2.0 * ((obs.bbox[2] - obs.bbox[0]) + (obs.bbox[3] - obs.bbox[1])))
    n = n_outline + len(obs.boundary)
    n_eff = max(4.0, 2.0 * perim / max(corr_px, 1e-3))
    return math.sqrt(max(1.0, n / n_eff))


def fit_pose(obs: MaskObservation, hull: Hull, R_init: np.ndarray, p_init: np.ndarray, motion: str = GROUND,
             ground_z: float = 0.0, prior: tuple[np.ndarray, np.ndarray] | None = None,
             n_outline: int = 120, sigma_px: float = 1.0, loss: str = "soft_l1", f_scale: float = 2.0,
             corr_px: float = 4.0, max_nfev: int = 200) -> PoseFit:
    """Fit the hull pose to one mask, starting at (R_init, p_init).

    ``prior = (mean, sigma)`` adds soft priors on the parameters (same layout as the solve).
    """
    if motion == GROUND:
        yaw0 = math.atan2(R_init[1, 0], R_init[0, 0])
        R_ref = rot_z(-yaw0) @ R_init
        x0 = np.array([p_init[0], p_init[1], yaw0])
        names = ("x", "y", "yaw")
        scale = [0.05, 0.05, 0.01]
    else:
        R_ref = np.asarray(R_init, float)
        x0 = np.r_[np.asarray(p_init, float), 0.0, 0.0, 0.0]
        names = ("x", "y", "z", "wx", "wy", "wz")
        scale = [0.05, 0.05, 0.05, 0.01, 0.01, 0.01]

    def resid(x: np.ndarray) -> np.ndarray:
        r = silhouette_residuals(x, hull, obs, motion, R_ref, ground_z, n_outline, sigma_px)
        if prior is not None:
            mu, sd = prior
            r = np.r_[r, (x - mu) / sd]
        return r

    infl = _correlation_inflation(obs, n_outline, corr_px)
    res = bounded_lsq(resid, x0, names, loss=loss, f_scale=f_scale, x_scale=scale, inflation=infl,
                      max_nfev=max_nfev)
    R, p = pose_from_params(res.x, motion, R_ref, ground_z)
    cov = np.zeros((6, 6))
    if motion == GROUND:
        cov[np.ix_([0, 1, 5], [0, 1, 5])] = res.cov
    else:
        cov[:] = res.cov
    n_sil = n_outline + len(obs.boundary)
    rms = float(np.sqrt(np.mean(res.residuals[:n_sil] ** 2))) * sigma_px
    return PoseFit(R, p, cov, res.x, rms, outline_iou(hull, R, p, obs), res.success, res.cost)


def initial_position(obs: MaskObservation, hull: Hull, motion: str, ground_z: float,
                     R: np.ndarray | None = None, depth_m: float | None = None) -> np.ndarray:
    """A starting position from the mask alone.

    ``ground``: the viewing ray through the mask centroid meets the plane at half the hull height,
    then drops to the ground. ``free``: the object sits on the centroid ray at ``depth_m`` if
    given, otherwise at the distance where the hull's size matches the mask's size.
    """
    rows, cols = np.nonzero(obs.mask)
    c = np.array([[cols.mean() + 0.5, rows.mean() + 0.5]])
    ray = obs.camera.rays(c)[0]
    C = obs.camera.centre
    if motion == GROUND:
        zc = ground_z + 0.5 * hull.height
        if ray[2] < -1e-6 and C[2] > zc:
            t = (zc - C[2]) / ray[2]
        else:
            t = obs.camera.intrinsics.f * max(hull.dims) / max(1.0, math.sqrt(obs.area))
        p = C + t * ray
        return np.array([p[0], p[1], ground_z])
    if depth_m is None:
        size_px = math.sqrt(max(obs.area, 1))
        size_m = math.sqrt(sorted(hull.dims)[1] * sorted(hull.dims)[2])
        depth_m = obs.camera.intrinsics.f * size_m / size_px
    cz = (obs.camera.R_cw @ ray)[2]
    t = depth_m / max(cz, 1e-3)
    centre = C + t * ray
    Rm = np.eye(3) if R is None else R
    return centre - Rm @ np.array([0.0, 0.0, 0.5 * hull.height])


def search_yaw(obs: MaskObservation, hull: Hull, p: np.ndarray, ground_z: float,
               yaws: Sequence[float]) -> list[tuple[float, float]]:
    """Silhouette cost of each yaw hypothesis at position ``p`` (lowest first)."""
    out = []
    for yaw in yaws:
        r = silhouette_residuals(np.array([p[0], p[1], yaw]), hull, obs, GROUND, np.eye(3), ground_z)
        out.append((float(np.sum(np.minimum(r * r, 25.0))), float(yaw)))
    return sorted(out)


def rotation_hypotheses(n_yaw: int = 8) -> list[np.ndarray]:
    """Attitudes covering the rotation group coarsely: each box face up, several headings."""
    ups = [(0.0, 0.0), (math.pi / 2, 0.0), (-math.pi / 2, 0.0), (0.0, math.pi / 2), (0.0, -math.pi / 2),
           (math.pi, 0.0)]
    out = []
    for roll, pitch in ups:
        for k in range(n_yaw):
            out.append(from_yaw_pitch_roll(2 * math.pi * k / n_yaw, pitch, roll))
    return out


def search_rotations(obs: MaskObservation, hull: Hull, p: np.ndarray, candidates: Sequence[np.ndarray],
                     ground_z: float, keep: int = 3) -> list[np.ndarray]:
    """The ``keep`` candidate attitudes with the lowest silhouette cost at position ``p``."""
    scored = []
    for R in candidates:
        r = silhouette_residuals(np.r_[p, 0.0, 0.0, 0.0], hull, obs, FREE, R, ground_z)
        scored.append((float(np.sum(np.minimum(r * r, 25.0))), R))
    scored.sort(key=lambda s: s[0])
    return [R for _, R in scored[:keep]]


def best_fit(fits: Sequence[PoseFit]) -> PoseFit:
    return min(fits, key=lambda f: (not f.success, f.cost))


def fit_with_hypotheses(obs: MaskObservation, hull: Hull, starts: Sequence[tuple[np.ndarray, np.ndarray]],
                        motion: str, ground_z: float, **kw: Any) -> PoseFit:
    return best_fit([fit_pose(obs, hull, R, p, motion, ground_z, **kw) for R, p in starts])
