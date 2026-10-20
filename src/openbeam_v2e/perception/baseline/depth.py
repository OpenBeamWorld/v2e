# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""Depth baseline and depth alignment.

* :class:`GroundPlaneDepth` (``baseline.ground_plane``): metric depth of the ground plane seen
  by a camera with known pose and intrinsics (NaN above the horizon; pixels that show something
  else get the depth of the ground behind them). It needs no model and serves as the metric
  reference for aligning model depth.
* :func:`align_depth`: fit the scale and shift that turn a relative depth map (or a
  disparity-like inverse depth map) into metres, using pixels whose metric depth is known (the
  ground plane, surveyed points). This is how monocular depth models with unknown scale become
  usable for reconstruction.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from openbeam_v2e.perception.interfaces import DepthEstimator
from openbeam_v2e.perception.types import DepthMap
from openbeam_v2e.reconstruct.geom import Camera


def ground_depth(camera: Camera, ground_z: float = 0.0) -> np.ndarray:
    """Depth along the optical axis of the plane ``z = ground_z`` per pixel (NaN where not seen)."""
    rays = camera.pixel_grid_rays()
    dz = rays[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (ground_z - camera.centre[2]) / dz
    t = np.where((dz < -1e-9) & (t > 0), t, np.nan)
    cam_z = rays @ camera.R_cw[2]
    return t * cam_z


class GroundPlaneDepth(DepthEstimator):
    """``baseline.ground_plane``: metric depth of the ground plane (needs the camera)."""

    name = "baseline.ground_plane"
    summary = "Metric depth of the ground plane from the reference camera (no model)."

    def __init__(self, ground_z: float = 0.0, **params: Any) -> None:
        super().__init__(ground_z=ground_z, **params)

    def estimate(self, image: np.ndarray, camera: Camera | None = None) -> DepthMap:
        if camera is None:
            raise ValueError("baseline.ground_plane needs the camera of the frame")
        return DepthMap(ground_depth(camera, float(self.params["ground_z"])), "metric", None, [self.provenance()])


def align_depth(depth: DepthMap, reference: np.ndarray, mask: np.ndarray | None = None,
                robust_iterations: int = 3) -> tuple[DepthMap, dict[str, float]]:
    """Convert a relative or inverse-relative depth map to metres using a metric reference.

    ``reference`` holds metric depth where known (NaN elsewhere), for example from
    :func:`ground_depth`; ``mask`` limits the pixels used. Relative depth is fitted as
    ``d = s r + t``; inverse-relative as ``1 / d = s r + t``. Iteratively reweighted to resist
    pixels where the reference does not apply (objects standing on the ground).
    """
    if depth.kind == "metric":
        return depth, {"scale": 1.0, "shift": 0.0, "rms_m": float("nan"), "pixels": 0}
    r = np.asarray(depth.depth, float)
    ref = np.asarray(reference, float)
    use = np.isfinite(r) & np.isfinite(ref) & (ref > 0)
    if mask is not None:
        use &= np.asarray(mask, bool)
    if use.sum() < 10:
        raise ValueError("too few reference pixels to align the depth map")
    x = r[use]
    y = ref[use] if depth.kind == "relative" else 1.0 / ref[use]
    w = np.ones_like(x)
    s, t = 1.0, 0.0
    for _ in range(max(1, robust_iterations)):
        A = np.c_[x, np.ones_like(x)] * w[:, None]
        (s, t), *_ = np.linalg.lstsq(A, y * w, rcond=None)
        res = y - (s * x + t)
        mad = 1.4826 * np.median(np.abs(res - np.median(res))) + 1e-12
        w = 1.0 / np.maximum(1.0, np.abs(res) / (2.5 * mad))
    lin = s * r + t
    metric = lin if depth.kind == "relative" else np.where(lin > 1e-6, 1.0 / np.where(lin > 1e-6, lin, 1.0), np.nan)
    err = metric[use] - ref[use]
    info = {"scale": float(s), "shift": float(t), "rms_m": float(np.sqrt(np.mean(err ** 2))),
            "pixels": int(use.sum())}
    return DepthMap(metric, "metric", None, list(depth.provenance)), info
