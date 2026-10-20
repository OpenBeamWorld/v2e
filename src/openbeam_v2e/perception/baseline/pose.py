# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""``baseline.silhouette``: rigid pose tracks from masks by template-hull silhouette fitting.

A thin model wrapper around :func:`openbeam_v2e.reconstruct.rigid_track.fit_rigid_track` so that
the pose step can be swapped like any other perception model (for example for a learned 6-DoF
pose estimator that returns the same :class:`~openbeam_v2e.perception.types.PoseTrack`).
"""

from __future__ import annotations

import math
from typing import Any, Sequence

from openbeam_v2e.perception.interfaces import PoseContext, PoseEstimator
from openbeam_v2e.perception.types import CameraTrack, MaskTrack, ObjectSpec, PoseTrack
from openbeam_v2e.reconstruct.rigid_track import TrackOptions, fit_rigid_track


class SilhouettePoseEstimator(PoseEstimator):
    """Template-hull silhouette fits per frame, joint dimensions, physics-prior smoothing (CPU)."""

    name = "baseline.silhouette"
    summary = ("Rigid pose tracks and dimensions by fitting template hulls to masks, smoothed with a "
               "constant-acceleration prior; covariance includes the calibration uncertainty (CPU).")

    def __init__(self, dims_rounds: int = 2, dims_frames: int = 10, typical_jerk: float | None = None,
                  typical_angular_jerk: float | None = None, min_area_px: int = 40,
                  smooth_attitude: bool = True, clipped_heading_sigma_deg: float | None = 15.0,
                  dims_max_rms_sigma: float | None = 2.0,
                  **params: Any) -> None:
        super().__init__(dims_rounds=dims_rounds, dims_frames=dims_frames, typical_jerk=typical_jerk,
                         typical_angular_jerk=typical_angular_jerk, min_area_px=min_area_px,
                         smooth_attitude=smooth_attitude, clipped_heading_sigma_deg=clipped_heading_sigma_deg,
                         dims_max_rms_sigma=dims_max_rms_sigma, **params)

    def fit(self, masks: MaskTrack, cameras: CameraTrack, spec: ObjectSpec, frames: Sequence[int],
            context: PoseContext) -> PoseTrack:
        p = self.params
        opts = TrackOptions(min_area_px=int(p["min_area_px"]), dims_rounds=int(p["dims_rounds"]),
                            dims_frames=int(p["dims_frames"]), typical_jerk=p["typical_jerk"],
                            dims_max_rms_sigma=(None if p["dims_max_rms_sigma"] is None else
                                                float(p["dims_max_rms_sigma"])),
                            typical_angular_jerk=p["typical_angular_jerk"],
                            smooth_attitude=bool(p["smooth_attitude"]),
                            clipped_heading_sigma_rad=(None if p["clipped_heading_sigma_deg"] is None else
                                                       math.radians(float(p["clipped_heading_sigma_deg"]))),
                            log=context.options.get("log"),
                            extra={"depth": context.depth} if context.depth is not None else {})
        prov = list(masks.provenance) + list(cameras.provenance) + [self.provenance()]
        return fit_rigid_track(masks, cameras, spec, frames, context.fps, context.ground_z, context.heldout,
                               opts, prov)
