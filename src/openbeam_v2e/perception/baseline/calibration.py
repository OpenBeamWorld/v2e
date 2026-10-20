# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""Camera calibration baselines.

* ``baseline.points``: world/pixel correspondences (surveyed points, known object corners, a
  checkerboard laid on the ground) -> intrinsics, distortion, pose with sigmas.
* ``baseline.vanishing``: annotated line segments along two perpendicular ground directions plus
  the camera height or one known ground distance.
* ``baseline.auto``: vanishing points found in the image itself (no annotation) plus the camera
  height or one known distance; refined with any correspondences that are also given. This is
  the least reliable start and says so in its sigmas.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from openbeam_v2e.perception.interfaces import CalibrationInputs, CameraCalibrator
from openbeam_v2e.reconstruct.calibrate import (
    CAMERA_PARAMS,
    CalibrationResult,
    calibrate_from_points,
    camera_from_vanishing_points,
    camera_height_from_distance,
    classify_vanishing_points,
    detect_vanishing_points,
    fit_vanishing_point,
    refine_camera,
)
from openbeam_v2e.reconstruct.geom import Camera
from openbeam_v2e.reconstruct.video import to_gray


def _metric_camera(cam: Camera, inputs: CalibrationInputs) -> Camera:
    """Rescale a unit-height camera with the camera height or one known ground distance."""
    if inputs.camera_height_m is not None:
        h = float(inputs.camera_height_m)
    elif inputs.known_distance is not None:
        a, b, d = inputs.known_distance
        h = camera_height_from_distance(cam, a, b, float(d), inputs.ground_z)
    else:
        raise ValueError("vanishing-point calibration needs camera_height_m or known_distance for scale")
    return camera_from_vanishing_points(*_vps_from_camera(cam), cam.width, cam.height, h, inputs.ground_z,
                                        (cam.intrinsics.cx, cam.intrinsics.cy), inputs.origin_pixel)


def _vps_from_camera(cam: Camera) -> tuple[np.ndarray, np.ndarray]:
    ux, _ = cam.project_directions([[1.0, 0.0, 0.0]])
    uy, _ = cam.project_directions([[0.0, 1.0, 0.0]])
    return ux[0], uy[0]


def _result(cam: Camera, method: str, sigmas: dict[str, float], n: int = 0) -> CalibrationResult:
    names = CAMERA_PARAMS
    cov = np.zeros((len(names), len(names)))
    for i, nme in enumerate(names):
        s = sigmas.get(nme, 0.0)
        cov[i, i] = (math.radians(s) if nme in ("pan", "tilt", "roll") else s) ** 2
    full = {n_: sigmas.get(n_, 0.0) for n_ in names}
    return CalibrationResult(cam, full, float("nan"), np.zeros((0, 2)), method, n, cov)


class PointCalibrator(CameraCalibrator):
    """``baseline.points``: DLT or plane-homography start, bounded least-squares refinement."""

    name = "baseline.points"
    summary = "Calibration from world/pixel point correspondences (>= 4 on the ground or >= 6 general)."

    def calibrate(self, inputs: CalibrationInputs) -> CalibrationResult:
        if inputs.world_points is None or inputs.image_points is None:
            raise ValueError("baseline.points needs world_points and image_points")
        pp = None
        if inputs.intrinsics is not None:
            pp = (inputs.intrinsics.cx, inputs.intrinsics.cy)
        res = calibrate_from_points(inputs.world_points, inputs.image_points, inputs.width, inputs.height,
                                    principal_point=pp, free_distortion=inputs.free_distortion,
                                    sigma_px=inputs.point_sigma_px,
                                    focal_guess=inputs.intrinsics.f if inputs.intrinsics else None)
        return res


class VanishingPointCalibrator(CameraCalibrator):
    """``baseline.vanishing``: two perpendicular ground directions from annotated segments."""

    name = "baseline.vanishing"
    summary = "Calibration from line segments along two perpendicular ground directions plus one scale fact."

    def __init__(self, angle_sigma_deg: float | None = None, **params: Any) -> None:
        super().__init__(angle_sigma_deg=angle_sigma_deg, **params)

    def calibrate(self, inputs: CalibrationInputs) -> CalibrationResult:
        if inputs.lines_x is None or inputs.lines_y is None:
            raise ValueError("baseline.vanishing needs lines_x and lines_y")
        vx, rx = fit_vanishing_point(inputs.lines_x)
        vy, ry = fit_vanishing_point(inputs.lines_y)
        pp = None if inputs.intrinsics is None else (inputs.intrinsics.cx, inputs.intrinsics.cy)
        unit = camera_from_vanishing_points(vx, vy, inputs.width, inputs.height, 1.0, inputs.ground_z, pp)
        cam = _metric_camera(unit, inputs)
        if inputs.world_points is not None and inputs.image_points is not None and len(inputs.world_points) >= 3:
            return refine_camera(inputs.world_points, inputs.image_points, cam,
                                 sigma_px=inputs.point_sigma_px, method="vanishing+refine")
        ang = self.params["angle_sigma_deg"]
        ang = float(ang) if ang is not None else max(0.2, float(np.hypot(rx, ry)))
        sig = {"pan": ang, "tilt": ang, "roll": ang, "f": cam.intrinsics.f * math.radians(ang) * 2.0,
               "cz_w": 0.02 * (inputs.camera_height_m or 1.0)}
        return _result(cam, "vanishing", sig)


class AutoCalibrator(CameraCalibrator):
    """``baseline.auto``: automatic vanishing points from image edges, then optional refinement."""

    name = "baseline.auto"
    summary = ("Calibration without annotation: dominant vanishing points from edges plus the camera "
               "height or one known distance (least reliable; refine with points when you can).")

    def __init__(self, angle_tol_deg: float = 1.5, iterations: int = 2000, seed: int = 0, n_vps: int = 5,
                 focal_agreement: float = 0.05, **params: Any) -> None:
        super().__init__(angle_tol_deg=angle_tol_deg, iterations=iterations, seed=seed, n_vps=n_vps,
                         focal_agreement=focal_agreement, **params)

    def calibrate(self, inputs: CalibrationInputs) -> CalibrationResult:
        if inputs.image is None:
            raise ValueError("baseline.auto needs the reference image")
        gray = to_gray(inputs.image)
        vps = detect_vanishing_points(gray, n=int(self.params["n_vps"]), angle_tol_deg=self.params["angle_tol_deg"],
                                      iterations=self.params["iterations"], seed=self.params["seed"],
                                      exclude=inputs.exclude_mask)
        _, horizontal = classify_vanishing_points(vps, inputs.width, inputs.height)
        pp = np.array([inputs.width / 2.0, inputs.height / 2.0]) if inputs.intrinsics is None else \
            np.array([inputs.intrinsics.cx, inputs.intrinsics.cy])
        # Every pair of finite horizontal vanishing points implies a focal length if the two
        # directions are perpendicular. Grids and road markings give several perpendicular pairs
        # (for a square grid also the two diagonals), which agree on the focal length, while
        # non-perpendicular pairs do not; so prefer the pair whose focal length other pairs share.
        pairs = []
        for i in range(len(horizontal)):
            for j in range(i + 1, len(horizontal)):
                a, b = horizontal[i].pixel(), horizontal[j].pixel()
                if a is None or b is None:
                    continue
                f2 = -float((a - pp) @ (b - pp))
                if f2 <= (0.2 * inputs.width) ** 2 or f2 >= (20.0 * inputs.width) ** 2:
                    continue
                pairs.append((math.sqrt(f2), min(horizontal[i].inliers, horizontal[j].inliers),
                              horizontal[i], horizontal[j]))
        if not pairs:
            raise ValueError("could not find two finite perpendicular vanishing points; annotate lines instead")
        tol = float(self.params["focal_agreement"])

        def agreement(c: tuple) -> int:
            return sum(1 for d in pairs if d is not c and abs(d[0] / c[0] - 1) < tol)

        # the best-supported pair wins; among pairs with nearly as much support (80 %), the one
        # whose focal length other pairs confirm
        top = max(c[1] for c in pairs)
        close = [c for c in pairs if c[1] >= 0.8 * top]
        pick = max(close, key=lambda c: (agreement(c), c[1]))
        best = (pick[1], pick[2], pick[3])
        _, va, vb = best
        # world +X towards the vanishing point further to the right
        if va.pixel()[0] < vb.pixel()[0]:
            va, vb = vb, va
        unit = camera_from_vanishing_points(va.point, vb.point, inputs.width, inputs.height, 1.0,
                                            inputs.ground_z, tuple(pp))
        cam = _metric_camera(unit, inputs)
        if inputs.world_points is not None and inputs.image_points is not None and len(inputs.world_points) >= 3:
            return refine_camera(inputs.world_points, inputs.image_points, cam,
                                 sigma_px=inputs.point_sigma_px, method="auto-vanishing+refine")
        ang = max(0.5, va.rms_deg, vb.rms_deg)
        sig = {"pan": ang, "tilt": ang, "roll": ang, "f": cam.intrinsics.f * math.radians(ang) * 3.0,
               "cz_w": 0.02 * (inputs.camera_height_m or 1.0)}
        return _result(cam, "auto-vanishing", sig)
