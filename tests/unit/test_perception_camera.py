# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""Camera calibration and camera tracking baselines recover a known synthetic camera.

Tolerances are stated per test; every clip is rendered from first principles with exact truth.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from _perception_clips import car_ptz, car_ptz_distorted, car_static
from openbeam_v2e.perception import registry
from openbeam_v2e.perception.interfaces import CalibrationInputs
from openbeam_v2e.reconstruct import calibrate, geom
from openbeam_v2e.reconstruct.video import to_gray


def _angles_deg(cam: geom.Camera) -> np.ndarray:
    return np.degrees(np.array(cam.ptz()))


def test_point_calibration_recovers_the_camera_with_honest_sigmas() -> None:
    clip = car_ptz()
    truth = clip.cameras[0]
    world, image = clip.landmark_observations(0, noise_px=0.5, seed=1)
    sel = np.random.default_rng(3).choice(len(world), 16, replace=False)
    res = registry.create("camera_calibration", "baseline.points").calibrate(
        CalibrationInputs(clip.frames.shape[2], clip.frames.shape[1], world_points=world[sel],
                          image_points=image[sel], point_sigma_px=0.5))
    # tolerances: 0.3 deg in angle, 1.5 % in focal length, 0.3 m in position, 1 px reprojection
    assert np.abs(_angles_deg(res.camera) - _angles_deg(truth)).max() < 0.3
    assert abs(res.camera.intrinsics.f / truth.intrinsics.f - 1) < 0.015
    assert np.linalg.norm(res.camera.centre - truth.centre) < 0.3
    assert res.rms_px < 1.0
    # the reported sigmas cover the actual errors (within 4 sigma)
    for k, name in enumerate(("pan", "tilt", "roll")):
        err = abs(_angles_deg(res.camera)[k] - _angles_deg(truth)[k])
        assert err < 4 * max(res.sigmas[name], 1e-3), name
    assert abs(res.camera.intrinsics.f - truth.intrinsics.f) < 4 * res.sigmas["f"]


def test_point_calibration_recovers_lens_distortion() -> None:
    clip = car_ptz_distorted()
    truth = clip.cameras[0]
    world, image = clip.landmark_observations(0, noise_px=0.3, seed=2)
    res = calibrate.calibrate_from_points(world, image, 320, 180, free_distortion=True, sigma_px=0.3)
    assert res.camera.intrinsics.k1 == pytest.approx(truth.intrinsics.k1, abs=0.01)
    assert abs(res.camera.intrinsics.f / truth.intrinsics.f - 1) < 0.01
    assert res.rms_px < 0.6


def test_ground_plane_only_calibration() -> None:
    clip = car_ptz()
    world, image = clip.landmark_observations(0, noise_px=0.3, seed=4)
    ground = world[:, 2] == 0
    res = calibrate.calibrate_from_points(world[ground], image[ground], 320, 180)
    assert res.method.startswith("plane-homography")
    assert np.abs(_angles_deg(res.camera) - _angles_deg(clip.cameras[0])).max() < 0.2


def test_vanishing_point_calibration_from_segments() -> None:
    clip = car_ptz()
    cam = clip.cameras[0]
    rng = np.random.default_rng(5)

    def segments(direction: np.ndarray) -> np.ndarray:
        out = []
        for _ in range(10):
            a = np.array([rng.uniform(-8, 8), rng.uniform(-4, 12), 0.0])
            b = a + 10.0 * direction
            (ua, ub), _ = cam.project(np.array([a, b]))
            out.append(np.r_[ua + rng.normal(0, 0.3, 2), ub + rng.normal(0, 0.3, 2)])
        return np.array(out)

    inputs = CalibrationInputs(320, 180, lines_x=segments(np.array([1.0, 0, 0])),
                               lines_y=segments(np.array([0, 1.0, 0])), camera_height_m=float(cam.centre[2]))
    res = registry.create("camera_calibration", "baseline.vanishing").calibrate(inputs)
    assert abs(res.camera.intrinsics.f / cam.intrinsics.f - 1) < 0.05
    assert abs(_angles_deg(res.camera)[1] - _angles_deg(cam)[1]) < 0.5      # tilt
    assert abs(_angles_deg(res.camera)[2] - _angles_deg(cam)[2]) < 0.5      # roll
    assert res.camera.centre[2] == pytest.approx(cam.centre[2])


def test_automatic_vanishing_points_on_a_tiled_ground() -> None:
    clip = car_ptz()
    cam = clip.cameras[0]
    vps = calibrate.detect_vanishing_points(to_gray(clip.frames[0]), n=4, exclude=clip.masks["car"][0])
    found = [v.pixel() for v in vps if v.pixel() is not None]
    truth = [cam.project_directions([d])[0][0] for d in ([1.0, 0, 0], [0, 1.0, 0])]
    # each true ground direction is found within 2 degrees (seen from the image centre)
    for t in truth:
        best = min(math.degrees(abs(math.atan2(f[1] - 90, f[0] - 160) - math.atan2(t[1] - 90, t[0] - 160)))
                   for f in found)
        assert best < 2.0
    res = registry.create("camera_calibration", "baseline.auto").calibrate(
        CalibrationInputs(320, 180, image=clip.frames[0], camera_height_m=float(cam.centre[2]),
                          exclude_mask=clip.masks["car"][0]))
    assert abs(res.camera.intrinsics.f / cam.intrinsics.f - 1) < 0.1
    assert abs(_angles_deg(res.camera)[1] - _angles_deg(cam)[1]) < 1.5


def test_rotation_zoom_tracker_follows_pan_tilt_roll_and_zoom() -> None:
    clip = car_ptz()
    tr = registry.create("camera_tracking", "baseline.rotation_zoom").track(
        clip.frame_source(), clip.cameras[0], range(clip.n_frames), exclude=lambda f: clip.masks["car"][f])
    truth = np.array([[*c.ptz(), c.intrinsics.f] for c in clip.cameras])
    est = np.c_[tr.pan, tr.tilt, tr.roll, tr.focal]
    err = est - truth
    # tolerances: pan/tilt 0.05 deg, roll 0.1 deg, focal 0.5 %, landmarks 0.5 px
    assert np.degrees(np.abs(err[:, :2])).max() < 0.05
    assert np.degrees(np.abs(err[:, 2])).max() < 0.1
    assert np.abs(err[:, 3] / truth[:, 3]).max() < 0.005
    for f in range(0, clip.n_frames, 5):
        world, uv = clip.landmark_observations(f)
        proj, _ = tr.camera(f).project(world)
        assert np.sqrt(np.mean(np.sum((proj - uv) ** 2, axis=1))) < 0.5
    assert np.all(tr.sigma[1:, :3] > 0)


def test_rotation_zoom_tracker_with_lens_distortion() -> None:
    clip = car_ptz_distorted()
    tr = registry.create("camera_tracking", "baseline.rotation_zoom").track(
        clip.frame_source(), clip.cameras[0], range(clip.n_frames), exclude=lambda f: clip.masks["car"][f])
    truth = np.array([c.ptz() for c in clip.cameras])
    assert np.degrees(np.abs(np.c_[tr.pan, tr.tilt] - truth[:, :2])).max() < 0.05


def test_tracker_on_a_still_camera_stays_still() -> None:
    clip = car_static()
    tr = registry.create("camera_tracking", "baseline.rotation_zoom").track(
        clip.frame_source(), clip.cameras[0], range(clip.n_frames))
    assert np.degrees(np.abs(tr.pan - tr.pan[0])).max() < 0.03
    assert np.abs(tr.focal / tr.focal[0] - 1).max() < 0.002
    static = registry.create("camera_tracking", "baseline.static").track(None, clip.cameras[0], range(5))
    assert len(static) == 5 and np.allclose(static.focal, clip.cameras[0].intrinsics.f)


def test_camera_tracking_from_any_point_tracker() -> None:
    clip = car_ptz()
    tr = registry.create("camera_tracking", "baseline.point_tracks", point_tracker="baseline.lk").track(
        clip.frame_source(), clip.cameras[0], range(clip.n_frames), exclude=lambda f: clip.masks["car"][f])
    truth = np.array([[*c.ptz(), c.intrinsics.f] for c in clip.cameras])
    err = np.c_[tr.pan, tr.tilt, tr.roll, tr.focal] - truth
    # tolerances: pan/tilt 0.1 deg, roll 0.2 deg, focal 0.5 %
    assert np.degrees(np.abs(err[:, :2])).max() < 0.1
    assert np.degrees(np.abs(err[:, 2])).max() < 0.2
    assert np.abs(err[:, 3] / truth[:, 3]).max() < 0.005
    assert {p.model for p in tr.provenance} == {"baseline.point_tracks", "baseline.lk"}


def test_points_stage_override_drives_camera_tracking() -> None:
    from openbeam_v2e.reconstruct.clip_config import ReconstructionConfig

    cfg = ReconstructionConfig.parse({})
    cfg.apply_models({"points": "cotracker"})
    assert cfg.tracking.model == "baseline.point_tracks" and cfg.tracking.params == {"point_tracker": "cotracker"}
