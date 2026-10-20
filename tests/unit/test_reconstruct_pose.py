# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""Rigid pose tracks, dimensions, physics fits and events recover synthetic truth.

These tests isolate the pose stage: masks and cameras are the exact synthetic ones, so the
tolerances measure the silhouette fit, the dimension fit and the smoother alone.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from _perception_clips import box_airborne, car_ptz
from openbeam_v2e.perception import registry
from openbeam_v2e.perception.interfaces import PoseContext
from openbeam_v2e.perception.types import CameraTrack, MaskTrack, PoseTrack
from openbeam_v2e.reconstruct import dynamics_fit, events, geom, hulls, objects, pose_fit, render


def _masks(clip, name: str) -> MaskTrack:
    w, h = clip.frames.shape[2], clip.frames.shape[1]
    mt = MaskTrack(name, w, h)
    for f in range(clip.n_frames):
        mt.set(f, clip.masks[name][f])
    return mt


def test_single_frame_fit_from_a_perturbed_start() -> None:
    clip = car_ptz()
    car = clip.object("car")
    f = 10
    obs = pose_fit.make_observation(f, clip.cameras[f], clip.masks["car"][f])
    R0 = geom.rot_z(math.radians(8.0)) @ car.poses.R_wb[f]
    p0 = car.poses.position[f] + np.array([0.4, -0.3, 0.0])
    fit = pose_fit.fit_pose(obs, car.hull, R0, p0, "ground")
    assert np.linalg.norm(fit.position - car.poses.position[f]) < 0.02
    assert math.degrees(geom.geodesic_angle(fit.R, car.poses.R_wb[f])) < 0.5
    assert fit.iou > 0.95
    assert fit.cov[2, 2] == 0.0 and fit.cov[0, 0] > 0.0   # ground motion: z is not free


@pytest.mark.parametrize("offset", [(-100, 0), (100, 0), (0, -75), (0, 75)])
def test_image_edge_does_not_become_an_observed_hull_edge(offset) -> None:
    # Stage-isolated synthetic fixture: exact masks/camera/dimensions and a perturbed-pose start.
    # Moving the principal point crops the same projection at each of the four image borders.
    intr = geom.Intrinsics.from_fov(240, 160, 50.0)
    intr = geom.Intrinsics(intr.width, intr.height, f=intr.f,
                           cx=intr.cx + offset[0], cy=intr.cy + offset[1])
    C = np.array([0.0, -12.0, 5.0])
    cam = geom.Camera(intr, geom.look_at(C, (0.0, 0.0, 0.7)), C)
    hull = hulls.make_hull("vehicle", (4.5, 1.8, 1.5))
    R, p = geom.rot_z(0.2), np.array([0.3, 0.4, 0.0])
    mask = render.render(cam, render.Scene(), [render.RenderObject(1, hull, R, p)]).mask(1)
    assert mask.any() and (mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())
    obs = pose_fit.make_observation(0, cam, mask)
    truth_residuals = pose_fit.silhouette_residuals(np.r_[p[:2], 0.2], hull, obs, "ground", np.eye(3), 0.0)
    assert np.sqrt(np.mean(truth_residuals ** 2)) < 0.5
    fit = pose_fit.fit_pose(obs, hull, geom.rot_z(0.27), p + [0.1, -0.1, 0.0], "ground")
    assert np.linalg.norm(fit.position - p) < 0.04
    assert math.degrees(geom.geodesic_angle(fit.R, R)) < 1.0
    assert fit.iou > 0.97


def test_fully_offscreen_pose_has_finite_nonzero_residuals() -> None:
    clip = car_ptz()
    car = clip.object("car")
    obs = pose_fit.make_observation(10, clip.cameras[10], clip.masks["car"][10])
    residuals = pose_fit.silhouette_residuals(np.array([100.0, 0.0, 0.0]), car.hull, obs,
                                            "ground", np.eye(3), 0.0)
    assert residuals.shape == (120 + len(obs.boundary),)
    assert np.isfinite(residuals).all() and np.mean(residuals ** 2) > 1.0


def test_ground_vehicle_track_dimensions_and_honest_sigmas() -> None:
    clip = car_ptz()
    car = clip.object("car")
    spec = objects.object_spec("car", "vehicle.car")
    est = registry.create("pose", "baseline.silhouette")
    held = list(range(2, clip.n_frames, 5))
    tr = est.fit(_masks(clip, "car"), CameraTrack.from_cameras(clip.cameras), spec, range(clip.n_frames),
                 PoseContext(fps=clip.fps, heldout=held))
    err = tr.position - car.poses.position
    yaw_t = np.array([geom.yaw_pitch_roll(R)[0] for R in car.poses.R_wb])
    yaw_e = np.array([geom.yaw_pitch_roll(geom.quat_to_matrix(q))[0] for q in tr.quat])
    # tolerances: 2 cm position RMS, 0.3 deg heading RMS, 3 cm per dimension
    assert np.sqrt(np.mean(np.sum(err ** 2, axis=1))) < 0.02
    assert np.degrees(np.sqrt(np.mean(geom.wrap_angle(yaw_e - yaw_t) ** 2))) < 0.3
    assert np.abs(tr.dims - np.array(car.hull.dims)).max() < 0.03
    # sigmas cover the errors, held-out frames are reported apart and fit well
    sig = tr.sigma_position()
    assert np.mean(np.abs(err[:, :2]) < 3 * sig[:, :2] + 1e-9) > 0.9
    assert tr.heldout.sum() == len(held) and not np.any(tr.fitted & tr.heldout)
    assert np.nanmedian(tr.iou[tr.heldout]) > 0.95
    speed_t = np.linalg.norm(np.gradient(car.poses.position, axis=0) * clip.fps, axis=1)
    assert np.abs(np.linalg.norm(tr.velocity, axis=1) - speed_t)[2:-2].max() < 0.3


def test_airborne_box_six_degrees_of_freedom() -> None:
    clip = box_airborne()
    box = clip.object("box")
    spec = objects.object_spec("box", "prop.box",
                               dims_prior={"length": (0.8, 0.3), "width": (0.6, 0.3), "height": (0.4, 0.3)})
    tr = registry.create("pose", "baseline.silhouette").fit(
        _masks(clip, "box"), CameraTrack.from_cameras(clip.cameras), spec, range(clip.n_frames),
        PoseContext(fps=clip.fps, heldout=[3, 9, 15]))
    R_est = np.array([geom.quat_to_matrix(q) for q in tr.quat])
    centre_t = box.poses.position + box.poses.R_wb @ np.array([0.0, 0.0, box.hull.height / 2])
    centre_e = tr.position + R_est @ np.array([0.0, 0.0, tr.dims[2] / 2])
    # tolerances: 3 cm centre RMS, 3 deg attitude modulo the box symmetry, 2 cm per dimension
    assert np.sqrt(np.mean(np.sum((centre_e - centre_t) ** 2, axis=1))) < 0.03
    flips = [np.diag([sx, sy, sx * sy]) for sx in (1, -1) for sy in (1, -1)]
    att = [min(geom.geodesic_angle(R_est[i] @ S, box.poses.R_wb[i]) for S in flips) for i in range(clip.n_frames)]
    assert np.degrees(max(att)) < 3.0
    assert np.abs(tr.dims - np.array(box.hull.dims)).max() < 0.02
    assert tr.symmetry == "box" and tr.motion == "free"


def test_ballistic_fit_recovers_launch_state() -> None:
    rng = np.random.default_rng(4)
    t = np.arange(15) / 30.0
    p0, v0 = np.array([1.0, 2.0, 1.5]), np.array([3.0, -1.0, 4.0])
    truth = p0 + v0 * t[:, None]
    truth[:, 2] -= 0.5 * dynamics_fit.G * t * t
    fit = dynamics_fit.fit_ballistic(t, truth + rng.normal(0, 0.01, truth.shape), np.full(truth.shape, 0.01))
    assert np.allclose(fit.p0, p0, atol=0.03) and np.allclose(fit.v0, v0, atol=0.15)
    assert 0.3 < fit.chi2_per_dof < 2.5


def _track(position: np.ndarray, velocity: np.ndarray, motion: str, dims=(1.0, 1.0, 1.0)) -> PoseTrack:
    n = len(position)
    cov = np.zeros((n, 6, 6))
    cov[:, :3, :3] = np.eye(3) * 1e-4
    return PoseTrack("obj", "prop.box", "box", {}, np.array(dims), np.zeros(3), "box", np.arange(n), position,
                     np.tile([0.0, 0.0, 0.0, 1.0], (n, 1)), cov, velocity, np.ones(n, bool), np.ones(n, bool),
                     np.zeros(n, bool), np.full(n, np.nan), motion)


def test_events_rest_take_off_touchdown_and_impact() -> None:
    fps = 30.0
    t = np.arange(60) / fps
    # a body that rests, is launched at frame 20 and lands at frame 40, then rests again
    z = np.zeros_like(t)
    vz = np.zeros_like(t)
    airborne = (t >= 20 / fps) & (t <= 40 / fps)
    tf = t[airborne] - 20 / fps
    z[airborne] = 3.0 * tf - 0.5 * 9.81 * tf * tf
    vz[airborne] = 3.0 - 9.81 * tf
    z = np.maximum(z, 0.0)
    pos = np.c_[np.zeros_like(t), np.zeros_like(t), z]
    vel = np.c_[np.zeros_like(t), np.zeros_like(t), vz]
    evs = events.detect_events(_track(pos, vel, "free"), fps)
    names = [e.name for e in evs]
    assert "take_off" in names and "touchdown" in names and names.count("rest") == 2
    take_off = next(e for e in evs if e.name == "take_off")
    assert take_off.frames[0] <= 21 <= take_off.frames[1] + 1
    assert any(e.name == "impact" for e in evs)   # the landing stops the vertical speed abruptly
    named = events.name_unique(evs)
    assert len({e.name for e in named}) == len(named)


def test_ground_heading_search_finds_the_heading_from_any_start() -> None:
    clip = car_ptz()
    car = clip.object("car")
    f = 20
    obs = pose_fit.make_observation(f, clip.cameras[f], clip.masks["car"][f])
    ranked = pose_fit.search_yaw(obs, car.hull, car.poses.position[f], 0.0, [k * math.pi / 6 for k in range(12)])
    yaw_t = geom.yaw_pitch_roll(car.poses.R_wb[f])[0]
    d = abs(geom.wrap_angle(ranked[0][1] - yaw_t))
    assert min(d, abs(d - math.pi)) < math.radians(16)


@pytest.mark.parametrize("reverse", [False, True])
def test_vehicle_heading_does_not_assume_forward_motion(reverse) -> None:
    # Reverse the synthetic clip in time: the same body heading now has negative forward speed.
    # Exact masks and cameras isolate pose fitting, not end-to-end video reconstruction.
    clip = car_ptz()
    car = clip.object("car")
    indices = list(range(12, 25))
    if reverse:
        indices.reverse()
    masks = MaskTrack("car", clip.frames.shape[2], clip.frames.shape[1])
    for f, i in enumerate(indices):
        masks.set(f, clip.masks["car"][i])
    cameras = CameraTrack.from_cameras([clip.cameras[i] for i in indices])
    tr = registry.create("pose", "baseline.silhouette").fit(
        masks, cameras, objects.object_spec("car", fit_dims=False), range(len(indices)),
        PoseContext(fps=clip.fps, heldout=[3, 8]))
    yaw = np.array([geom.yaw_pitch_roll(tr.rotation(f))[0] for f in range(len(indices))])
    truth = np.array([geom.yaw_pitch_roll(car.poses.R_wb[i])[0] for i in indices])
    assert np.degrees(np.max(np.abs(geom.wrap_angle(yaw - truth)))) < 1.0
    along_body = np.array([np.dot(tr.velocity[f], tr.rotation(f)[:, 0]) for f in range(len(indices))])
    assert ((-1 if reverse else 1) * along_body > 1.0).all()
    assert tr.heldout.sum() == 2 and not np.any(tr.fitted & tr.heldout)


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("scene_yaw", [0.0, math.pi - 0.1])
def test_partial_entry_uses_visible_shape_without_reversing_the_body_heading(reverse, scene_yaw) -> None:
    # Stage-isolated fixture: masks from a constant-heading vehicle entering the image. The
    # estimator receives exact masks/camera and a class dimension prior, never trajectory/heading.
    from openbeam_v2e.reconstruct.rigid_track import TrackOptions, fit_rigid_track

    n, fps = 36, 30.0
    intr = geom.Intrinsics.from_fov(240, 160, 50.0)
    scene_R = geom.rot_z(scene_yaw)
    C = scene_R @ np.array([0.0, -12.0, 5.0])
    cam = geom.Camera(intr, geom.look_at(C, (0.0, 0.0, 0.7)), C)
    hull = hulls.make_hull("vehicle", (4.5, 1.8, 1.5))
    R = scene_R @ geom.rot_z(0.2)
    sign = -1 if reverse else 1
    position = sign * (scene_R @ np.array([-10.0, -1.0, 0.0]) + 0.45 * np.arange(n)[:, None] * R[:, 0])
    masks = MaskTrack("car", cam.width, cam.height)
    for f, p in enumerate(position):
        masks.set(f, render.render(cam, render.Scene(), [render.RenderObject(1, hull, R, p)]).mask(1))
    held = masks.frames()[2::5]
    tr = fit_rigid_track(masks, CameraTrack.from_cameras([cam] * n),
                         objects.object_spec("car", fit_dims=False), range(n), fps, heldout=held,
                         options=TrackOptions(dims_rounds=0))
    yaw = np.array([geom.yaw_pitch_roll(tr.rotation(f))[0] for f in range(n)])
    assert np.degrees(np.max(np.abs(geom.wrap_angle(yaw - (scene_yaw + 0.2))))) < 5.0
    assert np.sqrt(np.mean(np.sum((tr.position - position) ** 2, axis=1))) < 0.1
    assert np.nanmedian(tr.iou[tr.heldout]) > 0.97
    assert np.all(tr.observed == np.array([f in masks.crops and masks.mask(f).sum() >= 40 for f in range(n)]))
    assert tr.heldout.sum() == len(held) and not np.any(tr.fitted & tr.heldout)
    assert sign * np.dot(tr.velocity[0], R[:, 0]) > 10.0
    init = tr.diagnostics["ground_initialization"]
    assert init["seed_untruncated"] and init["seed_frame"] not in held
    assert init["image_clipped_frames"] and not set(init["image_clipped_frames"]) & set(held)


def test_visible_evidence_can_override_the_clipped_heading_prior() -> None:
    from openbeam_v2e.reconstruct.rigid_track import TrackOptions, fit_rigid_track

    C = np.array([0.0, -12.0, 5.0])
    cam = geom.Camera(geom.Intrinsics.from_fov(240, 160, 50.0),
                      geom.look_at(C, (0.0, 0.0, 0.7)), C)
    hull = hulls.make_hull("vehicle", (4.5, 1.8, 1.5))
    masks = MaskTrack("car", cam.width, cam.height)
    positions = [np.array([0.0, 0.4, 0.0]), np.array([-4.5, 0.4, 0.0])]
    for f, (p, yaw) in enumerate(zip(positions, (0.0, 0.9))):
        masks.set(f, render.render(cam, render.Scene(), [render.RenderObject(1, hull, geom.rot_z(yaw), p)]).mask(1))
    tr = fit_rigid_track(masks, CameraTrack.from_cameras([cam] * 2), objects.object_spec("car", fit_dims=False),
                         range(2), 1.0, options=TrackOptions(smooth_attitude=False))
    assert tr.diagnostics["ground_initialization"]["image_clipped_frames"] == [1]
    assert abs(geom.wrap_angle(geom.yaw_pitch_roll(tr.rotation(1))[0] - 0.9)) < math.radians(3)
    assert np.linalg.norm(tr.position[1] - positions[1]) < 0.1


@pytest.mark.parametrize("sigma", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_clipped_heading_sigma_is_rejected(sigma) -> None:
    from openbeam_v2e.reconstruct.rigid_track import TrackOptions, fit_rigid_track

    with pytest.raises(ValueError, match="clipped heading sigma"):
        fit_rigid_track(None, None, None, [], 30.0, options=TrackOptions(clipped_heading_sigma_rad=sigma))


def test_all_clipped_track_reports_no_complete_seed_and_can_disable_the_prior() -> None:
    C = np.array([0.0, -12.0, 5.0])
    cam = geom.Camera(geom.Intrinsics.from_fov(240, 160, 50.0), geom.look_at(C, (0.0, 0.0, 0.7)), C)
    hull = hulls.make_hull("vehicle", (4.5, 1.8, 1.5))
    mask = render.render(cam, render.Scene(), [render.RenderObject(1, hull, geom.rot_z(0.2), [-5.0, 0.4, 0.0])]).mask(1)
    masks = MaskTrack("car", cam.width, cam.height)
    masks.set(0, mask)
    masks.set(1, mask)
    tr = registry.create("pose", "baseline.silhouette", clipped_heading_sigma_deg=None).fit(
        masks, CameraTrack.from_cameras([cam] * 2), objects.object_spec("car", fit_dims=False),
        range(2), PoseContext(fps=1.0))
    init = tr.diagnostics["ground_initialization"]
    assert init["image_clipped_frames"] == [0, 1] and not init["seed_untruncated"]
    assert init["seed_frame"] == 0 and init["clipped_heading_sigma_deg"] is None
    assert init["heading_prior"] == "disabled"
    assert tr.observed.all() and np.isfinite(tr.position).all() and np.isfinite(tr.quat).all()
