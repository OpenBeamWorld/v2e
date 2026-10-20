# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""Stage-isolated dimension fits: exact cameras/masks and perturbed size/pose starts.

The masks are rasterized hulls, not inferred video masks. No fixture pose is pinned in the
joint solve; the held-pose comparison demonstrates the size/pose coupling being tested.
"""

from __future__ import annotations

import numpy as np
import pytest

from openbeam_v2e.perception import registry
from openbeam_v2e.perception.interfaces import PoseContext
from openbeam_v2e.perception.types import CameraTrack, MaskTrack
from openbeam_v2e.reconstruct import geom, hulls, measure, objects, pose_fit, rigid_track, silhouette


def _fixture(scene_yaw=0.0, ground_z=0.0):
    dims = np.array([5.1, 2.1, 1.65])
    hull = hulls.make_hull("vehicle", dims)
    start_dims = np.array([4.5, 1.8, 1.5])
    start_hull = hulls.make_hull("vehicle", start_dims)
    scene_R = geom.rot_z(scene_yaw)
    observations, starts = [], []
    # Constant angular rate makes interpolation at held-out times identifiable in this fixture.
    for f, yaw in enumerate(np.linspace(-0.9, 1.35, 6)):
        C = scene_R @ np.array([0.0, -14.0, 6.0]) + [0.0, 0.0, ground_z]
        target = np.array([0.0, 0.0, ground_z + 0.8])
        cam = geom.Camera(geom.Intrinsics.from_fov(320, 200, 50.0), geom.look_at(C, target), C)
        R = scene_R @ geom.rot_z(yaw)
        p = scene_R @ np.array([0.3 * f, 0.2, 0.0]) + [0.0, 0.0, ground_z]
        mask = silhouette.polygon_mask(silhouette.hull_outline(hull, R, p, cam), cam.width, cam.height)
        obs = pose_fit.make_observation(f, cam, mask)
        # Starting poses are fitted with the wrong initial size, just as in an alternating track
        # solve. The truth pose is used only to perturb this stage-isolated test's initial guess.
        fitted = pose_fit.fit_pose(obs, start_hull, geom.rot_z(0.08) @ R,
                                   p + scene_R @ [0.15, -0.2, 0.0], "ground", ground_z)
        observations.append(obs)
        starts.append((fitted.R, fitted.position))
    prior = {name: (value, 0.8) for name, value in zip(measure.DIM_NAMES, start_dims)}
    return dims, start_dims, observations, starts, prior


@pytest.mark.parametrize("scene_yaw,ground_z", [(0.0, 0.0), (2.8, 1.3)])
def test_joint_ground_fit_recovers_size_instead_of_locking_in_the_starting_pose_bias(scene_yaw, ground_z):
    truth, dims0, obs, starts, prior = _fixture(scene_yaw, ground_z)
    conditional = measure.fit_dimensions("vehicle", {}, dims0, obs, starts, prior, ground_z)
    joint = measure.fit_dimensions("vehicle", {}, dims0, obs, starts, prior, ground_z, motion="ground")
    assert np.max(np.abs(joint.dims - truth)) < 0.08
    assert np.linalg.norm(joint.dims - truth) < np.linalg.norm(conditional.dims - truth) / 3
    assert joint.rms_px < conditional.rms_px / 2
    assert joint.success and joint.frames == list(range(6))
    assert joint.covariance_kind == "local_joint_ground_poses"
    assert len(joint.poses) == len(obs)
    assert all(p[2] == ground_z for _, p in joint.poses)


def test_joint_dimension_covariance_includes_local_pose_coupling():
    _, dims0, obs, starts, prior = _fixture()
    joint = measure.fit_dimensions("vehicle", {}, dims0, obs, starts, prior, motion="ground")
    conditional = measure.fit_dimensions("vehicle", {}, joint.dims, obs, joint.poses, prior)
    assert np.isfinite(joint.sigma).all() and (joint.sigma > 0).all()
    assert (joint.sigma > conditional.sigma).all()
    assert conditional.covariance_kind == "conditional_fixed_poses" and conditional.poses is None


@pytest.mark.parametrize("scene_yaw,ground_z", [(0.0, 0.0), (2.8, 1.3)])
def test_ground_track_learns_nonprior_dimensions_without_fixture_pose_inputs(scene_yaw, ground_z):
    truth, dims0, obs, _, prior = _fixture(scene_yaw, ground_z)
    masks = MaskTrack("car", obs[0].camera.width, obs[0].camera.height)
    for o in obs:
        masks.set(o.frame, o.mask)
    tr = rigid_track.fit_rigid_track(masks, CameraTrack.from_cameras([o.camera for o in obs]),
                                     objects.object_spec("car", dims_prior=prior), range(6), 1.0,
                                     ground_z, heldout=[1, 4])
    assert np.max(np.abs(tr.dims - truth)) < 0.08
    assert np.linalg.norm(tr.dims - truth) < np.linalg.norm(dims0 - truth) / 3
    assert np.nanmedian(tr.iou[tr.heldout]) > 0.95
    assert tr.heldout.sum() == 2 and not np.any(tr.fitted & tr.heldout)
    for record in tr.diagnostics["dims_rounds"]:
        assert not set(record["frames"]) & {1, 4}
        assert record["covariance_kind"] == "local_joint_ground_poses"


def test_free_motion_keeps_the_explicitly_conditional_dimension_solve():
    _, dims0, obs, starts, prior = _fixture()
    implicit = measure.fit_dimensions("vehicle", {}, dims0, obs, starts, prior)
    free = measure.fit_dimensions("vehicle", {}, dims0, obs, starts, prior, motion="free")
    assert np.array_equal(free.dims, implicit.dims)
    assert np.array_equal(free.sigma, implicit.sigma)
    assert free.poses is None and free.covariance_kind == "conditional_fixed_poses"


@pytest.mark.parametrize("n_obs,n_poses", [(0, 0), (1, 0), (0, 1)])
def test_dimension_fit_rejects_empty_or_mismatched_inputs(n_obs, n_poses):
    with pytest.raises(ValueError, match="matching nonempty"):
        measure.fit_dimensions("vehicle", {}, [4.5, 1.8, 1.5], [None] * n_obs, [None] * n_poses, {})


@pytest.mark.parametrize("sigma", [0.0, -1.0, np.nan, np.inf])
def test_dimension_fit_rejects_invalid_boundary_noise(sigma):
    with pytest.raises(ValueError, match="boundary sigma"):
        measure.fit_dimensions("vehicle", {}, [4.5, 1.8, 1.5], [None], [None], {}, sigma_px=sigma)


def _track_inputs(monkeypatch, rms):
    # Control-flow test doubles, not trajectory/size accuracy evidence.
    _, _, obs, _, _ = _fixture()
    masks = MaskTrack("car", obs[0].camera.width, obs[0].camera.height)
    for o in obs:
        masks.set(o.frame, o.mask)

    def fits(observations, hull, spec, ground_z, times, opts, previous=None):
        return {f: pose_fit.PoseFit(np.eye(3), np.array([0.1 * f, 0.2, ground_z]), np.eye(6) * 0.001,
                                    np.zeros(3), rms[f], 0.95, True, 1.0) for f in observations}

    monkeypatch.setattr(rigid_track, "_fit_frames", fits)
    return masks, CameraTrack.from_cameras([o.camera for o in obs])


@pytest.mark.parametrize("motion,limit,sigma_px,rejected", [
    ("ground", 2.0, 1.0, [2, 3]),
    ("ground", 2.0, 2.0, []),
    ("ground", None, 1.0, []),
    ("free", 2.0, 1.0, []),
])
def test_track_measures_only_compatible_fitting_frames(monkeypatch, motion, limit, sigma_px, rejected):
    masks, cameras = _track_inputs(monkeypatch, [0.5, 50.0, 2.5, 3.0, 50.0, 0.5])
    calls = []

    def dimensions(template, params, dims0, obs, poses, prior, ground_z, **kw):
        calls.append([o.frame for o in obs])
        assert kw == {"motion": motion, "sigma_px": sigma_px}
        return measure.DimensionFit(dims0.copy(), np.ones(3) * 0.1, 0.5, calls[-1], True)

    monkeypatch.setattr(rigid_track, "fit_dimensions", dimensions)
    tr = rigid_track.fit_rigid_track(masks, cameras, objects.object_spec("car", motion=motion), range(6),
                                     30.0, heldout=[1, 4], options=rigid_track.TrackOptions(
                                         dims_rounds=1, dims_max_rms_sigma=limit, sigma_px=sigma_px))
    selected = [f for f in [0, 2, 3, 5] if f not in rejected]
    assert calls == [selected]
    record = tr.diagnostics["dims_rounds"][0]
    assert record["frames"] == selected and record["rejected_frames"] == rejected
    assert record["rms_limit_px"] == (limit * sigma_px if motion == "ground" and limit is not None else None)
    assert record["updated"] and record["success"]
    assert tr.heldout.sum() == 2 and not np.any(tr.fitted & tr.heldout)


def test_no_compatible_frame_keeps_the_prior_size_and_uncertainty(monkeypatch):
    masks, cameras = _track_inputs(monkeypatch, [20.0] * 6)

    def bootstrap_only(template, params, dims0, obs, *args, **kw):
        assert "motion" not in kw  # a conditional proposal, never the final joint measurement
        return measure.DimensionFit(dims0.copy(), np.zeros(3), 20.0, [o.frame for o in obs], True)

    monkeypatch.setattr(rigid_track, "fit_dimensions", bootstrap_only)
    tr = registry.create("pose", "baseline.silhouette").fit(
        masks, cameras, objects.object_spec("car"), range(6), PoseContext(fps=30.0, heldout=[1, 4]))
    assert np.array_equal(tr.dims, [4.5, 1.8, 1.5])
    assert np.array_equal(tr.dims_sigma, [0.6, 0.2, 0.25])
    assert tr.diagnostics["dims_rounds"] == [{
        "frames": [], "rejected_frames": [0, 2, 3, 5], "rms_limit_px": 2.0,
        "motion": "ground", "updated": False, "reason": "no compatible fitting frames",
        "bootstrap": {"frames": [0, 2, 3, 5], "dims": [4.5, 1.8, 1.5], "success": True,
                      "kind": "conditional_size_proposal"}}]


def test_failed_dimension_solve_keeps_previous_dimensions_and_reports_the_failure(monkeypatch):
    masks, cameras = _track_inputs(monkeypatch, [0.5] * 6)

    def failed(template, params, dims0, obs, *args, **kw):
        return measure.DimensionFit(dims0 + 10.0, np.zeros(3), 3.0, [o.frame for o in obs], False)

    monkeypatch.setattr(rigid_track, "fit_dimensions", failed)
    tr = rigid_track.fit_rigid_track(masks, cameras, objects.object_spec("car"), range(6), 30.0)
    assert np.array_equal(tr.dims, [4.5, 1.8, 1.5])
    assert np.array_equal(tr.dims_sigma, [0.6, 0.2, 0.25])
    record, = tr.diagnostics["dims_rounds"]
    assert not record["success"] and not record["updated"] and "did not converge" in record["reason"]


def test_failed_conditional_bootstrap_cannot_replace_the_prior(monkeypatch):
    masks, cameras = _track_inputs(monkeypatch, [20.0] * 6)

    def failed(template, params, dims0, obs, *args, **kw):
        assert "motion" not in kw
        return measure.DimensionFit(dims0 + 10.0, np.zeros(3), 3.0, [o.frame for o in obs], False)

    monkeypatch.setattr(rigid_track, "fit_dimensions", failed)
    tr = rigid_track.fit_rigid_track(masks, cameras, objects.object_spec("car"), range(6), 30.0)
    assert np.array_equal(tr.dims, [4.5, 1.8, 1.5])
    assert np.array_equal(tr.dims_sigma, [0.6, 0.2, 0.25])
    record, = tr.diagnostics["dims_rounds"]
    assert not record["bootstrap"]["success"] and not record["updated"]


def test_nonfinite_fit_scores_never_enter_dimension_selection():
    def fit(rms, iou):
        return pose_fit.PoseFit(np.eye(3), np.zeros(3), np.eye(6), np.zeros(3), rms, iou, True, 1.0)

    fits = {0: fit(0.5, 0.9), 1: fit(np.nan, 0.99), 2: fit(0.5, np.inf), 3: fit(np.inf, 0.99)}
    assert rigid_track._dimension_frames(fits, 10, None) == ([0], [1, 2, 3])


@pytest.mark.parametrize("limit", [0.0, -1.0, np.nan, np.inf])
def test_invalid_dimension_rms_limit_is_rejected(limit):
    with pytest.raises(ValueError, match="dimension RMS limit"):
        rigid_track.fit_rigid_track(None, None, None, [], 30.0,
                                     options=rigid_track.TrackOptions(dims_max_rms_sigma=limit))


def test_registry_configuration_can_disable_the_ground_compatibility_gate(monkeypatch):
    masks, cameras = _track_inputs(monkeypatch, [20.0] * 6)

    def dimensions(template, params, dims0, obs, *args, **kw):
        return measure.DimensionFit(dims0.copy(), np.ones(3) * 0.1, 0.5, [o.frame for o in obs], True)

    monkeypatch.setattr(rigid_track, "fit_dimensions", dimensions)
    tr = registry.create("pose", "baseline.silhouette", dims_max_rms_sigma=None, dims_rounds=1).fit(
        masks, cameras, objects.object_spec("car"), range(6), PoseContext(fps=30.0, heldout=[1, 4]))
    record, = tr.diagnostics["dims_rounds"]
    assert record["rms_limit_px"] is None and record["rejected_frames"] == []
    assert record["frames"] == [0, 2, 3, 5]
