# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""Isolate optional physics refinement with supplied synthetic poses, masks and cameras.

The pipeline tests use registered pose/mask stand-ins, not RGB perception or benchmark truth.
They protect evidence selection, rejection, held-out independence and final residual reporting.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from openbeam_v2e.perception import interfaces, registry
from openbeam_v2e.perception.types import PoseTrack, Segmentation
from openbeam_v2e.reconstruct import dynamics_fit, events, geom, hulls, pipeline, pose_fit, silhouette
from openbeam_v2e.reconstruct.clip_config import ReconstructionConfig
from openbeam_v2e.reconstruct.video import ArrayFrames

FPS = 30.0


def _track(position, motion="free", sigma=0.01):
    position = np.asarray(position, float).copy()
    n = len(position)
    cov = np.tile(np.eye(6) * sigma ** 2, (n, 1, 1))
    return PoseTrack("obj", "prop.box", "box", {}, np.array([1.0, 0.8, 0.6]), np.full(3, 0.1),
                     "box", np.arange(n), position, np.tile([0.0, 0.0, 0.0, 1.0], (n, 1)), cov,
                     np.gradient(position, 1 / FPS, axis=0), np.ones(n, bool), np.ones(n, bool),
                     np.zeros(n, bool), np.full(n, 0.123), motion)


def _flight(n=18):
    t = np.arange(n) / FPS
    return np.c_[0.5 * t, -0.2 * t, 2.0 + 2.5 * t - 0.5 * dynamics_fit.G * t ** 2]


def _run(track, physics=None, ground_z=0.0, heldout_mask_shift=0):
    # Exact projected masks serve ONLY as residuals in this isolated physics stage. The pose
    # stand-in returns its supplied track without reading those masks (including held-out ones).
    cam = geom.Camera(geom.Intrinsics.from_fov(160, 120, 55.0),
                      geom.look_at((4.0, -8.0, 5.0), (0.0, 0.0, 1.0)), np.array([4.0, -8.0, 5.0]))
    hull = hulls.make_hull(track.template, track.dims)
    seg = Segmentation(cam.width, cam.height)
    for i, frame in enumerate(track.frames):
        polygons = silhouette.hull_outline(hull, track.rotation(int(frame)), track.position[i], cam)
        mask = silhouette.polygon_mask(polygons, cam.width, cam.height)
        if track.heldout[i]:
            mask = np.roll(mask, heldout_mask_shift, axis=1)
        seg.track("obj").set(int(frame), mask)

    class Masks(interfaces.Segmenter):
        name = "test.physics_masks"

        def segment(self, video, prompts, frames, cameras=None):
            seg.provenance = [self.provenance()]
            return seg

    class Poses(interfaces.PoseEstimator):
        name = "test.physics_poses"

        def fit(self, masks, cameras, spec, frames, context):
            return track

    registry.register(Masks)
    registry.register(Poses)
    try:
        cfg = ReconstructionConfig.parse({
            "camera": {"calibration": {"camera": cam.to_dict()}, "tracking": "static"},
            "segmentation": Masks.name, "pose": Poses.name,
            "objects": [{"id": "obj", "class": "prop.box"}], "ground_z": ground_z,
            "physics": physics or {}, "holdout": {"frames": track.frames[track.heldout].tolist()}})
        video = ArrayFrames(np.zeros((len(track), cam.height, cam.width, 3), np.uint8), FPS)
        return pipeline.run_pipeline(cfg, video)
    finally:
        registry.unregister("segmentation", Masks.name)
        registry.unregister("pose", Poses.name)


def test_missing_flight_events_do_not_refit_a_resting_free_object():
    tr = _track(np.zeros((24, 3)))
    before = tr.position.copy(), tr.velocity.copy(), tr.iou.copy()
    result = _run(tr)
    assert np.array_equal(tr.position, before[0]) and np.array_equal(tr.velocity, before[1])
    assert np.array_equal(tr.iou, before[2])
    assert not tr.diagnostics.get("ballistic_segments")
    assert not any(e.kind == "separation" for e in result.events)


def test_clearance_does_not_establish_gravity_only_motion():
    # A supported/held body above the reference plane has clearance, but is not in free fall.
    tr = _track(np.tile([0.0, 0.0, 2.0], (24, 1)))
    before = tr.position.copy(), tr.velocity.copy(), tr.iou.copy()
    _run(tr)
    assert np.array_equal(tr.position, before[0]) and np.array_equal(tr.velocity, before[1])
    assert np.array_equal(tr.iou, before[2])
    record, = tr.diagnostics["ballistic_segments"]
    assert not record["applied"] and record["chi2_per_dof"] > 4.0
    assert "consistency limit" in record["reason"]


def test_uncertain_clearance_does_not_select_a_whole_clip_fallback():
    tr = _track(np.tile([0.0, 0.0, 0.2], (18, 1)), sigma=0.2)
    before = tr.position.copy()
    _run(tr)
    assert np.array_equal(tr.position, before) and not tr.diagnostics.get("ballistic_segments")


@pytest.mark.parametrize("motion,enabled", [("ground", True), ("free", False)])
def test_explicit_ground_or_disabled_physics_is_authoritative(motion, enabled):
    tr = _track(_flight(), motion)
    before = tr.position.copy(), tr.velocity.copy(), tr.iou.copy()
    _run(tr, {"ballistic_segments": enabled})
    assert np.array_equal(tr.position, before[0]) and np.array_equal(tr.velocity, before[1])
    assert np.array_equal(tr.iou, before[2]) and not tr.diagnostics.get("ballistic_segments")


def test_ballistic_refinement_never_fits_heldout_or_unobserved_poses():
    positions = _flight()
    tr = _track(positions)
    tr.heldout[2::4] = True
    tr.fitted[tr.heldout] = False
    tr.observed[5] = tr.fitted[5] = False
    tr.position[tr.heldout] += [10.0, -20.0, 30.0]
    tr.position[5] -= 100.0
    flags = tr.observed.copy(), tr.fitted.copy(), tr.heldout.copy(), tr.cov.copy()
    fit = dynamics_fit.apply_ballistic(tr, 0, len(tr) - 1, FPS, max_chi2_per_dof=4.0)
    assert fit.applied and np.allclose(tr.position, positions, atol=1e-10)
    assert np.array_equal(tr.observed, flags[0]) and np.array_equal(tr.fitted, flags[1])
    assert np.array_equal(tr.heldout, flags[2]) and np.array_equal(tr.cov, flags[3])
    record, = tr.diagnostics["ballistic_segments"]
    assert record["fit_frames"] == tr.frames[tr.fitted].tolist()
    assert "not measured centre of mass" in record["reference_point"]
    assert "not calibrated" in record["covariance"]


def test_all_clearance_runs_are_refined_and_contact_frames_are_unchanged():
    t = np.arange(72) / FPS
    positions = np.c_[0.5 * t, np.zeros_like(t), np.zeros_like(t)]
    for start in (8, 40):
        dt = np.arange(19) / FPS
        positions[start:start + 19, 2] = 0.5 * dynamics_fit.G * dt * (18 / FPS - dt)
    tr = _track(positions)
    tr.heldout[2::5] = True
    tr.fitted[tr.heldout] = False
    selected = [(9, 25), (41, 57)]
    result = _run(tr)
    records = tr.diagnostics["ballistic_segments"]
    assert [r["frames"] for r in records] == [list(bounds) for bounds in selected]
    assert all(r["applied"] for r in records) and np.allclose(tr.position, positions, atol=1e-10)
    assert all(not set(r["fit_frames"]) & set(tr.frames[tr.heldout]) for r in records)
    assert len([e for e in result.events if "take_off" in e.name]) == 2
    assert len([e for e in result.events if "touchdown" in e.name]) == 2


def test_clip_boundary_flight_does_not_invent_events_and_refreshes_final_ious():
    tr = _track(_flight())
    tr.heldout[2::4] = True
    tr.fitted[tr.heldout] = False
    a, b = copy.deepcopy(tr), copy.deepcopy(tr)
    first = _run(a)
    _run(b, heldout_mask_shift=18)
    assert np.array_equal(a.position, b.position) and np.array_equal(a.velocity, b.velocity)
    assert not any("take_off" in e.name or "touchdown" in e.name for e in first.events)
    for i, frame in enumerate(a.frames):
        obs = pose_fit.make_observation(int(frame), first.camera_track.camera(int(frame)),
                                        first.segmentation.tracks["obj"].mask(int(frame)))
        hull = hulls.make_hull(a.template, a.dims)
        assert a.iou[i] == pose_fit.outline_iou(hull, a.rotation(int(frame)), a.position[i], obs)
    assert np.all(a.iou > 0.95)
    assert np.all(b.iou[b.heldout] < a.iou[a.heldout] - 0.1)
    assert np.array_equal(a.iou[a.fitted], b.iou[b.fitted])
    assert a.diagnostics["ballistic_segments"][0]["applied"]
    assert set(a.diagnostics["pre_ballistic_iou"].values()) == {0.123}


def test_insufficient_fitting_evidence_skips_refinement_without_breaking_pipeline():
    tr = _track(_flight())
    tr.fitted[2:] = False
    tr.heldout[2:] = True
    tr.position[2:] += [0.12, -0.1, 0.15]
    before = tr.position.copy()
    _run(tr)
    assert np.array_equal(tr.position, before)
    record, = tr.diagnostics["ballistic_segments"]
    assert not record["applied"] and "three fitted observations" in record["reason"]


def test_clip_can_begin_and_end_in_separate_airborne_segments():
    t = np.arange(48) / FPS
    positions = np.c_[0.5 * t, np.zeros_like(t), np.zeros_like(t)]
    dt = np.arange(16) / FPS
    positions[:16, 2] = 0.5 * dynamics_fit.G * (dt + 0.25) * (0.75 - dt)
    dt = np.arange(16) / FPS
    positions[32:, 2] = 0.5 * dynamics_fit.G * (dt + 0.05) * (0.75 - dt)
    tr = _track(positions)
    result = _run(tr)
    records = tr.diagnostics["ballistic_segments"]
    assert [r["frames"] for r in records] == [[0, 15], [32, 47]]
    assert all(r["applied"] for r in records)
    assert len([e for e in result.events if "take_off" in e.name]) == 1
    assert len([e for e in result.events if "touchdown" in e.name]) == 1


@pytest.mark.parametrize("ground_z", [0.0, 3.5])
def test_even_a_permissive_consistency_gate_cannot_accept_ground_penetration(ground_z):
    tr = _track(np.tile([0.0, 0.0, ground_z + 0.06], (24, 1)), sigma=1.0)
    before = tr.position.copy(), tr.velocity.copy()
    fit = dynamics_fit.apply_ballistic(tr, 0, len(tr) - 1, FPS, max_chi2_per_dof=None, ground_z=ground_z)
    assert not fit.applied and np.array_equal(tr.position, before[0])
    assert np.array_equal(tr.velocity, before[1])
    record, = tr.diagnostics["ballistic_segments"]
    assert record["min_predicted_height_m"] < 0 and "penetrates ground" in record["reason"]


def test_direct_refinement_cannot_override_explicit_ground_motion():
    with pytest.raises(ValueError, match="requires free motion"):
        dynamics_fit.apply_ballistic(_track(_flight(), "ground"), 0, 17, FPS)


@pytest.mark.parametrize("limit", [0.0, -1.0, np.nan, np.inf])
def test_invalid_ballistic_consistency_limit_is_rejected(limit):
    with pytest.raises(ValueError, match="chi-square limit"):
        ReconstructionConfig.parse({"physics": {"ballistic_max_chi2": limit}})
    with pytest.raises(ValueError, match="chi-square limit"):
        dynamics_fit.apply_ballistic(_track(_flight()), 0, 17, FPS, max_chi2_per_dof=limit)


def test_consistency_gate_is_configurable_but_cannot_disable_clearance_selection():
    assert ReconstructionConfig.parse({}).ballistic_max_chi2 == 4.0
    assert ReconstructionConfig.parse({"physics": {"ballistic_max_chi2": 2.0}}).ballistic_max_chi2 == 2.0
    tr = _track(np.zeros((18, 3)))
    _run(tr, {"ballistic_max_chi2": None})
    assert np.array_equal(tr.position, np.zeros((18, 3)))


@pytest.mark.parametrize("fps", [0.0, -1.0, np.nan, np.inf])
def test_invalid_refinement_fps_is_rejected(fps):
    with pytest.raises(ValueError, match="fps must be positive"):
        dynamics_fit.apply_ballistic(_track(_flight()), 0, 17, fps)


@pytest.mark.parametrize("kind", ["few", "shape", "nan", "times", "sigma"])
def test_invalid_ballistic_measurements_fail_clearly(kind):
    positions, t, sigmas = _flight(), np.arange(18) / FPS, np.full((18, 3), 0.01)
    if kind == "few":
        positions, t, sigmas = positions[:2], t[:2], sigmas[:2]
    elif kind == "shape":
        sigmas = sigmas[:, :2]
    elif kind == "nan":
        positions[0, 0] = np.nan
    elif kind == "times":
        t[2] = t[1]
    else:
        sigmas[0, 0] = -1.0
    with pytest.raises(ValueError, match="ballistic"):
        dynamics_fit.fit_ballistic(t, positions, sigmas)
