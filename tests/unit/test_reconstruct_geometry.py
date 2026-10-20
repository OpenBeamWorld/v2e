# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""Geometry building blocks of reconstruction: cameras, hulls, outlines, solvers, smoothing, I/O."""

from __future__ import annotations

import math

import numpy as np
import pytest

from openbeam_v2e.reconstruct import geom, hulls, render, silhouette, smoothing, solve, video


def test_ptz_angles_round_trip() -> None:
    for pan, tilt, roll in [(0.3, -0.2, 0.05), (-2.0, 0.4, -0.1), (3.0, -1.0, 0.2), (0.0, 0.0, 0.0)]:
        got = geom.ptz_angles(geom.ptz_rotation(pan, tilt, roll))
        assert np.allclose(got, (pan, tilt, roll), atol=1e-9)


def test_zero_ptz_looks_along_world_y_with_image_down_world_minus_z() -> None:
    R_wc = geom.ptz_rotation(0.0, 0.0, 0.0).T
    assert np.allclose(R_wc[:, 2], [0, 1, 0])   # optical axis
    assert np.allclose(R_wc[:, 0], [1, 0, 0])   # image right
    assert np.allclose(R_wc[:, 1], [0, 0, -1])  # image down


def test_distortion_round_trip_and_projection_rays() -> None:
    intr = geom.Intrinsics.from_fov(640, 360, 60.0, k1=-0.12, k2=0.02)
    xn = np.array([[0.3, 0.2], [-0.4, 0.1], [0.0, 0.0]])
    assert np.abs(intr.undistort(intr.distort(xn)) - xn).max() < 1e-9
    cam = geom.Camera(intr, geom.look_at((1.0, -10.0, 3.0), (0.0, 0.0, 0.5)), (1.0, -10.0, 3.0))
    pts = np.array([[0.0, 0.0, 0.0], [2.0, 3.0, 1.0], [-3.0, 1.0, 0.2]])
    uv, z = cam.project(pts)
    rays = cam.rays(uv)
    d = (pts - cam.centre) / np.linalg.norm(pts - cam.centre, axis=1, keepdims=True)
    assert np.allclose(rays, d, atol=1e-8)
    assert np.all(z > 0)


def test_homography_between_views_from_one_position_is_exact() -> None:
    intr = geom.Intrinsics.from_fov(400, 300, 50.0)
    C = np.array([0.0, -15.0, 4.0])
    a = geom.Camera.from_ptz(intr, C, 0.1, -0.2, 0.01)
    b = geom.Camera.from_ptz(intr.with_focal(intr.f * 1.2), C, 0.15, -0.18, 0.0)
    H = geom.homography_between(a, b)
    pts = np.array([[1.0, 2.0, 0.0], [-2.0, 5.0, 1.5], [0.5, 30.0, 0.0]])
    ua, _ = a.project(pts)
    ub, _ = b.project(pts)
    h = np.c_[ua, np.ones(len(ua))] @ H.T
    assert np.allclose(h[:, :2] / h[:, 2:], ub, atol=1e-8)


@pytest.mark.parametrize("template,dims", [("box", (1.0, 0.6, 0.4)), ("vehicle", (4.5, 1.8, 1.5)),
                                           ("cylinder", (0.6, 0.6, 1.0)), ("sphere", (0.5, 0.5, 0.5))])
def test_hull_outline_matches_ray_cast_mask(template: str, dims: tuple[float, float, float]) -> None:
    hull = hulls.make_hull(template, dims)
    assert hull.lowest_point(np.eye(3), (0.0, 0.0, 0.0)) == pytest.approx(0.0, abs=1e-9)
    intr = geom.Intrinsics.from_fov(240, 160, 50.0)
    C = np.array([0.0, -4.0 * max(dims), 1.5 * max(dims)])
    cam = geom.Camera(intr, geom.look_at(C, (0.0, 0.0, 0.3 * dims[2])), C)
    R = geom.from_yaw_pitch_roll(0.4, 0.1, -0.05)
    p = np.array([0.1, 0.2, 0.05])
    rr = render.render(cam, render.Scene(), [render.RenderObject(1, hull, R, p)])
    poly = silhouette.hull_mask(hull, R, p, cam)
    assert silhouette.iou(poly, rr.mask(1)) > 0.97


def test_symmetry_classes_follow_the_objects_schema() -> None:
    assert hulls.make_hull("box", (1, 2, 3)).symmetry == "box"
    assert hulls.make_hull("vehicle", (4.5, 1.8, 1.5)).symmetry == "none"
    assert hulls.make_hull("cylinder", (0.5, 0.5, 1)).symmetry == "cylinder_symmetric"
    assert hulls.make_hull("sphere", (0.3, 0.3, 0.3)).symmetry == "sphere"


def test_signed_distance_and_boundary_points_are_unbiased() -> None:
    m = np.zeros((40, 60), bool)
    m[10:30, 15:45] = True
    sdf = silhouette.signed_distance(m)
    assert sdf[20, 30] < 0 and sdf[0, 0] > 0
    pts = silhouette.boundary_edge_points(m)
    poly = np.array([[15.0, 10.0], [45.0, 10.0], [45.0, 30.0], [15.0, 30.0]])
    assert np.abs(silhouette.union_sdf(pts, [poly])).max() < 1e-9
    # the mask's own signed distance vanishes on its outline
    assert np.abs(silhouette.sample_field(sdf, pts)).max() < 0.51


@pytest.mark.parametrize("reverse", [False, True])
def test_clipping_outlines_to_the_view_preserves_visible_pixels(reverse) -> None:
    polygons = [np.array([[-4.0, -3.0], [15.0, 4.0], [30.0, 18.0], [2.0, 23.0]]),
                np.array([[11.0, 5.0], [18.0, 8.0], [16.0, 13.0]])]
    if reverse:
        polygons = [p[::-1] for p in polygons]
    clipped = silhouette.clip_polygons(polygons, 20, 16)
    assert np.array_equal(silhouette.polygon_mask(clipped, 20, 16),
                          silhouette.polygon_mask(polygons, 20, 16))
    for before, after in zip(polygons, clipped):
        assert np.sign(silhouette.polygon_area(before)) == np.sign(silhouette.polygon_area(after))
        assert (after >= 0).all() and (after <= [20, 16]).all()
    assert np.array_equal(clipped[1], polygons[1])  # interior outlines are unchanged


@pytest.mark.parametrize("poly", [[], [[-4, 0], [-2, 0], [-2, 5]],
                                  [[-4, 0], [0, 0], [0, 5]]])
def test_empty_or_zero_area_view_intersection_has_no_outline(poly) -> None:
    assert silhouette.clip_polygons([np.asarray(poly, float).reshape(-1, 2)], 20, 16) == []


def test_mask_to_polygons_round_trip() -> None:
    m = np.zeros((50, 70), bool)
    m[5:20, 5:40] = True
    m[25:45, 30:60] = True
    m[30:40, 40:50] = False  # a hole is filled in the outline
    polys = silhouette.mask_to_polygons(m, tolerance_px=0.0)
    assert len(polys) == 2
    back = silhouette.polygon_mask(polys, 70, 50)
    filled = m.copy()
    filled[30:40, 40:50] = True
    assert np.array_equal(back, filled)


def test_bounded_lsq_holds_pinned_parameters_and_reports_covariance() -> None:
    rng = np.random.default_rng(0)
    x = np.linspace(0, 1, 50)
    y = 2.0 + 3.0 * x + rng.normal(0, 0.1, len(x))

    def resid(p: np.ndarray) -> np.ndarray:
        return (p[0] + p[1] * x + p[2] * x * x - y) / 0.1

    res = solve.bounded_lsq(resid, [0.0, 0.0, 0.0], ("a", "b", "c"), lower={"c": 0.0}, upper={"c": 0.0})
    assert res.value("c") == 0.0 and res.sigma("c") == 0.0
    assert res.value("b") == pytest.approx(3.0, abs=4 * res.sigma("b"))
    assert 0.0 < res.sigma("a") < 0.1


def test_rts_smoother_recovers_a_trajectory_within_its_sigma() -> None:
    rng = np.random.default_rng(1)
    t = np.arange(90) / 30.0
    truth = 1.0 + 4.0 * t + 0.8 * t * t
    z = truth + rng.normal(0, 0.05, len(t))
    z[40:50] = np.nan  # a gap is bridged by the motion model
    s = smoothing.rts_smooth(t, z, np.full(len(t), 0.05 ** 2), smoothing.jerk_density(2.0, 1 / 30.0))
    err = s.value - truth
    assert np.sqrt(np.mean(err ** 2)) < 0.03
    assert np.mean(np.abs(err) < 3 * np.sqrt(s.var)) > 0.95
    assert np.abs(s.rate[45] - (4.0 + 1.6 * t[45])) < 0.3


def test_png_round_trip(tmp_path) -> None:
    rng = np.random.default_rng(2)
    for shape in [(7, 9), (7, 9, 3), (5, 4, 4)]:
        a = rng.integers(0, 256, size=shape, dtype=np.uint8)
        video.write_png(tmp_path / "x.png", a)
        assert np.array_equal(video.read_png(tmp_path / "x.png"), a)


def test_image_sequence_reads_numbered_frames(tmp_path) -> None:
    for i in (10, 2, 1):
        video.write_png(tmp_path / f"frame_{i}.png", np.full((4, 6, 3), i, np.uint8))
    seq = video.open_frames(tmp_path, fps=25.0)
    assert len(seq) == 3 and seq.size == (6, 4) and seq.fps == 25.0
    assert [int(seq.read(k)[0, 0, 0]) for k in range(3)] == [1, 2, 10]


def test_geodesic_angle_and_yaw_pitch_roll() -> None:
    R = geom.from_yaw_pitch_roll(0.3, -0.1, 0.2)
    assert np.allclose(geom.yaw_pitch_roll(R), (0.3, -0.1, 0.2))
    assert geom.geodesic_angle(R, R) == pytest.approx(0.0, abs=1e-7)
    assert geom.geodesic_angle(np.eye(3), geom.rot_z(math.radians(30))) == pytest.approx(math.radians(30))
