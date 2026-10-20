# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""Tests for the world-package format, the world model, cameras, rasterisation and export."""

from __future__ import annotations

import hashlib
import json
import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import openbeam_v2e.fourdcb.format as fmt
from openbeam_v2e.schema.runlog import ActorState, Row, RunHeader, Runlog
from openbeam_v2e.world import Camera, CameraTrack, read_world, runlog_to_world, write_world
from openbeam_v2e.world import geometry as geo
from openbeam_v2e.world.io import fill_gaps, to_base_package
from openbeam_v2e.world.raster import mask_bbox, mask_iou, mesh_silhouette, rasterize_polygons, rasterize_triangles
from openbeam_v2e.world.timeline import frame0_trigger, resample_poses

from world_fixtures import CAR_DIMS, FPS, camera, car_track, synthetic_world


# ----------------------------------------------------------------------------- geometry

@pytest.mark.parametrize("frames", [1, 2, 5])
def test_single_axis_rotation_fixture_keeps_the_batch_axis(frames: int) -> None:
    _, quat = car_track(frames, yaw_rate=0.4)
    assert quat.shape == (frames, 4)
    expected_yaw = 0.4 * np.arange(frames) / FPS
    assert np.allclose(quat[:, 2], np.sin(expected_yaw / 2))
    assert np.allclose(quat[:, 3], np.cos(expected_yaw / 2))


def test_quaternion_helpers_and_geodesic_angle() -> None:
    R = Rotation.from_euler("zyx", [30, 10, -5], degrees=True)
    q = geo.matrix_to_quat(R.as_matrix())
    assert q[3] >= 0
    assert np.allclose(geo.quat_to_matrix(q), R.as_matrix())
    assert np.all(np.isnan(geo.quat_to_matrix(np.array([[np.nan] * 4]))))
    Rb = Rotation.from_euler("z", 40, degrees=True).as_matrix() @ R.as_matrix()
    assert math.degrees(geo.geodesic_angle(R.as_matrix(), Rb)) == pytest.approx(40.0)
    assert math.degrees(geo.yaw_of(Rotation.from_euler("z", 25, degrees=True).as_matrix())) == pytest.approx(25.0)


def test_kabsch_recovers_a_rigid_motion() -> None:
    rng = np.random.default_rng(3)
    src = rng.normal(size=(40, 3))
    R = Rotation.from_rotvec([0.3, -0.2, 0.9]).as_matrix()
    t = np.array([1.0, -2.0, 0.5])
    R_fit, t_fit, rms = geo.kabsch(src, src @ R.T + t)
    assert np.allclose(R_fit, R) and np.allclose(t_fit, t) and rms < 1e-9


def test_principal_frame_orders_axes_by_extent() -> None:
    v, _ = geo.box_mesh(4.0, 2.0, 1.0)
    R = Rotation.from_euler("z", 35, degrees=True).as_matrix()
    frame, centre, extents = geo.principal_frame(v @ R.T + [3, 4, 5])
    assert np.allclose(extents, [4.0, 2.0, 1.0])
    assert np.allclose(centre, [3, 4, 5])
    assert abs(abs(frame[:, 0] @ R[:, 0]) - 1.0) < 1e-9
    assert np.linalg.det(frame) == pytest.approx(1.0)


def test_meshes_ground_height_and_distances() -> None:
    v, f = geo.cylinder_mesh(0.5, 2.0, axis="x")
    lo, hi = geo.mesh_bounds(v)
    assert np.allclose(hi - lo, [2.0, 1.0, 1.0], atol=1e-9)
    sv, sf = geo.sphere_mesh(1.0)
    assert np.allclose(np.linalg.norm(sv, axis=1), 1.0)
    assert sf.max() < len(sv)
    ramp = np.array([[0, -1, 0], [4, -1, 1], [4, 1, 1], [0, 1, 0]], dtype=float)
    rf = np.array([[0, 1, 2], [0, 2, 3]])
    z = geo.ground_height([[2.0, 0.0], [9.0, 0.0]], ramp, rf)
    assert z[0] == pytest.approx(0.5) and np.isnan(z[1])
    d = geo.points_to_triangles_distance([[2.0, 0.0, 2.0], [-1.0, 0.0, 0.0]], ramp, rf)
    assert d[0] == pytest.approx(1.5 / math.sqrt(1 + 1 / 16), rel=1e-6)
    assert d[1] == pytest.approx(1.0)


# ----------------------------------------------------------------------------- camera

def test_camera_projects_the_principal_point_and_round_trips() -> None:
    cam = Camera.from_fov(640, 480, 90.0, centre=(0, -10, 0), target=(0, 0, 0))
    uv, depth, valid = cam.project([[0, 0, 0], [0, 0, 5], [0, -20, 0]])
    assert np.allclose(uv[0], [320, 240]) and depth[0] == pytest.approx(10.0)
    assert uv[1, 1] < 240  # world up is image up
    assert not valid[2]
    distorted = Camera(640, 480, cam.fx, cam.fy, 320, 240, cam.R_wc, cam.centre, k1=0.1)
    uv_d, _, _ = distorted.project([[5, 0, 0]])
    uv_u, _, _ = cam.project([[5, 0, 0]])
    assert uv_d[0, 0] > uv_u[0, 0]  # barrel-free positive k1 pushes points outwards
    track = CameraTrack.from_cameras([cam, distorted])
    again = CameraTrack.from_json(json.loads(json.dumps(track.to_json())))
    assert again.camera(1).k1 == pytest.approx(0.1)
    assert np.allclose(again.R_wc, track.R_wc)
    with pytest.raises(ValueError):
        CameraTrack.from_cameras([cam, Camera.from_fov(640, 480, 90, centre=(1, -10, 0), target=(0, 0, 0))])
    assert cam.scaled(0.5).width == 320


# ----------------------------------------------------------------------------- raster

def test_rasterize_triangles_and_polygons_agree() -> None:
    square = np.array([[10, 10], [30, 10], [30, 30], [10, 30]], dtype=float)
    tri_mask, skipped = rasterize_triangles(square, [[0, 1, 2], [0, 2, 3]], 40, 40)
    poly_mask = rasterize_polygons([square], 40, 40)
    assert skipped == 0
    assert tri_mask.sum() == 400 and poly_mask.sum() == 400
    assert mask_iou(tri_mask, poly_mask) == 1.0
    assert mask_bbox(tri_mask) == (10, 10, 29, 29)
    assert mask_iou(np.zeros((4, 4), bool), np.zeros((4, 4), bool)) is None
    _, skipped = rasterize_triangles(square, [[0, 1, 2]], 40, 40, valid=[True, False, True, True])
    assert skipped == 1


def test_mesh_silhouette_of_a_box_is_where_the_camera_says() -> None:
    cam = camera()
    v, f = geo.box_mesh(1.0, 1.0, 1.0, centre=(2.0, 0.0, 0.8))
    mask, skipped = mesh_silhouette(cam, v, f)
    assert skipped == 0 and mask.any()
    uv, _, _ = cam.project([[2.0, 0.0, 0.8]])
    x0, y0, x1, y1 = mask_bbox(mask)
    assert x0 <= uv[0, 0] <= x1 + 1 and y0 <= uv[0, 1] <= y1 + 1


# ----------------------------------------------------------------------------- base layout

def test_base_layout_roundtrip_is_reproducible(tmp_path) -> None:
    world = synthetic_world(F=20)
    pkg, mesh_ids = to_base_package(world)
    assert fmt.check_model(pkg) == []
    fmt.write_package(pkg, tmp_path / "a")
    fmt.write_package(pkg, tmp_path / "b")
    for sub in ("meshes", "dynamics"):
        for p in sorted((tmp_path / "a" / sub).glob("*.npz")):
            q = tmp_path / "b" / sub / p.name
            assert hashlib.sha256(p.read_bytes()).digest() == hashlib.sha256(q.read_bytes()).digest()
    assert fmt.check_package(tmp_path / "a") == []
    again = fmt.read_package(tmp_path / "a")
    assert again.frame_count() == 20
    assert set(again.meshes) == set(pkg.meshes)
    car = again.dynamics["car"]
    assert car.pos.shape == (20, 8, 3) and car.ids.tolist() == [mesh_ids["car"]]
    assert fmt.check_package(tmp_path / "a", require_render=True)[0].where == "render.mp4"


def _write_bad(tmp_path, mutate):
    world = synthetic_world(F=6)
    pkg, _ = to_base_package(world)
    mutate(pkg)
    fmt.write_package(pkg, tmp_path, check=False)
    return [str(v) for v in fmt.check_package(tmp_path)]


@pytest.mark.parametrize(
    "mutate, fragment",
    [
        (lambda p: p.dynamics["car"].pos.__setitem__((2, 0, 0), np.nan), "mixes finite and NaN"),
        (lambda p: p.dynamics["car"].pos.__setitem__((slice(2, 3), slice(None)), np.nan), "more than one interval"),
        (lambda p: setattr(p.dynamics["car"], "ids", np.array([99], np.uint16)), "has no meshes/99.npz"),
        (lambda p: p.meshes[1].vertices.__setitem__(3, p.meshes[1].vertices[3] + 1.0), "is static"),
        (lambda p: setattr(p, "extrinsic", np.diag([1.0, 1.0, -1.0, 1.0])), "reflection"),
        (lambda p: setattr(p.dynamics["car"], "pos", p.dynamics["car"].pos[:3]), "has 3 frames, the sequence has 6"),
        (lambda p: p.meshes[2].faces.__setitem__(0, np.array([0, 1, 99], np.int32)), "does not exist"),
    ],
)
def test_check_package_reports_rule_violations(tmp_path, mutate, fragment) -> None:
    problems = _write_bad(tmp_path, mutate)
    assert any(fragment in p for p in problems), problems


def test_write_package_refuses_invalid_input(tmp_path) -> None:
    pkg, _ = to_base_package(synthetic_world(F=4))
    pkg.dynamics["car"].ids = np.array([77], np.uint16)
    with pytest.raises(fmt.FormatError):
        fmt.write_package(pkg, tmp_path)


def test_npz_reader_refuses_pickles_and_foreign_members(tmp_path) -> None:
    np.savez(tmp_path / "ok.npz", a=np.arange(3, dtype=np.int32))
    assert fmt.read_npz(tmp_path / "ok.npz")["a"].tolist() == [0, 1, 2]
    np.save(tmp_path / "obj.npy", np.array([{"x": 1}], dtype=object), allow_pickle=True)
    import zipfile

    with zipfile.ZipFile(tmp_path / "bad.npz", "w") as zf:
        zf.write(tmp_path / "obj.npy", "obj.npy")
    with pytest.raises(fmt.FormatError):
        fmt.read_npz(tmp_path / "bad.npz")
    with zipfile.ZipFile(tmp_path / "txt.npz", "w") as zf:
        zf.writestr("notes.txt", "hello")
    with pytest.raises(fmt.FormatError):
        fmt.read_npz(tmp_path / "txt.npz")


# ----------------------------------------------------------------------------- V2E world io

def test_world_roundtrip_keeps_tracks_sigmas_and_events(tmp_path) -> None:
    world = synthetic_world(F=30)
    car = world.object("car")
    car.position[10:12] = np.nan  # an occlusion gap
    car.quat_xyzw[10:12] = np.nan
    car.visible[10:12] = False
    world.outlines = {"car": {3: [np.array([[1, 1], [20, 1], [20, 9]], float)]}}
    write_world(world, tmp_path)
    assert fmt.check_package(tmp_path) == []
    back = read_world(tmp_path)
    assert back.fps == FPS and back.frame_count == 30 and back.kind == "reconstruction"
    car2 = back.object("car")
    assert np.allclose(car2.position, car.position, equal_nan=True)
    assert np.allclose(car2.quat_xyzw, car.quat_xyzw, equal_nan=True)
    assert car2.visible[10:12].tolist() == [False, False]
    assert car2.sigma_position_m[0] == pytest.approx(0.1)
    assert car2.dims_m["length"].value == pytest.approx(CAR_DIMS[0])
    assert car2.symmetry_class == "none" and back.object("crate").symmetry_class == "box"
    assert [e.name for e in back.events] == ["crate_contact", "car_rest"]
    assert {s.name for s in back.statics} == {"ground", "wall"}
    assert back.camera_sigma_px == pytest.approx(0.5)
    assert back.outlines["car"][3][0].shape == (3, 2)
    assert np.allclose(back.camera_track.camera(0).K, world.camera_track.camera(0).K)
    # the base layout shows the gap filled, so other tools see one contiguous lifetime
    base = fmt.read_package(tmp_path)
    assert np.all(np.isfinite(base.dynamics["car"].pos[10]))


def test_plain_package_import_recovers_rigid_poses(tmp_path) -> None:
    world = synthetic_world(F=25, car_kw={"yaw_rate": 0.4})
    pkg, _ = to_base_package(world)
    fmt.write_package(pkg, tmp_path)
    with pytest.raises(fmt.FormatError):
        read_world(tmp_path)  # no sidecars and no fps
    imported = read_world(tmp_path, fps=FPS)
    assert imported.kind == "imported" and len(imported.statics) == 2
    car = imported.object("car")
    original = world.object("car")
    assert np.allclose(car.centre_track(), original.centre_track(), atol=1e-4)
    extents = sorted(car.hull_extents(), reverse=True)
    assert np.allclose(extents, sorted(CAR_DIMS, reverse=True), atol=1e-4)
    # yaw changes by the same amount even though the imported body axes are chosen by PCA
    d_imp = geo.geodesic_angle(car.rotations()[0], car.rotations()[-1])
    d_org = geo.geodesic_angle(original.rotations()[0], original.rotations()[-1])
    assert d_imp == pytest.approx(d_org, abs=1e-4)
    assert car.annotation["rigid_fit_rms_max_m"] < 1e-4


def test_fill_gaps_interpolates_inside_only() -> None:
    p = np.array([[0, 0, 0], [np.nan] * 3, [2, 0, 0], [np.nan] * 3], float)
    q = np.array([[0, 0, 0, 1], [np.nan] * 4, [0, 0, 0, 1], [np.nan] * 4], float)
    p2, q2 = fill_gaps(p, q)
    assert p2[1, 0] == pytest.approx(1.0) and np.isnan(p2[3, 0])


# ----------------------------------------------------------------------------- timeline and export

def test_resample_and_frame0_trigger() -> None:
    t = np.array([0.0, 1.0, 2.0])
    p = np.array([[0, 0, 0], [1, 0, 0], [4, 0, 0]], float)
    q = Rotation.from_euler("z", [[0], [90], [90]], degrees=True).as_quat()
    p2, q2 = resample_poses(t, p, q, [0.5, 1.5, 3.0])
    assert p2[0, 0] == pytest.approx(0.5) and p2[1, 0] == pytest.approx(2.5) and np.isnan(p2[2, 0])
    assert math.degrees(geo.yaw_of(geo.quat_to_matrix(q2[0]))) == pytest.approx(45.0)
    assert frame0_trigger(t, p, [2.5, 3.0, 0.0], [1, 0, 0]) == pytest.approx(1.5)
    assert frame0_trigger(t, p, [9.0, 0.0, 0.0], [1, 0, 0]) is None
    assert frame0_trigger(t, p, [0.0, 0.0, 0.0], [1, 0, 0]) == 0.0


def test_export_refuses_to_invent_unobserved_attitude() -> None:
    reference = synthetic_world(F=2)
    log = Runlog(RunHeader(("crate",), attitude_observed=False),
                 (Row(0.0, {"crate": ActorState((0.0, 0.0, 2.0))}),))
    with pytest.raises(ValueError, match="attitude was not observed"):
        runlog_to_world(log, reference)


def test_runlog_export_lines_up_with_the_reference() -> None:
    reference = synthetic_world(F=30)
    car = reference.object("car")
    t0 = 2.0
    rows = []
    for k in range(-15, 15):  # half a second of run-up, then the first 15 video frames
        state = ActorState(tuple(car.position[max(k, 0)]), tuple(car.quat_xyzw[max(k, 0)]))
        events = ({"name": "impact", "actor": "car", "other": "wall"},) if k == 9 else ()
        rows.append(Row(t0 + k / FPS, {"car": state}, events=events))
    log = Runlog(RunHeader(("car",), backend="fake", regime="stepped", t0_s=t0), tuple(rows))
    run = runlog_to_world(log, reference)
    assert run.kind == "run_export" and run.frame_count == 30
    exported = run.object("car")
    assert np.allclose(exported.position[:15], car.position[:15], atol=1e-9)
    assert np.all(np.isnan(exported.position[15:]))  # the run ended before the clip did
    assert run.events[0].name == "impact" and run.events[0].kind == "contact"
    assert run.events[0].frame == (9, 9)
    assert run.events[0].actors == ("car", "wall")
    with pytest.raises(ValueError):
        runlog_to_world(Runlog(RunHeader(("ghost",)), (Row(0.0, {"ghost": rows[0].actors["car"]}),)), reference)
