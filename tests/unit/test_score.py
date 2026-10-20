# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""Tests for the generic scorer on closed-form synthetic worlds."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from openbeam_v2e.schema import validate_document
from openbeam_v2e.schema.scene import Scene
from openbeam_v2e.score import (
    ScoreOptions,
    claim_class,
    provenance_signature,
    repeat_report,
    robust_summary,
    score_run,
    score_worlds,
)
from openbeam_v2e.score.attitude import attitude_error, cylinder_axis, symmetry_group
from openbeam_v2e.score.claims import claim_block
from openbeam_v2e.score.events import contact_mask, detect_events
from openbeam_v2e.score.post_event import after_event, rollover
from openbeam_v2e.score.repeat import within_robust_interval
from openbeam_v2e.world import ObjectTrack, write_world
from openbeam_v2e.world.geometry import box_mesh
from openbeam_v2e.world.io import to_base_package

from world_fixtures import FPS, G, crate_track, synthetic_world


def scene_doc(**extra) -> dict:
    doc = {
        "schema": "v2e.scene/1", "world": "world/",
        "actors": [
            {"id": "car", "class": "vehicle.car", "seatable": True, "control": "driver",
             "initial_state": {"mode": "at_rest", "pose": "track:car@0"}},
            {"id": "crate", "class": "prop.box", "control": "passive",
             "initial_state": {"mode": "at_rest", "pose": "track:crate@0"}},
            {"id": "ground", "class": "static", "mesh": 1, "role": "drivable"},
            {"id": "wall", "class": "static", "mesh": 2},
        ],
        "events": [{"name": "crate_lands", "frame": [19, 20], "kind": "contact", "actors": ["crate", "ground"]},
                   {"name": "car_stops", "frame": [45, 46], "kind": "rest", "actors": ["car"]}],
        "phases": [{"name": "fall", "from": "start", "to": "crate_lands"},
                   {"name": "after", "from": "crate_lands", "to": "end"}],
        "targets": [
            {"metric": "first_contact_time", "actor": "crate", "value": [math.sqrt(2 * 2.0 / G), 1 / FPS],
             "source": "first_principles"},
            {"metric": "first_rebound_ratio", "actor": "crate", "value": [0.25, 0.02], "source": "design_value"},
            {"metric": "speed_at_event", "actor": "car", "event": "car_stops", "value": [0.0, 0.2],
             "source": "design_value"},
            {"metric": "time_of_event", "event": "car_stops", "value": [1.5, 0.05], "source": "design_value"},
            {"metric": "no_such_metric", "value": 1, "source": "design_value"},
        ],
    }
    doc.update(extra)
    return doc


# ----------------------------------------------------------------------------- attitude and statistics

def test_attitude_error_respects_symmetry() -> None:
    R0 = np.eye(3)[None]
    quarter = Rotation.from_euler("z", 90, degrees=True).as_matrix()[None]
    half_x = Rotation.from_euler("x", 180, degrees=True).as_matrix()[None]
    tilt = Rotation.from_euler("x", 30, degrees=True).as_matrix()[None]
    assert math.degrees(attitude_error(R0, quarter, "none")[0]) == pytest.approx(90.0)
    assert attitude_error(R0, quarter, "box", (1, 1, 1))[0] == pytest.approx(0.0, abs=1e-7)
    assert math.degrees(attitude_error(R0, quarter, "box", (2, 1, 0.5))[0]) == pytest.approx(90.0)
    assert attitude_error(R0, half_x, "box", (2, 1, 0.5))[0] == pytest.approx(0.0, abs=1e-7)
    assert len(symmetry_group("box", (1, 1, 1))) == 24
    assert len(symmetry_group("box", (1, 1, 2))) == 8
    assert len(symmetry_group("box", (3, 2, 1))) == 4
    # cylinder along z (diameter 1, length 2): spin is free, the axis sign only for symmetric ends
    assert cylinder_axis((1.0, 1.0, 2.0)).tolist() == [0, 0, 1]
    assert attitude_error(R0, quarter, "cylinder", (1, 1, 2))[0] == pytest.approx(0.0, abs=1e-7)
    assert math.degrees(attitude_error(R0, half_x, "cylinder", (1, 1, 2))[0]) == pytest.approx(180.0)
    assert attitude_error(R0, half_x, "cylinder_symmetric", (1, 1, 2))[0] == pytest.approx(0.0, abs=1e-7)
    assert math.degrees(attitude_error(R0, tilt, "cylinder_symmetric", (1, 1, 2))[0]) == pytest.approx(30.0)
    assert np.isnan(attitude_error(R0, tilt, "sphere")[0])


def test_robust_statistics_and_interval() -> None:
    s = robust_summary([1.0, 2.0, 3.0, 4.0, 100.0, None, float("nan")])
    assert s["n"] == 5 and s["median"] == 3.0
    assert s["mad"] == pytest.approx(1.4826 * 1.0)
    assert s["low"] == pytest.approx(3.0 - 3 * 1.4826)
    assert robust_summary([])["median"] is None
    values = [7 + d for d in (0, 1, -1, 0, 0.5)]
    assert within_robust_interval(9, values)
    assert not within_robust_interval(13, values)
    assert within_robust_interval(13, values, floor=6)


def test_claim_classes_and_signatures() -> None:
    native = [{"name": "drive", "source": "native"}, {"name": "contact", "source": "native"}]
    assert claim_class(native, ["drive", "contact"]) == "native"
    assert claim_class(native, ["drive", "contact", "brake"]) == "incomplete"
    assert claim_class(native + [{"name": "launch", "source": "native_substitute"}]) == "emulated"
    assert claim_class(native + [{"name": "lift", "source": "scripted"}]) == "scripted"
    assert claim_class([{"name": "drive", "source": "test_double"}]) == "test_double"
    assert claim_class([]) == "unknown"
    assert provenance_signature(native) == provenance_signature(list(reversed(native)))
    assert provenance_signature(native) != provenance_signature(native, [{"actor": "car", "kind": "stock"}])
    assert claim_block(None)["class"] == "unknown"
    assert claim_block({"effects": native}, ["drive"])["class"] == "native"


def test_contact_mask_catches_impacts_between_frames() -> None:
    gap = np.array([0.5, 0.25, 0.034, 0.083, 0.17, 0.3])
    assert contact_mask(gap, 0.03, 0.25).tolist() == [False, False, True, False, False, False]
    passing = np.array([1.0, 0.6, 0.4, 0.6, 1.0])  # a near miss far outside the cap
    assert not contact_mask(passing, 0.03, 0.25).any()


# ----------------------------------------------------------------------------- whole reports

def test_identical_worlds_score_perfectly() -> None:
    ref = synthetic_world()
    run = synthetic_world(kind="run_export")
    report = score_worlds(ref, run, Scene.from_dict(scene_doc()),
                          {"effects": [{"name": "drive", "source": "test_double"}]})
    assert validate_document(report) == []
    assert report["claim"]["class"] == "test_double"
    assert report["summary"]["position_rms_m_worst"] == 0.0
    assert report["summary"]["silhouette_iou_mean"] == pytest.approx(1.0)
    assert [p["name"] for p in report["phases"]] == ["fall", "after"]
    assert report["phases"][0]["frames"] == [0, 20] or report["phases"][0]["frames"] == [0, 19]
    lands = report["events"][0]
    assert lands["run_frame"] == 19 and lands["within_sigma"] is True
    targets = {t["metric"]: t for t in report["targets"]}
    assert targets["first_contact_time"]["pass"] is True
    assert targets["first_rebound_ratio"]["run"] == pytest.approx(0.25, abs=0.01)
    assert targets["first_rebound_ratio"]["pass"] is True
    assert targets["speed_at_event"]["run"] == pytest.approx(0.0, abs=0.2)
    assert targets["time_of_event"]["run"] == pytest.approx(45.5 / FPS, abs=1.0 / FPS)
    assert targets["time_of_event"]["pass"] is True
    assert targets["no_such_metric"]["status"] == "unsupported"
    assert report["summary"]["objective"] < 1.0  # only the targets' own sigmas contribute
    bounce = report["objects"]["crate"]["bounce"]
    assert bounce["apex_heights_m"]["reference"][0] == pytest.approx(crate_track(75)[2]["apex2"], abs=0.01)


def test_offsets_and_rotations_show_up_with_sigmas() -> None:
    ref = synthetic_world()
    run = synthetic_world(kind="run_export")
    run.object("car").position[:, 1] += 0.5
    q = Rotation.from_euler("z", 90, degrees=True).as_quat()
    run.object("crate").quat_xyzw[:] = q        # a cube turned a quarter turn looks the same
    report = score_worlds(ref, run)
    car = report["phases"][0]["objects"]["car"]
    assert car["position_rms_m"] == pytest.approx(0.5)
    assert car["position_rms_z"] == pytest.approx(5.0)
    assert car["silhouette_iou_mean"] < 0.95
    assert car["silhouette"]["shape_limit_mean"] == pytest.approx(1.0)
    crate = report["phases"][0]["objects"]["crate"]
    assert crate["attitude_rms_deg"] == pytest.approx(0.0, abs=1e-5)
    assert report["summary"]["objective"] > 1.0


def test_late_and_missing_objects_are_penalised() -> None:
    ref = synthetic_world()
    late = synthetic_world(kind="run_export", crate_kw={"drop": 2.6})  # falls further, lands later
    report = score_worlds(ref, late, Scene.from_dict(scene_doc()))
    lands = report["events"][0]
    assert lands["delta_frames"] > 0 and lands["within_sigma"] is False
    missing = synthetic_world(kind="run_export")
    missing.objects = [o for o in missing.objects if o.id != "crate"]
    report2 = score_worlds(ref, missing, Scene.from_dict(scene_doc()))
    assert report2["objects"]["crate"]["in_run"] is False
    assert report2["phases"][0]["objects"]["crate"]["coverage"] == 0.0
    assert report2["summary"]["objective"] > report["summary"]["objective"]


def test_timeline_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError):
        score_worlds(synthetic_world(F=75), synthetic_world(F=60))


def _barrier_crash(F: int, rebound: float) -> ObjectTrack:
    """8 m/s into a barrier at frame 30, then back at ``rebound`` m/s decaying to rest."""
    t = np.arange(F) / FPS
    x = np.empty(F)
    t_hit = 30 / FPS
    for i, ti in enumerate(t):
        if ti <= t_hit:
            x[i] = 8.0 * ti
        else:
            dt = min(ti - t_hit, rebound / 4.0)
            x[i] = 8.0 * t_hit - rebound * dt + 0.5 * 4.0 * dt * dt
    pos = np.column_stack([x, np.zeros(F), np.zeros(F)])
    yaw = np.where(t > t_hit, np.minimum(t - t_hit, 0.5) * math.radians(20.0), 0.0)
    quat = Rotation.from_euler("z", yaw[:, None]).as_quat()
    hv, hf = box_mesh(4.4, 1.8, 1.4, centre=(0, 0, 0.7))
    return ObjectTrack("car", "vehicle.car", hv, hf, pos, quat)


def test_after_event_measures_rebound_displacement_and_yaw() -> None:
    obj = _barrier_crash(90, rebound=2.0)
    out = after_event(obj, FPS, 30, rest=None)
    assert out["speed_before_mps"] == pytest.approx(8.0, abs=0.05)
    assert out["rebound_speed_mps"] == pytest.approx(2.0, abs=0.2)
    assert out["yaw_change_deg"] == pytest.approx(10.0, abs=0.5)
    assert out["displacement_m"] == pytest.approx(0.5, abs=0.1)


def test_rollover_counts_quarter_turns() -> None:
    F = 60
    t = np.arange(F) / FPS
    roll = np.minimum(t, 1.0) * math.pi          # half a turn in one second, then rest on the roof
    pos = np.column_stack([6.0 * np.minimum(t, 1.0), np.zeros(F), np.full(F, 1.0)])
    quat = Rotation.from_euler("x", roll[:, None]).as_quat()
    hv, hf = box_mesh(4.4, 1.8, 1.4)
    out = rollover(ObjectTrack("car", "vehicle.car", hv, hf, pos, quat), FPS)
    assert out["cumulative_roll_deg"] == pytest.approx(180.0, abs=7.0)
    assert out["quarter_turns"] == 2 and out["rest_attitude"] == "roof"
    assert out["roll_onset_s"] == pytest.approx(15.0 / 180.0, abs=2.0 / FPS)


def test_detected_events_on_the_synthetic_world() -> None:
    events = detect_events(synthetic_world())
    kinds = {(e.kind, e.actors[0]) for e in events}
    assert ("contact", "crate") in kinds and ("separation", "crate") in kinds
    assert ("rest", "car") in kinds and ("rest", "crate") in kinds
    assert not any(e.kind == "contact" and e.actors[0] == "car" for e in events)  # resting on its support


def test_repeat_report_over_noisy_runs() -> None:
    ref = synthetic_world()
    reports = []
    rng = np.random.default_rng(0)
    for i in range(9):
        run = synthetic_world(kind="run_export")
        run.object("car").position[:, 0] += rng.normal(0.0, 0.02)
        reports.append(score_worlds(ref, run, options=ScoreOptions(silhouettes=False)))
    out = repeat_report(reports + [None])
    assert validate_document(out) == []
    assert out["runs"] == 10 and out["failures"] == 1 and out["failure_fraction"] == pytest.approx(0.1)
    m = out["metrics"]["phases.all.car.position_rms_m"]
    assert m["n"] == 9 and 0.0 < m["median"] < 0.06


def test_score_run_from_directories_and_plain_packages(tmp_path) -> None:
    ref, run = synthetic_world(), synthetic_world(kind="run_export")
    run.object("car").position[:, 2] += 0.1
    write_world(ref, tmp_path / "world")
    write_world(run, tmp_path / "run")
    (tmp_path / "scene.json").write_text(json.dumps(scene_doc()))
    report = score_run(tmp_path / "run", scene=tmp_path / "scene.json", out=tmp_path / "out")
    assert json.loads((tmp_path / "out" / "score.json").read_text())["schema"] == "v2e.score/1"
    assert report["inputs"]["reference_kind"] == "reconstruction"
    assert report["phases"][1]["objects"]["car"]["position_rms_m"] == pytest.approx(0.1, abs=1e-6)
    # a plain package made by another tool can be the run: objects are matched by archive name
    import openbeam_v2e.fourdcb.format as fmt

    pkg, _ = to_base_package(run)
    fmt.write_package(pkg, tmp_path / "plain")
    from openbeam_v2e.world import read_world

    plain = read_world(tmp_path / "plain", fps=FPS)
    report = score_worlds(ref, plain)
    car = report["phases"][0]["objects"]["car"]
    assert car["position_rms_m"] == pytest.approx(0.1, abs=1e-4)
    assert report["inputs"] == {}
