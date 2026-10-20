# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""v2e-synth-1 generation on the fake connector: geometry, driver, truth and written files."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from openbeam_v2e.backends.base import StateRow
from openbeam_v2e.backends.fake import FAKE_CHANNELS
from openbeam_v2e.bench import drive, families, geometry, synth

TINY = synth.RenderSettings(width=96, height=54, noise_sigma=0.0, prompts=2)


def test_ramp_solid_top_surface_matches_collision_planks() -> None:
    case = families.FAMILIES["ramp_jump"].default_case()
    ramp = next(s for s in case.statics if s.id == "ramp")
    solid = geometry.solid_for(ramp)
    run, rise = families.profile_of(families.ramp_segments(case.params["angle_deg"]))
    assert solid.top_z == pytest.approx(rise)
    # The take-off edge is the plane x = 0 (the ramp's high end).
    assert solid.vertices[:, 0].max() == pytest.approx(0.0, abs=1e-9)
    assert len(solid.boxes) == len(families.ramp_segments(case.params["angle_deg"]))
    for part, box in zip(solid.parts, solid.boxes):
        # The plank's top face passes through the segment's top edge midpoint.
        top = part[part[:, 2] > 1e-9] if (part[:, 2] > 1e-9).any() else part
        from scipy.spatial.transform import Rotation

        n = Rotation.from_quat(box.quat_xyzw).as_matrix()[:, 2]
        face_centre = box.center + n * box.half_extents[2]
        assert n[2] > 0.9
        assert np.min(np.abs((top - face_centre) @ n)) < 1e-6


def test_incline_rises_toward_negative_x() -> None:
    case = families.FAMILIES["drop_and_bounce"].build(
        {"variant": "incline", "drop_height_m": None, "incline_deg": 30.0}, "tripod")
    solid = geometry.solid_for(next(s for s in case.statics if s.id == "incline"))
    high = solid.vertices[solid.vertices[:, 2] > solid.top_z - 1e-9]
    assert np.all(high[:, 0] < 0.0)


def _state(p: tuple[float, float, float], v: float, contacts: tuple[str, ...] = ()) -> StateRow:
    return StateRow("car", 0.0, p, (0.0, 0.0, 0.0, 1.0), (v, 0.0, 0.0), (0.0, 0.0, 0.0), contacts)


def test_cruise_controller_stages_and_channels() -> None:
    program = drive.cruise_program("car", 10.0, 0.0, [
        {"mode": "cruise", "until": {"contact": "barrier"}},
        {"mode": "fixed", "cmd": {"drive.throttle": 0.0, "drive.brake": 0.8}}])
    ctl = drive.controller_factory(program, FAKE_CHANNELS)
    slow = ctl.command(0.0, {"car": _state((0.0, 0.0, 0.7), 5.0)})
    assert slow["drive.throttle"] > 0.0 and slow["drive.brake"] == 0.0
    fast = ctl.command(0.01, {"car": _state((0.0, 0.0, 0.7), 14.0)})
    assert fast["drive.throttle"] == 0.0 and fast["drive.brake"] > 0.0
    left = ctl.command(0.02, {"car": _state((0.0, 0.5, 0.7), 10.0)})
    assert left["drive.steer"] > 0.0  # positive steer turns right, back toward the line
    hit = ctl.command(0.03, {"car": _state((0.0, 0.0, 0.7), 9.0, ("barrier",))})
    assert hit["drive.brake"] == 0.8 and hit["drive.throttle"] == 0.0
    assert set(hit) <= {c.id for c in FAKE_CHANNELS}


def test_unknown_models_go_to_the_default_factory() -> None:
    ctl = drive.controller_factory({"schema": "v2e.program/1", "actor": "car", "model": "passive"}, FAKE_CHANNELS)
    assert ctl.command(0.0, {}) == {}


def test_spawn_pose_starts_drive_in_on_a_runup() -> None:
    case = families.FAMILIES["barrier_crash"].default_case()
    car = case.actor("car")
    pose = synth.spawn_pose(car)
    assert pose.position_m[0] == pytest.approx(car.frame0_position_m[0] - synth.runup_length_m(car.speed_mps))
    assert pose.position_m[1:] == pytest.approx(car.frame0_position_m[1:])


@pytest.mark.slow
def test_generate_drop_case_writes_a_valid_truth_package(tmp_path) -> None:
    case = families.FAMILIES["drop_and_bounce"].default_case()
    out = synth.generate_case(case, tmp_path, TINY, physics_dt_s=0.004)
    d = out.directory
    assert len(list((d / "frames").glob("*.png"))) == case.frames
    truth = json.loads((d / "truth.json").read_text())
    h = case.params["drop_height_m"]
    for name in ("box", "ball", "cylinder"):
        t = truth["bench"]["events"][f"first_contact.{name}"]["t_s"]
        # The connector uses standard gravity; contact is within a few frames of free fall.
        assert t == pytest.approx(math.sqrt(2 * h / families.GRAVITY_MPS2), abs=2.0 / case.fps)
    prov = json.loads((d / "provenance.json").read_text())
    assert prov["claim_class"] == "test_double" and prov["valid_run"]
    assert prov["scene_schema_issues"] == []
    from openbeam_v2e.world import read_world

    world = read_world(d / "world")
    assert world.kind == "synthetic_truth" and world.frame_count == case.frames
    assert {o.id for o in world.objects} == {"box", "ball", "cylinder"}
    ann = json.loads((d / "annotations.json").read_text())
    assert ann["schema"] == "v2e.annotations/1"
    assert set(ann["objects"]) == {"box", "ball", "cylinder"}
    assert all(entry["prompts"] for entry in ann["objects"].values())
    from openbeam_v2e.reconstruct.annotate import load_annotations

    assert {p.object_id for p in load_annotations(d / "annotations.json").prompts} == {"box", "ball", "cylinder"}
    config = json.loads((d / "reconstruct.json").read_text())
    assert config["schema"] == "v2e.reconstruct/1"
    assert len(config["camera"]["calibration"]["points"]) >= 6
    assert {o["id"] for o in config["objects"]} == {"box", "ball", "cylinder"}
    assert json.loads((d / "programs.json").read_text()) == {}
    assert synth.load_case_dir(d)["programs"] == {}


@pytest.mark.slow
def test_generation_is_deterministic(tmp_path) -> None:
    case = families.FAMILIES["drop_and_bounce"].default_case()
    a = synth.generate_case(case, tmp_path / "a", TINY, physics_dt_s=0.004)
    b = synth.generate_case(case, tmp_path / "b", TINY, physics_dt_s=0.004)
    pa = json.loads((a.directory / "provenance.json").read_text())
    pb = json.loads((b.directory / "provenance.json").read_text())
    assert pa["frames_sha256"] == pb["frames_sha256"]
    assert pa["truth_sha256"] == pb["truth_sha256"]
