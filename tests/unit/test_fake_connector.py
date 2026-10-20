# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""The fake test connector: plausible rigid-body behaviour, determinism and ledger replay.

The expectations are first-principles bounds (free fall, ordering of restitution, the sign of a
turn), not engine numbers: the fake is a test double and its runs claim nothing about physics.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from connector_fixtures import car_scene, drop_scene, schedule

from openbeam_v2e.backends.base import ActorRef, Pose, RefusedCall, RunSpec
from openbeam_v2e.backends.fake import STANDARD_GRAVITY, FakeBackend, compile_scene
from openbeam_v2e.backends.harness import execute, negotiate
from openbeam_v2e.backends.ledger import read_ledger


def _run(scene, tmp_path: Path, name: str, programs=None, duration=2.0, telemetry_hz=100.0, statics=()):
    backend = FakeBackend()
    plan = negotiate(scene, backend.manifest(), programs=programs or {}, lockstep=True)
    assert plan.ok, plan
    spec = RunSpec(duration_s=duration, host=plan.host, regime=plan.regime, telemetry_hz=telemetry_hz)
    result = execute(backend, compile_scene(scene, statics=statics), plan, spec, tmp_path / name)
    lines = Path(result.rows).read_text().splitlines()
    rows = [json.loads(line) for line in lines[1:]]
    return result, rows


def _yaw(q) -> float:
    x, y, z, w = q
    return math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def test_free_fall_reaches_the_ground_at_the_closed_form_time(tmp_path: Path) -> None:
    height = 2.0
    result, rows = _run(drop_scene(height), tmp_path, "drop", duration=1.2, telemetry_hz=1000.0)
    expected = math.sqrt(2 * height / STANDARD_GRAVITY)
    assert result.events["first_contact.obj.ground"] == pytest.approx(expected, abs=0.003)
    before = [r for r in rows if r["t"] < expected - 0.01]
    z = np.array([r["actors"]["obj"]["p"][2] for r in before])
    t = np.array([r["t"] for r in before])
    # semi-implicit Euler at 1 ms lags the closed form by g * dt * t / 2 (about 3 mm at 0.6 s)
    assert np.allclose(z, 2.25 - 0.5 * STANDARD_GRAVITY * t ** 2, atol=0.5 * STANDARD_GRAVITY * 0.001 * 0.7)
    assert rows[-1]["actors"]["obj"]["p"][2] == pytest.approx(0.25, abs=0.01)
    assert result.claim_class() == "test_double" and result.valid()


def test_a_ball_bounces_higher_than_a_crate(tmp_path: Path) -> None:
    apex = {}
    for shape in ("prop.sphere", "prop.box"):
        result, rows = _run(drop_scene(1.0, shape), tmp_path, shape, duration=1.6)
        contact = result.events["first_contact.obj"]
        after = [r["actors"]["obj"]["p"][2] for r in rows if r["t"] > contact + 0.05]
        rest = 0.15 if shape == "prop.sphere" else 0.25
        apex[shape] = max(after) - rest
    assert apex["prop.sphere"] > apex["prop.box"] >= 0.0
    assert apex["prop.sphere"] < 1.0


def test_vehicle_drives_turns_right_with_positive_steer_and_brakes(tmp_path: Path) -> None:
    program = schedule("car_1", [{"t": 0.0, "cmd": {"drive.throttle": 0.6}},
                                 {"t": 1.5, "cmd": {"drive.throttle": 0.2, "drive.steer": 0.5}},
                                 {"t": 3.0, "cmd": {"drive.throttle": 0.0, "drive.steer": 0.0, "drive.brake": 1.0}}])
    result, rows = _run(car_scene(), tmp_path, "car", programs={"car_1": program}, duration=5.0, telemetry_hz=20.0)
    at = {round(r["t"], 2): r["actors"]["car_1"] for r in rows}
    assert at[1.5]["v"][0] > 2.0 and abs(at[1.5]["p"][1]) < 0.01
    assert _yaw(at[3.0]["q"]) < -10.0           # right turn: negative yaw in the Z-up frame
    assert math.hypot(*at[5.0]["v"]) < 0.05
    assert "rest.car_1" in result.events
    assert rows[0]["cmd"]["car_1"] == {"drive.throttle": 0.6}
    assert rows[-1]["cmd"]["car_1"]["drive.brake"] == 1.0


def test_barrier_stops_the_car(tmp_path: Path) -> None:
    scene = car_scene([{"id": "barrier", "class": "static", "mesh": 1}])
    statics = [{"name": "barrier", "kind": "box", "center": [12.0, 0.0, 0.75], "half_extents": [0.5, 3.0, 0.75]}]
    program = schedule("car_1", [{"t": 0.0, "cmd": {"drive.throttle": 1.0}}])
    result, rows = _run(scene, tmp_path, "crash", programs={"car_1": program}, duration=3.5, statics=statics,
                        telemetry_hz=50.0)
    assert "first_contact.car_1.barrier" in result.events
    assert max(r["actors"]["car_1"]["p"][0] for r in rows) < 12.0 - 0.5


def test_runs_are_bit_identical_and_replay_from_the_ledger(tmp_path: Path) -> None:
    program = {"car_1": schedule("car_1", [{"t": 0.0, "cmd": {"drive.throttle": 0.4, "drive.steer": -0.3}},
                                           {"t": 0.5, "cmd": {"drive.parking_brake": 1}}])}
    first, _ = _run(car_scene(), tmp_path, "a", programs=program, duration=1.0)
    second, _ = _run(car_scene(), tmp_path, "b", programs=program, duration=1.0)
    assert Path(first.rows).read_bytes() == Path(second.rows).read_bytes()
    rows = read_ledger(tmp_path / "a" / "ledger")
    assert any(r["method"] == "fake.actuate" and r["args"]["cmd"] == {"drive.parking_brake": 1.0} for r in rows)
    assert FakeBackend().replay(rows) == []
    tampered = [dict(r) for r in rows]
    for row in tampered:
        if row["method"] == "fake.actuate":
            row["args"] = {**row["args"], "cmd": {"drive.throttle": 0.9}}
            break
    assert FakeBackend().replay(tampered)


def test_the_fake_refuses_what_an_engine_could_not_honestly_do(tmp_path: Path) -> None:
    compiled = compile_scene(drop_scene(1.0))
    backend = FakeBackend()
    backend.install(compiled.packages)
    backend.open_scene(compiled.level)
    ref, record = backend.spawn(compiled.objects[0], compiled.spawns[0].pose, "obj")
    assert record.residual_m == 0.0 and record.residual_deg == pytest.approx(0.0, abs=1e-6)
    with pytest.raises(RefusedCall):
        backend.actuate(ref, {"drive.throttle": 0.5})          # a crate has no actuators
    with pytest.raises(RefusedCall):
        backend.actuate(ref, {"thrust.main": 1.0})              # undeclared channel
    backend.step(1)
    with pytest.raises(RefusedCall):
        backend.spawn(compiled.objects[0], Pose((5.0, 0.0, 1.0)), "late")
    with pytest.raises(RefusedCall):
        backend.install([compiled.packages[0].__class__("x", "vehicle", "")])
    assert ref == ActorRef("obj", "fake:0", compiled.objects[0].name, False)


def test_compile_scene_needs_explicit_poses_and_readable_statics(tmp_path: Path) -> None:
    scene = drop_scene()
    scene["actors"][0]["initial_state"] = {"mode": "at_rest", "pose": "track:obj@0"}
    with pytest.raises(ValueError, match="explicit pose"):
        compile_scene(scene)
    with pytest.raises(ValueError, match="no readable mesh"):
        compile_scene(car_scene([{"id": "wall", "class": "static", "mesh": 3}]), tmp_path)
    world = tmp_path / "world" / "meshes"
    world.mkdir(parents=True)
    corners = np.array([[x, y, z] for x in (10, 11) for y in (-2, 2) for z in (0, 1.5)], dtype=float)
    np.savez(world / "3.npz", vertices=corners, faces=np.zeros((0, 3), dtype=np.int64))
    compiled = compile_scene(car_scene([{"id": "wall", "class": "static", "mesh": 3}]), tmp_path)
    statics = compiled.packages[0].contents["statics"]
    assert statics == [{"name": "wall", "kind": "box", "center": [10.5, 0.0, 0.75], "half_extents": [0.5, 2.0, 0.75]}]
    assert any("assumed mass_kg" in note for note in compiled.notes)


def test_rows_are_valid_v2e_runlog(tmp_path: Path) -> None:
    runlog = pytest.importorskip("openbeam_v2e.schema.runlog")
    result, _ = _run(drop_scene(0.5), tmp_path, "rows", duration=0.6)
    log = runlog.read_runlog(Path(result.rows))
    assert log.header.backend == "fake" and log.header.actors == ("obj",)
    assert any(event["name"] == "first_contact.obj" for _, event in log.events())
