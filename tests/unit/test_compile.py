# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""Tests for the compiler: content choice, spawn at rest, run-up synthesis, timeline, refusals."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from openbeam_v2e.compile import CompileRefused, compile_scene
from openbeam_v2e.compile.content import DENSITY_CLASSES, GENERIC_VEHICLE_DESIGN_POINT
from openbeam_v2e.compile.level import obj_text
from openbeam_v2e.compile.runup import fit_entry_shape
from openbeam_v2e.schema import validate_document
from openbeam_v2e.schema.catalog import ContentCatalog
from openbeam_v2e.schema.scene import Scene
from openbeam_v2e.schema.templates import Templates
from openbeam_v2e.world import ObjectTrack, StaticMesh
from openbeam_v2e.world.geometry import box_mesh

from world_fixtures import CRATE, FPS, synthetic_world


def catalog(**kw) -> ContentCatalog:
    doc = {"schema": "v2e.catalog/1", "backend": "fake", "engine": "fake 0", "entries": [
        {"id": "compact", "kind": "vehicle", "seatable": True, "mass_kg": 1200,
         "dims_m": {"length": 4.0, "width": 1.75, "height": 1.45, "wheelbase": 2.5}},
        {"id": "estate", "kind": "vehicle", "seatable": True,
         "dims_m": {"length": 4.5, "width": 1.9, "height": 1.5, "wheelbase": 3.0}},
    ]}
    doc.update(kw)
    return ContentCatalog.from_dict(doc)


def scene(car_mode: str = "drive_in", crate_mode: str = "at_rest", extra: list | None = None, **kw) -> Scene:
    car_state = {"mode": "drive_in", "frame0_pose": "track:car@0", "speed_mps": [8.0, 0.2]} if car_mode == "drive_in" \
        else {"mode": "at_rest", "pose": "track:car@0"}
    crate_state = {"mode": crate_mode, ("pose" if crate_mode == "at_rest" else "frame0_pose"): "track:crate@0"}
    doc = {
        "schema": "v2e.scene/1", "world": "world/",
        "actors": [
            {"id": "car", "class": "vehicle.car", "seatable": True, "control": "driver", "initial_state": car_state},
            {"id": "crate", "class": "prop.box", "control": "passive", "initial_state": crate_state},
            {"id": "ground", "class": "static", "mesh": 1, "role": "drivable"},
            {"id": "wall", "class": "static", "mesh": 2},
            *(extra or []),
        ],
        "requires": {"effects": ["drive", "contact"], "controlled_actors": 1},
    }
    doc.update(kw)
    return Scene.from_dict(doc)


def resting_world(**kw):
    return synthetic_world(crate_kw={"drop": 0.0}, **kw)


def test_drive_in_scene_compiles_to_a_valid_request(tmp_path) -> None:
    result = compile_scene(scene(), Templates(), catalog(), resting_world())
    req = result.request
    assert validate_document(req) == []
    car_spawn = next(s for s in req["spawns"] if s["actor"] == "car")
    assert car_spawn["at_rest"] is True and car_spawn["initial_velocity_mps"] == 0
    assert car_spawn["mode"] == "drive_in"
    runup = result.runups["car"]
    assert runup.shape == "straight"
    assert runup.length_m == pytest.approx(8.0 ** 2 / 4.0 + 10.0)
    assert car_spawn["pose"]["position_m"][0] == pytest.approx(-12.0 - runup.length_m, abs=0.3)
    assert car_spawn["frame0_pose"]["position_m"][0] == pytest.approx(-12.0)
    t0 = 8.0 / 2.0 + 10.0 / 8.0
    assert req["timeline"]["t0_s"] == pytest.approx(t0)
    ctrl = next(c for c in req["controllers"] if c["actor"] == "car")
    assert ctrl["model"] == "time_indexed_tracker"
    ref = ctrl["reference"]
    assert ref["frame0_s_m"] == pytest.approx(runup.length_m)
    i0 = int(np.argmin(np.abs(np.array(ref["s_m"]) - ref["frame0_s_m"])))
    assert ref["t_s"][i0] == pytest.approx(t0, abs=1e-6) and ref["speed_mps"][i0] == pytest.approx(8.0)
    assert ref["t_s"] == sorted(ref["t_s"]) and ref["speed_mps"][0] == 0.0
    assert next(c for c in req["controllers"] if c["actor"] == "crate")["model"] == "passive"
    # synthesised ground beyond the observed ground plane, never marked observed
    ids = {m["id"]: m for m in req["level"]["collision"]}
    assert set(ids) == {"ground", "wall", "runup_car"}
    assert ids["runup_car"]["observed"] is False and ids["runup_car"]["source"] == "runup"
    assert ids["ground"]["observed"] is True
    # content records
    car = next(c for c in req["content"] if c["actor"] == "car")
    assert car["kind"] == "generated" and car["template"] == "vehicle.generic"
    assert car["parameters"]["length_m"]["status"] == "measured"
    assert car["parameters"]["length_m"]["value"] == pytest.approx(4.4)
    assert car["parameters"]["mass_kg"]["status"] == "assumed"
    crate = next(c for c in req["content"] if c["actor"] == "crate")
    density, _ = DENSITY_CLASSES["wood"]
    assert crate["parameters"]["mass_kg"]["value"] == pytest.approx(density * CRATE ** 3, rel=1e-3)
    # files
    path = result.write(tmp_path)
    assert json.loads(path.read_text())["schema"] == "v2e.package/1"
    assert (tmp_path / "level" / "runup_car.obj").read_text().startswith("# V2E level collision")


def test_request_is_deterministic() -> None:
    a = compile_scene(scene(), Templates(), catalog(), resting_world()).request
    b = compile_scene(scene(), Templates(), catalog(), resting_world()).request
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_blocked_corridor_is_refused() -> None:
    world = resting_world()
    v, f = box_mesh(1.0, 1.0, 1.0, centre=(-25.0, 0.5, 0.5))
    world.statics.append(StaticMesh(7, v, f, "post"))
    with pytest.raises(CompileRefused) as excinfo:
        compile_scene(scene(), Templates(), catalog(), world)
    assert [r.code for r in excinfo.value.refusals] == ["corridor_blocked"]
    # an object parked in the approach blocks it too
    world = resting_world()
    world.object("crate").position[:, :2] = [-20.0, 0.0]
    with pytest.raises(CompileRefused) as excinfo:
        compile_scene(scene(), Templates(), catalog(), world)
    assert "object 'crate'" in excinfo.value.refusals[0].message


def test_released_objects_set_t0_to_spawn_and_conflict_with_drive_in() -> None:
    world = synthetic_world()  # the crate hangs 2 m above the ground at frame 0
    result = compile_scene(scene(car_mode="at_rest"), Templates(), catalog(), world)
    assert result.request["timeline"]["t0_s"] == 0.0
    assert any("released at frame 0" in n for n in result.request["notes"])
    with pytest.raises(CompileRefused) as excinfo:
        compile_scene(scene(), Templates(), catalog(), world)
    assert "released_with_drive_in" in [r.code for r in excinfo.value.refusals]


def test_at_rest_scene_settles_before_frame0_and_lifts_sunk_poses() -> None:
    world = resting_world()
    world.object("car").position[0, 2] -= 0.1  # the reconstruction sank the car into the ground
    result = compile_scene(scene(car_mode="at_rest"), Templates(), catalog(), world)
    car = next(s for s in result.request["spawns"] if s["actor"] == "car")
    assert car["ground_offset_m"] == pytest.approx(0.1, abs=1e-6)
    assert car["pose"]["position_m"][2] == pytest.approx(0.0, abs=1e-6)
    assert result.request["timeline"]["t0_s"] == pytest.approx(0.5)
    ctrl = next(c for c in result.request["controllers"] if c["actor"] == "car")
    assert ctrl["reference"]["t_s"][0] == pytest.approx(0.5)


def test_stock_matching_records_residuals_and_refuses_out_of_tolerance() -> None:
    sc = scene(car_mode="at_rest")
    t = Templates.from_dict({"schema": "v2e.templates/1", "objects": {"car": {"template": "vehicle.stock"}}})
    result = compile_scene(sc, t, catalog(), resting_world())
    car = next(c for c in result.request["content"] if c["actor"] == "car")
    assert car["kind"] == "stock" and car["content_id"] == "estate"  # 4.5 m is nearest to 4.4 m
    assert car["residuals"]["length"]["relative"] == pytest.approx((4.5 - 4.4) / 4.4)
    assert car["parameters"]["length_m"]["status"] == "catalog"
    tight = Templates.from_dict({"schema": "v2e.templates/1", "objects": {
        "car": {"template": "vehicle.stock", "tolerance": {"length": 0.01}}}})
    with pytest.raises(CompileRefused) as excinfo:
        compile_scene(sc, tight, catalog(), resting_world())
    assert excinfo.value.refusals[0].code == "no_stock_match"
    fallback = Templates.from_dict({"schema": "v2e.templates/1", "objects": {
        "car": {"template": "vehicle.stock", "tolerance": {"length": 0.01}, "fallback": "vehicle.generic"}}})
    result = compile_scene(sc, fallback, catalog(), resting_world())
    car = next(c for c in result.request["content"] if c["actor"] == "car")
    assert car["template"] == "vehicle.generic" and car["fallback_from"] == "vehicle.stock"
    named = Templates.from_dict({"schema": "v2e.templates/1", "objects": {
        "car": {"template": "vehicle.stock", "rule": "content_id", "content_id": "compact"}}})
    car = next(c for c in compile_scene(sc, named, catalog(), resting_world()).request["content"] if c["actor"] == "car")
    assert car["content_id"] == "compact" and car["parameters"]["mass_kg"]["value"] == 1200
    with pytest.raises(CompileRefused):
        compile_scene(sc, t, catalog(entries=[]), resting_world())


def test_limits_templates_and_launchers_are_enforced() -> None:
    second = {"id": "car2", "class": "vehicle.car", "seatable": True, "control": "driver",
              "initial_state": {"mode": "at_rest", "pose": {"position_m": [0, -8, 0]}}}
    with pytest.raises(CompileRefused) as excinfo:
        compile_scene(scene(car_mode="at_rest", extra=[second], requires={"controlled_actors": 2}), Templates(),
                      catalog(limits={"max_seatable_vehicles": 1, "max_controlled_actors": 1}), resting_world())
    assert {r.code for r in excinfo.value.refusals} == {"seatable_limit", "controlled_limit"}
    with pytest.raises(CompileRefused) as excinfo:
        compile_scene(scene(car_mode="at_rest"), Templates(), catalog(generated_templates=["vehicle.generic"]),
                      resting_world())
    assert excinfo.value.refusals[0].code == "template_unavailable"
    with pytest.raises(CompileRefused) as excinfo:
        compile_scene(scene(car_mode="at_rest", crate_mode="moving"), Templates(), catalog(), resting_world())
    assert excinfo.value.refusals[0].code == "moving_without_launcher"
    sc = scene(car_mode="at_rest", crate_mode="moving")
    doc = sc.to_dict()
    doc["actors"][1]["initial_state"]["launcher"] = {"mechanism": "plugin.spring_plate"}
    result = compile_scene(Scene.from_dict(doc), Templates(), catalog(), resting_world())
    crate = next(s for s in result.request["spawns"] if s["actor"] == "crate")
    assert crate["mode"] == "moving" and crate["launcher"]["mechanism"] == "plugin.spring_plate"
    assert crate["initial_velocity_mps"] == 0


def test_missing_meshes_and_objects_are_refused() -> None:
    bad = scene(car_mode="at_rest", extra=[{"id": "ramp", "class": "static", "mesh": 9}])
    with pytest.raises(CompileRefused) as excinfo:
        compile_scene(bad, Templates(), catalog(), resting_world())
    assert excinfo.value.refusals[0].code == "missing_static_mesh"
    doc = scene(car_mode="at_rest").to_dict()
    doc["actors"][0]["initial_state"]["pose"] = "track:ghost@0"
    with pytest.raises(CompileRefused) as excinfo:
        compile_scene(Scene.from_dict(doc), Templates(), catalog(), resting_world())
    assert excinfo.value.refusals[0].code == "missing_world_object"


def _curved_car(F: int, radius: float = 40.0, speed: float = 8.0) -> ObjectTrack:
    t = np.arange(F) / FPS
    theta = -math.pi / 2 + speed * t / radius  # counter-clockwise around (0, radius)
    pos = np.column_stack([radius * np.cos(theta), radius + radius * np.sin(theta), np.zeros(F)])
    yaw = theta + math.pi / 2
    quat = np.column_stack([np.zeros(F), np.zeros(F), np.sin(yaw / 2), np.cos(yaw / 2)])
    hv, hf = box_mesh(4.4, 1.8, 1.4, centre=(0, 0, 0.7))
    return ObjectTrack("car", "vehicle.car", hv, hf, pos, quat)


def test_curved_entry_gets_an_arc_run_up() -> None:
    world = resting_world()
    world.objects[0] = _curved_car(world.frame_count)
    world.object("crate").position[:, :2] = [10.0, 30.0]
    result = compile_scene(scene(), Templates(), catalog(), world)
    runup = result.runups["car"]
    assert runup.shape == "arc"
    # every run-up point lies on the fitted circle of radius 40 m
    r = np.linalg.norm(runup.path_m[:, :2] - [0.0, 40.0], axis=1)
    assert np.allclose(r, 40.0, atol=0.05)
    assert fit_entry_shape(np.array([[0, 0], [1, 0], [2, 0]], float), np.array([1.0, 0.0]))["shape"] == "straight"


def test_several_drive_in_actors_reach_frame0_together() -> None:
    world = resting_world()
    fast = _curved_car(world.frame_count)
    fast.id = "car2"
    fast.position[:, 1] += 6.0  # a parallel lane, far from the first car
    fast.position[:, 0] -= 60.0
    world.objects.append(fast)
    extra = [{"id": "car2", "class": "vehicle.car", "seatable": True, "control": "driver",
              "initial_state": {"mode": "drive_in", "frame0_pose": "track:car2@0", "speed_mps": 12.0, "path": "straight"}}]
    result = compile_scene(scene(extra=extra, requires={"controlled_actors": 2}), Templates(), catalog(), world)
    t0 = result.request["timeline"]["t0_s"]
    natural = {k: r.natural_t0_s for k, r in result.runups.items()}
    assert t0 == pytest.approx(max(natural.values()))
    refs = {c["actor"]: c["reference"] for c in result.request["controllers"] if "reference" in c}
    slow_first = min(natural, key=natural.get)
    assert refs[slow_first]["t_s"][0] == pytest.approx(t0 - natural[slow_first])  # waits at rest
    for actor, ref in refs.items():
        i0 = int(np.argmin(np.abs(np.array(ref["s_m"]) - ref["frame0_s_m"])))
        assert ref["t_s"][i0] == pytest.approx(t0, abs=1e-6)


def test_generic_vehicle_scales_unmeasured_values_from_the_design_point() -> None:
    sc = scene(car_mode="at_rest")
    doc = sc.to_dict()
    doc["actors"][0]["dims_m"] = {"length": [4.75 * 1.1, 0.05]}
    world = resting_world()
    world.object("car").dims_m = {}
    result = compile_scene(Scene.from_dict(doc), Templates(), catalog(), world)
    params = next(c for c in result.request["content"] if c["actor"] == "car")["parameters"]
    assert params["length_m"]["source"] == "scene"
    assert params["width_m"]["source"] == "reconstruction hull extents"
    assert params["wheelbase_m"]["status"] == "assumed"
    assert params["wheelbase_m"]["value"] == pytest.approx(GENERIC_VEHICLE_DESIGN_POINT["wheelbase_m"] * 1.1, rel=1e-3)


def test_obj_text_is_one_based() -> None:
    text = obj_text([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, 2]], "x")
    assert text.splitlines()[-1] == "f 1 2 3" and text.startswith("# x\nv 0.000000")


def test_compile_scene_accepts_paths_and_writes_outputs(tmp_path) -> None:
    from openbeam_v2e.world import write_world

    write_world(resting_world(), tmp_path / "world")
    (tmp_path / "scene.json").write_text(json.dumps(scene(car_mode="at_rest").to_dict()))
    result = compile_scene(tmp_path / "scene.json", connector="openbeam", out=tmp_path / "out")
    assert result.request["inputs"]["backend"] == "openbeam"
    assert all(c["kind"] == "generated" for c in result.request["content"])
    assert (tmp_path / "out" / "package.json").is_file()
    assert (tmp_path / "out" / "level" / "ground.obj").is_file()


def test_dialect_lookup_and_dispatch(tmp_path) -> None:
    from openbeam_v2e.compile import dialects

    with pytest.raises(dialects.DialectUnavailable):
        dialects.find_dialect("no-such-engine")
    calls = {}

    def fake_dialect(request, world=None):
        calls["args"] = (request, world)
        return {"schema": "v2e.compiled/1"}

    dialects.BUILTIN_DIALECTS["unit-test"] = f"{__name__}:_UNUSED"
    try:
        import sys

        setattr(sys.modules[__name__], "_UNUSED", fake_dialect)
        out = dialects.build_compiled({"schema": "v2e.package/1"}, "unit-test", out=tmp_path, world="w")
    finally:
        dialects.BUILTIN_DIALECTS.pop("unit-test")
    assert out == {"schema": "v2e.compiled/1"}
    assert calls["args"] == ({"schema": "v2e.package/1"}, "w")  # 'out' is not offered to a dialect without it
