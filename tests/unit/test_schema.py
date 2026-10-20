# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""Tests for the V2E document schemas, the validator and the dataclasses."""

from __future__ import annotations

import copy
import json

import pytest

from openbeam_v2e import schema
from openbeam_v2e.schema import catalog, jsonio, runlog, scene, templates
from openbeam_v2e.schema.common import Measured, Pose, default_symmetry, parse_pose_ref
from openbeam_v2e.schema.validate import SchemaError, check, validate


def scene_doc() -> dict:
    return {
        "schema": "v2e.scene/1",
        "world": "world/",
        "actors": [
            {"id": "car_1", "class": "vehicle.car", "dims_m": {"length": [4.4, 0.05], "width": 1.8},
             "seatable": True, "control": "driver",
             "initial_state": {"mode": "drive_in", "frame0_pose": "track:car_1@0", "speed_mps": [12.0, 0.4]}},
            {"id": "crate", "class": "prop.box", "control": "passive",
             "initial_state": {"mode": "at_rest", "pose": {"position_m": [10, 0, 0.25]}}},
            {"id": "wall", "class": "static", "mesh": 2, "role": "collision"},
        ],
        "events": [{"name": "impact", "frame": [40, 41], "actors": ["car_1", "wall"]}],
        "phases": [{"name": "approach", "from": "start", "to": "impact"},
                   {"name": "after", "from": "impact", "to": "rest"}],
        "targets": [{"metric": "speed_at_event", "actor": "car_1", "event": "impact", "value": [12.0, 0.4],
                     "source": "design_value"}],
        "requires": {"effects": ["drive", "contact"], "observables": ["position", "attitude"],
                     "controlled_actors": 1, "seatable_vehicles": 1},
    }


# ----------------------------------------------------------------------------- validator

def test_validator_types_and_bool_is_not_a_number() -> None:
    assert validate(1, {"type": "integer"}) == []
    assert validate(1.0, {"type": "integer"}) == []
    assert validate(True, {"type": "number"})
    assert validate(True, {"type": "integer"})
    assert validate("x", {"type": ["number", "null"]})
    assert validate(None, {"type": ["number", "null"]}) == []


def test_validator_object_keywords() -> None:
    s = {"type": "object", "required": ["a"], "properties": {"a": {"type": "string"}},
         "patternProperties": {"^x_": {"type": "number"}}, "additionalProperties": False}
    assert validate({"a": "ok", "x_1": 2}, s) == []
    issues = validate({"b": 1, "x_1": "no"}, s)
    messages = " ".join(i.message for i in issues)
    assert "missing required member 'a'" in messages
    assert "is not an allowed member" in messages
    assert any(i.path == "/x_1" for i in issues)


def test_validator_arrays_refs_and_combinators() -> None:
    s = {
        "$defs": {"pair": {"type": "array", "prefixItems": [{"type": "number"}, {"type": "string"}],
                           "minItems": 2, "maxItems": 2}},
        "type": "array", "items": {"$ref": "#/$defs/pair"}, "uniqueItems": True,
    }
    assert validate([[1, "a"], [2, "b"]], s) == []
    assert validate([[1, "a"], [1, "a"]], s)
    assert validate([[1, 2]], s)
    assert validate(3, {"oneOf": [{"type": "integer"}, {"type": "number"}]})  # matches both
    assert validate("a", {"anyOf": [{"type": "integer"}, {"const": "a"}]}) == []
    assert validate(5, {"not": {"minimum": 3}})
    cond = {"if": {"properties": {"k": {"const": 1}}}, "then": {"required": ["v"]}}
    assert validate({"k": 1}, cond)
    assert validate({"k": 2}, cond) == []


def test_validator_strings_numbers_and_check() -> None:
    assert validate("abc", {"pattern": "^a", "minLength": 2, "maxLength": 3}) == []
    assert validate("zz", {"pattern": "^a"})
    assert validate(0, {"exclusiveMinimum": 0})
    assert validate(float("nan"), {"type": "number"})
    with pytest.raises(SchemaError) as excinfo:
        check({"a": 1}, {"required": ["b"]}, what="thing")
    assert "thing is invalid" in str(excinfo.value)


# ----------------------------------------------------------------------------- common

def test_measured_pose_and_refs() -> None:
    assert Measured.parse([1.5, 0.1]) == Measured(1.5, 0.1)
    assert Measured.parse(2) == Measured(2.0, None)
    assert Measured(1.0, 0.5).z(2.0) == pytest.approx(2.0)
    assert Measured(1.0, None).z(2.0) is None
    with pytest.raises(ValueError):
        Measured.parse(True)
    pose = Pose((1, 2, 3), (0, 0, 0, 1))
    assert pose.to_json() == {"position_m": [1.0, 2.0, 3.0], "quat_xyzw": [0.0, 0.0, 0.0, 1.0]}
    with pytest.raises(ValueError):
        Pose((0, 0, 0), (0, 0, 0, 2))
    ref = parse_pose_ref("track:car_1@12")
    assert (ref.object, ref.frame) == ("car_1", 12)
    assert ref.to_json() == "track:car_1@12"
    with pytest.raises(ValueError):
        parse_pose_ref("car_1@12")
    assert default_symmetry("prop.sphere") == "sphere"
    assert default_symmetry("vehicle.car") == "none"


def test_jsonio_is_strict_and_canonical(tmp_path) -> None:
    with pytest.raises(jsonio.JsonError):
        jsonio.loads('{"a": 1, "a": 2}')
    with pytest.raises(jsonio.JsonError):
        jsonio.loads('{"a": NaN}')
    with pytest.raises(jsonio.JsonError):
        jsonio.dumps({"a": float("inf")})
    assert jsonio.canonical_sha256({"b": 1, "a": [1, 2]}) == jsonio.canonical_sha256({"a": [1, 2], "b": 1})
    path = jsonio.dump({"x": (1, 2)}, tmp_path / "sub" / "x.json")
    assert jsonio.load(path) == {"x": [1, 2]}


# ----------------------------------------------------------------------------- scene

def test_scene_roundtrip_and_accessors() -> None:
    doc = scene_doc()
    sc = scene.Scene.from_dict(doc, base_dir="/data/clip")
    assert [a.id for a in sc.dynamic_actors] == ["car_1", "crate"]
    assert [a.id for a in sc.controlled_actors] == ["car_1"]
    assert sc.actor("wall").mesh == 2
    assert sc.event("impact").kind == "contact"  # inferred from the name
    assert sc.event("impact").sigma_frames == 0.5
    assert sc.actor("crate").symmetry_class == "box"
    assert str(sc.world_path()) == "/data/clip/world"
    again = scene.Scene.from_dict(sc.to_dict())
    assert again.to_dict() == sc.to_dict()
    assert sc.sha256() == again.sha256()
    assert schema.validate_document(sc.to_dict()) == []


@pytest.mark.parametrize(
    "mutate, fragment",
    [
        (lambda d: d["actors"][0]["initial_state"].update(velocity=[1, 0, 0]), "not an allowed member"),
        (lambda d: d["targets"][0].update(source="engine_run"), "must be one of"),
        (lambda d: d["phases"].append({"name": "x", "from": "nowhere", "to": "end"}), "neither an event"),
        (lambda d: d["events"].append({"name": "start", "frame": [1, 2]}), "reserved phase boundary"),
        (lambda d: d["events"].append({"name": "late", "frame": [9, 2]}), "first frame is after last"),
        (lambda d: d["actors"][1]["initial_state"].update(mode="drive_in", frame0_pose="track:crate@0",
                                                          speed_mps=1.0), "passive actor cannot drive in"),
        (lambda d: d["actors"].append(copy.deepcopy(d["actors"][1])), "used twice"),
        (lambda d: d["targets"].append({"metric": "m", "actor": "ghost", "value": 1, "source": "design_value"}),
         "unknown dynamic actor"),
        (lambda d: d["requires"].update(controlled_actors=0), "controlled actors"),
        (lambda d: d["actors"][1].update(seatable=True), "only vehicles can be seatable"),
        (lambda d: d["actors"][0]["initial_state"].pop("speed_mps"), "speed_mps"),
        (lambda d: d["actors"][2].pop("mesh"), "mesh"),
    ],
)
def test_scene_rejects(mutate, fragment) -> None:
    doc = scene_doc()
    mutate(doc)
    issues = scene.validate_scene(doc)
    assert issues, "expected the scene to be refused"
    assert any(fragment in issue.message for issue in issues), [str(i) for i in issues]


def test_load_scene_resolves_world(tmp_path) -> None:
    path = tmp_path / "scene.json"
    path.write_text(json.dumps(scene_doc()))
    sc = scene.load_scene(path)
    assert sc.world_path() == tmp_path / "world"


# ----------------------------------------------------------------------------- templates

def test_templates_resolution_order() -> None:
    doc = {"schema": "v2e.templates/1", "defaults": {"vehicle.truck": "vehicle.stock"},
           "objects": {"crate": {"template": "prop.box", "density_class": "wood",
                                 "overrides": {"mass_kg": {"value": 18.0, "sigma": 0.5, "status": "measured",
                                                           "reason": "weighed"}}}}}
    t = templates.Templates.from_dict(doc)
    assert t.rule_for("crate", "prop.box").overrides["mass_kg"].value == 18.0
    assert t.rule_for("truck_1", "vehicle.truck").template == "vehicle.stock"
    assert t.rule_for("car", "vehicle.car").template == "vehicle.generic"
    assert t.rule_for("ball", "prop.sphere").template == "prop.sphere"
    assert t.rule_for("thing", "prop.chair").template == "prop.mesh"
    assert t.rule_for("wall", "static").template == "static.mesh"
    assert templates.Templates.from_dict(t.to_dict()).to_dict() == t.to_dict()
    rule = templates.TemplateRule("vehicle.stock", tolerance={"length": 0.2})
    assert rule.effective_tolerance()["length"] == 0.2
    assert rule.effective_tolerance()["width"] == templates.DIMENSION_TOLERANCE["width"]


@pytest.mark.parametrize(
    "rule, fragment",
    [
        ({"template": "vehicle.stock", "rule": "content_id"}, "content_id"),
        ({"template": "prop.box", "overrides": {"mass_kg": {"value": 1, "status": "fitted", "reason": "x"}}},
         "nothing is fitted"),
        ({"template": "vehicle.generic", "density_class": "wood"}, "only props"),
        ({"template": "vehicle.generic", "rule": "nearest_by_dims"}, "only vehicle.stock"),
        ({"template": "vehicle.stock", "fallback": "vehicle.stock"}, "fallback"),
        ({"template": "rocket"}, "matches none"),
    ],
)
def test_templates_reject(rule, fragment) -> None:
    issues = templates.validate_templates({"schema": "v2e.templates/1", "objects": {"a": rule}})
    assert any(fragment in issue.message for issue in issues), [str(i) for i in issues]


# ----------------------------------------------------------------------------- runlog

def make_runlog_text() -> str:
    header = {"schema": "v2e-runlog/1", "frame": "z_up_level", "actors": ["a", "b"], "backend": "fake",
              "dt_s": 0.1}
    rows = [
        {"t": 0.0, "actors": {"a": {"p": [0, 0, 0], "q": [0, 0, 0, 1]}, "b": {"p": [1, 0, 0], "q": [0, 0, 0, 1]}}},
        {"t": 0.1, "phase": "go", "actors": {"a": {"p": [0.1, 0, 0], "q": [0, 0, 0, 1], "v": [1, 0, 0]}},
         "cmd": {"a": {"drive.throttle": 0.5}}, "events": [{"name": "contact", "actor": "a", "other": "b"}]},
    ]
    return "\n".join(json.dumps(x) for x in [header, *rows]) + "\n"


def test_runlog_roundtrip(tmp_path) -> None:
    log = runlog.read_runlog(make_runlog_text())
    assert log.header.actors == ("a", "b")
    assert log.times() == [0.0, 0.1]
    assert log.rows[1].cmd["a"]["drive.throttle"] == 0.5
    assert log.events()[0][1]["other"] == "b"
    path = runlog.write_runlog(log, tmp_path / "rows.jsonl")
    again = runlog.read_runlog(path)
    assert again == log
    assert schema.check_file(path) == []


def test_position_only_runlog_requires_an_explicit_header(tmp_path) -> None:
    header = {"schema": "v2e-runlog/1", "actors": ["crate"], "attitude_observed": False}
    rows = [{"t": 0.0, "actors": {"crate": {"p": [0, 0, 2], "v": [0, 0, 0]}}}]
    text = "\n".join(json.dumps(x) for x in [header, *rows])
    log = runlog.read_runlog(text)
    assert not log.header.attitude_observed
    assert log.rows[0].actors["crate"].q is None
    path = runlog.write_runlog(log, tmp_path / "position-only.jsonl")
    assert runlog.read_runlog(path) == log
    assert schema.check_file(path) == []
    assert "q" not in json.loads(path.read_text().splitlines()[1])["actors"]["crate"]
    for flag in (None, True):
        strict = {k: v for k, v in header.items() if k != "attitude_observed"}
        if flag is not None:
            strict["attitude_observed"] = flag
        assert runlog.validate_runlog(strict, rows)
    rows[0]["actors"]["crate"]["q"] = [0, 0, 0, 2]
    assert any("quaternion norm" in i.message for i in runlog.validate_runlog(header, rows))


@pytest.mark.parametrize(
    "edit, fragment",
    [
        (lambda h, r: r[1].update(t=0.0), "does not increase"),
        (lambda h, r: r[1]["actors"].update(c={"p": [0, 0, 0], "q": [0, 0, 0, 1]}), "not declared"),
        (lambda h, r: r[1]["actors"]["a"].update(q=[0, 0, 0, 2]), "quaternion norm"),
        (lambda h, r: r[1].update(cmd={"z": {"x": 1}}), "does not declare"),
        (lambda h, r: r[1]["actors"]["a"].update(force=[1, 2, 3]), "not an allowed member"),
        (lambda h, r: h.update(schema="other/1"), "must be"),
    ],
)
def test_runlog_rejects(edit, fragment) -> None:
    lines = [json.loads(line) for line in make_runlog_text().splitlines()]
    header, rows = lines[0], lines[1:]
    edit(header, rows)
    issues = runlog.validate_runlog(header, rows)
    assert any(fragment in issue.message for issue in issues), [str(i) for i in issues]
    with pytest.raises(SchemaError):
        runlog.read_runlog("\n".join(json.dumps(x) for x in [header, *rows]))


# ----------------------------------------------------------------------------- catalog and dispatch

def test_catalog_roundtrip_and_duplicates() -> None:
    doc = {"schema": "v2e.catalog/1", "backend": "fake", "limits": {"max_seatable_vehicles": 2},
           "entries": [{"id": "a", "kind": "vehicle", "seatable": True,
                        "dims_m": {"length": 4.0, "width": 1.7, "height": 1.4}}]}
    cat = catalog.ContentCatalog.from_dict(doc)
    assert cat.vehicles()[0].id == "a"
    assert cat.max_seatable_vehicles == 2
    assert catalog.ContentCatalog.from_dict(cat.to_dict()).to_dict() == cat.to_dict()
    doc["entries"].append(dict(doc["entries"][0]))
    assert any("used twice" in i.message for i in catalog.validate_catalog(doc))


def test_check_document_dispatch_and_export(tmp_path) -> None:
    assert schema.validate_document({"schema": "nope/1"})[0].message.startswith("unknown format")
    assert schema.validate_document([]) != []
    with pytest.raises(SchemaError):
        schema.check_document({"schema": "v2e.templates/1", "objects": {"a": {"template": "x"}}})
    written = schema.export_json_schemas(tmp_path)
    assert len(written) == len(schema.SCHEMAS)
    for path in written:
        loaded = json.loads(path.read_text())
        assert loaded.get("$schema", "").startswith("https://json-schema.org/")
