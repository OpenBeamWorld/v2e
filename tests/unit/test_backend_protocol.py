# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""The connector protocol: value checks, claim classes, provenance and the no-bypass rules."""

from __future__ import annotations

import dataclasses
import inspect
import math
from pathlib import Path

import pytest

from openbeam_v2e.backends import registry
from openbeam_v2e.backends.base import (
    CORE_CHANNEL_IDS,
    CORE_RECORDINGS,
    CORE_REGIMES,
    FORBIDDEN_OPERATIONS,
    STEPPED,
    ActuatorChannel,
    Attestation,
    Backend,
    BackendManifest,
    Capabilities,
    ChannelKind,
    CompiledScene,
    ContentOrigin,
    ControllerHost,
    Effect,
    Layer,
    LevelRef,
    Package,
    Plan,
    Pose,
    RegimeSupport,
    RunResult,
    RunSpec,
    Source,
    SpawnRecord,
    SpawnSpec,
    check_program,
    from_jsonable,
    sha256_json,
    to_jsonable,
)


def _manifest(kind: str = "engine") -> BackendManifest:
    caps = Capabilities(channels=(), soft_body=True, detachment="breakgroups",
                        regimes=(RegimeSupport(STEPPED, True),), physics_dt_s=0.001, recordings=("none",),
                        spawn_at_pose=True, max_controlled_actors=2, max_seatable_vehicles=2,
                        controller_hosts=frozenset({ControllerHost.CLIENT_LOCKSTEP}),
                        observables=frozenset({"position"}), camera_runtime_fov=False, instance_mask=False)
    return BackendManifest("x", kind, "Engine 1", "api", "dialect", caps)


def _result(effects, kind: str = "engine", content=(), **attest) -> RunResult:
    pose = Pose((0.0, 0.0, 1.0))
    spawn = SpawnRecord("a", pose, pose, 0.0, 0.0, "test")
    a = dict(spawns=(spawn,), initial_velocity_mps=0.0, teleports=0, in_run_resets=0, removed_actors=0,
             regime=STEPPED, programs=(), ledger=Path("ledger"), ledger_head_sha256="0" * 64, audit="clean")
    a.update(attest)
    plan = Plan("x", STEPPED, None, ("a",), (), (), {}, {}, tuple(effects))
    return RunResult(_manifest(kind), plan, tuple(effects), tuple(content), Attestation(**a), Path("rows.jsonl"),
                     {}, 0.0, None, None, 1.0, 0.1)


def test_continuous_channel_checks_range_and_type() -> None:
    ch = ActuatorChannel("drive.steer", "steer", ChannelKind.CONTINUOUS, (-1.0, 1.0))
    assert ch.check(0.5) == 0.5
    assert ch.check(1.0 + 1e-12) == 1.0
    for bad in (1.5, -2, float("nan"), True, "0.1", None):
        with pytest.raises(ValueError):
            ch.check(bad)


def test_toggle_and_discrete_channels() -> None:
    toggle = ActuatorChannel("drive.parking_brake", "parking_brake", ChannelKind.TOGGLE)
    assert toggle.check(1) == 1.0 and toggle.check(0) == 0.0
    with pytest.raises(ValueError):
        toggle.check(0.5)
    gear = ActuatorChannel("x.gear", "gear", ChannelKind.DISCRETE, levels=(-1.0, 0.0, 1.0, 2.0))
    assert gear.check(2) == 2.0
    with pytest.raises(ValueError):
        gear.check(3)


@pytest.mark.parametrize("sources,expected", [
    ([Source.NATIVE, Source.NATIVE], "native"),
    ([Source.NATIVE, Source.NATIVE_SUBSTITUTE], "emulated"),
    ([Source.NATIVE_SUBSTITUTE, Source.SCRIPTED], "scripted"),
    ([Source.SCRIPTED, Source.ABSENT], "incomplete"),
    ([Source.NATIVE, Source.TEST_DOUBLE], "test_double"),
    ([Source.TEST_DOUBLE, Source.ABSENT], "incomplete"),
])
def test_claim_class_is_computed_from_sources(sources, expected) -> None:
    effects = [Effect(f"e{i}", s, "m", Layer.ENGINE_CORE, "a" * 64 if s is Source.SCRIPTED else None)
               for i, s in enumerate(sources)]
    result = _result(effects)
    assert result.claim_class() == expected
    assert result.physics_statement().lower().startswith(expected.replace("_", " ").split()[0][:4])


def test_test_double_manifest_always_claims_test_double() -> None:
    assert _result([Effect("contact", Source.NATIVE, "m")], kind="test_double").claim_class() == "test_double"


def test_scripted_effect_must_pin_its_artefact() -> None:
    with pytest.raises(ValueError):
        Effect("thrust", Source.SCRIPTED, "plugin.x")


def test_provenance_signature_depends_on_effects_and_content_only() -> None:
    effects = [Effect("contact", Source.NATIVE, "m")]
    content = [ContentOrigin("a", "generated", "gen", "1", "f" * 64, {"length_m": 4.0})]
    base = _result(effects, content=content)
    same = _result(list(reversed(effects)), content=content, audit="different audit text")
    other = _result(effects, content=[dataclasses.replace(content[0], params_sha256="e" * 64)])
    assert base.provenance_signature() == same.provenance_signature()
    assert base.provenance_signature() != other.provenance_signature()


def test_validity_rules() -> None:
    effects = [Effect("contact", Source.NATIVE, "m")]
    assert _result(effects).valid()
    assert not _result(effects, audit="call 3: not allowed").valid()
    assert not _result(effects, removed_actors=1).valid()
    assert not _result(effects, teleports=1).valid()
    pose = Pose((0.0, 0.0, 1.0))
    off = SpawnRecord("a", pose, Pose((0.0, 0.1, 1.0)), 0.1, 0.0, "test")
    bad = _result(effects, spawns=(off,))
    assert not bad.valid() and "spawn residual" in bad.invalid_reasons()[0]


def test_run_result_json_round_trip(tmp_path: Path) -> None:
    result = _result([Effect("contact", Source.NATIVE, "m")])
    data = result.to_json()
    assert data["claim_class"] == "native" and data["valid"] is True
    back = RunResult.from_json(data, base=tmp_path)
    assert back.rows == tmp_path / "rows.jsonl"
    assert back.provenance_signature() == result.provenance_signature()
    assert back.manifest == result.manifest


def test_compiled_scene_round_trip_resolves_package_paths(tmp_path: Path) -> None:
    compiled = CompiledScene("fake", "d", LevelRef(package="lvl"),
                             (Package("lvl", "level", "a" * 64, path=Path("level.zip")),), (),
                             (SpawnSpec("a", "obj", Pose.from_yaw((1, 2, 3), 90.0)),))
    back = CompiledScene.from_json(compiled.to_json(), base=tmp_path)
    assert back.packages[0].path == tmp_path / "level.zip"
    assert back.spawns[0].pose.yaw_deg() == pytest.approx(90.0)
    with pytest.raises(ValueError):
        CompiledScene.from_json({**compiled.to_json(), "schema": "other"})


def test_pose_normalises_and_measures() -> None:
    pose = Pose((1, 2, 3), (0, 0, 0, 1.0004))
    assert pose.quat_xyzw == (0.0, 0.0, 0.0, 1.0)
    with pytest.raises(ValueError):
        Pose((0, 0, 0), (0, 0, 0, 0.5))
    with pytest.raises(ValueError):
        Pose((0, math.inf, 0))
    a, b = Pose.from_yaw((0, 0, 0), 10.0), Pose.from_yaw((3, 4, 0), 40.0)
    assert a.angle_to_deg(b) == pytest.approx(30.0)
    assert a.distance_to_m(b) == pytest.approx(5.0)
    assert Pose.parse({"position_m": [0, 0, 0], "yaw_deg": -45}).yaw_deg() == pytest.approx(-45.0)


def test_program_check_refuses_engine_code() -> None:
    good = {"schema": "v2e.program/1", "actor": "car_1", "model": "schedule", "params": {}}
    check_program(good)
    for bad in ({**good, "schema": "x"}, {**good, "actor": ""}, {**good, "code": "anything"},
                {**good, "params": []}):
        with pytest.raises(ValueError):
            check_program(bad)


def test_run_spec_rejects_nonsense() -> None:
    with pytest.raises(ValueError):
        RunSpec(duration_s=0, host=None, regime=STEPPED)
    with pytest.raises(ValueError):
        RunSpec(duration_s=1, host=None, regime=STEPPED, telemetry_hz=-1)


def test_json_helpers_round_trip_nested_dataclasses() -> None:
    caps = _manifest().capabilities
    back = from_jsonable(Capabilities, to_jsonable(caps))
    assert back == caps
    assert sha256_json({"b": 1, "a": [1.5]}) == sha256_json({"a": [1.5], "b": 1})
    with pytest.raises(ValueError):
        to_jsonable(float("nan"))


def _public_methods(cls: type) -> list[str]:
    return [name for name, _ in inspect.getmembers(cls, callable) if not name.startswith("_")]


@pytest.mark.parametrize("name", sorted(registry.BUILTIN))
def test_connectors_offer_no_physics_bypass(name: str) -> None:
    cls = registry.backend_class(name)
    for method in _public_methods(cls):
        lowered = method.lower()
        for word in FORBIDDEN_OPERATIONS:
            assert not (word in lowered and lowered.split("_")[0] in ("set", "apply", "add", "teleport", "reset",
                                                                     "put", "write")), \
                f"{name}.{method} looks like a physics bypass"
    # the interface itself spawns only at rest and resets only between runs
    abstract = set(Backend.__abstractmethods__)
    assert abstract == {"manifest", "mechanisms", "install", "open_scene", "spawn", "run", "reset_between_runs",
                        "close"}


def test_builtin_connectors_declare_only_core_values() -> None:
    from openbeam_v2e.backends.fake import FakeBackend
    from openbeam_v2e.backends.openbeam.backend import OpenBeamBackend
    from openbeam_v2e.backends.openbeam.cassette import ReplayClient

    for backend in (FakeBackend(), OpenBeamBackend(ReplayClient([]))):
        caps = backend.manifest().capabilities
        assert {c.id for c in caps.channels} <= set(CORE_CHANNEL_IDS)
        assert {r.regime for r in caps.regimes} <= set(CORE_REGIMES)
        assert set(caps.recordings) <= set(CORE_RECORDINGS)
        assert not caps.extras
