# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""Negotiation (plan or itemised refusal) and run execution through the harness."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest
from connector_fixtures import car_scene, drop_scene, openbeam_scene, schedule

from openbeam_v2e.backends.base import STEPPED, ControllerHost, Plan, Refusal, RunSpec
from openbeam_v2e.backends.fake import FakeBackend, compile_scene
from openbeam_v2e.backends.harness import RunDirectoryError, execute, negotiate
from openbeam_v2e.backends.openbeam.backend import OpenBeamBackend
from openbeam_v2e.backends.openbeam.cassette import ReplayClient


def _codes(result) -> set[str]:
    assert isinstance(result, Refusal), result
    return {r.code for r in result.reasons}


FAKE = FakeBackend().manifest()
PROGRAM = {"car_1": schedule("car_1", [{"t": 0.0, "cmd": {"drive.throttle": 0.5}}])}


def test_plan_for_a_feasible_scene() -> None:
    plan = negotiate(car_scene(), FAKE, programs=PROGRAM, lockstep=True)
    assert isinstance(plan, Plan) and plan.ok
    assert plan.regime == STEPPED and plan.host is ControllerHost.CLIENT_LOCKSTEP
    assert plan.controlled == ("car_1",) and "drive.parking_brake" in plan.channels["car_1"]
    assert {e.name for e in plan.effects} == {"drive", "brake", "steer", "contact"}
    assert Plan.from_json(json.loads(json.dumps(plan.to_json()))) == plan


def test_missing_program_and_program_for_a_passive_actor() -> None:
    assert "missing_program" in _codes(negotiate(car_scene(), FAKE, lockstep=True))
    extra = {**PROGRAM, "obj": schedule("obj", [{"t": 0.0, "cmd": {}}])}
    assert "program_for_passive_actor" in _codes(negotiate(car_scene(), FAKE, programs=extra, lockstep=True))
    wrong = {"car_1": {**PROGRAM["car_1"], "actor": "car_2"}}
    assert "bad_program" in _codes(negotiate(car_scene(), FAKE, programs=wrong, lockstep=True))


def test_missing_effects_observables_channels_and_regimes() -> None:
    scene = car_scene(requires={"effects": ["drive", "deformation", "thrust"], "observables": ["position", "nodes"],
                                "channels": [{"id": "drive.parking_brake", "kind": "continuous"},
                                             "plugin.fan"]})
    codes = _codes(negotiate(scene, FAKE, programs=PROGRAM, opt_in={"regime": "render_paced"}, lockstep=True))
    assert codes >= {"missing_effect", "missing_observable", "channel_kind", "missing_channel", "regime_unavailable"}


def test_actor_and_seatable_limits() -> None:
    caps = dataclasses.replace(FAKE.capabilities, max_controlled_actors=1, max_seatable_vehicles=1)
    manifest = dataclasses.replace(FAKE, capabilities=caps)
    second = {"id": "car_2", "class": "vehicle.car", "seatable": True, "control": "driver",
              "initial_state": {"mode": "at_rest", "pose": {"position_m": [10, 0, 0.725], "quat_xyzw": [0, 0, 0, 1]}}}
    programs = {**PROGRAM, "car_2": schedule("car_2", [{"t": 0.0, "cmd": {}}])}
    assert _codes(negotiate(car_scene([second]), manifest, programs=programs, lockstep=True)) >= {
        "actor_limit", "seatable_limit"}


def test_host_requires_lockstep_support() -> None:
    assert "no_controller_host" in _codes(negotiate(car_scene(), FAKE, programs=PROGRAM, lockstep=False))


def test_initial_states_that_cannot_spawn_at_rest_are_refused() -> None:
    scene = car_scene()
    scene["actors"][0]["initial_state"] = {"mode": "moving"}
    assert "initial_state" in _codes(negotiate(scene, FAKE, programs=PROGRAM, lockstep=True))
    passive = drop_scene()
    passive["actors"][0]["initial_state"] = {"mode": "drive_in"}
    assert "initial_state" in _codes(negotiate(passive, FAKE, lockstep=True))


def test_substitutes_need_a_declaration_by_the_connector() -> None:
    scene = drop_scene()
    scene["requires"]["effects"] = ["contact", "launch"]
    assert "substitute_not_declared" in _codes(negotiate(scene, FAKE, opt_in={"substitutes": {"launch": "x.spring"}},
                                                         lockstep=True))
    caps = dataclasses.replace(FAKE.capabilities, extras={"substitute.launch": "x.spring"})
    plan = negotiate(scene, dataclasses.replace(FAKE, capabilities=caps), lockstep=True,
                     opt_in={"substitutes": {"launch": "x.spring"}})
    assert plan.ok and any(e.name == "launch" for e in plan.effects)


def test_stepped_regime_needs_a_player_vehicle_or_an_anchor() -> None:
    manifest = OpenBeamBackend(ReplayClient([])).manifest()
    scene = openbeam_scene()
    plan = negotiate(scene, manifest)
    assert plan.ok and any("position only" in n for n in plan.notes)
    props_only = {**scene, "actors": [scene["actors"][0]]}
    assert "regime_condition" in _codes(negotiate(props_only, manifest))
    anchored = negotiate(props_only, manifest, opt_in={"add_anchor": True})
    assert anchored.ok and any("anchor" in n for n in anchored.notes)


def test_openbeam_without_a_host_cannot_drive() -> None:
    manifest = OpenBeamBackend(ReplayClient([])).manifest()
    codes = _codes(negotiate(car_scene(requires={"effects": ["drive"], "observables": ["position"]}), manifest,
                             programs=PROGRAM))
    assert {"missing_effect", "actor_limit", "no_controller_host"} <= codes


class _Spy(FakeBackend):
    def install(self, packages):
        self.plan_on_disk = (self._run_dir / "plan.json").is_file()
        return super().install(packages)


def test_execute_writes_the_plan_first_and_refuses_used_directories(tmp_path: Path) -> None:
    scene = drop_scene(0.5)
    plan = negotiate(scene, FAKE, lockstep=True)
    spec = RunSpec(duration_s=0.2, host=plan.host, regime=plan.regime)
    spy = _Spy()
    spy._run_dir = tmp_path / "run"
    result = execute(spy, compile_scene(scene), plan, spec, tmp_path / "run")
    assert spy.plan_on_disk
    for name in ("plan.json", "install.json", "run.json", "rows.jsonl", "ledger/calls.jsonl"):
        assert (tmp_path / "run" / name).is_file(), name
    assert result.valid() and result.attestation.audit == "clean"
    with pytest.raises(RunDirectoryError):
        execute(FakeBackend(), compile_scene(scene), plan, spec, tmp_path / "run")
    with pytest.raises(ValueError):
        execute(FakeBackend(), compile_scene(scene), plan, dataclasses.replace(spec, regime="render_paced"),
                tmp_path / "other")
