# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""End to end on the fake connector: world package -> scene -> compile -> run -> export.

The world is the closed-form synthetic one from ``world_fixtures`` (a car-sized box at rest on
flat ground, a wall, a crate dropped from 2 m). Nothing here depends on any footage. The test
checks the plumbing (frames, reference points, timeline, provenance), and the one physical
quantity a rigid stand-in must get right: the crate's first contact at sqrt(2h/g).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

world_fixtures = pytest.importorskip("world_fixtures")
compile_mod = pytest.importorskip("openbeam_v2e.compile")
world_io = pytest.importorskip("openbeam_v2e.world.io")
world_export = pytest.importorskip("openbeam_v2e.world.export")

from openbeam_v2e.agents import steps  # noqa: E402
from openbeam_v2e.agents.tools import AgentContext, call_tool  # noqa: E402
from openbeam_v2e.backends.base import RunSpec  # noqa: E402
from openbeam_v2e.backends.fake import STANDARD_GRAVITY, FakeBackend, compile_package  # noqa: E402
from openbeam_v2e.backends.harness import execute, negotiate  # noqa: E402


def _scene(tmp_path: Path) -> Path:
    world = world_fixtures.synthetic_world(crate_kw={"drop": 2.0})
    world_io.write_world(world, tmp_path / "world")
    scene = {
        "schema": "v2e.scene/1", "world": "world/",
        "actors": [
            {"id": "car", "class": "vehicle.car", "seatable": True, "control": "passive",
             "initial_state": {"mode": "at_rest", "pose": "track:car@0"}},
            {"id": "crate", "class": "prop.box", "control": "passive",
             "initial_state": {"mode": "at_rest", "pose": "track:crate@0"}},
            {"id": "ground", "class": "static", "mesh": 1, "role": "drivable"},
            {"id": "wall", "class": "static", "mesh": 2},
        ],
        "requires": {"effects": ["contact", "gravity"], "observables": ["position", "attitude"]},
    }
    path = tmp_path / "scene.json"
    path.write_text(json.dumps(scene), encoding="utf-8")
    return path


def test_world_to_export_round_trip(tmp_path: Path) -> None:
    scene_path = _scene(tmp_path)
    scene = json.loads(scene_path.read_text())
    compile_mod.compile_scene(scene_path, connector="fake", out=tmp_path / "pkg")
    compiled = compile_package(tmp_path / "pkg" / "package.json", scene=scene, world=tmp_path / "world")
    assert {s["name"].split("#")[0] for s in compiled.packages[0].contents["statics"]} >= {"ground", "wall"}
    assert {c.actor for c in compiled.content} == {"car", "crate"}

    backend = FakeBackend()
    plan = negotiate(scene, backend.manifest(), lockstep=True)
    assert plan.ok, plan
    spec = RunSpec(duration_s=2.4, host=plan.host, regime=plan.regime, telemetry_hz=120.0)
    result = execute(backend, compiled, plan, spec, tmp_path / "run")
    assert result.valid() and result.claim_class() == "test_double"
    assert result.t0_s == pytest.approx(compiled.t0_s)
    contact = result.events["first_contact.crate.ground"]
    assert contact == pytest.approx(math.sqrt(2 * 2.0 / STANDARD_GRAVITY), abs=1.5 / 120.0)

    world_export.export_run(result.rows, tmp_path / "world", tmp_path / "export", t0_s=result.t0_s)
    exported = world_io.read_world(tmp_path / "export")
    reference = world_io.read_world(tmp_path / "world")
    run_crate, ref_crate = exported.object("crate"), reference.object("crate")
    frames = int(math.floor(contact * reference.fps)) - 1
    # before the first contact both are free fall from the same pose: the fake matches the closed form
    assert np.allclose(run_crate.position[:frames], ref_crate.position[:frames], atol=0.03)
    # the parked car stays where the reconstruction put it (its reference point is the hull origin)
    assert np.allclose(exported.object("car").position[0], reference.object("car").position[0], atol=0.03)


def test_agent_tools_compile_and_run_a_world_scene(tmp_path: Path) -> None:
    scene_path = _scene(tmp_path)
    ctx = AgentContext(root=tmp_path)
    compiled = call_tool(ctx, "compile", {"scene": "scene.json", "out": "compiled"})
    assert compiled["ok"], compiled
    assert compiled["package"] == "compiled/package.json" and compiled["t0_s"] is not None
    run = call_tool(ctx, "run", {"compiled": "compiled", "out": "runs/one", "duration_s": 1.0})
    assert run["ok"], run
    assert run["valid"] and run["claim_class"] == "test_double"
    exported = call_tool(ctx, "export", {"run": "runs/one", "reference": "world", "out": "export"})
    assert exported["ok"], exported
    assert (tmp_path / "export").is_dir()
    assert scene_path.is_file()
    if steps.resolve("score") is not None:
        scored = call_tool(ctx, "score", {"run": "runs/one", "scene": "scene.json", "out": "score.json"})
        assert scored["ok"], scored
        assert (tmp_path / "score.json").is_file()
