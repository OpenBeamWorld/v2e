# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""World -> scene -> compile -> fake run -> export -> score, checked against first principles.

The reference is the closed-form synthetic world (no footage). The score must find the crate's
first contact at sqrt(2h/g) within a frame, carry the run's claim class, and keep the parked
car where the reconstruction put it.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from openbeam_v2e.compile import compile_scene
from openbeam_v2e.schema import validate_document
from openbeam_v2e.score import score_run
from openbeam_v2e.world import write_world

from world_fixtures import FPS, G, synthetic_world

fake = pytest.importorskip("openbeam_v2e.backends.fake")
harness = pytest.importorskip("openbeam_v2e.backends.harness")
base = pytest.importorskip("openbeam_v2e.backends.base")

DROP = 2.0


def _scene(tmp_path: Path) -> Path:
    write_world(synthetic_world(car_kw={"speed": 0.0}, crate_kw={"drop": DROP}), tmp_path / "world")  # parked car
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
        "events": [{"name": "crate_lands", "frame": [19, 20], "kind": "contact", "actors": ["crate", "ground"]}],
        "phases": [{"name": "fall", "from": "start", "to": "crate_lands"}],
        "targets": [{"metric": "first_contact_time", "actor": "crate", "value": [math.sqrt(2 * DROP / G), 1 / FPS],
                     "source": "first_principles"}],
        "requires": {"effects": ["contact", "gravity"], "observables": ["position", "attitude"]},
    }
    path = tmp_path / "scene.json"
    path.write_text(json.dumps(scene), encoding="utf-8")
    return path


def test_fake_run_scores_against_first_principles(tmp_path: Path) -> None:
    scene_path = _scene(tmp_path)
    scene = json.loads(scene_path.read_text())
    result = compile_scene(scene_path, connector="fake", out=tmp_path / "pkg")
    assert result.t0_s == 0.0  # the crate hangs in the air at frame 0, so frame 0 is spawn time
    compiled = fake.compile_package(tmp_path / "pkg" / "package.json", scene=scene, world=tmp_path / "world")
    backend = fake.FakeBackend()
    plan = harness.negotiate(scene, backend.manifest(), lockstep=True)
    spec = base.RunSpec(2.5, plan.host, plan.regime, telemetry_hz=120.0)  # run 2.5 s of simulated time
    run = harness.execute(backend, compiled, plan, spec, tmp_path / "run")
    assert run.valid()

    report = score_run(tmp_path / "run", scene=scene_path, out=tmp_path / "score.json")
    assert validate_document(report) == []
    assert report["claim"]["class"] == "test_double" and report["claim"]["valid"] is True
    target = report["targets"][0]
    assert target["status"] == "measured" and target["pass"] is True, target
    lands = report["events"][0]
    assert lands["run_frame"] is not None and abs(lands["delta_frames"]) <= 1.0
    fall = report["phases"][0]["objects"]
    assert fall["crate"]["position_rms_m"] < 0.05       # free fall matches the closed form
    assert fall["car"]["position_rms_m"] < 0.05         # the parked car stays put
    assert json.loads((tmp_path / "score.json").read_text())["schema"] == "v2e.score/1"
