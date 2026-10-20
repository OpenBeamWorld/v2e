# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""A separately authored drop scene through real CLI and stdio MCP processes.

This checks public-interface integration and the free-fall contact time, not a
native-engine or real-video gate. No prepared benchmark case is used.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from openbeam_v2e.schema.common import Measured
from openbeam_v2e.world import Camera, CameraTrack, ObjectTrack, StaticMesh, World, write_world
from openbeam_v2e.world.geometry import box_mesh

FPS = 24.0
DROP = 1.25
GRAVITY = 9.80665


def _write_scene(root: Path) -> None:
    frames = 24
    size = (0.5, 0.4, 0.3)
    vertices, faces = box_mesh(*size)
    t = np.arange(frames) / FPS
    positions = np.column_stack([
        np.full(frames, 0.7), np.full(frames, -0.3),
        size[2] / 2 + np.maximum(DROP - 0.5 * GRAVITY * t * t, 0),
    ])
    obj = ObjectTrack(
        "parcel", "prop.box", vertices, faces, positions,
        np.tile([0.0, 0.0, 0.0, 1.0], (frames, 1)),
        dims_m={k: Measured(v, 0.01) for k, v in zip(("length", "width", "height"), size)},
    )
    ground = StaticMesh(1, np.array([[-8, -8, 0], [8, -8, 0], [8, 8, 0], [-8, 8, 0]], float),
                        np.array([[0, 1, 2], [0, 2, 3]]), "floor", "drivable")
    camera = Camera.from_fov(160, 90, 65.0, centre=(1.0, -7.0, 3.0), target=(0.7, -0.3, 0.8))
    write_world(World(FPS, frames, CameraTrack.static(camera, frames), [ground], [obj],
                      kind="synthetic_truth"), root / "world")
    contact_frame = math.sqrt(2 * DROP / GRAVITY) * FPS
    scene = {
        "schema": "v2e.scene/1", "world": "world",
        "actors": [
            {"id": "parcel", "class": "prop.box", "control": "passive",
             "initial_state": {"mode": "at_rest", "pose": "track:parcel@0"}},
            {"id": "floor", "class": "static", "mesh": 1, "role": "drivable"},
        ],
        "events": [{"name": "lands", "frame": [math.floor(contact_frame), math.ceil(contact_frame)],
                    "kind": "contact", "actors": ["parcel", "floor"]}],
        "phases": [{"name": "fall", "from": "start", "to": "lands"}],
        "targets": [{"metric": "first_contact_time", "actor": "parcel",
                     "value": [math.sqrt(2 * DROP / GRAVITY), 1 / FPS], "source": "first_principles"}],
        "requires": {"effects": ["contact", "gravity"], "observables": ["position", "attitude"]},
    }
    (root / "scene.json").write_text(json.dumps(scene))


@pytest.mark.parametrize("interface", ["cli", "mcp"])
def test_new_drop_scene_through_public_processes(tmp_path: Path, interface: str) -> None:
    _write_scene(tmp_path)
    if interface == "cli":
        commands = [
            ["compile", "scene.json", "--out", "compiled"],
            ["run", "--compiled", "compiled", "--out", "run", "--duration", "1", "--telemetry-hz", "120"],
            ["export", "run", "--reference", "world", "--out", "exported"],
            ["score", "run", "--scene", "scene.json", "--reconstruction", "world", "--out", "score.json"],
            ["inspect", "run"],
        ]
        for command in commands:
            process = subprocess.run([sys.executable, "-m", "openbeam_v2e", *command],
                                     cwd=tmp_path, capture_output=True, text=True, timeout=60)
            assert process.returncode == 0, (command, process.stdout, process.stderr)
            assert json.loads(process.stdout).get("ok", True)
    else:
        calls = [
            ("compile", {"scene": "scene.json", "out": "compiled"}),
            ("run", {"compiled": "compiled", "out": "run", "duration_s": 1.0, "telemetry_hz": 120.0}),
            ("export", {"run": "run", "reference": "world", "out": "exported"}),
            ("score", {"run": "run", "scene": "scene.json", "reconstruction": "world", "out": "score.json"}),
            ("inspect", {"path": "run"}),
        ]
        messages = [
            {"jsonrpc": "2.0", "id": 1, "method": "initialize",
             "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                        "clientInfo": {"name": "offline-gate", "version": "1"}}},
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        ]
        messages.extend({"jsonrpc": "2.0", "id": i, "method": "tools/call",
                         "params": {"name": name, "arguments": args}}
                        for i, (name, args) in enumerate(calls, 3))
        process = subprocess.run([sys.executable, "-m", "openbeam_v2e", "mcp", "--root", str(tmp_path)],
                                 input="".join(json.dumps(m) + "\n" for m in messages),
                                 cwd=tmp_path, capture_output=True, text=True, timeout=90)
        assert process.returncode == 0, process.stderr
        answers = [json.loads(line) for line in process.stdout.splitlines()]
        assert [a["id"] for a in answers] == list(range(1, 8))
        assert len(answers[1]["result"]["tools"]) == 7
        for answer in answers[2:]:
            assert "error" not in answer, answer
            assert not answer["result"]["isError"], answer
        assert answers[3]["result"]["structuredContent"]["claim_class"] == "test_double"

    report = json.loads((tmp_path / "score.json").read_text())
    assert report["claim"]["class"] == "test_double" and report["claim"]["valid"]
    assert report["targets"][0]["pass"]
    assert report["phases"][0]["objects"]["parcel"]["position_rms_m"] < 0.05
    assert (tmp_path / "exported" / "v2e" / "world.json").is_file()


@pytest.mark.slow
@pytest.mark.parametrize("family", ["barrier_crash", "ramp_jump", "rollover", "drop_and_bounce"])
def test_generated_scene_through_the_general_compiler(tmp_path: Path, family: str) -> None:
    """Do not substitute bench.compile_case for the general scene/package compiler.

    This is a plumbing check using oracle worlds, not a reconstruction accuracy
    test. The separate benchmark reports reconstruction and physical target errors.
    """
    from openbeam_v2e.agents.tools import AgentContext, call_tool
    from openbeam_v2e.bench import families, synth
    from openbeam_v2e.world import read_world

    case = families.FAMILIES[family].default_case()
    generated = synth.generate_case(case, tmp_path / "cases",
                                    synth.RenderSettings(width=64, height=36, noise_sigma=0.0, prompts=2),
                                    physics_dt_s=0.004)
    ctx = AgentContext(root=tmp_path, connectors={"fake": {"physics_dt_s": 0.004}})
    case_path = generated.directory.relative_to(tmp_path).as_posix()
    programs = generated.directory / "programs.json"
    assert json.loads(programs.read_text()) == synth.programs_for(case)
    compiled = call_tool(ctx, "compile", {"scene": f"{case_path}/scene.json",
                                         "templates": f"{case_path}/templates.json", "out": "compiled"})
    assert compiled["ok"], compiled
    request = json.loads((tmp_path / "compiled" / "package.json").read_text())
    assert request["schema"] == "v2e.package/1"
    assert all(s["at_rest"] and s["initial_velocity_mps"] == 0 for s in request["spawns"])
    run = call_tool(ctx, "run", {"compiled": "compiled", "programs": f"{case_path}/programs.json",
                                  "out": "run", "duration_s": compiled["t0_s"] + case.frames / case.fps + 0.1,
                                  "telemetry_hz": 60.0})
    assert run["ok"] and run["valid"] and run["claim_class"] == "test_double", run
    exported = call_tool(ctx, "export", {"run": "run", "reference": f"{case_path}/world", "out": "exported"})
    assert exported["ok"], exported
    assert read_world(tmp_path / "exported").frame_count == case.frames
    scored = call_tool(ctx, "score", {"run": "run", "scene": f"{case_path}/scene.json",
                                      "reconstruction": f"{case_path}/world", "out": "score.json"})
    assert scored["ok"], scored
    assert scored["score"]["claim"]["class"] == "test_double"
    audit = call_tool(ctx, "inspect", {"path": "run"})
    assert audit["audit_replay"]["audit"] == "clean", audit
