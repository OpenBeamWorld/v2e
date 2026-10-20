# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""Scenes, programs and recorded OpenBeam cassettes shared by the connector and agent tests.

All values are round numbers chosen for the tests; none comes from a measurement. The OpenBeam
cassettes are hand-written responses shaped like the documented tool results.
"""

from __future__ import annotations

import copy
from typing import Any

from openbeam_v2e.backends.base import (
    CompiledScene,
    LevelRef,
    ObjectRef,
    Package,
    Pose,
    SpawnSpec,
)
from openbeam_v2e.backends.openbeam.backend import PHYSICS_DT_S, zup_to_yup


def at_rest(x: float, y: float, z: float, yaw_deg: float = 0.0) -> dict[str, Any]:
    quat = Pose.from_yaw((x, y, z), yaw_deg).quat_xyzw
    return {"mode": "at_rest", "pose": {"position_m": [x, y, z], "quat_xyzw": list(quat)}}


def drop_scene(height_m: float = 2.0, shape: str = "prop.box") -> dict[str, Any]:
    dims = {"diameter": [0.3, 0.01]} if shape == "prop.sphere" else {
        "length": [0.5, 0.01], "width": [0.5, 0.01], "height": [0.5, 0.01]}
    half = 0.15 if shape == "prop.sphere" else 0.25
    return {
        "schema": "v2e.scene/1", "world": "world/",
        "actors": [{"id": "obj", "class": shape, "dims_m": dims, "control": "passive",
                    "initial_state": at_rest(0.0, 0.0, height_m + half)}],
        "requires": {"effects": ["contact", "gravity"], "observables": ["position", "attitude"]},
    }


def car_scene(extra_actors: list[dict] | None = None, requires: dict | None = None) -> dict[str, Any]:
    return {
        "schema": "v2e.scene/1", "world": "world/",
        "actors": [{"id": "car_1", "class": "vehicle.car", "seatable": True, "control": "driver",
                    "dims_m": {"length": [4.5, 0.05], "width": [1.8, 0.03], "height": [1.45, 0.03]},
                    "initial_state": at_rest(0.0, 0.0, 0.725)}] + list(extra_actors or []),
        "requires": requires or {"effects": ["drive", "brake", "steer", "contact"],
                                 "observables": ["position", "attitude"]},
    }


def schedule(actor: str, entries: list[dict]) -> dict[str, Any]:
    return {"schema": "v2e.program/1", "actor": actor, "model": "schedule", "params": {"schedule": entries}}


# --------------------------------------------------------------------------- OpenBeam cassettes

SHA_LEVEL = "1" * 64
SHA_CRATE = "2" * 64
SHA_ANCHOR = "3" * 64
CRATE_BBOX = ((-0.25, -0.25, -0.25), (0.25, 0.25, 0.25))
ANCHOR_BBOX = ((-2.25, -0.9, -0.725), (2.25, 0.9, 0.725))
CRATE_POSE = Pose((0.0, 0.0, 2.25))
ANCHOR_POSE = Pose((40.0, 40.0, 0.725))


def openbeam_scene() -> dict[str, Any]:
    return {
        "schema": "v2e.scene/1", "world": "world/",
        "actors": [
            {"id": "crate", "class": "prop.box", "control": "passive", "seatable": False,
             "initial_state": {"mode": "at_rest", "pose": {"position_m": list(CRATE_POSE.position_m),
                                                           "quat_xyzw": [0, 0, 0, 1]}}},
            {"id": "anchor", "class": "vehicle.car", "control": "passive", "seatable": True,
             "initial_state": {"mode": "at_rest", "pose": {"position_m": list(ANCHOR_POSE.position_m),
                                                           "quat_xyzw": [0, 0, 0, 1]}}},
        ],
        "requires": {"effects": ["contact", "gravity"], "observables": ["position"]},
    }


def openbeam_compiled() -> CompiledScene:
    return CompiledScene(
        connector="openbeam", dialect="stock_jbeam/openbeam", level=LevelRef(package="level", filename="v2e-level.zip"),
        packages=(
            Package("level", "level", SHA_LEVEL, url="https://downloads.example.com/v2e-level.zip?X-Sig=secret",
                    filename="v2e-level.zip"),
            Package("crate", "prop", SHA_CRATE, url="https://downloads.example.com/crate.zip?X-Sig=secret",
                    filename="v2e-crate.zip"),
            Package("anchor", "vehicle", SHA_ANCHOR, url="https://downloads.example.com/anchor.zip?X-Sig=secret",
                    filename="v2e-anchor.zip"),
        ),
        objects=(
            ObjectRef("crate", "prop", package="crate", dims_m=(0.5, 0.5, 0.5), mass_kg=50.0,
                      bbox_min_m=CRATE_BBOX[0], bbox_max_m=CRATE_BBOX[1]),
            ObjectRef("anchor", "vehicle", package="anchor", seatable=True, dims_m=(4.5, 1.8, 1.45), mass_kg=1500.0,
                      bbox_min_m=ANCHOR_BBOX[0], bbox_max_m=ANCHOR_BBOX[1]),
        ),
        spawns=(SpawnSpec("anchor", "anchor", ANCHOR_POSE), SpawnSpec("crate", "crate", CRATE_POSE)),
    )


def _ok(data: dict) -> dict:
    return {"structuredContent": data, "isError": False}


def _engine_position(pose: Pose, bbox: tuple) -> list[float]:
    lo, hi = bbox
    offset = [(lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0, lo[2]]
    return zup_to_yup([pose.position_m[i] + offset[i] for i in range(3)])


def _actor(instance: int, file: str, position: list[float], velocity=(0.0, 0.0, 0.0), player=False) -> dict:
    return {"instance_id": instance, "file": file, "position": list(position), "velocity": list(velocity),
            "yaw": 0.0, "player": player, "robot": False, "asleep": False}


def openbeam_passive_cassette(duration_s: float = 0.1, telemetry_hz: float = 20.0,
                              remove_anchor: bool = False) -> list[dict]:
    """A recorded passive stepped run: install, load, spawn two actors, step, read poses."""
    anchor_pos = _engine_position(ANCHOR_POSE, ANCHOR_BBOX)
    crate_pos = _engine_position(CRATE_POSE, CRATE_BBOX)
    anchor = _actor(7, "v2e_anchor.jbeam", anchor_pos, player=True)
    crate = _actor(8, "v2e_crate.jbeam", crate_pos)
    calls: list[dict] = [
        {"tool": "list_mods", "arguments": {}, "result": _ok({"mods": [], "upload_directory": "uploads"})},
        {"tool": "upload_level",
         "arguments": {"sha256": SHA_LEVEL, "url": "https://downloads.example.com/v2e-level.zip?X-Sig=secret",
                       "filename": "v2e-level.zip"},
         "result": _ok({"level": {"filename": "v2e-level.zip", "levelIds": ["v2e_level"]}, "already_present": False,
                        "cache": {"refused": []}})},
        {"tool": "install_mod",
         "arguments": {"sha256": SHA_CRATE, "url": "https://downloads.example.com/crate.zip?X-Sig=secret",
                       "filename": "v2e-crate.zip"},
         "result": _ok({"mod": {"filename": "v2e-crate.zip"}, "vehicles": ["v2e_crate.jbeam"], "already_present": False,
                        "cache": {"refused": []}})},
        {"tool": "install_mod",
         "arguments": {"sha256": SHA_ANCHOR, "url": "https://downloads.example.com/anchor.zip?X-Sig=secret",
                       "filename": "v2e-anchor.zip"},
         "result": _ok({"mod": {"filename": "v2e-anchor.zip"}, "vehicles": ["v2e_anchor.jbeam"],
                        "already_present": False, "cache": {"refused": []}})},
        {"tool": "load_level", "arguments": {"wait_seconds": 180.0, "level_id": "v2e_level",
                                             "filename": "v2e-level.zip"},
         "result": _ok({"requested": "v2e_level", "loaded": True, "terrain": "v2e_level"})},
        {"tool": "actor_poses", "arguments": {}, "result": _ok({"actors": []})},
        {"tool": "spawn_vehicle", "arguments": {"file": "v2e_anchor.jbeam", "position": anchor_pos, "enter": True,
                                                "wait_seconds": 60.0},
         "result": _ok({"resolved": "v2e_anchor.jbeam", "spawned": True, "actor_count": 1})},
        {"tool": "actor_poses", "arguments": {}, "result": _ok({"actors": [anchor]})},
        {"tool": "spawn_vehicle", "arguments": {"file": "v2e_crate.jbeam", "position": crate_pos, "enter": False,
                                                "wait_seconds": 60.0},
         "result": _ok({"resolved": "v2e_crate.jbeam", "spawned": True, "actor_count": 2})},
        {"tool": "actor_poses", "arguments": {}, "result": _ok({"actors": [crate] if remove_anchor
                                                                else [anchor, crate]})},
        {"tool": "run_script", "match": "tool", "arguments": {}, "result": _ok({"return_code": 0, "succeeded": True})},
        {"tool": "sim_mode", "arguments": {"mode": "stepped", "idle_release_s": 120},
         "result": _ok({"mode": "stepped"})},
    ]
    per_row = max(1, int(round(1.0 / (telemetry_hz * PHYSICS_DT_S))))
    total = max(1, int(round(duration_s / PHYSICS_DT_S)))
    step = 0
    g = 9.80665
    while True:
        t = step * PHYSICS_DT_S
        fallen = copy.deepcopy(crate)
        fallen["position"][1] = crate_pos[1] - 0.5 * g * t * t      # Y is up in OpenBeam
        fallen["velocity"] = [0.0, -g * t, 0.0]
        actors = [fallen] if remove_anchor else [anchor, fallen]
        calls.append({"tool": "actor_poses", "arguments": {}, "result": _ok({"actors": actors})})
        if step >= total:
            break
        todo = min(per_row, total - step)
        calls.append({"tool": "sim_step", "arguments": {"steps": todo}, "result": _ok({"steps": todo, "wall_ms": 3})})
        step += todo
    calls += [
        {"tool": "sim_mode", "arguments": {"mode": "realtime"}, "result": _ok({"mode": "realtime"})},
        {"tool": "telemetry", "arguments": {}, "result": _ok({"name": "anchor", "nodes": 120, "beams": 400,
                                                              "broken_beams": 0})},
        {"tool": "run_script", "match": "tool", "arguments": {}, "result": _ok({"return_code": 0, "succeeded": True})},
    ]
    return calls
