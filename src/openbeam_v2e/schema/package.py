# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""``v2e.package/1``: the engine-neutral package request the compiler writes.

A package request says everything a connector's dialect needs to build engine packages,
and nothing engine-specific:

* ``level``: collision geometry in the Z-up level frame, one entry per static mesh of the world
  package plus any synthesised run-up ground (marked ``observed: false``; no score uses it),
  each written as a Wavefront OBJ file next to the request;
* ``content``: one content-origin record per dynamic actor: the template, whether the
  content is ``stock`` (with the catalog id and per-dimension residuals) or ``generated``
  (with generator parameters, each with sigma, status and source);
* ``spawns``: one pose per actor at which it spawns **at rest**; the schema has no field for
  an initial velocity;
* ``controllers``: what each controlled actor's program must achieve (for ``drive_in``: the
  run-up path, the speed profile and the frame-0 pose it must reach at t0);
* ``timeline``: ``t0_s``, the simulated time that corresponds to video frame 0.
"""

from __future__ import annotations

from typing import Any

from openbeam_v2e.schema.common import defs

__all__ = ["SCHEMA_ID", "PACKAGE_SCHEMA"]

SCHEMA_ID = "v2e.package/1"

_BOUNDS = {
    "type": "object",
    "required": ["min", "max"],
    "properties": {"min": {"$ref": "#/$defs/vec3"}, "max": {"$ref": "#/$defs/vec3"}},
    "additionalProperties": False,
}

_PARAMETER = {
    "type": "object",
    "required": ["value", "status", "source"],
    "properties": {
        "value": {"type": ["number", "string"]},
        "sigma": {"type": ["number", "null"], "minimum": 0},
        "status": {"enum": ["measured", "assumed", "fitted", "catalog"]},
        "source": {"type": "string"},
        "reason": {"type": "string"},
    },
    "additionalProperties": False,
}

PACKAGE_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://github.com/OpenBeamWorld/v2e/schemas/v2e.package-1.json",
    "title": "V2E engine package request (v2e.package/1)",
    "type": "object",
    "required": ["schema", "inputs", "timeline", "level", "content", "spawns", "controllers"],
    "properties": {
        "schema": {"const": SCHEMA_ID},
        "name": {"type": "string"},
        "compiler_version": {"type": "string"},
        "inputs": {
            "type": "object",
            "required": ["sha256", "backend"],
            "properties": {
                "sha256": {
                    "type": "object",
                    "required": ["scene", "templates", "catalog"],
                    "properties": {
                        "scene": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                        "templates": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                        "catalog": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                    },
                    "additionalProperties": False,
                },
                "backend": {"type": "string"},
                "engine": {"type": "string"},
                "world": {"type": "string"},
            },
            "additionalProperties": False,
        },
        "timeline": {
            "type": "object",
            "required": ["t0_s", "fps", "frame_count"],
            "properties": {
                "t0_s": {"type": "number", "minimum": 0},
                "fps": {"type": "number", "exclusiveMinimum": 0},
                "frame_count": {"type": "integer", "minimum": 1},
                "settle_s": {"type": "number", "minimum": 0},
            },
            "additionalProperties": False,
        },
        "level": {
            "type": "object",
            "required": ["frame", "collision"],
            "properties": {
                "frame": {"const": "z_up"},
                "bounds_m": _BOUNDS,
                "collision": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["id", "source", "observed", "file", "vertex_count", "face_count", "sha256"],
                        "properties": {
                            "id": {"$ref": "#/$defs/id"},
                            "source": {"enum": ["world", "runup"]},
                            "actor": {"type": ["string", "null"]},
                            "mesh_id": {"type": ["integer", "null"]},
                            "role": {"enum": ["collision", "drivable", "visual"]},
                            "observed": {"type": "boolean"},
                            "file": {"type": "string", "pattern": r"^level/[A-Za-z0-9_.\-]+\.obj$"},
                            "vertex_count": {"type": "integer", "minimum": 0},
                            "face_count": {"type": "integer", "minimum": 0},
                            "bounds_m": _BOUNDS,
                            "sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                        },
                        "additionalProperties": False,
                    },
                },
            },
            "additionalProperties": False,
        },
        "content": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["actor", "template", "kind", "parameters", "params_sha256"],
                "properties": {
                    "actor": {"$ref": "#/$defs/id"},
                    "class": {"$ref": "#/$defs/object_class"},
                    "template": {"type": "string"},
                    "kind": {"enum": ["generated", "stock"]},
                    "content_id": {"type": "string"},
                    "content_version": {"type": "string"},
                    "generator": {"type": "string"},
                    "generator_version": {"type": "string"},
                    "seatable": {"type": "boolean"},
                    "parameters": {"type": "object", "additionalProperties": _PARAMETER},
                    "residuals": {
                        "type": "object",
                        "additionalProperties": {
                            "type": "object",
                            "required": ["catalog", "measured", "relative"],
                            "properties": {"catalog": {"type": "number"}, "measured": {"type": "number"},
                                           "relative": {"type": "number"}, "tolerance": {"type": "number"}},
                            "additionalProperties": False,
                        },
                    },
                    "fallback_from": {"type": "string"},
                    "params_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                },
                "additionalProperties": False,
            },
        },
        "spawns": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["actor", "mode", "pose", "at_rest", "initial_velocity_mps"],
                "properties": {
                    "actor": {"$ref": "#/$defs/id"},
                    "mode": {"enum": ["at_rest", "drive_in", "moving"]},
                    "pose": {"$ref": "#/$defs/pose"},
                    "at_rest": {"const": True},
                    "initial_velocity_mps": {"const": 0},
                    "frame0_pose": {"$ref": "#/$defs/pose"},
                    "ground_offset_m": {"type": "number"},
                    "launcher": {"type": "object"},
                },
                "additionalProperties": False,
            },
        },
        "controllers": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["actor", "control", "model"],
                "properties": {
                    "actor": {"$ref": "#/$defs/id"},
                    "control": {"type": "string"},
                    "model": {"enum": ["time_indexed_tracker", "hold", "passive"]},
                    "reference": {
                        "type": "object",
                        "properties": {
                            "path_m": {"type": "array", "items": {"$ref": "#/$defs/vec3"}},
                            "s_m": {"type": "array", "items": {"type": "number"}},
                            "t_s": {"type": "array", "items": {"type": "number"}},
                            "speed_mps": {"type": "array", "items": {"type": "number", "minimum": 0}},
                            "frame0_s_m": {"type": "number"},
                            "t0_s": {"type": "number"},
                            "accel_mps2": {"type": "number", "exclusiveMinimum": 0},
                            "shape": {"enum": ["straight", "arc"]},
                        },
                        "additionalProperties": False,
                    },
                },
                "additionalProperties": False,
            },
        },
        "requires": {"type": "object"},
        "notes": {"type": "array", "items": {"type": "string"}},
    },
    "additionalProperties": False,
    "$defs": defs(),
}
