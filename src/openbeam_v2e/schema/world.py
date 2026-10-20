# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""Schemas of the V2E sidecars stored next to a world package.

A V2E world package is a world package in the 4DCodeBench layout (``camera.json``,
``meshes/``, ``dynamics/``, optional ``render.mp4``) plus a ``v2e/`` directory:

``v2e/world.json`` (``v2e.world/1``)
    Frame rate, frame count, image size, what produced the package (a reconstruction, a run
    export or synthetic truth) and the provenance of every perception model used.
``v2e/camera_track.json`` (``v2e.camera_track/1``)
    The per-frame camera of a fixed-position camera that may pan, tilt and zoom: one position,
    and per frame the intrinsics, the camera-to-world rotation and radial distortion. The
    base layout's ``camera.json`` can hold only one static camera; it keeps frame 0.
``v2e/objects.json`` (``v2e.objects/1``)
    Every dynamic object: class, symmetry class, measured dimensions with sigma, the mesh and
    dynamics archives that show it, its pose-track file, visibility intervals and annotation
    provenance.
``v2e/tracks/<object>.npz``
    Rigid pose track per object, one row per frame (``position`` (F, 3), ``quat_xyzw``
    (F, 4), ``visible`` (F,), ``sigma_position_m`` (F,), ``sigma_attitude_rad`` (F,)); rows the
    reconstruction does not know are NaN.
``v2e/uncertainty.json`` (``v2e.uncertainty/1``)
    Camera sigma per frame, event intervals, and fit residuals kept apart as ``fitted`` and
    ``held_out``.
``v2e/outlines.json`` (``v2e.outlines/1``, optional)
    Observed object outlines (pixel polygons per frame), used by silhouette scoring.
"""

from __future__ import annotations

from typing import Any

from openbeam_v2e.schema.common import defs

__all__ = [
    "WORLD_SCHEMA_ID",
    "CAMERA_TRACK_SCHEMA_ID",
    "OBJECTS_SCHEMA_ID",
    "UNCERTAINTY_SCHEMA_ID",
    "OUTLINES_SCHEMA_ID",
    "WORLD_SCHEMA",
    "CAMERA_TRACK_SCHEMA",
    "OBJECTS_SCHEMA",
    "UNCERTAINTY_SCHEMA",
    "OUTLINES_SCHEMA",
    "PACKAGE_KINDS",
]

WORLD_SCHEMA_ID = "v2e.world/1"
CAMERA_TRACK_SCHEMA_ID = "v2e.camera_track/1"
OBJECTS_SCHEMA_ID = "v2e.objects/1"
UNCERTAINTY_SCHEMA_ID = "v2e.uncertainty/1"
OUTLINES_SCHEMA_ID = "v2e.outlines/1"
PACKAGE_KINDS = ("reconstruction", "run_export", "synthetic_truth", "imported")


def _doc(schema_id: str, title: str, properties: dict[str, Any], required: list[str],
         extra_defs: dict[str, Any] | None = None) -> dict[str, Any]:
    d = defs()
    d.update(extra_defs or {})
    slug = schema_id.replace("/", "-")
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": f"https://github.com/OpenBeamWorld/v2e/schemas/{slug}.json",
        "title": title,
        "type": "object",
        "required": ["schema", *required],
        "properties": {"schema": {"const": schema_id}, **properties},
        "additionalProperties": False,
        "$defs": d,
    }


_PROVENANCE = {
    "type": "object",
    "required": ["id"],
    "properties": {
        "id": {"type": "string", "minLength": 1},
        "version": {"type": "string"},
        "task": {"type": "string"},
        "weights_sha256": {"type": ["string", "null"]},
        "parameters": {"type": "object"},
        "licence": {"type": "string"},
    },
    "additionalProperties": False,
}

WORLD_SCHEMA = _doc(
    WORLD_SCHEMA_ID,
    "V2E world package manifest (v2e.world/1)",
    {
        "kind": {"enum": list(PACKAGE_KINDS)},
        "fps": {"type": "number", "exclusiveMinimum": 0},
        "frame_count": {"type": "integer", "minimum": 1},
        "image_size": {"type": "array", "items": {"type": "integer", "minimum": 1}, "minItems": 2, "maxItems": 2},
        "frame": {"const": "z_up"},
        "units": {"const": "m"},
        "source_video": {
            "type": "object",
            "properties": {"name": {"type": "string"}, "sha256": {"type": "string"}, "licence": {"type": "string"}},
            "additionalProperties": False,
        },
        "models": {"type": "array", "items": {"$ref": "#/$defs/provenance"}},
        "producer": {"type": "object"},
        "notes": {"type": "string"},
    },
    ["kind", "fps", "frame_count"],
    {"provenance": _PROVENANCE},
)

_COLUMN_F = {"type": "array", "items": {"type": "number"}}
CAMERA_TRACK_SCHEMA = _doc(
    CAMERA_TRACK_SCHEMA_ID,
    "V2E camera track (v2e.camera_track/1)",
    {
        "model": {"const": "fixed_position_ptz"},
        "image_size": {"type": "array", "items": {"type": "integer", "minimum": 1}, "minItems": 2, "maxItems": 2},
        "position_m": {"$ref": "#/$defs/vec3"},
        "fx": _COLUMN_F,
        "fy": _COLUMN_F,
        "cx": _COLUMN_F,
        "cy": _COLUMN_F,
        "quat_xyzw": {"type": "array", "items": {"$ref": "#/$defs/quat_xyzw"}},
        "k1": _COLUMN_F,
        "k2": _COLUMN_F,
    },
    ["model", "image_size", "position_m", "fx", "fy", "cx", "cy", "quat_xyzw"],
)

OBJECTS_SCHEMA = _doc(
    OBJECTS_SCHEMA_ID,
    "V2E dynamic objects (v2e.objects/1)",
    {
        "objects": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "class", "mesh_id", "track"],
                "properties": {
                    "id": {"$ref": "#/$defs/id"},
                    "class": {"$ref": "#/$defs/object_class"},
                    "symmetry": {"$ref": "#/$defs/symmetry"},
                    "dims_m": {"$ref": "#/$defs/dims"},
                    "mass_kg": {"$ref": "#/$defs/measured"},
                    "mesh_id": {"type": "integer", "minimum": 1},
                    "dynamics": {"type": "string"},
                    "track": {"type": "string", "pattern": r"^v2e/tracks/[A-Za-z0-9_.\-]+\.npz$"},
                    "hull_frame": {"type": "integer", "minimum": 0},
                    "visibility": {"type": "array", "items": {"$ref": "#/$defs/frame_interval"}},
                    "annotation": {"type": "object"},
                },
                "additionalProperties": False,
            },
        },
        "statics": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["mesh_id"],
                "properties": {
                    "mesh_id": {"type": "integer", "minimum": 1},
                    "name": {"type": "string"},
                    "role": {"enum": ["collision", "drivable", "visual"]},
                    "sigma_m": {"type": "number", "minimum": 0},
                    "observed": {"type": "boolean"},
                },
                "additionalProperties": False,
            },
        },
    },
    ["objects"],
)

UNCERTAINTY_SCHEMA = _doc(
    UNCERTAINTY_SCHEMA_ID,
    "V2E uncertainty and events (v2e.uncertainty/1)",
    {
        "camera": {
            "type": "object",
            "properties": {
                "sigma_px": {"anyOf": [{"type": "number", "minimum": 0}, _COLUMN_F]},
                "reprojection_rms_px": {"type": "number", "minimum": 0},
            },
            "additionalProperties": False,
        },
        "events": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["name", "frame"],
                "properties": {
                    "name": {"$ref": "#/$defs/id"},
                    "frame": {"$ref": "#/$defs/frame_interval"},
                    "kind": {"enum": ["contact", "separation", "rest", "custom"]},
                    "actors": {"type": "array", "items": {"$ref": "#/$defs/id"}},
                    "evidence": {"type": "string"},
                },
                "additionalProperties": False,
            },
        },
        "residuals": {
            "type": "object",
            "properties": {
                "fitted": {"type": "object", "additionalProperties": {"type": "number"}},
                "held_out": {"type": "object", "additionalProperties": {"type": "number"}},
            },
            "additionalProperties": False,
        },
    },
    [],
)

OUTLINES_SCHEMA = _doc(
    OUTLINES_SCHEMA_ID,
    "V2E observed outlines (v2e.outlines/1)",
    {
        "image_size": {"type": "array", "items": {"type": "integer", "minimum": 1}, "minItems": 2, "maxItems": 2},
        "objects": {
            "type": "object",
            "propertyNames": {"$ref": "#/$defs/id"},
            "additionalProperties": {
                "type": "object",
                "propertyNames": {"pattern": r"^[0-9]+$"},
                "additionalProperties": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
                        "minItems": 3,
                    },
                },
            },
        },
        "source": {"type": "string"},
    },
    ["image_size", "objects"],
)
