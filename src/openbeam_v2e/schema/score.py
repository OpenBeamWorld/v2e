# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""``v2e.score/1`` and ``v2e.score_repeat/1``: score reports.

A score report compares one run (exported as a world package) with the reference world
package (the reconstruction, or synthetic truth). It holds per-phase, per-object metrics,
event timing, values at events, after-event measures, silhouette overlap, targets, the run's
claim class and a scalar objective for search. Every metric that has a reference sigma is
reported raw and as a z-score. ``v2e.score_repeat/1`` summarises the same scene over N runs:
median, MAD-based robust interval, IQR and failure fraction per metric.
"""

from __future__ import annotations

from typing import Any

__all__ = ["SCORE_SCHEMA_ID", "REPEAT_SCHEMA_ID", "CLAIM_CLASSES", "SCORE_SCHEMA", "REPEAT_SCHEMA"]

SCORE_SCHEMA_ID = "v2e.score/1"
REPEAT_SCHEMA_ID = "v2e.score_repeat/1"
CLAIM_CLASSES = ("native", "emulated", "scripted", "incomplete", "test_double", "unknown")

_NUMBER_OR_NULL = {"type": ["number", "null"]}

SCORE_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://github.com/OpenBeamWorld/v2e/schemas/v2e.score-1.json",
    "title": "V2E score report (v2e.score/1)",
    "type": "object",
    "required": ["schema", "scorer_version", "fps", "frame_count", "claim", "phases", "events", "objects",
                 "targets", "summary"],
    "properties": {
        "schema": {"const": SCORE_SCHEMA_ID},
        "scorer_version": {"type": "string"},
        "inputs": {"type": "object"},
        "fps": {"type": "number", "exclusiveMinimum": 0},
        "frame_count": {"type": "integer", "minimum": 1},
        "claim": {
            "type": "object",
            "required": ["class"],
            "properties": {
                "class": {"enum": list(CLAIM_CLASSES)},
                "provenance_signature": {"type": ["string", "null"]},
                "valid": {"type": ["boolean", "null"]},
                "effects": {"type": "array"},
                "content": {"type": "array"},
                "label": {"type": "string"},
            },
            "additionalProperties": False,
        },
        "alignment": {"type": "object"},
        "phases": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["name", "frames", "objects"],
                "properties": {
                    "name": {"type": "string"},
                    "frames": {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2},
                    "objects": {"type": "object"},
                },
                "additionalProperties": False,
            },
        },
        "events": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["name", "kind", "reference_frame", "run_frame", "delta_frames"],
                "properties": {
                    "name": {"type": "string"},
                    "kind": {"type": "string"},
                    "actors": {"type": "array", "items": {"type": "string"}},
                    "reference_frame": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
                    "reference_source": {"type": "string"},
                    "run_frame": _NUMBER_OR_NULL,
                    "delta_frames": _NUMBER_OR_NULL,
                    "delta_s": _NUMBER_OR_NULL,
                    "sigma_frames": _NUMBER_OR_NULL,
                    "z": _NUMBER_OR_NULL,
                    "within_sigma": {"type": ["boolean", "null"]},
                },
                "additionalProperties": False,
            },
        },
        "objects": {"type": "object"},
        "targets": {"type": "array", "items": {"type": "object"}},
        "summary": {
            "type": "object",
            "required": ["objective"],
            "properties": {"objective": {"type": "number", "minimum": 0}},
        },
        "warnings": {"type": "array", "items": {"type": "string"}},
    },
    "additionalProperties": False,
}

REPEAT_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://github.com/OpenBeamWorld/v2e/schemas/v2e.score_repeat-1.json",
    "title": "V2E repeat statistics (v2e.score_repeat/1)",
    "type": "object",
    "required": ["schema", "runs", "failures", "failure_fraction", "metrics"],
    "properties": {
        "schema": {"const": REPEAT_SCHEMA_ID},
        "runs": {"type": "integer", "minimum": 0},
        "failures": {"type": "integer", "minimum": 0},
        "failure_fraction": {"type": "number", "minimum": 0, "maximum": 1},
        "claim_classes": {"type": "array", "items": {"enum": list(CLAIM_CLASSES)}},
        "provenance_signatures": {"type": "array", "items": {"type": ["string", "null"]}},
        "metrics": {
            "type": "object",
            "additionalProperties": {
                "type": "object",
                "required": ["n", "median", "mad", "low", "high"],
                "properties": {
                    "n": {"type": "integer", "minimum": 0},
                    "median": _NUMBER_OR_NULL,
                    "mad": _NUMBER_OR_NULL,
                    "low": _NUMBER_OR_NULL,
                    "high": _NUMBER_OR_NULL,
                    "iqr": _NUMBER_OR_NULL,
                    "min": _NUMBER_OR_NULL,
                    "max": _NUMBER_OR_NULL,
                },
                "additionalProperties": False,
            },
        },
        "warnings": {"type": "array", "items": {"type": "string"}},
    },
    "additionalProperties": False,
}
