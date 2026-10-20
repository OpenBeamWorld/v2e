# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""Document formats of V2E: JSON Schemas, validators and dataclasses.

Every V2E document names its format in a ``schema`` member. This package holds the JSON
Schema of each format (draft 2020-12 vocabulary, so any conforming validator can check
them), a dependency-free validator for the subset the schemas use, cross-reference checks
the schema language cannot express, and dataclasses for the formats the pipeline reads:

* ``v2e.scene/1`` (:mod:`.scene`) and ``v2e.templates/1`` (:mod:`.templates`);
* ``v2e-runlog/1`` (:mod:`.runlog`), the row format every connector writes;
* the world-package sidecars ``v2e.world/1``, ``v2e.camera_track/1``, ``v2e.objects/1``,
  ``v2e.uncertainty/1`` and ``v2e.outlines/1`` (:mod:`.world`);
* ``v2e.catalog/1`` (:mod:`.catalog`), a connector's content catalog;
* ``v2e.package/1`` (:mod:`.package`), the compiler's engine-neutral package request;
* ``v2e.score/1`` and ``v2e.score_repeat/1`` (:mod:`.score`).

:func:`check_document` validates any of them by its ``schema`` member, and
:func:`export_json_schemas` writes the schemas as ``.json`` files for other tools.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping

from openbeam_v2e.schema import catalog, package, runlog, scene, score, templates, world
from openbeam_v2e.schema import jsonio
from openbeam_v2e.schema.common import Measured, Pose, PoseRef
from openbeam_v2e.schema.validate import Issue, SchemaError, validate

__all__ = [
    "SCHEMAS",
    "Issue",
    "SchemaError",
    "Measured",
    "Pose",
    "PoseRef",
    "check_document",
    "check_file",
    "validate_document",
    "export_json_schemas",
]

#: Format id -> JSON Schema.
SCHEMAS: dict[str, Mapping[str, Any]] = {
    scene.SCHEMA_ID: scene.SCENE_SCHEMA,
    templates.SCHEMA_ID: templates.TEMPLATES_SCHEMA,
    runlog.SCHEMA_ID + "#header": runlog.HEADER_SCHEMA,
    runlog.SCHEMA_ID + "#row": runlog.ROW_SCHEMA,
    world.WORLD_SCHEMA_ID: world.WORLD_SCHEMA,
    world.CAMERA_TRACK_SCHEMA_ID: world.CAMERA_TRACK_SCHEMA,
    world.OBJECTS_SCHEMA_ID: world.OBJECTS_SCHEMA,
    world.UNCERTAINTY_SCHEMA_ID: world.UNCERTAINTY_SCHEMA,
    world.OUTLINES_SCHEMA_ID: world.OUTLINES_SCHEMA,
    catalog.SCHEMA_ID: catalog.CATALOG_SCHEMA,
    package.SCHEMA_ID: package.PACKAGE_SCHEMA,
    score.SCORE_SCHEMA_ID: score.SCORE_SCHEMA,
    score.REPEAT_SCHEMA_ID: score.REPEAT_SCHEMA,
}

#: Format id -> validator including cross-reference checks, where one exists.
_SEMANTIC: dict[str, Callable[[Any], list[Issue]]] = {
    scene.SCHEMA_ID: scene.validate_scene,
    templates.SCHEMA_ID: templates.validate_templates,
    catalog.SCHEMA_ID: catalog.validate_catalog,
}


def validate_document(document: Any) -> list[Issue]:
    """Issues of any V2E JSON document, dispatched on its ``schema`` member."""
    if not isinstance(document, Mapping) or not isinstance(document.get("schema"), str):
        return [Issue("/schema", "a V2E document needs a string 'schema' member")]
    schema_id = document["schema"]
    if schema_id == runlog.SCHEMA_ID:
        return validate(document, runlog.HEADER_SCHEMA)
    if schema_id in _SEMANTIC:
        return _SEMANTIC[schema_id](document)
    if schema_id in SCHEMAS:
        return validate(document, SCHEMAS[schema_id])
    known = ", ".join(sorted(k for k in SCHEMAS if "#" not in k))
    return [Issue("/schema", f"unknown format {schema_id!r}; known formats: {known}")]


def check_document(document: Any, what: str = "document") -> None:
    """Raise :class:`SchemaError` when :func:`validate_document` finds issues."""
    issues = validate_document(document)
    if issues:
        raise SchemaError(issues, what)


def check_file(path: str | Path) -> list[Issue]:
    """Validate a JSON document or a ``v2e-runlog/1`` JSON Lines file."""
    path = Path(path)
    if path.suffix == ".jsonl":
        try:
            runlog.read_runlog(path)
        except SchemaError as exc:
            return exc.issues
        return []
    try:
        document = jsonio.load(path)
    except (jsonio.JsonError, OSError) as exc:
        return [Issue("/", str(exc))]
    return validate_document(document)


def export_json_schemas(directory: str | Path) -> list[Path]:
    """Write every schema as ``<format>.json`` (``/`` and ``#`` become ``-``)."""
    out = []
    for schema_id, schema in SCHEMAS.items():
        name = schema_id.replace("/", "-").replace("#", "-") + ".json"
        out.append(jsonio.dump(schema, Path(directory) / name))
    return out

