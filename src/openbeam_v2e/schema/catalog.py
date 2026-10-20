# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""``v2e.catalog/1``: the content a connector can load, as the compiler sees it.

A connector builds its catalog locally from the engine install it talks to (catalogs are
never redistributed). Each entry is a piece of stock content with measured dimensions; the
catalog also says which generated templates the connector can load and the engine's actor
limits. Example::

    {"schema": "v2e.catalog/1", "backend": "fake", "engine": "fake 0",
     "generated_templates": ["vehicle.generic", "prop.box", "prop.cylinder", "prop.sphere",
                             "prop.mesh", "static.mesh"],
     "limits": {"max_seatable_vehicles": null, "max_controlled_actors": 4},
     "entries": [{"id": "hatch_a", "kind": "vehicle", "seatable": true,
                  "dims_m": {"length": 4.0, "width": 2.0, "height": 1.5, "wheelbase": 2.5}}]}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from openbeam_v2e.schema import jsonio
from openbeam_v2e.schema.common import defs
from openbeam_v2e.schema.validate import Issue, SchemaError, validate

__all__ = ["SCHEMA_ID", "CATALOG_SCHEMA", "CatalogEntry", "ContentCatalog", "load_catalog", "validate_catalog"]

SCHEMA_ID = "v2e.catalog/1"

CATALOG_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://github.com/OpenBeamWorld/v2e/schemas/v2e.catalog-1.json",
    "title": "V2E content catalog (v2e.catalog/1)",
    "type": "object",
    "required": ["schema", "backend", "entries"],
    "properties": {
        "schema": {"const": SCHEMA_ID},
        "backend": {"type": "string", "minLength": 1},
        "engine": {"type": "string"},
        "built": {"type": "string"},
        "generated_templates": {"type": "array", "items": {"type": "string"}, "uniqueItems": True},
        "limits": {
            "type": "object",
            "properties": {
                "max_seatable_vehicles": {"type": ["integer", "null"], "minimum": 0},
                "max_controlled_actors": {"type": ["integer", "null"], "minimum": 0},
            },
            "additionalProperties": False,
        },
        "entries": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "kind", "dims_m"],
                "properties": {
                    "id": {"type": "string", "minLength": 1},
                    "kind": {"enum": ["vehicle", "prop"]},
                    "name": {"type": "string"},
                    "dims_m": {
                        "type": "object",
                        "required": ["length", "width", "height"],
                        "additionalProperties": {"type": "number", "exclusiveMinimum": 0},
                    },
                    "bbox_m": {
                        "type": "object",
                        "required": ["min", "max"],
                        "properties": {"min": {"$ref": "#/$defs/vec3"}, "max": {"$ref": "#/$defs/vec3"}},
                        "additionalProperties": False,
                    },
                    "mass_kg": {"type": "number", "exclusiveMinimum": 0},
                    "seatable": {"type": "boolean"},
                    "content_version": {"type": "string"},
                    "licence": {"type": "string"},
                },
                "additionalProperties": False,
            },
        },
    },
    "additionalProperties": False,
    "$defs": defs(),
}


@dataclass(frozen=True)
class CatalogEntry:
    id: str
    kind: str
    dims_m: Mapping[str, float]
    name: str = ""
    seatable: bool = False
    mass_kg: float | None = None
    content_version: str = ""
    bbox_min: tuple[float, float, float] | None = None
    bbox_max: tuple[float, float, float] | None = None
    licence: str = ""

    @classmethod
    def parse(cls, raw: Mapping[str, Any]) -> "CatalogEntry":
        bbox = raw.get("bbox_m")
        return cls(raw["id"], raw["kind"], dict(raw["dims_m"]), raw.get("name", ""), bool(raw.get("seatable", False)),
                   raw.get("mass_kg"), raw.get("content_version", ""),
                   tuple(bbox["min"]) if bbox else None, tuple(bbox["max"]) if bbox else None, raw.get("licence", ""))

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {"id": self.id, "kind": self.kind}
        if self.name:
            out["name"] = self.name
        out["dims_m"] = dict(self.dims_m)
        if self.bbox_min is not None and self.bbox_max is not None:
            out["bbox_m"] = {"min": list(self.bbox_min), "max": list(self.bbox_max)}
        if self.mass_kg is not None:
            out["mass_kg"] = self.mass_kg
        out["seatable"] = self.seatable
        if self.content_version:
            out["content_version"] = self.content_version
        if self.licence:
            out["licence"] = self.licence
        return out


@dataclass(frozen=True)
class ContentCatalog:
    backend: str
    entries: tuple[CatalogEntry, ...] = ()
    engine: str = ""
    generated_templates: tuple[str, ...] = ("vehicle.generic", "prop.box", "prop.cylinder", "prop.sphere",
                                            "prop.mesh", "static.mesh")
    max_seatable_vehicles: int | None = None
    max_controlled_actors: int | None = None
    built: str = ""
    extra: Mapping[str, Any] = field(default_factory=dict)

    def vehicles(self) -> tuple[CatalogEntry, ...]:
        return tuple(e for e in self.entries if e.kind == "vehicle")

    def entry(self, content_id: str) -> CatalogEntry:
        for entry in self.entries:
            if entry.id == content_id:
                return entry
        raise KeyError(f"catalog has no entry {content_id!r}")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any], check: bool = True) -> "ContentCatalog":
        if check:
            issues = validate_catalog(raw)
            if issues:
                raise SchemaError(issues, "catalog")
        limits = raw.get("limits", {})
        kwargs: dict[str, Any] = {}
        if "generated_templates" in raw:
            kwargs["generated_templates"] = tuple(raw["generated_templates"])
        return cls(raw["backend"], tuple(CatalogEntry.parse(e) for e in raw["entries"]), raw.get("engine", ""),
                   max_seatable_vehicles=limits.get("max_seatable_vehicles"),
                   max_controlled_actors=limits.get("max_controlled_actors"), built=raw.get("built", ""), **kwargs)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"schema": SCHEMA_ID, "backend": self.backend}
        if self.engine:
            out["engine"] = self.engine
        if self.built:
            out["built"] = self.built
        out["generated_templates"] = list(self.generated_templates)
        out["limits"] = {"max_seatable_vehicles": self.max_seatable_vehicles,
                         "max_controlled_actors": self.max_controlled_actors}
        out["entries"] = [e.to_json() for e in self.entries]
        return out

    def sha256(self) -> str:
        return jsonio.canonical_sha256(self.to_dict())


def validate_catalog(raw: Any) -> list[Issue]:
    issues = validate(raw, CATALOG_SCHEMA)
    if issues or not isinstance(raw, Mapping):
        return issues
    seen: set[str] = set()
    for index, entry in enumerate(raw["entries"]):
        if entry["id"] in seen:
            issues.append(Issue(f"/entries/{index}/id", f"entry id {entry['id']!r} is used twice"))
        seen.add(entry["id"])
    return issues


def load_catalog(path: str | Path) -> ContentCatalog:
    return ContentCatalog.from_dict(jsonio.load(path))
