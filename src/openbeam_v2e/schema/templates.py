# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""``v2e.templates/1``: which authored content stands in for each reconstructed object.

Templates in the first release:

``vehicle.generic``
    A vehicle generated from the measured dimensions. The default for every ``vehicle.*``
    class, and the only vehicle choice on engines whose stock content has no cars.
``vehicle.stock``
    A stock vehicle of the target engine, picked from the connector's content catalog by a
    declared rule: ``nearest_by_dims`` (weighted relative error over length, width, height and
    wheelbase; refused when a residual exceeds its tolerance) or ``content_id`` (a named entry).
    The compiler records the residual of every dimension.
``prop.box``, ``prop.cylinder``, ``prop.sphere``, ``prop.mesh``
    A prop sized by the measurement. Mass is a density class times the volume, or an explicit
    override, each with sigma.
``static.mesh``
    Level collision built from a static mesh of the world package.

Plugins may declare further templates as ``<plugin>.<name>``. Overrides (mass, tyre, stiffness,
bounce or friction class) are explicit and carry a status (``measured``, ``assumed`` or
``fitted``) and a reason; nothing is ``fitted`` for passive objects in the first release.

Example::

    {"schema": "v2e.templates/1",
     "defaults": {"vehicle.*": "vehicle.generic"},
     "objects": {
       "crate": {"template": "prop.box", "density_class": "wood",
                 "overrides": {"mass_kg": {"value": 18.0, "sigma": 0.5, "status": "measured",
                                           "reason": "weighed before the test"}}},
       "van": {"template": "vehicle.stock", "rule": "nearest_by_dims",
               "tolerance": {"length": 0.08, "width": 0.06}, "fallback": "vehicle.generic"}}}
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from openbeam_v2e.schema import jsonio
from openbeam_v2e.schema.common import defs
from openbeam_v2e.schema.validate import Issue, SchemaError, validate

__all__ = [
    "SCHEMA_ID",
    "TEMPLATES_SCHEMA",
    "CORE_TEMPLATES",
    "DEFAULT_TEMPLATES",
    "DIMENSION_WEIGHTS",
    "DIMENSION_TOLERANCE",
    "Override",
    "TemplateRule",
    "Templates",
    "validate_templates",
    "load_templates",
]

SCHEMA_ID = "v2e.templates/1"
CORE_TEMPLATES = (
    "vehicle.generic",
    "vehicle.stock",
    "prop.box",
    "prop.cylinder",
    "prop.sphere",
    "prop.mesh",
    "static.mesh",
)
#: Class pattern -> template, used when neither the object nor the document's defaults say.
DEFAULT_TEMPLATES: dict[str, str] = {
    "vehicle.*": "vehicle.generic",
    "prop.box": "prop.box",
    "prop.cylinder": "prop.cylinder",
    "prop.sphere": "prop.sphere",
    "prop.*": "prop.mesh",
    "static": "static.mesh",
}
#: ``nearest_by_dims`` weights and relative tolerances when a rule does not give its own.
DIMENSION_WEIGHTS: dict[str, float] = {"length": 1.0, "width": 1.0, "height": 0.5, "wheelbase": 0.5}
DIMENSION_TOLERANCE: dict[str, float] = {"length": 0.10, "width": 0.10, "height": 0.15, "wheelbase": 0.10}
OVERRIDE_STATUS = ("measured", "assumed", "fitted")


def _schema() -> dict[str, Any]:
    d = defs()
    template = {"anyOf": [{"enum": list(CORE_TEMPLATES)}, {"$ref": "#/$defs/namespaced"}]}
    d["template"] = template
    d["dim_map"] = {
        "type": "object",
        "propertyNames": {"pattern": r"^[a-z][a-z0-9_]*$"},
        "additionalProperties": {"type": "number", "minimum": 0},
    }
    d["override"] = {
        "type": "object",
        "required": ["value", "status", "reason"],
        "properties": {
            "value": {"type": ["number", "string"]},
            "sigma": {"$ref": "#/$defs/sigma"},
            "status": {"enum": list(OVERRIDE_STATUS)},
            "reason": {"type": "string", "minLength": 1},
        },
        "additionalProperties": False,
    }
    d["rule"] = {
        "type": "object",
        "required": ["template"],
        "properties": {
            "template": {"$ref": "#/$defs/template"},
            "rule": {"enum": ["nearest_by_dims", "content_id"]},
            "content_id": {"type": "string", "minLength": 1},
            "weights": {"$ref": "#/$defs/dim_map"},
            "tolerance": {"$ref": "#/$defs/dim_map"},
            "fallback": {"$ref": "#/$defs/template"},
            "density_class": {"type": "string", "minLength": 1},
            "overrides": {
                "type": "object",
                "propertyNames": {"pattern": r"^[a-z][a-z0-9_]*$"},
                "additionalProperties": {"$ref": "#/$defs/override"},
            },
        },
        "additionalProperties": False,
        "allOf": [{"if": {"properties": {"rule": {"const": "content_id"}}, "required": ["rule"]},
                   "then": {"required": ["content_id"]}}],
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://github.com/OpenBeamWorld/v2e/schemas/v2e.templates-1.json",
        "title": "V2E template mapping (v2e.templates/1)",
        "type": "object",
        "required": ["schema"],
        "properties": {
            "schema": {"const": SCHEMA_ID},
            "defaults": {
                "type": "object",
                "propertyNames": {"pattern": r"^(?:(?:vehicle|prop)\.(?:[a-z][a-z0-9_]*|\*)|static)$"},
                "additionalProperties": {"$ref": "#/$defs/template"},
            },
            "objects": {
                "type": "object",
                "propertyNames": {"$ref": "#/$defs/id"},
                "additionalProperties": {"$ref": "#/$defs/rule"},
            },
            "notes": {"type": "string"},
        },
        "additionalProperties": False,
        "$defs": d,
    }


TEMPLATES_SCHEMA: dict[str, Any] = _schema()


@dataclass(frozen=True)
class Override:
    value: float | str
    status: str
    reason: str
    sigma: float | None = None

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {"value": self.value}
        if self.sigma is not None:
            out["sigma"] = self.sigma
        out["status"] = self.status
        out["reason"] = self.reason
        return out


@dataclass(frozen=True)
class TemplateRule:
    template: str
    rule: str = "nearest_by_dims"
    content_id: str | None = None
    weights: Mapping[str, float] = field(default_factory=dict)
    tolerance: Mapping[str, float] = field(default_factory=dict)
    fallback: str | None = None
    density_class: str | None = None
    overrides: Mapping[str, Override] = field(default_factory=dict)

    @classmethod
    def parse(cls, raw: Mapping[str, Any]) -> "TemplateRule":
        return cls(
            template=raw["template"],
            rule=raw.get("rule", "nearest_by_dims"),
            content_id=raw.get("content_id"),
            weights=dict(raw.get("weights", {})),
            tolerance=dict(raw.get("tolerance", {})),
            fallback=raw.get("fallback"),
            density_class=raw.get("density_class"),
            overrides={k: Override(v["value"], v["status"], v["reason"], v.get("sigma"))
                       for k, v in raw.get("overrides", {}).items()},
        )

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {"template": self.template}
        if self.template == "vehicle.stock":
            out["rule"] = self.rule
        if self.content_id:
            out["content_id"] = self.content_id
        if self.weights:
            out["weights"] = dict(self.weights)
        if self.tolerance:
            out["tolerance"] = dict(self.tolerance)
        if self.fallback:
            out["fallback"] = self.fallback
        if self.density_class:
            out["density_class"] = self.density_class
        if self.overrides:
            out["overrides"] = {k: v.to_json() for k, v in self.overrides.items()}
        return out

    def effective_weights(self) -> dict[str, float]:
        return {**DIMENSION_WEIGHTS, **self.weights}

    def effective_tolerance(self) -> dict[str, float]:
        return {**DIMENSION_TOLERANCE, **self.tolerance}


@dataclass(frozen=True)
class Templates:
    defaults: Mapping[str, str] = field(default_factory=dict)
    objects: Mapping[str, TemplateRule] = field(default_factory=dict)
    notes: str = ""

    def rule_for(self, actor: str, object_class: str) -> TemplateRule:
        """The rule for one actor: its own entry, else the document defaults, else the built-in ones."""
        if actor in self.objects:
            return self.objects[actor]
        for table in (self.defaults, DEFAULT_TEMPLATES):
            if object_class in table:
                return TemplateRule(table[object_class])
            for pattern in sorted(table, key=len, reverse=True):
                if fnmatch.fnmatchcase(object_class, pattern):
                    return TemplateRule(table[pattern])
        raise KeyError(f"no template for class {object_class!r}")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any], check: bool = True) -> "Templates":
        if check:
            issues = validate_templates(raw)
            if issues:
                raise SchemaError(issues, "templates")
        return cls(dict(raw.get("defaults", {})),
                   {k: TemplateRule.parse(v) for k, v in raw.get("objects", {}).items()},
                   raw.get("notes", ""))

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"schema": SCHEMA_ID}
        if self.defaults:
            out["defaults"] = dict(self.defaults)
        if self.objects:
            out["objects"] = {k: v.to_json() for k, v in self.objects.items()}
        if self.notes:
            out["notes"] = self.notes
        return out

    def sha256(self) -> str:
        return jsonio.canonical_sha256(self.to_dict())


def validate_templates(raw: Any) -> list[Issue]:
    """Schema and consistency issues of a template mapping (empty when valid)."""
    issues = validate(raw, TEMPLATES_SCHEMA)
    if issues or not isinstance(raw, Mapping):
        return issues
    for aid, rule in raw.get("objects", {}).items():
        path = f"/objects/{aid}"
        template = rule["template"]
        if rule.get("fallback") == "vehicle.stock":
            issues.append(Issue(f"{path}/fallback", "a fallback must not need a catalog match itself"))
        if "rule" in rule and template != "vehicle.stock":
            issues.append(Issue(f"{path}/rule", "only vehicle.stock takes a selection rule"))
        if "density_class" in rule and not template.startswith("prop."):
            issues.append(Issue(f"{path}/density_class", "only props take a density class"))
        for name, override in rule.get("overrides", {}).items():
            if override["status"] == "fitted":
                issues.append(Issue(f"{path}/overrides/{name}",
                                    "nothing is fitted in this release; use 'measured' or 'assumed'"))
    return issues


def load_templates(path: str | Path) -> Templates:
    """Read and validate a ``v2e.templates/1`` file."""
    return Templates.from_dict(jsonio.load(path))
