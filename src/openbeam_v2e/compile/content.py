# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""Choose and size the authored content that stands in for each dynamic actor.

Every parameter of a content record carries a value, a sigma, a status and a source:

``measured``
    from the scene's ``dims_m`` / ``mass_kg`` or the reconstruction (object dimensions or
    hull extents);
``assumed``
    derived from a class default (the generic vehicle's design point, a density class);
``catalog``
    a property of a stock catalog entry.

Template overrides keep the status and reason the template file gives. The content-origin
record hashes its parameters, so runs whose content differs never share a provenance
signature.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping

from openbeam_v2e import __version__
from openbeam_v2e.schema import jsonio
from openbeam_v2e.schema.catalog import CatalogEntry, ContentCatalog
from openbeam_v2e.schema.common import Measured
from openbeam_v2e.schema.scene import Actor
from openbeam_v2e.schema.templates import TemplateRule

__all__ = [
    "GENERIC_VEHICLE_DESIGN_POINT",
    "DENSITY_CLASSES",
    "DEFAULT_DENSITY_CLASS",
    "Param",
    "ContentRecord",
    "Refusal",
    "measured_dims",
    "generic_vehicle",
    "stock_vehicle",
    "prop_content",
]

#: Design point of the generated generic vehicle: round values chosen for a mid-size car without
#: reference to any particular vehicle. Missing measurements scale from these.
GENERIC_VEHICLE_DESIGN_POINT: dict[str, float] = {
    "length_m": 4.75,
    "width_m": 1.80,
    "height_m": 1.45,
    "wheelbase_m": 2.75,
    "track_m": 1.60,
    "ride_height_m": 0.15,
    "wheel_radius_m": 0.32,
    "mass_kg": 1500.0,
}

#: Effective bulk density (kg/m^3) and its relative one-sigma uncertainty, per density class.
DENSITY_CLASSES: dict[str, tuple[float, float]] = {
    "foam": (30.0, 0.30),
    "cardboard": (150.0, 0.50),
    "hollow_plastic": (120.0, 0.50),
    "wood": (600.0, 0.25),
    "plastic": (950.0, 0.10),
    "rubber": (1100.0, 0.10),
    "water_filled": (1000.0, 0.05),
    "concrete": (2400.0, 0.05),
    "steel": (7850.0, 0.02),
}
DEFAULT_DENSITY_CLASS: dict[str, str] = {"prop.box": "wood", "prop.cylinder": "plastic", "prop.sphere": "rubber"}

_DIM_ALIASES = {"length": "length_m", "width": "width_m", "height": "height_m", "wheelbase": "wheelbase_m",
                "track": "track_m", "ride_height": "ride_height_m", "wheel_radius": "wheel_radius_m",
                "radius": "radius_m", "diameter": "diameter_m"}


@dataclass(frozen=True)
class Param:
    value: float | str
    status: str
    source: str
    sigma: float | None = None
    reason: str = ""

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {"value": self.value, "sigma": self.sigma, "status": self.status, "source": self.source}
        if self.reason:
            out["reason"] = self.reason
        return out


@dataclass(frozen=True)
class Refusal:
    """Why the compiler cannot produce an honest package for (part of) a scene."""

    code: str
    message: str
    actor: str | None = None

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.actor:
            out["actor"] = self.actor
        return out

    def __str__(self) -> str:
        return f"{self.actor + ': ' if self.actor else ''}{self.message} [{self.code}]"


@dataclass
class ContentRecord:
    actor: str
    object_class: str
    template: str
    kind: str                                  # "generated" | "stock"
    parameters: dict[str, Param] = field(default_factory=dict)
    content_id: str = ""
    content_version: str = ""
    generator: str = ""
    generator_version: str = ""
    seatable: bool = False
    residuals: dict[str, dict[str, float]] = field(default_factory=dict)
    fallback_from: str = ""

    def params_sha256(self) -> str:
        return jsonio.canonical_sha256({k: v.to_json() for k, v in sorted(self.parameters.items())})

    def value(self, name: str, default: float | None = None) -> float | None:
        p = self.parameters.get(name)
        return default if p is None or isinstance(p.value, str) else float(p.value)

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {"actor": self.actor, "class": self.object_class, "template": self.template,
                               "kind": self.kind}
        if self.content_id:
            out["content_id"] = self.content_id
        if self.content_version:
            out["content_version"] = self.content_version
        if self.generator:
            out["generator"] = self.generator
            out["generator_version"] = self.generator_version
        out["seatable"] = self.seatable
        out["parameters"] = {k: v.to_json() for k, v in sorted(self.parameters.items())}
        if self.residuals:
            out["residuals"] = self.residuals
        if self.fallback_from:
            out["fallback_from"] = self.fallback_from
        out["params_sha256"] = self.params_sha256()
        return out


def measured_dims(spec: Actor, world_dims: Mapping[str, Measured] | None = None,
                  hull_extents: Any | None = None) -> dict[str, Param]:
    """Dimensions known for an actor: scene first, then the reconstruction, then hull extents."""
    out: dict[str, Param] = {}
    for source, table in (("scene", spec.dims_m), ("reconstruction", world_dims or {})):
        for name, m in table.items():
            key = _DIM_ALIASES.get(name, name if name.endswith("_m") else f"{name}_m")
            if key not in out:
                out[key] = Param(float(m.value), "measured", source, m.sigma)
    if hull_extents is not None:  # body axes: x forward (length), y left (width), z up (height)
        ext = [float(e) for e in hull_extents]
        for key, value in zip(("length_m", "width_m", "height_m"), ext):
            if key not in out:
                out[key] = Param(value, "measured", "reconstruction hull extents", None)
    if spec.mass_kg is not None:
        out["mass_kg"] = Param(float(spec.mass_kg.value), "measured", "scene", spec.mass_kg.sigma)
    return out


def _apply_overrides(params: dict[str, Param], rule: TemplateRule) -> None:
    for name, o in rule.overrides.items():
        params[name] = Param(o.value, o.status, "template override", o.sigma, o.reason)


def generic_vehicle(spec: Actor, rule: TemplateRule, dims: dict[str, Param]) -> ContentRecord:
    """A generated vehicle sized by the measurements; unmeasured values scale from the design point."""
    dp = GENERIC_VEHICLE_DESIGN_POINT
    params = {k: v for k, v in dims.items() if k in dp}
    scale_l = params["length_m"].value / dp["length_m"] if "length_m" in params else 1.0
    scale_w = params["width_m"].value / dp["width_m"] if "width_m" in params else 1.0
    scale_h = params["height_m"].value / dp["height_m"] if "height_m" in params else 1.0
    src = "class default: generic vehicle design point"
    derived = {
        "length_m": (dp["length_m"] * scale_l, 0.10),
        "width_m": (dp["width_m"] * scale_w, 0.05),
        "height_m": (dp["height_m"] * scale_h, 0.05),
        "wheelbase_m": (dp["wheelbase_m"] * scale_l, 0.05),
        "track_m": (dp["track_m"] * scale_w, 0.05),
        "ride_height_m": (dp["ride_height_m"], 0.25),
        "wheel_radius_m": (dp["wheel_radius_m"] * min(max(scale_l, 0.8), 1.4), 0.10),
        "mass_kg": (dp["mass_kg"] * scale_l * scale_w * scale_h, 0.20),
    }
    for key, (value, rel_sigma) in derived.items():
        if key not in params:
            params[key] = Param(round(value, 4), "assumed", src, round(abs(value) * rel_sigma, 4))
    _apply_overrides(params, rule)
    return ContentRecord(spec.id, spec.cls, "vehicle.generic", "generated", params,
                         generator="openbeam_v2e.generators.vehicle", generator_version=__version__,
                         seatable=bool(spec.seatable))


def _relative_residuals(entry: CatalogEntry, dims: dict[str, Param], rule: TemplateRule) -> dict[str, dict[str, float]]:
    """Per dimension: catalog value, measured value, relative residual and the tolerance used."""
    tolerance = rule.effective_tolerance()
    out: dict[str, dict[str, float]] = {}
    for name, catalog_value in sorted(entry.dims_m.items()):
        key = _DIM_ALIASES.get(name, f"{name}_m")
        param = dims.get(key)
        if param is None or isinstance(param.value, str) or float(param.value) <= 0:
            continue
        measured = float(param.value)
        row = {"catalog": float(catalog_value), "measured": measured,
               "relative": (float(catalog_value) - measured) / measured}
        if name in tolerance:
            row["tolerance"] = float(tolerance[name])
        out[name] = row
    return out


def stock_vehicle(spec: Actor, rule: TemplateRule, dims: dict[str, Param],
                  catalog: ContentCatalog) -> ContentRecord | Refusal:
    """Pick a stock vehicle by ``nearest_by_dims`` or ``content_id``; refuse when nothing fits."""
    candidates = catalog.vehicles()
    if rule.rule == "content_id":
        try:
            entry = catalog.entry(rule.content_id or "")
        except KeyError:
            return Refusal("no_stock_match", f"catalog has no entry {rule.content_id!r}", spec.id)
        chosen, residuals = entry, _relative_residuals(entry, dims, rule)
    else:
        if not candidates:
            return Refusal("no_stock_match", f"the {catalog.backend} catalog lists no stock vehicles", spec.id)
        weights = rule.effective_weights()
        tolerance = rule.effective_tolerance()
        best: tuple[float, CatalogEntry, dict[str, dict[str, float]]] | None = None
        for entry in candidates:
            residuals = _relative_residuals(entry, dims, rule)
            if not residuals:
                continue
            if any(abs(r["relative"]) > tolerance.get(name, math.inf) for name, r in residuals.items()):
                continue
            num = sum(weights.get(name, 0.0) * r["relative"] ** 2 for name, r in residuals.items())
            den = sum(weights.get(name, 0.0) for name in residuals) or 1.0
            cost = math.sqrt(num / den)
            if best is None or cost < best[0] or (cost == best[0] and entry.id < best[1].id):
                best = (cost, entry, residuals)
        if best is None:
            return Refusal("no_stock_match",
                           "no stock vehicle is within the dimension tolerances "
                           f"{dict(sorted(tolerance.items()))}", spec.id)
        _, chosen, residuals = best
    params = {f"{name}_m": Param(float(v), "catalog", f"catalog entry {chosen.id}") for name, v in chosen.dims_m.items()}
    if chosen.mass_kg is not None:
        params["mass_kg"] = Param(float(chosen.mass_kg), "catalog", f"catalog entry {chosen.id}")
    _apply_overrides(params, rule)
    return ContentRecord(spec.id, spec.cls, "vehicle.stock", "stock", params, content_id=chosen.id,
                         content_version=chosen.content_version, seatable=bool(spec.seatable or chosen.seatable),
                         residuals=residuals)


def _volume(template: str, params: dict[str, Param]) -> tuple[float, str] | None:
    def get(name: str) -> float | None:
        p = params.get(name)
        return None if p is None or isinstance(p.value, str) else float(p.value)

    if template == "prop.sphere":
        r = get("radius_m") or (get("diameter_m") or get("length_m") or 0.0) / 2.0
        return (4.0 / 3.0 * math.pi * r ** 3, "4/3 pi r^3") if r else None
    if template == "prop.cylinder":
        r = get("radius_m") or (get("diameter_m") or 0.0) / 2.0
        h = get("length_m") or get("height_m")
        if not r and get("width_m"):
            r = get("width_m") / 2.0
        return (math.pi * r * r * h, "pi r^2 h") if r and h else None
    dims = [get("length_m"), get("width_m"), get("height_m")]
    if all(dims):
        return dims[0] * dims[1] * dims[2], "length x width x height"
    return None


def prop_content(spec: Actor, rule: TemplateRule, dims: dict[str, Param]) -> ContentRecord | Refusal:
    """A generated prop sized by its measurements; mass from a density class unless measured."""
    template = rule.template
    params = dict(dims)
    if "mass_kg" not in params and "mass_kg" not in rule.overrides:
        density_class = rule.density_class or DEFAULT_DENSITY_CLASS.get(template, "wood")
        if density_class not in DENSITY_CLASSES:
            return Refusal("unknown_density_class", f"unknown density class {density_class!r}; "
                           f"known: {', '.join(sorted(DENSITY_CLASSES))}", spec.id)
        volume = _volume(template, params)
        if volume is None:
            return Refusal("missing_dimensions", f"{template} needs dimensions to size it", spec.id)
        density, rel = DENSITY_CLASSES[density_class]
        mass = density * volume[0]
        params["mass_kg"] = Param(round(mass, 4), "assumed", f"density class {density_class} x volume ({volume[1]})",
                                  round(mass * rel, 4))
        params["density_class"] = Param(density_class, "assumed", "template" if rule.density_class else "class default")
    _apply_overrides(params, rule)
    return ContentRecord(spec.id, spec.cls, template, "generated", params,
                         generator="openbeam_v2e.generators.props", generator_version=__version__)
