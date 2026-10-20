# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""Object classes: default template, motion model and broad dimension priors per class.

The priors are deliberately wide (they keep a fit well-posed when the footage alone cannot
separate size from distance, for example for a free-flying object); measured dimensions come
from the silhouettes. A configuration can override any of them per object.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from openbeam_v2e.perception.types import ObjectSpec


@dataclass(frozen=True)
class ClassDefaults:
    template: str
    motion: str
    dims_prior: Mapping[str, tuple[float, float]] = field(default_factory=dict)  # (value, sigma) m
    typical_jerk: float = 10.0          # m/s^3, smoothing prior for the position track
    typical_angular_jerk: float = 20.0  # rad/s^3, smoothing prior for the attitude track


CLASS_DEFAULTS: dict[str, ClassDefaults] = {
    "vehicle.car": ClassDefaults("vehicle", "ground",
                                 {"length": (4.5, 0.6), "width": (1.8, 0.2), "height": (1.5, 0.25)},
                                 10.0, 10.0),
    "vehicle.truck": ClassDefaults("vehicle", "ground",
                                   {"length": (8.0, 3.0), "width": (2.5, 0.3), "height": (3.0, 0.8)},
                                   8.0, 8.0),
    "vehicle.van": ClassDefaults("vehicle", "ground",
                                 {"length": (5.2, 0.8), "width": (2.0, 0.2), "height": (2.2, 0.4)},
                                 10.0, 10.0),
    "prop.box": ClassDefaults("box", "free", {"length": (1.0, 1.0), "width": (1.0, 1.0), "height": (1.0, 1.0)},
                              20.0, 40.0),
    "prop.cylinder": ClassDefaults("cylinder", "free",
                                   {"length": (0.6, 0.6), "width": (0.6, 0.6), "height": (1.0, 1.0)}, 20.0, 40.0),
    "prop.sphere": ClassDefaults("sphere", "free",
                                 {"length": (0.5, 0.5), "width": (0.5, 0.5), "height": (0.5, 0.5)}, 20.0, 40.0),
}

GENERIC = ClassDefaults("box", "free", {"length": (1.0, 2.0), "width": (1.0, 2.0), "height": (1.0, 2.0)})


def defaults_for(object_class: str) -> ClassDefaults:
    if object_class in CLASS_DEFAULTS:
        return CLASS_DEFAULTS[object_class]
    if object_class.startswith("vehicle."):
        return CLASS_DEFAULTS["vehicle.car"]
    return GENERIC


def object_spec(object_id: str, object_class: str = "vehicle.car", template: str | None = None,
                motion: str | None = None, dims_prior: Mapping[str, tuple[float, float]] | None = None,
                template_params: Mapping[str, float] | None = None, fit_dims: bool = True) -> ObjectSpec:
    """An :class:`ObjectSpec` with class defaults filled in."""
    d = defaults_for(object_class)
    prior = dict(d.dims_prior)
    prior.update({k: (float(v[0]), float(v[1])) for k, v in (dims_prior or {}).items()})
    return ObjectSpec(object_id, object_class, template or d.template, dict(template_params or {}), prior,
                      motion or d.motion, fit_dims)


DIM_NAMES = ("length", "width", "height")
