# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""Building blocks shared by the V2E document schemas, and their Python counterparts.

Conventions used by every V2E document:

* the level and world frame is Z-up, right-handed, in metres; time is in seconds;
* orientations are unit quaternions in ``[x, y, z, w]`` order (scalar last);
* a measured quantity is written ``[value, sigma]`` (one standard deviation, same unit), or
  as a bare number when no uncertainty is known;
* identifiers start with a letter and use letters, digits, ``_``, ``.`` and ``-``;
* frame intervals are inclusive ``[first, last]`` video frame indices.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

__all__ = [
    "ID_PATTERN",
    "NAMESPACED_PATTERN",
    "TARGET_SOURCES",
    "FORBIDDEN_TARGET_SOURCES",
    "OBJECT_CLASS_PATTERN",
    "SYMMETRY_CLASSES",
    "Measured",
    "Pose",
    "PoseRef",
    "defs",
    "parse_pose_ref",
    "default_symmetry",
]

ID_PATTERN = r"^[A-Za-z][A-Za-z0-9_.\-]{0,63}$"
#: Plugin-declared values are namespaced: ``<plugin>.<name>``.
NAMESPACED_PATTERN = r"^[a-z][a-z0-9_\-]*\.[a-z0-9_.\-]+$"
#: Dynamic object classes are ``vehicle.<kind>`` or ``prop.<kind>``; static geometry is ``static``.
OBJECT_CLASS_PATTERN = r"^(?:(?:vehicle|prop)\.[a-z][a-z0-9_]*|static)$"

#: Where a target value may come from. Engine runs are never a source of targets.
TARGET_SOURCES = (
    "first_principles",
    "design_value",
    "published_measurement",
    "surveyed_clip",
    "independent_measurement",
    "reconstruction",
)
FORBIDDEN_TARGET_SOURCES = ("engine_run", "simulation", "run")

#: Symmetry classes used to compare attitudes modulo the rotations an object cannot show.
#: ``none``: every rotation is visible (vehicles). ``box``: rotations mapping the measured box to
#: itself. ``cylinder``: spin about the axis is invisible, the axis sign is visible.
#: ``cylinder_symmetric``: spin and axis sign are both invisible. ``sphere``: attitude is ignored.
SYMMETRY_CLASSES = ("none", "box", "cylinder", "cylinder_symmetric", "sphere")

_POSE_REF = re.compile(r"^track:(?P<object>[A-Za-z][A-Za-z0-9_.\-]{0,63})@(?P<frame>[0-9]+)$")


def default_symmetry(object_class: str) -> str:
    """Symmetry class assumed when a document does not declare one."""
    if object_class == "prop.box":
        return "box"
    if object_class == "prop.cylinder":
        return "cylinder_symmetric"
    if object_class == "prop.sphere":
        return "sphere"
    return "none"


@dataclass(frozen=True)
class Measured:
    """A value with an optional one-sigma uncertainty in the same unit."""

    value: float
    sigma: float | None = None

    @classmethod
    def parse(cls, raw: Any) -> "Measured":
        if isinstance(raw, Measured):
            return raw
        if isinstance(raw, bool):
            raise ValueError("a measured value cannot be a boolean")
        if isinstance(raw, (int, float)):
            return cls(float(raw), None)
        if isinstance(raw, Mapping):
            return cls(float(raw["value"]), None if raw.get("sigma") is None else float(raw["sigma"]))
        if isinstance(raw, Sequence) and len(raw) == 2:
            return cls(float(raw[0]), None if raw[1] is None else float(raw[1]))
        raise ValueError(f"cannot read a measured value from {raw!r}")

    def to_json(self) -> Any:
        return self.value if self.sigma is None else [self.value, self.sigma]

    def z(self, other: float) -> float | None:
        """(other - value) / sigma, or None when the sigma is unknown or zero."""
        if self.sigma is None or self.sigma <= 0.0:
            return None
        return (float(other) - self.value) / self.sigma


@dataclass(frozen=True)
class Pose:
    """A pose in the level frame: position in metres and an ``[x, y, z, w]`` unit quaternion."""

    position_m: tuple[float, float, float]
    quat_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)

    def __post_init__(self) -> None:
        if len(self.position_m) != 3 or len(self.quat_xyzw) != 4:
            raise ValueError("a pose needs a 3-vector position and a 4-vector quaternion")
        norm = math.sqrt(sum(float(c) * float(c) for c in self.quat_xyzw))
        if not math.isfinite(norm) or abs(norm - 1.0) > 1e-3:
            raise ValueError(f"quaternion {self.quat_xyzw!r} is not a unit quaternion")
        if not all(math.isfinite(float(c)) for c in self.position_m):
            raise ValueError("pose position must be finite")
        object.__setattr__(self, "position_m", tuple(float(c) for c in self.position_m))
        object.__setattr__(self, "quat_xyzw", tuple(float(c) / norm for c in self.quat_xyzw))

    @classmethod
    def parse(cls, raw: Mapping[str, Any]) -> "Pose":
        return cls(tuple(raw["position_m"]), tuple(raw.get("quat_xyzw", (0.0, 0.0, 0.0, 1.0))))

    def to_json(self) -> dict[str, Any]:
        return {"position_m": list(self.position_m), "quat_xyzw": list(self.quat_xyzw)}


@dataclass(frozen=True)
class PoseRef:
    """Either an explicit pose or a reference ``track:<object>@<frame>`` into a world package."""

    pose: Pose | None = None
    object: str | None = None
    frame: int | None = None

    def to_json(self) -> Any:
        if self.pose is not None:
            return self.pose.to_json()
        return f"track:{self.object}@{self.frame}"


def parse_pose_ref(raw: Any) -> PoseRef:
    """Read a pose reference (``"track:car_1@0"``) or an explicit pose object."""
    if isinstance(raw, PoseRef):
        return raw
    if isinstance(raw, str):
        match = _POSE_REF.match(raw)
        if not match:
            raise ValueError(f"pose reference {raw!r} must look like 'track:<object>@<frame>'")
        return PoseRef(object=match["object"], frame=int(match["frame"]))
    if isinstance(raw, Mapping):
        return PoseRef(pose=Pose.parse(raw))
    raise ValueError(f"cannot read a pose reference from {raw!r}")


def defs() -> dict[str, Any]:
    """JSON Schema ``$defs`` shared by the V2E document schemas."""
    number = {"type": "number"}
    return {
        "id": {"type": "string", "pattern": ID_PATTERN},
        "namespaced": {"type": "string", "pattern": NAMESPACED_PATTERN},
        "sigma": {"type": ["number", "null"], "minimum": 0},
        "measured": {
            "anyOf": [
                number,
                {"type": "array", "prefixItems": [number, {"$ref": "#/$defs/sigma"}], "minItems": 2, "maxItems": 2},
            ]
        },
        "vec3": {"type": "array", "items": number, "minItems": 3, "maxItems": 3},
        "quat_xyzw": {"type": "array", "items": number, "minItems": 4, "maxItems": 4},
        "mat3": {"type": "array", "items": {"$ref": "#/$defs/vec3"}, "minItems": 3, "maxItems": 3},
        "pose": {
            "type": "object",
            "required": ["position_m"],
            "properties": {"position_m": {"$ref": "#/$defs/vec3"}, "quat_xyzw": {"$ref": "#/$defs/quat_xyzw"}},
            "additionalProperties": False,
        },
        "pose_ref": {
            "anyOf": [
                {"type": "string", "pattern": r"^track:[A-Za-z][A-Za-z0-9_.\-]{0,63}@[0-9]+$"},
                {"$ref": "#/$defs/pose"},
            ]
        },
        "frame_interval": {
            "type": "array",
            "items": {"type": "integer", "minimum": 0},
            "minItems": 2,
            "maxItems": 2,
        },
        "object_class": {"type": "string", "pattern": OBJECT_CLASS_PATTERN},
        "symmetry": {"enum": list(SYMMETRY_CLASSES)},
        "dims": {
            "type": "object",
            "propertyNames": {"pattern": r"^[a-z][a-z0-9_]*$"},
            "additionalProperties": {"$ref": "#/$defs/measured"},
        },
    }
