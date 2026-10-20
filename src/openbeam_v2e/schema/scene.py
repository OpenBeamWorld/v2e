# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""``v2e.scene/1``: what happens in a clip, independent of any engine.

A scene names its world package and lists:

* **actors**: dynamic objects (``vehicle.*`` or ``prop.*``) with measured dimensions, a control
  mode (``driver``, ``passive`` or a plugin-declared ``<plugin>.<name>``) and an initial state,
  plus static geometry (``class: "static"``) that points at a mesh of the world package;
* **events**: named frame intervals such as a first contact, a take-off or the moment an
  object comes to rest, each with the evidence it rests on;
* **phases**: named windows between events (``start``, ``end`` and an undeclared ``rest`` are
  reserved boundaries); scoring is reported per phase;
* **targets**: values a run should reproduce, each with sigma and a source; engine runs are
  never a source;
* **requires**: the effects, observables and actor counts a connector must offer.

No start mode gives an actor speed for free. With ``at_rest`` the actor appears, motionless,
where its pose says. With ``drive_in`` it appears motionless at the far end of a generated
approach road, and its controller accelerates it so that it passes its observed frame-0 pose
at the observed speed. ``moving`` describes a passive object that is already in motion in the
first frame; the compiler refuses it unless the scene names an engine-native launcher built
from authored content.

Example::

    {"schema": "v2e.scene/1", "world": "world/",
     "actors": [
       {"id": "car_1", "class": "vehicle.car", "dims_m": {"length": [4.4, 0.05]},
        "seatable": true, "control": "driver",
        "initial_state": {"mode": "drive_in", "frame0_pose": "track:car_1@0", "speed_mps": [12.0, 0.4]}},
       {"id": "wall", "class": "static", "mesh": 2, "role": "collision"}],
     "events": [{"name": "impact", "frame": [40, 41], "kind": "contact", "actors": ["car_1", "wall"]}],
     "phases": [{"name": "approach", "from": "start", "to": "impact"},
                {"name": "after_impact", "from": "impact", "to": "rest"}],
     "targets": [{"metric": "speed_at_event", "actor": "car_1", "event": "impact",
                  "value": [12.0, 0.4], "source": "design_value"}],
     "requires": {"effects": ["drive", "brake", "steer", "contact"],
                  "observables": ["position", "attitude"], "controlled_actors": 1}}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from openbeam_v2e.schema import jsonio
from openbeam_v2e.schema.common import (
    FORBIDDEN_TARGET_SOURCES,
    TARGET_SOURCES,
    Measured,
    PoseRef,
    default_symmetry,
    defs,
    parse_pose_ref,
)
from openbeam_v2e.schema.validate import Issue, SchemaError, validate

__all__ = [
    "SCHEMA_ID",
    "SCENE_SCHEMA",
    "RESERVED_BOUNDARIES",
    "EVENT_KINDS",
    "infer_event_kind",
    "Actor",
    "StartState",
    "Event",
    "Phase",
    "Target",
    "Requires",
    "Scene",
    "validate_scene",
    "load_scene",
]

SCHEMA_ID = "v2e.scene/1"
RESERVED_BOUNDARIES = ("start", "end", "rest")
#: Event kinds the scorer can detect in a trajectory; ``custom`` events are compared by time only.
EVENT_KINDS = ("contact", "separation", "rest", "custom")

_VELOCITY_WORDS = ("velocity", "speed_vector", "angular_velocity", "impulse", "force", "momentum")


def _schema() -> dict[str, Any]:
    d = defs()
    d["initial_state"] = {
        "type": "object",
        "required": ["mode"],
        "properties": {
            "mode": {"enum": ["at_rest", "drive_in", "moving"]},
            "pose": {"$ref": "#/$defs/pose_ref"},
            "frame0_pose": {"$ref": "#/$defs/pose_ref"},
            "speed_mps": {"$ref": "#/$defs/measured"},
            "path": {"enum": ["auto", "straight", "arc"]},
            "launcher": {
                "type": "object",
                "required": ["mechanism"],
                "properties": {"mechanism": {"type": "string", "minLength": 1}, "content": {"type": "string"}},
                "additionalProperties": False,
            },
        },
        "additionalProperties": False,
        "allOf": [
            {"if": {"properties": {"mode": {"const": "at_rest"}}}, "then": {"required": ["pose"]}},
            {"if": {"properties": {"mode": {"const": "drive_in"}}}, "then": {"required": ["frame0_pose", "speed_mps"]}},
            {"if": {"properties": {"mode": {"const": "moving"}}}, "then": {"required": ["frame0_pose"]}},
        ],
    }
    d["dynamic_actor"] = {
        "type": "object",
        "required": ["id", "class", "control", "initial_state"],
        "properties": {
            "id": {"$ref": "#/$defs/id"},
            "class": {"type": "string", "pattern": r"^(?:vehicle|prop)\.[a-z][a-z0-9_]*$"},
            "object": {"$ref": "#/$defs/id"},
            "dims_m": {"$ref": "#/$defs/dims"},
            "mass_kg": {"$ref": "#/$defs/measured"},
            "seatable": {"type": "boolean"},
            "control": {
                "anyOf": [{"enum": ["driver", "passive"]}, {"$ref": "#/$defs/namespaced"}],
            },
            "symmetry": {"$ref": "#/$defs/symmetry"},
            "initial_state": {"$ref": "#/$defs/initial_state"},
            "notes": {"type": "string"},
        },
        "additionalProperties": False,
    }
    d["static_actor"] = {
        "type": "object",
        "required": ["id", "class", "mesh"],
        "properties": {
            "id": {"$ref": "#/$defs/id"},
            "class": {"const": "static"},
            "mesh": {"anyOf": [{"type": "integer", "minimum": 1}, {"type": "string", "pattern": r"^meshes/[1-9][0-9]*\.npz$"}]},
            "role": {"enum": ["collision", "drivable", "visual"]},
            "notes": {"type": "string"},
        },
        "additionalProperties": False,
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://github.com/OpenBeamWorld/v2e/schemas/v2e.scene-1.json",
        "title": "V2E scene description (v2e.scene/1)",
        "type": "object",
        "required": ["schema", "world", "actors"],
        "properties": {
            "schema": {"const": SCHEMA_ID},
            "name": {"type": "string"},
            "world": {"type": "string", "minLength": 1},
            "frame": {"type": "string"},
            "fps": {"type": "number", "exclusiveMinimum": 0},
            "actors": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "if": {"properties": {"class": {"const": "static"}}, "required": ["class"]},
                    "then": {"$ref": "#/$defs/static_actor"},
                    "else": {"$ref": "#/$defs/dynamic_actor"},
                },
            },
            "events": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["name", "frame"],
                    "properties": {
                        "name": {"$ref": "#/$defs/id"},
                        "frame": {"$ref": "#/$defs/frame_interval"},
                        "kind": {"enum": list(EVENT_KINDS)},
                        "actors": {"type": "array", "items": {"$ref": "#/$defs/id"}, "uniqueItems": True},
                        "evidence": {"type": "string"},
                    },
                    "additionalProperties": False,
                },
            },
            "phases": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["name", "from", "to"],
                    "properties": {
                        "name": {"$ref": "#/$defs/id"},
                        "from": {"$ref": "#/$defs/id"},
                        "to": {"$ref": "#/$defs/id"},
                    },
                    "additionalProperties": False,
                },
            },
            "targets": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["metric", "value", "source"],
                    "properties": {
                        "metric": {"type": "string", "pattern": r"^[a-z][a-z0-9_]*$"},
                        "actor": {"$ref": "#/$defs/id"},
                        "event": {"$ref": "#/$defs/id"},
                        "phase": {"$ref": "#/$defs/id"},
                        "value": {"$ref": "#/$defs/measured"},
                        "unit": {"type": "string"},
                        "source": {"enum": list(TARGET_SOURCES)},
                        "citation": {"type": "string"},
                        "tolerance_sigma": {"type": "number", "exclusiveMinimum": 0},
                    },
                    "additionalProperties": False,
                },
            },
            "requires": {
                "type": "object",
                "properties": {
                    "effects": {"type": "array", "items": {"type": "string", "minLength": 1}, "uniqueItems": True},
                    "observables": {"type": "array", "items": {"type": "string", "minLength": 1}, "uniqueItems": True},
                    "controlled_actors": {"type": "integer", "minimum": 0},
                    "seatable_vehicles": {"type": "integer", "minimum": 0},
                    "regime": {"type": "string"},
                },
                "additionalProperties": False,
            },
            "notes": {"type": "string"},
        },
        "additionalProperties": False,
        "$defs": d,
    }


SCENE_SCHEMA: dict[str, Any] = _schema()


@dataclass(frozen=True)
class StartState:
    mode: str
    pose: PoseRef | None = None
    frame0_pose: PoseRef | None = None
    speed_mps: Measured | None = None
    path: str = "auto"
    launcher: Mapping[str, str] | None = None

    @classmethod
    def parse(cls, raw: Mapping[str, Any]) -> "StartState":
        return cls(
            mode=raw["mode"],
            pose=parse_pose_ref(raw["pose"]) if "pose" in raw else None,
            frame0_pose=parse_pose_ref(raw["frame0_pose"]) if "frame0_pose" in raw else None,
            speed_mps=Measured.parse(raw["speed_mps"]) if "speed_mps" in raw else None,
            path=raw.get("path", "auto"),
            launcher=dict(raw["launcher"]) if "launcher" in raw else None,
        )

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {"mode": self.mode}
        if self.pose is not None:
            out["pose"] = self.pose.to_json()
        if self.frame0_pose is not None:
            out["frame0_pose"] = self.frame0_pose.to_json()
        if self.speed_mps is not None:
            out["speed_mps"] = self.speed_mps.to_json()
        if self.path != "auto":
            out["path"] = self.path
        if self.launcher is not None:
            out["launcher"] = dict(self.launcher)
        return out

    @property
    def reference_pose(self) -> PoseRef | None:
        """The pose the actor must have at video frame 0 (or at spawn, for ``at_rest``)."""
        return self.pose if self.mode == "at_rest" else self.frame0_pose


@dataclass(frozen=True)
class Actor:
    id: str
    cls: str
    control: str = "passive"
    initial_state: StartState | None = None
    dims_m: Mapping[str, Measured] = field(default_factory=dict)
    mass_kg: Measured | None = None
    seatable: bool = False
    symmetry: str | None = None
    object: str | None = None
    mesh: int | None = None
    role: str = "collision"
    notes: str = ""

    @property
    def is_static(self) -> bool:
        return self.cls == "static"

    @property
    def is_vehicle(self) -> bool:
        return self.cls.startswith("vehicle.")

    @property
    def is_controlled(self) -> bool:
        return not self.is_static and self.control != "passive"

    @property
    def object_id(self) -> str:
        """Object id in the world package (defaults to the actor id)."""
        return self.object or self.id

    @property
    def symmetry_class(self) -> str:
        return self.symmetry or default_symmetry(self.cls)

    @classmethod
    def parse(cls, raw: Mapping[str, Any]) -> "Actor":
        if raw["class"] == "static":
            mesh = raw["mesh"]
            if isinstance(mesh, str):
                mesh = int(mesh.split("/")[-1].split(".")[0])
            return cls(id=raw["id"], cls="static", mesh=int(mesh), role=raw.get("role", "collision"),
                       notes=raw.get("notes", ""))
        return cls(
            id=raw["id"],
            cls=raw["class"],
            control=raw["control"],
            initial_state=StartState.parse(raw["initial_state"]),
            dims_m={k: Measured.parse(v) for k, v in raw.get("dims_m", {}).items()},
            mass_kg=Measured.parse(raw["mass_kg"]) if "mass_kg" in raw else None,
            seatable=bool(raw.get("seatable", False)),
            symmetry=raw.get("symmetry"),
            object=raw.get("object"),
            notes=raw.get("notes", ""),
        )

    def to_json(self) -> dict[str, Any]:
        if self.is_static:
            out: dict[str, Any] = {"id": self.id, "class": "static", "mesh": self.mesh, "role": self.role}
        else:
            out = {"id": self.id, "class": self.cls}
            if self.object:
                out["object"] = self.object
            if self.dims_m:
                out["dims_m"] = {k: v.to_json() for k, v in self.dims_m.items()}
            if self.mass_kg is not None:
                out["mass_kg"] = self.mass_kg.to_json()
            out["seatable"] = self.seatable
            out["control"] = self.control
            if self.symmetry:
                out["symmetry"] = self.symmetry
            assert self.initial_state is not None
            out["initial_state"] = self.initial_state.to_json()
        if self.notes:
            out["notes"] = self.notes
        return out


@dataclass(frozen=True)
class Event:
    name: str
    frame: tuple[int, int]
    kind: str = "custom"
    actors: tuple[str, ...] = ()
    evidence: str = ""

    @property
    def centre(self) -> float:
        return 0.5 * (self.frame[0] + self.frame[1])

    @property
    def sigma_frames(self) -> float:
        """Half-width of the interval treated as one sigma, floored at half a frame."""
        return max(0.5, 0.5 * (self.frame[1] - self.frame[0]))

    @classmethod
    def parse(cls, raw: Mapping[str, Any]) -> "Event":
        kind = raw.get("kind") or infer_event_kind(raw["name"])
        return cls(raw["name"], (int(raw["frame"][0]), int(raw["frame"][1])), kind, tuple(raw.get("actors", ())),
                   raw.get("evidence", ""))

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {"name": self.name, "frame": list(self.frame), "kind": self.kind}
        if self.actors:
            out["actors"] = list(self.actors)
        if self.evidence:
            out["evidence"] = self.evidence
        return out


def infer_event_kind(name: str) -> str:
    """Event kind guessed from a name (``impact`` -> contact, ``takeoff`` -> separation, ...)."""
    lowered = name.lower()
    if any(word in lowered for word in ("contact", "impact", "touchdown", "landing", "hit", "collision")):
        return "contact"
    if any(word in lowered for word in ("separation", "takeoff", "take_off", "lift_off", "liftoff", "launch")):
        return "separation"
    if "rest" in lowered or "stop" in lowered:
        return "rest"
    return "custom"


@dataclass(frozen=True)
class Phase:
    name: str
    start: str
    end: str

    @classmethod
    def parse(cls, raw: Mapping[str, Any]) -> "Phase":
        return cls(raw["name"], raw["from"], raw["to"])

    def to_json(self) -> dict[str, Any]:
        return {"name": self.name, "from": self.start, "to": self.end}


@dataclass(frozen=True)
class Target:
    metric: str
    value: Measured
    source: str
    actor: str | None = None
    event: str | None = None
    phase: str | None = None
    unit: str = ""
    citation: str = ""
    tolerance_sigma: float = 2.0

    @classmethod
    def parse(cls, raw: Mapping[str, Any]) -> "Target":
        return cls(raw["metric"], Measured.parse(raw["value"]), raw["source"], raw.get("actor"), raw.get("event"),
                   raw.get("phase"), raw.get("unit", ""), raw.get("citation", ""),
                   float(raw.get("tolerance_sigma", 2.0)))

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {"metric": self.metric}
        for key in ("actor", "event", "phase"):
            if getattr(self, key):
                out[key] = getattr(self, key)
        out["value"] = self.value.to_json()
        if self.unit:
            out["unit"] = self.unit
        out["source"] = self.source
        if self.citation:
            out["citation"] = self.citation
        if self.tolerance_sigma != 2.0:
            out["tolerance_sigma"] = self.tolerance_sigma
        return out


@dataclass(frozen=True)
class Requires:
    effects: tuple[str, ...] = ()
    observables: tuple[str, ...] = ("position", "attitude")
    controlled_actors: int | None = None
    seatable_vehicles: int | None = None
    regime: str | None = None

    @classmethod
    def parse(cls, raw: Mapping[str, Any] | None) -> "Requires":
        raw = raw or {}
        return cls(tuple(raw.get("effects", ())), tuple(raw.get("observables", ("position", "attitude"))),
                   raw.get("controlled_actors"), raw.get("seatable_vehicles"), raw.get("regime"))

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {"effects": list(self.effects), "observables": list(self.observables)}
        if self.controlled_actors is not None:
            out["controlled_actors"] = self.controlled_actors
        if self.seatable_vehicles is not None:
            out["seatable_vehicles"] = self.seatable_vehicles
        if self.regime:
            out["regime"] = self.regime
        return out


@dataclass(frozen=True)
class Scene:
    world: str
    actors: tuple[Actor, ...]
    events: tuple[Event, ...] = ()
    phases: tuple[Phase, ...] = ()
    targets: tuple[Target, ...] = ()
    requires: Requires = field(default_factory=Requires)
    name: str = ""
    frame: str = "Z-up level frame, SI units"
    fps: float | None = None
    notes: str = ""
    base_dir: Path | None = None

    @property
    def dynamic_actors(self) -> tuple[Actor, ...]:
        return tuple(a for a in self.actors if not a.is_static)

    @property
    def static_actors(self) -> tuple[Actor, ...]:
        return tuple(a for a in self.actors if a.is_static)

    @property
    def controlled_actors(self) -> tuple[Actor, ...]:
        return tuple(a for a in self.actors if a.is_controlled)

    def actor(self, name: str) -> Actor:
        for item in self.actors:
            if item.id == name:
                return item
        raise KeyError(f"scene has no actor {name!r}")

    def event(self, name: str) -> Event:
        for event in self.events:
            if event.name == name:
                return event
        raise KeyError(f"scene has no event {name!r}")

    def world_path(self) -> Path:
        """The world package directory, resolved against the scene file's directory."""
        path = Path(self.world)
        if not path.is_absolute() and self.base_dir is not None:
            path = self.base_dir / path
        return path

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any], base_dir: str | Path | None = None, check: bool = True) -> "Scene":
        if check:
            issues = validate_scene(raw)
            if issues:
                raise SchemaError(issues, "scene")
        return cls(
            world=raw["world"],
            actors=tuple(Actor.parse(a) for a in raw["actors"]),
            events=tuple(Event.parse(e) for e in raw.get("events", ())),
            phases=tuple(Phase.parse(p) for p in raw.get("phases", ())),
            targets=tuple(Target.parse(t) for t in raw.get("targets", ())),
            requires=Requires.parse(raw.get("requires")),
            name=raw.get("name", ""),
            frame=raw.get("frame", "Z-up level frame, SI units"),
            fps=raw.get("fps"),
            notes=raw.get("notes", ""),
            base_dir=Path(base_dir) if base_dir is not None else None,
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"schema": SCHEMA_ID}
        if self.name:
            out["name"] = self.name
        out["world"] = self.world
        out["frame"] = self.frame
        if self.fps is not None:
            out["fps"] = self.fps
        out["actors"] = [a.to_json() for a in self.actors]
        out["events"] = [e.to_json() for e in self.events]
        out["phases"] = [p.to_json() for p in self.phases]
        out["targets"] = [t.to_json() for t in self.targets]
        out["requires"] = self.requires.to_json()
        if self.notes:
            out["notes"] = self.notes
        return out

    def sha256(self) -> str:
        return jsonio.canonical_sha256(self.to_dict())


def _semantic_issues(raw: Mapping[str, Any]) -> list[Issue]:
    issues: list[Issue] = []
    actors = raw.get("actors", [])
    ids = [a.get("id") for a in actors if isinstance(a, Mapping)]
    seen: set[str] = set()
    for index, aid in enumerate(ids):
        if aid in seen:
            issues.append(Issue(f"/actors/{index}/id", f"actor id {aid!r} is used twice"))
        seen.add(aid)
    dynamic = {a.get("id"): a for a in actors if isinstance(a, Mapping) and a.get("class") != "static"}

    for index, actor in enumerate(actors):
        if not isinstance(actor, Mapping) or actor.get("class") == "static":
            continue
        state = actor.get("initial_state", {})
        path = f"/actors/{index}"
        for key in state:
            if any(word in key for word in _VELOCITY_WORDS):
                issues.append(Issue(f"{path}/initial_state/{key}", "initial states never inject velocity"))
        mode = state["mode"] if "mode" in state else None
        if mode == "drive_in" and actor.get("control") == "passive":
            issues.append(Issue(f"{path}/initial_state/mode",
                                "a passive actor cannot drive in; start the scene earlier or declare a launcher"))
        if mode == "drive_in" and not str(actor.get("class", "")).startswith("vehicle."):
            issues.append(Issue(f"{path}/initial_state/mode", "only vehicles can drive in"))
        if mode == "moving" and actor.get("control") != "passive":
            issues.append(Issue(f"{path}/initial_state/mode",
                                "a controlled actor that is moving at frame 0 must use mode 'drive_in'"))
        if actor.get("seatable") and not str(actor.get("class", "")).startswith("vehicle."):
            issues.append(Issue(f"{path}/seatable", "only vehicles can be seatable"))
        speed = state.get("speed_mps")
        if speed is not None:
            value = speed[0] if isinstance(speed, list) else speed
            if isinstance(value, (int, float)) and value < 0:
                issues.append(Issue(f"{path}/initial_state/speed_mps", "speed must not be negative"))

    events = raw.get("events", [])
    event_names: set[str] = set()
    for index, event in enumerate(events):
        if not isinstance(event, Mapping):
            continue
        name = event.get("name")
        if name in RESERVED_BOUNDARIES and name != "rest":
            issues.append(Issue(f"/events/{index}/name", f"{name!r} is a reserved phase boundary"))
        if name in event_names:
            issues.append(Issue(f"/events/{index}/name", f"event {name!r} is declared twice"))
        event_names.add(name)
        frame = event.get("frame")
        if isinstance(frame, list) and len(frame) == 2 and all(isinstance(f, int) for f in frame) and frame[0] > frame[1]:
            issues.append(Issue(f"/events/{index}/frame", "first frame is after last frame"))
        for aid in event.get("actors", []):
            if aid not in seen:
                issues.append(Issue(f"/events/{index}/actors", f"unknown actor {aid!r}"))

    boundaries = event_names | set(RESERVED_BOUNDARIES)
    phase_names: set[str] = set()
    for index, phase in enumerate(raw.get("phases", [])):
        if not isinstance(phase, Mapping):
            continue
        if phase.get("name") in phase_names:
            issues.append(Issue(f"/phases/{index}/name", f"phase {phase.get('name')!r} is declared twice"))
        phase_names.add(phase.get("name"))
        for key in ("from", "to"):
            if phase.get(key) not in boundaries:
                issues.append(Issue(f"/phases/{index}/{key}", f"{phase.get(key)!r} is neither an event nor start/end/rest"))
        if phase.get("from") == "end" or phase.get("to") == "start":
            issues.append(Issue(f"/phases/{index}", "a phase must run forwards in time"))

    for index, target in enumerate(raw.get("targets", [])):
        if not isinstance(target, Mapping):
            continue
        path = f"/targets/{index}"
        if target.get("source") in FORBIDDEN_TARGET_SOURCES:
            issues.append(Issue(f"{path}/source", "engine runs are never a source of targets"))
        if "actor" in target and target["actor"] not in dynamic:
            issues.append(Issue(f"{path}/actor", f"unknown dynamic actor {target['actor']!r}"))
        if "event" in target and target["event"] not in boundaries:
            issues.append(Issue(f"{path}/event", f"unknown event {target['event']!r}"))
        if "phase" in target and target["phase"] not in phase_names:
            issues.append(Issue(f"{path}/phase", f"unknown phase {target['phase']!r}"))

    requires = raw.get("requires", {})
    if isinstance(requires, Mapping):
        controlled = [a for a in dynamic.values() if a.get("control") not in (None, "passive")]
        declared = requires.get("controlled_actors")
        if isinstance(declared, int) and declared < len(controlled):
            issues.append(Issue("/requires/controlled_actors",
                                f"declares {declared} but the scene has {len(controlled)} controlled actors"))
        seatable = [a for a in dynamic.values() if a.get("seatable")]
        declared = requires.get("seatable_vehicles")
        if isinstance(declared, int) and declared < len(seatable):
            issues.append(Issue("/requires/seatable_vehicles",
                                f"declares {declared} but the scene has {len(seatable)} seatable vehicles"))
    return issues


def validate_scene(raw: Any) -> list[Issue]:
    """Schema and cross-reference issues of a scene document (empty when valid)."""
    issues = validate(raw, SCENE_SCHEMA)
    if issues or not isinstance(raw, Mapping):
        return issues
    return _semantic_issues(raw)


def load_scene(path: str | Path) -> Scene:
    """Read and validate a ``v2e.scene/1`` file; relative world paths resolve against its directory."""
    path = Path(path)
    return Scene.from_dict(jsonio.load(path), base_dir=path.parent)
