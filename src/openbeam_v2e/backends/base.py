# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""The connector protocol: capabilities, provenance, attestation and the ``Backend`` interface.

A connector (the code calls it a *backend*) installs compiled packages, opens a scene, spawns
actors at rest at a pose, runs one controller program per controlled actor and reports what
happened. It never reconstructs, scores or searches.

Three rules shape every type in this module:

1. **Nothing here can push an object around behind the engine's back.** The types offer no way
   to apply a push or a twist, to move or re-orient a body, to change its speed, weight or the
   pull of gravity, or to put a body back mid-run; bodies enter the world motionless at a
   requested place. Controllers act only through channels the connector declares for each actor,
   with values of the declared kind and range, checked against the plan first.
2. **Capabilities are declared, never assumed.** Channels, timing regimes, recording classes,
   actor limits and observables come from the connector's manifest. Values beyond the core set
   are namespaced strings (``"<plugin>.<name>"``) that plugins declare; core code never names
   them.
3. **Provenance is computed, not narrated.** Every effect a run relies on has a source; the claim
   class (``native``, ``emulated``, ``scripted``, ``incomplete``, ``test_double``) is derived from
   those sources, and the provenance signature hashes the effects and the content origins so that
   runs with different content are never grouped together.

Frames and units: level and world frames are Z-up, right-handed, metres; time is simulated seconds
from spawn. Quaternions are ``[x, y, z, w]``.
"""

from __future__ import annotations

import abc
import collections.abc
import dataclasses
import enum
import hashlib
import json
import math
import types
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence, Union

__all__ = [
    # schema ids
    "SCHEMA_PROGRAM", "SCHEMA_COMPILED", "SCHEMA_PLAN", "SCHEMA_RUN", "SCHEMA_RUNLOG",
    # capabilities
    "Layer", "ChannelKind", "ActuatorChannel", "RENDER_PACED", "STEPPED", "CORE_REGIMES",
    "Repeatability", "RegimeSupport", "SENSOR_LOCKSTEP", "STEPPED_CAPTURE", "STREAM_CAPTURE",
    "RASTER_TEST_DOUBLE", "NO_RECORDING", "CORE_RECORDINGS", "ControllerHost", "Capabilities",
    "BackendManifest", "CORE_CHANNEL_IDS", "PHYSICS_EFFECTS", "FORBIDDEN_OPERATIONS",
    # provenance
    "Source", "Effect", "Pose", "SpawnRecord", "Attestation", "ContentOrigin", "CLAIM_CLASSES",
    # content and refs
    "Package", "InstallReceipt", "LevelRef", "ObjectRef", "ActorRef", "SpawnSpec", "CompiledScene",
    "CameraSpec", "StateRow", "Frame",
    # planning and runs
    "Plan", "RefusalReason", "Refusal", "RunSpec", "RunResult", "Actuation",
    # interfaces and errors
    "Backend", "Lockstep", "BackendError", "RefusedCall",
    # helpers
    "canonical_json", "sha256_json", "to_jsonable", "from_jsonable", "check_program",
]

SCHEMA_PROGRAM = "v2e.program/1"
SCHEMA_COMPILED = "v2e.compiled/1"
SCHEMA_PLAN = "v2e.plan/1"
SCHEMA_RUN = "v2e.run/1"
SCHEMA_RUNLOG = "v2e-runlog/1"


# --------------------------------------------------------------------------- JSON helpers


def to_jsonable(value: Any) -> Any:
    """Convert dataclasses, enums, paths, tuples and sets into plain JSON values."""
    if isinstance(value, enum.Enum):
        return value.value
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: to_jsonable(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (frozenset, set)):
        return sorted(to_jsonable(v) for v in value)
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite number {value!r} cannot be written as JSON")
    return value


def canonical_json(value: Any) -> bytes:
    """Deterministic JSON bytes (sorted keys, no whitespace) used for every hash in V2E."""
    return json.dumps(to_jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode("ascii")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


_NONE_TYPE = type(None)


def _convert(hint: Any, value: Any) -> Any:
    """Convert a JSON value into the Python type named by ``hint`` (dataclass field types)."""
    if hint is Any or value is None:
        return value
    origin = typing.get_origin(hint)
    args = typing.get_args(hint)
    if origin in (Union, types.UnionType):
        options = [a for a in args if a is not _NONE_TYPE]
        last_error: Exception | None = None
        for option in options:
            try:
                return _convert(option, value)
            except (TypeError, ValueError, KeyError) as exc:
                last_error = exc
        raise ValueError(f"cannot read {value!r} as {hint}") from last_error
    if origin is tuple:
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_convert(args[0], v) for v in value)
        if args:
            return tuple(_convert(a, v) for a, v in zip(args, value))
        return tuple(value)
    if origin in (frozenset, set):
        return frozenset(_convert(args[0], v) for v in value) if args else frozenset(value)
    if origin in (list, collections.abc.Sequence) or hint is list:
        return [_convert(args[0], v) for v in value] if args else list(value)
    if origin in (dict, collections.abc.Mapping) or hint in (dict, Mapping):
        if args:
            return {_convert(args[0], k): _convert(args[1], v) for k, v in value.items()}
        return dict(value)
    if isinstance(hint, type):
        if issubclass(hint, enum.Enum):
            return hint(value)
        if dataclasses.is_dataclass(hint):
            return from_jsonable(hint, value)
        if hint is Path:
            return Path(value)
        if hint is float:
            if isinstance(value, bool):
                raise TypeError("a boolean is not a number")
            return float(value)
        if hint is int:
            if isinstance(value, bool) or (isinstance(value, float) and not value.is_integer()):
                raise TypeError(f"{value!r} is not an integer")
            return int(value)
        if hint is str:
            if not isinstance(value, str):
                raise TypeError(f"{value!r} is not a string")
            return value
        if hint is bool:
            if not isinstance(value, bool):
                raise TypeError(f"{value!r} is not a boolean")
            return value
    return value


def from_jsonable(cls: type, data: Mapping[str, Any]) -> Any:
    """Build dataclass ``cls`` from the JSON form produced by :func:`to_jsonable`.

    Unknown keys are ignored so that newer files stay readable; missing keys fall back to the
    field defaults.
    """
    if not isinstance(data, Mapping):
        raise TypeError(f"{cls.__name__} needs a JSON object, got {type(data).__name__}")
    hints = typing.get_type_hints(cls)
    kwargs = {}
    for f in dataclasses.fields(cls):
        if not f.init or f.name not in data:
            continue
        kwargs[f.name] = _convert(hints[f.name], data[f.name])
    return cls(**kwargs)


# --------------------------------------------------------------------------- capabilities


class Layer(str, enum.Enum):
    """Which layer implements a mechanism. Always disclosed next to the claim class."""

    ENGINE_CORE = "engine_core"
    ENGINE_SHIPPED_SCRIPT = "engine_shipped_script"   # script code the engine itself ships
    PLUGIN_SCRIPT = "plugin_script"                   # shipped by a V2E plugin, never by this package


class ChannelKind(str, enum.Enum):
    CONTINUOUS = "continuous"   # a value in ``range``, held until changed
    TOGGLE = "toggle"           # a press (1) flips an engine-side state; 0 means "no press"
    DISCRETE = "discrete"       # one of ``levels`` (for example a gear)


@dataclass(frozen=True)
class ActuatorChannel:
    """One actuator a controller may command, per actor.

    ``id`` is a core id (``drive.throttle``, ``drive.brake``, ``drive.steer``,
    ``drive.parking_brake``) or a plugin id ``<plugin>.<name>``. ``drive.steer`` is negative to
    the left and positive to the right.
    """

    id: str
    effect: str
    kind: ChannelKind
    range: tuple[float, float] = (0.0, 1.0)
    levels: tuple[float, ...] = ()
    mechanism: str = ""
    layer: Layer = Layer.ENGINE_CORE

    def check(self, value: Any) -> float:
        """Return ``value`` as a float, or raise ``ValueError`` if this channel cannot take it."""
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{self.id}: {value!r} is not a number")
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"{self.id}: {value!r} is not finite")
        if self.kind is ChannelKind.TOGGLE:
            if value not in (0.0, 1.0):
                raise ValueError(f"{self.id}: a toggle takes 1 (press) or 0 (no press), not {value!r}")
        elif self.kind is ChannelKind.DISCRETE:
            if value not in self.levels:
                raise ValueError(f"{self.id}: {value!r} is not one of the levels {self.levels!r}")
        else:
            lo, hi = self.range
            if not (lo - 1e-9 <= value <= hi + 1e-9):
                raise ValueError(f"{self.id}: {value!r} is outside the range [{lo}, {hi}]")
            value = min(max(value, lo), hi)
        return value


#: Core actuator channel ids. Public connectors declare no channel outside this set.
CORE_CHANNEL_IDS = ("drive.throttle", "drive.brake", "drive.steer", "drive.parking_brake")

#: Effects provided by the engine's own physics rather than by an actuator channel.
PHYSICS_EFFECTS = frozenset({"contact", "gravity", "friction", "deformation", "detachment"})

#: Words that name a physics bypass. No connector method, channel or effect may be built on them;
#: a public test checks every connector's method names against this list.
FORBIDDEN_OPERATIONS = ("force", "impulse", "pose", "position", "velocity", "gravity", "mass",
                        "teleport", "reset_actor", "torque")

# Core regimes; plugins declare further ones as namespaced strings ("<plugin>.<name>").
RENDER_PACED = "render_paced"   # control once per rendered frame; steps per frame follow wall time
STEPPED = "stepped"             # physics advances only on explicit step calls
CORE_REGIMES = (RENDER_PACED, STEPPED)


class Repeatability(str, enum.Enum):
    BIT_IDENTICAL = "bit_identical"   # measured: repeats gave identical rows
    STATISTICAL = "statistical"       # measured: repeats differ; compare with robust intervals
    UNMEASURED = "unmeasured"


@dataclass(frozen=True)
class RegimeSupport:
    regime: str
    available: bool
    requires: tuple[str, ...] = ()        # scene conditions, for example ("player_vehicle",)
    repeatability: Repeatability = Repeatability.UNMEASURED
    evidence: str | None = None           # public report id; never typed by hand for engines


# Core recording classes; plugins declare further ones as namespaced strings.
SENSOR_LOCKSTEP = "sensor_lockstep"     # a camera sensor read in lockstep with stepping
STEPPED_CAPTURE = "stepped_capture"     # one documented frame grab per step
STREAM_CAPTURE = "stream_capture"       # a copy of an encoded stream; not frame-exact
RASTER_TEST_DOUBLE = "raster_test_double"
NO_RECORDING = "none"
CORE_RECORDINGS = (SENSOR_LOCKSTEP, STEPPED_CAPTURE, STREAM_CAPTURE, RASTER_TEST_DOUBLE, NO_RECORDING)


class ControllerHost(str, enum.Enum):
    ENGINE_SCRIPT = "engine_script"       # one engine-side host drives every controlled actor
    VEHICLE_SCRIPT = "vehicle_script"     # one host per vehicle, inside the engine
    CLIENT_LOCKSTEP = "client_lockstep"   # controllers run in Python between explicit steps


@dataclass(frozen=True)
class Capabilities:
    channels: tuple[ActuatorChannel, ...]
    soft_body: bool
    detachment: str                       # "breakgroups" | "none"
    regimes: tuple[RegimeSupport, ...]
    physics_dt_s: float | None
    recordings: tuple[str, ...]
    spawn_at_pose: bool                   # position AND rotation, at rest
    max_controlled_actors: int
    max_seatable_vehicles: int | None     # None = no known limit
    controller_hosts: frozenset[ControllerHost]
    observables: frozenset[str]
    camera_runtime_fov: bool
    instance_mask: bool
    extras: Mapping[str, str] = field(default_factory=dict)   # namespaced plugin keys only

    def channel(self, channel_id: str) -> ActuatorChannel | None:
        return next((c for c in self.channels if c.id == channel_id), None)

    def regime(self, name: str) -> RegimeSupport | None:
        return next((r for r in self.regimes if r.regime == name), None)


@dataclass(frozen=True)
class BackendManifest:
    backend: str          # "openbeam" | "fake" | plugin name
    kind: str             # "engine" | "test_double"
    engine: str           # public product name and public version only
    api: str              # for example "mcp (documented tools)"
    dialect: str
    capabilities: Capabilities
    plugins: tuple[str, ...] = ()
    licence_notice: str = ""


# --------------------------------------------------------------------------- provenance


class Source(str, enum.Enum):
    NATIVE = "native"                         # engine mechanism driven by authored content
    NATIVE_SUBSTITUTE = "native_substitute"   # a different native mechanism stands in
    SCRIPTED = "scripted"                     # force code shipped by a plugin; never this package
    ABSENT = "absent"
    TEST_DOUBLE = "test_double"


CLAIM_CLASSES = ("native", "emulated", "scripted", "incomplete", "test_double")


@dataclass(frozen=True)
class Effect:
    name: str
    source: Source
    mechanism: str
    layer: Layer = Layer.ENGINE_CORE
    artifact_sha256: str | None = None    # required when source is SCRIPTED

    def __post_init__(self) -> None:
        if self.source is Source.SCRIPTED and not self.artifact_sha256:
            raise ValueError(f"effect {self.name!r}: a scripted effect must pin its artefact by sha256")


@dataclass(frozen=True)
class Pose:
    """A pose in the level frame (Z-up, metres) with an ``[x, y, z, w]`` unit quaternion."""

    position_m: tuple[float, float, float]
    quat_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)

    def __post_init__(self) -> None:
        if len(self.position_m) != 3 or len(self.quat_xyzw) != 4:
            raise ValueError("a pose needs a 3-vector position and a 4-vector quaternion")
        pos = tuple(float(c) for c in self.position_m)
        quat = tuple(float(c) for c in self.quat_xyzw)
        if not all(math.isfinite(c) for c in pos + quat):
            raise ValueError("pose values must be finite")
        norm = math.sqrt(sum(c * c for c in quat))
        if abs(norm - 1.0) > 1e-3:
            raise ValueError(f"quaternion {quat!r} is not a unit quaternion")
        object.__setattr__(self, "position_m", pos)
        object.__setattr__(self, "quat_xyzw", tuple(c / norm for c in quat))

    @classmethod
    def from_yaw(cls, position_m: Sequence[float], yaw_deg: float = 0.0) -> "Pose":
        half = math.radians(yaw_deg) / 2.0
        return cls(tuple(position_m), (0.0, 0.0, math.sin(half), math.cos(half)))

    @classmethod
    def parse(cls, raw: Mapping[str, Any]) -> "Pose":
        """Read ``{"position_m": [...], "quat_xyzw": [...]}`` or ``{"position_m": ..., "yaw_deg": ...}``."""
        if "quat_xyzw" in raw:
            return cls(tuple(raw["position_m"]), tuple(raw["quat_xyzw"]))
        return cls.from_yaw(raw["position_m"], float(raw.get("yaw_deg", 0.0)))

    def yaw_deg(self) -> float:
        x, y, z, w = self.quat_xyzw
        return math.degrees(math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))

    def angle_to_deg(self, other: "Pose") -> float:
        """Geodesic angle between the two orientations."""
        dot = abs(sum(a * b for a, b in zip(self.quat_xyzw, other.quat_xyzw)))
        return math.degrees(2.0 * math.acos(min(1.0, dot)))

    def distance_to_m(self, other: "Pose") -> float:
        return math.dist(self.position_m, other.position_m)


@dataclass(frozen=True)
class SpawnRecord:
    actor: str
    requested: Pose
    actual: Pose               # read back before t = 0
    residual_m: float
    residual_deg: float
    method: str


@dataclass(frozen=True)
class Attestation:
    spawns: tuple[SpawnRecord, ...]
    initial_velocity_mps: float      # always 0.0: every actor spawns at rest
    teleports: int                   # 0 in a valid run
    in_run_resets: int               # 0 in a valid run
    removed_actors: int              # actors the engine removed; > 0 makes the run invalid
    regime: str
    programs: tuple[tuple[str, str], ...]   # (actor, sha256) of every program sent
    ledger: Path
    ledger_head_sha256: str
    audit: str                       # "clean" or the first violation (the run is then invalid)
    spawn_tolerance_m: float = 0.02
    spawn_tolerance_deg: float = 0.5


@dataclass(frozen=True)
class ContentOrigin:
    """Where an actor's content came from: a stock engine asset or V2E-generated content."""

    actor: str
    kind: str                                   # "stock" | "generated"
    generator: str = ""
    generator_version: str = ""
    params_sha256: str = ""
    parameters: Mapping[str, Any] = field(default_factory=dict)
    content_id: str = ""                        # stock: the engine's content id
    content_version: str = ""

    def __post_init__(self) -> None:
        if self.kind not in ("stock", "generated"):
            raise ValueError(f"content origin kind must be 'stock' or 'generated', not {self.kind!r}")


# --------------------------------------------------------------------------- content and references


@dataclass(frozen=True)
class Package:
    """A compiled package. Exactly how it reaches the engine is connector-specific.

    ``path`` is a local file; ``url`` a download address the user controls (stored in ledgers
    without its query string); ``game_path`` a file already on the engine's machine. ``contents``
    holds structured content for connectors that need no archive (the fake test connector).
    """

    name: str
    kind: str                       # "level" | "vehicle" | "prop" | "bundle" | connector-specific
    sha256: str = ""
    path: Path | None = None
    url: str | None = None
    filename: str | None = None
    game_path: str | None = None
    contents: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class InstallReceipt:
    package: str
    sha256: str
    status: str                     # "installed" | "already_present"
    engine_ids: tuple[str, ...] = ()   # what the engine calls the installed items
    details: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LevelRef:
    package: str | None = None
    level_id: str | None = None
    filename: str | None = None
    terrain: str | None = None
    spawn: str | None = None


@dataclass(frozen=True)
class ObjectRef:
    """Something that can be spawned: a vehicle or a prop from an installed package."""

    name: str
    kind: str                         # "vehicle" | "prop"
    package: str | None = None
    engine_file: str | None = None    # the engine's name for it, from the install receipt
    config: str | None = None
    seatable: bool = False
    dims_m: tuple[float, float, float] | None = None       # length, width, height
    mass_kg: float | None = None
    bbox_min_m: tuple[float, float, float] | None = None   # body frame, relative to the reference point
    bbox_max_m: tuple[float, float, float] | None = None
    geometry: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ActorRef:
    actor: str           # scene actor id
    handle: str          # connector instance id
    object: str
    seatable: bool = False


@dataclass(frozen=True)
class SpawnSpec:
    actor: str
    object: str
    pose: Pose


@dataclass(frozen=True)
class CompiledScene:
    """``v2e.compiled/1``: what a compiler hands to a connector."""

    connector: str
    dialect: str
    level: LevelRef
    packages: tuple[Package, ...]
    objects: tuple[ObjectRef, ...]
    spawns: tuple[SpawnSpec, ...]
    programs: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    content: tuple[ContentOrigin, ...] = ()
    scene: str | None = None
    t0_s: float | None = None        # simulated time of video frame 0, from the compiler's timeline
    notes: tuple[str, ...] = ()
    schema: str = SCHEMA_COMPILED

    def object(self, name: str) -> ObjectRef:
        for obj in self.objects:
            if obj.name == name:
                return obj
        raise KeyError(f"compiled scene has no object {name!r}")

    def to_json(self) -> dict[str, Any]:
        return to_jsonable(self)

    @classmethod
    def from_json(cls, data: Mapping[str, Any], base: Path | None = None) -> "CompiledScene":
        if data.get("schema") != SCHEMA_COMPILED:
            raise ValueError(f"not a {SCHEMA_COMPILED} document (schema={data.get('schema')!r})")
        compiled = from_jsonable(cls, data)
        if base is not None:   # package paths are relative to the compiled file
            packages = tuple(
                dataclasses.replace(p, path=(base / p.path) if p.path and not p.path.is_absolute() else p.path)
                for p in compiled.packages
            )
            compiled = dataclasses.replace(compiled, packages=packages)
        return compiled


@dataclass(frozen=True)
class CameraSpec:
    name: str
    width: int
    height: int
    fov_y_deg: float | None = None
    pose: Pose | None = None


@dataclass(frozen=True)
class StateRow:
    """One actor's state, as a lockstep connector reports it (level frame)."""

    actor: str
    t: float
    p: tuple[float, float, float]
    q: tuple[float, float, float, float] | None
    v: tuple[float, float, float] | None = None
    w: tuple[float, float, float] | None = None
    contacts: tuple[str, ...] = ()
    extras: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Frame:
    camera: str
    t: float
    width: int
    height: int
    path: Path | None = None
    meta: Mapping[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- planning and runs

Actuation = Mapping[str, Mapping[str, float]]   # actor id -> {channel id -> value}


def check_program(program: Mapping[str, Any]) -> None:
    """Raise ``ValueError`` unless ``program`` is a well-formed ``v2e.program/1`` mapping.

    A program carries parameters only, never engine code: a connector bundles its own audited
    host and sends it the parameter blob.
    """
    if not isinstance(program, Mapping):
        raise ValueError("a controller program must be a JSON object")
    if program.get("schema") != SCHEMA_PROGRAM:
        raise ValueError(f"controller program schema must be {SCHEMA_PROGRAM!r}")
    for key in ("actor", "model"):
        if not isinstance(program.get(key), str) or not program[key]:
            raise ValueError(f"controller program needs a non-empty string {key!r}")
    if not isinstance(program.get("params", {}), Mapping):
        raise ValueError("controller program 'params' must be an object")
    for key in ("code", "script", "source", "lua", "angelscript"):
        if key in program:
            raise ValueError(f"controller programs carry parameters only; {key!r} is not allowed")


@dataclass(frozen=True)
class Plan:
    """What the harness agreed to run, written into the run directory before anything is installed."""

    backend: str
    regime: str
    host: ControllerHost | None
    actors: tuple[str, ...]
    controlled: tuple[str, ...]
    seatable: tuple[str, ...]
    programs: Mapping[str, Mapping[str, Any]]
    channels: Mapping[str, tuple[str, ...]]     # actor -> channel ids its program may command
    effects: tuple[Effect, ...]
    requires: Mapping[str, Any] = field(default_factory=dict)
    opt_in: Mapping[str, Any] = field(default_factory=dict)
    modes: Mapping[str, str] = field(default_factory=dict)   # actor -> initial-state mode
    notes: tuple[str, ...] = ()
    schema: str = SCHEMA_PLAN

    ok = True

    def to_json(self) -> dict[str, Any]:
        return to_jsonable(self)

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "Plan":
        return from_jsonable(cls, data)


@dataclass(frozen=True)
class RefusalReason:
    code: str      # "missing_effect" | "missing_channel" | "missing_observable" | "actor_limit" | ...
    detail: str


@dataclass(frozen=True)
class Refusal:
    backend: str
    reasons: tuple[RefusalReason, ...]

    ok = False

    def __str__(self) -> str:
        return "; ".join(f"{r.code}: {r.detail}" for r in self.reasons)

    def to_json(self) -> dict[str, Any]:
        return {"refused": True, **to_jsonable(self)}


@dataclass(frozen=True)
class RunSpec:
    duration_s: float
    host: ControllerHost | None       # None only when no actor is controlled
    regime: str
    telemetry_hz: float | None = None
    cameras: tuple[CameraSpec, ...] = ()
    record: str = NO_RECORDING
    seed: int = 0
    control_hz: float | None = None
    t0_s: float | None = None        # simulated time of video frame 0 (the harness takes it from the compiler)

    def __post_init__(self) -> None:
        if not (math.isfinite(self.duration_s) and self.duration_s > 0):
            raise ValueError("duration_s must be a positive number")
        for name in ("telemetry_hz", "control_hz"):
            value = getattr(self, name)
            if value is not None and not (math.isfinite(value) and value > 0):
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class RunResult:
    manifest: BackendManifest
    plan: Plan
    effects: tuple[Effect, ...]
    content: tuple[ContentOrigin, ...]
    attestation: Attestation
    rows: Path                    # v2e-runlog/1 JSONL, Z-up level frame
    events: Mapping[str, float]
    t0_s: float | None
    video: Path | None
    frames: Path | None
    sim_s: float
    wall_s: float
    schema: str = SCHEMA_RUN

    def valid(self) -> bool:
        a = self.attestation
        spawns_ok = all(s.residual_m <= a.spawn_tolerance_m + 1e-12 and s.residual_deg <= a.spawn_tolerance_deg + 1e-9
                        for s in a.spawns)
        return (a.audit == "clean" and a.teleports == 0 and a.in_run_resets == 0 and a.removed_actors == 0
                and a.initial_velocity_mps == 0.0 and spawns_ok)

    def invalid_reasons(self) -> list[str]:
        a = self.attestation
        reasons = []
        if a.audit != "clean":
            reasons.append(f"ledger audit: {a.audit}")
        if a.teleports:
            reasons.append(f"{a.teleports} teleport(s)")
        if a.in_run_resets:
            reasons.append(f"{a.in_run_resets} in-run reset(s)")
        if a.removed_actors:
            reasons.append(f"the engine removed {a.removed_actors} actor(s)")
        if a.initial_velocity_mps != 0.0:
            reasons.append("an actor did not spawn at rest")
        for s in a.spawns:
            if s.residual_m > a.spawn_tolerance_m + 1e-12 or s.residual_deg > a.spawn_tolerance_deg + 1e-9:
                reasons.append(f"spawn residual of {s.actor}: {s.residual_m:.3f} m, {s.residual_deg:.2f} deg")
        return reasons

    def claim_class(self) -> str:
        """``incomplete`` > ``test_double`` > ``scripted`` > ``emulated`` > ``native`` (as ``score.claims``)."""
        sources = {e.source for e in self.effects}
        if Source.ABSENT in sources:
            return "incomplete"
        if self.manifest.kind == "test_double" or Source.TEST_DOUBLE in sources:
            return "test_double"
        if Source.SCRIPTED in sources:
            return "scripted"
        if Source.NATIVE_SUBSTITUTE in sources:
            return "emulated"
        return "native"

    def physics_statement(self) -> str:
        cls = self.claim_class()
        layers = sorted({e.layer.value for e in self.effects})
        if cls == "incomplete":
            missing = sorted(e.name for e in self.effects if e.source is Source.ABSENT)
            return f"Incomplete: required effects absent: {', '.join(missing)}."
        if cls == "test_double":
            return ("Test double: this run used a stand-in connector for tests. It makes no claim about any "
                    "physics engine.")
        if cls == "scripted":
            scripted = sorted(e.name for e in self.effects if e.source is Source.SCRIPTED)
            return (f"Scripted: effects {', '.join(scripted)} came from plugin code pinned by sha256, not from the "
                    f"engine's own mechanisms. Layers: {', '.join(layers)}.")
        if cls == "emulated":
            subs = sorted(f"{e.name} via {e.mechanism}" for e in self.effects if e.source is Source.NATIVE_SUBSTITUTE)
            return (f"Emulated: native mechanisms stood in for observed ones ({'; '.join(subs)}). "
                    f"Layers: {', '.join(layers)}.")
        return (f"Native: every effect came from the engine's own mechanisms driven by authored content "
                f"({self.manifest.engine}). Layers: {', '.join(layers)}.")

    def provenance_signature(self) -> str:
        effects = sorted((to_jsonable(e) for e in self.effects), key=lambda e: canonical_json(e))
        content = sorted((to_jsonable(c) for c in self.content), key=lambda c: canonical_json(c))
        return sha256_json({"effects": effects, "content": content})

    def to_json(self, relative_to: Path | None = None) -> dict[str, Any]:
        def rel(p: Path | None) -> str | None:
            if p is None:
                return None
            if relative_to is not None:
                try:
                    return str(Path(p).resolve().relative_to(Path(relative_to).resolve()))
                except ValueError:
                    pass
            return str(p)

        out = to_jsonable(self)
        out["rows"] = rel(self.rows)
        out["video"] = rel(self.video)
        out["frames"] = rel(self.frames)
        out["attestation"]["ledger"] = rel(self.attestation.ledger)
        out["claim_class"] = self.claim_class()
        out["valid"] = self.valid()
        out["invalid_reasons"] = self.invalid_reasons()
        out["physics_statement"] = self.physics_statement()
        out["provenance_signature"] = self.provenance_signature()
        return out

    @classmethod
    def from_json(cls, data: Mapping[str, Any], base: Path | None = None) -> "RunResult":
        result = from_jsonable(cls, data)
        if base is None:
            return result

        def absolute(p: Path | None) -> Path | None:
            return None if p is None else (p if p.is_absolute() else base / p)

        attestation = dataclasses.replace(result.attestation, ledger=absolute(result.attestation.ledger))
        return dataclasses.replace(result, rows=absolute(result.rows), video=absolute(result.video),
                                   frames=absolute(result.frames), attestation=attestation)


# --------------------------------------------------------------------------- interfaces


class BackendError(RuntimeError):
    """A connector could not do what was asked (the engine refused, a transport failed, ...)."""


class RefusedCall(BackendError):
    """A call was refused by policy before any engine or network I/O happened."""


class Backend(abc.ABC):
    """Install, run, report. Never reconstruct, score or search.

    Implementations must not add methods that set forces, impulses, poses, velocities, gravity or
    mass, or that teleport or reset actors during a run (see ``FORBIDDEN_OPERATIONS``).
    """

    #: Registry name, for example ``"fake"``.
    name: str = ""

    @abc.abstractmethod
    def manifest(self) -> BackendManifest: ...

    @abc.abstractmethod
    def mechanisms(self, plan: Plan) -> tuple[Effect, ...]:
        """The effect-level provenance this connector would deliver for ``plan``."""

    @abc.abstractmethod
    def install(self, packages: Sequence[Package]) -> list[InstallReceipt]:
        """Install packages; idempotent by sha256."""

    @abc.abstractmethod
    def open_scene(self, level: LevelRef) -> None: ...

    @abc.abstractmethod
    def spawn(self, obj: ObjectRef, pose: Pose, actor: str) -> tuple[ActorRef, SpawnRecord]:
        """Spawn ``obj`` at rest at ``pose`` for scene actor ``actor``; report requested versus actual."""

    @abc.abstractmethod
    def run(self, actors: Sequence[ActorRef], programs: Mapping[str, Mapping[str, Any]], spec: RunSpec,
            run_dir: Path) -> RunResult:
        """Run one program per controlled actor and write rows into ``run_dir``."""

    @abc.abstractmethod
    def reset_between_runs(self) -> None:
        """Return to the state after ``open_scene`` (never called during a run)."""

    @abc.abstractmethod
    def close(self) -> None: ...

    #: The call ledger of the current run; the harness attaches it with :meth:`use_ledger`.
    ledger: Any = None

    def use_ledger(self, ledger: Any) -> None:
        """Record every engine call of the next run into ``ledger`` (a ``CallLedger``)."""
        self.ledger = ledger

    @classmethod
    def audit_policy(cls) -> Any:
        """The ``AuditPolicy`` that ``openbeam-v2e audit`` replays this connector's ledger against."""
        raise NotImplementedError(f"{cls.__name__} declares no audit policy")

    def run_audit_policy(self) -> Any:
        """The policy for this instance's runs (may add snippets of an explicitly enabled host)."""
        return type(self).audit_policy()

    def __enter__(self) -> "Backend":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


class Lockstep(abc.ABC):
    """Optional mixin; required when ``CLIENT_LOCKSTEP`` is among the connector's controller hosts."""

    @abc.abstractmethod
    def step(self, steps: int) -> None: ...

    @abc.abstractmethod
    def read_state(self, actor: ActorRef) -> StateRow: ...

    @abc.abstractmethod
    def actuate(self, actor: ActorRef, cmd: Mapping[str, float]) -> None: ...

    @abc.abstractmethod
    def capture(self, camera: CameraSpec) -> Frame: ...
