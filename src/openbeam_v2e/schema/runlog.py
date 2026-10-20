# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""``v2e-runlog/1``: the canonical row format every connector writes.

A run log is JSON Lines. The first line is a header; every further line is one row::

    {"schema": "v2e-runlog/1", "frame": "z_up_level", "actors": ["car_1", "crate"],
     "backend": "fake", "regime": "stepped", "dt_s": 0.01}
    {"t": 0.0, "actors": {"car_1": {"p": [0, 0, 0.3], "q": [0, 0, 0, 1]}, ...}}
    {"t": 0.01, "frame": null, "phase": "approach",
     "actors": {"car_1": {"p": [...], "q": [...], "v": [...], "w": [...]}},
     "cmd": {"car_1": {"drive.throttle": 0.4}}, "events": [{"name": "contact", "actor": "car_1"}]}

Rows are in the Z-up level frame, metres and simulated seconds since spawn. ``p`` is the
position of the actor's reference point, ``q`` its ``[x, y, z, w]`` attitude; ``v`` and ``w``
(linear and angular velocity, world frame) and ``nodes`` (named points) are optional.
Connectors that cannot observe attitude must set ``attitude_observed`` to false in
the header and omit ``q`` rather than inventing an orientation.
Command channels are scoped by actor. Readers for other row formats are plugins
(``openbeam_v2e.runlog_readers`` entry point); this module reads only this format.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from openbeam_v2e.schema import jsonio
from openbeam_v2e.schema.common import defs
from openbeam_v2e.schema.validate import Issue, SchemaError, validate

__all__ = [
    "SCHEMA_ID",
    "HEADER_SCHEMA",
    "ROW_SCHEMA",
    "ActorState",
    "Row",
    "RunHeader",
    "Runlog",
    "validate_runlog",
    "read_runlog",
    "write_runlog",
]

SCHEMA_ID = "v2e-runlog/1"


def _header_schema() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://github.com/OpenBeamWorld/v2e/schemas/v2e-runlog-1.header.json",
        "title": "V2E run log header (v2e-runlog/1)",
        "type": "object",
        "required": ["schema", "actors"],
        "properties": {
            "schema": {"const": SCHEMA_ID},
            "frame": {"const": "z_up_level"},
            "actors": {"type": "array", "items": {"$ref": "#/$defs/id"}, "minItems": 1, "uniqueItems": True},
            "backend": {"type": "string"},
            "engine": {"type": "string"},
            "regime": {"type": "string"},
            "dt_s": {"type": "number", "exclusiveMinimum": 0},
            "t0_s": {"type": ["number", "null"]},
            "scene_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
            "notes": {"type": "string"},
            "attitude_observed": {"type": "boolean"},
        },
        "additionalProperties": False,
        "$defs": defs(),
    }


def _row_schema(attitude_observed: bool = True) -> dict[str, Any]:
    d = defs()
    d["state"] = {
        "type": "object",
        "required": ["p", "q"] if attitude_observed else ["p"],
        "properties": {
            "p": {"$ref": "#/$defs/vec3"},
            "q": {"$ref": "#/$defs/quat_xyzw"},
            "v": {"$ref": "#/$defs/vec3"},
            "w": {"$ref": "#/$defs/vec3"},
            "nodes": {"type": "object", "additionalProperties": {"$ref": "#/$defs/vec3"}},
        },
        "additionalProperties": False,
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://github.com/OpenBeamWorld/v2e/schemas/v2e-runlog-1.row.json",
        "title": "V2E run log row (v2e-runlog/1)",
        "type": "object",
        "required": ["t", "actors"],
        "properties": {
            "t": {"type": "number", "minimum": 0},
            "frame": {"type": ["integer", "null"], "minimum": 0},
            "phase": {"type": ["string", "null"]},
            "actors": {"type": "object", "propertyNames": {"$ref": "#/$defs/id"},
                       "additionalProperties": {"$ref": "#/$defs/state"}},
            "cmd": {
                "type": "object",
                "propertyNames": {"$ref": "#/$defs/id"},
                "additionalProperties": {"type": "object", "additionalProperties": {"type": "number"}},
            },
            "broken": {"type": "array", "items": {"type": "string"}},
            "events": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["name"],
                    "properties": {"name": {"type": "string", "minLength": 1}, "actor": {"$ref": "#/$defs/id"},
                                   "other": {"type": "string"}},
                    "additionalProperties": False,
                },
            },
        },
        "additionalProperties": False,
        "$defs": d,
    }


HEADER_SCHEMA: dict[str, Any] = _header_schema()
ROW_SCHEMA: dict[str, Any] = _row_schema()
_POSITION_ONLY_ROW_SCHEMA: dict[str, Any] = _row_schema(attitude_observed=False)


@dataclass(frozen=True)
class ActorState:
    p: tuple[float, float, float]
    q: tuple[float, float, float, float] | None = None
    v: tuple[float, float, float] | None = None
    w: tuple[float, float, float] | None = None
    nodes: Mapping[str, tuple[float, float, float]] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {"p": list(self.p)}
        if self.q is not None:
            out["q"] = list(self.q)
        if self.v is not None:
            out["v"] = list(self.v)
        if self.w is not None:
            out["w"] = list(self.w)
        if self.nodes:
            out["nodes"] = {k: list(v) for k, v in self.nodes.items()}
        return out


@dataclass(frozen=True)
class Row:
    t: float
    actors: Mapping[str, ActorState]
    frame: int | None = None
    phase: str | None = None
    cmd: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    broken: tuple[str, ...] = ()
    events: tuple[Mapping[str, str], ...] = ()

    @classmethod
    def parse(cls, raw: Mapping[str, Any]) -> "Row":
        actors = {
            k: ActorState(tuple(v["p"]), tuple(v["q"]) if "q" in v else None, tuple(v["v"]) if "v" in v else None,
                          tuple(v["w"]) if "w" in v else None,
                          {n: tuple(p) for n, p in v.get("nodes", {}).items()})
            for k, v in raw["actors"].items()
        }
        return cls(float(raw["t"]), actors, raw.get("frame"), raw.get("phase"),
                   {a: dict(c) for a, c in raw.get("cmd", {}).items()}, tuple(raw.get("broken", ())),
                   tuple(dict(e) for e in raw.get("events", ())))

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {"t": self.t}
        if self.frame is not None:
            out["frame"] = self.frame
        if self.phase is not None:
            out["phase"] = self.phase
        out["actors"] = {k: v.to_json() for k, v in self.actors.items()}
        if self.cmd:
            out["cmd"] = {a: dict(c) for a, c in self.cmd.items()}
        if self.broken:
            out["broken"] = list(self.broken)
        if self.events:
            out["events"] = [dict(e) for e in self.events]
        return out


@dataclass(frozen=True)
class RunHeader:
    actors: tuple[str, ...]
    backend: str = ""
    engine: str = ""
    regime: str = ""
    dt_s: float | None = None
    t0_s: float | None = None
    scene_sha256: str | None = None
    notes: str = ""
    attitude_observed: bool = True

    @classmethod
    def parse(cls, raw: Mapping[str, Any]) -> "RunHeader":
        return cls(tuple(raw["actors"]), raw.get("backend", ""), raw.get("engine", ""), raw.get("regime", ""),
                    raw.get("dt_s"), raw.get("t0_s"), raw.get("scene_sha256"), raw.get("notes", ""),
                    raw.get("attitude_observed", True))

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {"schema": SCHEMA_ID, "frame": "z_up_level", "actors": list(self.actors)}
        for key in ("backend", "engine", "regime", "notes"):
            if getattr(self, key):
                out[key] = getattr(self, key)
        if self.dt_s is not None:
            out["dt_s"] = self.dt_s
        if self.t0_s is not None:
            out["t0_s"] = self.t0_s
        if self.scene_sha256:
            out["scene_sha256"] = self.scene_sha256
        if not self.attitude_observed:
            out["attitude_observed"] = False
        return out


@dataclass(frozen=True)
class Runlog:
    header: RunHeader
    rows: tuple[Row, ...]

    def times(self) -> list[float]:
        return [row.t for row in self.rows]

    def events(self) -> list[tuple[float, Mapping[str, str]]]:
        """``(t, event)`` for every event any row reports."""
        return [(row.t, event) for row in self.rows for event in row.events]


def _row_issues(index: int, raw: Any, actors: set[str], previous_t: float | None,
               attitude_observed: bool = True) -> list[Issue]:
    where = f"/rows/{index}"
    schema = ROW_SCHEMA if attitude_observed else _POSITION_ONLY_ROW_SCHEMA
    issues = [Issue(f"{where}{i.path}", i.message) for i in validate(raw, schema)]
    if issues:
        return issues
    t = float(raw["t"])
    if previous_t is not None and t <= previous_t:
        issues.append(Issue(f"{where}/t", f"time {t} does not increase (previous {previous_t})"))
    for aid, state in raw["actors"].items():
        if aid not in actors:
            issues.append(Issue(f"{where}/actors/{aid}", "actor is not declared in the header"))
        if "q" in state:
            norm = math.sqrt(sum(c * c for c in state["q"]))
            if abs(norm - 1.0) > 1e-3:
                issues.append(Issue(f"{where}/actors/{aid}/q", f"quaternion norm {norm:.6f} is not 1"))
    for aid in raw.get("cmd", {}):
        if aid not in actors:
            issues.append(Issue(f"{where}/cmd/{aid}", "commands an actor the header does not declare"))
    return issues


def validate_runlog(header: Any, rows: Sequence[Any]) -> list[Issue]:
    """Issues of a parsed header and its raw rows (empty when valid)."""
    issues = [Issue(f"/header{i.path}", i.message) for i in validate(header, HEADER_SCHEMA)]
    if issues:
        return issues
    actors = set(header["actors"])
    previous: float | None = None
    for index, raw in enumerate(rows):
        issues.extend(_row_issues(index, raw, actors, previous, header.get("attitude_observed", True)))
        if isinstance(raw, Mapping) and isinstance(raw.get("t"), (int, float)):
            previous = float(raw["t"])
    if not rows:
        issues.append(Issue("/rows", "a run log needs at least one row"))
    return issues


def _iter_lines(source: str | Path | Iterable[str]) -> Iterator[str]:
    is_text = isinstance(source, str) and ("\n" in source or source.lstrip().startswith("{"))
    if isinstance(source, Path) or (isinstance(source, str) and not is_text):
        with open(source, encoding="utf-8") as handle:
            yield from handle
    elif isinstance(source, str):
        yield from io.StringIO(source)
    else:
        yield from source


def read_runlog(source: str | Path | Iterable[str], check: bool = True) -> Runlog:
    """Read a run log from a path, JSON Lines text, or an iterable of lines."""
    header: Any = None
    rows: list[Any] = []
    for number, line in enumerate(_iter_lines(source), start=1):
        if not line.strip():
            continue
        try:
            value = jsonio.loads(line)
        except jsonio.JsonError as exc:
            raise SchemaError([Issue(f"/line/{number}", str(exc))], "run log") from None
        if header is None:
            header = value
        else:
            rows.append(value)
    if header is None:
        raise SchemaError([Issue("/", "empty run log")], "run log")
    if check:
        issues = validate_runlog(header, rows)
        if issues:
            raise SchemaError(issues, "run log")
    return Runlog(RunHeader.parse(header), tuple(Row.parse(r) for r in rows))


def write_runlog(runlog: Runlog, path: str | Path) -> Path:
    """Write a run log as JSON Lines (header first)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as handle:
        handle.write(jsonio.dumps(runlog.header.to_json(), indent=None))
        for row in runlog.rows:
            handle.write(jsonio.dumps(row.to_json(), indent=None))
    return target
