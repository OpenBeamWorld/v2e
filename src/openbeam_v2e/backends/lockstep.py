# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""The client-lockstep controller host shared by connectors that implement ``Lockstep``.

Controllers run in Python between explicit physics steps: every control tick the host reads each
actor's state, asks each controlled actor's controller for a command, checks every channel id
and value against the plan and the connector's declared channels, sends the command, and steps.
Rows in the ``v2e-runlog/1`` format are written at the telemetry rate, with simulated time from
the step count.

Controllers come from ``openbeam_v2e.controllers`` when that package provides
``build_controller(program, channels)``. Without it, this module offers the two models every
connector test needs: ``passive`` (no commands) and ``schedule`` (timed commands, held or
linearly interpolated, with toggle presses at their entry times). The bundled ``bench.cruise``
model is loaded lazily so generated benchmark programs also run through CLI and MCP tools.
Any other model is refused.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from openbeam_v2e.backends.base import (
    SCHEMA_RUNLOG,
    ActorRef,
    ActuatorChannel,
    BackendError,
    ChannelKind,
    RefusedCall,
    RunSpec,
    StateRow,
    check_program,
)

__all__ = ["Controller", "ScheduleController", "PassiveController", "default_controller_factory",
           "LockstepOutcome", "run_lockstep", "RowWriter", "EventDetector"]


class Controller(Protocol):
    def command(self, t: float, observation: Mapping[str, StateRow]) -> Mapping[str, float]: ...


class PassiveController:
    def command(self, t: float, observation: Mapping[str, StateRow]) -> Mapping[str, float]:
        return {}


class ScheduleController:
    """Timed commands: ``{"schedule": [{"t": 0.0, "cmd": {...}}, ...], "interpolation": "hold"}``.

    Continuous and discrete channels hold the value of the latest entry (or interpolate linearly
    with ``"interpolation": "linear"``). Toggle channels are presses: an entry value of 1 presses
    once at that entry's time.
    """

    def __init__(self, params: Mapping[str, Any], channels: Sequence[ActuatorChannel]) -> None:
        entries = params.get("schedule", [])
        if not isinstance(entries, list) or not entries:
            raise ValueError("a schedule program needs a non-empty 'schedule' list")
        self.entries = sorted(((float(e["t"]), dict(e.get("cmd", {}))) for e in entries), key=lambda kv: kv[0])
        self.interpolation = params.get("interpolation", "hold")
        if self.interpolation not in ("hold", "linear"):
            raise ValueError("schedule interpolation must be 'hold' or 'linear'")
        self.kinds = {c.id: c.kind for c in channels}
        for _, cmd in self.entries:
            unknown = sorted(set(cmd) - set(self.kinds))
            if unknown:
                raise ValueError(f"schedule commands undeclared channel(s) {unknown}")
        self._pressed: set[int] = set()

    def command(self, t: float, observation: Mapping[str, StateRow]) -> Mapping[str, float]:
        out: dict[str, float] = {}
        for channel, kind in self.kinds.items():
            if kind is ChannelKind.TOGGLE:
                continue
            value = self._value(channel, t)
            if value is not None:
                out[channel] = value
        for index, (time_s, cmd) in enumerate(self.entries):
            if time_s <= t + 1e-12 and index not in self._pressed:
                for channel, value in cmd.items():
                    if self.kinds[channel] is ChannelKind.TOGGLE and float(value) == 1.0:
                        out[channel] = 1.0
                self._pressed.add(index)
        return out

    def _value(self, channel: str, t: float) -> float | None:
        before = [(ts, cmd[channel]) for ts, cmd in self.entries if channel in cmd and ts <= t + 1e-12]
        if not before:
            return None
        t0, v0 = before[-1]
        if self.interpolation == "linear" and self.kinds[channel] is ChannelKind.CONTINUOUS:
            after = [(ts, cmd[channel]) for ts, cmd in self.entries if channel in cmd and ts > t + 1e-12]
            if after:
                t1, v1 = after[0]
                return float(v0) + (float(v1) - float(v0)) * (t - t0) / (t1 - t0)
        return float(v0)


ControllerFactory = Callable[[Mapping[str, Any], Sequence[ActuatorChannel]], Controller]


def default_controller_factory(program: Mapping[str, Any], channels: Sequence[ActuatorChannel]) -> Controller:
    """Build a controller for a ``v2e.program/1`` program (see the module docstring)."""
    check_program(program)
    try:
        from openbeam_v2e import controllers as controllers_pkg
    except ImportError:   # pragma: no cover - the package always exists in this distribution
        controllers_pkg = None
    build = getattr(controllers_pkg, "build_controller", None)
    if callable(build):
        return build(program, channels)
    model = program["model"]
    if model == "passive":
        return PassiveController()
    if model == "schedule":
        return ScheduleController(program.get("params", {}), channels)
    if model == "bench.cruise":
        from openbeam_v2e.bench.drive import CruiseController

        return CruiseController(program, channels)
    raise BackendError(f"controller model {model!r} needs the reference driver models in openbeam_v2e.controllers")


@dataclass
class EventDetector:
    """Contact and rest events from state rows.

    Names follow the benchmark truth convention (``<event>.<actor>``), are unique per actor and use
    only the characters world-package event names allow: ``first_contact.<actor>`` (first contact
    with anything), ``first_contact.<actor>.<other>`` (first contact with one static part or actor)
    and ``rest.<actor>`` (speed and rate below thresholds for ``rest_hold_s`` after the actor has
    moved; the event time is when it became still).
    """

    rest_speed_mps: float = 0.05
    rest_rate_rad_s: float = 0.05
    rest_hold_s: float = 0.5
    events: dict[str, float] = field(default_factory=dict)
    _still_since: dict[str, float] = field(default_factory=dict)
    _moved: set = field(default_factory=set)

    def update(self, t: float, states: Mapping[str, StateRow]) -> list[dict[str, str]]:
        """Record new events at time ``t``; return them as ``v2e-runlog/1`` event objects."""
        new: list[dict[str, str]] = []
        for actor, state in sorted(states.items()):
            for other in state.contacts:
                for key, item in ((f"first_contact.{actor}", {"actor": actor}),
                                  (f"first_contact.{actor}.{other}", {"actor": actor, "other": other})):
                    if key not in self.events:
                        self.events[key] = t
                        new.append({"name": key, **item})
            speed = math.hypot(*state.v) if state.v else 0.0
            rate = math.hypot(*state.w) if state.w else 0.0
            if speed > 5 * self.rest_speed_mps:
                self._moved.add(actor)
            if speed < self.rest_speed_mps and rate < self.rest_rate_rad_s:
                since = self._still_since.setdefault(actor, t)
                key = f"rest.{actor}"
                if actor in self._moved and key not in self.events and t - since >= self.rest_hold_s - 1e-9:
                    self.events[key] = since
                    new.append({"name": key, "actor": actor})
            else:
                self._still_since.pop(actor, None)
        return new


def _vec(values: Sequence[float]) -> list[float]:
    return [float(v) for v in values]


class RowWriter:
    """Stream a ``v2e-runlog/1`` file: one header line, then one row per line.

    Rows are in the Z-up level frame with simulated time from spawn. An actor state carries ``p``
    and, when the connector observes attitude, ``q``; ``v`` and ``w`` when known.
    """

    def __init__(self, path: Path, *, actors: Sequence[str], backend: str, engine: str, regime: str,
                 dt_s: float | None, t0_s: float | None, notes: str = "",
                 attitude_observed: bool = True) -> None:
        self.path = Path(path)
        self._fh = self.path.open("w", encoding="utf-8")
        self.rows = 0
        self._last_t: float | None = None
        header: dict[str, Any] = {"schema": SCHEMA_RUNLOG, "frame": "z_up_level", "actors": sorted(actors),
                                  "backend": backend, "engine": engine, "regime": regime}
        if dt_s is not None:
            header["dt_s"] = dt_s
        if t0_s is not None:
            header["t0_s"] = t0_s
        if notes:
            header["notes"] = notes
        if not attitude_observed:
            header["attitude_observed"] = False
        self._write(header)

    def _write(self, obj: Mapping[str, Any]) -> None:
        self._fh.write(json.dumps(obj, sort_keys=True, separators=(",", ":")) + "\n")

    def write(self, t: float, states: Mapping[str, StateRow], cmd: Mapping[str, Mapping[str, float]] | None = None,
              events: Sequence[Mapping[str, str]] = ()) -> None:
        if self._last_t is not None and t <= self._last_t:
            raise ValueError(f"row time {t} does not increase")
        self._last_t = t
        actors = {}
        for actor, s in sorted(states.items()):
            state: dict[str, Any] = {"p": _vec(s.p)}
            for key in ("q", "v", "w"):
                value = getattr(s, key)
                if value is not None:
                    state[key] = _vec(value)
            actors[actor] = state
        row: dict[str, Any] = {"t": t, "actors": actors}
        commands = {a: dict(sorted(c.items())) for a, c in sorted((cmd or {}).items()) if c}
        if commands:
            row["cmd"] = commands
        if events:
            row["events"] = [dict(e) for e in events]
        self._write(row)
        self.rows += 1

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> "RowWriter":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


@dataclass(frozen=True)
class LockstepOutcome:
    rows: Path
    events: dict[str, float]
    sim_s: float
    steps: int
    rows_written: int


def run_lockstep(backend: Any, actors: Sequence[ActorRef], programs: Mapping[str, Mapping[str, Any]],
                 spec: RunSpec, run_dir: Path, *, plan_channels: Mapping[str, Sequence[str]],
                 channels: Sequence[ActuatorChannel], physics_dt_s: float,
                 controller_factory: ControllerFactory | None = None, engine: str = "",
                 t0_s: float | None = 0.0) -> LockstepOutcome:
    """Drive ``backend`` (a ``Lockstep`` connector) for ``spec.duration_s`` and write ``rows.jsonl``."""
    factory = controller_factory or default_controller_factory
    by_id = {c.id: c for c in channels}
    controllers: dict[str, Controller] = {}
    refs = {a.actor: a for a in actors}
    for actor, program in programs.items():
        if actor not in refs:
            raise RefusedCall(f"program for {actor!r}, which was not spawned")
        allowed = [by_id[c] for c in plan_channels.get(actor, ()) if c in by_id]
        controllers[actor] = factory(program, allowed)

    dt = float(physics_dt_s)
    total = max(1, int(round(spec.duration_s / dt)))
    control_every = max(1, int(round(1.0 / ((spec.control_hz or 100.0) * dt))))
    row_every = max(1, int(round(1.0 / ((spec.telemetry_hz or 60.0) * dt))))
    detector = EventDetector()
    last_cmd: dict[str, dict[str, float]] = {a: {} for a in controllers}
    rows_path = Path(run_dir) / "rows.jsonl"
    step = 0
    with RowWriter(rows_path, actors=[a.actor for a in actors], backend=getattr(backend, "name", ""), engine=engine,
                   regime=spec.regime, dt_s=dt, t0_s=t0_s) as writer:
        while True:
            t = round(step * dt, 9)
            control_tick = step % control_every == 0 and step < total
            row_tick = step % row_every == 0 or step == total
            states = {a.actor: backend.read_state(a) for a in actors} if (control_tick or row_tick) else {}
            new_events = detector.update(t, states) if row_tick else []
            if control_tick:
                for actor, controller in controllers.items():
                    raw = dict(controller.command(t, states))
                    cmd = {}
                    for channel_id, value in raw.items():
                        if channel_id not in plan_channels.get(actor, ()):
                            raise RefusedCall(f"{actor}: channel {channel_id!r} is not in the plan")
                        cmd[channel_id] = by_id[channel_id].check(value)
                    sent = {k: v for k, v in cmd.items()
                            if by_id[k].kind is ChannelKind.TOGGLE or last_cmd[actor].get(k) != v}
                    if sent:
                        backend.actuate(refs[actor], sent)
                    for k, v in cmd.items():
                        if by_id[k].kind is not ChannelKind.TOGGLE:
                            last_cmd[actor][k] = v
            if row_tick:
                writer.write(t, states, {a: dict(c) for a, c in last_cmd.items()}, new_events)
            if step >= total:
                break
            nxt = min(total, (step // control_every + 1) * control_every, (step // row_every + 1) * row_every)
            backend.step(nxt - step)
            step = nxt
    return LockstepOutcome(rows_path, dict(detector.events), round(total * dt, 9), total, writer.rows)
