# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""The benchmark's driver: a staged cruise controller for client-lockstep connectors.

Benchmark truth needs a vehicle that reaches a design speed on a straight line and then does
something simple when an event happens (coast after an impact, brake after landing). This module
provides that as a ``v2e.program/1`` model, ``bench.cruise``, served by a controller factory that
a lockstep connector accepts (``FakeBackend(controller_factory=controller_factory)``); every
other model is passed on to the default factory.

The program carries parameters only::

    {"schema": "v2e.program/1", "actor": "car", "model": "bench.cruise",
     "params": {"speed_mps": 11.1, "line_x_m": 0.0, "line_y_m": 0.0, "heading_deg": 0.0,
                "stages": [{"mode": "cruise", "until": {"contact": "barrier"}},
                           {"mode": "fixed", "cmd": {"drive.throttle": 0.0, "drive.brake": 0.0}}]}}

Stages run in order. ``cruise`` holds the speed with throttle and brake (proportional-integral on
forward speed) and the line with steering; ``fixed`` sends constant commands (steering still
holds the line unless the stage sets ``drive.steer``). A stage ends when its ``until`` condition
is met: ``{"contact": name}`` (the actor touches ``name``), ``{"leave": name}`` (it stops touching
``name`` after having touched it), ``{"airborne_s": s}`` (it has touched nothing for ``s``
seconds) or ``{"time_s": t}``. Contacts come from the connector's state
rows. Only the declared driver channels are commanded.

The line passes through ``(line_x_m, line_y_m)`` in the direction ``heading_deg``.
Both coordinates default to zero; older +X programs using only ``line_y_m`` are unchanged.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from openbeam_v2e.backends.base import ActuatorChannel, StateRow, check_program
from openbeam_v2e.backends.lockstep import default_controller_factory

__all__ = ["CRUISE_MODEL", "CruiseController", "controller_factory", "cruise_program"]

CRUISE_MODEL = "bench.cruise"

# Gains for a mid-size car on the fake connector's wheel model, set once by hand on a straight
# line; not fitted to any scene.
_KP_SPEED = 0.6
_KI_SPEED = 0.25
_KP_BRAKE = 0.5        # brake per unit of negative speed demand
_INTEGRATE_BAND = 1.5  # m/s: integrate the speed error only near the set point
_INTEGRAL_MAX = 3.0
_K_GRADE = 2.0         # throttle per unit of sin(nose-up pitch): grade feed-forward
_K_LATERAL = 0.12      # steer per metre of lateral offset
_K_HEADING = 0.9       # steer per radian of heading error
_K_YAW_RATE = 0.12     # steer per rad/s of yaw rate


def _stage_mode(stage: Mapping[str, Any]) -> str:
    return str(stage["mode"]) if "mode" in stage else "cruise"


def _yaw_from_quat(q: Sequence[float]) -> float:
    x, y, z, w = q
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _forward(q: Sequence[float]) -> tuple[float, float, float]:
    x, y, z, w = q
    return (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + w * z), 2.0 * (x * z - w * y))


class CruiseController:
    """Staged speed-and-line controller (see the module docstring)."""

    def __init__(self, program: Mapping[str, Any], channels: Sequence[ActuatorChannel]) -> None:
        params = dict(program.get("params", {}))
        self.actor = str(program["actor"])
        self.speed = float(params["speed_mps"])
        self.line_x = float(params.get("line_x_m", 0.0))
        self.line_y = float(params.get("line_y_m", 0.0))
        self.heading = math.radians(float(params.get("heading_deg", 0.0)))
        self.stages = list(params.get("stages", [{"mode": "cruise"}]))
        for stage in self.stages:
            if _stage_mode(stage) not in ("cruise", "fixed"):
                raise ValueError(f"unknown cruise stage mode {_stage_mode(stage)!r}")
        self.declared = {c.id for c in channels}
        self.stage = 0
        self.touched: set[str] = set()
        self.integral = 0.0
        self.last_t: float | None = None
        self.stage_log: list[tuple[int, float]] = [(0, 0.0)]
        self.free_since: float | None = None

    def _done(self, until: Mapping[str, Any] | None, t: float, contacts: set[str]) -> bool:
        if not until:
            return False
        if "contact" in until:
            return str(until["contact"]) in contacts
        if "leave" in until:
            name = str(until["leave"])
            return name in self.touched and name not in contacts
        if "airborne_s" in until:
            # Touching nothing for the given time: wheels bouncing over a ramp do not count.
            return self.free_since is not None and t - self.free_since >= float(until["airborne_s"]) - 1e-9
        if "time_s" in until:
            return t >= float(until["time_s"])
        raise ValueError(f"unknown stage condition {dict(until)!r}")

    def command(self, t: float, observation: Mapping[str, StateRow]) -> Mapping[str, float]:
        me = observation.get(self.actor)
        if me is None or me.q is None:
            return {}
        contacts = set(me.contacts)
        if contacts:
            self.free_since = None
        elif self.free_since is None:
            self.free_since = t
        while self.stage < len(self.stages) - 1 and self._done(self.stages[self.stage].get("until"), t, contacts):
            self.stage += 1
            self.integral = 0.0
            self.stage_log.append((self.stage, t))
        self.touched |= contacts
        dt = 0.0 if self.last_t is None else max(0.0, t - self.last_t)
        self.last_t = t
        stage = self.stages[self.stage]

        fwd = _forward(me.q)
        v = me.v or (0.0, 0.0, 0.0)
        v_fwd = sum(a * b for a, b in zip(fwd, v))
        yaw = _yaw_from_quat(me.q)
        err_heading = math.atan2(math.sin(yaw - self.heading), math.cos(yaw - self.heading))
        # Signed perpendicular distance to the requested line, including near-vertical headings.
        offset = (-math.sin(self.heading) * (me.p[0] - self.line_x)
                  + math.cos(self.heading) * (me.p[1] - self.line_y))
        yaw_rate = me.w[2] if me.w else 0.0
        steer = _K_LATERAL * offset + _K_HEADING * err_heading + _K_YAW_RATE * yaw_rate
        cmd: dict[str, float] = {"drive.steer": max(-1.0, min(1.0, steer))}

        if _stage_mode(stage) == "cruise":
            err = self.speed - v_fwd
            if abs(err) < _INTEGRATE_BAND:  # conditional integration: no wind-up while accelerating
                self.integral = max(-_INTEGRAL_MAX, min(_INTEGRAL_MAX, self.integral + err * dt))
            # fwd[2] is the sine of the body's nose-up pitch: feed the grade forward.
            u = _KP_SPEED * err + _KI_SPEED * self.integral + _K_GRADE * fwd[2]
            if u >= 0.0:
                cmd["drive.throttle"] = min(1.0, u)
                cmd["drive.brake"] = 0.0
            else:
                cmd["drive.throttle"] = 0.0
                cmd["drive.brake"] = min(1.0, -_KP_BRAKE * u)
        else:
            for channel, value in dict(stage.get("cmd", {})).items():
                cmd[str(channel)] = float(value)
        return {k: v for k, v in cmd.items() if k in self.declared}


def controller_factory(program: Mapping[str, Any], channels: Sequence[ActuatorChannel]) -> Any:
    """Serve ``bench.cruise``; pass every other model to the default factory."""
    check_program(program)
    if program["model"] == CRUISE_MODEL:
        return CruiseController(program, channels)
    return default_controller_factory(program, channels)


def cruise_program(actor: str, speed_mps: float, line_y_m: float, stages: Sequence[Mapping[str, Any]],
                   heading_deg: float = 0.0, *, line_x_m: float = 0.0) -> dict[str, Any]:
    """A ``bench.cruise`` program for ``actor``."""
    return {
        "schema": "v2e.program/1",
        "actor": actor,
        "model": CRUISE_MODEL,
        "params": {"speed_mps": float(speed_mps), "line_x_m": float(line_x_m),
                   "line_y_m": float(line_y_m), "heading_deg": float(heading_deg),
                   "stages": [dict(s) for s in stages]},
    }
