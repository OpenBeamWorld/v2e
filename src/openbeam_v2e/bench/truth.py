# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""Ground truth of a benchmark case: dense pose tracks, events detected from geometry, metrics.

A connector's run gives poses at its own row rate. This module turns those rows into the truth
a benchmark needs:

* :class:`Track`: one object's rigid pose track (hull origin and body-to-world rotation) at
  dense times, resampled to video frames on request;
* :func:`detect_events`: the case's events (first contact, take-off, landing, rest, ...) found
  from the tracks and the static geometry alone, so the same detector works on any connector's
  rows and on reconstructed tracks;
* :func:`truth_metrics`: the per-case truth values the scorer compares against (speeds at
  events, apex height, airborne time, cumulative roll, rebound ratios, rest pose).

Geometry is convex: an object is a union of convex parts (see
:mod:`openbeam_v2e.reconstruct.hulls`), static geometry is a list of convex parts in world
coordinates. A point's signed distance to a convex part is approximated by its largest
half-space distance, which is exact inside the part and near its faces, where contacts happen.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

__all__ = [
    "StaticPart",
    "Track",
    "EventTimes",
    "part_planes",
    "signed_distance_to_part",
    "clearance",
    "detect_events",
    "truth_metrics",
    "event_frames",
]

CONTACT_TOL_M = 0.03
"""Clearance below which two shapes count as touching (rigid contact keeps them a hair apart)."""

REST_SPEED_MPS = 0.05
REST_RATE_RAD_S = 0.05
REST_HOLD_S = 0.5
AIRBORNE_HOLD_S = 0.1
"""An object counts as airborne once it has touched nothing for this long (wheels bounce on ramps)."""


def part_planes(vertices: np.ndarray) -> np.ndarray:
    """Outward half-spaces ``n . x <= d`` of the convex hull of ``vertices`` as rows ``[n, d]``."""
    hull = ConvexHull(np.asarray(vertices, float))
    eq = hull.equations  # n . x + c <= 0 with unit outward n
    planes = np.concatenate([eq[:, :3], -eq[:, 3:4]], axis=1)
    # Drop duplicate facets (triangulated faces share a plane).
    rounded = np.round(planes, 9)
    _, keep = np.unique(rounded, axis=0, return_index=True)
    return planes[np.sort(keep)]


@dataclass(frozen=True)
class StaticPart:
    """One convex piece of static geometry in world coordinates."""

    static_id: str
    vertices: np.ndarray
    planes: np.ndarray = field(repr=False, default=None)

    def __post_init__(self) -> None:
        v = np.asarray(self.vertices, float).reshape(-1, 3)
        object.__setattr__(self, "vertices", v)
        if self.planes is None:
            object.__setattr__(self, "planes", part_planes(v))

    @property
    def top_z(self) -> float:
        return float(self.vertices[:, 2].max())


def signed_distance_to_part(points: np.ndarray, planes: np.ndarray) -> np.ndarray:
    """Approximate signed distance (negative inside) of ``points`` (N, 3) to a convex part."""
    p = np.atleast_2d(np.asarray(points, float))
    return np.max(p @ planes[:, :3].T - planes[:, 3][None, :], axis=1)


def clearance(vertices_a: Sequence[np.ndarray], part_b: StaticPart) -> float:
    """Smallest gap between an object (list of convex part vertex arrays) and a static part.

    Uses the object's vertices against the static part and the static part's vertices against
    each object part; negative means overlap.
    """
    best = math.inf
    for va in vertices_a:
        best = min(best, float(signed_distance_to_part(va, part_b.planes).min()))
        planes_a = part_planes(va)
        best = min(best, float(signed_distance_to_part(part_b.vertices, planes_a).min()))
    return best


@dataclass
class Track:
    """A rigid pose track: hull-origin positions (N, 3) and body-to-world rotations (N, 3, 3)."""

    object_id: str
    t: np.ndarray
    position: np.ndarray
    R_wb: np.ndarray
    hull_parts: tuple[np.ndarray, ...]  # convex part vertices in the body frame
    object_class: str = ""

    def __post_init__(self) -> None:
        self.t = np.asarray(self.t, float)
        self.position = np.asarray(self.position, float).reshape(-1, 3)
        self.R_wb = np.asarray(self.R_wb, float).reshape(-1, 3, 3)
        if not (len(self.t) == len(self.position) == len(self.R_wb)):
            raise ValueError("track times, positions and rotations must have the same length")

    def __len__(self) -> int:
        return len(self.t)

    def world_parts(self, i: int) -> list[np.ndarray]:
        R, p = self.R_wb[i], self.position[i]
        return [v @ R.T + p for v in self.hull_parts]

    def centre(self) -> np.ndarray:
        """World positions of the hull's bounding-box centre (N, 3)."""
        allv = np.concatenate(self.hull_parts)
        c_body = 0.5 * (allv.min(axis=0) + allv.max(axis=0))
        return self.position + np.einsum("nij,j->ni", self.R_wb, c_body)

    def lowest_z(self) -> np.ndarray:
        allv = np.concatenate(self.hull_parts)
        return np.array([float((allv @ R.T + p)[:, 2].min()) for R, p in zip(self.R_wb, self.position)])

    def velocity(self) -> np.ndarray:
        """Centre velocity by central differences (N, 3)."""
        c = self.centre()
        if len(self.t) < 2:
            return np.zeros_like(c)
        return np.gradient(c, self.t, axis=0)

    def angular_rate(self) -> np.ndarray:
        """World-frame angular velocity magnitude (N,) from successive rotations."""
        if len(self.t) < 2:
            return np.zeros(len(self.t))
        rots = Rotation.from_matrix(self.R_wb)
        rel = rots[1:] * rots[:-1].inv()
        w = rel.magnitude() / np.maximum(np.diff(self.t), 1e-12)
        return np.concatenate([w[:1], 0.5 * (w[1:] + w[:-1]), w[-1:]]) if len(w) > 1 else np.repeat(w, 2)

    def roll_pitch_yaw_deg(self) -> np.ndarray:
        """Intrinsic Z-Y-X angles (yaw, pitch, roll) unwrapped over time, returned as (N, 3) roll, pitch, yaw."""
        zyx = Rotation.from_matrix(self.R_wb).as_euler("ZYX")
        zyx = np.unwrap(zyx, axis=0)
        return np.degrees(zyx[:, ::-1])

    def cumulative_roll_deg(self) -> np.ndarray:
        """Signed integral of the body roll rate about the body x axis (N,)."""
        if len(self.t) < 2:
            return np.zeros(len(self.t))
        rots = Rotation.from_matrix(self.R_wb)
        rel = rots[:-1].inv() * rots[1:]  # body-frame increments
        dx = rel.as_rotvec()[:, 0]
        return np.degrees(np.concatenate([[0.0], np.cumsum(dx)]))

    def resample(self, times: np.ndarray) -> "Track":
        """Track at ``times`` (linear positions, spherical rotation interpolation, clamped ends)."""
        from scipy.spatial.transform import Slerp

        times = np.clip(np.asarray(times, float), self.t[0], self.t[-1])
        pos = np.stack([np.interp(times, self.t, self.position[:, k]) for k in range(3)], axis=1)
        slerp = Slerp(self.t, Rotation.from_matrix(self.R_wb))
        return Track(self.object_id, times, pos, slerp(times).as_matrix(), self.hull_parts, self.object_class)


@dataclass
class EventTimes:
    """Detected events: name -> time in seconds since the clip's frame 0 (or None if absent)."""

    times: dict[str, float | None]
    actors: dict[str, tuple[str, ...]]
    kinds: dict[str, str]

    def get(self, name: str) -> float | None:
        return self.times.get(name)


def _first(mask: np.ndarray) -> int | None:
    idx = np.flatnonzero(mask)
    return int(idx[0]) if len(idx) else None


def _contact_series(track: Track, part: StaticPart, tol: float) -> np.ndarray:
    return np.array([clearance(track.world_parts(i), part) <= tol for i in range(len(track))])


def _rest_index(track: Track, start: int | None) -> int | None:
    """First index at or after ``start`` from which the object stays still for the hold time.

    Near the end of the record, a still stretch counts when it covers at least half the hold.
    """
    if start is None:
        return None
    speed = np.linalg.norm(track.velocity(), axis=1)
    rate = track.angular_rate()
    still = (speed < REST_SPEED_MPS) & (rate < REST_RATE_RAD_S)
    n = len(track)
    for i in range(max(0, start), n):
        if not still[i]:
            continue
        j = int(np.searchsorted(track.t, track.t[i] + REST_HOLD_S, side="right"))
        if j >= n:
            return i if still[i:].all() and track.t[-1] - track.t[i] >= 0.5 * REST_HOLD_S else None
        if still[i:j].all():
            return i
    return None


def detect_events(family: str, tracks: Mapping[str, Track], statics: Sequence[StaticPart],
                  ground_z: float = 0.0, contact_tol_m: float = CONTACT_TOL_M,
                  wheel_gap_m: float = 0.0) -> EventTimes:
    """Find the events of a benchmark family from dense tracks and static geometry.

    ``wheel_gap_m`` is how far a vehicle's hull bottom sits above the surface it drives on (its
    ride height when the hull has no wheels); it is added to the contact tolerance for
    vehicle-surface contacts. Times are those of the tracks (seconds since frame 0).
    """
    by_id: dict[str, list[StaticPart]] = {}
    for part in statics:
        by_id.setdefault(part.static_id, []).append(part)

    def touching(track: Track, static_id: str, tol: float) -> np.ndarray:
        parts = by_id.get(static_id, [])
        if not parts:
            return np.zeros(len(track), bool)
        out = np.zeros(len(track), bool)
        for part in parts:
            out |= _contact_series(track, part, tol)
        return out

    times: dict[str, float | None] = {}
    actors: dict[str, tuple[str, ...]] = {}
    kinds: dict[str, str] = {}

    def put(name: str, track: Track | None, index: int | None, kind: str, who: tuple[str, ...]) -> None:
        times[name] = None if (index is None or track is None) else float(track.t[index])
        actors[name] = who
        kinds[name] = kind

    if family == "barrier_crash":
        car = tracks["car"]
        hit = touching(car, "barrier", contact_tol_m)
        i_hit = _first(hit)
        put("impact", car, i_hit, "contact", ("car", "barrier"))
        put("rest", car, _rest_index(car, i_hit), "rest", ("car",))
    elif family == "ramp_jump":
        car = tracks["car"]
        on_ramp = touching(car, "ramp", contact_tol_m + wheel_gap_m)
        on_ground = (car.lowest_z() - ground_z) <= contact_tol_m + wheel_gap_m
        i_ramp = _first(on_ramp)
        put("ramp_contact", car, i_ramp, "contact", ("car", "ramp"))
        free = ~on_ramp & ~on_ground
        i_off = None
        if i_ramp is not None:
            hold = max(1, int(np.searchsorted(car.t, car.t[0] + AIRBORNE_HOLD_S)))
            for i in range(i_ramp, len(car) - hold):
                if free[i:i + hold].all():
                    i_off = i
                    break
        put("takeoff", car, i_off, "separation", ("car", "ramp"))
        i_land = i_apex = None
        if i_off is not None:
            back = np.flatnonzero(~free[i_off:])
            i_land = int(i_off + back[0]) if len(back) else None
            end = i_land if i_land is not None else len(car)
            i_apex = int(i_off + np.argmax(car.centre()[i_off:end, 2]))
        put("apex", car, i_apex, "custom", ("car",))
        put("landing", car, i_land, "contact", ("car", "ground"))
    elif family == "rollover":
        car = tracks["car"]
        on_trip = touching(car, "trip_ramp", contact_tol_m + wheel_gap_m)
        i_on = _first(on_trip)
        put("trip_contact", car, i_on, "contact", ("car", "trip_ramp"))
        i_exit = None
        if i_on is not None:
            after = np.flatnonzero(~on_trip[i_on:])
            i_exit = int(i_on + after[0]) if len(after) else None
        put("trip_exit", car, i_exit, "separation", ("car", "trip_ramp"))
        put("rest", car, _rest_index(car, i_exit), "rest", ("car",))
    elif family == "drop_and_bounce":
        surface_ids = [sid for sid in by_id if sid != "ground"]
        for name, track in tracks.items():
            contact = np.zeros(len(track), bool)
            for sid in surface_ids:
                contact |= touching(track, sid, contact_tol_m)
            contact |= (track.lowest_z() - ground_z) <= contact_tol_m
            moved = np.linalg.norm(track.centre() - track.centre()[0], axis=1) > 0.01
            i_move = _first(moved)
            if "incline" in by_id:
                ground_hit = (track.lowest_z() - ground_z) <= contact_tol_m
                put(f"first_motion.{name}", track, i_move, "custom", (name,))
                i_ground = _first(ground_hit)
                put(f"ground_contact.{name}", track, i_ground, "contact", (name, "ground"))
                put(f"rest.{name}", track, _rest_index(track, i_ground if i_ground is not None else i_move),
                    "rest", (name,))
            else:
                i_contact = _first(contact)
                put(f"first_contact.{name}", track, i_contact, "contact", (name,))
                put(f"rest.{name}", track, _rest_index(track, i_contact), "rest", (name,))
    else:
        raise ValueError(f"no event detector for family {family!r}")
    return EventTimes(times, actors, kinds)


def event_frames(events: EventTimes, fps: float, n_frames: int) -> dict[str, tuple[int, int] | None]:
    """Inclusive frame intervals that bracket each event time (frame k is at time k / fps)."""
    out: dict[str, tuple[int, int] | None] = {}
    for name, t in events.times.items():
        if t is None:
            out[name] = None
            continue
        x = t * fps
        lo = int(math.floor(x + 1e-9))
        hi = int(math.ceil(x - 1e-9))
        if lo < 0 or hi > n_frames - 1:
            out[name] = None
            continue
        out[name] = (lo, max(lo, hi))
    return out


SPEED_WINDOW_S = 0.05
"""Speeds at events are mean speeds over this window just before the event (contacts change speed fast)."""


def _speed_at(track: Track, t: float | None, window_s: float = SPEED_WINDOW_S) -> float | None:
    """Mean speed of the bounding-box centre over ``[t - window_s, t]``."""
    if t is None:
        return None
    c = track.centre()
    t_a = max(float(track.t[0]), t - window_s)
    if t - t_a <= 1e-9:
        return None
    pa = np.array([np.interp(t_a, track.t, c[:, k]) for k in range(3)])
    pb = np.array([np.interp(t, track.t, c[:, k]) for k in range(3)])
    return float(np.linalg.norm(pb - pa) / (t - t_a))


def truth_metrics(family: str, tracks: Mapping[str, Track], events: EventTimes) -> dict[str, dict[str, Any]]:
    """Truth values per actor for the family's targets and reported metrics."""
    out: dict[str, dict[str, Any]] = {}
    if family in ("barrier_crash", "ramp_jump", "rollover"):
        car = tracks["car"]
        m: dict[str, Any] = {}
        first = {"barrier_crash": "impact", "ramp_jump": "takeoff", "rollover": "trip_contact"}[family]
        m["speed_at_event"] = _speed_at(car, events.get(first))
        if family == "barrier_crash":
            t_hit = events.get("impact")
            m["event_time"] = t_hit
            if t_hit is not None:
                i_hit = int(np.searchsorted(car.t, t_hit))
                yaw = car.roll_pitch_yaw_deg()[:, 2]
                m["yaw_change"] = float(yaw[-1] - yaw[i_hit])
                v = car.velocity()
                forward = v[i_hit] / max(1e-9, np.linalg.norm(v[i_hit]))
                window = (car.t >= t_hit) & (car.t <= t_hit + 1.0)
                along = v[window] @ forward
                m["rebound_speed"] = float(max(0.0, -along.min())) if len(along) else None
                m["post_impact_displacement"] = float(np.linalg.norm(car.centre()[-1] - car.centre()[i_hit]))
        if family == "ramp_jump":
            t_off, t_land = events.get("takeoff"), events.get("landing")
            t_apex = events.get("apex")
            if t_apex is not None and t_off is not None:
                cz = car.centre()[:, 2]
                m["apex_rise"] = float(np.interp(t_apex, car.t, cz) - np.interp(t_off, car.t, cz))
            m["airborne_time"] = (t_land - t_off) if (t_land is not None and t_off is not None) else None
            if t_land is not None:
                m["landing_distance"] = float(np.interp(t_land, car.t, car.centre()[:, 0]))
        if family == "rollover":
            roll = car.cumulative_roll_deg()
            m["cumulative_roll"] = float(roll[-1])
            m["quarter_turns"] = int(math.floor(abs(roll[-1]) / 90.0))
            up = car.R_wb[-1][:, 2]
            m["rest_attitude"] = "wheels_down" if up[2] > 0.7 else ("roof" if up[2] < -0.7 else "side")
            m["rolled_over"] = 1.0 if abs(roll).max() > 90.0 else 0.0
        m["rest_position"] = car.centre()[-1].tolist()
        out["car"] = m
    elif family == "drop_and_bounce":
        for name, track in tracks.items():
            m = {}
            t_contact = events.get(f"first_contact.{name}")
            m["event_time"] = t_contact
            z = track.centre()[:, 2]
            if t_contact is not None:
                i0 = int(np.searchsorted(track.t, t_contact))
                apexes = []
                for k in range(i0 + 1, len(z) - 1):
                    if z[k] > z[k - 1] and z[k] >= z[k + 1]:
                        apexes.append(float(z[k]))
                drop = float(z[0] - z[i0]) if i0 < len(z) else 0.0
                rise = [a - float(z[i0]) for a in apexes]
                m["bounce_apexes_m"] = rise
                m["rebound_ratio"] = (rise[0] / drop) if (rise and drop > 0) else 0.0
            m["rest_position"] = track.centre()[-1].tolist()
            out[name] = m
    return out
