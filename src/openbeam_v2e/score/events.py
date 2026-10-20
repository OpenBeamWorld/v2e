# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""Contact, separation and rest events detected from any world's object tracks.

Contacts are tested per pair: an object against each static mesh and against every other
object, using the exact distance from hull vertices to the other side's triangles (both ways),
within a tolerance; an impact that falls between two frames is caught as a local minimum of
the sampled distance (see :func:`contact_mask`). Search-censored neighbouring samples are
queried without a cutoff before that test; unknown poses remain unknown. A pair already touching
at the first known frame (an object resting on its support) starts in contact, so its first *new*
contact is the first one that begins later.
Rest is the start of the final stretch in which an object's speed and angular speed stay below
thresholds for at least a hold time.

These detectors work for any number and kind of objects; nothing is tuned to a scene.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from openbeam_v2e.world.geometry import (
    faces_in_box,
    mesh_bounds,
    points_to_triangles_distance,
    quat_to_matrix,
    triangle_bounds,
)
from openbeam_v2e.world.model import ObjectTrack, World, intervals_from_mask

__all__ = ["DetectedEvent", "ContactOptions", "static_name", "contact_mask", "contact_series", "detect_events",
           "rest_frame", "match_event"]


def static_name(mesh_id: int) -> str:
    """Name of a static mesh in contact pairs and event actor lists."""
    return f"static:{int(mesh_id)}"


@dataclass(frozen=True)
class ContactOptions:
    tolerance_m: float = 0.03
    between_frames_cap_m: float = 0.25
    rest_speed_mps: float = 0.05
    rest_angular_dps: float = 3.0
    rest_hold_s: float = 0.5


def contact_mask(distance: np.ndarray, tolerance: float, cap: float) -> np.ndarray:
    """Frames in contact from a sampled distance series.

    A frame is in contact when the distance is within ``tolerance``, or when it is a local
    minimum within ``tolerance`` plus half the larger neighbouring step (capped at ``cap``): a
    fast impact and rebound can fall between two frames, so the sampled distance never gets
    small even though the bodies touched.
    """
    d = np.asarray(distance, dtype=float)
    contact = np.isfinite(d) & (d <= tolerance)
    F = len(d)
    for t in range(F):
        if contact[t] or not np.isfinite(d[t]):
            continue
        prev = d[t - 1] if t > 0 else np.nan
        nxt = d[t + 1] if t + 1 < F else np.nan
        if not (np.isfinite(prev) and np.isfinite(nxt)) or d[t] > prev or d[t] > nxt:
            continue
        step = max(abs(d[t] - prev), abs(nxt - d[t]))
        if d[t] <= min(tolerance + 0.5 * step, cap):
            contact[t] = True
    return contact


@dataclass(frozen=True)
class DetectedEvent:
    kind: str                     # "contact" | "separation" | "rest"
    frame: int
    actors: tuple[str, ...]       # (object, other) for contact and separation; (object,) for rest

    def involves(self, names: Iterable[str]) -> bool:
        return set(names) <= set(self.actors)


@dataclass
class _Body:
    name: str
    vertices: np.ndarray          # (F, V, 3) world, NaN when unknown (dynamic) or (V, 3) static
    faces: np.ndarray
    static: bool = False
    lo: np.ndarray = field(default_factory=lambda: np.zeros(3))
    hi: np.ndarray = field(default_factory=lambda: np.zeros(3))


def _object_body(obj: ObjectTrack) -> _Body:
    R = quat_to_matrix(obj.quat_xyzw)
    v = np.einsum("fij,vj->fvi", R, obj.hull_vertices) + obj.position[:, None, :]
    return _Body(obj.id, v, obj.hull_faces)


def _pair_distance(a: np.ndarray, af: np.ndarray, b: np.ndarray, bf: np.ndarray, tol: float | None,
                    b_bounds: tuple[np.ndarray, np.ndarray] | None = None) -> float:
    """Vertex-to-triangle distance both ways; ``tol=None`` disables search culling.

    With a cutoff, infinity means search-censored, not an unknown pose or a measured distance.
    Distances above the cutoff need an uncapped query before use in a neighbouring-step test.
    """
    if tol is None:
        return min(float(np.min(points_to_triangles_distance(a, b, bf))),
                   float(np.min(points_to_triangles_distance(b, a, af))))
    alo, ahi = mesh_bounds(a)
    blo, bhi = mesh_bounds(b)
    if np.any(alo - tol > bhi) or np.any(blo - tol > ahi):
        return np.inf  # farther apart than ``tol``
    near_a = np.all((a >= blo - tol) & (a <= bhi + tol), axis=1)
    near_b = np.all((b >= alo - tol) & (b <= ahi + tol), axis=1)
    # only triangles near the other body can be closest to it (large static meshes stay cheap)
    bf_near = faces_in_box(b, bf, alo, ahi, tol, bounds=b_bounds)
    af_near = faces_in_box(a, af, blo, bhi, tol)
    d = np.inf
    if np.any(near_a) and len(bf_near):
        d = min(d, float(np.min(points_to_triangles_distance(a[near_a], b, bf_near))))
    if np.any(near_b) and len(af_near):
        d = min(d, float(np.min(points_to_triangles_distance(b[near_b], a, af_near))))
    if d == np.inf:  # overlapping boxes but no vertex near the other: test all vertices
        d = min(float(np.min(points_to_triangles_distance(a, b, bf))), float(np.min(points_to_triangles_distance(b, a, af))))
    return d


def contact_series(world: World, options: ContactOptions = ContactOptions()) -> dict[tuple[str, str], np.ndarray]:
    """Per pair ``(object, other)``: boolean contact per frame (False where the object is unknown)."""
    bodies = [_object_body(o) for o in world.objects]
    statics = [_Body(static_name(s.mesh_id), s.vertices, s.faces, True) for s in world.statics
               if s.role in ("collision", "drivable")]
    F = world.frame_count
    out: dict[tuple[str, str], np.ndarray] = {}
    for i, body in enumerate(bodies):
        known = np.all(np.isfinite(body.vertices), axis=(1, 2))
        for other in statics + bodies[i + 1:]:
            other_bounds = triangle_bounds(other.vertices, other.faces) if other.static else None
            pair_known = known if other.static else known & np.all(np.isfinite(other.vertices), axis=(1, 2))
            distance = np.full(F, np.nan)
            margin = max(options.tolerance_m, options.between_frames_cap_m)
            for t in np.flatnonzero(pair_known):
                ov = other.vertices if other.static else other.vertices[t]
                distance[t] = _pair_distance(body.vertices[t], body.faces, ov, other.faces, margin,
                                             other_bounds)
            # The broad phase can censor the approach/rebound samples needed by contact_mask.
            # Query only the one-frame halo of possible local minima, not every distant pair.
            # Never substitute infinity/the cap for a measured step or bridge an unknown pose.
            candidates = np.flatnonzero(np.isfinite(distance) & (distance > options.tolerance_m)
                                        & (distance <= options.between_frames_cap_m))
            halo: set[int] = set()
            for t in candidates:
                if t == 0 or t + 1 == F or not (pair_known[t - 1] and pair_known[t + 1]):
                    continue
                if distance[t] > min(distance[t - 1], distance[t + 1]):
                    continue
                halo.update(k for k in (t - 1, t + 1) if distance[k] > margin)
            for t in sorted(halo):
                ov = other.vertices if other.static else other.vertices[t]
                distance[t] = _pair_distance(body.vertices[t], body.faces, ov, other.faces, None, other_bounds)
            out[(body.name, other.name)] = contact_mask(distance, options.tolerance_m, options.between_frames_cap_m)
    return out


def rest_frame(obj: ObjectTrack, fps: float, options: ContactOptions = ContactOptions()) -> int | None:
    """First frame of the final at-rest stretch (None when the object never settles)."""
    speed = obj.speed(fps)
    omega = np.degrees(np.linalg.norm(obj.angular_velocity(fps), axis=1))
    still = obj.known & (np.nan_to_num(speed, nan=np.inf) <= options.rest_speed_mps) \
        & (np.nan_to_num(omega, nan=np.inf) <= options.rest_angular_dps)
    runs = intervals_from_mask(still)
    if not runs:
        return None
    last_known = int(np.flatnonzero(obj.known)[-1])
    a, b = runs[-1]
    hold = max(1, int(round(options.rest_hold_s * fps)))
    if b != last_known or (b - a + 1) < hold:
        return None
    return a


def detect_events(world: World, options: ContactOptions = ContactOptions()) -> list[DetectedEvent]:
    """Contact starts, separations and rests in ``world``, sorted by frame."""
    events: list[DetectedEvent] = []
    for (name, other), series in contact_series(world, options).items():
        known = world.object(name).known
        first = int(np.flatnonzero(known)[0]) if known.any() else 0
        for a, b in intervals_from_mask(series):
            if a > first:
                events.append(DetectedEvent("contact", a, (name, other)))
            if b + 1 < world.frame_count and known[b + 1]:
                events.append(DetectedEvent("separation", b + 1, (name, other)))
    for obj in world.objects:
        f = rest_frame(obj, world.fps, options)
        if f is not None:
            events.append(DetectedEvent("rest", f, (obj.id,)))
    return sorted(events, key=lambda e: (e.frame, e.kind, e.actors))


def match_event(events: list[DetectedEvent], kind: str, actors: Iterable[str], near: float) -> DetectedEvent | None:
    """The detected event of ``kind`` involving all ``actors`` closest to frame ``near``."""
    names = tuple(actors)
    candidates = [e for e in events if e.kind == kind and e.involves(names)]
    if not candidates:
        return None
    return min(candidates, key=lambda e: (abs(e.frame - near), e.frame))
