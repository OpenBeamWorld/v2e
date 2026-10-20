# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""Events from reconstructed pose tracks, as frame intervals with the evidence used.

* ``rest`` (kind ``rest``): speed below ``rest_speed_mps`` for at least ``min_rest_s``;
* ``take_off`` (kind ``separation``) and ``touchdown`` (kind ``contact``): the lowest point of
  the hull rises above or returns to the ground by more than ``air_height_m`` (and three
  position sigmas);
* ``impact`` (kind ``contact``): a peak of the smoothed acceleration above ``impact_accel_mps2``.

An interval ``[a, b]`` says the event happened between frames ``a`` and ``b`` inclusive; detectors
widen it where the evidence is ambiguous instead of guessing a frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from openbeam_v2e.perception.types import PoseTrack
from openbeam_v2e.reconstruct.geom import quat_to_matrix
from openbeam_v2e.reconstruct.hulls import make_hull


@dataclass
class Event:
    name: str
    kind: str                       # contact | separation | rest | custom
    frames: tuple[int, int]
    actors: list[str]
    evidence: str
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "kind": self.kind, "frame": [int(self.frames[0]), int(self.frames[1])],
                "actors": list(self.actors), "evidence": self.evidence}


@dataclass
class EventOptions:
    rest_speed_mps: float = 0.2
    min_rest_s: float = 0.3
    air_height_m: float = 0.05
    min_air_s: float = 0.1
    impact_accel_mps2: float = 30.0


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    out = []
    i = 0
    n = len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            out.append((i, j))
            i = j + 1
        else:
            i += 1
    return out


def lowest_point_heights(track: PoseTrack, ground_z: float = 0.0) -> np.ndarray:
    hull = make_hull(track.template, track.dims, **track.template_params)
    return np.array([hull.lowest_point(quat_to_matrix(q), p) - ground_z
                      for q, p in zip(track.quat, track.position)])


def airborne_segments(track: PoseTrack, fps: float, ground_z: float = 0.0,
                      options: EventOptions | None = None) -> list[tuple[int, int]]:
    """Frame bounds of sustained estimated clearance, not proof of gravity-only motion.

    The same conservative height/position-sigma rule drives separation events and optional
    refinement. Segments may touch the clip boundary without inventing a take-off/landing event.
    """
    if track.motion == "ground" or not len(track):
        return []
    o = options or EventOptions()
    heights = lowest_point_heights(track, ground_z)
    sigma_z = np.sqrt(np.maximum(track.cov[:, 2, 2], 0.0))
    air = np.isfinite(heights) & np.isfinite(sigma_z) & (heights > np.maximum(o.air_height_m, 3.0 * sigma_z))
    min_air = max(1, int(round(o.min_air_s * fps)))
    return [(int(track.frames[a]), int(track.frames[b])) for a, b in _runs(air) if b - a + 1 >= min_air]


def detect_events(track: PoseTrack, fps: float, ground_z: float = 0.0,
                  options: EventOptions | None = None) -> list[Event]:
    o = options or EventOptions()
    frames = track.frames
    events: list[Event] = []
    if len(frames) < 3:
        return events
    speed = np.linalg.norm(track.velocity, axis=1)
    min_rest = max(2, int(round(o.min_rest_s * fps)))
    for a, b in _runs(speed < o.rest_speed_mps):
        if b - a + 1 >= min_rest:
            events.append(Event("rest", "rest", (int(frames[a]), int(frames[b])), [track.object_id],
                                f"speed < {o.rest_speed_mps} m/s for {b - a + 1} frames",
                                {"max_speed_mps": float(speed[a:b + 1].max())}))
    if track.motion != "ground":
        h = lowest_point_heights(track, ground_z)
        for start, end in airborne_segments(track, fps, ground_z, o):
            a, b = track.index(start), track.index(end)
            if a > 0:
                events.append(Event("take_off", "separation", (int(frames[a - 1]), int(frames[a])),
                                    [track.object_id], "lowest hull point leaves the ground",
                                    {"height_m": float(h[a])}))
            if b < len(frames) - 1:
                events.append(Event("touchdown", "contact", (int(frames[b]), int(frames[b + 1])),
                                    [track.object_id], "lowest hull point returns to the ground",
                                    {"height_m": float(h[b])}))
    dt = 1.0 / fps
    acc = np.gradient(track.velocity, dt, axis=0)
    amag = np.linalg.norm(acc, axis=1)
    if track.motion != "ground":
        amag = np.linalg.norm(acc + np.array([0.0, 0.0, 9.81]), axis=1)  # gravity is not an impact
    for a, b in _runs(amag > o.impact_accel_mps2):
        k = a + int(np.argmax(amag[a:b + 1]))
        events.append(Event("impact", "contact", (int(frames[a]), int(frames[b])), [track.object_id],
                            f"acceleration peak {amag[k]:.1f} m/s^2", {"peak_frame": int(frames[k])}))
    events.sort(key=lambda e: e.frames[0])
    return events


def name_unique(events: Sequence[Event]) -> list[Event]:
    """Give repeated event names a numeric suffix (``rest``, ``rest_2``, ...) and the actor."""
    seen: dict[str, int] = {}
    out = []
    for e in events:
        base = f"{e.name}" if len(e.actors) != 1 else f"{e.actors[0]}_{e.name}"
        seen[base] = seen.get(base, 0) + 1
        name = base if seen[base] == 1 else f"{base}_{seen[base]}"
        out.append(Event(name, e.kind, e.frames, e.actors, e.evidence, e.data))
    return out
