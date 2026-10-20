# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""Measures of what happens after an event, computed the same way for reference and run.

After a contact (for example a vehicle meeting a barrier):
    rebound speed (peak speed against the pre-event direction of travel within 1 s), the
    displacement and yaw change from the event to rest, the rest pose and the time to rest.
Rollover (vehicles):
    cumulative roll (the integral of the body rotation rate about the initial horizontal travel
    direction), the number of completed quarter turns, the roll-onset time (cumulative roll
    first reaching 15 degrees) and the rest attitude (``wheels_down``, ``side`` or ``roof``).
Bounce (props):
    ground-contact start times, apex heights between contacts (lowest hull point above the
    ground), rebound ratios between successive apexes, the number of bounces above a height
    threshold, and the rest time and pose.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from openbeam_v2e.world.geometry import faces_in_box, ground_height, quat_to_matrix, triangle_bounds, yaw_of
from openbeam_v2e.world.model import ObjectTrack, World, intervals_from_mask

__all__ = ["after_event", "rollover", "bounce", "ground_gap", "ROLL_ONSET_DEG", "BOUNCE_MIN_APEX_M"]

ROLL_ONSET_DEG = 15.0
BOUNCE_MIN_APEX_M = 0.02
REBOUND_WINDOW_S = 1.0


def _wrap_deg(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def ground_gap(obj: ObjectTrack, world: World) -> np.ndarray:
    """(F,) height of the lowest hull point above the static surface under it (NaN when unknown)."""
    vertices, faces = world.static_geometry()
    out = np.full(obj.frame_count, np.nan)
    if len(faces) == 0:
        return out
    R = quat_to_matrix(obj.quat_xyzw)
    bounds = triangle_bounds(vertices, faces)
    for t in np.flatnonzero(obj.known):
        hull = obj.hull_vertices @ R[t].T + obj.position[t]
        near = faces_in_box(vertices, faces, hull.min(axis=0), hull.max(axis=0), 0.01, dims=2, bounds=bounds)
        g = ground_height(hull[:, :2], vertices, near, below=hull[:, 2] + 0.3)
        ok = np.isfinite(g)
        if np.any(ok):
            out[t] = float(np.min(hull[ok, 2] - g[ok]))
    return out


def after_event(obj: ObjectTrack, fps: float, frame: int, rest: int | None) -> dict[str, Any]:
    """Rebound speed, displacement, yaw change, rest pose and time to rest after ``frame``."""
    F = obj.frame_count
    if not 0 <= frame < F or not obj.known[frame]:
        return {"available": False}
    v = obj.velocity(fps)
    centre = obj.centre_track()
    before = [k for k in range(max(0, frame - 3), frame) if np.all(np.isfinite(v[k]))]
    out: dict[str, Any] = {"available": True}
    if before:
        v_pre = np.mean(v[before], axis=0)
        speed_pre = float(np.linalg.norm(v_pre))
        out["speed_before_mps"] = speed_pre
        if speed_pre > 1e-6:
            u = v_pre / speed_pre
            window = range(frame, min(F, frame + int(round(REBOUND_WINDOW_S * fps)) + 1))
            back = [-(v[k] @ u) for k in window if np.all(np.isfinite(v[k]))]
            out["rebound_speed_mps"] = float(max(0.0, max(back))) if back else None
    end = rest if rest is not None and rest >= frame else None
    if end is None:
        known_after = np.flatnonzero(obj.known[frame:])
        end_frame = frame + int(known_after[-1]) if len(known_after) else frame
        out["at_rest"] = False
    else:
        end_frame = end
        out["at_rest"] = True
        out["time_to_rest_s"] = (end - frame) / fps
    if obj.known[end_frame]:
        R = obj.rotations()
        out["displacement_m"] = float(np.linalg.norm(centre[end_frame] - centre[frame]))
        out["yaw_change_deg"] = _wrap_deg(math.degrees(yaw_of(R[end_frame]) - yaw_of(R[frame])))
        out["end_position_m"] = [float(x) for x in centre[end_frame]]
        out["end_yaw_deg"] = math.degrees(yaw_of(R[end_frame]))
    return out


def rollover(obj: ObjectTrack, fps: float, start: int = 0) -> dict[str, Any]:
    """Cumulative roll about the initial travel direction, quarter turns, onset and rest attitude."""
    known = np.flatnonzero(obj.known)
    known = known[known >= start]
    if len(known) < 2:
        return {"available": False}
    R = obj.rotations()
    v = obj.velocity(fps)
    first = int(known[0])
    travel = v[first] if np.all(np.isfinite(v[first])) and np.linalg.norm(v[first, :2]) > 0.5 else R[first][:, 0]
    h = np.array([travel[0], travel[1], 0.0])
    if np.linalg.norm(h) < 1e-9:
        h = np.array([1.0, 0.0, 0.0])
    h /= np.linalg.norm(h)
    omega = obj.angular_velocity(fps)
    rate = np.where(np.all(np.isfinite(omega), axis=1), omega @ h, 0.0)
    cumulative = np.cumsum(rate[first:]) / fps
    cum_deg = np.degrees(cumulative)
    onset = np.flatnonzero(np.abs(cum_deg) >= ROLL_ONSET_DEG)
    last = int(known[-1])
    up = R[last][2, 2]
    attitude = "wheels_down" if up > 0.5 else ("roof" if up < -0.5 else "side")
    total = float(cum_deg[last - first]) if last - first < len(cum_deg) else float(cum_deg[-1])
    return {
        "available": True,
        "cumulative_roll_deg": total,
        "max_abs_roll_deg": float(np.max(np.abs(cum_deg))),
        "quarter_turns": int(math.floor(abs(total) / 90.0)),
        "roll_onset_s": float((first + int(onset[0])) / fps) if len(onset) else None,
        "rest_attitude": attitude,
    }


def bounce(obj: ObjectTrack, fps: float, ground_gap: np.ndarray, contact_tol_m: float = 0.03,
           rest: int | None = None) -> dict[str, Any]:
    """Contacts, apexes and rebound ratios from the per-frame gap between hull and ground."""
    gap = np.asarray(ground_gap, dtype=float)
    if not np.any(np.isfinite(gap)):
        return {"available": False}
    from openbeam_v2e.score.events import contact_mask

    contact = contact_mask(gap, contact_tol_m, 0.25)
    runs = intervals_from_mask(contact)
    first_known = int(np.flatnonzero(np.isfinite(gap))[0])
    starts = [a for a, _ in runs if a > first_known]
    apexes = []
    for (a0, b0), (a1, _) in zip(runs, runs[1:]):
        seg = gap[b0 + 1:a1]
        if len(seg) and np.any(np.isfinite(seg)):
            apexes.append(float(np.nanmax(seg)))
    heights = ([float(gap[first_known])] if gap[first_known] > contact_tol_m and starts else []) + apexes
    ratios = [heights[i + 1] / heights[i] for i in range(len(heights) - 1) if heights[i] > 0]
    out: dict[str, Any] = {
        "available": True,
        "initial_height_m": float(gap[first_known]),
        "contact_times_s": [a / fps for a in starts],
        "first_contact_s": starts[0] / fps if starts else None,
        "apex_heights_m": apexes,
        "rebound_ratios": ratios,
        "bounces": int(sum(1 for h in apexes if h >= BOUNCE_MIN_APEX_M)),
        "rest_s": rest / fps if rest is not None else None,
    }
    if rest is not None and obj.known[rest]:
        out["rest_position_m"] = [float(x) for x in obj.centre_track()[rest]]
    return out
