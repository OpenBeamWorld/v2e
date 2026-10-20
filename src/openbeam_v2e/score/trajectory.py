# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""Per-object, per-phase trajectory and attitude metrics, and phase windows.

Positions are compared at each object's hull bounding-box centre, so the producers' choice of
body origin does not matter. Frames count when the reference pose is known and observed and
the run pose is known; ``coverage`` reports the fraction of the phase that counted. Every error
is reported raw and, where the reference has a per-frame sigma, as an RMS z-score.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from openbeam_v2e.score.attitude import attitude_error
from openbeam_v2e.world.model import ObjectTrack

__all__ = ["phase_windows", "trajectory_metrics", "state_at"]


def phase_windows(phases: Sequence[Any], boundaries: Mapping[str, float | None], frame_count: int) -> list[dict[str, Any]]:
    """Resolve named phases to inclusive frame windows.

    ``boundaries`` maps ``start``, ``end``, ``rest`` and event names to frames (None when the
    boundary is unknown). Phases with an unknown or inverted boundary are returned with
    ``frames: None`` and a reason.
    """
    out = []
    if not phases:
        return [{"name": "all", "frames": [0, frame_count - 1]}]
    for phase in phases:
        lo = boundaries.get(phase.start)
        hi = boundaries.get(phase.end)
        if lo is None or hi is None:
            missing = phase.start if lo is None else phase.end
            out.append({"name": phase.name, "frames": None, "reason": f"boundary {missing!r} is unknown"})
            continue
        a, b = int(round(lo)), int(round(hi))
        a, b = max(0, a), min(frame_count - 1, b)
        if b < a:
            out.append({"name": phase.name, "frames": None, "reason": "the phase ends before it starts"})
            continue
        out.append({"name": phase.name, "frames": [a, b]})
    return out


def _rms(x: np.ndarray) -> float | None:
    x = x[np.isfinite(x)]
    return float(math.sqrt(np.mean(x * x))) if len(x) else None


def trajectory_metrics(ref: ObjectTrack, run: ObjectTrack | None, fps: float, frames: Sequence[int]) -> dict[str, Any]:
    """Position, attitude and speed errors of one object over ``frames``."""
    frames = np.asarray(list(frames), dtype=int)
    eligible = frames[ref.known[frames] & ref.visible[frames]] if len(frames) else frames
    out: dict[str, Any] = {"frames": int(len(frames)), "reference_frames": int(len(eligible))}
    if run is None:
        out.update({"coverage": 0.0, "missing_in_run": True})
        return out
    used = eligible[run.known[eligible]] if len(eligible) else eligible
    out["coverage"] = float(len(used) / len(eligible)) if len(eligible) else None
    if len(used) == 0:
        return out
    c_ref = ref.centre_track()[used]
    c_run = run.centre_track()[used]
    err = np.linalg.norm(c_run - c_ref, axis=1)
    out["position_rms_m"] = _rms(err)
    out["position_max_m"] = float(np.max(err))
    out["position_mean_m"] = float(np.mean(err))
    sigma = ref.sigma_position_m[used]
    ok = np.isfinite(sigma) & (sigma > 0)
    out["position_rms_z"] = _rms(err[ok] / sigma[ok]) if np.any(ok) else None
    att = attitude_error(ref.rotations()[used], run.rotations()[used], ref.symmetry_class, ref.hull_extents())
    if np.any(np.isfinite(att)):
        out["attitude_rms_deg"] = math.degrees(_rms(att))
        out["attitude_max_deg"] = math.degrees(float(np.nanmax(att)))
        s_att = ref.sigma_attitude_rad[used]
        ok = np.isfinite(s_att) & (s_att > 0) & np.isfinite(att)
        out["attitude_rms_z"] = _rms(att[ok] / s_att[ok]) if np.any(ok) else None
    v_ref = ref.speed(fps)[used]
    v_run = run.speed(fps)[used]
    dv = v_run - v_ref
    if np.any(np.isfinite(dv)):
        out["speed_rms_mps"] = _rms(dv)
    out["symmetry"] = ref.symmetry_class
    return out


def state_at(obj: ObjectTrack, fps: float, frame: float) -> dict[str, Any] | None:
    """Centre position, speed and attitude of an object at a (possibly fractional) frame."""
    f = int(round(frame))
    if not 0 <= f < obj.frame_count or not obj.known[f]:
        return None
    R = obj.rotations()[f]
    yaw = math.degrees(math.atan2(R[1, 0], R[0, 0]))
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, -R[2, 0]))))
    roll = math.degrees(math.atan2(R[2, 1], R[2, 2]))
    speed = obj.speed(fps)[f]
    return {"frame": f, "position_m": [float(x) for x in obj.centre_track()[f]],
            "speed_mps": float(speed) if np.isfinite(speed) else None,
            "yaw_deg": yaw, "pitch_deg": pitch, "roll_deg": roll}
