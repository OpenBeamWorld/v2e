# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""Simulated time versus video time.

A run reports poses at simulated times since spawn; the footage has frames at ``frame / fps``.
For actors that spawn at rest in their frame-0 pose, video frame 0 is spawn time. For actors
that drive in, video frame 0 is the moment ``t0`` at which the actor's reference point first
crosses the plane through its frame-0 pose that is perpendicular to its path (a spatial
trigger, so the result does not depend on how long the run-up took). Later events are
compared on times relative to ``t0``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from openbeam_v2e.world.geometry import slerp_quats

__all__ = ["resample_poses", "frame0_trigger", "video_times"]


def video_times(frame_count: int, fps: float, t0_s: float = 0.0) -> np.ndarray:
    """Simulated time of each video frame."""
    return t0_s + np.arange(int(frame_count)) / float(fps)


def resample_poses(t_src: Any, position: Any, quat: Any, t_dst: Any) -> tuple[np.ndarray, np.ndarray]:
    """Positions (linear) and attitudes (spherical) at ``t_dst``; NaN outside the source span."""
    t_src = np.asarray(t_src, dtype=float)
    position = np.asarray(position, dtype=float)
    quat = np.asarray(quat, dtype=float)
    t_dst = np.asarray(t_dst, dtype=float)
    ok = np.all(np.isfinite(position), axis=1) & np.all(np.isfinite(quat), axis=1) & np.isfinite(t_src)
    t_src, position, quat = t_src[ok], position[ok], quat[ok]
    p = np.full((len(t_dst), 3), np.nan)
    if len(t_src) == 0:
        return p, np.full((len(t_dst), 4), np.nan)
    order = np.argsort(t_src, kind="stable")
    t_src, position, quat = t_src[order], position[order], quat[order]
    inside = (t_dst >= t_src[0]) & (t_dst <= t_src[-1])
    for axis in range(3):
        p[inside, axis] = np.interp(t_dst[inside], t_src, position[:, axis])
    q = slerp_quats(t_src, quat, t_dst)
    return p, q


def frame0_trigger(t: Any, position: Any, frame0_position: Any, direction: Any) -> float | None:
    """First time the path crosses the plane through ``frame0_position`` normal to ``direction``.

    The crossing time is interpolated linearly between the two rows around it. Returns None
    when the path never reaches the plane.
    """
    t = np.asarray(t, dtype=float)
    p = np.asarray(position, dtype=float)
    d = np.asarray(direction, dtype=float)
    norm = np.linalg.norm(d)
    if norm == 0 or len(t) == 0:
        return None
    d = d / norm
    s = (p - np.asarray(frame0_position, dtype=float)) @ d
    if s[0] >= 0:
        return float(t[0])
    hits = np.flatnonzero((s[:-1] < 0) & (s[1:] >= 0))
    if len(hits) == 0:
        return None
    i = int(hits[0])
    u = -s[i] / (s[i + 1] - s[i])
    return float(t[i] + u * (t[i + 1] - t[i]))
