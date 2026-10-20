# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""Benchmark cameras: per-frame cameras for a case's camera preset.

Both presets keep the camera at one position, which is the v0 reconstruction scope:

``tripod``
    Fixed orientation and focal length, no lens distortion.
``handheld_ptz``
    Pans and tilts to follow the subject with a first-order lag, zooms in linearly over the clip,
    has radial distortion and a small deterministic wobble, like an operator on a fluid head.

The cameras are :class:`openbeam_v2e.reconstruct.geom.Camera` objects, so the same model renders
the frames, is written into the truth package and is what reconstruction is scored against.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from openbeam_v2e.bench.families import CameraPreset
from openbeam_v2e.reconstruct.geom import Camera, Intrinsics, look_at
from openbeam_v2e.reconstruct.synthetic import ptz_follow

__all__ = ["camera_track", "follow_points", "WOBBLE_DEG"]

WOBBLE_DEG = 0.15
"""Peak amplitude of the handheld preset's deterministic pan and tilt wobble."""


def camera_track(preset: CameraPreset, width: int, height: int, fps: float,
                 subject_centres: np.ndarray | None = None) -> list[Camera]:
    """Cameras for every frame of a clip.

    ``subject_centres`` (F, 3) are the world positions the handheld preset follows (the scored
    subject's bounding-box centre, or the mean of several subjects); the tripod ignores them but
    uses their length as the frame count.
    """
    if subject_centres is None:
        raise ValueError("camera_track needs the subject centres (their length is the frame count)")
    targets = np.asarray(subject_centres, float).reshape(-1, 3)
    n = len(targets)
    centre = np.asarray(preset.position_m, float)
    intr = Intrinsics.from_fov(width, height, preset.hfov_deg, k1=preset.k1)
    if preset.name == "tripod":
        R = look_at(centre, preset.look_at_m)
        return [Camera(intr, R, centre) for _ in range(n)]
    if preset.name != "handheld_ptz":
        raise ValueError(f"unknown camera preset {preset.name!r}")
    # The operator has the subject framed from the first frame and follows it with a lag.
    aim = targets
    t = np.arange(n) / fps
    duration = max(t[-1], 1e-9) if n > 1 else 1.0
    focal = intr.f * (1.0 + (preset.zoom_end - 1.0) * t / duration)
    lag_frames = max(1.0, preset.lag_s * fps)
    cams = ptz_follow(centre, aim, focal, intr, lag_frames=lag_frames)
    # Deterministic wobble: two incommensurate sines per axis, no random state.
    out: list[Camera] = []
    for k, cam in enumerate(cams):
        pan, tilt, roll = cam.ptz()
        dp = math.radians(WOBBLE_DEG) * (0.6 * math.sin(2 * math.pi * 1.3 * t[k]) + 0.4 * math.sin(2 * math.pi * 0.47 * t[k]))
        dt = math.radians(WOBBLE_DEG) * (0.6 * math.sin(2 * math.pi * 0.9 * t[k] + 1.0) + 0.4 * math.sin(2 * math.pi * 0.31 * t[k]))
        out.append(Camera.from_ptz(cam.intrinsics, centre, pan + dp, tilt + dt, roll))
    return out


def follow_points(centres: Sequence[np.ndarray]) -> np.ndarray:
    """Mean of several subjects' centres per frame (what the handheld operator keeps in frame)."""
    stack = np.stack([np.asarray(c, float) for c in centres])
    return stack.mean(axis=0)
