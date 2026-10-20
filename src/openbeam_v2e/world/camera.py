# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""Pinhole cameras with radial distortion, and the camera track of a fixed-position camera.

Conventions (the same as the world-package layout): camera axes are x to the image right,
y down the image and z along the optical axis. ``R_wc`` is the camera-to-world rotation (its
columns are the camera axes in world coordinates) and ``centre`` the camera position, so a world
point ``p`` has camera coordinates ``R_wc.T @ (p - centre)``. Pixel ``(col, row)`` covers
``[col, col + 1) x [row, row + 1)``; its centre is ``(col + 0.5, row + 0.5)``. Distortion is
radial on normalised coordinates: ``x_d = x (1 + k1 r^2 + k2 r^4)``.

A :class:`CameraTrack` holds one camera per frame for a camera that stays at one position and
may pan, tilt, roll and zoom (the v0 scope); a static camera is a track whose frames are equal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from openbeam_v2e.schema.world import CAMERA_TRACK_SCHEMA_ID
from openbeam_v2e.world.geometry import matrix_to_quat, quat_to_matrix

__all__ = ["Camera", "CameraTrack", "look_at_rotation"]


def look_at_rotation(centre: Any, target: Any, up: Any = (0.0, 0.0, 1.0)) -> np.ndarray:
    """Camera-to-world rotation of a camera at ``centre`` looking at ``target`` with no roll."""
    fwd = np.asarray(target, dtype=float) - np.asarray(centre, dtype=float)
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, np.asarray(up, dtype=float))
    if np.linalg.norm(right) < 1e-9:  # looking straight up or down
        right = np.cross(fwd, np.array([0.0, 1.0, 0.0]))
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    return np.stack([right, down, fwd], axis=1)


@dataclass(frozen=True)
class Camera:
    """One pinhole camera with two-term radial distortion."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    R_wc: np.ndarray
    centre: np.ndarray
    k1: float = 0.0
    k2: float = 0.0

    @classmethod
    def from_fov(cls, width: int, height: int, hfov_deg: float, centre: Any, target: Any,
                 k1: float = 0.0, k2: float = 0.0) -> "Camera":
        f = 0.5 * width / math.tan(math.radians(hfov_deg) / 2.0)
        return cls(width, height, f, f, width / 2.0, height / 2.0, look_at_rotation(centre, target),
                   np.asarray(centre, dtype=float), k1, k2)

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]])

    @property
    def extrinsic(self) -> np.ndarray:
        """4x4 camera-to-world matrix, as stored in ``camera.json``."""
        E = np.eye(4)
        E[:3, :3] = self.R_wc
        E[:3, 3] = self.centre
        return E

    def to_camera(self, points: Any) -> np.ndarray:
        return (np.atleast_2d(np.asarray(points, dtype=float)) - self.centre) @ self.R_wc

    def project(self, points: Any, near: float = 1e-3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Project world points. Returns ``(uv (N, 2), depth (N,), valid (N,))``."""
        c = self.to_camera(points)
        depth = c[:, 2]
        valid = depth > near
        with np.errstate(invalid="ignore", divide="ignore"):
            x = c[:, 0] / depth
            y = c[:, 1] / depth
        r2 = x * x + y * y
        scale = 1.0 + self.k1 * r2 + self.k2 * r2 * r2
        uv = np.stack([self.fx * x * scale + self.cx, self.fy * y * scale + self.cy], axis=1)
        uv[~valid] = np.nan
        return uv, depth, valid

    def scaled(self, s: float) -> "Camera":
        """The same camera with its image resized by ``s``."""
        return Camera(max(1, int(round(self.width * s))), max(1, int(round(self.height * s))), self.fx * s,
                      self.fy * s, self.cx * s, self.cy * s, self.R_wc, self.centre, self.k1, self.k2)


@dataclass(frozen=True)
class CameraTrack:
    """Per-frame cameras of a camera that stays at one position (pan, tilt, roll and zoom)."""

    width: int
    height: int
    position_m: np.ndarray       # (3,)
    R_wc: np.ndarray             # (F, 3, 3)
    fx: np.ndarray               # (F,)
    fy: np.ndarray
    cx: np.ndarray
    cy: np.ndarray
    k1: np.ndarray
    k2: np.ndarray

    def __post_init__(self) -> None:
        F = len(self.R_wc)
        for name in ("fx", "fy", "cx", "cy", "k1", "k2"):
            if len(getattr(self, name)) != F:
                raise ValueError(f"camera track column {name!r} has {len(getattr(self, name))} rows, expected {F}")

    @property
    def frame_count(self) -> int:
        return len(self.R_wc)

    @classmethod
    def static(cls, camera: Camera, frame_count: int) -> "CameraTrack":
        F = int(frame_count)
        return cls(camera.width, camera.height, np.asarray(camera.centre, dtype=float),
                   np.repeat(np.asarray(camera.R_wc, dtype=float)[None], F, axis=0), np.full(F, camera.fx),
                   np.full(F, camera.fy), np.full(F, camera.cx), np.full(F, camera.cy), np.full(F, camera.k1),
                   np.full(F, camera.k2))

    @classmethod
    def from_cameras(cls, cameras: list[Camera]) -> "CameraTrack":
        if not cameras:
            raise ValueError("a camera track needs at least one camera")
        first = cameras[0]
        for cam in cameras[1:]:
            if np.linalg.norm(np.asarray(cam.centre) - np.asarray(first.centre)) > 1e-6:
                raise ValueError("v0 camera tracks keep one position; moving cameras are out of scope")
        return cls(first.width, first.height, np.asarray(first.centre, dtype=float),
                   np.stack([np.asarray(c.R_wc, dtype=float) for c in cameras]),
                   np.array([c.fx for c in cameras]), np.array([c.fy for c in cameras]),
                   np.array([c.cx for c in cameras]), np.array([c.cy for c in cameras]),
                   np.array([c.k1 for c in cameras]), np.array([c.k2 for c in cameras]))

    def camera(self, frame: int) -> Camera:
        i = int(frame)
        if not 0 <= i < self.frame_count:
            raise IndexError(f"frame {frame} is outside the camera track (0..{self.frame_count - 1})")
        return Camera(self.width, self.height, float(self.fx[i]), float(self.fy[i]), float(self.cx[i]),
                      float(self.cy[i]), self.R_wc[i], self.position_m, float(self.k1[i]), float(self.k2[i]))

    def project(self, points: Any, frame: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.camera(frame).project(points)

    def to_json(self) -> dict[str, Any]:
        def col(a: np.ndarray) -> list[float]:
            return [float(x) for x in a]

        return {
            "schema": CAMERA_TRACK_SCHEMA_ID,
            "model": "fixed_position_ptz",
            "image_size": [int(self.width), int(self.height)],
            "position_m": col(self.position_m),
            "fx": col(self.fx),
            "fy": col(self.fy),
            "cx": col(self.cx),
            "cy": col(self.cy),
            "quat_xyzw": [col(q) for q in matrix_to_quat(self.R_wc)],
            "k1": col(self.k1),
            "k2": col(self.k2),
        }

    @classmethod
    def from_json(cls, raw: Mapping[str, Any]) -> "CameraTrack":
        F = len(raw["fx"])
        zeros = [0.0] * F
        return cls(int(raw["image_size"][0]), int(raw["image_size"][1]), np.asarray(raw["position_m"], dtype=float),
                   quat_to_matrix(np.asarray(raw["quat_xyzw"], dtype=float).reshape(F, 4)),
                   np.asarray(raw["fx"], dtype=float), np.asarray(raw["fy"], dtype=float),
                   np.asarray(raw["cx"], dtype=float), np.asarray(raw["cy"], dtype=float),
                   np.asarray(raw.get("k1", zeros), dtype=float), np.asarray(raw.get("k2", zeros), dtype=float))
