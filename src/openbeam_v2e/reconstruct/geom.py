# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""Geometry building blocks: rotations, the pinhole camera with radial distortion, and the
pan/tilt/roll/zoom parameterisation of a camera that stays at one position.

Conventions used throughout ``openbeam_v2e``:

* **World frame:** right-handed, Z up, metres. The ground plane is ``z = ground_z`` (0 unless the
  configuration says otherwise).
* **Camera frame:** x to the right of the image, y down the image, z along the optical axis.
  A world point ``p`` has camera coordinates ``R_cw @ (p - centre)``.
* **Pixels:** ``u`` to the right, ``v`` down. Pixel ``(col, row)`` covers ``[col, col + 1) x
  [row, row + 1)``, so its centre is ``(col + 0.5, row + 0.5)`` and the default principal point
  ``(width / 2, height / 2)`` is the exact image centre.
* **Distortion:** two-term radial model on normalised coordinates,
  ``x_d = x (1 + k1 r^2 + k2 r^4)``.
* **Pan/tilt/roll:** at zero angles the camera looks along world +Y with image right = world +X
  and image down = world -Z. Pan turns the view about world +Z (counter-clockwise seen from
  above), tilt raises the optical axis, roll turns the image about the optical axis.
* **Quaternions** are ``(x, y, z, w)``, as in ``scipy.spatial.transform.Rotation``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Mapping

import numpy as np
from scipy.spatial.transform import Rotation

# Camera axes expressed in world coordinates at pan = tilt = roll = 0 (columns: x, y, z).
_BASE_WC = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])


def rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rotvec_to_matrix(w: Any) -> np.ndarray:
    return Rotation.from_rotvec(np.asarray(w, float)).as_matrix()


def matrix_to_rotvec(R: Any) -> np.ndarray:
    return Rotation.from_matrix(np.asarray(R, float)).as_rotvec()


def matrix_to_quat(R: Any) -> np.ndarray:
    """Rotation matrix -> quaternion (x, y, z, w) with w >= 0."""
    q = Rotation.from_matrix(np.asarray(R, float)).as_quat()
    return -q if q[3] < 0 else q


def quat_to_matrix(q: Any) -> np.ndarray:
    return Rotation.from_quat(np.asarray(q, float)).as_matrix()


def yaw_pitch_roll(R: Any) -> tuple[float, float, float]:
    """Body-to-world rotation -> (yaw, pitch, roll) in radians, intrinsic Z-Y-X order.

    Body frame: +x forward, +y left, +z up. Yaw is the heading of +x about world +Z, pitch is
    positive nose-down (right-handed about body +y), roll is right-handed about body +x.
    """
    yaw, pitch, roll = Rotation.from_matrix(np.asarray(R, float)).as_euler("ZYX")
    return float(yaw), float(pitch), float(roll)


def from_yaw_pitch_roll(yaw: float, pitch: float = 0.0, roll: float = 0.0) -> np.ndarray:
    return Rotation.from_euler("ZYX", [yaw, pitch, roll]).as_matrix()


def geodesic_angle(Ra: Any, Rb: Any) -> float:
    """Angle in radians of the rotation that takes ``Ra`` to ``Rb``."""
    M = np.asarray(Ra, float).T @ np.asarray(Rb, float)
    c = (np.trace(M) - 1.0) / 2.0
    return float(math.acos(max(-1.0, min(1.0, c))))


def wrap_angle(a: Any) -> Any:
    """Wrap angles to [-pi, pi)."""
    return (np.asarray(a, float) + math.pi) % (2.0 * math.pi) - math.pi


# --------------------------------------------------------------------------- pan / tilt / roll

def ptz_rotation(pan: float, tilt: float, roll: float = 0.0) -> np.ndarray:
    """World-to-camera rotation ``R_cw`` for the given pan, tilt and roll (radians)."""
    R_wc = rot_z(pan) @ rot_x(tilt) @ _BASE_WC @ rot_z(roll)
    return R_wc.T


def ptz_angles(R_cw: Any) -> tuple[float, float, float]:
    """Inverse of :func:`ptz_rotation`: ``R_cw`` -> (pan, tilt, roll) in radians."""
    R_wc = np.asarray(R_cw, float).T
    d = R_wc[:, 2]
    tilt = math.asin(max(-1.0, min(1.0, float(d[2]))))
    pan = math.atan2(-float(d[0]), float(d[1]))
    M = _BASE_WC.T @ rot_x(tilt).T @ rot_z(pan).T @ R_wc
    roll = math.atan2(float(M[1, 0]), float(M[0, 0]))
    return pan, tilt, roll


def look_at(centre: Any, target: Any, roll: float = 0.0) -> np.ndarray:
    """``R_cw`` of a camera at ``centre`` whose optical axis passes through ``target``."""
    d = np.asarray(target, float) - np.asarray(centre, float)
    d = d / np.linalg.norm(d)
    tilt = math.asin(max(-1.0, min(1.0, float(d[2]))))
    pan = math.atan2(-float(d[0]), float(d[1]))
    return ptz_rotation(pan, tilt, roll)


# --------------------------------------------------------------------------- camera

@dataclass(frozen=True)
class Intrinsics:
    """Pinhole intrinsics with square pixels by default (``aspect`` = fy / fx)."""

    width: int
    height: int
    f: float
    cx: float
    cy: float
    k1: float = 0.0
    k2: float = 0.0
    aspect: float = 1.0

    @classmethod
    def from_fov(cls, width: int, height: int, hfov_deg: float, **kw: float) -> "Intrinsics":
        f = 0.5 * width / math.tan(math.radians(hfov_deg) / 2.0)
        return cls(width, height, f, width / 2.0, height / 2.0, **kw)

    @property
    def fx(self) -> float:
        return self.f

    @property
    def fy(self) -> float:
        return self.f * self.aspect

    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]])

    def hfov_deg(self) -> float:
        return math.degrees(2.0 * math.atan(0.5 * self.width / self.f))

    def with_focal(self, f: float) -> "Intrinsics":
        return replace(self, f=float(f))

    def scaled(self, s: float) -> "Intrinsics":
        """Intrinsics of the same camera with the image resized by ``s``."""
        return replace(self, width=int(round(self.width * s)), height=int(round(self.height * s)),
                       f=self.f * s, cx=self.cx * s, cy=self.cy * s)

    # normalised <-> distorted normalised coordinates
    def distort(self, xn: np.ndarray) -> np.ndarray:
        if self.k1 == 0.0 and self.k2 == 0.0:
            return xn
        r2 = np.sum(xn * xn, axis=-1, keepdims=True)
        return xn * (1.0 + self.k1 * r2 + self.k2 * r2 * r2)

    def undistort(self, xd: np.ndarray, iterations: int = 8) -> np.ndarray:
        if self.k1 == 0.0 and self.k2 == 0.0:
            return xd
        x = np.array(xd, float, copy=True)
        for _ in range(iterations):  # fixed-point iteration; converges for moderate distortion
            r2 = np.sum(x * x, axis=-1, keepdims=True)
            x = xd / (1.0 + self.k1 * r2 + self.k2 * r2 * r2)
        return x

    def to_pixels(self, xn: np.ndarray) -> np.ndarray:
        xd = self.distort(xn)
        return np.stack([self.fx * xd[..., 0] + self.cx, self.fy * xd[..., 1] + self.cy], axis=-1)

    def to_normalised(self, uv: np.ndarray) -> np.ndarray:
        uv = np.asarray(uv, float)
        xd = np.stack([(uv[..., 0] - self.cx) / self.fx, (uv[..., 1] - self.cy) / self.fy], axis=-1)
        return self.undistort(xd)

    def to_dict(self) -> dict[str, Any]:
        return {"width": self.width, "height": self.height, "f": self.f, "cx": self.cx,
                "cy": self.cy, "k1": self.k1, "k2": self.k2, "aspect": self.aspect}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Intrinsics":
        return cls(int(d["width"]), int(d["height"]), float(d["f"]), float(d["cx"]), float(d["cy"]),
                   float(d.get("k1", 0.0)), float(d.get("k2", 0.0)), float(d.get("aspect", 1.0)))


class Camera:
    """A camera with known intrinsics, world-to-camera rotation ``R_cw`` and centre."""

    __slots__ = ("intrinsics", "R_cw", "centre")

    def __init__(self, intrinsics: Intrinsics, R_cw: Any, centre: Any) -> None:
        self.intrinsics = intrinsics
        self.R_cw = np.asarray(R_cw, float).reshape(3, 3)
        self.centre = np.asarray(centre, float).reshape(3)

    @classmethod
    def from_ptz(cls, intrinsics: Intrinsics, centre: Any, pan: float, tilt: float,
                 roll: float = 0.0) -> "Camera":
        return cls(intrinsics, ptz_rotation(pan, tilt, roll), centre)

    @property
    def width(self) -> int:
        return self.intrinsics.width

    @property
    def height(self) -> int:
        return self.intrinsics.height

    def ptz(self) -> tuple[float, float, float]:
        return ptz_angles(self.R_cw)

    def to_camera(self, points: Any) -> np.ndarray:
        P = np.atleast_2d(np.asarray(points, float))
        return (P - self.centre) @ self.R_cw.T

    def project(self, points: Any) -> tuple[np.ndarray, np.ndarray]:
        """World points (N, 3) -> pixels (N, 2) and depth along the optical axis (N,)."""
        c = self.to_camera(points)
        z = c[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            xn = c[:, :2] / z[:, None]
        return self.intrinsics.to_pixels(xn), z

    def project_directions(self, dirs: Any) -> tuple[np.ndarray, np.ndarray]:
        """World directions (points at infinity) -> pixels and the z component in camera frame."""
        c = np.atleast_2d(np.asarray(dirs, float)) @ self.R_cw.T
        z = c[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            xn = c[:, :2] / z[:, None]
        return self.intrinsics.to_pixels(xn), z

    def rays(self, uv: Any) -> np.ndarray:
        """Unit world-space ray directions through pixels ``uv`` (..., 2)."""
        xn = self.intrinsics.to_normalised(uv)
        d = np.concatenate([xn, np.ones(xn.shape[:-1] + (1,))], axis=-1)
        d = d @ self.R_cw  # R_cw^T applied to row vectors
        return d / np.linalg.norm(d, axis=-1, keepdims=True)

    def pixel_grid_rays(self, step: int = 1) -> np.ndarray:
        """Ray directions through the pixel centres, shape (H // step, W // step, 3)."""
        us = np.arange(0, self.width, step) + 0.5 * step
        vs = np.arange(0, self.height, step) + 0.5 * step
        uu, vv = np.meshgrid(us, vs)
        return self.rays(np.stack([uu, vv], axis=-1))

    def scaled(self, s: float) -> "Camera":
        return Camera(self.intrinsics.scaled(s), self.R_cw, self.centre)

    def with_intrinsics(self, intrinsics: Intrinsics) -> "Camera":
        return Camera(intrinsics, self.R_cw, self.centre)

    def to_dict(self) -> dict[str, Any]:
        pan, tilt, roll = self.ptz()
        return {"intrinsics": self.intrinsics.to_dict(), "centre": self.centre.tolist(),
                "R_cw": self.R_cw.tolist(),
                "pan_deg": math.degrees(pan), "tilt_deg": math.degrees(tilt),
                "roll_deg": math.degrees(roll)}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Camera":
        intr = Intrinsics.from_dict(d["intrinsics"])
        if "R_cw" in d:
            R = np.asarray(d["R_cw"], float)
        else:
            R = ptz_rotation(math.radians(d["pan_deg"]), math.radians(d["tilt_deg"]),
                             math.radians(d.get("roll_deg", 0.0)))
        return cls(intr, R, d["centre"])

    def __repr__(self) -> str:
        pan, tilt, roll = (math.degrees(a) for a in self.ptz())
        return (f"Camera(f={self.intrinsics.f:.1f}px, centre={np.round(self.centre, 3).tolist()}, "
                f"pan={pan:.2f}, tilt={tilt:.2f}, roll={roll:.2f} deg)")


def intersect_plane(origin: Any, dirs: Any, normal: Any = (0.0, 0.0, 1.0),
                    offset: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Intersect rays ``origin + t dirs`` with the plane ``normal . x = offset``.

    Returns the points and ``t``; rays parallel to or pointing away from the plane get
    ``t = inf`` and NaN points.
    """
    o = np.asarray(origin, float)
    d = np.asarray(dirs, float)
    n = np.asarray(normal, float)
    denom = d @ n
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (offset - o @ n) / denom
    t = np.where((np.abs(denom) > 1e-12) & (t > 0), t, np.inf)
    pts = o + d * np.where(np.isfinite(t), t, np.nan)[..., None]
    return pts, t


def homography_between(cam_a: Camera, cam_b: Camera) -> np.ndarray:
    """Pixel homography from camera ``a`` to camera ``b`` sharing one centre (no distortion).

    For two views from the same position (pan/tilt/roll/zoom only) this maps undistorted
    pixels exactly, whatever the depth of the scene point.
    """
    Ka, Kb = cam_a.intrinsics.K(), cam_b.intrinsics.K()
    return Kb @ cam_b.R_cw @ cam_a.R_cw.T @ np.linalg.inv(Ka)
