# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""Synthetic clips with exact ground truth, rendered by :mod:`openbeam_v2e.reconstruct.render`.

The generators build small scenes from first principles (no engine, no external data): a
textured ground plane with square tiles, a few upright posts that serve as landmarks, a camera at
a fixed position that pans, tilts and zooms, and moving objects whose poses come from simple
kinematics (a car following a smooth curve, a box thrown through the air). Every clip carries the true
camera per frame, the true pose of every object per frame and per-object masks, so tests and
benchmarks can measure how well a perception model or the whole reconstruction recovers them.

All numbers below are round design values chosen for test coverage; none is measured from any
footage.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from openbeam_v2e.reconstruct.geom import (
    Camera,
    Intrinsics,
    from_yaw_pitch_roll,
    look_at,
    ptz_angles,
    ptz_rotation,
    rotvec_to_matrix,
)
from openbeam_v2e.reconstruct.hulls import Hull, make_hull
from openbeam_v2e.reconstruct.render import RenderObject, Scene, render


# --------------------------------------------------------------------------- ground texture

def _hash01(i: np.ndarray, j: np.ndarray, salt: int) -> np.ndarray:
    """Deterministic pseudo-random values in [0, 1) per integer cell."""
    h = (i.astype(np.int64) * 73856093) ^ (j.astype(np.int64) * 19349663) ^ (salt * 83492791)
    h = (h ^ (h >> 13)) * 1274126177
    h = h ^ (h >> 16)
    return (h & 0xFFFF).astype(float) / 65536.0


def tiled_ground(tile_m: float = 2.0, seed: int = 0) -> Callable[[np.ndarray], np.ndarray]:
    """Checkered ground with per-tile brightness variation and small square marks.

    Tile corners sit at integer multiples of ``tile_m`` in x and y, which gives calibration tests
    exactly known world points, and the marks give camera trackers texture to follow.
    """

    def shade(points: np.ndarray) -> np.ndarray:
        x, y = points[:, 0] / tile_m, points[:, 1] / tile_m
        i, j = np.floor(x), np.floor(y)
        checker = ((i + j) % 2 == 0).astype(float)
        var = _hash01(i, j, seed)
        base = 0.32 + 0.30 * checker + 0.10 * (var - 0.5)
        # one small dark mark per tile at a pseudo-random spot
        mx = 0.2 + 0.6 * _hash01(i, j, seed + 1)
        my = 0.2 + 0.6 * _hash01(i, j, seed + 2)
        fx, fy = x - i, y - j
        mark = (np.abs(fx - mx) < 0.07) & (np.abs(fy - my) < 0.07)
        g = np.where(mark, base * 0.35, base)
        tint = np.stack([g * 1.0, g * 0.97, g * 0.9], axis=1)
        return np.clip(tint, 0.0, 1.0)

    return shade


# --------------------------------------------------------------------------- trajectories

@dataclass
class PoseSequence:
    """Poses per frame: body origin (T, 3) and body-to-world rotation (T, 3, 3)."""

    position: np.ndarray
    R_wb: np.ndarray

    def __len__(self) -> int:
        return len(self.position)


def unicycle_path(n_frames: int, fps: float, start_xy: Sequence[float], yaw0: float, speed0: float,
                  accel: float = 0.0, yaw_rate0: float = 0.0, yaw_accel: float = 0.0,
                  ground_z: float = 0.0, substeps: int = 20) -> PoseSequence:
    """A vehicle on flat ground: speed and yaw rate vary linearly with time (no slip)."""
    dt = 1.0 / fps / substeps
    x, y, yaw = float(start_xy[0]), float(start_xy[1]), float(yaw0)
    pos, rot = [], []
    t = 0.0
    for k in range(n_frames * substeps):
        if k % substeps == 0:
            pos.append([x, y, ground_z])
            rot.append(from_yaw_pitch_roll(yaw))
        v = max(0.0, speed0 + accel * t)
        w = yaw_rate0 + yaw_accel * t
        x += v * math.cos(yaw) * dt
        y += v * math.sin(yaw) * dt
        yaw += w * dt
        t += dt
    return PoseSequence(np.array(pos), np.array(rot))


def ballistic_path(n_frames: int, fps: float, p0: Sequence[float], v0: Sequence[float],
                   R0: Any, omega: Sequence[float], g: float = 9.81) -> PoseSequence:
    """Airborne motion: constant gravity, constant angular velocity (world frame)."""
    t = np.arange(n_frames) / fps
    p = np.asarray(p0, float)[None] + np.asarray(v0, float)[None] * t[:, None]
    p[:, 2] -= 0.5 * g * t * t
    R0 = np.asarray(R0, float)
    w = np.asarray(omega, float)
    R = np.array([rotvec_to_matrix(w * ti) @ R0 for ti in t])
    return PoseSequence(p, R)


# --------------------------------------------------------------------------- clip container

@dataclass
class SyntheticObject:
    name: str
    label: int
    hull: Hull
    poses: PoseSequence
    colour: tuple[float, float, float]
    object_class: str


@dataclass
class SyntheticClip:
    """Rendered frames plus exact truth."""

    frames: np.ndarray                    # (T, H, W, 3) uint8
    fps: float
    cameras: list[Camera]
    objects: list[SyntheticObject]
    masks: dict[str, np.ndarray]          # name -> (T, H, W) bool (visible pixels)
    landmarks: np.ndarray                 # (N, 3) world points with known coordinates
    ground_z: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n_frames(self) -> int:
        return len(self.frames)

    def object(self, name: str) -> SyntheticObject:
        for o in self.objects:
            if o.name == name:
                return o
        raise KeyError(name)

    def frame_source(self) -> Any:
        from openbeam_v2e.reconstruct.video import ArrayFrames

        return ArrayFrames(self.frames, fps=self.fps)

    def landmark_observations(self, frame: int, noise_px: float = 0.0, seed: int = 0,
                              margin: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
        """Landmarks visible in ``frame``: (world points, pixel observations with optional noise)."""
        cam = self.cameras[frame]
        uv, z = cam.project(self.landmarks)
        ok = (z > 0.1) & (uv[:, 0] > margin) & (uv[:, 0] < cam.width - margin) \
            & (uv[:, 1] > margin) & (uv[:, 1] < cam.height - margin)
        rng = np.random.default_rng(seed)
        obs = uv[ok] + rng.normal(0.0, noise_px, size=(int(ok.sum()), 2)) if noise_px > 0 else uv[ok]
        return self.landmarks[ok], obs

    def truth_dict(self) -> dict[str, Any]:
        cams = []
        for c in self.cameras:
            pan, tilt, roll = ptz_angles(c.R_cw)
            cams.append({"f": c.intrinsics.f, "pan_deg": math.degrees(pan),
                         "tilt_deg": math.degrees(tilt), "roll_deg": math.degrees(roll)})
        c0 = self.cameras[0]
        return {
            "schema": "v2e.synthetic-truth/1",
            "fps": self.fps,
            "n_frames": self.n_frames,
            "ground_z": self.ground_z,
            "camera": {"centre": c0.centre.tolist(), "intrinsics": c0.intrinsics.to_dict(),
                       "frames": cams},
            "objects": [{"name": o.name, "label": o.label, "class": o.object_class,
                         "hull": o.hull.to_dict(),
                         "position": o.poses.position.tolist(),
                         "R_wb": o.poses.R_wb.tolist()} for o in self.objects],
            "landmarks": self.landmarks.tolist(),
            "meta": self.meta,
        }

    def write(self, out_dir: str | Path, masks: bool = True) -> Path:
        """Write frames (PNG), masks (PNG per object and frame) and ``truth.json``."""
        from openbeam_v2e.reconstruct.video import write_png

        out = Path(out_dir)
        (out / "frames").mkdir(parents=True, exist_ok=True)
        for i, f in enumerate(self.frames):
            write_png(out / "frames" / f"{i:06d}.png", f)
        if masks:
            for name, m in self.masks.items():
                d = out / "masks" / name
                d.mkdir(parents=True, exist_ok=True)
                for i, mi in enumerate(m):
                    write_png(d / f"{i:06d}.png", (mi * 255).astype(np.uint8))
        (out / "truth.json").write_text(json.dumps(self.truth_dict(), indent=1))
        return out


def _box_down(a: np.ndarray, s: int) -> np.ndarray:
    h, w = a.shape[0] // s, a.shape[1] // s
    return a[: h * s, : w * s].reshape(h, s, w, s, *a.shape[2:]).mean(axis=(1, 3))


def _coverage_mask(sub: np.ndarray, s: int) -> np.ndarray:
    """Pixels covered by more than half; exact ties alternate in a checkerboard (no size bias)."""
    cov = _box_down(sub.astype(float), s)
    h, w = cov.shape
    parity = (np.add.outer(np.arange(h), np.arange(w)) % 2) == 0
    return (cov > 0.5 + 1e-9) | ((np.abs(cov - 0.5) <= 1e-9) & parity)


def render_clip(cameras: Sequence[Camera], objects: Sequence[SyntheticObject], scene: Scene,
                fps: float, landmarks: np.ndarray, noise_sigma: float = 1.5, seed: int = 0,
                meta: dict[str, Any] | None = None, supersample: int = 2) -> SyntheticClip:
    """Render every frame (anti-aliased by ``supersample`` x ``supersample`` rays per pixel).

    Masks mark the pixels that an object covers by more than half (ties alternate).
    """
    rng = np.random.default_rng(seed)
    frames = []
    masks = {o.name: [] for o in objects}
    s = max(1, int(supersample))
    for t, cam in enumerate(cameras):
        robj = [RenderObject(o.label, o.hull, o.poses.R_wb[t], o.poses.position[t], o.colour)
                for o in objects]
        res = render(cam.scaled(s) if s > 1 else cam, scene, robj)
        img = (_box_down(res.rgb, s) if s > 1 else res.rgb) * 255.0
        if noise_sigma > 0:
            img = img + rng.normal(0.0, noise_sigma, img.shape)
        frames.append(np.clip(np.round(img), 0, 255).astype(np.uint8))
        for o in objects:
            m = res.ids == o.label
            masks[o.name].append(_coverage_mask(m, s) if s > 1 else m)
    return SyntheticClip(np.array(frames), fps, list(cameras), list(objects),
                         {k: np.array(v) for k, v in masks.items()}, landmarks,
                         scene.ground_z, meta or {})


def _posts(positions: Sequence[Sequence[float]], size: float = 0.3, height: float = 2.0,
           first_label: int = 100) -> tuple[list[RenderObject], np.ndarray]:
    hull = make_hull("box", (size, size, height))
    objs, tops = [], []
    palette = [(0.9, 0.85, 0.2), (0.2, 0.75, 0.9), (0.95, 0.5, 0.15), (0.6, 0.3, 0.8)]
    for k, (x, y) in enumerate(positions):
        objs.append(RenderObject(first_label + k, hull, np.eye(3), np.array([x, y, 0.0]),
                                 palette[k % len(palette)]))
        h = size / 2
        tops += [[x + sx * h, y + sy * h, height] for sx in (-1, 1) for sy in (-1, 1)]
    return objs, np.array(tops)


def _ground_corners(x_range: Sequence[float], y_range: Sequence[float], tile: float) -> np.ndarray:
    xs = np.arange(math.ceil(x_range[0] / tile), math.floor(x_range[1] / tile) + 1) * tile
    ys = np.arange(math.ceil(y_range[0] / tile), math.floor(y_range[1] / tile) + 1) * tile
    return np.array([[x, y, 0.0] for x in xs for y in ys])


def ptz_follow(centre: np.ndarray, targets: np.ndarray, focal: np.ndarray, intr: Intrinsics,
               lag_frames: float = 3.0, roll_deg: float = 0.0, tilt_offset_deg: float = 0.0,
               jitter_deg: float = 0.0, seed: int = 0) -> list[Camera]:
    """Cameras at one position that follow ``targets`` with a smoothed lag, like an operator."""
    rng = np.random.default_rng(seed)
    T = len(targets)
    pans, tilts = [], []
    for t in range(T):
        pan, tilt, _ = ptz_angles(look_at(centre, targets[t]))
        pans.append(pan)
        tilts.append(tilt)
    pans_a = np.unwrap(np.array(pans))
    tilts_a = np.array(tilts)
    a = 1.0 / max(1.0, lag_frames)
    sp, st = [pans_a[0]], [tilts_a[0]]
    for t in range(1, T):
        sp.append(sp[-1] + a * (pans_a[t] - sp[-1]))
        st.append(st[-1] + a * (tilts_a[t] - st[-1]))
    jit = np.cumsum(rng.normal(0.0, math.radians(jitter_deg), size=(T, 2)), axis=0) if jitter_deg else np.zeros((T, 2))
    cams = []
    for t in range(T):
        R = ptz_rotation(sp[t] + jit[t, 0], st[t] + math.radians(tilt_offset_deg) + jit[t, 1],
                         math.radians(roll_deg))
        cams.append(Camera(intr.with_focal(float(focal[t])), R, centre))
    return cams


# --------------------------------------------------------------------------- scenarios

def ground_vehicle_clip(width: int = 480, height: int = 270, fps: float = 30.0, n_frames: int = 60,
                        hfov_deg: float = 50.0, zoom: float = 1.2, k1: float = 0.0,
                        roll_deg: float = 0.5, ptz: bool = True, noise_sigma: float = 1.5,
                        template: str = "vehicle", dims: Sequence[float] = (4.5, 1.8, 1.5),
                        seed: int = 0, supersample: int = 2) -> SyntheticClip:
    """A car on flat ground filmed from a raised fixed position by a panning, zooming camera.

    The car accelerates through a gentle curve; the camera follows it with a lag, zooms in by the
    factor ``zoom`` and has a small constant roll. With ``ptz=False`` the camera stays still.
    """
    intr = Intrinsics.from_fov(width, height, hfov_deg, k1=k1)
    centre = np.array([0.0, -20.0, 5.0])
    path = unicycle_path(n_frames, fps, start_xy=(-11.0, 0.0), yaw0=math.radians(5.0), speed0=8.0,
                         accel=1.0, yaw_rate0=0.12, yaw_accel=0.05)
    car = SyntheticObject("car", 1, make_hull(template, dims), path, (0.75, 0.12, 0.1), "vehicle.car")
    if ptz:
        targets = path.position + np.array([0.0, 0.0, 0.8])
        focal = intr.f * np.linspace(1.0, zoom, n_frames)
        cams = ptz_follow(centre, targets, focal, intr, lag_frames=4.0, roll_deg=roll_deg, seed=seed)
    else:
        R = look_at(centre, (0.0, 0.0, 0.5), math.radians(roll_deg))
        cams = [Camera(intr, R, centre)] * n_frames
    posts, tops = _posts([(-8.0, 6.0), (-2.0, 9.0), (4.0, 7.0), (9.0, 5.0)])
    scene = Scene(ground_shader=tiled_ground(2.0, seed), static_objects=posts)
    landmarks = np.concatenate([_ground_corners((-24, 24), (-12, 30), 2.0), tops])
    return render_clip(cams, [car], scene, fps, landmarks, noise_sigma, seed,
                       {"scenario": "ground_vehicle", "tile_m": 2.0, "post_tops": len(tops)},
                       supersample)


def airborne_box_clip(width: int = 400, height: int = 300, fps: float = 30.0, n_frames: int = 24,
                     hfov_deg: float = 45.0, noise_sigma: float = 1.5,
                     dims: Sequence[float] = (0.8, 0.6, 0.4), seed: int = 0,
                     supersample: int = 2) -> SyntheticClip:
    """A tumbling box thrown through the air in front of a still camera."""
    intr = Intrinsics.from_fov(width, height, hfov_deg)
    centre = np.array([0.0, -9.0, 1.8])
    R0 = from_yaw_pitch_roll(math.radians(20.0), math.radians(10.0), math.radians(5.0))
    path = ballistic_path(n_frames, fps, p0=(-2.2, 0.0, 1.2), v0=(5.0, 0.4, 3.2), R0=R0,
                          omega=(1.5, 2.5, 0.8))
    box = SyntheticObject("box", 1, make_hull("box", dims), path, (0.2, 0.45, 0.85), "prop.box")
    R = look_at(centre, (0.0, 0.0, 1.4))
    cams = [Camera(intr, R, centre)] * n_frames
    posts, tops = _posts([(-3.0, 5.0), (3.0, 6.0)])
    scene = Scene(ground_shader=tiled_ground(1.0, seed), static_objects=posts)
    landmarks = np.concatenate([_ground_corners((-6, 6), (-4, 10), 1.0), tops])
    return render_clip(cams, [box], scene, fps, landmarks, noise_sigma, seed,
                       {"scenario": "airborne_box", "tile_m": 1.0}, supersample)
