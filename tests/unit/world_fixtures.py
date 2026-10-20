# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""Small synthetic worlds for the core tests (generated here; no footage involved).

The default scene: a 30 fps fixed camera looking at flat ground, a wall, a car-sized box that
drives along +x, decelerates and stops, and a crate that falls from 2 m, bounces once and comes
to rest. All motion is written in closed form so tests know the exact answers.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.spatial.transform import Rotation

from openbeam_v2e.schema.common import Measured
from openbeam_v2e.world import Camera, CameraTrack, EventInterval, ObjectTrack, StaticMesh, World
from openbeam_v2e.world.geometry import box_mesh

FPS = 30.0
G = 9.81
CAR_DIMS = (4.4, 1.8, 1.4)
CRATE = 0.6


def camera() -> Camera:
    return Camera.from_fov(320, 180, 70.0, centre=(0.0, -22.0, 3.0), target=(2.0, 0.0, 0.8))


def ground() -> StaticMesh:
    v = np.array([[-40, -15, 0], [40, -15, 0], [40, 15, 0], [-40, 15, 0]], dtype=float)
    f = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    return StaticMesh(1, v, f, "ground", "drivable")


def wall(x: float = 18.0) -> StaticMesh:
    v, f = box_mesh(0.5, 8.0, 2.0, centre=(x, 0.0, 1.0))
    return StaticMesh(2, v, f, "wall", "collision")


def car_track(F: int, speed: float = 8.0, stop_frame: int = 45, yaw_rate: float = 0.0,
              start_x: float = -12.0) -> tuple[np.ndarray, np.ndarray]:
    """Constant speed along +x, then uniform deceleration to rest at ``stop_frame``."""
    t = np.arange(F) / FPS
    t_brake = (stop_frame - 15) / FPS
    t_stop = stop_frame / FPS
    decel = speed / (t_stop - t_brake)
    x = np.empty(F)
    for i, ti in enumerate(t):
        if ti <= t_brake:
            x[i] = start_x + speed * ti
        elif ti <= t_stop:
            dt = ti - t_brake
            x[i] = start_x + speed * t_brake + speed * dt - 0.5 * decel * dt * dt
        else:
            x[i] = start_x + speed * t_brake + 0.5 * speed * (t_stop - t_brake)
    pos = np.stack([x, np.zeros(F), np.zeros(F)], axis=1)
    yaw = yaw_rate * np.minimum(t, t_stop)
    quat = Rotation.from_euler("z", yaw[:, None]).as_quat()
    return pos, quat


def crate_track(F: int, drop: float = 2.0, x: float = 4.0, restitution: float = 0.5) -> tuple[np.ndarray, np.ndarray, dict]:
    """Free fall from ``drop`` metres, one bounce, then rest; the body origin is the box centre."""
    t = np.arange(F) / FPS
    h0 = CRATE / 2.0
    t1 = math.sqrt(2.0 * drop / G)
    v1 = G * t1 * restitution
    t2 = t1 + 2.0 * v1 / G
    z = np.empty(F)
    for i, ti in enumerate(t):
        if ti <= t1:
            z[i] = h0 + drop - 0.5 * G * ti * ti
        elif ti <= t2:
            dt = ti - t1
            z[i] = h0 + v1 * dt - 0.5 * G * dt * dt
        else:
            z[i] = h0
    pos = np.stack([np.full(F, x), np.full(F, 3.0), z], axis=1)
    quat = np.tile([0.0, 0.0, 0.0, 1.0], (F, 1))
    return pos, quat, {"t_contact1": t1, "t_contact2": t2, "apex2": v1 * v1 / (2 * G)}


def car_object(F: int, **kw) -> ObjectTrack:
    hv, hf = box_mesh(*CAR_DIMS, centre=(0.0, 0.0, CAR_DIMS[2] / 2))
    pos, quat = car_track(F, **kw)
    dims = {"length": Measured(CAR_DIMS[0], 0.05), "width": Measured(CAR_DIMS[1], 0.05),
            "height": Measured(CAR_DIMS[2], 0.05)}
    return ObjectTrack("car", "vehicle.car", hv, hf, pos, quat, sigma_position_m=np.full(F, 0.1),
                       sigma_attitude_rad=np.full(F, math.radians(2.0)), dims_m=dims)


def crate_object(F: int, **kw) -> ObjectTrack:
    hv, hf = box_mesh(CRATE, CRATE, CRATE)
    pos, quat, _ = crate_track(F, **kw)
    dims = {k: Measured(CRATE, 0.02) for k in ("length", "width", "height")}
    return ObjectTrack("crate", "prop.box", hv, hf, pos, quat, sigma_position_m=np.full(F, 0.05),
                       sigma_attitude_rad=np.full(F, math.radians(3.0)), dims_m=dims)


def synthetic_world(F: int = 75, kind: str = "reconstruction", car_kw: dict | None = None,
               crate_kw: dict | None = None) -> World:
    objects = [car_object(F, **(car_kw or {})), crate_object(F, **(crate_kw or {}))]
    t1 = crate_track(F, **(crate_kw or {}))[2]["t_contact1"]
    contact = int(math.floor(t1 * FPS))
    stop = (car_kw or {}).get("stop_frame", 45)
    events = [EventInterval("crate_contact", (contact, contact + 1), "contact", ("crate",), "synthetic"),
              EventInterval("car_rest", (stop, stop + 1), "rest", ("car",), "synthetic")]
    return World(FPS, F, CameraTrack.static(camera(), F), [ground(), wall()], objects, events, kind=kind,
                        camera_sigma_px=0.5)
