# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""The ``fake`` connector: a deterministic rigid-body stand-in for tests and the offline quickstart.

It is a test double, not a physics engine: every run it produces has the claim class
``test_double`` and supports no claim about any engine. It exists so that the whole pipeline
(negotiate, install, spawn, run, ledger, audit, export, score) can be exercised offline and in CI,
bit-identically.

What it simulates, with explicit semi-implicit Euler steps (default 1 ms):

* rigid boxes, spheres and cylinders, and simple four-wheel vehicles (a box body on four
  spring-damper wheels with drive, brake, steering and a toggle parking brake);
* gravity, penalty contacts with Coulomb friction against static planes and oriented boxes, and
  between actors; restitution is set per object through the contact damping;
* nothing else: no deformation, no detachment, no aerodynamics beyond a drag term on vehicles.

Content comes from :func:`compile_scene`, which turns a ``v2e.scene/1`` mapping into a
``v2e.compiled/1`` scene for this connector: an implicit ground plane, static actors as boxes
(from a primitive or a world-package mesh's bounding box), and one object per dynamic actor
sized from its measured dimensions. The reference point of every object (spawn poses and rows) is
the centre of its bounding box.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from openbeam_v2e import __version__
from openbeam_v2e.backends.base import (
    NO_RECORDING,
    STEPPED,
    ActorRef,
    ActuatorChannel,
    Attestation,
    Backend,
    BackendError,
    BackendManifest,
    CameraSpec,
    Capabilities,
    ChannelKind,
    CompiledScene,
    ContentOrigin,
    ControllerHost,
    Effect,
    Frame,
    InstallReceipt,
    Layer,
    LevelRef,
    Lockstep,
    ObjectRef,
    Package,
    Plan,
    Pose,
    RefusedCall,
    RegimeSupport,
    Repeatability,
    RunResult,
    RunSpec,
    Source,
    SpawnRecord,
    SpawnSpec,
    StateRow,
    sha256_json,
)
from openbeam_v2e.backends.ledger import AuditPolicy
from openbeam_v2e.backends.lockstep import run_lockstep

__all__ = ["FakeBackend", "compile_scene", "compile_package", "FAKE_CHANNELS", "STANDARD_GRAVITY"]

STANDARD_GRAVITY = 9.80665
FAKE_DIALECT = "fake/primitives"

FAKE_CHANNELS = (
    ActuatorChannel("drive.throttle", "drive", ChannelKind.CONTINUOUS, (0.0, 1.0), mechanism="fake.vehicle/throttle"),
    ActuatorChannel("drive.brake", "brake", ChannelKind.CONTINUOUS, (0.0, 1.0), mechanism="fake.vehicle/brake"),
    ActuatorChannel("drive.steer", "steer", ChannelKind.CONTINUOUS, (-1.0, 1.0), mechanism="fake.vehicle/steer"),
    ActuatorChannel("drive.parking_brake", "parking_brake", ChannelKind.TOGGLE, (0.0, 1.0),
                    mechanism="fake.vehicle/parking_brake"),
)

# Defaults for content the scene does not measure. Each is a round number chosen for a plausible
# test double, not taken from any measurement; compiled objects record them as "assumed".
_VEHICLE_DIMS = (4.5, 1.8, 1.45)
_VEHICLE_MASS = 1500.0
_PROP_DIMS = {"box": (0.5, 0.5, 0.5), "sphere": (0.3, 0.3, 0.3), "cylinder": (0.6, 0.3, 0.3)}
_PROP_DENSITY = 400.0          # kg/m^3
_RESTITUTION = {"vehicle": 0.2, "box": 0.3, "sphere": 0.6, "cylinder": 0.4}
_FRICTION = {"vehicle": 0.5, "box": 0.5, "sphere": 0.4, "cylinder": 0.5}
_GROUND_FRICTION = 0.8


# --------------------------------------------------------------------------- math helpers


def _rotmat(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ])


def _box_samples(center: np.ndarray, half: np.ndarray) -> np.ndarray:
    """Corners, edge midpoints and face centres of a box (26 points)."""
    pts = []
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            for k in (-1, 0, 1):
                if (i, j, k) != (0, 0, 0):
                    pts.append(center + half * np.array([i, j, k], dtype=float))
    return np.array(pts)


def _box_depth(local: np.ndarray, half: np.ndarray, radius: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Penetration depth (> 0 means contact) and outward normal of points with radius against a box."""
    sign = np.where(local >= 0.0, 1.0, -1.0)
    q = np.abs(local) - half
    outside = np.maximum(q, 0.0)
    dist_out = np.sqrt(np.sum(outside * outside, axis=1))
    is_out = dist_out > 1e-12
    axis = np.argmax(q, axis=1)
    inside_pen = -q[np.arange(len(q)), axis]
    depth = np.where(is_out, radius - dist_out, radius + inside_pen)
    n_out = sign * outside / np.where(is_out, dist_out, 1.0)[:, None]
    n_in = np.zeros_like(local)
    n_in[np.arange(len(q)), axis] = sign[np.arange(len(q)), axis]
    normal = np.where(is_out[:, None], n_out, n_in)
    return depth, normal


# --------------------------------------------------------------------------- world


@dataclass
class _Static:
    name: str
    kind: str                      # "plane" | "box" | "convex"
    mu: float
    point: np.ndarray | None = None
    normal: np.ndarray | None = None
    center: np.ndarray | None = None
    rot: np.ndarray | None = None
    half: np.ndarray | None = None
    planes: np.ndarray | None = None   # convex: rows [n, d] with n . x <= d inside

    def depth(self, pts: np.ndarray, radius: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.kind == "plane":
            signed = (pts - self.point) @ self.normal
            return radius - signed, np.broadcast_to(self.normal, pts.shape)
        if self.kind == "convex":
            # largest half-space distance: exact inside the part and near its faces, where contacts happen
            dist = pts @ self.planes[:, :3].T - self.planes[:, 3]
            idx = np.argmax(dist, axis=1)
            return radius - dist[np.arange(len(pts)), idx], self.planes[idx, :3]
        local = (pts - self.center) @ self.rot
        depth, n_local = _box_depth(local, self.half, radius)
        return depth, n_local @ self.rot.T


@dataclass
class _Vehicle:
    hubs: np.ndarray               # (4, 3) body frame, relative to the centre of mass
    front: np.ndarray              # (4,) bool
    rear: np.ndarray
    rest_length: float
    max_compression: float
    k_s: float
    c_s: float
    mu: float
    drive_max_n: float
    power_w: float
    brake_max_n: float
    park_max_n: float
    max_steer_rad: float
    crr: float
    cda_m2: float
    throttle: float = 0.0
    brake: float = 0.0
    steer: float = 0.0
    parking: bool = False


@dataclass
class _Body:
    actor: str
    handle: str
    object: str
    shape: str
    mass: float
    inertia: np.ndarray            # principal moments, body frame
    com_offset: np.ndarray         # reference point -> centre of mass, body frame
    box_center: np.ndarray         # collision box centre relative to the centre of mass
    half: np.ndarray               # collision box half extents
    radius: float                  # sphere radius (0 for other shapes)
    samples: np.ndarray            # collision sample points relative to the centre of mass
    sample_radius: np.ndarray
    k: float
    c: float
    mu: float
    p: np.ndarray
    q: np.ndarray
    v: np.ndarray = field(default_factory=lambda: np.zeros(3))
    w: np.ndarray = field(default_factory=lambda: np.zeros(3))
    vehicle: _Vehicle | None = None
    contacts: set = field(default_factory=set)

    def reference_pose(self) -> tuple[np.ndarray, np.ndarray]:
        rot = _rotmat(self.q)
        return self.p - rot @ self.com_offset, self.q


def _contact_gains(mass: float, shape: str, restitution: float, frequency_hz: float) -> tuple[float, float]:
    supports = 1.0 if shape == "sphere" else 4.0
    omega = 2.0 * math.pi * frequency_hz
    k = mass * omega * omega / supports
    e = min(max(restitution, 1e-3), 0.999)
    zeta = -math.log(e) / math.sqrt(math.pi ** 2 + math.log(e) ** 2)
    c = 2.0 * zeta * math.sqrt(k * mass / supports)
    return k, c


def _make_body(actor: str, handle: str, obj: ObjectRef, pose: Pose) -> _Body:
    geo = dict(obj.geometry)
    shape = geo.get("shape", "box")
    length, width, height = (float(v) for v in (obj.dims_m or _PROP_DIMS.get(shape, _VEHICLE_DIMS)))
    mass = float(obj.mass_kg or geo.get("mass_kg") or 1.0)
    restitution = float(geo.get("restitution", _RESTITUTION.get(shape, 0.3)))
    mu = float(geo.get("mu", _FRICTION.get(shape, 0.5)))
    vehicle = None
    radius = 0.0
    if shape == "vehicle":
        v = geo.get("vehicle", {})
        ride = float(v.get("ride_height_m", min(0.15, 0.2 * height)))
        cg = float(v.get("cg_height_m", 0.38 * height))
        wheel_r = float(v.get("wheel_radius_m", min(0.32, 0.3 * height)))
        wheelbase = float(v.get("wheelbase_m", 0.6 * length))
        track = float(v.get("track_m", 0.86 * width))
        com_offset = np.array([0.0, 0.0, cg - height / 2.0])
        box_center = np.array([0.0, 0.0, ride / 2.0]) - com_offset
        half = np.array([length / 2.0, width / 2.0, (height - ride) / 2.0])
        samples = _box_samples(box_center, half)
        inertia = mass / 12.0 * np.array([width ** 2 + (0.8 * height) ** 2, length ** 2 + (0.8 * height) ** 2,
                                          length ** 2 + width ** 2])
        static_comp = 0.08
        k_s = mass * STANDARD_GRAVITY / (4.0 * static_comp)
        hubs = np.array([[sx * wheelbase / 2.0, sy * track / 2.0, wheel_r - height / 2.0]
                         for sx in (1, -1) for sy in (1, -1)]) - com_offset
        vehicle = _Vehicle(
            hubs=hubs, front=np.array([True, True, False, False]), rear=np.array([False, False, True, True]),
            rest_length=wheel_r + static_comp, max_compression=static_comp + 0.12, k_s=k_s,
            c_s=2.0 * 0.35 * math.sqrt(k_s * mass / 4.0), mu=float(v.get("tyre_mu", 0.9)),
            drive_max_n=0.5 * STANDARD_GRAVITY * mass, power_w=float(v.get("power_w", 90000.0)),
            brake_max_n=0.9 * STANDARD_GRAVITY * mass, park_max_n=0.6 * STANDARD_GRAVITY * mass,
            max_steer_rad=float(v.get("max_steer_rad", 0.55)), crr=0.012, cda_m2=0.65)
        k, c = _contact_gains(mass, "box", restitution, 25.0)
        sample_radius = np.zeros(len(samples))
    elif shape == "sphere":
        radius = max(length, width, height) / 2.0
        com_offset = np.zeros(3)
        box_center = np.zeros(3)
        half = np.full(3, radius)
        samples = np.zeros((1, 3))
        sample_radius = np.array([radius])
        inertia = np.full(3, 0.4 * mass * radius * radius)
        k, c = _contact_gains(mass, "sphere", restitution, 30.0)
    elif shape == "cylinder":
        r = max(width, height) / 2.0
        com_offset = np.zeros(3)
        box_center = np.zeros(3)
        half = np.array([length / 2.0, r, r])
        angles = np.linspace(0.0, 2.0 * math.pi, 12, endpoint=False)
        rims = [np.array([sx * length / 2.0, r * math.cos(a), r * math.sin(a)]) for sx in (-1, 1) for a in angles]
        samples = np.array(rims + [np.array([length / 2.0, 0, 0]), np.array([-length / 2.0, 0, 0])])
        sample_radius = np.zeros(len(samples))
        inertia = np.array([0.5 * mass * r * r, mass * (3 * r * r + length ** 2) / 12.0,
                            mass * (3 * r * r + length ** 2) / 12.0])
        k, c = _contact_gains(mass, "box", restitution, 30.0)
    else:
        com_offset = np.zeros(3)
        box_center = np.zeros(3)
        half = np.array([length, width, height]) / 2.0
        samples = _box_samples(box_center, half)
        sample_radius = np.zeros(len(samples))
        inertia = mass / 12.0 * np.array([width ** 2 + height ** 2, length ** 2 + height ** 2,
                                          length ** 2 + width ** 2])
        k, c = _contact_gains(mass, "box", restitution, 30.0)
    # the reference point need not be the bounding-box centre (world-package hull origins rarely are)
    com_offset = com_offset + np.array(geo.get("ref_offset_m", (0.0, 0.0, 0.0)), dtype=float)
    q = np.array(pose.quat_xyzw, dtype=float)
    p = np.array(pose.position_m, dtype=float) + _rotmat(q) @ com_offset
    return _Body(actor=actor, handle=handle, object=obj.name, shape=shape, mass=mass, inertia=inertia,
                 com_offset=com_offset, box_center=box_center, half=half, radius=radius, samples=samples,
                 sample_radius=sample_radius, k=k, c=c, mu=mu, p=p, q=q, vehicle=vehicle)


class _World:
    def __init__(self, statics: Sequence[_Static], dt: float) -> None:
        self.statics = list(statics)
        self.dt = dt
        self.bodies: list[_Body] = []
        self.steps = 0

    @property
    def t(self) -> float:
        return self.steps * self.dt

    def digest(self) -> str:
        h = hashlib.sha256()
        for b in self.bodies:
            for arr in (b.p, b.q, b.v, b.w):
                h.update(np.ascontiguousarray(arr, dtype="<f8").tobytes())
        h.update(str(self.steps).encode())
        return h.hexdigest()

    def step(self) -> None:
        dt = self.dt
        forces = [np.array([0.0, 0.0, -STANDARD_GRAVITY * b.mass]) for b in self.bodies]
        torques = [np.zeros(3) for _ in self.bodies]
        rots = [_rotmat(b.q) for b in self.bodies]
        for b in self.bodies:
            b.contacts = set()

        def apply(i: int, point: np.ndarray, force: np.ndarray) -> None:
            forces[i] += force.sum(axis=0)
            torques[i] += np.cross(point - self.bodies[i].p, force).sum(axis=0)

        for i, b in enumerate(self.bodies):
            rot = rots[i]
            pts = b.p + b.samples @ rot.T
            for s in self.statics:
                depth, normal = s.depth(pts, b.sample_radius)
                hit = depth > 0.0
                if not np.any(hit):
                    continue
                b.contacts.add(s.name)
                cp = pts[hit] - normal[hit] * b.sample_radius[hit, None]
                vel = b.v + np.cross(b.w, cp - b.p)
                force = self._penalty(depth[hit], normal[hit], vel, b.k, b.c, math.sqrt(b.mu * s.mu))
                apply(i, cp, force)
            if b.vehicle is not None:
                self._wheels(i, b, rot, forces, torques)

        for i in range(len(self.bodies)):
            for j in range(i + 1, len(self.bodies)):
                self._pair(i, j, rots, apply)

        for i, b in enumerate(self.bodies):
            rot = rots[i]
            b.v = b.v + forces[i] / b.mass * dt
            b.p = b.p + b.v * dt
            inertia_world = rot @ np.diag(b.inertia) @ rot.T
            inv_world = rot @ np.diag(1.0 / b.inertia) @ rot.T
            b.w = b.w + inv_world @ (torques[i] - np.cross(b.w, inertia_world @ b.w)) * dt
            dq = _quat_mul(np.array([b.w[0], b.w[1], b.w[2], 0.0]), b.q) * (0.5 * dt)
            q = b.q + dq
            b.q = q / np.linalg.norm(q)
        self.steps += 1

    @staticmethod
    def _penalty(depth: np.ndarray, normal: np.ndarray, vel: np.ndarray, k: float, c: float,
                 mu: float) -> np.ndarray:
        vn = np.sum(vel * normal, axis=1)
        fn = np.maximum(k * depth - c * vn, 0.0)
        vt = vel - vn[:, None] * normal
        speed = np.sqrt(np.sum(vt * vt, axis=1) + 0.02 ** 2)
        return fn[:, None] * normal - (mu * fn / speed)[:, None] * vt

    def _pair(self, i: int, j: int, rots: list[np.ndarray], apply: Any) -> None:
        a, b = self.bodies[i], self.bodies[j]
        reach = np.linalg.norm(a.half) + np.linalg.norm(a.box_center) + np.linalg.norm(b.half) + \
            np.linalg.norm(b.box_center)
        if np.linalg.norm(a.p - b.p) > reach:
            return
        k = min(a.k, b.k)
        c = min(a.c, b.c)
        mu = math.sqrt(a.mu * b.mu)
        for src, dst, si, di in ((a, b, i, j), (b, a, j, i)):
            rs, rd = rots[si], rots[di]
            pts = src.p + src.samples @ rs.T
            if dst.shape == "sphere":
                delta = pts - dst.p
                dist = np.sqrt(np.sum(delta * delta, axis=1))
                depth = src.sample_radius + dst.radius - dist
                normal = delta / np.maximum(dist, 1e-12)[:, None]
            else:
                center = dst.p + rd @ dst.box_center
                depth, n_local = _box_depth((pts - center) @ rd, dst.half, src.sample_radius)
                normal = n_local @ rd.T
            hit = depth > 0.0
            if not np.any(hit):
                continue
            src.contacts.add(dst.actor)
            dst.contacts.add(src.actor)
            cp = pts[hit] - normal[hit] * src.sample_radius[hit, None]
            rel = (src.v + np.cross(src.w, cp - src.p)) - (dst.v + np.cross(dst.w, cp - dst.p))
            force = self._penalty(depth[hit], normal[hit], rel, k, c, mu)
            apply(si, cp, force)
            apply(di, cp, -force)

    def _wheels(self, i: int, b: _Body, rot: np.ndarray, forces: list, torques: list) -> None:
        veh = b.vehicle
        hubs = b.p + veh.hubs @ rot.T
        down = -rot[:, 2]
        bottoms = hubs + down * veh.rest_length
        forward_speed = float(b.v @ rot[:, 0])
        drive_total = veh.throttle * min(veh.drive_max_n, veh.power_w / max(abs(forward_speed), 1.0))
        delta = veh.steer * veh.max_steer_rad
        for s in self.statics:
            depth, normal = s.depth(bottoms, np.zeros(4))
            comp = np.minimum(depth, veh.rest_length + 0.05)
            hit = comp > 0.0
            if not np.any(hit):
                continue
            b.contacts.add(s.name)
            for w in np.nonzero(hit)[0]:
                n = normal[w]
                cp = bottoms[w] + n * comp[w]
                r = cp - b.p
                vel = b.v + np.cross(b.w, r)
                vn = float(vel @ n)
                fs = veh.k_s * comp[w] - veh.c_s * vn
                over = comp[w] - veh.max_compression
                if over > 0.0:
                    fs += 10.0 * veh.k_s * over
                fs = max(fs, 0.0)
                angle = -delta if veh.front[w] else 0.0
                heading = rot @ np.array([math.cos(angle), math.sin(angle), 0.0])
                heading = heading - (heading @ n) * n
                heading /= max(np.linalg.norm(heading), 1e-9)
                lateral = np.cross(n, heading)
                v_long = float(vel @ heading)
                v_lat = float(vel @ lateral)
                f_long = drive_total / 4.0
                f_long -= veh.brake * veh.brake_max_n / 4.0 * math.tanh(v_long / 0.3)
                if veh.parking and veh.rear[w]:
                    f_long -= veh.park_max_n / 2.0 * math.tanh(v_long / 0.1)
                f_long -= veh.crr * fs * math.tanh(v_long / 0.2)
                f_lat = -veh.mu * fs * math.tanh(v_lat / 0.4)
                limit = veh.mu * fs
                total = math.hypot(f_long, f_lat)
                if total > limit > 0.0:
                    f_long *= limit / total
                    f_lat *= limit / total
                force = fs * n + f_long * heading + f_lat * lateral
                forces[i] += force
                torques[i] += np.cross(r, force)
        speed = float(np.linalg.norm(b.v))
        forces[i] += -0.5 * 1.2 * veh.cda_m2 * speed * b.v


# --------------------------------------------------------------------------- connector


def _audit_actuate(args: Mapping[str, Any], row: Mapping[str, Any]) -> str | None:
    known = {c.id for c in FAKE_CHANNELS}
    unknown = sorted(set((args.get("cmd") or {})) - known)
    if unknown:
        return f"call {row.get('seq')}: actuate names undeclared channel(s) {unknown}"
    return None


def _audit_program(args: Mapping[str, Any], row: Mapping[str, Any]) -> str | None:
    if args.get("sha256") not in (row.get("blobs") or []):
        return f"call {row.get('seq')}: program {args.get('actor')!r} was not stored as a blob"
    return None


_POLICY = AuditPolicy(
    connector="fake",
    methods={
        "fake.install": frozenset({"name", "kind", "sha256", "contents"}),
        "fake.open_scene": frozenset({"package", "level_id"}),
        "fake.spawn": frozenset({"actor", "object", "pose"}),
        "fake.load_program": frozenset({"actor", "model", "sha256"}),
        "fake.actuate": frozenset({"actor", "cmd"}),
        "fake.step": frozenset({"steps"}),
        "fake.reset_between_runs": frozenset(),
    },
    checks={"fake.actuate": _audit_actuate, "fake.load_program": _audit_program},
)


class FakeBackend(Backend, Lockstep):
    """The offline test double (see the module docstring)."""

    name = "fake"

    def __init__(self, *, physics_dt_s: float = 0.001, controller_factory: Any = None) -> None:
        if not (math.isfinite(physics_dt_s) and 0.0 < physics_dt_s <= 0.01):
            raise ValueError("physics_dt_s must be in (0, 0.01]")
        self.physics_dt_s = float(physics_dt_s)
        self.controller_factory = controller_factory
        self._packages: dict[str, Package] = {}
        self._level: LevelRef | None = None
        self._world: _World | None = None
        self._spawns: list[SpawnRecord] = []
        self._objects: dict[str, ObjectRef] = {}
        self.ledger = None

    # ------------------------------------------------------------------ helpers

    def _log(self, method: str, args: Mapping[str, Any], result: Any = None, blobs: Sequence[str] = ()) -> None:
        if self.ledger is not None:
            sim_t = self._world.t if self._world is not None else None
            self.ledger.record(method, args, result, sim_t=sim_t, blobs=list(blobs))

    def _require_world(self) -> _World:
        if self._world is None:
            raise BackendError("open a scene first")
        return self._world

    def _body(self, actor: ActorRef) -> _Body:
        world = self._require_world()
        for body in world.bodies:
            if body.handle == actor.handle:
                return body
        raise BackendError(f"no actor with handle {actor.handle!r}")

    # ------------------------------------------------------------------ Backend

    def manifest(self) -> BackendManifest:
        caps = Capabilities(
            channels=FAKE_CHANNELS,
            soft_body=False,
            detachment="none",
            regimes=(RegimeSupport(STEPPED, True, (), Repeatability.BIT_IDENTICAL,
                                   "deterministic by construction; checked by the contract tests"),),
            physics_dt_s=self.physics_dt_s,
            recordings=(NO_RECORDING,),
            spawn_at_pose=True,
            max_controlled_actors=64,
            max_seatable_vehicles=None,
            controller_hosts=frozenset({ControllerHost.CLIENT_LOCKSTEP}),
            observables=frozenset({"position", "attitude", "velocity", "angular_velocity", "yaw", "contacts"}),
            camera_runtime_fov=False,
            instance_mask=False,
        )
        return BackendManifest(backend="fake", kind="test_double", engine=f"openbeam-v2e fake {__version__}",
                               api="python (in process)", dialect=FAKE_DIALECT, capabilities=caps)

    def mechanisms(self, plan: Plan) -> tuple[Effect, ...]:
        return tuple(dataclasses.replace(e, source=Source.TEST_DOUBLE, layer=Layer.ENGINE_CORE) for e in plan.effects)

    def install(self, packages: Sequence[Package]) -> list[InstallReceipt]:
        receipts = []
        for pkg in packages:
            if pkg.kind not in ("fake.level", "fake.objects"):
                raise RefusedCall(f"the fake connector installs only fake.level and fake.objects packages, "
                                  f"not {pkg.kind!r}")
            digest = sha256_json(pkg.contents)
            if pkg.sha256 and pkg.sha256 != digest:
                raise BackendError(f"package {pkg.name!r}: sha256 does not match its contents")
            present = pkg.name in self._packages and sha256_json(self._packages[pkg.name].contents) == digest
            self._log("fake.install", {"name": pkg.name, "kind": pkg.kind, "sha256": digest,
                                       "contents": dict(pkg.contents)})
            self._packages[pkg.name] = pkg
            ids = tuple(sorted(pkg.contents.get("objects", {}))) if pkg.kind == "fake.objects" else (pkg.name,)
            receipts.append(InstallReceipt(pkg.name, digest, "already_present" if present else "installed", ids))
        return receipts

    def open_scene(self, level: LevelRef) -> None:
        pkg = self._packages.get(level.package or "")
        if pkg is None or pkg.kind != "fake.level":
            raise BackendError(f"level package {level.package!r} is not installed")
        self._log("fake.open_scene", {"package": level.package, "level_id": level.level_id})
        self._level = level
        self._world = _World(_statics_from(pkg.contents), self.physics_dt_s)
        self._spawns = []

    def spawn(self, obj: ObjectRef, pose: Pose, actor: str) -> tuple[ActorRef, SpawnRecord]:
        world = self._require_world()
        if world.steps:
            raise RefusedCall("spawning after the run started would be an in-run change; spawn before running")
        if any(b.actor == actor for b in world.bodies):
            raise RefusedCall(f"actor {actor!r} is already spawned")
        installed = {name for p in self._packages.values() if p.kind == "fake.objects"
                     for name in p.contents.get("objects", {})}
        if obj.name not in installed:
            raise BackendError(f"object {obj.name!r} is not in an installed package")
        handle = f"fake:{len(world.bodies)}"
        body = _make_body(actor, handle, obj, pose)
        world.bodies.append(body)
        self._objects[handle] = obj
        p, q = body.reference_pose()
        actual = Pose(tuple(p), tuple(q))
        record = SpawnRecord(actor, pose, actual, pose.distance_to_m(actual), pose.angle_to_deg(actual),
                             "fake.spawn-at-rest")
        self._spawns.append(record)
        self._log("fake.spawn", {"actor": actor, "object": obj.name,
                                 "pose": {"position_m": list(pose.position_m), "quat_xyzw": list(pose.quat_xyzw)}},
                  world.digest())
        return ActorRef(actor, handle, obj.name, obj.seatable), record

    def run(self, actors: Sequence[ActorRef], programs: Mapping[str, Mapping[str, Any]], spec: RunSpec,
            run_dir: Path) -> RunResult:
        world = self._require_world()
        if spec.regime != STEPPED:
            raise RefusedCall(f"the fake connector runs only the {STEPPED!r} regime")
        if programs and spec.host is not ControllerHost.CLIENT_LOCKSTEP:
            raise RefusedCall("the fake connector hosts controllers in client lockstep only")
        started = time.perf_counter()
        program_hashes = []
        for actor, program in sorted(programs.items()):
            digest = self.ledger.store_blob(_canonical(program)) if self.ledger is not None else sha256_json(program)
            self._log("fake.load_program", {"actor": actor, "model": program.get("model"), "sha256": digest},
                      blobs=[digest])
            program_hashes.append((actor, digest))
        plan = _current_plan(run_dir)
        manifest = self.manifest()
        outcome = run_lockstep(self, actors, programs, spec, Path(run_dir),
                               plan_channels=plan.channels if plan else {a: tuple(c.id for c in FAKE_CHANNELS)
                                                                          for a in programs},
                               channels=FAKE_CHANNELS, physics_dt_s=self.physics_dt_s,
                               controller_factory=self.controller_factory, engine=manifest.engine,
                               t0_s=spec.t0_s if spec.t0_s is not None else _t0(plan))
        attestation = Attestation(
            spawns=tuple(self._spawns), initial_velocity_mps=0.0, teleports=0, in_run_resets=0, removed_actors=0,
            regime=spec.regime, programs=tuple(program_hashes),
            ledger=Path(self.ledger.directory) if self.ledger is not None else Path(run_dir) / "ledger",
            ledger_head_sha256=self.ledger.head_sha256 if self.ledger is not None else "", audit="pending")
        content = tuple(_content_origin(r.actor, self._objects[r.handle]) for r in actors)
        effects = self.mechanisms(plan) if plan else ()
        return RunResult(manifest=manifest, plan=plan, effects=effects, content=content, attestation=attestation,
                         rows=outcome.rows, events=outcome.events,
                         t0_s=spec.t0_s if spec.t0_s is not None else _t0(plan), video=None, frames=None,
                         sim_s=outcome.sim_s, wall_s=round(time.perf_counter() - started, 3))

    def reset_between_runs(self) -> None:
        self._log("fake.reset_between_runs", {})
        if self._level is not None:
            self.open_scene(self._level)

    def close(self) -> None:
        self._world = None

    @classmethod
    def audit_policy(cls) -> AuditPolicy:
        return _POLICY

    # ------------------------------------------------------------------ Lockstep

    def step(self, steps: int) -> None:
        world = self._require_world()
        if not isinstance(steps, int) or steps < 1:
            raise ValueError("steps must be a positive integer")
        for _ in range(steps):
            world.step()
        self._log("fake.step", {"steps": steps}, world.digest())

    def read_state(self, actor: ActorRef) -> StateRow:
        body = self._body(actor)
        world = self._require_world()
        p, q = body.reference_pose()
        rot = _rotmat(body.q)
        v_ref = body.v + np.cross(body.w, rot @ (-body.com_offset))
        return StateRow(actor=actor.actor, t=world.t, p=tuple(float(x) for x in p), q=tuple(float(x) for x in q),
                        v=tuple(float(x) for x in v_ref), w=tuple(float(x) for x in body.w),
                        contacts=tuple(sorted(body.contacts)))

    def actuate(self, actor: ActorRef, cmd: Mapping[str, float]) -> None:
        body = self._body(actor)
        checked = {}
        for channel_id, value in cmd.items():
            channel = next((c for c in FAKE_CHANNELS if c.id == channel_id), None)
            if channel is None:
                raise RefusedCall(f"undeclared channel {channel_id!r}")
            checked[channel_id] = channel.check(value)
        veh = body.vehicle
        if veh is None and checked:
            raise RefusedCall(f"{actor.actor} has no actuators (it is a {body.shape})")
        self._log("fake.actuate", {"actor": actor.actor, "cmd": checked})
        for channel_id, value in checked.items():
            if channel_id == "drive.throttle":
                veh.throttle = value
            elif channel_id == "drive.brake":
                veh.brake = value
            elif channel_id == "drive.steer":
                veh.steer = value
            elif channel_id == "drive.parking_brake" and value == 1.0:
                veh.parking = not veh.parking

    def capture(self, camera: CameraSpec) -> Frame:
        raise BackendError("the fake connector does not record frames")

    # ------------------------------------------------------------------ replay

    def replay(self, rows: Sequence[Mapping[str, Any]]) -> list[str]:
        """Re-issue a fake ledger's calls on this connector; return problems (empty when every digest matches)."""
        problems: list[str] = []
        handles: dict[str, ActorRef] = {}
        for row in rows:
            method, args = row["method"], row.get("args") or {}
            if method == "fake.install":
                self.install([Package(args["name"], args["kind"], args["sha256"], contents=args["contents"])])
            elif method == "fake.open_scene":
                self.open_scene(LevelRef(package=args["package"], level_id=args.get("level_id")))
            elif method == "fake.spawn":
                obj = self._find_object(args["object"])
                ref, _ = self.spawn(obj, Pose.parse(args["pose"]), args["actor"])
                handles[args["actor"]] = ref
            elif method == "fake.actuate":
                self.actuate(handles[args["actor"]], args["cmd"])
            elif method == "fake.step":
                self.step(int(args["steps"]))
            else:
                continue
            if row.get("result_sha256") is not None and method in ("fake.spawn", "fake.step"):
                expected = row["result_sha256"]
                actual = sha256_json(self._require_world().digest())
                if actual != expected:
                    problems.append(f"call {row['seq']} ({method}): state digest differs on replay")
        return problems

    def _find_object(self, name: str) -> ObjectRef:
        for pkg in self._packages.values():
            raw = pkg.contents.get("objects", {}).get(name)
            if raw is not None:
                return _object_from_contents(name, pkg.name, raw)
        raise BackendError(f"object {name!r} is not installed")


def _canonical(value: Any) -> bytes:
    from openbeam_v2e.backends.base import canonical_json
    return canonical_json(value)


def _current_plan(run_dir: Path) -> Plan | None:
    path = Path(run_dir) / "plan.json"
    if not path.is_file():
        return None
    return Plan.from_json(json.loads(path.read_text(encoding="utf-8")))


def _t0(plan: Plan | None) -> float | None:
    if plan is None:
        return 0.0
    return None if any(m == "drive_in" for m in plan.modes.values()) else 0.0


def _content_origin(actor: str, obj: ObjectRef) -> ContentOrigin:
    params = {"dims_m": list(obj.dims_m or ()), "mass_kg": obj.mass_kg, "geometry": dict(obj.geometry)}
    return ContentOrigin(actor=actor, kind="generated", generator="openbeam_v2e.backends.fake",
                         generator_version=__version__, params_sha256=sha256_json(params), parameters=params)


def _statics_from(contents: Mapping[str, Any]) -> list[_Static]:
    statics = []
    ground = contents.get("ground", {"z": 0.0})
    if ground is not None:
        statics.append(_Static("ground", "plane", float(ground.get("mu", _GROUND_FRICTION)),
                               point=np.array([0.0, 0.0, float(ground.get("z", 0.0))]),
                               normal=np.array([0.0, 0.0, 1.0])))
    for raw in contents.get("statics", []):
        mu = float(raw.get("mu", _GROUND_FRICTION))
        if raw["kind"] == "plane":
            normal = np.array(raw["normal"], dtype=float)
            statics.append(_Static(raw["name"], "plane", mu, point=np.array(raw["point"], dtype=float),
                                   normal=normal / np.linalg.norm(normal)))
        elif raw["kind"] == "box":
            q = np.array(raw.get("quat_xyzw", (0.0, 0.0, 0.0, 1.0)), dtype=float)
            statics.append(_Static(raw["name"], "box", mu, center=np.array(raw["center"], dtype=float),
                                   rot=_rotmat(q / np.linalg.norm(q)), half=np.array(raw["half_extents"], dtype=float)))
        elif raw["kind"] == "convex":
            planes = np.array(raw["planes"], dtype=float).reshape(-1, 4)
            norms = np.linalg.norm(planes[:, :3], axis=1, keepdims=True)
            statics.append(_Static(raw["name"], "convex", mu, planes=planes / norms))
        else:
            raise BackendError(f"unknown static primitive kind {raw['kind']!r}")
    return statics


def _object_from_contents(name: str, package: str, raw: Mapping[str, Any]) -> ObjectRef:
    return ObjectRef(name=name, kind="vehicle" if raw["shape"] == "vehicle" else "prop", package=package,
                     engine_file=name, seatable=bool(raw.get("seatable", False)),
                     dims_m=tuple(float(v) for v in raw["dims_m"]), mass_kg=float(raw["mass_kg"]),
                     geometry={k: v for k, v in raw.items() if k not in ("dims_m", "mass_kg", "seatable")})


# --------------------------------------------------------------------------- scene -> compiled scene


def _value(raw: Any) -> float:
    """A measured value ``[value, sigma]`` or a plain number."""
    if isinstance(raw, (list, tuple)):
        return float(raw[0])
    if isinstance(raw, Mapping):
        return float(raw["value"])
    return float(raw)


def _shape_for(cls: str) -> str:
    if cls.startswith("vehicle."):
        return "vehicle"
    return {"prop.sphere": "sphere", "prop.cylinder": "cylinder"}.get(cls, "box")


def _mesh_box(world_dir: Path, mesh: Any, name: str) -> dict[str, Any]:
    candidates = []
    if isinstance(mesh, str):
        candidates.append(world_dir / mesh)
    else:
        candidates += [world_dir / "meshes" / f"{int(mesh)}.npz", world_dir / f"{int(mesh)}.npz"]
    for path in candidates:
        if path.is_file():
            with np.load(path, allow_pickle=False) as data:
                for key in ("vertices", "verts", "V", "points", "positions"):
                    if key in data.files:
                        verts = np.asarray(data[key], dtype=float).reshape(-1, 3)
                        break
                else:
                    arrays = [np.asarray(data[k]) for k in data.files]
                    verts = next((a.astype(float).reshape(-1, 3) for a in arrays
                                  if a.ndim == 2 and a.shape[1] == 3 and a.dtype.kind == "f"), None)
            if verts is None or not len(verts):
                break
            lo, hi = verts.min(axis=0), verts.max(axis=0)
            return {"name": name, "kind": "box", "center": ((lo + hi) / 2.0).tolist(),
                    "half_extents": np.maximum((hi - lo) / 2.0, 1e-3).tolist()}
    raise ValueError(f"static actor {name!r}: no readable mesh among {[str(c) for c in candidates]}")


def compile_scene(scene: Mapping[str, Any], base_dir: Path | None = None, *, ground_z: float | None = 0.0,
                  statics: Sequence[Mapping[str, Any]] = ()) -> CompiledScene:
    """Compile a ``v2e.scene/1`` mapping into a ``v2e.compiled/1`` scene for the fake connector.

    Dynamic actors need an explicit ``initial_state.pose`` (``at_rest``); scenes that reference
    world-package tracks or drive in need the general compiler. Static actors become boxes from
    their world-package mesh's bounding box unless ``statics`` supplies primitives by name.
    """
    base_dir = Path(base_dir or ".")
    world_dir = base_dir / str(scene.get("world", "world"))
    given = {s["name"]: dict(s) for s in statics}
    level_statics = []
    objects: dict[str, dict[str, Any]] = {}
    refs: list[ObjectRef] = []
    spawns: list[SpawnSpec] = []
    notes: list[str] = []
    for spec in scene["actors"]:
        ident = str(spec["id"])
        cls = str(spec.get("class", ""))
        if cls == "static":
            level_statics.append(given.pop(ident) if ident in given else _mesh_box(world_dir, spec.get("mesh"),
                                                                                           ident))
            continue
        shape = _shape_for(cls)
        dims_raw = spec.get("dims_m", {})
        default = _VEHICLE_DIMS if shape == "vehicle" else _PROP_DIMS[shape]
        if shape == "sphere" and "diameter" in dims_raw:
            d = _value(dims_raw["diameter"])
            dims = (d, d, d)
        else:
            dims = tuple(_value(dims_raw[k]) if k in dims_raw else default[i]
                         for i, k in enumerate(("length", "width", "height")))
        assumed = [k for k in ("length", "width", "height") if k not in dims_raw]
        if "mass_kg" in spec:
            mass = _value(spec["mass_kg"])
        elif shape == "vehicle":
            mass = _VEHICLE_MASS
            assumed.append("mass_kg")
        else:
            volume = (math.pi / 6.0 * dims[0] ** 3 if shape == "sphere"
                      else math.pi * (dims[1] / 2.0) ** 2 * dims[0] if shape == "cylinder"
                      else dims[0] * dims[1] * dims[2])
            mass = _PROP_DENSITY * volume
            assumed.append("mass_kg")
        if assumed:
            notes.append(f"{ident}: assumed {', '.join(assumed)} (test-double defaults)")
        state = spec.get("initial_state") or {}
        mode = state["mode"] if "mode" in state else "at_rest"
        if mode != "at_rest":
            raise ValueError(f"actor {ident!r}: the fake compiler spawns at_rest actors only "
                             f"(mode {mode!r} needs the general compiler)")
        pose_raw = state.get("pose")
        if not isinstance(pose_raw, Mapping):
            raise ValueError(f"actor {ident!r}: the fake compiler needs an explicit pose "
                             f"({{'position_m': [...], 'quat_xyzw': [...]}}), not {pose_raw!r}")
        name = f"{ident}_object"
        objects[name] = {"shape": shape, "dims_m": list(dims), "mass_kg": mass, "seatable": bool(spec.get("seatable"))}
        refs.append(_object_from_contents(name, "fake-objects", objects[name]))
        spawns.append(SpawnSpec(ident, name, Pose.parse(pose_raw)))
    if given:
        raise ValueError(f"static primitives given for unknown actors {sorted(given)}")
    level_contents = {"ground": None if ground_z is None else {"z": float(ground_z), "mu": _GROUND_FRICTION},
                      "statics": level_statics}
    object_contents = {"objects": objects}
    packages = (
        Package("fake-level", "fake.level", sha256_json(level_contents), contents=level_contents),
        Package("fake-objects", "fake.objects", sha256_json(object_contents), contents=object_contents),
    )
    content = tuple(_content_origin(s.actor, next(r for r in refs if r.name == s.object)) for s in spawns)
    return CompiledScene(connector="fake", dialect=FAKE_DIALECT, level=LevelRef(package="fake-level",
                                                                                 level_id="fake-level"),
                         packages=packages, objects=tuple(refs), spawns=tuple(spawns), content=content,
                         notes=tuple(notes))


# --------------------------------------------------------------------------- v2e.package/1 -> compiled scene


def _read_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    verts: list[list[float]] = []
    faces: list[list[int]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "v" and len(parts) >= 4:
            verts.append([float(v) for v in parts[1:4]])
        elif parts[0] == "f" and len(parts) >= 4:
            idx = [int(token.split("/")[0]) for token in parts[1:]]
            idx = [i - 1 if i > 0 else len(verts) + i for i in idx]
            faces += [[idx[0], idx[k], idx[k + 1]] for k in range(1, len(idx) - 1)]
    return np.array(verts, dtype=float).reshape(-1, 3), np.array(faces, dtype=np.int64).reshape(-1, 3)


def _hull_planes(points: np.ndarray) -> np.ndarray:
    from scipy.spatial import ConvexHull

    hull = ConvexHull(points)
    planes = np.concatenate([hull.equations[:, :3], -hull.equations[:, 3:4]], axis=1)
    return np.unique(np.round(planes, 9), axis=0)


def _slab(tri: np.ndarray, normal: np.ndarray, thickness: float) -> np.ndarray:
    """Half-spaces of a triangle extruded by ``thickness`` against its normal."""
    planes = [np.concatenate([normal, [normal @ tri[0]]]), np.concatenate([-normal, [-(normal @ tri[0]) + thickness]])]
    centroid = tri.mean(axis=0)
    for a, b in ((0, 1), (1, 2), (2, 0)):
        side = np.cross(tri[b] - tri[a], normal)
        side /= max(np.linalg.norm(side), 1e-12)
        if side @ (centroid - tri[a]) > 0:
            side = -side
        planes.append(np.concatenate([side, [side @ tri[a]]]))
    return np.array(planes)


def _convex_parts(verts: np.ndarray, faces: np.ndarray, thickness: float = 0.5) -> list[np.ndarray]:
    """Convex parts for one collision mesh.

    A convex mesh is one part (its hull). A flat mesh (ground, a slab top) becomes a slab extruded
    downwards. Any other mesh becomes one slab per triangle, extruded against the triangle's
    normal, which follows open surfaces such as terrain or a curved ramp.
    """
    if len(verts) < 3:
        return []
    centred = verts - verts.mean(axis=0)
    _, singular, vt = np.linalg.svd(centred, full_matrices=False)
    if singular[-1] <= 1e-6 * max(singular[0], 1e-12):            # flat
        normal = vt[-1] if vt[-1][2] >= 0 else -vt[-1]
        return [_hull_planes(np.concatenate([verts, verts - normal * 1.0]))]
    planes = _hull_planes(verts)
    inside = verts @ planes[:, :3].T - planes[:, 3]
    on_hull = np.all(np.any(np.abs(inside) <= 1e-6, axis=1))
    if on_hull and len(faces):
        tri = verts[faces]
        normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        lens = np.linalg.norm(normals, axis=1)
        keep = lens > 1e-12
        normals = normals[keep] / lens[keep, None]
        offsets = np.einsum("ij,ij->i", normals, tri[keep, 0])
        signed = verts @ normals.T - offsets
        if np.all(signed <= 1e-6) or np.all(signed >= -1e-6):    # every face plane supports the mesh
            return [planes]
    parts = []
    for tri in verts[faces]:
        normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        length = np.linalg.norm(normal)
        if length > 1e-12:
            parts.append(_slab(tri, normal / length, thickness))
    return parts


def _param(params: Mapping[str, Any], key: str) -> float | None:
    raw = params.get(key)
    if raw is None:
        return None
    value = raw.get("value") if isinstance(raw, Mapping) else raw
    return float(value) if isinstance(value, (int, float)) else None


def compile_package(request: Path | Mapping[str, Any], base_dir: Path | None = None,
                    scene: Mapping[str, Any] | None = None, world: Any = None) -> CompiledScene:
    """Build the fake connector's content from a ``v2e.package/1`` request (the compiler's output).

    Level collision OBJ files become convex parts (see :func:`_convex_parts`); each actor's content
    record gives its shape, measured or assumed dimensions and mass; spawns keep the compiler's
    at-rest poses; the timeline's ``t0_s`` is kept. With the world package (``world``, a
    ``openbeam_v2e.world`` model or directory), each object's reference point is placed where the
    world's hull origin is, so rows line up with the reconstruction.
    """
    if not isinstance(request, Mapping):
        base_dir = Path(base_dir or Path(request).parent)
        request = json.loads(Path(request).read_text(encoding="utf-8"))
    base_dir = Path(base_dir or ".")
    if request.get("schema") != "v2e.package/1":
        raise ValueError("not a v2e.package/1 request")
    world_model = None
    if world is not None:
        from openbeam_v2e.world.io import read_world

        world_model = world if hasattr(world, "objects") else read_world(world)
    actor_class = {str(a["id"]): str(a.get("class", "")) for a in (scene or {}).get("actors", [])}
    actor_object = {str(a["id"]): str(a.get("object") or a["id"]) for a in (scene or {}).get("actors", [])}
    statics = []
    for entry in request["level"]["collision"]:
        if entry.get("role") == "visual":
            continue
        verts, faces = _read_obj(base_dir / entry["file"])
        for index, planes in enumerate(_convex_parts(verts, faces)):
            statics.append({"name": entry["id"] if index == 0 else f"{entry['id']}#{index}", "kind": "convex",
                            "planes": planes.round(9).tolist()})
    notes = [f"timeline t0_s={request['timeline']['t0_s']}"]
    ground = None
    if not statics:
        ground = {"z": 0.0, "mu": _GROUND_FRICTION}
        notes.append("the package has no collision geometry; the fake adds a ground plane at z=0")
    objects: dict[str, dict[str, Any]] = {}
    refs: list[ObjectRef] = []
    origins: list[ContentOrigin] = []
    for record in request["content"]:
        actor = str(record["actor"])
        cls = str(record.get("class") or actor_class.get(actor, ""))
        shape = _shape_for(cls)
        params = record.get("parameters", {})
        default = _VEHICLE_DIMS if shape == "vehicle" else _PROP_DIMS[shape]
        radius = _param(params, "radius_m") or ((_param(params, "diameter_m") or 0.0) / 2.0)
        if shape == "sphere" and radius:
            dims = (2 * radius,) * 3
        else:
            dims = tuple(_param(params, k) or default[i] for i, k in enumerate(("length_m", "width_m", "height_m")))
        mass = _param(params, "mass_kg") or (_VEHICLE_MASS if shape == "vehicle" else
                                             _PROP_DENSITY * dims[0] * dims[1] * dims[2])
        raw: dict[str, Any] = {"shape": shape, "dims_m": list(dims), "mass_kg": mass,
                               "seatable": bool(record.get("seatable", shape == "vehicle"))}
        if shape == "vehicle":
            raw["vehicle"] = {k: v for k, v in ((key, _param(params, key)) for key in (
                "wheelbase_m", "track_m", "ride_height_m", "wheel_radius_m")) if v is not None}
        if world_model is not None and world_model.has_object(actor_object.get(actor, actor)):
            hull = world_model.object(actor_object.get(actor, actor)).hull_vertices
            raw["ref_offset_m"] = ((hull.min(axis=0) + hull.max(axis=0)) / 2.0).round(9).tolist()
        name = f"{actor}_object"
        objects[name] = raw
        refs.append(_object_from_contents(name, "fake-objects", raw))
        kind = record.get("kind", "generated")
        origins.append(ContentOrigin(actor=actor, kind=kind if kind in ("stock", "generated") else "generated",
                                     generator=str(record.get("generator", "")),
                                     generator_version=str(record.get("generator_version", "")),
                                     params_sha256=str(record.get("params_sha256", "")),
                                     parameters=dict(params), content_id=str(record.get("content_id", "")),
                                     content_version=str(record.get("content_version", ""))))
    spawns = tuple(SpawnSpec(str(s["actor"]), f"{s['actor']}_object", Pose.parse(s["pose"]))
                   for s in request["spawns"])
    level_contents = {"ground": ground, "statics": statics}
    object_contents = {"objects": objects}
    packages = (
        Package("fake-level", "fake.level", sha256_json(level_contents), contents=level_contents),
        Package("fake-objects", "fake.objects", sha256_json(object_contents), contents=object_contents),
    )
    return CompiledScene(connector="fake", dialect=FAKE_DIALECT, level=LevelRef(package="fake-level",
                                                                                 level_id="fake-level"),
                         packages=packages, objects=tuple(refs), spawns=spawns, content=tuple(origins),
                         t0_s=float(request["timeline"]["t0_s"]), notes=tuple(notes + list(request.get("notes", []))))
