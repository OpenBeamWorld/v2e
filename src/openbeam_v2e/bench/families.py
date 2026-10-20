# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""Scene families of the synthetic benchmark ``v2e-synth-1``.

Four families, each a parameter grid over one kind of physical event built from generic
content only (a generated generic vehicle, generated props and simple level geometry):

``barrier_crash``
    A generic vehicle drives into a rigid barrier, head-on or with a partial overlap.
``ramp_jump``
    A generic vehicle drives up a plain wedge ramp, leaves its top edge and lands on flat
    ground.
``rollover``
    A generic vehicle drives straight with its left wheels over a one-sided trip ramp.
``drop_and_bounce``
    Props (box, ball, cylinder) dropped onto a slab, or a box sliding down an incline.
    Passive objects are score-only in v0.

Every number below is a round design value chosen for this benchmark from first principles
(typical passenger-car proportions, everyday speeds, standard gravity); none is measured from
any video. :data:`PROVENANCE` records where each kind of value comes from, and every target a
case declares names its source.

Conventions: level frame Z-up, right-handed, SI units. Vehicles travel along +X; +Y is to the
driver's left. A pose is ``(position_m, yaw_deg)`` for the content's reference point (the
bounding-box centre for vehicles and props).

A :class:`Case` is one grid point with one camera preset. It can describe itself as the
documents the rest of the pipeline reads: a ``v2e.scene/1`` scene, a ``v2e.templates/1``
mapping and a ``targets.json`` list with sources. Target metric names follow the package
scorer's (``speed_at_event``, ``time_of_event``, ``yaw_change_after_event``, ...) wherever it
measures the same thing; the benchmark adds ``apex_rise``, ``airborne_time``,
``landing_distance`` and ``roll_onset_allowed``, which only its own runner evaluates.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Mapping

__all__ = [
    "BENCHMARK_ID",
    "BENCHMARK_VERSION",
    "GRAVITY_MPS2",
    "GENERIC_VEHICLE",
    "PROVENANCE",
    "CameraPreset",
    "Static",
    "Actor",
    "Target",
    "Case",
    "Family",
    "FAMILIES",
    "family_names",
    "cases",
    "get_case",
    "kmh",
]

BENCHMARK_ID = "v2e-synth-1"
BENCHMARK_VERSION = "0.1.0"

GRAVITY_MPS2 = 9.81
"""Standard gravity rounded to three significant figures, used for every first-principles target."""

# Generic vehicle design point: round values for a mid-size passenger car, chosen for this
# benchmark without reference to any measured vehicle.
GENERIC_VEHICLE: Mapping[str, float] = {
    "length_m": 4.75,
    "width_m": 1.80,
    "height_m": 1.45,
    "wheelbase_m": 2.75,
    "track_m": 1.60,
    "ride_height_m": 0.15,
    "wheel_radius_m": 0.32,
    "mass_kg": 1500.0,
    # Assumed centre-of-mass height above the ground, used only for the rollover energy bound.
    "com_height_m": 0.55,
}

PROVENANCE: Mapping[str, str] = {
    "design_value": "round value chosen for v2e-synth-1 from first principles; not measured from any video",
    "first_principles": "computed from the design values with point-mass or rigid-body mechanics, symmetry "
                        "arguments and g = 9.81 m/s^2",
    "reported": "measured and reported only; no pass/fail bar in v0",
}

LABEL = "synthetic self-consistency"
"""Every figure computed from this benchmark carries this label: truth comes from a simulated run."""


def kmh(speed_kmh: float) -> float:
    """Convert km/h to m/s."""
    return speed_kmh / 3.6


# --------------------------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Static:
    """A piece of static level geometry described by a level-generator primitive.

    ``kind`` names a level primitive (``ground``, ``slab``, ``barrier``, ``wedge_ramp``,
    ``trip_ramp``, ``incline``, ``post``; see :mod:`openbeam_v2e.bench.geometry`); ``params`` are its dimensions in
    metres or degrees; ``position_m`` and ``yaw_deg`` place the primitive's reference point (the
    centre of its footprint at ground level unless the primitive says otherwise).
    """

    id: str
    kind: str
    params: Mapping[str, float]
    position_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    yaw_deg: float = 0.0
    observed: bool = True  # False for geometry the camera never shows (e.g. synthesized run-up)

    def document(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "params": dict(self.params),
            "position_m": list(self.position_m),
            "yaw_deg": self.yaw_deg,
            "observed": self.observed,
        }


@dataclass(frozen=True)
class Actor:
    """A dynamic object of a case.

    ``mode`` is ``drive_in`` (spawned at rest on a run-up, driven to ``frame0`` at ``speed_mps``)
    or ``at_rest`` (spawned at rest at ``frame0``; for a dropped prop that pose is in the air).
    ``drive`` holds the driver's stages for the benchmark controller (see
    :mod:`openbeam_v2e.bench.drive`): cruise at ``speed_mps`` on the line through ``frame0`` in
    direction ``frame0_yaw_deg``, then whatever the case does after its first event.
    """

    id: str
    cls: str  # "vehicle.car" | "prop.box" | "prop.sphere" | "prop.cylinder"
    template: str  # "vehicle.generic" | "prop.box" | "prop.sphere" | "prop.cylinder"
    dims_m: Mapping[str, float]
    mass_kg: float
    frame0_position_m: tuple[float, float, float]
    frame0_yaw_deg: float = 0.0
    frame0_pitch_deg: float = 0.0
    frame0_roll_deg: float = 0.0
    mode: str = "at_rest"
    speed_mps: float = 0.0
    control: str = "passive"  # "driver" | "passive"
    drive: Mapping[str, Any] = field(default_factory=dict)
    role: str = "subject"  # "subject" (scored) | "anchor" (off-camera, never scored)

    @property
    def seatable(self) -> bool:
        return self.cls.startswith("vehicle.")

    def scene_document(self, dims_sigma_m: float = 0.0) -> dict[str, Any]:
        dims = {name: [value, dims_sigma_m] for name, value in self.dims_m.items()}
        initial: dict[str, Any] = {"mode": self.mode, "frame0_pose": f"track:{self.id}@0"}
        if self.mode == "drive_in":
            initial["speed_mps"] = [round(self.speed_mps, 6), 0.0]
        doc: dict[str, Any] = {
            "id": self.id,
            "class": self.cls,
            "dims_m": dims,
            "seatable": self.seatable,
            "control": self.control,
            "initial_state": initial,
        }
        if self.role != "subject":
            doc["role"] = self.role
        return doc


@dataclass(frozen=True)
class CameraPreset:
    """A camera at a fixed position.

    ``tripod`` keeps its orientation and field of view. ``handheld_ptz`` stays at the same
    position but pans and tilts to follow the subject with a first-order lag of ``lag_s``,
    zooms in linearly to ``zoom_end`` times the starting focal length over the clip, and has
    radial distortion ``k1``.
    """

    name: str  # "tripod" | "handheld_ptz"
    position_m: tuple[float, float, float]
    look_at_m: tuple[float, float, float]
    hfov_deg: float
    k1: float = 0.0
    lag_s: float = 0.0
    zoom_end: float = 1.0

    def document(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "position_m": list(self.position_m),
            "look_at_m": list(self.look_at_m),
            "hfov_deg": self.hfov_deg,
            "k1": self.k1,
            "lag_s": self.lag_s,
            "zoom_end": self.zoom_end,
        }

    def handheld(self, k1: float = -0.05, lag_s: float = 0.3, zoom_end: float = 1.25) -> "CameraPreset":
        return CameraPreset("handheld_ptz", self.position_m, self.look_at_m, self.hfov_deg, k1, lag_s, zoom_end)


@dataclass(frozen=True)
class Target:
    """A target value, the half-width of its pass band and the source it comes from.

    In a ``v2e.scene/1`` document the band is written as the value's sigma with
    ``tolerance_sigma`` 1, so a run meets the target when it lands within the band.
    """

    metric: str
    actor: str
    value: float | None
    tolerance: float | None
    unit: str
    source: str  # a key of PROVENANCE
    event: str | None = None
    phase: str | None = None
    note: str = ""

    def document(self) -> dict[str, Any]:
        doc: dict[str, Any] = {"metric": self.metric, "actor": self.actor}
        if self.event is not None:
            doc["event"] = self.event
        if self.phase is not None:
            doc["phase"] = self.phase
        doc["value"] = None if self.value is None else [round(self.value, 6), self.tolerance]
        doc["unit"] = self.unit
        doc["source"] = self.source
        if self.note:
            doc["note"] = self.note
        return doc

    def scene_document(self) -> dict[str, Any]:
        """The target as a ``v2e.scene/1`` target (only for targets with a value)."""
        doc = self.document()
        doc.pop("note", None)
        if self.note:
            doc["citation"] = self.note
        doc["tolerance_sigma"] = 1.0
        return doc


@dataclass(frozen=True)
class Case:
    """One benchmark case: a grid point of a family seen by one camera preset."""

    family: str
    case_id: str
    params: Mapping[str, Any]
    statics: tuple[Static, ...]
    actors: tuple[Actor, ...]
    camera: CameraPreset
    clip_s: float
    fps: float
    events: tuple[str, ...]
    phases: tuple[tuple[str, str, str], ...]  # (name, from event, to event)
    targets: tuple[Target, ...]
    requires: Mapping[str, Any]
    score_only: bool = False
    expected_claim: str = "test_double"  # on the fake connector; "native" on an engine connector
    description: str = ""

    @property
    def frames(self) -> int:
        return int(round(self.clip_s * self.fps))

    @property
    def subjects(self) -> tuple[Actor, ...]:
        return tuple(a for a in self.actors if a.role == "subject")

    def actor(self, name: str) -> Actor:
        for candidate in self.actors:
            if candidate.id == name:
                return candidate
        raise KeyError(name)

    # ---- documents -----------------------------------------------------------------------

    def scene_document(self) -> dict[str, Any]:
        """The case as a ``v2e.scene/1`` document (truth values, sigma 0)."""
        actors = [a.scene_document() for a in self.actors]
        for k, static in enumerate(self.statics, start=1):
            # Mesh k of the case's world package (the generator writes statics in this order).
            actors.append({"id": static.id, "class": "static", "mesh": k, "role": "collision"})
        return {
            "schema": "v2e.scene/1",
            "world": "world/",
            "frame": "Z-up level frame, SI units",
            "actors": actors,
            "events": [{"name": name, "evidence": "ground truth"} for name in self.events],
            "phases": [{"name": n, "from": a, "to": b} for n, a, b in self.phases],
            "targets": [t.scene_document() for t in self.targets if t.value is not None],
            "requires": dict(self.requires),
        }

    def templates_document(self) -> dict[str, Any]:
        """The case's ``v2e.templates/1`` mapping: generic content for every object."""
        objects: dict[str, Any] = {}
        for obj in self.actors:
            entry: dict[str, Any] = {"template": obj.template}
            overrides = {"mass_kg": {"value": obj.mass_kg, "status": "assumed", "reason": "benchmark design value"}}
            entry["overrides"] = overrides
            objects[obj.id] = entry
        for static in self.statics:
            objects[static.id] = {"template": "static.mesh"}
        return {"schema": "v2e.templates/1", "objects": objects}

    def targets_document(self) -> dict[str, Any]:
        """``targets.json``: every target, including reported-only metrics, with its source."""
        return {
            "schema": "v2e.targets/1",
            "case": self.case_id,
            "label": LABEL,
            "sources": dict(PROVENANCE),
            "targets": [t.document() for t in self.targets],
        }

    def world_document(self) -> dict[str, Any]:
        """Static geometry and actor content as generator calls (input to the case compiler)."""
        return {
            "statics": [s.document() for s in self.statics],
            "actors": [
                {
                    "id": a.id,
                    "class": a.cls,
                    "template": a.template,
                    "dims_m": dict(a.dims_m),
                    "mass_kg": a.mass_kg,
                    "frame0": {
                        "position_m": list(a.frame0_position_m),
                        "yaw_deg": a.frame0_yaw_deg,
                        "pitch_deg": a.frame0_pitch_deg,
                        "roll_deg": a.frame0_roll_deg,
                    },
                    "mode": a.mode,
                    "speed_mps": a.speed_mps,
                    "control": a.control,
                    "drive": dict(a.drive),
                    "role": a.role,
                }
                for a in self.actors
            ],
        }

    def document(self) -> dict[str, Any]:
        """Everything about the case in one JSON-ready document (used by manifests and tests)."""
        return {
            "benchmark": BENCHMARK_ID,
            "version": BENCHMARK_VERSION,
            "family": self.family,
            "case": self.case_id,
            "description": self.description,
            "params": dict(self.params),
            "camera": self.camera.document(),
            "clip_s": self.clip_s,
            "fps": self.fps,
            "frames": self.frames,
            "score_only": self.score_only,
            "expected_claim": self.expected_claim,
            "world": self.world_document(),
            "scene": self.scene_document(),
            "templates": self.templates_document(),
            "targets": self.targets_document(),
        }


@dataclass(frozen=True)
class Family:
    """A family: a grid of parameters and a builder that turns one grid point into a case."""

    name: str
    summary: str
    grid: Mapping[str, tuple[Any, ...]]
    default: Mapping[str, Any]
    build: Any  # Callable[[Mapping[str, Any], str], Case]

    def points(self) -> Iterator[dict[str, Any]]:
        keys = list(self.grid)
        for values in itertools.product(*(self.grid[k] for k in keys)):
            yield dict(zip(keys, values))

    def cases(self, cameras: Iterable[str] = ("tripod", "handheld_ptz")) -> Iterator[Case]:
        for point in self.points():
            for camera in cameras:
                yield self.build(point, camera)

    def default_case(self, camera: str = "tripod") -> Case:
        return self.build(dict(self.default), camera)


def _vehicle(name: str, x: float, y: float, speed_mps: float, drive: Mapping[str, Any]) -> Actor:
    gv = GENERIC_VEHICLE
    dims = {
        "length": gv["length_m"],
        "width": gv["width_m"],
        "height": gv["height_m"],
        "wheelbase": gv["wheelbase_m"],
        "track": gv["track_m"],
    }
    return Actor(
        id=name,
        cls="vehicle.car",
        template="vehicle.generic",
        dims_m=dims,
        mass_kg=gv["mass_kg"],
        frame0_position_m=(x, y, gv["height_m"] / 2.0),
        frame0_yaw_deg=0.0,
        mode="drive_in",
        speed_mps=speed_mps,
        control="driver",
        drive=drive,
    )


def _camera(name: str, preset: CameraPreset) -> CameraPreset:
    if name == "tripod":
        return preset
    if name == "handheld_ptz":
        return preset.handheld()
    raise ValueError(f"unknown camera preset {name!r} (expected 'tripod' or 'handheld_ptz')")


def _fmt(value: float) -> str:
    return f"{value:g}".replace(".", "p")


POST = {"size_m": 0.2, "height_m": 1.5}


def _survey_posts(xs: Iterable[float], y: float) -> tuple[Static, ...]:
    """Thin upright posts at surveyed positions behind the action.

    Real recreations survey a few fixed points in view; the posts play that role here. Their
    corners are known world points for camera calibration, and they stand well clear of every
    object's path.
    """
    return tuple(Static(f"post_{i + 1}", "post", dict(POST), position_m=(float(x), float(y), 0.0))
                 for i, x in enumerate(xs))


_DRIVER_REQUIRES = {
    "effects": ["drive", "brake", "steer", "contact"],
    "observables": ["position", "attitude"],
    "controlled_actors": 1,
    "seatable_vehicles": 1,
}


# --------------------------------------------------------------------------------------------
# S1 barrier crash
# --------------------------------------------------------------------------------------------

BARRIER = {"depth_m": 0.8, "width_m": 4.0, "height_m": 1.0}


def _barrier_crash(p: Mapping[str, Any], camera: str) -> Case:
    v = kmh(p["speed_kmh"])
    overlap = p["overlap"]
    gv = GENERIC_VEHICLE
    length, width = gv["length_m"], gv["width_m"]
    lead_s = 1.0  # frame 0 is one second before first contact
    # The barrier's front face is the plane x = 0; it spans y in [-2, 2]. With a partial overlap
    # the vehicle's right side overlaps the barrier's left edge by `overlap` of its width.
    y = 0.0 if overlap >= 1.0 else BARRIER["width_m"] / 2.0 + width * (0.5 - overlap)
    x0 = -length / 2.0 - v * lead_s
    car = _vehicle("car", x0, y, v, drive={"stages": [
        {"mode": "cruise", "until": {"contact": "barrier"}},
        {"mode": "fixed", "cmd": {"drive.throttle": 0.0, "drive.brake": 0.8}}]})
    statics = (
        Static("ground", "ground", {"size_m": 200.0}),
        Static("barrier", "barrier", dict(BARRIER), position_m=(BARRIER["depth_m"] / 2.0, 0.0, 0.0)),
    ) + _survey_posts((-12.0, -6.0, 2.0, 7.0), 8.0)
    preset = CameraPreset("tripod", position_m=(-3.0, -16.0, 1.6), look_at_m=(-1.5, y, 0.7), hfov_deg=50.0)
    targets = [
        Target("speed_at_event", "car", v, 0.3, "m/s", "design_value", event="impact",
               note="the driver holds the design speed until first contact"),
        Target("time_of_event", "car", lead_s, 1.0 / 30.0, "s", "design_value", event="impact",
               note="time from frame 0 to first contact"),
    ]
    if overlap >= 1.0:
        targets.append(Target("yaw_change_after_event", "car", 0.0, 5.0, "deg", "first_principles", event="impact",
                              note="mirror symmetry: a head-on full-overlap impact has no preferred turning direction"))
    else:
        targets.append(Target("yaw_change_after_event", "car", None, None, "deg", "reported", event="impact"))
    targets += [
        Target("rebound_speed", "car", None, None, "m/s", "reported", event="impact"),
        Target("rest_position", "car", None, None, "m", "reported", event="rest"),
    ]
    return Case(
        family="barrier_crash",
        case_id=f"barrier_crash-v{_fmt(p['speed_kmh'])}-o{int(round(overlap * 100))}-{camera}",
        params=dict(p),
        statics=statics,
        actors=(car,),
        camera=_camera(camera, preset),
        clip_s=3.0,
        fps=30.0,
        events=("impact", "rest"),
        phases=(("approach", "start", "impact"), ("post_impact", "impact", "end")),
        targets=tuple(targets),
        requires=dict(_DRIVER_REQUIRES),
        description=f"generic vehicle into a rigid barrier at {p['speed_kmh']:g} km/h, {overlap:.0%} overlap",
    )


# --------------------------------------------------------------------------------------------
# S2 ramp jump
# --------------------------------------------------------------------------------------------

RAMP = {"slope_length_m": 4.0, "width_m": 3.0}
"""Total slope length and width of the jump ramp. Its slope steepens in three straight segments
(1 m at a third of the take-off angle, 1 m at two thirds, 2 m at the full angle), the polygonal
version of a launch ramp's curved transition, so ordinary ground clearance rolls onto it."""


def profile_of(segments: Iterable[tuple[float, float]]) -> tuple[float, float]:
    """Horizontal run and height of a ramp made of ``(slope_length_m, angle_deg)`` segments."""
    run = rise = 0.0
    for length, angle in segments:
        run += length * math.cos(math.radians(angle))
        rise += length * math.sin(math.radians(angle))
    return run, rise


def ramp_segments(angle_deg: float) -> tuple[tuple[float, float], ...]:
    total = RAMP["slope_length_m"]
    return ((0.25 * total, angle_deg / 3.0), (0.25 * total, 2.0 * angle_deg / 3.0), (0.5 * total, angle_deg))


def ramp_jump_prediction(speed_mps: float, angle_deg: float, edge_height_m: float) -> dict[str, float]:
    """Point-mass prediction for a take-off at ``speed_mps`` and ``angle_deg`` from height ``edge_height_m``.

    Returns the apex height above the take-off edge, the time to apex, the time from take-off to
    ground contact and the horizontal distance covered by then.
    """
    g = GRAVITY_MPS2
    vz = speed_mps * math.sin(math.radians(angle_deg))
    vx = speed_mps * math.cos(math.radians(angle_deg))
    t_apex = vz / g
    apex = vz * vz / (2.0 * g)
    t_land = (vz + math.sqrt(vz * vz + 2.0 * g * edge_height_m)) / g
    return {"apex_above_edge_m": apex, "time_to_apex_s": t_apex, "airborne_s": t_land, "distance_m": vx * t_land}


def _ramp_jump(p: Mapping[str, Any], camera: str) -> Case:
    v = kmh(p["speed_kmh"])
    angle = float(p["angle_deg"])
    gv = GENERIC_VEHICLE
    segments = ramp_segments(angle)
    run, rise = profile_of(segments)
    lead_s = 1.0  # frame 0 is one second before the front wheels reach the ramp foot
    foot_x = -run  # the ramp's top edge is the plane x = 0
    x0 = foot_x - v * lead_s - gv["wheelbase_m"] / 2.0  # front axle one second short of the foot
    car = _vehicle("car", x0, 0.0, v, drive={"stages": [
        {"mode": "cruise", "until": {"airborne_s": 0.1}},
        {"mode": "fixed", "cmd": {"drive.throttle": 0.0}, "until": {"contact": "ground"}},
        {"mode": "fixed", "cmd": {"drive.throttle": 0.0, "drive.brake": 0.3}}]})
    pred = ramp_jump_prediction(v, angle, rise)
    statics = (
        Static("ground", "ground", {"size_m": 200.0}),
        Static("ramp", "wedge_ramp", {"segments": [list(sg) for sg in segments], "width_m": RAMP["width_m"],
                                       "angle_deg": angle}, position_m=(-run / 2.0, 0.0, 0.0)),
    ) + _survey_posts((-8.0, 0.0, 8.0, 16.0), 9.0)
    preset = CameraPreset("tripod", position_m=(4.0, -22.0, 2.0), look_at_m=(5.0, 0.0, 1.2), hfov_deg=60.0)
    targets = (
        Target("speed_at_event", "car", v, 0.3, "m/s", "design_value", event="takeoff",
               note="the driver holds the design speed up the ramp"),
        Target("apex_rise", "car", pred["apex_above_edge_m"], max(0.05, 0.15 * pred["apex_above_edge_m"]),
               "m", "first_principles", phase="airborne",
               note="rise of the centre from take-off to apex: (v sin a)^2 / 2g; tolerance 15 % (at least 5 cm)"),
        Target("airborne_time", "car", pred["airborne_s"], 0.15 * pred["airborne_s"], "s", "first_principles",
               phase="airborne", note="point-mass time from take-off to ground contact; tolerance 15 %"),
        Target("landing_distance", "car", pred["distance_m"], 0.15 * pred["distance_m"], "m", "first_principles",
               phase="airborne", note="point-mass horizontal distance from the edge; tolerance 15 %"),
    )
    return Case(
        family="ramp_jump",
        case_id=f"ramp_jump-a{_fmt(angle)}-v{_fmt(p['speed_kmh'])}-{camera}",
        params=dict(p),
        statics=statics,
        actors=(car,),
        camera=_camera(camera, preset),
        clip_s=3.5,
        fps=30.0,
        events=("ramp_contact", "takeoff", "apex", "landing"),
        phases=(("approach", "start", "ramp_contact"), ("ramp", "ramp_contact", "takeoff"),
                ("airborne", "takeoff", "landing"), ("landing", "landing", "end")),
        targets=targets,
        requires=dict(_DRIVER_REQUIRES),
        description=f"generic vehicle off a {angle:g} deg wedge ramp at {p['speed_kmh']:g} km/h",
    )


# --------------------------------------------------------------------------------------------
# S3 rollover
# --------------------------------------------------------------------------------------------

TRIP_RAMP = {"segments": ((0.6, 8.0), (1.6, 22.0)), "width_m": 1.0}
"""A short one-sided ramp in two straight segments (gentle toe, steeper main face), wide enough
for one wheel track; its top edge drops straight back to the ground."""


def rollover_energy_bound(track_m: float, com_height_m: float, ramp_height_m: float, ramp_length_m: float,
                          width_m: float, height_m: float) -> dict[str, float]:
    """Lowest approach speed at which a rigid vehicle can roll past its tipping point.

    Model: at the end of the trip ramp the left wheels are ``ramp_height_m`` up and still rising
    at ``v * ramp_height_m / ramp_length_m``; the body pivots about the right wheels' contact line.
    It tips over when its roll kinetic energy about that line covers the rise of the centre of
    mass to the point vertically above the line. The body's own roll inertia is that of a
    uniform box ``width_m`` by ``height_m``. This is a bound for an idealised rigid body, not a
    prediction of what a suspended vehicle does.
    """
    g = GRAVITY_MPS2
    half = track_m / 2.0
    r = math.hypot(half, com_height_m)
    beta = math.atan2(com_height_m, half)
    phi0 = math.asin(min(1.0, ramp_height_m / track_m))
    rise = r - r * math.sin(beta + phi0)
    inertia_per_kg = (width_m ** 2 + height_m ** 2) / 12.0 + r * r
    omega_min = math.sqrt(2.0 * g * rise / inertia_per_kg)
    vz_min = omega_min * track_m * math.cos(phi0)
    v_min = vz_min * ramp_length_m / ramp_height_m
    return {"tip_rise_m": rise, "roll_rate_min_rad_s": omega_min, "speed_min_mps": v_min,
            "roll_at_ramp_end_deg": math.degrees(phi0)}


def _rollover(p: Mapping[str, Any], camera: str) -> Case:
    v = kmh(p["speed_kmh"])
    gv = GENERIC_VEHICLE
    lead_s = 1.0  # frame 0 is one second before the front-left wheel reaches the trip ramp
    ramp_start = 0.0
    front_axle_from_centre = gv["wheelbase_m"] / 2.0
    x0 = ramp_start - front_axle_from_centre - v * lead_s
    # The ramp sits under the left wheels: its centreline is the left wheel track.
    ramp_y = gv["track_m"] / 2.0
    car = _vehicle("car", x0, 0.0, v, drive={"stages": [
        {"mode": "cruise", "until": {"contact": "trip_ramp"}},
        {"mode": "fixed", "cmd": {"drive.throttle": 0.0, "drive.brake": 0.0, "drive.steer": 0.0}}]})
    trip_run, trip_rise = profile_of(TRIP_RAMP["segments"])
    last_len, last_angle = TRIP_RAMP["segments"][-1]
    bound = rollover_energy_bound(gv["track_m"], gv["com_height_m"], trip_rise,
                                  trip_rise / math.tan(math.radians(last_angle)), gv["width_m"], gv["height_m"])
    statics = (
        Static("ground", "ground", {"size_m": 200.0}),
        Static("trip_ramp", "trip_ramp", {"segments": [list(sg) for sg in TRIP_RAMP["segments"]],
                                          "width_m": TRIP_RAMP["width_m"]},
               position_m=(ramp_start + trip_run / 2.0, ramp_y, 0.0)),
    ) + _survey_posts((-6.0, 2.0, 10.0, 18.0), 9.0)
    preset = CameraPreset("tripod", position_m=(10.0, -18.0, 2.0), look_at_m=(6.0, 0.0, 0.8), hfov_deg=60.0)
    can_roll = v >= bound["speed_min_mps"]
    targets = (
        Target("speed_at_event", "car", v, 0.3, "m/s", "design_value", event="trip_contact"),
        Target("roll_onset_allowed", "car", 1.0 if can_roll else 0.0, 0.0, "bool", "first_principles",
               note=f"rigid-body energy bound: tipping needs at least {bound['speed_min_mps']:.1f} m/s; a run "
                    f"may roll over only where this is 1"),
        Target("cumulative_roll_deg", "car", None, None, "deg", "reported", phase="after_trip"),
        Target("quarter_turns", "car", None, None, "count", "reported", event="rest"),
        Target("rest_attitude", "car", None, None, "class", "reported", event="rest"),
    )
    return Case(
        family="rollover",
        case_id=f"rollover-v{_fmt(p['speed_kmh'])}-{camera}",
        params=dict(p),
        statics=statics,
        actors=(car,),
        camera=_camera(camera, preset),
        clip_s=4.0,
        fps=30.0,
        events=("trip_contact", "trip_exit", "rest"),
        phases=(("approach", "start", "trip_contact"), ("trip", "trip_contact", "trip_exit"),
                ("after_trip", "trip_exit", "end")),
        targets=targets,
        requires=dict(_DRIVER_REQUIRES),
        description=f"generic vehicle with its left wheels over a one-sided trip ramp at {p['speed_kmh']:g} km/h",
    )


# --------------------------------------------------------------------------------------------
# S4 drop and bounce (score-only)
# --------------------------------------------------------------------------------------------

SLAB = {"size_x_m": 6.0, "size_y_m": 4.0, "thickness_m": 0.25}
PROPS = {
    "box": {"cls": "prop.box", "dims": {"length": 0.5, "width": 0.5, "height": 0.5}, "mass_kg": 20.0},
    "ball": {"cls": "prop.sphere", "dims": {"diameter": 0.3}, "mass_kg": 2.0},
    # The cylinder lies on its side with its axis along +X: "length" is the axis length.
    "cylinder": {"cls": "prop.cylinder", "dims": {"length": 0.5, "diameter": 0.3}, "mass_kg": 8.0},
}


def _prop(name: str, x: float, y: float, bottom_z: float, pitch_deg: float = 0.0) -> Actor:
    spec = PROPS[name]
    dims = spec["dims"]
    half_h = (dims["height"] if "height" in dims else dims["diameter"]) / 2.0  # vertical half extent
    return Actor(
        id=name,
        cls=spec["cls"],
        template=spec["cls"],
        dims_m=dict(dims),
        mass_kg=spec["mass_kg"],
        frame0_position_m=(x, y, bottom_z + half_h),
        frame0_pitch_deg=pitch_deg,
        mode="at_rest",
        control="passive",
    )


def _drop_and_bounce(p: Mapping[str, Any], camera: str) -> Case:
    variant = p["variant"]
    g = GRAVITY_MPS2
    slab_top = SLAB["thickness_m"]
    statics: list[Static] = [Static("ground", "ground", {"size_m": 200.0})]
    targets: list[Target] = []
    if variant == "drop":
        h = float(p["drop_height_m"])
        statics.append(Static("slab", "slab", dict(SLAB), position_m=(0.0, 0.0, 0.0)))
        statics.extend(_survey_posts((-3.5, -1.2, 1.2, 3.5), 4.0))
        actors = (
            _prop("box", -1.5, 0.0, slab_top + h),
            _prop("ball", 0.0, 0.0, slab_top + h),
            _prop("cylinder", 1.5, 0.0, slab_top + h),
        )
        t_contact = math.sqrt(2.0 * h / g)
        for prop in actors:
            targets.append(Target("time_of_event", prop.id, t_contact, 1.0 / 30.0, "s", "first_principles",
                                  event=f"first_contact.{prop.id}",
                                  note="free fall from rest: sqrt(2 h / g) after frame 0"))
            targets.append(Target("first_rebound_ratio", prop.id, None, None, "ratio", "reported", phase="bounces"))
            targets.append(Target("rest_position", prop.id, None, None, "m", "reported", event=f"rest.{prop.id}"))
        case_id = f"drop_and_bounce-drop-h{_fmt(h)}-{camera}"
        description = f"box, ball and cylinder dropped {h:g} m onto a slab"
        events: tuple[str, ...] = tuple(f"{e}.{a.id}" for a in actors for e in ("first_contact", "rest"))
        phases: tuple[tuple[str, str, str], ...] = (("fall", "start", "first_contact.box"),
                                                    ("bounces", "first_contact.box", "end"))
        duration = 3.5  # long enough for the first bounces and, for most props, rest
        aim = (0.0, 0.0, slab_top + h / 2.0)
        preset = CameraPreset("tripod", position_m=(0.0, -9.0, 1.5 + h / 3.0), look_at_m=aim, hfov_deg=55.0)
    elif variant == "incline":
        angle = float(p["incline_deg"])
        length = 5.0
        statics.append(Static("incline", "incline", {"length_m": length, "width_m": 2.0, "angle_deg": angle},
                              position_m=(0.0, 0.0, 0.0)))
        statics.extend(_survey_posts((-3.5, -1.2, 1.2, 3.5), 4.0))
        # The incline's footprint is centred on x = 0 and its top is on the -X side. The box starts
        # at rest on the slope, a fifth of the way down from the top, with its bottom face on the slope.
        run = length * math.cos(math.radians(angle))
        rise = length * math.sin(math.radians(angle))
        x_s = -0.3 * run
        z_s = rise * (run / 2.0 - x_s) / run
        half_h = PROPS["box"]["dims"]["height"] / 2.0
        n = (math.sin(math.radians(angle)), math.cos(math.radians(angle)))
        box = Actor(
            id="box", cls="prop.box", template="prop.box", dims_m=dict(PROPS["box"]["dims"]),
            mass_kg=PROPS["box"]["mass_kg"],
            frame0_position_m=(x_s + half_h * n[0], 0.0, z_s + half_h * n[1]),
            frame0_pitch_deg=angle, mode="at_rest", control="passive")
        actors = (box,)
        targets.append(Target("slides", "box", None, None, "bool", "reported",
                              note="whether the box slides depends on the friction class; reported"))
        targets.append(Target("rest_position", "box", None, None, "m", "reported", event="rest.box"))
        case_id = f"drop_and_bounce-incline-a{_fmt(angle)}-{camera}"
        description = f"box released at rest on a {angle:g} deg incline"
        events = ("first_motion.box", "ground_contact.box", "rest.box")
        phases = (("slide", "start", "ground_contact.box"), ("after", "ground_contact.box", "end"))
        duration = 3.0
        preset = CameraPreset("tripod", position_m=(0.0, -9.0, 2.0), look_at_m=(0.5, 0.0, 0.8), hfov_deg=55.0)
    else:
        raise ValueError(f"unknown drop_and_bounce variant {variant!r}")
    requires = {
        "effects": ["contact"],
        "observables": ["position", "attitude"],
        "controlled_actors": 0,
        "seatable_vehicles": 0,
    }
    return Case(
        family="drop_and_bounce",
        case_id=case_id,
        params=dict(p),
        statics=tuple(statics),
        actors=actors,
        camera=_camera(camera, preset),
        clip_s=round(duration, 3),
        fps=30.0,
        events=events,
        phases=phases,
        targets=tuple(targets),
        requires=requires,
        score_only=True,
        description=description,
    )


def _drop_points() -> tuple[dict[str, Any], ...]:
    drops = tuple({"variant": "drop", "drop_height_m": h, "incline_deg": None} for h in (1.0, 2.0, 4.0))
    inclines = tuple({"variant": "incline", "drop_height_m": None, "incline_deg": a} for a in (20.0, 30.0))
    return drops + inclines


@dataclass(frozen=True)
class _ListFamily(Family):
    """A family whose grid is an explicit list of points rather than a product."""

    explicit: tuple[Mapping[str, Any], ...] = ()

    def points(self) -> Iterator[dict[str, Any]]:
        for point in self.explicit:
            yield dict(point)


FAMILIES: Mapping[str, Family] = {
    "barrier_crash": Family(
        "barrier_crash",
        "a generated car driven into a fixed concrete block",
        {"speed_kmh": (30.0, 40.0, 50.0), "overlap": (1.0, 0.4)},
        {"speed_kmh": 40.0, "overlap": 1.0},
        _barrier_crash,
    ),
    "ramp_jump": Family(
        "ramp_jump",
        "a generated car jumping from a straight-sloped wedge and landing on level ground",
        {"angle_deg": (10.0, 15.0, 20.0), "speed_kmh": (40.0, 55.0)},
        {"angle_deg": 15.0, "speed_kmh": 55.0},
        _ramp_jump,
    ),
    "rollover": Family(
        "rollover",
        "a generated car tipped by a short ramp under its left-hand wheels",
        {"speed_kmh": (40.0, 50.0, 60.0)},
        {"speed_kmh": 60.0},
        _rollover,
    ),
    "drop_and_bounce": _ListFamily(
        "drop_and_bounce",
        "a box, a ball and a cylinder falling onto a slab, or a box let go on a slope (scored, not fitted)",
        {"variant": ("drop", "incline")},
        {"variant": "drop", "drop_height_m": 2.0, "incline_deg": None},
        _drop_and_bounce,
        explicit=_drop_points(),
    ),
}


def family_names() -> tuple[str, ...]:
    return tuple(FAMILIES)


def cases(subset: str = "default", families: Iterable[str] | None = None,
          cameras: Iterable[str] | None = None) -> list[Case]:
    """Cases of the benchmark.

    ``subset="default"`` gives one case per family at its default grid point (tripod camera unless
    ``cameras`` says otherwise): small enough for a laptop CPU and for CI. ``subset="full"`` gives
    every grid point with both camera presets.
    """
    names = list(families) if families is not None else list(FAMILIES)
    unknown = [n for n in names if n not in FAMILIES]
    if unknown:
        raise KeyError(f"unknown families {unknown}; known: {list(FAMILIES)}")
    if subset == "default":
        cams = tuple(cameras) if cameras is not None else ("tripod",)
        return [FAMILIES[n].default_case(c) for n in names for c in cams]
    if subset == "full":
        cams = tuple(cameras) if cameras is not None else ("tripod", "handheld_ptz")
        return [case for n in names for case in FAMILIES[n].cases(cams)]
    raise ValueError(f"unknown subset {subset!r} (expected 'default' or 'full')")


def get_case(case_id: str) -> Case:
    """Look a case up by id across the full grid."""
    for case in cases("full"):
        if case.case_id == case_id:
            return case
    raise KeyError(f"no case {case_id!r} in {BENCHMARK_ID}")
