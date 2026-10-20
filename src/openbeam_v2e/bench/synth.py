# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""Generate v2e-synth-1 cases: simulated truth, rendered frames and the files a pipeline reads.

For one :class:`~openbeam_v2e.bench.families.Case` the generator

1. builds the case's content for a connector (level primitives, a generic vehicle or props) and
   spawns every actor **at rest**: a vehicle that enters moving starts on a run-up behind its
   frame-0 pose and the benchmark driver (:mod:`openbeam_v2e.bench.drive`) brings it to the design
   speed with throttle, brake and steering only;
2. negotiates a plan and runs it through the harness, which writes the plan, the call ledger and
   the ``v2e-runlog/1`` rows (the ``generation/`` directory of the case);
3. aligns video frame 0 with simulated time (the instant the subject crosses its frame-0 pose, or
   spawn for scenes without a driven vehicle), detects the events from the rows and the geometry,
   and computes the truth metrics;
4. renders the clip with the in-package ray caster through the case's camera (a tripod, or a
   panning, tilting and zooming handheld camera at one position), with per-object masks;
5. writes frames, masks, the reconstruction inputs a person would provide (``annotations.json``:
   a few outlines per moving object; ``reconstruct.json``: surveyed points, classes), the
   truth as a world package (``world/``, kind ``synthetic_truth``), ``truth.json``,
    ``scene.json``, ``templates.json``, ``programs.json``, ``targets.json`` and ``provenance.json``.

In the first release the connector is ``fake`` (a deterministic rigid-body test double), so every
case is bit-reproducible on any machine. The same steps run with an engine connector that offers
the stepped regime and client lockstep or an engine-side host; see
``benchmarks/v2e-synth-1/REGENERATE_WITH_OPENBEAM.md``.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from openbeam_v2e import __version__
from openbeam_v2e.backends import harness
from openbeam_v2e.backends.base import (
    STEPPED,
    CompiledScene,
    LevelRef,
    ObjectRef,
    Package,
    Pose,
    RunSpec,
    SpawnSpec,
    sha256_json,
)
from openbeam_v2e.bench import drive
from openbeam_v2e.bench.cameras import camera_track, follow_points
from openbeam_v2e.bench.families import BENCHMARK_ID, BENCHMARK_VERSION, GENERIC_VEHICLE, LABEL, Actor, Case
from openbeam_v2e.bench.geometry import StaticSolid, solids_for
from openbeam_v2e.bench.truth import (
    EventTimes,
    StaticPart,
    Track,
    detect_events,
    event_frames,
    truth_metrics,
)
from openbeam_v2e.reconstruct.hulls import ConvexPart, Hull, make_hull
from openbeam_v2e.reconstruct.render import RenderObject, Scene
from openbeam_v2e.reconstruct.synthetic import PoseSequence, SyntheticClip, SyntheticObject, render_clip, tiled_ground

__all__ = [
    "RenderSettings",
    "ObjectModel",
    "TruthRun",
    "GeneratedCase",
    "object_model",
    "compile_case",
    "simulate_case",
    "render_case",
    "generate_case",
    "load_case_dir",
    "run",
]

RUNUP_ACCEL_MPS2 = 2.0
"""Reference acceleration for run-up length: L = v^2 / (2 a) + settle distance."""
RUNUP_SETTLE_M = 15.0
CASE_FORMAT = 5
"""Version of the generated case layout; a runner regenerates cases written in another format."""
TELEMETRY_HZ = 240.0
CONTROL_HZ = 100.0

_COLOURS = {
    "car": (0.78, 0.13, 0.11),
    "box": (0.20, 0.42, 0.85),
    "ball": (0.95, 0.75, 0.10),
    "cylinder": (0.15, 0.65, 0.35),
}
_STATIC_COLOURS = {
    "barrier": (0.72, 0.71, 0.68),
    "wedge_ramp": (0.62, 0.55, 0.46),
    "trip_ramp": (0.62, 0.55, 0.46),
    "incline": (0.60, 0.58, 0.55),
    "slab": (0.66, 0.66, 0.70),
    "post": (0.95, 0.50, 0.12),
}
_FIRST_STATIC_LABEL = 100


@dataclass(frozen=True)
class RenderSettings:
    width: int = 320
    height: int = 180
    noise_sigma: float = 1.0
    tile_m: float = 2.0
    seed: int = 0
    prompts: int = 3
    write_masks: bool = True
    write_mp4: bool = False


# --------------------------------------------------------------------------------------------
# Objects: connector content, render hull and the transform between their frames
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ObjectModel:
    """How one actor appears to a box-based connector and to the renderer.

    The connector's reference point is the centre of the object's bounding box. The render hull's
    origin is the centre of its bottom face (see :mod:`openbeam_v2e.reconstruct.hulls`); for a
    cylinder lying along +X the hull's axis (body z) is turned onto +X by ``R_body_hull``.
    """

    actor: str
    shape: str                       # "vehicle" | "box" | "sphere" | "cylinder"
    dims_lwh: tuple[float, float, float]
    mass_kg: float
    contents: Mapping[str, Any]      # the fake connector's object entry
    hull: Hull
    R_body_hull: np.ndarray          # hull frame -> connector body frame
    hull_origin_body: np.ndarray     # hull origin in the connector body frame (relative to the reference point)
    colour: tuple[float, float, float]

    def hull_pose(self, p_ref: np.ndarray, R_wb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Hull origin and hull-to-world rotation for a connector pose (reference point, body rotation)."""
        return p_ref + R_wb @ self.hull_origin_body, R_wb @ self.R_body_hull


def object_model(spec: Actor) -> ObjectModel:
    d = spec.dims_m
    colour = _COLOURS.get(spec.id, (0.5, 0.5, 0.5))
    if spec.cls.startswith("vehicle."):
        gv = GENERIC_VEHICLE
        lwh = (float(d["length"]), float(d["width"]), float(d["height"]))
        contents = {
            "shape": "vehicle", "dims_m": list(lwh), "mass_kg": spec.mass_kg, "seatable": True,
            "vehicle": {"wheelbase_m": float(d.get("wheelbase", gv["wheelbase_m"])),
                        "track_m": float(d.get("track", gv["track_m"])),
                        "ride_height_m": gv["ride_height_m"], "wheel_radius_m": gv["wheel_radius_m"],
                        "cg_height_m": gv["com_height_m"]},
        }
        hull = make_hull("vehicle", lwh)
        return ObjectModel(spec.id, "vehicle", lwh, spec.mass_kg, contents, hull, np.eye(3),
                           np.array([0.0, 0.0, -lwh[2] / 2.0]), colour)
    if spec.cls == "prop.sphere":
        dia = float(d["diameter"])
        lwh = (dia, dia, dia)
        contents = {"shape": "sphere", "dims_m": list(lwh), "mass_kg": spec.mass_kg, "seatable": False}
        return ObjectModel(spec.id, "sphere", lwh, spec.mass_kg, contents, make_hull("sphere", lwh), np.eye(3),
                           np.array([0.0, 0.0, -dia / 2.0]), colour)
    if spec.cls == "prop.cylinder":
        length, dia = float(d["length"]), float(d["diameter"])
        lwh = (length, dia, dia)  # connector body frame: axis along x
        contents = {"shape": "cylinder", "dims_m": list(lwh), "mass_kg": spec.mass_kg, "seatable": False}
        hull = make_hull("cylinder", (dia, dia, length))  # hull frame: axis along z
        R_bh = Rotation.from_euler("y", 90.0, degrees=True).as_matrix()  # hull z -> body x
        origin = -R_bh @ np.array([0.0, 0.0, length / 2.0])
        return ObjectModel(spec.id, "cylinder", lwh, spec.mass_kg, contents, hull, R_bh, origin, colour)
    if spec.cls == "prop.box":
        lwh = (float(d["length"]), float(d["width"]), float(d["height"]))
        contents = {"shape": "box", "dims_m": list(lwh), "mass_kg": spec.mass_kg, "seatable": False}
        return ObjectModel(spec.id, "box", lwh, spec.mass_kg, contents, make_hull("box", lwh), np.eye(3),
                           np.array([0.0, 0.0, -lwh[2] / 2.0]), colour)
    raise ValueError(f"no object model for class {spec.cls!r}")


def _attitude(spec: Actor) -> np.ndarray:
    return Rotation.from_euler("ZYX", [spec.frame0_yaw_deg, spec.frame0_pitch_deg, spec.frame0_roll_deg],
                               degrees=True).as_matrix()


def runup_length_m(speed_mps: float) -> float:
    return speed_mps * speed_mps / (2.0 * RUNUP_ACCEL_MPS2) + RUNUP_SETTLE_M


def spawn_pose(spec: Actor) -> Pose:
    """Where the actor spawns at rest: its frame-0 pose, or the start of its run-up when it drives in."""
    R = _attitude(spec)
    q = Rotation.from_matrix(R).as_quat()
    x, y, z = spec.frame0_position_m
    if spec.mode == "drive_in":
        heading = R[:, 0]
        start = np.array([x, y, z]) - heading * runup_length_m(spec.speed_mps)
        return Pose(tuple(float(v) for v in start), tuple(float(v) for v in q))
    return Pose((float(x), float(y), float(z)), tuple(float(v) for v in q))


# --------------------------------------------------------------------------------------------
# Compile for the fake connector and simulate
# --------------------------------------------------------------------------------------------


def compile_case(case: Case, solids: Sequence[StaticSolid] | None = None) -> tuple[CompiledScene, dict[str, ObjectModel]]:
    """The case as a ``v2e.compiled/1`` scene for the ``fake`` connector, plus each actor's object model."""
    solids = list(solids if solids is not None else solids_for(case.statics))
    statics = [prim for solid in solids for prim in solid.fake_primitives()]
    level_contents = {"ground": {"z": 0.0}, "statics": statics}
    models = {a.id: object_model(a) for a in case.actors}
    objects = {f"{name}_object": dict(m.contents) for name, m in models.items()}
    object_contents = {"objects": objects}
    packages = (
        Package("bench-level", "fake.level", sha256_json(level_contents), contents=level_contents),
        Package("bench-objects", "fake.objects", sha256_json(object_contents), contents=object_contents),
    )
    refs = []
    for name, m in models.items():
        raw = objects[f"{name}_object"]
        refs.append(ObjectRef(name=f"{name}_object", kind="vehicle" if m.shape == "vehicle" else "prop",
                              package="bench-objects", engine_file=f"{name}_object", seatable=bool(raw["seatable"]),
                              dims_m=tuple(raw["dims_m"]), mass_kg=float(raw["mass_kg"]),
                              geometry={k: v for k, v in raw.items() if k not in ("dims_m", "mass_kg", "seatable")}))
    spawns = tuple(SpawnSpec(a.id, f"{a.id}_object", spawn_pose(a)) for a in case.actors)
    compiled = CompiledScene(connector="fake", dialect="fake/primitives",
                             level=LevelRef(package="bench-level", level_id="bench-level"),
                             packages=packages, objects=tuple(refs), spawns=spawns,
                             programs=programs_for(case), scene=case.case_id,
                             notes=(f"{BENCHMARK_ID} case {case.case_id}",))
    return compiled, models


def programs_for(case: Case) -> dict[str, dict[str, Any]]:
    out = {}
    for a in case.actors:
        if a.control == "driver":
            out[a.id] = drive.cruise_program(
                a.id, a.speed_mps, a.frame0_position_m[1], a.drive.get("stages", ()),
                heading_deg=a.frame0_yaw_deg, line_x_m=a.frame0_position_m[0])
    return out


def generation_scene(case: Case) -> dict[str, Any]:
    """The scene the harness negotiates for truth generation: every actor at rest at its spawn pose."""
    actors = []
    for a in case.actors:
        pose = spawn_pose(a)
        actors.append({
            "id": a.id, "class": a.cls, "seatable": a.seatable,
            "control": a.control,
            "initial_state": {"mode": "at_rest",
                              "pose": {"position_m": list(pose.position_m), "quat_xyzw": list(pose.quat_xyzw)}},
        })
    return {"schema": "v2e.scene/1", "world": "world/", "actors": actors, "requires": dict(case.requires)}


@dataclass
class TruthRun:
    """The generation run of a case, aligned to video time."""

    t0_s: float                              # simulated time of video frame 0
    tracks: dict[str, Track]                 # dense, times relative to frame 0
    frame_tracks: dict[str, Track]           # one row per video frame
    events: EventTimes
    frames: dict[str, tuple[int, int] | None]
    metrics: dict[str, dict[str, Any]]
    run_dir: Path
    claim_class: str
    valid: bool
    invalid_reasons: list[str]
    backend_events: Mapping[str, float]
    manifest: Mapping[str, Any]
    wall_s: float


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return [r for r in rows if "t" in r and "actors" in r]


def _frame0_time(rows: list[dict[str, Any]], spec: Actor) -> float:
    """First simulated time at which ``actor`` crosses the plane through its frame-0 pose."""
    R = _attitude(spec)
    heading = R[:, 0]
    target = np.asarray(spec.frame0_position_m, float)
    prev_t, prev_s = None, None
    for row in rows:
        state = row["actors"].get(spec.id)
        if not state or state.get("p") is None:
            continue
        s = float(np.dot(np.asarray(state["p"], float) - target, heading))
        if prev_s is not None and prev_s < 0.0 <= s:
            frac = -prev_s / (s - prev_s) if s != prev_s else 0.0
            return float(prev_t + frac * (row["t"] - prev_t))
        prev_t, prev_s = float(row["t"]), s
    raise RuntimeError(f"{spec.id} never reached its frame-0 pose; lengthen the run or check the driver")


def _record_t0(run_dir: Path, t0: float) -> None:
    """Write the frame-0 trigger time into the run record, so ``openbeam-v2e score`` aligns the run."""
    path = Path(run_dir) / "run.json"
    if not path.is_file():
        return
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["t0_s"] = float(t0)
    path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _tracks_from_rows(rows: list[dict[str, Any]], models: Mapping[str, ObjectModel], t0: float,
                      classes: Mapping[str, str]) -> dict[str, Track]:
    tracks = {}
    for name, model in models.items():
        ts, ps, Rs = [], [], []
        for row in rows:
            state = row["actors"].get(name)
            if not state or state.get("p") is None or state.get("q") is None:
                continue
            R_wb = Rotation.from_quat(state["q"]).as_matrix()
            origin, R_wh = model.hull_pose(np.asarray(state["p"], float), R_wb)
            ts.append(float(row["t"]) - t0)
            ps.append(origin)
            Rs.append(R_wh)
        tracks[name] = Track(name, np.array(ts), np.array(ps), np.array(Rs),
                             tuple(part.vertices for part in model.hull.parts), classes.get(name, ""))
    return tracks


def _static_parts(solids: Sequence[StaticSolid]) -> list[StaticPart]:
    return [StaticPart(s.static_id, v) for s in solids if s.kind != "ground" for v in s.parts]


def _run_duration(case: Case) -> float:
    lead = 0.0
    for a in case.actors:
        if a.mode == "drive_in":
            v = max(a.speed_mps, 1.0)
            accel_s = v / 2.5
            lead = max(lead, accel_s + (runup_length_m(v) - v * accel_s / 2.0) / v + 2.0)
    return round(lead + case.clip_s + 0.5, 3)


def simulate_case(case: Case, work_dir: str | Path, *, physics_dt_s: float = 0.001,
                  telemetry_hz: float = TELEMETRY_HZ) -> TruthRun:
    """Run the case on the ``fake`` connector through the harness and extract aligned truth."""
    from openbeam_v2e.backends.fake import FakeBackend

    started = time.perf_counter()
    solids = solids_for(case.statics)
    compiled, models = compile_case(case, solids)
    backend = FakeBackend(physics_dt_s=physics_dt_s, controller_factory=drive.controller_factory)
    scene = generation_scene(case)
    plan = harness.negotiate(scene, backend.manifest(), programs=compiled.programs, lockstep=True,
                             opt_in={"regime": STEPPED})
    if not getattr(plan, "ok", False):
        raise RuntimeError(f"the fake connector refused {case.case_id}: {plan}")
    spec = RunSpec(duration_s=_run_duration(case), host=plan.host, regime=STEPPED, telemetry_hz=telemetry_hz,
                   control_hz=CONTROL_HZ)
    run_dir = Path(work_dir)
    if run_dir.exists():
        shutil.rmtree(run_dir)
    result = harness.execute(backend, compiled, plan, spec, run_dir)
    rows = _read_rows(result.rows)
    driven = [a for a in case.actors if a.mode == "drive_in"]
    t0 = _frame0_time(rows, driven[0]) if driven else 0.0
    _record_t0(run_dir, t0)
    classes = {a.id: a.cls for a in case.actors}
    tracks = _tracks_from_rows(rows, models, t0, classes)
    keep = {name: tr.t >= -0.5 for name, tr in tracks.items()}
    tracks = {name: Track(name, tr.t[keep[name]], tr.position[keep[name]], tr.R_wb[keep[name]], tr.hull_parts,
                          tr.object_class) for name, tr in tracks.items()}
    events = detect_events(case.family, {k: v.resample(v.t[v.t >= 0.0]) for k, v in tracks.items()},
                           _static_parts(solids))
    frame_times = np.arange(case.frames) / case.fps
    frame_tracks = {name: tr.resample(frame_times) for name, tr in tracks.items()}
    frames = event_frames(events, case.fps, case.frames)
    clipped = {name: tr.resample(tr.t[(tr.t >= 0.0) & (tr.t <= case.clip_s + 1e-9)]) for name, tr in tracks.items()}
    metrics = truth_metrics(case.family, clipped, events)
    return TruthRun(t0_s=t0, tracks=tracks, frame_tracks=frame_tracks, events=events, frames=frames,
                    metrics=metrics, run_dir=run_dir, claim_class=result.claim_class(), valid=result.valid(),
                    invalid_reasons=result.invalid_reasons(), backend_events=dict(result.events),
                    manifest={"backend": result.manifest.backend, "kind": result.manifest.kind,
                              "engine": result.manifest.engine, "dialect": result.manifest.dialect,
                              "physics_dt_s": physics_dt_s},
                    wall_s=round(time.perf_counter() - started, 3))


# --------------------------------------------------------------------------------------------
# Render
# --------------------------------------------------------------------------------------------


def _static_render_objects(solids: Sequence[StaticSolid]) -> list[RenderObject]:
    out = []
    for k, s in enumerate(solids):
        if s.kind == "ground":
            continue
        lo, hi = s.vertices.min(axis=0), s.vertices.max(axis=0)
        hull = Hull("static", tuple(float(v) for v in (hi - lo)), tuple(ConvexPart(v) for v in s.parts), "none")
        out.append(RenderObject(_FIRST_STATIC_LABEL + k, hull, np.eye(3), np.zeros(3),
                                _STATIC_COLOURS.get(s.kind, (0.6, 0.6, 0.6))))
    return out


def landmarks_for(solids: Sequence[StaticSolid], tile_m: float, extent: tuple[float, float, float, float]) -> np.ndarray:
    """Known world points: tile corners of the ground texture in ``extent`` and static solids' top corners."""
    x0, x1, y0, y1 = extent
    xs = np.arange(math.ceil(x0 / tile_m), math.floor(x1 / tile_m) + 1) * tile_m
    ys = np.arange(math.ceil(y0 / tile_m), math.floor(y1 / tile_m) + 1) * tile_m
    pts = [[x, y, 0.0] for x in xs for y in ys]
    for s in solids:
        if s.kind == "ground":
            continue
        for part in s.parts:
            top = part[:, 2].max()
            pts += [list(v) for v in part if v[2] >= top - 1e-6]
    return np.unique(np.round(np.asarray(pts, float), 9), axis=0)


def render_case(case: Case, truth: TruthRun, settings: RenderSettings = RenderSettings()) -> SyntheticClip:
    solids = solids_for(case.statics)
    models = {a.id: object_model(a) for a in case.actors}
    subjects = [a.id for a in case.subjects]
    centres = follow_points([truth.frame_tracks[s].centre() for s in subjects])
    cams = camera_track(case.camera, settings.width, settings.height, case.fps, centres)
    objects = []
    for label, a in enumerate(case.actors, start=1):
        tr = truth.frame_tracks[a.id]
        objects.append(SyntheticObject(a.id, label, models[a.id].hull, PoseSequence(tr.position, tr.R_wb),
                                       models[a.id].colour, a.cls))
    scene = Scene(ground_shader=tiled_ground(settings.tile_m, settings.seed),
                  static_objects=_static_render_objects(solids))
    all_pos = np.concatenate([truth.frame_tracks[a.id].position for a in case.actors])
    extent = (float(all_pos[:, 0].min()) - 20.0, float(all_pos[:, 0].max()) + 20.0,
              float(all_pos[:, 1].min()) - 12.0, float(all_pos[:, 1].max()) + 14.0)
    landmarks = landmarks_for(solids, settings.tile_m, extent)
    meta = {"benchmark": BENCHMARK_ID, "case": case.case_id, "label": LABEL, "camera_preset": case.camera.name,
            "tile_m": settings.tile_m}
    return render_clip(cams, objects, scene, case.fps, landmarks, settings.noise_sigma, settings.seed, meta)


# --------------------------------------------------------------------------------------------
# Assisted annotation inputs
# --------------------------------------------------------------------------------------------


def _outline(mask: np.ndarray) -> list[list[float]] | None:
    """Convex outline of a mask in pixel coordinates (pixel centres), or None if empty."""
    rows, cols = np.nonzero(mask)
    if len(rows) < 3:
        return None
    from openbeam_v2e.reconstruct.silhouette import convex_hull_2d

    pts = np.stack([cols + 0.5, rows + 0.5], axis=1)
    hull = convex_hull_2d(pts)
    return np.round(hull, 2).tolist()


SURVEY_SIGMA_PX = 0.5
"""Pixel noise added to the surveyed points' image positions, as when a person clicks them."""


def annotations_for(case: Case, clip: SyntheticClip, prompts: int = 3) -> dict[str, Any]:
    """Assisted-annotation input (``v2e.annotations/1``): a few outlines per moving object.

    This is what a person provides for assisted reconstruction: the outline of each moving object
    in a few frames (the first, middle and last frame in which it is well in view).
    """
    objects = {}
    for a in case.subjects:
        masks = clip.masks[a.id]
        areas = np.array([int(m.sum()) for m in masks])
        # Outline frames where the object is well in view (at least half its largest area), as a
        # person would pick them, not slivers at the image edge.
        visible = [i for i in range(len(masks)) if areas[i] >= max(4, 0.5 * areas.max())]
        if not visible:
            continue
        picks = sorted({visible[int(round(k * (len(visible) - 1) / max(1, prompts - 1)))] for k in range(prompts)})
        outlines = []
        for i in picks:
            poly = _outline(masks[i])
            if poly is not None:
                outlines.append({"frame": int(i), "polygon": poly})
        objects[a.id] = {"class": a.cls, "prompts": outlines}
    width, height = clip.cameras[0].width, clip.cameras[0].height
    return {"schema": "v2e.annotations/1", "image_size": [int(width), int(height)], "objects": objects}


def survey_points(clip: SyntheticClip, seed: int = 0) -> list[dict[str, Any]]:
    """Surveyed points seen in frame 0: known world coordinates and their (clicked) image positions."""
    world, pix = clip.landmark_observations(0, noise_px=SURVEY_SIGMA_PX, seed=seed)
    return [{"world": [round(float(v), 6) for v in w], "image": [round(float(v), 3) for v in uv]}
            for w, uv in zip(world, pix)]


def reconstruct_config(case: Case, clip: SyntheticClip, seed: int = 0) -> dict[str, Any]:
    """The case's ``v2e.reconstruct/1`` configuration for the baseline (CPU) perception models.

    It holds what a person would know about their own clip: the frame rate, the ground height,
    surveyed points for calibration, which objects to reconstruct and their classes, and the
    annotation file. Dimensions are not given: they are measured.
    """
    return {
        "schema": "v2e.reconstruct/1",
        "video": {"path": "frames", "fps": case.fps},
        "ground_z": 0.0,
        "camera": {"calibration": {"model": "baseline.points", "frame": 0, "sigma_px": SURVEY_SIGMA_PX,
                                   "points": survey_points(clip, seed)},
                   "tracking": "baseline"},
        "segmentation": "baseline",
        "pose": "baseline",
        "objects": [{"id": a.id, "class": a.cls} for a in case.subjects],
        "annotations": "annotations.json",
    }


# --------------------------------------------------------------------------------------------
# Writing a case
# --------------------------------------------------------------------------------------------


@dataclass
class GeneratedCase:
    case: Case
    directory: Path
    truth: TruthRun
    clip: SyntheticClip
    files: dict[str, Path] = field(default_factory=dict)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialise {type(value).__name__}")


def _dump(path: Path, data: Any) -> Path:
    path.write_text(json.dumps(data, indent=1, sort_keys=True, default=_jsonable) + "\n", encoding="utf-8")
    return path


_DRIVABLE = {"ground", "wedge_ramp", "trip_ramp", "incline", "slab"}


def _merge_meshes(meshes: Sequence[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    verts, faces, offset = [], [], 0
    for v, f in meshes:
        verts.append(np.asarray(v, float))
        faces.append(np.asarray(f, np.int32) + offset)
        offset += len(v)
    return np.concatenate(verts), np.concatenate(faces)


def _hull_mesh(hull: Hull) -> tuple[np.ndarray, np.ndarray]:
    from scipy.spatial import ConvexHull

    meshes = []
    for part in hull.parts:
        v = np.asarray(part.vertices, float)
        h = ConvexHull(v)
        tris = []
        for simplex, eq in zip(h.simplices, h.equations):
            a, b, c = v[simplex]
            if np.dot(np.cross(b - a, c - a), eq[:3]) < 0:
                simplex = simplex[[0, 2, 1]]
            tris.append(simplex)
        meshes.append((v, np.asarray(tris, np.int32)))
    return _merge_meshes(meshes)


def body_frame_object(spec: Actor, model: ObjectModel, track: Track, **extra: Any) -> Any:
    """A world-package object track in the connector's body frame.

    Positions are the connector's reference point (the bounding-box centre) and rotations the
    connector body's, with the hull mesh expressed in that frame. A run exported from the
    connector's rows (``openbeam_v2e.world.export``) therefore lines up with this object exactly.
    """
    from openbeam_v2e.world import ObjectTrack

    hv, hf = _hull_mesh(model.hull)
    hv_body = hv @ model.R_body_hull.T + model.hull_origin_body
    R_wb = np.einsum("nij,jk->nik", track.R_wb, model.R_body_hull.T)
    p_ref = track.position - np.einsum("nij,j->ni", R_wb, model.hull_origin_body)
    return ObjectTrack(id=spec.id, cls=spec.cls, hull_vertices=hv_body, hull_faces=hf, position=p_ref,
                       quat_xyzw=Rotation.from_matrix(R_wb).as_quat(), **extra)


def truth_world(case: Case, truth: TruthRun, clip: SyntheticClip) -> tuple[Any, dict[str, int]]:
    """The case's truth as a world package model (kind ``synthetic_truth``) and the static mesh ids.

    Object poses are those of the connector's reference point (the bounding-box centre) in the
    connector's body frame, with the hull mesh stored in that frame (see :func:`body_frame_object`).
    """
    from openbeam_v2e.schema.common import Measured
    from openbeam_v2e.world import Camera as WorldCamera
    from openbeam_v2e.world import CameraTrack, EventInterval, StaticMesh, World

    cams = []
    for c in clip.cameras:
        i = c.intrinsics
        cams.append(WorldCamera(i.width, i.height, i.fx, i.fy, i.cx, i.cy, np.asarray(c.R_cw).T,
                                np.asarray(c.centre, float), i.k1, i.k2))
    statics, mesh_ids = [], {}
    for k, solid in enumerate(solids_for(case.statics), start=1):
        v, f = _merge_meshes(solid.meshes())
        statics.append(StaticMesh(k, v, f, name=solid.static_id,
                                  role="drivable" if solid.kind in _DRIVABLE else "collision",
                                  sigma_m=0.0, observed=solid.observed))
        mesh_ids[solid.static_id] = k
    objects = []
    models = {a.id: object_model(a) for a in case.actors}
    for a in case.actors:
        tr = truth.frame_tracks[a.id]
        visible = np.array([bool(m.any()) for m in clip.masks[a.id]])
        objects.append(body_frame_object(
            a, models[a.id], tr, visible=visible, sigma_position_m=np.zeros(len(tr.t)),
            sigma_attitude_rad=np.zeros(len(tr.t)), dims_m={k: Measured(float(v), 0.0) for k, v in a.dims_m.items()},
            mass_kg=Measured(float(a.mass_kg), 0.0),
            annotation={"source": "simulated truth", "benchmark": BENCHMARK_ID,
                        "pose_reference": "bounding-box centre, connector body frame"}))
    events = [EventInterval(name, interval, truth.events.kinds.get(name, "custom"),
                            tuple(truth.events.actors.get(name, ())), "simulated truth")
              for name, interval in truth.frames.items() if interval is not None]
    world = World(fps=case.fps, frame_count=case.frames, camera_track=CameraTrack.from_cameras(cams),
                  statics=statics, objects=objects, events=events, kind="synthetic_truth",
                  camera_sigma_px=0.0, source_video={"name": "frames/%06d.png", "licence": "CC-BY-4.0"},
                  notes=f"{BENCHMARK_ID} {case.case_id}: {LABEL}")
    return world, mesh_ids


def truth_scene(case: Case, truth: TruthRun, mesh_ids: Mapping[str, int]) -> dict[str, Any]:
    """The case's ``v2e.scene/1`` over its truth world package (``world/``).

    Static actors point at the package's meshes; dynamic actors' frame-0 poses are references
    ``track:<id>@0`` into its object tracks; events carry the truth frame intervals.
    """
    doc = case.scene_document()
    for actor_doc in doc["actors"]:
        if actor_doc.get("class") == "static":
            actor_doc["mesh"] = int(mesh_ids[actor_doc["id"]])
            actor_doc["role"] = "collision"
            continue
        a = case.actor(actor_doc["id"])
        state = actor_doc["initial_state"]
        if state["mode"] == "at_rest":
            state.pop("frame0_pose", None)
            state["pose"] = f"track:{a.id}@0"
        actor_doc["mass_kg"] = a.mass_kg
    events = []
    for name in case.events:
        interval = truth.frames.get(name)
        if interval is None:
            continue
        events.append({"name": name, "frame": list(interval), "kind": truth.events.kinds.get(name, "custom"),
                       "actors": list(truth.events.actors.get(name, ())), "evidence": "simulated truth"})
    doc["events"] = events
    known = {e["name"] for e in events} | {"start", "end", "rest"}
    doc["phases"] = [ph for ph in doc["phases"] if ph["from"] in known and ph["to"] in known]
    doc["fps"] = case.fps
    doc["name"] = case.case_id
    return doc


def _schema_issues(scene: Mapping[str, Any]) -> list[str]:
    try:
        from openbeam_v2e.schema.scene import validate_scene
    except ImportError:  # pragma: no cover - the schema package ships with the distribution
        return []
    return [str(i) for i in validate_scene(scene)]


def _encode_mp4(frames_dir: Path, fps: float, out: Path) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return False
    cmd = [ffmpeg, "-loglevel", "error", "-y", "-framerate", f"{fps:g}", "-i", str(frames_dir / "%06d.png"),
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-map_metadata", "-1", "-fflags", "+bitexact",
           "-bsf:v", "filter_units=remove_types=6", str(out)]
    return subprocess.run(cmd, check=False).returncode == 0


def generate_case(case: Case, out_dir: str | Path, settings: RenderSettings = RenderSettings(), *,
                  physics_dt_s: float = 0.001) -> GeneratedCase:
    """Simulate, render and write one case into ``out_dir/<case_id>/``."""
    directory = Path(out_dir) / case.case_id
    directory.mkdir(parents=True, exist_ok=True)
    truth = simulate_case(case, directory / "generation", physics_dt_s=physics_dt_s)
    clip = render_case(case, truth, settings)
    clip.write(directory, masks=settings.write_masks)  # frames/, masks/, truth.json (cameras, poses, landmarks)
    files: dict[str, Path] = {"frames": directory / "frames", "truth": directory / "truth.json"}
    if settings.write_masks:
        files["masks"] = directory / "masks"

    # Extend truth.json with the bench's alignment, events and metrics.
    truth_doc = json.loads(files["truth"].read_text(encoding="utf-8"))
    truth_doc["bench"] = {
        "benchmark": BENCHMARK_ID,
        "version": BENCHMARK_VERSION,
        "case": case.case_id,
        "label": LABEL,
        "pose_reference": "hull origin: centre of the object's bottom face (openbeam_v2e.reconstruct.hulls)",
        "t0_sim_s": truth.t0_s,
        "events": {name: {"t_s": t, "frame": truth.frames.get(name), "kind": truth.events.kinds.get(name),
                          "actors": list(truth.events.actors.get(name, ()))}
                   for name, t in truth.events.times.items()},
        "metrics": truth.metrics,
        "claim_class": truth.claim_class,
        "valid_run": truth.valid,
    }
    _dump(files["truth"], truth_doc)

    world, mesh_ids = truth_world(case, truth, clip)
    from openbeam_v2e.world import write_world

    files["world"] = write_world(world, directory / "world")
    scene = truth_scene(case, truth, mesh_ids)
    files["scene"] = _dump(directory / "scene.json", scene)
    files["templates"] = _dump(directory / "templates.json", case.templates_document())
    files["programs"] = _dump(directory / "programs.json", programs_for(case))
    files["targets"] = _dump(directory / "targets.json", case.targets_document())
    files["case"] = _dump(directory / "case.json", case.document())
    files["annotations"] = _dump(directory / "annotations.json", annotations_for(case, clip, settings.prompts))
    files["reconstruct"] = _dump(directory / "reconstruct.json", reconstruct_config(case, clip, settings.seed))
    if settings.write_mp4 and _encode_mp4(directory / "frames", case.fps, directory / "video.mp4"):
        files["video"] = directory / "video.mp4"

    frame_hashes = hashlib.sha256()
    for path in sorted((directory / "frames").glob("*.png")):
        frame_hashes.update(_sha256_file(path).encode())
    provenance = {
        "schema": "v2e.bench-provenance/1",
        "benchmark": BENCHMARK_ID,
        "benchmark_version": BENCHMARK_VERSION,
        "case": case.case_id,
        "label": LABEL,
        "generator": "openbeam_v2e.bench.synth",
        "generator_version": __version__,
        "case_format": CASE_FORMAT,
        "connector": truth.manifest,
        "claim_class": truth.claim_class,
        "valid_run": truth.valid,
        "invalid_reasons": truth.invalid_reasons,
        "renderer": "openbeam_v2e.reconstruct.render (ray caster)",
        "render": {"width": settings.width, "height": settings.height, "noise_sigma": settings.noise_sigma,
                   "tile_m": settings.tile_m, "seed": settings.seed},
        "case_sha256": sha256_json(case.document()),
        "frames_sha256": frame_hashes.hexdigest(),
        "truth_sha256": _sha256_file(files["truth"]),
        "scene_schema_issues": _schema_issues(scene),
        "assets": "all geometry and textures are generated by openbeam-v2e; no external assets",
        "licence": "CC-BY-4.0 (data); generated by Apache-2.0 code",
        "wall_s": truth.wall_s,
    }
    files["provenance"] = _dump(directory / "provenance.json", provenance)
    return GeneratedCase(case, directory, truth, clip, files)


def load_case_dir(directory: str | Path) -> dict[str, Any]:
    """Read a generated case directory's JSON files (frames are read on demand by the caller)."""
    d = Path(directory)
    out: dict[str, Any] = {"directory": d}
    for name in ("case", "truth", "scene", "templates", "programs", "targets", "annotations", "reconstruct", "provenance"):
        path = d / f"{name}.json"
        if path.is_file():
            out[name] = json.loads(path.read_text(encoding="utf-8"))
    return out


def run(out: str | Path, connector: str | None = None, subset: str = "default",
        families: Sequence[str] | None = None, cameras: Sequence[str] | None = None,
        reconstruction: Sequence[str] | str | None = None, width: int = RenderSettings.width,
        height: int = RenderSettings.height, physics_dt_s: float = 0.002, regenerate: bool = False,
        models: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Generate and run v2e-synth-1 into ``out``; return the report as JSON (agent-tool entry point).

    ``reconstruction`` is ``"baseline"``, ``"truth"`` or both (the default); ``models`` overrides
    the perception model of a reconstruction stage (``{"masks": "sam2"}``). The report files
    ``report.md``, ``report.csv`` and ``report.json`` are written into ``out``.
    """
    from openbeam_v2e.bench.runner import RECONSTRUCTIONS, run_bench

    if isinstance(reconstruction, str):
        reconstruction = [r.strip() for r in reconstruction.split(",") if r.strip()]
    report = run_bench(out, subset=subset, families=families, cameras=cameras,
                       reconstructions=tuple(reconstruction or RECONSTRUCTIONS), connector=connector or "fake",
                       settings=RenderSettings(width=int(width), height=int(height)), physics_dt_s=physics_dt_s,
                       regenerate=regenerate, models=models)
    doc = report.to_json()
    doc["files"] = {k: str(Path(out) / f"report.{ext}") for k, ext in (("markdown", "md"), ("csv", "csv"),
                                                                        ("json", "json"))}
    return doc
