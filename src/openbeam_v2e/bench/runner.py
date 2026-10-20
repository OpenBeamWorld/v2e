# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""Run v2e-synth-1 end to end and tabulate the results.

For every case the runner

1. **generates** the case (or reuses an earlier generation with the same case hash and render
   settings): frames, masks, annotations and the synthetic-truth world package;
2. **reconstructs** the clip, by one of two sources:

   * ``baseline``: the package's reconstruction step with the CPU baseline perception models,
     given the frames, the assisted-annotation outlines and the surveyed points; this measures
     perception;
   * ``truth``: the truth world itself (an oracle reconstruction), which isolates what the
     recreation and scoring steps lose even with perfect perception;

3. **compares** the reconstruction with the truth (camera reprojection, focal length,
   dimensions, trajectories, attitudes, events, silhouettes);
4. **recreates** the scene on a connector from the reconstruction: each actor's measured size,
   frame-0 pose and entry speed go into the case's content and controller programs, every actor
   spawns at rest, and the connector runs it;
5. **scores** the run against the truth with the same comparisons, plus the case's
   first-principles targets.

A step that is not available in the installation, or that fails, marks the row
``skipped:<step>`` or ``failed:<step>`` with the reason; it never becomes a number.

v0 limits (stated in the report): the recreation uses the case's level geometry for static
objects (assisted annotation of static structure), and only the ``fake`` connector is wired for
the offline benchmark.
"""

from __future__ import annotations

import dataclasses
import importlib
import inspect
import json
import math
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from openbeam_v2e.bench import families as fam
from openbeam_v2e.bench import metrics
from openbeam_v2e.bench.report import BenchReport, BenchRow
from openbeam_v2e.bench.synth import (
    GeneratedCase,
    RenderSettings,
    TruthRun,
    _attitude,
    generate_case,
    object_model,
    simulate_case,
)
from openbeam_v2e.bench.truth import EventTimes, Track, detect_events, truth_metrics

__all__ = ["RECONSTRUCTIONS", "run_bench", "run_case", "case_from_world", "evaluate_targets", "StepUnavailable"]

RECONSTRUCTIONS = ("baseline", "truth")
MIN_ENTRY_SPEED_MPS = 0.5
"""A reconstructed entry speed below this recreates the vehicle standing at its frame-0 pose."""
CONNECTORS = ("fake",)

_RECONSTRUCT_ENTRY_POINTS = (("openbeam_v2e.reconstruct.pipeline", "reconstruct"),
                             ("openbeam_v2e.reconstruct", "reconstruct"))


class StepUnavailable(RuntimeError):
    """A pipeline step this installation does not provide."""


# --------------------------------------------------------------------------------------------
# Reconstruction
# --------------------------------------------------------------------------------------------


def _resolve(entry_points: Sequence[tuple[str, str]]) -> Callable[..., Any] | None:
    for module_name, attr in entry_points:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        fn = getattr(module, attr, None)
        if callable(fn):
            return fn
    return None


def _call_with_accepted(fn: Callable[..., Any], **kwargs: Any) -> Any:
    params = inspect.signature(fn).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return fn(**kwargs)
    return fn(**{k: v for k, v in kwargs.items() if k in params})


def reconstruct_baseline(case: fam.Case, case_dir: Path, out_dir: Path,
                         models: Mapping[str, str] | None = None) -> Any:
    """Reconstruct a generated case with the package's reconstruction step; return its world.

    ``models`` overrides the perception model of a stage, for example ``{"masks": "sam2"}``.
    """
    fn = _resolve(_RECONSTRUCT_ENTRY_POINTS)
    if fn is None:
        raise StepUnavailable("the reconstruction step is not installed "
                              "(needs openbeam_v2e.reconstruct.pipeline:reconstruct)")
    from openbeam_v2e.world import read_world

    out_dir.mkdir(parents=True, exist_ok=True)
    result = _call_with_accepted(fn, video=case_dir / "frames", out_dir=out_dir,
                                 config=case_dir / "reconstruct.json", models=dict(models) if models else None)
    if hasattr(result, "objects") and hasattr(result, "camera_track"):
        return result
    world_dir = out_dir
    if isinstance(result, (str, Path)):
        world_dir = Path(result)
    elif isinstance(result, Mapping):
        for key in ("world", "out", "path", "directory"):
            if isinstance(result.get(key), (str, Path)):
                world_dir = Path(result[key])
                break
    return read_world(world_dir)


def _world_tracks(world: Any, ids: Iterable[str]) -> dict[str, Track]:
    t = np.arange(world.frame_count) / world.fps
    out = {}
    for oid in ids:
        if not world.has_object(oid):
            continue
        obj = world.object(oid)
        known = obj.known
        if not np.any(known):
            continue
        out[oid] = Track(oid, t[known], obj.position[known], obj.rotations()[known], (obj.hull_vertices,), obj.cls)
    return out


def _statics_parts(case: fam.Case) -> list:
    from openbeam_v2e.bench.geometry import solids_for
    from openbeam_v2e.bench.truth import StaticPart

    return [StaticPart(s.static_id, v) for s in solids_for(case.statics) if s.kind != "ground" for v in s.parts]


def world_events(case: fam.Case, world: Any) -> dict[str, float | None]:
    """Event times of a world: its own event intervals by name, otherwise detected from its tracks."""
    own = {ev.name: ev.centre / world.fps for ev in getattr(world, "events", [])}
    tracks = _world_tracks(world, [a.id for a in case.actors])
    detected: dict[str, float | None] = {}
    if len(tracks) == len(case.actors):
        try:
            detected = dict(detect_events(case.family, tracks, _statics_parts(case)).times)
        except (KeyError, ValueError):
            detected = {}
    names = set(case.events) | set(detected)
    return {name: own.get(name, detected.get(name)) for name in names}


# --------------------------------------------------------------------------------------------
# Recreation from a reconstruction
# --------------------------------------------------------------------------------------------


def _body_from_world_frame(spec: fam.Actor, obj: Any) -> np.ndarray:
    """Rotation from the connector body frame to the world object's body frame.

    Vehicles, boxes and spheres share axes in every V2E frame. A cylinder's axis is the connector
    body's x axis; a world object may put it on another axis (the reconstruction hulls use z),
    which is recognised from the hull's extents.
    """
    if spec.cls != "prop.cylinder":
        return np.eye(3)
    ext = obj.hull_extents()
    diffs = [abs(ext[k] - 0.5 * (ext[(k + 1) % 3] + ext[(k + 2) % 3])) for k in range(3)]
    axis = int(np.argmax(diffs))
    if axis == 0:
        return np.eye(3)
    # The map from connector body coordinates to the object's: body x goes onto the object's axis.
    if axis == 2:
        return Rotation.from_euler("y", -90.0, degrees=True).as_matrix()
    return Rotation.from_euler("z", 90.0, degrees=True).as_matrix()


def _dims_from_world(spec: fam.Actor, obj: Any, R_fix: np.ndarray) -> dict[str, float]:
    """The actor's dimensions as the reconstruction measured them.

    Vehicles keep the measured names (length, width, height; wheelbase and track when measured).
    Props are read from the hull's extents along the connector body axes, because tools name a
    cylinder's or a sphere's dimensions differently.
    """
    measured = {k: float(v.value) for k, v in obj.dims_m.items()} if spec.cls.startswith("vehicle.") else {}
    body_ext = np.abs(R_fix.T @ obj.hull_extents())  # extents along the connector body axes
    out = {}
    for name in spec.dims_m:
        if name in measured:
            out[name] = measured[name]
        elif name == "length":
            out[name] = float(body_ext[0])
        elif name == "width":
            out[name] = float(body_ext[1])
        elif name == "height":
            out[name] = float(body_ext[2])
        elif name == "diameter":
            out[name] = float(np.mean(body_ext[1:]))
        else:
            out[name] = float(spec.dims_m[name])
    return out


def case_from_world(case: fam.Case, world: Any, speed_frames: int = 6) -> fam.Case:
    """A copy of ``case`` whose actors take their size, frame-0 pose and entry speed from ``world``.

    The frame-0 pose is the reconstructed pose at the first known frame (expressed for the
    connector's reference point, the bounding-box centre); the entry speed of a vehicle that
    drives in is its centre's speed along its heading over the first ``speed_frames`` frames.
    """
    actors = []
    for spec in case.actors:
        if not world.has_object(spec.id):
            raise KeyError(f"the reconstruction has no object {spec.id!r}")
        obj = world.object(spec.id)
        known = np.flatnonzero(obj.known)
        if not len(known):
            raise ValueError(f"the reconstruction never located {spec.id!r}")
        R_fix = _body_from_world_frame(spec, obj)
        dims = _dims_from_world(spec, obj, R_fix)
        trial = dataclasses.replace(spec, dims_m=dims)
        i0 = int(known[0])
        R_body = obj.rotations()[i0] @ R_fix
        centre = obj.centre_track()[i0]
        yaw, pitch, roll = Rotation.from_matrix(R_body).as_euler("ZYX", degrees=True)
        speed = spec.speed_mps
        mode = spec.mode
        if spec.mode == "drive_in":
            # A vehicle that drives in starts upright on flat ground: keep the reconstructed heading,
            # not the small pitch and roll a fit reports at frame 0.
            pitch = roll = 0.0
            heading = Rotation.from_euler("z", yaw, degrees=True).as_matrix()[:, 0]
            k = known[known <= i0 + speed_frames]
            if len(k) >= 2:
                c = obj.centre_track()
                speed = float(np.dot(c[k[-1]] - c[k[0]], heading) / ((k[-1] - k[0]) / world.fps))
            if speed < MIN_ENTRY_SPEED_MPS:
                mode, speed = "at_rest", 0.0  # effectively standing still at frame 0
            centre = np.array([centre[0], centre[1], dims.get("height", spec.dims_m.get("height", 1.0)) / 2.0])
        actors.append(dataclasses.replace(
            trial, frame0_position_m=tuple(float(v) for v in centre), frame0_yaw_deg=float(yaw),
            frame0_pitch_deg=float(pitch), frame0_roll_deg=float(roll), speed_mps=max(0.0, speed), mode=mode))
    return dataclasses.replace(case, actors=tuple(actors))


def run_world_from_truth(case: fam.Case, run: TruthRun, reference: Any) -> Any:
    """The run on the reference's video timeline as a ``run_export`` world (connector body frame)."""
    from openbeam_v2e.bench.synth import body_frame_object
    from openbeam_v2e.world import World

    objects = []
    for spec in case.actors:
        ref = reference.object(spec.id) if reference.has_object(spec.id) else None
        objects.append(body_frame_object(spec, object_model(spec), run.frame_tracks[spec.id],
                                         symmetry=ref.symmetry if ref is not None else None,
                                         annotation={"source": "run export", "run_actor": spec.id}))
    return World(reference.fps, reference.frame_count, reference.camera_track, list(reference.statics), objects,
                 [], kind="run_export", notes=f"bench run of {case.case_id}")


# --------------------------------------------------------------------------------------------
# Targets
# --------------------------------------------------------------------------------------------


_METRIC_KEYS = {
    "yaw_change_after_event": "yaw_change",
    "first_rebound_ratio": "rebound_ratio",
    "cumulative_roll_deg": "cumulative_roll",
}


def _target_value(t: fam.Target, run_metrics: Mapping[str, Mapping[str, Any]],
                  run_events: Mapping[str, float | None]) -> Any:
    got = run_metrics.get(t.actor) or {}
    if t.metric == "time_of_event":
        return run_events.get(t.event) if t.event else None
    return got.get(_METRIC_KEYS.get(t.metric, t.metric))


def evaluate_targets(case: fam.Case, run_metrics: Mapping[str, Mapping[str, Any]],
                     run_events: Mapping[str, float | None] | None = None) -> tuple[int, int, list[dict[str, Any]]]:
    """Check the case's sourced targets against a run's measured metrics and events.

    Returns ``(met, total, rows)``. A target whose value the run did not produce counts as not met.
    """
    rows = []
    met = total = 0
    events = dict(run_events or {})
    for t in case.targets:
        if t.value is None:
            continue
        if t.metric == "roll_onset_allowed":
            rolled = (run_metrics.get(t.actor) or {}).get("rolled_over")
            ok = None if rolled is None else (rolled <= t.value)
            value = rolled
        else:
            value = _target_value(t, run_metrics, events)
            ok = None if value is None else abs(float(value) - float(t.value)) <= float(t.tolerance or 0.0) + 1e-9
        total += 1
        met += 1 if ok else 0
        rows.append({"metric": t.metric, "actor": t.actor, "event": t.event, "target": t.value,
                     "tolerance": t.tolerance, "source": t.source, "run": value, "met": ok})
    return met, total, rows


# --------------------------------------------------------------------------------------------
# One case, all rows
# --------------------------------------------------------------------------------------------


def _load_or_generate(case: fam.Case, cases_dir: Path, settings: RenderSettings, physics_dt_s: float,
                      regenerate: bool) -> Path:
    from openbeam_v2e.backends.base import sha256_json

    case_dir = cases_dir / case.case_id
    prov_path = case_dir / "provenance.json"
    if not regenerate and prov_path.is_file():
        try:
            prov = json.loads(prov_path.read_text(encoding="utf-8"))
            from openbeam_v2e.bench.synth import CASE_FORMAT

            same = (prov.get("case_sha256") == sha256_json(case.document())
                    and prov.get("case_format") == CASE_FORMAT
                    and prov.get("render", {}).get("width") == settings.width
                    and prov.get("render", {}).get("height") == settings.height
                    and prov.get("connector", {}).get("physics_dt_s") == physics_dt_s)
            if same and (case_dir / "world").is_dir():
                return case_dir
        except (OSError, ValueError):
            pass
    generate_case(case, cases_dir, settings, physics_dt_s=physics_dt_s)
    return case_dir


def run_case(case: fam.Case, out: Path, *, reconstructions: Sequence[str] = RECONSTRUCTIONS,
             settings: RenderSettings = RenderSettings(), physics_dt_s: float = 0.002,
             regenerate: bool = False, models: Mapping[str, str] | None = None) -> list[BenchRow]:
    from openbeam_v2e.world import read_world

    started = time.perf_counter()
    rows: list[BenchRow] = []
    try:
        case_dir = _load_or_generate(case, out / "cases", settings, physics_dt_s, regenerate)
        truth_world = read_world(case_dir / "world")
        truth_doc = json.loads((case_dir / "truth.json").read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - reported in the row
        row = BenchRow(case.case_id, case.family, status="failed:generate", error=f"{type(exc).__name__}: {exc}")
        row.values["seconds"] = time.perf_counter() - started
        return [row]
    gen_seconds = time.perf_counter() - started
    truth_events = {k: v["t_s"] for k, v in truth_doc["bench"]["events"].items()}
    landmarks = np.asarray(truth_doc.get("landmarks", []), float)
    subjects = [a.id for a in case.subjects]

    for source in reconstructions:
        t_row = time.perf_counter()
        row = BenchRow(case.case_id, case.family)
        row.values["recon"] = source if (source != "baseline" or not models) else \
            "baseline+" + ",".join(f"{k}={v}" for k, v in sorted(models.items()))
        row.detail["generation_s"] = round(gen_seconds, 3)
        # ---- reconstruct
        try:
            if source == "truth":
                recon = truth_world
            elif source == "baseline":
                recon = reconstruct_baseline(case, case_dir, out / "recon" / case.case_id, models)
            else:
                raise ValueError(f"unknown reconstruction source {source!r}")
        except StepUnavailable as exc:
            row.status, row.error = "skipped:reconstruct", str(exc)
            row.values["seconds"] = time.perf_counter() - t_row
            rows.append(row)
            continue
        except Exception as exc:  # noqa: BLE001
            row.status, row.error = "failed:reconstruct", f"{type(exc).__name__}: {exc}"
            row.detail["traceback"] = traceback.format_exc(limit=4)
            row.values["seconds"] = time.perf_counter() - t_row
            rows.append(row)
            continue
        # ---- compare the reconstruction with the truth
        try:
            recon_events = world_events(case, recon)
            t1 = metrics.compare_worlds(truth_world, recon, subjects, truth_events, recon_events,
                                        points=landmarks if len(landmarks) else None)
            if t1.camera is not None:
                row.values["cam_rms_px"] = t1.camera.rms_px
                row.values["focal_err_pct"] = t1.camera.focal_err_pct
            dims = [o.dims_err_max_m for o in t1.objects.values() if o.dims_err_max_m is not None]
            row.values["dims_err_cm"] = 100.0 * float(np.mean(dims)) if dims else None
            row.values["recon_traj_rms_m"] = t1.mean("traj_rms_m", subjects)
            row.values["recon_att_deg"] = t1.mean("att_mean_deg", subjects)
            row.values["recon_event_frames"] = t1.event_error_mean()
            row.detail["reconstruction"] = t1.to_json()
        except Exception as exc:  # noqa: BLE001
            row.status, row.error = "failed:compare_reconstruction", f"{type(exc).__name__}: {exc}"
            row.detail["traceback"] = traceback.format_exc(limit=4)
            row.values["seconds"] = time.perf_counter() - t_row
            rows.append(row)
            continue
        # ---- recreate on the connector and score
        try:
            run_case_ = case_from_world(case, recon)
            run_dir = out / "runs" / case.case_id / source
            run = simulate_case(run_case_, run_dir, physics_dt_s=physics_dt_s)
            run_world = run_world_from_truth(run_case_, run, truth_world)
            t4 = metrics.compare_worlds(truth_world, run_world, subjects, truth_events, run.events.times, dims=False)
            row.values["claim"] = run.claim_class
            row.values["run_traj_rms_m"] = t4.mean("traj_rms_m", subjects)
            row.values["run_event_frames"] = t4.event_error_mean()
            row.values["silhouette_iou"] = t4.mean("iou_mean", subjects)
            met, total, target_rows = evaluate_targets(case, run.metrics, run.events.times)
            row.values["targets"] = f"{met}/{total}"
            row.detail["run"] = t4.to_json()
            row.detail["run_valid"] = run.valid
            row.detail["run_dir"] = str(run_dir)
            row.detail["targets"] = target_rows
            row.detail["recreated_from"] = {a.id: {"dims_m": dict(a.dims_m), "frame0_position_m": a.frame0_position_m,
                                                   "frame0_yaw_deg": a.frame0_yaw_deg, "speed_mps": a.speed_mps}
                                            for a in run_case_.actors}
            row.detail["score"] = _score_detail(truth_world, run_world, case_dir)
            if not run.valid:
                row.status, row.error = "failed:run", "; ".join(run.invalid_reasons)
        except Exception as exc:  # noqa: BLE001
            row.status, row.error = "failed:run", f"{type(exc).__name__}: {exc}"
            row.detail["traceback"] = traceback.format_exc(limit=4)
        row.values["seconds"] = time.perf_counter() - t_row
        rows.append(row)
    return rows


def _score_detail(truth_world: Any, run_world: Any, case_dir: Path) -> Any:
    """The package scorer's report for the run, when the scorer is installed (else ``None``)."""
    fn = _resolve((("openbeam_v2e.score.scorer", "score_worlds"), ("openbeam_v2e.score", "score_worlds")))
    if fn is None:
        return None
    try:
        from openbeam_v2e.schema.scene import load_scene

        scene = load_scene(case_dir / "scene.json")
        report = _call_with_accepted(fn, reference=truth_world, run=run_world, scene=scene)
    except Exception as exc:  # noqa: BLE001 - the bench's own comparison already ran
        return {"error": f"{type(exc).__name__}: {exc}"}
    if isinstance(report, Mapping):
        return {k: report[k] for k in ("objective", "claim", "targets") if k in report}
    return None


# --------------------------------------------------------------------------------------------
# The benchmark
# --------------------------------------------------------------------------------------------


def run_bench(out: str | Path, *, subset: str = "default", families: Iterable[str] | None = None,
              cameras: Iterable[str] | None = None, reconstructions: Sequence[str] = RECONSTRUCTIONS,
              connector: str = "fake", settings: RenderSettings = RenderSettings(), physics_dt_s: float = 0.002,
              regenerate: bool = False, models: Mapping[str, str] | None = None,
              progress: Callable[[str], None] | None = None) -> BenchReport:
    """Run the benchmark and write ``report.md``, ``report.csv`` and ``report.json`` into ``out``."""
    if connector not in CONNECTORS:
        raise ValueError(f"the offline benchmark runs on {CONNECTORS}; connector {connector!r} is planned "
                         f"(see benchmarks/v2e-synth-1/REGENERATE_WITH_OPENBEAM.md)")
    unknown = sorted(set(reconstructions) - set(RECONSTRUCTIONS))
    if unknown:
        raise ValueError(f"unknown reconstruction sources {unknown}; known: {RECONSTRUCTIONS}")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cases = fam.cases(subset, families=families, cameras=cameras)
    report = BenchReport(fam.BENCHMARK_ID, fam.LABEL, {
        "subset": subset, "connector": connector, "reconstruction": ",".join(reconstructions),
        "render": f"{settings.width}x{settings.height}", "physics_dt_s": physics_dt_s,
        "statics": "case level geometry (v0)",
        "models": ",".join(f"{k}={v}" for k, v in sorted((models or {}).items())) or "baseline"})
    for case in cases:
        if progress:
            progress(f"{case.case_id}: generating and running")
        for row in run_case(case, out, reconstructions=reconstructions, settings=settings, physics_dt_s=physics_dt_s,
                            regenerate=regenerate, models=models):
            report.add(row)
            if progress:
                progress(f"  {row.values.get('recon')}: {row.status}{' ' + row.error if row.error else ''}")
    report.write(out)
    return report
