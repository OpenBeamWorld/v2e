# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""Score a run world against a reference world (``v2e.score/1``).

:func:`score_worlds` takes two worlds on the same video timeline (the reconstruction or
synthetic truth, and a run exported with :func:`openbeam_v2e.world.export.runlog_to_world`),
optionally the scene (phases, events, targets) and what is known about the run (effects,
content, validity), and reports for every object:

* per phase: position, attitude (modulo symmetry) and speed error, raw and as z-scores, and
  silhouette overlap through the reference camera, raw and over its shape limit;
* at each event: position, speed and attitude in both worlds and their differences;
* after contact events: rebound, displacement, yaw change and time to rest; rollover measures
  for vehicles; bounce measures for props;
* the frame-0 alignment of actors that drive in;

plus event timing (each scene event matched to the run's reported or detected event), the
scene's targets, the claim class and a scalar ``objective`` (mean of squared normalised errors,
lower is better) for search. Nothing here depends on a particular scene or clip.

:func:`score_run` is the file-level entry point used by the CLI and the agent tools.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from openbeam_v2e import __version__
from openbeam_v2e.schema import jsonio
from openbeam_v2e.schema.scene import Scene, load_scene
from openbeam_v2e.schema.score import SCORE_SCHEMA, SCORE_SCHEMA_ID
from openbeam_v2e.schema.validate import check
from openbeam_v2e.score.attitude import attitude_error
from openbeam_v2e.score.claims import claim_block
from openbeam_v2e.score.events import ContactOptions, DetectedEvent, detect_events, match_event, rest_frame, static_name
from openbeam_v2e.score.post_event import after_event, bounce, ground_gap, rollover
from openbeam_v2e.score.silhouette import silhouette_scores
from openbeam_v2e.score.trajectory import phase_windows, state_at, trajectory_metrics
from openbeam_v2e.world.geometry import yaw_of
from openbeam_v2e.world.model import ObjectTrack, World

__all__ = ["ScoreOptions", "score_worlds", "score_run", "TARGET_METRICS"]

MISSING_PENALTY = 100.0


@dataclass(frozen=True)
class ScoreOptions:
    silhouettes: bool = True
    silhouette_width_px: int = 320
    silhouette_max_frames: int = 40
    contact: ContactOptions = ContactOptions()
    position_floor_m: float = 0.05     # objective normaliser when the reference has no sigma
    attitude_floor_deg: float = 2.0
    event_floor_frames: float = 0.5


@dataclass
class _EventRow:
    name: str
    kind: str
    actors: tuple[str, ...]        # world names (objects and static:<mesh>)
    ref_lo: float
    ref_hi: float
    ref_source: str
    run_frame: float | None

    @property
    def ref_centre(self) -> float:
        return 0.5 * (self.ref_lo + self.ref_hi)

    @property
    def sigma(self) -> float:
        return max(0.5, 0.5 * (self.ref_hi - self.ref_lo))


def _f(x: Any) -> float | None:
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def _delta(a: Any, b: Any) -> float | None:
    a, b = _f(a), _f(b)
    return None if a is None or b is None else b - a


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _diff_dict(ref: Mapping[str, Any], run: Mapping[str, Any]) -> dict[str, Any]:
    """Members of both measure blocks as ``{name: {"reference", "run"[, "delta"]}}``."""
    out: dict[str, Any] = {}
    for key in sorted((set(ref) | set(run)) - {"available"}):
        a, b = ref.get(key), run.get(key)
        if _is_number(a) or _is_number(b):
            out[key] = {"reference": _f(a) if _is_number(a) else None, "run": _f(b) if _is_number(b) else None,
                        "delta": _delta(a, b) if _is_number(a) and _is_number(b) else None}
        else:
            out[key] = {"reference": a, "run": b}
    return out


def _object_ids(reference: World, scene: Scene | None) -> list[tuple[str, str]]:
    """(actor name, object id) pairs to score."""
    if scene is None:
        return [(o.id, o.id) for o in reference.objects]
    return [(a.id, a.object_id) for a in scene.dynamic_actors if reference.has_object(a.object_id)]


def _world_names(scene: Scene | None) -> dict[str, str]:
    """Scene actor id -> name used in detected events (objects keep their id, statics become static:<mesh>)."""
    if scene is None:
        return {}
    names = {a.id: a.object_id for a in scene.dynamic_actors}
    names.update({a.id: static_name(a.mesh) for a in scene.static_actors if a.mesh is not None})
    return names


def _event_rows(reference: World, run: World, scene: Scene | None, run_detected: list[DetectedEvent],
                names: Mapping[str, str]) -> list[_EventRow]:
    reported = {e.name: e for e in run.events}
    if scene is not None and scene.events:
        source = [(e.name, e.kind, e.actors, e.frame, "scene") for e in scene.events]
    else:
        source = [(e.name, e.kind, e.actors, e.frame, "reconstruction") for e in reference.events]
    rows = []
    for name, kind, actors, frame, origin in source:
        mapped = tuple(names.get(a, a) for a in actors)
        row = _EventRow(name, kind, mapped, float(frame[0]), float(frame[1]), origin, None)
        if name in reported:
            row.run_frame = float(reported[name].centre)
        elif kind in ("contact", "separation", "rest"):
            match = match_event(run_detected, kind, mapped[:1] if kind == "rest" else mapped, row.ref_centre)
            row.run_frame = float(match.frame) if match is not None else None
        rows.append(row)
    return rows


def _rest_boundary(world: World, objects: list[str], options: ContactOptions) -> float | None:
    frames = []
    for oid in objects:
        if not world.has_object(oid):
            return None
        f = rest_frame(world.object(oid), world.fps, options)
        if f is None:
            return None
        frames.append(f)
    return float(max(frames)) if frames else None


def _median_sigma(values: np.ndarray, floor: float) -> float:
    v = values[np.isfinite(values) & (values > 0)]
    return max(float(np.median(v)), floor) if len(v) else floor


# ----------------------------------------------------------------------------- targets

def _run_value_lookup(report: Mapping[str, Any]) -> dict[str, Callable[[Any], tuple[Any, Any]]]:
    objects = report["objects"]
    phases = {p["name"]: p for p in report["phases"]}
    events = {e["name"]: e for e in report["events"]}

    def obj(t: Any) -> Mapping[str, Any]:
        return objects.get(t.actor, {})

    def at_event(t: Any, key: str) -> tuple[Any, Any]:
        entry = obj(t).get("at_events", {}).get(t.event, {})
        return (entry.get("reference") or {}).get(key), (entry.get("run") or {}).get(key)

    def after(t: Any, key: str) -> tuple[Any, Any]:
        entry = obj(t).get("after_events", {}).get(t.event, {}).get(key, {})
        return entry.get("reference"), entry.get("run")

    def block(t: Any, name: str, key: str, index: int | None = None) -> tuple[Any, Any]:
        entry = obj(t).get(name, {}).get(key, {})
        ref, run = entry.get("reference"), entry.get("run")
        if index is not None:
            ref = ref[index] if isinstance(ref, list) and len(ref) > index else None
            run = run[index] if isinstance(run, list) and len(run) > index else None
        return ref, run

    def phase_metric(t: Any, key: str) -> tuple[Any, Any]:
        p = phases.get(t.phase or "all", {})
        return None, p.get("objects", {}).get(t.actor, {}).get(key)

    def event_time(t: Any) -> tuple[Any, Any]:
        e = events.get(t.event, {})
        fps = report["fps"]
        ref = 0.5 * sum(e["reference_frame"]) / fps if e else None
        return ref, (e.get("run_frame") / fps if e and e.get("run_frame") is not None else None)

    return {
        "speed_at_event": lambda t: at_event(t, "speed_mps"),
        "yaw_at_event": lambda t: at_event(t, "yaw_deg"),
        "time_of_event": event_time,
        "rebound_speed": lambda t: after(t, "rebound_speed_mps"),
        "displacement_after_event": lambda t: after(t, "displacement_m"),
        "yaw_change_after_event": lambda t: after(t, "yaw_change_deg"),
        "time_to_rest_after_event": lambda t: after(t, "time_to_rest_s"),
        "cumulative_roll_deg": lambda t: block(t, "rollover", "cumulative_roll_deg"),
        "quarter_turns": lambda t: block(t, "rollover", "quarter_turns"),
        "roll_onset_time": lambda t: block(t, "rollover", "roll_onset_s"),
        "first_contact_time": lambda t: block(t, "bounce", "first_contact_s"),
        "first_apex_height": lambda t: block(t, "bounce", "apex_heights_m", 0),
        "first_rebound_ratio": lambda t: block(t, "bounce", "rebound_ratios", 0),
        "bounce_count": lambda t: block(t, "bounce", "bounces"),
        "rest_time": lambda t: block(t, "bounce", "rest_s"),
        "trajectory_rms": lambda t: phase_metric(t, "position_rms_m"),
        "attitude_rms": lambda t: phase_metric(t, "attitude_rms_deg"),
        "silhouette_iou": lambda t: phase_metric(t, "silhouette_iou_mean"),
    }


#: Target metrics :func:`score_worlds` can evaluate.
TARGET_METRICS = tuple(sorted(_run_value_lookup({"objects": {}, "phases": [], "events": [], "fps": 1.0})))


def _evaluate_targets(report: Mapping[str, Any], scene: Scene | None) -> list[dict[str, Any]]:
    if scene is None:
        return []
    lookup = _run_value_lookup(report)
    out = []
    for t in scene.targets:
        row: dict[str, Any] = {"metric": t.metric, "actor": t.actor, "event": t.event, "phase": t.phase,
                               "value": t.value.value, "sigma": t.value.sigma, "source": t.source}
        if t.citation:
            row["citation"] = t.citation
        if t.metric not in lookup:
            row.update({"status": "unsupported", "run": None, "z": None, "pass": None})
        else:
            ref, run = lookup[t.metric](t)
            run = _f(run) if not isinstance(run, bool) else None
            row["reference"] = _f(ref) if isinstance(ref, (int, float)) and not isinstance(ref, bool) else None
            row["run"] = run
            if run is None:
                row.update({"status": "not_measured", "z": None, "pass": False})
            else:
                z = t.value.z(run)
                row["z"] = z
                if z is None:
                    row["pass"] = bool(abs(run - t.value.value) <= 1e-9)
                else:
                    row["pass"] = bool(abs(z) <= t.tolerance_sigma)
                row["status"] = "measured"
        out.append(row)
    return out


# ----------------------------------------------------------------------------- main entry

def score_worlds(reference: World, run: World, scene: Scene | None = None,
                 run_info: Mapping[str, Any] | None = None, options: ScoreOptions = ScoreOptions(),
                 inputs: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Compare ``run`` with ``reference`` and return a ``v2e.score/1`` report."""
    if run.frame_count != reference.frame_count or abs(run.fps - reference.fps) > 1e-9:
        raise ValueError("the run must be exported onto the reference timeline (same fps and frame count); "
                         "use openbeam_v2e.world.export.runlog_to_world")
    fps, F = reference.fps, reference.frame_count
    warnings: list[str] = []
    if reference.kind == "run_export":
        warnings.append("the reference is itself an engine run: this comparison supports no accuracy claim")
    if reference.kind == "synthetic_truth":
        warnings.append("synthetic self-consistency (engine-rendered truth): not an accuracy claim")

    pairs = _object_ids(reference, scene)
    names = _world_names(scene)
    ref_detected = detect_events(reference, options.contact)
    run_detected = detect_events(run, options.contact)
    rows = _event_rows(reference, run, scene, run_detected, names)

    boundaries: dict[str, float | None] = {"start": 0.0, "end": float(F - 1)}
    for r in rows:
        boundaries[r.name] = r.ref_centre
    if "rest" not in boundaries:
        boundaries["rest"] = _rest_boundary(reference, [oid for _, oid in pairs], options.contact)
    windows = phase_windows(scene.phases if scene is not None else (), boundaries, F)

    report: dict[str, Any] = {
        "schema": SCORE_SCHEMA_ID,
        "scorer_version": __version__,
        "inputs": dict(inputs or {}),
        "fps": float(fps),
        "frame_count": int(F),
        "claim": claim_block(run_info, scene.requires.effects if scene is not None else ()),
        "alignment": {},
        "phases": [],
        "events": [],
        "objects": {},
        "targets": [],
        "summary": {"objective": 0.0},
        "warnings": warnings,
    }
    terms: list[float] = []

    # events
    for r in rows:
        delta = None if r.run_frame is None else r.run_frame - r.ref_centre
        within = None if r.run_frame is None else bool(r.ref_lo - 0.5 <= r.run_frame <= r.ref_hi + 0.5)
        z = None if delta is None else delta / r.sigma
        report["events"].append({
            "name": r.name, "kind": r.kind, "actors": list(r.actors), "reference_frame": [r.ref_lo, r.ref_hi],
            "reference_source": r.ref_source, "run_frame": r.run_frame, "delta_frames": delta,
            "delta_s": None if delta is None else delta / fps, "sigma_frames": r.sigma, "z": z,
            "within_sigma": within})
        terms.append(MISSING_PENALTY if delta is None else (delta / max(r.sigma, options.event_floor_frames)) ** 2)

    # phases
    for w in windows:
        entry: dict[str, Any] = {"name": w["name"], "frames": w["frames"] or [0, -1], "objects": {}}
        if w["frames"] is None:
            report["warnings"].append(f"phase {w['name']!r} skipped: {w['reason']}")
            report["phases"].append(entry)
            continue
        frames = range(w["frames"][0], w["frames"][1] + 1)
        for actor, oid in pairs:
            ref_obj = reference.object(oid)
            run_obj = run.object(oid) if run.has_object(oid) else None
            m = trajectory_metrics(ref_obj, run_obj, fps, frames)
            if options.silhouettes:
                sil = silhouette_scores(reference, run, oid, frames, options.silhouette_width_px,
                                        options.silhouette_max_frames)
                if sil.get("frames"):
                    m["silhouette_iou_mean"] = sil["iou_mean"]
                    m["silhouette_iou_over_shape_limit"] = sil["iou_over_shape_limit_mean"]
                    m["silhouette"] = sil
            entry["objects"][actor] = m
            if m.get("reference_frames"):
                sp = _median_sigma(ref_obj.sigma_position_m[list(frames)], options.position_floor_m)
                sa = math.degrees(_median_sigma(ref_obj.sigma_attitude_rad[list(frames)],
                                                math.radians(options.attitude_floor_deg)))
                if m.get("position_rms_m") is not None:
                    terms.append((m["position_rms_m"] / sp) ** 2)
                if m.get("attitude_rms_deg") is not None:
                    terms.append((m["attitude_rms_deg"] / sa) ** 2)
                cov = m.get("coverage")
                if cov is not None and cov < 1.0:
                    terms.append((10.0 * (1.0 - cov)) ** 2)
        report["phases"].append(entry)

    # objects: values at events, after-event measures, rollover and bounce
    for actor, oid in pairs:
        ref_obj = reference.object(oid)
        run_obj = run.object(oid) if run.has_object(oid) else None
        block: dict[str, Any] = {"object": oid, "class": ref_obj.cls, "symmetry": ref_obj.symmetry_class,
                                 "in_run": run_obj is not None, "at_events": {}, "after_events": {}}
        ref_rest = rest_frame(ref_obj, fps, options.contact)
        run_rest = rest_frame(run_obj, fps, options.contact) if run_obj is not None else None
        for r in rows:
            if r.actors and oid not in r.actors:
                continue
            ref_state = state_at(ref_obj, fps, r.ref_centre)
            run_frame = r.run_frame if r.run_frame is not None else r.ref_centre
            run_state = state_at(run_obj, fps, run_frame) if run_obj is not None else None
            at = {"reference": ref_state, "run": run_state}
            if ref_state and run_state:
                at["position_error_m"] = float(np.linalg.norm(np.subtract(run_state["position_m"],
                                                                          ref_state["position_m"])))
                at["speed_delta_mps"] = _delta(ref_state["speed_mps"], run_state["speed_mps"])
                err = attitude_error(ref_obj.rotations()[[ref_state["frame"]]], run_obj.rotations()[[run_state["frame"]]],
                                     ref_obj.symmetry_class, ref_obj.hull_extents())[0]
                at["attitude_error_deg"] = math.degrees(err) if np.isfinite(err) else None
            block["at_events"][r.name] = at
            if r.kind == "contact":
                ref_after = after_event(ref_obj, fps, int(round(r.ref_centre)), ref_rest)
                run_after = after_event(run_obj, fps, int(round(run_frame)), run_rest) if run_obj is not None else {}
                block["after_events"][r.name] = _diff_dict(ref_after, run_after)
        if ref_obj.cls.startswith("vehicle."):
            block["rollover"] = _diff_dict(rollover(ref_obj, fps), rollover(run_obj, fps) if run_obj is not None else {})
        if ref_obj.cls.startswith("prop."):
            ref_b = bounce(ref_obj, fps, ground_gap(ref_obj, reference), options.contact.tolerance_m, ref_rest)
            run_b = bounce(run_obj, fps, ground_gap(run_obj, run), options.contact.tolerance_m, run_rest) \
                if run_obj is not None else {}
            block["bounce"] = _diff_dict(ref_b, run_b)
        block["rest_frame"] = {"reference": ref_rest, "run": run_rest}
        if run_obj is None:
            terms.append(MISSING_PENALTY)
        report["objects"][actor] = block

    # frame-0 alignment of actors that drive in
    if scene is not None:
        for a in scene.dynamic_actors:
            state = a.initial_state
            if state is None or state.mode != "drive_in" or not reference.has_object(a.object_id):
                continue
            ref_obj = reference.object(a.object_id)
            run_obj = run.object(a.object_id) if run.has_object(a.object_id) else None
            if run_obj is None or not (ref_obj.known[0] and run_obj.known[0]):
                report["alignment"][a.id] = {"available": False}
                continue
            pos = float(np.linalg.norm(run_obj.centre_track()[0] - ref_obj.centre_track()[0]))
            heading = math.degrees(yaw_of(run_obj.rotations()[0]) - yaw_of(ref_obj.rotations()[0]))
            heading = (heading + 180.0) % 360.0 - 180.0
            target_speed = state.speed_mps.value if state.speed_mps is not None else _f(ref_obj.speed(fps)[0])
            run_speed = _f(run_obj.speed(fps)[0])
            report["alignment"][a.id] = {"available": True, "position_m": pos, "heading_deg": heading,
                                         "speed_mps": _delta(target_speed, run_speed)}

    report["targets"] = _evaluate_targets(report, scene)
    for t in report["targets"]:
        if t.get("z") is not None:
            terms.append(float(t["z"]) ** 2)

    # summary
    pos = [o.get("position_rms_m") for p in report["phases"] for o in p["objects"].values() if o.get("position_rms_m") is not None]
    att = [o.get("attitude_rms_deg") for p in report["phases"] for o in p["objects"].values() if o.get("attitude_rms_deg") is not None]
    iou = [o.get("silhouette_iou_mean") for p in report["phases"] for o in p["objects"].values() if o.get("silhouette_iou_mean") is not None]
    timed = [e for e in report["events"] if e["within_sigma"] is not None]
    report["summary"] = {
        "objective": float(np.mean(terms)) if terms else 0.0,
        "terms": len(terms),
        "position_rms_m_worst": max(pos) if pos else None,
        "attitude_rms_deg_worst": max(att) if att else None,
        "silhouette_iou_mean": float(np.mean(iou)) if iou else None,
        "events_found": sum(1 for e in report["events"] if e["run_frame"] is not None),
        "events_total": len(report["events"]),
        "events_within_sigma": sum(1 for e in timed if e["within_sigma"]),
        "targets_passed": sum(1 for t in report["targets"] if t.get("pass")),
        "targets_total": len(report["targets"]),
        "detected_reference_events": len(ref_detected),
        "detected_run_events": len(run_detected),
    }
    check(jsonio.loads(jsonio.dumps(report)), SCORE_SCHEMA, "score report")
    return report


def _run_from_directory(run_dir: Path, reference: World) -> tuple[World, dict[str, Any]] | None:
    """A connector run directory (``run.json`` + rows): export its rows and collect the run record."""
    if not (run_dir / "run.json").is_file():
        return None
    from openbeam_v2e.backends.harness import load_run  # optional: only for connector run directories
    from openbeam_v2e.schema.runlog import read_runlog
    from openbeam_v2e.world.export import runlog_to_world

    result = load_run(run_dir)
    rows = read_runlog(result.rows)
    world = runlog_to_world(rows, reference, t0_s=result.t0_s)
    effects = [jsonio.loads(jsonio.dumps(_plain(e))) for e in result.effects]
    content = [jsonio.loads(jsonio.dumps(_plain(c))) for c in result.content]
    info = {"effects": effects, "content": content, "claim_class": result.claim_class(),
            "provenance_signature": result.provenance_signature(), "valid": bool(result.valid())}
    return world, info


def _plain(value: Any) -> Any:
    import dataclasses
    import enum

    if dataclasses.is_dataclass(value):
        return {f.name: _plain(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, Path):
        return value.as_posix()
    return value


def score_run(run: str | Path | World, reference: str | Path | World | None = None,
              scene: str | Path | Scene | None = None, out: str | Path | None = None,
              run_info: Mapping[str, Any] | None = None, options: ScoreOptions = ScoreOptions()) -> dict[str, Any]:
    """Score a run (world package directory, connector run directory or :class:`World`).

    ``reference`` defaults to the scene's world package. The report is written to ``out`` (a
    ``.json`` path, or a directory that receives ``score.json``) when given.
    """
    from openbeam_v2e.world.io import read_world

    scene_obj = scene if isinstance(scene, Scene) or scene is None else load_scene(scene)
    if reference is None:
        if scene_obj is None:
            raise ValueError("score_run needs a reference world or a scene that names one")
        reference = scene_obj.world_path()
    ref_world = reference if isinstance(reference, World) else read_world(reference)
    info = dict(run_info) if run_info else None
    if isinstance(run, World):
        run_world = run
    else:
        loaded = _run_from_directory(Path(run), ref_world)
        if loaded is not None:
            run_world, found = loaded
            info = info or found
        else:
            run_world = read_world(run)
    inputs = {"run": str(run) if not isinstance(run, World) else "<memory>",
              "reference": str(reference) if not isinstance(reference, World) else "<memory>",
              "scene_sha256": scene_obj.sha256() if scene_obj is not None else None,
              "reference_kind": ref_world.kind, "run_kind": run_world.kind}
    report = score_worlds(ref_world, run_world, scene_obj, info, options, inputs)
    if out is not None:
        target = Path(out)
        if target.suffix != ".json":
            target = target / "score.json"
        jsonio.dump(report, target)
    return report
