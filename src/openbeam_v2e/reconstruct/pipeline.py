# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""The reconstruction pipeline: video in, world package with uncertainty out.

Stages, each run by whichever perception model the configuration names
(:mod:`openbeam_v2e.reconstruct.clip_config`):

1. **calibration** of the reference frame (intrinsics, distortion, camera position and attitude);
2. **camera tracking**: pan, tilt, roll and zoom per frame from the fixed position, forwards and
   backwards from the reference frame;
3. **segmentation**: masks per object per frame, prompted by the annotations;
4. **depth** (optional): per-frame depth maps, aligned to metres on the ground plane;
5. **pose**: a rigid pose track with covariance and measured dimensions per object, with some
   observed frames held out of the fit for honest residuals;
6. **physics and events**: ballistic (airborne) segments refitted ballistically, then events (rest,
   take-off, touchdown, impact) as frame intervals.

The result is written as a world package (:func:`openbeam_v2e.world.write_world`: the
4DCodeBench layout plus the ``v2e/`` sidecars) and ``v2e/reconstruct_report.json`` with the full
detail (covariances, calibration, per-frame fits, provenance of every model).

:func:`reconstruct` is the entry point used by the CLI and the agent tools.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from openbeam_v2e import __version__
from openbeam_v2e.perception import registry
from openbeam_v2e.perception.interfaces import CalibrationInputs, PoseContext
from openbeam_v2e.perception.types import (
    CameraTrack,
    DepthMap,
    ModelProvenance,
    ObjectPrompt,
    PoseTrack,
    Segmentation,
)
from openbeam_v2e.reconstruct import cache as cache_mod
from openbeam_v2e.reconstruct.annotate import Annotations, load_annotations, parse_annotations
from openbeam_v2e.reconstruct.calibrate import CAMERA_PARAMS, CalibrationResult
from openbeam_v2e.reconstruct.clip_config import ReconstructionConfig
from openbeam_v2e.reconstruct.dynamics_fit import apply_ballistic
from openbeam_v2e.reconstruct.events import Event, EventOptions, airborne_segments, detect_events, name_unique
from openbeam_v2e.reconstruct.geom import Camera, Intrinsics
from openbeam_v2e.reconstruct.geom import quat_to_matrix as geom_quat_to_matrix
from openbeam_v2e.reconstruct.hulls import make_hull
from openbeam_v2e.reconstruct.objects import object_spec
from openbeam_v2e.reconstruct.pose_fit import make_observation, outline_iou
from openbeam_v2e.reconstruct.silhouette import mask_to_polygons
from openbeam_v2e.reconstruct.video import FrameSource, open_frames, write_png

RESULT_SCHEMA = "v2e.reconstruction/1"
SUMMARY_SCHEMA = "v2e.reconstruct-summary/1"
ASSUMED_CAMERA_HEIGHT_M = (1.6, 0.5)   # used only when the configuration gives no scale at all


class _Window:
    """A frame source restricted to ``frames`` and renumbered 0..n-1 (stride and offset)."""

    def __init__(self, src: FrameSource, frames: Sequence[int], fps: float) -> None:
        self.src = src
        self.frames = list(frames)
        self.fps = fps
        self.path = getattr(src, "path", None)
        self.files = [src.files[f] for f in self.frames] if getattr(src, "files", None) else None

    def __len__(self) -> int:
        return len(self.frames)

    @property
    def size(self) -> tuple[int, int]:
        return self.src.size

    def read(self, index: int) -> np.ndarray:
        return self.src.read(self.frames[index])


@dataclass
class Reconstruction:
    """Everything the pipeline produced, before it is written."""

    fps: float
    frame_count: int
    image_size: tuple[int, int]
    video_frames: list[int]
    calibration: CalibrationResult
    camera_track: CameraTrack
    segmentation: Segmentation
    tracks: dict[str, PoseTrack]
    events: list[Event]
    models: list[ModelProvenance]
    config: dict[str, Any]
    source_video: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)
    cache_use: dict[str, str] = field(default_factory=dict)
    ground_z: float = 0.0

    # ---------------------------------------------------------------- world package
    def to_world(self) -> Any:
        """The reconstruction as a :class:`openbeam_v2e.world.World`."""
        from openbeam_v2e.schema.common import Measured
        from openbeam_v2e.world.camera import Camera as WCamera
        from openbeam_v2e.world.camera import CameraTrack as WTrack
        from openbeam_v2e.world.model import EventInterval, ObjectTrack, StaticMesh, World

        F = self.frame_count
        cams = []
        for f in range(F):
            c = self.camera_track.camera(f)
            i = c.intrinsics
            cams.append(WCamera(i.width, i.height, i.fx, i.fy, i.cx, i.cy, c.R_cw.T.copy(), c.centre.copy(),
                                i.k1, i.k2))
        wtrack = WTrack.from_cameras(cams)
        objects = []
        residuals: dict[str, dict[str, float]] = {"fitted": {}, "held_out": {}}
        outlines: dict[str, dict[int, list[np.ndarray]]] = {}
        for oid, tr in self.tracks.items():
            pos = np.full((F, 3), np.nan)
            quat = np.full((F, 4), np.nan)
            sp = np.full(F, np.nan)
            sa = np.full(F, np.nan)
            vis = np.zeros(F, bool)
            idx = tr.frames.astype(int)
            pos[idx] = tr.position
            quat[idx] = tr.quat
            sp[idx] = np.linalg.norm(tr.sigma_position(), axis=1)
            sa[idx] = np.radians(np.linalg.norm(tr.sigma_rotation_deg(), axis=1))
            vis[idx] = tr.observed
            hull = make_hull(tr.template, tr.dims, **tr.template_params)
            verts, faces = _hull_mesh(hull)
            # world packages put the body origin at the hull's bounding-box centre (the connectors'
            # reference point); fits use the centre of the bottom face, so shift both
            centre = 0.5 * (verts.min(axis=0) + verts.max(axis=0))
            verts = verts - centre
            R_all = np.array([geom_quat_to_matrix(q) for q in tr.quat])
            pos[idx] = tr.position + R_all @ centre
            names = ("length", "width", "height")
            dims = {n: Measured(float(tr.dims[k]), float(tr.dims_sigma[k])) for k, n in enumerate(names)}
            ann = {"source": "reconstruction", "template": tr.template, "motion": tr.motion,
                   "body_origin": "hull bounding-box centre",
                   "models": [p.model for p in tr.provenance]}
            objects.append(ObjectTrack(oid, tr.object_class, verts, faces, pos, quat, vis, sp, sa, dims,
                                       tr.symmetry, None, None, ann))
            fitted = tr.iou[tr.fitted & np.isfinite(tr.iou)]
            held = tr.iou[tr.heldout & np.isfinite(tr.iou)]
            if len(fitted):
                residuals["fitted"][f"{oid}.silhouette_iou_median"] = float(np.median(fitted))
            if len(held):
                residuals["held_out"][f"{oid}.silhouette_iou_median"] = float(np.median(held))
            mt = self.segmentation.tracks.get(oid)
            if mt is not None:
                outlines[oid] = {}
                for f in mt.frames():
                    if f < F:
                        polys = mask_to_polygons(mt.mask(f), tolerance_px=0.9)
                        if polys:
                            outlines[oid][int(f)] = polys
        if np.isfinite(self.calibration.rms_px):
            residuals["fitted"]["camera.reprojection_rms_px"] = float(self.calibration.rms_px)
        statics = [StaticMesh(1, *self._ground_quad(), name="ground", role="drivable",
                              sigma_m=self._ground_sigma(), observed=True)]
        cam_sigma = self.camera_track.focal * np.sqrt(self.camera_track.sigma[:, 0] ** 2 +
                                                      self.camera_track.sigma[:, 1] ** 2)
        events = [EventInterval(e.name, e.frames, e.kind, tuple(e.actors), e.evidence) for e in self.events]
        models = [_schema_provenance(p) for p in self.models]
        return World(self.fps, F, wtrack, statics, objects, events, "reconstruction", models, residuals,
                     cam_sigma, outlines, dict(self.source_video),
                     f"openbeam-v2e {__version__} reconstruction; video frames "
                     f"{self.video_frames[0]}..{self.video_frames[-1]} step "
                     f"{(self.video_frames[1] - self.video_frames[0]) if len(self.video_frames) > 1 else 1}")

    def _ground_quad(self) -> tuple[np.ndarray, np.ndarray]:
        pts = [tr.position[:, :2] for tr in self.tracks.values() if len(tr.position)]
        if pts:
            P = np.concatenate(pts)
            lo, hi = P.min(axis=0) - 20.0, P.max(axis=0) + 20.0
        else:
            c = self.camera_track.centre[:2]
            lo, hi = c - 50.0, c + 50.0
        z = self.ground_z
        v = np.array([[lo[0], lo[1], z], [hi[0], lo[1], z], [hi[0], hi[1], z], [lo[0], hi[1], z]])
        return v, np.array([[0, 1, 2], [0, 2, 3]])

    def _ground_sigma(self) -> float:
        cov = self.camera_track.calibration_cov
        return float(math.sqrt(max(cov[5, 5], 0.0))) if cov is not None else 0.0

    # ---------------------------------------------------------------- full detail
    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": RESULT_SCHEMA,
            "producer": {"package": "openbeam-v2e", "version": __version__},
            "fps": self.fps, "frame_count": self.frame_count, "image_size": list(self.image_size),
            "video_frames": [int(f) for f in self.video_frames],
            "ground_z": self.ground_z,
            "source_video": dict(self.source_video),
            "calibration": {**self.calibration.to_dict(),
                            "cov": None if self.calibration.cov is None else np.asarray(self.calibration.cov).tolist(),
                            "cov_names": list(CAMERA_PARAMS)},
            "camera_track": self.camera_track.to_dict(),
            "objects": {oid: tr.to_dict() for oid, tr in self.tracks.items()},
            "events": [e.to_dict() for e in self.events],
            "segmentation": {oid: {"frames": len(t.crops), "provenance": [p.to_dict() for p in t.provenance]}
                             for oid, t in self.segmentation.tracks.items()},
            "models": [p.to_dict() for p in self.models],
            "config": self.config,
            "warnings": list(self.warnings),
            "timings_s": dict(self.timings),
            "cache": dict(self.cache_use),
        }

    def summary(self, out: Path | None = None) -> dict[str, Any]:
        cal = self.calibration
        return {
            "schema": SUMMARY_SCHEMA,
            "out": None if out is None else str(out),
            "frames": self.frame_count, "fps": self.fps, "image_size": list(self.image_size),
            "camera": {"method": cal.method, "reprojection_rms_px": None if not np.isfinite(cal.rms_px) else cal.rms_px,
                       "focal_px": [float(self.camera_track.focal.min()), float(self.camera_track.focal.max())],
                       "centre_m": cal.camera.centre.tolist(),
                       "sigmas": {k: v for k, v in cal.sigmas.items() if v}},
            "objects": [tr.summary() for tr in self.tracks.values()],
            "events": [e.to_dict() for e in self.events],
            "models": sorted({f"{p.kind}:{p.model}" for p in self.models}),
            "warnings": list(self.warnings),
            "timings_s": {k: round(v, 3) for k, v in self.timings.items()},
        }

    def write(self, out_dir: str | Path) -> Path:
        """Write the world package and ``v2e/reconstruct_report.json``."""
        from openbeam_v2e.schema import jsonio
        from openbeam_v2e.world.io import write_world

        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        write_world(self.to_world(), out)
        jsonio.dump(_json_safe(self.to_dict()), out / "v2e" / "reconstruct_report.json")
        if self.config.get("output", {}).get("masks"):
            for oid, t in self.segmentation.tracks.items():
                d = out / "v2e" / "masks" / oid
                d.mkdir(parents=True, exist_ok=True)
                for f in t.frames():
                    write_png(d / f"{f:06d}.png", t.mask(f))
        return out


def _json_safe(o: Any) -> Any:
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, Mapping):
        return {str(k): _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, np.ndarray):
        return _json_safe(o.tolist())
    if isinstance(o, (np.floating, np.integer)):
        return _json_safe(o.item())
    if isinstance(o, Path):
        return str(o)
    return o


def _hull_mesh(hull: Any) -> tuple[np.ndarray, np.ndarray]:
    verts, faces, off = [], [], 0
    for part in hull.parts:
        v, f = part.triangles()
        verts.append(v)
        faces.append(f + off)
        off += len(v)
    return np.concatenate(verts), np.concatenate(faces)


def _schema_provenance(p: ModelProvenance) -> dict[str, Any]:
    """The ``v2e.world/1`` provenance shape (id, version, task, weights hash, parameters)."""
    params = dict(_json_safe(dict(p.params)))
    params["implementation"] = p.implementation
    if p.weights:
        params["weights"] = p.weights
    if p.libraries:
        params["libraries"] = dict(p.libraries)
    return {"id": p.model, "version": p.version, "task": p.kind, "weights_sha256": p.weights_sha256,
            "parameters": params}


# --------------------------------------------------------------------------- stages

def _calibrate(cfg: ReconstructionConfig, video: _Window, ref: int, exclude: np.ndarray | None,
               warnings: list[str]) -> tuple[CalibrationResult, list[ModelProvenance]]:
    cal = cfg.calibration
    w, h = video.size
    if "camera" in cal:
        cam = Camera.from_dict(cal["camera"])
        sig = dict(cal.get("sigmas", {}))
        cov = np.diag([(math.radians(sig.get(n, 0.0)) if n in ("pan", "tilt", "roll") else sig.get(n, 0.0)) ** 2
                       for n in CAMERA_PARAMS])
        res = CalibrationResult(cam, {n: float(sig.get(n, 0.0)) for n in CAMERA_PARAMS}, float("nan"),
                                np.zeros((0, 2)), "given", 0, cov)
        prov = ModelProvenance("given", "camera_calibration", __version__, params={"sigmas": sig})
        return res, [prov]
    stage = cfg.calibration_stage
    assert stage is not None
    intr = None
    if cal.get("intrinsics"):
        d = dict(cal["intrinsics"])
        d.setdefault("width", w)
        d.setdefault("height", h)
        d.setdefault("cx", w / 2.0)
        d.setdefault("cy", h / 2.0)
        intr = Intrinsics.from_dict(d)
    pts = cal.get("points") or []
    world = np.array([p["world"] for p in pts], float) if pts else None
    image = np.array([p["image"] for p in pts], float) if pts else None
    kd = cal.get("known_distance")
    height = cal.get("camera_height_m")
    if stage.model in ("baseline.auto", "baseline.vanishing") and height is None and kd is None:
        height = ASSUMED_CAMERA_HEIGHT_M[0]
        warnings.append(f"no scale given for the calibration: camera height assumed {ASSUMED_CAMERA_HEIGHT_M[0]} m "
                        f"(+-{ASSUMED_CAMERA_HEIGHT_M[1]} m); metric results scale with it. Give camera.calibration "
                        "points, camera_height_m or known_distance for metric output.")
    inputs = CalibrationInputs(
        w, h, image=video.read(ref), world_points=world, image_points=image,
        point_sigma_px=float(cal.get("sigma_px", 1.0)),
        lines_x=None if cal.get("lines_x") is None else np.asarray(cal["lines_x"], float),
        lines_y=None if cal.get("lines_y") is None else np.asarray(cal["lines_y"], float),
        camera_height_m=None if height is None else float(height),
        known_distance=None if kd is None else (kd["a"], kd["b"], float(kd["m"])),
        origin_pixel=cal.get("origin_pixel"), intrinsics=intr, ground_z=cfg.ground_z,
        free_distortion=bool(cal.get("free_distortion", False)), exclude_mask=exclude)
    model = registry.create("camera_calibration", stage.model, **stage.params)
    res = model.calibrate(inputs)
    if height is not None and cal.get("camera_height_m") is None and kd is None and res.cov is not None:
        # an assumed height: its uncertainty scales the whole scene
        k = CAMERA_PARAMS.index("cz_w")
        res.cov[k, k] = max(res.cov[k, k], ASSUMED_CAMERA_HEIGHT_M[1] ** 2)
        res.sigmas["cz_w"] = math.sqrt(res.cov[k, k])
    return res, [model.provenance()]


def _track_camera(cfg: ReconstructionConfig, video: _Window, ref: int, ref_cam: Camera,
                  exclude: Callable[[int], np.ndarray | None] | None) -> CameraTrack:
    stage = cfg.tracking
    assert stage is not None
    model = registry.create("camera_tracking", stage.model, **stage.params)
    n = len(video)
    fwd = model.track(video, ref_cam, list(range(ref, n)), exclude)
    if ref == 0:
        return fwd
    back = model.track(video, ref_cam, list(range(ref, -1, -1)), exclude)
    order = list(range(0, n))

    def col(name: str) -> np.ndarray:
        b = getattr(back, name)[::-1][:-1]
        return np.concatenate([b, getattr(fwd, name)])

    sigma = np.concatenate([back.sigma[::-1][:-1], fwd.sigma])
    quality = None
    if fwd.quality is not None and back.quality is not None:
        quality = np.concatenate([np.asarray(back.quality)[::-1][:-1], np.asarray(fwd.quality)])
    return CameraTrack(fwd.intrinsics, fwd.centre, order, col("pan"), col("tilt"), col("roll"), col("focal"),
                       sigma, quality=quality, provenance=fwd.provenance)


def _prompts_for(annotations: Annotations, frames: Sequence[int], video_frames: list[int]) -> list[ObjectPrompt]:
    """Annotation prompts renumbered from video frames to window frames."""
    index = {vf: i for i, vf in enumerate(video_frames)}
    out = []
    for p in annotations.prompts:
        if p.frame in index:
            out.append(ObjectPrompt(p.object_id, index[p.frame], p.box, p.polygon, p.mask, p.points, p.labels))
    return out


def _segmentation_from_cache(raw: Mapping[str, np.ndarray], w: int, h: int,
                             prov: list[ModelProvenance]) -> Segmentation:
    seg = Segmentation(w, h, provenance=prov)
    ids = [str(x) for x in raw["ids"]]
    for k, oid in enumerate(ids):
        t = seg.track(oid)
        frames = raw[f"o{k}_frames"]
        boxes = raw[f"o{k}_boxes"]
        bits = raw[f"o{k}_bits"]
        conf = raw[f"o{k}_conf"]
        off = 0
        for f, (x0, y0, x1, y1), c in zip(frames, boxes, conf):
            n = int((x1 - x0) * (y1 - y0))
            m = np.unpackbits(bits[off // 8: (off + n + 7) // 8 + 1])[off % 8: off % 8 + n].astype(bool)
            off += ((n + 7) // 8) * 8
            full = np.zeros((h, w), bool)
            full[y0:y1, x0:x1] = m.reshape(y1 - y0, x1 - x0)
            t.set(int(f), full, float(c))
    return seg


def _segmentation_to_cache(seg: Segmentation) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {"ids": np.array(list(seg.tracks), dtype=str)}
    for k, (oid, t) in enumerate(seg.tracks.items()):
        frames = t.frames()
        boxes, chunks, conf = [], [], []
        for f in frames:
            c = t.crops[f]
            boxes.append(c.bbox())
            m = c.mask.ravel()
            pad = (-len(m)) % 8
            chunks.append(np.packbits(np.r_[m, np.zeros(pad, bool)]))
            conf.append(t.confidence.get(f, 1.0))
        out[f"o{k}_frames"] = np.array(frames, np.int64)
        out[f"o{k}_boxes"] = np.array(boxes, np.int64).reshape(-1, 4)
        out[f"o{k}_bits"] = np.concatenate(chunks) if chunks else np.zeros(0, np.uint8)
        out[f"o{k}_conf"] = np.array(conf, float)
    return out


def _holdout(cfg: ReconstructionConfig, observed: Sequence[int]) -> list[int]:
    if cfg.holdout_frames is not None:
        return [f for f in cfg.holdout_frames if f in set(observed)]
    if cfg.holdout_every <= 1:
        return []
    obs = sorted(observed)
    return [f for i, f in enumerate(obs) if i % cfg.holdout_every == cfg.holdout_offset % cfg.holdout_every]


def _physics_and_events(tracks: Mapping[str, PoseTrack], segmentation: Segmentation, cameras: CameraTrack,
                        fps: float, cfg: ReconstructionConfig) -> list[Event]:
    opts = EventOptions(**{k: v for k, v in cfg.events.items() if k in EventOptions.__dataclass_fields__})
    events: list[Event] = []
    for oid, track in tracks.items():
        if cfg.ballistic and track.motion == "free":
            # Missing event pairs are NOT evidence that the entire clip is gravity-only motion.
            # Clearance can establish a candidate interval, never the absence of other forces.
            candidates = airborne_segments(track, fps, cfg.ground_z, opts)
            track.diagnostics["ballistic_selection"] = {
                "candidate_segments": [list(bounds) for bounds in candidates],
                "evidence": "sustained hull clearance above the height and position-sigma thresholds",
                "limit": "clearance is not proof of gravity-only motion or calibrated contact uncertainty",
            }
            before_iou = track.iou.copy()
            applied = False
            for start, end in candidates:
                try:
                    fit = apply_ballistic(track, start, end, fps, max_chi2_per_dof=cfg.ballistic_max_chi2,
                                          ground_z=cfg.ground_z)
                except (ValueError, np.linalg.LinAlgError) as exc:
                    track.diagnostics.setdefault("ballistic_segments", []).append(
                        {"frames": [start, end], "applied": False, "reason": str(exc)})
                    continue
                applied |= fit.applied
            if applied:
                track.diagnostics["pre_ballistic_iou"] = {
                    int(f): float(value) for f, value in zip(track.frames, before_iou) if np.isfinite(value)}
                hull = make_hull(track.template, track.dims, **track.template_params)
                for i, frame in enumerate(track.frames):
                    if not track.observed[i]:
                        continue
                    mask = segmentation.tracks[oid].mask(int(frame))
                    obs = None if mask is None else make_observation(int(frame), cameras.camera(int(frame)), mask)
                    # Held-out masks score the final prediction only; they never select a refinement.
                    track.iou[i] = np.nan if obs is None else outline_iou(hull, track.rotation(int(frame)),
                                                                        track.position[i], obs)
        events.extend(detect_events(track, fps, cfg.ground_z, opts))
    return name_unique(events)


# --------------------------------------------------------------------------- entry points

def run_pipeline(cfg: ReconstructionConfig, video: FrameSource | None = None,
                 log: Callable[[str], None] | None = None) -> Reconstruction:
    """Run every stage for a parsed configuration and return the in-memory result."""
    say = log or (lambda m: None)
    timings: dict[str, float] = {}
    warnings: list[str] = []
    t_all = time.perf_counter()
    if video is None:
        if cfg.video_path is None:
            raise ValueError("no video given (pass video= or set video.path in the configuration)")
        video = open_frames(cfg.video_path, cfg.fps)
    fps_src = float(cfg.fps or getattr(video, "fps", 30.0))
    start, end = cfg.frame_range
    end = len(video) if end is None else min(end, len(video))
    video_frames = list(range(start, end, cfg.stride))
    if len(video_frames) < 3:
        raise ValueError(f"the frame range {start}..{end} step {cfg.stride} has fewer than three frames")
    fps = fps_src / cfg.stride
    win = _Window(video, video_frames, fps)
    w, h = win.size
    stage_cache = cache_mod.StageCache(cfg.cache, say)
    vfp = cache_mod.video_fingerprint(win) if cfg.cache else ""

    # annotations
    ann = Annotations()
    if cfg.annotations is not None:
        ann = load_annotations(cfg.annotations) if isinstance(cfg.annotations, Path) else \
            parse_annotations(cfg.annotations, cfg.base_dir)
    prompts = _prompts_for(ann, range(len(win)), video_frames)
    det_prov: list[ModelProvenance] = []
    if not prompts and cfg.detection is not None:
        from openbeam_v2e.perception.adapters.hf_detection import detections_to_prompts

        detector = registry.create("detection", cfg.detection.model, **cfg.detection.params)
        det_frame = video_frames.index(int(cfg.calibration.get("frame", video_frames[0])))
        dets = detector.detect(win.read(det_frame), cfg.detection_classes or None)
        prompts, det_classes = detections_to_prompts(dets, det_frame)
        for oid, c in det_classes.items():
            ann.classes.setdefault(oid, c)
        det_prov = [detector.provenance()]
        say(f"detection: {len(prompts)} objects on frame {video_frames[det_frame]}")
    fit_ids = [o.id for o in cfg.objects] or list(dict.fromkeys(ann.objects() + [p.object_id for p in prompts]))

    # 1. calibration
    t0 = time.perf_counter()
    ref_video = int(cfg.calibration.get("frame", video_frames[0]))
    if ref_video not in video_frames:
        raise ValueError(f"calibration frame {ref_video} is not in the reconstructed frame range")
    ref = video_frames.index(ref_video)
    ref_exclude = None
    ref_prompts = [p.region_mask(w, h) for p in prompts if p.frame == ref and p.region_mask(w, h) is not None]
    if ref_prompts:
        ref_exclude = np.logical_or.reduce(ref_prompts)
    calib, cal_prov = _calibrate(cfg, win, ref, ref_exclude, warnings)
    timings["calibration"] = time.perf_counter() - t0
    say(f"calibration: {calib.method}, rms {calib.rms_px:.2f} px, {calib.camera}")

    # 2. camera tracking (cached)
    t0 = time.perf_counter()
    prompt_masks: dict[int, np.ndarray] = {}
    for p in prompts:
        m = p.region_mask(w, h)
        if m is not None:
            prompt_masks[p.frame] = m | prompt_masks.get(p.frame, np.zeros((h, w), bool))
    assert cfg.tracking is not None
    track_cls = registry.get("camera_tracking", cfg.tracking.model)
    key_track = stage_cache.key("camera_track", cache_mod.code_digest([track_cls]),
                                {"video": vfp, "model": cfg.tracking.model, "params": cfg.tracking.params,
                                 "ref": ref, "camera": calib.camera.to_dict(), "n": len(win)})
    cached = stage_cache.get_json("camera_track", key_track)
    if cached is not None:
        cam_track = CameraTrack.from_dict(cached)
    else:
        cam_track = _track_camera(cfg, win, ref, calib.camera, lambda f: prompt_masks.get(f))
        stage_cache.put_json("camera_track", key_track, _json_safe(cam_track.to_dict()))
    cov = np.asarray(calib.cov) if calib.cov is not None else np.zeros((len(CAMERA_PARAMS),) * 2)
    cam_track.calibration_cov = cov[:7, :7].copy()
    cam_track.centre_sigma = np.sqrt(np.maximum(np.diag(cov)[3:6], 0.0))
    timings["camera_tracking"] = time.perf_counter() - t0
    say(f"camera tracking: {cam_track.provenance[0].model if cam_track.provenance else '?'}; "
        f"focal {cam_track.focal.min():.1f}-{cam_track.focal.max():.1f} px")

    # 3. segmentation (cached)
    t0 = time.perf_counter()
    assert cfg.segmentation is not None
    seg_cls = registry.get("segmentation", cfg.segmentation.model)
    seg_params = dict(cfg.segmentation.params)
    if cfg.segmentation.model == "baseline.files":
        seg_params.setdefault("root", str(cfg.base_dir))
        seg_params.setdefault("objects", fit_ids)
    key_seg = stage_cache.key("segmentation", cache_mod.code_digest([seg_cls]),
                              {"video": vfp, "model": cfg.segmentation.model, "params": seg_params,
                               "camera_track": key_track, "prompts": [
                                   (p.object_id, p.frame, p.box, None if p.polygon is None else np.asarray(p.polygon).tolist(),
                                    None if p.mask is None else cache_mod.digest(np.packbits(p.mask).tolist()),
                                    None if p.points is None else np.asarray(p.points).tolist())
                                   for p in prompts]})
    seg_model = registry.create("segmentation", cfg.segmentation.model, **seg_params)
    raw_seg = stage_cache.get_npz("segmentation", key_seg)
    if raw_seg is not None:
        seg = _segmentation_from_cache(raw_seg, w, h, [seg_model.provenance()])
    else:
        seg = seg_model.segment(win, prompts, list(range(len(win))), cam_track)
        stage_cache.put_npz("segmentation", key_seg, _segmentation_to_cache(seg))
    timings["segmentation"] = time.perf_counter() - t0
    if not fit_ids:
        fit_ids = list(seg.tracks)
    say(f"segmentation: {cfg.segmentation.model}; objects {[(o, len(seg.tracks[o].crops)) for o in seg.tracks]}")

    # 4. depth (optional, lazy per frame)
    depth_fn = None
    depth_prov: list[ModelProvenance] = []
    if cfg.depth is not None:
        from openbeam_v2e.perception.baseline.depth import align_depth, ground_depth

        depth_model = registry.create("depth", cfg.depth.model, **cfg.depth.params)
        depth_prov = [depth_model.provenance()]
        memo: dict[int, DepthMap | None] = {}

        def depth_fn(f: int) -> DepthMap | None:  # noqa: F811 - closure assigned once
            if f not in memo:
                cam = cam_track.camera(f)
                dm = depth_model.estimate(win.read(f), cam)
                if dm.kind != "metric":
                    try:
                        dm, _ = align_depth(dm, ground_depth(cam, cfg.ground_z))
                    except ValueError:
                        dm = None
                memo[f] = dm
            return memo[f]

    # 5. pose per object
    t0 = time.perf_counter()
    assert cfg.pose is not None
    pose_model = registry.create("pose", cfg.pose.model, **cfg.pose.params)
    obj_cfg = {o.id: o for o in cfg.objects}
    tracks: dict[str, PoseTrack] = {}
    for oid in fit_ids:
        mt = seg.tracks.get(oid)
        if mt is None or len(mt.crops) < 2:
            warnings.append(f"object {oid!r}: fewer than two masks, no pose track")
            continue
        oc = obj_cfg.get(oid)
        cls = oc.object_class if oc else ann.classes.get(oid, cfg.default_class)
        spec = object_spec(oid, cls, oc.template if oc else None, oc.motion if oc else None,
                           oc.dims_prior if oc else None, oc.template_params if oc else None,
                           oc.fit_dims if oc else True)
        held = _holdout(cfg, mt.frames())
        ctx = PoseContext(cfg.ground_z, fps, held, depth_fn, {"log": say})
        try:
            tr = pose_model.fit(mt, cam_track, spec, list(range(len(win))), ctx)
        except ValueError as exc:
            warnings.append(f"object {oid!r}: {exc}")
            continue
        tr.provenance = list(cal_prov) + list(tr.provenance) + depth_prov
        tracks[oid] = tr
        say(f"pose {oid}: dims {np.round(tr.dims, 3).tolist()} +- {np.round(tr.dims_sigma, 3).tolist()}")
    timings["pose"] = time.perf_counter() - t0

    # 6. physics and events
    events = _physics_and_events(tracks, seg, cam_track, fps, cfg)

    models: list[ModelProvenance] = list(cal_prov) + list(cam_track.provenance) + det_prov \
        + list(seg.provenance) + [pose_model.provenance()] + depth_prov
    timings["total"] = time.perf_counter() - t_all
    src = {}
    if cfg.video_path is not None:
        src["name"] = Path(cfg.video_path).name
        if Path(cfg.video_path).is_file():
            src["sha256"] = cache_mod.file_digest(cfg.video_path)
    return Reconstruction(fps, len(win), (w, h), video_frames, calib, cam_track, seg, tracks, events, models,
                          _json_safe(cfg.raw), src, warnings, timings, dict(stage_cache.used), cfg.ground_z)


def reconstruct(video: str | Path | None = None, out_dir: str | Path | None = None,
                config: Mapping[str, Any] | str | Path | None = None, config_path: str | Path | None = None,
                annotations: Any = None, annotations_path: str | Path | None = None,
                models: Mapping[str, str] | None = None, log: Callable[[str], None] | None = None,
                frames: FrameSource | None = None) -> dict[str, Any]:
    """Reconstruct a video into a world package and return a JSON-ready summary.

    ``config`` is a ``v2e.reconstruct/1`` document or a path to one (``config_path`` is the same
    as a path); ``video`` overrides the configured video; ``annotations`` (or
    ``annotations_path``) gives keyframe prompts; ``models`` maps stages to models, for example
    ``{"masks": "sam2", "camera": "static"}``. ``frames`` passes an in-memory frame source
    instead of a file. The world package is written to ``out_dir`` when given.
    """
    path = config_path or (config if isinstance(config, (str, Path)) else None)
    if path is not None:
        cfg = ReconstructionConfig.load(path)
    elif isinstance(config, Mapping):
        cfg = ReconstructionConfig.parse(config, Path.cwd())
    else:
        cfg = ReconstructionConfig.parse({}, Path.cwd())
    if video is not None:
        cfg.video_path = Path(video)
    ann = annotations_path or annotations
    if ann is not None:
        cfg.annotations = Path(ann) if isinstance(ann, (str, Path)) else ann
    cfg.apply_models(models)
    result = run_pipeline(cfg, frames, log)
    out = None
    if out_dir is not None:
        out = result.write(out_dir)
    return result.summary(out)


def load_reconstruction(path: str | Path) -> dict[str, Any]:
    """Read ``v2e/reconstruct_report.json`` from a world package directory (or the file itself)."""
    from openbeam_v2e.schema import jsonio

    p = Path(path)
    if p.is_dir():
        p = p / "v2e" / "reconstruct_report.json"
    return jsonio.load(p)
