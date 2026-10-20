# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""Benchmark comparisons between two worlds on the same video timeline.

The benchmark compares a world with its truth twice: the reconstruction (did perception recover
the camera, the objects' sizes, paths, attitudes and the events?) and the run exported from a
connector (did the recreated scene play out like the truth?). Both comparisons use the same
definitions:

* camera: reprojection RMS of known world points between the truth camera and the other camera,
  frame by frame, and the relative focal-length error;
* dimensions: absolute error of each measured dimension (vehicles: length, width and height;
  props: the hull's extents from smallest to largest);
* trajectory: RMS distance between the objects' bounding-box centres over frames both know;
* attitude: geodesic angle between the attitudes, taken modulo the object's symmetry
  (:func:`openbeam_v2e.score.attitude.attitude_error`);
* events: frame difference between same-named events;
* silhouette: intersection over union of the two objects' outlines drawn through the truth
  camera.

All numbers come back raw; the benchmark report labels them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

__all__ = ["CameraErrors", "ObjectErrors", "WorldComparison", "compare_cameras", "compare_objects",
           "compare_events", "compare_worlds"]


@dataclass
class CameraErrors:
    rms_px: float | None
    max_px: float | None
    focal_err_pct: float | None
    frames: int


@dataclass
class ObjectErrors:
    object_id: str
    traj_rms_m: float | None
    att_mean_deg: float | None
    dims_err_m: dict[str, float] = field(default_factory=dict)
    iou_mean: float | None = None
    frames: int = 0

    @property
    def dims_err_max_m(self) -> float | None:
        return max(self.dims_err_m.values()) if self.dims_err_m else None


@dataclass
class WorldComparison:
    camera: CameraErrors | None
    objects: dict[str, ObjectErrors]
    event_frames: dict[str, float]
    missing_events: list[str]

    def mean(self, attr: str, objects: Iterable[str] | None = None) -> float | None:
        ids = list(objects) if objects is not None else list(self.objects)
        vals = [getattr(self.objects[i], attr) for i in ids if i in self.objects]
        vals = [v for v in vals if v is not None and math.isfinite(v)]
        return float(np.mean(vals)) if vals else None

    def event_error_mean(self) -> float | None:
        vals = list(self.event_frames.values())
        return float(np.mean(vals)) if vals else None

    def to_json(self) -> dict[str, Any]:
        cam = None if self.camera is None else dict(self.camera.__dict__)
        objs = {}
        for k, o in self.objects.items():
            objs[k] = {"traj_rms_m": o.traj_rms_m, "att_mean_deg": o.att_mean_deg, "dims_err_m": o.dims_err_m,
                       "iou_mean": o.iou_mean, "frames": o.frames}
        return {"camera": cam, "objects": objs, "event_frames": self.event_frames,
                "missing_events": self.missing_events}


def compare_cameras(truth: Any, other: Any, points: np.ndarray, margin_px: float = 0.0) -> CameraErrors:
    """Reprojection RMS of ``points`` between two camera tracks (``openbeam_v2e.world.CameraTrack``)."""
    pts = np.asarray(points, float).reshape(-1, 3)
    F = min(truth.frame_count, other.frame_count)
    errs, focal = [], []
    for f in range(F):
        a = truth.camera(f)
        b = other.camera(f)
        uv_a, _, ok_a = a.project(pts)
        uv_b, _, ok_b = b.project(pts)
        inside = ok_a & ok_b & (uv_a[:, 0] >= margin_px) & (uv_a[:, 0] < a.width - margin_px) \
            & (uv_a[:, 1] >= margin_px) & (uv_a[:, 1] < a.height - margin_px)
        if np.any(inside):
            # Compare in the truth image's pixel units even if the other track has another size.
            scale = a.width / max(1, b.width)
            d = np.linalg.norm(uv_a[inside] - uv_b[inside] * scale, axis=1)
            errs.append(d)
        focal.append(abs(b.fx * (a.width / max(1, b.width)) - a.fx) / a.fx)
    if not errs:
        return CameraErrors(None, None, float(100.0 * np.mean(focal)) if focal else None, F)
    allv = np.concatenate(errs)
    return CameraErrors(float(np.sqrt(np.mean(allv ** 2))), float(allv.max()), float(100.0 * np.mean(focal)), F)


def _attitude_errors(ref: Any, other: Any) -> np.ndarray:
    try:
        from openbeam_v2e.score.attitude import attitude_error
    except ImportError:  # pragma: no cover - the score package ships with the distribution
        attitude_error = None
    Ra, Rb = ref.rotations(), other.rotations()
    if attitude_error is not None:
        return attitude_error(Ra, Rb, ref.symmetry_class, ref.hull_extents())
    from scipy.spatial.transform import Rotation

    ok = np.all(np.isfinite(Ra.reshape(len(Ra), -1)), axis=1) & np.all(np.isfinite(Rb.reshape(len(Rb), -1)), axis=1)
    out = np.full(len(Ra), np.nan)
    if np.any(ok):
        out[ok] = (Rotation.from_matrix(Ra[ok]).inv() * Rotation.from_matrix(Rb[ok])).magnitude()
    return out


def _silhouette_iou(world_ref: Any, ref: Any, other: Any, frames: Sequence[int]) -> float | None:
    from openbeam_v2e.world.raster import mask_iou, mesh_silhouette

    vals = []
    for f in frames:
        va, vb = ref.hull_world(f), other.hull_world(f)
        if va is None or vb is None:
            continue
        cam = world_ref.camera_track.camera(f)
        ma, _ = mesh_silhouette(cam, va, ref.hull_faces)
        mb, _ = mesh_silhouette(cam, vb, other.hull_faces)
        if not ma.any():
            continue
        v = mask_iou(ma, mb)
        if v is not None:
            vals.append(v)
    return float(np.mean(vals)) if vals else None


def compare_objects(truth: Any, other: Any, ids: Iterable[str], dims: bool = True,
                    silhouette_frames: int = 12) -> dict[str, ObjectErrors]:
    """Per-object errors of ``other`` against ``truth`` (both ``openbeam_v2e.world.World``)."""
    out: dict[str, ObjectErrors] = {}
    for oid in ids:
        if not truth.has_object(oid):
            continue
        ref = truth.object(oid)
        if not other.has_object(oid):
            out[oid] = ObjectErrors(oid, None, None)
            continue
        obj = other.object(oid)
        F = min(ref.frame_count, obj.frame_count)
        ca, cb = ref.centre_track()[:F], obj.centre_track()[:F]
        both = np.all(np.isfinite(ca), axis=1) & np.all(np.isfinite(cb), axis=1)
        traj = float(np.sqrt(np.mean(np.sum((ca[both] - cb[both]) ** 2, axis=1)))) if np.any(both) else None
        att = _attitude_errors(ref, obj)[:F]
        att_ok = att[np.isfinite(att)]
        att_deg = float(np.degrees(att_ok.mean())) if len(att_ok) else None
        dims_err: dict[str, float] = {}
        if dims:
            if ref.cls.startswith("vehicle."):
                for name, m in ref.dims_m.items():
                    if name in obj.dims_m and name in ("length", "width", "height"):
                        dims_err[name] = abs(float(obj.dims_m[name].value) - float(m.value))
            else:
                # Props: compare the hulls' extents, smallest to largest, so that naming and body-frame
                # conventions (a cylinder's axis on x or on z) do not matter.
                ea, eb = np.sort(ref.hull_extents()), np.sort(obj.hull_extents())
                dims_err = {f"extent_{k}": abs(float(eb[k] - ea[k])) for k in range(3)}
        known = np.flatnonzero(both)
        frames = known[np.linspace(0, len(known) - 1, min(silhouette_frames, len(known))).astype(int)] if len(known) else []
        out[oid] = ObjectErrors(oid, traj, att_deg, dims_err, _silhouette_iou(truth, ref, obj, list(frames)),
                                int(np.count_nonzero(both)))
    return out


def compare_events(truth_events: Mapping[str, float | None], other_events: Mapping[str, float | None],
                   fps: float) -> tuple[dict[str, float], list[str]]:
    """Absolute frame difference per event present in both; names missing from ``other``."""
    errors, missing = {}, []
    for name, t in truth_events.items():
        if t is None:
            continue
        o = other_events.get(name)
        if o is None:
            missing.append(name)
            continue
        errors[name] = abs(float(o) - float(t)) * fps
    return errors, missing


def compare_worlds(truth: Any, other: Any, ids: Iterable[str], truth_events: Mapping[str, float | None],
                   other_events: Mapping[str, float | None], points: np.ndarray | None = None,
                   dims: bool = True) -> WorldComparison:
    cam = compare_cameras(truth.camera_track, other.camera_track, points) if points is not None else None
    objs = compare_objects(truth, other, ids, dims=dims)
    ev, missing = compare_events(truth_events, other_events, truth.fps)
    return WorldComparison(cam, objs, ev, missing)
