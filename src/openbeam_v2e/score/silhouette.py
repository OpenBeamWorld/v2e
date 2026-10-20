# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""Projected silhouette overlap through the reconstructed camera.

For each scored frame the run object's hull, at the run's pose, is drawn through the
reference camera of that frame and compared with the reference outline: the observed outline
(``v2e/outlines.json``) where the reconstruction has one, otherwise the reference hull drawn at
the reference pose. The raw IoU is reported next to a **shape-limited** value: the IoU the
run's content would reach if it sat exactly at the reference pose. Template shape mismatch
caps the raw IoU at that value, so ``iou / shape_limit`` isolates the pose error.

Masks are drawn at a reduced working width (default 320 px) and frames are subsampled to at
most ``frame_budget`` per phase; both choices are reported.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from openbeam_v2e.world.camera import Camera
from openbeam_v2e.world.geometry import quat_to_matrix
from openbeam_v2e.world.model import ObjectTrack, World
from openbeam_v2e.world.raster import mask_iou, mesh_silhouette, rasterize_polygons

__all__ = ["silhouette_scores", "frame_sample"]


def frame_sample(frames: Sequence[int], frame_budget: int) -> list[int]:
    """At most ``frame_budget`` frames spread evenly over ``frames`` (first and last kept)."""
    frames = list(frames)
    if len(frames) <= frame_budget:
        return frames
    idx = np.unique(np.round(np.linspace(0, len(frames) - 1, frame_budget)).astype(int))
    return [frames[i] for i in idx]


def _hull_at(obj: ObjectTrack, position: np.ndarray, quat: np.ndarray) -> np.ndarray:
    return obj.hull_vertices @ quat_to_matrix(quat).T + position


def _reference_mask(reference: World, ref_obj: ObjectTrack, frame: int, cam: Camera, scale: float) -> tuple[np.ndarray | None, str]:
    outlines = reference.outlines.get(ref_obj.id, {})
    if frame in outlines and outlines[frame]:
        polys = [np.asarray(p, dtype=float) * scale for p in outlines[frame]]
        return rasterize_polygons(polys, cam.width, cam.height), "observed_outline"
    if not ref_obj.known[frame]:
        return None, "none"
    mask, _ = mesh_silhouette(cam, _hull_at(ref_obj, ref_obj.position[frame], ref_obj.quat_xyzw[frame]),
                              ref_obj.hull_faces)
    return mask, "reference_hull"


def silhouette_scores(reference: World, run: World, object_id: str, frames: Sequence[int],
                      working_width: int = 320, frame_budget: int = 40) -> dict[str, Any]:
    """Raw and shape-limited IoU of one object over ``frames``."""
    ref_obj = reference.object(object_id)
    run_obj = run.object(object_id) if run.has_object(object_id) else None
    picked = frame_sample([f for f in frames if ref_obj.known[f] or f in reference.outlines.get(object_id, {})],
                          frame_budget)
    ious, limits, sources, skipped = [], [], set(), 0
    for f in picked:
        full = reference.camera_track.camera(f)
        scale = min(1.0, working_width / float(full.width))
        cam = full.scaled(scale)
        ref_mask, source = _reference_mask(reference, ref_obj, f, cam, scale)
        if ref_mask is None or not ref_mask.any():
            continue
        sources.add(source)
        if run_obj is None or not run_obj.known[f]:
            ious.append(0.0)
        else:
            mask, s = mesh_silhouette(cam, _hull_at(run_obj, run_obj.position[f], run_obj.quat_xyzw[f]),
                                      run_obj.hull_faces)
            skipped += s
            ious.append(mask_iou(mask, ref_mask) or 0.0)
        if ref_obj.known[f]:
            shape_hull = run_obj if run_obj is not None else ref_obj
            limit_mask, _ = mesh_silhouette(cam, _hull_at(shape_hull, ref_obj.position[f], ref_obj.quat_xyzw[f]),
                                            shape_hull.hull_faces)
            limits.append(mask_iou(limit_mask, ref_mask) or 0.0)
        else:
            limits.append(np.nan)
    if not ious:
        return {"frames": 0}
    iou = np.array(ious)
    limit = np.array(limits)
    with np.errstate(invalid="ignore", divide="ignore"):
        normalised = np.where(limit > 0, np.minimum(iou / limit, 1.0), np.nan)
    return {
        "frames": int(len(iou)),
        "iou_mean": float(np.mean(iou)),
        "iou_min": float(np.min(iou)),
        "shape_limit_mean": float(np.nanmean(limit)) if np.any(np.isfinite(limit)) else None,
        "iou_over_shape_limit_mean": float(np.nanmean(normalised)) if np.any(np.isfinite(normalised)) else None,
        "reference": sorted(sources),
        "working_width_px": int(min(working_width, reference.camera_track.width)),
        "triangles_behind_camera": int(skipped),
    }
