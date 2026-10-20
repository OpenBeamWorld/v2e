# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""Reject point-tracker drift using separately observed object masks.

Mask agreement is a necessary image-support check, not proof that a point kept
its material identity. A newly visible patch must not be substituted for a lost
corner. Consumers can use the separate segmentation for object-pose fitting.
"""

from __future__ import annotations

import hashlib
from typing import Mapping

import numpy as np
from scipy.ndimage import distance_transform_edt

from openbeam_v2e import __version__
from openbeam_v2e.perception.types import MaskTrack, ModelProvenance, PointTracks


def filter_by_masks(tracks: PointTracks, support: Mapping[int, MaskTrack], *,
                    margin_px: float = 2.0) -> tuple[PointTracks, np.ndarray]:
    """Return copied tracks and an (N,T) mask of rejected visible predictions.

    ``support`` assigns query indices to independently observed object masks.
    A missing/empty mask rejects that query in that frame, regardless of model
    confidence. Queries absent from ``support`` retain their original validity.
    Coordinates use V2E pixel centres at +0.5. ``margin_px`` dilates support by
    Euclidean pixel-centre distance; it never fills missing observations.
    """
    if not np.isfinite(margin_px) or margin_px < 0:
        raise ValueError("margin_px must be finite and nonnegative")
    frames = np.asarray(tracks.frames)
    points = np.asarray(tracks.points, float).copy()
    visible = np.asarray(tracks.visible, bool).copy()
    if (frames.ndim != 1 or frames.dtype.kind not in "iu"
            or np.any(frames[1:] <= frames[:-1])
            or points.ndim != 3 or points.shape[1:] != (len(frames), 2)
            or visible.shape != points.shape[:2]):
        raise ValueError("Expected increasing source frames and points (N,T,2), visible (N,T)")
    original_visible = visible.copy()
    visible &= np.isfinite(points).all(axis=-1)
    groups: dict[int, tuple[MaskTrack, list[int]]] = {}
    for index, masks in support.items():
        if isinstance(index, bool) or not isinstance(index, (int, np.integer)) or not 0 <= index < len(points):
            raise ValueError("Support query index is outside the point array")
        if masks.width < 1 or masks.height < 1:
            raise ValueError("Support masks need a positive image size")
        groups.setdefault(id(masks), (masks, []))[1].append(int(index))
    provenance = []
    for masks, queries in groups.values():
        digest = hashlib.sha256(f"{masks.width}:{masks.height}".encode())
        for ti, frame in enumerate(frames):
            mask = masks.mask(int(frame))
            digest.update(f"{int(frame)}:".encode())
            if mask is None:
                digest.update(b"missing")
                visible[queries, ti] = False
                continue
            mask = np.asarray(mask, bool)
            if mask.shape != (masks.height, masks.width):
                raise ValueError("Support mask shape differs from its image size")
            digest.update(mask.tobytes())
            if not mask.any():
                visible[queries, ti] = False
                continue
            allowed = mask if margin_px == 0 else distance_transform_edt(~mask) <= margin_px
            for qi in queries:
                if not visible[qi, ti]:
                    continue
                u, v = points[qi, ti]
                if not (0 <= u < masks.width and 0 <= v < masks.height):
                    visible[qi, ti] = False
                else:
                    visible[qi, ti] = allowed[int(np.floor(v)), int(np.floor(u))]
        provenance.append({"query_indices": queries, "object_id": masks.object_id,
                           "image_size": [masks.width, masks.height], "mask_sha256": digest.hexdigest()})
    points[~visible] = np.nan
    record = ModelProvenance(
        "validation.mask_support", "point_tracking", __version__,
        implementation=__name__ + ":filter_by_masks",
        params={"margin_px": float(margin_px), "support": provenance,
                "missing_masks": "reject", "material_identity_validated": False})
    result = PointTracks(frames.copy(), points, visible, [*tracks.provenance, record])
    return result, original_visible & ~visible
