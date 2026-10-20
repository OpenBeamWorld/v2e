# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""Mask support prevents confident background drift from entering point fitting."""

import numpy as np
import pytest

from openbeam_v2e.perception.point_support import filter_by_masks
from openbeam_v2e.perception.types import MaskTrack, PointTracks


def test_occlusion_and_reappearance_do_not_preserve_confident_background_tracks():
    # Object moves from x=2 to x=8 while hidden. Two old point tracks remain on
    # the background with full confidence; a separately seeded point reacquires.
    frames = np.array([0, 4, 8])
    xy = np.array([[[2.5, 3.5], [2.5, 3.5], [2.5, 3.5]],
                   [[2.5, 3.5], [2.5, 3.5], [8.5, 3.5]],
                   [[1.5, 1.5], [1.5, 1.5], [1.5, 1.5]]])
    tracks = PointTracks(frames, xy.copy(), np.ones((3, 3), bool))
    masks = MaskTrack("cube", 12, 8)
    for frame, x in [(0, 2), (8, 8)]:
        mask = np.zeros((8, 12), bool)
        mask[3, x] = True
        masks.set(frame, mask)
    result, rejected = filter_by_masks(tracks, {0: masks, 1: masks}, margin_px=0)
    assert result.visible.tolist() == [[True, False, False], [True, False, True], [True, True, True]]
    assert rejected.tolist() == [[False, True, True], [False, True, False], [False, False, False]]
    assert np.isnan(result.points[~result.visible]).all()
    np.testing.assert_array_equal(tracks.points, xy)
    assert tracks.visible.all()
    assert result.provenance[-1].params["support"][0]["query_indices"] == [0, 1]


def test_mask_holes_are_not_treated_as_solid_bounding_boxes_and_margin_is_bounded():
    mask = np.zeros((9, 9), bool)
    mask[1:8, 1:8] = True
    mask[2:7, 2:7] = False
    masks = MaskTrack("ring", 9, 9)
    masks.set(0, mask)
    tracks = PointTracks(np.array([0]), np.array([[[4.5, 4.5]], [[2.5, 4.5]], [[1.5, 4.5]]]),
                         np.ones((3, 1), bool))
    support = {i: masks for i in range(3)}
    result, _ = filter_by_masks(tracks, support, margin_px=0)
    assert result.visible[:, 0].tolist() == [False, False, True]
    result, _ = filter_by_masks(tracks, support, margin_px=1)
    assert result.visible[:, 0].tolist() == [False, True, True]
    # Mask provenance must reflect a changed observation.
    previous = result.provenance[-1].params["support"][0]["mask_sha256"]
    mask[4, 4] = True
    masks.set(0, mask)
    result, _ = filter_by_masks(tracks, support, margin_px=1)
    assert result.provenance[-1].params["support"][0]["mask_sha256"] != previous


def test_visibility_is_never_invented_by_a_support_mask():
    masks = MaskTrack("object", 8, 6)
    masks.set(0, np.ones((6, 8), bool))
    tracks = PointTracks(np.array([0]), np.array([[[2.5, 3.5]], [[np.nan, 3.5]], [[-1, 2]]]),
                         np.array([[False], [True], [True]]))
    result, _ = filter_by_masks(tracks, {i: masks for i in range(3)})
    assert not result.visible.any()
    assert np.isnan(result.points).all()
    with pytest.raises(ValueError, match="index"):
        filter_by_masks(tracks, {3: masks})
    with pytest.raises(ValueError, match="margin"):
        filter_by_masks(tracks, {0: masks}, margin_px=np.nan)
