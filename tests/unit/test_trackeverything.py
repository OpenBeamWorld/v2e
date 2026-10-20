# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""Geometry and frame-grid contracts; model accuracy needs separate GPU evidence."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from openbeam_v2e.perception import registry
from openbeam_v2e.perception.adapters.trackeverything import (
    TrackEverythingPointTracker,
    project_query_tracks,
)
from openbeam_v2e.reconstruct.video import ArrayFrames


def test_projection_camera_direction_resize_and_normalization():
    # A 90-degree world-to-camera rotation and a nonzero camera translation.
    E = np.array([[[0., -1., 0., .5], [1., 0., 0., -.25],
                   [0., 0., 1., 1.], [0., 0., 0., 1.]]])
    K = np.array([[[100., 0., 80.], [0., 120., 60.], [0., 0., 1.]]])
    xyz = np.array([[[.25, -.3, 1.]]])  # camera coordinates (.8, 0, 2)
    xy, visible = project_query_tracks(xyz, [[9]], K, E, (121, 161), (361, 641))
    np.testing.assert_allclose(xy, [[[480.5, 180.5]]])
    assert visible.tolist() == [[True]]
    # Joint normalization of points AND translations must not change pixels.
    scaled = E.copy()
    scaled[:, :3, 3] /= 7
    xy2, _ = project_query_tracks(xyz / 7, [[9]], K, scaled, (121, 161), (361, 641))
    np.testing.assert_allclose(xy, xy2)


def test_lost_nonfinite_behind_camera_and_offscreen_points_stay_missing():
    xyz = np.array([[[0., 0., 1.], [0., 0., 1.], [0., 0., -1.],
                     [np.nan, 0., 1.], [20., 0., 1.], [0., 0., 1.]]])
    K = np.array([[[10., 0., 10.], [0., 10., 10.], [0., 0., 1.]]])
    logits = [[2., 0., 2., 2., 2., np.inf]]
    xy, visible = project_query_tracks(xyz, logits, K, np.eye(4)[None], (21, 21), (21, 21))
    assert visible[:, 0].tolist() == [True, False, False, False, False, False]
    np.testing.assert_allclose(xy[0, 0], [10.5, 10.5])
    assert np.isnan(xy[1:]).all()
    _, higher = project_query_tracks(xyz, logits, K, np.eye(4)[None], (21, 21), (21, 21),
                                     visibility_threshold=.9)
    assert not higher.any()


class Tensor:
    def __init__(self, data):
        self.data = np.asarray(data)

    def detach(self):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.data


def prediction(T, N):
    coords = np.zeros((1, T, N, 3))
    coords[..., 2] = 1
    return SimpleNamespace(
        query_coords=Tensor(coords), query_visibs=Tensor(np.full((1, T, N), 4.)),
        intrinsics=Tensor(np.tile(np.eye(3), (T, 1, 1))),
        extrinsics=Tensor(np.tile(np.eye(4), (T, 1, 1))),
        rgb=np.zeros((T, 3, 6, 8)), norm_factor=3.)


def test_registry_adapter_remaps_nonuniform_frames_once_and_keeps_raw_evidence(tmp_path, monkeypatch):
    cls = registry.get("point_tracking", "trackeverything")
    assert cls is TrackEverythingPointTracker
    output = tmp_path / "result"
    tracker = cls(tracker_source="trackeverything", output_dir=str(output), visibility_threshold=.1)
    tracker._weights_sha256, tracker._pi3_sha256 = "a" * 64, "b" * 64
    seen = {}

    def infer(**kwargs):
        seen.update(kwargs)
        return prediction(3, 2)

    monkeypatch.setattr(tracker, "_model", lambda: SimpleNamespace(infer=infer))
    # RGB channel values identify source frame and channel ordering.
    frames = np.zeros((10, 6, 8, 3), np.uint8)
    frames[..., 0] = np.arange(10)[:, None, None]
    frames[..., 1], frames[..., 2] = 21, 42
    source = ArrayFrames(frames, fps=24)
    queries = np.array([[2., .5, .5], [9., 7.5, 5.5]])
    result = tracker.track(source, queries, [2, 5, 9])
    np.testing.assert_array_equal(seen["video"][:, 0, 0], [[2, 21, 42], [5, 21, 42], [9, 21, 42]])
    np.testing.assert_array_equal(seen["query_points_2d"], [[0, 0, 0], [2, 7, 5]])
    np.testing.assert_array_equal(queries, [[2, .5, .5], [9, 7.5, 5.5]])
    assert "frame_stride" not in seen and "start_frame" not in seen
    assert result.frames.tolist() == [2, 5, 9]
    assert result.visible.tolist() == [[True, True, True], [False, False, True]]
    assert np.isnan(result.points[1, :2]).all()
    np.testing.assert_allclose(result.points[0], .5)
    with np.load(output / "query-tracks.npz", allow_pickle=False) as raw:
        assert raw["coords"].shape == (3, 2, 3)
        np.testing.assert_array_equal(raw["source_frames"], result.frames)
        np.testing.assert_allclose(raw["source_time_s"], np.array([2, 5, 9]) / 24)
        np.testing.assert_allclose(raw["points"], result.points, equal_nan=True)
        assert raw["norm_factor"] == 3
    receipt = json.loads((output / "tracking-result.json").read_text())
    assert receipt["inference_completed"] and not receipt["metric_scale_known"]
    assert receipt["source_fps"] == 24
    assert not receipt["tracking_accuracy_validated"] and not receipt["physical_recreation_validated"]
    assert receipt["provenance"]["weights_sha256"] == "a" * 64
    assert receipt["provenance"]["params"]["pi3_sha256"] == "b" * 64
    with pytest.raises(FileExistsError):
        tracker.track(source, queries, [2, 5, 9])


@pytest.mark.parametrize("frames,queries", [
    ([0, 2], [[1, 1, 1]]), ([0, 2], [[.5, 1, 1]]),
    ([0, 2], [[0, 8, 1]]), ([0, 2], [[0, np.nan, 1]]),
    ([2, 0], [[0, 1, 1]]), ([0, 0], [[0, 1, 1]]),
    ([0., 2.], [[0, 1, 1]]), ([0, 4], [[0, 1, 1]]),
])
def test_bad_frame_grid_or_queries_fail_before_loading_models(frames, queries, monkeypatch):
    tracker = TrackEverythingPointTracker(tracker_source="unused")
    monkeypatch.setattr(tracker, "_model", lambda: pytest.fail("model must not load"))
    with pytest.raises(ValueError):
        tracker.track(ArrayFrames(np.zeros((4, 6, 8, 3), np.uint8)), np.array(queries), frames)


def test_incomplete_prediction_leaves_failed_receipt(tmp_path, monkeypatch):
    output = tmp_path / "failed"
    tracker = TrackEverythingPointTracker(tracker_source="unused", output_dir=str(output))
    monkeypatch.setattr(tracker, "_model", lambda: SimpleNamespace(infer=lambda **kw: prediction(1, 1)))
    with pytest.raises(ValueError, match="Incomplete"):
        tracker.track(ArrayFrames(np.zeros((2, 6, 8, 3), np.uint8)), np.array([[0, .5, .5]]), [0, 1])
    receipt = json.loads((output / "tracking-result.json").read_text())
    assert not receipt["inference_completed"] and receipt["error_type"] == "ValueError"
    assert not (output / "query-tracks.npz").exists()
