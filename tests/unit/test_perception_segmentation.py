# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""Segmentation and depth baselines on synthetic clips with exact masks and depth."""

from __future__ import annotations

import numpy as np
import pytest

from _perception_clips import car_ptz, car_static
from openbeam_v2e.perception import registry
from openbeam_v2e.perception.baseline.depth import align_depth, ground_depth
from openbeam_v2e.perception.types import CameraTrack, DepthMap, ObjectPrompt, Segmentation
from openbeam_v2e.reconstruct import render, silhouette, synthetic
from openbeam_v2e.reconstruct.video import ArrayFrames, write_png


def _prompt_box(mask: np.ndarray, pad: int = 3) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = silhouette.mask_bbox(mask)
    return (x0 - pad, y0 - pad, x1 + pad, y1 + pad)


def test_background_subtraction_follows_a_car_with_a_panning_zooming_camera() -> None:
    clip = car_ptz()
    cams = CameraTrack.from_cameras(clip.cameras)
    seg = registry.create("segmentation", "baseline.background").segment(
        clip.frame_source(), [ObjectPrompt("car", 0, box=_prompt_box(clip.masks["car"][0]))],
        range(clip.n_frames), cams)
    track = seg.tracks["car"]
    ious = [silhouette.iou(track.mask(f), clip.masks["car"][f]) if track.mask(f) is not None else 0.0
            for f in range(clip.n_frames)]
    # tolerance: every frame segmented with IoU >= 0.9, median >= 0.95
    assert min(ious) >= 0.9
    assert np.median(ious) >= 0.95


def test_background_subtraction_finds_objects_without_prompts() -> None:
    clip = car_static()
    seg = registry.create("segmentation", "baseline.background").segment(
        clip.frame_source(), [], range(clip.n_frames), CameraTrack.static(clip.cameras[0], range(clip.n_frames)))
    assert list(seg.tracks) == ["object_1"]
    t = seg.tracks["object_1"]
    ious = [silhouette.iou(t.mask(f), clip.masks["car"][f]) for f in t.frames()]
    assert len(ious) >= clip.n_frames - 2 and np.median(ious) > 0.9


def test_mask_files_and_prompt_masks(tmp_path) -> None:
    clip = car_static()
    (tmp_path / "masks" / "car").mkdir(parents=True)
    for f in (0, 3, 7):
        write_png(tmp_path / "masks" / "car" / f"{f:06d}.png", clip.masks["car"][f])
    seg = registry.create("segmentation", "baseline.files", root=str(tmp_path), objects=["car"]).segment(
        clip.frame_source(), [], range(clip.n_frames))
    assert seg.tracks["car"].frames() == [0, 3, 7]
    assert np.array_equal(seg.tracks["car"].mask(3), clip.masks["car"][3])
    poly = silhouette.mask_to_polygons(clip.masks["car"][5])[0]
    seg2 = registry.create("segmentation", "baseline.prompts").segment(
        clip.frame_source(), [ObjectPrompt("car", 5, polygon=poly)], range(clip.n_frames))
    assert seg2.tracks["car"].frames() == [5]
    assert silhouette.iou(seg2.tracks["car"].mask(5), clip.masks["car"][5]) > 0.95


def test_ground_plane_depth_matches_the_rendered_depth() -> None:
    clip = car_static()
    cam = clip.cameras[0]
    rr = render.render(cam, render.Scene())
    rays = cam.pixel_grid_rays()
    truth = rr.depth * (rays @ cam.R_cw[2])          # ray length -> depth along the optical axis
    est = registry.create("depth", "baseline.ground_plane").estimate(clip.frames[0], cam).depth
    ok = np.isfinite(truth) & np.isfinite(est)
    assert ok.sum() > 1000
    assert np.abs(est[ok] / truth[ok] - 1).max() < 1e-6


@pytest.mark.parametrize("kind", ["relative", "inverse_relative"])
def test_align_depth_recovers_scale_and_shift(kind: str) -> None:
    cam = car_static().cameras[0]
    metric = ground_depth(cam)
    rel = 0.37 * metric + 2.0 if kind == "relative" else 5.0 / metric - 0.1
    rel = np.where(np.isfinite(metric), rel, np.nan)
    aligned, info = align_depth(DepthMap(rel, kind), metric)
    ok = np.isfinite(metric)
    assert np.allclose(aligned.depth[ok], metric[ok], rtol=1e-6)
    assert info["rms_m"] < 1e-6


def test_synthetic_masks_have_no_size_bias() -> None:
    """Ties in pixel coverage alternate, so truth masks are neither grown nor shrunk on average."""
    clip = synthetic.ground_vehicle_clip(width=160, height=90, n_frames=3, seed=7)
    car = clip.object("car")
    ratios = []
    for f in range(3):
        cam = clip.cameras[f].scaled(8.0)  # the outline sampled finely approximates its exact area
        fine = silhouette.hull_mask(car.hull, car.poses.R_wb[f], car.poses.position[f], cam)
        ratios.append(clip.masks["car"][f].sum() / (fine.sum() / 64.0))
    assert 0.97 < np.mean(ratios) < 1.03


def test_background_subtraction_tracks_before_the_first_prompt() -> None:
    clip = car_static()
    anchor = clip.n_frames // 2
    seg = registry.create("segmentation", "baseline.background").segment(
        clip.frame_source(), [ObjectPrompt("car", anchor, mask=clip.masks["car"][anchor])],
        range(clip.n_frames), CameraTrack.static(clip.cameras[0], range(clip.n_frames)))
    track = seg.tracks["car"]
    ious = [silhouette.iou(track.mask(f), clip.masks["car"][f]) if track.mask(f) is not None else 0.0
            for f in range(anchor)]
    assert min(ious) > 0.9


def test_box_prediction_uses_signed_time_when_tracking_backwards() -> None:
    from openbeam_v2e.perception.baseline.segmentation import _predict_box

    history = [(10, (20, 10, 30, 20)), (7, (14, 10, 24, 20))]
    assert _predict_box(history, 4) == pytest.approx((8, 10, 18, 20))


def test_prompt_guided_background_keeps_an_object_that_stops() -> None:
    frames, masks = [], []
    for f in range(30):
        mask = np.zeros((32, 96), bool)
        x = 8 + 5 * min(f, 8)
        mask[10:22, x:x + 12] = True
        image = np.full((32, 96, 3), 180, np.uint8)
        image[mask] = [210, 40, 50]
        frames.append(image)
        masks.append(mask)
    prompts = [ObjectPrompt("object", f, mask=masks[f]) for f in (0, 8, 29)]
    seg = registry.create("segmentation", "baseline.background").segment(
        ArrayFrames(frames), prompts, range(len(frames)))
    track = seg.tracks["object"]
    # The stationary object occupies the same pixels in a majority of the video. Only the
    # three prompts are supplied to the model; all other masks are held-back test truth.
    ious = [silhouette.iou(track.mask(f), masks[f]) if track.mask(f) is not None else 0.0
            for f in range(9, 29)]
    assert min(ious) > 0.9


@pytest.mark.parametrize("prompt_exclusion", [True, False])
def test_prompt_exclusions_are_not_observed_masks(monkeypatch, prompt_exclusion) -> None:
    model = registry.create("segmentation", "baseline.background", prompt_exclusion=prompt_exclusion)
    seen = {}

    def no_foreground(video, frames, cameras, cache, exclude=None):
        if exclude is not None:
            seen.update(exclude)
        return {f: np.zeros((24, 40), bool) for f in frames}

    monkeypatch.setattr(model, "foreground", no_foreground)
    mask = np.zeros((24, 40), bool)
    mask[8:16, 10:18] = True
    seg = model.segment(ArrayFrames(np.zeros((12, 24, 40, 3), np.uint8)),
                        [ObjectPrompt("object", f, mask=mask) for f in (2, 9)], range(12))
    assert seg.tracks["object"].frames() == [2, 9]
    assert bool(seen[5][12, 14]) is prompt_exclusion
    assert not seen[0].any() and not seen[11].any()


def test_prompt_exclusion_does_not_invent_a_clean_background() -> None:
    mask = np.zeros((24, 40), bool)
    mask[8:16, 10:18] = True
    image = np.full((24, 40, 3), 180, np.uint8)
    image[mask] = [210, 40, 50]
    seg = registry.create("segmentation", "baseline.background").segment(
        ArrayFrames([image] * 12), [ObjectPrompt("object", f, mask=mask) for f in (0, 11)], range(12))
    # An object that never uncovers the background cannot be recovered by subtraction.
    assert seg.tracks["object"].frames() == [0, 11]


@pytest.mark.parametrize("kind", ["box", "mask", "polygon"])
def test_prompt_bounds_interpolate_only_between_anchors(kind) -> None:
    model = registry.create("segmentation", "baseline.background", search_margin=0.0)

    def prompt(f, x):
        if kind == "box":
            return ObjectPrompt("object", f, box=(x, 8, x + 8, 16))
        if kind == "polygon":
            return ObjectPrompt("object", f, polygon=[[x, 8], [x + 8, 8], [x + 8, 16], [x, 16]])
        mask = np.zeros((24, 40), bool)
        mask[8:16, x:x + 8] = True
        return ObjectPrompt("object", f, mask=mask)

    bounds = model._prompt_bounds([prompt(2, 4), prompt(8, 16)], range(12))
    assert sorted(bounds) == list(range(2, 9))
    assert bounds[5] == [(8.0, 6.0, 20.0, 18.0)]


def test_point_prompts_have_no_background_exclusion_area() -> None:
    model = registry.create("segmentation", "baseline.background")
    prompts = [ObjectPrompt("object", f, points=[[10, 12]], labels=[1]) for f in (2, 9)]
    assert model._prompt_bounds(prompts, range(12)) == {}


def test_foreground_morphology_preserves_a_clipped_silhouette() -> None:
    model = registry.create("segmentation", "baseline.background")
    background = np.full((24, 40, 3), 0.7, np.float32)
    image = background.copy()
    image[8:18, :12] = [0.8, 0.1, 0.1]
    foreground = model._one(image, np.stack([background] * 3), 3)
    expected = np.zeros((24, 40), bool)
    expected[8:18, :12] = True
    assert foreground[10:16, 0].all()
    assert not foreground[:6, 0].any()
    assert silhouette.iou(foreground, expected) > 0.95


def test_automatic_segmentation_second_pass_uses_the_downscaled_grid() -> None:
    clip = car_static()
    seg = registry.create("segmentation", "baseline.background", max_width=128).segment(
        clip.frame_source(), [], range(clip.n_frames), CameraTrack.static(clip.cameras[0], range(clip.n_frames)))
    assert list(seg.tracks) == ["object_1"]
    track = seg.tracks["object_1"]
    assert len(track.frames()) >= clip.n_frames - 2
    assert track.mask(30).shape == clip.masks["car"][30].shape
    assert silhouette.iou(track.mask(30), clip.masks["car"][30]) > 0.9


@pytest.mark.parametrize("kind", ["mask", "box", "polygon"])
def test_later_prompt_recovers_a_stopped_object_after_a_visibility_gap(kind, monkeypatch) -> None:
    images, masks = [], []
    for f in range(30):
        mask = np.zeros((40, 144), bool)
        x = 12 + 8 * min(f, 4)
        if f not in (5, 6, 7):
            mask[12:26, x:x + 20] = True
        image = np.full((40, 144, 3), 180, np.uint8)
        image[mask] = [210, 40, 50]
        images.append(image)
        masks.append(mask)
    prompts = []
    for f in (0, 29):
        if kind == "mask":
            prompts.append(ObjectPrompt("object", f, mask=masks[f]))
        elif kind == "box":
            prompts.append(ObjectPrompt("object", f, box=silhouette.mask_bbox(masks[f])))
        else:
            prompts.append(ObjectPrompt("object", f, polygon=silhouette.mask_to_polygons(masks[f])[0]))
    model = registry.create("segmentation", "baseline.background")

    def colour_foreground(video, frames, cameras, cache=None, exclude=None):
        # Isolate association from background estimation with an RGB-only foreground stand-in.
        # The production background model is exercised separately on generated benchmark clips.
        return {f: video.read(f)[..., 0] > 200 for f in frames}

    monkeypatch.setattr(model, "foreground", colour_foreground)
    track = model.segment(ArrayFrames(images), prompts, range(30)).tracks["object"]
    # Only the first and last annotation reach the model. Missing masks count as zero;
    # a later annotation must not turn the intervening fully hidden frames into observations.
    ious = [silhouette.iou(track.mask(f), masks[f]) if track.mask(f) is not None else 0.0
            for f in range(8, 29)]
    assert min(ious) > 0.9
    assert all(track.mask(f) is None for f in (5, 6, 7))


def test_backward_reacquisition_preserves_existing_forward_observations() -> None:
    model = registry.create("segmentation", "baseline.background")
    first, last = np.zeros((24, 80), bool), np.zeros((24, 80), bool)
    first[6:18, 4:16] = True
    last[6:18, 52:64] = True
    comps = {f: [(m, silhouette.mask_bbox(m)) for m in (first, last)] for f in range(7)}
    seg = Segmentation(80, 24)
    model._assisted(seg, [ObjectPrompt("object", 0, mask=first), ObjectPrompt("object", 6, mask=last)],
                    list(range(7)), comps, 80, 24)
    assert np.array_equal(seg.tracks["object"].mask(3), first)
    assert np.array_equal(seg.tracks["object"].mask(6), last)


def test_explicit_prompt_pixels_win_over_propagated_masks() -> None:
    model = registry.create("segmentation", "baseline.background")
    mask = np.zeros((24, 40), bool)
    mask[6:18, 10:22] = True
    comps = {f: [(mask, silhouette.mask_bbox(mask))] for f in range(7)}
    seg = Segmentation(40, 24)
    # Both tracks can follow the same component; annotation evidence must outrank a perfect
    # prediction match, regardless of object iteration order.
    model._assisted(seg, [ObjectPrompt("first", 6, mask=mask), ObjectPrompt("second", 3, mask=mask)],
                    list(range(7)), comps, 40, 24)
    assert np.array_equal(seg.tracks["second"].mask(3), mask)
    assert seg.tracks["first"].mask(3) is None
    assert np.array_equal(seg.tracks["first"].mask(6), mask)
    assert seg.tracks["second"].mask(6) is None


def test_unsampled_prompt_can_seed_only_requested_foreground_frames() -> None:
    model = registry.create("segmentation", "baseline.background")
    mask = np.zeros((24, 40), bool)
    mask[6:18, 10:22] = True
    frames = list(range(2, 6))
    comps = {f: [(mask, silhouette.mask_bbox(mask))] for f in frames}
    seg = Segmentation(40, 24)
    model._assisted(seg, [ObjectPrompt("object", 0, mask=mask), ObjectPrompt("object", 7, mask=mask)],
                    frames, comps, 40, 24)
    assert seg.tracks["object"].frames() == frames
    assert all(np.array_equal(seg.tracks["object"].mask(f), mask) for f in frames)


def test_three_overlapping_tracks_can_remove_a_whole_mask_safely() -> None:
    model = registry.create("segmentation", "baseline.background")
    mask = np.zeros((24, 40), bool)
    mask[6:18, 10:22] = True
    frames = list(range(7))
    comps = {f: [(mask, silhouette.mask_bbox(mask))] for f in frames}
    seg = Segmentation(40, 24)
    model._assisted(seg, [ObjectPrompt("first", 6, mask=mask), ObjectPrompt("second", 3, mask=mask),
                          ObjectPrompt("third", 4, mask=mask)], frames, comps, 40, 24)
    for oid, f in (("first", 6), ("second", 3), ("third", 4)):
        assert np.array_equal(seg.tracks[oid].mask(f), mask)
    for f in frames:
        observed = [t.mask(f) for t in seg.tracks.values() if t.mask(f) is not None]
        assert len(observed) == 1
