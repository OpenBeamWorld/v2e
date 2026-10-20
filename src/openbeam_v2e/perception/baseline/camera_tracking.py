# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""Camera tracking baselines for a camera at one position.

:class:`RotationZoomTracker` (``baseline.rotation_zoom``) follows pan, tilt, roll and zoom with
sparse features, using the fact that views from one position are related exactly by a rotation
and a focal length change:

1. corners are detected in a keyframe (moving objects excluded) and their viewing rays computed
   from the keyframe camera;
2. each new frame, the corners are tracked from the previous frame with pyramidal Lucas-Kanade,
   starting from where a constant-rate motion model predicts them;
3. the frame's (pan, tilt, roll, focal) is solved from the keyframe rays and the tracked pixels
   by robust (Huber) bounded least squares, with outliers rejected by residual;
4. the corners are then tracked again directly from the keyframe image, starting at the
   predicted positions, which removes the drift of frame-to-frame chaining, and the state is
   solved once more;
5. when too few corners survive, or the zoom has changed by more than ``max_zoom_change`` since
   the keyframe (template matching degrades with scale), the current frame becomes the keyframe.

The optional roll smoothing (Whittaker, weighted by sigma) keeps roll noise on frames with few
features from leaking into pan and tilt; the frames are then re-solved with the smoothed roll
held.

:class:`StaticCamera` (``baseline.static``) returns the reference camera for every frame.

:class:`PointTrackRotationZoom` (``baseline.point_tracks``) solves the same camera model from the
tracks of any registered point-tracking model, so a learned tracker such as CoTracker can drive
camera tracking.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
from scipy import ndimage

from openbeam_v2e.perception.baseline.point_tracking import _pyramid, good_features, lucas_kanade
from openbeam_v2e.perception.interfaces import CameraTracker, ExcludeFn
from openbeam_v2e.perception.types import CameraTrack
from openbeam_v2e.reconstruct.camera_track import (
    camera_from_state,
    predict_state,
    smooth_track,
    solve_from_rays,
    state_from_camera,
)
from openbeam_v2e.reconstruct.geom import Camera
from openbeam_v2e.reconstruct.video import to_gray


class StaticCamera(CameraTracker):
    """``baseline.static``: the camera does not move; every frame gets the reference camera."""

    name = "baseline.static"
    summary = "Fixed camera: repeats the reference camera for every frame."

    def __init__(self, sigma_deg: float = 0.0, **params: Any) -> None:
        super().__init__(sigma_deg=sigma_deg, **params)

    def track(self, video: Any, reference: Camera, frames: Sequence[int],
              exclude: ExcludeFn | None = None) -> CameraTrack:
        s = math.radians(self.params["sigma_deg"])
        return CameraTrack.static(reference, frames, (s, s, s, 0.0), [self.provenance()])


class RotationZoomTracker(CameraTracker):
    """``baseline.rotation_zoom``: sparse-feature pan/tilt/roll/zoom tracking (CPU)."""

    name = "baseline.rotation_zoom"
    summary = ("Pan/tilt/roll/zoom from corners tracked with Lucas-Kanade and solved as a pure "
               "rotation plus zoom about the fixed camera position (NumPy/SciPy, CPU).")

    def __init__(self, max_features: int = 300, min_distance: int = 8, quality: float = 0.01,
                 win: int = 7, levels: int = 3, min_inlier_fraction: float = 0.45,
                 min_inliers: int = 25, max_zoom_change: float = 0.12, huber_px: float = 1.0,
                 reject_px: float = 2.5, estimate_roll: bool = True, estimate_zoom: bool = True,
                 smooth_roll: float | None = None, exclude_dilate_px: int = 6, **params: Any) -> None:
        super().__init__(max_features=max_features, min_distance=min_distance, quality=quality, win=win,
                         levels=levels, min_inlier_fraction=min_inlier_fraction, min_inliers=min_inliers,
                         max_zoom_change=max_zoom_change, huber_px=huber_px, reject_px=reject_px,
                         estimate_roll=estimate_roll, estimate_zoom=estimate_zoom, smooth_roll=smooth_roll,
                         exclude_dilate_px=exclude_dilate_px, **params)

    def _exclude(self, exclude: ExcludeFn | None, frame: int, shape: tuple[int, int]) -> np.ndarray | None:
        if exclude is None:
            return None
        m = exclude(frame)
        if m is None:
            return None
        d = int(self.params["exclude_dilate_px"])
        m = np.asarray(m, bool)
        return ndimage.binary_dilation(m, iterations=d) if d > 0 else m

    def track(self, video: Any, reference: Camera, frames: Sequence[int],
              exclude: ExcludeFn | None = None) -> CameraTrack:
        p = self.params
        frames = [int(f) for f in frames]
        intr = reference.intrinsics
        centre = reference.centre
        free = ["pan", "tilt"]
        if p["estimate_roll"]:
            free.append("roll")
        if p["estimate_zoom"]:
            free.append("f")
        T = len(frames)
        states = np.zeros((T, 4))
        sigmas = np.zeros((T, 4))
        quality = np.zeros(T)
        states[0] = state_from_camera(reference)
        sigmas[0] = 0.0

        def new_keyframe(ti: int, gray: np.ndarray) -> dict[str, Any]:
            cam = camera_from_state(intr, centre, states[ti])
            pts = good_features(gray, p["max_features"], p["quality"], p["min_distance"],
                                self._exclude(exclude, frames[ti], gray.shape), border=p["win"] + 2)
            return {"ti": ti, "gray": gray, "pyr": _pyramid(gray, p["levels"]), "uv": pts,
                    "rays": cam.rays(pts) if len(pts) else np.zeros((0, 3)), "state": states[ti].copy(),
                    "alive": np.ones(len(pts), bool), "cur": pts.copy()}

        gray0 = to_gray(video.read(frames[0]))
        key = new_keyframe(0, gray0)
        quality[0] = len(key["uv"])
        prev_gray, prev_pyr = gray0, key["pyr"]
        history = [states[0]]
        for ti in range(1, T):
            gray = to_gray(video.read(frames[ti]))
            pyr = _pyramid(gray, p["levels"])
            guess_state = predict_state(history[-2:])
            alive = key["alive"]
            idx = np.nonzero(alive)[0]
            solved = None
            if len(idx) >= 4:
                # (2) frame-to-frame tracking from the previous positions, started at the prediction
                pred_uv = camera_from_state(intr, centre, guess_state).project_directions(key["rays"][idx])[0]
                prev_uv = key["cur"][idx]
                start = np.where(np.isfinite(pred_uv), prev_uv + (pred_uv - camera_from_state(
                    intr, centre, history[-1]).project_directions(key["rays"][idx])[0]), prev_uv)
                start = np.where(np.isfinite(start), start, prev_uv)
                uv1, ok1, _ = lucas_kanade(prev_gray, gray, prev_uv, guess=start, win=p["win"],
                                           levels=p["levels"], pyramids=(prev_pyr, pyr))
                obs = np.where(ok1[:, None], uv1, np.nan)
                # (3) solve from keyframe rays
                s1 = solve_from_rays(intr, key["rays"][idx], obs, guess_state, free, p["huber_px"],
                                     0.5, p["reject_px"])
                if s1.success and s1.inliers.sum() >= 4:
                    # (4) re-track from the keyframe image at the predicted positions (drift-free)
                    pred2 = camera_from_state(intr, centre, s1.state).project_directions(key["rays"][idx])[0]
                    good_pred = np.all(np.isfinite(pred2), axis=1)
                    uv2, ok2, _ = lucas_kanade(key["gray"], gray, key["uv"][idx],
                                               guess=np.where(good_pred[:, None], pred2, key["uv"][idx]),
                                               win=p["win"], levels=1, pyramids=(key["pyr"], pyr))
                    ok2 &= good_pred & (np.linalg.norm(uv2 - pred2, axis=1) < 3.0 * p["reject_px"])
                    obs2 = np.where(ok2[:, None], uv2, np.nan)
                    s2 = solve_from_rays(intr, key["rays"][idx], obs2, s1.state, free, p["huber_px"],
                                         0.5, p["reject_px"])
                    solved = s2 if (s2.success and s2.inliers.sum() >= max(4, 0.5 * s1.inliers.sum())) else s1
                    used_obs = obs2 if solved is s2 else obs
                    inl = solved.inliers & np.all(np.isfinite(used_obs), axis=1)
                    key["cur"][idx] = np.where(inl[:, None], used_obs, key["cur"][idx])
                    alive_new = np.zeros_like(alive)
                    alive_new[idx[inl]] = True
                    key["alive"] = alive_new
            if solved is None:
                states[ti] = guess_state
                sigmas[ti] = np.inf
                quality[ti] = 0
            else:
                states[ti] = solved.state
                sigmas[ti] = solved.sigma
                quality[ti] = int(solved.inliers.sum())
            history.append(states[ti])
            n_alive = int(key["alive"].sum())
            zoom_change = abs(states[ti][3] / key["state"][3] - 1.0)
            if (n_alive < max(p["min_inliers"], p["min_inlier_fraction"] * len(key["uv"]))
                    or zoom_change > p["max_zoom_change"]):
                key = new_keyframe(ti, gray)
            prev_gray, prev_pyr = gray, pyr

        if p["smooth_roll"] and T >= 3 and "roll" in free:
            roll_s = smooth_track(states[:, 2], np.maximum(sigmas[:, 2], 1e-6), float(p["smooth_roll"]))
            states[:, 2] = roll_s
            # the per-frame pan/tilt/zoom stay as solved; roll smoothing is a small correction
        sigmas[0] = np.nanmedian(np.where(np.isfinite(sigmas[1:]), sigmas[1:], np.nan), axis=0) if T > 1 else 0.0
        sigmas = np.where(np.isfinite(sigmas), sigmas, 1e3)
        return CameraTrack(intr, centre, frames, states[:, 0], states[:, 1], states[:, 2], states[:, 3],
                           sigmas, quality=quality, provenance=[self.provenance()])


class PointTrackRotationZoom(CameraTracker):
    """``baseline.point_tracks``: pan/tilt/roll/zoom from any point-tracking model.

    Corners are detected on anchor frames spread over the clip (moving objects excluded) and handed to
    the point tracker named by ``point_tracker`` (``baseline.lk``, ``cotracker`` or any registered
    model) in one call. Frames are then solved in order: every query whose anchor frame is already
    solved contributes its viewing ray, and the frame's state is solved from the visible queries
    with the same robust rotation-plus-zoom model as ``baseline.rotation_zoom``.
    """

    name = "baseline.point_tracks"
    summary = ("Pan/tilt/roll/zoom from the tracks of any registered point tracker (for example CoTracker), "
               "solved as a rotation plus zoom about the fixed camera position.")

    def __init__(self, point_tracker: str = "baseline.lk", point_tracker_params: dict | None = None,
                 keyframe_every: int = 15, max_features: int = 200, min_distance: int = 8,
                 quality: float = 0.01, huber_px: float = 1.0, reject_px: float = 2.5,
                 estimate_roll: bool = True, estimate_zoom: bool = True, exclude_dilate_px: int = 6,
                 **params: Any) -> None:
        super().__init__(point_tracker=point_tracker, point_tracker_params=dict(point_tracker_params or {}),
                         keyframe_every=keyframe_every, max_features=max_features, min_distance=min_distance,
                         quality=quality, huber_px=huber_px, reject_px=reject_px, estimate_roll=estimate_roll,
                         estimate_zoom=estimate_zoom, exclude_dilate_px=exclude_dilate_px, **params)

    def track(self, video: Any, reference: Camera, frames: Sequence[int],
              exclude: ExcludeFn | None = None) -> CameraTrack:
        from openbeam_v2e.perception import registry

        p = self.params
        frames = [int(f) for f in frames]
        T = len(frames)
        intr, centre = reference.intrinsics, reference.centre
        free = ["pan", "tilt"] + (["roll"] if p["estimate_roll"] else []) + (["f"] if p["estimate_zoom"] else [])
        keys = list(range(0, T, max(1, int(p["keyframe_every"]))))
        queries, owner = [], []
        for k in keys:
            gray = to_gray(video.read(frames[k]))
            ex = None
            if exclude is not None and exclude(frames[k]) is not None:
                ex = ndimage.binary_dilation(np.asarray(exclude(frames[k]), bool),
                                             iterations=int(p["exclude_dilate_px"]))
            pts = good_features(gray, p["max_features"], p["quality"], p["min_distance"], ex)
            queries += [[frames[k], u, v] for u, v in pts]
            owner += [k] * len(pts)
        tracker = registry.create("point_tracking", p["point_tracker"], **p["point_tracker_params"])
        tracks = tracker.track(video, np.asarray(queries, float), frames)
        owner_arr = np.asarray(owner)
        states = np.zeros((T, 4))
        sigmas = np.zeros((T, 4))
        quality = np.zeros(T)
        states[0] = state_from_camera(reference)
        rays = np.full((len(queries), 3), np.nan)
        history = [states[0]]
        solved = {0}
        sel0 = owner_arr == 0
        rays[sel0] = reference.rays(np.asarray(queries, float)[sel0, 1:])
        for ti in range(1, T):
            guess = predict_state(history[-2:])
            ok = np.isfinite(rays[:, 0]) & tracks.visible[:, ti]
            if ok.sum() >= 4:
                s = solve_from_rays(intr, rays[ok], tracks.points[ok, ti], guess, free, p["huber_px"], 0.5,
                                    p["reject_px"])
                states[ti], sigmas[ti], quality[ti] = s.state, s.sigma, int(s.inliers.sum())
            else:
                states[ti], sigmas[ti] = guess, np.inf
            history.append(states[ti])
            if ti in keys and ti not in solved:
                sel = owner_arr == ti
                rays[sel] = camera_from_state(intr, centre, states[ti]).rays(np.asarray(queries, float)[sel, 1:])
                solved.add(ti)
        sigmas[0] = np.nanmedian(np.where(np.isfinite(sigmas[1:]), sigmas[1:], np.nan), axis=0) if T > 1 else 0.0
        sigmas = np.where(np.isfinite(sigmas), sigmas, 1e3)
        prov = [self.provenance()] + list(tracks.provenance)
        return CameraTrack(intr, centre, frames, states[:, 0], states[:, 1], states[:, 2], states[:, 3], sigmas,
                           quality=quality, provenance=prov)
