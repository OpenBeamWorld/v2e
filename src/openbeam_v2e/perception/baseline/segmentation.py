# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""Segmentation baselines: background subtraction, mask files and prompt masks.

:class:`BackgroundSubtraction` (``baseline.background``) needs no model. For every frame it builds
a background image from neighbouring frames warped into the frame's view. Views from one camera
position map onto each other exactly (rotation plus zoom), so a panning, zooming camera works as
well as a still one once the camera track is known. Moving objects are where the frame differs
from the per-pixel median of the warped neighbours; shadows (darker, same colour) are suppressed;
connected regions are then assigned to objects:

* **assisted** (prompts given): each object starts from its prompt boxes, outlines or masks;
  between prompts its box is predicted from the previous frames and regions overlapping the
  prediction are merged into its mask; a prompt mask or outline at a frame is used as is.
  A reverse pass from the last prompt fills missing observations using foreground evidence,
  preserving forward observations and resetting motion prediction at each annotation.
  The second pass excludes interpolated prompt boxes from background samples, so an object
  that stops is not immediately absorbed into the background. These boxes are not observations;
* **automatic** (no prompts): regions are linked frame to frame by box overlap, and tracks that
  last long enough become objects ``object_1``, ``object_2``, ... by size.

:class:`MaskFiles` (``baseline.files``) reads masks produced by any external tool, and
:class:`PromptMasks` (``baseline.prompts``) uses only the annotated outlines and masks.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

from openbeam_v2e.perception.interfaces import Segmenter
from openbeam_v2e.perception.types import CameraTrack, ObjectPrompt, Segmentation
from openbeam_v2e.reconstruct.silhouette import bbox_iou
from openbeam_v2e.reconstruct.video import read_mask


class _FrameCache:
    """Recently used frames as float32 RGB in [0, 1], optionally box-downscaled by ``factor``."""

    def __init__(self, video: Any, size: int = 32, factor: int = 1) -> None:
        self.video = video
        self.size = size
        self.factor = max(1, int(factor))
        self._d: OrderedDict[int, np.ndarray] = OrderedDict()

    def get(self, i: int) -> np.ndarray:
        if i in self._d:
            self._d.move_to_end(i)
            return self._d[i]
        img = np.asarray(self.video.read(i), np.float32) / 255.0
        s = self.factor
        if s > 1:
            h, w = img.shape[0] // s, img.shape[1] // s
            img = img[: h * s, : w * s].reshape(h, s, w, s, -1).mean(axis=(1, 3))
        self._d[i] = img
        if len(self._d) > self.size:
            self._d.popitem(last=False)
        return img


class _Warp:
    """Bilinear resampling of one view into another (cameras sharing one position)."""

    def __init__(self, src_cam: Any, dst_rays: np.ndarray, src_shape: tuple[int, int]) -> None:
        self.shape = dst_rays.shape[:2]
        H, W = src_shape
        uv, z = src_cam.project_directions(dst_rays.reshape(-1, 3))
        col = uv[:, 0] - 0.5
        row = uv[:, 1] - 0.5
        self.valid = (z > 0) & np.isfinite(col) & (col >= 0) & (col <= W - 1) & (row >= 0) & (row <= H - 1)
        col = np.where(self.valid, col, 0.0)
        row = np.where(self.valid, row, 0.0)
        c0 = np.clip(np.floor(col), 0, max(W - 2, 0)).astype(np.int64)
        r0 = np.clip(np.floor(row), 0, max(H - 2, 0)).astype(np.int64)
        fx = (col - c0).astype(np.float32)[:, None]
        fy = (row - r0).astype(np.float32)[:, None]
        self.i00 = r0 * W + c0
        self.W = W
        self.w = ((1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        img = image.reshape(-1, image.shape[2] if image.ndim == 3 else 1)
        i, W = self.i00, self.W
        out = (img[i] * self.w[0] + img[i + 1] * self.w[1] + img[i + W] * self.w[2]
               + img[i + W + 1] * self.w[3])
        out[~self.valid] = np.nan
        return out.reshape(*self.shape, -1)


def warp_to_view(image: np.ndarray, src_cam: Any, dst_rays: np.ndarray) -> np.ndarray:
    """Resample ``image`` (seen by ``src_cam``) into the view whose pixel rays are ``dst_rays``.

    Exact for cameras that share one position. Pixels the source does not see are NaN.
    """
    im = image if image.ndim == 3 else image[..., None]
    return _Warp(src_cam, dst_rays, im.shape[:2])(im.astype(np.float32))


def nanmedian0(stack: np.ndarray) -> np.ndarray:
    """Median over axis 0 ignoring NaN (much faster than ``np.nanmedian`` for image stacks)."""
    srt = np.sort(stack, axis=0)                      # NaN sorts last
    n = np.sum(np.isfinite(stack), axis=0)
    lo = np.clip((n - 1) // 2, 0, len(stack) - 1)
    hi = np.clip(n // 2, 0, len(stack) - 1)
    a = np.take_along_axis(srt, lo[None], axis=0)[0]
    b = np.take_along_axis(srt, hi[None], axis=0)[0]
    med = 0.5 * (a + b)
    return np.where(n > 0, med, np.nan)


def _components(fg: np.ndarray, min_area: int) -> list[tuple[np.ndarray, tuple[int, int, int, int]]]:
    lab, n = ndimage.label(fg)
    out = []
    for k, sl in enumerate(ndimage.find_objects(lab), start=1):
        if sl is None:
            continue
        comp = lab[sl] == k
        area = int(comp.sum())
        if area < min_area:
            continue
        full = np.zeros(fg.shape, bool)
        full[sl] = comp
        out.append((full, (sl[1].start, sl[0].start, sl[1].stop, sl[0].stop)))
    return out


def _expand(box: Sequence[float], frac: float, px: float) -> tuple[float, float, float, float]:
    w, h = box[2] - box[0], box[3] - box[1]
    dx, dy = frac * w + px, frac * h + px
    return (box[0] - dx, box[1] - dy, box[2] + dx, box[3] + dy)


def _overlap_frac(mask: np.ndarray, box: Sequence[float]) -> float:
    x0, y0 = max(0, int(np.floor(box[0]))), max(0, int(np.floor(box[1])))
    x1, y1 = min(mask.shape[1], int(np.ceil(box[2]))), min(mask.shape[0], int(np.ceil(box[3])))
    area = int(mask.sum())
    if area == 0 or x1 <= x0 or y1 <= y0:
        return 0.0
    return float(mask[y0:y1, x0:x1].sum()) / area


class BackgroundSubtraction(Segmenter):
    """``baseline.background``: motion-compensated background subtraction (CPU, no model)."""

    name = "baseline.background"
    summary = ("Moving-object masks from the median of neighbouring frames warped into each view, "
               "with shadow suppression and prompt-guided or automatic object tracking (CPU).")

    def __init__(self, min_gap_s: float = 0.15, neighbours: int = 16, min_overlap: float = 0.1,
                 min_samples: int = 3, threshold_sigma: float = 5.0, min_threshold: float = 0.08,
                 min_area_px: int = 30, open_px: int = 1, close_px: int = 2,
                 shadow_suppression: bool = True, shadow_ratio: tuple[float, float] = (0.35, 0.95),
                 shadow_chroma: float = 0.06, search_margin: float = 0.35, min_track_frames: int = 5,
                 second_pass: bool = True, prompt_exclusion: bool = True,
                 max_width: int = 640, **params: Any) -> None:
        super().__init__(min_gap_s=min_gap_s, neighbours=neighbours, min_overlap=min_overlap,
                         min_samples=min_samples, threshold_sigma=threshold_sigma,
                         min_threshold=min_threshold, min_area_px=min_area_px, open_px=open_px,
                         close_px=close_px, shadow_suppression=shadow_suppression,
                         shadow_ratio=tuple(shadow_ratio), shadow_chroma=shadow_chroma,
                         search_margin=search_margin, min_track_frames=min_track_frames,
                         second_pass=second_pass, prompt_exclusion=prompt_exclusion,
                         max_width=max_width, **params)

    # ---------------------------------------------------------------- foreground
    def _neighbours(self, f: int, frames: list[int], fps: float, cameras: CameraTrack | None) -> list[int]:
        """Background sample frames for ``f``: up to ``neighbours`` frames spread evenly in time over
        the frames that are at least ``min_gap_s`` away and still see most of its view.

        A pixel is background in the median as long as moving objects cover it in fewer than half
        of the samples; spreading the samples over the clip keeps that true for objects that move
        out of their own footprint during the clip.
        """
        p = self.params
        min_gap = max(1, int(round(float(p["min_gap_s"]) * fps)))
        cand = [g for g in frames if abs(g - f) >= min_gap]
        if cameras is not None and cand:
            cam_f = cameras.camera(f)
            us = (np.arange(8) + 0.5) * cam_f.width / 8
            vs = (np.arange(8) + 0.5) * cam_f.height / 8
            rays = cam_f.rays(np.stack(np.meshgrid(us, vs), axis=-1).reshape(-1, 2))
            keep = []
            for g in cand:
                uv, z = cameras.camera(g).project_directions(rays)
                inside = (z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < cam_f.width) & (uv[:, 1] >= 0) \
                    & (uv[:, 1] < cam_f.height)
                if inside.mean() >= float(p["min_overlap"]):
                    keep.append(g)
            cand = keep
        k = int(p["neighbours"])
        cand.sort()
        if len(cand) > k:
            idx = np.linspace(0, len(cand) - 1, k).round().astype(int)
            nb = [cand[i] for i in idx]
        else:
            nb = cand
        if len(nb) < int(p["min_samples"]):
            nb = sorted((g for g in frames if g != f), key=lambda g: -abs(g - f))[:k]
        return nb

    def _one(self, img: np.ndarray, stack: np.ndarray, min_samples: int) -> np.ndarray:
        p = self.params
        count = np.sum(np.all(np.isfinite(stack), axis=3), axis=0)
        known = count >= min_samples
        bg = nanmedian0(stack)
        bg = np.where(known[..., None] & np.isfinite(bg), bg, img)
        diff = np.max(np.abs(img - bg), axis=2)
        d = diff[known]
        mad = float(np.median(np.abs(d - np.median(d)))) if d.size else 0.0
        thr = max(p["min_threshold"], p["threshold_sigma"] * 1.4826 * mad)
        fg = (diff > thr) & known
        if p["shadow_suppression"]:
            li = img.mean(axis=2)
            lb = np.maximum(bg.mean(axis=2), 1e-3)
            ratio = li / lb
            ci = img / np.maximum(li[..., None], 1e-3)
            cb = bg / lb[..., None]
            chroma = np.max(np.abs(ci - cb), axis=2) * np.minimum(li, lb)
            lo, hi = p["shadow_ratio"]
            fg &= ~((ratio > lo) & (ratio < hi) & (chroma < p["shadow_chroma"]))
        # Do not erode objects just because their silhouette crosses an image boundary.
        # Edge padding extends only foreground already touching that edge, not the whole border.
        opening, closing = int(p["open_px"]), int(p["close_px"])
        pad = 2 * max(opening, closing)
        if pad > 0:
            fg = np.pad(fg, pad, mode="edge")
            if opening > 0:
                fg = ndimage.binary_opening(fg, iterations=opening)
            if closing > 0:
                fg = ndimage.binary_closing(fg, iterations=closing)
            fg = fg[pad:-pad, pad:-pad]
        return ndimage.binary_fill_holes(fg)

    def foreground(self, video: Any, frames: Sequence[int], cameras: CameraTrack | None,
                   cache: _FrameCache | None = None,
                   exclude: dict[int, np.ndarray] | None = None) -> dict[int, np.ndarray]:
        """Foreground mask per frame (moving pixels, shadows removed), at the cache's scale.

        ``exclude`` maps frames to masks (at the cache's scale) of likely object pixels in
        that frame; those pixels are left out when the frame serves as a background sample for
        another frame. Where no clean sample remains, the decision with all samples is kept.
        """
        p = self.params
        frames = list(frames)
        cache = cache or _FrameCache(video, size=2 * int(p["neighbours"]) + 8)
        s = cache.factor
        fps = float(getattr(video, "fps", 30.0))
        out: dict[int, np.ndarray] = {}

        def cam(f: int) -> Any:
            c = cameras.camera(f)  # type: ignore[union-attr]
            return c.scaled(1.0 / s) if s > 1 else c

        for f in frames:
            nb = self._neighbours(f, frames, fps, cameras)
            img = cache.get(f)
            rays = cam(f).pixel_grid_rays() if cameras is not None else None
            if rays is not None and rays.shape[:2] != img.shape[:2]:
                rays = rays[: img.shape[0], : img.shape[1]]
            layers, movings = [], []
            for g in nb:
                src = cache.get(g)
                if cameras is not None:
                    warp = _Warp(cam(g), rays, src.shape[:2])
                    layers.append(warp(src))
                    if exclude is not None and g in exclude:
                        movings.append(warp(exclude[g].astype(np.float32)[..., None])[..., 0] > 0.01)
                    else:
                        movings.append(None)
                else:
                    layers.append(src)
                    movings.append(exclude[g] if exclude is not None and g in exclude else None)
            stack = np.stack(layers)
            if exclude is None:
                out[f] = self._one(img, stack, int(p["min_samples"]))
                continue
            clean = stack.copy()
            for k, mv in enumerate(movings):
                if mv is not None:
                    clean[k][mv] = np.nan
            n_clean = np.sum(np.all(np.isfinite(clean), axis=3), axis=0)
            # pixels without a clean sample fall back to all samples
            fallback = n_clean < 1
            if fallback.any():
                clean[:, fallback] = stack[:, fallback]
            out[f] = self._one(img, clean, 1)
        return out

    # ---------------------------------------------------------------- association
    def segment(self, video: Any, prompts: Sequence[ObjectPrompt], frames: Sequence[int],
                cameras: CameraTrack | None = None) -> Segmentation:
        frames = sorted(int(f) for f in frames)
        w, h = video.size
        factor = max(1, int(np.ceil(w / float(self.params["max_width"])))) if self.params["max_width"] else 1
        cache = _FrameCache(video, size=2 * int(self.params["neighbours"]) + 8, factor=factor)

        def upscale(m: np.ndarray) -> np.ndarray:
            if factor == 1:
                return m
            big = np.repeat(np.repeat(m, factor, axis=0), factor, axis=1)
            out = np.zeros((h, w), bool)
            out[: min(h, big.shape[0]), : min(w, big.shape[1])] = big[:h, :w]
            return out

        def associate(fg: dict[int, np.ndarray]) -> Segmentation:
            seg = Segmentation(w, h, provenance=[self.provenance()])
            min_area = max(1, int(self.params["min_area_px"]) // (factor * factor))
            comps = {f: [(upscale(m), tuple(v * factor for v in b)) for m, b in _components(fg[f], min_area)]
                     for f in frames}
            if prompts:
                self._assisted(seg, prompts, frames, comps, w, h)
            else:
                self._automatic(seg, frames, comps)
            return seg

        fg = self.foreground(video, frames, cameras, cache)
        seg = associate(fg)
        if self.params["second_pass"]:
            # Sparse prompts can identify a stationary object that the first pass missed.
            # Exclude its likely footprint between anchors, but still require image evidence
            # to emit a mask. No boxes are extrapolated beyond the annotation interval.
            bounds = self._prompt_bounds(prompts, frames) if self.params["prompt_exclusion"] else {}
            exclude = {}
            for f in frames:
                if prompts:
                    masks = [t.mask(f) for t in seg.tracks.values() if f in t.crops]
                    m = np.logical_or.reduce(masks) if masks else np.zeros((h, w), bool)
                else:
                    m = fg[f]
                for box in bounds.get(f, []):
                    x0, y0 = max(0, int(np.floor(box[0]))), max(0, int(np.floor(box[1])))
                    x1, y1 = min(w, int(np.ceil(box[2]))), min(h, int(np.ceil(box[3])))
                    if x1 > x0 and y1 > y0:
                        m[y0:y1, x0:x1] = True
                if factor > 1 and prompts:
                    hh, ww = h // factor, w // factor
                    m = m[: hh * factor, : ww * factor].reshape(hh, factor, ww, factor).any(axis=(1, 3))
                exclude[f] = ndimage.binary_dilation(m, iterations=3)
            seg = associate(self.foreground(video, frames, cameras, cache, exclude))
        return seg

    def _prompt_bounds(self, prompts: Sequence[ObjectPrompt], frames: Sequence[int]
                       ) -> dict[int, list[tuple[float, float, float, float]]]:
        """Coarse exclusion regions between annotations, never synthetic observed masks.

        Interpolation only chooses which background samples to ignore. If an object's
        footprint was never seen unobstructed, foreground estimation keeps its usual fallback.
        Point prompts alone do not define an area to exclude.
        """
        anchors: dict[str, dict[int, tuple[float, float, float, float]]] = {}
        for prompt in prompts:
            if prompt.box is None and prompt.polygon is None and prompt.mask is None:
                continue
            box = prompt.bbox()
            if box is None or not np.isfinite(box).all() or box[2] <= box[0] or box[3] <= box[1]:
                continue
            anchors.setdefault(prompt.object_id, {})[int(prompt.frame)] = box
        out: dict[int, list[tuple[float, float, float, float]]] = {}
        for samples in anchors.values():
            times = sorted(samples)
            boxes = np.array([samples[f] for f in times], float)
            for f in frames:
                if times[0] <= f <= times[-1]:
                    box = [float(np.interp(f, times, boxes[:, k])) for k in range(4)]
                    out.setdefault(int(f), []).append(_expand(box, float(self.params["search_margin"]), 2.0))
        return out

    def _assisted(self, seg: Segmentation, prompts: Sequence[ObjectPrompt], frames: list[int],
                  comps: dict[int, list[tuple[np.ndarray, tuple[int, int, int, int]]]], w: int, h: int) -> None:
        by_obj: dict[str, dict[int, ObjectPrompt]] = {}
        for pr in prompts:
            by_obj.setdefault(pr.object_id, {})[int(pr.frame)] = pr
        margin = float(self.params["search_margin"])
        for oid, prs in by_obj.items():
            track = seg.track(oid)
            for direction in (1, -1):
                start = min(prs) if direction > 0 else max(prs)
                seq = [f for f in frames if (f >= start if direction > 0 else f <= start)]
                if direction < 0:
                    seq = seq[::-1]
                history: list[tuple[int, tuple[float, float, float, float]]] = []
                if start not in seq:
                    # A prompt outside the requested sampling grid can seed a prediction,
                    # but only the requested frames with image evidence become observations.
                    anchor = prs[start].bbox()
                    if anchor is not None:
                        history.append((start, tuple(float(v) for v in anchor)))
                for f in seq:
                    pr = prs.get(f)
                    if pr is not None:
                        # Re-anchoring must not carry velocity across a gap or annotation jump.
                        anchor = pr.bbox()
                        history = [(f, anchor)] if anchor is not None else []
                    region = pr.region_mask(w, h) if pr is not None else None
                    if region is not None:
                        track.set(f, region, 1.0)
                        b = track.bbox(f)
                        if b is not None:
                            history = [(f, tuple(float(v) for v in b))]
                        continue
                    if pr is not None and pr.bbox() is not None:
                        pred = pr.bbox()
                        search = _expand(pred, 0.05, 2.0)
                    elif history:
                        pred = _predict_box(history, f)
                        search = _expand(pred, margin, 4.0)
                    else:
                        continue
                    cands = [(m, b) for m, b in comps.get(f, []) if bbox_iou(b, search) > 0 and
                             _overlap_frac(m, search) > 0.5]
                    if not cands:
                        continue
                    # the region that overlaps the prediction most, plus fragments that touch it
                    # (within a few pixels) and lie inside the prediction
                    tight = _expand(pred, 0.1, 2.0)
                    main = max(cands, key=lambda c: _overlap_frac(c[0], pred) * c[0].sum())
                    mb = main[1]
                    reach = max(3, int(round(0.05 * max(mb[2] - mb[0], mb[3] - mb[1]))))
                    near = ndimage.binary_dilation(main[0], iterations=reach)
                    chosen = [main[0]] + [m for m, b in cands if m is not main[0] and (m & near).any()
                                          and _overlap_frac(m, tight) > 0.8]
                    mask = np.logical_or.reduce(chosen)
                    if pr is not None and pr.box is not None:
                        x0, y0, x1, y1 = (int(round(v)) for v in pr.box)
                        clip = np.zeros_like(mask)
                        clip[max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = True
                        mask &= clip
                    b = _bbox(mask)
                    if b is None:
                        continue
                    conf = bbox_iou(b, pred)
                    # The reverse pass recovers losses from later anchors, never replaces
                    # existing forward observations or fabricates masks across a visibility gap.
                    if direction > 0 or pr is not None or f not in track.crops:
                        track.set(f, mask, conf)
                    if history and history[-1][0] == f:
                        history.pop()
                    history.append((f, b))
        # Explicit annotation evidence outranks propagated masks, including perfect prediction
        # matches. Otherwise pixels go to the object whose prediction they fit better.
        ids = list(seg.tracks)
        for f in frames:
            present = [o for o in ids if f in seg.tracks[o].crops]
            for i in range(len(present)):
                for j in range(i + 1, len(present)):
                    a, b = seg.tracks[present[i]], seg.tracks[present[j]]
                    ma, mb = a.mask(f), b.mask(f)
                    if ma is None or mb is None:
                        continue  # an earlier pair can have removed an object's entire mask
                    both = ma & mb
                    if both.any():
                        rank_a = (f in by_obj[present[i]], a.confidence.get(f, 0))
                        rank_b = (f in by_obj[present[j]], b.confidence.get(f, 0))
                        if rank_a >= rank_b:
                            b.set(f, mb & ~both, b.confidence.get(f, 0))
                        else:
                            a.set(f, ma & ~both, a.confidence.get(f, 0))

    def _automatic(self, seg: Segmentation, frames: list[int],
                   comps: dict[int, list[tuple[np.ndarray, tuple[int, int, int, int]]]]) -> None:
        tracks: list[dict[str, Any]] = []
        active: list[int] = []
        for f in frames:
            cur = comps.get(f, [])
            if active and cur:
                cost = np.ones((len(active), len(cur)))
                for i, ti in enumerate(active):
                    pb = _predict_box(tracks[ti]["boxes"], f)
                    for j, (_, b) in enumerate(cur):
                        cost[i, j] = 1.0 - bbox_iou(_expand(pb, 0.2, 2.0), b)
                rows, cols = linear_sum_assignment(cost)
                used = set()
                still = []
                for r, c in zip(rows, cols):
                    if cost[r, c] < 0.95:
                        tracks[active[r]]["masks"][f] = cur[c][0]
                        tracks[active[r]]["boxes"].append((f, cur[c][1]))
                        used.add(c)
                        still.append(active[r])
                new = [j for j in range(len(cur)) if j not in used]
                active = still
            else:
                new = list(range(len(cur)))
                active = []
            for j in new:
                tracks.append({"masks": {f: cur[j][0]}, "boxes": [(f, cur[j][1])]})
                active.append(len(tracks) - 1)
        good = [t for t in tracks if len(t["masks"]) >= int(self.params["min_track_frames"])]
        good.sort(key=lambda t: -np.median([m.sum() for m in t["masks"].values()]))
        for k, t in enumerate(good, start=1):
            tr = seg.track(f"object_{k}")
            for f, m in t["masks"].items():
                tr.set(f, m, 1.0)


def _bbox(mask: np.ndarray) -> tuple[float, float, float, float] | None:
    rows = np.nonzero(mask.any(axis=1))[0]
    if len(rows) == 0:
        return None
    cols = np.nonzero(mask.any(axis=0))[0]
    return float(cols[0]), float(rows[0]), float(cols[-1] + 1), float(rows[-1] + 1)


def _predict_box(history: Sequence[tuple[int, Sequence[float]]], frame: int) -> tuple[float, float, float, float]:
    """Constant-velocity prediction of a box from the last observations."""
    if len(history) == 1:
        return tuple(float(v) for v in history[-1][1])  # type: ignore[return-value]
    (fa, ba), (fb, bb) = history[-2], history[-1]
    dt = fb - fa
    if dt == 0:
        return tuple(float(v) for v in bb)  # type: ignore[return-value]
    step = (frame - fb) / dt
    return tuple(float(bb[i] + step * (bb[i] - ba[i])) for i in range(4))  # type: ignore[return-value]


class MaskFiles(Segmenter):
    """``baseline.files``: masks from files written by any tool (one image per object and frame).

    ``pattern`` is a path template with ``{object}`` and ``{frame}`` fields, for example
    ``masks/{object}/{frame:06d}.png``. Object ids come from the prompts or from ``objects``.
    """

    name = "baseline.files"
    summary = "Reads per-object mask images written by any external segmenter or annotation tool."

    def __init__(self, pattern: str = "masks/{object}/{frame:06d}.png", root: str = ".",
                 objects: Sequence[str] = (), threshold: int = 127, **params: Any) -> None:
        super().__init__(pattern=pattern, root=root, objects=list(objects), threshold=threshold, **params)

    def segment(self, video: Any, prompts: Sequence[ObjectPrompt], frames: Sequence[int],
                cameras: CameraTrack | None = None) -> Segmentation:
        w, h = video.size
        seg = Segmentation(w, h, provenance=[self.provenance()])
        ids = list(dict.fromkeys(list(self.params["objects"]) + [p.object_id for p in prompts]))
        root = Path(self.params["root"])
        for oid in ids:
            track = seg.track(oid)
            for f in frames:
                path = root / self.params["pattern"].format(object=oid, frame=int(f))
                if path.is_file():
                    m = read_mask(path, int(self.params["threshold"]))
                    if m.shape != (h, w):
                        raise ValueError(f"{path}: mask is {m.shape[::-1]}, video is {(w, h)}")
                    track.set(int(f), m, 1.0)
        return seg


class PromptMasks(Segmenter):
    """``baseline.prompts``: only the annotated outlines and masks (frames without one are unobserved)."""

    name = "baseline.prompts"
    summary = "Uses the keyframe outlines or masks of the annotation as the only observations."

    def segment(self, video: Any, prompts: Sequence[ObjectPrompt], frames: Sequence[int],
                cameras: CameraTrack | None = None) -> Segmentation:
        w, h = video.size
        seg = Segmentation(w, h, provenance=[self.provenance()])
        keep = set(int(f) for f in frames)
        for pr in prompts:
            if int(pr.frame) not in keep:
                continue
            m = pr.region_mask(w, h)
            if m is not None:
                seg.track(pr.object_id).set(int(pr.frame), m, 1.0)
        return seg
