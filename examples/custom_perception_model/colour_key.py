# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""An example perception model: a colour-key segmenter.

It shows everything a perception model needs and nothing more:

* subclass the interface of its kind (here :class:`~openbeam_v2e.perception.interfaces.Segmenter`);
* give it a ``name`` (unique within the kind) and a one-line ``summary``;
* implement the interface's one method and return the shared result type
  (:class:`~openbeam_v2e.perception.types.Segmentation`), with a provenance record;
* import heavy libraries lazily inside the method (this one needs only NumPy and SciPy).

The model learns each object's colour from its annotated outline, then in every frame keeps the
pixels of that colour, choosing the connected region nearest to where the object was last seen.
Colours are compared as chromaticity (each channel divided by the pixel's brightness), so the
lit and shaded faces of an object count as the same colour.
It works on footage where each moving object has a distinctive, roughly uniform colour, such as
the synthetic benchmark's clips.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from openbeam_v2e.perception.interfaces import Segmenter
from openbeam_v2e.perception.types import CameraTrack, ObjectPrompt, Segmentation


class ColourKeySegmenter(Segmenter):
    """``example.colour_key``: masks from each object's annotated colour."""

    name = "example.colour_key"
    summary = "Keeps pixels of each object's annotated colour; for footage with distinctly coloured objects."

    def __init__(self, tolerance: float = 0.06, min_pixels: int = 4, **params: Any) -> None:
        super().__init__(tolerance=tolerance, min_pixels=min_pixels, **params)

    def segment(self, video: Any, prompts: Sequence[ObjectPrompt], frames: Sequence[int],
                cameras: CameraTrack | None = None) -> Segmentation:
        from scipy import ndimage

        width, height = video.size
        seg = Segmentation(width, height, provenance=[self.provenance()])
        tolerance = float(self.params["tolerance"])
        min_pixels = int(self.params["min_pixels"])

        def chroma(image: np.ndarray) -> np.ndarray:
            rgb = image.astype(float)
            return rgb / np.maximum(rgb.sum(axis=-1, keepdims=True), 1.0)

        # 1. Each object's colour: the median chromaticity inside its first outlined region.
        colours: dict[str, np.ndarray] = {}
        seeds: dict[str, tuple[int, np.ndarray]] = {}
        for prompt in sorted(prompts, key=lambda p: p.frame):
            region = prompt.region_mask(width, height)
            if prompt.object_id in colours or region is None or not region.any():
                continue
            colours[prompt.object_id] = np.median(chroma(video.read(int(prompt.frame)))[region], axis=0)
            rows, cols = np.nonzero(region)
            seeds[prompt.object_id] = (int(prompt.frame), np.array([cols.mean(), rows.mean()]))

        # 2. Every frame: pixels close to the colour; keep the region nearest the last position,
        #    walking forwards and then backwards from the outlined frame.
        for object_id, colour in colours.items():
            seed_frame, seed_centre = seeds[object_id]
            forwards = [f for f in sorted(frames) if f >= seed_frame]
            backwards = [f for f in sorted(frames, reverse=True) if f < seed_frame]
            for walk in (forwards, backwards):
                last = seed_centre
                for f in walk:
                    close = np.linalg.norm(chroma(video.read(int(f))) - colour, axis=2) < tolerance
                    labels, count = ndimage.label(close)
                    if count == 0:
                        continue
                    index = range(1, count + 1)
                    centres = ndimage.center_of_mass(close, labels, index)
                    sizes = ndimage.sum(close, labels, index)
                    best, best_d = None, np.inf
                    for k, ((r, c), size) in zip(index, zip(centres, sizes)):
                        d = float(np.hypot(c - last[0], r - last[1]))
                        if size >= min_pixels and d < best_d:
                            best, best_d = k, d
                    if best is None:
                        continue
                    mask = labels == best
                    seg.track(object_id).set(int(f), mask, 1.0)
                    rows, cols = np.nonzero(mask)
                    last = np.array([cols.mean(), rows.mean()])
        return seg
