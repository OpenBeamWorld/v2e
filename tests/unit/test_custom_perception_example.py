# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""The custom perception model example registers and segments like a built-in model."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from openbeam_v2e.perception import registry
from openbeam_v2e.perception.interfaces import Segmenter
from openbeam_v2e.perception.types import ObjectPrompt
from openbeam_v2e.reconstruct.synthetic import airborne_box_clip

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "custom_perception_model" / "colour_key.py"


def _load():
    spec = importlib.util.spec_from_file_location("v2e_example_colour_key", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_colour_key_segmenter_registers_and_tracks_the_object() -> None:
    module = _load()
    cls = module.ColourKeySegmenter
    assert issubclass(cls, Segmenter) and cls.kind == "segmentation"
    registry.register(cls)
    try:
        assert registry.get("segmentation", "example.colour_key") is cls
        model = registry.create("segmentation", "example.colour_key")
        clip = airborne_box_clip(width=160, height=120, n_frames=12, noise_sigma=0.0)
        truth = clip.masks["box"]
        prompt = ObjectPrompt("box", 0, mask=truth[0])
        seg = model.segment(clip.frame_source(), [prompt], list(range(clip.n_frames)))
        track = seg.track("box")
        ious = []
        for f in range(clip.n_frames):
            mask = track.mask(f)
            if mask is None or not truth[f].any():
                continue
            ious.append(np.count_nonzero(mask & truth[f]) / np.count_nonzero(mask | truth[f]))
        assert len(ious) >= clip.n_frames - 2
        assert np.mean(ious) > 0.8
        assert seg.provenance[0].model == "example.colour_key"
    finally:
        registry.unregister("segmentation", "example.colour_key")
