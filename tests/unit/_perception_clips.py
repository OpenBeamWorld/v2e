# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""Synthetic clips shared by the perception and reconstruction tests (rendered once per session).

Small images and few frames keep the suite fast on a laptop CPU; the scenes are the
first-principles generators in :mod:`openbeam_v2e.reconstruct.synthetic`.
"""

from __future__ import annotations

from functools import lru_cache

from openbeam_v2e.reconstruct import synthetic


@lru_cache(maxsize=None)
def car_ptz() -> synthetic.SyntheticClip:
    """A car through a curve, filmed by a panning, zooming camera with a little roll."""
    return synthetic.ground_vehicle_clip(width=320, height=180, n_frames=40, seed=0)


@lru_cache(maxsize=None)
def car_ptz_distorted() -> synthetic.SyntheticClip:
    return synthetic.ground_vehicle_clip(width=320, height=180, n_frames=24, k1=-0.08, seed=1)


@lru_cache(maxsize=None)
def car_static() -> synthetic.SyntheticClip:
    """A still camera; the car crosses most of the view (two seconds), as background models need."""
    return synthetic.ground_vehicle_clip(width=256, height=144, n_frames=60, ptz=False, roll_deg=0.0, seed=2)


@lru_cache(maxsize=None)
def box_airborne() -> synthetic.SyntheticClip:
    """A tumbling box thrown through the air in front of a still camera."""
    return synthetic.airborne_box_clip(width=320, height=240, n_frames=20, seed=3)
