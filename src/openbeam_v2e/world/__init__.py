# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""The V2E world: one model for reconstructions, run exports and synthetic truth.

* :mod:`.model`: :class:`World` (fps, frame count, camera track, static meshes, rigid
  object tracks with per-frame sigma, events, residuals, observed outlines);
* :mod:`.camera`: pinhole cameras with radial distortion and the per-frame track of a
  fixed-position pan/tilt/zoom camera;
* :mod:`.io`: read and write world packages (the 4DCodeBench layout plus V2E sidecars), and
  import plain packages made by other tools;
* :mod:`.geometry`, :mod:`.raster`: rotations, rigid fits, simple meshes, distance queries
  and silhouette rasterisation;
* :mod:`.timeline`, :mod:`.export`: simulated versus video time, the frame-0 trigger, and the
  export of a ``v2e-runlog/1`` run onto the reference timeline.
"""

from __future__ import annotations

from openbeam_v2e.world.camera import Camera, CameraTrack, look_at_rotation
from openbeam_v2e.world.export import export_run, runlog_to_world
from openbeam_v2e.world.io import read_world, write_world
from openbeam_v2e.world.model import EventInterval, ObjectTrack, StaticMesh, World

__all__ = [
    "Camera",
    "CameraTrack",
    "EventInterval",
    "ObjectTrack",
    "StaticMesh",
    "World",
    "export_run",
    "look_at_rotation",
    "read_world",
    "runlog_to_world",
    "write_world",
]
