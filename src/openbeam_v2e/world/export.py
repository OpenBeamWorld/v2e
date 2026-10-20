# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""Export a run (``v2e-runlog/1``) as a world package on the footage's own timeline.

The run's rows are resampled at the simulated times of the video frames (``t0 + frame / fps``),
so the export lines up frame for frame with the reconstruction and can be scored against it.
The camera track and static geometry are taken from the reference world (the run is seen
through the reconstructed camera), each object keeps the hull of the content that ran (or the
reference hull when none is given), and events the connector reported become event intervals.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from openbeam_v2e.schema.runlog import Runlog, read_runlog
from openbeam_v2e.schema.scene import infer_event_kind
from openbeam_v2e.world.io import read_world, write_world
from openbeam_v2e.world.model import EventInterval, ObjectTrack, StaticMesh, World
from openbeam_v2e.world.timeline import resample_poses, video_times

__all__ = ["runlog_to_world", "export_run"]


def runlog_to_world(runlog: Runlog, reference: World, t0_s: float | None = None,
                    actor_objects: Mapping[str, str] | None = None,
                    hulls: Mapping[str, tuple[Any, Any]] | None = None,
                    notes: str = "") -> World:
    """Resample ``runlog`` onto ``reference``'s frames and return a ``run_export`` world.

    ``actor_objects`` maps run actor ids to object ids of the reference (default: the same id).
    ``hulls`` gives the body-frame hull ``(vertices, faces)`` of the content each actor ran.
    """
    t0 = t0_s if t0_s is not None else (runlog.header.t0_s if runlog.header.t0_s is not None else 0.0)
    fps, F = reference.fps, reference.frame_count
    t_video = video_times(F, fps, t0)
    mapping = dict(actor_objects or {})
    hulls = dict(hulls or {})
    objects: list[ObjectTrack] = []
    for actor in runlog.header.actors:
        object_id = mapping.get(actor, actor)
        rows = [(row.t, row.actors[actor]) for row in runlog.rows if actor in row.actors]
        if not rows:
            continue
        if any(state.q is None for _, state in rows):
            raise ValueError(f"cannot export actor {actor!r}: attitude was not observed; "
                             "a world package needs measured poses, not an assumed orientation")
        t = np.array([r[0] for r in rows])
        p = np.array([r[1].p for r in rows], dtype=float)
        q = np.array([r[1].q for r in rows], dtype=float)
        position, quat = resample_poses(t, p, q, t_video)
        ref = reference.object(object_id) if reference.has_object(object_id) else None
        if actor in hulls:
            hv, hf = hulls[actor]
        elif ref is not None:
            hv, hf = ref.hull_vertices, ref.hull_faces
        else:
            raise ValueError(f"no hull for run actor {actor!r}: pass hulls or map it to a reference object")
        objects.append(ObjectTrack(
            object_id, ref.cls if ref is not None else "prop.mesh", hv, hf, position, quat,
            dims_m=dict(ref.dims_m) if ref is not None else {}, symmetry=ref.symmetry if ref is not None else None,
            annotation={"source": "run export", "run_actor": actor}))
    events: list[EventInterval] = []
    seen: set[str] = set()
    for t, event in runlog.events():
        name = event["name"]
        if name in seen:
            continue
        seen.add(name)
        frame = int(round((t - t0) * fps))
        actors = tuple(mapping.get(a, a) for a in (event.get("actor"), event.get("other")) if a)
        events.append(EventInterval(name, (frame, frame), infer_event_kind(name), actors, "reported by the connector"))
    statics = [StaticMesh(s.mesh_id, s.vertices, s.faces, s.name, s.role, s.sigma_m, s.observed)
               for s in reference.statics]
    header = runlog.header
    note = notes or f"run export: backend={header.backend or 'unknown'} regime={header.regime or 'unknown'} t0_s={t0:g}"
    return World(fps, F, reference.camera_track, statics, objects, events, kind="run_export", notes=note)


def export_run(runlog_path: str | Path, reference_dir: str | Path, out_dir: str | Path,
               t0_s: float | None = None, actor_objects: Mapping[str, str] | None = None) -> Path:
    """Read a run log and a reference world package, write the run as a world package."""
    reference = read_world(reference_dir)
    world = runlog_to_world(read_runlog(runlog_path), reference, t0_s, actor_objects)
    return write_world(world, out_dir)
