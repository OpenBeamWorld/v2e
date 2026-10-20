# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""Read and write V2E world packages, and read plain 4DCodeBench-layout packages.

:func:`write_world` writes the base layout (so any tool that reads the layout can use the
package) and the V2E sidecars under ``v2e/`` (schemas in :mod:`openbeam_v2e.schema.world`).
Static meshes are written for every frame; each rigid object becomes a mesh archive with its
hull in world coordinates per frame plus a dynamics archive whose columns are the hull
vertices, both over the object's known interval (gaps inside it are interpolated for the base
layout; the sidecar track keeps them as NaN).

:func:`read_world` reads a V2E package exactly. Given a plain package without sidecars, it
imports what the layout can say: static meshes, one static camera, and one rigid object per
surface-meshed dynamics archive, with a pose per frame from a rigid fit of the hull vertices
to the first frame in which they are all known. Point-only (particle) and volumetric archives
are listed in ``notes`` and skipped, since v0 scores rigid objects only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from openbeam_v2e.fourdcb.format import (
    MotionArchive,
    FormatError,
    MeshSequence,
    Package,
    read_npz,
    read_package,
    write_npz,
    write_package,
)
from openbeam_v2e.schema import jsonio
from openbeam_v2e.schema.common import Measured
from openbeam_v2e.schema.validate import check
from openbeam_v2e.schema.world import (
    CAMERA_TRACK_SCHEMA,
    OBJECTS_SCHEMA,
    OBJECTS_SCHEMA_ID,
    OUTLINES_SCHEMA,
    OUTLINES_SCHEMA_ID,
    UNCERTAINTY_SCHEMA,
    UNCERTAINTY_SCHEMA_ID,
    WORLD_SCHEMA,
    WORLD_SCHEMA_ID,
)
from openbeam_v2e.world.camera import Camera, CameraTrack
from openbeam_v2e.world.geometry import kabsch, matrix_to_quat, principal_frame, quat_to_matrix, slerp_quats
from openbeam_v2e.world.model import EventInterval, ObjectTrack, StaticMesh, World, intervals_from_mask

__all__ = ["write_world", "read_world", "fill_gaps", "to_base_package"]


def fill_gaps(position: np.ndarray, quat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate unknown poses inside the known interval (linear position, spherical attitude)."""
    known = np.all(np.isfinite(position), axis=1) & np.all(np.isfinite(quat), axis=1)
    idx = np.flatnonzero(known)
    p, q = position.copy(), quat.copy()
    if len(idx) < 2:
        return p, q
    inside = np.arange(idx[0], idx[-1] + 1)
    missing = inside[~known[inside]]
    if len(missing):
        for axis in range(3):
            p[missing, axis] = np.interp(missing, idx, position[idx, axis])
        q[missing] = slerp_quats(idx.astype(float), quat[idx], missing.astype(float))
    return p, q


def _assign_mesh_ids(world: World) -> dict[str, int]:
    used = {s.mesh_id for s in world.statics}
    if len(used) != len(world.statics):
        raise ValueError("static mesh ids must be unique")
    ids: dict[str, int] = {}
    for obj in world.objects:
        if obj.mesh_id is not None:
            if obj.mesh_id in used:
                raise ValueError(f"mesh id {obj.mesh_id} of object {obj.id!r} is already used")
            ids[obj.id] = obj.mesh_id
            used.add(obj.mesh_id)
    next_id = 1
    for obj in world.objects:
        if obj.id in ids:
            continue
        while next_id in used:
            next_id += 1
        ids[obj.id] = next_id
        used.add(next_id)
    for obj_id, mesh_id in ids.items():
        if mesh_id > 0xFFFF:
            raise ValueError(f"mesh id {mesh_id} of object {obj_id!r} does not fit the layout's uint16 ids")
    return ids


def to_base_package(world: World) -> tuple[Package, dict[str, int]]:
    """The base-layout view of a world (camera at frame 0, meshes and dynamics archives)."""
    cam0 = world.camera_track.camera(0)
    pkg = Package(cam0.K, cam0.extrinsic)
    F = world.frame_count
    for s in world.statics:
        pkg.meshes[s.mesh_id] = MeshSequence.static(s.mesh_id, s.vertices, s.faces, F)
    mesh_ids = _assign_mesh_ids(world)
    for obj in world.objects:
        interval = obj.known_interval()
        if interval is None:
            continue
        p, q = fill_gaps(obj.position, obj.quat_xyzw)
        a, b = interval
        R = quat_to_matrix(q[a:b + 1])
        world_v = np.einsum("fij,vj->fvi", R, obj.hull_vertices) + p[a:b + 1, None, :]
        mesh_id = mesh_ids[obj.id]
        faces = obj.hull_faces.astype(np.int32)
        pkg.meshes[mesh_id] = MeshSequence(mesh_id, a, [v.astype(np.float32) for v in world_v], [faces] * (b - a + 1))
        pos = np.full((F, len(obj.hull_vertices), 3), np.nan, dtype=np.float32)
        pos[a:b + 1] = world_v.astype(np.float32)
        pkg.dynamics[obj.id] = MotionArchive(obj.id, pos, np.array([mesh_id], dtype=np.uint16), faces)
    return pkg, mesh_ids


def _measured_json(dims: dict[str, Measured]) -> dict[str, Any]:
    return {k: v.to_json() for k, v in dims.items()}


def write_world(world: World, directory: str | Path, compressed: bool = True) -> Path:
    """Write ``world`` as a base-layout package plus V2E sidecars."""
    root = Path(directory)
    pkg, mesh_ids = to_base_package(world)
    write_package(pkg, root, compressed=compressed, check=True)
    side = root / "v2e"
    tracks = side / "tracks"
    tracks.mkdir(parents=True, exist_ok=True)
    for stale in tracks.glob("*.npz"):
        stale.unlink()
    manifest: dict[str, Any] = {
        "schema": WORLD_SCHEMA_ID, "kind": world.kind, "fps": float(world.fps), "frame_count": int(world.frame_count),
        "image_size": [int(world.image_size[0]), int(world.image_size[1])], "frame": "z_up", "units": "m",
    }
    if world.source_video:
        manifest["source_video"] = dict(world.source_video)
    if world.models:
        manifest["models"] = [dict(m) for m in world.models]
    if world.notes:
        manifest["notes"] = world.notes
    check(manifest, WORLD_SCHEMA, "v2e/world.json")
    jsonio.dump(manifest, side / "world.json")
    jsonio.dump(world.camera_track.to_json(), side / "camera_track.json")

    objects = []
    for obj in world.objects:
        entry: dict[str, Any] = {"id": obj.id, "class": obj.cls, "symmetry": obj.symmetry_class}
        if obj.dims_m:
            entry["dims_m"] = _measured_json(obj.dims_m)
        if obj.mass_kg is not None:
            entry["mass_kg"] = obj.mass_kg.to_json()
        entry["mesh_id"] = mesh_ids[obj.id]
        entry["dynamics"] = f"dynamics/{obj.id}.npz"
        entry["track"] = f"v2e/tracks/{obj.id}.npz"
        entry["visibility"] = [list(iv) for iv in intervals_from_mask(obj.visible)]
        if obj.annotation:
            entry["annotation"] = dict(obj.annotation)
        objects.append(entry)
        write_npz(tracks / f"{obj.id}.npz", {
            "position": obj.position.astype("<f8"),
            "quat_xyzw": obj.quat_xyzw.astype("<f8"),
            "visible": obj.visible.astype("<u1"),
            "sigma_position_m": obj.sigma_position_m.astype("<f8"),
            "sigma_attitude_rad": obj.sigma_attitude_rad.astype("<f8"),
            "hull_vertices": obj.hull_vertices.astype("<f8"),
            "hull_faces": obj.hull_faces.astype("<i4"),
        }, compressed)
    statics = []
    for s in world.statics:
        item: dict[str, Any] = {"mesh_id": s.mesh_id, "role": s.role, "observed": bool(s.observed)}
        if s.name:
            item["name"] = s.name
        if s.sigma_m is not None:
            item["sigma_m"] = float(s.sigma_m)
        statics.append(item)
    doc = {"schema": OBJECTS_SCHEMA_ID, "objects": objects, "statics": statics}
    check(doc, OBJECTS_SCHEMA, "v2e/objects.json")
    jsonio.dump(doc, side / "objects.json")

    unc: dict[str, Any] = {"schema": UNCERTAINTY_SCHEMA_ID}
    if world.camera_sigma_px is not None:
        sigma = world.camera_sigma_px
        unc["camera"] = {"sigma_px": float(sigma) if np.ndim(sigma) == 0 else [float(x) for x in np.asarray(sigma)]}
    unc["events"] = [ev.to_json() for ev in world.events]
    unc["residuals"] = {"fitted": dict(world.residuals.get("fitted", {})),
                        "held_out": dict(world.residuals.get("held_out", {}))}
    check(unc, UNCERTAINTY_SCHEMA, "v2e/uncertainty.json")
    jsonio.dump(unc, side / "uncertainty.json")

    outline_path = side / "outlines.json"
    if world.outlines:
        doc = {"schema": OUTLINES_SCHEMA_ID, "image_size": list(world.image_size),
               "objects": {oid: {str(f): [np.asarray(p, dtype=float).tolist() for p in polys]
                                 for f, polys in sorted(frames.items())}
                           for oid, frames in world.outlines.items()}}
        check(doc, OUTLINES_SCHEMA, "v2e/outlines.json")
        jsonio.dump(doc, outline_path)
    elif outline_path.exists():
        outline_path.unlink()
    return root


def _camera_from_base(pkg: Package, frame_count: int) -> CameraTrack:
    K, E = np.asarray(pkg.intrinsics, dtype=float), np.asarray(pkg.extrinsic, dtype=float)
    width, height = max(1, int(round(2.0 * K[0, 2]))), max(1, int(round(2.0 * K[1, 2])))
    cam = Camera(width, height, float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2]), E[:3, :3], E[:3, 3])
    return CameraTrack.static(cam, frame_count)


def _static_from_base(pkg: Package, mesh_id: int, role: str = "collision", name: str = "",
                      sigma: float | None = None, observed: bool = True) -> StaticMesh:
    seq = pkg.meshes[mesh_id]
    return StaticMesh(mesh_id, np.asarray(seq.vertices[0], dtype=float), np.asarray(seq.faces[0]), name, role, sigma,
                      observed)


def _import_rigid(d: MotionArchive, F: int) -> ObjectTrack | None:
    pos = np.asarray(d.pos, dtype=float)
    complete = np.all(np.isfinite(pos), axis=(1, 2))
    if d.faces is None or pos.shape[1] < 3 or not np.any(complete):
        return None
    first = int(np.flatnonzero(complete)[0])
    R0, c0, extents = principal_frame(pos[first])
    hull = (pos[first] - c0) @ R0
    position = np.full((F, 3), np.nan)
    quat = np.full((F, 4), np.nan)
    worst = 0.0
    for t in np.flatnonzero(complete):
        R, trans, rms = kabsch(hull, pos[t])
        position[t] = trans
        quat[t] = matrix_to_quat(R)
        worst = max(worst, rms)
    dims = {"length": Measured(float(extents[0])), "width": Measured(float(extents[1])),
            "height": Measured(float(extents[2]))}
    return ObjectTrack(d.name, "prop.mesh", hull, np.asarray(d.faces), position, quat, dims_m=dims,
                       mesh_id=int(d.ids[0]) if len(d.ids) else None,
                       annotation={"source": "imported from a base-layout dynamics archive",
                                   "rigid_fit_rms_max_m": round(worst, 6)})


def read_world(directory: str | Path, fps: float | None = None) -> World:
    """Read a V2E world package, or import a plain base-layout package (``fps`` required then)."""
    root = Path(directory)
    pkg = read_package(root)
    side = root / "v2e"
    if not (side / "world.json").is_file():
        if fps is None:
            raise FormatError(f"{root} has no v2e/world.json; pass fps to import a plain package")
        F = pkg.frame_count()
        if F <= 0:
            raise FormatError(f"{root} holds no frames")
        notes, objects = [], []
        dynamic_ids = pkg.dynamic_mesh_ids()
        for name, d in sorted(pkg.dynamics.items()):
            obj = _import_rigid(d, F)
            if obj is None:
                notes.append(f"dynamics/{name}.npz skipped: not a rigid surface mesh")
            else:
                objects.append(obj)
        statics = [_static_from_base(pkg, mid) for mid in sorted(pkg.meshes) if mid not in dynamic_ids]
        return World(float(fps), F, _camera_from_base(pkg, F), statics, objects, kind="imported",
                            notes="; ".join(notes))

    manifest = jsonio.load(side / "world.json")
    check(manifest, WORLD_SCHEMA, "v2e/world.json")
    F = int(manifest["frame_count"])
    if (side / "camera_track.json").is_file():
        raw = jsonio.load(side / "camera_track.json")
        check(raw, CAMERA_TRACK_SCHEMA, "v2e/camera_track.json")
        camera_track = CameraTrack.from_json(raw)
    else:
        camera_track = _camera_from_base(pkg, F)
    objects_doc = jsonio.load(side / "objects.json") if (side / "objects.json").is_file() else {
        "schema": OBJECTS_SCHEMA_ID, "objects": []}
    check(objects_doc, OBJECTS_SCHEMA, "v2e/objects.json")
    objects = []
    for entry in objects_doc["objects"]:
        members = read_npz(root / entry["track"])
        visible = members.get("visible")
        objects.append(ObjectTrack(
            entry["id"], entry["class"], members["hull_vertices"], members["hull_faces"], members["position"],
            members["quat_xyzw"], None if visible is None else visible.astype(bool),
            members.get("sigma_position_m"), members.get("sigma_attitude_rad"),
            {k: Measured.parse(v) for k, v in entry.get("dims_m", {}).items()}, entry.get("symmetry"),
            Measured.parse(entry["mass_kg"]) if "mass_kg" in entry else None, entry["mesh_id"],
            dict(entry.get("annotation", {}))))
    object_meshes = {o.mesh_id for o in objects} | pkg.dynamic_mesh_ids()
    listed = {s["mesh_id"]: s for s in objects_doc.get("statics", [])}
    statics = []
    for mesh_id in sorted(pkg.meshes):
        if mesh_id in object_meshes:
            continue
        info = listed.get(mesh_id, {})
        statics.append(_static_from_base(pkg, mesh_id, info.get("role", "collision"), info.get("name", ""),
                                         info.get("sigma_m"), info.get("observed", True)))
    events, residuals, camera_sigma = [], {"fitted": {}, "held_out": {}}, None
    if (side / "uncertainty.json").is_file():
        unc = jsonio.load(side / "uncertainty.json")
        check(unc, UNCERTAINTY_SCHEMA, "v2e/uncertainty.json")
        events = [EventInterval.from_json(e) for e in unc.get("events", [])]
        residuals = {"fitted": dict(unc.get("residuals", {}).get("fitted", {})),
                     "held_out": dict(unc.get("residuals", {}).get("held_out", {}))}
        sigma = unc.get("camera", {}).get("sigma_px")
        camera_sigma = None if sigma is None else (float(sigma) if np.ndim(sigma) == 0 else np.asarray(sigma, float))
    outlines: dict[str, dict[int, list[np.ndarray]]] = {}
    if (side / "outlines.json").is_file():
        doc = jsonio.load(side / "outlines.json")
        check(doc, OUTLINES_SCHEMA, "v2e/outlines.json")
        outlines = {oid: {int(f): [np.asarray(p, dtype=float) for p in polys] for f, polys in frames.items()}
                    for oid, frames in doc["objects"].items()}
    return World(float(manifest["fps"]), F, camera_track, statics, objects, events, manifest["kind"],
                        list(manifest.get("models", [])), residuals, camera_sigma, outlines,
                        dict(manifest.get("source_video", {})), manifest.get("notes", ""))
