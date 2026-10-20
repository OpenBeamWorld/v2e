# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""Scene + templates + content catalog + world package -> ``v2e.package/1`` request.

:func:`compile_scene` is engine-free. It

1. builds level collision from the world's static meshes (``visual`` meshes are left out);
2. picks content for every dynamic actor from its template rule: a generated generic vehicle,
   the nearest stock vehicle of the connector's catalog (with per-dimension residuals), or a
   generated prop sized by its measurements, each with a content-origin record;
3. places every actor **at rest**: ``at_rest`` actors at their reconstructed pose (lifted out of
   the ground when the reconstruction sinks them), ``drive_in`` actors at the start of a
   synthesised run-up, ``moving`` passive objects only with a declared engine-native launcher;
4. sets the timeline: video frame 0 is simulated time ``t0``, the moment every drive-in actor
   reaches its frame-0 pose (actors with a shorter run-up wait at rest), or a short settle time
   when nobody drives in;
5. states what each controlled actor's program must achieve (time-indexed reference paths).

Anything the compiler cannot do honestly is collected as a :class:`Refusal`; if there is any,
:class:`CompileRefused` is raised with all of them, and no package is produced.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from openbeam_v2e import __version__
from openbeam_v2e.compile.content import (
    ContentRecord,
    Param,
    Refusal,
    generic_vehicle,
    measured_dims,
    prop_content,
    stock_vehicle,
)
from openbeam_v2e.compile.level import CollisionMesh, write_obj
from openbeam_v2e.compile.runup import A_REF_MPS2, SETTLE_M, Runup, synthesize_runup
from openbeam_v2e.schema import jsonio
from openbeam_v2e.schema.catalog import ContentCatalog, load_catalog
from openbeam_v2e.schema.common import Pose, PoseRef
from openbeam_v2e.schema.package import PACKAGE_SCHEMA, SCHEMA_ID
from openbeam_v2e.schema.scene import Actor, Scene, load_scene
from openbeam_v2e.schema.templates import TemplateRule, Templates, load_templates
from openbeam_v2e.schema.validate import check
from openbeam_v2e.world.geometry import ground_height, quat_to_matrix
from openbeam_v2e.world.io import read_world
from openbeam_v2e.world.model import ObjectTrack, World

__all__ = ["CompileRefused", "CompileResult", "compile_scene", "compile_files", "content_parameters",
           "generated_only_catalog", "SETTLE_S", "AIRBORNE_M"]

#: Simulated time between spawn and video frame 0 when no actor drives in (suspensions settle).
SETTLE_S = 0.5
#: An object whose lowest point is this far above the ground at frame 0 is released in the air.
AIRBORNE_M = 0.05
PENETRATION_M = 0.005


class CompileRefused(Exception):
    """The scene cannot be compiled honestly; ``refusals`` says why."""

    def __init__(self, refusals: list[Refusal]) -> None:
        self.refusals = list(refusals)
        super().__init__("compile refused:\n" + "\n".join(f"  {r}" for r in self.refusals))


@dataclass
class CompileResult:
    request: dict[str, Any]
    collision: list[CollisionMesh]
    content: list[ContentRecord]
    runups: dict[str, Runup] = field(default_factory=dict)

    @property
    def t0_s(self) -> float:
        return float(self.request["timeline"]["t0_s"])

    def write(self, out_dir: str | Path) -> Path:
        """Write ``package.json`` and the level OBJ files into ``out_dir``; returns the request path."""
        out = Path(out_dir)
        for mesh in self.collision:
            write_obj(mesh, out)
        return jsonio.dump(self.request, out / "package.json")


def _pose_of(ref: PoseRef | None, world: World, spec: Actor) -> tuple[Pose | None, int | None, Refusal | None]:
    if ref is None:
        return None, None, Refusal("pose_unknown", "the initial state names no pose", spec.id)
    if ref.pose is not None:
        return ref.pose, None, None
    if not world.has_object(ref.object or ""):
        return None, None, Refusal("missing_world_object", f"the world package has no object {ref.object!r}", spec.id)
    obj = world.object(ref.object or "")
    frame = int(ref.frame or 0)
    if frame >= obj.frame_count or not obj.known[frame]:
        return None, frame, Refusal("pose_unknown", f"object {ref.object!r} has no pose at frame {frame}", spec.id)
    return Pose(tuple(obj.position[frame]), tuple(obj.quat_xyzw[frame])), frame, None


def _ground_contact(obj: ObjectTrack | None, pose: Pose, vertices: np.ndarray, faces: np.ndarray) -> tuple[float, float]:
    """(penetration, gap) of the object's hull at ``pose`` against the static surface under it."""
    if obj is None or len(faces) == 0:
        return 0.0, 0.0
    R = quat_to_matrix(np.asarray(pose.quat_xyzw))
    hull = obj.hull_vertices @ R.T + np.asarray(pose.position_m)
    ground = ground_height(hull[:, :2], vertices, faces, below=hull[:, 2] + 0.3)
    ok = np.isfinite(ground)
    if not np.any(ok):
        return 0.0, 0.0
    diff = hull[ok, 2] - ground[ok]
    return float(max(0.0, -diff.min())), float(max(0.0, diff.min()))


def _observed_reference(obj: ObjectTrack, frame0: int, fps: float, t0: float) -> tuple[list, list, list]:
    """Reference-point positions, times and speeds from frame0 until the first unknown frame."""
    speeds = np.linalg.norm(obj.velocity(fps, centre=False), axis=1)
    path, times, v = [], [], []
    for k in range(frame0, obj.frame_count):
        if not obj.known[k]:
            break
        path.append(obj.position[k].tolist())
        times.append(t0 + (k - frame0) / fps)
        v.append(float(speeds[k]) if np.isfinite(speeds[k]) else 0.0)
    return path, times, v


def _arc_length(path: list[list[float]], start: float = 0.0) -> list[float]:
    s, out = start, []
    for i, p in enumerate(path):
        if i:
            s += math.dist(p, path[i - 1])
        out.append(s)
    return out


def _r(x: float, n: int = 6) -> float:
    return float(round(float(x), n))


def generated_only_catalog(backend: str) -> ContentCatalog:
    """A catalog with no stock content: every actor gets generated content (the OpenBeam default)."""
    return ContentCatalog(backend=backend, engine="", entries=())


def compile_scene(scene: Scene | str | Path, templates: Templates | str | Path | None = None,
                  catalog: ContentCatalog | str | Path | None = None, world: World | str | Path | None = None, *,
                  connector: str | None = None, out: str | Path | None = None, out_dir: str | Path | None = None,
                  accel_mps2: float = A_REF_MPS2, settle_m: float = SETTLE_M, settle_s: float = SETTLE_S,
                  name: str = "") -> CompileResult:
    """Compile a scene into an engine-neutral package request, or raise :class:`CompileRefused`.

    Inputs may be objects or file paths. Without a catalog, the connector name gets a catalog with
    no stock content (generated content only). With ``out`` (or ``out_dir``) the request and the
    level files are written there as well.
    """
    scene = scene if isinstance(scene, Scene) else load_scene(scene)
    if templates is None:
        templates = Templates()
    elif not isinstance(templates, Templates):
        templates = load_templates(templates)
    if catalog is None:
        catalog = generated_only_catalog(connector or "generic")
    elif not isinstance(catalog, ContentCatalog):
        catalog = load_catalog(catalog)
    if world is None:
        world = read_world(scene.world_path())
    elif not isinstance(world, World):
        world = read_world(world)
    result = _compile(scene, templates, catalog, world, accel_mps2, settle_m, settle_s, name)
    target = out_dir if out_dir is not None else out
    if target is not None:
        result.write(target)
    return result


def _compile(scene: Scene, templates: Templates, catalog: ContentCatalog, world: World, accel_mps2: float,
             settle_m: float, settle_s: float, name: str) -> CompileResult:
    refusals: list[Refusal] = []
    notes: list[str] = []

    # limits the connector declares
    seatable = [a for a in scene.dynamic_actors if a.seatable]
    if catalog.max_seatable_vehicles is not None and len(seatable) > catalog.max_seatable_vehicles:
        refusals.append(Refusal("seatable_limit", f"the scene has {len(seatable)} seatable vehicles; {catalog.backend} "
                                f"allows {catalog.max_seatable_vehicles}"))
    controlled = scene.controlled_actors
    if catalog.max_controlled_actors is not None and len(controlled) > catalog.max_controlled_actors:
        refusals.append(Refusal("controlled_limit", f"the scene has {len(controlled)} controlled actors; "
                                f"{catalog.backend} allows {catalog.max_controlled_actors}"))

    # level collision
    static_by_mesh = {a.mesh: a for a in scene.static_actors}
    world_meshes = {s.mesh_id for s in world.statics}
    for a in scene.static_actors:
        if a.mesh not in world_meshes:
            refusals.append(Refusal("missing_static_mesh", f"the world package has no static mesh {a.mesh}", a.id))
    collision: list[CollisionMesh] = []
    for s in sorted(world.statics, key=lambda m: m.mesh_id):
        named = static_by_mesh.get(s.mesh_id)
        role = named.role if named is not None else s.role
        if role == "visual":
            continue
        collision.append(CollisionMesh(named.id if named is not None else f"static_{s.mesh_id}", s.vertices, s.faces,
                                       "world", role, s.observed, s.mesh_id, None))
        if named is None:
            notes.append(f"static mesh {s.mesh_id} ({s.name or 'unnamed'}) is not named in the scene; kept as {role}")
    static_v, static_f = world.static_geometry()

    # content and spawn poses
    content: list[ContentRecord] = []
    spawns: dict[str, dict[str, Any]] = {}
    runups: dict[str, Runup] = {}
    released: list[str] = []
    frame0_of: dict[str, int] = {}
    frame0_hulls = []
    for a in scene.dynamic_actors:
        if world.has_object(a.object_id):
            obj = world.object(a.object_id)
            hull = obj.hull_world(0)
            if hull is not None:
                frame0_hulls.append((a.id, hull))

    for a in scene.dynamic_actors:
        obj = world.object(a.object_id) if world.has_object(a.object_id) else None
        rule = templates.rule_for(a.id, a.cls)
        template = rule.template
        dims = measured_dims(a, obj.dims_m if obj else None, obj.hull_extents() if obj else None)
        record: ContentRecord | Refusal
        if template != "vehicle.stock" and template.split(".")[0] in ("vehicle", "prop", "static") \
                and template not in catalog.generated_templates:
            record = Refusal("template_unavailable", f"{catalog.backend} cannot load generated {template!r} content", a.id)
        elif template == "vehicle.generic":
            record = generic_vehicle(a, rule, dims)
        elif template == "vehicle.stock":
            record = stock_vehicle(a, rule, dims, catalog)
            if isinstance(record, Refusal) and rule.fallback:
                fallback = TemplateRule(rule.fallback, overrides=rule.overrides, density_class=rule.density_class)
                notes.append(f"{a.id}: {record.message}; using the fallback {rule.fallback}")
                record = generic_vehicle(a, fallback, dims) if rule.fallback == "vehicle.generic" else \
                    prop_content(a, fallback, dims)
                if isinstance(record, ContentRecord):
                    record.fallback_from = "vehicle.stock"
        elif template.startswith("prop."):
            record = prop_content(a, rule, dims)
        elif template == "static.mesh":
            record = Refusal("template_unavailable", "a dynamic actor cannot use static.mesh", a.id)
        else:  # plugin template: pass the measurements through; the plugin's dialect builds it
            record = ContentRecord(a.id, a.cls, template, "generated", dict(dims), generator=template,
                                   generator_version="plugin", seatable=a.seatable)
        if isinstance(record, Refusal):
            refusals.append(record)
        else:
            content.append(record)

        state = a.initial_state
        assert state is not None
        pose, frame, problem = _pose_of(state.reference_pose, world, a)
        if problem is not None:
            refusals.append(problem)
            continue
        assert pose is not None
        frame0_of[a.id] = frame or 0
        if state.mode == "at_rest":
            penetration, gap = _ground_contact(obj, pose, static_v, static_f)
            offset = penetration if penetration > PENETRATION_M else 0.0
            if offset:
                pose = Pose((pose.position_m[0], pose.position_m[1], pose.position_m[2] + offset), pose.quat_xyzw)
                notes.append(f"{a.id}: lifted {offset:.3f} m out of the static surface")
            if gap > AIRBORNE_M:
                released.append(a.id)
                notes.append(f"{a.id}: starts {gap:.2f} m above the surface (released at frame 0)")
            spawns[a.id] = {"actor": a.id, "mode": "at_rest", "pose": pose.to_json(), "at_rest": True,
                            "initial_velocity_mps": 0, "ground_offset_m": _r(offset)}
        elif state.mode == "drive_in":
            if obj is None:
                refusals.append(Refusal("missing_world_object", "drive_in needs the reconstructed track", a.id))
                continue
            width = None
            for rec in content:
                if rec.actor == a.id:
                    width = rec.value("width_m")
            others = [(n, h) for n, h in frame0_hulls if n != a.id]
            plan = synthesize_runup(obj, world.fps, float(state.speed_mps.value), frame or 0, static_v, static_f,
                                    others, width, state.path, accel_mps2, settle_m, actor=a.id)
            if isinstance(plan, Refusal):
                refusals.append(plan)
                continue
            runups[a.id] = plan
            notes.extend(f"{a.id}: {n}" for n in plan.notes)
            if plan.ground is not None:
                collision.append(plan.ground)
            spawns[a.id] = {"actor": a.id, "mode": "drive_in", "pose": plan.start_pose.to_json(), "at_rest": True,
                            "initial_velocity_mps": 0, "frame0_pose": pose.to_json(), "ground_offset_m": 0.0}
        else:  # moving
            if state.launcher is None:
                refusals.append(Refusal(
                    "moving_without_launcher",
                    "a passive object already moving at frame 0 cannot be given speed; start the scene earlier so "
                    "the launch is visible, declare an engine-native launcher from authored content "
                    "(initial_state.launcher), or leave the object out", a.id))
                continue
            spawns[a.id] = {"actor": a.id, "mode": "moving", "pose": pose.to_json(), "at_rest": True,
                            "initial_velocity_mps": 0, "frame0_pose": pose.to_json(), "ground_offset_m": 0.0,
                            "launcher": dict(state.launcher)}
            notes.append(f"{a.id}: launched by {state.launcher['mechanism']} (native substitute; the run is emulated)")

    # timeline
    if runups and released:
        refusals.append(Refusal("released_with_drive_in",
                                "objects released in the air at frame 0 and actors that drive in would need a "
                                f"delayed spawn, which connectors cannot express yet ({', '.join(released)})"))
    if refusals:
        raise CompileRefused(refusals)
    if runups:
        t0 = max(r.natural_t0_s for r in runups.values())
    elif released:
        t0 = 0.0
    else:
        t0 = float(settle_s)

    controllers = []
    for a in scene.dynamic_actors:
        obj = world.object(a.object_id) if world.has_object(a.object_id) else None
        if not a.is_controlled:
            controllers.append({"actor": a.id, "control": a.control, "model": "passive"})
            continue
        frame0 = frame0_of.get(a.id, 0)
        if a.id in runups:
            plan = runups[a.id]
            delay = t0 - plan.natural_t0_s
            path = plan.path_m.tolist()
            times = (plan.t_s + delay).tolist()
            speeds = plan.v_mps.tolist()
            obs_path, obs_t, obs_v = _observed_reference(obj, frame0, world.fps, t0) if obj is not None else ([], [], [])
            path += obs_path[1:]
            times += obs_t[1:]
            speeds += obs_v[1:]
            ref = {"path_m": [[_r(c) for c in p] for p in path], "s_m": [_r(x) for x in _arc_length(path)],
                   "t_s": [_r(x) for x in times], "speed_mps": [_r(max(0.0, x)) for x in speeds],
                   "frame0_s_m": _r(plan.length_m), "t0_s": _r(t0), "accel_mps2": plan.accel_mps2,
                   "shape": plan.shape}
            controllers.append({"actor": a.id, "control": a.control, "model": "time_indexed_tracker", "reference": ref})
        elif obj is not None:
            obs_path, obs_t, obs_v = _observed_reference(obj, frame0, world.fps, t0)
            ref = {"path_m": [[_r(c) for c in p] for p in obs_path], "s_m": [_r(x) for x in _arc_length(obs_path)],
                   "t_s": [_r(x) for x in obs_t], "speed_mps": [_r(max(0.0, x)) for x in obs_v],
                   "frame0_s_m": 0.0, "t0_s": _r(t0)}
            controllers.append({"actor": a.id, "control": a.control, "model": "time_indexed_tracker", "reference": ref})
        else:
            controllers.append({"actor": a.id, "control": a.control, "model": "hold"})

    request: dict[str, Any] = {
        "schema": SCHEMA_ID,
        "name": name or scene.name or "scene",
        "compiler_version": __version__,
        "inputs": {"sha256": {"scene": scene.sha256(), "templates": templates.sha256(), "catalog": jsonio.canonical_sha256(catalog.to_dict())},
                   "backend": catalog.backend, "engine": catalog.engine, "world": str(scene.world)},
        "timeline": {"t0_s": _r(t0), "fps": float(world.fps), "frame_count": int(world.frame_count),
                     "settle_s": float(settle_s)},
        "level": {"frame": "z_up", "collision": [m.to_json() for m in collision]},
        "content": [c.to_json() for c in content],
        "spawns": [spawns[a.id] for a in scene.dynamic_actors if a.id in spawns],
        "controllers": controllers,
        "requires": scene.requires.to_json(),
        "notes": notes,
    }
    if collision:
        lo = np.min([m.to_json()["bounds_m"]["min"] for m in collision], axis=0)
        hi = np.max([m.to_json()["bounds_m"]["max"] for m in collision], axis=0)
        request["level"]["bounds_m"] = {"min": [_r(x) for x in lo], "max": [_r(x) for x in hi]}
    check(request, PACKAGE_SCHEMA, "package request")
    return CompileResult(request, collision, content, runups)


def compile_files(scene_path: str | Path, templates_path: str | Path | None, catalog_path: str | Path,
                  out_dir: str | Path, world_path: str | Path | None = None) -> Path:
    """File-level entry point: read the inputs, compile, write ``package.json`` and the level files."""
    result = compile_scene(scene_path, templates_path, catalog_path, world_path)
    return result.write(out_dir)


def content_parameters(result: CompileResult, actor: str) -> Mapping[str, Param]:
    """Parameters of one actor's content record (helper for dialects and tests)."""
    for record in result.content:
        if record.actor == actor:
            return record.parameters
    raise KeyError(actor)
