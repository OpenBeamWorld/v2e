# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""A small NumPy ray caster for scenes made of a ground plane and convex parts.

It renders synthetic test footage with exact ground truth (camera, poses, per-object masks) and
draws reconstructed hulls for overlays. Every pixel centre is traced through the camera model,
lens distortion included, so silhouettes are exact for convex parts: a ray hits a part where it
has entered every half-space of the part before leaving any of them.

This is a test and visualisation tool, not a production renderer: no shadows, no textures on
objects, flat shading with one directional light.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

from openbeam_v2e.reconstruct.geom import Camera
from openbeam_v2e.reconstruct.hulls import Hull

GroundShader = Callable[[np.ndarray], np.ndarray]  # world points (N, 3) -> RGB in [0, 1] (N, 3)
SkyShader = Callable[[np.ndarray], np.ndarray]     # ray directions (N, 3) -> RGB (N, 3)


@dataclass
class RenderObject:
    """One object to draw: a hull at a pose with a flat colour; ``object_id`` > 0 labels its mask."""

    object_id: int
    hull: Hull
    R_wb: np.ndarray
    position: np.ndarray
    colour: tuple[float, float, float] = (0.8, 0.2, 0.2)


@dataclass
class RenderResult:
    rgb: np.ndarray        # (H, W, 3) float in [0, 1]
    ids: np.ndarray        # (H, W) int: 0 = background (ground or sky), else object_id
    depth: np.ndarray      # (H, W) ray length in metres (inf for sky)

    def mask(self, object_id: int) -> np.ndarray:
        return self.ids == object_id

    def to_uint8(self) -> np.ndarray:
        return np.clip(np.round(self.rgb * 255.0), 0, 255).astype(np.uint8)


@dataclass
class Scene:
    """Ground plane, sky and lighting for :func:`render`."""

    ground_z: float = 0.0
    ground_shader: GroundShader | None = None
    sky_shader: SkyShader | None = None
    light_dir: Sequence[float] = (0.35, -0.45, 0.82)   # direction towards the light
    ambient: float = 0.35
    static_objects: list[RenderObject] = field(default_factory=list)


def default_sky(dirs: np.ndarray) -> np.ndarray:
    t = np.clip(dirs[:, 2], 0.0, 1.0)[:, None]
    return (1 - t) * np.array([0.78, 0.82, 0.88]) + t * np.array([0.45, 0.6, 0.85])


def default_ground(points: np.ndarray) -> np.ndarray:
    return np.tile(np.array([0.45, 0.45, 0.42]), (len(points), 1))


def _bbox_of_projection(camera: Camera, verts: np.ndarray, pad: int = 2) -> tuple[int, int, int, int] | None:
    uv, z = camera.project(verts)
    if np.any(z <= 1e-3):
        return 0, 0, camera.width, camera.height  # straddles the camera plane: trace everything
    if not np.all(np.isfinite(uv)):
        return None
    x0 = max(0, int(np.floor(uv[:, 0].min())) - pad)
    y0 = max(0, int(np.floor(uv[:, 1].min())) - pad)
    x1 = min(camera.width, int(np.ceil(uv[:, 0].max())) + pad)
    y1 = min(camera.height, int(np.ceil(uv[:, 1].max())) + pad)
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def _trace_part(origin: np.ndarray, dirs: np.ndarray, planes_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Entry distance (inf when missed) and entry plane index for rays against a convex part."""
    n = planes_world[:, :3]
    d = planes_world[:, 3]
    nd = dirs @ n.T                      # (N, P)
    no = origin @ n.T                    # (P,)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (d - no)[None, :] / nd
    entering = nd < -1e-12
    leaving = nd > 1e-12
    t_in = np.where(entering, t, -np.inf)
    t_out = np.where(leaving, t, np.inf)
    # rays parallel to a plane and outside it never hit
    parallel_out = (~entering & ~leaving) & ((no - d)[None, :] > 0)
    tin_max = t_in.max(axis=1)
    tout_min = t_out.min(axis=1)
    hit = (tin_max <= tout_min) & (tout_min > 0) & ~parallel_out.any(axis=1)
    t_hit = np.where(hit, np.maximum(tin_max, 0.0), np.inf)
    return t_hit, np.argmax(t_in, axis=1)


def render(camera: Camera, scene: Scene, objects: Sequence[RenderObject] = ()) -> RenderResult:
    """Ray-cast one frame."""
    H, W = camera.height, camera.width
    dirs = camera.pixel_grid_rays().reshape(-1, 3)
    origin = camera.centre
    depth = np.full(H * W, np.inf)
    ids = np.zeros(H * W, int)
    normal = np.zeros((H * W, 3))
    albedo = np.zeros((H * W, 3))

    # ground plane z = ground_z
    dz = dirs[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        tg = (scene.ground_z - origin[2]) / dz
    ground = (dz < -1e-9) & (tg > 0)
    depth[ground] = tg[ground]
    if np.any(ground):
        pts = origin + dirs[ground] * tg[ground, None]
        shader = scene.ground_shader or default_ground
        albedo[ground] = shader(pts)
        normal[ground] = (0.0, 0.0, 1.0)

    for obj in list(scene.static_objects) + list(objects):
        R = np.asarray(obj.R_wb, float)
        t = np.asarray(obj.position, float)
        for part in obj.hull.parts:
            verts = part.vertices @ R.T + t
            box = _bbox_of_projection(camera, verts)
            if box is None:
                continue
            x0, y0, x1, y1 = box
            rows = np.arange(y0, y1)
            cols = np.arange(x0, x1)
            idx = (rows[:, None] * W + cols[None, :]).ravel()
            n_world = part.planes[:, :3] @ R.T
            d_world = part.planes[:, 3] + n_world @ t
            planes = np.concatenate([n_world, d_world[:, None]], axis=1)
            th, k = _trace_part(origin, dirs[idx], planes)
            closer = th < depth[idx]
            if not np.any(closer):
                continue
            sel = idx[closer]
            depth[sel] = th[closer]
            ids[sel] = obj.object_id
            normal[sel] = n_world[k[closer]]
            albedo[sel] = obj.colour

    light = np.asarray(scene.light_dir, float)
    light = light / np.linalg.norm(light)
    shade = scene.ambient + (1.0 - scene.ambient) * np.clip(normal @ light, 0.0, 1.0)
    rgb = albedo * shade[:, None]
    sky = ~np.isfinite(depth)
    if np.any(sky):
        rgb[sky] = (scene.sky_shader or default_sky)(dirs[sky])
    return RenderResult(np.clip(rgb, 0, 1).reshape(H, W, 3), ids.reshape(H, W), depth.reshape(H, W))
