# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""Template hulls: the shapes reconstruction fits to object outlines.

A hull is a union of convex parts in the **body frame** (+x forward, +y left, +z up, origin at
the centre of the hull's bottom face, so a hull resting on flat ground at ``z = 0`` has its
origin on the ground). Convex parts make two operations exact and cheap:

* the silhouette of a part is the convex hull of its projected vertices, so the outline of the
  whole object is a union of convex polygons;
* a ray hits a part where it enters all of the part's half-spaces, which is what the synthetic
  renderer uses.

Templates are parameterised by measured dimensions (length along x, width along y, height along
z). Built-in templates: ``box``, ``vehicle`` (lower body plus a cabin with sloped screens),
``cylinder`` (axis along z) and ``sphere``. Each declares the symmetry class (``none``, ``box``,
``cylinder``, ``cylinder_symmetric`` or ``sphere``, as in the ``v2e.objects/1`` sidecar) that
scoring uses to compare attitudes; for ``box`` the scorer derives the exact rotation group from
the measured dimensions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import numpy as np
from scipy.spatial import ConvexHull


@dataclass(frozen=True, eq=False)
class ConvexPart:
    """A convex polytope given by its vertices (body frame)."""

    vertices: np.ndarray
    planes: np.ndarray = field(repr=False, compare=False, default=None)  # (P, 4): n . x <= d as [n, d]

    def __post_init__(self) -> None:
        v = np.asarray(self.vertices, float).reshape(-1, 3)
        object.__setattr__(self, "vertices", v)
        if self.planes is None:
            hull = ConvexHull(v)
            eq = hull.equations  # n . x + c <= 0 inside
            planes = np.concatenate([eq[:, :3], -eq[:, 3:]], axis=1)
            # merge near-duplicate facets (triangulated quads)
            keep: list[np.ndarray] = []
            for p in planes:
                if not any(np.allclose(p, k, atol=1e-9) for k in keep):
                    keep.append(p)
            object.__setattr__(self, "planes", np.array(keep))

    def triangles(self) -> tuple[np.ndarray, np.ndarray]:
        """Outward-wound triangles of the part's convex hull: (vertices, faces)."""
        hull = ConvexHull(self.vertices)
        faces = []
        centroid = self.vertices.mean(axis=0)
        for simplex in hull.simplices:
            a, b, c = self.vertices[simplex]
            n = np.cross(b - a, c - a)
            faces.append(simplex if np.dot(n, a - centroid) >= 0 else simplex[::-1])
        return self.vertices, np.array(faces, int)


@dataclass(frozen=True, eq=False)
class Hull:
    """A template hull at given dimensions."""

    template: str
    dims: tuple[float, float, float]  # length (x), width (y), height (z), metres
    parts: tuple[ConvexPart, ...]
    symmetry: str
    params: Mapping[str, float] = field(default_factory=dict)

    @property
    def length(self) -> float:
        return self.dims[0]

    @property
    def width(self) -> float:
        return self.dims[1]

    @property
    def height(self) -> float:
        return self.dims[2]

    def vertices(self) -> np.ndarray:
        return np.concatenate([p.vertices for p in self.parts])

    def world_parts(self, R_wb: Any, position: Any) -> list[np.ndarray]:
        """Vertices of every part in world coordinates for the pose (R_wb, position)."""
        R = np.asarray(R_wb, float)
        t = np.asarray(position, float)
        return [p.vertices @ R.T + t for p in self.parts]

    def lowest_point(self, R_wb: Any, position: Any) -> float:
        """World z of the lowest hull vertex at the pose."""
        return float(min(v[:, 2].min() for v in self.world_parts(R_wb, position)))

    def volume(self) -> float:
        return float(sum(ConvexHull(p.vertices).volume for p in self.parts))

    def to_dict(self) -> dict[str, Any]:
        return {"template": self.template, "dims_m": list(self.dims), "symmetry": self.symmetry,
                "params": dict(self.params)}


# --------------------------------------------------------------------------- built-in templates

def _box_vertices(x0: float, x1: float, y0: float, y1: float, z0: float, z1: float) -> np.ndarray:
    return np.array([[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)], float)


def box_hull(length: float, width: float, height: float) -> Hull:
    L, W, H = float(length), float(width), float(height)
    part = ConvexPart(_box_vertices(-L / 2, L / 2, -W / 2, W / 2, 0.0, H))
    return Hull("box", (L, W, H), (part,), "box")


VEHICLE_DEFAULTS: dict[str, float] = {
    # Proportions of a generic passenger car; every value is a ratio of the overall dimensions.
    "body_height": 0.55,      # top of the lower body / overall height
    "cabin_rear": -0.38,      # rear end of the cabin roof base / length (from the centre)
    "cabin_front": 0.22,      # front end of the cabin base / length
    "roof_inset_rear": 0.08,  # how far the roof's rear edge sits in from the base / length
    "roof_inset_front": 0.14, # how far the roof's front edge sits in from the base / length
    "cabin_width": 0.86,      # cabin width / overall width
}


def vehicle_hull(length: float, width: float, height: float, **ratios: float) -> Hull:
    """A generic road vehicle: a lower body box plus a cabin with sloped front and rear screens."""
    p = dict(VEHICLE_DEFAULTS)
    unknown = set(ratios) - set(p)
    if unknown:
        raise ValueError(f"unknown vehicle hull parameters: {sorted(unknown)}")
    p.update({k: float(v) for k, v in ratios.items()})
    L, W, H = float(length), float(width), float(height)
    zb = p["body_height"] * H
    body = ConvexPart(_box_vertices(-L / 2, L / 2, -W / 2, W / 2, 0.0, zb))
    xr, xf = p["cabin_rear"] * L, p["cabin_front"] * L
    xr_top, xf_top = xr + p["roof_inset_rear"] * L, xf - p["roof_inset_front"] * L
    if not (xr < xr_top < xf_top < xf):
        raise ValueError("vehicle hull ratios give an empty roof")
    wc = p["cabin_width"] * W / 2
    cabin = ConvexPart(np.array([[x, y, z]
                                 for (x, z) in ((xr, zb), (xf, zb), (xr_top, H), (xf_top, H))
                                 for y in (-wc, wc)], float))
    return Hull("vehicle", (L, W, H), (body, cabin), "none", p)


def cylinder_hull(length: float, width: float, height: float, segments: int = 24) -> Hull:
    """An elliptic cylinder with its axis along body z (length and width are the diameters)."""
    a = np.linspace(0.0, 2.0 * math.pi, int(segments), endpoint=False)
    ring = np.stack([0.5 * length * np.cos(a), 0.5 * width * np.sin(a)], axis=1)
    v = np.concatenate([np.c_[ring, np.zeros(len(a))], np.c_[ring, np.full(len(a), height)]])
    # a round cylinder's two ends look alike, so its axis sign is ambiguous too
    sym = "cylinder_symmetric" if abs(length - width) <= 0.02 * max(length, width) else "box"
    return Hull("cylinder", (float(length), float(width), float(height)), (ConvexPart(v),), sym)


def sphere_hull(length: float, width: float, height: float, rings: int = 10, segments: int = 20) -> Hull:
    """An ellipsoid resting on its lowest point (a sphere when all diameters are equal)."""
    pts = [[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]]
    for i in range(1, rings):
        th = math.pi * i / rings
        for j in range(segments):
            ph = 2.0 * math.pi * j / segments
            pts.append([math.sin(th) * math.cos(ph), math.sin(th) * math.sin(ph), -math.cos(th)])
    u = np.array(pts)
    v = u * np.array([length, width, height]) / 2.0 + np.array([0.0, 0.0, height / 2.0])
    equal = max(length, width, height) - min(length, width, height) <= 0.02 * max(length, width, height)
    return Hull("sphere", (float(length), float(width), float(height)), (ConvexPart(v),),
                "sphere" if equal else "box")


TEMPLATES: dict[str, Callable[..., Hull]] = {
    "box": box_hull,
    "vehicle": vehicle_hull,
    "cylinder": cylinder_hull,
    "sphere": sphere_hull,
}


def make_hull(template: str, dims: Any, **params: float) -> Hull:
    """Build a registered template hull at ``dims`` = (length, width, height)."""
    try:
        factory = TEMPLATES[template]
    except KeyError:
        raise ValueError(f"unknown hull template {template!r}; known: {sorted(TEMPLATES)}") from None
    L, W, H = (float(x) for x in dims)
    if min(L, W, H) <= 0:
        raise ValueError(f"hull dimensions must be positive, got {dims}")
    return factory(L, W, H, **params)


def register_template(name: str, factory: Callable[..., Hull]) -> None:
    """Register a further hull template (``factory(length, width, height, **params) -> Hull``)."""
    TEMPLATES[name] = factory
