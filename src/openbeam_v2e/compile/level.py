# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""Level collision from the static meshes of a world package.

Each static mesh becomes one collision entry written as a Wavefront OBJ file (Z-up, metres,
one-based indices), a format every engine toolchain can convert. Synthesised geometry (the
run-up ground of actors that drive in) is a separate entry marked ``observed: false``; it is
never used by a score.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from openbeam_v2e.world.geometry import mesh_bounds

__all__ = ["CollisionMesh", "obj_text", "write_obj"]


@dataclass
class CollisionMesh:
    id: str
    vertices: np.ndarray
    faces: np.ndarray
    source: str = "world"            # "world" | "runup"
    role: str = "collision"
    observed: bool = True
    mesh_id: int | None = None
    actor: str | None = None

    @property
    def file(self) -> str:
        return f"level/{self.id}.obj"

    def text(self) -> str:
        return obj_text(self.vertices, self.faces, f"V2E level collision '{self.id}' ({self.source}), Z-up metres")

    def to_json(self) -> dict[str, Any]:
        lo, hi = mesh_bounds(self.vertices)
        return {
            "id": self.id,
            "source": self.source,
            "actor": self.actor,
            "mesh_id": self.mesh_id,
            "role": self.role,
            "observed": self.observed,
            "file": self.file,
            "vertex_count": int(len(self.vertices)),
            "face_count": int(len(self.faces)),
            "bounds_m": {"min": [round(float(x), 6) for x in lo], "max": [round(float(x), 6) for x in hi]},
            "sha256": hashlib.sha256(self.text().encode("utf-8")).hexdigest(),
        }


def obj_text(vertices: Any, faces: Any, comment: str = "") -> str:
    """Deterministic OBJ text (six decimals, one-based face indices)."""
    v = np.asarray(vertices, dtype=float).reshape(-1, 3)
    f = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    lines = [f"# {comment}"] if comment else []
    lines += [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in v]
    lines += [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in f]
    return "\n".join(lines) + "\n"


def write_obj(mesh: CollisionMesh, out_dir: str | Path) -> Path:
    target = Path(out_dir) / mesh.file
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(mesh.text(), encoding="utf-8")
    return target
