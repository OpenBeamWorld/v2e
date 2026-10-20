# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""Assisted annotation: boxes, outlines, masks and clicks per object on chosen frames.

Annotation files use the ``v2e.annotations/1`` format::

    {"schema": "v2e.annotations/1",
     "image_size": [1280, 720],
     "objects": {
       "car_1": {"class": "vehicle.car",
                 "prompts": [
                   {"frame": 0,  "box": [412, 300, 655, 410]},
                   {"frame": 30, "polygon": [[500, 300], [700, 300], [700, 400], [500, 400]]},
                   {"frame": 45, "mask": "masks/car_1_000045.png"},
                   {"frame": 60, "points": [[610, 350]], "labels": [1]}]}}}

Paths are relative to the annotation file. A ``v2e.outlines/1`` document (observed outlines per
frame) is accepted too and read as outline prompts. Any tool that writes this format, a
labelling program, a detector or a person, can drive assisted reconstruction.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from openbeam_v2e.perception.types import ObjectPrompt
from openbeam_v2e.reconstruct.video import read_mask, write_png

SCHEMA = "v2e.annotations/1"


@dataclass
class Annotations:
    prompts: list[ObjectPrompt] = field(default_factory=list)
    classes: dict[str, str] = field(default_factory=dict)
    image_size: tuple[int, int] | None = None

    def objects(self) -> list[str]:
        return list(dict.fromkeys([p.object_id for p in self.prompts] + list(self.classes)))

    def for_object(self, object_id: str) -> list[ObjectPrompt]:
        return [p for p in self.prompts if p.object_id == object_id]


def _prompt(oid: str, kf: Mapping[str, Any], base: Path) -> ObjectPrompt:
    if "frame" not in kf:
        raise ValueError(f"annotation prompt of {oid!r} needs a frame")
    mask = None
    if kf.get("mask") is not None:
        m = kf["mask"]
        mask = read_mask(base / m) if isinstance(m, str) else np.asarray(m, bool)
    return ObjectPrompt(
        oid, int(kf["frame"]),
        box=None if kf.get("box") is None else tuple(float(v) for v in kf["box"]),  # type: ignore[arg-type]
        polygon=None if kf.get("polygon") is None else np.asarray(kf["polygon"], float),
        mask=mask,
        points=None if kf.get("points") is None else np.asarray(kf["points"], float),
        labels=None if kf.get("labels") is None else np.asarray(kf["labels"], int))


def parse_annotations(doc: Any, base: str | Path = ".") -> Annotations:
    """Read annotations from a parsed document (see the module docstring)."""
    base = Path(base)
    out = Annotations()
    if isinstance(doc, Mapping) and doc.get("schema") == "v2e.outlines/1":
        out.image_size = tuple(doc.get("image_size", ())) or None  # type: ignore[assignment]
        for oid, frames in doc.get("objects", {}).items():
            for f, polys in frames.items():
                if polys:
                    biggest = max(polys, key=lambda p: len(p))
                    out.prompts.append(ObjectPrompt(oid, int(f), polygon=np.asarray(biggest, float)))
        return out
    if isinstance(doc, Mapping):
        if doc.get("schema", SCHEMA) != SCHEMA:
            raise ValueError(f"unsupported annotation format {doc.get('schema')!r}")
        if doc.get("image_size"):
            out.image_size = (int(doc["image_size"][0]), int(doc["image_size"][1]))
        for oid, entry in doc.get("objects", {}).items():
            if entry.get("class"):
                out.classes[oid] = str(entry["class"])
            for kf in entry.get("prompts", []):
                out.prompts.append(_prompt(oid, kf, base))
        return out
    if isinstance(doc, Sequence):
        for kf in doc:
            oid = kf.get("object") or kf.get("object_id")
            if not oid:
                raise ValueError("each annotation entry in a list needs 'object'")
            out.prompts.append(_prompt(str(oid), kf, base))
            if kf.get("class"):
                out.classes[str(oid)] = str(kf["class"])
        return out
    raise ValueError("annotations must be a v2e.annotations/1 object, a v2e.outlines/1 object or a list")


def load_annotations(path: str | Path) -> Annotations:
    p = Path(path)
    return parse_annotations(json.loads(p.read_text()), p.parent)


def save_annotations(ann: Annotations, path: str | Path, write_masks: bool = True) -> Path:
    """Write annotations as ``v2e.annotations/1``; masks go next to the file as PNG."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    objs: dict[str, Any] = {}
    for pr in ann.prompts:
        entry = objs.setdefault(pr.object_id, {"prompts": []})
        if pr.object_id in ann.classes:
            entry["class"] = ann.classes[pr.object_id]
        kf: dict[str, Any] = {"frame": int(pr.frame)}
        if pr.box is not None:
            kf["box"] = [float(v) for v in pr.box]
        if pr.polygon is not None:
            kf["polygon"] = np.asarray(pr.polygon, float).tolist()
        if pr.points is not None:
            kf["points"] = np.asarray(pr.points, float).tolist()
            if pr.labels is not None:
                kf["labels"] = np.asarray(pr.labels, int).tolist()
        if pr.mask is not None and write_masks:
            rel = Path("masks") / f"{pr.object_id}_{int(pr.frame):06d}.png"
            (p.parent / rel).parent.mkdir(parents=True, exist_ok=True)
            write_png(p.parent / rel, np.asarray(pr.mask, bool))
            kf["mask"] = rel.as_posix()
        entry["prompts"].append(kf)
    doc: dict[str, Any] = {"schema": SCHEMA, "objects": objs}
    if ann.image_size:
        doc["image_size"] = list(ann.image_size)
    p.write_text(json.dumps(doc, indent=1))
    return p
