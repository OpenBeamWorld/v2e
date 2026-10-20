# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""Per-clip reconstruction configuration (``v2e.reconstruct/1``).

Everything that is specific to one clip lives here, never in code: the frame range, the
calibration evidence, the objects and their classes, which perception model runs each stage and
with which parameters, held-out frames, and event thresholds. Example::

    {"schema": "v2e.reconstruct/1",
     "video": {"path": "clip.mp4", "fps": 30, "frames": [0, 240], "stride": 1},
     "ground_z": 0.0,
     "camera": {
       "calibration": {"model": "baseline.points", "frame": 0, "sigma_px": 1.0,
                       "points": [{"world": [0, 0, 0], "image": [612.4, 388.0]}, ...]},
       "tracking": {"model": "baseline.rotation_zoom"}},
     "segmentation": {"model": "baseline.background"},
     "depth": null,
     "pose": {"model": "baseline.silhouette"},
     "detection": null,
     "objects": [{"id": "car_1", "class": "vehicle.car",
                  "dims_prior": {"length": [4.6, 0.4]}}],
     "annotations": "annotations.json",
     "holdout": {"every": 5, "offset": 2},
     "events": {"rest_speed_mps": 0.2},
     "cache": ".cache/v2e"}

``detection`` (for example ``{"model": "hf.detection", "classes": ["car"]}``) seeds prompts
automatically on the calibration frame when there are no annotations.

Calibration evidence can instead be ``lines_x``/``lines_y`` segments with ``camera_height_m``
or ``known_distance`` (``baseline.vanishing``), only a scale fact (``baseline.auto``), or a
complete ``camera`` (intrinsics and pose) when it is already known.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

SCHEMA = "v2e.reconstruct/1"

#: Stage names accepted in ``models`` overrides, mapped to perception kinds.
STAGE_ALIASES: dict[str, str] = {
    "calibration": "camera_calibration", "camera_calibration": "camera_calibration",
    "camera": "camera_tracking", "tracking": "camera_tracking", "camera_tracking": "camera_tracking",
    "masks": "segmentation", "segmentation": "segmentation", "segmenter": "segmentation",
    "depth": "depth",
    "pose": "pose",
    "points": "point_tracking", "point_tracking": "point_tracking",
    "detection": "detection", "detector": "detection",
}

#: The model each kind uses when a configuration says only "baseline".
DEFAULT_MODELS: dict[str, str] = {
    "camera_calibration": "baseline.points",
    "camera_tracking": "baseline.rotation_zoom",
    "segmentation": "baseline.background",
    "depth": "baseline.ground_plane",
    "pose": "baseline.silhouette",
    "point_tracking": "baseline.lk",
}

SHORT_NAMES: dict[tuple[str, str], str] = {
    ("camera_tracking", "static"): "baseline.static",
    ("segmentation", "files"): "baseline.files",
    ("segmentation", "prompts"): "baseline.prompts",
    ("segmentation", "annotations"): "baseline.prompts",
    ("camera_calibration", "points"): "baseline.points",
    ("camera_calibration", "vanishing"): "baseline.vanishing",
    ("camera_calibration", "auto"): "baseline.auto",
}


def resolve_model_name(kind: str, name: str) -> str:
    if name in ("baseline", "default", ""):
        return DEFAULT_MODELS[kind]
    return SHORT_NAMES.get((kind, name), name)


@dataclass
class StageConfig:
    model: str
    params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def parse(cls, kind: str, raw: Any, default: str | None) -> "StageConfig | None":
        if raw is None:
            return None if default is None else cls(default)
        if isinstance(raw, str):
            return cls(resolve_model_name(kind, raw))
        if isinstance(raw, Mapping):
            name = raw.get("model", default or "baseline")
            return cls(resolve_model_name(kind, str(name)), dict(raw.get("params", {})))
        raise ValueError(f"stage {kind}: expected a model name or an object, got {raw!r}")


@dataclass
class ObjectConfig:
    id: str
    object_class: str = "vehicle.car"
    template: str | None = None
    template_params: dict[str, float] = field(default_factory=dict)
    motion: str | None = None
    dims_prior: dict[str, tuple[float, float]] = field(default_factory=dict)
    fit_dims: bool = True

    @classmethod
    def parse(cls, raw: Mapping[str, Any]) -> "ObjectConfig":
        if "id" not in raw:
            raise ValueError("every object needs an id")
        prior = {k: (float(v[0]), float(v[1])) for k, v in dict(raw.get("dims_prior", {})).items()}
        motion = raw.get("motion")
        if motion not in (None, "ground", "free"):
            raise ValueError(f"object {raw['id']!r}: motion must be 'ground' or 'free'")
        return cls(str(raw["id"]), str(raw.get("class", "vehicle.car")), raw.get("template"),
                   {k: float(v) for k, v in dict(raw.get("template_params", {})).items()}, motion, prior,
                   bool(raw.get("fit_dims", True)))


@dataclass
class ReconstructionConfig:
    video_path: Path | None = None
    fps: float | None = None
    frame_range: tuple[int, int | None] = (0, None)
    stride: int = 1
    ground_z: float = 0.0
    calibration: dict[str, Any] = field(default_factory=dict)
    calibration_stage: StageConfig | None = None
    tracking: StageConfig | None = None
    segmentation: StageConfig | None = None
    depth: StageConfig | None = None
    pose: StageConfig | None = None
    detection: StageConfig | None = None
    detection_classes: list[str] = field(default_factory=list)
    objects: list[ObjectConfig] = field(default_factory=list)
    default_class: str = "vehicle.car"
    annotations: Any = None            # path, document or None
    holdout_every: int = 5
    holdout_offset: int = 2
    holdout_frames: list[int] | None = None
    events: dict[str, float] = field(default_factory=dict)
    ballistic: bool = True
    cache: Path | None = None
    write_masks: bool = False
    base_dir: Path = field(default_factory=lambda: Path("."))
    raw: dict[str, Any] = field(default_factory=dict)
    ballistic_max_chi2: float | None = 4.0

    def path(self, p: Any) -> Path:
        q = Path(p)
        return q if q.is_absolute() else self.base_dir / q

    @classmethod
    def parse(cls, raw: Mapping[str, Any], base_dir: str | Path = ".") -> "ReconstructionConfig":
        if raw.get("schema", SCHEMA) != SCHEMA:
            raise ValueError(f"unsupported configuration format {raw.get('schema')!r} (expected {SCHEMA})")
        base = Path(base_dir)
        cfg = cls(base_dir=base, raw=dict(raw))
        video = raw.get("video") or {}
        if isinstance(video, str):
            video = {"path": video}
        if video.get("path"):
            cfg.video_path = cfg.path(video["path"])
        cfg.fps = None if video.get("fps") is None else float(video["fps"])
        fr = video.get("frames")
        if fr is not None:
            cfg.frame_range = (int(fr[0]), None if len(fr) < 2 or fr[1] is None else int(fr[1]))
        cfg.stride = max(1, int(video.get("stride", 1)))
        cfg.ground_z = float(raw.get("ground_z", 0.0))
        cam = raw.get("camera") or {}
        cal = dict(cam.get("calibration") or {})
        cfg.calibration = cal
        default_cal = None
        if "camera" not in cal:
            if cal.get("points"):
                default_cal = "baseline.points"
            elif cal.get("lines_x") and cal.get("lines_y"):
                default_cal = "baseline.vanishing"
            else:
                default_cal = "baseline.auto"
        cfg.calibration_stage = StageConfig.parse("camera_calibration", cal.get("model"), default_cal) \
            if default_cal or cal.get("model") else None
        cfg.tracking = StageConfig.parse("camera_tracking", cam.get("tracking", "baseline"), "baseline.rotation_zoom")
        cfg.segmentation = StageConfig.parse("segmentation", raw.get("segmentation", "baseline"), "baseline.background")
        cfg.depth = StageConfig.parse("depth", raw.get("depth"), None)
        cfg.pose = StageConfig.parse("pose", raw.get("pose", "baseline"), "baseline.silhouette")
        det = raw.get("detection")
        cfg.detection = StageConfig.parse("detection", det, None)
        if isinstance(det, Mapping):
            cfg.detection_classes = [str(c) for c in det.get("classes", [])]
        cfg.objects = [ObjectConfig.parse(o) for o in raw.get("objects", [])]
        ids = [o.id for o in cfg.objects]
        if len(set(ids)) != len(ids):
            raise ValueError("object ids must be unique")
        cfg.default_class = str(raw.get("default_class", "vehicle.car"))
        ann = raw.get("annotations")
        cfg.annotations = cfg.path(ann) if isinstance(ann, str) else ann
        hold = raw.get("holdout", {"every": 5, "offset": 2})
        if hold is None or hold is False:
            cfg.holdout_every = 0
        elif "frames" in hold:
            cfg.holdout_frames = [int(f) for f in hold["frames"]]
        else:
            cfg.holdout_every = int(hold.get("every", 5))
            cfg.holdout_offset = int(hold.get("offset", 2))
        cfg.events = {k: float(v) for k, v in dict(raw.get("events", {})).items()}
        physics = dict(raw.get("physics", {}))
        cfg.ballistic = bool(physics.get("ballistic_segments", True))
        limit = physics.get("ballistic_max_chi2", 4.0)
        cfg.ballistic_max_chi2 = None if limit is None else float(limit)
        if cfg.ballistic_max_chi2 is not None and (not math.isfinite(cfg.ballistic_max_chi2)
                                                   or cfg.ballistic_max_chi2 <= 0):
            raise ValueError("ballistic chi-square limit must be positive and finite, or null")
        cache = raw.get("cache")
        cfg.cache = cfg.path(cache) if cache else None
        cfg.write_masks = bool(dict(raw.get("output", {})).get("masks", False))
        return cfg

    @classmethod
    def load(cls, path: str | Path) -> "ReconstructionConfig":
        p = Path(path)
        return cls.parse(json.loads(p.read_text()), p.parent)

    def apply_models(self, models: Mapping[str, str] | None) -> None:
        """Override stage models, e.g. ``{"masks": "sam2", "camera": "static"}``."""
        for stage, name in (models or {}).items():
            kind = STAGE_ALIASES.get(stage)
            if kind is None:
                raise ValueError(f"unknown stage {stage!r}; known: {sorted(STAGE_ALIASES)}")
            sc = StageConfig(resolve_model_name(kind, str(name)))
            if kind == "camera_calibration":
                self.calibration_stage = sc
            elif kind == "camera_tracking":
                self.tracking = sc
            elif kind == "segmentation":
                self.segmentation = sc
            elif kind == "depth":
                self.depth = sc
            elif kind == "pose":
                self.pose = sc
            elif kind == "detection":
                self.detection = sc
            elif kind == "point_tracking":
                # point trackers feed camera tracking: track the camera from their tracks
                self.tracking = StageConfig("baseline.point_tracks", {"point_tracker": sc.model})
            else:
                raise ValueError(f"stage {stage!r} is not configurable in the reconstruction pipeline")
