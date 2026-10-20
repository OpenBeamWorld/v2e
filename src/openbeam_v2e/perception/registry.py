# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""Registry of perception models, built-in and from other packages.

Models are looked up by ``(kind, name)``. Built-ins are registered by import path, so listing
them imports nothing heavy. Other packages add models through the ``openbeam_v2e.perception``
entry point group; each entry point names a :class:`~openbeam_v2e.perception.interfaces.PerceptionModel`
subclass, whose ``kind`` and ``name`` attributes say where it goes::

    # pyproject.toml of a third-party package
    [project.entry-points."openbeam_v2e.perception"]
    my_depth = "my_package.depth:MyDepthModel"

At run time, :func:`register` adds a class directly (useful in notebooks and tests).
"""

from __future__ import annotations

import importlib
import logging
from importlib import metadata
from typing import Any

from openbeam_v2e.perception.interfaces import INTERFACES, KINDS, PerceptionModel

log = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "openbeam_v2e.perception"

_BASE = "openbeam_v2e.perception.baseline"
_ADAPT = "openbeam_v2e.perception.adapters"

#: (kind, name) -> "module:Class" for the models that ship with the package.
BUILTINS: dict[tuple[str, str], str] = {
    ("camera_calibration", "baseline.points"): f"{_BASE}.calibration:PointCalibrator",
    ("camera_calibration", "baseline.vanishing"): f"{_BASE}.calibration:VanishingPointCalibrator",
    ("camera_calibration", "baseline.auto"): f"{_BASE}.calibration:AutoCalibrator",
    ("camera_tracking", "baseline.rotation_zoom"): f"{_BASE}.camera_tracking:RotationZoomTracker",
    ("camera_tracking", "baseline.static"): f"{_BASE}.camera_tracking:StaticCamera",
    ("camera_tracking", "baseline.point_tracks"): f"{_BASE}.camera_tracking:PointTrackRotationZoom",
    ("point_tracking", "baseline.lk"): f"{_BASE}.point_tracking:LucasKanadeTracker",
    ("segmentation", "baseline.background"): f"{_BASE}.segmentation:BackgroundSubtraction",
    ("segmentation", "baseline.files"): f"{_BASE}.segmentation:MaskFiles",
    ("segmentation", "baseline.prompts"): f"{_BASE}.segmentation:PromptMasks",
    ("depth", "baseline.ground_plane"): f"{_BASE}.depth:GroundPlaneDepth",
    ("pose", "baseline.silhouette"): f"{_BASE}.pose:SilhouettePoseEstimator",
    ("segmentation", "sam2"): f"{_ADAPT}.sam2:Sam2Segmenter",
    ("point_tracking", "cotracker"): f"{_ADAPT}.cotracker:CoTrackerPointTracker",
    ("point_tracking", "trackeverything"): f"{_ADAPT}.trackeverything:TrackEverythingPointTracker",
    ("depth", "hf.depth"): f"{_ADAPT}.hf_depth:TransformersDepth",
    ("detection", "hf.detection"): f"{_ADAPT}.hf_detection:TransformersDetector",
}

_runtime: dict[tuple[str, str], type[PerceptionModel]] = {}
_entry_points: dict[tuple[str, str], Any] | None = None


def register(cls: type[PerceptionModel]) -> type[PerceptionModel]:
    """Register a model class (usable as a decorator)."""
    if cls.kind not in INTERFACES:
        raise ValueError(f"unknown perception kind {cls.kind!r}; expected one of {KINDS}")
    if not cls.name:
        raise ValueError("a perception model needs a name")
    if not issubclass(cls, INTERFACES[cls.kind]):
        raise TypeError(f"{cls.__qualname__} must subclass {INTERFACES[cls.kind].__qualname__}")
    _runtime[(cls.kind, cls.name)] = cls
    return cls


def unregister(kind: str, name: str) -> None:
    _runtime.pop((kind, name), None)


def _load_path(path: str) -> type[PerceptionModel]:
    module, _, attr = path.partition(":")
    return getattr(importlib.import_module(module), attr)


def _discover() -> dict[tuple[str, str], Any]:
    global _entry_points
    if _entry_points is None:
        found: dict[tuple[str, str], Any] = {}
        try:
            eps = metadata.entry_points(group=ENTRY_POINT_GROUP)
        except TypeError:  # pragma: no cover - Python < 3.10 API
            eps = metadata.entry_points().get(ENTRY_POINT_GROUP, [])  # type: ignore[attr-defined]
        for ep in eps:
            try:
                cls = ep.load()
                key = (cls.kind, cls.name)
                if not issubclass(cls, INTERFACES.get(cls.kind, object)):
                    raise TypeError("not a perception model of its declared kind")
                found[key] = cls
            except Exception as exc:  # a broken plugin must not break the registry
                log.warning("perception entry point %s failed to load: %s", ep.name, exc)
        _entry_points = found
    return _entry_points


def get(kind: str, name: str) -> type[PerceptionModel]:
    """The model class registered as ``(kind, name)``."""
    if (kind, name) in _runtime:
        return _runtime[(kind, name)]
    if (kind, name) in BUILTINS:
        return _load_path(BUILTINS[(kind, name)])
    plugins = _discover()
    if (kind, name) in plugins:
        return plugins[(kind, name)]
    known = sorted(n for k, n in names() if k == kind)
    raise KeyError(f"no {kind} model named {name!r}; known: {known}")


def create(kind: str, name: str, **params: Any) -> PerceptionModel:
    """Instantiate a registered model, checking that its libraries are importable first."""
    cls = get(kind, name)
    ok, why = cls.check()
    if not ok:
        raise RuntimeError(f"{kind} model {name!r} is not available: {why}")
    return cls(**params)


def names() -> list[tuple[str, str]]:
    keys = set(BUILTINS) | set(_runtime) | set(_discover())
    return sorted(keys)


def describe(kind: str | None = None) -> list[dict[str, Any]]:
    """JSON-ready descriptions of the registered models (kind, name, extras, availability)."""
    out = []
    for k, n in names():
        if kind is not None and k != kind:
            continue
        try:
            out.append(get(k, n).describe())
        except Exception as exc:  # report broken entries instead of failing the listing
            out.append({"kind": k, "name": n, "available": False, "status": f"failed to load: {exc}"})
    return out
