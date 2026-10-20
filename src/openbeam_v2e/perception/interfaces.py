# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""The model-agnostic perception interfaces.

Reconstruction asks for a capability, never for a specific model. Each capability is an abstract
class with one method; any model that implements it can be swapped in through the registry
(:mod:`openbeam_v2e.perception.registry`), including models from other packages registered
under the ``openbeam_v2e.perception`` entry point group.

=====================  ==========================  ===========================================
kind                   interface                   method
=====================  ==========================  ===========================================
camera_calibration     :class:`CameraCalibrator`   ``calibrate(inputs) -> CalibrationResult``
camera_tracking        :class:`CameraTracker`      ``track(video, reference, frames, ...)``
detection              :class:`Detector`           ``detect(image, classes) -> [Detection]``
segmentation           :class:`Segmenter`          ``segment(video, prompts, frames, ...)``
point_tracking         :class:`PointTracker`       ``track(video, queries, frames)``
depth                  :class:`DepthEstimator`     ``estimate(image, camera) -> DepthMap``
pose                   :class:`PoseEstimator`      ``fit(masks, cameras, spec, ...)``
=====================  ==========================  ===========================================

A model declares ``name`` (unique within its kind; built-ins use ``baseline.*``), the pip extras
it needs (``extras``) and how to check that its libraries are importable (``check``). Models must
import heavy libraries lazily, inside their methods, so that listing models never imports them.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar, Mapping, Sequence

import numpy as np

from openbeam_v2e import __version__
from openbeam_v2e.perception.types import (
    CameraTrack,
    DepthMap,
    Detection,
    MaskTrack,
    ModelProvenance,
    ObjectPrompt,
    ObjectSpec,
    PointTracks,
    PoseTrack,
    Segmentation,
)
from openbeam_v2e.reconstruct.geom import Camera, Intrinsics

KINDS = ("camera_calibration", "camera_tracking", "detection", "segmentation", "point_tracking",
         "depth", "pose")


class PerceptionModel(abc.ABC):
    """Base class of every perception model."""

    kind: ClassVar[str] = ""
    name: ClassVar[str] = ""
    version: ClassVar[str] = __version__
    extras: ClassVar[tuple[str, ...]] = ()          # pip extras, e.g. ("sam2",)
    requires: ClassVar[tuple[str, ...]] = ()        # importable modules the model needs
    summary: ClassVar[str] = ""

    def __init__(self, **params: Any) -> None:
        self.params: dict[str, Any] = dict(params)

    @classmethod
    def check(cls) -> tuple[bool, str]:
        """Whether the model's libraries are importable (without importing them)."""
        import importlib.util

        missing = [m for m in cls.requires if importlib.util.find_spec(m) is None]
        if missing:
            hint = f" (pip install 'openbeam-v2e[{','.join(cls.extras)}]')" if cls.extras else ""
            return False, f"missing {', '.join(missing)}{hint}"
        return True, "ok"

    def provenance(self, **extra: Any) -> ModelProvenance:
        cls = type(self)
        fields: dict[str, Any] = {"weights": None, "weights_sha256": None, "libraries": {}}
        fields.update(extra)
        return ModelProvenance(cls.name, cls.kind, cls.version, f"{cls.__module__}:{cls.__qualname__}",
                               fields["weights"], fields["weights_sha256"], dict(self.params),
                               dict(fields["libraries"]))

    @classmethod
    def describe(cls) -> dict[str, Any]:
        ok, why = cls.check()
        return {"kind": cls.kind, "name": cls.name, "version": cls.version, "extras": list(cls.extras),
                "available": ok, "status": why, "summary": cls.summary,
                "implementation": f"{cls.__module__}:{cls.__qualname__}"}


# --------------------------------------------------------------------------- calibration

@dataclass
class CalibrationInputs:
    """Evidence for calibrating the reference frame's camera. Any subset may be given."""

    width: int
    height: int
    image: np.ndarray | None = None                       # RGB frame (for automatic estimators)
    world_points: np.ndarray | None = None                # (N, 3)
    image_points: np.ndarray | None = None                # (N, 2)
    point_sigma_px: float = 1.0
    lines_x: np.ndarray | None = None                     # (N, 4) segments along world X on the ground
    lines_y: np.ndarray | None = None                     # (N, 4) segments along world Y on the ground
    camera_height_m: float | None = None
    known_distance: tuple[Sequence[float], Sequence[float], float] | None = None  # two ground pixels, metres
    origin_pixel: Sequence[float] | None = None
    intrinsics: Intrinsics | None = None                  # known or guessed intrinsics
    ground_z: float = 0.0
    free_distortion: bool = False
    exclude_mask: np.ndarray | None = None                # pixels to ignore (moving objects)


class CameraCalibrator(PerceptionModel):
    kind = "camera_calibration"

    @abc.abstractmethod
    def calibrate(self, inputs: CalibrationInputs) -> Any:
        """Return a :class:`openbeam_v2e.reconstruct.calibrate.CalibrationResult`."""


# --------------------------------------------------------------------------- camera tracking

ExcludeFn = Callable[[int], "np.ndarray | None"]


class CameraTracker(PerceptionModel):
    kind = "camera_tracking"

    @abc.abstractmethod
    def track(self, video: Any, reference: Camera, frames: Sequence[int],
              exclude: ExcludeFn | None = None) -> CameraTrack:
        """Track a camera that stays at ``reference.centre`` over ``frames``.

        ``reference`` is the camera of ``frames[0]`` from calibration; ``exclude(frame)`` may return a
        mask of pixels to ignore (moving objects).
        """


# --------------------------------------------------------------------------- 2-D perception

class Detector(PerceptionModel):
    kind = "detection"

    @abc.abstractmethod
    def detect(self, image: np.ndarray, classes: Sequence[str] | None = None) -> list[Detection]:
        """Detect objects in one RGB image."""


class Segmenter(PerceptionModel):
    kind = "segmentation"

    @abc.abstractmethod
    def segment(self, video: Any, prompts: Sequence[ObjectPrompt], frames: Sequence[int],
                cameras: CameraTrack | None = None) -> Segmentation:
        """Masks per object per frame. ``prompts`` name the objects (assisted annotation)."""


class PointTracker(PerceptionModel):
    kind = "point_tracking"

    @abc.abstractmethod
    def track(self, video: Any, queries: np.ndarray, frames: Sequence[int]) -> PointTracks:
        """Track query points ``(frame, u, v)`` (N, 3) through ``frames``."""


class DepthEstimator(PerceptionModel):
    kind = "depth"

    @abc.abstractmethod
    def estimate(self, image: np.ndarray, camera: Camera | None = None) -> DepthMap:
        """Depth for one RGB image."""


# --------------------------------------------------------------------------- pose

DepthFn = Callable[[int], "DepthMap | None"]


@dataclass
class PoseContext:
    """Scene facts a pose estimator may use."""

    ground_z: float = 0.0
    fps: float = 30.0
    heldout: Sequence[int] = ()                    # observed frames to leave out of the fit
    depth: DepthFn | None = None
    options: Mapping[str, Any] = field(default_factory=dict)


class PoseEstimator(PerceptionModel):
    kind = "pose"

    @abc.abstractmethod
    def fit(self, masks: MaskTrack, cameras: CameraTrack, spec: ObjectSpec, frames: Sequence[int],
            context: PoseContext) -> PoseTrack:
        """Fit a rigid pose track (with covariance and dimensions) to an object's masks."""


INTERFACES: Mapping[str, type[PerceptionModel]] = {
    "camera_calibration": CameraCalibrator,
    "camera_tracking": CameraTracker,
    "detection": Detector,
    "segmentation": Segmenter,
    "point_tracking": PointTracker,
    "depth": DepthEstimator,
    "pose": PoseEstimator,
}
