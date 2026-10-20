# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""Result types shared by every perception model.

Every model, whether the CPU baseline or an adapter around a large network, returns these types,
so reconstruction never depends on which model produced a result. Each result carries a
:class:`ModelProvenance` record (model name, version, weights and their hash, parameters) and,
where the model can say anything about it, an uncertainty.

Types:

* :class:`CameraTrack`: intrinsics, a fixed camera position and per-frame pan, tilt, roll and
  focal length with sigmas;
* :class:`ObjectPrompt`: assisted annotation for one object in one frame (box, polygon, mask or
  points);
* :class:`MaskTrack` and :class:`Segmentation`: per-object masks per frame, stored as crops;
* :class:`Detection`: one detected object (box, label, score, optional mask);
* :class:`PointTracks`: 2-D point tracks with visibility and an optional per-point confidence;
* :class:`DepthMap`: per-pixel depth, metric or relative;
* :class:`ObjectSpec` and :class:`PoseTrack`: what to fit for an object and the fitted rigid
  pose track with covariance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from openbeam_v2e.reconstruct.geom import Camera, Intrinsics, ptz_rotation


def _plain(x: Any) -> Any:
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, Mapping):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    return x


@dataclass(frozen=True)
class ModelProvenance:
    """Which model produced a result, with enough detail to reproduce it."""

    model: str                                   # registry name, e.g. "baseline.rotation_zoom"
    kind: str                                    # capability, e.g. "camera_tracking"
    version: str                                 # version of the model code
    implementation: str = ""                     # "package.module:Class"
    weights: str | None = None                   # model id or weights file name
    weights_sha256: str | None = None
    params: Mapping[str, Any] = field(default_factory=dict)
    libraries: Mapping[str, str] = field(default_factory=dict)  # third-party library versions

    def to_dict(self) -> dict[str, Any]:
        return {"model": self.model, "kind": self.kind, "version": self.version,
                "implementation": self.implementation, "weights": self.weights,
                "weights_sha256": self.weights_sha256, "params": _plain(dict(self.params)),
                "libraries": dict(self.libraries)}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ModelProvenance":
        return cls(d["model"], d["kind"], d["version"], d.get("implementation", ""), d.get("weights"),
                   d.get("weights_sha256"), dict(d.get("params", {})), dict(d.get("libraries", {})))


# --------------------------------------------------------------------------- camera

@dataclass
class CameraTrack:
    """A camera at one position: per-frame pan, tilt, roll (radians) and focal length (pixels).

    ``intrinsics`` holds the image size, principal point and lens distortion shared by all
    frames (its ``f`` is the focal length of the first frame). ``sigma`` holds one-sigma values
    per frame for (pan, tilt, roll, focal).
    """

    intrinsics: Intrinsics
    centre: np.ndarray
    frames: np.ndarray
    pan: np.ndarray
    tilt: np.ndarray
    roll: np.ndarray
    focal: np.ndarray
    sigma: np.ndarray
    centre_sigma: np.ndarray = field(default_factory=lambda: np.zeros(3))
    quality: np.ndarray | None = None           # per frame: inlier count or similar
    provenance: list[ModelProvenance] = field(default_factory=list)
    #: Covariance (7x7) of the reference calibration over (pan, tilt, roll [rad], centre x, y, z [m],
    #: focal [px]); it moves every frame together and is propagated into object poses.
    calibration_cov: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.centre = np.asarray(self.centre, float)
        self.frames = np.asarray(self.frames, int)
        for name in ("pan", "tilt", "roll", "focal"):
            setattr(self, name, np.asarray(getattr(self, name), float))
        self.sigma = np.asarray(self.sigma, float).reshape(len(self.frames), 4)
        self.centre_sigma = np.asarray(self.centre_sigma, float)
        self._index = {int(f): i for i, f in enumerate(self.frames)}

    def __len__(self) -> int:
        return len(self.frames)

    def index(self, frame: int) -> int:
        return self._index[int(frame)]

    def has(self, frame: int) -> bool:
        return int(frame) in self._index

    def camera(self, frame: int) -> Camera:
        i = self.index(frame)
        return Camera(self.intrinsics.with_focal(float(self.focal[i])),
                      ptz_rotation(self.pan[i], self.tilt[i], self.roll[i]), self.centre)

    def cameras(self) -> list[Camera]:
        return [self.camera(int(f)) for f in self.frames]

    @classmethod
    def from_cameras(cls, cameras: Sequence[Camera], frames: Iterable[int] | None = None,
                     sigma: Any = 0.0, provenance: Sequence[ModelProvenance] = ()) -> "CameraTrack":
        cams = list(cameras)
        fr = np.arange(len(cams)) if frames is None else np.asarray(list(frames), int)
        ang = np.array([c.ptz() for c in cams])
        sig = np.broadcast_to(np.asarray(sigma, float), (len(cams), 4)).copy()
        return cls(cams[0].intrinsics, cams[0].centre, fr, ang[:, 0], ang[:, 1], ang[:, 2],
                   np.array([c.intrinsics.f for c in cams]), sig, provenance=list(provenance))

    @classmethod
    def static(cls, camera: Camera, frames: Iterable[int], sigma: Sequence[float] = (0, 0, 0, 0),
               provenance: Sequence[ModelProvenance] = ()) -> "CameraTrack":
        fr = list(frames)
        return cls.from_cameras([camera] * len(fr), fr, sigma, provenance)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "v2e.camera-track/1",
            "model": "fixed_position_ptz",
            "intrinsics": self.intrinsics.to_dict(),
            "centre_m": self.centre.tolist(),
            "centre_sigma_m": self.centre_sigma.tolist(),
            "frames": self.frames.tolist(),
            "pan_deg": np.degrees(self.pan).tolist(),
            "tilt_deg": np.degrees(self.tilt).tolist(),
            "roll_deg": np.degrees(self.roll).tolist(),
            "focal_px": self.focal.tolist(),
            "sigma": {"pan_deg": np.degrees(self.sigma[:, 0]).tolist(),
                      "tilt_deg": np.degrees(self.sigma[:, 1]).tolist(),
                      "roll_deg": np.degrees(self.sigma[:, 2]).tolist(),
                      "focal_px": self.sigma[:, 3].tolist()},
            "quality": None if self.quality is None else np.asarray(self.quality).tolist(),
            "calibration_cov": None if self.calibration_cov is None else np.asarray(self.calibration_cov).tolist(),
            "provenance": [p.to_dict() for p in self.provenance],
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "CameraTrack":
        s = d["sigma"]
        sig = np.stack([np.radians(s["pan_deg"]), np.radians(s["tilt_deg"]), np.radians(s["roll_deg"]),
                        np.asarray(s["focal_px"], float)], axis=1)
        return cls(Intrinsics.from_dict(d["intrinsics"]), d["centre_m"], d["frames"],
                   np.radians(d["pan_deg"]), np.radians(d["tilt_deg"]), np.radians(d["roll_deg"]),
                   d["focal_px"], sig, d.get("centre_sigma_m", [0, 0, 0]),
                   None if d.get("quality") is None else np.asarray(d["quality"]),
                   [ModelProvenance.from_dict(p) for p in d.get("provenance", [])],
                   None if d.get("calibration_cov") is None else np.asarray(d["calibration_cov"], float))


# --------------------------------------------------------------------------- prompts and masks

@dataclass
class ObjectPrompt:
    """Assisted annotation of one object in one frame. Give at least one of the fields."""

    object_id: str
    frame: int
    box: tuple[float, float, float, float] | None = None     # x0, y0, x1, y1 (pixels)
    polygon: np.ndarray | None = None                          # (N, 2) outline (pixels)
    mask: np.ndarray | None = None                             # (H, W) bool
    points: np.ndarray | None = None                           # (N, 2) clicks (pixels)
    labels: np.ndarray | None = None                           # (N,) 1 = on the object, 0 = not

    def region_mask(self, width: int, height: int) -> np.ndarray | None:
        """A filled mask from ``mask`` or ``polygon`` (``None`` for boxes and points only)."""
        if self.mask is not None:
            return np.asarray(self.mask, bool)
        if self.polygon is not None:
            from openbeam_v2e.reconstruct.silhouette import polygon_mask

            return polygon_mask([np.asarray(self.polygon, float)], width, height)
        return None

    def bbox(self) -> tuple[float, float, float, float] | None:
        if self.box is not None:
            return tuple(float(v) for v in self.box)  # type: ignore[return-value]
        if self.polygon is not None:
            P = np.asarray(self.polygon, float)
            return float(P[:, 0].min()), float(P[:, 1].min()), float(P[:, 0].max()), float(P[:, 1].max())
        if self.mask is not None:
            from openbeam_v2e.reconstruct.silhouette import mask_bbox

            b = mask_bbox(np.asarray(self.mask, bool))
            return None if b is None else tuple(float(v) for v in b)  # type: ignore[return-value]
        if self.points is not None:
            P = np.asarray(self.points, float)
            if self.labels is not None:
                P = P[np.asarray(self.labels) > 0]
            if len(P):
                return float(P[:, 0].min()), float(P[:, 1].min()), float(P[:, 0].max()), float(P[:, 1].max())
        return None


@dataclass
class MaskCrop:
    """A mask stored as its bounding-box crop."""

    x0: int
    y0: int
    mask: np.ndarray

    @classmethod
    def from_full(cls, full: np.ndarray, pad: int = 0) -> "MaskCrop | None":
        m = np.asarray(full, bool)
        rows = np.nonzero(m.any(axis=1))[0]
        if len(rows) == 0:
            return None
        cols = np.nonzero(m.any(axis=0))[0]
        y0, y1 = max(0, rows[0] - pad), min(m.shape[0], rows[-1] + 1 + pad)
        x0, x1 = max(0, cols[0] - pad), min(m.shape[1], cols[-1] + 1 + pad)
        return cls(int(x0), int(y0), m[y0:y1, x0:x1].copy())

    def full(self, width: int, height: int) -> np.ndarray:
        out = np.zeros((height, width), bool)
        h, w = self.mask.shape
        out[self.y0:self.y0 + h, self.x0:self.x0 + w] = self.mask[: height - self.y0, : width - self.x0]
        return out

    def bbox(self) -> tuple[int, int, int, int]:
        h, w = self.mask.shape
        return self.x0, self.y0, self.x0 + w, self.y0 + h

    def area(self) -> int:
        return int(np.count_nonzero(self.mask))


@dataclass
class MaskTrack:
    """Masks of one object per frame (frames without a mask: not observed)."""

    object_id: str
    width: int
    height: int
    crops: dict[int, MaskCrop] = field(default_factory=dict)
    confidence: dict[int, float] = field(default_factory=dict)
    provenance: list[ModelProvenance] = field(default_factory=list)

    def frames(self) -> list[int]:
        return sorted(self.crops)

    def mask(self, frame: int) -> np.ndarray | None:
        c = self.crops.get(int(frame))
        return None if c is None else c.full(self.width, self.height)

    def set(self, frame: int, mask: np.ndarray, confidence: float = 1.0) -> None:
        crop = MaskCrop.from_full(mask)
        if crop is None:
            self.crops.pop(int(frame), None)
            self.confidence.pop(int(frame), None)
        else:
            self.crops[int(frame)] = crop
            self.confidence[int(frame)] = float(confidence)

    def bbox(self, frame: int) -> tuple[int, int, int, int] | None:
        c = self.crops.get(int(frame))
        return None if c is None else c.bbox()


@dataclass
class Segmentation:
    width: int
    height: int
    tracks: dict[str, MaskTrack] = field(default_factory=dict)
    provenance: list[ModelProvenance] = field(default_factory=list)

    def track(self, object_id: str) -> MaskTrack:
        if object_id not in self.tracks:
            self.tracks[object_id] = MaskTrack(object_id, self.width, self.height,
                                               provenance=list(self.provenance))
        return self.tracks[object_id]


@dataclass
class Detection:
    box: tuple[float, float, float, float]
    label: str
    score: float
    mask: np.ndarray | None = None


@dataclass
class PointTracks:
    """Pixel positions (N, T, 2) of N points over the frames, NaN where a point is lost.

    ``frames`` (T,) lists the frame numbers in the order they were tracked (possibly descending)
    and ``visible`` (N, T) says where each point was seen; ``points`` is NaN wherever ``visible``
    is False. ``confidence`` is optional: (N, T) values in [0, 1] taken from the model's own
    outputs, NaN wherever the point is not visible. It is ``None`` when the model reports no such
    value (``baseline.lk`` and ``cotracker`` leave it ``None``).
    """

    frames: np.ndarray
    points: np.ndarray
    visible: np.ndarray
    provenance: list[ModelProvenance] = field(default_factory=list)
    confidence: np.ndarray | None = None


@dataclass
class DepthMap:
    """Per-pixel depth along the optical axis.

    ``kind`` is ``"metric"`` (metres), ``"relative"`` (an unknown scale and shift) or
    ``"inverse_relative"`` (disparity-like: larger is nearer, unknown scale and shift).
    """

    depth: np.ndarray
    kind: str
    sigma: np.ndarray | None = None
    provenance: list[ModelProvenance] = field(default_factory=list)


# --------------------------------------------------------------------------- objects

@dataclass
class ObjectSpec:
    """What reconstruction should fit for one moving object."""

    object_id: str
    object_class: str = "vehicle.car"
    template: str = "vehicle"
    template_params: dict[str, float] = field(default_factory=dict)
    dims_prior: dict[str, tuple[float, float]] = field(default_factory=dict)  # name -> (value, sigma)
    motion: str = "ground"            # "ground": on the ground plane, yaw only; "free": 6-DoF
    fit_dims: bool = True


@dataclass
class PoseTrack:
    """A fitted rigid pose track.

    Positions are of the body origin (centre of the hull's bottom face) in the world frame;
    ``quat`` is body-to-world (x, y, z, w). ``cov`` is the 6x6 covariance per frame of
    (position, rotation vector of a small world-frame rotation applied to the estimate).
    ``observed`` marks frames with a mask, ``fitted`` the frames used in the fit and ``heldout``
    the observed frames deliberately left out of it.
    """

    object_id: str
    object_class: str
    template: str
    template_params: dict[str, float]
    dims: np.ndarray                  # (3,) length, width, height
    dims_sigma: np.ndarray            # (3,)
    symmetry: str
    frames: np.ndarray
    position: np.ndarray              # (T, 3)
    quat: np.ndarray                  # (T, 4)
    cov: np.ndarray                   # (T, 6, 6)
    velocity: np.ndarray              # (T, 3)
    observed: np.ndarray              # (T,) bool
    fitted: np.ndarray                # (T,) bool
    heldout: np.ndarray               # (T,) bool
    iou: np.ndarray                   # (T,) silhouette IoU of the final pose against the mask (NaN if none)
    motion: str = "ground"
    provenance: list[ModelProvenance] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.frames)

    def index(self, frame: int) -> int:
        hits = np.nonzero(self.frames == int(frame))[0]
        if len(hits) == 0:
            raise KeyError(frame)
        return int(hits[0])

    def rotation(self, frame: int) -> np.ndarray:
        from openbeam_v2e.reconstruct.geom import quat_to_matrix

        return quat_to_matrix(self.quat[self.index(frame)])

    def sigma_position(self) -> np.ndarray:
        return np.sqrt(np.maximum(np.diagonal(self.cov[:, :3, :3], axis1=1, axis2=2), 0.0))

    def sigma_rotation_deg(self) -> np.ndarray:
        return np.degrees(np.sqrt(np.maximum(np.diagonal(self.cov[:, 3:, 3:], axis1=1, axis2=2), 0.0)))

    def summary(self) -> dict[str, Any]:
        obs = self.observed
        fitted_iou = self.iou[self.fitted & np.isfinite(self.iou)]
        held_iou = self.iou[self.heldout & np.isfinite(self.iou)]
        return {
            "object_id": self.object_id,
            "frames": [int(self.frames[0]), int(self.frames[-1])] if len(self.frames) else [],
            "observed_frames": int(obs.sum()),
            "dims_m": self.dims.tolist(),
            "dims_sigma_m": self.dims_sigma.tolist(),
            "median_sigma_position_m": float(np.median(np.linalg.norm(self.sigma_position(), axis=1))) if len(self) else math.nan,
            "iou_fitted_median": float(np.median(fitted_iou)) if len(fitted_iou) else None,
            "iou_heldout_median": float(np.median(held_iou)) if len(held_iou) else None,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "object_id": self.object_id, "class": self.object_class, "template": self.template,
            "template_params": dict(self.template_params), "dims_m": self.dims.tolist(),
            "dims_sigma_m": self.dims_sigma.tolist(), "symmetry": self.symmetry,
            "motion": self.motion, "frames": self.frames.tolist(),
            "position_m": self.position.tolist(), "quat_xyzw": self.quat.tolist(),
            "cov": self.cov.tolist(), "velocity_mps": self.velocity.tolist(),
            "observed": self.observed.tolist(), "fitted": self.fitted.tolist(),
            "heldout": self.heldout.tolist(),
            "iou": [None if not np.isfinite(v) else float(v) for v in self.iou],
            "provenance": [p.to_dict() for p in self.provenance],
            "diagnostics": _plain(self.diagnostics),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "PoseTrack":
        return cls(d["object_id"], d["class"], d["template"], dict(d.get("template_params", {})),
                   np.asarray(d["dims_m"], float), np.asarray(d["dims_sigma_m"], float), d["symmetry"],
                   np.asarray(d["frames"], int), np.asarray(d["position_m"], float),
                   np.asarray(d["quat_xyzw"], float), np.asarray(d["cov"], float),
                   np.asarray(d["velocity_mps"], float), np.asarray(d["observed"], bool),
                   np.asarray(d["fitted"], bool), np.asarray(d["heldout"], bool),
                   np.array([np.nan if v is None else v for v in d["iou"]], float),
                   d.get("motion", "ground"),
                   [ModelProvenance.from_dict(p) for p in d.get("provenance", [])],
                   dict(d.get("diagnostics", {})))
