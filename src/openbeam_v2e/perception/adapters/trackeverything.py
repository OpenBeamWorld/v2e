# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""TrackEverything/Pi3 point tracking from a pinned external Linux/CUDA checkout.

The V2E interface receives projected pixels with gaps; optional artifacts retain
the estimated, non-metric 3-D world and cameras. No model code is vendored here.
"""

from __future__ import annotations

from dataclasses import replace
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Sequence

import numpy as np

from openbeam_v2e.perception.adapters._common import require, sha256_of, versions
from openbeam_v2e.perception.interfaces import PointTracker
from openbeam_v2e.perception.types import PointTracks

CODE_REVISION = "f0c18dd9714b9f85825c45a1c3aae685da4e6222"
MODEL_REPOSITORY = "katefgroup/TrackEverything"
MODEL_REVISION = "1957810f46ef684154fec91a9b8c0e6a68cd695e"
PI3_REPOSITORY = "yyfz233/Pi3"
PI3_REVISION = "ae722e7039287d0c8fde9f11f197f804f44b510c"


def project_query_tracks(coords: np.ndarray, visibility_logits: np.ndarray,
                         intrinsics: np.ndarray, extrinsics: np.ndarray,
                         inference_hw: Sequence[int], source_hw: Sequence[int],
                         *, visibility_threshold: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Project normalized-world (T,N,3) tracks into V2E (N,T,2) pixels.

    Extrinsics map world to camera; their translations already share the tracks'
    normalization. Upstream uses pixel-index coordinates and (size - 1) resize
    scaling; V2E uses pixel centres at +0.5. Missing predictions stay NaN.
    """
    if not 0 < visibility_threshold < 1:
        raise ValueError("visibility_threshold must be between 0 and 1")
    xyz, logits = np.asarray(coords, float), np.asarray(visibility_logits, float)
    K, E = np.asarray(intrinsics, float), np.asarray(extrinsics, float)
    if xyz.ndim != 3 or xyz.shape[-1] != 3 or logits.shape != xyz.shape[:2]:
        raise ValueError("Expected coords (T,N,3) and visibility logits (T,N)")
    T = len(xyz)
    if K.shape != (T, 3, 3) or E.shape != (T, 4, 4):
        raise ValueError("Camera shapes do not match the track frame count")
    if not np.isfinite(K).all() or not np.isfinite(E).all():
        raise ValueError("Camera predictions must be finite")
    sizes = np.asarray([inference_hw, source_hw], float)
    if (sizes.shape != (2, 2) or not np.isfinite(sizes).all()
            or np.any(sizes < 2) or np.any(sizes != np.floor(sizes))):
        raise ValueError("Image sizes must be integer (height, width) pairs greater than 1")
    camera = np.einsum("tij,tnj->tni", E[:, :3, :3], xyz) + E[:, None, :3, 3]
    homogeneous = np.einsum("tij,tnj->tni", K, camera)
    with np.errstate(divide="ignore", invalid="ignore"):
        xy = homogeneous[..., :2] / homogeneous[..., 2:]
    xy *= ((sizes[1] - 1) / (sizes[0] - 1))[::-1]
    xy += 0.5
    height, width = sizes[1]
    threshold = np.log(visibility_threshold / (1 - visibility_threshold))
    visible = (np.isfinite(xyz).all(axis=-1) & np.isfinite(xy).all(axis=-1)
               & np.isfinite(logits) & (logits > threshold) & (camera[..., 2] > 0)
               & (xy[..., 0] >= 0) & (xy[..., 0] < width)
               & (xy[..., 1] >= 0) & (xy[..., 1] < height))
    xy[~visible] = np.nan
    return xy.transpose(1, 0, 2), visible.T


class TrackEverythingPointTracker(PointTracker):
    """``trackeverything``: pinned sparse query tracking with Pi3 geometry.

    ``tracker_source`` names the external Git checkout (directory name must be
    ``trackeverything``). ``output_dir``, when supplied, must be a new directory.
    Inference is forward from each query's seed frame, with no gap filling.
    """

    name = "trackeverything"
    requires = ("torch", "huggingface_hub")
    summary = "TrackEverything/Pi3 query tracks (external pinned checkout and CUDA required)."

    def __init__(self, tracker_source: str, device: str = "cuda",
                 visibility_threshold: float = 0.5, output_dir: str | None = None) -> None:
        if not 0 < visibility_threshold < 1:
            raise ValueError("visibility_threshold must be between 0 and 1")
        if device != "cuda" and not (device.startswith("cuda:") and device[5:].isdigit()):
            raise ValueError("TrackEverything requires a CUDA device")
        super().__init__(tracker_source=str(Path(tracker_source).expanduser().resolve()), device=device,
                         visibility_threshold=float(visibility_threshold),
                         output_dir=None if output_dir is None else str(Path(output_dir).expanduser().resolve()),
                         tracker_code_revision=CODE_REVISION, model_revision=MODEL_REVISION,
                         geometry_backend="pi3", pi3_revision=PI3_REVISION,
                         mode="sparse", use_conf_mask=True, mem_efficient=True)
        self._loaded: Any = None
        self._weights_sha256: str | None = None
        self._pi3_sha256: str | None = None

    def _model(self) -> Any:
        if self._loaded is not None:
            return self._loaded
        source = Path(self.params["tracker_source"])
        if source.name != "trackeverything" or not (source / "demo_hf.py").is_file():
            raise ValueError("tracker_source must name the pinned checkout directory trackeverything")

        def git(*args: str) -> str:
            return subprocess.check_output(["git", "-C", str(source), *args], text=True).strip()

        if (Path(git("rev-parse", "--show-toplevel")).resolve() != source
                or git("rev-parse", "HEAD") != CODE_REVISION
                or git("status", "--porcelain", "--untracked-files=no")):
            raise ValueError(f"Use an unmodified TrackEverything checkout at {CODE_REVISION}")
        torch = require("torch", "perception")
        if not torch.cuda.is_available():
            raise RuntimeError("Run TrackEverything in its upstream Linux/CUDA environment")
        hub = require("huggingface_hub", "perception")
        parent = str(source.parent)
        sys.path.insert(0, parent)
        try:
            upstream = importlib.import_module("trackeverything.demo_hf")
        finally:
            if parent in sys.path:
                sys.path.remove(parent)
        if Path(upstream.__file__).resolve() != source / "demo_hf.py":
            raise RuntimeError("A different TrackEverything checkout is already imported; use a fresh process")
        checkpoint = hub.hf_hub_download(MODEL_REPOSITORY, "model.pth", revision=MODEL_REVISION)
        geometry = hub.snapshot_download(PI3_REPOSITORY, revision=PI3_REVISION,
                                         allow_patterns=["config.json", "model.safetensors"])
        self._weights_sha256 = sha256_of(checkpoint)
        self._pi3_sha256 = sha256_of(Path(geometry) / "model.safetensors")
        if self._weights_sha256 is None or self._pi3_sha256 is None:
            raise RuntimeError("Downloaded model weights are missing")
        self._loaded = upstream.TrackEverything.from_pretrained(
            checkpoint, device=self.params["device"], geometry_backend="pi3", pi3_checkpoint=geometry)
        return self._loaded

    def track(self, video: Any, queries: np.ndarray, frames: Sequence[int]) -> PointTracks:
        fr = np.asarray(frames)
        if (fr.ndim != 1 or not len(fr) or fr.dtype.kind not in "iu"
                or np.any(fr[1:] <= fr[:-1]) or fr[0] < 0 or fr[-1] >= len(video)):
            raise ValueError("frames must be strictly increasing source-frame integers within the video")
        width, height = video.size
        fps = float(video.fps)
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("FrameSource fps must be finite and positive")
        q = np.asarray(queries, float)
        if (q.ndim != 2 or q.shape[1] != 3 or not len(q) or not np.isfinite(q).all()
                or np.any(q[:, 0] != np.floor(q[:, 0])) or not np.isin(q[:, 0], fr).all()
                or np.any(q[:, 1:] < 0) or np.any(q[:, 1:] >= [width, height])):
            raise ValueError("queries must be finite (source frame, u, v) seeds on the selected frame grid and image")
        if min(width, height) < 2:
            raise ValueError("Images must be at least 2 by 2 pixels")
        remapped = q.copy()
        remapped[:, 0] = np.searchsorted(fr, q[:, 0])
        remapped[:, 1:] -= 0.5  # V2E pixel centres -> upstream pixel indices
        # Materialize the selected frames exactly once. Do not apply upstream's
        # start_frame/stride again or query times will refer to the wrong images.
        rgb = np.empty((len(fr), height, width, 3), dtype=np.uint8)
        for i, frame in enumerate(fr):
            image = np.asarray(video.read(int(frame)))
            if image.shape != rgb.shape[1:] or image.dtype != np.uint8:
                raise ValueError("FrameSource must return fixed-size uint8 RGB images")
            rgb[i] = image
        out = Path(self.params["output_dir"]) if self.params["output_dir"] else None
        if out:
            out.mkdir(parents=True, exist_ok=False)
        receipt: dict[str, Any] = {
            "schema": "v2e.trackeverything/1", "inference_completed": False,
            "source_frames": fr.tolist(), "source_image_hw": [height, width],
            "source_fps": fps,
            "input_rgb_sha256": hashlib.sha256(rgb).hexdigest(),
            "queries": q.tolist(), "pixel_convention": "V2E pixel centres at +0.5",
            "coordinate_system": "estimated normalized world; extrinsics map world to camera",
            "metric_scale_known": False, "camera_calibrated": False,
            "tracking_accuracy_validated": False, "physical_recreation_validated": False,
        }
        try:
            model = self._model()
            output = model.infer(video=rgb, mode="sparse", query_points_2d=remapped.astype(np.float32),
                                 use_conf_mask=True, mem_efficient=True)

            def cpu(value: Any) -> np.ndarray:
                return value.detach().float().cpu().numpy()

            coords, logits = cpu(output.query_coords), cpu(output.query_visibs)
            shape = (1, len(fr), len(q))
            if coords.shape != (*shape, 3) or logits.shape != shape:
                raise ValueError("Incomplete TrackEverything query prediction")
            K, E = cpu(output.intrinsics), cpu(output.extrinsics)
            infer_hw = tuple(output.rgb.shape[-2:])
            norm = float(output.norm_factor)
            if not np.isfinite(norm) or norm <= 0:
                raise ValueError("Invalid scene normalization factor")
            points, visible = project_query_tracks(coords[0], logits[0], K, E, infer_hw, (height, width),
                                                   visibility_threshold=self.params["visibility_threshold"])
            visible &= fr[None, :] >= q[:, 0, None]  # no pre-seed observations from forward inference
            points[~visible] = np.nan
            provenance = self.provenance(weights=MODEL_REPOSITORY, weights_sha256=self._weights_sha256,
                                         libraries=versions("torch", "huggingface_hub", "numpy"))
            provenance = replace(provenance, params={**provenance.params, "pi3_repository": PI3_REPOSITORY,
                                                     "pi3_sha256": self._pi3_sha256})
            receipt.update(provenance=provenance.to_dict(), frames=len(fr), queries_count=len(q),
                           inference_image_hw=infer_hw, norm_factor=norm)
            if out:
                np.savez_compressed(out / "query-tracks.npz", coords=coords[0], visibility_logits=logits[0],
                                    intrinsics=K, extrinsics=E, norm_factor=norm, inference_image_hw=infer_hw,
                                    source_image_hw=[height, width], source_frames=fr,
                                    source_time_s=fr.astype(float) / fps, queries=q,
                                    input_queries=remapped, points=points, visible=visible)
                receipt["tracks_sha256"] = sha256_of(out / "query-tracks.npz")
            receipt["inference_completed"] = True
            return PointTracks(fr.copy(), points, visible, [provenance])
        except Exception as error:
            receipt["error_type"] = type(error).__name__
            raise
        finally:
            if out:
                (out / "tracking-result.json").write_text(json.dumps(receipt, indent=2) + "\n")
