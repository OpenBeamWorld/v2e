# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""``sam2``: video object segmentation with Meta's SAM 2 (optional extra ``[sam2]``).

Prompts (boxes, outlines turned into masks, masks, clicks) seed one SAM 2 object per V2E object;
masks are then propagated forwards and backwards through the frame range. The model and its
weights come from the ``sam2`` package (Apache-2.0) and are loaded only when ``segment`` runs:

* ``model_id`` (default ``facebook/sam2.1-hiera-tiny``) loads weights from the Hugging Face Hub
  through ``SAM2VideoPredictor.from_pretrained``; or
* ``config`` plus ``checkpoint`` builds the predictor from a local checkpoint, whose sha256 is
  recorded in the provenance.

Nothing from SAM 2 is vendored here. Frames are handed to SAM 2 as a temporary directory of
JPEG files, which is the input its video predictor reads.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from openbeam_v2e.perception.adapters._common import pick_device, require, sha256_of, versions
from openbeam_v2e.perception.interfaces import Segmenter
from openbeam_v2e.perception.types import CameraTrack, ObjectPrompt, Segmentation


class Sam2Segmenter(Segmenter):
    """Prompted video segmentation with SAM 2."""

    name = "sam2"
    extras = ("sam2",)
    requires = ("sam2", "torch", "PIL")
    summary = "SAM 2 video segmentation and mask tracking from keyframe prompts (GPU recommended)."

    def __init__(self, model_id: str = "facebook/sam2.1-hiera-tiny", config: str | None = None,
                 checkpoint: str | None = None, device: str | None = None, jpeg_quality: int = 95,
                 mask_threshold: float = 0.0, **params: Any) -> None:
        super().__init__(model_id=model_id, config=config, checkpoint=checkpoint, device=device,
                         jpeg_quality=jpeg_quality, mask_threshold=mask_threshold, **params)

    def _predictor(self) -> Any:
        p = self.params
        device = pick_device(p["device"])
        if p["config"] and p["checkpoint"]:
            build = require("sam2.build_sam", "sam2")
            return build.build_sam2_video_predictor(p["config"], p["checkpoint"], device=device)
        mod = require("sam2.sam2_video_predictor", "sam2")
        return mod.SAM2VideoPredictor.from_pretrained(p["model_id"], device=device)

    def provenance(self, **extra: Any):  # type: ignore[override]
        p = self.params
        return super().provenance(weights=p["checkpoint"] or p["model_id"],
                                  weights_sha256=sha256_of(p["checkpoint"]),
                                  libraries=versions("sam2", "torch"), **extra)

    @staticmethod
    def _write_frames(video: Any, frames: Sequence[int], directory: Path, quality: int) -> None:
        Image = require("PIL.Image", "sam2")
        for k, f in enumerate(frames):
            Image.fromarray(np.asarray(video.read(f), np.uint8)).save(directory / f"{k:05d}.jpg", quality=quality)

    def segment(self, video: Any, prompts: Sequence[ObjectPrompt], frames: Sequence[int],
                cameras: CameraTrack | None = None) -> Segmentation:
        torch = require("torch", "sam2")
        frames = sorted(int(f) for f in frames)
        index = {f: k for k, f in enumerate(frames)}
        w, h = video.size
        seg = Segmentation(w, h, provenance=[self.provenance()])
        ids = list(dict.fromkeys(p.object_id for p in prompts))
        if not ids:
            raise ValueError("sam2 needs at least one prompt (box, outline, mask or click) per object")
        obj_num = {oid: k + 1 for k, oid in enumerate(ids)}
        predictor = self._predictor()
        thr = float(self.params["mask_threshold"])
        with tempfile.TemporaryDirectory(prefix="v2e-sam2-") as tmp:
            self._write_frames(video, frames, Path(tmp), int(self.params["jpeg_quality"]))
            with torch.inference_mode():
                state = predictor.init_state(video_path=tmp)
                first = len(frames)
                for pr in prompts:
                    if pr.frame not in index:
                        continue
                    k = index[pr.frame]
                    first = min(first, k)
                    region = pr.region_mask(w, h)
                    if region is not None:
                        predictor.add_new_mask(inference_state=state, frame_idx=k, obj_id=obj_num[pr.object_id],
                                               mask=region)
                        continue
                    kw: dict[str, Any] = {}
                    if pr.box is not None:
                        kw["box"] = np.asarray(pr.box, np.float32)
                    if pr.points is not None:
                        kw["points"] = np.asarray(pr.points, np.float32)
                        lab = pr.labels if pr.labels is not None else np.ones(len(pr.points))
                        kw["labels"] = np.asarray(lab, np.int32)
                    if kw:
                        predictor.add_new_points_or_box(inference_state=state, frame_idx=k,
                                                        obj_id=obj_num[pr.object_id], **kw)
                if first == len(frames):
                    raise ValueError("no prompt falls inside the frame range")
                num_obj = {v: k for k, v in obj_num.items()}
                for reverse in (False, True):
                    for k, out_ids, logits in predictor.propagate_in_video(state, start_frame_idx=first,
                                                                           reverse=reverse):
                        for j, n in enumerate(list(out_ids)):
                            m = np.asarray((logits[j] > thr).cpu().numpy()).reshape(h, w)
                            if m.any():
                                seg.track(num_obj[int(n)]).set(frames[int(k)], m, 1.0)
        return seg
