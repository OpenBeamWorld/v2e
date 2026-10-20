# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""``hf.depth``: monocular depth from any Hugging Face ``depth-estimation`` model (extra ``[perception]``).

Works with the depth models the ``transformers`` pipeline supports, for example Depth Anything V2
(``depth-anything/Depth-Anything-V2-Small-hf``, relative inverse depth) or its metric variants
(``...-Metric-Outdoor-Small-hf``, metres), DPT or ZoeDepth. Relative output is returned with
``kind="inverse_relative"`` (or ``"relative"``); the reconstruction aligns it to metres on the
ground plane (:func:`openbeam_v2e.perception.baseline.depth.align_depth`). Model licences differ:
check the model card before you use its output.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from openbeam_v2e.perception.adapters._common import pick_device, require, versions
from openbeam_v2e.perception.interfaces import DepthEstimator
from openbeam_v2e.perception.types import DepthMap
from openbeam_v2e.reconstruct.geom import Camera


class TransformersDepth(DepthEstimator):
    """Depth maps from a ``transformers`` depth-estimation pipeline."""

    name = "hf.depth"
    extras = ("perception",)
    requires = ("transformers", "PIL")
    summary = "Monocular depth from a Hugging Face depth-estimation model (relative or metric)."

    def __init__(self, model_id: str = "depth-anything/Depth-Anything-V2-Small-hf", kind: str | None = None,
                 revision: str | None = None, device: str | None = None, **params: Any) -> None:
        if kind is None:
            kind = "metric" if "metric" in model_id.lower() else "inverse_relative"
        if kind not in ("metric", "relative", "inverse_relative"):
            raise ValueError("kind must be metric, relative or inverse_relative")
        super().__init__(model_id=model_id, kind=kind, revision=revision, device=device, **params)
        self._pipe: Any = None

    def provenance(self, **extra: Any):  # type: ignore[override]
        p = self.params
        w = p["model_id"] + (f"@{p['revision']}" if p["revision"] else "")
        return super().provenance(weights=w, libraries=versions("transformers", "torch"), **extra)

    def _pipeline(self) -> Any:
        if self._pipe is None:
            tf = require("transformers", "perception")
            dev = pick_device(self.params["device"])
            kw: dict[str, Any] = {"model": self.params["model_id"]}
            if self.params["revision"]:
                kw["revision"] = self.params["revision"]
            self._pipe = tf.pipeline("depth-estimation", device=dev, **kw)
        return self._pipe

    def estimate(self, image: np.ndarray, camera: Camera | None = None) -> DepthMap:
        Image = require("PIL.Image", "perception")
        img = np.asarray(image, np.uint8)
        out = self._pipeline()(Image.fromarray(img))
        pred = out["predicted_depth"]
        if hasattr(pred, "detach"):
            pred = pred.detach()
        if hasattr(pred, "cpu"):
            pred = pred.cpu().numpy()
        d = np.squeeze(np.asarray(pred, float))
        if d.shape != img.shape[:2]:
            # the model's output resolution differs from the frame: resample bilinearly
            from scipy import ndimage

            d = ndimage.zoom(d, (img.shape[0] / d.shape[0], img.shape[1] / d.shape[1]), order=1)
            d = d[: img.shape[0], : img.shape[1]]
        return DepthMap(d, self.params["kind"], None, [self.provenance()])
