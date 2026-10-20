# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""``hf.detection``: object detection with Hugging Face models (extra ``[perception]``).

Uses the ``transformers`` ``object-detection`` pipeline (for example DETR,
``facebook/detr-resnet-50``) or, when class names are given and the model is open-vocabulary,
the ``zero-shot-object-detection`` pipeline (for example OWLv2,
``google/owlv2-base-patch16-ensemble``). Detections become prompts for a segmenter, which turns
assisted reconstruction into automatic reconstruction for clips where a detector sees the
objects (:func:`detections_to_prompts`).
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from openbeam_v2e.perception.adapters._common import pick_device, require, versions
from openbeam_v2e.perception.interfaces import Detector
from openbeam_v2e.perception.types import Detection, ObjectPrompt

#: Detector labels (COCO and common open-vocabulary words) mapped to V2E object classes.
LABEL_TO_CLASS: dict[str, str] = {
    "car": "vehicle.car", "truck": "vehicle.truck", "bus": "vehicle.truck", "van": "vehicle.van",
    "motorcycle": "vehicle.motorcycle", "bicycle": "vehicle.bicycle", "box": "prop.box",
    "suitcase": "prop.box", "sports ball": "prop.sphere", "ball": "prop.sphere", "barrel": "prop.cylinder",
}


class TransformersDetector(Detector):
    """Boxes from a ``transformers`` object-detection or zero-shot detection pipeline."""

    name = "hf.detection"
    extras = ("perception",)
    requires = ("transformers", "PIL")
    summary = "Object detection (closed-set or open-vocabulary) to seed prompts automatically."

    def __init__(self, model_id: str = "facebook/detr-resnet-50", revision: str | None = None,
                 min_score: float = 0.5, zero_shot: bool | None = None, device: str | None = None,
                 **params: Any) -> None:
        if zero_shot is None:
            zero_shot = any(k in model_id.lower() for k in ("owl", "grounding"))
        super().__init__(model_id=model_id, revision=revision, min_score=min_score, zero_shot=zero_shot,
                         device=device, **params)
        self._pipe: Any = None

    def provenance(self, **extra: Any):  # type: ignore[override]
        p = self.params
        w = p["model_id"] + (f"@{p['revision']}" if p["revision"] else "")
        return super().provenance(weights=w, libraries=versions("transformers", "torch"), **extra)

    def _pipeline(self) -> Any:
        if self._pipe is None:
            tf = require("transformers", "perception")
            task = "zero-shot-object-detection" if self.params["zero_shot"] else "object-detection"
            kw: dict[str, Any] = {"model": self.params["model_id"]}
            if self.params["revision"]:
                kw["revision"] = self.params["revision"]
            self._pipe = tf.pipeline(task, device=pick_device(self.params["device"]), **kw)
        return self._pipe

    def detect(self, image: np.ndarray, classes: Sequence[str] | None = None) -> list[Detection]:
        Image = require("PIL.Image", "perception")
        pil = Image.fromarray(np.asarray(image, np.uint8))
        if self.params["zero_shot"]:
            if not classes:
                raise ValueError("an open-vocabulary detector needs class names")
            raw = self._pipeline()(pil, candidate_labels=list(classes))
        else:
            raw = self._pipeline()(pil)
        out = []
        for d in raw:
            score = float(d["score"])
            label = str(d["label"])
            if score < float(self.params["min_score"]):
                continue
            if classes and not self.params["zero_shot"] and label not in classes:
                continue
            b = d["box"]
            out.append(Detection((float(b["xmin"]), float(b["ymin"]), float(b["xmax"]), float(b["ymax"])),
                                 label, score))
        out.sort(key=lambda d: -d.score)
        return out


def detections_to_prompts(detections: Sequence[Detection], frame: int,
                          label_to_class: Mapping[str, str] | None = None,
                          prefix: str = "object") -> tuple[list[ObjectPrompt], dict[str, str]]:
    """One box prompt per detection (ids ``object_1``, ...) and the V2E class of each."""
    mapping = dict(LABEL_TO_CLASS)
    mapping.update(label_to_class or {})
    prompts, classes = [], {}
    for k, d in enumerate(detections, start=1):
        oid = f"{prefix}_{k}"
        prompts.append(ObjectPrompt(oid, int(frame), box=d.box))
        cls = mapping.get(d.label.lower())
        if cls:
            classes[oid] = cls
    return prompts, classes
