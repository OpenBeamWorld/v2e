# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""``cotracker``: dense point tracking with Meta's CoTracker (optional, installed separately).

CoTracker is not published on PyPI; install it from its repository (``pip install
git+https://github.com/facebookresearch/co-tracker``) together with the ``[perception]`` extra
for PyTorch. The adapter then builds ``cotracker.predictor.CoTrackerPredictor`` from a local
checkpoint (its sha256 goes into the provenance).

Loading through ``torch.hub`` downloads and runs code from the CoTracker repository at run time,
so it happens only when you pass ``allow_hub=True`` deliberately.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from openbeam_v2e.perception.adapters._common import pick_device, require, sha256_of, versions
from openbeam_v2e.perception.interfaces import PointTracker
from openbeam_v2e.perception.types import PointTracks


class CoTrackerPointTracker(PointTracker):
    """Point tracking with CoTracker (offline model, forwards and backwards)."""

    name = "cotracker"
    extras = ("perception",)
    requires = ("torch",)
    summary = "CoTracker point tracking (install CoTracker from its repository; GPU recommended)."

    def __init__(self, checkpoint: str | None = None, allow_hub: bool = False,
                 hub_model: str = "cotracker3_offline", device: str | None = None,
                 visibility_threshold: float = 0.5, **params: Any) -> None:
        super().__init__(checkpoint=checkpoint, allow_hub=allow_hub, hub_model=hub_model, device=device,
                         visibility_threshold=visibility_threshold, **params)

    def provenance(self, **extra: Any):  # type: ignore[override]
        p = self.params
        return super().provenance(weights=p["checkpoint"] or f"torch.hub:{p['hub_model']}",
                                  weights_sha256=sha256_of(p["checkpoint"]),
                                  libraries=versions("torch", "cotracker"), **extra)

    def _model(self) -> Any:
        p = self.params
        if p["checkpoint"]:
            pred = require("cotracker.predictor", "perception")
            return pred.CoTrackerPredictor(checkpoint=p["checkpoint"])
        if p["allow_hub"]:
            torch = require("torch", "perception")
            return torch.hub.load("facebookresearch/co-tracker", p["hub_model"])
        raise RuntimeError("cotracker needs checkpoint=<path> (or allow_hub=True to fetch code and weights "
                           "from the CoTracker repository at run time)")

    def track(self, video: Any, queries: np.ndarray, frames: Sequence[int]) -> PointTracks:
        torch = require("torch", "perception")
        frames = [int(f) for f in frames]
        index = {f: k for k, f in enumerate(frames)}
        Q = np.asarray(queries, float).reshape(-1, 3)
        keep = np.array([int(q[0]) in index for q in Q], bool)
        q_local = np.c_[[index[int(t)] for t in Q[keep, 0]], Q[keep, 1:]] if keep.any() else np.zeros((0, 3))
        device = pick_device(self.params["device"])
        model = self._model().to(device)
        clip = np.stack([np.asarray(video.read(f), np.float32) for f in frames])     # T, H, W, 3
        vid = torch.from_numpy(clip).permute(0, 3, 1, 2)[None].to(device)            # 1, T, 3, H, W
        # CoTracker uses pixel coordinates with the first pixel centre at 0; V2E puts it at 0.5
        qt = torch.from_numpy(np.c_[q_local[:, :1], q_local[:, 1:] - 0.5].astype(np.float32))[None].to(device)
        with torch.no_grad():
            tracks, vis = model(vid, queries=qt, backward_tracking=True)
        tr = tracks[0].cpu().numpy() + 0.5                                           # T, N, 2
        vv = vis[0].cpu().numpy()
        if vv.dtype != bool:
            vv = vv > float(self.params["visibility_threshold"])
        n = len(Q)
        pts = np.full((n, len(frames), 2), np.nan)
        visible = np.zeros((n, len(frames)), bool)
        pts[keep] = np.transpose(tr, (1, 0, 2))
        visible[keep] = vv.T
        pts[~visible] = np.nan
        return PointTracks(np.asarray(frames), pts, visible, [self.provenance()])
