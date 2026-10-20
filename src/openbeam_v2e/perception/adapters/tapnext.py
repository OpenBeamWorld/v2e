# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""``tapnext``: online point tracking with Google DeepMind's TAPNext++ (optional, installed separately).

TAPNext and TAPNext++ are part of DeepMind's ``tapnet`` repository, which is not published on
PyPI. Install it from the repository together with the ``[perception]`` extra for PyTorch::

    pip install "openbeam-v2e[perception]" \\
        "tapnet[torch] @ git+https://github.com/google-deepmind/tapnet.git" einops

``tapnet[torch]`` adds torchvision, which TAPNext needs; ``einops`` is imported by TAPNext but not
declared by that extra. The tapnet code and its published checkpoints are released under
Apache-2.0.

The adapter never downloads weights. Download a checkpoint linked from the tapnet README and pass
its local path as ``checkpoint`` (its sha256 goes into the provenance):

* a TAPNext++ PyTorch checkpoint, loaded with tapnet's ``TAPNextPP.from_checkpoint``:
  ``tapnextpp_ckpt.pt`` (256 x 256, the default ``resolution=256``) or ``tapnextpp_512.ckpt``
  (fine-tuned at 512 x 512: pass ``resolution=512``);
* a TAPNext JAX checkpoint (``bootstapnext_ckpt.npz``, 256 x 256), converted on load with
  tapnet's ``restore_model_from_jax_checkpoint``.

The network runs online, one frame per call, carrying its recurrent state from frame to frame,
so memory stays bounded on long clips. Conventions handled here:

* frames are resized to ``resolution`` x ``resolution`` (bilinear with half-pixel centres and no
  antialiasing, like tapnet's own preprocessing; the aspect ratio is not kept) and scaled to
  [-1, 1];
* query and output positions are in TAPNext's 256 x 256 query space at every ``resolution``, in
  (y, x) order, and query frames are indices into the frames fed to the network. V2E and TAPNext
  both put the first pixel centre at 0.5 (TAPNext samples its query embedding with
  ``align_corners=False`` and adds 0.5 to its output bin), so positions are scaled between the
  frame size and the query space and swapped between (u, v) and (y, x), with no half-pixel shift;
* a point is visible where ``sigmoid(visible logit) > visibility_threshold`` (0.5 by default,
  which is the positive-logit test of tapnet's demos); with ``use_certainty=True`` the sigmoid is
  multiplied by the certainty first, like the demos' certainty option;
* ``PointTracks.confidence`` is tapnet's ``tracker_certainty``: the probability mass of the
  predicted position distribution within ``certainty_radius`` pixels (of the 256 x 256 query
  space) of the predicted position. It is NaN where a point is not visible.
"""

from __future__ import annotations

from importlib import metadata
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from openbeam_v2e.perception.adapters._common import pick_device, require, sha256_of, versions
from openbeam_v2e.perception.interfaces import PointTracker
from openbeam_v2e.perception.types import PointTracks

#: Side of TAPNext's query and output space (the model's ``image_size``) for the published checkpoints.
QUERY_SIZE = 256

_INSTALL = ('install tapnet from its repository: pip install '
            '"tapnet[torch] @ git+https://github.com/google-deepmind/tapnet.git" einops')


def _tapnet(module: str) -> Any:
    """Import a tapnet module, naming the repository install when it is missing."""
    try:
        return require(module, "perception")
    except ImportError as exc:
        raise ImportError(f"{exc}; {_INSTALL}") from exc


def _vcs_commit(dist: str) -> str | None:
    """The commit a distribution was installed from (pip records it for ``git+`` installs)."""
    import json

    try:
        text = metadata.distribution(dist).read_text("direct_url.json")
        return str(json.loads(text)["vcs_info"]["commit_id"]) if text else None
    except Exception:
        return None


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(x, float)))


def model_frame(rgb: np.ndarray, size: int) -> np.ndarray:
    """An RGB ``uint8`` frame (H, W, 3) as TAPNext input: (size, size, 3) ``float32`` in [-1, 1].

    Bilinear resampling with half-pixel centres and no antialiasing, the sampling of
    ``torch.nn.functional.interpolate(..., mode="bilinear", align_corners=False)``.
    """
    img = np.asarray(rgb)
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=2)
    img = img[..., :3]
    h, w = img.shape[:2]

    def taps(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        src = np.clip((np.arange(size) + 0.5) * (n / size) - 0.5, 0.0, n - 1)
        i0 = np.floor(src).astype(int)
        return i0, np.minimum(i0 + 1, n - 1), (src - i0).astype(np.float32)

    y0, y1, wy = taps(h)
    x0, x1, wx = taps(w)
    wy, wx = wy[:, None, None], wx[None, :, None]
    rows = img[y0].astype(np.float32) * (1 - wy) + img[y1].astype(np.float32) * wy
    out = rows[:, x0] * (1 - wx) + rows[:, x1] * wx
    return (out / 127.5 - 1.0).astype(np.float32)


class TapNextPointTracker(PointTracker):
    """Point tracking with TAPNext++ (online and causal).

    Points are tracked through ``frames`` in the order given, which may be descending (the
    reconstruction pipeline's backward pass). As with ``baseline.lk``, a point has no position
    before its own query frame in that order: those columns are NaN and not visible (TAPNext only
    emits placeholders there). To track backwards from a query frame, pass the frames in reverse
    order. All queries share one online pass, which starts at the earliest query frame; a query whose
    frame is not in ``frames`` gives an all-NaN row.
    """

    name = "tapnext"
    extras = ("perception",)
    requires = ("torch", "torchvision", "einops", "tapnet")
    summary = ("TAPNext++ online point tracking (install tapnet from its repository; local checkpoint; "
               "GPU recommended).")

    def __init__(self, checkpoint: str | None = None, resolution: int = 256, device: str | None = None,
                 visibility_threshold: float = 0.5, use_certainty: bool = False, certainty_radius: int = 8,
                 autocast: bool = True, **params: Any) -> None:
        resolution = int(resolution)
        if resolution <= 0 or resolution % 8:
            raise ValueError(f"tapnext resolution must be a positive multiple of 8 (the patch size), got {resolution}")
        super().__init__(checkpoint=None if checkpoint is None else str(checkpoint), resolution=resolution,
                         device=device, visibility_threshold=float(visibility_threshold),
                         use_certainty=bool(use_certainty), certainty_radius=int(certainty_radius),
                         autocast=bool(autocast), **params)
        self._net: tuple[str, Any] | None = None          # (device, network), kept out of the params
        self._certainty: Any = None
        self._sha: tuple[tuple[str, int, int], str | None] | None = None

    @classmethod
    def check(cls) -> tuple[bool, str]:
        ok, why = super().check()
        return (True, why) if ok else (False, f"{why}; {_INSTALL}")

    def provenance(self, **extra: Any):  # type: ignore[override]
        libraries = versions("torch", "torchvision", "einops", "tapnet")
        commit = _vcs_commit("tapnet")
        if commit and "tapnet" in libraries:
            libraries["tapnet"] = f"{libraries['tapnet']} (git {commit})"
        return super().provenance(weights=self.params["checkpoint"], weights_sha256=self._weights_sha256(),
                                  libraries=libraries, **extra)

    def _weights_sha256(self) -> str | None:
        path = self.params["checkpoint"]
        if not path:
            return None
        try:
            st = Path(path).stat()
        except OSError:
            return None
        key = (path, st.st_size, st.st_mtime_ns)
        if self._sha is None or self._sha[0] != key:      # hash a multi-GB checkpoint once
            self._sha = (key, sha256_of(path))
        return self._sha[1]

    def _model(self, device: str) -> Any:
        """The TAPNext network on ``device``, loaded once per instance."""
        p = self.params
        path = p["checkpoint"]
        if not path:
            raise RuntimeError("tapnext needs checkpoint=<local path> to a TAPNext++ .pt/.ckpt or TAPNext .npz "
                               "file linked from the tapnet README (the adapter never downloads weights)")
        if not Path(path).is_file():
            raise FileNotFoundError(f"tapnext checkpoint not found: {path}")
        if self._net is not None and self._net[0] == device:
            return self._net[1]
        require("torch", "perception")
        utils = _tapnet("tapnet.tapnext.tapnext_torch_utils")
        if Path(path).suffix.lower() == ".npz":
            arch = _tapnet("tapnet.tapnext.tapnext_torch")
            net = arch.TAPNext(image_size=(QUERY_SIZE, QUERY_SIZE))
            utils.restore_model_from_jax_checkpoint(net, path)
            net = net.to(device).eval()
        else:
            pp = _tapnet("tapnet.tapnextpp.votsp2026.model")
            wrapper = pp.TAPNextPP.from_checkpoint(path, device=device, input_resolution=p["resolution"])
            net = getattr(wrapper, "_model", None)        # the TAPNext network, as tapnet's TAPNext++ demo uses it
            if net is None:
                raise RuntimeError("this tapnet version's TAPNextPP does not expose its TAPNext network (_model)")
        self._certainty = getattr(utils, "tracker_certainty", None)
        if p["use_certainty"] and self._certainty is None:
            raise RuntimeError("use_certainty=True needs tapnet's tracker_certainty, which this tapnet version lacks")
        self._net = (device, net)
        return net

    def _autocast(self, torch: Any, device: str) -> Any:
        import contextlib

        if self.params["autocast"] and str(device).startswith("cuda"):
            return torch.amp.autocast("cuda", dtype=torch.float16)    # as tapnet's demos run on CUDA
        return contextlib.nullcontext()

    def _step(self, torch: Any, net: Any, frame: np.ndarray, queries: np.ndarray | None, state: Any,
              device: str) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, Any]:
        """One online step of the network.

        ``frame`` is (S, S, 3) ``float32`` in [-1, 1]; ``queries`` (Q, 3) ``float32`` as (t, y, x) in
        query space, given on the first step only. Returns positions (Q, 2) as (y, x) in query space,
        visibility logits (Q,), the certainty (Q,) or ``None``, and the new recurrent state.
        """
        video = torch.from_numpy(np.ascontiguousarray(frame[None, None])).to(device)    # 1, 1, S, S, 3
        with torch.no_grad(), self._autocast(torch, device):
            if state is None:
                qp = torch.from_numpy(np.ascontiguousarray(queries[None])).to(device)      # 1, Q, 3
                tracks, track_logits, vis_logits, state = net(video=video, query_points=qp)
            else:
                tracks, track_logits, vis_logits, state = net(video=video, state=state)
            cert = None
            if self._certainty is not None:
                cert = self._certainty(tracks.float(), track_logits.float(),
                                       radius=int(self.params["certainty_radius"]))
        yx = tracks[0, 0].float().cpu().numpy().astype(float)                            # Q, 2
        vis = vis_logits[0, 0, :, 0].float().cpu().numpy().astype(float)                 # Q
        c = None if cert is None else cert[0, 0, :, 0].float().cpu().numpy().astype(float)
        return yx, vis, c, state

    def track(self, video: Any, queries: np.ndarray, frames: Sequence[int]) -> PointTracks:
        p = self.params
        frames = [int(f) for f in frames]
        index = {f: k for k, f in enumerate(frames)}
        Q = np.asarray(queries, float).reshape(-1, 3)
        n, T = len(Q), len(frames)
        pts = np.full((n, T, 2), np.nan)
        visible = np.zeros((n, T), bool)
        conf = np.full((n, T), np.nan)
        rows = np.array([k for k in range(n) if int(Q[k, 0]) in index], int)
        has_conf = False
        if len(rows):
            torch = require("torch", "perception")
            start = np.array([index[int(Q[k, 0])] for k in rows], int)
            k0 = int(start.min())                         # frames before every query carry no output
            device = pick_device(p["device"])
            net = self._model(device)
            qh, qw = (float(s) for s in getattr(net, "image_size", (QUERY_SIZE, QUERY_SIZE)))
            thr = float(p["visibility_threshold"])
            state, qm, sx, sy = None, None, 1.0, 1.0
            for ti in range(k0, T):
                rgb = np.asarray(video.read(frames[ti]))
                if state is None:
                    h, w = rgb.shape[:2]
                    sx, sy = qw / w, qh / h
                    # (t, y, x): t indexes the frames fed to the network, starting at frames[k0]
                    qm = np.stack([start - k0, Q[rows, 2] * sy, Q[rows, 1] * sx], axis=1).astype(np.float32)
                yx, logit, cert, state = self._step(torch, net, model_frame(rgb, p["resolution"]),
                                                    qm if state is None else None, state, device)
                score = sigmoid(logit)
                if p["use_certainty"]:
                    score = score * cert
                uv = np.stack([yx[:, 1] / sx, yx[:, 0] / sy], axis=1)
                on = (ti >= start) & (score > thr) & np.all(np.isfinite(uv), axis=1)
                pts[rows[on], ti] = uv[on]
                visible[rows[on], ti] = True
                if cert is not None:
                    has_conf = True
                    conf[rows[on], ti] = np.clip(cert[on], 0.0, 1.0)
        pts[~visible] = np.nan
        conf[~visible] = np.nan
        return PointTracks(np.asarray(frames), pts, visible, [self.provenance()], conf if has_conf else None)
