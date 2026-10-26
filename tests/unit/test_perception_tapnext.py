# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""The ``tapnext`` point-tracking adapter's glue, with stand-ins for PyTorch and tapnet.

The stand-in network follows TAPNext's documented online call pattern (queries as (t, y, x) in
256 x 256 query space on the first call, the recurrent state afterwards, outputs as (y, x)) and
moves each point by a known amount per frame, so the tests check the conversion between V2E's
queries and tracks and the network's inputs and outputs, not the network itself.
"""

from __future__ import annotations

import hashlib
import sys
import types
from contextlib import contextmanager
from typing import Any, Callable

import numpy as np
import pytest

from openbeam_v2e.perception import registry
from openbeam_v2e.perception.types import PointTracks

W, H = 200, 100


class _T:
    """A NumPy-backed stand-in for the few tensor methods the adapter uses."""

    def __init__(self, a: Any) -> None:
        self.a = np.asarray(a)

    def to(self, device: str) -> "_T":
        return self

    def float(self) -> "_T":
        return _T(self.a.astype(np.float32))

    def cpu(self) -> "_T":
        return self

    def numpy(self) -> np.ndarray:
        return self.a

    def __getitem__(self, k: Any) -> "_T":
        return _T(self.a[k])


def _fake_torch() -> types.ModuleType:
    torch = types.ModuleType("torch")

    @contextmanager
    def no_grad() -> Any:
        yield

    torch.no_grad = no_grad  # type: ignore[attr-defined]
    torch.from_numpy = lambda a: _T(np.array(a))  # type: ignore[attr-defined]
    return torch


class _Video:
    """Constant frames whose grey level encodes the frame number (40 + 10 f)."""

    size = (W, H)
    fps = 30.0

    def __init__(self, n: int = 12) -> None:
        self.n = n
        self.reads: list[int] = []

    def __len__(self) -> int:
        return self.n

    def read(self, f: int) -> np.ndarray:
        self.reads.append(f)
        return np.full((H, W, 3), 40 + 10 * f, np.uint8)


class _Net:
    """Stand-in TAPNext: each point moves by ``velocity[j]`` (dy, dx) query-space pixels per frame number."""

    image_size = (256, 256)

    def __init__(self, velocity: list[tuple[float, float]] | None = None,
                 logit: Callable[[int, int], float] | None = None) -> None:
        self.velocity = velocity
        self.logit = logit or (lambda j, f: 3.0)
        self.calls: list[dict[str, Any]] = []

    def to(self, device: str) -> "_Net":
        self.device = device
        return self

    def eval(self) -> "_Net":
        self.evaluated = True
        return self

    def __call__(self, video: _T, query_points: _T | None = None, state: Any = None) -> tuple:
        v = video.a
        assert v.shape[:2] == (1, 1) and v.shape[2] == v.shape[3] and v.shape[4] == 3
        assert v.dtype == np.float32 and -1.0 <= v.min() <= v.max() <= 1.0
        f = int(round(((float(v[0, 0, 0, 0, 0]) + 1.0) * 127.5 - 40.0) / 10.0))
        if state is None:
            assert query_points is not None
            qp, step, origin = query_points.a[0].astype(float), 0, {}
        else:
            assert query_points is None, "queries cannot be added to an existing state"
            qp, step, origin = state.query_points, state.step, state.origin
        self.calls.append({"frame": f, "size": v.shape[2], "step": step,
                           "queries": None if state is not None else qp.copy()})
        q = len(qp)
        tracks = np.zeros((1, 1, q, 2))
        vis = np.zeros((1, 1, q, 1))
        for j in range(q):
            t = qp[j, 0] - step
            if t > 0:                                   # not started: placeholders that look visible
                tracks[0, 0, j] = (-999.0, 999.0)
                vis[0, 0, j, 0] = 9.0
                continue
            if t == 0:
                origin[j] = f
            vel = np.asarray(self.velocity[j] if self.velocity else (0.0, 0.0))
            tracks[0, 0, j] = qp[j, 1:] + (f - origin[j]) * vel
            vis[0, 0, j, 0] = self.logit(j, f)
        new_state = types.SimpleNamespace(step=step + 1, query_points=qp, origin=origin)
        return _T(tracks.astype(np.float32)), _T(np.zeros((1, 1, q, 512))), _T(vis.astype(np.float32)), new_state


def _install(monkeypatch: pytest.MonkeyPatch, net: _Net, certainty: list[float] | None = None) -> list[tuple]:
    """Stand-in ``torch`` and ``tapnet`` modules; returns a log of the loader calls."""
    log: list[tuple] = []
    names = ["tapnet", "tapnet.tapnext", "tapnet.tapnext.tapnext_torch", "tapnet.tapnext.tapnext_torch_utils",
             "tapnet.tapnextpp", "tapnet.tapnextpp.votsp2026", "tapnet.tapnextpp.votsp2026.model"]
    mods = {name: types.ModuleType(name) for name in names}

    def make_tapnext(image_size: tuple[int, int], **kw: Any) -> _Net:
        log.append(("TAPNext", tuple(image_size)))
        return net

    def restore(model: Any, ckpt_path: str) -> Any:
        log.append(("npz", str(ckpt_path)))
        return model

    class TAPNextPP:
        @classmethod
        def from_checkpoint(cls, path: str, device: str = "cuda", half_precision: bool = False,
                            compile_model: bool = False, input_resolution: int = 256) -> Any:
            log.append(("pt", str(path), device, input_resolution))
            return types.SimpleNamespace(_model=net)

    mods["tapnet.tapnext.tapnext_torch"].TAPNext = make_tapnext  # type: ignore[attr-defined]
    utils = mods["tapnet.tapnext.tapnext_torch_utils"]
    utils.restore_model_from_jax_checkpoint = restore  # type: ignore[attr-defined]
    if certainty is not None:
        def tracker_certainty(coord_yx: _T, track_logits: _T, radius: int = 8) -> _T:
            assert coord_yx.a.shape[-1] == 2 and track_logits.a.shape[-1] == 512
            log.append(("radius", radius))
            return _T(np.asarray(certainty, np.float32).reshape(1, 1, -1, 1))

        utils.tracker_certainty = tracker_certainty  # type: ignore[attr-defined]
    mods["tapnet.tapnextpp.votsp2026.model"].TAPNextPP = TAPNextPP  # type: ignore[attr-defined]
    for name, mod in mods.items():
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.setitem(sys.modules, "torch", _fake_torch())
    return log


@pytest.fixture
def ckpt(tmp_path: Any) -> Any:
    path = tmp_path / "tapnextpp_ckpt.pt"
    path.write_bytes(b"stand-in weights")
    return path


def _tracker(**kw: Any) -> Any:
    from openbeam_v2e.perception.adapters.tapnext import TapNextPointTracker

    return TapNextPointTracker(device="cpu", **kw)


@pytest.mark.parametrize("resolution", [256, 512])
def test_queries_and_tracks_are_scaled_and_swapped_both_ways(monkeypatch: pytest.MonkeyPatch, ckpt: Any,
                                                              resolution: int) -> None:
    net = _Net(velocity=[(2.0, 5.0), (-3.0, 1.0)])
    log = _install(monkeypatch, net)
    queries = np.array([[0, 50.5, 20.5], [0, 120.25, 70.75]])
    tracks = _tracker(checkpoint=str(ckpt), resolution=resolution).track(_Video(), queries, [0, 1, 2, 3])

    assert log == [("pt", str(ckpt), "cpu", resolution)]
    # V2E (u, v) with pixel centres at 0.5 -> TAPNext (t, y, x) in 256 x 256 query space: scaled, not shifted
    sx, sy = 256 / W, 256 / H
    assert np.allclose(net.calls[0]["queries"], [[0, 20.5 * sy, 50.5 * sx], [0, 70.75 * sy, 120.25 * sx]])
    assert [c["size"] for c in net.calls] == [resolution] * 4          # frames at the model resolution
    assert [c["queries"] is None for c in net.calls] == [False, True, True, True]
    assert np.allclose(tracks.points[:, 0], queries[:, 1:], atol=1e-4)   # the round trip is exact
    f = np.arange(4.0)
    assert np.allclose(tracks.points[0, :, 0], 50.5 + 5.0 * f / sx, atol=1e-4)
    assert np.allclose(tracks.points[0, :, 1], 20.5 + 2.0 * f / sy, atol=1e-4)
    assert np.allclose(tracks.points[1, :, 0], 120.25 + 1.0 * f / sx, atol=1e-4)
    assert np.allclose(tracks.points[1, :, 1], 70.75 - 3.0 * f / sy, atol=1e-4)
    assert tracks.visible.all() and list(tracks.frames) == [0, 1, 2, 3]


def test_descending_frames_and_queries_starting_mid_window(monkeypatch: pytest.MonkeyPatch, ckpt: Any) -> None:
    net = _Net(velocity=[(0.0, 4.0), (2.0, 0.0)])
    _install(monkeypatch, net)
    video = _Video()
    frames = [9, 8, 7, 6, 5]
    queries = np.array([[9, 30.0, 40.0], [7, 60.0, 20.0], [3, 10.0, 10.0]])   # the last is outside frames
    tracks = _tracker(checkpoint=str(ckpt)).track(video, queries, frames)

    assert video.reads == frames and [c["frame"] for c in net.calls] == frames
    assert np.allclose(net.calls[0]["queries"][:, 0], [0, 2])                 # local indices in the given order
    assert len(net.calls[0]["queries"]) == 2
    assert list(tracks.frames) == frames
    sx, sy = 256 / W, 256 / H
    assert np.allclose(tracks.points[0, :, 0], 30.0 + 4.0 * (np.array(frames) - 9) / sx)
    assert np.allclose(tracks.points[0, :, 1], 40.0)
    # before its query frame (in the given order) a point has no position, even where the network
    # emitted placeholder outputs that look visible
    assert not tracks.visible[1, :2].any() and np.isnan(tracks.points[1, :2]).all()
    assert tracks.visible[1, 2:].all()
    assert np.allclose(tracks.points[1, 2:, 1], 20.0 + 2.0 * (np.array([7, 6, 5]) - 7) / sy)
    assert not tracks.visible[2].any() and np.isnan(tracks.points[2]).all()


def test_online_pass_starts_at_the_earliest_query_frame(monkeypatch: pytest.MonkeyPatch, ckpt: Any) -> None:
    net = _Net()
    _install(monkeypatch, net)
    video = _Video()
    tracks = _tracker(checkpoint=str(ckpt)).track(video, np.array([[3, 10.5, 10.5]]), range(6))
    assert video.reads == [3, 4, 5] and net.calls[0]["queries"][0, 0] == 0
    assert np.isnan(tracks.points[0, :3]).all() and tracks.visible[0, 3:].all()


def test_visibility_threshold_and_nan_where_not_visible(monkeypatch: pytest.MonkeyPatch, ckpt: Any) -> None:
    logits = [0.3, -0.3, 2.0]                     # sigmoid: just above, just below, well above 0.5
    queries = np.array([[0, 10.0, 10.0], [0, 20.0, 20.0], [0, 30.0, 30.0]])
    _install(monkeypatch, _Net(logit=lambda j, f: logits[j]))
    default = _tracker(checkpoint=str(ckpt)).track(_Video(), queries, [0, 1])
    assert default.visible[:, 1].tolist() == [True, False, True]
    _install(monkeypatch, _Net(logit=lambda j, f: logits[j]))
    strict = _tracker(checkpoint=str(ckpt), visibility_threshold=0.6).track(_Video(), queries, [0, 1])
    assert strict.visible[:, 1].tolist() == [False, False, True]
    for t in (default, strict):
        assert np.isnan(t.points[~t.visible]).all() and np.isfinite(t.points[t.visible]).all()
        assert t.confidence is None                # this stand-in tapnet has no tracker_certainty


def test_confidence_is_the_models_certainty(monkeypatch: pytest.MonkeyPatch, ckpt: Any) -> None:
    queries = np.array([[0, 10.0, 10.0], [0, 20.0, 20.0], [0, 30.0, 30.0]])
    logits = [2.0, 2.0, -1.0]                     # sigmoid: well above, well above, below 0.5
    certainty = [0.9, 0.4, 0.8]
    log = _install(monkeypatch, _Net(logit=lambda j, f: logits[j]), certainty)
    tracks = _tracker(checkpoint=str(ckpt), certainty_radius=6).track(_Video(), queries, [0, 1, 2])
    assert ("radius", 6) in log
    assert tracks.visible.tolist() == [[True] * 3, [True] * 3, [False] * 3]
    assert np.allclose(tracks.confidence[:2], [[0.9] * 3, [0.4] * 3]) and np.isnan(tracks.confidence[2]).all()

    _install(monkeypatch, _Net(logit=lambda j, f: logits[j]), certainty)
    both = _tracker(checkpoint=str(ckpt), use_certainty=True).track(_Video(), queries, [0, 1, 2])
    assert both.visible[:, 0].tolist() == [True, False, False]   # sigmoid(2) * 0.4 falls below 0.5


def test_use_certainty_needs_tracker_certainty(monkeypatch: pytest.MonkeyPatch, ckpt: Any) -> None:
    _install(monkeypatch, _Net())
    with pytest.raises(RuntimeError, match="tracker_certainty"):
        _tracker(checkpoint=str(ckpt), use_certainty=True).track(_Video(), np.array([[0, 1.0, 1.0]]), [0])


def test_jax_checkpoint_loads_through_tapnets_converter(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    path = tmp_path / "bootstapnext_ckpt.npz"
    path.write_bytes(b"stand-in")
    net = _Net()
    log = _install(monkeypatch, net)
    tracker = _tracker(checkpoint=str(path))
    tracker.track(_Video(), np.array([[0, 5.0, 5.0]]), [0, 1])
    tracker.track(_Video(), np.array([[0, 5.0, 5.0]]), [0, 1])
    assert log == [("TAPNext", (256, 256)), ("npz", str(path))]       # loaded once per instance
    assert net.device == "cpu" and net.evaluated


def test_provenance_records_the_checkpoint_hash(monkeypatch: pytest.MonkeyPatch, ckpt: Any) -> None:
    from openbeam_v2e.perception.adapters import tapnext

    _install(monkeypatch, _Net())
    tracks = _tracker(checkpoint=str(ckpt), resolution=512).track(_Video(), np.array([[0, 5.0, 5.0]]), [0])
    prov = tracks.provenance[0]
    assert prov.model == "tapnext" and prov.kind == "point_tracking"
    assert prov.implementation == "openbeam_v2e.perception.adapters.tapnext:TapNextPointTracker"
    assert prov.weights == str(ckpt)
    assert prov.weights_sha256 == hashlib.sha256(b"stand-in weights").hexdigest()
    assert prov.params["resolution"] == 512 and prov.params["visibility_threshold"] == 0.5
    assert prov.to_dict()["params"]["checkpoint"] == str(ckpt)

    class Dist:
        def read_text(self, name: str) -> str:
            return ('{"url": "https://github.com/google-deepmind/tapnet.git", '
                    '"vcs_info": {"vcs": "git", "commit_id": "abc123"}}')

    monkeypatch.setattr(tapnext.metadata, "distribution", lambda name: Dist())
    monkeypatch.setattr(tapnext, "versions", lambda *names: {"torch": "2.9.0", "tapnet": "0.1.0"})
    assert _tracker(checkpoint=str(ckpt)).provenance().libraries == {"torch": "2.9.0", "tapnet": "0.1.0 (git abc123)"}


def test_weights_must_be_a_local_file_and_nothing_is_downloaded(tmp_path: Any) -> None:
    with pytest.raises(RuntimeError, match="checkpoint=<local path>"):
        _tracker()._model("cpu")
    with pytest.raises(FileNotFoundError):
        _tracker(checkpoint=str(tmp_path / "missing.pt"))._model("cpu")
    with pytest.raises(ValueError, match="multiple of 8"):
        _tracker(resolution=300)


def test_missing_libraries_name_the_install(monkeypatch: pytest.MonkeyPatch, ckpt: Any) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(ImportError, match=r"openbeam-v2e\[perception\]"):
        _tracker(checkpoint=str(ckpt))._model("cpu")
    monkeypatch.setitem(sys.modules, "torch", _fake_torch())
    monkeypatch.setitem(sys.modules, "tapnet", None)
    with pytest.raises(ImportError, match=r"git\+https://github.com/google-deepmind/tapnet.git"):
        _tracker(checkpoint=str(ckpt))._model("cpu")

    cls = registry.get("point_tracking", "tapnext")
    monkeypatch.setattr(cls, "requires", ("a_module_that_is_not_installed",))
    with pytest.raises(RuntimeError, match=r"openbeam-v2e\[perception\].*google-deepmind/tapnet"):
        registry.create("point_tracking", "tapnext")


def test_registry_lists_and_describes_tapnext() -> None:
    from openbeam_v2e.perception.adapters.tapnext import TapNextPointTracker
    from openbeam_v2e.reconstruct.clip_config import ReconstructionConfig

    assert registry.get("point_tracking", "tapnext") is TapNextPointTracker
    assert ("point_tracking", "tapnext") in registry.names()
    d = next(m for m in registry.describe("point_tracking") if m["name"] == "tapnext")
    assert d["extras"] == ["perception"] and isinstance(d["available"], bool)
    assert d["implementation"] == "openbeam_v2e.perception.adapters.tapnext:TapNextPointTracker"
    cfg = ReconstructionConfig.parse({})
    cfg.apply_models({"points": "tapnext"})
    assert cfg.tracking.model == "baseline.point_tracks" and cfg.tracking.params == {"point_tracker": "tapnext"}


def test_nothing_is_loaded_when_no_query_is_in_the_frames() -> None:
    tracks = _tracker().track(_Video(), np.array([[20, 1.0, 1.0]]), [0, 1])
    assert np.isnan(tracks.points).all() and not tracks.visible.any() and tracks.confidence is None
    assert tracks.provenance[0].weights is None


def test_model_frame_matches_half_pixel_bilinear_resampling() -> None:
    from openbeam_v2e.perception.adapters.tapnext import model_frame

    rgb = np.random.default_rng(0).integers(0, 256, (8, 16, 3)).astype(np.uint8)
    out = model_frame(rgb, 4)                     # 2 x 4 blocks: the mean of the middle 2 x 2
    blocks = rgb.astype(float).reshape(4, 2, 4, 4, 3)[:, :, :, 1:3].mean(axis=(1, 3))
    assert out.shape == (4, 4, 3) and out.dtype == np.float32
    assert np.allclose(out, blocks / 127.5 - 1.0, atol=1e-5)
    same = np.random.default_rng(1).integers(0, 256, (8, 8, 3)).astype(np.uint8)
    assert np.allclose(model_frame(same, 8), same / 127.5 - 1.0, atol=1e-6)
    assert model_frame(np.zeros((5, 7), np.uint8), 8).shape == (8, 8, 3)


def test_point_tracks_confidence_is_optional() -> None:
    from openbeam_v2e.perception.baseline.point_tracking import LucasKanadeTracker
    from openbeam_v2e.reconstruct.video import ArrayFrames

    t = PointTracks(np.arange(2), np.zeros((1, 2, 2)), np.ones((1, 2), bool), [])
    assert t.confidence is None
    rgb = (np.random.default_rng(2).random((3, 48, 64, 3)) * 255).astype(np.uint8)
    lk = LucasKanadeTracker().track(ArrayFrames(rgb), np.array([[0, 30.5, 20.5]]), [0, 1, 2])
    assert lk.confidence is None


@pytest.mark.parametrize("radius", [0, 9])
def test_certainty_radius_outside_tapnets_window_is_refused(radius: int) -> None:
    with pytest.raises(ValueError, match="certainty_radius"):
        _tracker(certainty_radius=radius)


def test_queries_run_in_groups_each_from_its_own_first_frame(monkeypatch: pytest.MonkeyPatch, ckpt: Any) -> None:
    net = _Net()
    _install(monkeypatch, net)
    video = _Video()
    queries = np.array([[5, 10.5, 10.5], [1, 20.5, 20.5], [1, 30.5, 30.5]])
    tracks = _tracker(checkpoint=str(ckpt), max_queries=2).track(video, queries, range(8))
    starts = [c for c in net.calls if c["queries"] is not None]
    # sorted by start frame: the two frame-1 queries share a pass from frame 1, the frame-5 one starts at 5
    assert [(c["frame"], len(c["queries"])) for c in starts] == [(1, 2), (5, 1)]
    assert video.reads == list(range(1, 8)) + list(range(5, 8))
    assert tracks.visible[1:, 1:].all() and not tracks.visible[1:, 0].any()
    assert tracks.visible[0, 5:].all() and not tracks.visible[0, :5].any()


def test_non_finite_queries_are_left_out(monkeypatch: pytest.MonkeyPatch, ckpt: Any) -> None:
    net = _Net()
    _install(monkeypatch, net)
    queries = np.array([[0, 10.5, 10.5], [0, np.nan, 20.5], [0, 30.5, np.inf]])
    tracks = _tracker(checkpoint=str(ckpt)).track(_Video(), queries, [0, 1, 2])
    assert len(net.calls[0]["queries"]) == 1 and np.isfinite(net.calls[0]["queries"]).all()
    assert tracks.visible[0].all()
    assert not tracks.visible[1:].any() and np.isnan(tracks.points[1:]).all()


def test_network_and_checkpoint_hash_are_shared_across_trackers(monkeypatch: pytest.MonkeyPatch, ckpt: Any) -> None:
    from openbeam_v2e.perception.adapters import tapnext as T

    log = _install(monkeypatch, _Net())
    calls = []
    monkeypatch.setattr(T, "sha256_of", lambda path: calls.append(path) or "0" * 64)
    for _ in range(3):
        tr = _tracker(checkpoint=str(ckpt))
        tr.track(_Video(), np.array([[0, 10.5, 10.5]]), [0, 1])
        assert tr.provenance().weights_sha256 == "0" * 64
    assert log.count(("pt", str(ckpt), "cpu", 256)) == 1 and len(calls) == 1
