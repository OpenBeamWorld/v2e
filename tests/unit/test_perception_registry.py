# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""The perception registry, third-party models and the optional adapters' glue.

The adapter tests replace the heavy libraries with small stand-ins that follow the libraries'
documented call patterns; they check the conversion between V2E types and the library's inputs
and outputs, not the models themselves (those need the optional extras and their weights).
"""

from __future__ import annotations

import ast
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pytest

from openbeam_v2e.perception import interfaces, registry
from openbeam_v2e.perception.types import CameraTrack, ObjectPrompt, Segmentation

SRC = Path(__file__).resolve().parents[2] / "src" / "openbeam_v2e"


def test_every_builtin_model_loads_and_implements_its_interface() -> None:
    for (kind, name), _ in registry.BUILTINS.items():
        cls = registry.get(kind, name)
        assert issubclass(cls, interfaces.INTERFACES[kind]), name
        assert cls.kind == kind and cls.name == name
        d = cls.describe()
        assert d["kind"] == kind and isinstance(d["available"], bool)


def test_baselines_need_only_the_base_install() -> None:
    for (kind, name), path in registry.BUILTINS.items():
        if name.startswith("baseline."):
            cls = registry.get(kind, name)
            assert cls.requires == () and cls.extras == ()
            assert cls.check() == (True, "ok")


def test_adapters_import_their_libraries_lazily() -> None:
    for path in sorted((SRC / "perception" / "adapters").glob("*.py")):
        tree = ast.parse(path.read_text())
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                mod = node.names[0].name if isinstance(node, ast.Import) else (node.module or "")
                assert mod.split(".")[0] in {"__future__", "openbeam_v2e", "numpy", "typing", "tempfile",
                                             "pathlib", "hashlib", "importlib"}, (path.name, mod)


def test_unavailable_model_is_refused_with_the_extra_to_install(monkeypatch: pytest.MonkeyPatch) -> None:
    cls = registry.get("segmentation", "sam2")
    monkeypatch.setattr(cls, "requires", ("a_module_that_is_not_installed",))
    with pytest.raises(RuntimeError, match=r"openbeam-v2e\[sam2\]"):
        registry.create("segmentation", "sam2")
    with pytest.raises(KeyError, match="known"):
        registry.get("segmentation", "no.such.model")


class _HalfMasks(interfaces.Segmenter):
    """A third-party segmenter: returns each prompt's region on its own frame only."""

    name = "test.half"

    def segment(self, video: Any, prompts: Sequence[ObjectPrompt], frames: Sequence[int],
                cameras: CameraTrack | None = None) -> Segmentation:
        w, h = video.size
        seg = Segmentation(w, h, provenance=[self.provenance()])
        for p in prompts:
            m = np.zeros((h, w), bool)
            x0, y0, x1, y1 = (int(v) for v in p.box)
            m[y0:y1, x0:x1] = True
            seg.track(p.object_id).set(p.frame, m)
        return seg


def test_runtime_registration_and_entry_points(monkeypatch: pytest.MonkeyPatch) -> None:
    registry.register(_HalfMasks)
    try:
        model = registry.create("segmentation", "test.half")
        assert model.provenance().model == "test.half"
        assert ("segmentation", "test.half") in registry.names()
    finally:
        registry.unregister("segmentation", "test.half")

    class EP:
        name = "plugin_depth"

        def load(self) -> type:
            class PluginDepth(interfaces.DepthEstimator):
                name = "plugin.depth"

                def estimate(self, image: np.ndarray, camera: Any = None) -> Any:
                    raise NotImplementedError

            return PluginDepth

    class Broken:
        name = "broken"

        def load(self) -> type:
            raise ImportError("missing dependency of a plugin")

    monkeypatch.setattr(registry, "_entry_points", None)
    monkeypatch.setattr(registry.metadata, "entry_points", lambda group=None: [EP(), Broken()])
    assert registry.get("depth", "plugin.depth").name == "plugin.depth"
    assert any(d["name"] == "plugin.depth" for d in registry.describe("depth"))
    with pytest.raises(TypeError):
        registry.register(type("NotAModel", (interfaces.Detector,), {"name": "x", "kind": "segmentation"}))


# --------------------------------------------------------------------------- adapter glue


class _FakeTensor:
    def __init__(self, a: np.ndarray) -> None:
        self.a = np.asarray(a)

    def cpu(self) -> "_FakeTensor":
        return self

    def numpy(self) -> np.ndarray:
        return self.a

    def __gt__(self, other: float) -> "_FakeTensor":
        return _FakeTensor(self.a > other)

    def __getitem__(self, k: Any) -> "_FakeTensor":
        return _FakeTensor(self.a[k])


def _fake_torch() -> types.ModuleType:
    torch = types.ModuleType("torch")

    @contextmanager
    def inference_mode() -> Any:
        yield

    torch.inference_mode = inference_mode  # type: ignore[attr-defined]
    torch.no_grad = inference_mode  # type: ignore[attr-defined]
    return torch


def test_sam2_adapter_converts_prompts_and_masks(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("PIL")
    from openbeam_v2e.reconstruct import synthetic, silhouette

    clip = synthetic.ground_vehicle_clip(width=96, height=64, n_frames=4, ptz=False)
    calls: dict[str, Any] = {"points_or_box": [], "mask": []}

    class Predictor:
        def init_state(self, video_path: str) -> dict:
            calls["frames"] = sorted(p.name for p in Path(video_path).iterdir())
            return {}

        def add_new_points_or_box(self, inference_state: dict, frame_idx: int, obj_id: int, **kw: Any) -> None:
            calls["points_or_box"].append((frame_idx, obj_id, sorted(kw)))

        def add_new_mask(self, inference_state: dict, frame_idx: int, obj_id: int, mask: np.ndarray) -> None:
            calls["mask"].append((frame_idx, obj_id, mask.shape))

        def propagate_in_video(self, state: dict, start_frame_idx: int = 0, reverse: bool = False) -> Any:
            order = range(start_frame_idx, -1, -1) if reverse else range(start_frame_idx, 4)
            for k in order:
                logits = np.where(clip.masks["car"][k], 5.0, -5.0)[None, None]
                yield k, [1, 2], _FakeTensor(np.concatenate([logits, -np.ones_like(logits)]))

    vp = types.ModuleType("sam2.sam2_video_predictor")
    vp.SAM2VideoPredictor = types.SimpleNamespace(from_pretrained=lambda model_id, device: Predictor())
    monkeypatch.setitem(sys.modules, "sam2", types.ModuleType("sam2"))
    monkeypatch.setitem(sys.modules, "sam2.sam2_video_predictor", vp)
    monkeypatch.setitem(sys.modules, "torch", _fake_torch())
    from openbeam_v2e.perception.adapters.sam2 import Sam2Segmenter

    box = silhouette.mask_bbox(clip.masks["car"][1])
    poly = silhouette.mask_to_polygons(clip.masks["car"][2])[0]
    prompts = [ObjectPrompt("car", 1, box=box), ObjectPrompt("other", 2, polygon=poly)]
    seg = Sam2Segmenter(device="cpu").segment(clip.frame_source(), prompts, range(4))
    assert calls["frames"] == ["00000.jpg", "00001.jpg", "00002.jpg", "00003.jpg"]
    assert calls["points_or_box"] == [(1, 1, ["box"])] and calls["mask"] == [(2, 2, (64, 96))]
    assert seg.tracks["car"].frames() == [0, 1, 2, 3] and "other" not in seg.tracks
    assert np.array_equal(seg.tracks["car"].mask(3), clip.masks["car"][3])
    assert seg.provenance[0].weights == "facebook/sam2.1-hiera-tiny"


def test_transformers_depth_and_detection_adapters(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("PIL")
    tf = types.ModuleType("transformers")
    seen: dict[str, Any] = {}

    def pipeline(task: str, device: str, model: str, **kw: Any) -> Any:
        seen.setdefault("tasks", []).append((task, model))
        if task == "depth-estimation":
            return lambda img: {"predicted_depth": _FakeTensor(np.arange(12.0).reshape(1, 3, 4))}
        return lambda img, **k: [{"score": 0.9, "label": "car", "box": {"xmin": 1, "ymin": 2, "xmax": 30, "ymax": 20}},
                                 {"score": 0.2, "label": "car", "box": {"xmin": 0, "ymin": 0, "xmax": 5, "ymax": 5}},
                                 {"score": 0.8, "label": "person", "box": {"xmin": 9, "ymin": 9, "xmax": 12, "ymax": 30}}]

    tf.pipeline = pipeline  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", tf)
    from openbeam_v2e.perception.adapters.hf_depth import TransformersDepth
    from openbeam_v2e.perception.adapters.hf_detection import TransformersDetector, detections_to_prompts

    img = np.zeros((30, 40, 3), np.uint8)
    dm = TransformersDepth(device="cpu").estimate(img)
    assert dm.depth.shape == (30, 40) and dm.kind == "inverse_relative"
    assert TransformersDepth(model_id="x/Depth-Anything-V2-Metric-Outdoor-Small-hf").params["kind"] == "metric"
    dets = TransformersDetector(device="cpu").detect(img, classes=["car"])
    assert [d.label for d in dets] == ["car"] and dets[0].box == (1.0, 2.0, 30.0, 20.0)
    prompts, classes = detections_to_prompts(dets, frame=3)
    assert prompts[0].object_id == "object_1" and prompts[0].frame == 3 and classes == {"object_1": "vehicle.car"}
    assert ("depth-estimation", "depth-anything/Depth-Anything-V2-Small-hf") in seen["tasks"]


def test_cotracker_requires_an_explicit_weights_source() -> None:
    from openbeam_v2e.perception.adapters.cotracker import CoTrackerPointTracker

    with pytest.raises(RuntimeError, match="allow_hub"):
        CoTrackerPointTracker()._model()
