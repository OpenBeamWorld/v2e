# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""End to end: a synthetic clip on disk -> ``reconstruct`` -> world package -> compared with truth.

The clip is written as PNG frames (as a user's frames would be) and reconstructed with the CPU
baselines from a ``v2e.reconstruct/1`` configuration: 16 noisy surveyed points calibrate the
reference frame, one box prompt names the car, everything else is automatic. Tolerances:

* camera: landmarks reproject within 1 px RMS on every tenth frame;
* car: centre trajectory within 0.15 m RMS, heading within 1 degree RMS, each dimension within
  0.1 m, and the reported position sigmas not smaller than a third of the actual error;
* held-out frames: median silhouette IoU at least 0.9;
* the world package passes the ``v2e.*`` schema checks and reads back through
  :func:`openbeam_v2e.world.read_world`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from _perception_clips import car_ptz
from openbeam_v2e.perception import interfaces, registry
from openbeam_v2e.perception.types import Segmentation
from openbeam_v2e.reconstruct import geom, pipeline, silhouette
from openbeam_v2e.schema import check_file


def _write_case(tmp: Path) -> Path:
    clip = car_ptz()
    clip.write(tmp / "clip", masks=False)
    world, image = clip.landmark_observations(0, noise_px=0.5, seed=1)
    sel = np.random.default_rng(3).choice(len(world), 16, replace=False)
    x0, y0, x1, y1 = silhouette.mask_bbox(clip.masks["car"][0])
    (tmp / "annotations.json").write_text(json.dumps({
        "schema": "v2e.annotations/1",
        "objects": {"car_1": {"class": "vehicle.car",
                              "prompts": [{"frame": 0, "box": [x0 - 3, y0 - 3, x1 + 3, y1 + 3]}]}}}))
    cfg = {"schema": "v2e.reconstruct/1",
           "video": {"path": "clip/frames", "fps": clip.fps},
           "camera": {"calibration": {"sigma_px": 0.5, "points": [
               {"world": world[i].tolist(), "image": image[i].tolist()} for i in sel]}},
           "segmentation": {"model": "baseline.background", "params": {"neighbours": 12}},
           "annotations": "annotations.json",
           "holdout": {"every": 5, "offset": 2},
           "cache": "cache"}
    path = tmp / "reconstruct.json"
    path.write_text(json.dumps(cfg))
    return path


@pytest.fixture(scope="module")
def case(tmp_path_factory: pytest.TempPathFactory) -> dict:
    tmp = tmp_path_factory.mktemp("v2e_reconstruct")
    cfg = _write_case(tmp)
    summary = pipeline.reconstruct(config_path=cfg, out_dir=tmp / "world")
    return {"tmp": tmp, "cfg": cfg, "summary": summary, "out": tmp / "world"}


def test_summary_reports_models_objects_and_residuals(case: dict) -> None:
    s = case["summary"]
    assert s["schema"] == "v2e.reconstruct-summary/1" and s["frames"] == 40
    assert s["camera"]["method"].endswith("+refine") and s["camera"]["reprojection_rms_px"] < 1.0
    assert {"camera_calibration:baseline.points", "camera_tracking:baseline.rotation_zoom",
            "segmentation:baseline.background", "pose:baseline.silhouette"} <= set(s["models"])
    (obj,) = s["objects"]
    assert obj["object_id"] == "car_1" and obj["iou_heldout_median"] >= 0.9
    assert not s["warnings"]


def test_world_package_passes_the_schemas_and_reads_back(case: dict) -> None:
    from openbeam_v2e.world import read_world

    out: Path = case["out"]
    for name in ("world.json", "camera_track.json", "objects.json", "uncertainty.json", "outlines.json"):
        assert check_file(out / "v2e" / name) == [], name
    w = read_world(out)
    assert w.kind == "reconstruction" and w.frame_count == 40 and w.has_object("car_1")
    assert {m["id"] for m in w.models} >= {"baseline.points", "baseline.background"}
    assert w.residuals["held_out"]["car_1.silhouette_iou_median"] >= 0.9
    report = pipeline.load_reconstruction(out)
    assert report["schema"] == "v2e.reconstruction/1"
    assert np.asarray(report["objects"]["car_1"]["cov"]).shape == (40, 6, 6)


def test_reconstruction_matches_truth(case: dict) -> None:
    from openbeam_v2e.world import read_world

    clip = car_ptz()
    w = read_world(case["out"])
    for f in range(0, 40, 10):
        world, uv = clip.landmark_observations(f)
        cam = w.camera_track.camera(f)
        proj, _, _ = cam.project(world)
        assert np.sqrt(np.mean(np.sum((proj - uv) ** 2, axis=1))) < 1.0
    car = clip.object("car")
    obj = w.object("car_1")
    truth_centre = car.poses.position + np.einsum("fij,j->fi", car.poses.R_wb, [0.0, 0.0, car.hull.height / 2])
    err = obj.centre_track() - truth_centre
    rms = float(np.sqrt(np.mean(np.sum(err ** 2, axis=1))))
    assert rms < 0.15
    yaw_t = np.array([geom.yaw_pitch_roll(R)[0] for R in car.poses.R_wb])
    yaw_e = np.array([geom.yaw_pitch_roll(R)[0] for R in obj.rotations()])
    assert np.degrees(np.sqrt(np.mean(geom.wrap_angle(yaw_e - yaw_t) ** 2))) < 1.0
    dims = np.array([obj.dims_m[k].value for k in ("length", "width", "height")])
    assert np.abs(dims - np.array(car.hull.dims)).max() < 0.1
    # the calibration's uncertainty is carried into the track: sigmas are not overconfident
    assert np.median(obj.sigma_position_m) > rms / 3


def test_second_run_reuses_cached_stages(case: dict) -> None:
    cfg = pipeline.ReconstructionConfig.load(case["cfg"])
    result = pipeline.run_pipeline(cfg)
    assert result.cache_use == {"camera_track": "cache", "segmentation": "cache"}
    assert np.allclose(result.tracks["car_1"].position,
                       np.asarray(pipeline.load_reconstruction(case["out"])["objects"]["car_1"]["position_m"]))


class _TruthMasks(interfaces.Segmenter):
    """Stands in for any external segmentation model."""

    name = "test.truth_masks"

    def segment(self, video, prompts, frames, cameras=None) -> Segmentation:
        clip = car_ptz()
        seg = Segmentation(*video.size, provenance=[self.provenance()])
        for f in frames:
            seg.track("car_1").set(f, clip.masks["car"][f])
        return seg


def test_any_registered_model_can_replace_a_stage(case: dict) -> None:
    registry.register(_TruthMasks)
    try:
        s = pipeline.reconstruct(config_path=case["cfg"], models={"masks": "test.truth_masks", "camera": "static"})
    finally:
        registry.unregister("segmentation", "test.truth_masks")
    assert "segmentation:test.truth_masks" in s["models"] and "camera_tracking:baseline.static" in s["models"]


def test_configuration_errors_are_clear(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsupported configuration format"):
        pipeline.ReconstructionConfig.parse({"schema": "v2e.reconstruct/9"})
    with pytest.raises(ValueError, match="unknown stage"):
        pipeline.ReconstructionConfig.parse({}).apply_models({"no_such_stage": "baseline"})
    with pytest.raises(ValueError, match="no video"):
        pipeline.reconstruct(config={"schema": "v2e.reconstruct/1"})
