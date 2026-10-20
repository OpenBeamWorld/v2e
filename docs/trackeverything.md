# TrackEverything in the video-import workflow

`trackeverything` is a V2E **point-tracking model**. The registry calls the upstream
Python API on a CUDA worker, then returns `PointTracks` to V2E. Optional artifacts
retain the raw 3-D queries and Pi3 cameras for inspection and subsequent fitting.
This is an offline perception integration; the game consumes reconstructed content
and actuator goals through its existing import/connector interfaces.

## Environment

Use the upstream Linux/Python 3.10/PyTorch 2.8/CUDA 12.9 environment. The base V2E
installation still needs only NumPy/SciPy. Install the upstream dependencies in a
dedicated environment, alongside V2E:

```sh
git clone https://github.com/ayushjain1144/trackeverything.git /path/to/deps/trackeverything
git -C /path/to/deps/trackeverything checkout --detach f0c18dd9714b9f85825c45a1c3aae685da4e6222
python -m pip install -r /path/to/deps/trackeverything/requirements.txt
python -m pip install -e /path/to/v2e
hf auth login
```

The checkout must keep the directory name `trackeverything` and unmodified tracked
files. The adapter verifies its revision and the imported module's location. Model
listing does not import Torch or download weights. Inference downloads these pinned
snapshots into the usual Hugging Face cache:

| Model | Revision |
| --- | --- |
| `katefgroup/TrackEverything` | `1957810f46ef684154fec91a9b8c0e6a68cd695e` |
| `yyfz233/Pi3` | `ae722e7039287d0c8fde9f11f197f804f44b510c` |

TrackEverything model access must be granted to the authenticated account. Pi3 is
selected explicitly because the pinned upstream default VGGT-Ω configuration uses
an author-local checkpoint. Model code and weights remain external, under their
upstream licences (TrackEverything: CC-BY-NC-4.0).

## Track a clip directly

```python
import numpy as np
from openbeam_v2e.perception import registry
from openbeam_v2e.reconstruct.video import open_frames

video = open_frames("clip.mp4")  # alternatively an image-only frame directory
tracker = registry.create(
    "point_tracking", "trackeverything",
    tracker_source="/path/to/deps/trackeverything",
    output_dir="/path/to/new-tracking-result",  # optional; must not exist yet
)
# Source-frame IDs, followed by V2E image coordinates (pixel centres at +0.5).
queries = np.array([[0, 248.5, 153.5], [30, 300.5, 200.5]])
tracks = tracker.track(video, queries, range(len(video)))
print(tracks.points.shape)  # (2, frame_count, 2)
print(tracks.visible.shape) # (2, frame_count)
```

Frames must be increasing source-frame integers. A seed must belong to the
selected frame grid, including when cropping or subsampling. The adapter selects
frames once and remaps seed times to upstream indices; it does not apply a second
stride. Source-frame IDs and query order are preserved in the result.

Projection uses the predicted world-to-camera transform and the upstream
`(size - 1)` resize convention. The adapter converts between upstream pixel indices
and V2E's pixel centres. Scene normalization is already shared by coordinates and
camera translation and is not applied again. Returned pixels are NaN when the
visibility probability is at or below `visibility_threshold` (default 0.5), before
the seed frame, behind the camera, outside the image, or nonfinite. These are model
predictions, not independently measured visibility or accuracy.

With `output_dir`, the run writes:

- `query-tracks.npz`: raw `coords` `(T,N,3)`, `visibility_logits` `(T,N)`, inference
  intrinsics/extrinsics, normalization, image sizes, source-frame IDs/times, original and
  remapped queries, and V2E `points`/`visible` arrays. Load with `allow_pickle=False`.
- `tracking-result.json`: completion state, source RGB hash, query/frame identity and FPS,
  code/model pins, weight hashes, library versions, and the NPZ hash. A failed model
  call leaves a failed receipt rather than a completed prediction.

## Select it in reconstruction

The existing `baseline.point_tracks` camera stage can consume the adapter directly:

```json
{"camera": {"tracking": {
  "model": "baseline.point_tracks",
  "params": {
    "point_tracker": "trackeverything",
    "point_tracker_params": {"tracker_source": "/path/to/deps/trackeverything"}
  }
}}}
```

Include this section in the normal `v2e.reconstruct/1` configuration, alongside
calibration evidence and object annotations, then run `openbeam-v2e reconstruct`.
The camera stage supplies background feature seeds and solves V2E's fixed-position
pan/tilt/zoom model from the projected tracks. Raw Pi3 camera estimates are not
silently substituted for calibrated metric cameras.

## Check object tracks against independent masks

Predicted visibility alone does not exclude drift onto the background. For queries
that belong to an object, apply `perception.point_support.filter_by_masks` using
masks observed independently of those tracks. For example, after the direct call
above, load masks prepared by an external segmenter or annotation tool:

```python
from openbeam_v2e.perception.point_support import filter_by_masks

segmentation = registry.create(
    "segmentation", "baseline.files", root="/path/to/observations", objects=["cube"],
).segment(video, [], tracks.frames)
# For this example both queries refer to the cube; other queries can be omitted.
supported, rejected = filter_by_masks(
    tracks, {0: segmentation.track("cube"), 1: segmentation.track("cube")}, margin_px=2.0,
)
```

Mask files use `masks/cube/{source_frame:06d}.png` at the source resolution.
Missing/empty masks reject the assigned points even when the tracker is confident.
The helper returns copied tracks with NaN gaps and a separate rejection array;
provenance records mask hashes, query assignments and the support margin. Queries
are rasterized into their containing pixel; the margin dilates masks by Euclidean
pixel-centre distance. No point is moved onto the mask or made visible by it.

This is an explicit observation check before fitting, not a replacement tracker.
An independently reacquired visible patch can support object observations, but
cannot restore the identity of a hidden corner. Mask agreement is necessary image
support, not a material-point accuracy label. Use independently validated masks;
the filter cannot detect an incorrectly segmented object.

## SO-101 boundary

Tracking supplies image correspondences and estimated 3-D motion. Recreating the
specific arm, cube, tray and mat still requires their geometry, metric alignment
and materials. The SO-101's linked parts require URDF-constrained joint fitting;
V2E's existing rigid-object silhouette fitter does not perform that articulated
fit. Physical carrying must be measured in OpenBeam after the inferred goals run
through native servos and contacts.

The initial SO-101 research runs (138 frames at 10 fps and 414 at 30 fps, ten
manually seeded queries) completed on an L40S, but both lost the cube during its
transfer. Comparing the four cube-query mean with the visible green-patch centroid
after frame 300 gave approximately 205 px and 148 px mean disagreement respectively.
This is a diagnostic mismatch, not a material-point accuracy benchmark. Model
visibility sometimes remained high despite the drift. Those results do not qualify
an exact scene reconstruction or an actuator trajectory.

Geometry/frame-grid unit tests run offline:

```sh
python -m pytest tests/unit/test_trackeverything.py tests/unit/test_perception_registry.py tests/unit/test_point_support.py
```

They cover camera direction, normalization invariance, non-square resizing,
pixel origins, source-frame remapping, missing predictions and artifact receipts.
Support tests cover confident background drift, occlusion/reappearance, mask holes,
bounded margins and preservation of the input arrays.
GPU inference and physical recreation are separate checks.

### Direct adapter GPU smoke (2026-10-20)

The V2E registry adapter processed the full 414-frame/30 fps SO-101 episode with
ten queries on an L40S, producing `(10, 414, 2)` pixel tracks and 1,794 predicted
visible samples. Input images, source-frame IDs, weight hashes and artifact hashes
were checked; the 414-frame H.264 overlay fully decoded. This establishes the
direct inference/conversion path, not tracking accuracy. The cube was again lost
in transfer (178.45 px mean late-clip green-patch diagnostic mismatch).

Local evidence: `so101-trackeverything-run-03/` beside the SO-101 checkout, including
`result/v2e-integration-check.json`, `result/tracking-result.json`, `run-receipt.json`
and `review/trackeverything-overlay.mp4`. The GPU worker was stopped and its pool
reservation released. Reproducibility identifiers:

- Adapter snapshot SHA-256: `a98c9e2dfc072170b86d8f4b75bf63757855d7bacf78ae8d666ac952701277df`.
- Track NPZ SHA-256: `ac5c730343f0a7051589821e0cdc124571893d4a9cd2f6407722cbd4fe5245ec`.
- Source video SHA-256: `90dd68acdd40f80de7ccb6149da106d1e3a71e51fde358bf8c6433525201a73e`.

Source FPS validation and timestamp export were added after that GPU snapshot and
checked offline. The inference/projection code is unchanged; recomputing the
projection with current code matches the saved V2E pixels and visibility. The
snapshot delta is retained in `gpu-source-delta.json`. No game scene or joint
trajectory was produced by this smoke run.

### SO-101 cube recovery review

The case-specific recovery tool lives in the
[OpenBeam SO-101 workflow](https://github.com/behoyh/openbeam/blob/agent/so101-teleop-completion/doc/nextgen/SO101_VIDEO_RECONSTRUCTION.md).
It verifies the prepared images, queries and raw predictions before applying the
generic support helper to the four cube seeds. Its per-frame colour detector
recovers a unique visible patch in **340/414 frames**, retaining **74 gaps**.
With a 2 px support margin, **379 of 605 predicted-visible cube samples are
rejected**; 226 retain image support. All 149 predicted-visible samples after
frame 300 are rejected. Separate patch observations recover the cube in the tray
from frame 283; they do not become new observations of the four original corners.

Evidence is in sibling `so101-cube-recovery-02/`: raw predictions remain in run 03,
while the recovery directory contains copied filtered tracks, per-frame PNG masks,
patch observations, provenance and `cube-recovery.mp4`. The 1280×576 H.264 comparison
fully decodes to 414 frames at 30 fps (13.8 s); reviewed samples include transfer,
occlusion and the cube resting in the tray. This is a tracking-review video;
metric scene alignment, articulated fitting and physical recreation are pending.
