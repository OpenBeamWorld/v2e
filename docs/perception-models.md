# Adding a perception model

Reconstruction never calls a specific model. It asks for a capability (calibrate the camera,
track it, segment the objects, estimate depth, fit poses) and the perception registry hands it
whichever model the configuration names. A model from another package plugs in the same way as
the built-in ones, so you can try a new segmenter, tracker or depth network on any clip and on
the benchmark without touching V2E's code.

How the stages fit together, and the configuration that selects them, is described in
[Reconstruction](reconstruction.md); this guide is about writing and plugging in a model.

## The capabilities

Each capability is an abstract class in `openbeam_v2e.perception.interfaces` with one method.
Every result type lives in `openbeam_v2e.perception.types` and carries a provenance record (model,
version, weights and their hash, parameters) and, where the model can give one, an uncertainty.

| Kind | Interface | Method | Returns |
| --- | --- | --- | --- |
| `camera_calibration` | `CameraCalibrator` | `calibrate(inputs)` | the reference frame's camera with sigmas |
| `camera_tracking` | `CameraTracker` | `track(video, reference, frames, ...)` | `CameraTrack`: pan, tilt, roll and focal length per frame, with sigmas |
| `detection` | `Detector` | `detect(image, classes)` | `Detection` list (seeds prompts when there are no annotations) |
| `segmentation` | `Segmenter` | `segment(video, prompts, frames, cameras)` | `Segmentation`: masks per object per frame |
| `point_tracking` | `PointTracker` | `track(video, queries, frames)` | `PointTracks` with visibility and, where the model gives one, a per-point confidence |
| `depth` | `DepthEstimator` | `estimate(image, camera)` | `DepthMap`, metric or relative |
| `pose` | `PoseEstimator` | `fit(masks, cameras, spec, frames, context)` | `PoseTrack`: rigid poses with covariance and measured dimensions |

`video` is any frame source: `len(video)`, `video.fps`, `video.size` (width, height) and
`video.read(i)` returning an RGB `uint8` array. `prompts` are the assisted-annotation prompts
(`ObjectPrompt`: a box, polygon, mask or clicks for one object in one frame).

## What ships

The base install has CPU baselines that need only NumPy and SciPy: `baseline.points`,
`baseline.vanishing` and `baseline.auto` (calibration), `baseline.rotation_zoom` and
`baseline.static` (camera tracking), `baseline.background`, `baseline.prompts` and
`baseline.files` (segmentation), `baseline.lk` (point tracking), `baseline.ground_plane` (depth)
and `baseline.silhouette` (pose). Adapters for heavier models install as extras: `sam2`
(segmentation, `pip install "openbeam-v2e[sam2]"`), `hf.depth` and `hf.detection`
(`[perception]`) and `cotracker` and `tapnext` (point tracking). List what your installation has
with:

```python
from openbeam_v2e.perception import registry
for model in registry.describe():
    print(model["kind"], model["name"], model["available"], model["status"])
```

### Choosing a point tracker

- `baseline.lk` is the CPU default: pyramidal Lucas-Kanade in NumPy and SciPy, tracking forward
  from each query frame.
- `tapnext` runs TAPNext++ (or the earlier BootsTAPNext) from Google DeepMind's
  [tapnet](https://github.com/google-deepmind/tapnet) repository. It is the first learned point
  tracker here whose code and weights are both released under Apache-2.0. tapnet is not on PyPI:
  install `[perception]` plus `"tapnet[torch] @ git+https://github.com/google-deepmind/tapnet.git"`
  and `einops` (tapnet's own install also pulls in JAX and its other dependencies). The adapter
  never downloads weights. Download one of the checkpoints it supports from the tapnet README,
  `tapnextpp_ckpt.pt` (256 x 256), `tapnextpp_512.ckpt` (pass `resolution: 512`) or
  `bootstapnext_ckpt.npz` (256 x 256), and pass its path as `checkpoint`. The model runs online,
  one frame at a time; queries are tracked in groups of at most `max_queries` (default 256), each
  group starting at its own earliest query frame, so memory stays bounded on long clips.
  Like `baseline.lk`, it tracks forward through the frames in the order given: a point has no
  position before its query frame. It fills `PointTracks.confidence` from tapnet's own certainty
  estimate. The adapter is tested with stand-ins for PyTorch and tapnet, not with the real
  weights.
- `cotracker` runs CoTracker3 (offline, forwards and backwards), installed from its repository.
  Most of CoTracker's code, and the CoTracker3 weights, are released under CC BY-NC 4.0:
  non-commercial use only.
- Mask trackers such as SAM 2 (and the older Track-Anything) follow object masks, not points.
  They belong to the segmentation stage (`sam2`), not to `point_tracking`.

## Writing a model

The [custom perception model example](../examples/custom_perception_model/) is a complete
segmenter in about 90 lines. The parts every model has:

```python
from openbeam_v2e.perception.interfaces import Segmenter
from openbeam_v2e.perception.types import Segmentation

class ColourKeySegmenter(Segmenter):          # 1. subclass the interface of its kind
    name = "example.colour_key"               # 2. a name unique within the kind
    summary = "Keeps pixels of each object's annotated colour."
    extras = ()                               #    pip extras it needs, e.g. ("sam2",)
    requires = ()                             #    importable modules it needs, e.g. ("torch",)

    def __init__(self, tolerance=0.06, **params):
        super().__init__(tolerance=tolerance, **params)   # parameters land in the provenance

    def segment(self, video, prompts, frames, cameras=None):   # 3. the interface's method
        width, height = video.size
        seg = Segmentation(width, height, provenance=[self.provenance()])  # 4. say who made it
        ...                                    # seg.track(object_id).set(frame, mask, confidence)
        return seg
```

Rules that keep the base install light and results reproducible:

- **Import heavy libraries inside the method**, never at module level. Listing models must not
  import PyTorch. Declare the modules in `requires` and the extra in `extras`: `check()` then
  reports a missing library with the install command instead of failing at import.
- **Record what produced the result.** Pass `weights` and `weights_sha256` to
  `self.provenance(...)` when the model loads weights, and library versions in `libraries`.
  The reconstruction stores every record, so a world package says which models made it.
- **Report uncertainty where you can**: per-frame sigmas in a `CameraTrack`, mask confidence in a
  `Segmentation`, covariance in a `PoseTrack`. Downstream steps turn them into z-scores.
- **Download nothing unasked.** Weights are fetched only when the user runs the model.
- **Keep clip-specific numbers out of the code.** Thresholds and settings are parameters, set per
  clip in the configuration.

## Registering it

In a notebook or a test, register the class at run time:

```python
from openbeam_v2e.perception import registry
registry.register(ColourKeySegmenter)
```

In a package, declare an entry point; the registry finds it once the package is installed:

```toml
[project.entry-points."openbeam_v2e.perception"]
colour_key = "my_package.colour_key:ColourKeySegmenter"
```

A plugin that fails to load is logged and skipped; it never breaks the registry.

## Choosing it

Per stage, in three equivalent ways:

```sh
openbeam-v2e reconstruct clip/frames --config clip/reconstruct.json --model masks=example.colour_key --out clip/world
```

```json
{"schema": "v2e.reconstruct/1", "segmentation": {"model": "example.colour_key", "params": {"tolerance": 0.04}}}
```

```python
from openbeam_v2e.reconstruct.pipeline import reconstruct
reconstruct("clip/frames", "clip/world", config="clip/reconstruct.json", models={"masks": "example.colour_key"})
```

Stage names: `calibration`, `camera` (tracking), `masks` (segmentation), `depth`, `pose`,
`points` (point tracking) and `detection`.

## Measuring it

The synthetic benchmark runs any model on the same cases with exact truth, next to the baseline:

```sh
python -m openbeam_v2e.bench synth run --out runs/baseline
python -m openbeam_v2e.bench synth run --out runs/colour-key --model masks=example.colour_key
```

Each report's `recon` column names the models used (`baseline+masks=example.colour_key`). The
reconstruction columns (camera reprojection, dimensions, path, attitude, events) show what the
model changed; the run columns show whether the recreated scene played out better. The figures
are synthetic self-consistency: for accuracy on real footage, use clips with independent
measurements (planned for a later release).
