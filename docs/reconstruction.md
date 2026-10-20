# Reconstruction

Reconstruction turns a video of a physical event into a world package: the camera for every
frame, the ground, and each moving object with its class, measured dimensions, a rigid pose
track and events. Every quantity carries an uncertainty, and every result records which
perception model produced it. This guide covers running it, its inputs and outputs, what the
CPU baseline does at each stage, how uncertainty is computed, and the limits of v0. To plug in a
different model for a stage, see [Adding a perception model](perception-models.md).

## Running it

```sh
openbeam-v2e reconstruct clip.mp4 --config reconstruct.json --annotations annotations.json --out world/
```

```python
from openbeam_v2e.reconstruct.pipeline import reconstruct
summary = reconstruct("clip.mp4", "world/", config="reconstruct.json", annotations="annotations.json",
                      models={"masks": "sam2"})   # optional: a model per stage
```

AI agents call the same step through the `reconstruct` tool of the CLI and the MCP server
([AI agents](agents.md)). The return value is a JSON summary: the camera calibration and its
reprojection error, each object's dimensions with sigma and its fitted and held-out silhouette
overlap, the events, the models used and any warnings.

The video can be a file (decoded by an external `ffmpeg`, which V2E does not bundle), a
directory of numbered frames (PNG is read without extra packages; JPEG and other formats need
Pillow) or a `.npy` array of frames.

## Inputs

### Configuration (`v2e.reconstruct/1`)

Everything specific to one clip lives in its configuration, never in code:

```json
{"schema": "v2e.reconstruct/1",
 "video": {"path": "clip.mp4", "fps": 30, "frames": [0, 240], "stride": 1},
 "ground_z": 0.0,
 "camera": {
   "calibration": {"frame": 0, "sigma_px": 1.0,
                   "points": [{"world": [0.0, 0.0, 0.0], "image": [612.0, 388.0]},
                              {"world": [4.0, 0.0, 0.0], "image": [800.0, 390.0]}]},
   "tracking": "baseline.rotation_zoom"},
 "segmentation": {"model": "baseline.background", "params": {"neighbours": 16}},
 "depth": null,
 "detection": null,
 "pose": "baseline.silhouette",
 "objects": [{"id": "car_1", "class": "vehicle.car", "dims_prior": {"length": [4.6, 0.4]}}],
 "annotations": "annotations.json",
 "holdout": {"every": 5, "offset": 2},
 "events": {"rest_speed_mps": 0.2},
 "cache": ".cache/v2e"}
```

| Key | Meaning |
| --- | --- |
| `video` | Source, frame rate (overrides the file's), frame range `[first, end)` and stride. |
| `ground_z` | Height of the ground plane in the world frame. |
| `camera.calibration` | Evidence for the reference frame's camera (see below), or a complete `camera` (intrinsics and pose) with optional `sigmas` when it is already known. |
| `camera.tracking` | `baseline.rotation_zoom` for a camera that pans, tilts or zooms; `baseline.static` for a camera that does not move; `baseline.point_tracks` to solve the same camera model from any point tracker's tracks (`{"model": "baseline.point_tracks", "params": {"point_tracker": "cotracker"}}`, or the stage override `points=cotracker`). |
| `segmentation`, `depth`, `detection`, `pose` | The model per stage, as a name or `{"model": ..., "params": {...}}`. `depth` and `detection` are optional. |
| `objects` | The objects to fit: id, class (`vehicle.car`, `vehicle.truck`, `prop.box`, `prop.cylinder`, `prop.sphere`, ...), and optionally `template`, `template_params`, `motion` (`ground` or `free`), `dims_prior` (`{name: [value, sigma]}`) and `fit_dims`. Without this list the annotated objects are fitted, and without annotations the segmenter's own objects. |
| `annotations` | A `v2e.annotations/1` file or document (below). |
| `holdout` | Observed frames left out of the pose fit so that its residuals can be checked honestly: every n-th frame, or an explicit `frames` list. `null` holds nothing out. |
| `events` | Thresholds of the event detectors. |
| `physics.ballistic_segments` | Try ballistic refinement only on sustained-clearance segments of free objects (default on). No missing-event whole-clip fallback. |
| `physics.ballistic_max_chi2` | Maximum weighted residual per degree of freedom for a ballistic proposal (default 4; null disables this consistency screen, not clearance/ground checks). This is an uncalibrated heuristic for correlated poses. |
| `cache` | Directory of the stage cache; omit it to recompute everything. |
| `output.masks` | Also write the masks as PNG files. |

Calibration evidence, from most to least reliable:

- **Points** (`baseline.points`): pixels of points whose world coordinates you know, for
  example surveyed marks, the corners of a parking bay, a checkerboard laid on the ground. Four
  points on the ground or six in general position; more is better. `free_distortion: true` also
  fits radial lens distortion (needs points spread to the image edges).
- **Line segments** (`baseline.vanishing`): segments `[x1, y1, x2, y2]` along two perpendicular
  ground directions (`lines_x`, `lines_y`: lane markings, kerbs, tiles) plus one scale fact:
  `camera_height_m` or `known_distance` (`{"a": [u, v], "b": [u, v], "m": 10.0}`, two ground pixels
  a known distance apart). Points, if also given, refine the result.
- **Nothing but a scale fact** (`baseline.auto`): vanishing points found in the image's straight
  edges. This is the weakest start; use it to get going and add points when you can. Without
  any scale fact V2E assumes a camera height of 1.6 m with a sigma of 0.5 m and says so in the
  warnings: everything metric then scales with that assumption.

### Annotations (`v2e.annotations/1`)

Reconstruction in v0 is assisted: tell it which objects to follow with a prompt on at least one
frame per object. A prompt is a box, an outline (polygon), a mask (an image file) or clicks:

```json
{"schema": "v2e.annotations/1",
 "objects": {
   "car_1": {"class": "vehicle.car",
             "prompts": [
               {"frame": 0,  "box": [412, 300, 655, 410]},
               {"frame": 30, "polygon": [[500, 300], [700, 300], [700, 400], [500, 400]]},
               {"frame": 45, "mask": "masks/car_1_000045.png"},
               {"frame": 60, "points": [[610, 350]], "labels": [1]}]}}}
```

A `v2e.outlines/1` document is accepted too. With a `detection` model configured and no
annotations, detections on the calibration frame become box prompts (for example
`{"model": "hf.detection", "classes": ["car"]}`).

## Outputs

The output directory is a world package: the 4DCodeBench layout (`camera.json`, `meshes/`,
`dynamics/`) written by `openbeam_v2e.world`, plus the V2E sidecars in `v2e/` (see
[World packages](world-package.md)):

| File | Content |
| --- | --- |
| `v2e/world.json` | Frame rate, frame count, image size, the provenance of every model used. |
| `v2e/camera_track.json` | Per frame: intrinsics, camera-to-world rotation (camera axes: x right, y down, z forward) and distortion; one camera position. |
| `v2e/objects.json`, `v2e/tracks/<id>.npz` | Each object's class, symmetry class, dimensions with sigma, visibility, and its pose track with per-frame position and attitude sigmas. |
| `v2e/uncertainty.json` | Camera sigma per frame (pixels), events as frame intervals, fitted and held-out residuals. |
| `v2e/outlines.json` | The observed outline of each object per frame (from the masks), for silhouette scoring. |
| `v2e/reconstruct_report.json` | The full detail: calibration with its covariance, the camera track with sigmas, each pose track with its 6x6 covariance per frame, velocities, per-frame fit diagnostics, the configuration, timings and the provenance records. |

Conventions: the world frame is right-handed, Z up, in metres. An object's body frame is +x
forward, +y left, +z up. In the world package its origin is the hull's bounding-box centre (the
reference point connectors report); the fits and `v2e/reconstruct_report.json` use the centre of
the hull's bottom face instead (a car resting on the ground has that origin on the ground). Pixel `(col, row)` covers `[col, col + 1)`, so pixel
centres sit at `+0.5`. Frames are renumbered from 0 for the first reconstructed frame; the
report lists the source frame of each.

## What the CPU baseline does

Each stage below is the default model. All of them run on a laptop CPU with NumPy and SciPy.

**Calibration** starts from a direct linear transform (points in general position) or a ground
homography with the focal length solved from the rotation constraints (points on the ground),
tries every start that applies, and refines the best by bounded robust least squares over pan,
tilt, roll, position and focal length (and distortion when asked). Automatic calibration fits
straight segments to edges, finds vanishing points by RANSAC and picks the pair of ground
directions that is best supported and agrees with other pairs on the focal length.

**Camera tracking** (`baseline.rotation_zoom`) uses the fact that views from one position differ
only by a rotation and a zoom, whatever the depth of the scene. Corners in a keyframe are tracked
with pyramidal Lucas-Kanade, the frame's pan, tilt, roll and focal length are solved from the
viewing rays of the keyframe with a robust loss, and the corners are then matched again directly
against the keyframe, which removes the drift of frame-to-frame chaining. A new keyframe starts
when too few corners survive or the zoom has changed by more than 12 %. Moving objects are
excluded where prompts mark them, and outvoted by the robust loss elsewhere.

**Segmentation** (`baseline.background`) builds a background for every frame from up to 16
other frames spread over the clip, each warped into the frame's view (exact for a camera at one
position). Pixels that differ from the per-pixel median are moving; shadows (darker, same
colour) are dropped; regions are assigned to objects by following each object's box from its
prompts, forwards from the first annotation and backwards from the last. The reverse pass fills
missing observations without replacing existing forward observations; each annotation resets
motion prediction, and annotated pixels take precedence over propagated masks. A prompt outside
the requested frame sampling grid can seed a box prediction, but is not emitted as an observation.
Missing foreground stays unobserved, even between annotations. A second pass leaves each object's
observed pixels and interpolated prompt boxes out of the background samples. This can retain an
object that stops, even when it occupies the same pixels for most of the clip. The interpolated
boxes only exclude samples: they do not become observed masks, and are not extrapolated beyond
the first or last annotation. Set `prompt_exclusion: false` in the segmentation model's `params`
to disable this extra exclusion. Pixels with no clean background sample fall back to the original
samples; a car that never uncovers its background still needs a prompt-driven segmenter such as
SAM 2. Morphological cleanup preserves silhouettes clipped by the image edge.

**Pose** (`baseline.silhouette`) fits a template hull to every mask: a box, a generic vehicle
(lower body plus a cabin), a cylinder or a sphere, sized by the object's dimensions. The fit
minimises two residuals, the mask's signed distance along the hull's projected outline and the
outline's signed distance at the mask boundary, by bounded robust least squares. Projected
polygons are clipped to the image rectangle before comparison; an object extending outside the
view is not penalised for its unseen outline. Ground objects have three degrees of freedom
(x, y, heading), free objects six. Headings start from the direction of travel (both ways are
tried, including reversing) or a search. Ground fits start at the first untruncated fitting mask
and revisit earlier masks backwards, with signed-time position seeds from fitted neighbours.
A weak adjacent-attitude prior stabilises ambiguous cropped outlines without assuming forward
motion. `clipped_heading_sigma_deg` in the pose model's `params` defaults to 15 degrees, combines
with the neighbour's yaw variance, and can be `null` to disable the prior. Visible shape evidence
can override it. The report's `ground_initialization` diagnostic records the seed, cropped fitting
frames and prior; if no complete mask exists it explicitly reports that fallback. This is a prior,
not new observed attitude evidence. Held-out masks are never used to select the seed or fit it.
Free objects search a coarse set of
attitudes on their first frame. Dimensions are then refitted jointly over a spread of frames
with the poses held, alternating twice. A constant-acceleration Rauch-Tung-Striebel smoother per
coordinate, weighted by each frame's covariance and driven by a jerk prior for the object class,
gives the final track, velocities and covariances for every frame, including frames without a
mask.

**Physics and events**: sustained estimated hull clearance above the height/position-sigma
thresholds selects candidate segments of free objects for an optional known-gravity fit. Missing
take-off/touchdown events never cause a whole-clip fallback. Every candidate needs at least three
fitted, observed poses; held-out and interpolated poses are prediction times, not measurements.
The weighted residual must pass `physics.ballistic_max_chi2`, and the proposed hull must not
penetrate the ground plane. Rejected candidates leave positions, velocities and residuals unchanged.
All candidate intervals are considered, including separate flights at clip boundaries, without
inventing boundary events. Accepted refinements refresh final silhouette IoUs (held-out masks only
score predictions); pre-refinement IoUs and candidate decisions remain in diagnostics.

Clearance alone does not identify gravity-only motion: an object can be held above the plane.
The consistency ratio is not a calibrated p-value because smoothed poses are correlated. This
simple model acts on the tracked body origin, not a measured centre of mass; rotation and other
forces can invalidate it. Input pose covariance is retained, not replaced by a falsely precise
conditional ballistic covariance. None of these screens classifies ground vehicles as airborne.
Events are frame intervals: `rest` (speed below a threshold long enough),
`take_off` and `touchdown` (the hull's lowest point leaves or returns to the ground) and
`impact` (an acceleration peak).

**Depth** is optional. `baseline.ground_plane` gives the metric depth of the ground; a model
with relative output (for example `hf.depth`) is aligned to metres on the ground plane, and its
depth inside an object's mask starts the pose fit of free objects.

## Uncertainty

- Every fit reports `s^2 (J^T J)^-1` from its Jacobian, with the residual scale measured robustly
  and inflated for the correlation between neighbouring outline samples.
- The calibration's own covariance moves every frame's camera together, so it is propagated into
  each object's track as a systematic term: the poses of representative frames are refitted
  with the camera perturbed along each principal axis of the calibration covariance. The same
  error scales the scene, so it widens the dimension sigmas too.
- The smoother turns per-frame covariances into covariances of the final track; frames without
  a mask get the model's (larger) uncertainty.
- Held-out frames are never used in the fit. Their silhouette overlap is reported apart from the
  fitted frames' (`residuals.held_out` in `v2e/uncertainty.json`).

What the sigmas do not include: a template that does not match the object's real shape (the
fitted and held-out overlaps show how well it matches), errors of the segmentation that are
consistent across frames (for example a shadow that is always attached), and moving cameras,
which are out of scope. A local fit covariance is not a calibrated description of multiple
plausible headings from a cropped or mismatched silhouette. Whole-track heading ambiguity and
long extrapolations outside visibility remain accuracy limitations.

## Choosing and installing models

Every stage takes any registered model. The optional adapters install as extras and import their
libraries only when they run:

| Model | Stage | Install |
| --- | --- | --- |
| `sam2` | segmentation | `pip install "openbeam-v2e[sam2]"` |
| `hf.depth` | depth | `pip install "openbeam-v2e[perception]"` |
| `hf.detection` | detection | `pip install "openbeam-v2e[perception]"` |
| `cotracker` | point tracking | `[perception]` plus CoTracker from its repository |
| `tapnext` | point tracking | `[perception]` plus tapnet from its repository: `pip install "tapnet[torch] @ git+https://github.com/google-deepmind/tapnet.git" einops` |

Model weights are downloaded from their hosts only when you run a model that needs them; check
each model's licence on its model card before you use its output. `tapnext` never downloads
weights: pass the local path of a checkpoint linked from the tapnet README as `checkpoint`
(with `resolution: 512` for the 512 x 512 TAPNext++ checkpoint). `registry.describe()` lists
what your installation can run.

## The stage cache

With `cache` set, camera tracking and segmentation results are stored under a key that hashes
the video, the model, its parameters, the code of the model and the upstream results. A second
run with the same inputs reads them back bit for bit; changing any of them recomputes.

## Synthetic clips

`openbeam_v2e.reconstruct.synthetic` renders clips with exact truth from first principles (a
tiled ground, posts as landmarks, a car on a curve, a box thrown through the air, a camera at
one position that pans, tilts and zooms, optional lens distortion and noise). The tests use
them to check each stage against stated tolerances; the benchmark builds on the same renderer.
These are self-consistency checks on synthetic footage, not accuracy claims for real video.

## Limits of v0

- One camera at a fixed position. Pan, tilt, zoom and lens distortion are tracked; a camera
  that moves through the scene (dashcam, drone, walking handheld) is not.
- Rigid objects. Deformation, crush and detachment are not reconstructed.
- Assisted: each object needs a prompt on at least one frame (or a detection model).
- Metric scale comes from the calibration evidence. For free objects, size and distance trade
  off along the viewing ray, so their dimension prior matters unless the object is also seen
  against something of known size.
- Background subtraction needs objects to move out of their own footprint during the clip and
  can follow shadows that are not darker versions of the ground colour.
- Automatic calibration needs straight, converging ground lines in view and is the least
  reliable start.
