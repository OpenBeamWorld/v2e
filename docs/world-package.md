# World packages

Every stage of V2E that describes "what happened" uses one format: the reconstruction of a
clip, the export of an engine run and synthetic ground truth are all world packages, so any
two of them can be scored against each other. In Python they are `openbeam_v2e.world.World`
objects; on disk they are directories.

## Layout

A V2E world package is a world package in the 4DCodeBench layout plus a `v2e/` directory:

```text
world/
  camera.json              intrinsics K (3x3) and extrinsic E (4x4, camera-to-world) of frame 0
  meshes/<id>.npz          vertices_TTTT (V, 3) float32 and faces_TTTT (T, 3) int32 per frame
  dynamics/<object>.npz    pos (F, V, 3) float32 hull vertices per frame, ids, faces
  render.mp4               optional
  v2e/world.json           v2e.world/1: fps, frame_count, image_size, kind, model provenance
  v2e/camera_track.json    v2e.camera_track/1: one position, per-frame intrinsics, rotation, k1, k2
  v2e/objects.json         v2e.objects/1: class, symmetry, dimensions with sigma, visibility
  v2e/tracks/<object>.npz  position (F, 3), quat_xyzw (F, 4), visible, per-frame sigmas, hull
  v2e/uncertainty.json     v2e.uncertainty/1: camera sigma, events, fitted / held-out residuals
  v2e/outlines.json        v2e.outlines/1 (optional): observed pixel outlines per object and frame
```

Conventions: metres, Z up, right-handed; quaternions `[x, y, z, w]`; object body frames are
x forward, y left, z up; camera axes are x right, y down, z forward; pixel `(col, row)` has its
centre at `(col + 0.5, row + 0.5)`; lens distortion is radial,
`x_d = x (1 + k1 r^2 + k2 r^4)`. Frames run `0 .. frame_count - 1` and time is `frame / fps`.
Unknown poses are NaN rows in the track archive; the base layout shows each object over one
contiguous interval with interior gaps interpolated, as that layout requires.

The `fourdcb` module reads, writes and checks the base layout. It is an independent
implementation written from the published description of the layout and is not affiliated
with its authors.

## Writing a world from your own model

Any reconstruction model can produce a world package; nothing in the format depends on the
models V2E ships. Build the objects and write them:

```python
import numpy as np
from openbeam_v2e.schema.common import Measured
from openbeam_v2e.world import Camera, CameraTrack, ObjectTrack, StaticMesh, World, write_world
from openbeam_v2e.world.geometry import box_mesh

F, fps = 90, 30.0
cam = Camera.from_fov(1920, 1080, 60.0, centre=(0, -25, 4), target=(0, 0, 1))
ground = StaticMesh(1, [[-50, -20, 0], [50, -20, 0], [50, 20, 0], [-50, 20, 0]], [[0, 1, 2], [0, 2, 3]],
                    "ground", "drivable")
hull_v, hull_f = box_mesh(4.5, 1.8, 1.5, centre=(0, 0, 0.75))     # body frame
position = np.zeros((F, 3)); position[:, 0] = np.linspace(-10, 10, F)
quat = np.tile([0, 0, 0, 1.0], (F, 1))
car = ObjectTrack("car", "vehicle.car", hull_v, hull_f, position, quat,
                  sigma_position_m=np.full(F, 0.1),
                  dims_m={"length": Measured(4.5, 0.05), "width": Measured(1.8, 0.05)})
world = World(fps, F, CameraTrack.static(cam, F), [ground], [car])
write_world(world, "world/")
```

For a camera that pans, tilts or zooms, build one `Camera` per frame (same centre) and use
`CameraTrack.from_cameras`. Record the models you used in `World.models` (id, version,
weights hash, parameters) so the package states its provenance.

## Reading packages made by other tools

`read_world(path)` reads V2E packages exactly. A plain base-layout package without `v2e/`
can be imported with `read_world(path, fps=...)`: static meshes become statics, `camera.json`
becomes a static camera, and every dynamics archive with surface faces becomes a rigid object
whose pose per frame comes from a rigid fit of its vertices (the import records the worst fit
residual, which shows when an object was not rigid). Point-only and volumetric archives are
skipped and listed in `World.notes`.

## Runs as world packages

Connectors write runs as `v2e-runlog/1` JSON Lines (simulated time, actor poses in the Z-up
level frame). When a connector cannot observe full attitude, it sets
`attitude_observed: false` in the header and omits `q` in its rows. Those logs can be read
and audited, but cannot be exported as full-pose worlds without measured orientations.
The default header still requires a unit quaternion in every actor state.
`openbeam_v2e.world.export.runlog_to_world` resamples a full-pose run onto the reference
timeline (`t0 + frame / fps`, where `t0` is the simulated time of video frame 0), keeps the
reference camera and statics, and returns a `run_export` world ready to score;
`export_run(rows, reference_dir, out_dir, t0_s)` does the same from files.
