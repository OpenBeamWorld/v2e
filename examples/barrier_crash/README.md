# Example: barrier crash

A generated mid-size car drives straight into a fixed concrete block at 40 km/h with its whole
front face (100 % overlap), then the driver brakes. The camera stands on a tripod 16 m to the
side. This is the `barrier_crash` family of [v2e-synth-1](../../benchmarks/v2e-synth-1/) at its
default grid point.

What it exercises: a vehicle that enters the clip already moving (spawned at rest on a run-up and
driven to its frame-0 pose and speed with throttle, brake and steering only), a contact event
between a vehicle and static geometry, and the measures taken after an impact (rebound speed,
displacement, yaw change, time to rest).

## Files

| File | What it holds |
| --- | --- |
| `case.json` | The whole case: parameters, camera preset, level primitives, the vehicle, events, phases and the `v2e.scene/1`, `v2e.templates/1` and targets documents. |
| `templates.json` | Content for each object: the `vehicle.generic` template for the car, `static.mesh` for the level. |
| `targets.json` | Every target with its source (see below). |
| `walkthrough.py` | Runs the steps below and narrates them. |

The clip itself is not stored here: it is generated on your machine in a few seconds, exactly the
same every time.

## Run it

```sh
python examples/barrier_crash/walkthrough.py --out runs/barrier
```

The walkthrough prints seven steps:

1. **The case**: parameters and targets with their sources.
2. **Generate**: the `fake` connector simulates the case and the built-in ray caster renders 90
   frames (3 s at 30 fps). The vehicle spawns at rest about 46 m behind its frame-0 pose; frame 0 is the
   instant it crosses that pose, one second before it reaches the barrier.
3. **Inspect** the truth package `runs/barrier/cases/<case>/world/` (a world package of kind
   `synthetic_truth`): the car's dimensions, the static meshes and the camera.
4. **Reconstruct** from the frames, the outlines in `annotations.json` and the surveyed points,
   with the CPU baseline models. Until the reconstruction step is installed the walkthrough says
   so and uses the truth as an oracle reconstruction.
5. **Compare** the reconstruction with the truth: camera reprojection, focal length, dimensions,
   path, attitude.
6. **Recreate** the scene on the `fake` connector from the reconstruction: measured size,
   frame-0 pose and entry speed; every actor spawns at rest.
7. **Score** the run against the truth: path error, silhouette overlap, event timing and the
   targets.

The same steps with the package commands (paths from step 2):

```sh
C=runs/barrier/cases/barrier_crash-v40-o100-tripod
openbeam-v2e inspect $C/world                                    # the truth package
openbeam-v2e reconstruct $C/frames --config $C/reconstruct.json --out runs/barrier/recon-cli
openbeam-v2e compile $C/scene.json --templates $C/templates.json --out runs/barrier/compiled
openbeam-v2e score runs/barrier/runs/barrier_crash-v40-o100-tripod/baseline \
    --scene $C/scene.json --reconstruction $C/world                # score the recreated run
```

`reconstruct.json` holds the surveyed points, the object list and the name of the annotation
file; `compile` writes the engine-neutral package request and the `fake` connector's
`compiled.json`. The run directory scored in the last line is the one the walkthrough wrote in
step 6 (`truth` instead of `baseline` when it fell back to the oracle).

## Targets and where they come from

| Target | Value | Source |
| --- | --- | --- |
| Speed at first contact | 11.1 m/s (40 km/h), within 0.3 m/s | design value: the driver holds the design speed until contact |
| Time of first contact after frame 0 | 1.000 s, within one frame | design value: frame 0 is placed one second before contact |
| Yaw change after the impact | 0 deg, within 5 deg | first principles: a head-on, full-overlap impact has no preferred turning direction |
| Rebound speed, rest position | reported | measured, no bar in v0 |

Engine runs are never a source of targets. The run's claim class on `fake` is `test_double`: the
connector is a rigid stand-in for tests, so no figure here describes a physics engine.

## Variations

The family's grid covers 30, 40 and 50 km/h at 100 % and 40 % overlap, each with the tripod and a
handheld camera that pans, tilts and zooms from one spot:

```sh
python -m openbeam_v2e.bench synth list --family barrier_crash --subset full
python -m openbeam_v2e.bench synth generate --case barrier_crash-v50-o40-handheld_ptz --out runs/barrier-grid
```

At 40 % overlap the impact is off-centre, so the yaw change is reported instead of held to the
symmetry target.

## Limits

The car is a rigid body here: the `fake` connector has no deformation, so crush is neither
simulated nor scored (deformation is out of scope for v0 everywhere). The recreation uses the
case's level geometry for the barrier and the ground.
