# Example: rollover

A generated mid-size car drives straight at 60 km/h with its left wheels over a short, steep
one-sided ramp (a gentle toe, then a 22 deg face, 0.68 m high, 1 m wide). The left side is thrown
up, the car rolls toward its right and comes to rest. The camera stands on a tripod 18 m to the
side. This is the `rollover` family of [v2e-synth-1](../../benchmarks/v2e-synth-1/) at its default
grid point.

What it exercises: a contact under one side only, roll about the travel direction, attitude far
from upright (scored modulo nothing, since a vehicle has no symmetry), and the rollover measures:
cumulative roll, quarter turns and rest attitude.

## Files

| File | What it holds |
| --- | --- |
| `case.json` | The case: parameters, camera, the trip ramp, the car, events, phases, scene, templates and targets. |
| `templates.json` | `vehicle.generic` for the car, `static.mesh` for the level. |
| `targets.json` | Every target with its source. |
| `walkthrough.py` | Runs and narrates the pipeline steps. |

## Run it

```sh
python examples/rollover/walkthrough.py --out runs/rollover
```

The steps are those of the [barrier crash example](../barrier_crash/README.md#run-it). The clip is
120 frames (4 s); the events are `trip_contact`, `trip_exit` and `rest`. The driver holds speed and
line until the left wheels touch the ramp, then lets go of every control.

## Targets and where they come from

| Target | Value | Source |
| --- | --- | --- |
| Speed when the left wheels reach the ramp | 16.67 m/s, within 0.3 m/s | design value |
| Whether a rollover is allowed at all | 1 (allowed) | first principles: a rigid-body energy bound (below) |
| Cumulative roll, quarter turns, rest attitude | reported | measured, no bar in v0 |

The energy bound treats the car as a rigid box pivoting about its right wheels' contact line. At
the top of the ramp the left wheels are 0.68 m up and still rising at v tan(22 deg); the car can
tip only if its roll energy about the pivot covers the climb of its centre of mass to the point
straight above the pivot. With the design proportions (track 1.6 m, centre of mass assumed
0.55 m up) that needs at least about 4.9 m/s, so every speed of the grid allows a rollover; a run
that rolls where the bound forbids it fails the target. The bound says when rolling is possible,
not that it happens: whether and how far the car rolls is reported.

## Variations

The grid covers 40, 50 and 60 km/h with both cameras:

```sh
python -m openbeam_v2e.bench synth list --family rollover --subset full
python -m openbeam_v2e.bench synth generate --case rollover-v40-handheld_ptz --out runs/rollover-grid
```

## Limits

Roof crush and body deformation are out of scope for v0; the `fake` connector's car is rigid. The
recreation uses the case's level geometry for the trip ramp and the ground.
