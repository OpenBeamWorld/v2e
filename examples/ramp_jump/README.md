# Example: ramp jump

A generated mid-size car drives at 55 km/h up a 4 m launch ramp whose slope steepens to 15 deg
at the top, leaves the top edge, flies and lands on level ground, then brakes. The camera stands
on a tripod 22 m to the side. This is the `ramp_jump` family of
[v2e-synth-1](../../benchmarks/v2e-synth-1/) at its default grid point.

What it exercises: driving onto sloped static geometry, a separation event (take-off), an airborne
phase in which only gravity acts, a landing contact, and attitude changes in pitch.

## Files

| File | What it holds |
| --- | --- |
| `case.json` | The case: parameters, camera, the ramp's segments (1 m at 5 deg, 1 m at 10 deg, 2 m at 15 deg), the car, events, phases, scene, templates and targets. |
| `templates.json` | `vehicle.generic` for the car, `static.mesh` for the level. |
| `targets.json` | Every target with its source. |
| `walkthrough.py` | Runs and narrates the pipeline steps. |

The ramp steepens in straight segments, the polygonal form of the curved transition real launch
ramps have, so a car with ordinary ground clearance rolls onto it instead of striking its foot.

## Run it

```sh
python examples/ramp_jump/walkthrough.py --out runs/ramp
```

The steps are those of the [barrier crash example](../barrier_crash/README.md#run-it): generate
(105 frames, 3.5 s), inspect the truth package, reconstruct (baseline, or the truth as an oracle
until the reconstruction step is installed), compare, recreate on the `fake` connector, score.
The events are `ramp_contact`, `takeoff` (the car has touched nothing for 0.1 s), `apex` and
`landing`.

## Targets and where they come from

All three airborne targets are point-mass predictions for a take-off at the design speed v and
the last segment's angle a, from the top edge's height h, with g = 9.81 m/s^2:

| Target | Value | Source |
| --- | --- | --- |
| Speed at take-off | 15.28 m/s, within 0.3 m/s | design value: the driver holds the design speed up the ramp |
| Rise of the centre from take-off to apex | (v sin a)^2 / 2g = 0.80 m, within 15 % | first principles |
| Time from take-off to landing | (v sin a + sqrt((v sin a)^2 + 2 g h)) / g = 0.97 s, within 15 % | first principles |
| Horizontal distance from the edge at landing | v cos a times that time = 14.3 m, within 15 % | first principles |

On the `fake` connector the car reaches the top edge well below the design speed: the test
double's wheels are points on spring-dampers, and they lose speed each time they cross a change
of slope. The take-off target is therefore missed, and the three airborne targets, which assume
the design speed, are missed with it. This is the kind of mismatch the targets exist to expose;
an engine connector is held to the same numbers.

## Variations

The grid covers final angles of 10, 15 and 20 deg at 40 and 55 km/h, each with both cameras:

```sh
python -m openbeam_v2e.bench synth list --family ramp_jump --subset full
python -m openbeam_v2e.bench synth generate --case ramp_jump-a20-v55-handheld_ptz --out runs/ramp-grid
```

## Limits

Suspension travel and tyre deformation on landing are those of the rigid test double. The
recreation uses the case's level geometry for the ramp and the ground.
