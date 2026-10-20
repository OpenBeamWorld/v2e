# Example: drop and bounce

A 0.5 m box, a 0.3 m ball and a 0.5 m cylinder lying on its side are let go at rest 2 m above a
concrete slab and fall, bounce and settle. The camera stands on a tripod 9 m in front of the slab.
This is the `drop_and_bounce` family of [v2e-synth-1](../../benchmarks/v2e-synth-1/) at its
default grid point; the family also has a variant where a box is let go on a 20 or 30 deg slope.

What it exercises: several passive objects at once, objects that start at rest in the air (no
velocity is ever given to anything), repeated contacts, and attitude scored modulo each shape's
symmetry: a ball's attitude is ignored, a cylinder's spin about its axis is ignored, a cube is
compared up to the rotations that map it onto itself.

Passive objects are **scored, not fitted** in v0: the recreation uses each prop's measured size
and a mass and bounce class from its template; nothing is tuned to make the bounces match.

## Files

| File | What it holds |
| --- | --- |
| `case.json` | The case: parameters, camera, slab and props, events, phases, scene, templates and targets. |
| `templates.json` | `prop.box`, `prop.sphere` and `prop.cylinder` with the masses used (20, 2 and 8 kg, marked `assumed`). |
| `targets.json` | Every target with its source. |
| `walkthrough.py` | Runs and narrates the pipeline steps. |

## Run it

```sh
python examples/drop_and_bounce/walkthrough.py --out runs/drop
```

The steps are those of the [barrier crash example](../barrier_crash/README.md#run-it). The clip is
105 frames (3.5 s); frame 0 is the instant the props are let go (their spawn), so no alignment run-up is
needed. Events are named per object: `first_contact.box`, `rest.box`, and so on.

## Targets and where they come from

| Target | Value | Source |
| --- | --- | --- |
| First contact of each prop after frame 0 | sqrt(2 h / g) = 0.639 s for h = 2 m, within one frame | first principles: free fall from rest |
| Rebound ratio, rest position | reported | measured, no bar in v0 |

## Variations

```sh
python -m openbeam_v2e.bench synth list --family drop_and_bounce --subset full
python -m openbeam_v2e.bench synth generate --case drop_and_bounce-drop-h4-tripod --out runs/drop-grid
python -m openbeam_v2e.bench synth generate --case drop_and_bounce-incline-a30-tripod --out runs/drop-grid
```

Drop heights of 1, 2 and 4 m, and the box on a 20 or 30 deg slope, each with both cameras.

## Limits

On an engine connector that offers stepped physics only with a player vehicle in the scene, a
props-only scene needs an off-camera anchor vehicle (the harness asks you to opt in to it). The
`fake` connector needs none.
