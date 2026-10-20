# v2e-synth-1

A synthetic benchmark for the whole V2E pipeline: four families of physical events, each a
parameter grid, generated with exact truth (camera, every object's pose in every frame, events)
and run through reconstruction, recreation on a connector and scoring.

> **Label: synthetic self-consistency.** The truth is a simulated run, so every figure says how
> well the pipeline recovers what a simulator produced. None is a claim about accuracy on real
> footage; those come only from real clips with independent measurements (planned for a later release).

## Families

| Family | Event | Grid | Cases (x 2 cameras) |
| --- | --- | --- | --- |
| `barrier_crash` | A generated car drives into a fixed concrete block. | 30, 40, 50 km/h; 100 % and 40 % overlap | 6 |
| `ramp_jump` | A generated car leaves a launch ramp and lands on level ground. | final slope 10, 15, 20 deg; 40 and 55 km/h | 6 |
| `rollover` | A short, steep ramp under the left wheels tips a generated car over. | 40, 50, 60 km/h | 3 |
| `drop_and_bounce` | A box, a ball and a cylinder fall onto a slab, or a box is let go on a slope (scored, not fitted). | drops of 1, 2, 4 m; slopes of 20 and 30 deg | 5 |

Each grid point is rendered with two cameras that stay in one place, the v0 scope:

- `tripod`: fixed orientation and focal length, no lens distortion;
- `handheld_ptz`: pans and tilts after the subject with a lag, zooms in by a quarter over the
  clip, has radial distortion and a small deterministic wobble.

That makes 40 cases (`--subset full`); the default subset is one case per family with the
tripod camera, small enough for a laptop CPU and for CI. Every number in the case definitions is
a round design value chosen from first principles (typical car proportions, everyday speeds,
standard gravity); none is measured from any video. The definitions are code, in
`openbeam_v2e.bench.families`; `python -m openbeam_v2e.bench synth list --json` prints them.

Content is generic: a generated mid-size car (4.75 x 1.80 x 1.45 m, 1500 kg), generated props and
simple level geometry (ground, a concrete block, ramps that steepen in straight segments, a slab,
a slope) plus four survey posts per scene whose corners serve as surveyed points.

## How a case is generated

1. The case is compiled into content for the connector; every actor spawns **at rest**. A car
   that enters the clip moving starts on a run-up behind its frame-0 pose and the benchmark
   driver brings it to the design speed with throttle, brake and steering only.
2. The harness negotiates a plan and runs it (stepped physics), writing the plan, the
   replayable call ledger and the rows.
3. Video frame 0 is the instant the car crosses its frame-0 pose (for props: the instant they are
   let go). Events (first contact, take-off, landing, rest, ...) are detected from the poses and
   the geometry.
4. The clip is rendered with the in-package ray caster through the case's camera, with a mask
   per object.

In the first release the connector is `fake`, a deterministic rigid-body test double, so a case
regenerates bit for bit on any machine (`provenance.json` records the hashes of the frames and the
truth). [REGENERATE_WITH_OPENBEAM.md](REGENERATE_WITH_OPENBEAM.md) describes the planned
regeneration from OpenBeam renders through the documented MCP tools.

## What a generated case holds

| File | Contents |
| --- | --- |
| `frames/` | the clip, PNG frames |
| `masks/<object>/` | the visible mask of each object per frame (for mask-based tasks) |
| `annotations.json` | `v2e.annotations/1`: the outline of each moving object in three frames, the assisted-annotation input |
| `reconstruct.json` | `v2e.reconstruct/1`: surveyed points with their pixel positions in frame 0 (with 0.5 px click noise), object ids and classes, baseline models |
| `world/` | the truth as a world package of kind `synthetic_truth`: camera track, static meshes, object tracks, events |
| `scene.json` | `v2e.scene/1` over the truth package: actors, events, phases, targets with sources |
| `templates.json`, `targets.json` | content for each object; every target with its source |
| `programs.json` | explicit controller programs (`bench.cruise` for cars, an empty map for passive scenes) |
| `truth.json` | cameras, poses, landmarks, the frame-0 alignment, events and truth metrics |
| `generation/` | the run that produced the truth: plan, ledger, rows, `run.json` |
| `provenance.json` | generator version, connector, render settings, hashes, label and licence |

## Tasks and the report

`python -m openbeam_v2e.bench synth run --out DIR` (or `openbeam-v2e bench synth --out DIR`)
runs each selected case with two reconstruction sources:

- **`baseline`**: the reconstruction step with the CPU baseline models, from the frames, the
  outlines and the surveyed points. `--model STAGE=MODEL` replaces the model of one stage.
- **`truth`**: the truth package used as the reconstruction (an oracle), isolating what
  recreation and scoring lose with perfect perception.

and writes `report.md`, `report.csv` and `report.json`:

| Column | Meaning |
| --- | --- |
| camera RMS, focal err | reprojection RMS of the known world points between truth and reconstructed cameras over all frames; relative focal-length error |
| dims err | largest dimension error of each subject, averaged (vehicles: length, width, height; props: hull extents) |
| recon traj RMS, recon attitude | RMS distance between bounding-box centres; mean attitude error modulo the object's symmetry |
| recon events | mean frame error of the events the reconstruction implies |
| claim | the recreated run's claim class (`test_double` on `fake`) |
| run traj RMS, run events, IoU | the recreated run against the truth: centre path, event timing, silhouette overlap through the truth camera |
| targets met | the case's sourced targets met by the recreated run |

A step that is not installed or fails leaves its cells empty and names the step; it never turns
into a number. `report.json` keeps per-object detail, the recreated parameters and, when the
scorer is installed, its objective.

`status: ok` means the steps completed, **not** that the accuracy or physical targets passed.
Read the error and target columns. The benchmark's recreation path builds fake primitives
directly; the contract suite separately checks all four scene families through the general
scene compiler, package dialect, run, export, score and audit tools.

To use a generated case with the normal tools, run these commands from its directory:

```sh
openbeam-v2e compile scene.json --templates templates.json --out compiled
openbeam-v2e run --compiled compiled --programs programs.json --out run --duration 15
openbeam-v2e export run --reference world --out exported
openbeam-v2e score run --scene scene.json --reconstruction world
```

The run duration includes the run-up: choose at least the compiler's `t0_s` plus the clip
duration, rather than assuming video frame 0 is spawn. These commands use the oracle world;
they check integration, not perception accuracy.

Targets are design values and first-principles predictions (free-fall contact times, projectile
apex and range, a rigid-body rollover energy bound, symmetry of a head-on impact). The `fake`
connector's car slows down where the ramp's slope changes, reaches the top edge slower than the
design speed, and so falls short of every ramp-jump target built on that speed. Exposing such gaps
is what the targets are for.

## v0 limits

- The recreation takes the static geometry from the case (as if the level were surveyed); static
  structure reconstruction from annotated outlines is not scored yet.
- Only the `fake` connector is wired for the offline benchmark. Engine-rendered truth, the
  repeatability task on OpenBeam (nine stepped runs per case, robust intervals) and the
  parameter-search task are planned.
- Deformation is neither simulated by the test double nor scored.

## Licence

Data generated by this benchmark: CC BY 4.0, "v2e-synth-1 (c) OpenBeam, LLC". Generated by
Apache-2.0 code; all geometry and textures are generated, no external assets.
