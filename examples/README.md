# Examples

Each example is a small, complete scene you can run offline on a laptop CPU. The four scene
examples are the default cases of the [v2e-synth-1 benchmark](../benchmarks/v2e-synth-1/), one per
kind of event, and each comes with a walkthrough script that takes it through every pipeline step
and explains what happens.

| Example | What it shows | Run |
| --- | --- | --- |
| [`quickstart_fake`](quickstart_fake/) | The connector path alone: negotiate a plan, spawn at rest, run a keyframed program, audit the ledger. | see its README |
| [`barrier_crash`](barrier_crash/) | A generated car drives into a fixed concrete block; impact and after-impact measures. | `python examples/barrier_crash/walkthrough.py --out runs/barrier` |
| [`ramp_jump`](ramp_jump/) | A generated car leaves a launch ramp, flies and lands on level ground. | `python examples/ramp_jump/walkthrough.py --out runs/ramp` |
| [`rollover`](rollover/) | A short ramp under the left wheels tips a generated car over. | `python examples/rollover/walkthrough.py --out runs/rollover` |
| [`drop_and_bounce`](drop_and_bounce/) | A box, a ball and a cylinder fall onto a slab and bounce; scored, not fitted. | `python examples/drop_and_bounce/walkthrough.py --out runs/drop` |
| [`custom_perception_model`](custom_perception_model/) | A complete segmenter plug-in (`example.colour_key`) and how to run it on any clip or on the benchmark. | see its README |

Every scene example folder holds:

- `case.json`: the scene's parameters, camera, level primitives, objects, events and phases,
  with the `v2e.scene/1`, `v2e.templates/1` and targets documents;
- `templates.json`: which content stands in for each object;
- `targets.json`: every target value with its source: a design value, physics from first
  principles, or "reported" when v0 sets no bar. Engine runs are never a source;
- `walkthrough.py`: the narrated run;
- `README.md`: the scene, the steps, the targets and the scene's limits.

The clips and truth packages are generated when you run an example (a few seconds to a minute on
a laptop); they are deterministic, so two machines produce the same frames and the same truth.

## Expected claim class

On the `fake` connector every run is `test_double`: the connector is a rigid-body stand-in for
tests, so its runs say nothing about any physics engine. The same scenes on an engine connector
are expected to be `native` (every effect from the engine's own mechanisms driven by authored
content).

## Your own footage

These examples start from generated clips so that the truth is exact. To recreate your own clip,
follow [Getting started](../docs/getting-started.md): the same steps apply, with your video, a few
outlines per moving object and some surveyed points in place of the generated inputs.
