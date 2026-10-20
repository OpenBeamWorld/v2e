# Getting started

This guide installs OpenBeam V2E, runs a complete example on a laptop CPU, runs the synthetic
benchmark, and then shows how the same steps apply to your own clip. Everything here is offline:
no account, no GPU and no engine install is needed until you choose an engine connector.

## Install

Python 3.10 or newer. From a checkout of the repository:

```sh
python -m pip install -e ".[test]"
openbeam-v2e --version
python -m pytest -q
```

The base install depends only on NumPy and SciPy, and includes the MCP server for AI agents
(`openbeam-v2e mcp`). Optional extras add heavier pieces:

| Extra | Adds |
| --- | --- |
| `perception` | PyTorch, Transformers and the Hugging Face hub, for the depth and detection adapters; PyTorch for the `cotracker` and `tapnext` point trackers, whose libraries install from their repositories |
| `sam2` | SAM 2 for segmentation and mask tracking |

FFmpeg is used as an external program when you read MP4 files or ask for an MP4 of a generated
clip; it is not bundled. Folders of PNG frames work without it.

## 1. Run one example end to end

```sh
python examples/barrier_crash/walkthrough.py --out runs/barrier
```

In under a minute this generates a 3-second clip of a generated car hitting a concrete block,
with exact truth; reconstructs the clip; recreates the scene on the `fake` connector; and scores
the run against the truth. Each step is printed with what it produced. The
[barrier crash README](../examples/barrier_crash/README.md) explains every step and every
target. The other scene examples (`ramp_jump`, `rollover`, `drop_and_bounce`) work the same way.

Look at what was written:

| Path | What it is |
| --- | --- |
| `runs/barrier/cases/<case>/frames/` | the rendered clip as PNG frames |
| `runs/barrier/cases/<case>/annotations.json` | outlines of the moving object in a few frames (assisted annotation) |
| `runs/barrier/cases/<case>/reconstruct.json` | the reconstruction settings: surveyed points, object classes, models |
| `runs/barrier/cases/<case>/world/` | the truth as a world package (camera, static meshes, object tracks, events) |
| `runs/barrier/cases/<case>/scene.json` | the scene (`v2e.scene/1`): actors, events, phases, targets |
| `runs/barrier/cases/<case>/programs.json` | controller programs accepted by the normal CLI and MCP run tools |
| `runs/barrier/recon/<case>/` | the reconstruction, as a world package |
| `runs/barrier/runs/<case>/<source>/` | the connector run: plan, call ledger, rows, `run.json` |

## 2. Inspect, compile, run and score with the package command

The walkthrough calls the same code as the `openbeam-v2e` command. With the case folder from
step 1 (`C=runs/barrier/cases/barrier_crash-v40-o100-tripod`):

```sh
openbeam-v2e inspect                       # connectors and the steps this installation has
openbeam-v2e inspect $C/world              # what a world package holds
openbeam-v2e reconstruct $C/frames --config $C/reconstruct.json --out runs/barrier/recon2
openbeam-v2e compile $C/scene.json --templates $C/templates.json --out runs/barrier/compiled
openbeam-v2e score runs/barrier/runs/barrier_crash-v40-o100-tripod/truth \
    --scene $C/scene.json --reconstruction $C/world
```

`compile` writes an engine-neutral package request and the connector's `compiled.json`.
Running it needs a controller program for each controlled actor. The generated case's
`programs.json` supplies the benchmark cruise driver through the same CLI and MCP interfaces;
choose a run duration that covers the compiled `t0_s` run-up plus the clip duration. The
[quickstart_fake example](../examples/quickstart_fake/README.md) instead uses a keyframed program.

## 3. Run the synthetic benchmark

```sh
python -m openbeam_v2e.bench synth list                  # the cases (add --subset full for all 40)
python -m openbeam_v2e.bench synth run --out runs/synth  # one case per family, with a table
openbeam-v2e bench synth --out runs/synth                # the same through the package command
```

The table compares the baseline reconstruction and an oracle (the truth used as the
reconstruction) on every case: camera reprojection, dimensions, paths, attitudes and events of
the reconstruction, then the recreated run's path error, event timing, silhouette overlap and the
targets it meets. `--model masks=<name>` swaps a perception model for one stage, so you can
compare models on the same cases. Every figure is labelled **synthetic self-consistency**: the
truth is simulated, so the benchmark tests the pipeline, not accuracy on real footage. See
[v2e-synth-1](../benchmarks/v2e-synth-1/README.md).

## 4. Your own clip

v0 handles footage from **one camera that stays where it is** (it may pan, tilt and zoom), with
vehicles and props treated as rigid bodies. You provide what a person can provide quickly:

1. **The video**: an MP4 (needs FFmpeg) or a folder of numbered frames.
2. **Outlines** of each moving object in a few frames (`v2e.annotations/1`): a box, a polygon, a
   mask or a few clicks per keyframe. Any labelling tool that writes this JSON works.
3. **Calibration evidence** in `reconstruct.json` (`v2e.reconstruct/1`): a handful of points
   whose world coordinates you measured (corners of road markings, posts, a tape-measured
   rectangle), with their pixel positions in one frame; or straight edges along two ground
   directions plus the camera height or one known distance.
4. **The objects** to reconstruct and their classes (`vehicle.car`, `prop.box`, ...).

Then:

```sh
openbeam-v2e reconstruct my_clip/frames --config my_clip/reconstruct.json --out my_clip/world
openbeam-v2e inspect my_clip/world
```

The reconstruction is a world package with uncertainty on every quantity. Write a scene
(`v2e.scene/1`) over it, naming the events and phases you care about and the targets you have
independent evidence for (see [Scenes and templates](scene-and-templates.md)), compile it for a
connector, run it and score it. To recreate it in OpenBeam, configure the OpenBeam connector
([OpenBeam connector](openbeam.md)).

## Where next

- [Reconstruction](reconstruction.md): the configuration and annotation formats, what each
  baseline stage does, and the uncertainty it reports.
- [Adding a perception model](perception-models.md): plug in another segmentation, tracking,
  depth or pose model.
- [Connectors](connectors.md): the connector interface, capability flags, refusals and how to
  write a connector for another engine.
- [Using V2E from AI agents](agents.md): the same tools over MCP.
- [Scoring](scoring.md) and [World packages](world-package.md): what the numbers mean and where
  they are stored.
