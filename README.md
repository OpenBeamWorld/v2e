# OpenBeam V2E (Video-to-Engine)

OpenBeam V2E is a general framework for recreating a filmed physical event in a physics engine
and measuring how closely the engine run matches the footage. It reconstructs the video, compiles
the reconstruction into content the engine can load, runs it through a connector and scores the
result, with uncertainty at every step. It is not one perception model: videos within the v0 scope 
use the same pipeline, perception models can be replaced per stage, and any MCP-capable AI agent 
can drive the tools.

The Python distribution is `openbeam-v2e`, the import package is `openbeam_v2e` and the command
is `openbeam-v2e`.

> **Status: private v0 preview (pre-alpha).**

## Contents

- [The pipeline](#the-pipeline)
  1. [Reconstruction](#1-reconstruction)
  2. [Scene and templates](#2-scene-and-templates)
  3. [Compiler](#3-compiler)
  4. [Connectors](#4-connectors)
  5. [Run and export](#5-run-and-export)
  6. [Scoring](#6-scoring)
  7. [Using it from AI agents](#7-using-it-from-ai-agents)
  8. [Benchmark and examples](#8-benchmark-and-examples)
- [Install](#install)
- [Quickstart](#quickstart)
- [Status of this preview](#status-of-this-preview)
- [Scope of v0](#scope-of-v0)
- [What is not included](#what-is-not-included)
- [Licence](#licence)

## The pipeline

```
 video ──► 1 reconstruct ──► world package ──► 2 scene + templates ──► 3 compile ──► package
                (perception models)                                     (connector catalog)
                                                                                     │
 score report ◄── 6 score ◄── world package ◄── 5 export ◄── run dir ◄── 4 connector run
```

Every step reads and writes plain files in documented formats (JSON, NPZ arrays, OBJ meshes),
so you can enter or leave the pipeline at any step: score worlds produced by another tool,
compile a scene you wrote by hand, or write a connector for any engine.

### 1. Reconstruction

Video in, **world package** out. The world package holds:

- the **camera track**: intrinsics, lens distortion and pan, tilt, roll and zoom for every frame
  of a camera that stays in one place;
- the **static geometry**: the ground plane from calibration, plus any static meshes you supply
  (surveyed or modelled; recovering other structure from the footage is not built yet);
- each **moving object**: its class, symmetry class, measured dimensions with sigmas, a rigid
  pose track (position and attitude per frame, with covariance) and its observed outline;
- the **events** (contact, take-off, landing, impact, rest) as frame intervals;
- the fit's **uncertainty**, fitted and held-out residuals, and model provenance.

You provide the video (an MP4 through an external FFmpeg, or a folder of frames), an outline,
box, mask or a few clicks per moving object on a few frames (`v2e.annotations/1`), and calibration
evidence in a configuration file (`v2e.reconstruct/1`): surveyed points, or straight edges along
two ground directions plus one known distance.

```sh
openbeam-v2e reconstruct clip.mp4 --config reconstruct.json --annotations annotations.json --out world/
```

**Perception models are pluggable.** Reconstruction asks the perception registry for a
capability (calibration, camera tracking, detection, segmentation and mask tracking, point
tracking, depth, pose fitting) and gets whichever model the configuration names. The base
install ships CPU baselines that need only NumPy and SciPy. Optional adapters load SAM 2,
CoTracker, and Hugging Face depth and detection models only when used; no model code or weights
are copied into this package. Your own model registers through the `openbeam_v2e.perception`
entry point group and can be chosen per stage (`--model masks=my.segmenter`).

Guides: [Reconstruction](docs/reconstruction.md), [Adding a perception model](docs/perception-models.md),
[World packages](docs/world-package.md).

### 2. Scene and templates

A **scene** (`v2e.scene/1`) says what to recreate: actors and initial states, events, phases,
targets with their sources, and required connector capabilities. Poses can be explicit or
references into the world package (`track:car@0`).

A **template mapping** (`v2e.templates/1`) chooses content for each object: a generic vehicle
generated from its measurements, the nearest stock vehicle a connector offers, a generated prop
with a mass from a density class, or static collision geometry.

Guide: [Scenes and templates](docs/scene-and-templates.md).

### 3. Compiler

The compiler turns **scene + templates + a connector's content catalog** (`v2e.catalog/1`)
into an engine-neutral package request (`v2e.package/1`), or coded refusals:

- level collision is written as OBJ meshes from the world package's static geometry;
- each actor gets generated content or the nearest stock match, with per-dimension residuals;
- **every actor spawns at rest.** Drive-in actors get a generated straight or arc run-up whose
  corridor is checked against static geometry and other objects;
- the request carries the frame-0 timeline and a reference path for every controlled actor.

A connector **dialect** turns the request into what that connector loads. Dialects register
through the `openbeam_v2e.dialect_packs` entry point group; the `fake` connector ships its own.

```sh
openbeam-v2e compile scene.json --templates templates.json --connector fake --out compiled/
```

Guide: [The compiler](docs/compiler.md).

### 4. Connectors

A connector is the only part of V2E that talks to a physics engine. It installs compiled content,
opens the scene, spawns actors at rest, runs a program per controlled actor and reports what
happened. Reconstruction, compiling and scoring stay engine-free.

| Connector | Status |
| --- | --- |
| `openbeam` | Included. Uses documented customer MCP tools. Passive scenes only without a separately accepted script host; full attitude is not observed. Spawn orientation remains unverified until its convention is configured against a real game. No compiled-content dialect yet. |
| `fake` | Included: deterministic rigid-body test double used by tests, examples and the benchmark. Results are labelled `test_double`, never native engine evidence. |
| BeamNG | Planned for a later release. |

**Capability flags.** Each connector declares actuator channels (effect, kind, range and public
mechanism), timing regimes and measured repeatability, recording modes, actor limits, controller
hosts and observables. Negotiation returns a plan or an itemised refusal; a connector is never
asked to run a scene beyond its declared capabilities.

**No physics bypass.** The connector interface has no way to set a force, impulse, pose,
velocity, gravity or mass, to teleport an actor or to reset it during a run. The only spawn is
at rest at a pose; controllers act through declared channels such as `drive.throttle`,
`drive.brake` and `drive.steer`. Connector contracts test this boundary.

**Write your own connector:** implement `openbeam_v2e.backends.base.Backend`, declare a manifest,
record engine calls in the ledger and report only measured values. An orientation you cannot
read back must not be copied from the request. Register it under the `openbeam_v2e.backends`
entry point group and run the connector contract tests.

Guides: [Connectors](docs/connectors.md), [OpenBeam limitations and configuration](docs/openbeam.md).

### 5. Run and export

`run` negotiates a plan, writes it before installing anything, and produces a run directory:

- `ledger/calls.jsonl`: a hash-chained call ledger with arguments and result digests; credentials
  are redacted and URLs lose query strings. `openbeam-v2e audit` checks it offline;
- `rows.jsonl` (`v2e-runlog/1`): observed actor states and commands at each sample. Unobserved
  attitude is explicit, never a fabricated orientation;
- `run.json`: requested versus actual spawn poses, validity and a computed claim class.

The claim class records how the run was produced: `native`, `emulated`, `scripted`, `incomplete`
or `test_double`. The last is mandatory for `fake`; it is not evidence about an engine.
See the [connector guide](docs/connectors.md#runs-the-call-ledger-and-claim-classes) for the rules.

Runs that fail the audit, lose an actor or exceed spawn tolerances are marked invalid and not
scored. `export` writes the run as a world package on the footage's timeline through its
reconstructed camera. Full-pose export refuses logs with missing attitude.

```sh
openbeam-v2e run --compiled compiled/ --programs programs.json --out runs/one --duration 6
openbeam-v2e audit runs/one
openbeam-v2e export runs/one --reference world/ --out runs/one-world
```

### 6. Scoring

The scorer compares a run or world package with the reconstruction:

- **per phase and object**: hull-centre position error, attitude error modulo object symmetry,
  z-scores against reconstruction uncertainty, speed error and coverage;
- **silhouettes**: the run's hull through the reconstructed camera against the observed outline,
  alongside the best overlap the content's shape could reach;
- **events**: detected contacts, separations and rest matched to the scene, with timing error
  and uncertainty, including between-frame impacts;
- **after events**: rebound speed, displacement and yaw change, rollover and bounce measures;
- **targets**, claim class, provenance, a search objective and robust repeat statistics.

```sh
openbeam-v2e score runs/one --scene scene.json --reconstruction world/
```

Guide: [Scoring](docs/scoring.md).

### 7. Using it from AI agents

The same seven tools back the CLI and stdio MCP server: `reconstruct`, `compile`, `run`,
`export`, `score`, `inspect` and `bench`. Arguments have JSON Schemas, results are JSON objects,
and failures carry codes an agent can act on.

```sh
openbeam-v2e mcp --root /path/to/workspace
```

Register the command with any MCP-capable client, for example:

```json
{"mcpServers": {"openbeam-v2e": {"command": "openbeam-v2e",
                                 "args": ["mcp", "--root", "/path/to/workspace"]}}}
```

Paths are confined to the workspace. Connectors and credentials come from the operator, never
tool arguments. The server performs no network I/O except through configured connectors.

Guide: [Using V2E from AI agents](docs/agents.md).

### 8. Benchmark and examples

- **[v2e-synth-1](benchmarks/v2e-synth-1/README.md)**: four families (barrier crash, ramp jump,
  rollover, drop and bounce), with a 40-case grid over tripod and fixed-position pan/tilt/zoom
  cameras. Generic content produces deterministic truth, then reconstruction (CPU baseline,
  replacement models or truth as an oracle), recreation and scoring produce a report. Every
  figure is **synthetic self-consistency**; completion does not mean accuracy targets passed.
- **[Examples](examples/README.md)**: one narrated walkthrough per family, the connector
  quickstart, and a complete custom perception model plug-in.
- Generated cases include `programs.json`, so their benchmark cruise driver also works through
  the normal compiler, CLI and MCP run tools.

```sh
python -m openbeam_v2e.bench synth run --out runs/synth   # one case per family, with a table
openbeam-v2e bench synth --out runs/synth                 # the same through the package command
```

## Install

Python 3.10 or newer. From a checkout of this repository:

```sh
python -m pip install -e ".[test]"
openbeam-v2e --version
python -m pytest -q
```

The base install depends only on NumPy and SciPy. Optional extras:

| Extra | Adds |
| --- | --- |
| `test` | pytest and Pillow for image-fixture coverage |
| `perception` | PyTorch, Transformers and the Hugging Face hub, for depth and detection adapters |
| `sam2` | SAM 2 for segmentation and mask tracking |

The MCP server needs no extra. FFmpeg is an external program for MP4 input/output, not bundled;
folders of PNG frames work without it.

## Quickstart

Run one complete example offline: generate a synthetic clip with exact truth, reconstruct it,
recreate it on `fake` and score it.

```sh
python examples/barrier_crash/walkthrough.py --out runs/barrier
```

Then try the connector path on its own:

```sh
openbeam-v2e inspect
openbeam-v2e inspect examples/quickstart_fake/scene.json
openbeam-v2e run --scene examples/quickstart_fake/scene.json \
    --programs examples/quickstart_fake/programs.json --out runs/quickstart --duration 6
openbeam-v2e inspect runs/quickstart
```

For your own clip, follow [Getting started](docs/getting-started.md#4-your-own-clip).
All guides are listed in the [documentation index](docs/README.md).

## Status of this preview

**Verified offline:** CPU reconstruction on synthetic clips; schemas and world-package round
trips; compiler/run-up synthesis and the fake dialect; connector negotiation, ledger and audit;
CLI and stdio MCP; all four families through the general compiler, run, export and score tools;
wheel build/install checks. Offline CI covers Python 3.10 and 3.12.

Latest measurements and reproduction commands: [Validation and remaining gates](docs/validation.md).

## Scope of v0

- Footage from a single camera that stays where it is; panning, tilting, zooming and lens
  distortion are handled.
- Vehicles and props are tracked as rigid bodies. The engine still simulates deformation, but
  V2E neither reconstructs nor scores it yet.
- Annotation is assisted: a prompt on at least one frame per moving object, or a detection model.
- Control goes through actuators only: inputs plus whatever channels a connector declares.

## What is not included

- **No engine, game or engine source.**
- **No BeamNG connector** in this release; one is planned for later.
- **No model weights** and no copied model code; adapters load libraries you install.
- **No real footage** or data derived from third-party video. Examples and benchmark frames are
  generated by this package from generic content.
- Not in v0: travelling cameras (dashcam, drone, walking handheld), cameras riding on an object,
  several cameras at once, deformation reconstruction, or passive-object material fitting.

## Licence

Code: [Apache-2.0](LICENSE). Data that OpenBeam, LLC makes for this project (benchmark truth,
annotations, example scenes): [CC BY 4.0](LICENSE-DATA). Third-party data keeps its own licence,
recorded next to it.

World packages use the 4DCodeBench layout. V2E's `fourdcb` module reads and writes that layout
and was written separately from its published description; the 4DCodeBench authors have not
reviewed or endorsed it, and V2E has no affiliation with them.

Trademarks: the OpenBeam name is used by OpenBeam, LLC as a trademark. BeamNG GmbH owns the
BeamNG.drive® and BeamNG.tech® marks; it has not endorsed V2E and has no affiliation with it.

Security reports: see [SECURITY.md](SECURITY.md). Contributions: see [CONTRIBUTING.md](CONTRIBUTING.md).
