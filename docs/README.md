# Documentation

Start with [Getting started](getting-started.md). Guides without a link are planned and land
with the code they describe.

| Guide | Covers |
| --- | --- |
| [Getting started](getting-started.md) | Install, run an example end to end, run the benchmark, and recreate your own clip. |
| [Validation and remaining gates](validation.md) | What the offline checks establish, benchmark quality failures, and work still needed before release. |
| Concepts | The pipeline end to end: reconstruction, scene and templates, compiler, connectors, run and export, scoring. |
| Scope | What v0 can and cannot reconstruct, and what is planned after v0. |
| [Reconstruction](reconstruction.md) | Running reconstruction; the `v2e.reconstruct/1` configuration and `v2e.annotations/1` prompts; outputs; what the CPU baseline does at each stage; how uncertainty is computed; installing model adapters; the limits of v0. |
| [Adding a perception model](perception-models.md) | The perception capabilities and result types, writing and registering a model, choosing it per stage, and measuring it on the benchmark. |
| [Scenes and templates](scene-and-templates.md) | `v2e.scene/1` and `v2e.templates/1`, run-up synthesis and frame-0 alignment, with examples. |
| [The compiler](compiler.md) | Scene + templates + a connector's content catalog to a `v2e.package/1` request: generated vehicles and props, stock matches, level collision, run-ups, refusals and connector dialects. |
| [Connectors](connectors.md) | The `Backend` interface, capability flags, negotiation and refusal, the call ledger, and how to write a new connector. |
| [OpenBeam connector](openbeam.md) | Connecting to a local game or a cloud session through the documented MCP tools; installing compiled packages. |
| Honesty | Claim classes, implementation layers, timing regimes and measured repeatability (for now in [Connectors](connectors.md#runs-the-call-ledger-and-claim-classes)). |
| [Scoring](scoring.md) | Metrics per phase, events, silhouettes, post-event metrics, targets, uncertainty and repeat statistics. |
| [World packages](world-package.md) | The world-package layout and V2E sidecars, writing a world from your own model, reading packages made by other tools, and runs as world packages. |
| [AI agents](agents.md) | The CLI and the MCP server: tool list, schemas and an example session. |
| [Examples](../examples/README.md) | One narrated walkthrough per kind of event (barrier crash, ramp jump, rollover, drop and bounce), the connector quickstart and a custom perception model. |
| [Benchmark v2e-synth-1](../benchmarks/v2e-synth-1/README.md) | The synthetic benchmark: families, cameras, generated files, tasks and report columns. |
| Data licensing | CC BY 4.0 for data we make; rules for third-party clips. |
