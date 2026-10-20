# Using V2E from AI agents

V2E exposes its pipeline as a small set of JSON-schema tools. The same tools back the
`openbeam-v2e` command line and an MCP server, so a person, a script and any MCP-capable model
drive the pipeline the same way.

| Tool | Step | What it does |
| --- | --- | --- |
| `reconstruct` | 1 | Video in, world package out: camera track, static geometry, objects with measured dimensions and rigid pose tracks, events, all with uncertainty. `models` picks a perception model per stage. |
| `compile` | 3 | Scene (plus templates) to a `v2e.package/1` request, then to the target connector's `compiled.json`. |
| `run` | 4-5 | Negotiate a plan or an itemised refusal, install, spawn at rest, run one program per controlled actor, write the run directory (plan, ledger, rows, `run.json`). |
| `export` | 5 | A run as a world package on the footage's timeline, seen through the reconstructed camera. |
| `score` | 6 | A run against its reconstruction and scene. |
| `inspect` | any | Summarise a run (validity, claim class, ledger audit replay), a scene (and whether each configured connector can run it), a compiled scene or a report; describe a connector; or, with no arguments, list connectors and the steps this installation has. |
| `bench` | 8 | `repeat`: run one plan N times and report measured repeatability. `synth`: the synthetic benchmark. |

Every tool answers with a JSON object. Failures have `"ok": false` and an `error.code`:
`bad_arguments`, `refused` (with the connector's reasons), `step_unavailable` (this installation
lacks a step, and the answer says which entry point is missing), `connector_not_configured`,
`compile_first` or `failed`.

## Command line

```sh
openbeam-v2e inspect                                   # connectors and available steps
openbeam-v2e compile scene.json --out compiled          # --connector fake by default
openbeam-v2e run --compiled compiled --programs programs.json --out runs/one --duration 6
openbeam-v2e inspect runs/one                          # validity, claim class, audit replay
openbeam-v2e audit runs/one
openbeam-v2e export runs/one --reference world --out runs/one-world
openbeam-v2e score runs/one --scene scene.json --reconstruction world
openbeam-v2e bench repeat --scene scene.json --programs programs.json --out bench -n 9
```

Exit status: 0 on success, 1 when a tool fails or a connector refuses, 2 for usage errors and
steps that are not available. Connectors other than `fake` are configured with
`--connector-config FILE` (see [OpenBeam connector](openbeam.md)).

## MCP server

```sh
openbeam-v2e mcp --root /path/to/workspace [--connector-config connectors.json]
```

The server speaks MCP over stdio (protocol revisions 2025-06-18, 2025-03-26 and 2024-11-05) and
needs nothing beyond the base install. Register the command with your client; for example, in a
client that reads an `mcpServers` map:

```json
{"mcpServers": {"openbeam-v2e": {"command": "openbeam-v2e",
                                 "args": ["mcp", "--root", "/path/to/workspace"]}}}
```

Tool results carry the JSON answer as `structuredContent` and as a text block. A failed step is
a result with `isError: true`.

### Safety rules

- **The workspace is a sandbox.** Every path argument is resolved inside `--root`; absolute paths
  elsewhere and `..` escapes are refused. Outputs go to new or empty directories.
- **Connectors come from the operator.** A tool may name a connector, but only one configured
  when the server started. Endpoints and credentials are never tool arguments.
- **No network I/O of its own.** The server reaches an engine only through a configured
  connector. Everything a pipeline step prints goes to stderr, never into the protocol stream.
- **No physics bypass.** No tool can set forces, poses or velocities; runs carry a computed claim
  class and an auditable call ledger.

### An example session

1. `inspect` with `{}`: which connectors are configured and which steps are installed.
2. `inspect` with `{"path": "scene.json"}`: the actors, what the scene requires, and per
   connector either "ok" or the reasons it would refuse.
3. `compile` with `{"scene": "scene.json", "out": "compiled"}`.
4. `run` with `{"compiled": "compiled", "programs": "programs.json", "out": "runs/one",
   "duration_s": 6}`.
5. `inspect` with `{"path": "runs/one"}`: check `valid`, `claim_class` and `audit_replay`.
6. `score` with `{"run": "runs/one", "scene": "scene.json", "reconstruction": "world"}`.

## Pipeline step entry points

The tools find steps owned by other subpackages through these functions; a step is available
when its function exists. They are called with keyword arguments, and only the keywords a
function declares are passed.

| Step | Entry point |
| --- | --- |
| reconstruct | `openbeam_v2e.reconstruct.pipeline:reconstruct` |
| compile | `openbeam_v2e.compile:compile_scene` |
| dialect | `openbeam_v2e.compile.dialects:build_compiled` |
| export | `openbeam_v2e.world.export:export_run` |
| score | `openbeam_v2e.score:score_run` |
| synth | `openbeam_v2e.bench.runner:run_bench` |
