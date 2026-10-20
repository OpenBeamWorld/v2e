# OpenBeam connector

The `openbeam` connector runs compiled scenes in OpenBeam through the documented MCP tools only.
It loads authored content compiled by V2E and nothing else.

## What it uses

The connector may call 25 tools, fixed in code as `PUBLIC_TOOLS` and checked against the
published tool reference:

- account tools (alphabetical): `cloud_game_call`, `cloud_game_tools`, `cloud_recording_download`,
  `cloud_recording_list`, `cloud_recording_start`, `cloud_recording_stop`, `cloud_session_end`,
  `cloud_session_list`, `cloud_session_start`;
- game tools (alphabetical): `actor_poses`, `camera_frame`, `delete_mod`, `install_mod`, `list_mods`,
  `load_level`, `quit`, `read_log`, `run_script`, `screenshot`, `sim_mode`, `sim_step`,
  `spawn_vehicle`, `status`, `telemetry`, `upload_level`.

Each tool may carry only its documented argument names (or fewer: `sim_step` carries only
`steps`). A call outside the list, or with another argument, is refused before any network I/O.
`run_script` bodies must be renderings of audited snippets; this release has one, which writes a
marker line to the script log. Every call is written to the run's ledger.

## Pipeline steps through the documented tools

| Step | Tools |
| --- | --- |
| install | `list_mods` (skip archives already present, by sha256), then `install_mod` for vehicles and props or `upload_level` for levels, from a URL you control plus the archive's sha256, or from a path on the game's machine |
| open scene | `load_level` with an explicit wait |
| spawn at rest | `spawn_vehicle` at a position pre-compensated from the object's bounding box; `actor_poses` reads the result back |
| run (passive scenes) | `sim_mode` stepped, then `sim_step` and `actor_poses` per row; optional `camera_frame` per row |
| end of run | `sim_mode` realtime, `telemetry` |

Coordinates: OpenBeam reports positions with Y up; V2E's level frame is Z up. The connector maps
`(x, y, z)` in V2E to `(x, z, -y)` in OpenBeam and back.

## What this release can and cannot do on OpenBeam

- **Passive scenes run** in the documented stepped mode (simulated time from the step count).
  Stepped mode needs a player vehicle in the scene and, in the cloud, a training session.
- **Rows report position and velocity** of each actor's reference point. The documented tools
  report no full attitude, so `q` is absent from OpenBeam rows and scenes that require the
  `attitude` observable are refused. The run-log header explicitly sets
  `attitude_observed: false`; the normal reader accepts these position-only rows.
  Exporting them as full-pose world packages (and scoring them that way) is refused
  rather than inventing an orientation.
- **Orientation at spawn is not settable** through `spawn_vehicle`, and the yaw that
  `actor_poses` reports is used only when you configure its convention (`yaw_convention`) after
  checking it against your game version. Without it, the spawn's orientation residual is recorded
  as unverified and the run is marked invalid.
- **Controlled actors need an engine-side host** written against a script API reference you
  accept explicitly in the connector configuration. This release bundles none, so the connector
  declares no actuator channels and refuses scenes with controlled actors.
- **At most two seatable vehicles.** OpenBeam removes the oldest seatable vehicle when a third
  spawns; the connector refuses a third and marks a run invalid if any actor disappears.
- **Archives are installed from URLs you control.** The documented tools cannot upload local
  files into a cloud session.

## Connecting

Credentials come from the `OPENBEAM_TOKEN` environment variable or from a token file
(default `$XDG_CONFIG_HOME/openbeam-v2e/token`, mode 0600). They travel only in the
`Authorization` header and never appear in arguments, ledgers or logs.

Sign in with the documented device-code flow:

```sh
openbeam-v2e connectors login openbeam
```

Open the printed link, approve the code, and the token is stored. Then describe the connection in
a connector configuration file and pass it with `--connector-config`:

```json
{"openbeam": {"session_id": "YOUR_SESSION_ID", "training": true,
              "yaw_convention": {"units": "deg", "sign": 1, "offset_deg": 0}}}
```

| Key | Meaning |
| --- | --- |
| `session_id` | A running cloud session, reached at `https://cloud.openbeam.world/v1/sessions/<id>/mcp`. |
| `account_session_id` | A cloud session reached through the account endpoint's `cloud_game_call` (needs a token your MCP client obtained for the account endpoint). |
| `endpoint` | The MCP address of a game you run yourself (plain `http` only on this machine). |
| `training` | The session is a training session, so stepped mode is available. |
| `yaw_convention` | How to read `actor_poses` yaw; leave it out until you have checked it. |
| `cassette` | Replay a recorded cassette instead of connecting (offline; used by the tests). |
| `token_file`, `base_url` | Where the token lives; the OpenBeam Cloud address. |

## Recording and replaying calls

`openbeam_v2e.backends.openbeam.cassette.RecordingClient` wraps a live client and writes a
`v2e.cassette/1` file (images reduced to their size and sha256, URLs without query strings);
`ReplayClient` serves it back in order and fails on any call that differs. The offline tests use
hand-written cassettes shaped like the documented tool results.

Live tests are opt-in: `python -m pytest -m openbeam` with your own session.
