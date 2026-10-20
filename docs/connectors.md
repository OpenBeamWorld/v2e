# Connectors

A connector is the only part of V2E that talks to a physics engine. It installs compiled
packages, opens a scene, spawns every actor at rest at a pose, runs one controller program per
controlled actor and reports what happened. It never reconstructs, scores or searches; those
steps are engine-free and live in the rest of the package.

In the code a connector is a *backend*: the interface is `openbeam_v2e.backends.base.Backend`
and plugins register through the `openbeam_v2e.backends` entry point group.

## Connectors in this release

| Connector | What it is | Claim class of its runs |
| --- | --- | --- |
| `openbeam` | An OpenBeam game or cloud session, reached only through the documented MCP tools (see [OpenBeam connector](openbeam.md)). | `native` when every required effect is native, otherwise as computed |
| `fake` | A deterministic rigid-body stand-in for tests and the offline quickstart. | always `test_double` |

A BeamNG connector is planned for a later release.

## Rules every connector follows

1. **Install, run, report.** Nothing else.
2. **No physics bypass can be expressed.** The interface has no way to set a force, impulse,
   pose, velocity, gravity or mass, to teleport an actor or to reset it during a run. The only
   spawn is "at rest at a pose". A public test checks every connector's method names.
3. **Actuation goes through declared channels.** A controller commands actor-scoped channel ids
   such as `drive.throttle` with values of the channel's kind and range; the harness checks each
   command against the plan before it reaches the connector.
4. **Capabilities are declared, never assumed.** The harness reads them from the connector's
   manifest and refuses a scene the connector cannot run honestly.
5. **Provenance is computed, not narrated.** Every effect has a source; the claim class follows
   from the sources.

## The manifest

`Backend.manifest()` returns a `BackendManifest`: the connector's name, whether it is an engine
or a test double, the public engine name and version, the API it uses, its content dialect, and
its `Capabilities`:

| Field | Meaning |
| --- | --- |
| `channels` | Actuator channels: id, effect (`drive`, `brake`, `steer`, ...), kind (`continuous`, `toggle`, `discrete`), range or levels, the public mechanism behind it and the implementation layer. Core ids: `drive.throttle` (0..1), `drive.brake` (0..1), `drive.steer` (-1 left .. +1 right), `drive.parking_brake`. Plugins add `<plugin>.<name>` channels. |
| `soft_body`, `detachment` | Whether deformation and part detachment can happen at all. |
| `regimes` | Timing regimes (`render_paced`, `stepped`, or plugin regimes), each with availability, the scene conditions it needs (for example a player vehicle) and its **measured** repeatability (`bit_identical`, `statistical` or `unmeasured`) with the report it came from. |
| `physics_dt_s` | The engine's physics step, when known. |
| `recordings` | How frames can be captured: `stepped_capture`, `stream_capture`, `sensor_lockstep`, `raster_test_double` or `none`. |
| `spawn_at_pose` | Whether position **and** rotation can be requested at spawn. |
| `max_controlled_actors`, `max_seatable_vehicles` | How many actors may run a program at once, and the engine's limit on seatable vehicles. |
| `controller_hosts` | Where controllers run: `engine_script`, `vehicle_script` or `client_lockstep` (Python between explicit steps). |
| `observables` | What rows can report: `position`, `attitude`, `velocity`, `angular_velocity`, ... |
| `camera_runtime_fov`, `instance_mask` | Camera features. |

## Negotiation and refusal

`harness.negotiate(scene, manifest, opt_in=..., programs=...)` returns a `Plan` or a `Refusal`
listing every reason, for example:

- a required effect, channel (of the required kind) or observable is missing;
- more controlled actors or seatable vehicles than the connector allows;
- the requested regime is unavailable, or its scene conditions are not met (a stepped regime
  that needs a player vehicle can use an off-camera anchor vehicle only if you opt in with
  `add_anchor`, and the anchor counts toward the seatable limit);
- a controlled actor has no program, or a passive actor has one;
- an initial state cannot be spawned at rest.

A substitute mechanism is used only if you name it (`opt_in.substitutes`) **and** the connector
declares it; the run is then labelled `emulated`.

## Runs, the call ledger and claim classes

`harness.execute(backend, compiled, plan, spec, run_dir)` writes:

| File | Content |
| --- | --- |
| `plan.json` | The negotiated plan, written before anything is installed. |
| `install.json` | One receipt per installed package. |
| `ledger/calls.jsonl` | One line per engine call: sequence number, wall time, simulated time, the call and its **full arguments**, a digest of the result and the hash of the previous line (a hash chain). URLs are stored without query strings; credentials never appear. |
| `ledger/blobs/<sha256>` | Program texts and script bodies, stored once by content. |
| `rows.jsonl` | `v2e-runlog/1`: a header line, then one row per sample with each actor's pose in the Z-up level frame and the commands in force. |
| `run.json` | The run record: manifest, plan, effects, content origins, spawn attestation (requested versus actual pose of every actor), events, `t0_s`, validity, claim class, physics statement and provenance signature. |

`openbeam-v2e audit RUN` replays the ledger offline against the connector's allow list: the hash
chain must be intact, every call must be allowed with only its documented arguments, every
script body must come from the audited snippet set and every blob must match its hash.

A run is **valid** only if the audit is clean, no actor teleported, was reset or was removed by
the engine, every actor spawned at rest, and every spawn residual is within tolerance (2 cm,
0.5 degrees). Invalid runs are never scored.

Claim classes:

| Class | When |
| --- | --- |
| `native` | Every required effect is the engine's own mechanism driven by authored content. |
| `emulated` | At least one effect uses a declared substitute mechanism; nothing is scripted. |
| `scripted` | At least one effect comes from plugin code pinned by sha256. This package never ships such code. |
| `incomplete` | A required effect is absent. |
| `test_double` | The run used a test double. It supports no claim about any engine. |

Runs whose effects or content differ get different provenance signatures and are never grouped.

## Writing a connector

1. Subclass `Backend` (and `Lockstep` if controllers should run in Python between steps):

   ```python
   from openbeam_v2e.backends.base import Backend, BackendManifest, Capabilities

   class MyEngineBackend(Backend):
       name = "myengine"

       @classmethod
       def from_config(cls, config):           # endpoints and credentials come from here only
           return cls(**config)

       def manifest(self) -> BackendManifest: ...
       def mechanisms(self, plan): ...         # one Effect per plan effect, with its true source
       def install(self, packages): ...        # idempotent by sha256
       def open_scene(self, level): ...
       def spawn(self, obj, pose, actor): ...  # at rest; return (ActorRef, SpawnRecord with the actual pose)
       def run(self, actors, programs, spec, run_dir): ...   # write rows.jsonl, return a RunResult
       def reset_between_runs(self): ...
       def close(self): ...

       @classmethod
       def audit_policy(cls): ...              # the calls and argument names the ledger audit allows
   ```

2. Record every engine call with `self.ledger.record(method, args, result)` (the harness attaches
   a fresh ledger to each run) and store program texts with `self.ledger.store_blob(...)`.
3. Write rows with `openbeam_v2e.backends.lockstep.RowWriter`; for a `Lockstep` connector,
   `run_lockstep(...)` drives the controllers, checks every command and writes the rows for you.
4. Report only what you measured: an orientation you cannot read back is an unverified residual,
   not a copy of the request.
5. Register the connector:

   ```toml
   [project.entry-points."openbeam_v2e.backends"]
   myengine = "my_package.connector:MyEngineBackend"
   ```

6. Add a case to `tests/contract/test_connector_contract.py` (a scene and compiled content your
   connector can run, with a recorded cassette if it talks to a remote service) and run the
   suite.

The compiler writes an engine-neutral `v2e.package/1` request; a connector's dialect turns it
into a `v2e.compiled/1` scene (`openbeam_v2e.backends.base.CompiledScene`): the packages to
install, the objects they contain, one at-rest spawn per actor, the programs, the content-origin
records and the timeline's `t0_s`. The fake connector's dialect is
`openbeam_v2e.backends.fake.compile_package`.
