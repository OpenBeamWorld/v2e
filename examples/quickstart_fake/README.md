# quickstart_fake

The whole connector path offline, on the `fake` test connector: negotiate a plan, spawn every
actor at rest, run a timed controller program for the car, write the run directory with its
replayable call ledger, and audit it. It needs only the base install and runs in a few seconds.

| File | What it holds |
| --- | --- |
| `scene.json` | A `v2e.scene/1` scene: a generic car at rest on flat ground, a 0.5 m crate held 2.0 m up and a 0.4 m ball held 1.0 m up (both released at t = 0). Poses are explicit, so no world package is needed. |
| `programs.json` | One `v2e.program/1` program for the car (`schedule` model): throttle, then a gentle right turn, then full brake. |
| `targets.json` | First-principles checks (free-fall contact times) and the expected claim class. |

```sh
openbeam-v2e inspect examples/quickstart_fake/scene.json      # can each connector run it?
openbeam-v2e run --scene examples/quickstart_fake/scene.json \
    --programs examples/quickstart_fake/programs.json --out runs/quickstart --duration 6
openbeam-v2e inspect runs/quickstart                           # valid, claim class, audit replay
openbeam-v2e bench repeat --scene examples/quickstart_fake/scene.json \
    --programs examples/quickstart_fake/programs.json --out runs/quickstart-repeat -n 3 --duration 2
```

Expect `"claim_class": "test_double"`: the fake connector is a stand-in for tests, so its runs
support no claim about any physics engine. The repeat bench reports `bit_identical`.

With an OpenBeam connector configured (`--connector-config`), `inspect` on the scene reports that
OpenBeam would refuse it today, and why: the car is a controlled actor, and the OpenBeam connector
in this release bundles no engine-side controller host.
