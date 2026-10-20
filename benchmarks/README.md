# Benchmarks

| Benchmark | What it measures | Status |
| --- | --- | --- |
| [`v2e-synth-1`](v2e-synth-1/) | Reconstruction (camera, dimensions, trajectories, events) and the full round trip (reconstruct, compile, run, score) on four families of simulated events with exact truth. | First release: generated offline with the `fake` connector and the built-in ray caster. Engine-rendered truth is planned. |

Every figure from `v2e-synth-1` is labelled **synthetic self-consistency**: its truth is a
simulated run, so it says how well the pipeline recovers what a simulator produced, not how
accurate it is on real footage. Accuracy claims come only from real clips with independent
measurements (planned for a later release).
