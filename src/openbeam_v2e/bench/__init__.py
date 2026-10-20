# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""Step 8: benchmarks.

The synthetic benchmark ``v2e-synth-1``: four families of physical events (barrier crash, ramp
jump, rollover, drop and bounce) generated with exact truth and run through the whole pipeline.

* :mod:`.families`: the scene families, their parameter grids, camera presets and
  first-principles targets with sources;
* :mod:`.geometry`: level primitives as rendered solids and connector collision boxes;
* :mod:`.drive`: the benchmark's staged cruise driver (actuator channels only);
* :mod:`.synth`: case generation (simulate on a connector, align, detect events, render, write);
* :mod:`.truth`: dense truth tracks, geometric event detection and truth metrics;
* :mod:`.metrics`: world-to-world comparisons (camera, dimensions, paths, attitudes, events,
  silhouettes);
* :mod:`.runner`: generate, reconstruct, recreate, score and tabulate;
* :mod:`.report`: Markdown, CSV and JSON reports;
* :mod:`.walkthrough`: the narrated per-family runs behind ``examples/<family>/walkthrough.py``;
* :mod:`.cli`: ``python -m openbeam_v2e.bench synth ...``.

Every figure from this benchmark is labelled "synthetic self-consistency": the truth is a
simulated run, so it supports no claim about accuracy on real footage.
"""
