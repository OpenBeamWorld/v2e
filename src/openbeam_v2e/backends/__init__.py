# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""Step 4: connectors (engine backends).

A connector installs packages, opens a scene, spawns actors at rest at a pose, runs one
controller program per actor and reports what happened. It never reconstructs, scores or
searches. This package holds the engine-agnostic ``Backend`` interface, the harness that
negotiates a plan or a refusal from a scene's requirements and a connector's declared
capabilities, the replayable call ledger, and the ``fake`` test connector.

Capabilities are declared, never assumed: actuator channels with their kind and range,
timing regimes with measured repeatability, recording classes, actor limits and
observables. No connector can express a physics bypass (force, impulse, pose or velocity
setters, teleports, in-run resets). Connectors for other engines register through the
``openbeam_v2e.backends`` entry point group.
"""
