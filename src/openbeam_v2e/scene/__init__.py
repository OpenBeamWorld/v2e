# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""Step 2: scene description and template mapping.

Schemas and helpers for ``v2e.scene/1`` (actors, roles, initial states, events, phases,
targets with sigma and source, required effects and observables) and
``v2e.templates/1`` (which content represents each reconstructed object: a generated
generic vehicle, a stock vehicle where the target engine has one, a generated prop, or
static collision). Also run-up synthesis for actors that enter the scene moving, and the
frame-0 alignment between video time and simulated time.

Initial states never inject velocity: actors spawn at rest and controllers bring them to
their observed state.
"""
