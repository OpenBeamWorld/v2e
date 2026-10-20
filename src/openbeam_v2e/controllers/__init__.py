# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""Controller programs and reference driver models.

``v2e.program/1`` describes one engine-neutral program per controlled actor, addressing
actor-scoped actuator channels by id. The Python reference models (command schedule, path
follower, time-indexed tracker, passive) define the behaviour that engine-side hosts must
reproduce on synthetic golden traces. Controllers act only through declared actuator
channels; there is no way to set forces, poses or velocities.
"""
