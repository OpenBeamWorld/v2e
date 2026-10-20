# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""Cruise control follows the requested line in any horizontal coordinate frame."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from openbeam_v2e.backends.base import StateRow
from openbeam_v2e.backends.fake import FAKE_CHANNELS
from openbeam_v2e.bench import drive, families, synth


@pytest.mark.parametrize("heading_deg", [0.0, 30.0, -78.0, 90.0, 135.0, 180.0])
@pytest.mark.parametrize("offset", [-0.8, 0.8])
def test_cruise_commands_are_rotation_and_translation_equivariant(heading_deg, offset) -> None:
    def command(heading, origin):
        angle = math.radians(heading)
        forward = np.array([math.cos(angle), math.sin(angle)])
        left = np.array([-forward[1], forward[0]])
        p = np.asarray(origin) - 15.0 * forward + offset * left
        yaw = angle + 0.08
        q = (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))
        v = (8.0 * math.cos(yaw), 8.0 * math.sin(yaw), 0.0)
        state = StateRow("car", 0.0, (float(p[0]), float(p[1]), 0.7), q, v, (0.0, 0.0, 0.02))
        program = drive.cruise_program("car", 10.0, origin[1], [{"mode": "cruise"}], heading_deg=heading)
        program["params"]["line_x_m"] = origin[0]
        controller = drive.controller_factory(program, FAKE_CHANNELS)
        return controller.command(0.0, {"car": state})

    assert command(heading_deg, (7.0, -3.0)) == pytest.approx(command(0.0, (0.0, 0.0)), abs=1e-12)


def test_generated_cruise_line_goes_through_the_frame0_pose() -> None:
    case = families.FAMILIES["barrier_crash"].default_case()
    actor = dataclasses.replace(case.actor("car"), frame0_position_m=(7.0, -3.0, 0.725), frame0_yaw_deg=35.0)
    program = synth.programs_for(dataclasses.replace(case, actors=(actor,)))["car"]
    assert program["params"]["line_x_m"] == 7.0
    assert program["params"]["line_y_m"] == -3.0
    assert program["params"]["heading_deg"] == 35.0


def test_legacy_cruise_program_needs_no_line_x_coordinate() -> None:
    program = drive.cruise_program("car", 8.0, -3.0, [{"mode": "cruise"}])
    program["params"].pop("line_x_m")
    state = StateRow("car", 0.0, (100.0, -3.0, 0.7), (0.0, 0.0, 0.0, 1.0),
                     (8.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    controller = drive.controller_factory(program, FAKE_CHANNELS)
    assert controller.command(0.0, {"car": state})["drive.steer"] == 0.0


@pytest.mark.parametrize("heading_deg", [-78.0, 90.0])
def test_rotated_drive_in_reaches_its_frame0_plane(tmp_path, heading_deg) -> None:
    case = families.FAMILIES["barrier_crash"].default_case()
    actor = dataclasses.replace(case.actor("car"), frame0_position_m=(7.0, -3.0, 0.725),
                                frame0_yaw_deg=heading_deg, speed_mps=8.0)
    # Isolate the controller from contact physics: no obstacles, unchanged driver channels.
    case = dataclasses.replace(case, actors=(actor,), statics=(), targets=(), clip_s=0.3)
    run = synth.simulate_case(case, tmp_path / "rotated", physics_dt_s=0.004)
    assert run.valid and run.claim_class == "test_double"
    assert run.t0_s > 0.0
    centre = run.frame_tracks["car"].centre()[0]
    assert centre[:2] == pytest.approx(actor.frame0_position_m[:2], abs=0.03)
