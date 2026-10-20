# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""The v2e-synth-1 scene families: grid, documents, sources and first-principles targets."""

from __future__ import annotations

import json
import math

import pytest

from openbeam_v2e.bench import families as fam


def test_four_families_with_distinct_events() -> None:
    assert set(fam.family_names()) == {"barrier_crash", "ramp_jump", "rollover", "drop_and_bounce"}
    events = {name: fam.FAMILIES[name].default_case().events for name in fam.family_names()}
    assert len({tuple(e) for e in events.values()}) == 4


def test_default_subset_has_one_case_per_family() -> None:
    cases = fam.cases()
    assert [c.family for c in cases] == list(fam.family_names())
    assert all(c.camera.name == "tripod" for c in cases)


def test_full_grid_size_and_unique_ids() -> None:
    cases = fam.cases("full")
    # 6 barrier + 6 ramp + 3 rollover + 5 drop/incline grid points, each with two cameras.
    assert len(cases) == (6 + 6 + 3 + 5) * 2
    ids = [c.case_id for c in cases]
    assert len(ids) == len(set(ids))
    assert fam.get_case(ids[7]).case_id == ids[7]


def test_unknown_inputs_are_refused() -> None:
    with pytest.raises(KeyError):
        fam.cases(families=["skydive"])
    with pytest.raises(ValueError):
        fam.cases(subset="most")
    with pytest.raises(KeyError):
        fam.get_case("nope")


@pytest.mark.parametrize("case", fam.cases("full"), ids=lambda c: c.case_id)
def test_case_documents_are_json_and_sourced(case: fam.Case) -> None:
    doc = case.document()
    json.dumps(doc)
    for target in case.targets:
        assert target.source in fam.PROVENANCE
        if target.value is None:
            assert target.source == "reported"
        else:
            assert target.source != "reported"
            assert math.isfinite(target.value)
    # Initial states never carry a velocity vector; drive_in carries a scalar speed only.
    for actor in doc["scene"]["actors"]:
        state = actor.get("initial_state", {})
        assert set(state) <= {"mode", "pose", "frame0_pose", "speed_mps"}
    assert case.frames == round(case.clip_s * case.fps)


def test_drop_contact_time_is_free_fall() -> None:
    case = fam.FAMILIES["drop_and_bounce"].default_case()
    h = case.params["drop_height_m"]
    for target in case.targets:
        if target.metric == "time_of_event":
            assert target.value == pytest.approx(math.sqrt(2 * h / fam.GRAVITY_MPS2))
            assert target.source == "first_principles"


def test_ramp_prediction_matches_projectile_algebra() -> None:
    v, a, h0 = 15.0, 20.0, 1.2
    pred = fam.ramp_jump_prediction(v, a, h0)
    vz = v * math.sin(math.radians(a))
    t = pred["airborne_s"]
    assert h0 + vz * t - 0.5 * fam.GRAVITY_MPS2 * t * t == pytest.approx(0.0, abs=1e-9)
    assert pred["apex_above_edge_m"] == pytest.approx(vz * vz / (2 * fam.GRAVITY_MPS2))


def test_rollover_bound_orders_the_speed_grid() -> None:
    gv = fam.GENERIC_VEHICLE
    _, rise = fam.profile_of(fam.TRIP_RAMP["segments"])
    run_eq = rise / math.tan(math.radians(fam.TRIP_RAMP["segments"][-1][1]))
    bound = fam.rollover_energy_bound(gv["track_m"], gv["com_height_m"], rise, run_eq, gv["width_m"], gv["height_m"])
    possible = {c.params["speed_kmh"]: next(t.value for t in c.targets if t.metric == "roll_onset_allowed")
                for c in fam.FAMILIES["rollover"].cases(("tripod",))}
    assert possible == {s: (1.0 if fam.kmh(s) >= bound["speed_min_mps"] else 0.0) for s in possible}
    # A wider track or a lower centre of mass needs more speed to tip.
    wider = fam.rollover_energy_bound(gv["track_m"] * 1.2, gv["com_height_m"], rise, run_eq, gv["width_m"],
                                      gv["height_m"])
    assert wider["speed_min_mps"] > bound["speed_min_mps"]


def test_barrier_overlap_offsets_the_vehicle() -> None:
    full = fam.FAMILIES["barrier_crash"].build({"speed_kmh": 40.0, "overlap": 1.0}, "tripod")
    part = fam.FAMILIES["barrier_crash"].build({"speed_kmh": 40.0, "overlap": 0.4}, "tripod")
    y_full = full.actor("car").frame0_position_m[1]
    y_part = part.actor("car").frame0_position_m[1]
    assert y_full == 0.0
    half_barrier = fam.BARRIER["width_m"] / 2
    width = fam.GENERIC_VEHICLE["width_m"]
    # The overlapping width is the barrier edge minus the vehicle's right side.
    assert half_barrier - (y_part - width / 2) == pytest.approx(0.4 * width)


def test_handheld_preset_keeps_position() -> None:
    tri = fam.FAMILIES["ramp_jump"].default_case("tripod").camera
    ptz = fam.FAMILIES["ramp_jump"].default_case("handheld_ptz").camera
    assert ptz.position_m == tri.position_m
    assert ptz.k1 != 0.0 and ptz.zoom_end > 1.0
