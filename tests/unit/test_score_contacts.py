# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""Sampled contact geometry with closed-form gaps, not inferred poses or footage."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from openbeam_v2e.schema.scene import Scene
from openbeam_v2e.score import ScoreOptions, score_worlds
from openbeam_v2e.score import events
from openbeam_v2e.world import CameraTrack, EventInterval, ObjectTrack, World
from openbeam_v2e.world.geometry import box_mesh, cylinder_mesh, sphere_mesh

from world_fixtures import camera, ground


def _world(gap, shape="box", fps=30.0, moving_support=False) -> World:
    gap = np.asarray(gap, dtype=float)
    F = len(gap)
    if shape == "sphere":
        vertices, faces = sphere_mesh(0.2, rings=4, segments=8)
    elif shape == "cylinder":
        vertices, faces = cylinder_mesh(0.2, 0.4, segments=8)
    else:
        vertices, faces = box_mesh(0.4, 0.3, 0.2)
    position = np.column_stack([np.zeros(F), np.zeros(F), gap - vertices[:, 2].min()])
    quat = np.tile([0.0, 0.0, 0.0, 1.0], (F, 1))
    obj = ObjectTrack("body", f"prop.{shape}", vertices, faces, position, quat.copy())
    if moving_support:
        shift = np.arange(F)[:, None] * np.array([0.07, -0.04, 0.02])
        obj.position += shift
        sv, sf = box_mesh(3.0, 3.0, 0.4, centre=(0.0, 0.0, -0.2))
        support = ObjectTrack("support", "prop.box", sv, sf, shift, quat.copy())
        objects, statics = [obj, support], []
    else:
        objects, statics = [obj], [ground()]
    return World(fps, F, CameraTrack.static(camera(), F), statics, objects)


def _series(world, options=events.ContactOptions()):
    other = "support" if world.has_object("support") else "static:1"
    return events.contact_series(world, options)[("body", other)]


@pytest.mark.parametrize("shape", ["box", "sphere", "cylinder"])
@pytest.mark.parametrize("fps,speed,offset", [
    (24.0, 9.0, 0.2), (30.0, 12.0, -0.3), (60.0, 27.0, 0.35),
    (24.0, 15.0, -0.2), (48.0, 23.0, 0.37), (60.0, 19.0, -0.35),
])
def test_fast_rebound_matches_full_gap_at_different_sampling_offsets(shape, fps, speed, offset):
    # Piecewise-linear approach/rebound; the exact sampled gap is known independently.
    delta = (np.arange(9) - (4.0 + offset)) / fps
    gap = speed * np.where(delta < 0.0, -delta, 0.7 * delta)
    expected = events.contact_mask(gap, 0.03, 0.25)
    assert gap.min() > 0.03 and expected[np.argmin(gap)]
    actual = _series(_world(gap, shape, fps))
    np.testing.assert_array_equal(actual, expected)


def test_moving_body_pair_uses_relative_geometry_and_keeps_actor_pair():
    gap = [1.0, 0.55, 0.08, 0.35, 0.8]
    world = _world(gap, moving_support=True)
    np.testing.assert_array_equal(_series(world), events.contact_mask(gap, 0.03, 0.25))
    hits = [e for e in events.detect_events(world) if e.kind == "contact"]
    assert hits == [events.DetectedEvent("contact", 2, ("body", "support"))]


def test_contact_geometry_is_rigid_frame_invariant():
    world = _world([1.0, 0.55, 0.08, 0.35, 0.8])
    R = Rotation.from_euler("xyz", [28.0, -19.0, 63.0], degrees=True).as_matrix()
    shift = np.array([2.0, -3.0, 4.0])
    obj = world.object("body")
    obj.position = obj.position @ R.T + shift
    obj.quat_xyzw[:] = Rotation.from_matrix(R).as_quat()
    world.statics = [replace(s, vertices=s.vertices @ R.T + shift) for s in world.statics]
    assert _series(world).tolist() == [False, False, True, False, False]


@pytest.mark.parametrize("missing_frame", [1, 3])
@pytest.mark.parametrize("missing_body", ["body", "support"])
def test_unknown_neighbour_is_not_replaced_with_a_distance(missing_frame, missing_body):
    world = _world([1.0, 0.55, 0.08, 0.35, 0.8], moving_support=True)
    world.object(missing_body).position[missing_frame] = np.nan
    assert not _series(world).any()


@pytest.mark.parametrize("gap", [
    [0.6, 0.4, 0.22, 0.25, 0.5],  # finite exact steps do not justify a contact
    [0.9, 0.6, 0.3, 0.6, 0.9],    # local minimum outside the existing cap
    [0.5, 0.15, 0.12, 0.15, 0.5], # slowly varying near miss
    [0.1] * 5,                    # hovering, not touching
    [0.08, 0.35, 0.8],            # no two-sided evidence at a clip boundary
    [0.8, 0.35, 0.08],
])
def test_censored_distances_do_not_create_unjustified_contacts(gap):
    assert not events.contact_mask(np.asarray(gap), 0.03, 0.25).any()
    assert not _series(_world(gap)).any()


def test_only_censored_neighbours_of_candidates_need_full_queries(monkeypatch):
    original = events._pair_distance
    full_queries = []

    def counted(a, af, b, bf, tol, b_bounds=None):
        if tol is None:
            full_queries.append(float(a[:, 2].min()))
        return original(a, af, b, bf, tol, b_bounds)

    monkeypatch.setattr(events, "_pair_distance", counted)
    assert _series(_world([1.0, 0.55, 0.08, 0.35, 0.8]))[2]
    assert sorted(full_queries) == pytest.approx([0.35, 0.55])
    full_queries.clear()
    assert not _series(_world([10.0, 9.0, 8.0, 7.0, 6.0])).any()
    assert not full_queries  # retain the broad-phase shortcut for entirely distant pairs


def test_contact_tolerance_can_exceed_between_frame_cap():
    options = events.ContactOptions(tolerance_m=0.09, between_frames_cap_m=0.04)
    assert _series(_world([0.5, 0.08, 0.5]), options).tolist() == [False, True, False]


def test_initial_support_is_not_a_new_contact_but_later_rebound_is():
    world = _world([0.0, 0.0, 0.5, 0.35, 0.045, 0.25, 0.6])
    hits = [e.frame for e in events.detect_events(world) if e.kind == "contact"]
    assert hits == [4]


def test_named_contacts_stay_distinct_and_missing_event_target_stays_unmeasured():
    world = _world([1.0, 0.6, 0.3, 0.05, 0.15, 0.5, 0.9, 0.6, 0.2, 0.02, 0.2, 0.5, 1.0])
    scene = Scene.from_dict({
        "schema": "v2e.scene/1", "world": "world/",
        "actors": [
            {"id": "body", "class": "prop.box", "control": "passive",
             "initial_state": {"mode": "at_rest", "pose": "track:body@0"}},
            {"id": "floor", "class": "static", "mesh": 1},
            {"id": "absent_wall", "class": "static", "mesh": 2},
        ],
        "events": [
            {"name": name, "kind": "contact", "frame": [frame, frame], "actors": ["body", other]}
            for name, frame, other in [("first_hit", 3, "floor"), ("second_hit", 9, "floor"),
                                       ("wall_hit", 3, "absent_wall")]
        ] + [{"name": "missing", "kind": "custom", "frame": [0, 0], "actors": ["body"]}],
        "targets": [
            {"metric": "time_of_event", "event": name, "value": [frame / 30.0, 0.01],
             "source": "design_value"}
            for name, frame in [("first_hit", 3), ("second_hit", 9), ("wall_hit", 3), ("missing", 0)]
        ],
    })
    report = score_worlds(world, world, scene, options=ScoreOptions(silhouettes=False))
    assert {e["name"]: e["run_frame"] for e in report["events"]} == {
        "first_hit": 3.0, "second_hit": 9.0, "wall_hit": None, "missing": None}
    targets = {t["event"]: t for t in report["targets"]}
    assert targets["first_hit"]["pass"] and targets["second_hit"]["pass"]
    for name in ("wall_hit", "missing"):
        assert targets[name]["run"] is None and not targets[name]["pass"]
    # An explicitly reported event remains authoritative by name, not overwritten by geometry.
    run = replace(world, events=[EventInterval("first_hit", (5, 5), "contact", ("body", "static:1"))])
    report = score_worlds(world, run, scene, options=ScoreOptions(silhouettes=False))
    assert {e["name"]: e["run_frame"] for e in report["events"]} == {
        "first_hit": 5.0, "second_hit": 9.0, "wall_hit": None, "missing": None}
