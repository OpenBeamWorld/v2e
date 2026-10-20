# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""The v2e-synth-1 runner: oracle round trip, recreation from a world, targets and the CLI."""

from __future__ import annotations

import json

import numpy as np
import pytest

from openbeam_v2e.bench import cli, families, runner, synth

TINY = synth.RenderSettings(width=96, height=54, noise_sigma=0.0, prompts=2)


def test_targets_are_checked_against_run_metrics() -> None:
    case = families.FAMILIES["barrier_crash"].default_case()
    v = families.kmh(case.params["speed_kmh"])
    met, total, rows = runner.evaluate_targets(case, {"car": {"speed_at_event": v + 0.1, "yaw_change": 20.0}},
                                               {"impact": 1.0 + 1.0 / 60.0})
    assert total == 3 and met == 2
    assert {r["metric"]: r["met"] for r in rows} == {"speed_at_event": True, "time_of_event": True,
                                                     "yaw_change_after_event": False}
    roll = families.FAMILIES["rollover"].build({"speed_kmh": 60.0}, "tripod")
    allowed = next(t.value for t in roll.targets if t.metric == "roll_onset_allowed")
    met, _, _ = runner.evaluate_targets(roll, {"car": {"speed_at_event": families.kmh(60.0), "rolled_over": 1.0}})
    assert met == (2 if allowed == 1.0 else 1)


def test_unknown_connector_and_source_are_refused(tmp_path) -> None:
    with pytest.raises(ValueError):
        runner.run_bench(tmp_path, connector="someengine")
    with pytest.raises(ValueError):
        runner.run_bench(tmp_path, reconstructions=("psychic",))


def test_cli_lists_cases(capsys) -> None:
    assert cli.main(["synth", "list"]) == 0
    out = capsys.readouterr().out
    for name in families.family_names():
        assert name in out
    assert cli.main(["synth", "list", "--subset", "full", "--json", "--family", "rollover"]) == 0
    docs = json.loads(capsys.readouterr().out)
    assert len(docs) == 6 and all(d["family"] == "rollover" for d in docs)


@pytest.mark.slow
def test_oracle_round_trip_on_the_fake_connector(tmp_path) -> None:
    report = runner.run_bench(tmp_path, families=["drop_and_bounce"], reconstructions=("truth",), settings=TINY,
                              physics_dt_s=0.004)
    (row,) = report.rows
    assert row.status == "ok", row.error
    assert row.values["recon"] == "truth"
    assert row.values["cam_rms_px"] == pytest.approx(0.0, abs=1e-6)
    assert row.values["dims_err_cm"] == pytest.approx(0.0, abs=1e-9)
    assert row.values["claim"] == "test_double"
    # Same content, same frame-0 state, deterministic connector: the recreation repeats the truth.
    assert row.values["run_traj_rms_m"] < 0.02
    assert row.values["silhouette_iou"] > 0.9
    for ext in ("md", "csv", "json"):
        assert (tmp_path / f"report.{ext}").is_file()
    doc = json.loads((tmp_path / "report.json").read_text())
    assert doc["label"] == families.LABEL
    # A second run reuses the generated case.
    again = runner.run_bench(tmp_path, families=["drop_and_bounce"], reconstructions=("truth",), settings=TINY,
                             physics_dt_s=0.004)
    assert again.rows[0].values["run_traj_rms_m"] == pytest.approx(row.values["run_traj_rms_m"])


@pytest.mark.slow
def test_case_from_world_recovers_the_case(tmp_path) -> None:
    from openbeam_v2e.world import read_world

    case = families.FAMILIES["barrier_crash"].default_case()
    g = synth.generate_case(case, tmp_path, TINY, physics_dt_s=0.004)
    world = read_world(g.directory / "world")
    rebuilt = runner.case_from_world(case, world)
    car, car2 = case.actor("car"), rebuilt.actor("car")
    assert car2.dims_m == pytest.approx(car.dims_m)
    assert np.allclose(car2.frame0_position_m, car.frame0_position_m, atol=0.05)
    assert car2.speed_mps == pytest.approx(car.speed_mps, abs=0.3)


@pytest.mark.slow
def test_case_from_world_reads_cylinders_in_either_body_frame(tmp_path) -> None:
    import dataclasses

    from scipy.spatial.transform import Rotation

    from openbeam_v2e.world import read_world

    case = families.FAMILIES["drop_and_bounce"].default_case()
    g = synth.generate_case(case, tmp_path, TINY, physics_dt_s=0.004)
    world = read_world(g.directory / "world")
    direct = runner.case_from_world(case, world).actor("cylinder")
    # Re-express the cylinder with its axis on body z, as the reconstruction hulls do.
    obj = world.object("cylinder")
    M = Rotation.from_euler("y", -90.0, degrees=True).as_matrix()   # connector body -> object frame
    R = obj.rotations() @ M.T
    turned = dataclasses.replace(obj, hull_vertices=obj.hull_vertices @ M.T, quat_xyzw=Rotation.from_matrix(R).as_quat())
    world.objects = [turned if o.id == "cylinder" else o for o in world.objects]
    via_z = runner.case_from_world(case, world).actor("cylinder")
    assert via_z.dims_m == pytest.approx(direct.dims_m, abs=1e-6)
    assert direct.dims_m == pytest.approx({"length": 0.5, "diameter": 0.3}, abs=0.02)
    assert np.allclose(via_z.frame0_position_m, direct.frame0_position_m, atol=1e-6)
    Ra = Rotation.from_euler("ZYX", [direct.frame0_yaw_deg, direct.frame0_pitch_deg, direct.frame0_roll_deg], degrees=True)
    Rb = Rotation.from_euler("ZYX", [via_z.frame0_yaw_deg, via_z.frame0_pitch_deg, via_z.frame0_roll_deg], degrees=True)
    assert (Ra.inv() * Rb).magnitude() < 1e-6


def test_agent_step_entry_point_is_the_benchmark() -> None:
    import inspect

    from openbeam_v2e.agents import steps

    fn = steps.resolve("synth")
    assert fn in (runner.run_bench, synth.run)
    params = inspect.signature(fn).parameters
    assert {"out", "connector", "subset", "families"} <= set(params)
    # The tool serialises whatever the entry point returns through its to_json().
    assert callable(getattr(report_type(), "to_json", None))
    with pytest.raises(ValueError):
        fn("unused-out", connector="someengine")


def report_type():
    from openbeam_v2e.bench.report import BenchReport

    return BenchReport("v2e-synth-1", families.LABEL, {})


def test_scene_targets_use_the_scorers_metric_names() -> None:
    from openbeam_v2e.score.scorer import TARGET_METRICS

    bench_only = {"apex_rise", "airborne_time", "landing_distance", "roll_onset_allowed"}
    for case in families.cases("full", cameras=("tripod",)):
        for target in case.scene_document()["targets"]:
            assert target["metric"] in set(TARGET_METRICS) | bench_only, target["metric"]


@pytest.mark.parametrize("events", [None, {}, {"impact": None}])
def test_missing_target_event_does_not_fall_back_to_actor_metric(events) -> None:
    case = families.FAMILIES["barrier_crash"].default_case()
    v = families.kmh(case.params["speed_kmh"])
    # A stale actor-level value must not fabricate an observation of the named event.
    met, total, rows = runner.evaluate_targets(
        case, {"car": {"speed_at_event": v, "event_time": 1.0, "time_of_event": 1.0, "yaw_change": 0.0}}, events)
    assert (met, total) == (2, 3)
    timing = next(r for r in rows if r["metric"] == "time_of_event")
    assert timing["event"] == "impact"
    assert timing["run"] is None and timing["met"] is None


def test_drop_targets_use_each_props_named_contact_time() -> None:
    case = families.FAMILIES["drop_and_bounce"].default_case()
    timing = {t.actor: t for t in case.targets if t.metric == "time_of_event"}
    met, total, rows = runner.evaluate_targets(case, {}, {
        "first_contact.box": timing["box"].value,
        "first_contact.ball": timing["ball"].value + 1.0,
        "first_contact.cylinder": None,
    })
    assert (met, total) == (1, 3)
    assert {r["actor"]: r["met"] for r in rows} == {"box": True, "ball": False, "cylinder": None}
