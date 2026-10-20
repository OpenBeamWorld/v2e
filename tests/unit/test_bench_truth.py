# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""Benchmark truth: geometric event detection and truth metrics on analytic tracks."""

from __future__ import annotations

import math

import numpy as np
import pytest

from openbeam_v2e.bench import truth
from openbeam_v2e.bench.report import BenchReport, BenchRow, format_value

G = 9.81


def box_vertices(lx: float, ly: float, lz: float, x0: float = 0.0, y0: float = 0.0, z0: float = 0.0) -> np.ndarray:
    """Box with its bottom-face centre at (x0, y0, z0)."""
    return np.array([[x0 + sx * lx / 2, y0 + sy * ly / 2, z0 + h] for sx in (-1, 1) for sy in (-1, 1) for h in (0.0, lz)])


def test_signed_distance_inside_and_outside() -> None:
    part = truth.StaticPart("slab", box_vertices(2.0, 2.0, 1.0))
    d = truth.signed_distance_to_part(np.array([[0, 0, 0.5], [0, 0, 1.25], [2.0, 0, 0.5]]), part.planes)
    assert d[0] == pytest.approx(-0.5)
    assert d[1] == pytest.approx(0.25)
    assert d[2] == pytest.approx(1.0)
    assert part.top_z == pytest.approx(1.0)


def _drop_track(h: float, slab_top: float, dt: float = 0.002, restitution: float = 0.4, t_end: float = 3.0) -> truth.Track:
    t = np.arange(0.0, t_end, dt)
    z = np.empty_like(t)
    zc, vz = slab_top + h, 0.0
    for i in range(len(t)):
        z[i] = zc
        vz -= G * dt
        zc += vz * dt
        if zc < slab_top:
            zc = slab_top
            vz = -vz * restitution if abs(vz) > 0.3 else 0.0
    pos = np.stack([np.zeros_like(t), np.zeros_like(t), z], axis=1)
    R = np.repeat(np.eye(3)[None], len(t), axis=0)
    return truth.Track("box", t, pos, R, (box_vertices(0.5, 0.5, 0.5),), "prop.box")


def test_drop_first_contact_matches_free_fall() -> None:
    slab = truth.StaticPart("slab", box_vertices(6.0, 4.0, 0.3))
    track = _drop_track(2.0, 0.3)
    events = truth.detect_events("drop_and_bounce", {"box": track}, [slab], contact_tol_m=0.0)
    t_contact = events.get("first_contact.box")
    assert t_contact == pytest.approx(math.sqrt(2 * 2.0 / G), abs=0.01)
    assert events.kinds["first_contact.box"] == "contact"
    t_rest = events.get("rest.box")
    assert t_rest is not None and t_rest > t_contact
    metrics = truth.truth_metrics("drop_and_bounce", {"box": track}, events)
    assert 0.1 < metrics["box"]["rebound_ratio"] < 0.25  # restitution 0.4 -> height ratio 0.16


def test_barrier_impact_time() -> None:
    barrier = truth.StaticPart("barrier", box_vertices(0.8, 4.0, 1.0, x0=0.4))
    v, length = 10.0, 4.0
    t = np.arange(0.0, 2.0, 0.002)
    x = -length / 2 - v * 1.0 + v * t
    x = np.minimum(x, -length / 2)  # stops dead at the face
    pos = np.stack([x, np.zeros_like(t), np.zeros_like(t)], axis=1)
    R = np.repeat(np.eye(3)[None], len(t), axis=0)
    car = truth.Track("car", t, pos, R, (box_vertices(length, 1.8, 1.4),), "vehicle.car")
    events = truth.detect_events("barrier_crash", {"car": car}, [barrier], contact_tol_m=0.03)
    assert events.get("impact") == pytest.approx(1.0 - 0.03 / v, abs=0.003)
    assert events.get("rest") == pytest.approx(1.0, abs=0.01)
    metrics = truth.truth_metrics("barrier_crash", {"car": car}, events)
    assert metrics["car"]["speed_at_event"] == pytest.approx(v, rel=0.02)
    assert metrics["car"]["yaw_change"] == pytest.approx(0.0, abs=1e-9)


def test_event_frames_bracket_times() -> None:
    ev = truth.EventTimes({"a": 1.0, "b": 1.0 + 1.0 / 90.0, "c": None, "d": 99.0}, {}, {})
    frames = truth.event_frames(ev, fps=30.0, n_frames=60)
    assert frames["a"] == (30, 30)
    assert frames["b"] == (30, 31)
    assert frames["c"] is None and frames["d"] is None


def test_cumulative_roll_counts_full_turns() -> None:
    t = np.linspace(0.0, 1.0, 401)
    angles = np.radians(450.0) * t
    R = np.array([[[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]] for a in angles])
    track = truth.Track("car", t, np.zeros((len(t), 3)), R, (box_vertices(4.0, 1.8, 1.4),))
    assert track.cumulative_roll_deg()[-1] == pytest.approx(450.0, abs=0.5)


def test_report_formats_missing_values() -> None:
    report = BenchReport("v2e-synth-1", "synthetic self-consistency", {"backend": "fake"})
    report.add(BenchRow("case-a", "drop_and_bounce", values={"cam_rms_px": 0.5, "claim": "test_double"}))
    report.add(BenchRow("case-b", "rollover", status="failed:reconstruct", error="no landmarks"))
    md = report.markdown()
    assert "| case-a | - | ok | 0.50 |" in md
    assert "failed:reconstruct" in md and "no landmarks" in md
    assert "not accuracy on real footage" in md
    assert format_value(float("nan")) == "-"
    assert report.csv().splitlines()[0].startswith("case,recon,status,")
    assert not report.ok
