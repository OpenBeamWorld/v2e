# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""The connector contract: what every connector must do, run against each built-in connector.

The fake runs in process; OpenBeam runs from a recorded cassette (no network). A new connector
adds a case to ``CASES`` with a scene and compiled content it can run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "unit"))
from connector_fixtures import drop_scene, openbeam_compiled, openbeam_passive_cassette, openbeam_scene  # noqa: E402

from openbeam_v2e.backends import registry  # noqa: E402
from openbeam_v2e.backends.base import (  # noqa: E402
    CLAIM_CLASSES,
    Backend,
    ChannelKind,
    Lockstep,
    RunSpec,
)
from openbeam_v2e.backends.fake import FakeBackend, compile_scene  # noqa: E402
from openbeam_v2e.backends.harness import audit_run, execute, negotiate  # noqa: E402
from openbeam_v2e.backends.openbeam.backend import OpenBeamBackend  # noqa: E402
from openbeam_v2e.backends.openbeam.cassette import ReplayClient  # noqa: E402


def _fake_case() -> tuple[Backend, dict, Any]:
    scene = drop_scene(0.5)
    return FakeBackend(), scene, compile_scene(scene)


def _openbeam_case() -> tuple[Backend, dict, Any]:
    backend = OpenBeamBackend(ReplayClient(openbeam_passive_cassette()), yaw_convention={"units": "deg"})
    return backend, openbeam_scene(), openbeam_compiled()


CASES: dict[str, Callable[[], tuple[Backend, dict, Any]]] = {"fake": _fake_case, "openbeam": _openbeam_case}


def test_every_builtin_connector_has_a_contract_case() -> None:
    assert set(CASES) == set(registry.BUILTIN)


@pytest.mark.parametrize("name", sorted(CASES))
def test_manifest_is_complete_and_honest(name: str) -> None:
    backend, _, _ = CASES[name]()
    manifest = backend.manifest()
    caps = manifest.capabilities
    assert manifest.backend == name == backend.name
    assert manifest.kind in ("engine", "test_double") and manifest.engine and manifest.dialect
    for channel in caps.channels:
        lo, hi = channel.range
        assert lo < hi and channel.mechanism
        if channel.kind is ChannelKind.TOGGLE:
            assert (lo, hi) == (0.0, 1.0)
    assert caps.max_controlled_actors >= 0
    assert all(r.regime for r in caps.regimes)
    if any(h.value == "client_lockstep" for h in caps.controller_hosts):
        assert isinstance(backend, Lockstep)


@pytest.mark.parametrize("name", sorted(CASES))
def test_a_passive_scene_runs_validly_and_audits_clean(name: str, tmp_path: Path) -> None:
    backend, scene, compiled = CASES[name]()
    plan = negotiate(scene, backend.manifest(), lockstep=isinstance(backend, Lockstep))
    assert plan.ok, plan
    spec = RunSpec(duration_s=0.1, host=plan.host, regime=plan.regime, telemetry_hz=20.0)
    result = execute(backend, compiled, plan, spec, tmp_path / "run")
    assert result.valid(), result.invalid_reasons()
    assert result.claim_class() in CLAIM_CLASSES
    assert result.attestation.initial_velocity_mps == 0.0
    assert {s.actor for s in result.attestation.spawns} == set(plan.actors)
    lines = Path(result.rows).read_text().splitlines()
    header = json.loads(lines[0])
    assert header["schema"] == "v2e-runlog/1" and header["frame"] == "z_up_level"
    assert set(header["actors"]) == set(plan.actors)
    times = [json.loads(line)["t"] for line in lines[1:]]
    assert times == sorted(times) and times[0] == 0.0 and times[-1] == pytest.approx(0.1)
    report = audit_run(tmp_path / "run")
    assert report["audit"] == "clean" and report["matches_run_record"]
    saved = json.loads((tmp_path / "run" / "run.json").read_text())
    assert saved["provenance_signature"] == result.provenance_signature()
