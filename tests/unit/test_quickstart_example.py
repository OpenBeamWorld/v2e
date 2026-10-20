# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""The quickstart example runs as its README says and meets its first-principles targets."""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

from openbeam_v2e.agents.tools import AgentContext, call_tool

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "quickstart_fake"


def test_quickstart_meets_its_targets(tmp_path: Path) -> None:
    shutil.copytree(EXAMPLE, tmp_path / "quickstart")
    targets = json.loads((EXAMPLE / "targets.json").read_text())
    ctx = AgentContext(root=tmp_path)
    run = call_tool(ctx, "run", {"scene": "quickstart/scene.json", "programs": "quickstart/programs.json",
                                 "out": "runs/q", "duration_s": 6.0, "telemetry_hz": 200.0})
    assert run["ok"], run
    assert run["valid"] and run["audit"] == "clean"
    assert run["claim_class"] == targets["expected_claim_class"]
    for target in targets["targets"]:
        assert abs(run["events"][target["event"]] - target["value_s"]) <= target["tolerance_s"], target
    rows = [json.loads(line) for line in (tmp_path / "runs/q/rows.jsonl").read_text().splitlines()[1:]]
    car = [r["actors"]["car_1"] for r in rows]
    x, y, z, w = car[-1]["q"]
    yaw = math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
    assert car[-1]["p"][0] > 5.0 and yaw < -5.0
    assert "rest.car_1" in run["events"]
    refused = call_tool(ctx, "inspect", {"path": "quickstart/scene.json"})
    assert refused["connectors"]["fake"]["ok"]
