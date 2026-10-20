# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""The scene examples stay in step with the benchmark families, and their walkthrough runs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openbeam_v2e.bench import families
from openbeam_v2e.bench.walkthrough import walkthrough

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


@pytest.mark.parametrize("family", families.family_names())
def test_example_documents_match_the_family_default_case(family: str) -> None:
    folder = EXAMPLES / family
    case = families.FAMILIES[family].default_case()
    expected = {"case.json": case.document(), "templates.json": case.templates_document(),
                "targets.json": case.targets_document()}
    for name, doc in expected.items():
        on_disk = json.loads((folder / name).read_text(encoding="utf-8"))
        assert on_disk == json.loads(json.dumps(doc)), (
            f"examples/{family}/{name} is stale; regenerate it from openbeam_v2e.bench.families")
    script = (folder / "walkthrough.py").read_text(encoding="utf-8")
    assert f'main("{family}"' in script
    readme = (folder / "README.md").read_text(encoding="utf-8")
    assert f"examples/{family}/walkthrough.py" in readme


@pytest.mark.slow
def test_walkthrough_runs_offline(tmp_path) -> None:
    lines: list[str] = []
    summary = walkthrough("drop_and_bounce", tmp_path, width=64, height=36, reconstruction="truth",
                          physics_dt_s=0.004, echo=lines.append)
    assert summary["reconstruction"] == "truth"
    assert summary["run_valid"] and summary["claim_class"] == "test_double"
    assert summary["targets_total"] == 3
    assert any(line.startswith("== 7.") for line in lines)
    assert (tmp_path / f"{summary['case']}.summary.json").is_file()
