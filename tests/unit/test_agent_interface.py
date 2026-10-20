# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""The agent surface: JSON-schema tools, the stdio MCP server and the CLI, end to end on the fake."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from openbeam_v2e import cli
from openbeam_v2e.agents import steps
from openbeam_v2e.agents.mcp_server import SUPPORTED_PROTOCOLS, McpServer
from openbeam_v2e.agents.tools import TOOLS, AgentContext, call_tool, tool_specs, validate_arguments

ROOT = Path(__file__).resolve().parents[2]
QUICKSTART = ROOT / "examples" / "quickstart_fake"


@pytest.fixture()
def workspace(tmp_path: Path) -> Path:
    shutil.copytree(QUICKSTART, tmp_path / "quickstart")
    return tmp_path


def _run_args(**extra) -> dict:
    return {"scene": "quickstart/scene.json", "programs": "quickstart/programs.json", "out": "runs/a",
            "duration_s": 1.5, **extra}


def test_tool_specs_are_well_formed() -> None:
    names = [spec["name"] for spec in tool_specs()]
    assert names == ["reconstruct", "compile", "run", "export", "score", "inspect", "bench"]
    for spec in tool_specs():
        schema = spec["inputSchema"]
        assert schema["type"] == "object" and schema["additionalProperties"] is False
        assert set(schema.get("required", [])) <= set(schema["properties"])
        assert spec["description"] and spec["annotations"]["destructiveHint"] is False
    assert {t.name for t in TOOLS if t.read_only} == {"inspect"}


def test_argument_validation() -> None:
    schema = next(t.input_schema for t in TOOLS if t.name == "run")
    assert validate_arguments(schema, _run_args()) == []
    problems = validate_arguments(schema, {"duration_s": -1, "teleport": True, "record": "video"})
    assert any("/out: required" in p for p in problems)
    assert any("unknown argument" in p for p in problems)
    assert any("duration_s" in p for p in problems) and any("record" in p for p in problems)


def test_run_inspect_and_bench_on_the_fake(workspace: Path) -> None:
    ctx = AgentContext(root=workspace)
    run = call_tool(ctx, "run", _run_args())
    assert run["ok"], run
    assert run["claim_class"] == "test_double" and run["valid"] and run["audit"] == "clean"
    assert run["run"] == "runs/a" and run["row_count"] > 10
    inspected = call_tool(ctx, "inspect", {"path": "runs/a"})
    assert inspected["kind"] == "run" and inspected["audit_replay"]["valid"]
    scene = call_tool(ctx, "inspect", {"path": "quickstart/scene.json"})
    assert scene["kind"] == "scene" and scene["connectors"]["fake"]["ok"]
    overview = call_tool(ctx, "inspect", {})
    assert overview["steps"]["run"] and "openbeam" in overview["connectors"]
    connector = call_tool(ctx, "inspect", {"connector": "openbeam"})
    assert connector["configured"] is False and "mcp/sim_step" in connector["allowed_calls"]
    bench = call_tool(ctx, "bench", {"kind": "repeat", "scene": "quickstart/scene.json",
                                     "programs": "quickstart/programs.json", "out": "bench", "n": 2,
                                     "duration_s": 0.5})
    assert bench["ok"], bench
    assert bench["report"]["repeatability"] == "bit_identical"


def test_tools_refuse_honestly(workspace: Path) -> None:
    ctx = AgentContext(root=workspace)
    no_program = call_tool(ctx, "run", {"scene": "quickstart/scene.json", "out": "runs/x"})
    assert no_program["error"]["code"] == "refused"
    assert any(r["code"] == "missing_program" for r in no_program["reasons"])
    escape = call_tool(ctx, "run", _run_args(scene="../outside.json"))
    assert escape["error"]["code"] == "bad_arguments" and "outside the workspace" in escape["error"]["message"]
    not_configured = call_tool(ctx, "run", _run_args(connector="openbeam", out="runs/y"))
    assert not_configured["error"]["code"] in ("connector_not_configured", "compile_first")
    used = call_tool(ctx, "run", _run_args())
    assert used["ok"]
    again = call_tool(ctx, "run", _run_args())
    assert again["error"]["code"] == "bad_arguments" and "new or empty" in again["error"]["message"]
    assert call_tool(ctx, "nope", {})["error"]["code"] == "unknown_tool"
    if steps.resolve("score") is None:
        assert call_tool(ctx, "score", {"run": "runs/a"})["error"]["code"] == "step_unavailable"


def test_mcp_server_protocol(workspace: Path) -> None:
    server = McpServer(AgentContext(root=workspace), log=open(os.devnull, "w"))
    init = server.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                          "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                                     "clientInfo": {"name": "test", "version": "0"}}})
    assert init["result"]["protocolVersion"] == "2025-06-18"
    assert init["result"]["capabilities"] == {"tools": {"listChanged": False}}
    assert server.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None
    listed = server.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    assert [t["name"] for t in listed["result"]["tools"]][:3] == ["reconstruct", "compile", "run"]
    call = server.handle({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                          "params": {"name": "run", "arguments": _run_args()}})
    result = call["result"]
    assert result["isError"] is False and result["structuredContent"]["claim_class"] == "test_double"
    assert json.loads(result["content"][0]["text"])["ok"] is True
    failed = server.handle({"jsonrpc": "2.0", "id": 4, "method": "tools/call",
                            "params": {"name": "run", "arguments": {"out": "runs/b"}}})
    assert failed["result"]["isError"] is True
    assert server.handle({"jsonrpc": "2.0", "id": 5, "method": "tools/call",
                          "params": {"name": "teleport"}})["error"]["code"] == -32602
    assert server.handle({"jsonrpc": "2.0", "id": 6, "method": "nope"})["error"]["code"] == -32601
    assert server.handle({"jsonrpc": "2.0", "id": 7, "method": "ping"})["result"] == {}
    old = McpServer(AgentContext(root=workspace), log=open(os.devnull, "w"))
    old.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2024-11-05"}})
    assert "outputSchema" not in old.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})["result"]["tools"][0]
    assert "2024-11-05" in SUPPORTED_PROTOCOLS


def test_mcp_server_over_stdio(workspace: Path) -> None:
    messages = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize",
         "params": {"protocolVersion": "2025-03-26", "capabilities": {}, "clientInfo": {"name": "t", "version": "0"}}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "inspect", "arguments": {}}},
    ]
    stdin = "".join(json.dumps(m) + "\n" for m in messages) + "not json\n"
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    proc = subprocess.run([sys.executable, "-m", "openbeam_v2e", "mcp", "--root", str(workspace)], input=stdin,
                          capture_output=True, text=True, env=env, timeout=60, check=True)
    answers = [json.loads(line) for line in proc.stdout.splitlines()]
    assert [a.get("id") for a in answers] == [1, 2, 3, None]
    assert answers[0]["result"]["protocolVersion"] == "2025-03-26"
    assert len(answers[1]["result"]["tools"]) == len(TOOLS)
    overview = json.loads(answers[2]["result"]["content"][0]["text"])
    assert overview["kind"] == "overview"
    assert answers[3]["error"]["code"] == -32700
    assert "serving" in proc.stderr


def test_cli_end_to_end(workspace: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.chdir(workspace)
    code = cli.main(["run", "--scene", "quickstart/scene.json", "--programs", "quickstart/programs.json",
                     "--out", "runs/cli", "--duration", "1.0"])
    out = json.loads(capsys.readouterr().out)
    assert code == 0 and out["valid"] and out["claim_class"] == "test_double"
    assert cli.main(["audit", "runs/cli"]) == 0
    assert json.loads(capsys.readouterr().out)["audit"] == "clean"
    assert cli.main(["inspect", "runs/cli"]) == 0
    assert json.loads(capsys.readouterr().out)["kind"] == "run"
    assert cli.main(["run", "--scene", "quickstart/scene.json", "--out", "runs/refused"]) == 1
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "refused"
    assert cli.main(["connectors", "list"]) == 0
    listed = json.loads(capsys.readouterr().out)["connectors"]
    assert set(listed) >= {"fake", "openbeam"} and listed["fake"]["configured"]
    assert cli.main(["compile", "quickstart/scene.json", "--out", "compiled"]) == 0
    compiled = json.loads(capsys.readouterr().out)
    assert compiled["connector"] == "fake" and compiled["spawns"] == ["car_1", "crate", "ball"]
    assert cli.main(["run", "--compiled", "compiled", "--programs", "quickstart/programs.json", "--out", "runs/c",
                     "--duration", "0.5"]) == 0
    assert json.loads(capsys.readouterr().out)["valid"]


def test_cli_rejects_a_bad_connector_config(workspace: Path, monkeypatch: pytest.MonkeyPatch,
                                            capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.chdir(workspace)
    (workspace / "bad.json").write_text("[1, 2]")
    assert cli.main(["inspect", "--connector-config", "bad.json"]) == 2
    assert "connector-config" in capsys.readouterr().err
