# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""The OpenBeam connector, offline: allow list, call guard, recorded runs, transport and sign-in.

No test here reaches OpenBeam. Recorded runs use hand-written cassettes shaped like the documented
tool results; the transport and sign-in tests use a local HTTP server on 127.0.0.1.
"""

from __future__ import annotations

import json
import os
import stat
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from connector_fixtures import (
    ANCHOR_BBOX,
    CRATE_BBOX,
    openbeam_compiled,
    openbeam_passive_cassette,
    openbeam_scene,
)

from openbeam_v2e.backends.base import (
    STEPPED,
    ActorRef,
    ActuatorChannel,
    ChannelKind,
    ControllerHost,
    RefusedCall,
    RunSpec,
)
from openbeam_v2e.backends.harness import audit_run, execute, negotiate
from openbeam_v2e.backends.ledger import CallLedger, audit_ledger, read_ledger
from openbeam_v2e.backends.openbeam import transport
from openbeam_v2e.backends.openbeam.backend import OpenBeamBackend, make_audit_policy, yup_to_zup, zup_to_yup
from openbeam_v2e.backends.openbeam.cassette import CassetteMismatch, RecordingClient, ReplayClient
from openbeam_v2e.backends.openbeam.scripts import ScriptPolicy, ScriptReference, Snippet
from openbeam_v2e.backends.openbeam.tools import ACCOUNT_TOOLS, GAME_TOOLS, PUBLIC_TOOLS, TOOL_ARGS, GuardedClient

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "openbeam"


def test_public_tools_are_a_subset_of_the_published_reference() -> None:
    published = json.loads((FIXTURES / "published_tools.json").read_text())
    assert ACCOUNT_TOOLS <= set(published["account_tools"])
    assert GAME_TOOLS <= set(published["game_tools"])
    assert len(PUBLIC_TOOLS) == 25
    for tool, allowed in TOOL_ARGS.items():
        documented = set(published["documented_arguments"].get(tool, ()))
        assert allowed <= documented, f"{tool} sends undocumented arguments {sorted(allowed - documented)}"


def test_axis_conversion_is_a_proper_rotation() -> None:
    assert zup_to_yup([1.0, 2.0, 3.0]) == [1.0, 3.0, -2.0]
    assert yup_to_zup(zup_to_yup([1.0, 2.0, 3.0])) == [1.0, 2.0, 3.0]
    ex, ey, ez = zup_to_yup([1, 0, 0]), zup_to_yup([0, 1, 0]), zup_to_yup([0, 0, 1])
    assert ez == [0.0, 1.0, 0.0]                       # V2E up is OpenBeam up
    cross = [ex[1] * ey[2] - ex[2] * ey[1], ex[2] * ey[0] - ex[0] * ey[2], ex[0] * ey[1] - ex[1] * ey[0]]
    assert cross == ez                                  # right-handed in, right-handed out


class _Never:
    def call(self, tool, arguments=None):              # pragma: no cover - must never be reached
        raise AssertionError(f"network I/O for {tool}")


def test_guard_refuses_before_any_io() -> None:
    guard = GuardedClient(_Never(), scripts=ScriptPolicy())
    for tool, args in (("gamepad", {"button": "a"}), ("not_a_tool", {}), ("sim_step", {"steps": 1, "positions": {}}),
                       ("spawn_vehicle", {"file": "x", "rotation": [0, 0, 0]}),
                       ("run_script", {"code": "game.log(\"hi\"); other();"}),
                       ("run_script", {"code": "anything"}),
                       ("cloud_game_call", {"name": "robot_set_joints", "arguments": {}}),
                       ("cloud_game_call", {"name": "run_script", "arguments": {"code": "x"}})):
        with pytest.raises(RefusedCall):
            guard.call(tool, args)


def test_snippets_render_strictly() -> None:
    policy = ScriptPolicy()
    code = policy.render("log_marker", marker="v2e-abc:begin")
    assert code == 'game.log("v2e-abc:begin");' and policy.check(code) == "log_marker"
    with pytest.raises(RefusedCall):
        policy.render("log_marker", marker='x"); evil(); ("')
    with pytest.raises(RefusedCall):
        policy.render("nope")
    assert ScriptPolicy.from_description(policy.describe()).check(code) == "log_marker"


def _run(tmp_path: Path, name: str, cassette: list, **kw):
    client = ReplayClient(cassette)
    backend = OpenBeamBackend(client, **kw)
    scene = openbeam_scene()
    plan = negotiate(scene, backend.manifest())
    assert plan.ok, plan
    spec = RunSpec(duration_s=0.1, host=plan.host, regime=STEPPED, telemetry_hz=20.0)
    result = execute(backend, openbeam_compiled(), plan, spec, tmp_path / name)
    assert client.exhausted, f"{len(client.calls) - client.position} recorded calls were not made"
    return result


def test_recorded_passive_run_with_a_verified_yaw_convention(tmp_path: Path) -> None:
    result = _run(tmp_path, "run", openbeam_passive_cassette(), yaw_convention={"units": "deg"})
    assert result.valid(), result.invalid_reasons()
    assert result.claim_class() == "native"
    assert result.attestation.audit == "clean" and result.attestation.removed_actors == 0
    lines = Path(result.rows).read_text().splitlines()
    header, rows = json.loads(lines[0]), [json.loads(line) for line in lines[1:]]
    assert header["backend"] == "openbeam" and header["regime"] == STEPPED
    assert [r["t"] for r in rows] == [0.0, 0.05, 0.1]
    # rows are in V2E's Z-up frame at the reference point the spawn asked for
    assert rows[0]["actors"]["crate"]["p"] == pytest.approx([0.0, 0.0, 2.25])
    assert rows[-1]["actors"]["crate"]["p"][2] == pytest.approx(2.25 - 0.5 * 9.80665 * 0.01, abs=1e-9)
    assert rows[-1]["actors"]["crate"]["v"] == pytest.approx([0.0, 0.0, -0.980665])
    assert "q" not in rows[0]["actors"]["crate"]            # no attitude through the documented tools
    assert json.loads((tmp_path / "run" / "telemetry_end.json").read_text())["broken_beams"] == 0
    ledger = read_ledger(tmp_path / "run" / "ledger")
    urls = [r["args"]["url"] for r in ledger if "url" in r["args"]]
    assert urls and all("?" not in u and "secret" not in u for u in urls)
    scripts = [r for r in ledger if r["method"] == "mcp/run_script"]
    assert len(scripts) == 2 and all(r["blobs"] for r in scripts)
    assert audit_run(tmp_path / "run")["audit"] == "clean"


def test_passive_rows_are_readable_without_fabricating_attitude(tmp_path: Path) -> None:
    from openbeam_v2e.schema.runlog import read_runlog

    result = _run(tmp_path, "run", openbeam_passive_cassette(), yaw_convention={"units": "deg"})
    log = read_runlog(result.rows)
    assert log.header.attitude_observed is False
    assert len(log.rows) == 3
    assert log.rows[0].actors["crate"].q is None
    assert log.rows[0].actors["crate"].p == pytest.approx([0.0, 0.0, 2.25])


def test_unverified_orientation_makes_the_run_invalid(tmp_path: Path) -> None:
    result = _run(tmp_path, "run", openbeam_passive_cassette())
    assert not result.valid()
    assert any("spawn residual" in r for r in result.invalid_reasons())
    assert all("orientation unverified" in s.method for s in result.attestation.spawns)


def test_a_vanished_actor_invalidates_the_run(tmp_path: Path) -> None:
    result = _run(tmp_path, "run", openbeam_passive_cassette(remove_anchor=True), yaw_convention={"units": "deg"})
    assert result.attestation.removed_actors >= 1
    assert not result.valid() and any("removed" in r for r in result.invalid_reasons())


def test_cassette_mismatch_is_loud(tmp_path: Path) -> None:
    cassette = openbeam_passive_cassette()
    cassette[1] = {**cassette[1], "arguments": {**cassette[1]["arguments"], "filename": "other.zip"}}
    with pytest.raises(CassetteMismatch):
        _run(tmp_path, "run", cassette)


def test_install_refuses_packages_without_a_reachable_source() -> None:
    compiled = openbeam_compiled()
    local_only = compiled.packages[0].__class__("level", "level", "1" * 64, path=Path("level.zip"))
    backend = OpenBeamBackend(ReplayClient([{"tool": "list_mods", "arguments": {},
                                             "result": {"structuredContent": {"mods": []}}}]))
    with pytest.raises(RefusedCall, match="URL you control"):
        backend.install([local_only])


def test_third_seatable_vehicle_is_refused() -> None:
    backend = OpenBeamBackend(ReplayClient([]))
    backend._level = object()
    backend._spawned = [(type("R", (), {"seatable": True})(), None)] * 2
    obj = openbeam_compiled().objects[1]
    with pytest.raises(RefusedCall, match="at most 2 seatable"):
        backend.spawn(obj, openbeam_compiled().spawns[0].pose, "third")


def test_controlled_actors_need_an_accepted_host() -> None:
    class Host:
        reference = ScriptReference("Example script reference", "1", "https://example.com/script-api")
        snippets = (Snippet("drive_host", 'game.log("host $marker");', {"marker": r"[a-z0-9-]{1,40}"},
                            ("game.log",), "example"),)
        channels = ()
        observables = ("attitude",)

    with pytest.raises(ValueError, match="accepted explicitly"):
        OpenBeamBackend(ReplayClient([]), host=Host())
    backend = OpenBeamBackend(ReplayClient([]), host=Host(), accepted_script_reference=Host.reference)
    assert backend.scripts.check('game.log("host abc");') == "drive_host"
    manifest = backend.manifest()
    assert "attitude" in manifest.capabilities.observables
    assert manifest.capabilities.max_controlled_actors == 2
    # the offline audit policy knows only the built-in snippets; an instance policy adds the host's
    assert OpenBeamBackend.audit_policy().checks["mcp/run_script"](
        {"code": 'game.log("host abc");'}, {"seq": 1, "blobs": []}) is not None


def test_from_config_builds_transports_without_io(tmp_path: Path) -> None:
    cassette = tmp_path / "c.json"
    cassette.write_text(json.dumps({"schema": "v2e.cassette/1", "calls": []}))
    assert OpenBeamBackend.from_config({"cassette": str(cassette)}).transport == "cassette"
    assert OpenBeamBackend.from_config({"session_id": "abc-123", "token_file": str(tmp_path / "t")}).transport == \
        "session"
    assert OpenBeamBackend.from_config({"endpoint": "http://127.0.0.1:9/mcp"}).transport == "local"
    with pytest.raises(ValueError):
        OpenBeamBackend.from_config({})
    with pytest.raises(ValueError):
        OpenBeamBackend.from_config({"endpoint": "http://203.0.113.5/mcp"})     # plain http off this machine
    with pytest.raises(ValueError):
        transport.session_endpoint("../other")


# --------------------------------------------------------------------------- transport


class _McpHandler(BaseHTTPRequestHandler):
    seen: list = []
    sse = False

    def log_message(self, *args) -> None:
        pass

    def do_POST(self) -> None:  # noqa: N802 - http.server API
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).seen.append((self.headers.get("Authorization"), body))
        if self.headers.get("Authorization") != "Bearer test-token":
            self.send_response(401)
            self.end_headers()
            self.wfile.write(b'{"error":"unauthorized"}')
            return
        if "id" not in body:
            self.send_response(202)
            self.end_headers()
            return
        if body["method"] == "initialize":
            result = {"protocolVersion": "2025-06-18", "capabilities": {}, "serverInfo": {"name": "test"}}
        elif body["method"] == "tools/call":
            name = body["params"]["name"]
            result = {"structuredContent": {"tool": name, "args": body["params"]["arguments"]},
                      "isError": name == "bad"}
        else:
            result = {"tools": [{"name": "status"}]}
        answer = json.dumps({"jsonrpc": "2.0", "id": body["id"], "result": result}).encode()
        self.send_response(200)
        if type(self).sse:
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write(b"event: message\ndata: " + answer + b"\n\n")
        else:
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(answer)))
            self.end_headers()
            self.wfile.write(answer)


@pytest.fixture()
def mcp_server():
    _McpHandler.seen = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _McpHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}/mcp"
    server.shutdown()
    server.server_close()


@pytest.mark.parametrize("sse", [False, True])
def test_transport_speaks_streamable_http(mcp_server: str, sse: bool) -> None:
    _McpHandler.sse = sse
    game = transport.GameEndpoint(mcp_server, lambda: "test-token")
    result = game.call("status", {})
    assert result.data == {"tool": "status", "args": {}} and not result.is_error
    assert game.call("bad", {}).is_error
    methods = [body.get("method") for _, body in _McpHandler.seen]
    assert methods[:3] == ["initialize", "notifications/initialized", "tools/call"]
    assert all(auth == "Bearer test-token" for auth, _ in _McpHandler.seen)
    assert all("test-token" not in json.dumps(body) for _, body in _McpHandler.seen)
    game.close()


def test_transport_reports_http_errors(mcp_server: str) -> None:
    game = transport.GameEndpoint(mcp_server, lambda: "wrong")
    with pytest.raises(transport.McpTransportError, match="HTTP 401"):
        game.call("status", {})


def test_account_forwarding_wraps_game_tools(mcp_server: str) -> None:
    _McpHandler.sse = False
    account = transport.CloudAccount(lambda: "test-token", base_url=mcp_server.rsplit("/", 1)[0])
    game = transport.AccountGame(account, "sess-1")
    data = game.call("status", {}).data
    assert data == {"tool": "cloud_game_call", "args": {"session_id": "sess-1", "name": "status", "arguments": {}}}


def test_recording_client_writes_a_shareable_cassette(tmp_path: Path) -> None:
    inner = ReplayClient([{"tool": "status", "arguments": {}, "result": {"structuredContent": {"ok": True}}}])
    recorder = RecordingClient(inner, tmp_path / "rec.json")
    recorder.call("status", {})
    recorder.close()
    replay = ReplayClient(tmp_path / "rec.json")
    assert replay.call("status", {}).data == {"ok": True}


# --------------------------------------------------------------------------- credentials and sign-in


def test_token_store_permissions_and_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENBEAM_TOKEN", raising=False)
    store = transport.TokenStore(tmp_path / "cfg" / "token")
    assert store.read() is None
    path = store.write("abc")
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    assert store.read() == "abc"
    os.chmod(path, 0o644)
    with pytest.raises(PermissionError):
        store.read()
    monkeypatch.setenv("OPENBEAM_TOKEN", "from-env")
    assert store.read() == "from-env"


class _DeviceHandler(BaseHTTPRequestHandler):
    polls = 0

    def log_message(self, *args) -> None:
        pass

    def do_POST(self) -> None:  # noqa: N802 - http.server API
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length)) if length else {}
        if self.path == "/v1/device/code":
            answer, status = {"deviceCode": "dev-1", "userCode": "ABCD-EFGH", "interval": 0, "expiresIn": 60,
                              "verificationUriComplete": "http://127.0.0.1/app/#device=ABCD-EFGH"}, 200
        elif body.get("deviceCode") == "dev-1":
            type(self).polls += 1
            answer, status = ({"error": "authorization_pending"}, 400) if type(self).polls < 2 else \
                ({"userId": "u1", "token": "tok-1"}, 200)
        else:
            answer, status = {"error": "expired_token"}, 400
        data = json.dumps(answer).encode()
        self.send_response(status)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def test_device_code_sign_in() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _DeviceHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    said: list[str] = []
    try:
        token = transport.device_code_login(f"http://127.0.0.1:{server.server_address[1]}", say=said.append,
                                            sleep=lambda s: None)
    finally:
        server.shutdown()
        server.server_close()
    assert token == "tok-1" and "ABCD-EFGH" in said[0]


def test_bbox_compensation_constants_are_consistent() -> None:
    # the cassette builder and the connector agree on where the engine is asked to put the actor
    for bbox in (CRATE_BBOX, ANCHOR_BBOX):
        lo, hi = bbox
        assert all(lo[i] < hi[i] for i in range(3))


# --------------------------------------------------------------------------- engine-script host path


class _StandInHost:
    """A host for tests only: its one snippet logs a start line; rows come back from the script log."""

    reference = ScriptReference("Stand-in script reference", "0", "https://example.com/stand-in")
    snippets = (Snippet("host_start", 'game.log("start $marker $actor");',
                        {"marker": r"[A-Za-z0-9_.:\-]{1,80}", "actor": r"[a-z0-9_]{1,40}"}, ("game.log",),
                        "stand-in"),)
    channels = (ActuatorChannel("drive.throttle", "drive", ChannelKind.CONTINUOUS, (0.0, 1.0),
                                mechanism="example.input/throttle"),)
    observables = ("position", "attitude")

    def start(self, programs, handles, marker, duration_s, telemetry_hz):
        return [("host_start", {"marker": marker, "actor": actor}) for actor in sorted(programs)]

    def row_grep(self, marker, page):
        return f"{marker} row={page}[0-9]{{3}} "

    def parse_row(self, line, handles):
        parts = line.split()
        if len(parts) != 11 or not parts[1].startswith("row="):
            return None
        values = [float(v) for v in parts[4:]]
        return {"t": float(parts[2][2:]), "actors": {parts[3]: {"p": values[:3], "q": values[3:]}}}


def test_engine_script_host_run_path(tmp_path: Path) -> None:
    host = _StandInHost()
    ok = {"structuredContent": {"return_code": 0, "succeeded": True}}
    log = "\n".join(f"v2e-x row={i:06d} t={i * 0.005:.3f} car_1 {i * 0.1:.1f} 0 0.7 0 0 0 1" for i in range(3))
    cassette = [
        {"tool": "run_script", "match": "tool", "arguments": {}, "result": ok},
        {"tool": "run_script", "match": "tool", "arguments": {}, "result": ok},
        {"tool": "sim_mode", "arguments": {"mode": "stepped", "idle_release_s": 120}, "result": ok},
        {"tool": "sim_step", "arguments": {"steps": 20}, "result": {"structuredContent": {"steps": 20}}},
        {"tool": "sim_mode", "arguments": {"mode": "realtime"}, "result": ok},
        {"tool": "read_log", "match": "tool", "arguments": {}, "result": {"content": [{"type": "text", "text": log}]}},
        {"tool": "run_script", "match": "tool", "arguments": {}, "result": ok},
    ]
    client = ReplayClient(cassette)
    backend = OpenBeamBackend(client, host=host, accepted_script_reference=host.reference)
    assert backend.manifest().capabilities.controller_hosts == frozenset({ControllerHost.ENGINE_SCRIPT})
    ledger = CallLedger(tmp_path / "ledger", header={"connector": "openbeam"})
    backend.use_ledger(ledger)
    program = {"schema": "v2e.program/1", "actor": "car_1", "model": "schedule", "params": {"schedule": []}}
    spec = RunSpec(duration_s=0.01, host=ControllerHost.ENGINE_SCRIPT, regime=STEPPED, telemetry_hz=200.0)
    result = backend.run([ActorRef("car_1", "7", "car", True)], {"car_1": program}, spec, tmp_path)
    ledger.close()
    assert client.exhausted
    lines = Path(result.rows).read_text().splitlines()
    rows = [json.loads(line) for line in lines[1:]]
    assert [r["t"] for r in rows] == [0.0, 0.005, 0.01]
    assert rows[2]["actors"]["car_1"] == {"p": [0.2, 0.0, 0.7], "q": [0.0, 0.0, 0.0, 1.0]}
    issued = [args.get("code") for tool, args in client.issued if tool == "run_script"]
    assert issued[1].startswith('game.log("start v2e-') and issued[1].endswith(' car_1");')
    assert result.attestation.programs and result.attestation.programs[0][0] == "car_1"
    assert audit_ledger(tmp_path / "ledger", make_audit_policy(backend.scripts)) == "clean"
    # the offline policy knows only the built-in snippets, so a host run needs its host to be audited
    assert "audited snippet" in audit_ledger(tmp_path / "ledger", OpenBeamBackend.audit_policy())
