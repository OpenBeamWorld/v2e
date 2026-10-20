# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""A minimal MCP transport for OpenBeam (standard library only).

OpenBeam's documented endpoints speak MCP over Streamable HTTP (JSON-RPC 2.0 in ``POST``
requests, answered with JSON or a short event stream):

* a **session endpoint**, ``https://cloud.openbeam.world/v1/sessions/<session id>/mcp``, which
  exposes the game tools of one running cloud session; it takes the access token from the
  documented device-code sign-in as a bearer token;
* the **account endpoint**, ``https://cloud.openbeam.world/mcp``, which exposes the account tools
  (``cloud_session_*``, ``cloud_game_call``, ``cloud_recording_*``) to a bearer token your MCP
  client obtained for the account;
* a game you run yourself, if it serves MCP at an address and token you configure.

Credentials are read from the ``OPENBEAM_TOKEN`` environment variable or from a token file
(default ``$XDG_CONFIG_HOME/openbeam-v2e/token``, written with mode 0600). They travel only in the
``Authorization`` header and never appear in tool arguments, ledgers or logs.
"""

from __future__ import annotations

import base64
import http.client
import json
import os
import stat
import threading
import time
import urllib.parse
from pathlib import Path
from typing import Any, Callable, Mapping

from openbeam_v2e import __version__
from openbeam_v2e.backends.base import BackendError
from openbeam_v2e.backends.openbeam.tools import ToolResult

__all__ = ["PROTOCOL_VERSION", "DEFAULT_BASE_URL", "McpHttp", "McpTransportError", "GameEndpoint", "CloudAccount",
           "AccountGame", "TokenStore", "session_endpoint", "list_sessions", "device_code_login", "decode_tool_result"]

PROTOCOL_VERSION = "2025-06-18"
DEFAULT_BASE_URL = "https://cloud.openbeam.world"
USER_AGENT = f"openbeam-v2e/{__version__}"


class McpTransportError(BackendError):
    """A JSON-RPC error or an HTTP or network failure."""


def decode_tool_result(result: Mapping[str, Any], seconds: float = 0.0) -> ToolResult:
    content = list(result.get("content") or [])
    texts = [c.get("text", "") for c in content if c.get("type") == "text"]
    images = [(c.get("mimeType", ""), base64.b64decode(c.get("data", ""))) for c in content if c.get("type") == "image"]
    data = result.get("structuredContent")
    if data is None:
        try:
            data = json.loads(texts[0]) if texts else {}
        except ValueError:
            data = {"text": texts[0]}
    if not isinstance(data, dict):
        data = {"value": data}
    return ToolResult(data=data, text=texts[0] if texts else "", images=images, is_error=bool(result.get("isError")),
                      seconds=seconds)


def _parse_body(raw: str, method: str) -> Any:
    raw = raw.strip()
    if not raw:
        return None
    if raw.startswith("{") or raw.startswith("["):
        return json.loads(raw)
    answer = None
    for block in raw.split("\n\n"):
        data = "\n".join(line[5:].lstrip() for line in block.splitlines() if line.startswith("data:"))
        if data:
            answer = json.loads(data)
    if answer is None:
        raise McpTransportError(f"{method}: the server sent no JSON-RPC answer")
    return answer


class McpHttp:
    """JSON-RPC over Streamable HTTP: one persistent connection, JSON or event-stream answers."""

    def __init__(self, url: str, token: Callable[[], str | None], *, timeout_s: float = 120.0) -> None:
        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise ValueError(f"MCP endpoint must be an http(s) URL, not {url!r}")
        if parsed.scheme == "http" and parsed.hostname not in ("127.0.0.1", "localhost", "::1"):
            raise ValueError("plain http is allowed only for a game on this machine; use https")
        self.url = url
        self._parsed = parsed
        self._token = token
        self.timeout_s = timeout_s
        self._id = 0
        self._lock = threading.Lock()
        self._conn: http.client.HTTPConnection | None = None
        self.session_id: str | None = None
        self.initialized = False

    def _connection(self) -> http.client.HTTPConnection:
        if self._conn is None:
            cls = http.client.HTTPSConnection if self._parsed.scheme == "https" else http.client.HTTPConnection
            self._conn = cls(self._parsed.hostname, self._parsed.port, timeout=self.timeout_s)
        return self._conn

    def rpc(self, method: str, params: Mapping[str, Any] | None = None, *, notification: bool = False) -> Any:
        with self._lock:
            body: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
            if not notification:
                self._id += 1
                body["id"] = self._id
            if params is not None:
                body["params"] = dict(params)
            headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream",
                       "MCP-Protocol-Version": PROTOCOL_VERSION, "User-Agent": USER_AGENT}
            token = self._token()
            if token:
                headers["Authorization"] = f"Bearer {token}"
            if self.session_id:
                headers["Mcp-Session-Id"] = self.session_id
            raw = self._post(json.dumps(body).encode("utf-8"), headers, method)
        if notification:
            return None
        answer = _parse_body(raw, method)
        if answer is None:
            raise McpTransportError(f"{method}: empty answer")
        if "error" in answer:
            err = answer["error"] or {}
            raise McpTransportError(f"{method}: JSON-RPC error {err.get('code')}: {err.get('message')}")
        return answer.get("result")

    def _post(self, data: bytes, headers: Mapping[str, str], method: str) -> str:
        path = self._parsed.path or "/"
        if self._parsed.query:
            path += "?" + self._parsed.query
        for attempt in (0, 1):
            reused = self._conn is not None
            conn = self._connection()
            try:
                conn.request("POST", path, body=data, headers=dict(headers))
                response = conn.getresponse()
                raw = response.read().decode("utf-8", "replace")
            except (http.client.HTTPException, OSError) as exc:
                conn.close()
                self._conn = None
                # retry once only on a reused connection the server closed while idle; a tool call
                # that may have reached the server is never sent twice
                if attempt == 0 and reused and isinstance(exc, (http.client.RemoteDisconnected, BrokenPipeError,
                                                                 ConnectionResetError)):
                    continue
                raise McpTransportError(f"{method}: {type(exc).__name__}: {exc}") from None
            sid = response.getheader("Mcp-Session-Id")
            if sid:
                self.session_id = sid
            if (response.getheader("Connection") or "").lower() == "close":
                conn.close()
                self._conn = None
            if response.status >= 400:
                raise McpTransportError(f"{method}: HTTP {response.status}: {raw[:300]}")
            return raw
        raise McpTransportError(f"{method}: no answer")

    def initialize(self) -> dict:
        result = self.rpc("initialize", {"protocolVersion": PROTOCOL_VERSION, "capabilities": {},
                                         "clientInfo": {"name": "openbeam-v2e", "version": __version__}})
        self.rpc("notifications/initialized", notification=True)
        self.initialized = True
        return result or {}

    def call_tool(self, name: str, arguments: Mapping[str, Any] | None = None) -> ToolResult:
        if not self.initialized:
            self.initialize()
        started = time.perf_counter()
        result = self.rpc("tools/call", {"name": name, "arguments": dict(arguments or {})}) or {}
        return decode_tool_result(result, time.perf_counter() - started)

    def list_tools(self) -> list[dict]:
        if not self.initialized:
            self.initialize()
        return list((self.rpc("tools/list", {}) or {}).get("tools", []))

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None


class GameEndpoint:
    """Game tools at one MCP endpoint: a cloud session endpoint or a game you run yourself."""

    def __init__(self, url: str, token: Callable[[], str | None], *, timeout_s: float = 120.0) -> None:
        self.mcp = McpHttp(url, token, timeout_s=timeout_s)

    def call(self, tool: str, arguments: Mapping[str, Any] | None = None) -> ToolResult:
        return self.mcp.call_tool(tool, arguments)

    def close(self) -> None:
        self.mcp.close()


class CloudAccount:
    """The account endpoint: ``cloud_session_*``, ``cloud_game_call`` and ``cloud_recording_*``."""

    def __init__(self, token: Callable[[], str | None], *, base_url: str = DEFAULT_BASE_URL,
                 timeout_s: float = 180.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.mcp = McpHttp(f"{self.base_url}/mcp", token, timeout_s=timeout_s)

    def call(self, tool: str, arguments: Mapping[str, Any] | None = None) -> ToolResult:
        return self.mcp.call_tool(tool, arguments)

    def close(self) -> None:
        self.mcp.close()


class AccountGame:
    """Game tools of one cloud session, forwarded through the account's ``cloud_game_call``."""

    def __init__(self, account: CloudAccount, session_id: str) -> None:
        self.account = account
        self.session_id = session_id

    def call(self, tool: str, arguments: Mapping[str, Any] | None = None) -> ToolResult:
        return self.account.call("cloud_game_call", {"session_id": self.session_id, "name": tool,
                                                     "arguments": dict(arguments or {})})

    def close(self) -> None:
        self.account.close()


def session_endpoint(session_id: str, base_url: str = DEFAULT_BASE_URL) -> str:
    if not session_id or not all(c.isalnum() or c in "_-" for c in session_id) or len(session_id) > 128:
        raise ValueError("session ids are 1-128 letters, digits, underscores or hyphens")
    return f"{base_url.rstrip('/')}/v1/sessions/{session_id}/mcp"


class TokenStore:
    """Read and write the access token: ``OPENBEAM_TOKEN`` first, then a 0600 file."""

    ENV = "OPENBEAM_TOKEN"

    def __init__(self, path: Path | None = None) -> None:
        if path is None:
            base = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
            path = base / "openbeam-v2e" / "token"
        self.path = Path(path)

    def __call__(self) -> str | None:
        return self.read()

    def read(self) -> str | None:
        token = os.environ.get(self.ENV)
        if token:
            return token.strip()
        try:
            mode = self.path.stat().st_mode
        except OSError:
            return None
        if mode & (stat.S_IRWXG | stat.S_IRWXO):
            raise PermissionError(f"{self.path} is readable by other users; run chmod 600 on it")
        return self.path.read_text(encoding="utf-8").strip() or None

    def write(self, token: str) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(token.strip() + "\n")
        os.chmod(self.path, 0o600)
        return self.path


def _rest(method: str, url: str, *, token: str | None = None, body: Mapping[str, Any] | None = None,
          timeout_s: float = 30.0) -> tuple[int, Any]:
    parsed = urllib.parse.urlsplit(url)
    cls = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
    conn = cls(parsed.hostname, parsed.port, timeout=timeout_s)
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    data = None
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        conn.request(method, parsed.path or "/", body=data, headers=headers)
        response = conn.getresponse()
        raw = response.read().decode("utf-8", "replace")
    except (http.client.HTTPException, OSError) as exc:
        raise McpTransportError(f"{method} {parsed.path}: {exc}") from None
    finally:
        conn.close()
    try:
        payload = json.loads(raw) if raw.strip() else None
    except ValueError:
        payload = {"text": raw[:300]}
    return response.status, payload


def list_sessions(token: str, base_url: str = DEFAULT_BASE_URL) -> list[dict]:
    """``GET /v1/sessions``: the account's sessions (``sessionId``, ``state``, ...)."""
    status, payload = _rest("GET", f"{base_url.rstrip('/')}/v1/sessions", token=token)
    if status != 200:
        raise McpTransportError(f"listing sessions failed with HTTP {status}")
    return list(payload or [])


def device_code_login(base_url: str = DEFAULT_BASE_URL, *, say: Callable[[str], None] = print,
                      sleep: Callable[[float], None] = time.sleep, max_wait_s: float = 600.0) -> str:
    """The documented device-code sign-in: request a code, wait for browser approval, return the token."""
    base = base_url.rstrip("/")
    status, code = _rest("POST", f"{base}/v1/device/code")
    if status != 200 or not isinstance(code, dict) or "deviceCode" not in code:
        raise McpTransportError(f"requesting a device code failed with HTTP {status}")
    say(f"Open {code.get('verificationUriComplete')} and approve code {code.get('userCode')}.")
    interval = float(code.get("interval", 5))
    deadline = time.monotonic() + min(max_wait_s, float(code.get("expiresIn", 600)))
    while time.monotonic() < deadline:
        sleep(interval)
        status, answer = _rest("POST", f"{base}/v1/device/token", body={"deviceCode": code["deviceCode"]})
        answer = answer or {}
        if status == 200 and answer.get("token"):
            return str(answer["token"])
        error = answer.get("error") or answer.get("code") or ""
        if error == "authorization_pending":
            continue
        if error == "slow_down":
            interval += 5.0
            continue
        raise McpTransportError(f"device sign-in ended: {error or f'HTTP {status}'}")
    raise McpTransportError("device sign-in timed out; request a new code")
