# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""The OpenBeam tools this connector may call, and the guard that enforces the list.

``PUBLIC_TOOLS`` is fixed in code. Every name in it appears in OpenBeam's published MCP tool
reference, and a unit test checks that against a copy of the reference's tool index. For each
tool, ``TOOL_ARGS`` lists the argument names the connector may send; it is the documented set or
a narrower one (for example, ``sim_step`` may carry only ``steps``: V2E never sends joint
targets).

:class:`GuardedClient` wraps any client object with a ``call(tool, arguments)`` method (the
transport in this package, a recorded cassette, or another MCP client) and refuses, before any
network I/O, a tool outside the list, an undocumented argument, or a ``run_script`` body that is
not a rendering of an audited snippet. Every allowed call is written to the run's call ledger.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

from openbeam_v2e.backends.base import BackendError, RefusedCall

__all__ = ["ACCOUNT_TOOLS", "GAME_TOOLS", "PUBLIC_TOOLS", "TOOL_ARGS", "ToolResult", "GameClient", "GuardedClient",
           "ToolError"]

ACCOUNT_TOOLS = frozenset((
    "cloud_game_call", "cloud_game_tools", "cloud_recording_download", "cloud_recording_list",
    "cloud_recording_start", "cloud_recording_stop", "cloud_session_end", "cloud_session_list",
    "cloud_session_start",
))

GAME_TOOLS = frozenset((
    "actor_poses", "camera_frame", "delete_mod", "install_mod", "list_mods", "load_level", "quit", "read_log",
    "run_script", "screenshot", "sim_mode", "sim_step", "spawn_vehicle", "status", "telemetry", "upload_level",
))

#: Every tool the connector may call (25). A subset of the published tool reference.
PUBLIC_TOOLS = ACCOUNT_TOOLS | GAME_TOOLS

_ARCHIVE_ARGS = frozenset({"url", "path", "sha256", "filename", "replace"})

TOOL_ARGS: Mapping[str, frozenset[str]] = {
    # account tools
    "cloud_session_list": frozenset(),
    "cloud_session_start": frozenset({"width", "height", "fps", "bitrate_kbps", "kind"}),
    "cloud_session_end": frozenset({"session_id"}),
    "cloud_game_tools": frozenset({"session_id"}),
    "cloud_game_call": frozenset({"name", "arguments", "session_id"}),
    "cloud_recording_start": frozenset({"session_id", "label"}),
    "cloud_recording_stop": frozenset({"session_id"}),
    "cloud_recording_list": frozenset(),
    "cloud_recording_download": frozenset({"session_id", "episode_id", "file"}),
    # game tools
    "status": frozenset(),
    "telemetry": frozenset(),
    "read_log": frozenset({"log", "lines", "grep"}),
    "install_mod": _ARCHIVE_ARGS,
    "upload_level": _ARCHIVE_ARGS,
    "list_mods": frozenset(),
    "delete_mod": frozenset({"filename"}),
    "load_level": frozenset({"level_id", "filename", "archive", "sha256", "terrain", "spawn", "wait_seconds"}),
    "spawn_vehicle": frozenset({"file", "config", "position", "enter", "wait_seconds"}),
    "run_script": frozenset({"code"}),
    "actor_poses": frozenset(),
    "camera_frame": frozenset({"max_width", "format"}),
    "screenshot": frozenset({"wait_seconds", "keep"}),
    "sim_mode": frozenset({"mode", "idle_release_s"}),
    "sim_step": frozenset({"steps"}),
    "quit": frozenset(),
}

assert set(TOOL_ARGS) == set(PUBLIC_TOOLS)


class ToolError(BackendError):
    """The game or the account service answered a tool call with ``isError``."""

    def __init__(self, tool: str, message: str, data: Any = None) -> None:
        super().__init__(f"{tool}: {message}")
        self.tool = tool
        self.data = data


@dataclass
class ToolResult:
    data: dict                                   # structuredContent, or the text parsed as JSON
    text: str = ""
    images: list = field(default_factory=list)   # [(mime type, bytes)]
    is_error: bool = False
    seconds: float = 0.0


class GameClient(Protocol):
    def call(self, tool: str, arguments: Mapping[str, Any] | None = None) -> ToolResult: ...


def _check_args(tool: str, arguments: Mapping[str, Any]) -> None:
    if tool not in PUBLIC_TOOLS:
        raise RefusedCall(f"{tool!r} is not in the OpenBeam connector's documented tool list")
    extra = sorted(set(arguments) - set(TOOL_ARGS[tool]))
    if extra:
        raise RefusedCall(f"{tool}: argument(s) {extra} are not in the documented set {sorted(TOOL_ARGS[tool])}")


class GuardedClient:
    """Enforce ``PUBLIC_TOOLS``, ``TOOL_ARGS`` and the script snippet policy, then call and record."""

    def __init__(self, client: GameClient, *, scripts: Any, ledger: Any = None, raise_on_error: bool = True) -> None:
        self.client = client
        self.scripts = scripts
        self.ledger = ledger
        self.raise_on_error = raise_on_error
        self.sim_t: float | None = None

    def check(self, tool: str, arguments: Mapping[str, Any]) -> list[str]:
        """Raise ``RefusedCall`` unless the call is allowed; return blob payloads to store with it."""
        _check_args(tool, arguments)
        blobs: list[str] = []
        if tool == "cloud_game_call":
            inner = arguments.get("name")
            if inner not in GAME_TOOLS:
                raise RefusedCall(f"cloud_game_call may forward only documented game tools, not {inner!r}")
            blobs += self.check(str(inner), dict(arguments.get("arguments") or {}))
        if tool == "run_script":
            code = arguments.get("code")
            if not isinstance(code, str):
                raise RefusedCall("run_script needs a string 'code'")
            self.scripts.check(code)          # raises RefusedCall for anything outside the snippet set
            blobs.append(code)
        return blobs

    def call(self, tool: str, arguments: Mapping[str, Any] | None = None, *, raise_on_error: bool | None = None,
             ledger_method: str | None = None) -> ToolResult:
        arguments = dict(arguments or {})
        payloads = self.check(tool, arguments)
        digests = []
        for payload in payloads:
            if self.ledger is not None:
                digests.append(self.ledger.store_blob(payload))
            else:
                digests.append(hashlib.sha256(payload.encode("utf-8")).hexdigest())
        error = None
        result: ToolResult | None = None
        try:
            result = self.client.call(tool, arguments)
            if result.is_error:
                error = _message(result)
        except BackendError as exc:
            error = str(exc)
            raise
        finally:
            if self.ledger is not None:
                payload = None if result is None else {"data": result.data, "text": result.text,
                                                       "images": [img for _, img in result.images],
                                                       "is_error": result.is_error}
                self.ledger.record(ledger_method or f"mcp/{tool}", arguments, payload, sim_t=self.sim_t,
                                   error=error, blobs=digests)
        should_raise = self.raise_on_error if raise_on_error is None else raise_on_error
        if result.is_error and should_raise:
            raise ToolError(tool, error or "tool error", result.data)
        return result


def _message(result: ToolResult) -> str:
    data = result.data or {}
    message = data.get("message") or data.get("error") or result.text or "tool error"
    return message if isinstance(message, str) else str(message)
