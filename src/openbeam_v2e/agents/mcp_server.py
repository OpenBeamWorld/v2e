# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""An MCP server for the V2E tools over stdio (standard library only).

Start it with ``openbeam-v2e mcp`` and register that command with any MCP-capable client. The
server speaks the stdio transport of the Model Context Protocol: one JSON-RPC 2.0 message per line
on stdin and stdout, logs on stderr. It implements ``initialize``, ``ping``, ``tools/list`` and
``tools/call`` (protocol revisions 2025-06-18, 2025-03-26 and 2024-11-05) and exposes the tools in
:mod:`openbeam_v2e.agents.tools`.

Tool results carry the JSON answer twice, as ``structuredContent`` and as a text block, so clients
without structured-output support still see it. A failed step is a tool result with
``isError: true``; protocol problems (malformed JSON, unknown methods or tools) are JSON-RPC
errors.
"""

from __future__ import annotations

import contextlib
import json
import sys
from typing import IO, Any, Mapping

from openbeam_v2e import __version__
from openbeam_v2e.agents.tools import AgentContext, call_tool, tool_specs

__all__ = ["SUPPORTED_PROTOCOLS", "McpServer", "serve"]

SUPPORTED_PROTOCOLS = ("2025-06-18", "2025-03-26", "2024-11-05")

PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602

INSTRUCTIONS = (
    "OpenBeam V2E recreates a filmed physical event in a physics engine and scores the run against the footage. "
    "Typical order: inspect (no arguments) to see connectors and available steps; reconstruct a video; compile a "
    "v2e.scene/1 scene for a connector; run it; inspect the run (validity, claim class, ledger audit); score it. "
    "The 'fake' connector is an offline test double: its runs are labelled test_double and support no physics claim. "
    "Paths are relative to the server's workspace; outputs go to new directories."
)


class McpServer:
    def __init__(self, ctx: AgentContext, *, log: IO[str] | None = None) -> None:
        self.ctx = ctx
        self.log = log or sys.stderr
        self.protocol: str | None = None
        self.initialized = False

    def _say(self, message: str) -> None:
        print(f"openbeam-v2e mcp: {message}", file=self.log, flush=True)

    @staticmethod
    def _error(msg_id: Any, code: int, message: str) -> dict:
        return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": message}}

    @staticmethod
    def _result(msg_id: Any, result: Mapping[str, Any]) -> dict:
        return {"jsonrpc": "2.0", "id": msg_id, "result": dict(result)}

    def handle(self, message: Any) -> dict | list | None:
        """Answer one decoded JSON-RPC message (or batch); None for notifications."""
        if isinstance(message, list):
            answers = [a for a in (self.handle(m) for m in message) if a is not None]
            return answers or None
        if not isinstance(message, dict) or message.get("jsonrpc") != "2.0" or not isinstance(message.get("method"),
                                                                                                 str):
            return self._error(message.get("id") if isinstance(message, dict) else None, INVALID_REQUEST,
                               "not a JSON-RPC 2.0 request")
        method = message["method"]
        msg_id = message.get("id")
        params = message.get("params") or {}
        is_notification = "id" not in message
        if is_notification:
            if method == "notifications/initialized":
                self.initialized = True
            return None
        if not isinstance(params, dict):
            return self._error(msg_id, INVALID_PARAMS, "params must be an object")
        if method == "initialize":
            requested = params.get("protocolVersion")
            self.protocol = requested if requested in SUPPORTED_PROTOCOLS else SUPPORTED_PROTOCOLS[0]
            return self._result(msg_id, {
                "protocolVersion": self.protocol,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "openbeam-v2e", "title": "OpenBeam V2E", "version": __version__},
                "instructions": INSTRUCTIONS,
            })
        if method == "ping":
            return self._result(msg_id, {})
        if method == "tools/list":
            specs = tool_specs()
            if self.protocol and self.protocol < "2025-06-18":
                specs = [{k: v for k, v in s.items() if k not in ("outputSchema", "title")} for s in specs]
            return self._result(msg_id, {"tools": specs})
        if method == "tools/call":
            name = params.get("name")
            if not isinstance(name, str) or name not in {s["name"] for s in tool_specs()}:
                return self._error(msg_id, INVALID_PARAMS, f"unknown tool {name!r}")
            arguments = params.get("arguments")
            answer = call_tool(self.ctx, name, arguments, on_error=self._say)
            result: dict[str, Any] = {
                "content": [{"type": "text", "text": json.dumps(answer, indent=2, sort_keys=True)}],
                "isError": not answer.get("ok", False),
            }
            if not self.protocol or self.protocol >= "2025-06-18":
                result["structuredContent"] = answer
            return self._result(msg_id, result)
        if method in ("resources/list", "prompts/list"):
            return self._error(msg_id, METHOD_NOT_FOUND, f"{method} is not offered by this server")
        return self._error(msg_id, METHOD_NOT_FOUND, f"unknown method {method!r}")

    def serve(self, stdin: IO[bytes], stdout: IO[bytes]) -> int:
        self._say(f"serving {len(tool_specs())} tools over stdio; workspace {self.ctx.root}; connectors "
                  f"{sorted(self.ctx.connectors)}")
        while True:
            line = stdin.readline()
            if not line:
                return 0
            text = line.decode("utf-8", "replace").strip()
            if not text:
                continue
            try:
                message = json.loads(text)
            except ValueError:
                answer: Any = self._error(None, PARSE_ERROR, "invalid JSON")
            else:
                answer = self.handle(message)
            if answer is not None:
                stdout.write((json.dumps(answer, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8"))
                stdout.flush()


def serve(ctx: AgentContext) -> int:
    """Serve on the process's stdin and stdout until stdin closes.

    While serving, anything a pipeline step prints goes to stderr, so it cannot corrupt the
    protocol stream on stdout.
    """
    protocol_out = sys.stdout.buffer
    with contextlib.redirect_stdout(sys.stderr):
        return McpServer(ctx).serve(sys.stdin.buffer, protocol_out)
