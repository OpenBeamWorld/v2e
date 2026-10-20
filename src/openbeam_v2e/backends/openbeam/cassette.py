# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""Record and replay OpenBeam tool calls (``v2e.cassette/1``) for offline tests.

A cassette is a JSON file listing tool calls in order, each with the arguments sent and the raw
MCP ``tools/call`` result received::

    {"schema": "v2e.cassette/1", "calls": [
      {"tool": "status", "arguments": {}, "result": {"structuredContent": {...}, "isError": false}}]}

:class:`ReplayClient` serves a cassette in order and fails on any call that differs from the
recording (set ``"match": "tool"`` on an entry to compare the tool name only).
:class:`RecordingClient` wraps a live client and writes what it sees; images are kept only as
their size and sha256, and URLs lose their query strings, so a cassette can be shared.
"""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from openbeam_v2e.backends.base import BackendError
from openbeam_v2e.backends.ledger import redact
from openbeam_v2e.backends.openbeam.tools import GameClient, ToolResult
from openbeam_v2e.backends.openbeam.transport import decode_tool_result

__all__ = ["SCHEMA_CASSETTE", "ReplayClient", "RecordingClient", "CassetteMismatch", "load_cassette"]

SCHEMA_CASSETTE = "v2e.cassette/1"


class CassetteMismatch(BackendError):
    pass


def load_cassette(source: Path | Mapping[str, Any]) -> list[dict]:
    data = json.loads(Path(source).read_text(encoding="utf-8")) if isinstance(source, (str, Path)) else dict(source)
    if data.get("schema") != SCHEMA_CASSETTE:
        raise ValueError(f"not a {SCHEMA_CASSETTE} file")
    return [dict(c) for c in data.get("calls", [])]


class ReplayClient:
    """Serve recorded tool results in order; every call must match the recording."""

    def __init__(self, cassette: Path | Mapping[str, Any] | list[dict]) -> None:
        self.calls = list(cassette) if isinstance(cassette, list) else load_cassette(cassette)
        self.position = 0
        self.issued: list[tuple[str, dict]] = []

    def call(self, tool: str, arguments: Mapping[str, Any] | None = None) -> ToolResult:
        arguments = dict(arguments or {})
        self.issued.append((tool, arguments))
        if self.position >= len(self.calls):
            raise CassetteMismatch(f"unexpected call {tool} after the end of the cassette")
        entry = self.calls[self.position]
        self.position += 1
        if entry["tool"] != tool:
            raise CassetteMismatch(f"call {self.position - 1}: expected {entry['tool']}, got {tool}")
        if entry.get("match", "all") == "all" and dict(entry.get("arguments", {})) != arguments:
            raise CassetteMismatch(f"call {self.position - 1} ({tool}): arguments differ from the recording: "
                                   f"{arguments!r} != {entry.get('arguments')!r}")
        return decode_tool_result(entry.get("result", {}))

    @property
    def exhausted(self) -> bool:
        return self.position == len(self.calls)

    def close(self) -> None:
        pass


class RecordingClient:
    """Wrap a live client and record every call into a cassette."""

    def __init__(self, client: GameClient, path: Path) -> None:
        self.client = client
        self.path = Path(path)
        self.calls: list[dict] = []

    def call(self, tool: str, arguments: Mapping[str, Any] | None = None) -> ToolResult:
        result = self.client.call(tool, arguments)
        content: list[dict] = []
        if result.text:
            content.append({"type": "text", "text": result.text})
        for mime, data in result.images:
            content.append({"type": "text", "text": json.dumps({"image": mime, "bytes": len(data),
                                                                "sha256": hashlib.sha256(data).hexdigest()})})
        self.calls.append({"tool": tool, "arguments": redact(dict(arguments or {})),
                           "result": {"structuredContent": redact(result.data), "content": content,
                                      "isError": result.is_error}})
        return result

    def save(self) -> Path:
        self.path.write_text(json.dumps({"schema": SCHEMA_CASSETTE, "calls": self.calls}, indent=2) + "\n",
                             encoding="utf-8")
        return self.path

    def close(self) -> None:
        self.save()
        close = getattr(self.client, "close", None)
        if callable(close):
            close()


def image_content(data: bytes, mime: str = "image/png") -> dict:
    """An MCP image content item, for hand-written cassettes."""
    return {"type": "image", "mimeType": mime, "data": base64.b64encode(data).decode("ascii")}
