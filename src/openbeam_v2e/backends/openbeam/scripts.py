# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""Audited script snippets for ``run_script``, and the engine-script host interface.

The connector never sends free-form script code. Every ``run_script`` body must be a rendering of
a registered :class:`Snippet`: a fixed template whose parameters match strict patterns. A body
that matches no snippet is refused before any network I/O, and the ledger audit re-checks every
recorded body offline.

This release registers one snippet, ``log_marker``, which writes a marker line to the script log
with ``game.log`` as shown in OpenBeam's published tool reference. Driving vehicles needs an
engine-side controller host (see :class:`ScriptHost`) written against a script API reference the
user has accepted; this release bundles none, so on OpenBeam it runs passive scenes only, through
the documented stepped-physics tools. A host is enabled only with an explicit acceptance of its
script reference in the connector configuration.
"""

from __future__ import annotations

import re
import string
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

from openbeam_v2e.backends.base import ActuatorChannel, RefusedCall

__all__ = ["Snippet", "ScriptReference", "ScriptPolicy", "ScriptHost", "BUILTIN_SNIPPETS", "MARKER_PATTERN"]

MARKER_PATTERN = r"[A-Za-z0-9_.:\-]{1,80}"


@dataclass(frozen=True)
class ScriptReference:
    """A published script API reference that a host's calls are checked against."""

    name: str
    version: str
    url: str


@dataclass(frozen=True)
class Snippet:
    id: str
    template: str                       # ``string.Template`` syntax: ``$name`` placeholders
    params: Mapping[str, str]           # placeholder -> regular expression for allowed values
    calls: tuple[str, ...]              # script API calls the snippet makes
    reference: str                      # where those calls are documented

    def render(self, **values: Any) -> str:
        missing = sorted(set(self.params) - set(values))
        extra = sorted(set(values) - set(self.params))
        if missing or extra:
            raise RefusedCall(f"snippet {self.id}: missing {missing}, unexpected {extra}")
        for name, value in values.items():
            if not re.fullmatch(self.params[name], str(value)):
                raise RefusedCall(f"snippet {self.id}: value for {name!r} does not match its pattern")
        return string.Template(self.template).substitute({k: str(v) for k, v in values.items()})

    def pattern(self) -> re.Pattern[str]:
        """A regular expression matching exactly the renderings of this snippet."""
        parts = []
        pos = 0
        for match in string.Template.pattern.finditer(self.template):
            parts.append(re.escape(self.template[pos:match.start()]))
            if match.group("escaped") is not None:
                parts.append(re.escape("$"))
            else:
                key = match.group("named") or match.group("braced")
                if not key or key not in self.params:
                    raise ValueError(f"snippet {self.id}: bad placeholder in its template")
                parts.append(f"(?:{self.params[key]})")
            pos = match.end()
        parts.append(re.escape(self.template[pos:]))
        return re.compile("".join(parts))


BUILTIN_SNIPPETS = (
    Snippet(
        id="log_marker",
        template='game.log("$marker");',
        params={"marker": MARKER_PATTERN},
        calls=("game.log",),
        reference="OpenBeam MCP tool reference: run_script example",
    ),
)


class ScriptHost(Protocol):
    """An engine-side controller host for the ``engine_script`` controller host.

    A host declares the script reference its calls are written against, the snippets it needs
    (templates only; no free-form code), the actuator channels and observables it provides, and
    how to start a run and read its rows back from the script log.
    """

    reference: ScriptReference
    snippets: Sequence[Snippet]
    channels: Sequence[ActuatorChannel]
    observables: Sequence[str]

    def start(self, programs: Mapping[str, Mapping[str, Any]], handles: Mapping[str, str], marker: str,
              duration_s: float, telemetry_hz: float) -> list[tuple[str, Mapping[str, Any]]]:
        """Return ``(snippet id, values)`` pairs that start the host for this run."""

    def row_grep(self, marker: str, page: int) -> str:
        """Regular expression selecting page ``page`` of this run's row lines in the script log."""

    def parse_row(self, line: str, handles: Mapping[str, str]) -> Mapping[str, Any] | None:
        """One ``v2e-runlog/1`` row from a log line, or None for lines that are not rows."""


@dataclass
class ScriptPolicy:
    """The set of snippets ``run_script`` bodies are checked against."""

    snippets: dict[str, Snippet] = field(default_factory=lambda: {s.id: s for s in BUILTIN_SNIPPETS})

    def add(self, snippets: Sequence[Snippet]) -> None:
        for snippet in snippets:
            if snippet.id in self.snippets and self.snippets[snippet.id] != snippet:
                raise ValueError(f"snippet id {snippet.id!r} is already registered with a different template")
            self.snippets[snippet.id] = snippet

    def render(self, snippet_id: str, **values: Any) -> str:
        if snippet_id not in self.snippets:
            raise RefusedCall(f"unknown snippet {snippet_id!r}")
        return self.snippets[snippet_id].render(**values)

    def match(self, code: str) -> str | None:
        for snippet in self.snippets.values():
            if snippet.pattern().fullmatch(code):
                return snippet.id
        return None

    def check(self, code: str) -> str:
        snippet_id = self.match(code)
        if snippet_id is None:
            raise RefusedCall("run_script body is not a rendering of an audited snippet")
        return snippet_id

    def describe(self) -> dict[str, Any]:
        return {s.id: {"template": s.template, "params": dict(s.params), "calls": list(s.calls),
                       "reference": s.reference} for s in self.snippets.values()}

    @classmethod
    def from_description(cls, description: Mapping[str, Any]) -> "ScriptPolicy":
        policy = cls(snippets={})
        policy.add([Snippet(sid, d["template"], dict(d["params"]), tuple(d["calls"]), d["reference"])
                    for sid, d in description.items()])
        return policy
