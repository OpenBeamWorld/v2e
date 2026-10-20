# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""Claim classes: what kind of physics a run's result rests on, computed from its effects.

Each effect a scene requires (``drive``, ``brake``, ``contact``, ...) is provided by a source:
``native`` (the engine's own mechanism driven by authored content), ``native_substitute`` (a
different native mechanism standing in for the observed one), ``scripted`` (force code that only
a plugin can ship), ``absent`` or ``test_double``. The run's class follows:

* ``incomplete`` when a required effect is absent or missing;
* ``test_double`` when any effect comes from a test double;
* ``scripted`` when any effect is scripted;
* ``emulated`` when any effect is a native substitute;
* ``native`` otherwise.

The provenance signature hashes the effects and the content-origin records, so runs whose
physics or content differ are never grouped together.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from openbeam_v2e.schema import jsonio

__all__ = ["SOURCES", "claim_class", "provenance_signature", "claim_block"]

SOURCES = ("native", "native_substitute", "scripted", "absent", "test_double")


def _source(effect: Mapping[str, Any]) -> str:
    source = effect.get("source", "absent")
    return getattr(source, "value", source)


def claim_class(effects: Iterable[Mapping[str, Any]], required: Sequence[str] = ()) -> str:
    """Claim class from effect records ``{"name", "source", ...}`` and the required effect names."""
    effects = list(effects)
    by_name = {e.get("name"): _source(e) for e in effects}
    if any(by_name.get(name, "absent") == "absent" for name in required):
        return "incomplete"
    sources = set(by_name.values())
    if "test_double" in sources:
        return "test_double"
    if "scripted" in sources:
        return "scripted"
    if "native_substitute" in sources:
        return "emulated"
    if not effects:
        return "unknown"
    return "native"


def provenance_signature(effects: Iterable[Mapping[str, Any]], content: Iterable[Mapping[str, Any]] = ()) -> str:
    """SHA-256 over the sorted effects and content-origin records."""
    eff = sorted(({k: getattr(v, "value", v) for k, v in e.items()} for e in effects),
                 key=lambda e: jsonio.canonical_bytes(e))
    con = sorted((dict(c) for c in content), key=lambda c: jsonio.canonical_bytes(c))
    return jsonio.canonical_sha256({"effects": eff, "content": con})


def claim_block(run_info: Mapping[str, Any] | None, required: Sequence[str] = ()) -> dict[str, Any]:
    """The ``claim`` member of a score report from what is known about the run."""
    if not run_info:
        return {"class": "unknown", "provenance_signature": None, "valid": None,
                "label": "claim class unknown: no run record was given"}
    effects = [dict(e) for e in run_info.get("effects", [])]
    content = [dict(c) for c in run_info.get("content", [])]
    cls = run_info.get("claim_class") or claim_class(effects, required)
    signature = run_info.get("provenance_signature") or (provenance_signature(effects, content) if effects else None)
    block = {"class": cls, "provenance_signature": signature, "valid": run_info.get("valid"),
             "effects": effects, "content": content}
    if cls != "native":
        block["label"] = f"claim class: {cls}"
    return block
