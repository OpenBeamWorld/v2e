# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""Strict JSON reading and canonical JSON writing for V2E documents.

Reading rejects duplicate object keys and the non-standard constants ``NaN`` and
``Infinity``, so a document means the same thing to every reader. Writing never emits
non-finite numbers. :func:`canonical_sha256` hashes a document independently of key order
and whitespace; content-origin records and package requests use it.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

__all__ = ["JsonError", "loads", "load", "dumps", "dump", "canonical_bytes", "canonical_sha256", "to_jsonable"]


class JsonError(ValueError):
    """A document is not strict JSON."""


def _pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in items:
        if key in out:
            raise JsonError(f"duplicate object key {key!r}")
        out[key] = value
    return out


def _bad_constant(name: str) -> Any:
    raise JsonError(f"non-standard JSON constant {name}")


def loads(text: str) -> Any:
    """Parse strict JSON text."""
    try:
        return json.loads(text, object_pairs_hook=_pairs, parse_constant=_bad_constant)
    except json.JSONDecodeError as exc:
        raise JsonError(str(exc)) from None


def load(path: str | Path) -> Any:
    """Read a strict JSON file (UTF-8)."""
    return loads(Path(path).read_text(encoding="utf-8"))


def to_jsonable(value: Any) -> Any:
    """Convert NumPy scalars and arrays, tuples and paths into plain JSON values."""
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, Path):
        return value.as_posix()
    if hasattr(value, "tolist") and callable(value.tolist):  # numpy arrays and scalars
        return to_jsonable(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        raise JsonError("refusing to write a non-finite number")
    return value


def dumps(value: Any, indent: int | None = 2) -> str:
    """Serialise ``value`` as JSON text with a trailing newline."""
    return json.dumps(to_jsonable(value), indent=indent, allow_nan=False, ensure_ascii=False) + "\n"


def dump(value: Any, path: str | Path, indent: int | None = 2) -> Path:
    """Write ``value`` as UTF-8 JSON, creating parent directories."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(dumps(value, indent), encoding="utf-8")
    return target


def canonical_bytes(value: Any) -> bytes:
    """Key-sorted, whitespace-free UTF-8 encoding used for hashing."""
    return json.dumps(to_jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False,
                      ensure_ascii=False).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    """SHA-256 of :func:`canonical_bytes` as lowercase hex."""
    return hashlib.sha256(canonical_bytes(value)).hexdigest()
