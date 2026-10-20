# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""The replayable call ledger.

Every engine call a connector makes is written as one JSON line: sequence number, wall time,
simulated time when known, the method or tool name, the **full arguments**, a digest of the
result, and the sha256 of the previous line, so that the file is a hash chain. Program texts and
other large payloads are stored once, content-addressed, under ``blobs/<sha256>``.

Two rules keep ledgers safe to share:

* credentials live in transport headers and never in arguments; as a second line of defence,
  argument keys that look like credentials are replaced with ``"<redacted>"``;
* URLs are stored without their query string and fragment (pre-signed download links carry their
  signature there); the package sha256 is stored next to them.

``audit_ledger`` replays a ledger offline against a connector's audit policy: the hash chain must
be intact, every call must be on the policy's allow list with only documented argument names,
and every stored blob must match its hash.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

from openbeam_v2e.backends.base import canonical_json, sha256_json, to_jsonable

__all__ = ["CallLedger", "LedgerCheck", "AuditPolicy", "read_ledger", "verify_ledger", "audit_ledger",
           "redact", "GENESIS"]

GENESIS = "0" * 64
LEDGER_FILE = "calls.jsonl"
BLOB_DIR = "blobs"

_SECRET_KEY = re.compile(r"(token|secret|password|passwd|authorization|api[_-]?key|bearer|cookie|credential)",
                         re.IGNORECASE)
_URL = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*://")


def _strip_url(value: str) -> str:
    if not _URL.match(value):
        return value
    parts = urllib.parse.urlsplit(value)
    netloc = parts.hostname or ""
    if parts.port:
        netloc = f"{netloc}:{parts.port}"
    return urllib.parse.urlunsplit((parts.scheme, netloc, parts.path, "", ""))


def redact(value: Any, key: str = "") -> Any:
    """Return ``value`` with credential-like keys masked and URL queries, fragments and user info removed."""
    if key and _SECRET_KEY.search(key):
        return "<redacted>"
    if isinstance(value, Mapping):
        return {str(k): redact(v, str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact(v) for v in value]
    if isinstance(value, str):
        return _strip_url(value)
    return to_jsonable(value)


def _row_hash(row: Mapping[str, Any]) -> str:
    body = {k: v for k, v in row.items() if k != "sha256"}
    return hashlib.sha256(canonical_json(body)).hexdigest()


class CallLedger:
    """Append-only, hash-chained JSONL ledger with a content-addressed blob store."""

    def __init__(self, directory: Path, header: Mapping[str, Any] | None = None) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / BLOB_DIR).mkdir(exist_ok=True)
        self.path = self.directory / LEDGER_FILE
        if self.path.exists() and self.path.stat().st_size:
            raise FileExistsError(f"{self.path} already holds a ledger; use a fresh run directory")
        self._fh = self.path.open("w", encoding="ascii")
        self._seq = 0
        self._head = GENESIS
        self._t0 = time.perf_counter()
        self.record("ledger.header", dict(header or {}))

    @property
    def head_sha256(self) -> str:
        return self._head

    @property
    def calls(self) -> int:
        return self._seq

    def store_blob(self, data: bytes | str) -> str:
        """Store ``data`` once under its sha256 and return the hash."""
        raw = data.encode("utf-8") if isinstance(data, str) else bytes(data)
        digest = hashlib.sha256(raw).hexdigest()
        target = self.directory / BLOB_DIR / digest
        if not target.exists():
            target.write_bytes(raw)
        return digest

    def record(self, method: str, args: Mapping[str, Any] | None = None, result: Any = None, *,
               sim_t: float | None = None, error: str | None = None, result_digest: str | None = None,
               blobs: list[str] | tuple[str, ...] = ()) -> dict:
        """Append one call. ``result`` is digested, never stored; pass ``result_digest`` to supply one.

        ``blobs`` lists the sha256 of payloads stored with :meth:`store_blob` that this call used
        (for example the program text a script call carried).
        """
        if self._fh.closed:
            raise ValueError("ledger is closed")
        if result_digest is None:
            result_digest = None if result is None else sha256_json(_digestable(result))
        row = {
            "seq": self._seq,
            "wall_s": round(time.perf_counter() - self._t0, 6),
            "sim_t": sim_t,
            "method": method,
            "args": redact(dict(args or {})),
            "result_sha256": result_digest,
            "blobs": list(blobs),
            "error": error,
            "prev_sha256": self._head,
        }
        row["sha256"] = _row_hash(row)
        self._fh.write(json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n")
        self._fh.flush()
        self._head = row["sha256"]
        self._seq += 1
        return row

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> "CallLedger":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def _digestable(result: Any) -> Any:
    """Results may hold bytes (images); hash those instead of serialising them."""
    if isinstance(result, (bytes, bytearray, memoryview)):
        return {"bytes_sha256": hashlib.sha256(bytes(result)).hexdigest()}
    if isinstance(result, Mapping):
        return {str(k): _digestable(v) for k, v in result.items()}
    if isinstance(result, (list, tuple)):
        return [_digestable(v) for v in result]
    return to_jsonable(result)


def read_ledger(directory: Path) -> list[dict]:
    path = Path(directory) / LEDGER_FILE
    rows = []
    with path.open("r", encoding="ascii") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


@dataclass(frozen=True)
class LedgerCheck:
    ok: bool
    rows: int
    head_sha256: str
    problems: tuple[str, ...] = ()

    @property
    def first_problem(self) -> str:
        return self.problems[0] if self.problems else "clean"


def verify_ledger(directory: Path) -> LedgerCheck:
    """Check the hash chain, sequence numbers and blob hashes of a ledger."""
    directory = Path(directory)
    problems: list[str] = []
    try:
        rows = read_ledger(directory)
    except (OSError, ValueError) as exc:
        return LedgerCheck(False, 0, GENESIS, (f"cannot read ledger: {exc}",))
    head = GENESIS
    for index, row in enumerate(rows):
        if row.get("seq") != index:
            problems.append(f"row {index}: sequence number {row.get('seq')!r}")
        if row.get("prev_sha256") != head:
            problems.append(f"row {index}: hash chain broken")
        if _row_hash(row) != row.get("sha256"):
            problems.append(f"row {index}: row hash does not match its content")
        head = row.get("sha256", "")
    if rows and rows[0].get("method") != "ledger.header":
        problems.append("row 0 is not the ledger header")
    blob_dir = directory / BLOB_DIR
    if blob_dir.is_dir():
        for blob in sorted(blob_dir.iterdir()):
            if hashlib.sha256(blob.read_bytes()).hexdigest() != blob.name:
                problems.append(f"blob {blob.name[:12]}... does not match its hash")
    return LedgerCheck(not problems, len(rows), head, tuple(problems))


ArgCheck = Callable[[Mapping[str, Any], Mapping[str, Any]], "str | None"]


@dataclass(frozen=True)
class AuditPolicy:
    """What a connector may call. ``methods`` maps a method name to its allowed argument names.

    ``checks`` maps a method name to a function ``(args, row) -> problem or None`` for deeper
    checks (for example: a script body must come from the audited snippet set, and must be stored
    as a blob).
    """

    connector: str
    methods: Mapping[str, frozenset[str]]
    checks: Mapping[str, ArgCheck] = field(default_factory=dict)

    def problem(self, row: Mapping[str, Any]) -> str | None:
        method = row.get("method")
        if method == "ledger.header":
            return None
        if method not in self.methods:
            return f"call {row.get('seq')}: {method!r} is not on the {self.connector} allow list"
        args = row.get("args") or {}
        extra = sorted(set(args) - set(self.methods[method]))
        if extra:
            return f"call {row.get('seq')}: {method} with undocumented argument(s) {extra}"
        check = self.checks.get(method)
        if check is not None:
            return check(args, row)
        return None


def iter_problems(rows: list[dict], policy: AuditPolicy) -> Iterator[str]:
    for row in rows:
        problem = policy.problem(row)
        if problem:
            yield problem


def audit_ledger(directory: Path, policy: AuditPolicy) -> str:
    """Return ``"clean"`` or the first violation found when replaying the ledger against ``policy``."""
    check = verify_ledger(directory)
    if not check.ok:
        return check.first_problem
    rows = read_ledger(directory)
    header = rows[0].get("args", {}) if rows else {}
    if header.get("connector") not in (None, policy.connector):
        return f"ledger was written by {header.get('connector')!r}, not {policy.connector!r}"
    blob_dir = Path(directory) / BLOB_DIR
    for row in rows:
        for digest in row.get("blobs") or []:
            if not (blob_dir / str(digest)).is_file():
                return f"call {row.get('seq')}: blob {str(digest)[:12]}... is missing"
    return next(iter_problems(rows, policy), "clean")
