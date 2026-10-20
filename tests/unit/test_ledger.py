# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""The replayable call ledger: hash chain, redaction, blobs and policy audits."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openbeam_v2e.backends.ledger import (
    AuditPolicy,
    CallLedger,
    audit_ledger,
    read_ledger,
    redact,
    verify_ledger,
)


def _ledger(tmp_path: Path) -> CallLedger:
    ledger = CallLedger(tmp_path / "ledger", header={"connector": "demo"})
    digest = ledger.store_blob("program text")
    ledger.record("demo.load", {"name": "a", "sha256": digest}, {"ok": True}, blobs=[digest])
    ledger.record("demo.step", {"steps": 10}, b"\x00\x01", sim_t=0.01)
    ledger.close()
    return ledger


def _policy() -> AuditPolicy:
    return AuditPolicy("demo", {"demo.load": frozenset({"name", "sha256"}), "demo.step": frozenset({"steps"})})


def test_chain_verifies_and_head_matches(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    check = verify_ledger(tmp_path / "ledger")
    assert check.ok and check.rows == 3
    assert check.head_sha256 == ledger.head_sha256
    rows = read_ledger(tmp_path / "ledger")
    assert rows[0]["method"] == "ledger.header" and rows[1]["blobs"]
    assert audit_ledger(tmp_path / "ledger", _policy()) == "clean"


def test_tampering_is_detected(tmp_path: Path) -> None:
    _ledger(tmp_path)
    path = tmp_path / "ledger" / "calls.jsonl"
    lines = path.read_text().splitlines()
    row = json.loads(lines[2])
    row["args"]["steps"] = 11
    path.write_text("\n".join([lines[0], lines[1], json.dumps(row)]) + "\n")
    assert not verify_ledger(tmp_path / "ledger").ok
    path.write_text("\n".join([lines[0], lines[2]]) + "\n")
    problems = verify_ledger(tmp_path / "ledger").problems
    assert any("chain" in p or "sequence" in p for p in problems)


def test_blob_tampering_and_missing_blobs(tmp_path: Path) -> None:
    _ledger(tmp_path)
    blob = next((tmp_path / "ledger" / "blobs").iterdir())
    blob.write_text("changed")
    assert "does not match" in verify_ledger(tmp_path / "ledger").first_problem
    blob.unlink()
    assert "missing" in audit_ledger(tmp_path / "ledger", _policy())


def test_policy_violations(tmp_path: Path) -> None:
    ledger = CallLedger(tmp_path / "l1", header={"connector": "demo"})
    ledger.record("demo.step", {"steps": 1, "force_n": 5.0})
    ledger.close()
    assert "undocumented argument" in audit_ledger(tmp_path / "l1", _policy())
    ledger = CallLedger(tmp_path / "l2", header={"connector": "demo"})
    ledger.record("demo.teleport", {})
    ledger.close()
    assert "not on the demo allow list" in audit_ledger(tmp_path / "l2", _policy())
    ledger = CallLedger(tmp_path / "l3", header={"connector": "other"})
    ledger.close()
    assert "written by 'other'" in audit_ledger(tmp_path / "l3", _policy())


def test_redaction_strips_url_secrets_and_credential_keys() -> None:
    raw = {"url": "https://user:pw@bucket.example.com/a/b.zip?X-Amz-Signature=abc#frag", "token": "t",
           "nested": {"Authorization": "Bearer x", "list": ["https://h.example/p?q=1", 3]}, "steps": 4}
    out = redact(raw)
    assert out["url"] == "https://bucket.example.com/a/b.zip"
    assert out["token"] == "<redacted>" and out["nested"]["Authorization"] == "<redacted>"
    assert out["nested"]["list"] == ["https://h.example/p", 3]
    assert out["steps"] == 4


def test_ledger_refuses_to_append_to_an_existing_run(tmp_path: Path) -> None:
    _ledger(tmp_path)
    with pytest.raises(FileExistsError):
        CallLedger(tmp_path / "ledger")
