from __future__ import annotations

import os
import stat

import pytest

from core.memory.session_pin_cipher import SESSION_PIN_ENVELOPE_SCHEMA
from core.memory.session_pin_ledger import (
    SESSION_PIN_LEDGER_FILENAME,
    SessionPinLedger,
    SessionPinLedgerError,
)


def _record(record_id: str = "a" * 32) -> dict[str, str]:
    return {
        "schema": SESSION_PIN_ENVELOPE_SCHEMA,
        "key_id": "sha256:" + "b" * 64,
        "record_id": record_id,
        "nonce_b64": "bm9uY2U=",
        "ciphertext_b64": "Y2lwaGVydGV4dA==",
    }


def test_session_pin_ledger_rejects_arbitrary_filename(tmp_path):
    with pytest.raises(ValueError, match="filename must be"):
        SessionPinLedger(tmp_path / "arbitrary.jsonl")


def test_session_pin_ledger_replaces_through_private_atomic_owner(tmp_path):
    path = tmp_path / SESSION_PIN_LEDGER_FILENAME
    ledger = SessionPinLedger(path)

    with ledger.transaction():
        ledger.commit_records([_record()])
        snapshot = ledger.read_snapshot()

    assert snapshot.truncated is False
    assert snapshot.permissions_repair_required is False
    assert len(snapshot.lines) == 1
    assert SESSION_PIN_ENVELOPE_SCHEMA in snapshot.lines[0]
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_session_pin_ledger_refuses_symlink_reads(tmp_path):
    target = tmp_path / "other.jsonl"
    target.write_text("secret\n", encoding="utf-8")
    path = tmp_path / SESSION_PIN_LEDGER_FILENAME
    path.symlink_to(target)
    ledger = SessionPinLedger(path)

    with ledger.transaction(), pytest.raises(
        SessionPinLedgerError,
        match="open_failed",
    ):
        ledger.read_snapshot()


def test_session_pin_ledger_bounds_tail_and_marks_rewrite(monkeypatch, tmp_path):
    from core.memory import session_pin_ledger as ledger_module

    monkeypatch.setattr(ledger_module, "SESSION_PIN_LEDGER_MAX_BYTES", 96)
    monkeypatch.setattr(ledger_module, "SESSION_PIN_LEDGER_MAX_RECORDS", 2)
    path = tmp_path / SESSION_PIN_LEDGER_FILENAME
    path.write_text("first\nsecond\n" + "x" * 128 + "\nlast\n", encoding="utf-8")
    ledger = SessionPinLedger(path)

    with ledger.transaction():
        snapshot = ledger.read_snapshot()

    assert snapshot.truncated is True
    assert len(snapshot.lines) <= 2
    assert snapshot.lines[-1] == "last"


def test_session_pin_ledger_rejects_non_envelope_records(tmp_path):
    ledger = SessionPinLedger(tmp_path / SESSION_PIN_LEDGER_FILENAME)

    with ledger.transaction(), pytest.raises(
        SessionPinLedgerError,
        match="record_schema_invalid",
    ):
        ledger.commit_records([{"schema": "plaintext", "content": "secret"}])


@pytest.mark.skipif(not hasattr(os, "O_NOFOLLOW"), reason="requires O_NOFOLLOW")
def test_session_pin_ledger_repairs_permissive_legacy_mode(tmp_path):
    path = tmp_path / SESSION_PIN_LEDGER_FILENAME
    path.write_text("", encoding="utf-8")
    path.chmod(0o644)
    ledger = SessionPinLedger(path)

    with ledger.transaction():
        before = ledger.read_snapshot()
        ledger.commit_records([])
        after = ledger.read_snapshot()

    assert before.permissions_repair_required is True
    assert after.permissions_repair_required is False
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
