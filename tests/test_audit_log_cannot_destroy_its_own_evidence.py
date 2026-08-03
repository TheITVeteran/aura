"""An audit log that deletes itself when tampered with is not an audit log.

CP126 on core/audit/__init__.py — Aura's record of what she did, registered
at boot. Four criticals:

* on corruption it renamed the database, DELETED the WAL and SHM sidecars,
  and if the rename failed deleted everything — then created a fresh empty
  log. That destroys the evidence at exactly the moment tampering must be
  investigated, and the WAL holds the most recent entries;
* every handled insert, commit, healing and retry failure fell through to
  ``return entry_id``, so a caller could not tell a durable receipt from a
  lost event;
* the schema comment promised redacted JSON and the code stored the
  caller's dictionary verbatim;
* the table had no update/delete prevention, signature, hash chain or
  verifier, so any process with database access could alter rows undetected.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from core.audit import AuditLog


@pytest.fixture
def audit(tmp_path):
    return AuditLog(db_path=str(tmp_path / "audit.db"))


# ------------------------------------------------------- evidence survives


def test_a_corrupt_database_is_quarantined_never_deleted(tmp_path):
    db = tmp_path / "audit.db"
    log = AuditLog(db_path=str(db))
    log.record("skill_call", "before corruption")
    log.close()

    db.write_bytes(b"this is not a sqlite database at all, not even close")
    Path(str(db) + "-wal").write_bytes(b"newest entries live here")

    AuditLog(db_path=str(db))

    quarantined = list(tmp_path.glob("audit.db.corrupt.*"))
    assert quarantined, "the corrupt database was destroyed instead of quarantined"
    assert any(p.name.endswith("-wal") for p in tmp_path.glob("audit.db.corrupt.*-wal")), (
        "the WAL was deleted; it holds the most recent entries, which is "
        "exactly what an investigator needs"
    )


def test_quarantine_preserves_the_original_bytes(tmp_path):
    db = tmp_path / "audit.db"
    AuditLog(db_path=str(db)).close()
    marker = b"corrupt-but-evidential-0123456789"
    db.write_bytes(marker)

    AuditLog(db_path=str(db))

    quarantined = [p for p in tmp_path.glob("audit.db.corrupt.*") if not p.name.endswith(("-wal", "-shm"))]
    assert quarantined
    assert quarantined[0].read_bytes() == marker


def test_a_quarantine_that_cannot_complete_refuses_rather_than_overwriting(tmp_path, monkeypatch):
    db = tmp_path / "audit.db"
    AuditLog(db_path=str(db)).close()
    db.write_bytes(b"corrupt")

    def _cannot_rename(self, target):
        raise OSError("read-only volume")

    monkeypatch.setattr(Path, "rename", _cannot_rename)
    log = AuditLog(db_path=str(db))

    assert db.read_bytes() == b"corrupt", (
        "the corrupt database was overwritten by a fresh log after the "
        "quarantine failed; the evidence is gone"
    )
    assert log.record("skill_call", "anything") == "", (
        "auditing must report unavailable rather than silently accept writes "
        "it cannot chain to the preserved history"
    )


# ------------------------------------------------------- honest receipts


def test_a_successful_record_returns_an_id(audit):
    assert audit.record("skill_call", "did a thing")


def test_a_lost_event_returns_no_receipt(audit, monkeypatch):
    """The caller must be able to tell a durable receipt from a lost event."""

    def _explode(*args, **kwargs):
        raise sqlite3.DatabaseError("disk went away")

    monkeypatch.setattr(audit, "_insert_locked", _explode)
    assert audit.record("skill_call", "this never landed") == ""


def test_the_entry_id_is_not_truncated_to_48_bits(audit):
    """A collision became a database error reported as a valid id."""
    assert len(audit.record("skill_call", "x")) == 32


# ------------------------------------------------------------- redaction


def test_parameters_are_redacted_not_merely_promised(audit):
    audit.record(
        "skill_call",
        "called an api",
        params={"api_key": "sk-live-secret-value-01234567890", "user": "bryan"},
    )
    rows = audit.get_recent(limit=1)
    assert "sk-live-secret-value" not in str(rows), (
        "the schema comment promised redacted JSON and the code stored the "
        "caller's dictionary verbatim"
    )


def test_redaction_keeps_the_non_secret_fields(audit):
    audit.record("skill_call", "x", params={"api_key": "sk-abcdefghijklmnop", "tool": "grep"})
    assert "grep" in str(audit.get_recent(limit=1))


# ---------------------------------------------------------- the hash chain


def test_a_clean_log_verifies(audit):
    for index in range(5):
        audit.record("skill_call", f"entry {index}")
    report = audit.verify_chain()
    assert report["verified"] is True
    assert report["entries"] == 5


def test_an_edited_row_breaks_the_chain(audit, tmp_path):
    for index in range(3):
        audit.record("skill_call", f"entry {index}")
    audit.close()

    con = sqlite3.connect(str(tmp_path / "audit.db"))
    con.execute("UPDATE audit_log SET description='tampered' WHERE seq=2")
    con.commit()
    con.close()

    report = audit.verify_chain()
    assert report["verified"] is False
    assert report["reason"] == "entry_modified_after_write"


def test_a_deleted_row_breaks_the_chain(audit, tmp_path):
    for index in range(4):
        audit.record("skill_call", f"entry {index}")
    audit.close()

    con = sqlite3.connect(str(tmp_path / "audit.db"))
    con.execute("DELETE FROM audit_log WHERE seq=2")
    con.commit()
    con.close()

    report = audit.verify_chain()
    assert report["verified"] is False
    assert "sequence_gap" in report["reason"]


def test_the_first_entry_commits_to_a_genesis_anchor(audit, tmp_path):
    """Otherwise the log can be truncated to a shorter, still-valid chain."""
    audit.record("skill_call", "first")
    audit.close()
    con = sqlite3.connect(str(tmp_path / "audit.db"))
    con.row_factory = sqlite3.Row
    row = con.execute("SELECT prev_hash FROM audit_log WHERE seq=1").fetchone()
    con.close()
    assert row["prev_hash"] == "aura.audit.chain.genesis.v1"


def test_a_legacy_database_is_migrated_not_recreated(tmp_path):
    """Recreating an audit log to change its shape is the same act as deleting it."""
    db = tmp_path / "audit.db"
    con = sqlite3.connect(str(db))
    con.executescript(
        """
        CREATE TABLE audit_log (
            id TEXT PRIMARY KEY, action_type TEXT NOT NULL, description TEXT NOT NULL,
            actor TEXT NOT NULL, skill_name TEXT, params TEXT, result_ok INTEGER,
            cid TEXT, session_id TEXT, created_at REAL NOT NULL
        );
        """
    )
    con.execute(
        "INSERT INTO audit_log VALUES ('old1','skill_call','historic','user',NULL,NULL,NULL,NULL,NULL,1.0)"
    )
    con.commit()
    con.close()

    log = AuditLog(db_path=str(db))
    assert any(row["id"] == "old1" for row in log.get_recent(limit=10)), (
        "the pre-chain history was discarded by the migration"
    )
    log.record("skill_call", "new entry")
    report = log.verify_chain()
    assert report["verified"] is True
    assert report["unchained_legacy_rows"] == 1, (
        "rows written before the chain existed are genuinely unprotected and "
        "must be reported as such, not counted as verified"
    )


# ------------------------------------------------------- input validation


def test_a_negative_limit_does_not_return_the_entire_history(audit):
    """In SQLite a negative LIMIT means NO limit."""
    for index in range(20):
        audit.record("skill_call", f"entry {index}")
    assert len(audit.get_recent(limit=-1)) <= 20
    assert len(audit.get_recent(limit=5)) == 5


@pytest.mark.parametrize("bad", [None, "many", float("nan")])
def test_a_malformed_limit_falls_back_rather_than_raising(audit, bad):
    audit.record("skill_call", "x")
    assert isinstance(audit.get_recent(limit=bad), list)


def test_an_absurd_limit_is_capped(audit):
    audit.record("skill_call", "x")
    assert len(audit.get_recent(limit=10**9)) == 1
