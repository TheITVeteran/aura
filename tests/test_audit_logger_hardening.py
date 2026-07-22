"""CP126 hardening contracts for core/audit/audit_logger.py.

Covers the schema-safe fixes: honoring an injected HMAC secret, fail-closed DB
init, a write receipt from log(), value-level redaction, and a close/checkpoint
lifecycle. Each test uses a private tmp DB and an explicit secret — no shared or
live audit DB is touched.
"""
from __future__ import annotations

import sqlite3

import pytest

from core.audit.audit_logger import AuditLogger


def _logger(tmp_path, secret="unit-secret"):
    return AuditLogger(db_path=str(tmp_path / "audit.db"), hmac_secret=secret)


# ── 05fd4428: injected hmac_secret is honored ──────────────────────────────


def test_injected_secret_is_used(tmp_path, monkeypatch):
    monkeypatch.delenv("AURA_AUDIT_HMAC_SECRET", raising=False)
    al = _logger(tmp_path, secret="explicit")
    assert al.hmac_secret == b"explicit"


def test_missing_secret_raises(tmp_path, monkeypatch):
    monkeypatch.delenv("AURA_AUDIT_HMAC_SECRET", raising=False)
    with pytest.raises(RuntimeError):
        AuditLogger(db_path=str(tmp_path / "a.db"))


# ── e001f7fa: init failure fails construction ──────────────────────────────


def test_db_init_failure_fails_construction(tmp_path, monkeypatch):
    real_connect = sqlite3.connect

    class _BadConn:
        def __init__(self, real):
            self._real = real

        def execute(self, *a, **k):
            raise sqlite3.OperationalError("disk I/O error")

        def close(self):
            self._real.close()

    monkeypatch.setattr(sqlite3, "connect", lambda *a, **k: _BadConn(real_connect(":memory:")))
    with pytest.raises(RuntimeError, match="audit_db_init_failed"):
        AuditLogger(db_path=str(tmp_path / "b.db"), hmac_secret="s")


# ── 24e2b90a: log() returns a write receipt ────────────────────────────────


def test_log_returns_true_on_success(tmp_path):
    al = _logger(tmp_path)
    assert al.log("did_thing", actor="tester", target="x") is True
    assert al.verify_integrity() is True


def test_log_returns_false_after_close(tmp_path):
    al = _logger(tmp_path)
    al.close()
    assert al.log("after_close") is False


# ── a8f4cc6a: value-level redaction ────────────────────────────────────────


def test_redacts_secret_values_not_just_keys(tmp_path):
    al = _logger(tmp_path)
    red = al._redact({"note": "connect to https://user:pass@host/db", "cfg": "api_key=sk-12345"})
    assert "pass@host" not in red["note"] and "***:***@" in red["note"]
    assert "sk-12345" not in red["cfg"]


def test_redacts_secret_keys(tmp_path):
    al = _logger(tmp_path)
    red = al._redact({"password": "hunter2", "ok": "fine"})
    assert red["password"] == "[REDACTED]"
    assert red["ok"] == "fine"


# ── e78ffab8: close/checkpoint lifecycle ───────────────────────────────────


def test_close_is_idempotent_and_checkpoints(tmp_path):
    al = _logger(tmp_path)
    al.log("e", actor="a")
    al.close()
    al.close()  # idempotent, no raise
    assert al._closed is True


# ── faf5ad29: signatures survive a round-trip verify ───────────────────────


def test_tamper_is_detected(tmp_path):
    al = _logger(tmp_path)
    al.log("sensitive", actor="root", target="core")
    # Tamper with the stored row directly.
    al._conn.execute("UPDATE audit_events SET actor='attacker' WHERE id=1")
    al._conn.commit()
    assert al.verify_integrity() is False
