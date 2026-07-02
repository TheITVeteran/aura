"""Outcome Ledger: delayed-receipt credit assignment with expected-vs-observed + persistence."""
from __future__ import annotations

import pytest

from core.cognition.outcome_ledger import (
    CreditSource,
    OutcomeLedger,
    get_outcome_ledger,
)


@pytest.fixture
def ledger(tmp_path):
    return OutcomeLedger(db_path=str(tmp_path / "ledger.db"), default_horizon_s=100.0)


def test_open_then_resolve_computes_delay_and_prediction_error(ledger):
    rid = ledger.open(
        "ship the reply", expected=0.7,
        sources=[CreditSource("policy", "continue_goal", 1.0)],
        now=1000.0,
    )
    assert len(ledger.pending()) == 1

    receipt = ledger.resolve(rid, observed=0.9, now=1042.0)
    assert receipt is not None
    assert receipt.status == "resolved"
    assert receipt.delay == pytest.approx(42.0)
    assert receipt.prediction_error == pytest.approx(0.9 - 0.7)
    assert len(ledger.pending()) == 0


def test_resolve_unknown_receipt_returns_none(ledger):
    assert ledger.resolve("nope", observed=1.0) is None


def test_sweep_expires_stale_receipts_as_failures(ledger):
    ledger.open("slow goal", expected=0.8, horizon_s=10.0, now=1000.0)
    # before horizon → not swept
    assert ledger.sweep(now=1005.0) == []
    assert len(ledger.pending()) == 1
    # past horizon → expired as observed=0
    expired = ledger.sweep(now=1011.0)
    assert len(expired) == 1
    assert expired[0].status == "expired"
    assert expired[0].observed == 0.0
    assert len(ledger.pending()) == 0


def test_credit_flows_to_sources(ledger):
    ledger.open(
        "good tool call", expected=0.5,
        sources=[CreditSource("tool", "web_search", 2.0), CreditSource("memory", "m123", 1.0)],
        now=1000.0,
    )
    rid = ledger.pending()[0]["receipt_id"]
    ledger.resolve(rid, observed=1.0, now=1001.0)

    credit = ledger.credit_by_source(now=1002.0)
    # both sources earned positive credit; the higher-weight tool earned more
    assert credit["web_search"] > 0
    assert credit["m123"] > 0
    assert credit["web_search"] > credit["m123"]


def test_expectation_calibration_tracks_error(ledger):
    r1 = ledger.open("a", expected=0.5, now=1000.0)
    ledger.resolve(r1, observed=0.5, now=1001.0)  # perfect → err 0
    assert ledger.expectation_calibration() == pytest.approx(0.0)

    r2 = ledger.open("b", expected=0.0, now=1002.0)
    ledger.resolve(r2, observed=1.0, now=1003.0)  # err 1.0
    # mean of [0.0, 1.0] = 0.5
    assert ledger.expectation_calibration() == pytest.approx(0.5)


def test_pending_receipts_survive_reopen(tmp_path):
    path = str(tmp_path / "persist.db")
    l1 = OutcomeLedger(db_path=path, default_horizon_s=1_000_000_000_000.0)
    rid = l1.open(
        "long horizon action",
        expected=0.6,
        horizon_s=1_000_000_000_000.0,
        now=1000.0,
    )

    # New ledger instance (simulating a restart) recovers the pending receipt …
    l2 = OutcomeLedger(db_path=path)
    pend = l2.pending()
    assert any(p["receipt_id"] == rid for p in pend)
    # … and can resolve it in the "next session".
    resolved = l2.resolve(rid, observed=0.8, now=5000.0)
    assert resolved is not None and resolved.delay == pytest.approx(4000.0)


def test_startup_expires_stale_pending_receipts_before_recovery(tmp_path):
    path = str(tmp_path / "persist.db")
    l1 = OutcomeLedger(db_path=path)
    rid = l1.open("expired startup action", expected=0.7, horizon_s=1.0, now=1000.0)

    l2 = OutcomeLedger(db_path=path)

    assert l2.pending() == []
    assert l2.stats()["startup_expired_count"] == 1
    assert l2.resolve(rid, observed=1.0) is None


def test_pending_load_cap_does_not_prevent_resolution_by_id(tmp_path, monkeypatch):
    path = str(tmp_path / "persist.db")
    monkeypatch.setattr(OutcomeLedger, "MAX_PENDING_LOAD", 2)
    l1 = OutcomeLedger(db_path=path, default_horizon_s=1_000_000_000_000.0)
    receipt_ids = [
        l1.open(
            f"long horizon action {idx}",
            expected=0.5,
            horizon_s=1_000_000_000_000.0,
            now=1000.0 + idx,
        )
        for idx in range(5)
    ]

    l2 = OutcomeLedger(db_path=path, default_horizon_s=1_000_000_000_000.0)

    assert len(l2.pending()) == 2
    assert l2.stats()["pending_db_count"] == 5
    assert l2.stats()["pending_load_truncated"] is True
    resolved = l2.resolve(receipt_ids[0], observed=0.9, now=1010.0)
    assert resolved is not None
    assert resolved.receipt_id == receipt_ids[0]
    assert l2.stats()["pending_db_count"] == 4


def test_singleton_is_stable():
    assert get_outcome_ledger() is get_outcome_ledger()
