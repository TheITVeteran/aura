"""Scientific engine: hypothesis → experiment (ledger receipt) → belief update."""
from __future__ import annotations

import pytest

from core.cognition.scientific_engine import ScientificEngine


@pytest.fixture
def engine(tmp_path):
    return ScientificEngine(db_path=str(tmp_path / "sci.db"), learning_rate=0.5)


def test_form_hypothesis_starts_open(engine):
    hid = engine.form_hypothesis(
        "retry fixes flaky tool", predicted_observable="success_rate",
        expected=0.8, prior_confidence=0.5,
    )
    h = engine.get(hid)
    assert h.status == "open"
    assert h.confidence == pytest.approx(0.5)


def test_confirming_evidence_raises_confidence(engine):
    hid = engine.form_hypothesis("X works", predicted_observable="rate", expected=0.9, prior_confidence=0.5)
    before = engine.belief(hid)
    engine.observe(hid, observed=0.9)  # matches expectation
    assert engine.belief(hid) > before


def test_refuting_evidence_lowers_confidence(engine):
    hid = engine.form_hypothesis("X works", predicted_observable="rate", expected=0.9, prior_confidence=0.7)
    before = engine.belief(hid)
    engine.observe(hid, observed=0.1)  # opposite of expectation
    assert engine.belief(hid) < before


def test_repeated_support_settles_to_supported(engine):
    hid = engine.form_hypothesis("Y holds", predicted_observable="rate", expected=0.8, prior_confidence=0.55)
    for _ in range(4):
        engine.observe(hid, observed=0.8)
    h = engine.get(hid)
    assert h.status == "supported"
    assert h.supports == h.trials


def test_repeated_refutation_settles_to_refuted(engine):
    hid = engine.form_hypothesis("Z holds", predicted_observable="rate", expected=0.9, prior_confidence=0.4)
    for _ in range(4):
        engine.observe(hid, observed=0.1)
    h = engine.get(hid)
    assert h.status == "refuted"


def test_experiment_opens_a_real_ledger_receipt(engine, tmp_path, monkeypatch):
    # Point the ledger singleton at a temp db so the experiment receipt is real but isolated.
    import core.cognition.outcome_ledger as ol
    monkeypatch.setattr(ol, "_ledger", ol.OutcomeLedger(db_path=str(tmp_path / "ledger.db")))

    hid = engine.form_hypothesis("ledger-backed", predicted_observable="rate", expected=0.7)
    receipt_id = engine.run_experiment(hid)
    assert receipt_id is not None
    assert any(p["receipt_id"] == receipt_id for p in ol.get_outcome_ledger().pending())

    # Observing resolves that receipt (expected-vs-observed lands in the ledger).
    engine.observe(hid, observed=0.95)
    assert all(p["receipt_id"] != receipt_id for p in ol.get_outcome_ledger().pending())
    assert ol.get_outcome_ledger().expectation_calibration() >= 0.0


def test_hypotheses_persist_across_instances(tmp_path):
    path = str(tmp_path / "persist_sci.db")
    e1 = ScientificEngine(db_path=path)
    hid = e1.form_hypothesis("durable claim", predicted_observable="rate", expected=0.6)

    e2 = ScientificEngine(db_path=path)
    assert e2.get(hid) is not None
    assert e2.get(hid).claim == "durable claim"
