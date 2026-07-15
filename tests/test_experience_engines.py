"""Experience-grounded verifiers: prediction resolution, outcome pricing,
rubric ensembles — the boundary movers that let non-seed domains earn
foundry admission."""
from __future__ import annotations

import asyncio

import pytest

from core.brain.verifiers.experience_engines import (
    OutcomeLedgerVerifier,
    PredictionResolutionVerifier,
    RubricEnsembleVerifier,
)
from core.brain.verifiers.foundry import VerifierFoundry

pytestmark = pytest.mark.unit


class _FakeLedger:
    def __init__(self):
        self.opened: dict[str, dict] = {}
        self.resolved: dict[str, float] = {}
        self._n = 0

    def open(self, action, expected, *, category="action", horizon_s=None,
             context=None):
        self._n += 1
        rid = f"rcpt-{self._n}"
        self.opened[rid] = {"action": action, "expected": expected}
        return rid

    def resolve(self, receipt_id, observed, *, note=""):
        if receipt_id not in self.opened:
            return None
        self.resolved[receipt_id] = observed
        return object()


# ─────────────────────────────────────────────────────────────────────────────
# PredictionResolutionVerifier
# ─────────────────────────────────────────────────────────────────────────────

def test_prediction_is_registered_not_verified():
    ledger = _FakeLedger()
    v = PredictionResolutionVerifier(ledger=ledger)
    res = asyncio.run(v.verify("The build will pass by tomorrow morning.",
                               context={"confidence": 0.8}))
    assert res.checked is False            # the future has not happened
    assert res.ok is True
    rid = res.detail["prediction_receipt"]
    assert rid and ledger.opened[rid]["expected"] == 0.8
    assert rid in v.open_predictions()


def test_non_prediction_text_is_a_noop():
    v = PredictionResolutionVerifier(ledger=_FakeLedger())
    res = asyncio.run(v.verify("Paris is the capital of France."))
    assert res.checked is False
    assert res.detail == {}


def test_resolution_grades_the_linked_foundry_verdict(tmp_path, monkeypatch):
    ledger = _FakeLedger()
    foundry = VerifierFoundry(root=tmp_path / "foundry")
    try:
        monkeypatch.setattr(
            "core.runtime.service_access.optional_service",
            lambda name, default=None: foundry if name == "verifier_foundry"
            else default,
        )
        v = PredictionResolutionVerifier(ledger=ledger)
        res = asyncio.run(v.verify("I predict that memory will stay flat.",
                                   context={"confidence": 0.7}))
        rid = res.detail["prediction_receipt"]
        vid = foundry.record_verdict(verifier=v.name, domain="prediction",
                                     hard_pass=True, score=0.7, checked=True)
        v.link_foundry_verdict(rid, vid)

        assert v.resolve(rid, happened=False)   # reality disagreed
        cell = foundry.reliability("prediction", "prediction")
        assert cell.graded == 1
        assert cell.false_passes == 1           # the optimistic verdict was wrong
        assert ledger.resolved[rid] == 0.0
        assert rid not in v.open_predictions()
    finally:
        foundry.close()


def test_resolution_of_unknown_receipt_is_false():
    v = PredictionResolutionVerifier(ledger=_FakeLedger())
    assert v.resolve("rcpt-nope", happened=True) is False


# ─────────────────────────────────────────────────────────────────────────────
# OutcomeLedgerVerifier
# ─────────────────────────────────────────────────────────────────────────────

def test_outcome_verifier_needs_history_before_checking():
    v = OutcomeLedgerVerifier()
    res = asyncio.run(v.verify("plan: refactor the parser",
                               context={"action_family": "refactor"}))
    assert res.checked is False
    for _ in range(4):
        v.record_outcome("refactor", success=True)
    res = asyncio.run(v.verify("plan", context={"action_family": "refactor"}))
    assert res.checked is False               # 4 < minimum history


def test_outcome_verifier_prices_by_empirical_rate():
    v = OutcomeLedgerVerifier()
    for i in range(10):
        v.record_outcome("deploy", success=(i < 8))
    res = asyncio.run(v.verify("plan: deploy tonight",
                               context={"action_family": "deploy"}))
    assert res.checked is True
    assert res.ok is True                      # priors price, never disprove
    assert res.score == pytest.approx(0.8, abs=0.01)
    assert res.detail["n"] == 10


def test_outcome_verifier_without_family_is_noop():
    v = OutcomeLedgerVerifier()
    res = asyncio.run(v.verify("do something"))
    assert res.checked is False


# ─────────────────────────────────────────────────────────────────────────────
# RubricEnsembleVerifier
# ─────────────────────────────────────────────────────────────────────────────

def test_rubric_without_scorer_refuses_to_pretend():
    v = RubricEnsembleVerifier(scorer=None)
    res = asyncio.run(v.verify("A lovely essay about rivers."))
    assert res.checked is False


def test_rubric_ensemble_scores_and_passes_good_text():
    v = RubricEnsembleVerifier(scorer=lambda text, q: 0.9, ensemble_k=3)
    res = asyncio.run(v.verify("A clear, grounded, consistent answer."))
    assert res.checked is True and res.ok is True
    assert res.score > 0.8
    assert set(res.detail["item_scores"]) >= {"clarity", "internal_consistency"}


def test_rubric_hard_fails_on_consistency_collapse():
    def scorer(text, question):
        return 0.05 if "self-contradiction" in question else 0.9

    v = RubricEnsembleVerifier(scorer=scorer, ensemble_k=3)
    res = asyncio.run(v.verify("The sky is green and also not green."))
    assert res.checked is True
    assert res.ok is False
    assert any("internal_consistency" in i for i in res.issues)


def test_rubric_median_tames_noisy_judges():
    calls = {"n": 0}

    def noisy(text, question):
        calls["n"] += 1
        return 0.0 if calls["n"] % 3 == 0 else 0.9   # one outlier per item

    v = RubricEnsembleVerifier(scorer=noisy, ensemble_k=3)
    res = asyncio.run(v.verify("Decent text."))
    assert res.ok is True
    assert res.score > 0.8                      # medians shrugged the outliers off


def test_rubric_accepts_context_items():
    v = RubricEnsembleVerifier(scorer=lambda t, q: 0.8, ensemble_k=1)
    res = asyncio.run(v.verify("text", context={"rubric_items": ["Cites sources?"]}))
    assert len(res.detail["item_scores"]) == 5


# ─────────────────────────────────────────────────────────────────────────────
# Registry hosts the new engines
# ─────────────────────────────────────────────────────────────────────────────

def test_registry_selects_experience_engines():
    from core.brain.verifiers.registry import VerifierRegistry

    registry = VerifierRegistry()
    assert any(v.name == "prediction" for v in registry.select("forecast"))
    assert any(v.name == "rubric" for v in registry.select("writing"))
    assert any(v.name == "outcome" for v in registry.select("planning"))
