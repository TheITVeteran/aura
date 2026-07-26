"""CP126 hardening contracts for core/brain/verifiers/citation_engine.py.

This engine decides whether a claim is grounded, so its own honesty matters:
`checked` must mean a claim was ACTUALLY tested, grounding must not be satisfied
by substring accidents, arbitrary objects must not become evidence, hedged
claims must still be contradiction-checked, and a corpus retrieval FAILURE must
be distinguishable from an empty corpus.
"""
from __future__ import annotations

import asyncio

import core.brain.verifiers.citation_engine as ce
from core.brain.verifiers.citation_engine import (
    CitationEngine,
    _coerce_evidence,
    _contradicts,
    _grounded,
)


def _verify(candidate, **ctx):
    return asyncio.run(CitationEngine().verify(candidate, context=ctx))


# ── de6c00b6: `checked` means a claim was actually examined ───────────────


def test_checked_is_false_when_no_claim_overlapped_evidence():
    # Confident-sounding sentences exist, but they share nothing with the
    # evidence — nothing was actually tested, so `checked` must be False.
    res = _verify(
        "Jupiter is the largest planet. Saturn has beautiful rings.",
        evidence=["The retry budget is three attempts, then it fails closed."],
    )
    assert res.checked is False


def test_checked_is_true_when_a_claim_overlapped_evidence():
    res = _verify(
        "The retry budget is three attempts.",
        evidence=["The retry budget is three attempts, then it fails closed."],
    )
    assert res.checked is True


# ── 5a486091: grounding uses whole tokens, not substrings ─────────────────


def test_substring_accident_is_not_grounding():
    # "cat"/"age" appear inside "catalogue"/"storage" but are not the same word.
    tokens = ce._content_words("the catalogue lives in storage")
    assert _grounded("cats age", tokens) is False


def test_real_token_overlap_still_grounds():
    tokens = ce._content_words("the retry budget is three attempts")
    assert _grounded("the retry budget is three attempts", tokens) is True


# ── 5cbb1ea0: arbitrary objects do not become evidence ────────────────────


def test_arbitrary_objects_are_dropped_not_stringified():
    class Thing:
        pass

    items, dropped = _coerce_evidence([Thing(), object(), "real evidence text"])
    assert items == ["real evidence text"]
    assert dropped == 2


def test_mapping_evidence_uses_text_fields():
    items, dropped = _coerce_evidence([{"title": "T", "snippet": "the budget is three"}])
    assert items == ["T: the budget is three"] and dropped == 0


def test_dropped_evidence_is_disclosed():
    class Thing:
        pass

    res = _verify(
        "The retry budget is three attempts.",
        evidence=[Thing(), "The retry budget is three attempts, then it fails closed."],
    )
    assert res.detail["dropped_evidence_items"] == 1
    assert any("dropped" in i for i in res.issues)


# ── ac741810: hedged claims are still contradiction-checked ───────────────


def test_hedged_claim_is_still_contradiction_checked():
    res = _verify(
        "I think the retry budget is unlimited and reboots forever.",
        evidence=["The retry budget is three attempts, then it fails closed."],
    )
    assert res.ok is False, "a hedge must not exempt a contradicting claim"
    assert res.detail["hedged_claims"] == 1


def test_hedged_claim_is_not_penalised_for_being_ungrounded():
    res = _verify(
        "I think Jupiter is the largest planet.",
        evidence=["The retry budget is three attempts."],
    )
    assert res.ok is True  # hedged + merely unsupported → not a failure
    assert res.detail["ungrounded"] == 0


# ── 21dc697f: a bare polarity flip needs a wider shared subject ───────────


def test_thin_negation_overlap_is_not_a_contradiction():
    # Only 2 shared content words and a negation mismatch — too weak to call
    # a contradiction now.
    assert _contradicts("the retry budget works", ["the retry budget is not documented"]) is False


def test_absolute_vs_bounded_still_contradicts_at_low_overlap():
    assert _contradicts(
        "The retry budget is unlimited and reboots forever.",
        ["The retry budget is three attempts, then it fails closed."],
    ) is True


# ── e18291df: retrieval failure is distinguishable from absence ───────────


def test_corpus_failure_is_reported_distinctly(monkeypatch):
    from core.knowledge import local_corpus

    def _boom(*a, **k):
        raise RuntimeError("corpus unavailable")

    monkeypatch.setattr(local_corpus, "get_local_corpus_store", _boom)
    res = _verify("The retry budget is three attempts.", objective="what is the retry budget")
    assert res.checked is False
    assert res.detail["evidence_retrieval_failed"] is True
    assert any("retrieval failed" in i for i in res.issues)


def test_empty_corpus_is_not_reported_as_failure(monkeypatch):
    from core.knowledge import local_corpus

    class _Empty:
        def search(self, *a, **k):
            return []

    monkeypatch.setattr(local_corpus, "get_local_corpus_store", lambda *a, **k: _Empty())
    res = _verify("The retry budget is three attempts.", objective="what is the retry budget")
    assert res.detail["evidence_retrieval_failed"] is False
    assert res.ok is True and res.checked is False


# ── 406fff00 / 961804bb: work bounds and disclosed truncation ─────────────


def test_evidence_items_are_bounded(monkeypatch):
    monkeypatch.setattr(ce, "_MAX_EVIDENCE_ITEMS", 3)
    items, dropped = _coerce_evidence(["e"] * 10)
    assert len(items) == 3 and dropped == 7


def test_truncated_issue_lists_are_disclosed():
    many = ". ".join(f"Fact number {i} is definitely wrong here" for i in range(20)) + "."
    res = _verify(many, evidence=["Totally unrelated grounding evidence about pears."])
    assert any("[truncated]" in i for i in res.issues)


def test_sentence_count_is_bounded(monkeypatch):
    monkeypatch.setattr(ce, "_MAX_SENTENCES", 5)
    assert len(ce._sentences(". ".join(["This is a sentence here"] * 50) + ".")) == 5
