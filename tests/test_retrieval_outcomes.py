"""Memories that can be found harmful.

Every other signal in retrieval is positive or neutral, so a memory that
consistently misled could only be out-competed, never demoted. These tests
pin the one input that pushes downward — and the limits that stop it
becoming a way to make inconvenient memories disappear.
"""
from __future__ import annotations

import pytest

from core.memory.intentional_retrieval import MemoryHit
from core.memory.retrieval_outcomes import (
    MemoryCategory,
    OutcomeLedger,
    RetrievalVerdict,
    apply_influence,
    get_outcome_ledger,
)


@pytest.fixture
def ledger() -> OutcomeLedger:
    return OutcomeLedger()


@pytest.fixture(autouse=True)
def _clean_global():
    get_outcome_ledger().reset_for_test()
    yield
    get_outcome_ledger().reset_for_test()


def test_an_ungraded_memory_has_no_influence(ledger):
    ledger.note_retrieved("m1")
    assert ledger.influence("m1") == 1.0


def test_influence_waits_for_enough_gradings(ledger):
    """Two bad outcomes is unlucky; it is not yet evidence."""
    for _ in range(2):
        ledger.grade("m1", RetrievalVerdict.HARMFUL)
    assert ledger.influence("m1") == 1.0


def test_repeated_harm_demotes_a_memory(ledger):
    for _ in range(5):
        ledger.grade("m1", RetrievalVerdict.HARMFUL)
    assert ledger.influence("m1") < 1.0


def test_repeated_help_promotes_a_memory(ledger):
    for _ in range(5):
        ledger.grade("m1", RetrievalVerdict.HELPFUL)
    assert ledger.influence("m1") > 1.0


def test_a_harmful_memory_is_never_suppressed_to_zero(ledger):
    """Unfindable is also unattributable; the record must survive."""
    for _ in range(200):
        ledger.grade("m1", RetrievalVerdict.HARMFUL)
    assert ledger.influence("m1") == pytest.approx(0.35, abs=1e-6)
    assert ledger.influence("m1") > 0.0


def test_harm_pulls_harder_than_help_lifts(ledger):
    for _ in range(10):
        ledger.grade("helpful", RetrievalVerdict.HELPFUL)
    for _ in range(10):
        ledger.grade("harmful", RetrievalVerdict.HARMFUL)
    lift = ledger.influence("helpful") - 1.0
    drop = 1.0 - ledger.influence("harmful")
    assert drop > lift


def test_neutral_gradings_count_toward_the_sample(ledger):
    for _ in range(3):
        ledger.grade("m1", RetrievalVerdict.NEUTRAL)
    assert ledger.stats_for("m1").graded == 3
    assert ledger.influence("m1") == 1.0


def test_silence_is_not_recorded_as_help(ledger):
    """Retrieval alone must never improve a memory's standing."""
    for _ in range(50):
        ledger.note_retrieved("m1")
    stats = ledger.stats_for("m1")
    assert stats.retrieved == 50
    assert stats.graded == 0
    assert ledger.influence("m1") == 1.0


def test_harmful_memories_are_listable(ledger):
    for _ in range(4):
        ledger.grade("bad", RetrievalVerdict.HARMFUL)
    for _ in range(4):
        ledger.grade("good", RetrievalVerdict.HELPFUL)
    listed = [m["key"] for m in ledger.harmful_memories()]
    assert listed == ["bad"]


def test_pitfall_is_a_first_class_category(ledger):
    ledger.note_retrieved("p1", category=MemoryCategory.PITFALL)
    ledger.note_retrieved("e1", category=MemoryCategory.EPISODE)
    assert [p["key"] for p in ledger.pitfalls()] == ["p1"]


def test_unknown_category_is_ignored_not_fatal(ledger):
    ledger.note_retrieved("m1", category="nonsense")
    assert ledger.stats_for("m1").category is MemoryCategory.EPISODE


def test_unknown_verdict_is_ignored(ledger):
    assert ledger.grade("m1", "sideways") is None


def test_apply_influence_reranks_by_track_record():
    ledger = get_outcome_ledger()
    for _ in range(6):
        ledger.grade("misleading", RetrievalVerdict.HARMFUL)
    hits = [
        MemoryHit(content="a", score=0.9, store_type="episodic", source="misleading"),
        MemoryHit(content="b", score=0.8, store_type="episodic", source="fine"),
    ]
    ranked = apply_influence(hits)
    # The higher raw score loses to the clean one once harm is counted.
    assert [h.source for h in ranked] == ["fine", "misleading"]


def test_apply_influence_leaves_untracked_hits_alone():
    hits = [
        MemoryHit(content="a", score=0.9, store_type="episodic", source="x"),
        MemoryHit(content="b", score=0.8, store_type="episodic", source="y"),
    ]
    ranked = apply_influence(hits)
    assert [h.score for h in ranked] == [0.9, 0.8]


def test_apply_influence_survives_a_malformed_hit():
    class Broken:
        score = "not a number"

    ranked = apply_influence([Broken()])
    assert len(ranked) == 1


def test_retriever_merge_applies_the_ledger():
    """The wiring, not just the module: a demoted memory falls in real ranking."""
    from core.memory.intentional_retrieval import IntentionalRetriever

    ledger = get_outcome_ledger()
    for _ in range(6):
        ledger.grade("bad-source", RetrievalVerdict.HARMFUL)
    hits = [
        MemoryHit(content="strong but harmful", score=0.95, store_type="episodic",
                  source="bad-source"),
        MemoryHit(content="weaker but clean", score=0.7, store_type="episodic",
                  source="ok-source"),
    ]
    merged = IntentionalRetriever._merge(hits, limit=2)
    assert merged[0].source == "ok-source"
