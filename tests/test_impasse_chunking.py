"""Impasse detection and chunking, including the utility problem.

Soar's chunking is easy to implement and easy to implement wrongly. The two
documented ways it goes wrong both have tests here rather than a comment:

* the utility problem — learned rules add match cost to every decision, so
  indiscriminate learning makes an experienced system slower than a naive one;
* over-generalisation — a chunk fires outside the situation that produced it
  and is confidently wrong.

``ChunkStore.prune`` catches both with one derived rule, expected value <= 0,
and ``test_an_expensive_chunk_is_retracted`` /
``test_an_over_general_chunk_is_retracted`` are the two populations.
"""

from __future__ import annotations

import pytest

from core.cognition.impasse import (
    ChunkStore,
    Impasse,
    ImpasseType,
    classify,
    situation_signature,
)


# --------------------------------------------------------------------------
# Detection
# --------------------------------------------------------------------------


def test_a_clear_winner_is_not_an_impasse():
    assert classify(["a", "b"], scores={"a": 0.9, "b": 0.4}) is None


def test_equal_scores_are_a_tie_impasse():
    """The workspace case: sorting and taking [0] hides this completely."""
    imp = classify(["a", "b", "c"], scores={"a": 0.9, "b": 0.9, "c": 0.2})
    assert imp is not None
    assert imp.type is ImpasseType.TIE
    assert imp.candidates == ("a", "b")


def test_near_ties_need_a_tolerance_to_count():
    scores = {"a": 0.900, "b": 0.899}
    assert classify(["a", "b"], scores=scores) is None
    imp = classify(["a", "b"], scores=scores, tolerance=0.01)
    assert imp is not None and imp.type is ImpasseType.TIE


def test_no_candidates_is_rejection_not_tie():
    imp = classify([], scores={})
    assert imp is not None and imp.type is ImpasseType.REJECTION
    assert "no candidates were proposed" in imp.detail


def test_all_rejected_is_rejection():
    imp = classify(["a", "b"], scores={"a": 1.0, "b": 1.0}, rejected=["a", "b"])
    assert imp is not None and imp.type is ImpasseType.REJECTION
    assert "every candidate was rejected" in imp.detail


def test_contradictory_preferences_are_a_conflict():
    imp = classify(["a", "b"], scores={"a": 1.0, "b": 1.0}, preferences=[("a", "b"), ("b", "a")])
    assert imp is not None and imp.type is ImpasseType.CONFLICT


def test_conflict_outranks_tie():
    """Contradictory preferences are a worse failure than an absent one."""
    imp = classify(
        ["a", "b"],
        scores={"a": 0.5, "b": 0.5},
        preferences=[("a", "b"), ("b", "a")],
    )
    assert imp is not None and imp.type is ImpasseType.CONFLICT


def test_a_choice_that_changes_nothing_is_a_no_change_impasse():
    imp = classify(["a"], scores={"a": 1.0}, changed=False)
    assert imp is not None and imp.type is ImpasseType.NO_CHANGE


def test_changed_true_is_not_an_impasse():
    assert classify(["a"], scores={"a": 1.0}, changed=True) is None


def test_signature_separates_situations_with_the_same_options():
    """Without context in the key, a chunk over-generalises across situations."""
    a = situation_signature({"mode": "idle"}, ["x", "y"])
    b = situation_signature({"mode": "urgent"}, ["x", "y"])
    assert a != b


def test_signature_is_order_independent():
    assert situation_signature({"a": 1, "b": 2}, ["y", "x"]) == situation_signature(
        {"b": 2, "a": 1}, ["x", "y"]
    )


# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------


def _tie() -> Impasse:
    imp = classify(["a", "b"], scores={"a": 0.5, "b": 0.5}, context={"mode": "idle"})
    assert imp is not None
    return imp


def test_a_chunk_resolves_the_same_impasse_next_time():
    store = ChunkStore()
    imp = store.record_impasse(_tie())
    store.learn(imp, "a", cost_saved_per_use=0.20, match_cost=0.001)

    hit = store.recall(imp.signature)
    assert hit is not None and hit.resolution == "a"
    assert hit.uses == 1


def test_an_unseen_situation_has_no_chunk():
    store = ChunkStore()
    assert store.recall("never-seen") is None


def test_relearning_the_same_resolution_keeps_the_statistics():
    store = ChunkStore()
    imp = _tie()
    store.learn(imp, "a", cost_saved_per_use=0.2, match_cost=0.001)
    store.recall(imp.signature)
    store.record_outcome(imp.signature, correct=True)

    store.learn(imp, "a", cost_saved_per_use=0.3, match_cost=0.001)
    chunk = store.chunks()[0]
    assert chunk.correct == 1, "evidence about this answer was thrown away"
    assert chunk.cost_saved_per_use == 0.3


def test_relearning_a_different_resolution_resets_the_statistics():
    """The evidence was gathered about a different answer and does not transfer."""
    store = ChunkStore()
    imp = _tie()
    store.learn(imp, "a", cost_saved_per_use=0.2, match_cost=0.001)
    store.record_outcome(imp.signature, correct=True)
    store.learn(imp, "b", cost_saved_per_use=0.2, match_cost=0.001)
    chunk = store.chunks()[0]
    assert chunk.resolution == "b"
    assert chunk.correct == 0 and chunk.incorrect == 0


def test_recording_an_outcome_for_an_unknown_chunk_is_harmless():
    ChunkStore().record_outcome("nothing", correct=False)


# --------------------------------------------------------------------------
# The utility problem
# --------------------------------------------------------------------------


def test_a_paying_chunk_is_kept():
    store = ChunkStore()
    imp = _tie()
    store.learn(imp, "a", cost_saved_per_use=0.20, match_cost=0.001)
    assert store.prune() == []
    assert len(store.chunks()) == 1


def test_an_expensive_chunk_is_retracted():
    """Population one: match cost exceeds what it saves.

    This is the utility problem in miniature — the chunk is perfectly correct
    and still makes the system slower.
    """
    store = ChunkStore()
    imp = _tie()
    chunk = store.learn(imp, "a", cost_saved_per_use=0.001, match_cost=0.010)
    assert chunk.p_correct == 1.0, "the chunk is not wrong, it is expensive"
    retracted = store.prune()
    assert [c.signature for c in retracted] == [imp.signature]
    assert store.chunks() == []


def test_an_over_general_chunk_is_retracted():
    """Population two: cheap to match, but wrong often enough to stop paying."""
    store = ChunkStore()
    imp = _tie()
    store.learn(imp, "a", cost_saved_per_use=0.010, match_cost=0.005)
    for _ in range(7):
        store.record_outcome(imp.signature, correct=False)
    for _ in range(3):
        store.record_outcome(imp.signature, correct=True)
    # EV = 0.3*0.010 - 0.005 = -0.002
    assert store.prune()
    assert store.chunks() == []


def test_a_mostly_correct_cheap_chunk_survives():
    store = ChunkStore()
    imp = _tie()
    store.learn(imp, "a", cost_saved_per_use=0.010, match_cost=0.001)
    store.record_outcome(imp.signature, correct=False)
    for _ in range(9):
        store.record_outcome(imp.signature, correct=True)
    # EV = 0.9*0.010 - 0.001 = +0.008
    assert store.prune() == []


def test_total_match_cost_makes_the_slowdown_visible():
    """The number that turns 'learning is free' into a measurement."""
    store = ChunkStore()
    for i in range(50):
        imp = classify(
            ["a", "b"], scores={"a": 0.5, "b": 0.5}, context={"i": i}
        )
        assert imp is not None
        store.learn(imp, "a", cost_saved_per_use=0.0005, match_cost=0.001)
    assert store.total_match_cost() == pytest.approx(0.05)
    assert store.net_value() < 0.0, "50 non-paying chunks should show as net negative"
    assert len(store.prune()) == 50


def test_retraction_records_why():
    store = ChunkStore()
    imp = _tie()
    store.learn(imp, "a", cost_saved_per_use=0.001, match_cost=0.010)
    store.prune()
    report = store.report()
    assert report["chunks"] == 0
    assert len(report["retracted"]) == 1
    assert "EV=" in report["retracted"][0][1]


def test_impasse_counts_are_reported_by_type():
    store = ChunkStore()
    store.record_impasse(_tie())
    rejection = classify([], scores={})
    assert rejection is not None
    store.record_impasse(rejection)
    counts = store.impasse_counts()
    assert counts["tie"] == 1
    assert counts["rejection"] == 1
    assert counts["conflict"] == 0
