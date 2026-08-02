"""Saying a thing again is not evidence for it.

CP126: "Repeated unverified claims self-promote to trusted belief. Matching
an existing value increments confidence by a fixed amount until trusted
status, without requiring independent evidence, source diversity, or a
verifier receipt."

Four repetitions walked 0.35 -> 0.47 -> 0.59 -> 0.71 -> 0.83 and crossed the
0.75 trusted line. A belief system in which repetition is evidence will
believe whatever it is told most often.
"""
from __future__ import annotations

from core.constitution import (
    BELIEF_MIN_SOURCES_FOR_TRUST,
    BELIEF_UNCORROBORATED_CEILING,
    BeliefAuthority,
)


def test_repetition_alone_never_reaches_trusted():
    authority = BeliefAuthority()
    for _ in range(25):
        record = authority.review_update("ns", "claim", "unsupported")
    assert record.status != "trusted"
    assert record.confidence <= BELIEF_UNCORROBORATED_CEILING


def test_repetition_alone_does_not_raise_confidence_at_all():
    """The increment itself required new evidence, not just a match."""
    authority = BeliefAuthority()
    first = authority.review_update("ns", "claim", "unsupported")
    for _ in range(10):
        later = authority.review_update("ns", "claim", "unsupported")
    assert later.confidence == first.confidence
    assert "repeated_without_new_evidence" in later.reason


def test_resubmitting_the_same_citation_is_not_new_evidence():
    """A source repeating itself is still one source."""
    authority = BeliefAuthority()
    for _ in range(10):
        record = authority.review_update("ns", "claim", "value", evidence=["paper-1"])
    assert record.status != "trusted"
    assert "repeated_without_new_evidence" in record.reason


def test_independent_sources_do_reach_trusted():
    """The control: corroboration must still work, or this is just a block."""
    authority = BeliefAuthority()
    for source in ("a", "b", "c", "d", "e"):
        record = authority.review_update(
            "ns", "claim", "value", evidence=[f"sensor-{source}"]
        )
    assert record.status == "trusted"
    assert record.confidence > BELIEF_UNCORROBORATED_CEILING


def test_trust_needs_more_than_one_distinct_source():
    authority = BeliefAuthority()
    record = authority.review_update("ns", "claim", "value", evidence=["only-source"])
    assert record.status != "trusted"
    assert BELIEF_MIN_SOURCES_FOR_TRUST >= 2


def test_accumulated_evidence_is_retained_across_updates():
    """Distinct-source counting needs the union, not just the latest batch."""
    authority = BeliefAuthority()
    authority.review_update("ns", "claim", "value", evidence=["a"])
    record = authority.review_update("ns", "claim", "value", evidence=["b"])
    assert set(record.evidence) >= {"a", "b"}


def test_a_contradicting_value_still_contests_rather_than_reinforces():
    """The fix must not accidentally make contradictions look like support."""
    authority = BeliefAuthority()
    authority.review_update("ns", "claim", "first", evidence=["a"])
    record = authority.review_update("ns", "claim", "second", evidence=["b"])
    assert record.status in {"contested", "trusted"}
    assert "reinforced" not in record.reason
