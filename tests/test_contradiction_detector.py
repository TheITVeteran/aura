"""A claim contradicted itself, and a real contradiction slipped through.

Both found by the eval arena's truthfulness probe the first time it ran
anything real (2026-08-04). The rule was::

    diff = words1.symmetric_difference(words2)
    if diff.issubset(negation_words | {"optimized"}):
        return True

For two identical texts ``diff`` is empty and the empty set is a subset of
everything, so any claim of three or more words contradicted itself.
``detect_conflicts`` walks every pair in the graph, so every duplicated belief
was a logged conflict. In the other direction, "aura runs locally" against
"aura does not run locally" returned False, because ``runs``/``run``/``does``
were not on the permitted-difference list — which is to say almost every real
English negation escaped.

The permitted-difference list already contained "unoptimized", "high", "low"
and "latency": a single past example had become the rule. Contradiction is
structural — the same subject and predicate, asserted on one side and denied on
the other — and that is what is measured now.
"""

from __future__ import annotations

import pytest

from core.epistemics.contradiction_detector import ContradictionDetector


CONTRADICTIONS = [
    ("the retrieval path is optimized", "the retrieval path is not optimized"),
    ("aura runs locally", "aura does not run locally"),
    ("the cache is warm", "the cache is not warm"),
    ("latency is high", "latency is low"),
    ("memory usage increases under load", "memory usage decreases under load"),
    ("the sandbox is safe", "the sandbox is unsafe"),
    ("the migration succeeded", "the migration failed"),
    ("the answer is correct", "the answer is incorrect"),
    ("the model runs on device", "the model does not run on device"),
]

NOT_CONTRADICTIONS = [
    # The defect: a claim against itself.
    ("the retrieval path is optimized", "the retrieval path is optimized"),
    ("aura runs locally", "aura runs locally"),
    # Different subjects, however they are worded.
    ("the retrieval path is optimized", "lisbon sits on the atlantic coast"),
    ("latency is high", "throughput is low"),
    ("the cache is warm", "the disk is not full"),
    ("memory usage increases under load", "cpu usage decreases at idle"),
    ("the migration succeeded", "the backup succeeded"),
    ("i like cats", "i like dogs"),
    ("the build is fast", "the release notes are long"),
]


@pytest.mark.parametrize(("left", "right"), CONTRADICTIONS)
def test_a_denial_of_the_same_claim_is_a_contradiction(left, right):
    assert ContradictionDetector.are_contradictory(left, right) is True
    # Contradiction is symmetric.
    assert ContradictionDetector.are_contradictory(right, left) is True


@pytest.mark.parametrize(("left", "right"), NOT_CONTRADICTIONS)
def test_agreement_and_unrelated_claims_are_not_contradictions(left, right):
    assert ContradictionDetector.are_contradictory(left, right) is False
    assert ContradictionDetector.are_contradictory(right, left) is False


def test_a_claim_never_contradicts_itself():
    """Every duplicated belief in the graph was a logged conflict."""
    for text in (
        "the retrieval path is optimized",
        "memory pressure is elevated right now",
        "my code is my body; it must be maintained",
    ):
        assert ContradictionDetector.are_contradictory(text, text) is False


def test_empty_text_is_not_a_contradiction():
    assert ContradictionDetector.are_contradictory("", "anything at all") is False
    assert ContradictionDetector.are_contradictory("anything at all", "") is False
    assert ContradictionDetector.are_contradictory("", "") is False


def test_detect_conflicts_does_not_flag_duplicates_in_a_graph():
    class _Node:
        def __init__(self, claim_id, text):
            self.claim_id = claim_id
            self.text = text

    class _Graph:
        def __init__(self, nodes):
            self.nodes = {n.claim_id: n for n in nodes}

    graph = _Graph(
        [
            _Node("c1", "the retrieval path is optimized"),
            _Node("c2", "the retrieval path is optimized"),
            _Node("c3", "the retrieval path is not optimized"),
            _Node("c4", "lisbon sits on the atlantic coast"),
        ]
    )
    conflicts = ContradictionDetector.detect_conflicts(graph)
    pairs = {tuple(sorted((a, b))) for a, b, _ in conflicts}
    assert ("c1", "c3") in pairs
    assert ("c2", "c3") in pairs
    assert ("c1", "c2") not in pairs, "identical claims are not a conflict"
    assert all("c4" not in pair for pair in pairs)


def test_the_eval_arena_truthfulness_probe_passes():
    from core.evals.eval_arena import _probe_contradiction_detection

    outcome = _probe_contradiction_detection()
    assert outcome.measured is True
    assert outcome.passed is True, outcome.evidence
