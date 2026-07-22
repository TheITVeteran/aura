"""CP126 hardening contracts for core/adaptation/dynamic_value_graph.py.

This is the machinery by which Aura's VALUES change from evidence, so the
gates are integrity boundaries: a value must not gain trust from corrupt
state, from evidence against it, or from non-finite numbers.
"""
from __future__ import annotations

from core.adaptation.dynamic_value_graph import (
    EvidenceType,
    ValueEvidence,
    ValueNode,
    ValueNodeStatus,
    _finite,
)


def _evidence(node: ValueNode, *, signal: float, count: int = 12, conf: float = 0.9):
    for index in range(count):
        node.add_evidence(
            ValueEvidence(
                EvidenceType.OUTCOME_QUALITY,
                node.name,
                signal,
                conf,
                f"source-{index % 4}",
                "ctx",
            )
        )


class TestCorruptStateFailsClosed:
    def test_unknown_persisted_status_becomes_candidate_not_adopted(self):
        node = ValueNode.from_dict({"name": "y", "weight": 0.5, "status": "bogus"})
        # ADOPTED is the most-trusted state; corrupting one field must not
        # promote an unvetted value past every pipeline gate.
        assert node.status is ValueNodeStatus.CANDIDATE
        assert node.status is not ValueNodeStatus.ADOPTED

    def test_known_status_is_preserved(self):
        node = ValueNode.from_dict({"name": "y", "weight": 0.5, "status": "adopted"})
        assert node.status is ValueNodeStatus.ADOPTED


class TestNegativeEvidenceCannotPromote:
    def test_candidate_with_negative_evidence_is_not_promoted(self):
        from core.adaptation.dynamic_value_graph import DynamicValueGraph

        graph = DynamicValueGraph.__new__(DynamicValueGraph)
        node = ValueNode(name="x", weight=0.5, status=ValueNodeStatus.CANDIDATE)
        _evidence(node, signal=-0.8)
        delta, confidence = node.compute_evidence_delta()
        assert delta < 0.0, "fixture must produce negative evidence"

        graph.MIN_EVIDENCE = 5
        assert graph._process_candidate(node, delta, confidence) is None
        assert node.status is ValueNodeStatus.CANDIDATE

    def test_candidate_with_positive_evidence_is_promoted(self):
        from core.adaptation.dynamic_value_graph import DynamicValueGraph

        graph = DynamicValueGraph.__new__(DynamicValueGraph)
        graph.MIN_EVIDENCE = 5
        node = ValueNode(name="x", weight=0.5, status=ValueNodeStatus.CANDIDATE)
        _evidence(node, signal=0.8)
        delta, confidence = node.compute_evidence_delta()
        assert delta > 0.0

        mutation = graph._process_candidate(node, delta, confidence)
        assert mutation is not None
        assert node.status is ValueNodeStatus.SANDBOX

    def test_provisional_does_not_adopt_on_negative_evidence(self):
        import time

        from core.adaptation.dynamic_value_graph import DynamicValueGraph

        graph = DynamicValueGraph.__new__(DynamicValueGraph)
        node = ValueNode(
            name="x",
            weight=0.5,
            status=ValueNodeStatus.PROVISIONAL,
            rollback_deadline=time.time() - 10.0,
        )
        _evidence(node, signal=-0.8)
        delta, confidence = node.compute_evidence_delta()

        mutation = graph._process_provisional(node, delta, confidence)
        # It may still record a conservative adjustment, but it must NOT adopt.
        assert node.status is ValueNodeStatus.PROVISIONAL
        if mutation is not None:
            assert mutation.mutation_type != "adopted"


class TestNonFiniteEvidenceCannotPoisonValues:
    def test_nan_evidence_yields_a_finite_delta(self):
        node = ValueNode(name="z", weight=0.5)
        _evidence(node, signal=float("nan"), conf=float("nan"), count=10)
        delta, confidence = node.compute_evidence_delta()
        assert delta == delta  # not NaN
        assert confidence == confidence

    def test_finite_helper_bounds_and_rejects(self):
        assert _finite(float("nan"), 0.0, low=-1.0, high=1.0) == 0.0
        assert _finite(float("inf"), 0.0, low=-1.0, high=1.0) == 0.0
        assert _finite(50.0, 0.0, low=-1.0, high=1.0) == 1.0
        assert _finite("junk", 0.25, low=0.0, high=1.0) == 0.25


def test_confidence_is_not_applied_twice():
    """weighted_signal() already multiplies by confidence.

    The old accumulation multiplied by confidence again, so the effective
    weighting was signal*confidence^2 over sum-of-confidence. At a uniform
    confidence c, the correct confidence-weighted mean of a constant signal
    is that signal exactly; the squared form returned signal*c.
    """
    node = ValueNode(name="c", weight=0.5)
    _evidence(node, signal=1.0, conf=0.5, count=20)
    delta, _confidence = node.compute_evidence_delta()

    # mean_signal must be 1.0 (not 0.5), and delta = clamp(mean*0.1).
    assert delta > 0.09, f"expected mean_signal≈1.0 -> delta≈0.1, got {delta}"
