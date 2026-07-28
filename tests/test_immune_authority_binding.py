"""A Will approval must cover the action, not its category.

CP126 81f0c6a0: Will saw component, kind, danger and lineage — so an
approval for "restart component X" covered any payload that later arrived
under that description. The exact parameters, the behavioural rule that
produced them, the cell generation and the repair strategy were all outside
the decision.
"""
from __future__ import annotations

import pytest

from core.adaptation import adaptive_immunity as mod
from core.adaptation.adaptive_immunity import (
    Antigen,
    EffectorArtifact,
    EffectorKind,
    _artifact_payload_digest,
    get_adaptive_immune_system,
)


def _artifact(payload=None, kind=EffectorKind.CLEAR_CACHE, component="memory"):
    return EffectorArtifact(
        artifact_id="a1",
        kind=kind,
        component=component,
        confidence=0.8,
        source_cell_id="c1",
        lineage_id="l1",
        bounded_payload=payload if payload is not None else {"scope": "warm"},
    )


def _antigen():
    return Antigen.from_dict(
        {"antigen_id": "ag", "subsystem": "memory", "vector": [0.5] * 16, "danger": 0.6}
    )


# --- the digest identifies the concrete effect ---------------------------


def test_the_same_payload_digests_the_same():
    assert _artifact_payload_digest(_artifact()) == _artifact_payload_digest(_artifact())


def test_a_different_payload_digests_differently():
    assert _artifact_payload_digest(_artifact({"scope": "warm"})) != _artifact_payload_digest(
        _artifact({"scope": "ALL"})
    )


def test_a_different_component_digests_differently():
    assert _artifact_payload_digest(_artifact(component="memory")) != _artifact_payload_digest(
        _artifact(component="scheduler")
    )


def test_a_different_kind_digests_differently():
    assert _artifact_payload_digest(
        _artifact(kind=EffectorKind.CLEAR_CACHE)
    ) != _artifact_payload_digest(_artifact(kind=EffectorKind.HALT_RUNAWAY))


def test_an_unserializable_payload_still_digests():
    assert _artifact_payload_digest(_artifact({"obj": object()}))


# --- the approval is bound to it -----------------------------------------


def test_the_payload_reaches_the_will_decision(monkeypatch):
    captured = {}

    class _Decision:
        @staticmethod
        def is_approved():
            return True

    class _Will:
        @staticmethod
        def decide(**kwargs):
            captured.update(kwargs)
            return _Decision()

    monkeypatch.setattr("core.will.get_will", lambda: _Will())
    immune = get_adaptive_immune_system()

    assert immune._authorize_protected_action(_artifact(), _antigen()) is True

    evidence = captured["context"].get("evidence", captured["context"])
    rendered = str(evidence)
    assert "payload_digest" in rendered
    assert "repair_strategy" in rendered
    assert "source_cell_generation" in rendered


def test_a_payload_changed_after_approval_is_refused(monkeypatch):
    """The approval described a different effect than the one that would run."""
    artifact = _artifact()

    class _Decision:
        @staticmethod
        def is_approved():
            # Mutate between the decision and the post-check.
            artifact.bounded_payload["scope"] = "ALL"
            return True

    monkeypatch.setattr(
        "core.will.get_will", lambda: type("W", (), {"decide": staticmethod(lambda **k: _Decision())})()
    )
    immune = get_adaptive_immune_system()

    assert immune._authorize_protected_action(artifact, _antigen()) is False


def test_a_denied_decision_is_still_denied(monkeypatch):
    class _Decision:
        @staticmethod
        def is_approved():
            return False

    monkeypatch.setattr(
        "core.will.get_will", lambda: type("W", (), {"decide": staticmethod(lambda **k: _Decision())})()
    )
    immune = get_adaptive_immune_system()

    assert immune._authorize_protected_action(_artifact(), _antigen()) is False


def test_unavailable_authority_fails_closed(monkeypatch):
    monkeypatch.setattr(
        "core.will.get_will", lambda: (_ for _ in ()).throw(RuntimeError("will down"))
    )
    immune = get_adaptive_immune_system()

    assert immune._authorize_protected_action(_artifact(), _antigen()) is False


def test_a_missing_cell_reports_generation_minus_one():
    immune = get_adaptive_immune_system()

    assert immune._cell_generation("not-a-real-cell") == -1
