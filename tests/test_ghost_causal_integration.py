"""Tests for core/ghost/causal_integration.py — system-Φ over the real
inter-subsystem consequence stream.

These pin the honest "unity vs federation" instrument: an interleaved,
feedback-cyclic stream of subsystem activity must read as integrated; a
single-organ or block-structured stream must read as federated. They also fix
the information-theoretic primitives (entropy, mutual information, bipartition
enumeration) so the irreducibility core cannot silently drift.
"""
from __future__ import annotations

import math

import pytest

from core.ghost.causal_integration import (
    SystemIntegration,
    SystemIntegrationReport,
    _entropy,
    _mutual_information,
    get_system_integration,
    reset_system_integration,
)
from core.runtime.consequence_bus import ConsequenceBus


@pytest.fixture()
def bus():
    ConsequenceBus.reset()
    b = ConsequenceBus.get()
    yield b
    ConsequenceBus.reset()
    reset_system_integration()


def _publish(bus, source: str, domain: str = "cognition") -> None:
    bus.publish_action(source=source, domain=domain, action_content=f"{source}:{domain}")


# ── information primitives ───────────────────────────────────────────────────

def test_entropy_uniform_and_degenerate():
    assert _entropy([]) == 0.0
    assert _entropy(["a", "a", "a"]) == 0.0
    assert _entropy(["a", "b"]) == pytest.approx(1.0)  # 1 bit
    assert _entropy(["a", "b", "c", "d"]) == pytest.approx(2.0)


def test_mutual_information_dependent_vs_independent():
    # Perfectly dependent: MI equals the entropy of the shared variable.
    xs = [0, 1, 0, 1, 0, 1]
    assert _mutual_information(xs, xs) == pytest.approx(_entropy(xs), abs=1e-9)
    # Independent: MI ~ 0.
    a = [0, 0, 1, 1]
    b = [0, 1, 0, 1]
    assert _mutual_information(a, b) == pytest.approx(0.0, abs=1e-9)
    # Mismatched lengths are handled, not crashed.
    assert _mutual_information([1, 2], [1]) == 0.0


def test_partition_masks_enumeration_is_deduplicated():
    eng = SystemIntegration()
    for m in range(2, 8):
        masks = eng._partition_masks(m)
        # Each partition counted once (item 0 fixed to side A), no all-A mask.
        assert len(masks) == (1 << (m - 1)) - 1
        assert all(mask & 1 for mask in masks)
        assert all(mask != (1 << m) - 1 for mask in masks)
    assert eng._partition_masks(1) == []


# ── the instrument ───────────────────────────────────────────────────────────

def test_insufficient_history_is_honest(bus):
    eng = SystemIntegration(bus=bus, min_events=8)
    _publish(bus, "affect")
    _publish(bus, "memory")
    rep = eng.report()
    assert isinstance(rep, SystemIntegrationReport)
    assert rep.label == "insufficient_history"
    assert rep.phi_system == 0.0
    assert rep.events == 2


def test_interleaved_feedback_stream_reads_integrated(bus):
    # A rotating hand-off across five organs: every consecutive pair crosses a
    # boundary (cross≈1) and the rotation closes a loop (recurrence≈1).
    organs = ["affect", "memory", "will", "world_model", "self_model"]
    for i in range(40):
        _publish(bus, organs[i % len(organs)])
    eng = SystemIntegration(bus=bus)
    rep = eng.report(force=True)
    assert rep.label == "integrated"
    assert rep.phi_system > 0.55
    assert rep.cross_subsystem_influence > 0.8
    assert rep.feedback_recurrence > 0.8
    assert set(rep.subsystems) == set(organs)


def test_single_organ_stream_reads_federated(bus):
    # One organ talking only to itself, with a lone non-recurrent tail — islands.
    for _ in range(35):
        _publish(bus, "solver")
    for _ in range(5):
        _publish(bus, "logger")
    eng = SystemIntegration(bus=bus)
    rep = eng.report(force=True)
    assert rep.phi_system < 0.4
    assert rep.label in {"federated", "loosely_coupled"}
    assert rep.feedback_recurrence == 0.0  # solver→logger has no return edge


def test_integrated_beats_federated_monotonically(bus):
    for _ in range(30):
        _publish(bus, "solver")
    federated = SystemIntegration(bus=bus).report(force=True)

    ConsequenceBus.reset()
    b2 = ConsequenceBus.get()
    organs = ["a", "b", "c", "d"]
    for i in range(40):
        _publish(b2, organs[i % len(organs)])
    integrated = SystemIntegration(bus=b2).report(force=True)

    assert integrated.phi_system > federated.phi_system


def test_ttl_cache_returns_same_object(bus):
    organs = ["a", "b", "c"]
    for i in range(20):
        _publish(bus, organs[i % len(organs)])
    eng = SystemIntegration(bus=bus, ttl=1000.0)
    first = eng.report()
    for _ in range(30):
        _publish(bus, "d")
    second = eng.report()  # within TTL → memoised, ignores new events
    assert first is second
    third = eng.report(force=True)  # force → recompute
    assert third is not first
    assert "d" in third.subsystems


def test_idle_gap_does_not_count_as_handoff(bus):
    # Two organs separated by a long idle gap must not read as a causal hand-off.
    eng = SystemIntegration(bus=bus)
    now = 1_000_000.0
    events = []

    class _E:
        def __init__(self, source, ts):
            self.source = source
            self.domain = "cognition"
            self.timestamp = ts
            self.actual_outcome = ""

    # 8 'perception' events, a 10-minute gap, then 8 'planner' events.
    for i in range(8):
        events.append(_E("perception", now + i))
    for i in range(8):
        events.append(_E("planner", now + 600 + i))
    rep = eng._compute(events, now=now + 620)
    # The only cross-boundary adjacency spans the gap and is discounted.
    assert rep.cross_subsystem_influence == 0.0


def test_min_partition_mi_detects_irreducible_coactivation(bus):
    # Build a stream where {a,b} always co-fire and {c,d} always co-fire, but the
    # two pairs alternate over time. No bipartition is clean: separating a from b
    # (or c from d) leaves high MI. The minimum stays strictly positive.
    pattern = ["a", "b", "a", "b", "c", "d", "c", "d"] * 6
    for s in pattern:
        _publish(bus, s)
    rep = SystemIntegration(bus=bus, co_activation_window=4).report(force=True)
    assert rep.min_partition_mi > 0.0
    assert math.isfinite(rep.min_partition_mi)


def test_singleton_accessor_is_stable():
    reset_system_integration()
    a = get_system_integration()
    b = get_system_integration()
    assert a is b
    reset_system_integration()
    assert get_system_integration() is not a
