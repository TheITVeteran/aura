"""Tests for engram retrieval resolved by the plasticity competition field."""
from __future__ import annotations

from core.memory.engram_plasticity import EngramPlasticityField


def _field() -> EngramPlasticityField:
    return EngramPlasticityField(settle_steps=24)


def test_empty_and_single_candidate():
    f = _field()
    r0 = f.compete([])
    assert r0.order == [] and r0.winner == -1
    r1 = f.compete([0.8])
    assert r1.order == [0] and r1.winner == 0


def test_strong_salience_wins_weak_is_gated_out():
    f = _field()
    # candidate 1 has the strongest drive; candidate 3 is near-silent.
    r = f.compete([0.3, 0.95, 0.4, 0.02])
    assert r.winner == 1
    assert r.order[0] == 1
    assert 3 in r.gated_out                    # sub-threshold trace contributes nothing
    assert 1 not in r.gated_out


def test_anti_confabulation_relevance_beats_raw_strength():
    """A vivid-but-irrelevant trace (low salience) must lose to the relevant one.

    In recall_similar the salience already folds relevance × strength, so a
    high-strength low-relevance engram arrives with LOW salience and must not
    win — the competition picks the trace that actually matches.
    """
    f = _field()
    # index 0 = relevant match (high salience), index 1 = vivid but off-topic.
    r = f.compete([0.9, 0.15])
    assert r.winner == 0
    assert r.weights[0] > r.weights[1]


def test_arousal_lowers_threshold_keeps_more_engrams_alive():
    f = _field()
    sal = [0.55, 0.45, 0.40, 0.35]
    calm = f.compete(sal, arousal=0.0, valence=0.0)
    aroused = f.compete(sal, arousal=1.0, valence=0.0)
    # Emotional arousal lowers θ, so fewer candidates fall below threshold.
    assert len(aroused.gated_out) <= len(calm.gated_out)


def test_dominant_attractor_raises_pressure_signal():
    f = _field()
    # One overwhelming trace among many weak ones drives total activation hot.
    f.compete([1.0, 0.95, 0.9, 0.85, 0.8, 0.75])
    sig = f.governance_signal()
    assert sig["homeostatic_pressure"] >= 1.0
    assert "governance_breach" in sig
    assert sig["safe_bound"] > 0


def test_weights_normalised_and_ordered():
    f = _field()
    r = f.compete([0.2, 0.9, 0.5])
    assert abs(sum(r.weights) - 1.0) < 1e-6 or sum(r.weights) == 0.0
    # order is sorted by settled activation, strongest first
    assert r.order[0] == 1


def test_determinism():
    a = _field().compete([0.3, 0.7, 0.5, 0.1], arousal=0.4, valence=0.2)
    b = _field().compete([0.3, 0.7, 0.5, 0.1], arousal=0.4, valence=0.2)
    assert a.order == b.order
    assert a.weights == b.weights
