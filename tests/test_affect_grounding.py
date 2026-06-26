"""Affect grounding: labels from sustained multi-signal conditions, not a single float threshold."""
from __future__ import annotations

import pytest

from core.affect.affect_grounding import (
    AffectGroundingEngine,
    get_affect_grounding_engine,
)


@pytest.fixture
def engine():
    return AffectGroundingEngine(window=12, min_samples=4)


def _feed(engine, n, **signals):
    for _ in range(n):
        engine.observe(**signals)


# ── refuses to assert a label without enough history ─────────────────────────

def test_no_label_without_minimum_history(engine):
    engine.observe(novelty=0.0, prediction_error=0.0, arousal=-0.5)
    assert engine.assess() == []          # one sample is not a feeling
    assert engine.dominant() is None


def test_single_transient_does_not_dominate(engine):
    _feed(engine, 8, novelty=0.6, valence=0.3, prediction_error=0.5)  # curious context
    engine.observe(novelty=0.0, prediction_error=0.0, arousal=-0.6)   # one bored blip
    dom = engine.dominant()
    assert dom is not None and dom.label != "boredom"  # a blip can't flip the read


# ── grounded boredom: multiple signals, sustained, explained ────────────────

def test_sustained_multisignal_boredom(engine):
    _feed(engine, 10, novelty=0.05, prediction_error=0.05, arousal=-0.5, idle=0.6)
    dom = engine.dominant()
    assert dom is not None and dom.label == "boredom"
    assert len(dom.factors) >= 2                      # not a single threshold
    assert dom.persistence > 0.8                       # sustained across the window
    assert any("novel" in f for f in dom.factors)


def test_boredom_needs_more_than_one_signal(engine):
    # Low novelty ONLY (arousal/PE neutral-positive) should not name boredom on its own.
    _feed(engine, 10, novelty=0.05, prediction_error=0.6, arousal=0.4)
    dom = engine.dominant()
    assert dom is None or dom.label != "boredom"


# ── other grounded affects ───────────────────────────────────────────────────

def test_curiosity_from_novelty_and_learnability(engine):
    _feed(engine, 10, novelty=0.8, valence=0.4, prediction_error=0.5)
    dom = engine.dominant()
    assert dom is not None and dom.label == "curiosity"
    assert any("novelty" in f for f in dom.factors)


def test_flow_from_learnable_challenge(engine):
    _feed(engine, 10, prediction_error=0.45, arousal=0.5, valence=0.3, control=0.7)
    labels = [a.label for a in engine.assess()]
    assert "flow" in labels


def test_anxiety_from_arousal_negative_valence_low_control(engine):
    _feed(engine, 10, arousal=0.7, valence=-0.4, control=0.2, social_threat=0.6)
    dom = engine.dominant()
    assert dom is not None and dom.label == "anxiety"


def test_frustration_from_pain_and_low_control(engine):
    _feed(engine, 10, pain=0.8, control=0.2, valence=-0.3)
    labels = [a.label for a in engine.assess()]
    assert "frustration" in labels


def test_contentment_from_positive_low_arousal_no_pain(engine):
    _feed(engine, 10, valence=0.5, arousal=0.0, pain=0.0)
    labels = [a.label for a in engine.assess()]
    assert "contentment" in labels


# ── explanation + confidence ─────────────────────────────────────────────────

def test_assessment_is_explained_and_serializable(engine):
    _feed(engine, 10, novelty=0.05, prediction_error=0.05, arousal=-0.5, idle=0.6)
    dom = engine.dominant()
    d = dom.to_dict()
    assert d["factors"] and d["intensity"] > 0 and 0 <= d["confidence"] <= 1


def test_confidence_grows_with_evidence():
    short = AffectGroundingEngine(window=20, min_samples=4)
    _feed(short, 5, novelty=0.05, prediction_error=0.05, arousal=-0.5, idle=0.6)
    few = short.dominant().confidence
    _feed(short, 15, novelty=0.05, prediction_error=0.05, arousal=-0.5, idle=0.6)
    many = short.dominant().confidence
    assert many > few


# ── singleton ────────────────────────────────────────────────────────────────

def test_singleton_is_stable():
    assert get_affect_grounding_engine() is get_affect_grounding_engine()
