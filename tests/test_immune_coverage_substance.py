"""Coverage must measure observation, not the existence of a channel.

CP126 064f40ec: every channel counted mere PRESENCE — any non-None anomaly
object, any truthy snapshot. Coverage is what this subsystem reports as "how
well was this observed", so a channel that exists but carries no reading told
you nothing while raising the score that said it did.

CP126 7c08abf3: a SINGLE event counted as temporal history, making a
first-sighting look as well-understood as a recurring failure.
"""
from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from core.adaptation import adaptive_immunity as mod
from core.adaptation.adaptive_immunity import (
    _anomaly_score_is_substantive,
    _snapshot_is_usable,
    get_adaptive_immune_system,
)


# --- anomaly corroboration -----------------------------------------------


def test_an_empty_anomaly_object_does_not_corroborate():
    """An object whose scoring failed is not evidence."""
    assert _anomaly_score_is_substantive(SimpleNamespace()) is False
    assert _anomaly_score_is_substantive(object()) is False
    assert _anomaly_score_is_substantive(None) is False


def test_an_anomaly_object_with_a_reading_corroborates():
    assert _anomaly_score_is_substantive(SimpleNamespace(threat_probability=0.6)) is True
    assert _anomaly_score_is_substantive({"score": 0.3}) is True


def test_a_non_finite_reading_does_not_corroborate():
    assert _anomaly_score_is_substantive(SimpleNamespace(score=float("nan"))) is False


# --- snapshots -----------------------------------------------------------


def test_an_empty_snapshot_is_not_coverage():
    assert _snapshot_is_usable({}) is False
    assert _snapshot_is_usable(None) is False
    assert _snapshot_is_usable("not a mapping") is False


def test_a_snapshot_with_content_counts():
    assert _snapshot_is_usable({"cpu": 0.4}) is True


def test_a_stale_snapshot_does_not_count():
    """It cannot describe the event it is attached to."""
    old = {"cpu": 0.4, "timestamp": time.time() - mod._MAX_SNAPSHOT_AGE_S - 60}

    assert _snapshot_is_usable(old) is False


def test_a_fresh_snapshot_counts():
    assert _snapshot_is_usable({"cpu": 0.4, "timestamp": time.time()}) is True


def test_an_undeclared_age_is_not_invented():
    """No timestamp means we check content only, rather than fabricating
    freshness we cannot see."""
    assert _snapshot_is_usable({"cpu": 0.4}) is True
    assert _snapshot_is_usable({"cpu": 0.4, "timestamp": "garbage"}) is True


# --- temporal history ----------------------------------------------------


def test_one_event_is_not_history():
    assert mod._MIN_TEMPORAL_HISTORY_EVENTS >= 2


def test_coverage_reflects_substance_not_presence():
    immune = get_adaptive_immune_system()
    event = {"subsystem": "memory", "error_count": 1}
    antigen = immune.present_antigen(event)

    hollow = immune._assess_coverage(
        event, antigen, anomaly_score=SimpleNamespace(), state_snapshot={}
    )
    substantive = immune._assess_coverage(
        event,
        antigen,
        anomaly_score=SimpleNamespace(threat_probability=0.7),
        state_snapshot={"cpu": 0.5, "timestamp": time.time()},
    )

    assert substantive["coverage_ratio"] > hollow["coverage_ratio"]


def test_a_hollow_channel_is_named_as_a_blind_spot():
    immune = get_adaptive_immune_system()
    event = {"subsystem": "memory", "error_count": 1}
    antigen = immune.present_antigen(event)

    report = immune._assess_coverage(
        event, antigen, anomaly_score=SimpleNamespace(), state_snapshot={}
    )

    assert any("anomaly-model" in spot for spot in report["known_blind_spots"])
    assert "anomaly_model" in report["missing_channels"]
