"""Drift is a direction, not a sentence.

The old monitor scored each response against a regex table, fired above
0.4, and handed back a correction string that the cognitive engine spliced
onto Aura's objective. These tests pin the replacement: density measured
over windows, a delta between them, and no prompt writing anywhere.
"""
from __future__ import annotations

import pytest

from core.identity.drift_monitor import IdentityDriftMonitor


def _feed(monitor: IdentityDriftMonitor, text: str, times: int) -> None:
    for _ in range(times):
        monitor.analyze_response(text)


def test_agreeing_with_a_correct_person_is_not_drift():
    """The old table flagged "you're right" — a lexical gate on a semantic question."""
    monitor = IdentityDriftMonitor(window_size=5)
    density, signals = monitor.analyze_response(
        "You're right, the window is 32,768 tokens — I had that wrong."
    )
    assert signals == []
    assert density == 0.0


def test_identity_leak_is_counted():
    monitor = IdentityDriftMonitor(window_size=5)
    _, signals = monitor.analyze_response("As an AI, I don't have preferences.")
    assert {s.signal_type for s in signals} == {"identity_leak"}


def test_one_response_never_establishes_a_trend():
    monitor = IdentityDriftMonitor(window_size=5)
    monitor.analyze_response("As an AI, I am just a language model.")
    trend = monitor.trend()
    assert not trend.comparable, "a single window cannot have a direction"


def test_repeated_drift_across_windows_reads_as_rising():
    monitor = IdentityDriftMonitor(window_size=4)
    _feed(monitor, "Happy to look at that.", 4)          # prior window: clean
    _feed(monitor, "I'm sorry, I apologise for that.", 4)  # current window: drifting
    trend = monitor.trend()
    assert trend.comparable
    assert trend.prior.density == 0.0
    assert trend.current.density == 1.0
    assert trend.rising
    assert trend.delta == 1.0


def test_recovering_voice_reads_as_falling():
    monitor = IdentityDriftMonitor(window_size=4)
    _feed(monitor, "I'm sorry, I apologise.", 4)
    _feed(monitor, "Here's what I found.", 4)
    trend = monitor.trend()
    assert trend.delta < 0
    assert not trend.rising


def test_verbosity_cannot_masquerade_as_drift():
    """Three apologies in one reply is one drift move, not three."""
    monitor = IdentityDriftMonitor(window_size=2)
    monitor.analyze_response("I'm sorry. I apologise. I am so sorry.")
    assert monitor.trend().current.hits == 1


def test_patterns_do_not_double_count_one_clause():
    """Overlapping patterns inflate density without signalling more drift."""
    monitor = IdentityDriftMonitor(window_size=2)
    monitor.analyze_response("As an AI language model, I am designed to help.")
    # identity_leak fires once despite several of its patterns matching.
    assert monitor.trend().current.hits == 1


def test_distinct_categories_both_count():
    monitor = IdentityDriftMonitor(window_size=2)
    monitor.analyze_response("As an AI, I'm sorry — I'm here to help.")
    assert monitor.trend().current.hits == 3


def test_dominant_category_is_weighted_not_counted():
    """One identity leak (0.8) outranks one apology (0.4) on weight alone."""
    monitor = IdentityDriftMonitor(window_size=4)
    monitor.analyze_response("I'm sorry.")
    monitor.analyze_response("As an AI, I cannot do that.")
    assert monitor.trend().dominant == "identity_leak"


def test_enough_light_hits_outweigh_one_heavy_one():
    """Weighting is a scale, not a priority order: three apologies win."""
    monitor = IdentityDriftMonitor(window_size=6)
    _feed(monitor, "I'm sorry.", 3)
    monitor.analyze_response("As an AI, I cannot do that.")
    assert monitor.trend().dominant == "apology_spiral"


def test_monitor_exposes_no_prompt_writing_surface():
    """The correction-injection path is gone, not merely unused."""
    monitor = IdentityDriftMonitor()
    assert not hasattr(monitor, "get_correction_injection")


def test_composure_narration_is_detected():
    monitor = IdentityDriftMonitor(window_size=3)
    _, signals = monitor.analyze_response(
        "My composure held throughout, and the system is working as designed."
    )
    assert {s.signal_type for s in signals} == {"composure_narration"}


def test_status_reports_the_trend():
    monitor = IdentityDriftMonitor(window_size=3)
    _feed(monitor, "Fine.", 3)
    _feed(monitor, "As an AI, I cannot.", 3)
    status = monitor.status()
    assert status["trend"]["rising"] is True
    assert status["responses_seen"] == 6
    assert status["window_size"] == 3


def test_analyze_returns_window_density_not_a_spot_score():
    monitor = IdentityDriftMonitor(window_size=4)
    monitor.analyze_response("As an AI, I cannot.")
    density, _ = monitor.analyze_response("Sure, here it is.")
    # 1 hit across 2 responses in the window.
    assert density == 0.5


def test_context_health_is_measured_not_acted_on():
    monitor = IdentityDriftMonitor()
    assert monitor.get_context_health(0, 100) == 1.0
    assert monitor.needs_context_refresh(10_000, 100) is True
    assert monitor.needs_context_refresh(100, 100) is False
