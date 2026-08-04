"""Two reasons a delivered reply stops being "high confidence", now testable.

This decision lived inside ``api_chat`` — 4,488 lines, 633 branches — where the
only way to exercise it was to drive a whole HTTP turn against a live model. It
has two independent limbs and neither had a test:

  1. the reply contradicts itself (false inability claims, commitment
     contradictions);
  2. the generation came from a FALLBACK lane during this turn, so the answer is
     real but not from the mind that was asked.

Extracting it does not make ``api_chat`` meaningfully shorter — the call site is
about as long as the block it replaced. What it buys is that the rule can be
stated and checked directly, which is the part that was missing.
"""

from __future__ import annotations

import pytest

from interface.routes.chat_quality import assess_post_response_confidence


class TestOnlyHighIsDowngraded:
    @pytest.mark.parametrize("confidence", ["degraded", "failed", "low", ""])
    def test_an_already_degraded_reply_is_left_alone(self, confidence):
        result = assess_post_response_confidence(
            confidence, self_consistent=False, used_fallback_lane=True,
            generation_happened_this_turn=True,
        )
        assert result.confidence == confidence
        assert result.downgraded is False

    def test_nothing_here_promotes_anything(self):
        result = assess_post_response_confidence(
            "degraded", self_consistent=True, used_fallback_lane=False
        )
        assert result.confidence == "degraded"


class TestSelfConsistency:
    def test_a_contradicting_reply_is_downgraded(self):
        result = assess_post_response_confidence(
            "high",
            self_consistent=False,
            inconsistency_reason="claimed inability it does not have",
        )
        assert result.confidence == "degraded"
        assert result.reason == "claimed inability it does not have"
        assert result.downgraded is True

    def test_an_unnamed_inconsistency_still_names_something(self):
        result = assess_post_response_confidence("high", self_consistent=False)
        assert result.reason == "self_inconsistent"

    def test_consistency_alone_keeps_it_high(self):
        result = assess_post_response_confidence("high", self_consistent=True)
        assert result.confidence == "high"
        assert result.downgraded is False


class TestFallbackLane:
    def test_a_fallback_generation_in_this_turn_is_downgraded(self):
        result = assess_post_response_confidence(
            "high",
            self_consistent=True,
            used_fallback_lane=True,
            generation_happened_this_turn=True,
            actual_endpoint="Brainstem",
            desired_endpoint="Cortex",
        )
        assert result.confidence == "degraded"
        assert result.reason == "fallback_lane_generation"
        assert "Brainstem" in result.lane_warning
        assert "Cortex" in result.lane_warning

    def test_a_fallback_from_an_EARLIER_turn_does_not_taint_this_one(self):
        """The lane status is global; only this turn's generation counts."""
        result = assess_post_response_confidence(
            "high",
            self_consistent=True,
            used_fallback_lane=True,
            generation_happened_this_turn=False,
        )
        assert result.confidence == "high"
        assert result.downgraded is False

    def test_a_primary_lane_generation_stays_high(self):
        result = assess_post_response_confidence(
            "high",
            self_consistent=True,
            used_fallback_lane=False,
            generation_happened_this_turn=True,
        )
        assert result.confidence == "high"

    def test_the_warning_survives_missing_endpoint_names(self):
        result = assess_post_response_confidence(
            "high",
            self_consistent=True,
            used_fallback_lane=True,
            generation_happened_this_turn=True,
        )
        assert "fallback" in result.lane_warning
        assert "primary" in result.lane_warning


def test_inconsistency_is_checked_before_the_lane():
    """Both limbs true: the reply's own contradiction is the better reason."""
    result = assess_post_response_confidence(
        "high",
        self_consistent=False,
        inconsistency_reason="contradicted its own commitment",
        used_fallback_lane=True,
        generation_happened_this_turn=True,
    )
    assert result.reason == "contradicted its own commitment"


def test_the_call_site_uses_it():
    import inspect

    from interface.routes import chat as chat_routes

    source = inspect.getsource(chat_routes.api_chat)
    assert "assess_post_response_confidence(" in source
