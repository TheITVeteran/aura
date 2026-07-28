"""Context dropped on the way into the workspace must be recorded.

CP126 (medium), core/brain/llm/latent_cortex/engine.py: "Cognitive context
is silently dropped at three independent limits. Only six items are
considered, each text is truncated to 400 characters, and each embedding is
truncated again to 64 tokens, but the receipt does not state requested
versus admitted context or dropped content. Consumers can believe the full
context influenced reasoning when it did not."

Two of the three limits turned out to already do the right thing:
``normalize_cognitive_context`` RAISES past six items and past 400
characters, so a caller cannot quietly over-send. Those are verified here so
a future change to truncate instead of reject fails loudly.

The token limit was the real one. Encoding a 400-character memory can easily
exceed 64 tokens, and the tail was dropped with nothing recorded — a slot
seeded from half a memory was indistinguishable from one seeded from all of
it, and the reasoning that follows differs.
"""
from __future__ import annotations

import pytest

from core.brain.llm.latent_cortex.cognitive_context import (
    MAX_COGNITIVE_CONTEXT_CHARS,
    MAX_COGNITIVE_CONTEXT_ITEMS,
    CognitiveContextError,
    normalize_cognitive_context,
)
from core.brain.llm.latent_cortex.engine import LatentCortexEngine


@pytest.fixture
def engine():
    e = LatentCortexEngine.__new__(LatentCortexEngine)
    e._cognitive_context_truncations = []
    return e


class TestTwoLimitsRejectRatherThanTruncate:
    """Pinned so a future change to silently truncate fails here."""

    def test_too_many_items_is_refused(self):
        payload = [
            {"source": f"s{i}", "text": "x"}
            for i in range(MAX_COGNITIVE_CONTEXT_ITEMS + 1)
        ]
        with pytest.raises(CognitiveContextError):
            normalize_cognitive_context(payload)

    def test_overlong_text_is_refused(self):
        payload = [{"source": "s", "text": "x" * (MAX_COGNITIVE_CONTEXT_CHARS + 1)}]
        with pytest.raises(CognitiveContextError):
            normalize_cognitive_context(payload)

    def test_a_conforming_payload_is_accepted(self):
        payload = [{"source": "s", "text": "a real observation"}]
        assert normalize_cognitive_context(payload) == payload


class TestTokenTruncationIsReceipted:
    def test_a_clean_episode_reports_complete(self, engine):
        admission = engine.cognitive_context_admission()
        assert admission["complete"] is True
        assert admission["truncated_items"] == 0
        assert admission["dropped_tokens_total"] == 0

    def test_a_truncated_item_is_counted(self, engine):
        engine._cognitive_context_truncations.append(
            {
                "source": "episodic",
                "context_role": "memory_observation",
                "requested_tokens": 97,
                "admitted_tokens": 64,
                "dropped_tokens": 33,
            }
        )
        admission = engine.cognitive_context_admission()
        assert admission["complete"] is False
        assert admission["truncated_items"] == 1
        assert admission["dropped_tokens_total"] == 33

    def test_the_receipt_names_which_item_lost_content(self, engine):
        engine._cognitive_context_truncations.append(
            {
                "source": "episodic",
                "context_role": "memory_observation",
                "requested_tokens": 97,
                "admitted_tokens": 64,
                "dropped_tokens": 33,
            }
        )
        entry = engine.cognitive_context_admission()["truncations"][0]
        assert entry["source"] == "episodic"
        assert entry["requested_tokens"] > entry["admitted_tokens"]

    def test_the_receipt_states_the_limits_it_applied(self, engine):
        admission = engine.cognitive_context_admission()
        assert admission["max_tokens_per_item"] == 64
        assert admission["max_items"] == MAX_COGNITIVE_CONTEXT_ITEMS
        assert admission["max_chars"] == MAX_COGNITIVE_CONTEXT_CHARS

    def test_the_receipt_is_serializable(self, engine):
        assert engine.cognitive_context_admission()["schema"] == (
            "aura.cognitive_context_admission.v1"
        )


class TestTheRecordIsPerEpisode:
    def test_validating_context_clears_the_previous_record(self, engine):
        engine._cognitive_context_truncations.append(
            {"source": "old", "context_role": "untyped",
             "requested_tokens": 9, "admitted_tokens": 4, "dropped_tokens": 5}
        )
        engine._validate_cognitive_context([{"source": "s", "text": "fresh"}])
        assert engine.cognitive_context_admission()["complete"] is True

    def test_reset_clears_the_record(self, engine):
        engine._cognitive_context_truncations.append(
            {"source": "x", "context_role": "untyped",
             "requested_tokens": 9, "admitted_tokens": 4, "dropped_tokens": 5}
        )
        engine.reset_cognitive_context_admission()
        assert engine.cognitive_context_admission()["truncated_items"] == 0
