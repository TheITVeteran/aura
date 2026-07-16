"""Contract tests for receipt-bound latent-context compaction."""

from __future__ import annotations

import pytest

from core.brain.llm.latent_cortex.context_compaction import (
    POLICY_VERSION,
    compact_latent_messages,
)


def test_compaction_is_bounded_salient_and_deterministic():
    messages = [
        {"role": "system", "content": "canonical system " + "s" * 5000},
        {"role": "assistant", "content": "unrelated history " + "x" * 1800},
        {
            "role": "system",
            "content": (
                "CURRENT FUNCTIONAL STATE duplicate generation cancellation "
                "worker restart "
                + "e" * 1200
            ),
        },
        {
            "role": "user",
            "content": (
                "Explain duplicate generation under cancellation and worker restart. "
                + "u" * 4000
            ),
        },
    ]

    compacted, receipt = compact_latent_messages(messages, max_chars=2400)
    repeated, repeated_receipt = compact_latent_messages(messages, max_chars=2400)

    assert compacted == repeated
    assert receipt == repeated_receipt
    assert sum(len(item["content"]) for item in compacted) <= 2400
    assert compacted[0]["role"] == "system"
    assert compacted[-1]["role"] == "user"
    assert any("CURRENT FUNCTIONAL STATE" in item["content"] for item in compacted)
    assert receipt["schema"] == "aura.latent_context_compaction.v1"
    assert receipt["policy"] == POLICY_VERSION
    assert receipt["applied"] is True
    assert receipt["compacted_char_count"] <= receipt["max_chars"]
    assert receipt["omitted_char_count"] > 0
    assert len(receipt["original_sha256"]) == 64
    assert len(receipt["compacted_sha256"]) == 64


def test_single_user_message_uses_the_entire_budget_once():
    messages = [{"role": "user", "content": "a" * 4000}]

    compacted, receipt = compact_latent_messages(messages, max_chars=2048)

    assert compacted[0]["role"] == "user"
    assert len(compacted[0]["content"]) == 2048
    assert receipt["compacted_char_count"] == 2048
    assert receipt["compacted_message_count"] == 1


def test_context_under_budget_is_preserved_without_claiming_compaction():
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "question"},
    ]

    compacted, receipt = compact_latent_messages(messages, max_chars=2048)

    assert compacted == messages
    assert receipt["applied"] is False
    assert receipt["original_sha256"] == receipt["compacted_sha256"]
    assert receipt["omitted_char_count"] == 0


@pytest.mark.parametrize(
    "messages",
    [
        [],
        [None, "text"],
        [{"role": "tool", "content": "ignored"}],
        [{"role": "user", "content": None}],
    ],
)
def test_context_without_valid_messages_is_rejected(messages):
    with pytest.raises(ValueError, match="no valid messages"):
        compact_latent_messages(messages, max_chars=2048)
