from __future__ import annotations

import pytest

from core.brain.llm.mlx_worker import _classify_generation_stop_reason
from core.conversation.response_reliability import (
    _has_truncated_tail,
    assess_model_text_integrity,
    assess_user_facing_reply,
)


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"soft_cancelled": True}, "soft_cancelled"),
        ({"deadline_hit": True}, "deadline_exceeded"),
        ({"sentinel_aborted": True}, "sentinel_abort"),
        ({"role_continuation_hit": True}, "role_continuation"),
        ({"configured_stop_hit": True}, "configured_stop"),
        ({"hard_token_limit_hit": True}, "hard_token_limit"),
        ({"generated_tokens": 128}, "max_tokens"),
        ({}, "eos"),
    ],
)
def test_worker_reports_the_exact_decode_termination(overrides, expected):
    state = {
        "soft_cancelled": False,
        "deadline_hit": False,
        "sentinel_aborted": False,
        "role_continuation_hit": False,
        "configured_stop_hit": False,
        "hard_token_limit_hit": False,
        "generated_tokens": 47,
        "max_tokens": 128,
    }
    state.update(overrides)

    assert _classify_generation_stop_reason(**state) == expected


UNPUNCTUATED_COMPLETE_PROSE = (
    "The runtime has finished its current work and retained the complete result "
    "in the active conversation record"
)


@pytest.mark.parametrize("stop_reason", ["eos", "configured_stop", "role_continuation"])
def test_intentional_termination_does_not_turn_missing_punctuation_into_truncation(
    stop_reason,
):
    assert not _has_truncated_tail(
        UNPUNCTUATED_COMPLETE_PROSE,
        generation_stop_reason=stop_reason,
    )
    assessment = assess_model_text_integrity(
        UNPUNCTUATED_COMPLETE_PROSE,
        prompt="What is the current result?",
        user_facing=True,
        generation_stop_reason=stop_reason,
    )
    assert "truncated_tail" not in assessment.reasons


@pytest.mark.parametrize(
    "stop_reason",
    ["", "max_tokens", "deadline_exceeded", "soft_cancelled", "hard_token_limit"],
)
def test_exhaustion_keeps_unpunctuated_prose_incomplete(stop_reason):
    assert _has_truncated_tail(
        UNPUNCTUATED_COMPLETE_PROSE,
        generation_stop_reason=stop_reason,
    )


def test_objectively_dangling_syntax_remains_incomplete_under_eos():
    clipped = (
        "The function updates each balance and removes names whose balance "
        "reaches zero from the"
    )

    assert _has_truncated_tail(clipped, generation_stop_reason="eos")


def test_user_facing_assessor_consumes_the_worker_termination_receipt():
    assessment = assess_user_facing_reply(
        "What is the current result?",
        UNPUNCTUATED_COMPLETE_PROSE,
        generation_stop_reason="eos",
    )

    assert "truncated_tail" not in assessment.reasons
