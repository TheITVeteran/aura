from __future__ import annotations

import math

import pytest

from core.brain.llm.mlx_worker import (
    _build_semantic_completion_eos_guard,
    _classify_generation_stop_reason,
    _semantic_surface_stop_ready,
    _surface_quality_candidate,
)
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
        ({"semantic_contract_satisfied": True}, "semantic_contract_satisfied"),
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


def test_continuation_quality_evaluates_the_complete_authored_candidate():
    partial = "I feel steady right now. What I know comes from current state readings;"
    tail = " what remains subjective is something I can only infer."
    job = {
        "user_surface_continuation_contract": True,
        "user_surface_continuation_partial": partial,
    }

    assert _surface_quality_candidate(job, tail) == partial + tail


def test_semantic_stop_waits_for_all_requested_epistemic_facets():
    prompt = (
        "How are you doing right now? Distinguish what you know from what "
        "you can only infer."
    )
    job = {
        "clean_user_surface_contract": True,
        "semantic_completion_contract": True,
        "self_condition_contract": True,
        "user_surface_validation_prompt": prompt,
    }
    incomplete = (
        "I feel steady and attentive right now. That condition is supported "
        "by my current affect and coherence readings."
    )
    complete = (
        incomplete
        + " Whether that amounts to subjective feeling is something I can only infer."
    )

    assert not _semantic_surface_stop_ready(job, incomplete, generated_tokens=40)
    assert _semantic_surface_stop_ready(job, complete, generated_tokens=64)


def test_semantic_eos_guard_blocks_termination_until_coverage_is_complete():
    mx = pytest.importorskip("mlx.core")

    class Tokenizer:
        eos_token_ids = {9}

        @staticmethod
        def decode(token_ids):
            return "".join(chr(int(token_id)) for token_id in token_ids)

    prompt = "Give one complete response with (1) alpha and (2) beta."
    job = {
        "clean_user_surface_contract": True,
        "semantic_completion_contract": True,
        "user_surface_validation_prompt": prompt,
    }
    guard = _build_semantic_completion_eos_guard(
        Tokenizer(),
        job,
        prompt_token_count=2,
        tensor_ops=mx,
    )
    assert guard is not None

    incomplete = [1, 2, *map(ord, "1. Alpha is covered but the second")]
    logits = mx.zeros((1, 128))
    blocked = guard(mx.array(incomplete), logits)
    assert math.isinf(float(blocked[0, 9]))
    assert float(blocked[0, 9]) < 0

    complete = [
        1,
        2,
        *map(ord, "1. Alpha is covered. 2. Beta is covered."),
    ]
    allowed = guard(mx.array(complete), logits)
    assert float(allowed[0, 9]) == 0.0
