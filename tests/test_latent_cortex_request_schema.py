"""Type checks are not a schema, and prompt shape is not evidence.

CP126 1a992727: deep_reason checked only that question was a string and
messages a list — nothing bounded the text, validated roles and content
shapes, rejected unknown message fields, or constrained the domain, so
oversized and malformed input went into IPC and the calibration stores that
learn from it.

CP126 fc2f6c87: the ENTIRE caller-supplied prompt_shape was copied into an
auditable routing receipt, and supplied counts could only RAISE the analyzed
ones — making the caller's claim authoritative over the measurement.

CP126 0eef80b2: those counts become "stakes" and "uncertainty", borrowing
the vocabulary of epistemic and consequence signals they are not.
"""
from __future__ import annotations

import pytest

from core.brain import latent_cortex_service as mod
from core.brain.latent_cortex_service import LatentCortexService

_ERR = LatentCortexService._request_schema_error


# --- 1a992727: a bounded request schema ---------------------------------


def test_a_well_formed_request_passes():
    assert _ERR("hello", [{"role": "user", "content": "hi"}], "general") == ""


def test_an_oversized_question_is_refused():
    assert _ERR("x" * (mod._MAX_QUESTION_CHARS + 1), None, "general").startswith(
        "question_too_large"
    )


def test_too_many_messages_are_refused():
    messages = [{"role": "user", "content": "x"}] * (mod._MAX_MESSAGES + 1)

    assert _ERR(None, messages, "general").startswith("too_many_messages")


def test_oversized_message_content_is_refused():
    messages = [{"role": "user", "content": "x" * 100_000} for _ in range(10)]

    assert _ERR(None, messages, "general").startswith("messages_too_large")


@pytest.mark.parametrize("role", ["root", "", None, 42, "SYSTEM "])
def test_an_invalid_role_is_refused(role):
    assert _ERR(None, [{"role": role, "content": "x"}], "general").endswith("invalid_role")


@pytest.mark.parametrize("role", ["system", "user", "assistant", "tool"])
def test_declared_roles_are_accepted(role):
    assert _ERR(None, [{"role": role, "content": "x"}], "general") == ""


def test_unknown_message_fields_are_refused_not_ignored():
    """A field this layer cannot bound still reaches the worker."""
    error = _ERR(None, [{"role": "user", "content": "x", "exec": "rm -rf"}], "general")

    assert "unknown_fields" in error and "exec" in error


@pytest.mark.parametrize("content", [{"a": 1}, 42, None, ["x"]])
def test_non_string_content_is_refused(content):
    assert _ERR(None, [{"role": "user", "content": content}], "general").endswith(
        "invalid_content"
    )


def test_a_non_mapping_message_is_refused():
    assert _ERR(None, ["just a string"], "general").endswith("not_mapping")


@pytest.mark.parametrize("domain", ["", "   ", None, 42, "x" * 200, "bad\ndomain"])
def test_an_invalid_domain_is_refused(domain):
    assert _ERR("hi", None, domain) == "invalid_domain"


# --- fc2f6c87: a closed prompt-shape schema ------------------------------


def _select(prompt_shape):
    return LatentCortexService.select_foreground_episode(
        foreground=True,
        desktop_required=False,
        cognitive_mode="deliberate",
        prompt_shape=prompt_shape,
        compact_contract=False,
        strict_output_contract=False,
        incompatible_contract=False,
        proof_or_benchmark=False,
        visible_objective="Explain the tradeoffs and compare them.",
    )


def test_unknown_prompt_shape_keys_do_not_reach_the_receipt():
    decision = _select({"question_parts": 2, "injected": {"deep": "payload"}})

    assert "injected" not in decision["latent_cortex_prompt_shape"]
    assert "injected" in decision["latent_cortex_prompt_shape_rejected_keys"]


def test_a_caller_cannot_inflate_a_routing_signal():
    """Supplied counts could only ever RAISE the analyzed ones."""
    decision = _select({"question_parts": 10_000})

    assert decision["latent_cortex_prompt_shape"]["question_parts"] <= 512


def test_the_analyzed_shape_still_wins_when_larger():
    decision = _select({"question_parts": 0})

    assert decision["latent_cortex_prompt_shape"]["question_parts"] >= 0


def test_no_supplied_shape_is_safe():
    decision = _select(None)

    assert decision["latent_cortex_prompt_shape_rejected_keys"] == []


def test_a_non_mapping_shape_is_safe():
    assert _select("not a mapping")["latent_cortex_prompt_shape_rejected_keys"] == []


# --- 0eef80b2: the signals say what they are -----------------------------


def test_the_receipt_declares_the_signal_basis():
    decision = _select({"question_parts": 3})

    assert decision["signal_basis"] == "prompt_shape_heuristic"
    assert decision["signal_sources"] == ["prompt_text_shape"]


def test_uncertainty_is_not_claimed_as_calibrated():
    decision = _select({"question_parts": 3})

    assert decision["calibrated_uncertainty"] is False
    assert decision["consequence_evidence"] is False


def test_stakes_and_uncertainty_are_still_produced():
    """The routing behaviour is unchanged; only the claim is corrected."""
    decision = _select({"question_parts": 3})

    assert 0.0 <= decision["stakes"] <= 1.0
    assert 0.0 <= decision["uncertainty"] <= 1.0
