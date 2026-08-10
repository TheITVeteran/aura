"""Test doubles for the reasoning amplifier that use the REAL result types.

Two tests hand-rolled ``SimpleNamespace(receipt=SimpleNamespace(to_dict=lambda:
{...}))`` in place of ``AmplifiedAnswer`` and ``ReasoningReceipt``. When
``promotion_authority`` was added as an adoption precondition, both fakes kept
returning receipts without it, so the phase correctly declined to adopt and
both tests began asserting composed answers on a path they never reached. They
failed for a reason unrelated to what they were written to check.

A hand-built double cannot drift from a contract it never referenced. Building
the real dataclasses means a new required field is a construction error here
rather than a silent behavioural difference, and the default for
``promotion_authority`` is production's default rather than an omission.

``promotion_authority`` is deliberately explicit at every call site: whether a
verifier promoted an answer is the precondition under test, not a detail to
inherit from a helper.
"""
from __future__ import annotations

from typing import Any

from core.brain.reasoning_amplifier_v2 import AmplifiedAnswer, ReasoningReceipt


def amplifier_receipt(
    *,
    promotion_authority: str,
    strategy_used: str = "test_strategy",
    task_type: str = "planning",
    winning_candidate_id: int | None = 0,
    valid_candidates: int = 1,
    confidence: float = 0.95,
    **overrides: Any,
) -> ReasoningReceipt:
    """A real ReasoningReceipt. `promotion_authority` is required on purpose."""
    fields: dict[str, Any] = {
        "mode": "test",
        "strategy_used": strategy_used,
        "task_type": task_type,
        "num_candidates": 1,
        "verifiers_run": ["test_verifier"],
        "valid_candidates": valid_candidates,
        "winning_candidate_id": winning_candidate_id,
        "confidence": confidence,
        "agreement": 1.0,
        "epistemic_status": "verified",
        "promotion_authority": promotion_authority,
    }
    fields.update(overrides)
    return ReasoningReceipt(**fields)


def amplified_answer(
    answer: str,
    *,
    promotion_authority: str,
    confidence: float = 0.95,
    verified: bool = True,
    calibrated: bool = True,
    generation_metadata: dict[str, Any] | None = None,
    source_answer: str | None = None,
    **receipt_overrides: Any,
) -> AmplifiedAnswer:
    """A real AmplifiedAnswer carrying a real receipt."""
    return AmplifiedAnswer(
        answer=answer,
        confidence=confidence,
        verified=verified,
        calibrated=calibrated,
        receipt=amplifier_receipt(
            promotion_authority=promotion_authority,
            confidence=confidence,
            **receipt_overrides,
        ),
        generation_metadata=dict(generation_metadata or {}),
        source_answer=source_answer if source_answer is not None else answer,
    )


def test_the_double_matches_the_production_contract():
    """A double that omits a required field is the drift this file prevents.

    Constructing the real dataclass already fails loudly on a missing
    argument; this asserts the fields the ADOPTION path reads are actually
    present in the receipt dict a phase receives, since that path reads a
    plain dict and cannot type-check anything.
    """
    receipt = amplified_answer("x", promotion_authority="checked_verifier").receipt.to_dict()

    # The precondition response_generation checks before adopting an answer.
    assert receipt["promotion_authority"] == "checked_verifier"
    assert "winning_candidate_id" in receipt
    assert "confidence" in receipt


def test_an_unpromoted_answer_is_representable():
    """The default case must be expressible, or tests only cover adoption."""
    receipt = amplified_answer("x", promotion_authority="none").receipt.to_dict()

    assert receipt["promotion_authority"] == "none"
