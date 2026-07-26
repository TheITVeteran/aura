"""SPARK-053 principled terminal reason and language-causality contracts."""

from __future__ import annotations

import copy
import hashlib
import json

import pytest

from core.brain.llm.latent_cortex.loop_core import canonical_sha256
from core.brain.llm.latent_cortex.terminal_disposition import (
    COMPUTE_BUDGET,
    IRREDUCIBLE_UNCERTAINTY,
    LOW_VALUE,
    RECURRENCE_BUDGET,
    VERIFIED_CONVERGENCE,
    WALL_BUDGET,
    classify_terminal_disposition,
    finalize_terminal_disposition_receipt,
    validate_terminal_disposition_receipt,
)


def _sha(value: dict) -> str:
    return canonical_sha256(value)


def _loop(*, fixed: bool = True) -> dict:
    payload = {
        "selected_branch": 0,
        "branches": [
            {
                "final_residual": 0.01,
                "transitions": [
                    {
                        "fixed_point_candidate": fixed,
                    }
                ],
            }
        ],
        "all_finite": True,
        "all_accepted_states_anchor_bounded": True,
    }
    return {**payload, "receipt_sha256": _sha(payload)}


def _halting(*, low_value: bool = False) -> dict:
    decision = {
        "halt": low_value,
        "reason": "learned_stop" if low_value else "residual_policy",
        "evidence_ready": low_value,
        "features": {"expected_net_value": -0.2 if low_value else 0.2},
        "features_sha256": "a" * 64,
    }
    payload = {
        "head_was_causal": low_value,
        "branches": [{"decisions": [decision]}],
    }
    return {**payload, "receipt_sha256": _sha(payload)}


def _trace(*, action: str, mode: str, uncertainty: float = 0.2) -> list[dict]:
    return [
        {
            "decision": {"action": action, "mode": mode},
            "state_signal": {
                "answer_verified": mode in {"verified_stop", "verified_execute"},
                "uncertainty": uncertainty,
                "verifier_score": 1.0 if mode == "verified_stop" else 0.0,
                "budget_remaining_fraction": 0.5,
            },
        }
    ]


def _budget(*, exhausted: bool = False, wall: bool = False) -> dict:
    return {
        "max_layer_apps": 10_000,
        "spent_layer_apps": 10_000 if exhausted else 5_000,
        "wall_clock_s": 120.0,
        "elapsed_s": 121.0 if wall else 30.0,
        "exhausted": exhausted,
    }


def test_required_terminal_reasons_are_distinct_and_precedence_is_principled():
    convergence = classify_terminal_disposition(
        halting_reason="converged",
        halting=_halting(),
        loop_stability=_loop(),
        cognitive_action_trace=[],
        budget=_budget(),
    )
    assert (convergence.reason, convergence.disposition) == (
        VERIFIED_CONVERGENCE,
        "answer",
    )

    low_value = classify_terminal_disposition(
        halting_reason="learned_stop",
        halting=_halting(low_value=True),
        loop_stability=_loop(fixed=False),
        cognitive_action_trace=[],
        budget=_budget(),
    )
    assert (low_value.reason, low_value.disposition) == (LOW_VALUE, "answer")

    irreducible = classify_terminal_disposition(
        halting_reason="value_controller_abstain",
        halting=_halting(),
        loop_stability=_loop(fixed=False),
        cognitive_action_trace=_trace(
            action="abstain",
            mode="irreducible_abstain",
            uncertainty=1.0,
        ),
        budget=_budget(),
    )
    assert (irreducible.reason, irreducible.disposition) == (
        IRREDUCIBLE_UNCERTAINTY,
        "abstain",
    )

    recurrence_budget = classify_terminal_disposition(
        halting_reason="value_controller_abstain",
        halting=_halting(),
        loop_stability=_loop(fixed=False),
        cognitive_action_trace=_trace(action="abstain", mode="budget_abstain"),
        budget=_budget(),
    )
    assert (recurrence_budget.reason, recurrence_budget.disposition) == (
        RECURRENCE_BUDGET,
        "abstain",
    )

    compute = classify_terminal_disposition(
        halting_reason="max_steps",
        halting=_halting(),
        loop_stability=_loop(fixed=False),
        cognitive_action_trace=[],
        budget=_budget(exhausted=True),
    )
    assert compute.reason == COMPUTE_BUDGET

    wall = classify_terminal_disposition(
        halting_reason="max_steps",
        halting=_halting(),
        loop_stability=_loop(fixed=False),
        cognitive_action_trace=[],
        budget=_budget(exhausted=True, wall=True),
    )
    assert wall.reason == WALL_BUDGET

    # Irreducible evidence dominates a coincident budget flag: giving a
    # bounded answer here would turn a proved inability to discriminate into
    # unsupported confidence.
    irreducible_at_budget = classify_terminal_disposition(
        halting_reason="value_controller_abstain",
        halting=_halting(),
        loop_stability=_loop(fixed=False),
        cognitive_action_trace=_trace(
            action="abstain",
            mode="irreducible_abstain",
            uncertainty=1.0,
        ),
        budget=_budget(exhausted=True, wall=True),
    )
    assert irreducible_at_budget.reason == IRREDUCIBLE_UNCERTAINTY


def test_convergence_requires_actual_fixed_point_and_stability_evidence():
    with pytest.raises(ValueError, match="fixed-point"):
        classify_terminal_disposition(
            halting_reason="converged",
            halting=_halting(),
            loop_stability=_loop(fixed=False),
            cognitive_action_trace=[],
            budget=_budget(),
        )


def test_receipt_binds_reason_instruction_model_tokens_and_final_output():
    loop = _loop()
    halting = _halting()
    budget_at_decision = _budget()
    decision = classify_terminal_disposition(
        halting_reason="converged",
        halting=halting,
        loop_stability=loop,
        cognitive_action_trace=[],
        budget=budget_at_decision,
    )
    instruction_tokens = [11, 12, 13]
    bridge_tokens = [1, 2, *instruction_tokens]
    output_tokens = [91, 92]
    output_text = "The supported answer."
    receipt = finalize_terminal_disposition_receipt(
        decision,
        instruction_tokens=instruction_tokens,
        full_bridge_tokens=bridge_tokens,
        output_tokens=output_tokens,
        output_text=output_text,
        output_source="resident_model_decode",
    )
    final_budget = {**budget_at_decision, "spent_layer_apps": 5_500, "elapsed_s": 31.0}
    bridge_sha = hashlib.sha256(
        json.dumps(bridge_tokens, separators=(",", ":")).encode("ascii")
    ).hexdigest()
    assert (
        validate_terminal_disposition_receipt(
            receipt,
            halting_reason="converged",
            halting=halting,
            loop_stability=loop,
            cognitive_action_trace=[],
            budget=final_budget,
            output_tokens=output_tokens,
            output_text=output_text,
            full_bridge_tokens_sha256=bridge_sha,
        )
        == receipt
    )
    assert receipt["language"]["model_generated"] is True
    assert receipt["language"]["instruction_applied"] is True
    assert receipt["language"]["full_bridge_tokens"][-3:] == instruction_tokens
    assert receipt["language"]["output_text_sha256"] != receipt["language"]["instruction_sha256"]

    forged = copy.deepcopy(receipt)
    forged["reason"] = IRREDUCIBLE_UNCERTAINTY
    forged_payload = {key: value for key, value in forged.items() if key != "receipt_sha256"}
    forged["receipt_sha256"] = canonical_sha256(forged_payload)
    with pytest.raises(ValueError, match="identity"):
        validate_terminal_disposition_receipt(
            forged,
            halting_reason="converged",
            halting=halting,
            loop_stability=loop,
            cognitive_action_trace=[],
            budget=final_budget,
            output_tokens=output_tokens,
            output_text=output_text,
            full_bridge_tokens_sha256=bridge_sha,
        )

    detached_instruction = copy.deepcopy(receipt)
    detached_instruction["language"]["instruction_tokens"] = [21, 22, 23]
    detached_instruction["language"]["instruction_tokens_sha256"] = canonical_sha256(
        [21, 22, 23]
    )
    detached_payload = {
        key: value for key, value in detached_instruction.items() if key != "receipt_sha256"
    }
    detached_instruction["receipt_sha256"] = canonical_sha256(detached_payload)
    with pytest.raises(ValueError, match="identity"):
        validate_terminal_disposition_receipt(
            detached_instruction,
            halting_reason="converged",
            halting=halting,
            loop_stability=loop,
            cognitive_action_trace=[],
            budget=final_budget,
            output_tokens=output_tokens,
            output_text=output_text,
            full_bridge_tokens_sha256=bridge_sha,
        )


def test_resident_language_cannot_claim_an_unapplied_instruction():
    decision = classify_terminal_disposition(
        halting_reason="converged",
        halting=_halting(),
        loop_stability=_loop(),
        cognitive_action_trace=[],
        budget=_budget(),
    )
    with pytest.raises(ValueError, match="lacks its language instruction"):
        finalize_terminal_disposition_receipt(
            decision,
            instruction_tokens=[],
            full_bridge_tokens=[],
            output_tokens=[1],
            output_text="answer",
            output_source="resident_model_decode",
        )
