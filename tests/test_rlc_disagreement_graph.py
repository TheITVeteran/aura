from __future__ import annotations

import copy
import hashlib

import pytest

mx = pytest.importorskip("mlx.core")

from core.brain.llm.latent_cortex.cognitive_operators import (  # noqa: E402
    CognitiveOperator,
    execute_cognitive_operator,
)
from core.brain.llm.latent_cortex.disagreement_graph import (  # noqa: E402
    build_disagreement_graph_receipt,
    decompose_branch_candidates,
    validate_disagreement_graph_receipt,
)
from core.brain.llm.latent_cortex.structural_diversity import (  # noqa: E402
    build_structural_diversity_receipt,
)


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _action_trace() -> list[dict]:
    return [
        {
            "transition": {
                "step_index": 0,
                "action": "decompose",
                "outcome": "completed",
            },
            "state_before": {"residual": 0.8, "disagreement": 0.2},
            "state_after": {"residual": 0.6, "disagreement": 0.3},
        },
        {
            "transition": {
                "step_index": 1,
                "action": "check_assumption",
                "outcome": "completed",
            },
            "state_before": {"residual": 0.6, "disagreement": 0.3},
            "state_after": {"residual": 0.4, "disagreement": 0.5},
        },
    ]


def _operator_row(
    *,
    branch: int,
    step: int,
    action: str,
    role: str,
    operator: CognitiveOperator,
) -> dict:
    anchor = mx.ones((1, 5, 8))
    state = anchor + (branch + step + 1) * 0.01 * mx.arange(40).reshape(1, 5, 8)
    _output, receipt = execute_cognitive_operator(
        state,
        anchor,
        mx.arange(8, dtype=mx.float32),
        operator=operator,
        role=role,
        branch_index=branch,
        action=action,
        action_step=step,
        protected_slots=(4,),
    )
    return receipt


def _operator_trace(*, diverge: bool = True) -> list[dict]:
    rows = [
        _operator_row(
            branch=branch,
            step=0,
            action="decompose",
            role="direct_derivation",
            operator=CognitiveOperator.DIRECT_DERIVATION,
        )
        for branch in range(2)
    ]
    rows.append(
        _operator_row(
            branch=0,
            step=1,
            action="check_assumption",
            role="direct_derivation",
            operator=CognitiveOperator.DIRECT_DERIVATION,
        )
    )
    rows.append(
        _operator_row(
            branch=1,
            step=1,
            action="check_assumption",
            role="formalization" if diverge else "direct_derivation",
            operator=(
                CognitiveOperator.FORMALIZATION
                if diverge
                else CognitiveOperator.DIRECT_DERIVATION
            ),
        )
    )
    return rows


def _isolation() -> dict:
    return {
        "candidates": [
            {"index": index, "candidate_sha256": _digest(f"latent-{index}")}
            for index in range(2)
        ]
    }


def _structure(operator_trace: list[dict]) -> dict:
    return build_structural_diversity_receipt(
        n_branches=2,
        cognitive_slots=[],
        operator_trace=operator_trace,
        action_trace=_action_trace(),
        branch_isolation=_isolation(),
    )


def _candidate_inputs() -> tuple[dict[str, dict], dict]:
    candidates = {
        0: (
            "The inputs remain ordered. "
            "Assuming the operation is commutative, the answer is 37."
        ),
        1: (
            "The inputs remain ordered. "
            "Preserve operand order, so the answer is 42."
        ),
    }
    decompositions = decompose_branch_candidates(
        candidates,
        objective="Determine the result while respecting operand order.",
    )
    blind = {
        "rows": [
            {
                "branch": index,
                "candidate_sha256": _digest(candidate),
            }
            for index, candidate in candidates.items()
        ]
    }
    return decompositions, blind


def _build(*, include_candidates: bool = True, diverge: bool = True) -> dict:
    operators = _operator_trace(diverge=diverge)
    decompositions, blind = _candidate_inputs()
    return build_disagreement_graph_receipt(
        n_branches=2,
        operator_trace=operators,
        action_trace=_action_trace(),
        structural_diversity=_structure(operators),
        candidate_decompositions=decompositions if include_candidates else {},
        blind_review=blind if include_candidates else {},
    )


def test_localizes_first_causal_transition_and_exact_disputed_assumption():
    receipt = _build()

    pair = receipt["pairwise"][0]
    assert pair["shared_causal_program_prefix"] == 1
    assert pair["causal_divergence"]["action_step"] == 1
    assert pair["causal_divergence"]["differing_fields"] == [
        "role",
        "operator",
        "transform",
        "changed_slots",
    ]
    assert pair["shared_exact_claim_prefix"] == 1
    assert pair["candidate_divergence"]["atom_ordinal"] == 1
    assert pair["candidate_divergence"]["kind"] == "assumption"
    assert pair["candidate_divergence"]["exact_content_committed"] is True
    assert pair["candidate_divergence"]["semantic_equivalence_claimed"] is False
    assert receipt["global_earliest"] == {
        "causal": {"action_step": 1, "left": 0, "right": 1},
        "candidate": {"atom_ordinal": 1, "left": 0, "right": 1},
    }
    assert receipt["selection_effect"] == receipt["repair_effect"] == "none"


def test_without_decoded_candidates_retains_causal_localization_only():
    receipt = _build(include_candidates=False)

    pair = receipt["pairwise"][0]
    assert pair["causal_divergence"]["available"] is True
    assert pair["candidate_divergence"] == {
        "available": False,
        "reason": "decoded_candidates_unavailable",
    }
    assert receipt["candidate_decompositions"] == {}
    assert receipt["candidate_binding_sha256"] == ""


def test_equal_causal_programs_do_not_count_distinct_tensor_states_as_disagreement():
    receipt = _build(include_candidates=False, diverge=False)

    pair = receipt["pairwise"][0]
    assert pair["shared_causal_program_prefix"] == 2
    assert pair["causal_divergence"] == {
        "available": False,
        "reason": "causal_programs_exactly_equal",
    }
    assert receipt["localized"] is False


def test_atomic_candidates_must_match_blind_review_commitments():
    operators = _operator_trace()
    decompositions, blind = _candidate_inputs()
    blind["rows"][0]["candidate_sha256"] = _digest("different candidate")

    with pytest.raises(ValueError, match="differs from blind-review"):
        build_disagreement_graph_receipt(
            n_branches=2,
            operator_trace=operators,
            action_trace=_action_trace(),
            structural_diversity=_structure(operators),
            candidate_decompositions=decompositions,
            blind_review=blind,
        )


def test_validator_reconstructs_graph_and_rejects_authority_escalation():
    operators = _operator_trace()
    structure = _structure(operators)
    decompositions, blind = _candidate_inputs()
    receipt = build_disagreement_graph_receipt(
        n_branches=2,
        operator_trace=operators,
        action_trace=_action_trace(),
        structural_diversity=structure,
        candidate_decompositions=decompositions,
        blind_review=blind,
    )
    validate_disagreement_graph_receipt(
        receipt,
        n_branches=2,
        operator_trace=operators,
        action_trace=_action_trace(),
        structural_diversity=structure,
        blind_review=blind,
    )

    tampered = copy.deepcopy(receipt)
    tampered["selection_effect"] = "winner_replaced"
    with pytest.raises(ValueError, match="differs from reconstruction"):
        validate_disagreement_graph_receipt(
            tampered,
            n_branches=2,
            operator_trace=operators,
            action_trace=_action_trace(),
            structural_diversity=structure,
            blind_review=blind,
        )


def test_candidate_paraphrase_never_becomes_claimed_semantic_equivalence():
    receipt = _build()

    assert receipt["semantic_equivalence_claimed"] is False
    assert receipt["surface_wording_authority"] == "none"
    assert all(
        row["candidate_divergence"].get("semantic_equivalence_claimed") is False
        for row in receipt["pairwise"]
        if row["candidate_divergence"]["available"]
    )
