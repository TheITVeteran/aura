from __future__ import annotations

import copy
import hashlib

import pytest

mx = pytest.importorskip("mlx.core")

from core.brain.llm.latent_cortex.cognitive_operators import (  # noqa: E402
    CognitiveOperator,
    execute_cognitive_operator,
)
from core.brain.llm.latent_cortex.structural_diversity import (  # noqa: E402
    build_structural_diversity_receipt,
    validate_structural_diversity_receipt,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _action_trace(*, surface_text: str = "") -> list[dict]:
    return [
        {
            "surface_text": surface_text,
            "transition": {
                "step_index": 0,
                "action": "decompose",
                "outcome": "completed",
            },
            "state_before": {"residual": 0.8, "disagreement": 0.1},
            "state_after": {"residual": 0.5, "disagreement": 0.3},
        }
    ]


def _isolation(roles: list[str]) -> dict:
    return {
        "certified": True,
        "candidates": [
            {"index": index, "candidate_sha256": _digest(f"candidate-{index}")}
            for index, _role in enumerate(roles)
        ]
    }


def _operator_rows(roles: list[str], operators: list[CognitiveOperator]) -> list[dict]:
    rows = []
    anchor = mx.ones((1, 5, 8))
    control = mx.arange(8, dtype=mx.float32)
    for index, (role, operator) in enumerate(zip(roles, operators, strict=True)):
        state = anchor + (index + 1) * 0.05 * mx.arange(40).reshape(1, 5, 8)
        _output, receipt = execute_cognitive_operator(
            state,
            anchor,
            control,
            operator=operator,
            role=role,
            branch_index=index,
            action="decompose",
            action_step=0,
            protected_slots=(4,),
        )
        rows.append(receipt)
    return rows


def _build(roles: list[str], operators: list[CognitiveOperator], *, wording: str = ""):
    return build_structural_diversity_receipt(
        n_branches=len(roles),
        cognitive_slots=[
            {
                "slot": 4,
                "source": "world_model",
                "text_sha256": _digest("same admitted premise"),
            }
        ],
        operator_trace=_operator_rows(roles, operators),
        action_trace=_action_trace(surface_text=wording),
        branch_isolation=_isolation(roles),
    )


def test_distinct_causal_structures_create_independent_support_classes():
    roles = ["constructive_solution", "counterexample_search", "causal_reconstruction"]
    operators = [
        CognitiveOperator.CONSTRUCTIVE_SOLUTION,
        CognitiveOperator.COUNTEREXAMPLE,
        CognitiveOperator.CAUSAL_SIMULATION,
    ]

    receipt = _build(roles, operators)

    assert receipt["certified"] is True
    assert receipt["wording_counted"] is False
    assert receipt["independent_support_count"] == 3
    assert receipt["duplicate_groups"] == []
    assert all(row["independent"] for row in receipt["pairwise"])
    assert all("algorithms" in row["differing_facets"] for row in receipt["pairwise"])


def test_incomplete_branch_isolation_is_diagnostic_but_never_certified():
    roles = ["constructive_solution", "counterexample_search"]
    operators = [
        CognitiveOperator.CONSTRUCTIVE_SOLUTION,
        CognitiveOperator.COUNTEREXAMPLE,
    ]
    isolation = {
        "certified": False,
        "reason": "isolation_incomplete",
        "candidates": [
            {"index": 0, "candidate_sha256": _digest("candidate-0")},
            {"index": 1, "candidate_sha256": ""},
        ],
    }

    receipt = build_structural_diversity_receipt(
        n_branches=2,
        cognitive_slots=[],
        operator_trace=_operator_rows(roles, operators),
        action_trace=_action_trace(),
        branch_isolation=isolation,
    )

    assert receipt["certified"] is False
    assert receipt["reason"] == "branch_isolation_unproven"
    assert receipt["branch_isolation_certified"] is False
    assert receipt["candidate_commitments_complete"] is False
    assert receipt["structural_independence_observed"] is True
    assert len(receipt["branches"]) == 2
    with pytest.raises(ValueError, match="not independently certified"):
        validate_structural_diversity_receipt(
            receipt,
            n_branches=2,
            cognitive_slots=[],
            operator_trace=_operator_rows(roles, operators),
            action_trace=_action_trace(),
            branch_isolation=isolation,
        )


def test_surface_paraphrase_cannot_create_or_change_structural_support():
    roles = ["constructive_solution", "counterexample_search"]
    operators = [
        CognitiveOperator.CONSTRUCTIVE_SOLUTION,
        CognitiveOperator.COUNTEREXAMPLE,
    ]

    terse = _build(roles, operators, wording="Therefore the answer is four.")
    ornate = _build(
        roles,
        operators,
        wording="Consequently, one may elegantly conclude that the result equals four.",
    )

    assert terse == ornate


def test_duplicate_causal_programs_collapse_even_when_states_differ():
    roles = ["direct_derivation", "direct_derivation"]
    operators = [CognitiveOperator.DIRECT_DERIVATION] * 2

    receipt = _build(roles, operators)

    assert receipt["certified"] is False
    assert receipt["independent_support_count"] == 1
    assert receipt["duplicate_groups"] == [[0, 1]]
    assert receipt["branches"][0]["state_commitment_sha256"] != receipt["branches"][1][
        "state_commitment_sha256"
    ]
    assert receipt["branches"][0]["structural_sha256"] == receipt["branches"][1][
        "structural_sha256"
    ]


def test_service_reconstruction_rejects_tampered_diversity_claim():
    roles = ["constructive_solution", "counterexample_search"]
    operators = [
        CognitiveOperator.CONSTRUCTIVE_SOLUTION,
        CognitiveOperator.COUNTEREXAMPLE,
    ]
    operator_trace = _operator_rows(roles, operators)
    action_trace = _action_trace()
    isolation = _isolation(roles)
    receipt = build_structural_diversity_receipt(
        n_branches=2,
        cognitive_slots=[],
        operator_trace=operator_trace,
        action_trace=action_trace,
        branch_isolation=isolation,
    )
    validate_structural_diversity_receipt(
        receipt,
        n_branches=2,
        cognitive_slots=[],
        operator_trace=operator_trace,
        action_trace=action_trace,
        branch_isolation=isolation,
    )

    tampered = copy.deepcopy(receipt)
    tampered["independent_support_count"] = 1
    with pytest.raises(ValueError, match="differs from reconstruction"):
        validate_structural_diversity_receipt(
            tampered,
            n_branches=2,
            cognitive_slots=[],
            operator_trace=operator_trace,
            action_trace=action_trace,
            branch_isolation=isolation,
        )


def test_mid_episode_role_shift_is_an_explicit_program_path():
    roles = ["constructive_solution"]
    first = _operator_rows(roles, [CognitiveOperator.CONSTRUCTIVE_SOLUTION])[0]
    anchor = mx.ones((1, 5, 8))
    state = anchor + 0.1 * mx.arange(40).reshape(1, 5, 8)
    control = mx.arange(8, dtype=mx.float32)
    _output, second = execute_cognitive_operator(
        state,
        anchor,
        control,
        operator=CognitiveOperator.COUNTEREXAMPLE,
        role="counterexample_search",
        branch_index=0,
        action="falsify",
        action_step=1,
        protected_slots=(4,),
    )
    action_trace = _action_trace()
    action_trace.append(
        {
            "transition": {
                "step_index": 1,
                "action": "falsify",
                "outcome": "verified_progress_saved",
            },
            "state_before": {"residual": 0.5, "disagreement": 0.3},
            "state_after": {"residual": 0.4, "disagreement": 0.4},
        }
    )

    receipt = build_structural_diversity_receipt(
        n_branches=1,
        cognitive_slots=[],
        operator_trace=[first, second],
        action_trace=action_trace,
        branch_isolation=_isolation(roles),
    )

    assert receipt["certified"] is True
    assert receipt["branches"][0]["role_path"] == [
        "constructive_solution",
        "counterexample_search",
    ]
    assert receipt["branches"][0]["operator_path"] == [
        "constructive_solution",
        "counterexample",
    ]
