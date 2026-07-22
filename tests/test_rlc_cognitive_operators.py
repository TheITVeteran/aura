"""Contracts for mechanically distinct recurrent cognitive operators."""
from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")

from core.brain.llm.latent_cortex.cognitive_operators import (  # noqa: E402
    OPERATOR_SPECS,
    CognitiveOperator,
    execute_cognitive_operator,
    validate_operator_receipt,
)

ROLE_BY_OPERATOR = {
    CognitiveOperator.DIRECT_DERIVATION: "direct_derivation",
    CognitiveOperator.CONSTRUCTIVE_SOLUTION: "constructive_solution",
    CognitiveOperator.COUNTEREXAMPLE: "counterexample_search",
    CognitiveOperator.INVERSE_REASONING: "inverse_reasoning",
    CognitiveOperator.CAUSAL_SIMULATION: "causal_simulation",
    CognitiveOperator.FORMALIZATION: "formalization",
    CognitiveOperator.ANALOGY_MAPPING: "analogy_mapping",
    CognitiveOperator.ASSUMPTION_REMOVAL: "assumption_removal",
    CognitiveOperator.BOUNDARY_CASE: "boundary_case_analysis",
}


def _states():
    z = mx.arange(1 * 6 * 16, dtype=mx.float32).reshape(1, 6, 16) / 100.0
    anchor = mx.concatenate(
        [z[:, index : index + 1, :] for index in reversed(range(6))],
        axis=1,
    ) + 0.25
    control = mx.linspace(-1.0, 1.0, 16).reshape(1, 1, 16)
    mx.eval(z, anchor, control)
    return z, anchor, control


def test_nine_operator_programs_are_causal_distinct_and_context_safe():
    z, anchor, control = _states()
    outputs = {}
    transforms = set()
    for index, operator in enumerate(CognitiveOperator):
        output, receipt = execute_cognitive_operator(
            z,
            anchor,
            control,
            operator=operator,
            role=ROLE_BY_OPERATOR[operator],
            branch_index=index,
            action="blind_resolve",
            action_step=0,
            protected_slots=(4, 5),
            comm_slot=0,
        )
        validate_operator_receipt(receipt)
        assert bool(mx.allclose(output[:, 4:, :], z[:, 4:, :]))
        assert receipt["causal"] is True
        assert receipt["changed_slots"]
        assert not set(receipt["changed_slots"]) & {4, 5}
        outputs[operator] = receipt["output_sha256"]
        transforms.add(receipt["transform"])

    assert len(outputs) == 9
    assert len(set(outputs.values())) == 9
    assert len(transforms) == 9
    assert len({spec.strength for spec in OPERATOR_SPECS.values()}) > 1


def test_operator_receipt_rejects_role_digest_and_causality_tampering():
    z, anchor, control = _states()
    _, receipt = execute_cognitive_operator(
        z,
        anchor,
        control,
        operator=CognitiveOperator.COUNTEREXAMPLE,
        role="counterexample_search",
        branch_index=0,
        action="falsify",
        action_step=2,
        protected_slots=(5,),
    )
    for patch in (
        {"role": "constructive_solution"},
        {"output_sha256": receipt["input_sha256"]},
        {"receipt_sha256": "0" * 64},
        {"changed_slots": [5]},
        {"strength": None},
    ):
        with pytest.raises(ValueError):
            validate_operator_receipt({**receipt, **patch})


def test_unknown_or_mismatched_role_cannot_become_an_operator():
    z, anchor, control = _states()
    with pytest.raises(ValueError, match="unknown executable cognitive role"):
        execute_cognitive_operator(
            z,
            anchor,
            control,
            operator=CognitiveOperator.DIRECT_DERIVATION,
            role="creative_label_only",
            branch_index=0,
            action="blind_resolve",
            action_step=0,
        )
    with pytest.raises(ValueError, match="disagree"):
        execute_cognitive_operator(
            z,
            anchor,
            control,
            operator=CognitiveOperator.FORMALIZATION,
            role="constructive_solution",
            branch_index=0,
            action="formalize",
            action_step=0,
        )
