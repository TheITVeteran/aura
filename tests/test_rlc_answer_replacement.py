from __future__ import annotations

import copy
import hashlib

import pytest

from core.brain.llm.latent_cortex.answer_replacement import (
    MAX_BASELINE_EVIDENCE_TOKENS,
    MAX_REPLACEMENT_OUTPUT_TOKENS,
    build_answer_replacement_receipt,
    validate_answer_replacement_receipt,
)
from core.brain.llm.latent_cortex.atomic_decomposition import (
    build_atomic_decomposition,
)
from core.brain.llm.latent_cortex.diagnostic_action_selector import (
    build_candidate_routes,
    build_diagnostic_action_selector_receipt,
)
from core.brain.llm.latent_cortex.epistemic_state import OperationKind
from core.brain.llm.latent_cortex.local_repair import (
    build_local_repair_receipt,
    prepare_local_repair_requests,
)
from core.brain.llm.latent_cortex.value_of_computation import (
    build_evidence_snapshot,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _scenario(
    *,
    left: str,
    right: str,
    repaired: str,
) -> tuple[str, dict[int, str], dict, dict, dict, dict[str, dict]]:
    objective = "Return the exactly correct arithmetic answer."
    candidates = {0: left, 1: right}
    decompositions = {
        str(index): build_atomic_decomposition(text, objective=objective)
        for index, text in candidates.items()
    }
    graph_payload = {
        "n_branches": 2,
        "candidate_decompositions": decompositions,
        "branches": [
            {
                "index": index,
                "operator_transition_count": 1,
                "operator_program_sha256": _digest(f"program-{index}"),
                "candidate_decomposition_sha256": decompositions[str(index)]["receipt_sha256"],
            }
            for index in range(2)
        ],
        "pairwise": [
            {
                "left": 0,
                "right": 1,
                "localized": True,
                "causal_divergence": {
                    "available": True,
                    "kind": "causal_transition",
                    "action_step": 1,
                },
                "candidate_divergence": {
                    "available": True,
                    "kind": "atomic_claim",
                    "atom_ordinal": 0,
                    "left": {
                        "atom_id": "a000",
                        "text_sha256": decompositions["0"]["atoms"][0]["text_sha256"],
                    },
                    "right": {
                        "atom_id": "a000",
                        "text_sha256": decompositions["1"]["atoms"][0]["text_sha256"],
                    },
                },
            }
        ],
    }
    graph = {**graph_payload, "receipt_sha256": _digest("answer-graph")}
    routes = build_candidate_routes(
        candidates,
        objective=objective,
        candidate_decompositions=decompositions,
    )
    snapshot = build_evidence_snapshot(bucket="answer-replacement", cells={})
    selector = build_diagnostic_action_selector_receipt(
        disagreement_graph=graph,
        candidate_routes=routes,
        action_policy_evidence=snapshot,
        value_policy={
            "snapshot_sha256": snapshot["snapshot_sha256"],
            "executors": [
                OperationKind.CHECK_ASSUMPTION.value,
                OperationKind.REGENERATE_FROM_PREFIX.value,
            ],
        },
        action_trace=[
            {
                "state_signal": {
                    "has_memory": False,
                    "has_evidence": False,
                    "has_verifier": True,
                    "has_savepoint": True,
                }
            }
        ],
    )
    requests = prepare_local_repair_requests(
        disagreement_graph=graph,
        diagnostic_selection=selector,
        branch_candidates=candidates,
        objective=objective,
    )
    generated = (
        {
            requests[0]["request_id"]: {
                "candidate": repaired,
                "generation_context": {
                    "prompt_sha256": requests[0]["prompt_sha256"],
                    "generated_token_count": 16,
                    "termination": "contract_complete",
                    "initial_cache_offsets": [0, 0],
                    "final_cache_offsets": [16, 16],
                    "all_initial_offsets_zero": True,
                    "solver_context_imported": False,
                    "parameter_relation": "shared_resident_checkpoint",
                },
            }
        }
        if requests
        else {}
    )
    local_repair = build_local_repair_receipt(
        disagreement_graph=graph,
        diagnostic_selection=selector,
        branch_candidates=candidates,
        objective=objective,
        generated_repairs=generated,
    )
    return objective, candidates, graph, selector, local_repair, generated


def _encode(value: str) -> list[int]:
    return list(value.encode("utf-8"))


def _decode(tokens) -> str:
    return bytes(tokens).decode("utf-8")


def _build(
    *,
    left: str = "2 + 2 = 5.",
    right: str = "2 + 2 = 4.",
    repaired: str = "2 + 2 = 4.",
    selected_branch: int = 0,
    enabled: bool = True,
    baseline_text: str | None = None,
):
    objective, candidates, graph, selector, local_repair, generated = _scenario(
        left=left,
        right=right,
        repaired=repaired,
    )
    baseline_text = left if baseline_text is None else baseline_text
    receipt, tokens, private = build_answer_replacement_receipt(
        disagreement_graph=graph,
        diagnostic_selection=selector,
        local_repair=local_repair,
        selected_branch=selected_branch,
        branch_candidates=candidates,
        generated_repairs=generated,
        objective=objective,
        baseline_text=baseline_text,
        baseline_tokens=_encode(baseline_text),
        encode=_encode,
        decode=_decode,
        enabled=enabled,
        margin=0.05,
        max_output_tokens=64,
    )
    return receipt, tokens, graph, selector, local_repair, private, objective


def test_complete_exact_repair_replaces_only_after_nonoverlap_margin():
    (
        receipt,
        tokens,
        graph,
        selector,
        local_repair,
        private,
        objective,
    ) = _build()

    assert receipt["intended_decision"] == "replace"
    assert receipt["decision"] == "replace"
    assert receipt["answer_selection_effect"] == "replaced"
    candidate = receipt["candidates"][0]
    assert candidate["source_branch_quality"]["lower_bound"] == 0.0
    assert candidate["source_branch_quality"]["upper_bound"] == 0.0
    assert candidate["replacement_quality"]["lower_bound"] == 1.0
    assert candidate["replacement_quality"]["upper_bound"] == 1.0
    assert candidate["same_verifier_class"] is True
    assert candidate["dominates"] is True
    assert _decode(tokens) == "2 + 2 = 4."
    validate_answer_replacement_receipt(
        receipt,
        disagreement_graph=graph,
        diagnostic_selection=selector,
        local_repair=local_repair,
        private_evidence=private,
        expected_objective=objective,
        expected_selected_branch=0,
        expected_enabled=True,
        expected_margin=0.05,
        expected_max_output_tokens=64,
        expected_output_text="2 + 2 = 4.",
        expected_output_tokens=tokens,
    )


def test_unknown_claim_keeps_repair_interval_open_and_forces_abstention():
    receipt, tokens, *_ = _build(
        left="2 + 2 = 5. This is the answer.",
        right="2 + 2 = 4. This is the answer.",
        repaired="2 + 2 = 4. This is the answer.",
    )

    assert receipt["candidates"][0]["replacement_quality"]["basis"] == (
        "incomplete_semantic_exact_coverage"
    )
    assert receipt["candidates"][0]["replacement_quality"]["upper_bound"] == 1.0
    assert receipt["decision"] == "abstain"
    assert receipt["answer_selection_effect"] == "abstained"
    assert tokens == []


def test_refutation_on_nonselected_branch_does_not_replace_selected_answer():
    receipt, tokens, *_ = _build(
        selected_branch=1,
        baseline_text="2 + 2 = 4.",
    )

    assert receipt["decision"] == "retain"
    assert receipt["reason"] == "final_decode_already_exactly_verified"
    assert _decode(tokens) == "2 + 2 = 4."


def test_explicit_disable_retains_baseline_without_borrowing_authority():
    receipt, tokens, *_ = _build(enabled=False)

    assert receipt["intended_decision"] == "retain"
    assert receipt["decision"] == "retain"
    assert receipt["reason"] == "answer_replacement_disabled"
    assert _decode(tokens) == "2 + 2 = 5."


def test_no_repair_candidate_is_public_noop_without_private_evidence():
    objective, candidates, graph, selector, local_repair, generated = _scenario(
        left="2 + 2 = 4.",
        right="2 + 2 = 4.",
        repaired="2 + 2 = 4.",
    )
    assert local_repair["requests"] == []
    baseline = "2 + 2 = 4."
    receipt, tokens, private = build_answer_replacement_receipt(
        disagreement_graph=graph,
        diagnostic_selection=selector,
        local_repair=local_repair,
        selected_branch=0,
        branch_candidates=candidates,
        generated_repairs=generated,
        objective=objective,
        baseline_text=baseline,
        baseline_tokens=_encode(baseline),
        encode=_encode,
        decode=_decode,
        enabled=True,
        margin=0.05,
        max_output_tokens=64,
    )

    assert private == {}
    assert receipt["private_evidence_required"] is False
    assert receipt["candidates"] == []
    assert receipt["decision"] == "retain"
    assert receipt["reason"] == "no_local_repair_candidates"
    validate_answer_replacement_receipt(
        receipt,
        disagreement_graph=graph,
        diagnostic_selection=selector,
        local_repair=local_repair,
        private_evidence={},
        expected_objective=objective,
        expected_selected_branch=0,
        expected_enabled=True,
        expected_margin=0.05,
        expected_max_output_tokens=64,
        expected_output_text=baseline,
        expected_output_tokens=tokens,
    )
    with pytest.raises(ValueError, match="retained private evidence"):
        validate_answer_replacement_receipt(
            receipt,
            disagreement_graph=graph,
            diagnostic_selection=selector,
            local_repair=local_repair,
            private_evidence={"unexpected": "candidate"},
            expected_objective=objective,
            expected_selected_branch=0,
            expected_enabled=True,
            expected_margin=0.05,
            expected_max_output_tokens=64,
            expected_output_text=baseline,
            expected_output_tokens=tokens,
        )


def test_no_repair_budget_never_returns_a_deterministically_refuted_decode():
    objective, candidates, graph, selector, local_repair, generated = _scenario(
        left="2 + 2 = 4.",
        right="2 + 2 = 4.",
        repaired="2 + 2 = 4.",
    )
    assert local_repair["requests"] == []
    baseline = "2 + 2 = 5."

    receipt, tokens, private = build_answer_replacement_receipt(
        disagreement_graph=graph,
        diagnostic_selection=selector,
        local_repair=local_repair,
        selected_branch=0,
        branch_candidates=candidates,
        generated_repairs=generated,
        objective=objective,
        baseline_text=baseline,
        baseline_tokens=_encode(baseline),
        encode=_encode,
        decode=_decode,
        enabled=True,
        margin=0.05,
        max_output_tokens=64,
    )

    assert private["baseline_text"] == baseline
    assert receipt["private_evidence_required"] is True
    assert receipt["baseline_quality"]["basis"] == "deterministic_exact_refutation"
    assert receipt["decision"] == "abstain"
    assert receipt["reason"] == "known_refutation_has_no_dominant_repair"
    assert tokens == []


def test_output_text_tamper_is_rejected_by_service_reconstruction():
    receipt, tokens, graph, selector, local_repair, private, objective = _build()

    with pytest.raises(ValueError, match="output binding"):
        validate_answer_replacement_receipt(
            receipt,
            disagreement_graph=graph,
            diagnostic_selection=selector,
            local_repair=local_repair,
            private_evidence=private,
            expected_objective=objective,
            expected_selected_branch=0,
            expected_enabled=True,
            expected_margin=0.05,
            expected_max_output_tokens=64,
            expected_output_text="2 + 2 = 9.",
            expected_output_tokens=tokens,
        )


def test_policy_margin_tamper_cannot_create_replacement_authority():
    receipt, tokens, graph, selector, local_repair, private, objective = _build()
    tampered = copy.deepcopy(receipt)
    tampered["policy"]["margin"] = 0.0

    with pytest.raises(ValueError, match="commitment"):
        validate_answer_replacement_receipt(
            tampered,
            disagreement_graph=graph,
            diagnostic_selection=selector,
            local_repair=local_repair,
            private_evidence=private,
            expected_objective=objective,
            expected_selected_branch=0,
            expected_enabled=True,
            expected_margin=0.05,
            expected_max_output_tokens=64,
            expected_output_text="2 + 2 = 4.",
            expected_output_tokens=tokens,
        )


def test_stale_local_repair_commitment_is_rejected():
    receipt, tokens, graph, selector, local_repair, private, objective = _build()
    stale = copy.deepcopy(local_repair)
    stale["receipt_sha256"] = "0" * 64

    with pytest.raises(ValueError):
        validate_answer_replacement_receipt(
            receipt,
            disagreement_graph=graph,
            diagnostic_selection=selector,
            local_repair=stale,
            private_evidence=private,
            expected_objective=objective,
            expected_selected_branch=0,
            expected_enabled=True,
            expected_margin=0.05,
            expected_max_output_tokens=64,
            expected_output_text="2 + 2 = 4.",
            expected_output_tokens=tokens,
        )


def test_failed_tokenizer_roundtrip_abstains_instead_of_silent_retain():
    objective, candidates, graph, selector, local_repair, generated = _scenario(
        left="2 + 2 = 5.",
        right="2 + 2 = 4.",
        repaired="2 + 2 = 4.",
    )
    receipt, tokens, _private = build_answer_replacement_receipt(
        disagreement_graph=graph,
        diagnostic_selection=selector,
        local_repair=local_repair,
        selected_branch=0,
        branch_candidates=candidates,
        generated_repairs=generated,
        objective=objective,
        baseline_text="2 + 2 = 5.",
        baseline_tokens=[1],
        encode=lambda _value: [1],
        decode=lambda _tokens: "different text",
        max_output_tokens=64,
    )

    assert receipt["intended_decision"] == "replace"
    assert receipt["decision"] == "abstain"
    assert receipt["reason"] == "dominant_repair_output_binding_failed"
    assert tokens == []


def test_tokenizer_expansion_beyond_output_ceiling_fails_closed():
    objective, candidates, graph, selector, local_repair, generated = _scenario(
        left="2 + 2 = 5.",
        right="2 + 2 = 4.",
        repaired="2 + 2 = 4.",
    )
    oversized = list(range(65))
    receipt, tokens, _private = build_answer_replacement_receipt(
        disagreement_graph=graph,
        diagnostic_selection=selector,
        local_repair=local_repair,
        selected_branch=0,
        branch_candidates=candidates,
        generated_repairs=generated,
        objective=objective,
        baseline_text="2 + 2 = 5.",
        baseline_tokens=[1],
        encode=lambda _value: oversized,
        decode=lambda _tokens: "2 + 2 = 4.",
        max_output_tokens=64,
    )

    assert receipt["decision"] == "abstain"
    assert receipt["accepted_output"]["binding_status"] == "failed_closed"
    assert tokens == []


def test_true_arithmetic_with_false_prose_cannot_receive_certain_interval():
    receipt, tokens, *_ = _build(
        left="2 + 2 = 5 and Earth is flat.",
        right="2 + 2 = 4 and Earth is flat.",
        repaired="2 + 2 = 4 and Earth is flat.",
    )

    quality = receipt["candidates"][0]["replacement_quality"]
    assert quality["basis"] == "incomplete_semantic_exact_coverage"
    assert quality["lower_bound"] == 0.0
    assert quality["upper_bound"] == 1.0
    assert receipt["decision"] == "abstain"
    assert tokens == []


def test_python_parse_success_is_syntax_evidence_not_semantic_certainty():
    receipt, tokens, *_ = _build(
        left="```python\nif True print('bad')\n```",
        right="```python\nprint('valid')\n```",
        repaired="```python\nprint('valid')\n```",
    )

    quality = receipt["candidates"][0]["replacement_quality"]
    assert quality["basis"] == "incomplete_semantic_exact_coverage"
    assert quality["semantic_exact_verified_count"] == 0
    assert receipt["decision"] == "abstain"
    assert tokens == []


def test_actual_final_decode_is_the_comparator_not_short_branch_probe():
    receipt, tokens, *_ = _build(
        baseline_text="2 + 2 = 4.",
    )

    assert receipt["selected_branch_quality"]["basis"] == ("deterministic_exact_refutation")
    assert receipt["baseline_quality"]["basis"] == ("full_span_semantic_exact_complete")
    assert receipt["decision"] == "retain"
    assert receipt["reason"] == "final_decode_already_exactly_verified"
    assert _decode(tokens) == "2 + 2 = 4."


def test_refuted_selected_branch_abstains_when_request_budget_omits_it():
    objective, candidates, graph, selector, local_repair, generated = _scenario(
        left="2 + 2 = 5.",
        right="3 + 3 = 7.",
        repaired="2 + 2 = 4.",
    )
    assert local_repair["request_count"] == 1
    assert local_repair["requests"][0]["branch"] == 0
    receipt, tokens, _private = build_answer_replacement_receipt(
        disagreement_graph=graph,
        diagnostic_selection=selector,
        local_repair=local_repair,
        selected_branch=1,
        branch_candidates=candidates,
        generated_repairs=generated,
        objective=objective,
        baseline_text="ordinary unverified decode",
        baseline_tokens=[1],
        encode=_encode,
        decode=_decode,
        max_output_tokens=64,
    )

    assert receipt["selected_branch_quality"]["basis"] == ("deterministic_exact_refutation")
    assert receipt["decision"] == "abstain"
    assert tokens == []


def test_service_reexecution_rejects_tampered_private_baseline():
    receipt, tokens, graph, selector, local_repair, private, objective = _build()
    tampered_private = copy.deepcopy(private)
    tampered_private["baseline_text"] = "2 + 2 = 4."

    with pytest.raises(ValueError, match="private evidence"):
        validate_answer_replacement_receipt(
            receipt,
            disagreement_graph=graph,
            diagnostic_selection=selector,
            local_repair=local_repair,
            private_evidence=tampered_private,
            expected_objective=objective,
            expected_selected_branch=0,
            expected_enabled=True,
            expected_margin=0.05,
            expected_max_output_tokens=64,
            expected_output_text="2 + 2 = 4.",
            expected_output_tokens=tokens,
        )


def test_service_reexecution_rejects_tampered_private_baseline_tokens():
    receipt, tokens, graph, selector, local_repair, private, objective = _build()
    tampered_private = copy.deepcopy(private)
    tampered_private["baseline_tokens"] = [9, 9, 9]

    with pytest.raises(ValueError, match="private evidence"):
        validate_answer_replacement_receipt(
            receipt,
            disagreement_graph=graph,
            diagnostic_selection=selector,
            local_repair=local_repair,
            private_evidence=tampered_private,
            expected_objective=objective,
            expected_selected_branch=0,
            expected_enabled=True,
            expected_margin=0.05,
            expected_max_output_tokens=64,
            expected_output_text="2 + 2 = 4.",
            expected_output_tokens=tokens,
        )


def test_private_baseline_evidence_accepts_engine_decode_beyond_replacement_limit():
    objective, candidates, graph, selector, local_repair, generated = _scenario(
        left="2 + 2 = 5.",
        right="2 + 2 = 4.",
        repaired="2 + 2 = 4.",
    )
    baseline_tokens = [1] * (MAX_REPLACEMENT_OUTPUT_TOKENS + 1)

    receipt, accepted_tokens, private = build_answer_replacement_receipt(
        disagreement_graph=graph,
        diagnostic_selection=selector,
        local_repair=local_repair,
        selected_branch=0,
        branch_candidates=candidates,
        generated_repairs=generated,
        objective=objective,
        baseline_text="ordinary unverified decode",
        baseline_tokens=baseline_tokens,
        encode=_encode,
        decode=_decode,
        max_output_tokens=64,
    )

    assert len(private["baseline_tokens"]) == MAX_REPLACEMENT_OUTPUT_TOKENS + 1
    assert receipt["baseline_decode"]["token_count"] == len(baseline_tokens)
    assert receipt["decision"] == "abstain"
    assert accepted_tokens == []
    validate_answer_replacement_receipt(
        receipt,
        disagreement_graph=graph,
        diagnostic_selection=selector,
        local_repair=local_repair,
        private_evidence=private,
        expected_objective=objective,
        expected_selected_branch=0,
        expected_enabled=True,
        expected_margin=0.05,
        expected_max_output_tokens=64,
        expected_output_text="",
        expected_output_tokens=accepted_tokens,
    )


def test_private_baseline_evidence_rejects_tokens_beyond_engine_decode_envelope():
    objective, candidates, graph, selector, local_repair, generated = _scenario(
        left="2 + 2 = 5.",
        right="2 + 2 = 4.",
        repaired="2 + 2 = 4.",
    )

    with pytest.raises(ValueError, match="baseline token limit exceeded"):
        build_answer_replacement_receipt(
            disagreement_graph=graph,
            diagnostic_selection=selector,
            local_repair=local_repair,
            selected_branch=0,
            branch_candidates=candidates,
            generated_repairs=generated,
            objective=objective,
            baseline_text="ordinary unverified decode",
            baseline_tokens=[1] * (MAX_BASELINE_EVIDENCE_TOKENS + 1),
            encode=_encode,
            decode=_decode,
            max_output_tokens=64,
        )


def test_rejected_generated_repair_has_no_private_authority_or_fallback():
    objective, candidates, graph, selector, local_repair, generated = _scenario(
        left="2 + 2 = 5.",
        right="2 + 2 = 4.",
        repaired="2 + 2 = 5.",
    )
    assert local_repair["transactions"][0]["status"] == ("repaired_candidate_rejected")
    receipt, tokens, private = build_answer_replacement_receipt(
        disagreement_graph=graph,
        diagnostic_selection=selector,
        local_repair=local_repair,
        selected_branch=0,
        branch_candidates=candidates,
        generated_repairs=generated,
        objective=objective,
        baseline_text="ordinary unverified decode",
        baseline_tokens=[1],
        encode=_encode,
        decode=_decode,
        max_output_tokens=64,
    )

    assert private["generated_repairs"] == {}
    assert receipt["decision"] == "abstain"
    assert tokens == []
