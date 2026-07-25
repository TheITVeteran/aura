from __future__ import annotations

import copy
import hashlib
import json

import pytest

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
    parse_local_repair_generation,
    prepare_local_repair_requests,
    validate_local_repair_receipt,
)
from core.brain.llm.latent_cortex.value_of_computation import (
    build_evidence_snapshot,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _inputs(
    *,
    left: str = (
        "Prelude stays fixed. Therefore 2 + 2 = 5. Thus 10 + 1 = 11."
    ),
    right: str = (
        "Prelude stays fixed. Therefore 2 + 2 = 4. Thus 10 + 1 = 11."
    ),
) -> tuple[str, dict[int, str], dict, dict]:
    objective = "Check the arithmetic while preserving every valid prior claim."
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
                "candidate_decomposition_sha256": decompositions[str(index)][
                    "receipt_sha256"
                ],
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
                    "kind": "dependency_transition",
                    "atom_ordinal": 1,
                    "left": {
                        "atom_id": "a001",
                        "text_sha256": decompositions["0"]["atoms"][1]["text_sha256"],
                    },
                    "right": {
                        "atom_id": "a001",
                        "text_sha256": decompositions["1"]["atoms"][1]["text_sha256"],
                    },
                },
            }
        ],
    }
    graph = {**graph_payload, "receipt_sha256": _digest("graph")}
    routes = build_candidate_routes(
        candidates,
        objective=objective,
        candidate_decompositions=decompositions,
    )
    snapshot = build_evidence_snapshot(bucket="diagnostic", cells={})
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
    return objective, candidates, graph, selector


def _context(prompt_sha256: str) -> dict:
    return {
        "prompt_sha256": prompt_sha256,
        "generated_token_count": 24,
        "termination": "contract_complete",
        "initial_cache_offsets": [0, 0],
        "final_cache_offsets": [80, 80],
        "all_initial_offsets_zero": True,
        "solver_context_imported": False,
        "parameter_relation": "shared_resident_checkpoint",
    }


def _admitted_receipt() -> tuple[dict, dict, dict]:
    objective, candidates, graph, selector = _inputs()
    requests = prepare_local_repair_requests(
        disagreement_graph=graph,
        diagnostic_selection=selector,
        branch_candidates=candidates,
        objective=objective,
    )
    request = requests[0]
    repaired = parse_local_repair_generation(
        'FINAL_ANSWER: {"replacement_suffix":'
        '"Therefore 2 + 2 = 4. Thus 10 + 1 = 11."}',
        prefix=request["prefix"],
    )
    receipt = build_local_repair_receipt(
        disagreement_graph=graph,
        diagnostic_selection=selector,
        branch_candidates=candidates,
        objective=objective,
        generated_repairs={
            request["request_id"]: {
                "candidate": repaired,
                "generation_context": _context(request["prompt_sha256"]),
            }
        },
    )
    return receipt, graph, selector


def test_exact_refutation_invalidates_only_failed_node_and_descendants():
    objective, candidates, graph, selector = _inputs()

    requests = prepare_local_repair_requests(
        disagreement_graph=graph,
        diagnostic_selection=selector,
        branch_candidates=candidates,
        objective=objective,
    )

    assert len(requests) == 1
    request = requests[0]
    assert request["branch"] == 0
    assert request["failed_atom_id"] == "a001"
    assert request["last_valid_atom_id"] == "a000"
    assert request["invalidated_atom_ids"] == ["a001", "a002"]
    assert [row["atom_id"] for row in request["preserved_prefix_atoms"]] == [
        "a000"
    ]
    assert request["verified_ancestor_routes"] == []


def test_exactly_verified_ancestor_is_named_and_preserved():
    objective, candidates, graph, selector = _inputs(
        left="1 + 1 = 2. Therefore 2 + 2 = 5.",
        right="1 + 1 = 2. Therefore 2 + 2 = 4.",
    )

    request = prepare_local_repair_requests(
        disagreement_graph=graph,
        diagnostic_selection=selector,
        branch_candidates=candidates,
        objective=objective,
    )[0]

    assert request["verified_ancestor_routes"] == [
        {
            "atom_id": "a000",
            "verifier": "exact_integer_arithmetic",
            "route_sha256": request["verified_ancestor_routes"][0][
                "route_sha256"
            ],
        }
    ]
    assert len(request["verified_ancestor_routes"][0]["route_sha256"]) == 64


def test_repaired_suffix_is_admitted_without_mutating_original_branches():
    receipt, graph, selector = _admitted_receipt()

    transaction = receipt["transactions"][0]
    assert transaction["status"] == "repaired_candidate_admitted"
    assert transaction["preserved_prefix_unchanged"] is True
    assert transaction["failed_verifier_passed"] is True
    assert receipt["repair_effect"] == "candidate_pool_addition"
    assert receipt["original_branch_commitments_before"] == (
        receipt["original_branch_commitments_after"]
    )
    assert receipt["answer_selection_effect"] == "none"
    assert receipt["latent_state_effect"] == "none"
    validate_local_repair_receipt(
        receipt,
        disagreement_graph=graph,
        diagnostic_selection=selector,
    )


def test_changed_prefix_cannot_be_admitted_as_local_repair():
    objective, candidates, graph, selector = _inputs()
    request = prepare_local_repair_requests(
        disagreement_graph=graph,
        diagnostic_selection=selector,
        branch_candidates=candidates,
        objective=objective,
    )[0]
    receipt = build_local_repair_receipt(
        disagreement_graph=graph,
        diagnostic_selection=selector,
        branch_candidates=candidates,
        objective=objective,
        generated_repairs={
            request["request_id"]: {
                "candidate": (
                    "Prelude was silently changed. "
                    "Therefore 2 + 2 = 4. Thus 10 + 1 = 11."
                ),
                "generation_context": _context(request["prompt_sha256"]),
            }
        },
    )

    assert receipt["transactions"][0]["status"] == "repaired_candidate_rejected"
    assert receipt["transactions"][0]["reason"] == "preserved_prefix_changed"
    assert receipt["repair_effect"] == "none"


def test_replacement_that_remains_refuted_is_rejected():
    objective, candidates, graph, selector = _inputs()
    request = prepare_local_repair_requests(
        disagreement_graph=graph,
        diagnostic_selection=selector,
        branch_candidates=candidates,
        objective=objective,
    )[0]
    receipt = build_local_repair_receipt(
        disagreement_graph=graph,
        diagnostic_selection=selector,
        branch_candidates=candidates,
        objective=objective,
        generated_repairs={
            request["request_id"]: {
                "candidate": candidates[0],
                "generation_context": _context(request["prompt_sha256"]),
            }
        },
    )

    transaction = receipt["transactions"][0]
    assert transaction["status"] == "repaired_candidate_rejected"
    assert transaction["failed_verifier_passed"] is False
    assert transaction["repair_candidate_effect"] == "none"


def test_repair_cannot_rewrite_later_atom_outside_dependency_closure():
    objective, candidates, graph, selector = _inputs(
        left=(
            "Prelude stays fixed. Therefore 2 + 2 = 5. "
            "Independent fact remains."
        ),
        right=(
            "Prelude stays fixed. Therefore 2 + 2 = 4. "
            "Independent fact remains."
        ),
    )
    request = prepare_local_repair_requests(
        disagreement_graph=graph,
        diagnostic_selection=selector,
        branch_candidates=candidates,
        objective=objective,
    )[0]
    assert request["invalidated_atom_ids"] == ["a001"]
    receipt = build_local_repair_receipt(
        disagreement_graph=graph,
        diagnostic_selection=selector,
        branch_candidates=candidates,
        objective=objective,
        generated_repairs={
            request["request_id"]: {
                "candidate": (
                    "Prelude stays fixed. Therefore 2 + 2 = 4. "
                    "Independent fact was changed."
                ),
                "generation_context": _context(request["prompt_sha256"]),
            }
        },
    )

    transaction = receipt["transactions"][0]
    assert transaction["status"] == "repaired_candidate_rejected"
    assert transaction["reason"] == "unrelated_atom_changed"
    assert transaction["unrelated_work_unchanged"] is False


def test_absent_generation_is_an_honest_non_execution_not_a_repair():
    objective, candidates, graph, selector = _inputs()
    receipt = build_local_repair_receipt(
        disagreement_graph=graph,
        diagnostic_selection=selector,
        branch_candidates=candidates,
        objective=objective,
        execution_failures={},
    )

    assert receipt["attempted_count"] == 0
    assert receipt["admitted_count"] == 0
    assert receipt["transactions"][0]["status"] == "repair_not_executed"
    assert receipt["repair_effect"] == "none"


def test_repair_request_count_is_bounded_before_private_generation():
    objective, candidates, graph, selector = _inputs(
        left="2 + 2 = 5. Therefore 3 + 3 = 7.",
        right="2 + 2 = 4. Therefore 3 + 3 = 6.",
    )

    assert (
        prepare_local_repair_requests(
            disagreement_graph=graph,
            diagnostic_selection=selector,
            branch_candidates=candidates,
            objective=objective,
            max_requests=0,
        )
        == []
    )
    with pytest.raises(ValueError, match="request budget"):
        prepare_local_repair_requests(
            disagreement_graph=graph,
            diagnostic_selection=selector,
            branch_candidates=candidates,
            objective=objective,
            max_requests=9,
        )


@pytest.mark.parametrize(
    "value",
    [
        '{"replacement_suffix":"fixed"}',
        'prefix FINAL_ANSWER: {"replacement_suffix":"fixed"}',
        'FINAL_ANSWER: {"replacement_suffix":""}',
        'FINAL_ANSWER: {"replacement_suffix":"fixed","extra":true}',
        'FINAL_ANSWER: {"replacement_suffix":"fixed"} trailing',
    ],
)
def test_generation_contract_rejects_ambiguous_or_extra_material(value: str):
    with pytest.raises(ValueError, match="local repair generation"):
        parse_local_repair_generation(value, prefix="kept ")


def test_validator_rejects_tampered_invalidation_and_authority():
    receipt, graph, selector = _admitted_receipt()

    tampered = copy.deepcopy(receipt)
    tampered["transactions"][0]["invalidated_atom_ids"] = ["a001"]
    payload = {
        key: value for key, value in tampered.items() if key != "receipt_sha256"
    }
    tampered["receipt_sha256"] = hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        .encode("utf-8")
    ).hexdigest()
    with pytest.raises(ValueError, match="transaction binding"):
        validate_local_repair_receipt(
            tampered,
            disagreement_graph=graph,
            diagnostic_selection=selector,
        )

    authority = copy.deepcopy(receipt)
    authority["accepted_answer_effect"] = "replaced"
    authority_payload = {
        key: value for key, value in authority.items() if key != "receipt_sha256"
    }
    authority["receipt_sha256"] = hashlib.sha256(
        json.dumps(
            authority_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        .encode("utf-8")
    ).hexdigest()
    with pytest.raises(ValueError, match="summary or authority"):
        validate_local_repair_receipt(
            authority,
            disagreement_graph=graph,
            diagnostic_selection=selector,
        )


def test_stale_selector_cannot_be_rebound_to_another_graph():
    receipt, graph, selector = _admitted_receipt()
    changed_graph = copy.deepcopy(graph)
    changed_graph["receipt_sha256"] = _digest("changed")

    with pytest.raises(ValueError, match="upstream binding"):
        validate_local_repair_receipt(
            receipt,
            disagreement_graph=changed_graph,
            diagnostic_selection=selector,
        )
