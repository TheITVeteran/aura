from __future__ import annotations

import copy

import pytest

from core.brain.llm.latent_cortex.contract_repair import (
    build_contract_repair_receipt,
    parse_contract_repair_generation,
    prepare_contract_repair_requests,
    validate_contract_repair_receipt,
)


def _context(prompt_sha256: str) -> dict:
    return {
        "prompt_sha256": prompt_sha256,
        "generated_token_count": 12,
        "termination": "contract_complete",
        "initial_cache_offsets": [0, 0],
        "final_cache_offsets": [20, 20],
        "all_initial_offsets_zero": True,
        "solver_context_imported": False,
        "parameter_relation": "shared_resident_checkpoint",
    }


def test_only_invalid_private_candidates_receive_bounded_repair_requests():
    candidates = {
        0: "The answer is four.",
        1: 'FINAL_ANSWER: {"answer": 4}',
    }

    requests = prepare_contract_repair_requests(
        branch_candidates=candidates,
        objective="Compute two plus two.",
        max_requests=2,
    )

    assert [row["branch"] for row in requests] == [0]
    assert "The answer is four." not in str(
        {key: value for key, value in requests[0].items() if key != "prompt"}
    )
    assert requests[0]["prompt"].endswith("Begin the complete contract response now.")


def test_object_only_continuation_restores_only_the_fixed_protocol_frame():
    assert parse_contract_repair_generation('{"answer": 4}') == (
        'FINAL_ANSWER: {"answer": 4}'
    )
    with pytest.raises(ValueError, match="contract repair generation is invalid"):
        parse_contract_repair_generation("the answer is four")


def test_one_unique_json_object_can_be_canonicalized_but_ambiguity_cannot():
    duplicated = (
        '{"choice":"H","confidence":"medium"} '
        'FINAL_ANSWER: {"confidence":"medium","choice":"H"}'
    )
    assert parse_contract_repair_generation(duplicated) == (
        'FINAL_ANSWER: {"choice":"H","confidence":"medium"}'
    )
    with pytest.raises(ValueError, match="contract repair generation is invalid"):
        parse_contract_repair_generation(
            '{"answer":4} and maybe {"answer":5}'
        )
    with pytest.raises(ValueError, match="contract repair generation is invalid"):
        parse_contract_repair_generation('[{"answer":4}]')


def test_fresh_strict_contract_candidate_is_admitted_without_answer_authority():
    candidates = {0: "The answer is four."}
    request = prepare_contract_repair_requests(
        branch_candidates=candidates,
        objective="Compute two plus two.",
        max_requests=1,
    )[0]
    repaired = 'FINAL_ANSWER: {"answer": 4}'

    receipt = build_contract_repair_receipt(
        branch_candidates=candidates,
        objective="Compute two plus two.",
        generated_repairs={
            request["request_id"]: {
                "candidate": repaired,
                "generation_context": _context(request["prompt_sha256"]),
            }
        },
        max_requests=1,
        max_tokens=128,
    )

    assert receipt["attempted_count"] == 1
    assert receipt["admitted_count"] == 1
    assert receipt["candidate_effect"] == "contract_valid_candidate_pool_addition"
    assert receipt["answer_selection_effect"] == "none"
    assert repaired not in str(receipt)
    validate_contract_repair_receipt(receipt)


def test_task_shape_is_bound_into_request_and_required_for_admission():
    candidates = {0: 'FINAL_ANSWER: {"items": [1, 2], "total": 3}'}
    response_contract = '{"sequence":list[int],"checksum":int}'
    request = prepare_contract_repair_requests(
        branch_candidates=candidates,
        objective="Return the sequence and checksum.",
        response_contract=response_contract,
        max_requests=1,
    )[0]

    assert response_contract in request["prompt"]
    assert request["original_contract_reason"].startswith("response_contract:")
    with pytest.raises(ValueError, match="violates response contract"):
        parse_contract_repair_generation(
            candidates[0],
            response_contract=response_contract,
        )

    repaired = 'FINAL_ANSWER: {"sequence": [1, 2], "checksum": 3}'
    receipt = build_contract_repair_receipt(
        branch_candidates=candidates,
        objective="Return the sequence and checksum.",
        response_contract=response_contract,
        generated_repairs={
            request["request_id"]: {
                "candidate": repaired,
                "generation_context": _context(request["prompt_sha256"]),
            }
        },
        max_requests=1,
        max_tokens=128,
    )
    assert receipt["response_contract_required"] is True
    assert receipt["admitted_count"] == 1
    validate_contract_repair_receipt(receipt)


def test_unambiguous_array_is_wrapped_without_inventing_free_values():
    response_contract = (
        '{"returns":list[{"state":list[[str,int]],"pressure":list[int]}],'
        '"time_complexity":"O(n^2)"}'
    )
    raw = '[{"state": [["a", 1]], "pressure": [2]}]'
    repaired = parse_contract_repair_generation(
        raw,
        response_contract=response_contract,
    )
    assert repaired == (
        'FINAL_ANSWER: {"returns":[{"pressure":[2],"state":[["a",1]]}],'
        '"time_complexity":"O(n^2)"}'
    )

    with pytest.raises(ValueError, match="contract repair generation is invalid"):
        parse_contract_repair_generation(
            "[1, 2]",
            response_contract='{"left":list[int],"right":list[int]}',
        )
    with pytest.raises(ValueError, match="contract repair generation is invalid"):
        parse_contract_repair_generation(
            "[1, 2]",
            response_contract='{"items":list[int],"count":int}',
        )


def test_one_tuple_is_wrapped_as_one_list_item_without_changing_values():
    response_contract = (
        '{"returns":list[{"state":list[[str,int]],"pressure":list[int]}],'
        '"time_complexity":"O(n^2)"}'
    )
    raw = (
        '{"returns":[{"state":["ax",-1],"pressure":[1]}],'
        '"time_complexity":"O(n^2)"}\nTrailing explanation.'
    )
    repaired = parse_contract_repair_generation(
        raw,
        response_contract=response_contract,
    )
    assert repaired == (
        'FINAL_ANSWER: {"returns":[{"pressure":[1],"state":[["ax",-1]]}],'
        '"time_complexity":"O(n^2)"}'
    )


def test_schema_coercion_is_disclosed_in_the_public_receipt():
    candidates = {0: "not contracted"}
    response_contract = '{"items":list[int],"kind":"fixed"}'
    request = prepare_contract_repair_requests(
        branch_candidates=candidates,
        objective="Return the items.",
        response_contract=response_contract,
        max_requests=1,
    )[0]
    receipt = build_contract_repair_receipt(
        branch_candidates=candidates,
        objective="Return the items.",
        response_contract=response_contract,
        generated_repairs={
            request["request_id"]: {
                "candidate": "[1, 2]",
                "generation_context": _context(request["prompt_sha256"]),
            }
        },
        max_requests=1,
        max_tokens=128,
    )
    assert receipt["transactions"][0]["schema_coercion_applied"] is True
    validate_contract_repair_receipt(receipt)


def test_canonicalized_object_can_bind_an_irrecoverable_raw_termination():
    candidates = {0: "The answer is four."}
    request = prepare_contract_repair_requests(
        branch_candidates=candidates,
        objective="Compute two plus two.",
        max_requests=1,
    )[0]
    context = _context(request["prompt_sha256"])
    context["termination"] = "contract_irrecoverable"
    receipt = build_contract_repair_receipt(
        branch_candidates=candidates,
        objective="Compute two plus two.",
        generated_repairs={
            request["request_id"]: {
                "candidate": '{"answer":4} FINAL_ANSWER: invalid',
                "generation_context": context,
            }
        },
        max_requests=1,
        max_tokens=128,
    )
    assert receipt["admitted_count"] == 1
    validate_contract_repair_receipt(receipt)


def test_independently_salvaged_object_discloses_contract_incomplete_termination():
    candidates = {0: "not contracted"}
    response_contract = '{"items":list[int],"kind":"fixed"}'
    request = prepare_contract_repair_requests(
        branch_candidates=candidates,
        objective="Return the items.",
        response_contract=response_contract,
        max_requests=1,
    )[0]
    context = _context(request["prompt_sha256"])
    context["termination"] = "token_limit_contract_incomplete"
    receipt = build_contract_repair_receipt(
        branch_candidates=candidates,
        objective="Return the items.",
        response_contract=response_contract,
        generated_repairs={
            request["request_id"]: {
                "candidate": '{"items":[1],"kind":"fixed"}\nTrailing prose.',
                "generation_context": context,
            }
        },
        max_requests=1,
        max_tokens=128,
    )
    assert receipt["admitted_count"] == 1
    assert receipt["transactions"][0]["generation_context"]["termination"] == (
        "token_limit_contract_incomplete"
    )
    validate_contract_repair_receipt(receipt)


def test_malformed_generation_and_tampered_receipt_fail_closed():
    with pytest.raises(ValueError, match="contract repair generation is invalid"):
        parse_contract_repair_generation("FINAL_ANSWER: not-json")

    candidates = {0: "The answer is four."}
    request = prepare_contract_repair_requests(
        branch_candidates=candidates,
        objective="Compute two plus two.",
        max_requests=1,
    )[0]
    receipt = build_contract_repair_receipt(
        branch_candidates=candidates,
        objective="Compute two plus two.",
        execution_failures={request["request_id"]: "generation_contract_invalid"},
        max_requests=1,
        max_tokens=128,
    )
    assert receipt["admitted_count"] == 0
    assert receipt["transactions"][0]["candidate_effect"] == "none"

    tampered = copy.deepcopy(receipt)
    tampered["transactions"][0]["answer_selection_effect"] = "replaced"
    with pytest.raises(ValueError, match="commitment mismatch"):
        validate_contract_repair_receipt(tampered)


def test_zero_request_budget_produces_a_true_noop():
    receipt = build_contract_repair_receipt(
        branch_candidates={0: "not contracted"},
        objective="Answer.",
        max_requests=0,
        max_tokens=128,
    )
    assert receipt["requests"] == []
    assert receipt["transactions"] == []
    assert receipt["attempted_count"] == 0
    validate_contract_repair_receipt(receipt)
