from __future__ import annotations

import pytest

from core.brain.llm.unified_recurrent_shadow_probe_contract import (
    RECEIPT_SCHEMA,
    seal_shadow_probe_receipt,
    seal_shadow_probe_request,
    shadow_probe_receipt_errors,
    shadow_probe_request_errors,
    token_sequence_sha256,
)


def _completed_body(request_sha256: str) -> dict[str, object]:
    return {
        "schema": RECEIPT_SCHEMA,
        "request_sha256": request_sha256,
        "status": "completed",
        "reason": "matched_shadow_probe_completed",
        "package_id": "fixture",
        "controller_sha256": "c" * 64,
        "family": "khop",
        "recurrence_depth": 4,
        "input_token_count": 3,
        "expected_token_count": 2,
        "max_tokens": 2,
        "base_token_count": 2,
        "base_output_sha256": "a" * 64,
        "base_exact_match": False,
        "base_stopped_on_eos": True,
        "base_latency_ms": 10,
        "shadow_token_count": 2,
        "shadow_output_sha256": "b" * 64,
        "shadow_exact_match": True,
        "shadow_stopped_on_eos": True,
        "shadow_latency_ms": 20,
        "outputs_equal": False,
        "output_exposed": False,
        "serving_authority": False,
    }


def test_probe_request_is_bounded_and_tamper_evident() -> None:
    request = seal_shadow_probe_request([1, 2, 3], [4, 5], max_tokens=2)
    assert shadow_probe_request_errors(request) == []

    request["public_token_ids"][0] = 9
    assert "unified_recurrent_shadow_probe_request_commitment_differs" in (
        shadow_probe_request_errors(request)
    )


def test_completed_receipt_contains_measurements_but_no_output() -> None:
    request = seal_shadow_probe_request([1, 2, 3], [4, 5], max_tokens=2)
    receipt = seal_shadow_probe_receipt(_completed_body(request["request_sha256"]))

    assert shadow_probe_receipt_errors(
        receipt,
        expected_request_sha256=request["request_sha256"],
        expected_package_id="fixture",
        expected_controller_sha256="c" * 64,
    ) == []
    assert "text" not in receipt
    assert "tokens" not in receipt
    assert receipt["output_exposed"] is False
    assert receipt["serving_authority"] is False


def test_parent_binding_rejects_a_replayed_receipt() -> None:
    request = seal_shadow_probe_request([1, 2, 3], [4, 5], max_tokens=2)
    receipt = seal_shadow_probe_receipt(_completed_body(request["request_sha256"]))

    assert "unified_recurrent_shadow_probe_package_binding_differs" in (
        shadow_probe_receipt_errors(receipt, expected_package_id="another-package")
    )


def test_output_commitment_is_keyed_and_not_a_reusable_small_answer_hash() -> None:
    token_ids = [12, 999]
    first_key = b"a" * 32
    second_key = b"b" * 32

    assert token_sequence_sha256(token_ids, key=first_key) == token_sequence_sha256(
        token_ids,
        key=first_key,
    )
    assert token_sequence_sha256(token_ids, key=first_key) != token_sequence_sha256(
        token_ids,
        key=second_key,
    )

    with pytest.raises(
        ValueError,
        match="unified_recurrent_shadow_probe_output_tokens_invalid",
    ):
        token_sequence_sha256(token_ids, key=b"short")
