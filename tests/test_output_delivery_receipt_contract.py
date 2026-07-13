from __future__ import annotations

import copy

import pytest

from core.runtime.receipts import (
    OutputReceipt,
    digest_output_content,
    digest_principal_binding,
    validate_transport_output_receipt,
)


def _valid_receipt() -> OutputReceipt:
    return OutputReceipt(
        receipt_id="output-contract-1",
        cause="test",
        origin="api",
        target="primary",
        digest=digest_output_content("delivered response"),
        metadata={
            "delivery_stage": "transport_accepted",
            "accepted_sinks": ["http_response_body"],
            "recipient_principal_digest": digest_principal_binding("bryan"),
        },
    )


def test_transport_output_receipt_requires_exact_content_and_principal() -> None:
    receipt = _valid_receipt()

    assert validate_transport_output_receipt(
        receipt,
        content="delivered response",
        principal="bryan",
    )
    assert not validate_transport_output_receipt(
        receipt,
        content="different response",
        principal="bryan",
    )
    assert not validate_transport_output_receipt(
        receipt,
        content="delivered response",
        principal="alice",
    )


@pytest.mark.parametrize(
    "tamper",
    [
        "missing_receipt_id",
        "wrong_kind",
        "secondary_target",
        "unaccepted_stage",
        "empty_sinks",
        "scalar_sinks",
        "unknown_sink",
        "non_string_sink",
    ],
)
def test_transport_output_receipt_rejects_malformed_delivery_claim(tamper: str) -> None:
    receipt = copy.deepcopy(_valid_receipt())
    if tamper == "missing_receipt_id":
        receipt.receipt_id = ""
    elif tamper == "wrong_kind":
        receipt.kind = "turn"
    elif tamper == "secondary_target":
        receipt.target = "secondary"
    elif tamper == "unaccepted_stage":
        receipt.metadata["delivery_stage"] = "queued"
    elif tamper == "empty_sinks":
        receipt.metadata["accepted_sinks"] = []
    elif tamper == "scalar_sinks":
        receipt.metadata["accepted_sinks"] = "http_response_body"
    elif tamper == "unknown_sink":
        receipt.metadata["accepted_sinks"] = ["unverified_transport"]
    elif tamper == "non_string_sink":
        receipt.metadata["accepted_sinks"] = [42]

    assert not validate_transport_output_receipt(
        receipt,
        content="delivered response",
        principal="bryan",
    )


def test_principal_binding_fails_closed_for_empty_or_unencodable_identity() -> None:
    assert digest_principal_binding("") == ""
    assert digest_principal_binding("\ud800") == ""
