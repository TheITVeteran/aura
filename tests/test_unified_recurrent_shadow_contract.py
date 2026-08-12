from __future__ import annotations

import pytest

from core.brain.llm.unified_recurrent_shadow_contract import (
    LOAD_SCHEMA,
    seal_shadow_load_receipt,
    shadow_load_receipt_errors,
)


def _body(*, configured: bool = False, loaded: bool = False) -> dict[str, object]:
    return {
        "schema": LOAD_SCHEMA,
        "configured": configured,
        "loaded": loaded,
        "reason": "certified_shadow_package_loaded" if loaded else "not_configured",
        "package_id": "fixture" if loaded else "",
        "manifest_sha256": "a" * 64 if loaded else "",
        "checkpoint_sha256": "b" * 64 if loaded else "",
        "controller_sha256": "c" * 64 if loaded else "",
        "families": ["khop"] if loaded else [],
        "task_depths": [1, 2, 4] if loaded else [],
        "recurrence_depth": 4 if loaded else 0,
        "model_identity_strength": ("config_behavior_hash_and_weight_extent" if loaded else "none"),
        "mode": "shadow_only",
        "serving_authority": False,
    }


@pytest.mark.parametrize("configured,loaded", [(False, False), (True, True)])
def test_sealed_shadow_receipts_validate(configured: bool, loaded: bool) -> None:
    receipt = seal_shadow_load_receipt(_body(configured=configured, loaded=loaded))

    assert shadow_load_receipt_errors(receipt) == []


def test_configured_but_unloaded_shadow_is_never_ready() -> None:
    body = _body(configured=True, loaded=False)
    body["reason"] = "load_failed"

    with pytest.raises(
        ValueError,
        match="configured_unified_recurrent_shadow_not_loaded",
    ):
        seal_shadow_load_receipt(body)


def test_shadow_receipt_cannot_claim_serving_authority() -> None:
    body = _body(configured=True, loaded=True)
    body["serving_authority"] = True

    with pytest.raises(ValueError, match="unified_recurrent_shadow_receipt_invalid"):
        seal_shadow_load_receipt(body)


def test_receipt_tamper_is_detected() -> None:
    receipt = seal_shadow_load_receipt(_body(configured=True, loaded=True))
    receipt["recurrence_depth"] = 8

    assert "unified_recurrent_shadow_receipt_invalid" in shadow_load_receipt_errors(receipt)
