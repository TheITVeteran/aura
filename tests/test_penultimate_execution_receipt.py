"""SPARK-066: the CP227 defect, made unable to happen again.

CP227's accuracy gate was voided because `_decode` ran outside
`recurrence_adapter_scope`: both arms decoded the bare base model and the
matching numbers looked like a clean null. These tests reconstruct that exact
shape and require it to be refused before it can become a verdict.
"""

from __future__ import annotations

import hashlib

import pytest

from core.brain.llm.latent_cortex.penultimate_execution_receipt import (
    ORDINARY_GENERATION,
    PROVEN,
    RECURRENT_LATENT,
    REFUSED,
    SHALLOW_ORCHESTRATION,
    PenultimateReceiptError,
    latent_execution_verdict,
    penultimate_execution_receipt,
    recurrent_pass,
    validate_penultimate_receipt,
)

_LAYERS = 64


def _d(tag: str) -> str:
    return hashlib.sha256(tag.encode("utf-8")).hexdigest()


def _identity() -> dict:
    return {
        "checkpoint_sha256": _d("qwen2.5-32b"),
        "tokenizer_sha256": _d("tokenizer"),
        "parameter_count": 32_000_000_000,
        "quantization": "4bit",
        "layer_count": _LAYERS,
    }


def _adapter(attached: bool = True, activated: list[int] | None = None) -> dict:
    blocks = [40, 41, 42, 43]
    return {
        "adapter_sha256": _d("rlc-adapter") if attached else None,
        "attached": attached,
        "expected_blocks": blocks if attached else [],
        "activated_blocks": (
            (blocks if activated is None else activated) if attached else []
        ),
    }


def _passes(count: int, *, identical: bool = False) -> list[dict]:
    return [
        recurrent_pass(
            ordinal=index,
            state_sha256=_d("state" if identical else f"state-{index}"),
            delta_l2=0.0 if identical else 0.31 - 0.05 * index,
        )
        for index in range(count)
    ]


def _receipt(**overrides) -> dict:
    passes = overrides.pop("passes", _passes(4))
    kwargs = {
        "mechanism": RECURRENT_LATENT,
        "identity": _identity(),
        "adapter": _adapter(),
        "layer_index": _LAYERS - 2,
        "passes": passes,
        "decode_state_sha256": (
            passes[-1]["state_sha256"] if passes else _d("no-passes")
        ),
        "decoded_token_count": 128,
        "answer_sha256": _d("answer"),
        "fallback_occurred": False,
        "fallback_reason": None,
    }
    kwargs.update(overrides)
    return penultimate_execution_receipt(**kwargs)


# --- the healthy case -------------------------------------------------------


def test_a_real_latent_run_is_proven():
    verdict = latent_execution_verdict(_receipt(), require_adapter=True)
    assert verdict["verdict"] == PROVEN
    assert verdict["reasons"] == []
    assert verdict["passes"] == 4
    assert verdict["distinct_pass_states"] == 4


def test_a_receipt_re_derives_from_its_own_fields():
    receipt = _receipt()
    assert validate_penultimate_receipt(receipt) == receipt


# --- 1. the adapter that was never attached (the CP227 shape) ---------------


def test_an_adapter_that_claims_attachment_but_fired_nowhere_is_refused():
    with pytest.raises(PenultimateReceiptError) as excinfo:
        _receipt(adapter=_adapter(attached=True, activated=[]))
    assert "adapter_did_not_activate" in str(excinfo.value)


def test_an_adapter_that_fired_in_only_some_expected_blocks_is_refused():
    with pytest.raises(PenultimateReceiptError) as excinfo:
        _receipt(adapter=_adapter(attached=True, activated=[40, 41]))
    assert "adapter_did_not_activate" in str(excinfo.value)


def test_the_cp227_shape_cannot_produce_a_proven_verdict():
    # Both arms decoding the bare base model: no adapter attached at all, yet
    # the run still wants to be counted as a treatment.
    bare = _receipt(adapter=_adapter(attached=False))
    verdict = latent_execution_verdict(bare, require_adapter=True)
    assert verdict["verdict"] == REFUSED
    assert {row["reason"] for row in verdict["reasons"]} == {
        "treatment_adapter_not_attached"
    }


def test_an_unattached_adapter_may_not_carry_an_identity():
    with pytest.raises(PenultimateReceiptError):
        _receipt(
            adapter={
                "adapter_sha256": _d("rlc-adapter"),
                "attached": False,
                "expected_blocks": [],
                "activated_blocks": [],
            }
        )


def test_a_control_arm_without_an_adapter_is_a_valid_receipt():
    # The control legitimately has no treatment; it just cannot claim one.
    control = _receipt(adapter=_adapter(attached=False))
    assert latent_execution_verdict(control, require_adapter=False)["verdict"] == PROVEN


# --- 2. the depth that never recurred ---------------------------------------


def test_four_passes_that_all_landed_on_the_same_state_did_not_recur():
    receipt = _receipt(passes=_passes(4, identical=True))
    verdict = latent_execution_verdict(receipt, require_adapter=True)
    assert verdict["verdict"] == REFUSED
    assert any(row["reason"] == "no_recurrence_occurred" for row in verdict["reasons"])


def test_a_single_pass_is_not_accused_of_failing_to_recur():
    receipt = _receipt(passes=_passes(1))
    verdict = latent_execution_verdict(receipt, require_adapter=True)
    assert verdict["verdict"] == PROVEN


def test_a_run_shallower_than_required_is_refused():
    receipt = _receipt(passes=_passes(2))
    verdict = latent_execution_verdict(receipt, require_adapter=True, minimum_passes=4)
    assert verdict["verdict"] == REFUSED
    assert verdict["reasons"][0]["reason"] == "fewer_passes_than_required"


# --- 3. the state that was computed and then dropped ------------------------


def test_decoding_from_a_state_the_recurrence_did_not_produce_is_refused():
    passes = _passes(4)
    receipt = _receipt(passes=passes, decode_state_sha256=_d("somewhere-else"))
    verdict = latent_execution_verdict(receipt, require_adapter=True)
    assert verdict["verdict"] == REFUSED
    assert any(
        row["reason"] == "decode_did_not_consume_final_state"
        for row in verdict["reasons"]
    )


def test_decoding_from_an_earlier_pass_is_refused():
    passes = _passes(4)
    receipt = _receipt(passes=passes, decode_state_sha256=passes[0]["state_sha256"])
    verdict = latent_execution_verdict(receipt, require_adapter=True)
    assert verdict["verdict"] == REFUSED


# --- 4. the fallback wearing the latent name --------------------------------


def test_a_fallback_cannot_be_recorded_as_a_latent_run():
    with pytest.raises(PenultimateReceiptError) as excinfo:
        _receipt(fallback_occurred=True, fallback_reason="worker timed out")
    assert "fallback_claims_latent" in str(excinfo.value)


def test_a_fallback_needs_a_reason():
    with pytest.raises(PenultimateReceiptError) as excinfo:
        _receipt(
            mechanism=ORDINARY_GENERATION,
            fallback_occurred=True,
            fallback_reason=None,
        )
    assert "fallback_reason_missing" in str(excinfo.value)


def test_an_ordinary_generation_refuses_the_latent_claim_without_erroring():
    receipt = _receipt(
        mechanism=ORDINARY_GENERATION,
        fallback_occurred=True,
        fallback_reason="latent worker unavailable",
    )
    verdict = latent_execution_verdict(receipt, require_adapter=True)
    assert verdict["verdict"] == REFUSED
    assert {row["reason"] for row in verdict["reasons"]} >= {
        "mechanism_is_not_latent",
        "fallback_occurred",
    }


def test_shallow_orchestration_is_named_rather_than_absorbed():
    receipt = _receipt(mechanism=SHALLOW_ORCHESTRATION)
    verdict = latent_execution_verdict(receipt, require_adapter=True)
    assert verdict["verdict"] == REFUSED
    assert verdict["reasons"][0]["mechanism"] == SHALLOW_ORCHESTRATION


# --- position, identity, and tampering --------------------------------------


def test_penultimate_is_a_position_not_a_label():
    with pytest.raises(PenultimateReceiptError) as excinfo:
        _receipt(layer_index=10)
    assert "is_not_penultimate" in str(excinfo.value)


def test_a_receipt_with_no_passes_is_refused():
    with pytest.raises(PenultimateReceiptError):
        _receipt(passes=[], decode_state_sha256=_d("nothing"))


def test_passes_out_of_order_are_refused():
    passes = _passes(3)
    passes[0], passes[2] = passes[2], passes[0]
    with pytest.raises(PenultimateReceiptError) as excinfo:
        _receipt(passes=passes, decode_state_sha256=passes[-1]["state_sha256"])
    assert "out_of_order" in str(excinfo.value)


def test_editing_the_model_identity_breaks_the_receipt_digest():
    receipt = dict(_receipt())
    identity = dict(receipt["model_identity"])
    identity["quantization"] = "8bit"
    receipt["model_identity"] = identity
    with pytest.raises(PenultimateReceiptError):
        validate_penultimate_receipt(receipt)


def test_editing_the_activated_blocks_breaks_the_receipt_digest():
    receipt = dict(_receipt())
    adapter = dict(receipt["adapter"])
    adapter["activated_blocks"] = [40, 41, 42, 43, 44]
    receipt["adapter"] = adapter
    with pytest.raises(PenultimateReceiptError):
        validate_penultimate_receipt(receipt)
