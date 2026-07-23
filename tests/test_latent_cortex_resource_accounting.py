from __future__ import annotations

import hashlib
from copy import deepcopy
from types import SimpleNamespace

import pytest

from core.brain.llm.latent_cortex.resource_accounting import (
    ModelComputeProfile,
    ResourceLedger,
    build_information_receipt,
    certify_comparison_accounting,
    policy_sha256,
    triangular_attention_pairs,
    validate_comparison_accounting_certificate,
    validate_information_receipt,
    validate_resource_receipt,
)
from core.brain.llm.latent_cortex.types import ComputeBudget


def _model():
    args = SimpleNamespace(
        model_type="qwen2",
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=256,
        head_dim=None,
    )
    inner = SimpleNamespace(args=args, layers=[object() for _ in range(8)])
    return SimpleNamespace(args=args, model=inner)


def _information(*, prompt: str = "question", verifier: str = "same"):
    payload = prompt.encode()
    return build_information_receipt(
        sources=[
            {
                "source_id": "prompt",
                "kind": "model_input_tokens",
                "content_sha256": hashlib.sha256(payload).hexdigest(),
                "byte_count": len(payload),
                "token_count": 3,
            }
        ],
        policies={
            "decode": policy_sha256({"temperature": 0.0}),
            "verifier": policy_sha256({"identity": verifier}),
            "tools": policy_sha256({"allow": []}),
        },
    )


def _resource(
    *,
    tensor_ops: int = 100,
    tensor_reads: int = 500,
    verifier_calls: int = 2,
    unknown: bool = False,
):
    ledger = ResourceLedger(ModelComputeProfile.from_model(_model()))
    ledger.charge(
        "prefill",
        transformer_layer_apps=80,
        attention_query_key_pairs=440,
        output_head_tokens=1,
    )
    ledger.charge(
        "operators",
        tensor_element_reads=tensor_reads,
        tensor_element_writes=100,
        tensor_scalar_ops=tensor_ops,
    )
    ledger.charge(
        "verifier",
        verifier_calls=verifier_calls,
        verifier_input_bytes=400 * verifier_calls,
        verifier_output_bytes=8 * verifier_calls,
        host_scalar_ops=400 * verifier_calls,
    )
    if unknown:
        ledger.mark_unknown("mystery_probe")
    return ledger.to_receipt()


def test_model_profile_binds_derived_estimator_constants():
    profile = ModelComputeProfile.from_model(_model())
    receipt = profile.to_receipt()
    assert profile.head_dim == 16
    assert profile.dense_flops_per_token_layer > 0
    assert ModelComputeProfile.from_receipt(receipt) == profile

    tampered = dict(receipt)
    tampered["dense_flops_per_token_layer"] += 1
    with pytest.raises(ValueError, match="digest differs"):
        ModelComputeProfile.from_receipt(tampered)


def test_resource_receipt_reconstructs_totals_and_rejects_tampering():
    receipt = _resource()
    validated = validate_resource_receipt(receipt)
    assert validated["accounting_complete"] is True
    assert validated["estimated_flops"] > 0
    assert validated["totals"]["transformer_layer_apps"] == 80
    assert validated["totals"]["tensor_scalar_ops"] == 100

    tampered = {**receipt, "estimated_flops": receipt["estimated_flops"] + 1}
    with pytest.raises(ValueError, match="digest differs"):
        validate_resource_receipt(tampered)


def test_budget_separates_admission_units_from_measured_work():
    budget = ComputeBudget(max_layer_apps=10_000)
    budget.bind_model(_model())
    budget.charge(
        10,
        8,
        operation="prefill",
        attention_pairs=triangular_attention_pairs(10) * 8,
        output_head_tokens=1,
    )
    budget.charge_proxy_work(
        "latent_proxy",
        layer_app_equivalents=500,
        scalar_ops=1_000,
    )
    receipt = budget.to_receipt()
    accounting = validate_resource_receipt(receipt["resource_accounting"])
    assert budget.spent_layer_apps == 580
    assert accounting["totals"]["transformer_layer_apps"] == 80
    assert accounting["totals"]["tensor_scalar_ops"] == 1_000


def test_equal_accounting_and_information_admit_comparison():
    certificate = certify_comparison_accounting(
        treatment_resource=_resource(),
        control_resource=_resource(),
        treatment_information=_information(),
        control_information=_information(),
    )
    assert certificate["admitted"] is True
    assert certificate["information_matched"] is True
    assert not certificate["reasons"]
    assert validate_comparison_accounting_certificate(certificate) == certificate


def _rehash_certificate(certificate):
    body = {
        key: value
        for key, value in certificate.items()
        if key != "certificate_sha256"
    }
    certificate["certificate_sha256"] = policy_sha256(body)


def test_comparison_validator_recomputes_dimension_verdict_after_rehash():
    certificate = certify_comparison_accounting(
        treatment_resource=_resource(),
        control_resource=_resource(),
        treatment_information=_information(),
        control_information=_information(),
    )
    tampered = deepcopy(certificate)
    tampered["resource_dimensions"]["estimated_flops"]["within_tolerance"] = False
    _rehash_certificate(tampered)
    with pytest.raises(ValueError, match="tolerance verdict differs"):
        validate_comparison_accounting_certificate(tampered)


def test_comparison_validator_rejects_invented_reason_after_rehash():
    certificate = certify_comparison_accounting(
        treatment_resource=_resource(),
        control_resource=_resource(),
        treatment_information=_information(),
        control_information=_information(),
    )
    tampered = deepcopy(certificate)
    tampered["reasons"] = ["trust_me"]
    tampered["admitted"] = False
    _rehash_certificate(tampered)
    with pytest.raises(ValueError, match="reason is invalid"):
        validate_comparison_accounting_certificate(tampered)


@pytest.mark.parametrize(
    ("treatment", "control", "reason"),
    [
        (
            _resource(tensor_ops=10_000_000),
            _resource(tensor_ops=100),
            "resource_mismatch:estimated_flops",
        ),
        (
            _resource(tensor_reads=50_000),
            _resource(tensor_reads=500),
            "resource_mismatch:tensor_element_reads",
        ),
        (
            _resource(verifier_calls=8),
            _resource(verifier_calls=2),
            "resource_mismatch:verifier_calls",
        ),
        (
            _resource(unknown=True),
            _resource(),
            "treatment_resource_accounting_incomplete",
        ),
    ],
)
def test_hidden_or_unequal_work_refuses_comparison(treatment, control, reason):
    certificate = certify_comparison_accounting(
        treatment_resource=treatment,
        control_resource=control,
        treatment_information=_information(),
        control_information=_information(),
    )
    assert certificate["admitted"] is False
    assert reason in certificate["reasons"]


def test_information_or_policy_advantage_refuses_comparison():
    prompt_mismatch = certify_comparison_accounting(
        treatment_resource=_resource(),
        control_resource=_resource(),
        treatment_information=_information(prompt="private hint"),
        control_information=_information(),
    )
    verifier_mismatch = certify_comparison_accounting(
        treatment_resource=_resource(),
        control_resource=_resource(),
        treatment_information=_information(verifier="stronger"),
        control_information=_information(),
    )
    assert "information_or_policy_mismatch" in prompt_mismatch["reasons"]
    assert "information_or_policy_mismatch" in verifier_mismatch["reasons"]


def test_information_receipt_is_order_stable_and_tamper_evident():
    receipt = _information()
    assert validate_information_receipt(receipt) == receipt
    tampered = dict(receipt)
    tampered["unknown_accesses"] = ["unlogged_retrieval"]
    with pytest.raises(ValueError, match="differs from its sources"):
        validate_information_receipt(tampered)


def test_aggregate_preserves_every_sample_as_a_distinct_operation():
    aggregate = ResourceLedger.aggregate([_resource(), _resource()]).to_receipt()
    assert aggregate["accounting_complete"] is True
    assert aggregate["totals"]["transformer_layer_apps"] == 160
    assert set(aggregate["operations"]) == {
        "sample_0:operators",
        "sample_0:prefill",
        "sample_0:verifier",
        "sample_1:operators",
        "sample_1:prefill",
        "sample_1:verifier",
    }


def test_triangular_attention_pairs_includes_existing_context():
    assert triangular_attention_pairs(4) == 10
    assert triangular_attention_pairs(4, context_tokens=7) == 38
