"""SPARK-056 measured runtime-integrity contracts."""

from __future__ import annotations

import pytest

from core.brain.llm.latent_cortex.runtime_integrity import (
    bind_worker_runtime_integrity,
    build_fast_weight_cleanup_proof,
    canonical_sha256,
    runtime_integrity_claim_verdict,
    runtime_integrity_safe,
    validate_fast_weight_cleanup_proof,
    validate_runtime_integrity_receipt,
)
from tests.fixtures.rlc_runtime_integrity import (
    accepted_fast_weight_learning,
    bound_runtime_integrity,
    complete_worker_identity,
    engine_runtime_integrity,
)

EPISODE_ID = "episode-spark-056"
INPUT_SHA256 = "7" * 64


def _recommit(value: dict) -> dict:
    payload = {
        key: item
        for key, item in value.items()
        if key != "receipt_sha256"
    }
    value["receipt_sha256"] = canonical_sha256(payload)
    return value


def test_engine_measurements_bind_to_exact_worker_and_reconstruct():
    engine = engine_runtime_integrity(
        episode_id=EPISODE_ID,
        input_tokens_sha256=INPUT_SHA256,
    )
    assert runtime_integrity_safe(engine, require_worker=False)
    assert runtime_integrity_safe(engine, require_worker=True) is False

    worker = complete_worker_identity()
    bound = bind_worker_runtime_integrity(
        engine,
        worker_identity=worker,
    )
    assert validate_runtime_integrity_receipt(
        bound,
        require_worker=True,
        expected_episode_id=EPISODE_ID,
        expected_input_tokens_sha256=INPUT_SHA256,
        expected_worker_identity=worker,
        expected_fast_weights_applied=False,
    ) == bound
    assert runtime_integrity_safe(
        bound,
        expected_worker_identity=worker,
        expected_fast_weights_applied=False,
    )


@pytest.mark.parametrize(
    ("section", "reason"),
    [
        ("parameters", "parameter_canary_changed"),
        ("adapted_layers", "adapted_layer_identity_changed"),
    ],
)
def test_permanent_parameter_mutation_is_measured_not_asserted(
    section: str,
    reason: str,
):
    proof = engine_runtime_integrity(
        episode_id=EPISODE_ID,
        input_tokens_sha256=INPUT_SHA256,
    )
    proof[section]["after"]["sha256"] = "8" * 64
    proof[section]["unchanged"] = False
    proof["verdict"] = {
        "engine_measurements_complete": False,
        "worker_bound": False,
        "safe_to_continue": False,
        "reasons": [reason, "worker_identity_unbound"],
    }
    _recommit(proof)

    assert validate_runtime_integrity_receipt(
        proof,
        require_worker=False,
    ) == proof
    assert runtime_integrity_safe(proof, require_worker=False) is False


def test_tokenizer_adapter_or_quantization_stack_change_refutes_reuse():
    proof = engine_runtime_integrity(
        episode_id=EPISODE_ID,
        input_tokens_sha256=INPUT_SHA256,
    )
    after = proof["serving_stack"]["after"]
    after["identity"]["worker_quantization"]["bits"] = 8
    after["identity_sha256"] = canonical_sha256(after["identity"])
    proof["serving_stack"]["unchanged"] = False
    proof["verdict"] = {
        "engine_measurements_complete": False,
        "worker_bound": False,
        "safe_to_continue": False,
        "reasons": ["serving_stack_changed", "worker_identity_unbound"],
    }
    _recommit(proof)

    assert validate_runtime_integrity_receipt(
        proof,
        require_worker=False,
    ) == proof
    assert runtime_integrity_claim_verdict(
        proof,
        "params_unchanged",
    ) == "unproven"
    assert runtime_integrity_safe(proof, require_worker=False) is False


def test_fast_weight_erase_and_cache_invalidation_are_both_required():
    learning = accepted_fast_weight_learning(
        episode_id=EPISODE_ID,
        input_tokens_sha256=INPUT_SHA256,
    )
    valid = engine_runtime_integrity(
        episode_id=EPISODE_ID,
        input_tokens_sha256=INPUT_SHA256,
        fast_weights_applied=True,
        fast_weight_learning=learning,
        probe_cache={
            "entries": 0,
            "invalidations": [
                "fast_weights_attached",
                "fast_weights_detached",
            ],
        },
    )
    assert runtime_integrity_safe(valid, require_worker=False)

    stale_cache = engine_runtime_integrity(
        episode_id=EPISODE_ID,
        input_tokens_sha256=INPUT_SHA256,
        fast_weights_applied=True,
        fast_weight_learning=learning,
        probe_cache={"entries": 1, "invalidations": []},
    )
    assert stale_cache["cache"]["safe"] is False
    assert runtime_integrity_safe(stale_cache, require_worker=False) is False

    failed_cleanup = build_fast_weight_cleanup_proof(
        episode_id=EPISODE_ID,
        input_tokens_sha256=INPUT_SHA256,
        detached=True,
        erase_proven=False,
        lease_released=True,
        conflicts=0,
        pre_probe_sha256="8" * 64,
        post_probe_sha256="9" * 64,
        layer_ids=["layers.1.o_proj", "layers.2.o_proj"],
    )
    mismatched_erase = engine_runtime_integrity(
        episode_id=EPISODE_ID,
        input_tokens_sha256=INPUT_SHA256,
        fast_weights_applied=True,
        fast_weight_learning=learning,
        fast_weight_cleanup=failed_cleanup,
        probe_cache={
            "entries": 0,
            "invalidations": [
                "fast_weights_attached",
                "fast_weights_detached",
            ],
        },
    )
    assert validate_runtime_integrity_receipt(
        mismatched_erase,
        require_worker=False,
        expected_fast_weights_applied=True,
    ) == mismatched_erase
    assert runtime_integrity_safe(
        mismatched_erase,
        require_worker=False,
        expected_fast_weights_applied=True,
    ) is False


def test_cleanup_proof_is_episode_bound_and_schema_complete():
    cleanup = build_fast_weight_cleanup_proof(
        episode_id=EPISODE_ID,
        input_tokens_sha256=INPUT_SHA256,
        detached=True,
        erase_proven=True,
        lease_released=True,
        conflicts=0,
        pre_probe_sha256="8" * 64,
        post_probe_sha256="8" * 64,
        layer_ids=["layers.1.o_proj"],
    )
    with pytest.raises(ValueError, match="episode mismatch"):
        validate_fast_weight_cleanup_proof(
            cleanup,
            expected_episode_id="substituted-episode",
            expected_input_tokens_sha256=INPUT_SHA256,
        )
    with pytest.raises(ValueError, match="input mismatch"):
        validate_fast_weight_cleanup_proof(
            cleanup,
            expected_episode_id=EPISODE_ID,
            expected_input_tokens_sha256="9" * 64,
        )

    missing_field = dict(cleanup)
    missing_field.pop("lease_released")
    missing_field["receipt_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in missing_field.items()
            if key != "receipt_sha256"
        }
    )
    with pytest.raises(ValueError, match="fields are invalid"):
        validate_fast_weight_cleanup_proof(missing_field)


def test_learning_and_cleanup_disagreement_cannot_authorize_reuse():
    learning = accepted_fast_weight_learning(
        episode_id=EPISODE_ID,
        input_tokens_sha256=INPUT_SHA256,
    )
    conflicting_cleanup = build_fast_weight_cleanup_proof(
        episode_id=EPISODE_ID,
        input_tokens_sha256=INPUT_SHA256,
        detached=True,
        erase_proven=True,
        lease_released=True,
        conflicts=0,
        pre_probe_sha256="9" * 64,
        post_probe_sha256="9" * 64,
        layer_ids=["layers.1.o_proj", "layers.2.o_proj"],
    )
    proof = engine_runtime_integrity(
        episode_id=EPISODE_ID,
        input_tokens_sha256=INPUT_SHA256,
        fast_weights_applied=True,
        fast_weight_learning=learning,
        fast_weight_cleanup=conflicting_cleanup,
        probe_cache={
            "entries": 0,
            "invalidations": [
                "fast_weights_attached",
                "fast_weights_detached",
            ],
        },
    )
    assert proof["fast_weight_erase"]["learning_agrees"] is False
    assert proof["fast_weight_erase"]["exact"] is False
    assert runtime_integrity_safe(proof, require_worker=False) is False


def test_cleanup_transaction_cannot_be_hidden_outside_fast_weight_scope():
    cleanup = build_fast_weight_cleanup_proof(
        episode_id=EPISODE_ID,
        input_tokens_sha256=INPUT_SHA256,
        detached=True,
        erase_proven=True,
        lease_released=True,
        conflicts=0,
        pre_probe_sha256="8" * 64,
        post_probe_sha256="8" * 64,
        layer_ids=["layers.1.o_proj"],
    )
    proof = engine_runtime_integrity(
        episode_id=EPISODE_ID,
        input_tokens_sha256=INPUT_SHA256,
        fast_weights_applied=False,
        fast_weight_cleanup=cleanup,
    )
    assert proof["fast_weight_erase"]["required"] is False
    assert proof["fast_weight_erase"]["exact"] is False
    assert runtime_integrity_safe(proof, require_worker=False) is False


def test_absent_cleanup_proof_remains_an_explicit_negative_measurement():
    proof = engine_runtime_integrity(
        episode_id=EPISODE_ID,
        input_tokens_sha256=INPUT_SHA256,
        fast_weights_applied=True,
        fast_weight_cleanup={},
    )
    assert validate_runtime_integrity_receipt(
        proof,
        require_worker=False,
        expected_fast_weights_applied=True,
    ) == proof
    assert proof["fast_weight_erase"]["cleanup_receipt_sha256"] == ""
    assert proof["fast_weight_erase"]["exact"] is False
    assert proof["verdict"]["engine_measurements_complete"] is False


def test_self_rehashed_verdict_and_worker_substitution_do_not_gain_authority():
    proof = engine_runtime_integrity(
        episode_id=EPISODE_ID,
        input_tokens_sha256=INPUT_SHA256,
    )
    proof["parameters"]["after"]["sha256"] = "8" * 64
    proof["parameters"]["unchanged"] = False
    proof["verdict"]["safe_to_continue"] = True
    _recommit(proof)
    with pytest.raises(ValueError, match="verdict does not reconstruct"):
        validate_runtime_integrity_receipt(proof, require_worker=False)

    expected = complete_worker_identity()
    substituted = complete_worker_identity(boot_id="9" * 32)
    bound = bound_runtime_integrity(
        episode_id=EPISODE_ID,
        input_tokens_sha256=INPUT_SHA256,
        worker_identity=substituted,
    )
    with pytest.raises(ValueError, match="differs from expected worker"):
        validate_runtime_integrity_receipt(
            bound,
            require_worker=True,
            expected_worker_identity=expected,
        )


def test_empty_measurements_and_incomplete_adapter_bytes_cannot_self_certify():
    empty_canary = engine_runtime_integrity(
        episode_id=EPISODE_ID,
        input_tokens_sha256=INPUT_SHA256,
    )
    for side in ("before", "after"):
        empty_canary["parameters"][side]["parameter_leaf_count"] = 0
        empty_canary["parameters"][side]["sampled_tensor_count"] = 0
        empty_canary["parameters"][side]["sampled_element_count"] = 0
    _recommit(empty_canary)
    with pytest.raises(ValueError, match="parameter canary is incomplete"):
        validate_runtime_integrity_receipt(
            empty_canary,
            require_worker=False,
        )

    incomplete_adapter = engine_runtime_integrity(
        episode_id=EPISODE_ID,
        input_tokens_sha256=INPUT_SHA256,
    )
    for side in ("before", "after"):
        measurement = incomplete_adapter["serving_stack"][side]
        measurement["identity"]["worker_adapters"] = [
            {
                "name": "layers.1.self_attn.o_proj",
                "type": "LoRALinear",
                "rank": 8,
                "scale": 1.0,
                "parameter_sha256": "",
                "parameter_scope": "adapter_owned_excluding_wrapped_base_v1",
            }
        ]
        measurement["identity"]["worker_adapter_stack_sha256"] = (
            canonical_sha256(measurement["identity"]["worker_adapters"])
        )
        measurement["identity_sha256"] = canonical_sha256(
            measurement["identity"]
        )
    _recommit(incomplete_adapter)
    with pytest.raises(ValueError, match="serving identity is incomplete"):
        validate_runtime_integrity_receipt(
            incomplete_adapter,
            require_worker=False,
        )


def test_checkpoint_identity_must_match_independent_expectation():
    proof = bound_runtime_integrity(
        episode_id=EPISODE_ID,
        input_tokens_sha256=INPUT_SHA256,
    )
    with pytest.raises(ValueError, match="checkpoint fingerprint mismatch"):
        validate_runtime_integrity_receipt(
            proof,
            require_worker=True,
            expected_checkpoint_fingerprint="0" * 64,
        )


def test_no_fast_weights_is_explicitly_not_required_not_fake_erasure():
    proof = bound_runtime_integrity(
        episode_id=EPISODE_ID,
        input_tokens_sha256=INPUT_SHA256,
    )
    erase = proof["fast_weight_erase"]
    assert erase == {
        "required": False,
        "learning_receipt_sha256": "",
        "cleanup_receipt_sha256": "",
        "admitted": False,
        "detached": False,
        "erase_proven": False,
        "lease_released": False,
        "conflicts": 0,
        "pre_probe_sha256": "",
        "post_probe_sha256": "",
        "layer_ids": [],
        "learning_agrees": True,
        "exact": True,
    }
    assert runtime_integrity_claim_verdict(
        proof,
        "fast_weights_erased",
        expected_fast_weights_applied=False,
    ) == "proven"
