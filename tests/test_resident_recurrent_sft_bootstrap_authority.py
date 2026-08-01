from __future__ import annotations

import copy
import json

import pytest

from core.learning.recurrence_curriculum import RECURRENCE_TRAINING_FAMILIES
from core.learning.resident_recurrent_sft_bootstrap_authority import (
    CLAIMS_NOT_SUPPORTED,
    REQUIRED_SOURCE_ROLES,
    TRAINING_AUTHORITY,
    ResidentSFTBootstrapAuthorityError,
    ResidentSFTBootstrapConfig,
    authorize_bound_artifacts,
    build_authority,
    build_dataset_commitment,
    canonical_dataset_payloads,
    sha256_bytes,
    sha256_json,
    validate_authority,
)

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
FAMILY = sorted(RECURRENCE_TRAINING_FAMILIES)[0]


def _row(task_id: str, prompt: str, *, depth: int = 2) -> dict[str, object]:
    return {
        "task_id": task_id,
        "family": FAMILY,
        "depth": depth,
        "prompt": prompt,
        "answer": 'FINAL_ANSWER: {"value":1}',
    }


def _binding(path: str, payload: bytes) -> dict[str, object]:
    return {
        "path": path,
        "sha256": sha256_bytes(payload),
        "size_bytes": len(payload),
    }


def _source_payloads() -> dict[str, bytes]:
    return {role: f"source:{role}\n".encode() for role in REQUIRED_SOURCE_ROLES}


def _authority() -> tuple[dict[str, object], bytes, bytes, dict[str, bytes]]:
    train_rows = [_row("train.1", "Solve the first recurrence task.")]
    validation_rows = [_row("validation.1", "Solve a disjoint recurrence task.")]
    train_payload, validation_payload = canonical_dataset_payloads(
        train_rows,
        validation_rows,
    )
    dataset = build_dataset_commitment(train_rows, validation_rows)
    sources = _source_payloads()
    authority = build_authority(
        campaign_id="resident-32b-recurrent-sft-bootstrap-cp782",
        committed_at="2026-08-01T01:00:00-07:00",
        expires_at="2026-08-08T01:00:00-07:00",
        model_path="training/fused-model/aura-32b",
        model_identity={"fingerprint": SHA_A, "method": "sha256", "files": 9},
        behavior_identity={
            "bundle_sha256": SHA_B,
            "file_count": 3,
            "files": [],
        },
        personality_identity={
            "identity_sha256": SHA_C,
            "present": False,
        },
        tokenizer_identity={
            "identity_sha256": SHA_A,
            "artifact_sha256": SHA_B,
            "runtime_sha256": SHA_C,
        },
        execution_spec={
            "path": "config/latent_cortex/resident.json",
            "sha256": SHA_A,
            "size_bytes": 100,
            "semantic_sha256": SHA_B,
        },
        dataset=dataset,
        dataset_artifacts={
            "train": _binding("artifacts/cp782/train.json", train_payload),
            "validation": _binding(
                "artifacts/cp782/validation.json",
                validation_payload,
            ),
        },
        sources={
            role: _binding(f"sources/{role}.py", payload)
            for role, payload in sources.items()
        },
        runtime_identity={
            "identity_sha256": SHA_C,
            "python": "3.12.0",
            "mlx": "0.test",
        },
        trust_policy={
            "path": "artifacts/cp782/trust-policy.json",
            "sha256": SHA_B,
            "size_bytes": 100,
            "semantic_sha256": SHA_C,
        },
        artifact_root="artifacts/cp782",
        config=ResidentSFTBootstrapConfig(seed=2026080107),
    )
    return authority, train_payload, validation_payload, sources


def _recompute_authority_sha256(authority: dict[str, object]) -> None:
    material = dict(authority)
    material.pop("authority_sha256")
    authority["authority_sha256"] = sha256_json(material)


def test_resident_authority_binds_nonpromotable_cached_bootstrap() -> None:
    authority, _train, _validation, _sources = _authority()

    validated = validate_authority(
        authority,
        expected_authority_sha256=authority["authority_sha256"],
    )

    assert validated["training_authority"] == TRAINING_AUTHORITY
    assert validated["trainer"]["objective"] == "cached_supervised_live_path_ce.v1"
    assert validated["model"]["base_checkpoint_immutable"] is True
    assert validated["post_training_gate"]["grpo_admission_before_gate"] is False
    assert validated["claims_not_supported"] == list(CLAIMS_NOT_SUPPORTED)
    assert validated["claim_state"]["promotion_allowed"] is False


def test_dataset_commitment_rejects_identity_and_prompt_overlap() -> None:
    with pytest.raises(
        ResidentSFTBootstrapAuthorityError,
        match="train_validation_id_overlap",
    ):
        build_dataset_commitment(
            [_row("same.1", "First prompt")],
            [_row("same.1", "Second prompt")],
        )
    with pytest.raises(
        ResidentSFTBootstrapAuthorityError,
        match="train_validation_prompt_overlap",
    ):
        build_dataset_commitment(
            [_row("train.1", "Same prompt")],
            [_row("validation.1", "Same prompt")],
        )


def test_dataset_commitment_requires_training_only_family_and_answer_contract() -> None:
    invalid_family = _row("train.1", "Prompt")
    invalid_family["family"] = "frontier_reserved"
    with pytest.raises(ResidentSFTBootstrapAuthorityError, match="not_training_only"):
        build_dataset_commitment(
            [invalid_family],
            [_row("validation.1", "Validation")],
        )
    invalid_answer = _row("train.1", "Prompt")
    invalid_answer["answer"] = "1"
    with pytest.raises(ResidentSFTBootstrapAuthorityError, match="answer_contract"):
        build_dataset_commitment(
            [invalid_answer],
            [_row("validation.1", "Validation")],
        )


def test_authority_digest_and_claim_boundary_are_both_enforced() -> None:
    authority, _train, _validation, _sources = _authority()
    tampered = copy.deepcopy(authority)
    tampered["claim_state"]["promotion_allowed"] = True
    with pytest.raises(ResidentSFTBootstrapAuthorityError, match="digest_mismatch"):
        validate_authority(tampered)

    _recompute_authority_sha256(tampered)
    with pytest.raises(ResidentSFTBootstrapAuthorityError, match="claim_boundary"):
        validate_authority(tampered)


def test_authority_rejects_recomputed_config_weakening() -> None:
    authority, _train, _validation, _sources = _authority()
    weakened = copy.deepcopy(authority)
    weakened["trainer"]["checkpoint_every"] = 2
    _recompute_authority_sha256(weakened)

    with pytest.raises(ResidentSFTBootstrapAuthorityError, match="checkpoint_must"):
        validate_authority(weakened)


def test_observed_identity_drift_is_rejected_after_valid_authority() -> None:
    authority, _train, _validation, _sources = _authority()
    changed_model = dict(authority["model"]["base_checkpoint"])
    changed_model["fingerprint"] = SHA_B

    with pytest.raises(ResidentSFTBootstrapAuthorityError, match="model_binding_drift"):
        validate_authority(authority, observed_model_identity=changed_model)


def test_bound_artifact_authorization_accepts_exact_bytes_only() -> None:
    authority, train, validation, sources = _authority()

    validated = authorize_bound_artifacts(
        authority,
        train_payload=train,
        validation_payload=validation,
        source_payloads=sources,
        expected_authority_sha256=authority["authority_sha256"],
    )
    assert validated["dataset"]["train_sha256"] == sha256_bytes(train)

    with pytest.raises(ResidentSFTBootstrapAuthorityError, match="train_artifact_binding_drift"):
        authorize_bound_artifacts(
            authority,
            train_payload=train + b" ",
            validation_payload=validation,
            source_payloads=sources,
            expected_authority_sha256=authority["authority_sha256"],
        )


def test_bound_source_authorization_rejects_missing_and_mutated_roles() -> None:
    authority, train, validation, sources = _authority()
    missing = dict(sources)
    missing.pop("trainer")
    with pytest.raises(ResidentSFTBootstrapAuthorityError, match="source_payload_roles"):
        authorize_bound_artifacts(
            authority,
            train_payload=train,
            validation_payload=validation,
            source_payloads=missing,
            expected_authority_sha256=authority["authority_sha256"],
        )

    mutated = dict(sources)
    mutated["trainer"] += b"mutation"
    with pytest.raises(ResidentSFTBootstrapAuthorityError, match="source_trainer_binding_drift"):
        authorize_bound_artifacts(
            authority,
            train_payload=train,
            validation_payload=validation,
            source_payloads=mutated,
            expected_authority_sha256=authority["authority_sha256"],
        )


def test_small_checkpoint_authority_still_rejects_resident_scale(tmp_path) -> None:
    from core.learning.structured_sft_research_authority import small_model_identity

    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen2",
                "architectures": ["Qwen2ForCausalLM"],
                "hidden_size": 5_120,
                "num_hidden_layers": 64,
                "num_attention_heads": 40,
                "num_key_value_heads": 8,
                "intermediate_size": 27_648,
                "vocab_size": 152_064,
            }
        ),
        encoding="ascii",
    )
    (tmp_path / "model.safetensors").write_bytes(b"bounded-test-weight")

    with pytest.raises(ValueError, match="model_not_small_checkpoint"):
        small_model_identity(tmp_path)
