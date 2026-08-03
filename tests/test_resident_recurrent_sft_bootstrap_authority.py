from __future__ import annotations

import copy
import json
from datetime import datetime

import pytest

from core.learning.recurrence_curriculum import RECURRENCE_TRAINING_FAMILIES
from core.learning.recurrence_native_objective_v5 import (
    GeneratedRollinSelectionConfig,
)
from core.learning.recurrence_native_objective_v6 import BranchSpecializationConfig
from core.learning.resident_recurrent_sft_bootstrap_authority import (
    AUTHORITY_SCHEMA,
    CLAIMS_NOT_SUPPORTED,
    GENERATED_ROLLIN_AUTHORITY_SCHEMA,
    GENERATED_ROLLIN_REQUIRED_SOURCE_ROLES,
    LEGACY_AUTHORITY_SCHEMA,
    LEGACY_REQUIRED_SOURCE_ROLES,
    OBJECTIVE_NAME_V2,
    OBJECTIVE_NAME_V3,
    PREVIOUS_AUTHORITY_SCHEMA,
    PREVIOUS_REQUIRED_SOURCE_ROLES,
    REQUIRED_SOURCE_ROLES,
    TRAINER_CONFIG_SCHEMA_V2,
    TRAINER_CONFIG_SCHEMA_V3,
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
from core.learning.resident_recurrent_sft_bootstrap_state import (
    authority_state_bindings,
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
        campaign_scope="full_bootstrap",
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
            role: _binding(f"sources/{role}.py", payload) for role, payload in sources.items()
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
        artifact_root_identity={"st_dev": 1, "st_ino": 2},
        config=ResidentSFTBootstrapConfig(
            seed=2026080107,
            max_steps=9,
            schema=TRAINER_CONFIG_SCHEMA_V3,
            objective=OBJECTIVE_NAME_V3,
            generated_rollin=GeneratedRollinSelectionConfig(),
            branch_specialization=BranchSpecializationConfig(),
            structural_warmup_steps=4,
            structural_warmup_learning_rate=1e-4,
            role_conditioned_branches=2,
            branch_indices=(0, 1),
        ),
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
    assert validated["campaign_scope"] == "full_bootstrap"
    assert validated["trainer"]["objective"] == OBJECTIVE_NAME_V3
    assert validated["model"]["base_checkpoint_immutable"] is True
    assert validated["post_training_gate"]["grpo_admission_before_gate"] is False
    assert validated["claims_not_supported"] == list(CLAIMS_NOT_SUPPORTED)
    assert validated["claim_state"]["promotion_allowed"] is False


def test_current_authority_requires_depth_conditioned_source_closure() -> None:
    authority, _train, _validation, _sources = _authority()

    assert authority["schema"] == AUTHORITY_SCHEMA
    assert set(authority["sources"]) == REQUIRED_SOURCE_ROLES
    assert {
        "scoped_recurrence_adapter",
        "depth_conditioning",
        "loop_core",
        "adapter_package_identity",
        "adapter_materializer",
        "paired_campaign_loader",
        "objective_policy",
        "specialization_objective",
        "role_conditioned_adapter",
    } <= set(authority["sources"])

    authority["sources"].pop("depth_conditioning")
    _recompute_authority_sha256(authority)
    with pytest.raises(ResidentSFTBootstrapAuthorityError, match="source_roles_invalid"):
        validate_authority(authority)


def test_v2_trainer_config_requires_and_binds_generated_rollin_policy() -> None:
    rollin = GeneratedRollinSelectionConfig(
        student_forcing_probability=0.7,
        sampling_temperature=0.6,
        branch_softmin_temperature=0.4,
    )
    config = ResidentSFTBootstrapConfig(
        seed=17,
        schema=TRAINER_CONFIG_SCHEMA_V2,
        objective=OBJECTIVE_NAME_V2,
        generated_rollin=rollin,
    )

    assert ResidentSFTBootstrapConfig.from_dict(config.to_dict()) == config
    assert config.to_dict()["generated_rollin"] == rollin.to_dict()
    with pytest.raises(
        ResidentSFTBootstrapAuthorityError,
        match="rollin_required",
    ):
        ResidentSFTBootstrapConfig(
            seed=17,
            schema=TRAINER_CONFIG_SCHEMA_V2,
            objective=OBJECTIVE_NAME_V2,
        )
    with pytest.raises(
        ResidentSFTBootstrapAuthorityError,
        match="rollin_not_supported",
    ):
        ResidentSFTBootstrapConfig(
            seed=17,
            generated_rollin=rollin,
        )


def test_v3_trainer_config_binds_role_specialization_and_warmup() -> None:
    specialization = BranchSpecializationConfig(
        weight=8.0,
        target_separation=0.3,
    )
    config = ResidentSFTBootstrapConfig(
        seed=17,
        max_steps=9,
        schema=TRAINER_CONFIG_SCHEMA_V3,
        objective=OBJECTIVE_NAME_V3,
        generated_rollin=GeneratedRollinSelectionConfig(),
        branch_specialization=specialization,
        structural_warmup_steps=4,
        structural_warmup_learning_rate=1e-4,
        role_conditioned_branches=2,
        branch_indices=(0, 1),
    )

    assert ResidentSFTBootstrapConfig.from_dict(config.to_dict()) == config
    assert config.to_dict()["branch_specialization"] == specialization.to_dict()
    with pytest.raises(
        ResidentSFTBootstrapAuthorityError,
        match="specialization_required",
    ):
        ResidentSFTBootstrapConfig(
            seed=17,
            max_steps=9,
            schema=TRAINER_CONFIG_SCHEMA_V3,
            objective=OBJECTIVE_NAME_V3,
            generated_rollin=GeneratedRollinSelectionConfig(),
            structural_warmup_steps=4,
            structural_warmup_learning_rate=1e-4,
            role_conditioned_branches=2,
            branch_indices=(0, 1),
        )
    with pytest.raises(
        ResidentSFTBootstrapAuthorityError,
        match="branch_count_mismatch",
    ):
        ResidentSFTBootstrapConfig(
            seed=17,
            max_steps=9,
            schema=TRAINER_CONFIG_SCHEMA_V3,
            objective=OBJECTIVE_NAME_V3,
            generated_rollin=GeneratedRollinSelectionConfig(),
            branch_specialization=specialization,
            structural_warmup_steps=4,
            structural_warmup_learning_rate=1e-4,
            role_conditioned_branches=3,
            branch_indices=(0, 1),
        )


def test_historical_v3_authority_retains_generated_rollin_source_closure() -> None:
    authority, train, validation, sources = _authority()
    authority["schema"] = GENERATED_ROLLIN_AUTHORITY_SCHEMA
    authority["sources"] = {
        role: authority["sources"][role]
        for role in sorted(GENERATED_ROLLIN_REQUIRED_SOURCE_ROLES)
    }
    historical_sources = {
        role: sources[role] for role in GENERATED_ROLLIN_REQUIRED_SOURCE_ROLES
    }
    _recompute_authority_sha256(authority)

    validated = validate_authority(authority)
    assert set(validated["sources"]) == GENERATED_ROLLIN_REQUIRED_SOURCE_ROLES
    authorize_bound_artifacts(
        validated,
        train_payload=train,
        validation_payload=validation,
        source_payloads=historical_sources,
        expected_authority_sha256=validated["authority_sha256"],
    )


def test_historical_v2_authority_retains_pre_policy_source_closure() -> None:
    authority, train, validation, sources = _authority()
    authority["schema"] = PREVIOUS_AUTHORITY_SCHEMA
    authority["sources"] = {
        role: authority["sources"][role]
        for role in sorted(PREVIOUS_REQUIRED_SOURCE_ROLES)
    }
    previous_sources = {
        role: sources[role] for role in PREVIOUS_REQUIRED_SOURCE_ROLES
    }
    _recompute_authority_sha256(authority)

    validated = validate_authority(authority)
    assert set(validated["sources"]) == PREVIOUS_REQUIRED_SOURCE_ROLES
    authorize_bound_artifacts(
        validated,
        train_payload=train,
        validation_payload=validation,
        source_payloads=previous_sources,
        expected_authority_sha256=validated["authority_sha256"],
    )


def test_historical_v1_authority_retains_its_original_source_closure() -> None:
    authority, train, validation, sources = _authority()
    authority["schema"] = LEGACY_AUTHORITY_SCHEMA
    authority["sources"] = {
        role: authority["sources"][role] for role in sorted(LEGACY_REQUIRED_SOURCE_ROLES)
    }
    legacy_sources = {role: sources[role] for role in LEGACY_REQUIRED_SOURCE_ROLES}
    _recompute_authority_sha256(authority)

    validated = authorize_bound_artifacts(
        authority,
        train_payload=train,
        validation_payload=validation,
        source_payloads=legacy_sources,
        expected_authority_sha256=authority["authority_sha256"],
    )

    assert validated["schema"] == LEGACY_AUTHORITY_SCHEMA
    assert set(validated["sources"]) == LEGACY_REQUIRED_SOURCE_ROLES


def test_resident_authority_derives_complete_checkpoint_bindings() -> None:
    authority, _train, _validation, _sources = _authority()

    bindings = authority_state_bindings(authority)

    assert bindings["authority_sha256"] == authority["authority_sha256"]
    assert bindings["dataset_sha256"] == authority["dataset"]["dataset_sha256"]
    assert (
        bindings["behavior_identity_sha256"]
        == authority["model"]["behavior_bundle"]["bundle_sha256"]
    )
    assert bindings["source_closure_sha256"] == sha256_json(authority["sources"])
    assert bindings["trainer_config_sha256"] == sha256_json(authority["trainer"])
    assert bindings["campaign_scope_sha256"] == sha256_json("full_bootstrap")
    assert bindings["artifact_root_identity_sha256"] == sha256_json(
        authority["artifact_root_identity"]
    )


def test_resident_authority_rejects_malformed_artifact_root_identity() -> None:
    authority, _train, _validation, _sources = _authority()
    authority["artifact_root_identity"] = {"st_dev": 1, "st_ino": -1}
    _recompute_authority_sha256(authority)

    with pytest.raises(
        ResidentSFTBootstrapAuthorityError,
        match="artifact_root_identity_invalid",
    ):
        validate_authority(authority)


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


def test_resident_config_requires_deterministic_adapter_initialization() -> None:
    with pytest.raises(ResidentSFTBootstrapAuthorityError, match="dropout_must_be_zero"):
        ResidentSFTBootstrapConfig(seed=1, lora_dropout=0.1)
    with pytest.raises(
        ResidentSFTBootstrapAuthorityError,
        match="lora_initialization_seed_invalid",
    ):
        ResidentSFTBootstrapConfig(seed=1, lora_initialization_seed=2**32)


def test_authority_time_window_is_enforced_for_fresh_run_and_resume() -> None:
    authority, _train, _validation, _sources = _authority()
    during = datetime.fromisoformat("2026-08-02T01:00:00-07:00")
    after = datetime.fromisoformat("2026-08-09T01:00:00-07:00")
    before = datetime.fromisoformat("2026-07-31T01:00:00-07:00")

    validate_authority(authority, now=during)
    with pytest.raises(ResidentSFTBootstrapAuthorityError, match="expired"):
        validate_authority(authority, now=after)
    validate_authority(authority, now=after, allow_expired_resume=True)
    with pytest.raises(ResidentSFTBootstrapAuthorityError, match="not_yet_valid"):
        validate_authority(authority, now=before)


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
