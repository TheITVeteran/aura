from __future__ import annotations

import base64
import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from core.brain.llm.latent_cortex.campaign_journal import canonical_json_bytes
from core.brain.llm.latent_cortex.campaign_trust import (
    CAMPAIGN_TRUST_POLICY_SCHEMA,
    CAMPAIGN_TRUST_ROLES,
    CONTAMINATION_AUDITOR,
    EVIDENCE_VERIFIER,
    build_role_attestation,
    policy_signed_payload,
    validate_campaign_trust_policy,
)
from core.learning.structured_sft import (
    StructuredSFTCurriculumSpec,
    build_structured_sft_custody_bundles,
    validate_candidate_dataset_artifacts,
    validate_structured_sft_custody_pair,
)
from core.learning.structured_sft_admission import (
    STRUCTURED_SFT_CONTAMINATION_AUDIT_SCHEMA,
    STRUCTURED_SFT_EVIDENCE_AUDIT_SCHEMA,
    STRUCTURED_SFT_PRIVACY_AUDIT_SCHEMA,
    STRUCTURED_SFT_TRAINER_BINDING_SCHEMA,
    ZERO_SHA256,
    StructuredSFTAdmissionError,
    build_structured_sft_admission_bundle,
    structured_sft_admission_payloads,
    structured_sft_admission_protocol,
    validate_structured_sft_admission_bundle,
)
from tools.build_structured_sft_dataset import build_custodied_dataset_directories
from tools.manage_structured_sft_admission import (
    StructuredSFTAdmissionToolError,
    _read_json,
)

NOW = 1_800_000_300
SIGNED_AT = 1_800_000_200
SHA = "a" * 64


def _raw_public(key: Ed25519PrivateKey) -> bytes:
    return key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )


def _public_pem(key: Ed25519PrivateKey) -> bytes:
    return key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )


def _sha(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _commit(body: dict[str, Any], field: str) -> dict[str, Any]:
    return {**body, field: _sha(body)}


def _tokenizer_validation(candidate: dict[str, Any], custody: dict[str, Any]) -> dict[str, Any]:
    tokenizer_sha = _sha({"tokenizer": "fixture"})
    runtime_sha = _sha({"runtime": "fixture"})
    snapshot_sha = _sha({"snapshot": "fixture"})
    body = {
        "schema": "aura.rlc.structured_sft_tokenizer_validation_bundle.v3",
        "projection_schema": "aura.rlc.structured_sft_tokenization.v1",
        "projection_curriculum_sha256": _sha({"projection": "curriculum"}),
        "projection_report_sha256": _sha({"projection": "report"}),
        "candidate_curriculum_commitment_sha256": candidate["curriculum_manifest"][
            "curriculum_sha256"
        ],
        "tokenization_scope": "candidate_train_validation_only",
        "candidate_package_sha256": candidate["package_sha256"],
        "candidate_validation_scope": candidate["validation_scope"],
        "candidate_custody_attestation": {
            "schema": "aura.rlc.structured_sft_candidate_custody.v1",
            "generation_id": "fixture-generation",
            "candidate_package_sha256": candidate["package_sha256"],
            "evaluator_package_sha256": custody["evaluator_package_sha256"],
            "custody_root_sha256": custody["custody_root_sha256"],
            "custody_report_sha256": custody["custody_report_sha256"],
            "commit_sha256": _sha({"commit": "fixture"}),
            "evaluator_filesystem_accessed": False,
        },
        "tokenizer": {
            "files": [],
            "sha256": tokenizer_sha,
            "loaded_from_persistent_content_addressed_snapshot": True,
            "snapshot_path": "/content-addressed/test-fixture",
            "snapshot_manifest": {
                "snapshot_manifest_sha256": snapshot_sha,
            },
            "runtime": {"sha256": runtime_sha},
        },
        "trainer_binding_contract": {
            "tokenizer_path": "/content-addressed/test-fixture",
            "tokenizer_identity_sha256": tokenizer_sha,
            "runtime_identity_sha256": runtime_sha,
            "snapshot_manifest_sha256": snapshot_sha,
            "revalidate_in_trainer_process": True,
            "candidate_only_revalidation": True,
            "evaluator_filesystem_access_required": False,
            "path_substitution_allowed": False,
        },
        "trainer": candidate["trainer_contract"]["trainer"],
        "mask_prompt": True,
        "max_seq_length": candidate["trainer_contract"]["max_seq_length"],
        "truncation_allowed": False,
        "rows_with_truncation": 0,
        "holdout_tokenized": False,
        "rows_checked": custody["visible_example_count"],
        "groups": {},
        "projection_receipts_sha256": _sha({"receipts": "fixture"}),
        "status": "passed_exact_masked_prefix",
    }
    return _commit(body, "validation_bundle_sha256")


def _role_pin(role: str, key: Ed25519PrivateKey) -> dict[str, str]:
    raw = _raw_public(key)
    return {
        "signer_id": f"{role}-fixture-signer",
        "organization_id": f"{role}-fixture-organization",
        "public_key_b64": base64.b64encode(raw).decode("ascii"),
        "key_id": hashlib.sha256(raw).hexdigest(),
        "implementation_sha256": _sha({"implementation": role}),
        "release_sha256": _sha({"release": role}),
        "custody_class": "external_service",
        "custody_evidence_sha256": _sha({"custody": role}),
    }


def _signed_policy(
    root: Ed25519PrivateKey,
    role_keys: dict[str, Ed25519PrivateKey],
    *,
    campaign_name: str,
    protocol_sha256: str,
) -> dict[str, Any]:
    body = {
        "schema": CAMPAIGN_TRUST_POLICY_SCHEMA,
        "policy_id": "structured-sft-admission-test-fixture",
        "policy_revision": 7,
        "campaign_name": campaign_name,
        "protocol_sha256": protocol_sha256,
        "previous_policy_sha256": _sha({"previous": "policy"}),
        "revoked_key_ids": [],
        "issued_at_unix": 1_800_000_000,
        "not_before_unix": 1_800_000_100,
        "expires_at_unix": 1_800_086_400,
        "roles": {role: _role_pin(role, role_keys[role]) for role in CAMPAIGN_TRUST_ROLES},
    }
    signed = canonical_json_bytes(body)
    root_raw = _raw_public(root)
    return {
        **body,
        "root_signature": {
            "algorithm": "Ed25519",
            "key_id": hashlib.sha256(root_raw).hexdigest(),
            "signature_b64": base64.b64encode(root.sign(signed)).decode("ascii"),
            "signed_payload_sha256": hashlib.sha256(signed).hexdigest(),
        },
    }


def _resign_policy(policy: dict[str, Any], root: Ed25519PrivateKey) -> None:
    signed = canonical_json_bytes(policy_signed_payload(policy))
    policy["root_signature"]["signature_b64"] = base64.b64encode(root.sign(signed)).decode("ascii")
    policy["root_signature"]["signed_payload_sha256"] = hashlib.sha256(signed).hexdigest()


@pytest.fixture(scope="module")
def admission_inputs() -> dict[str, Any]:
    bundles = build_structured_sft_custody_bundles(
        StructuredSFTCurriculumSpec(
            seed=396,
            train_cases_per_family=1,
            validation_cases_per_family=1,
            holdout_cases_per_family=1,
        ),
        holdout_seed=bytes(range(32)),
    )
    candidate = validate_candidate_dataset_artifacts(bundles.candidate_artifacts)
    custody = validate_structured_sft_custody_pair(
        bundles.candidate_artifacts,
        bundles.evaluator_artifacts,
    )
    tokenizer = _tokenizer_validation(candidate, custody)
    root = Ed25519PrivateKey.generate()
    role_keys = {role: Ed25519PrivateKey.generate() for role in CAMPAIGN_TRUST_ROLES}
    policy_document = _signed_policy(
        root,
        role_keys,
        campaign_name=f"structured-sft:{candidate['package_sha256']}",
        protocol_sha256=structured_sft_admission_protocol()["protocol_sha256"],
    )
    policy = validate_campaign_trust_policy(
        policy_document,
        trusted_root_public_key_pem=_public_pem(root),
        now_unix=NOW,
    )
    contamination_pin = policy.role_pin(CONTAMINATION_AUDITOR)
    evidence_pin = policy.role_pin(EVIDENCE_VERIFIER)
    privacy = _commit(
        {
            "schema": STRUCTURED_SFT_PRIVACY_AUDIT_SCHEMA,
            "candidate_package_sha256": candidate["package_sha256"],
            "evaluator_package_sha256": custody["evaluator_package_sha256"],
            "custody_root_sha256": custody["custody_root_sha256"],
            "implementation_sha256": evidence_pin["implementation_sha256"],
            "release_sha256": evidence_pin["release_sha256"],
            "controls": {
                name: "passed"
                for name in (
                    "origin_classification",
                    "pii_scan",
                    "secret_scan",
                    "consent_basis",
                    "license_basis",
                    "tenant_boundary",
                    "retention_policy",
                    "revocation_lineage",
                    "deletion_lineage",
                    "remote_sync_policy",
                )
            },
            "contains_user_content": False,
            "pii_findings": 0,
            "secret_findings": 0,
            "tenant_violations": 0,
            "license_violations": 0,
            "consent_violations": 0,
            "retention_violations": 0,
            "revocation_violations": 0,
            "deletion_violations": 0,
            "remote_sync_violations": 0,
            "status": "passed_synthetic_non_user_data",
        },
        "report_sha256",
    )
    contamination = _commit(
        {
            "schema": STRUCTURED_SFT_CONTAMINATION_AUDIT_SCHEMA,
            "candidate_package_sha256": candidate["package_sha256"],
            "evaluator_package_sha256": custody["evaluator_package_sha256"],
            "custody_root_sha256": custody["custody_root_sha256"],
            "implementation_sha256": contamination_pin["implementation_sha256"],
            "release_sha256": contamination_pin["release_sha256"],
            "surfaces": [
                "prompt",
                "target",
                "public_derivation",
                "tool_input",
                "tool_output",
                "normalized_code",
                "normalized_json",
                "adapter_corpus",
                "training_corpus",
                "evaluation_corpus",
            ],
            "methods": [
                "exact_sha256",
                "normalized_sha256",
                "token_shingle_jaccard",
                "character_shingle_jaccard",
                "canonical_ast_sha256",
                "canonical_json_sha256",
            ],
            "reference_corpora": [
                {
                    "name": kind,
                    "kind": kind,
                    "manifest_sha256": _sha({"corpus": kind}),
                    "entry_count": 1,
                }
                for kind in ("adapter", "training", "evaluation")
            ],
            "pre_augmentation_partition_sha256": _sha({"partition": "before-augmentation"}),
            "semantic_dedup_manifest_sha256": _sha({"dedup": "multisurface"}),
            "exact_overlap_count": 0,
            "near_duplicate_overlap_count": 0,
            "cross_split_lineage_overlap_count": 0,
            "uncovered_surface_count": 0,
            "status": "passed_zero_multisurface_overlap",
        },
        "report_sha256",
    )
    evidence = _commit(
        {
            "schema": STRUCTURED_SFT_EVIDENCE_AUDIT_SCHEMA,
            "candidate_package_sha256": candidate["package_sha256"],
            "evaluator_package_sha256": custody["evaluator_package_sha256"],
            "custody_root_sha256": custody["custody_root_sha256"],
            "source_closure_sha256": candidate["curriculum_manifest"]["source_binding"]["sha256"],
            "tokenizer_validation_bundle_sha256": tokenizer["validation_bundle_sha256"],
            "tokenizer_identity_sha256": tokenizer["tokenizer"]["sha256"],
            "tokenizer_runtime_identity_sha256": tokenizer["tokenizer"]["runtime"]["sha256"],
            "implementation_sha256": evidence_pin["implementation_sha256"],
            "release_sha256": evidence_pin["release_sha256"],
            "checks": {
                name: "passed"
                for name in (
                    "independent_semantic_replay",
                    "proof_kernel_replay",
                    "program_execution_replay",
                    "repair_ast_single_substitution",
                    "tool_result_schema_v3",
                    "tool_execution_receipt_binding",
                    "prompt_injection_screen",
                    "data_poisoning_screen",
                    "verifier_gaming_screen",
                    "split_lineage_disjointness",
                    "candidate_holdout_noncontainment",
                )
            },
            "visible_examples_replayed": custody["visible_example_count"],
            "holdout_examples_replayed": custody["holdout_example_count"],
            "tool_traces_reexecuted": 1,
            "failed_check_count": 0,
            "status": "passed_independent_reverification",
        },
        "report_sha256",
    )
    tokenizer_record = tokenizer["tokenizer"]
    trainer = _commit(
        {
            "schema": STRUCTURED_SFT_TRAINER_BINDING_SCHEMA,
            "candidate_package_sha256": candidate["package_sha256"],
            "custody_root_sha256": custody["custody_root_sha256"],
            "source_closure_sha256": candidate["curriculum_manifest"]["source_binding"]["sha256"],
            "tokenizer_validation_bundle_sha256": tokenizer["validation_bundle_sha256"],
            "tokenizer_identity_sha256": tokenizer_record["sha256"],
            "tokenizer_runtime_identity_sha256": tokenizer_record["runtime"]["sha256"],
            "tokenizer_snapshot_manifest_sha256": tokenizer_record["snapshot_manifest"][
                "snapshot_manifest_sha256"
            ],
            "trainer": candidate["trainer_contract"]["trainer"],
            "mask_prompt": True,
            "supervised_region": candidate["trainer_contract"]["supervised_region"],
            "max_seq_length": candidate["curriculum_manifest"]["spec"]["max_seq_length"],
            "truncation_allowed": False,
            "revalidate_in_trainer_process": True,
            "candidate_only_revalidation": True,
            "path_substitution_allowed": False,
            "model_sha256": _sha({"model": "fixture"}),
            "adapter_base_sha256": _sha({"adapter": "fixture"}),
            "recurrence_program_sha256": _sha({"recurrence": "fixture"}),
            "optimizer_config_sha256": _sha({"optimizer": "fixture"}),
            "scheduler_config_sha256": _sha({"scheduler": "fixture"}),
            "rng_manifest_sha256": _sha({"rng": "fixture"}),
            "compute_budget_sha256": _sha({"compute": "fixture"}),
            "execution_authorized": False,
            "status": "bound_pending_transfer_authority",
        },
        "binding_sha256",
    )
    return {
        "candidate_artifacts": bundles.candidate_artifacts,
        "evaluator_artifacts": bundles.evaluator_artifacts,
        "tokenizer_validation": tokenizer,
        "privacy_report": privacy,
        "contamination_report": contamination,
        "evidence_report": evidence,
        "trainer_binding": trainer,
        "root": root,
        "root_pem": _public_pem(root),
        "role_keys": role_keys,
        "policy": policy,
    }


def _attestations(
    inputs: dict[str, Any], *, signed_at: int = SIGNED_AT
) -> dict[str, dict[str, Any]]:
    payloads = structured_sft_admission_payloads(
        candidate_artifacts=inputs["candidate_artifacts"],
        evaluator_artifacts=inputs["evaluator_artifacts"],
        tokenizer_validation=inputs["tokenizer_validation"],
        privacy_report=inputs["privacy_report"],
        contamination_report=inputs["contamination_report"],
        evidence_report=inputs["evidence_report"],
        trainer_binding=inputs["trainer_binding"],
        sequence=1,
        previous_admission_sha256=ZERO_SHA256,
    )
    return {
        name: build_role_attestation(
            inputs["policy"],
            role=role,
            payload=payloads[name],
            signed_at_unix=signed_at,
            private_key=inputs["role_keys"][role],
        )
        for name, role in {
            "package_declaration": "task_issuer",
            "contamination_audit": "contamination_auditor",
            "evidence_audit": "evidence_verifier",
            "trainer_binding": "campaign_runner",
        }.items()
    }


def _build(inputs: dict[str, Any], *, signed_at: int = SIGNED_AT) -> dict[str, Any]:
    return build_structured_sft_admission_bundle(
        **{
            key: inputs[key]
            for key in (
                "candidate_artifacts",
                "evaluator_artifacts",
                "tokenizer_validation",
                "privacy_report",
                "contamination_report",
                "evidence_report",
                "trainer_binding",
            )
        },
        policy=inputs["policy"],
        attestations=_attestations(inputs, signed_at=signed_at),
        sequence=1,
        previous_admission_sha256=ZERO_SHA256,
        observed_at_unix=NOW,
    )


def _validate(bundle: dict[str, Any], inputs: dict[str, Any], **overrides):
    arguments = {
        key: inputs[key]
        for key in (
            "candidate_artifacts",
            "evaluator_artifacts",
            "tokenizer_validation",
            "privacy_report",
            "contamination_report",
            "evidence_report",
            "trainer_binding",
        )
    }
    arguments.update(
        {
            "trusted_root_public_key_pem": inputs["root_pem"],
            "expected_sequence": 1,
            "expected_previous_admission_sha256": ZERO_SHA256,
            "expected_policy_sha256": inputs["policy"].policy_sha256,
            "minimum_policy_revision": 7,
            "now_unix": NOW,
        }
    )
    arguments.update(overrides)
    return validate_structured_sft_admission_bundle(bundle, **arguments)


def test_external_admission_round_trip_never_authorizes_training(
    admission_inputs,
) -> None:
    bundle = _build(admission_inputs)

    assert _validate(bundle, admission_inputs) == bundle
    assert bundle["admission"]["trainer_ready"] is False
    assert bundle["admission"]["training_authority"].startswith("none_")
    assert bundle["admission"]["status"].endswith("no_training_authority")
    assert "resident_training_execution_receipt" in bundle["admission"]["remaining_gates"]


@pytest.mark.parametrize(
    ("report_name", "field", "value", "error"),
    [
        ("privacy_report", "pii_findings", 1, "privacy_report_failed"),
        (
            "privacy_report",
            "contains_user_content",
            True,
            "privacy_report_failed",
        ),
        (
            "contamination_report",
            "near_duplicate_overlap_count",
            1,
            "contamination_report_failed",
        ),
        (
            "contamination_report",
            "uncovered_surface_count",
            1,
            "contamination_report_failed",
        ),
        (
            "evidence_report",
            "failed_check_count",
            1,
            "evidence_report_failed",
        ),
        (
            "trainer_binding",
            "execution_authorized",
            True,
            "trainer_binding_failed",
        ),
    ],
)
def test_reports_fail_closed_even_when_recommitted(
    admission_inputs, report_name, field, value, error
) -> None:
    attacked = copy.deepcopy(admission_inputs)
    report = attacked[report_name]
    digest_field = "binding_sha256" if report_name == "trainer_binding" else "report_sha256"
    report[field] = value
    body = dict(report)
    body.pop(digest_field)
    report[digest_field] = _sha(body)

    with pytest.raises(StructuredSFTAdmissionError, match=error):
        _build(attacked)


def test_tokenizer_bundle_requires_exact_reconstructed_commitment(
    admission_inputs,
) -> None:
    attacked = copy.deepcopy(admission_inputs)
    attacked["tokenizer_validation"]["rows_checked"] += 1

    with pytest.raises(StructuredSFTAdmissionError, match="tokenizer_commitment_invalid"):
        _build(attacked)


def test_auditor_report_provenance_must_match_policy_pin(
    admission_inputs,
) -> None:
    attacked = copy.deepcopy(admission_inputs)
    report = attacked["contamination_report"]
    report["implementation_sha256"] = SHA
    body = dict(report)
    body.pop("report_sha256")
    report["report_sha256"] = _sha(body)

    with pytest.raises(StructuredSFTAdmissionError, match="auditor_provenance_mismatch"):
        _build(attacked)


def test_admission_rejects_nonexternal_role_custody(admission_inputs) -> None:
    attacked = copy.deepcopy(admission_inputs)
    policy_document = copy.deepcopy(attacked["policy"].document)
    policy_document["roles"]["task_issuer"]["custody_class"] = "local_software"
    _resign_policy(policy_document, attacked["root"])
    attacked["policy"] = validate_campaign_trust_policy(
        policy_document,
        trusted_root_public_key_pem=attacked["root_pem"],
        now_unix=NOW,
    )

    with pytest.raises(StructuredSFTAdmissionError, match="policy_invalid"):
        _build(attacked)


def test_admission_rejects_future_or_stale_role_attestations(
    admission_inputs,
) -> None:
    with pytest.raises(StructuredSFTAdmissionError, match="too_late"):
        _build(admission_inputs, signed_at=NOW + 1)

    stale_inputs = copy.deepcopy(admission_inputs)
    policy_document = copy.deepcopy(stale_inputs["policy"].document)
    policy_document["issued_at_unix"] = NOW - 900_000
    policy_document["not_before_unix"] = NOW - 800_000
    _resign_policy(policy_document, stale_inputs["root"])
    stale_inputs["policy"] = validate_campaign_trust_policy(
        policy_document,
        trusted_root_public_key_pem=stale_inputs["root_pem"],
        now_unix=NOW,
    )
    with pytest.raises(StructuredSFTAdmissionError, match="too_early"):
        _build(stale_inputs, signed_at=NOW - 700_000)


def test_verifier_requires_exact_external_sequence_chain_and_policy_pins(
    admission_inputs,
) -> None:
    bundle = _build(admission_inputs)

    with pytest.raises(StructuredSFTAdmissionError, match="rollback"):
        _validate(bundle, admission_inputs, expected_sequence=2)
    with pytest.raises(StructuredSFTAdmissionError, match="chain_mismatch"):
        _validate(
            bundle,
            admission_inputs,
            expected_previous_admission_sha256="b" * 64,
        )
    with pytest.raises(StructuredSFTAdmissionError, match="policy_pin_mismatch"):
        _validate(
            bundle,
            admission_inputs,
            expected_policy_sha256="b" * 64,
        )
    with pytest.raises(StructuredSFTAdmissionError, match="observation_time_mismatch"):
        _validate(bundle, admission_inputs, now_unix=NOW + 1)


def test_bundle_rejects_substitution_unknown_fields_and_authority_escalation(
    admission_inputs,
) -> None:
    bundle = _build(admission_inputs)
    attacked = copy.deepcopy(bundle)
    attacked["unknown"] = True
    with pytest.raises(StructuredSFTAdmissionError, match="schema_invalid"):
        _validate(attacked, admission_inputs)

    attacked = copy.deepcopy(bundle)
    attacked["admission"]["trainer_ready"] = True
    body = dict(attacked)
    body.pop("bundle_sha256")
    attacked["bundle_sha256"] = _sha(body)
    with pytest.raises(StructuredSFTAdmissionError, match="reconstruction_mismatch"):
        _validate(attacked, admission_inputs)

    with pytest.raises(StructuredSFTAdmissionError):
        _validate(
            bundle,
            admission_inputs,
            trusted_root_public_key_pem=_public_pem(Ed25519PrivateKey.generate()),
        )


def test_protocol_rejects_nonfinite_json_before_signing(admission_inputs) -> None:
    attacked = copy.deepcopy(admission_inputs)
    attacked["privacy_report"]["pii_findings"] = float("nan")

    with pytest.raises(StructuredSFTAdmissionError, match="report_invalid"):
        _build(attacked)


def test_admission_tool_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    document = tmp_path / "duplicate.json"
    document.write_text('{"schema":"first","schema":"second"}')

    with pytest.raises(StructuredSFTAdmissionToolError, match="json_invalid"):
        _read_json(document, role="bundle")


def test_private_key_free_cli_assembles_and_reverifies_exact_bundle(
    admission_inputs,
    tmp_path: Path,
) -> None:
    candidate_dir = tmp_path / "candidate"
    evaluator_dir = tmp_path / "evaluator"
    build_custodied_dataset_directories(
        candidate_directory=candidate_dir,
        evaluator_directory=evaluator_dir,
        spec=StructuredSFTCurriculumSpec(
            seed=396,
            train_cases_per_family=1,
            validation_cases_per_family=1,
            holdout_cases_per_family=1,
        ),
        holdout_seed=bytes(range(32)),
    )

    documents = {
        "tokenizer-validation": admission_inputs["tokenizer_validation"],
        "privacy-report": admission_inputs["privacy_report"],
        "contamination-report": admission_inputs["contamination_report"],
        "evidence-report": admission_inputs["evidence_report"],
        "trainer-binding": admission_inputs["trainer_binding"],
        "policy": admission_inputs["policy"].document,
    }
    paths: dict[str, Path] = {}
    for name, document in documents.items():
        path = tmp_path / f"{name}.json"
        path.write_bytes(canonical_json_bytes(document) + b"\n")
        paths[name] = path
    root_path = tmp_path / "root.pem"
    root_path.write_bytes(admission_inputs["root_pem"])
    attestation_paths: dict[str, Path] = {}
    for name, attestation in _attestations(admission_inputs).items():
        path = tmp_path / f"{name}-attestation.json"
        path.write_bytes(canonical_json_bytes(attestation) + b"\n")
        attestation_paths[name] = path

    tool = Path(__file__).resolve().parents[1] / "tools/manage_structured_sft_admission.py"
    common = [
        "--candidate-dir",
        str(candidate_dir),
        "--evaluator-dir",
        str(evaluator_dir),
        "--tokenizer-validation",
        str(paths["tokenizer-validation"]),
        "--privacy-report",
        str(paths["privacy-report"]),
        "--contamination-report",
        str(paths["contamination-report"]),
        "--evidence-report",
        str(paths["evidence-report"]),
        "--trainer-binding",
        str(paths["trainer-binding"]),
        "--policy",
        str(paths["policy"]),
        "--root",
        str(root_path),
        "--minimum-policy-revision",
        "7",
        "--observed-at",
        str(NOW),
    ]
    attestation_arguments = [
        value
        for name in (
            "package_declaration",
            "contamination_audit",
            "evidence_audit",
            "trainer_binding",
        )
        for value in (
            f"--{name.replace('_', '-')}-attestation",
            str(attestation_paths[name]),
        )
    ]
    bundle_path = tmp_path / "admission-bundle.json"
    assembled = subprocess.run(
        [
            sys.executable,
            str(tool),
            "assemble",
            *common,
            "--sequence",
            "1",
            "--previous-admission-sha256",
            ZERO_SHA256,
            *attestation_arguments,
            "--out",
            str(bundle_path),
        ],
        cwd=tool.parents[1],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert assembled.returncode == 0, assembled.stdout + assembled.stderr
    assembled_bundle = json.loads(bundle_path.read_bytes())

    verified_path = tmp_path / "verified-bundle.json"
    verified = subprocess.run(
        [
            sys.executable,
            str(tool),
            "verify",
            *common,
            "--bundle",
            str(bundle_path),
            "--expected-sequence",
            "1",
            "--expected-previous-admission-sha256",
            ZERO_SHA256,
            "--expected-policy-sha256",
            admission_inputs["policy"].policy_sha256,
            "--out",
            str(verified_path),
        ],
        cwd=tool.parents[1],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert verified.returncode == 0, verified.stdout + verified.stderr
    assert json.loads(verified_path.read_bytes()) == assembled_bundle
