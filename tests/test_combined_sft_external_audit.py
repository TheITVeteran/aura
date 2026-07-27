from __future__ import annotations

import base64
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from core.brain.llm.latent_cortex.campaign_journal import canonical_json_bytes
from core.brain.llm.latent_cortex.campaign_trust import (
    CAMPAIGN_RUNNER,
    CAMPAIGN_TRUST_POLICY_SCHEMA,
    CAMPAIGN_TRUST_ROLES,
    CONTAMINATION_AUDITOR,
    EVIDENCE_VERIFIER,
    TASK_ISSUER,
    build_role_attestation,
    validate_campaign_trust_policy,
)
from core.learning.combined_sft_external_audit import (
    COMBINED_SFT_CONTAMINATION_REPORT_SCHEMA,
    COMBINED_SFT_EXECUTION_REPORT_SCHEMA,
    COMBINED_SFT_PRIVACY_REPORT_SCHEMA,
    COMBINED_SFT_RUNNER_BINDING_SCHEMA,
    ZERO_SHA256,
    CombinedSFTExternalAuditError,
    build_combined_sft_external_audit_bundle,
    combined_sft_external_audit_payloads,
    combined_sft_external_audit_protocol,
    validate_combined_sft_external_audit_bundle,
)
from core.learning.combined_sft_lineage import (
    COMBINED_SFT_LINEAGE_EVALUATOR_FILES,
    build_combined_sft_lineage_bundle,
)
from core.learning.combined_sft_lineage_publication import (
    publish_combined_sft_lineage_custody,
)
from tools.manage_combined_sft_external_audit import main as external_audit_cli_main

pytest_plugins = ("test_combined_sft_lineage",)

NOW = 1_800_000_300
SIGNED_AT = 1_800_000_200


def _sha(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _commit(body: dict[str, Any], field: str) -> dict[str, Any]:
    return {**body, field: _sha(body)}


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


def _role_pin(role: str, key: Ed25519PrivateKey) -> dict[str, str]:
    raw = _raw_public(key)
    return {
        "signer_id": f"{role}-external-signer",
        "organization_id": f"{role}-external-organization",
        "public_key_b64": base64.b64encode(raw).decode("ascii"),
        "key_id": hashlib.sha256(raw).hexdigest(),
        "implementation_sha256": _sha({"implementation": role}),
        "release_sha256": _sha({"release": role}),
        "custody_class": "external_service",
        "custody_evidence_sha256": _sha({"custody": role}),
    }


def _policy_document(
    root: Ed25519PrivateKey,
    role_keys: dict[str, Ed25519PrivateKey],
    *,
    commitment_sha256: str,
) -> dict[str, Any]:
    body = {
        "schema": CAMPAIGN_TRUST_POLICY_SCHEMA,
        "policy_id": "combined-sft-external-audit-fixture",
        "policy_revision": 1,
        "campaign_name": f"combined-sft:{commitment_sha256}",
        "protocol_sha256": combined_sft_external_audit_protocol()["protocol_sha256"],
        "previous_policy_sha256": _sha({"previous": "policy"}),
        "revoked_key_ids": [],
        "issued_at_unix": 1_800_000_000,
        "not_before_unix": 1_800_000_100,
        "expires_at_unix": 1_800_086_400,
        "roles": {role: _role_pin(role, role_keys[role]) for role in CAMPAIGN_TRUST_ROLES},
    }
    signed = canonical_json_bytes(body)
    raw = _raw_public(root)
    return {
        **body,
        "root_signature": {
            "algorithm": "Ed25519",
            "key_id": hashlib.sha256(raw).hexdigest(),
            "signature_b64": base64.b64encode(root.sign(signed)).decode("ascii"),
            "signed_payload_sha256": hashlib.sha256(signed).hexdigest(),
        },
    }


@pytest.fixture(scope="module")
def external_audit_inputs(lineage_inputs):
    structured, replay, evaluations = lineage_inputs
    bundle = build_combined_sft_lineage_bundle(
        structured_candidate_artifacts=structured.candidate_artifacts,
        structured_evaluator_artifacts=structured.evaluator_artifacts,
        replay_candidate_artifacts=replay.candidate_artifacts,
        replay_evaluator_artifacts=replay.evaluator_artifacts,
        external_evaluation_records=evaluations,
        required_evaluation_corpora=sorted({row["corpus"] for row in evaluations}),
        dedup_key=b"combined-external-audit-dedup-key" * 2,
    )
    custody = bundle.custody_report
    manifest = __import__("json").loads(
        bundle.evaluator_artifacts[COMBINED_SFT_LINEAGE_EVALUATOR_FILES[0]]
    )
    root = Ed25519PrivateKey.generate()
    role_keys = {role: Ed25519PrivateKey.generate() for role in CAMPAIGN_TRUST_ROLES}
    policy_document = _policy_document(
        root,
        role_keys,
        commitment_sha256=custody["commitment_sha256"],
    )
    policy = validate_campaign_trust_policy(
        policy_document,
        trusted_root_public_key_pem=_public_pem(root),
        now_unix=NOW,
    )
    evidence_pin = policy.role_pin(EVIDENCE_VERIFIER)
    contamination_pin = policy.role_pin(CONTAMINATION_AUDITOR)
    runner_pin = policy.role_pin(CAMPAIGN_RUNNER)
    privacy_controls = {
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
            "derived_artifact_lineage",
            "remote_sync_policy",
        )
    }
    zero_counts = {
        name: 0
        for name in (
            "unresolved_pii_findings",
            "unresolved_secret_findings",
            "tenant_violations",
            "license_violations",
            "consent_violations",
            "retention_violations",
            "revocation_violations",
            "deletion_violations",
            "lineage_violations",
            "remote_sync_violations",
        )
    }
    structured_binding = manifest["structured_binding"]
    replay_binding = manifest["verified_replay_binding"]
    common = {
        "commitment_sha256": custody["commitment_sha256"],
        "manifest_sha256": custody["manifest_sha256"],
        "combined_semantic_index_sha256": custody["combined_semantic_index_sha256"],
    }
    privacy = _commit(
        {
            "schema": COMBINED_SFT_PRIVACY_REPORT_SCHEMA,
            **common,
            "implementation_sha256": evidence_pin["implementation_sha256"],
            "release_sha256": evidence_pin["release_sha256"],
            "sources": [
                {
                    "source_class": "structured_synthetic",
                    "candidate_package_sha256": structured_binding["candidate_package_sha256"],
                    "evaluator_package_sha256": structured_binding["evaluator_package_sha256"],
                    "custody_root_sha256": structured_binding["custody_root_sha256"],
                    "privacy_manifest_sha256": ZERO_SHA256,
                    "contains_user_content": False,
                    "controls": privacy_controls,
                    **zero_counts,
                    "status": "passed_synthetic_non_user_data",
                },
                {
                    "source_class": "verified_replay_user_content",
                    "candidate_package_sha256": replay_binding["candidate_package_sha256"],
                    "evaluator_package_sha256": replay_binding["evaluator_package_sha256"],
                    "custody_root_sha256": replay_binding["custody_root_sha256"],
                    "privacy_manifest_sha256": replay_binding["privacy_manifest_sha256"],
                    "contains_user_content": True,
                    "controls": privacy_controls,
                    **zero_counts,
                    "status": "passed_user_content_governance",
                },
            ],
            "status": "passed_source_class_specific_privacy_audit",
        },
        "report_sha256",
    )
    inventory = {
        "coverage": manifest["coverage"],
        "required_evaluation_corpora": manifest["required_evaluation_corpora"],
        "record_counts": manifest["record_counts"],
    }
    references = [
        {
            "name": "resident-adapter-base",
            "kind": "adapter",
            "manifest_sha256": _sha({"adapter": 1}),
            "entry_count": 1,
        },
        {
            "name": "combined-training-corpus",
            "kind": "training",
            "manifest_sha256": _sha({"training": 1}),
            "entry_count": manifest["combined_semantic_index"]["record_count"],
        },
        *[
            {
                "name": name,
                "kind": "evaluation",
                "manifest_sha256": _sha({"evaluation": name}),
                "entry_count": 1,
            }
            for name in manifest["required_evaluation_corpora"]
        ],
    ]
    contamination = _commit(
        {
            "schema": COMBINED_SFT_CONTAMINATION_REPORT_SCHEMA,
            **common,
            "implementation_sha256": contamination_pin["implementation_sha256"],
            "release_sha256": contamination_pin["release_sha256"],
            "inventory_sha256": _sha(inventory),
            **inventory,
            "methods": [
                "exact_sha256",
                "normalized_sha256",
                "token_shingle_jaccard",
                "character_shingle_jaccard",
                "canonical_ast_sha256",
                "canonical_json_sha256",
                "causal_lineage_identity",
            ],
            "reference_corpora": references,
            "exact_overlap_count": 0,
            "near_duplicate_overlap_count": 0,
            "cross_split_lineage_overlap_count": 0,
            "uncovered_surface_count": 0,
            "status": "passed_complete_zero_multisurface_overlap",
        },
        "report_sha256",
    )
    execution = _commit(
        {
            "schema": COMBINED_SFT_EXECUTION_REPORT_SCHEMA,
            **common,
            "implementation_sha256": evidence_pin["implementation_sha256"],
            "release_sha256": evidence_pin["release_sha256"],
            "structured_custody_root_sha256": structured_binding["custody_root_sha256"],
            "verified_replay_custody_root_sha256": replay_binding["custody_root_sha256"],
            "checks": {
                name: "passed"
                for name in (
                    "structured_source_reconstruction",
                    "verified_replay_source_reconstruction",
                    "combined_lineage_reconstruction",
                    "proof_kernel_replay",
                    "sandbox_program_reexecution",
                    "tool_result_schema_v3",
                    "tool_receipt_result_binding",
                    "repair_ast_single_substitution",
                    "prompt_injection_screen",
                    "data_poisoning_screen",
                    "verifier_gaming_screen",
                    "candidate_holdout_noncontainment",
                )
            },
            "visible_examples_replayed": sum(
                item["count"] for item in manifest["record_counts"] if item["split"] != "holdout"
            ),
            "holdout_examples_replayed": sum(
                item["count"] for item in manifest["record_counts"] if item["split"] == "holdout"
            ),
            "tool_traces_reexecuted": sum(
                item["count"]
                for item in manifest["record_counts"]
                if item["corpus"]
                in {"structured_sft:code_tool", "structured_sft:code_tool_repair"}
            ),
            "failed_check_count": 0,
            "status": "passed_independent_source_and_execution_replay",
        },
        "report_sha256",
    )
    runner = _commit(
        {
            "schema": COMBINED_SFT_RUNNER_BINDING_SCHEMA,
            **common,
            "implementation_sha256": runner_pin["implementation_sha256"],
            "release_sha256": runner_pin["release_sha256"],
            "candidate_inputs_loadable": False,
            "execution_authorized": False,
            "trainer_ready": False,
            "training_authority": "none_pending_tokenizer_and_transfer_gates",
            "status": "bound_quarantine_no_training_authority",
        },
        "binding_sha256",
    )
    return {
        "bundle": bundle,
        "root": root,
        "root_pem": _public_pem(root),
        "role_keys": role_keys,
        "policy": policy,
        "privacy_report": privacy,
        "contamination_report": contamination,
        "execution_report": execution,
        "runner_binding": runner,
    }


def _documents(inputs: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_artifacts": inputs["bundle"].candidate_artifacts,
        "evaluator_artifacts": inputs["bundle"].evaluator_artifacts,
        "privacy_report": inputs["privacy_report"],
        "contamination_report": inputs["contamination_report"],
        "execution_report": inputs["execution_report"],
        "runner_binding": inputs["runner_binding"],
    }


def _attestations(inputs: dict[str, Any]) -> dict[str, dict[str, Any]]:
    payloads = combined_sft_external_audit_payloads(
        **_documents(inputs),
        sequence=1,
        previous_audit_sha256=ZERO_SHA256,
    )
    roles = {
        "package_declaration": TASK_ISSUER,
        "contamination_audit": CONTAMINATION_AUDITOR,
        "evidence_audit": EVIDENCE_VERIFIER,
        "runner_binding": CAMPAIGN_RUNNER,
    }
    return {
        name: build_role_attestation(
            inputs["policy"],
            role=role,
            payload=payloads[name],
            signed_at_unix=SIGNED_AT,
            private_key=inputs["role_keys"][role],
        )
        for name, role in roles.items()
    }


def _build(inputs: dict[str, Any]) -> dict[str, Any]:
    return build_combined_sft_external_audit_bundle(
        **_documents(inputs),
        policy=inputs["policy"],
        attestations=_attestations(inputs),
        sequence=1,
        previous_audit_sha256=ZERO_SHA256,
        observed_at_unix=NOW,
    )


def test_external_audit_round_trip_never_grants_training_authority(
    external_audit_inputs,
) -> None:
    bundle = _build(external_audit_inputs)
    audit = bundle["audit"]
    assert audit["status"] == "external_combined_evidence_verified_no_training_authority"
    assert audit["trainer_ready"] is False
    assert audit["training_authority"].startswith("none_pending_")
    assert len({pin["organization_id"] for pin in bundle["policy"]["roles"].values()}) == 4

    rebuilt = validate_combined_sft_external_audit_bundle(
        bundle,
        **_documents(external_audit_inputs),
        trusted_root_public_key_pem=external_audit_inputs["root_pem"],
        expected_sequence=1,
        expected_previous_audit_sha256=ZERO_SHA256,
        expected_policy_sha256=external_audit_inputs["policy"].policy_sha256,
        minimum_policy_revision=1,
        now_unix=NOW,
    )
    assert rebuilt == bundle


def test_privacy_source_classes_cannot_be_conflated(external_audit_inputs) -> None:
    attacked = dict(external_audit_inputs)
    attacked["privacy_report"] = copy.deepcopy(external_audit_inputs["privacy_report"])
    report = attacked["privacy_report"]
    replay = next(
        source for source in report["sources"] if source["source_class"].startswith("verified")
    )
    replay["contains_user_content"] = False
    replay["status"] = "passed_synthetic_non_user_data"
    body = dict(report)
    body.pop("report_sha256")
    report["report_sha256"] = _sha(body)
    with pytest.raises(CombinedSFTExternalAuditError, match="privacy_source_failed"):
        combined_sft_external_audit_payloads(
            **_documents(attacked),
            sequence=1,
            previous_audit_sha256=ZERO_SHA256,
        )


def test_every_declared_evaluation_corpus_requires_reference_evidence(
    external_audit_inputs,
) -> None:
    attacked = dict(external_audit_inputs)
    attacked["contamination_report"] = copy.deepcopy(
        external_audit_inputs["contamination_report"]
    )
    report = attacked["contamination_report"]
    report["reference_corpora"] = [
        item for item in report["reference_corpora"] if item["name"] != "eval:fresh-tools"
    ]
    body = dict(report)
    body.pop("report_sha256")
    report["report_sha256"] = _sha(body)
    with pytest.raises(CombinedSFTExternalAuditError, match="contamination_report_failed"):
        combined_sft_external_audit_payloads(
            **_documents(attacked),
            sequence=1,
            previous_audit_sha256=ZERO_SHA256,
        )


def test_role_attestations_are_not_interchangeable(external_audit_inputs) -> None:
    attestations = _attestations(external_audit_inputs)
    attestations["contamination_audit"] = attestations["package_declaration"]
    with pytest.raises(CombinedSFTExternalAuditError, match="contamination_audit"):
        build_combined_sft_external_audit_bundle(
            **_documents(external_audit_inputs),
            policy=external_audit_inputs["policy"],
            attestations=attestations,
            sequence=1,
            previous_audit_sha256=ZERO_SHA256,
            observed_at_unix=NOW,
        )


def test_partial_execution_replay_cannot_pass_as_complete(external_audit_inputs) -> None:
    attacked = dict(external_audit_inputs)
    attacked["execution_report"] = copy.deepcopy(external_audit_inputs["execution_report"])
    report = attacked["execution_report"]
    report["tool_traces_reexecuted"] -= 1
    body = dict(report)
    body.pop("report_sha256")
    report["report_sha256"] = _sha(body)
    with pytest.raises(CombinedSFTExternalAuditError, match="execution_report_failed"):
        combined_sft_external_audit_payloads(
            **_documents(attacked),
            sequence=1,
            previous_audit_sha256=ZERO_SHA256,
        )


def test_tampered_combined_custody_has_stable_audit_failure(external_audit_inputs) -> None:
    attacked = _documents(external_audit_inputs)
    attacked["candidate_artifacts"] = dict(attacked["candidate_artifacts"])
    name = next(iter(attacked["candidate_artifacts"]))
    attacked["candidate_artifacts"][name] += b" "
    with pytest.raises(CombinedSFTExternalAuditError, match="custody_invalid"):
        combined_sft_external_audit_payloads(
            **attacked,
            sequence=1,
            previous_audit_sha256=ZERO_SHA256,
        )


@pytest.mark.parametrize(
    ("sequence", "previous", "reason"),
    [(2, ZERO_SHA256, "chain_origin"), (1, "f" * 64, "chain_origin")],
)
def test_chain_origin_is_not_self_asserted(
    external_audit_inputs,
    sequence: int,
    previous: str,
    reason: str,
) -> None:
    with pytest.raises(CombinedSFTExternalAuditError, match=reason):
        combined_sft_external_audit_payloads(
            **_documents(external_audit_inputs),
            sequence=sequence,
            previous_audit_sha256=previous,
        )


def test_bundle_tampering_cannot_escalate_trainer_authority(external_audit_inputs) -> None:
    bundle = _build(external_audit_inputs)
    attacked = copy.deepcopy(bundle)
    attacked["audit"]["trainer_ready"] = True
    audit_body = dict(attacked["audit"])
    audit_body.pop("audit_sha256")
    attacked["audit"]["audit_sha256"] = _sha(audit_body)
    bundle_body = dict(attacked)
    bundle_body.pop("bundle_sha256")
    attacked["bundle_sha256"] = _sha(bundle_body)
    with pytest.raises(CombinedSFTExternalAuditError):
        validate_combined_sft_external_audit_bundle(
            attacked,
            **_documents(external_audit_inputs),
            trusted_root_public_key_pem=external_audit_inputs["root_pem"],
            expected_sequence=1,
            expected_previous_audit_sha256=ZERO_SHA256,
            expected_policy_sha256=external_audit_inputs["policy"].policy_sha256,
            minimum_policy_revision=1,
            now_unix=NOW,
        )


def _write_json(path: Path, value: Any) -> None:
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def test_private_key_free_cli_assembles_and_reverifies_committed_publication(
    external_audit_inputs,
    tmp_path: Path,
    capsysbinary,
) -> None:
    publication_root = tmp_path / "published"
    publish_combined_sft_lineage_custody(
        bundle=external_audit_inputs["bundle"],
        publication_root=publication_root,
    )
    paths = {
        "privacy-report": external_audit_inputs["privacy_report"],
        "contamination-report": external_audit_inputs["contamination_report"],
        "execution-report": external_audit_inputs["execution_report"],
        "runner-binding": external_audit_inputs["runner_binding"],
        "policy": external_audit_inputs["policy"].document,
    }
    arguments = []
    for name, document in paths.items():
        path = tmp_path / f"{name}.json"
        _write_json(path, document)
        arguments.extend((f"--{name}", str(path)))
    root_path = tmp_path / "trusted-root.pem"
    root_path.write_bytes(external_audit_inputs["root_pem"])
    attestation_arguments = []
    for name, attestation in _attestations(external_audit_inputs).items():
        path = tmp_path / f"{name}.json"
        _write_json(path, attestation)
        attestation_arguments.extend((f"--{name.replace('_', '-')}-attestation", str(path)))
    common = [
        "--candidate-dir",
        str(publication_root / "candidate"),
        "--evaluator-dir",
        str(publication_root / "evaluator"),
        *arguments,
        "--root",
        str(root_path),
        "--minimum-policy-revision",
        "1",
        "--observed-at",
        str(NOW),
    ]
    bundle_path = tmp_path / "external-audit.json"
    assembled_code = external_audit_cli_main(
        [
            "assemble",
            *common,
            "--sequence",
            "1",
            "--previous-audit-sha256",
            ZERO_SHA256,
            *attestation_arguments,
            "--out",
            str(bundle_path),
        ]
    )
    assembled_output = capsysbinary.readouterr()
    assert assembled_code == 0, assembled_output.out + assembled_output.err
    assembled_bundle = json.loads(assembled_output.out)
    verified_path = tmp_path / "verified.json"
    verified_code = external_audit_cli_main(
        [
            "verify",
            *common,
            "--bundle",
            str(bundle_path),
            "--expected-sequence",
            "1",
            "--expected-previous-audit-sha256",
            ZERO_SHA256,
            "--expected-policy-sha256",
            external_audit_inputs["policy"].policy_sha256,
            "--out",
            str(verified_path),
        ]
    )
    verified_output = capsysbinary.readouterr()
    assert verified_code == 0, verified_output.out + verified_output.err
    assert json.loads(verified_output.out) == assembled_bundle
    assert json.loads(verified_path.read_bytes()) == assembled_bundle
