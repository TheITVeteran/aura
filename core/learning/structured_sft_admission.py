"""Externally rooted admission contract for structured-SFT candidates.

The curriculum producer cannot authorize its own output.  This module binds a
validated custody pair and tokenizer projection to Aura's existing root-signed,
role-separated campaign policy.  Four independently pinned organizations must
attest to the package declaration, contamination audit, evidence audit, and
trainer binding.  The caller must also supply a trusted monotonic floor; a
self-contained bundle is never allowed to declare that an older bundle is
current.

Passing this contract still does not authorize training.  Transfer,
non-inferiority, resident execution, and promotion remain separate gates.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final, Never

from core.brain.llm.latent_cortex.campaign_journal import canonical_json_bytes
from core.brain.llm.latent_cortex.campaign_trust import (
    CAMPAIGN_RUNNER,
    CONTAMINATION_AUDITOR,
    EVIDENCE_VERIFIER,
    TASK_ISSUER,
    CampaignTrustError,
    VerifiedCampaignTrustPolicy,
    externally_custodied_roles,
    validate_campaign_trust_policy,
    verify_role_attestation,
)
from core.learning.structured_sft import (
    validate_candidate_dataset_artifacts,
    validate_structured_sft_custody_pair,
)
from core.runtime.file_read_gateway import read_stable_bytes

STRUCTURED_SFT_ADMISSION_SCHEMA: Final = "aura.rlc.structured_sft_admission.v1"
STRUCTURED_SFT_ADMISSION_BUNDLE_SCHEMA: Final = "aura.rlc.structured_sft_admission_bundle.v1"
STRUCTURED_SFT_PRIVACY_AUDIT_SCHEMA: Final = "aura.rlc.structured_sft_privacy_audit.v1"
STRUCTURED_SFT_CONTAMINATION_AUDIT_SCHEMA: Final = (
    "aura.rlc.structured_sft_multisurface_contamination_audit.v1"
)
STRUCTURED_SFT_EVIDENCE_AUDIT_SCHEMA: Final = "aura.rlc.structured_sft_evidence_audit.v1"
STRUCTURED_SFT_TRAINER_BINDING_SCHEMA: Final = "aura.rlc.structured_sft_trainer_binding.v1"
STRUCTURED_SFT_ADMISSION_VERSION: Final = "2026.07.26.1"

ZERO_SHA256: Final = "0" * 64
_MAX_SOURCE_BYTES = 4 * 1024 * 1024
_MAX_JSON_NODES = 250_000
_MAX_JSON_DEPTH = 128
_MAX_ATTESTATION_AGE_S = 7 * 24 * 60 * 60

_PROTOCOL_SOURCES: Final = (
    "core/learning/structured_sft_admission.py",
    "core/learning/structured_sft.py",
    "core/brain/llm/latent_cortex/campaign_trust.py",
    "tools/build_structured_sft_dataset.py",
    "tools/manage_campaign_trust.py",
    "tools/manage_structured_sft_admission.py",
    "tools/validate_structured_sft_tokenization.py",
)
_REQUIRED_CONTAMINATION_SURFACES: Final = (
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
)
_REQUIRED_CONTAMINATION_METHODS: Final = (
    "exact_sha256",
    "normalized_sha256",
    "token_shingle_jaccard",
    "character_shingle_jaccard",
    "canonical_ast_sha256",
    "canonical_json_sha256",
)
_REQUIRED_PRIVACY_CONTROLS: Final = (
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
_REQUIRED_EVIDENCE_CHECKS: Final = (
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
_POST_ADMISSION_GATES: Final = (
    "replay_transfer_noninferiority_evaluation",
    "external_replay_sft_authority",
    "resident_training_execution_receipt",
    "independent_heldout_promotion_protocol",
)

_PRIVACY_FIELDS = {
    "schema",
    "candidate_package_sha256",
    "evaluator_package_sha256",
    "custody_root_sha256",
    "implementation_sha256",
    "release_sha256",
    "controls",
    "contains_user_content",
    "pii_findings",
    "secret_findings",
    "tenant_violations",
    "license_violations",
    "consent_violations",
    "retention_violations",
    "revocation_violations",
    "deletion_violations",
    "remote_sync_violations",
    "status",
    "report_sha256",
}
_CONTAMINATION_FIELDS = {
    "schema",
    "candidate_package_sha256",
    "evaluator_package_sha256",
    "custody_root_sha256",
    "implementation_sha256",
    "release_sha256",
    "surfaces",
    "methods",
    "reference_corpora",
    "pre_augmentation_partition_sha256",
    "semantic_dedup_manifest_sha256",
    "exact_overlap_count",
    "near_duplicate_overlap_count",
    "cross_split_lineage_overlap_count",
    "uncovered_surface_count",
    "status",
    "report_sha256",
}
_EVIDENCE_FIELDS = {
    "schema",
    "candidate_package_sha256",
    "evaluator_package_sha256",
    "custody_root_sha256",
    "source_closure_sha256",
    "tokenizer_validation_bundle_sha256",
    "tokenizer_identity_sha256",
    "tokenizer_runtime_identity_sha256",
    "implementation_sha256",
    "release_sha256",
    "checks",
    "visible_examples_replayed",
    "holdout_examples_replayed",
    "tool_traces_reexecuted",
    "failed_check_count",
    "status",
    "report_sha256",
}
_TRAINER_FIELDS = {
    "schema",
    "candidate_package_sha256",
    "custody_root_sha256",
    "source_closure_sha256",
    "tokenizer_validation_bundle_sha256",
    "tokenizer_identity_sha256",
    "tokenizer_runtime_identity_sha256",
    "tokenizer_snapshot_manifest_sha256",
    "trainer",
    "mask_prompt",
    "supervised_region",
    "max_seq_length",
    "truncation_allowed",
    "revalidate_in_trainer_process",
    "candidate_only_revalidation",
    "path_substitution_allowed",
    "model_sha256",
    "adapter_base_sha256",
    "recurrence_program_sha256",
    "optimizer_config_sha256",
    "scheduler_config_sha256",
    "rng_manifest_sha256",
    "compute_budget_sha256",
    "execution_authorized",
    "status",
    "binding_sha256",
}
_BUNDLE_FIELDS = {
    "schema",
    "admission",
    "policy",
    "attestations",
    "bundle_sha256",
}
_ATTESTATION_ROLES = {
    "package_declaration": TASK_ISSUER,
    "contamination_audit": CONTAMINATION_AUDITOR,
    "evidence_audit": EVIDENCE_VERIFIER,
    "trainer_binding": CAMPAIGN_RUNNER,
}
_TOKENIZER_VALIDATION_FIELDS = {
    "schema",
    "projection_schema",
    "projection_curriculum_sha256",
    "projection_report_sha256",
    "candidate_curriculum_commitment_sha256",
    "tokenization_scope",
    "candidate_package_sha256",
    "candidate_validation_scope",
    "candidate_custody_attestation",
    "tokenizer",
    "trainer_binding_contract",
    "trainer",
    "mask_prompt",
    "max_seq_length",
    "truncation_allowed",
    "rows_with_truncation",
    "holdout_tokenized",
    "rows_checked",
    "groups",
    "projection_receipts_sha256",
    "status",
    "validation_bundle_sha256",
}
_TOKENIZER_VALIDATION_SCHEMA = "aura.rlc.structured_sft_tokenizer_validation_bundle.v3"


class StructuredSFTAdmissionError(ValueError):
    """Stable fail-closed admission error."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _fail(code: str) -> Never:
    error = StructuredSFTAdmissionError(code)
    raise error


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_sha(value: Any, *, field: str) -> str:
    if not _is_sha256(value):
        _fail(f"structured_sft_admission_{field}_invalid")
    return value


def _normalized(value: Any, *, code: str) -> Any:
    nodes = 0
    stack = [(value, 0)]
    while stack:
        current, depth = stack.pop()
        nodes += 1
        if nodes > _MAX_JSON_NODES or depth > _MAX_JSON_DEPTH:
            _fail(code)
        if isinstance(current, Mapping):
            if any(not isinstance(key, str) for key in current):
                _fail(code)
            stack.extend((child, depth + 1) for child in current.values())
        elif isinstance(current, list):
            stack.extend((child, depth + 1) for child in current)
        elif current is not None:
            if not isinstance(current, (str, int, float, bool)):
                _fail(code)
            if isinstance(current, float) and not math.isfinite(current):
                _fail(code)
    try:
        return json.loads(canonical_json_bytes(value))
    except (TypeError, ValueError, RecursionError, OverflowError):
        _fail(code)


def _committed_report(
    raw: Any,
    *,
    fields: set[str],
    schema: str,
    digest_field: str,
    code: str,
) -> dict[str, Any]:
    if not isinstance(raw, Mapping) or set(raw) != fields:
        _fail(code)
    report = _normalized(raw, code=code)
    if not isinstance(report, dict) or report.get("schema") != schema:
        _fail(code)
    body = dict(report)
    digest = body.pop(digest_field, None)
    if not _is_sha256(digest) or hashlib.sha256(canonical_json_bytes(body)).hexdigest() != digest:
        _fail(f"{code}_commitment_invalid")
    return report


def structured_sft_admission_protocol() -> dict[str, Any]:
    """Bind every executable source that defines this admission protocol."""

    root = Path(__file__).resolve().parents[2]
    records: list[dict[str, Any]] = []
    for relative in _PROTOCOL_SOURCES:
        path = root / relative
        payload = read_stable_bytes(path, max_bytes=_MAX_SOURCE_BYTES)
        records.append(
            {
                "path": relative,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }
        )
    body = {
        "schema": "aura.rlc.structured_sft_admission_protocol.v1",
        "version": STRUCTURED_SFT_ADMISSION_VERSION,
        "sources": records,
        "required_roles": dict(_ATTESTATION_ROLES),
        "required_contamination_surfaces": list(_REQUIRED_CONTAMINATION_SURFACES),
        "required_contamination_methods": list(_REQUIRED_CONTAMINATION_METHODS),
        "required_privacy_controls": list(_REQUIRED_PRIVACY_CONTROLS),
        "required_evidence_checks": list(_REQUIRED_EVIDENCE_CHECKS),
        "post_admission_gates": list(_POST_ADMISSION_GATES),
    }
    return {
        **body,
        "protocol_sha256": hashlib.sha256(canonical_json_bytes(body)).hexdigest(),
    }


def _package_binding(
    candidate_artifacts: Mapping[str, bytes],
    evaluator_artifacts: Mapping[str, bytes],
    tokenizer_validation: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    candidate = validate_candidate_dataset_artifacts(candidate_artifacts)
    custody = validate_structured_sft_custody_pair(
        candidate_artifacts,
        evaluator_artifacts,
    )
    tokenizer = _normalized(
        tokenizer_validation,
        code="structured_sft_admission_tokenizer_validation_invalid",
    )
    if not isinstance(tokenizer, dict):
        _fail("structured_sft_admission_tokenizer_validation_invalid")
    if set(tokenizer) != _TOKENIZER_VALIDATION_FIELDS:
        _fail("structured_sft_admission_tokenizer_validation_invalid")
    tokenizer_body = dict(tokenizer)
    tokenizer_bundle_sha = tokenizer_body.pop("validation_bundle_sha256", None)
    if (
        not _is_sha256(tokenizer_bundle_sha)
        or hashlib.sha256(canonical_json_bytes(tokenizer_body)).hexdigest() != tokenizer_bundle_sha
    ):
        _fail("structured_sft_admission_tokenizer_commitment_invalid")
    attestation = tokenizer.get("candidate_custody_attestation")
    tokenizer_record = tokenizer.get("tokenizer")
    trainer_contract = tokenizer.get("trainer_binding_contract")
    trainer = candidate["trainer_contract"]
    if (
        tokenizer.get("schema") != _TOKENIZER_VALIDATION_SCHEMA
        or tokenizer.get("status") != "passed_exact_masked_prefix"
        or tokenizer.get("projection_schema") != "aura.rlc.structured_sft_tokenization.v1"
        or not _is_sha256(tokenizer.get("projection_curriculum_sha256"))
        or not _is_sha256(tokenizer.get("projection_report_sha256"))
        or not _is_sha256(tokenizer.get("projection_receipts_sha256"))
        or tokenizer.get("candidate_curriculum_commitment_sha256")
        != candidate["curriculum_manifest"]["curriculum_sha256"]
        or tokenizer.get("tokenization_scope") != "candidate_train_validation_only"
        or tokenizer.get("candidate_validation_scope") != candidate["validation_scope"]
        or tokenizer.get("trainer") != trainer["trainer"]
        or tokenizer.get("mask_prompt") is not True
        or tokenizer.get("max_seq_length") != trainer["max_seq_length"]
        or tokenizer.get("truncation_allowed") is not False
        or tokenizer.get("rows_with_truncation") != 0
        or tokenizer.get("holdout_tokenized") is not False
        or tokenizer.get("rows_checked") != custody["visible_example_count"]
        or tokenizer.get("candidate_package_sha256") != candidate["package_sha256"]
        or not isinstance(attestation, dict)
        or attestation.get("candidate_package_sha256") != candidate["package_sha256"]
        or attestation.get("evaluator_filesystem_accessed") is not False
        or attestation.get("evaluator_package_sha256") != custody["evaluator_package_sha256"]
        or attestation.get("custody_root_sha256") != custody["custody_root_sha256"]
        or not isinstance(tokenizer_record, dict)
        or not _is_sha256(tokenizer_record.get("sha256"))
        or not isinstance(tokenizer_record.get("runtime"), dict)
        or not _is_sha256(tokenizer_record["runtime"].get("sha256"))
        or not isinstance(tokenizer_record.get("snapshot_manifest"), dict)
        or not _is_sha256(tokenizer_record["snapshot_manifest"].get("snapshot_manifest_sha256"))
        or not isinstance(trainer_contract, dict)
        or trainer_contract.get("tokenizer_identity_sha256") != tokenizer_record["sha256"]
        or trainer_contract.get("runtime_identity_sha256") != tokenizer_record["runtime"]["sha256"]
        or trainer_contract.get("snapshot_manifest_sha256")
        != tokenizer_record["snapshot_manifest"]["snapshot_manifest_sha256"]
        or trainer_contract.get("revalidate_in_trainer_process") is not True
        or trainer_contract.get("candidate_only_revalidation") is not True
        or trainer_contract.get("evaluator_filesystem_access_required") is not False
        or trainer_contract.get("path_substitution_allowed") is not False
    ):
        _fail("structured_sft_admission_tokenizer_binding_invalid")
    return candidate, custody, tokenizer


def _validate_privacy_report(
    raw: Any, *, candidate: Mapping[str, Any], custody: Mapping[str, Any]
) -> dict[str, Any]:
    report = _committed_report(
        raw,
        fields=_PRIVACY_FIELDS,
        schema=STRUCTURED_SFT_PRIVACY_AUDIT_SCHEMA,
        digest_field="report_sha256",
        code="structured_sft_admission_privacy_report_invalid",
    )
    controls = report.get("controls")
    zero_fields = (
        "pii_findings",
        "secret_findings",
        "tenant_violations",
        "license_violations",
        "consent_violations",
        "retention_violations",
        "revocation_violations",
        "deletion_violations",
        "remote_sync_violations",
    )
    if (
        report.get("candidate_package_sha256") != candidate["package_sha256"]
        or report.get("evaluator_package_sha256") != custody["evaluator_package_sha256"]
        or report.get("custody_root_sha256") != custody["custody_root_sha256"]
        or not _is_sha256(report.get("implementation_sha256"))
        or not _is_sha256(report.get("release_sha256"))
        or not isinstance(controls, dict)
        or set(controls) != set(_REQUIRED_PRIVACY_CONTROLS)
        or any(controls[name] != "passed" for name in _REQUIRED_PRIVACY_CONTROLS)
        or report.get("contains_user_content") is not False
        or any(report.get(field) != 0 for field in zero_fields)
        or report.get("status") != "passed_synthetic_non_user_data"
    ):
        _fail("structured_sft_admission_privacy_report_failed")
    return report


def _validate_contamination_report(
    raw: Any, *, candidate: Mapping[str, Any], custody: Mapping[str, Any]
) -> dict[str, Any]:
    report = _committed_report(
        raw,
        fields=_CONTAMINATION_FIELDS,
        schema=STRUCTURED_SFT_CONTAMINATION_AUDIT_SCHEMA,
        digest_field="report_sha256",
        code="structured_sft_admission_contamination_report_invalid",
    )
    references = report.get("reference_corpora")
    if (
        report.get("candidate_package_sha256") != candidate["package_sha256"]
        or report.get("evaluator_package_sha256") != custody["evaluator_package_sha256"]
        or report.get("custody_root_sha256") != custody["custody_root_sha256"]
        or not _is_sha256(report.get("implementation_sha256"))
        or not _is_sha256(report.get("release_sha256"))
        or report.get("surfaces") != list(_REQUIRED_CONTAMINATION_SURFACES)
        or report.get("methods") != list(_REQUIRED_CONTAMINATION_METHODS)
        or not isinstance(references, list)
        or len(references) != 3
        or any(
            not isinstance(item, dict)
            or set(item) != {"name", "kind", "manifest_sha256", "entry_count"}
            or not isinstance(item.get("name"), str)
            or not item["name"]
            or item["name"] != item["name"].strip()
            or len(item["name"]) > 200
            or item.get("kind") not in {"adapter", "training", "evaluation"}
            or not _is_sha256(item.get("manifest_sha256"))
            or isinstance(item.get("entry_count"), bool)
            or not isinstance(item.get("entry_count"), int)
            or item["entry_count"] <= 0
            for item in references
        )
        or len({item["name"] for item in references}) != len(references)
        or {item["kind"] for item in references} != {"adapter", "training", "evaluation"}
        or not _is_sha256(report.get("pre_augmentation_partition_sha256"))
        or not _is_sha256(report.get("semantic_dedup_manifest_sha256"))
        or any(
            report.get(field) != 0
            for field in (
                "exact_overlap_count",
                "near_duplicate_overlap_count",
                "cross_split_lineage_overlap_count",
                "uncovered_surface_count",
            )
        )
        or report.get("status") != "passed_zero_multisurface_overlap"
    ):
        _fail("structured_sft_admission_contamination_report_failed")
    return report


def _validate_evidence_report(
    raw: Any,
    *,
    candidate: Mapping[str, Any],
    custody: Mapping[str, Any],
    tokenizer: Mapping[str, Any],
) -> dict[str, Any]:
    report = _committed_report(
        raw,
        fields=_EVIDENCE_FIELDS,
        schema=STRUCTURED_SFT_EVIDENCE_AUDIT_SCHEMA,
        digest_field="report_sha256",
        code="structured_sft_admission_evidence_report_invalid",
    )
    checks = report.get("checks")
    source_sha = candidate["curriculum_manifest"]["source_binding"]["sha256"]
    tokenizer_record = tokenizer["tokenizer"]
    if (
        report.get("candidate_package_sha256") != candidate["package_sha256"]
        or report.get("evaluator_package_sha256") != custody["evaluator_package_sha256"]
        or report.get("custody_root_sha256") != custody["custody_root_sha256"]
        or report.get("source_closure_sha256") != source_sha
        or report.get("tokenizer_validation_bundle_sha256") != tokenizer["validation_bundle_sha256"]
        or report.get("tokenizer_identity_sha256") != tokenizer_record["sha256"]
        or report.get("tokenizer_runtime_identity_sha256") != tokenizer_record["runtime"]["sha256"]
        or not _is_sha256(report.get("implementation_sha256"))
        or not _is_sha256(report.get("release_sha256"))
        or not isinstance(checks, dict)
        or set(checks) != set(_REQUIRED_EVIDENCE_CHECKS)
        or any(checks[name] != "passed" for name in _REQUIRED_EVIDENCE_CHECKS)
        or report.get("visible_examples_replayed") != custody["visible_example_count"]
        or report.get("holdout_examples_replayed") != custody["holdout_example_count"]
        or isinstance(report.get("tool_traces_reexecuted"), bool)
        or not isinstance(report.get("tool_traces_reexecuted"), int)
        or report["tool_traces_reexecuted"] <= 0
        or report.get("failed_check_count") != 0
        or report.get("status") != "passed_independent_reverification"
    ):
        _fail("structured_sft_admission_evidence_report_failed")
    return report


def _validate_trainer_binding(
    raw: Any,
    *,
    candidate: Mapping[str, Any],
    custody: Mapping[str, Any],
    tokenizer: Mapping[str, Any],
) -> dict[str, Any]:
    binding = _committed_report(
        raw,
        fields=_TRAINER_FIELDS,
        schema=STRUCTURED_SFT_TRAINER_BINDING_SCHEMA,
        digest_field="binding_sha256",
        code="structured_sft_admission_trainer_binding_invalid",
    )
    source_sha = candidate["curriculum_manifest"]["source_binding"]["sha256"]
    tokenizer_record = tokenizer["tokenizer"]
    snapshot = tokenizer_record["snapshot_manifest"]
    trainer_contract = candidate["trainer_contract"]
    if (
        binding.get("candidate_package_sha256") != candidate["package_sha256"]
        or binding.get("custody_root_sha256") != custody["custody_root_sha256"]
        or binding.get("source_closure_sha256") != source_sha
        or binding.get("tokenizer_validation_bundle_sha256")
        != tokenizer["validation_bundle_sha256"]
        or binding.get("tokenizer_identity_sha256") != tokenizer_record["sha256"]
        or binding.get("tokenizer_runtime_identity_sha256") != tokenizer_record["runtime"]["sha256"]
        or binding.get("tokenizer_snapshot_manifest_sha256") != snapshot["snapshot_manifest_sha256"]
        or binding.get("trainer") != trainer_contract["trainer"]
        or binding.get("mask_prompt") is not True
        or binding.get("supervised_region") != trainer_contract["supervised_region"]
        or binding.get("max_seq_length")
        != candidate["curriculum_manifest"]["spec"]["max_seq_length"]
        or binding.get("truncation_allowed") is not False
        or binding.get("revalidate_in_trainer_process") is not True
        or binding.get("candidate_only_revalidation") is not True
        or binding.get("path_substitution_allowed") is not False
        or any(
            not _is_sha256(binding.get(field))
            for field in (
                "model_sha256",
                "adapter_base_sha256",
                "recurrence_program_sha256",
                "optimizer_config_sha256",
                "scheduler_config_sha256",
                "rng_manifest_sha256",
                "compute_budget_sha256",
            )
        )
        or binding.get("execution_authorized") is not False
        or binding.get("status") != "bound_pending_transfer_authority"
    ):
        _fail("structured_sft_admission_trainer_binding_failed")
    return binding


def _admission_payloads_from_validated(
    *,
    candidate: Mapping[str, Any],
    custody: Mapping[str, Any],
    tokenizer: Mapping[str, Any],
    privacy: Mapping[str, Any],
    contamination: Mapping[str, Any],
    evidence: Mapping[str, Any],
    trainer: Mapping[str, Any],
    sequence: int,
    previous_admission_sha256: str,
) -> dict[str, dict[str, Any]]:
    package_declaration = {
        "schema": "aura.rlc.structured_sft_package_declaration.v1",
        "sequence": sequence,
        "previous_admission_sha256": previous_admission_sha256,
        "candidate_package_sha256": candidate["package_sha256"],
        "evaluator_package_sha256": custody["evaluator_package_sha256"],
        "custody_root_sha256": custody["custody_root_sha256"],
        "source_closure_sha256": candidate["curriculum_manifest"]["source_binding"]["sha256"],
        "tokenizer_validation_bundle_sha256": tokenizer["validation_bundle_sha256"],
        "privacy_report_sha256": privacy["report_sha256"],
        "contamination_report_sha256": contamination["report_sha256"],
        "evidence_report_sha256": evidence["report_sha256"],
        "trainer_binding_sha256": trainer["binding_sha256"],
        "producer_training_authority": "none",
    }
    return {
        "package_declaration": package_declaration,
        "contamination_audit": dict(contamination),
        "evidence_audit": {
            "schema": "aura.rlc.structured_sft_combined_evidence.v1",
            "privacy_report": dict(privacy),
            "evidence_report": dict(evidence),
        },
        "trainer_binding": dict(trainer),
    }


def build_structured_sft_admission_bundle(
    *,
    candidate_artifacts: Mapping[str, bytes],
    evaluator_artifacts: Mapping[str, bytes],
    tokenizer_validation: Mapping[str, Any],
    policy: VerifiedCampaignTrustPolicy,
    privacy_report: Mapping[str, Any],
    contamination_report: Mapping[str, Any],
    evidence_report: Mapping[str, Any],
    trainer_binding: Mapping[str, Any],
    attestations: Mapping[str, Any],
    sequence: int,
    previous_admission_sha256: str,
    observed_at_unix: int,
) -> dict[str, Any]:
    """Assemble and verify a role-separated, non-training admission bundle."""

    if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence <= 0:
        _fail("structured_sft_admission_sequence_invalid")
    if (
        isinstance(observed_at_unix, bool)
        or not isinstance(observed_at_unix, int)
        or observed_at_unix <= 0
    ):
        _fail("structured_sft_admission_observed_at_invalid")
    previous = _require_sha(
        previous_admission_sha256,
        field="previous_admission_sha256",
    )
    if (sequence == 1) != (previous == ZERO_SHA256):
        _fail("structured_sft_admission_chain_origin_invalid")
    candidate, custody, tokenizer = _package_binding(
        candidate_artifacts,
        evaluator_artifacts,
        tokenizer_validation,
    )
    privacy = _validate_privacy_report(privacy_report, candidate=candidate, custody=custody)
    contamination = _validate_contamination_report(
        contamination_report, candidate=candidate, custody=custody
    )
    evidence = _validate_evidence_report(
        evidence_report,
        candidate=candidate,
        custody=custody,
        tokenizer=tokenizer,
    )
    trainer = _validate_trainer_binding(
        trainer_binding,
        candidate=candidate,
        custody=custody,
        tokenizer=tokenizer,
    )
    protocol = structured_sft_admission_protocol()
    expected_campaign = f"structured-sft:{candidate['package_sha256']}"
    if (
        policy.document["campaign_name"] != expected_campaign
        or policy.document["protocol_sha256"] != protocol["protocol_sha256"]
        or not externally_custodied_roles(policy)
    ):
        _fail("structured_sft_admission_policy_invalid")
    contamination_pin = policy.role_pin(CONTAMINATION_AUDITOR)
    evidence_pin = policy.role_pin(EVIDENCE_VERIFIER)
    if (
        contamination["implementation_sha256"] != contamination_pin["implementation_sha256"]
        or contamination["release_sha256"] != contamination_pin["release_sha256"]
        or privacy["implementation_sha256"] != evidence_pin["implementation_sha256"]
        or privacy["release_sha256"] != evidence_pin["release_sha256"]
        or evidence["implementation_sha256"] != evidence_pin["implementation_sha256"]
        or evidence["release_sha256"] != evidence_pin["release_sha256"]
    ):
        _fail("structured_sft_admission_auditor_provenance_mismatch")
    expected_payloads = _admission_payloads_from_validated(
        candidate=candidate,
        custody=custody,
        tokenizer=tokenizer,
        privacy=privacy,
        contamination=contamination,
        evidence=evidence,
        trainer=trainer,
        sequence=sequence,
        previous_admission_sha256=previous,
    )
    if not isinstance(attestations, Mapping) or set(attestations) != set(_ATTESTATION_ROLES):
        _fail("structured_sft_admission_attestation_set_invalid")
    signed_times: list[int] = []
    for name, role in _ATTESTATION_ROLES.items():
        try:
            signed = verify_role_attestation(
                policy,
                attestations[name],
                role=role,
                expected_payload=expected_payloads[name],
                not_before_unix=max(
                    policy.document["not_before_unix"],
                    observed_at_unix - _MAX_ATTESTATION_AGE_S,
                ),
                not_after_unix=observed_at_unix,
            )
        except CampaignTrustError as exc:
            raise StructuredSFTAdmissionError(
                f"structured_sft_admission_{name}_{exc.code}"
            ) from exc
        signed_times.append(int(signed["signed_at_unix"]))
    if max(signed_times) - min(signed_times) > _MAX_ATTESTATION_AGE_S:
        _fail("structured_sft_admission_attestation_window_invalid")

    admission_body = {
        "schema": STRUCTURED_SFT_ADMISSION_SCHEMA,
        "sequence": sequence,
        "previous_admission_sha256": previous,
        "observed_at_unix": observed_at_unix,
        "policy_sha256": policy.policy_sha256,
        "policy_revision": policy.document["policy_revision"],
        "root_key_id": policy.root_key_id,
        "protocol_sha256": protocol["protocol_sha256"],
        "candidate_package_sha256": candidate["package_sha256"],
        "evaluator_package_sha256": custody["evaluator_package_sha256"],
        "custody_root_sha256": custody["custody_root_sha256"],
        "tokenizer_validation_bundle_sha256": tokenizer["validation_bundle_sha256"],
        "privacy_report_sha256": privacy["report_sha256"],
        "contamination_report_sha256": contamination["report_sha256"],
        "evidence_report_sha256": evidence["report_sha256"],
        "trainer_binding_sha256": trainer["binding_sha256"],
        "attestation_sha256": {
            name: hashlib.sha256(canonical_json_bytes(attestations[name])).hexdigest()
            for name in _ATTESTATION_ROLES
        },
        "status": "external_pretraining_evidence_verified_no_training_authority",
        "trainer_ready": False,
        "training_authority": "none_pending_transfer_and_promotion_gates",
        "remaining_gates": list(_POST_ADMISSION_GATES),
    }
    admission = {
        **admission_body,
        "admission_sha256": hashlib.sha256(canonical_json_bytes(admission_body)).hexdigest(),
    }
    bundle_body = {
        "schema": STRUCTURED_SFT_ADMISSION_BUNDLE_SCHEMA,
        "admission": admission,
        "policy": policy.document,
        "attestations": _normalized(
            attestations,
            code="structured_sft_admission_attestation_set_invalid",
        ),
    }
    return {
        **bundle_body,
        "bundle_sha256": hashlib.sha256(canonical_json_bytes(bundle_body)).hexdigest(),
    }


def validate_structured_sft_admission_bundle(
    raw: Any,
    *,
    candidate_artifacts: Mapping[str, bytes],
    evaluator_artifacts: Mapping[str, bytes],
    tokenizer_validation: Mapping[str, Any],
    trusted_root_public_key_pem: bytes,
    privacy_report: Mapping[str, Any],
    contamination_report: Mapping[str, Any],
    evidence_report: Mapping[str, Any],
    trainer_binding: Mapping[str, Any],
    expected_sequence: int,
    expected_previous_admission_sha256: str,
    expected_policy_sha256: str,
    minimum_policy_revision: int,
    now_unix: int,
) -> dict[str, Any]:
    """Rebuild a bundle against caller-pinned trust and monotonic state."""

    if not isinstance(raw, Mapping) or set(raw) != _BUNDLE_FIELDS:
        _fail("structured_sft_admission_bundle_schema_invalid")
    bundle = _normalized(raw, code="structured_sft_admission_bundle_invalid")
    if (
        not isinstance(bundle, dict)
        or bundle.get("schema") != STRUCTURED_SFT_ADMISSION_BUNDLE_SCHEMA
    ):
        _fail("structured_sft_admission_bundle_schema_invalid")
    body = dict(bundle)
    bundle_sha = body.pop("bundle_sha256", None)
    if (
        not _is_sha256(bundle_sha)
        or hashlib.sha256(canonical_json_bytes(body)).hexdigest() != bundle_sha
    ):
        _fail("structured_sft_admission_bundle_commitment_invalid")
    admission = bundle.get("admission")
    if not isinstance(admission, dict):
        _fail("structured_sft_admission_bundle_invalid")
    sequence = admission.get("sequence")
    if (
        isinstance(expected_sequence, bool)
        or not isinstance(expected_sequence, int)
        or expected_sequence <= 0
        or isinstance(sequence, bool)
        or not isinstance(sequence, int)
        or sequence != expected_sequence
    ):
        _fail("structured_sft_admission_rollback")
    expected_previous = _require_sha(
        expected_previous_admission_sha256,
        field="expected_previous_admission_sha256",
    )
    if admission.get("previous_admission_sha256") != expected_previous:
        _fail("structured_sft_admission_chain_mismatch")
    policy_sha = _require_sha(
        expected_policy_sha256,
        field="expected_policy_sha256",
    )
    if admission.get("policy_sha256") != policy_sha:
        _fail("structured_sft_admission_policy_pin_mismatch")
    if admission.get("observed_at_unix") != now_unix:
        _fail("structured_sft_admission_observation_time_mismatch")
    candidate = validate_candidate_dataset_artifacts(candidate_artifacts)
    expected_campaign = f"structured-sft:{candidate['package_sha256']}"
    protocol = structured_sft_admission_protocol()
    try:
        policy = validate_campaign_trust_policy(
            bundle.get("policy"),
            trusted_root_public_key_pem=trusted_root_public_key_pem,
            expected_campaign_name=expected_campaign,
            expected_policy_sha256=policy_sha,
            expected_protocol_sha256=protocol["protocol_sha256"],
            minimum_policy_revision=minimum_policy_revision,
            now_unix=now_unix,
        )
    except CampaignTrustError as exc:
        raise StructuredSFTAdmissionError(f"structured_sft_admission_{exc.code}") from exc
    rebuilt = build_structured_sft_admission_bundle(
        candidate_artifacts=candidate_artifacts,
        evaluator_artifacts=evaluator_artifacts,
        tokenizer_validation=tokenizer_validation,
        policy=policy,
        privacy_report=privacy_report,
        contamination_report=contamination_report,
        evidence_report=evidence_report,
        trainer_binding=trainer_binding,
        attestations=bundle["attestations"],
        sequence=sequence,
        previous_admission_sha256=expected_previous,
        observed_at_unix=now_unix,
    )
    if rebuilt != bundle:
        _fail("structured_sft_admission_reconstruction_mismatch")
    if admission.get("trainer_ready") is not False:
        _fail("structured_sft_admission_training_authority_escalation")
    return rebuilt


def structured_sft_admission_payloads(
    *,
    candidate_artifacts: Mapping[str, bytes],
    evaluator_artifacts: Mapping[str, bytes],
    tokenizer_validation: Mapping[str, Any],
    privacy_report: Mapping[str, Any],
    contamination_report: Mapping[str, Any],
    evidence_report: Mapping[str, Any],
    trainer_binding: Mapping[str, Any],
    sequence: int,
    previous_admission_sha256: str,
) -> dict[str, dict[str, Any]]:
    """Return exact payloads that independent role signers must attest."""

    candidate, custody, tokenizer = _package_binding(
        candidate_artifacts,
        evaluator_artifacts,
        tokenizer_validation,
    )
    privacy = _validate_privacy_report(privacy_report, candidate=candidate, custody=custody)
    contamination = _validate_contamination_report(
        contamination_report, candidate=candidate, custody=custody
    )
    evidence = _validate_evidence_report(
        evidence_report,
        candidate=candidate,
        custody=custody,
        tokenizer=tokenizer,
    )
    trainer = _validate_trainer_binding(
        trainer_binding,
        candidate=candidate,
        custody=custody,
        tokenizer=tokenizer,
    )
    previous = _require_sha(
        previous_admission_sha256,
        field="previous_admission_sha256",
    )
    if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence <= 0:
        _fail("structured_sft_admission_sequence_invalid")
    if (sequence == 1) != (previous == ZERO_SHA256):
        _fail("structured_sft_admission_chain_origin_invalid")
    return _admission_payloads_from_validated(
        candidate=candidate,
        custody=custody,
        tokenizer=tokenizer,
        privacy=privacy,
        contamination=contamination,
        evidence=evidence,
        trainer=trainer,
        sequence=sequence,
        previous_admission_sha256=previous,
    )


__all__ = [
    "STRUCTURED_SFT_ADMISSION_BUNDLE_SCHEMA",
    "STRUCTURED_SFT_ADMISSION_SCHEMA",
    "STRUCTURED_SFT_ADMISSION_VERSION",
    "STRUCTURED_SFT_CONTAMINATION_AUDIT_SCHEMA",
    "STRUCTURED_SFT_EVIDENCE_AUDIT_SCHEMA",
    "STRUCTURED_SFT_PRIVACY_AUDIT_SCHEMA",
    "STRUCTURED_SFT_TRAINER_BINDING_SCHEMA",
    "StructuredSFTAdmissionError",
    "ZERO_SHA256",
    "build_structured_sft_admission_bundle",
    "structured_sft_admission_payloads",
    "structured_sft_admission_protocol",
    "validate_structured_sft_admission_bundle",
]
