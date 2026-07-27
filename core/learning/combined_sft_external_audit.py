"""Externally rooted privacy, contamination, and execution audit for SFT.

The combined lineage manifest proves that declared structured, replay, and
evaluation records share one pre-augmentation semantic partition.  It cannot
prove that the producer declared every corpus, inspected private replay under
an independent privacy authority, or independently replayed executable traces.
This module binds those claims to Aura's root-signed, role-separated campaign
trust contract.  Passing it never grants training authority.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

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
from core.learning.combined_sft_lineage import (
    COMBINED_SFT_LINEAGE_EVALUATOR_FILES,
    CombinedSFTLineageError,
    validate_combined_sft_lineage_custody,
)
from core.runtime.file_read_gateway import read_stable_bytes

COMBINED_SFT_EXTERNAL_AUDIT_SCHEMA: Final = "aura.rlc.combined_sft_external_audit.v1"
COMBINED_SFT_EXTERNAL_AUDIT_BUNDLE_SCHEMA: Final = (
    "aura.rlc.combined_sft_external_audit_bundle.v1"
)
COMBINED_SFT_PRIVACY_REPORT_SCHEMA: Final = "aura.rlc.combined_sft_privacy_report.v1"
COMBINED_SFT_CONTAMINATION_REPORT_SCHEMA: Final = (
    "aura.rlc.combined_sft_contamination_report.v1"
)
COMBINED_SFT_EXECUTION_REPORT_SCHEMA: Final = "aura.rlc.combined_sft_execution_report.v1"
COMBINED_SFT_RUNNER_BINDING_SCHEMA: Final = "aura.rlc.combined_sft_runner_binding.v1"
COMBINED_SFT_EXTERNAL_AUDIT_VERSION: Final = "2026.07.26.1"

ZERO_SHA256: Final = "0" * 64
_MAX_SOURCE_BYTES = 4 * 1024 * 1024
_MAX_JSON_NODES = 300_000
_MAX_JSON_DEPTH = 128
_MAX_ATTESTATION_AGE_S = 7 * 24 * 60 * 60
_SHA_FIELDS = (
    "commitment_sha256",
    "manifest_sha256",
    "combined_semantic_index_sha256",
)
_PROTOCOL_SOURCES = (
    "core/learning/combined_sft_external_audit.py",
    "core/learning/combined_sft_lineage.py",
    "core/learning/combined_sft_lineage_publication.py",
    "core/learning/structured_sft.py",
    "core/learning/verified_replay_sft.py",
    "core/brain/llm/latent_cortex/campaign_trust.py",
)
_ROLE_PAYLOADS = {
    "package_declaration": TASK_ISSUER,
    "contamination_audit": CONTAMINATION_AUDITOR,
    "evidence_audit": EVIDENCE_VERIFIER,
    "runner_binding": CAMPAIGN_RUNNER,
}
_PRIVACY_CONTROLS = (
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
_PRIVACY_ZERO_FIELDS = (
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
_CONTAMINATION_METHODS = (
    "exact_sha256",
    "normalized_sha256",
    "token_shingle_jaccard",
    "character_shingle_jaccard",
    "canonical_ast_sha256",
    "canonical_json_sha256",
    "causal_lineage_identity",
)
_EXECUTION_CHECKS = (
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
_COMMON_REPORT_FIELDS = {
    "commitment_sha256",
    "manifest_sha256",
    "combined_semantic_index_sha256",
    "implementation_sha256",
    "release_sha256",
}
_PRIVACY_FIELDS = _COMMON_REPORT_FIELDS | {
    "schema",
    "sources",
    "report_sha256",
    "status",
}
_CONTAMINATION_FIELDS = _COMMON_REPORT_FIELDS | {
    "schema",
    "inventory_sha256",
    "coverage",
    "required_evaluation_corpora",
    "record_counts",
    "methods",
    "reference_corpora",
    "exact_overlap_count",
    "near_duplicate_overlap_count",
    "cross_split_lineage_overlap_count",
    "uncovered_surface_count",
    "report_sha256",
    "status",
}
_EXECUTION_FIELDS = _COMMON_REPORT_FIELDS | {
    "schema",
    "structured_custody_root_sha256",
    "verified_replay_custody_root_sha256",
    "checks",
    "visible_examples_replayed",
    "holdout_examples_replayed",
    "tool_traces_reexecuted",
    "failed_check_count",
    "report_sha256",
    "status",
}
_RUNNER_FIELDS = _COMMON_REPORT_FIELDS | {
    "schema",
    "candidate_inputs_loadable",
    "execution_authorized",
    "trainer_ready",
    "training_authority",
    "implementation_sha256",
    "release_sha256",
    "binding_sha256",
    "status",
}
_BUNDLE_FIELDS = {"schema", "audit", "policy", "attestations", "bundle_sha256"}


class CombinedSFTExternalAuditError(ValueError):
    """Stable fail-closed combined audit error."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _error(code: str) -> CombinedSFTExternalAuditError:
    return CombinedSFTExternalAuditError(code)


def _is_sha(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _normalized(value: Any, *, code: str) -> Any:
    nodes = 0
    stack = [(value, 0)]
    while stack:
        current, depth = stack.pop()
        nodes += 1
        if nodes > _MAX_JSON_NODES or depth > _MAX_JSON_DEPTH:
            raise _error(code)
        if isinstance(current, Mapping):
            if any(not isinstance(key, str) for key in current):
                raise _error(code)
            stack.extend((child, depth + 1) for child in current.values())
        elif isinstance(current, list):
            stack.extend((child, depth + 1) for child in current)
        elif current is not None:
            if not isinstance(current, (str, int, float, bool)):
                raise _error(code)
            if isinstance(current, float) and not math.isfinite(current):
                raise _error(code)
    try:
        return json.loads(canonical_json_bytes(value))
    except (OverflowError, RecursionError, TypeError, ValueError) as exc:
        raise _error(code) from exc


def _sha(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _committed_report(
    raw: Any,
    *,
    fields: set[str],
    schema: str,
    digest_field: str,
    code: str,
) -> dict[str, Any]:
    if not isinstance(raw, Mapping) or set(raw) != fields:
        raise _error(code)
    report = _normalized(raw, code=code)
    if not isinstance(report, dict) or report.get("schema") != schema:
        raise _error(code)
    body = dict(report)
    digest = body.pop(digest_field, None)
    if not _is_sha(digest) or _sha(body) != digest:
        raise _error(f"{code}_commitment_invalid")
    return report


def combined_sft_external_audit_protocol() -> dict[str, Any]:
    """Commit the executable protocol and every mandatory audit dimension."""

    root = Path(__file__).resolve().parents[2]
    records = []
    for relative in _PROTOCOL_SOURCES:
        payload = read_stable_bytes(root / relative, max_bytes=_MAX_SOURCE_BYTES)
        records.append(
            {"path": relative, "sha256": hashlib.sha256(payload).hexdigest(), "size_bytes": len(payload)}
        )
    body = {
        "schema": "aura.rlc.combined_sft_external_audit_protocol.v1",
        "version": COMBINED_SFT_EXTERNAL_AUDIT_VERSION,
        "sources": records,
        "required_roles": dict(_ROLE_PAYLOADS),
        "required_privacy_controls": list(_PRIVACY_CONTROLS),
        "required_contamination_methods": list(_CONTAMINATION_METHODS),
        "required_execution_checks": list(_EXECUTION_CHECKS),
    }
    return {**body, "protocol_sha256": _sha(body)}


def _manifest(
    candidate_artifacts: Mapping[str, bytes],
    evaluator_artifacts: Mapping[str, bytes],
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        custody = validate_combined_sft_lineage_custody(
            candidate_artifacts,
            evaluator_artifacts,
        )
    except CombinedSFTLineageError as exc:
        raise CombinedSFTExternalAuditError(
            "combined_sft_external_audit_custody_invalid"
        ) from exc
    try:
        manifest = json.loads(
            evaluator_artifacts[COMBINED_SFT_LINEAGE_EVALUATOR_FILES[0]]
        )
    except (KeyError, RecursionError, UnicodeDecodeError, ValueError) as exc:
        raise CombinedSFTExternalAuditError("combined_sft_external_audit_manifest_invalid") from exc
    if not isinstance(manifest, dict):
        raise _error("combined_sft_external_audit_manifest_invalid")
    return custody, manifest


def _binds_custody(report: Mapping[str, Any], custody: Mapping[str, Any]) -> bool:
    return all(report.get(field) == custody[field] for field in _SHA_FIELDS)


def _privacy_report(
    raw: Any,
    *,
    custody: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    report = _committed_report(
        raw,
        fields=_PRIVACY_FIELDS,
        schema=COMBINED_SFT_PRIVACY_REPORT_SCHEMA,
        digest_field="report_sha256",
        code="combined_sft_external_audit_privacy_report_invalid",
    )
    sources = report.get("sources")
    expected_bindings = {
        "structured_synthetic": manifest["structured_binding"],
        "verified_replay_user_content": manifest["verified_replay_binding"],
    }
    if (
        not _binds_custody(report, custody)
        or not _is_sha(report.get("implementation_sha256"))
        or not _is_sha(report.get("release_sha256"))
        or not isinstance(sources, list)
        or len(sources) != 2
        or report.get("status") != "passed_source_class_specific_privacy_audit"
    ):
        raise _error("combined_sft_external_audit_privacy_report_failed")
    observed: dict[str, Mapping[str, Any]] = {}
    source_fields = {
        "source_class",
        "candidate_package_sha256",
        "evaluator_package_sha256",
        "custody_root_sha256",
        "privacy_manifest_sha256",
        "contains_user_content",
        "controls",
        *_PRIVACY_ZERO_FIELDS,
        "status",
    }
    for source in sources:
        if not isinstance(source, Mapping) or set(source) != source_fields:
            raise _error("combined_sft_external_audit_privacy_source_invalid")
        source_class = source.get("source_class")
        if source_class not in expected_bindings or source_class in observed:
            raise _error("combined_sft_external_audit_privacy_source_invalid")
        observed[source_class] = source
    if set(observed) != set(expected_bindings):
        raise _error("combined_sft_external_audit_privacy_source_missing")
    for source_class, binding in expected_bindings.items():
        source = observed[source_class]
        expected_user_content = source_class == "verified_replay_user_content"
        expected_privacy = (
            binding["privacy_manifest_sha256"] if expected_user_content else ZERO_SHA256
        )
        if (
            source.get("candidate_package_sha256") != binding["candidate_package_sha256"]
            or source.get("evaluator_package_sha256") != binding["evaluator_package_sha256"]
            or source.get("custody_root_sha256") != binding["custody_root_sha256"]
            or source.get("privacy_manifest_sha256") != expected_privacy
            or source.get("contains_user_content") is not expected_user_content
            or not isinstance(source.get("controls"), dict)
            or set(source["controls"]) != set(_PRIVACY_CONTROLS)
            or any(source["controls"][name] != "passed" for name in _PRIVACY_CONTROLS)
            or any(source.get(field) != 0 for field in _PRIVACY_ZERO_FIELDS)
            or source.get("status")
            != (
                "passed_user_content_governance"
                if expected_user_content
                else "passed_synthetic_non_user_data"
            )
        ):
            raise _error("combined_sft_external_audit_privacy_source_failed")
    return report


def _inventory(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "coverage": manifest["coverage"],
        "required_evaluation_corpora": manifest["required_evaluation_corpora"],
        "record_counts": manifest["record_counts"],
    }


def _contamination_report(
    raw: Any,
    *,
    custody: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    report = _committed_report(
        raw,
        fields=_CONTAMINATION_FIELDS,
        schema=COMBINED_SFT_CONTAMINATION_REPORT_SCHEMA,
        digest_field="report_sha256",
        code="combined_sft_external_audit_contamination_report_invalid",
    )
    references = report.get("reference_corpora")
    required = manifest["required_evaluation_corpora"]
    if (
        not _binds_custody(report, custody)
        or report.get("inventory_sha256") != _sha(_inventory(manifest))
        or report.get("coverage") != manifest["coverage"]
        or report.get("required_evaluation_corpora") != required
        or report.get("record_counts") != manifest["record_counts"]
        or report.get("methods") != list(_CONTAMINATION_METHODS)
        or not _is_sha(report.get("implementation_sha256"))
        or not _is_sha(report.get("release_sha256"))
        or not isinstance(references, list)
        or len(references) < len(required) + 2
        or any(
            not isinstance(item, Mapping)
            or set(item) != {"name", "kind", "manifest_sha256", "entry_count"}
            or not isinstance(item.get("name"), str)
            or not item["name"]
            or item["name"] != item["name"].strip()
            or item.get("kind") not in {"adapter", "training", "evaluation"}
            or not _is_sha(item.get("manifest_sha256"))
            or type(item.get("entry_count")) is not int
            or item["entry_count"] < 1
            for item in references
        )
        or len({(item["kind"], item["name"]) for item in references}) != len(references)
        or {item["kind"] for item in references} != {"adapter", "training", "evaluation"}
        or not set(required)
        <= {item["name"] for item in references if item["kind"] == "evaluation"}
        or any(
            report.get(field) != 0
            for field in (
                "exact_overlap_count",
                "near_duplicate_overlap_count",
                "cross_split_lineage_overlap_count",
                "uncovered_surface_count",
            )
        )
        or report.get("status") != "passed_complete_zero_multisurface_overlap"
    ):
        raise _error("combined_sft_external_audit_contamination_report_failed")
    return report


def _execution_report(
    raw: Any,
    *,
    custody: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    report = _committed_report(
        raw,
        fields=_EXECUTION_FIELDS,
        schema=COMBINED_SFT_EXECUTION_REPORT_SCHEMA,
        digest_field="report_sha256",
        code="combined_sft_external_audit_execution_report_invalid",
    )
    checks = report.get("checks")
    structured = manifest["structured_binding"]
    replay = manifest["verified_replay_binding"]
    counts = manifest["record_counts"]
    expected_visible = sum(item["count"] for item in counts if item["split"] != "holdout")
    expected_holdout = sum(item["count"] for item in counts if item["split"] == "holdout")
    expected_tool_traces = sum(
        item["count"]
        for item in counts
        if item["corpus"] in {"structured_sft:code_tool", "structured_sft:code_tool_repair"}
    )
    if (
        not _binds_custody(report, custody)
        or report.get("structured_custody_root_sha256") != structured["custody_root_sha256"]
        or report.get("verified_replay_custody_root_sha256") != replay["custody_root_sha256"]
        or not _is_sha(report.get("implementation_sha256"))
        or not _is_sha(report.get("release_sha256"))
        or not isinstance(checks, dict)
        or set(checks) != set(_EXECUTION_CHECKS)
        or any(checks[name] != "passed" for name in _EXECUTION_CHECKS)
        or report.get("visible_examples_replayed") != expected_visible
        or report.get("holdout_examples_replayed") != expected_holdout
        or expected_visible < 1
        or expected_holdout < 1
        or report.get("tool_traces_reexecuted") != expected_tool_traces
        or expected_tool_traces < 1
        or report.get("failed_check_count") != 0
        or report.get("status") != "passed_independent_source_and_execution_replay"
    ):
        raise _error("combined_sft_external_audit_execution_report_failed")
    return report


def _runner_binding(raw: Any, *, custody: Mapping[str, Any]) -> dict[str, Any]:
    report = _committed_report(
        raw,
        fields=_RUNNER_FIELDS,
        schema=COMBINED_SFT_RUNNER_BINDING_SCHEMA,
        digest_field="binding_sha256",
        code="combined_sft_external_audit_runner_binding_invalid",
    )
    if (
        not _binds_custody(report, custody)
        or not _is_sha(report.get("implementation_sha256"))
        or not _is_sha(report.get("release_sha256"))
        or report.get("candidate_inputs_loadable") is not False
        or report.get("execution_authorized") is not False
        or report.get("trainer_ready") is not False
        or report.get("training_authority") != "none_pending_tokenizer_and_transfer_gates"
        or report.get("status") != "bound_quarantine_no_training_authority"
    ):
        raise _error("combined_sft_external_audit_runner_binding_failed")
    return report


def _validated_reports(
    *,
    candidate_artifacts: Mapping[str, bytes],
    evaluator_artifacts: Mapping[str, bytes],
    privacy_report: Mapping[str, Any],
    contamination_report: Mapping[str, Any],
    execution_report: Mapping[str, Any],
    runner_binding: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    custody, manifest = _manifest(candidate_artifacts, evaluator_artifacts)
    return (
        custody,
        manifest,
        _privacy_report(privacy_report, custody=custody, manifest=manifest),
        _contamination_report(contamination_report, custody=custody, manifest=manifest),
        _execution_report(execution_report, custody=custody, manifest=manifest),
        _runner_binding(runner_binding, custody=custody),
    )


def _payloads(
    *,
    custody: Mapping[str, Any],
    manifest: Mapping[str, Any],
    privacy: Mapping[str, Any],
    contamination: Mapping[str, Any],
    execution: Mapping[str, Any],
    runner: Mapping[str, Any],
    sequence: int,
    previous_audit_sha256: str,
) -> dict[str, dict[str, Any]]:
    return {
        "package_declaration": {
            "schema": "aura.rlc.combined_sft_package_declaration.v1",
            "sequence": sequence,
            "previous_audit_sha256": previous_audit_sha256,
            **{field: custody[field] for field in _SHA_FIELDS},
            "structured_candidate_package_sha256": manifest["structured_binding"][
                "candidate_package_sha256"
            ],
            "verified_replay_candidate_package_sha256": manifest[
                "verified_replay_binding"
            ]["candidate_package_sha256"],
            "inventory_sha256": _sha(_inventory(manifest)),
            "privacy_report_sha256": privacy["report_sha256"],
            "contamination_report_sha256": contamination["report_sha256"],
            "execution_report_sha256": execution["report_sha256"],
            "runner_binding_sha256": runner["binding_sha256"],
            "producer_training_authority": "none",
        },
        "contamination_audit": dict(contamination),
        "evidence_audit": {
            "schema": "aura.rlc.combined_sft_privacy_execution_evidence.v1",
            "privacy_report": dict(privacy),
            "execution_report": dict(execution),
        },
        "runner_binding": dict(runner),
    }


def combined_sft_external_audit_payloads(
    *,
    candidate_artifacts: Mapping[str, bytes],
    evaluator_artifacts: Mapping[str, bytes],
    privacy_report: Mapping[str, Any],
    contamination_report: Mapping[str, Any],
    execution_report: Mapping[str, Any],
    runner_binding: Mapping[str, Any],
    sequence: int,
    previous_audit_sha256: str,
) -> dict[str, dict[str, Any]]:
    """Return exact payloads for four independent detached signatures."""

    if type(sequence) is not int or sequence <= 0 or not _is_sha(previous_audit_sha256):
        raise _error("combined_sft_external_audit_chain_invalid")
    if (sequence == 1) != (previous_audit_sha256 == ZERO_SHA256):
        raise _error("combined_sft_external_audit_chain_origin_invalid")
    custody, manifest, privacy, contamination, execution, runner = _validated_reports(
        candidate_artifacts=candidate_artifacts,
        evaluator_artifacts=evaluator_artifacts,
        privacy_report=privacy_report,
        contamination_report=contamination_report,
        execution_report=execution_report,
        runner_binding=runner_binding,
    )
    return _payloads(
        custody=custody,
        manifest=manifest,
        privacy=privacy,
        contamination=contamination,
        execution=execution,
        runner=runner,
        sequence=sequence,
        previous_audit_sha256=previous_audit_sha256,
    )


def build_combined_sft_external_audit_bundle(
    *,
    candidate_artifacts: Mapping[str, bytes],
    evaluator_artifacts: Mapping[str, bytes],
    privacy_report: Mapping[str, Any],
    contamination_report: Mapping[str, Any],
    execution_report: Mapping[str, Any],
    runner_binding: Mapping[str, Any],
    policy: VerifiedCampaignTrustPolicy,
    attestations: Mapping[str, Any],
    sequence: int,
    previous_audit_sha256: str,
    observed_at_unix: int,
) -> dict[str, Any]:
    """Verify externally custodied roles and assemble a no-authority audit."""

    payloads = combined_sft_external_audit_payloads(
        candidate_artifacts=candidate_artifacts,
        evaluator_artifacts=evaluator_artifacts,
        privacy_report=privacy_report,
        contamination_report=contamination_report,
        execution_report=execution_report,
        runner_binding=runner_binding,
        sequence=sequence,
        previous_audit_sha256=previous_audit_sha256,
    )
    custody, _manifest_document = _manifest(candidate_artifacts, evaluator_artifacts)
    protocol = combined_sft_external_audit_protocol()
    if (
        not isinstance(policy, VerifiedCampaignTrustPolicy)
        or policy.document.get("campaign_name")
        != f"combined-sft:{custody['commitment_sha256']}"
        or policy.document.get("protocol_sha256") != protocol["protocol_sha256"]
        or not externally_custodied_roles(policy)
        or type(observed_at_unix) is not int
        or observed_at_unix <= 0
        or not isinstance(attestations, Mapping)
        or set(attestations) != set(_ROLE_PAYLOADS)
    ):
        raise _error("combined_sft_external_audit_policy_invalid")
    organizations: set[str] = set()
    keys: set[str] = set()
    signed_times = []
    for name, role in _ROLE_PAYLOADS.items():
        pin = policy.role_pin(role)
        if pin["organization_id"] in organizations or pin["key_id"] in keys:
            raise _error("combined_sft_external_audit_role_separation_invalid")
        organizations.add(pin["organization_id"])
        keys.add(pin["key_id"])
        try:
            verified = verify_role_attestation(
                policy,
                attestations[name],
                role=role,
                expected_payload=payloads[name],
                not_before_unix=max(
                    policy.document["not_before_unix"],
                    observed_at_unix - _MAX_ATTESTATION_AGE_S,
                ),
                not_after_unix=observed_at_unix,
            )
        except CampaignTrustError as exc:
            raise CombinedSFTExternalAuditError(
                f"combined_sft_external_audit_{name}_{exc.code}"
            ) from exc
        signed_times.append(int(verified["signed_at_unix"]))
    if max(signed_times) - min(signed_times) > _MAX_ATTESTATION_AGE_S:
        raise _error("combined_sft_external_audit_attestation_window_invalid")
    evidence_pin = policy.role_pin(EVIDENCE_VERIFIER)
    contamination_pin = policy.role_pin(CONTAMINATION_AUDITOR)
    runner_pin = policy.role_pin(CAMPAIGN_RUNNER)
    if (
        privacy_report.get("implementation_sha256") != evidence_pin["implementation_sha256"]
        or privacy_report.get("release_sha256") != evidence_pin["release_sha256"]
        or execution_report.get("implementation_sha256")
        != evidence_pin["implementation_sha256"]
        or execution_report.get("release_sha256") != evidence_pin["release_sha256"]
        or contamination_report.get("implementation_sha256")
        != contamination_pin["implementation_sha256"]
        or contamination_report.get("release_sha256") != contamination_pin["release_sha256"]
        or runner_binding.get("implementation_sha256") != runner_pin["implementation_sha256"]
        or runner_binding.get("release_sha256") != runner_pin["release_sha256"]
    ):
        raise _error("combined_sft_external_audit_release_identity_mismatch")
    audit_body = {
        "schema": COMBINED_SFT_EXTERNAL_AUDIT_SCHEMA,
        "sequence": sequence,
        "previous_audit_sha256": previous_audit_sha256,
        "observed_at_unix": observed_at_unix,
        "policy_sha256": policy.policy_sha256,
        "policy_revision": policy.document["policy_revision"],
        "root_key_id": policy.root_key_id,
        "protocol_sha256": protocol["protocol_sha256"],
        **{field: custody[field] for field in _SHA_FIELDS},
        "payload_sha256": {name: _sha(payloads[name]) for name in _ROLE_PAYLOADS},
        "attestation_sha256": {
            name: _sha(attestations[name]) for name in _ROLE_PAYLOADS
        },
        "status": "external_combined_evidence_verified_no_training_authority",
        "trainer_ready": False,
        "training_authority": "none_pending_tokenizer_and_transfer_gates",
        "remaining_gates": [
            "resident_replay_tokenizer_validation",
            "resumable_trainer_receipt_authority",
            "small_checkpoint_transfer_falsification",
            "resident_equal_compute_promotion",
        ],
    }
    audit = {**audit_body, "audit_sha256": _sha(audit_body)}
    bundle_body = {
        "schema": COMBINED_SFT_EXTERNAL_AUDIT_BUNDLE_SCHEMA,
        "audit": audit,
        "policy": policy.document,
        "attestations": _normalized(
            attestations,
            code="combined_sft_external_audit_attestations_invalid",
        ),
    }
    return {**bundle_body, "bundle_sha256": _sha(bundle_body)}


def validate_combined_sft_external_audit_bundle(
    raw: Any,
    *,
    candidate_artifacts: Mapping[str, bytes],
    evaluator_artifacts: Mapping[str, bytes],
    privacy_report: Mapping[str, Any],
    contamination_report: Mapping[str, Any],
    execution_report: Mapping[str, Any],
    runner_binding: Mapping[str, Any],
    trusted_root_public_key_pem: bytes,
    expected_sequence: int,
    expected_previous_audit_sha256: str,
    expected_policy_sha256: str,
    minimum_policy_revision: int,
    now_unix: int,
) -> dict[str, Any]:
    """Reconstruct an audit against caller-pinned trust and chain state."""

    if not isinstance(raw, Mapping) or set(raw) != _BUNDLE_FIELDS:
        raise _error("combined_sft_external_audit_bundle_schema_invalid")
    bundle = _normalized(raw, code="combined_sft_external_audit_bundle_invalid")
    if not isinstance(bundle, dict) or bundle.get("schema") != COMBINED_SFT_EXTERNAL_AUDIT_BUNDLE_SCHEMA:
        raise _error("combined_sft_external_audit_bundle_schema_invalid")
    body = dict(bundle)
    bundle_sha = body.pop("bundle_sha256", None)
    if not _is_sha(bundle_sha) or _sha(body) != bundle_sha:
        raise _error("combined_sft_external_audit_bundle_commitment_invalid")
    audit = bundle.get("audit")
    if not isinstance(audit, dict):
        raise _error("combined_sft_external_audit_bundle_invalid")
    if type(expected_sequence) is not int or audit.get("sequence") != expected_sequence:
        raise _error("combined_sft_external_audit_rollback")
    if not _is_sha(expected_previous_audit_sha256) or audit.get(
        "previous_audit_sha256"
    ) != expected_previous_audit_sha256:
        raise _error("combined_sft_external_audit_chain_mismatch")
    if not _is_sha(expected_policy_sha256) or audit.get("policy_sha256") != expected_policy_sha256:
        raise _error("combined_sft_external_audit_policy_pin_mismatch")
    if audit.get("observed_at_unix") != now_unix:
        raise _error("combined_sft_external_audit_observation_time_mismatch")
    custody, _manifest_document = _manifest(candidate_artifacts, evaluator_artifacts)
    protocol = combined_sft_external_audit_protocol()
    try:
        policy = validate_campaign_trust_policy(
            bundle.get("policy"),
            trusted_root_public_key_pem=trusted_root_public_key_pem,
            expected_campaign_name=f"combined-sft:{custody['commitment_sha256']}",
            expected_policy_sha256=expected_policy_sha256,
            expected_protocol_sha256=protocol["protocol_sha256"],
            minimum_policy_revision=minimum_policy_revision,
            now_unix=now_unix,
        )
    except CampaignTrustError as exc:
        raise CombinedSFTExternalAuditError(
            f"combined_sft_external_audit_{exc.code}"
        ) from exc
    rebuilt = build_combined_sft_external_audit_bundle(
        candidate_artifacts=candidate_artifacts,
        evaluator_artifacts=evaluator_artifacts,
        privacy_report=privacy_report,
        contamination_report=contamination_report,
        execution_report=execution_report,
        runner_binding=runner_binding,
        policy=policy,
        attestations=bundle["attestations"],
        sequence=expected_sequence,
        previous_audit_sha256=expected_previous_audit_sha256,
        observed_at_unix=now_unix,
    )
    if rebuilt != bundle:
        raise _error("combined_sft_external_audit_reconstruction_mismatch")
    if audit.get("trainer_ready") is not False:
        raise _error("combined_sft_external_audit_training_authority_escalation")
    return rebuilt


__all__ = [
    "COMBINED_SFT_CONTAMINATION_REPORT_SCHEMA",
    "COMBINED_SFT_EXECUTION_REPORT_SCHEMA",
    "COMBINED_SFT_EXTERNAL_AUDIT_BUNDLE_SCHEMA",
    "COMBINED_SFT_EXTERNAL_AUDIT_SCHEMA",
    "COMBINED_SFT_PRIVACY_REPORT_SCHEMA",
    "COMBINED_SFT_RUNNER_BINDING_SCHEMA",
    "CombinedSFTExternalAuditError",
    "ZERO_SHA256",
    "build_combined_sft_external_audit_bundle",
    "combined_sft_external_audit_payloads",
    "combined_sft_external_audit_protocol",
    "validate_combined_sft_external_audit_bundle",
]
