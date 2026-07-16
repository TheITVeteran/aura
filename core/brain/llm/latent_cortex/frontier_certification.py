"""Machine-checkable release gate for Recursive Latent Cortex frontier gains.

Unit tests can prove this verifier rejects weak evidence. Only a completed
resident-32B bundle can prove the capability claim itself.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from typing import Any

from core.brain.frontier_evidence_v5 import verify_signed_envelope
from core.brain.llm.latent_cortex.experiments import (
    CONJECTURE,
    PROVEN,
    PairedObservation,
    grade_paired_treatment_vs_control,
)

SCHEMA = "aura.latent_cortex.frontier_gain_bundle.v2"
CERTIFICATE_SCHEMA = "aura.latent_cortex.frontier_gain_certificate.v2"
INDEPENDENT_ATTESTATION_SCHEMA = (
    "aura.latent_cortex.frontier_gain_independent_attestation.v1"
)
TASK_COMMITMENT_ATTESTATION_SCHEMA = (
    "aura.latent_cortex.frontier_gain_task_commitment.v1"
)
_COMPARISON_KINDS = {
    "resident_32b_vs_vanilla_same_checkpoint",
    "resident_32b_vs_external_frontier",
}
_SHA256_LENGTH = 64
_MAX_LAYER_APPS = (1 << 63) - 1


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def evidence_payload_sha256(bundle: dict[str, Any]) -> str:
    return canonical_sha256(
        {
            "schema": bundle.get("schema"),
            "producer_id": bundle.get("producer_id"),
            "preregistration": bundle.get("preregistration"),
            "preregistration_sha256": bundle.get("preregistration_sha256"),
            "resident_model": bundle.get("resident_model"),
            "task_commitment_sha256": bundle.get("task_commitment_sha256"),
            "task_commitment": bundle.get("task_commitment"),
            "raw_artifact_manifest_sha256": bundle.get(
                "raw_artifact_manifest_sha256"
            ),
            "trials": bundle.get("trials"),
        }
    )


def _finite_number(value: Any, *, positive: bool = False) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return False
    return math.isfinite(number) and (number > 0.0 if positive else True)


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != _SHA256_LENGTH:
        return False
    return all(character in "0123456789abcdef" for character in value)


def _is_git_sha(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_preregistration(prereg: Any, reasons: list[str]) -> dict[str, Any]:
    if not isinstance(prereg, dict):
        reasons.append("missing_preregistration")
        return {}
    domains = prereg.get("domains")
    if (
        not isinstance(domains, list)
        or len(domains) < 2
        or any(not isinstance(domain, str) or not domain.strip() for domain in domains)
        or len(set(domains)) != len(domains)
    ):
        reasons.append("invalid_preregistered_domains")
    if prereg.get("comparison_kind") not in _COMPARISON_KINDS:
        reasons.append("invalid_comparison_kind")
    if not _is_sha256(prereg.get("architecture_freeze_sha256")):
        reasons.append("missing_architecture_freeze_sha256")
    if not _is_sha256(prereg.get("tool_policy_sha256")):
        reasons.append("missing_tool_policy_sha256")
    if not _is_sha256(prereg.get("compute_estimator_sha256")):
        reasons.append("missing_compute_estimator_sha256")
    if not _is_sha256(prereg.get("treatment_checkpoint_fingerprint")):
        reasons.append("missing_treatment_checkpoint_fingerprint")
    if not _finite_number(prereg.get("frozen_at"), positive=True):
        reasons.append("invalid_freeze_timestamp")
    min_trials = prereg.get("min_trials_per_domain")
    if type(min_trials) is not int or min_trials < 30:
        reasons.append("min_trials_per_domain_below_30")
    alpha = prereg.get("alpha")
    if not _finite_number(alpha) or not 0.0 < float(alpha) <= 0.05:
        reasons.append("invalid_alpha")
    minimum_effect = prereg.get("minimum_effect")
    if not _finite_number(minimum_effect) or not 0.0 < float(minimum_effect) <= 0.5:
        reasons.append("invalid_minimum_effect")
    compute_tolerance = prereg.get("compute_tolerance")
    if not _finite_number(compute_tolerance) or not 0.0 <= float(compute_tolerance) <= 0.05:
        reasons.append("invalid_compute_tolerance")
    if prereg.get("compute_metric") != "estimated_flops":
        reasons.append("compute_metric_must_be_estimated_flops")
    comparison_kind = prereg.get("comparison_kind")
    if comparison_kind == "resident_32b_vs_vanilla_same_checkpoint":
        if not _is_sha256(prereg.get("control_checkpoint_fingerprint")) or prereg.get(
            "control_checkpoint_fingerprint"
        ) != prereg.get("treatment_checkpoint_fingerprint"):
            reasons.append("same_checkpoint_control_not_bound")
    elif comparison_kind == "resident_32b_vs_external_frontier":
        if not str(prereg.get("control_model_id") or ""):
            reasons.append("external_frontier_control_model_missing")
        if not str(prereg.get("control_model_build_fingerprint") or ""):
            reasons.append("external_frontier_control_build_missing")
        if not str(prereg.get("control_provider") or ""):
            reasons.append("external_frontier_control_provider_missing")
    return prereg


def _validate_resident_model(
    resident: Any, prereg: dict[str, Any], reasons: list[str]
) -> dict[str, Any]:
    if not isinstance(resident, dict):
        reasons.append("missing_resident_model_receipt")
        return {}
    parameter_count = resident.get("parameter_count")
    if type(parameter_count) is not int or not 30_000_000_000 <= parameter_count <= 40_000_000_000:
        reasons.append("resident_model_not_32b_class")
    fingerprint = str(resident.get("checkpoint_fingerprint") or "")
    if not _is_sha256(fingerprint) or fingerprint != str(
        prereg.get("treatment_checkpoint_fingerprint") or ""
    ):
        reasons.append("resident_checkpoint_mismatch")
    if resident.get("checkpoint_fingerprint_method") != "sha256":
        reasons.append("resident_checkpoint_not_fully_hashed")
    if type(resident.get("checkpoint_file_count")) is not int or resident[
        "checkpoint_file_count"
    ] <= 0:
        reasons.append("resident_checkpoint_file_count_missing")
    if resident.get("latent_cortex_architecture_sha256") != prereg.get(
        "architecture_freeze_sha256"
    ):
        reasons.append("resident_architecture_freeze_mismatch")
    if not str(resident.get("worker_boot_id") or ""):
        reasons.append("missing_resident_worker_boot_id")
    if resident.get("loaded_in_installed_app") is not True:
        reasons.append("installed_app_residency_unproven")
    if not str(resident.get("installed_app_bundle_id") or ""):
        reasons.append("installed_app_bundle_id_missing")
    if not _is_sha256(resident.get("installed_app_build_sha256")):
        reasons.append("installed_app_build_unproven")
    if not _is_sha256(resident.get("worker_binary_sha256")):
        reasons.append("resident_worker_binary_unproven")
    if not _is_sha256(resident.get("build_provenance_sha256")):
        reasons.append("installed_app_provenance_unproven")
    if not _is_git_sha(resident.get("source_commit")):
        reasons.append("installed_app_source_commit_invalid")
    return resident


def _trial_compute(
    trial: dict[str, Any],
    arm: str,
    expected_estimator: str,
    reasons: list[str],
) -> tuple[float | None, int | None]:
    compute = trial.get(f"{arm}_compute")
    trial_id = str(trial.get("trial_id") or "unknown")
    if not isinstance(compute, dict):
        reasons.append(f"{trial_id}:{arm}_compute_missing")
        return None, None
    flops = compute.get("estimated_flops")
    layer_apps = compute.get("layer_apps")
    if not _finite_number(flops, positive=True):
        reasons.append(f"{trial_id}:{arm}_flops_invalid")
        flops = None
    if (
        type(layer_apps) is not int
        or layer_apps <= 0
        or layer_apps > _MAX_LAYER_APPS
    ):
        reasons.append(f"{trial_id}:{arm}_layer_apps_invalid")
        layer_apps = None
    estimator = compute.get("estimator_sha256")
    if not _is_sha256(estimator):
        reasons.append(f"{trial_id}:{arm}_compute_estimator_unidentified")
    elif estimator != expected_estimator:
        reasons.append(f"{trial_id}:{arm}_compute_estimator_mismatch")
    return float(flops) if flops is not None else None, layer_apps


def _validate_treatment_receipt(
    trial: dict[str, Any],
    checkpoint: str,
    worker_boot_id: str,
    installed_app_build_sha256: str,
    reasons: list[str],
) -> str:
    trial_id = str(trial.get("trial_id") or "unknown")
    receipt = trial.get("treatment_receipt")
    if not isinstance(receipt, dict):
        reasons.append(f"{trial_id}:treatment_receipt_missing")
        return ""
    required_truths = {
        "params_unchanged": True,
        "latent_opt_applied": True,
        "fast_weights_applied": True,
        "fast_weights_erased": True,
    }
    for key, expected in required_truths.items():
        if receipt.get(key) is not expected:
            reasons.append(f"{trial_id}:{key}_unproven")
    if str(receipt.get("checkpoint_fingerprint") or "") != checkpoint:
        reasons.append(f"{trial_id}:treatment_checkpoint_mismatch")
    if receipt.get("checkpoint_fingerprint_method") != "sha256" or type(
        receipt.get("checkpoint_file_count")
    ) is not int or receipt["checkpoint_file_count"] <= 0:
        reasons.append(f"{trial_id}:treatment_checkpoint_identity_incomplete")
    if str(receipt.get("worker_boot_id") or "") != worker_boot_id:
        reasons.append(f"{trial_id}:treatment_worker_boot_mismatch")
    if receipt.get("installed_app_build_sha256") != installed_app_build_sha256:
        reasons.append(f"{trial_id}:treatment_app_build_mismatch")
    episode_id = str(receipt.get("episode_id") or "")
    if not episode_id or not _is_sha256(receipt.get("schedule_hash")):
        reasons.append(f"{trial_id}:treatment_identity_incomplete")
    if receipt.get("latent_opt_mode") != "gradient":
        reasons.append(f"{trial_id}:latent_opt_mode_not_gradient")
    for metric in (
        "n_slots",
        "n_branches",
        "steps_taken",
        "latent_opt_attempts",
        "latent_opt_steps",
        "fast_weights_layers",
        "fast_weight_optimization_attempts",
        "fast_weight_optimized_steps",
    ):
        if type(receipt.get(metric)) is not int or receipt[metric] <= 0:
            reasons.append(f"{trial_id}:treatment_{metric}_unproven")
    for metric in ("latent_opt_rejected", "fast_weight_rejected_steps"):
        if type(receipt.get(metric)) is not int or receipt[metric] < 0:
            reasons.append(f"{trial_id}:treatment_{metric}_invalid")
    if (
        type(receipt.get("latent_opt_attempts")) is int
        and type(receipt.get("latent_opt_steps")) is int
        and type(receipt.get("latent_opt_rejected")) is int
        and receipt["latent_opt_attempts"]
        != receipt["latent_opt_steps"] + receipt["latent_opt_rejected"]
    ):
        reasons.append(f"{trial_id}:latent_opt_accounting_mismatch")
    if (
        type(receipt.get("fast_weight_optimization_attempts")) is int
        and type(receipt.get("fast_weight_optimized_steps")) is int
        and type(receipt.get("fast_weight_rejected_steps")) is int
        and receipt["fast_weight_optimization_attempts"]
        != receipt["fast_weight_optimized_steps"]
        + receipt["fast_weight_rejected_steps"]
    ):
        reasons.append(f"{trial_id}:fast_weight_accounting_mismatch")
    if receipt.get("latent_opt_budget_exhausted") is not False:
        reasons.append(f"{trial_id}:latent_opt_budget_exhausted")
    if receipt.get("fast_weight_budget_exhausted") is not False:
        reasons.append(f"{trial_id}:fast_weight_budget_exhausted")
    raw_flags = receipt.get("honest_flags")
    if not isinstance(raw_flags, list):
        reasons.append(f"{trial_id}:treatment_honest_flags_invalid")
        flags: list[str] = []
    else:
        flags = [str(flag) for flag in raw_flags]
    if flags:
        reasons.append(f"{trial_id}:treatment_honest_flags_present")
    return episode_id


def _validate_control_receipt(
    trial: dict[str, Any],
    prereg: dict[str, Any],
    worker_boot_id: str,
    installed_app_build_sha256: str,
    reasons: list[str],
) -> str:
    trial_id = str(trial.get("trial_id") or "unknown")
    receipt = trial.get("control_receipt")
    if not isinstance(receipt, dict):
        reasons.append(f"{trial_id}:control_receipt_missing")
        return ""
    request_id = str(receipt.get("request_id") or "")
    if not request_id:
        reasons.append(f"{trial_id}:control_request_id_missing")
    comparison_kind = prereg.get("comparison_kind")
    if comparison_kind == "resident_32b_vs_vanilla_same_checkpoint":
        if receipt.get("mode") != "vanilla" or receipt.get("latent_cortex_enabled") is not False:
            reasons.append(f"{trial_id}:control_not_vanilla")
        if receipt.get("params_unchanged") is not True:
            reasons.append(f"{trial_id}:control_params_unchanged_unproven")
        if str(receipt.get("checkpoint_fingerprint") or "") != str(
            prereg.get("control_checkpoint_fingerprint") or ""
        ):
            reasons.append(f"{trial_id}:control_checkpoint_mismatch")
        if str(receipt.get("worker_boot_id") or "") != worker_boot_id:
            reasons.append(f"{trial_id}:control_worker_boot_mismatch")
        if receipt.get("checkpoint_fingerprint_method") != "sha256" or type(
            receipt.get("checkpoint_file_count")
        ) is not int or receipt["checkpoint_file_count"] <= 0:
            reasons.append(f"{trial_id}:control_checkpoint_identity_incomplete")
        if receipt.get("installed_app_build_sha256") != installed_app_build_sha256:
            reasons.append(f"{trial_id}:control_app_build_mismatch")
    elif comparison_kind == "resident_32b_vs_external_frontier":
        if receipt.get("model_id") != prereg.get("control_model_id"):
            reasons.append(f"{trial_id}:external_control_model_mismatch")
        if receipt.get("model_build_fingerprint") != prereg.get(
            "control_model_build_fingerprint"
        ):
            reasons.append(f"{trial_id}:external_control_build_mismatch")
        if receipt.get("provider") != prereg.get("control_provider"):
            reasons.append(f"{trial_id}:external_control_provider_mismatch")
        if not _is_sha256(receipt.get("provider_receipt_sha256")):
            reasons.append(f"{trial_id}:external_control_receipt_unproven")
    return request_id


def _validate_trial_lineage(trial: dict[str, Any], reasons: list[str]) -> None:
    trial_id = str(trial.get("trial_id") or "unknown")
    for field in (
        "task_payload_sha256",
        "treatment_output_sha256",
        "control_output_sha256",
        "scorer_config_sha256",
        "verifier_receipt_sha256",
    ):
        if not _is_sha256(trial.get(field)):
            reasons.append(f"{trial_id}:{field}_invalid")


def _task_manifest_sha256(trials: list[Any]) -> str:
    rows: list[dict[str, Any]] = []
    for trial in trials:
        if not isinstance(trial, dict):
            raise ValueError("task manifest contains a non-mapping trial")
        rows.append(
            {
                "trial_id": trial.get("trial_id"),
                "task_id": trial.get("task_id"),
                "domain": trial.get("domain"),
                "task_payload_sha256": trial.get("task_payload_sha256"),
                "task_generated_at": trial.get("task_generated_at"),
            }
        )
    rows.sort(key=lambda row: str(row["trial_id"] or ""))
    return canonical_sha256(rows)


def _validate_task_commitment(
    bundle: dict[str, Any],
    prereg: dict[str, Any],
    trials: list[Any],
    *,
    latest_task_generated_at: float,
    earliest_evaluation_started_at: float,
    trusted_task_issuers: Mapping[str, Mapping[str, str]] | None,
    reasons: list[str],
) -> tuple[str, str]:
    raw = bundle.get("task_commitment")
    producer_id = str(bundle.get("producer_id") or "")
    signer = raw.get("signer") if isinstance(raw, dict) else None
    signer_id = str(signer.get("signer_id") or "") if isinstance(signer, dict) else ""
    try:
        pin = (
            trusted_task_issuers.get(signer_id)
            if trusted_task_issuers is not None
            else None
        )
    except (AttributeError, TypeError, ValueError):
        pin = None
    if not isinstance(pin, Mapping) or set(pin) != {
        "public_key_b64",
        "implementation_sha256",
        "release_sha256",
    }:
        reasons.append("task_issuer_trust_pin_missing")
        return "", ""
    public_key = pin.get("public_key_b64")
    implementation_sha256 = pin.get("implementation_sha256")
    release_sha256 = pin.get("release_sha256")
    if (
        not isinstance(public_key, str)
        or not _is_sha256(implementation_sha256)
        or not _is_sha256(release_sha256)
    ):
        reasons.append("task_issuer_trust_pin_invalid")
        return "", ""
    try:
        envelope, payload, verified_signer_id = verify_signed_envelope(
            raw,
            schema=TASK_COMMITMENT_ATTESTATION_SCHEMA,
            trusted_keys={signer_id: public_key},
            role="latent cortex task commitment issuer",
        )
        manifest_sha256 = _task_manifest_sha256(trials)
        expected_commitment = canonical_sha256(
            {
                "architecture_freeze_sha256": prereg.get(
                    "architecture_freeze_sha256"
                ),
                "preregistration_sha256": bundle.get("preregistration_sha256"),
                "task_count": len(trials),
                "task_manifest_sha256": manifest_sha256,
            }
        )
    except (TypeError, ValueError, OverflowError, RecursionError):
        reasons.append("task_commitment_signature_or_manifest_invalid")
        return "", ""
    expected_fields = {
        "architecture_freeze_sha256",
        "preregistration_sha256",
        "task_commitment_sha256",
        "task_manifest_sha256",
        "task_count",
        "issuer_implementation_sha256",
        "issuer_release_sha256",
        "committed_at",
    }
    committed_at = payload.get("committed_at")
    invalid = (
        set(payload) != expected_fields
        or verified_signer_id == producer_id
        or payload.get("architecture_freeze_sha256")
        != prereg.get("architecture_freeze_sha256")
        or payload.get("preregistration_sha256")
        != bundle.get("preregistration_sha256")
        or payload.get("task_manifest_sha256") != manifest_sha256
        or payload.get("task_commitment_sha256") != expected_commitment
        or bundle.get("task_commitment_sha256") != expected_commitment
        or payload.get("task_count") != len(trials)
        or payload.get("issuer_implementation_sha256") != implementation_sha256
        or payload.get("issuer_release_sha256") != release_sha256
        or not _finite_number(committed_at, positive=True)
        or float(committed_at or 0.0) <= latest_task_generated_at
        or not math.isfinite(earliest_evaluation_started_at)
        or float(committed_at or 0.0) >= earliest_evaluation_started_at
    )
    if invalid:
        reasons.append("task_commitment_invalid")
        return verified_signer_id, ""
    try:
        return verified_signer_id, canonical_sha256(envelope)
    except (TypeError, ValueError, OverflowError, RecursionError):
        reasons.append("task_commitment_not_canonical_json")
        return verified_signer_id, ""


def _validate_independent_attestation(
    bundle: dict[str, Any],
    *,
    evidence_hash: str,
    latest_task_generated_at: float,
    trusted_verifiers: Mapping[str, Mapping[str, str]] | None,
    reasons: list[str],
) -> tuple[str, str]:
    raw = bundle.get("independent_verifier")
    producer_id = str(bundle.get("producer_id") or "")
    if not producer_id:
        reasons.append("producer_id_missing")
    signer = raw.get("signer") if isinstance(raw, dict) else None
    signer_id = str(signer.get("signer_id") or "") if isinstance(signer, dict) else ""
    try:
        pin = trusted_verifiers.get(signer_id) if trusted_verifiers is not None else None
    except (AttributeError, TypeError, ValueError):
        pin = None
    if not isinstance(pin, Mapping) or set(pin) != {
        "public_key_b64",
        "implementation_sha256",
        "release_sha256",
    }:
        reasons.append("independent_verifier_trust_pin_missing")
        return "", ""
    public_key = pin.get("public_key_b64")
    implementation_sha256 = pin.get("implementation_sha256")
    release_sha256 = pin.get("release_sha256")
    if (
        not isinstance(public_key, str)
        or not _is_sha256(implementation_sha256)
        or not _is_sha256(release_sha256)
    ):
        reasons.append("independent_verifier_trust_pin_invalid")
        return "", ""
    try:
        envelope, payload, verified_signer_id = verify_signed_envelope(
            raw,
            schema=INDEPENDENT_ATTESTATION_SCHEMA,
            trusted_keys={signer_id: public_key},
            role="latent cortex independent verifier",
        )
    except (TypeError, ValueError, OverflowError, RecursionError):
        reasons.append("independent_verifier_signature_invalid")
        return "", ""
    expected_fields = {
        "accepted",
        "claim_tier",
        "producer_id",
        "evidence_payload_sha256",
        "preregistration_sha256",
        "implementation_sha256",
        "verifier_release_sha256",
        "raw_artifact_manifest_sha256",
        "task_commitment_sha256",
        "verified_at",
    }
    invalid = (
        set(payload) != expected_fields
        or payload.get("accepted") is not True
        or payload.get("claim_tier") != PROVEN
        or verified_signer_id == producer_id
        or payload.get("producer_id") != producer_id
        or payload.get("evidence_payload_sha256") != evidence_hash
        or payload.get("preregistration_sha256")
        != bundle.get("preregistration_sha256")
        or payload.get("implementation_sha256") != implementation_sha256
        or payload.get("verifier_release_sha256") != release_sha256
        or payload.get("raw_artifact_manifest_sha256")
        != bundle.get("raw_artifact_manifest_sha256")
        or payload.get("task_commitment_sha256")
        != bundle.get("task_commitment_sha256")
        or not _finite_number(payload.get("verified_at"), positive=True)
        or float(payload.get("verified_at") or 0.0) <= latest_task_generated_at
    )
    if invalid:
        reasons.append("independent_verification_invalid")
        return verified_signer_id, ""
    try:
        return verified_signer_id, canonical_sha256(envelope)
    except (TypeError, ValueError, OverflowError, RecursionError):
        reasons.append("independent_attestation_not_canonical_json")
        return verified_signer_id, ""


def verify_frontier_gain_bundle(
    bundle: Any,
    *,
    trusted_verifiers: Mapping[str, Mapping[str, str]] | None = None,
    trusted_task_issuers: Mapping[str, Mapping[str, str]] | None = None,
) -> dict[str, Any]:
    """Return a deterministic certificate; never infer missing evidence."""
    reasons: list[str] = []
    if not isinstance(bundle, dict):
        bundle = {}
        reasons.append("bundle_not_mapping")
    if bundle.get("schema") != SCHEMA:
        reasons.append("schema_mismatch")
    prereg = _validate_preregistration(bundle.get("preregistration"), reasons)
    try:
        expected_prereg_hash = canonical_sha256(prereg) if prereg else ""
    except (TypeError, ValueError, OverflowError, RecursionError):
        expected_prereg_hash = ""
        reasons.append("preregistration_not_canonical_json")
    if str(bundle.get("preregistration_sha256") or "") != expected_prereg_hash:
        reasons.append("preregistration_hash_mismatch")
    resident = _validate_resident_model(bundle.get("resident_model"), prereg, reasons)
    checkpoint = str(resident.get("checkpoint_fingerprint") or "")
    worker_boot_id = str(resident.get("worker_boot_id") or "")
    installed_app_build_sha256 = str(
        resident.get("installed_app_build_sha256") or ""
    )
    expected_estimator = str(prereg.get("compute_estimator_sha256") or "")
    if not _is_sha256(bundle.get("task_commitment_sha256")):
        reasons.append("task_commitment_missing")
    if not _is_sha256(bundle.get("raw_artifact_manifest_sha256")):
        reasons.append("raw_artifact_manifest_missing")

    trials = bundle.get("trials")
    if not isinstance(trials, list) or not trials:
        reasons.append("trials_missing")
        trials = []
    raw_domains = prereg.get("domains")
    domains = (
        [
            domain
            for domain in raw_domains
            if isinstance(domain, str) and domain.strip()
        ]
        if isinstance(raw_domains, list)
        else []
    )
    raw_min_trials = prereg.get("min_trials_per_domain")
    min_trials = raw_min_trials if type(raw_min_trials) is int else 30
    paired: dict[str, list[PairedObservation]] = {domain: [] for domain in domains}
    seen_trial_ids: set[str] = set()
    seen_task_ids: set[str] = set()
    seen_task_payloads: set[str] = set()
    seen_episode_ids: set[str] = set()
    seen_control_request_ids: set[str] = set()
    order_counts = {"treatment_first": 0, "control_first": 0}
    frozen_at = (
        float(prereg["frozen_at"])
        if _finite_number(prereg.get("frozen_at"), positive=True)
        else 0.0
    )
    latest_task_generated_at = 0.0
    earliest_evaluation_started_at = float("inf")
    latest_evaluation_started_at = 0.0
    tolerance = (
        float(prereg["compute_tolerance"])
        if _finite_number(prereg.get("compute_tolerance"))
        else 0.0
    )
    for trial in trials:
        if not isinstance(trial, dict):
            reasons.append("trial_not_mapping")
            continue
        trial_id = str(trial.get("trial_id") or "")
        task_id = str(trial.get("task_id") or "")
        domain = str(trial.get("domain") or "")
        if not trial_id or trial_id in seen_trial_ids:
            reasons.append("duplicate_or_missing_trial_id")
            continue
        if not task_id or task_id in seen_task_ids:
            reasons.append(f"{trial_id}:duplicate_or_missing_task_id")
            continue
        seen_trial_ids.add(trial_id)
        seen_task_ids.add(task_id)
        if domain not in paired:
            reasons.append(f"{trial_id}:unregistered_domain")
            continue
        if trial.get("held_out") is not True or trial.get("contamination_scan_passed") is not True:
            reasons.append(f"{trial_id}:heldout_or_contamination_unproven")
        if not _finite_number(trial.get("task_generated_at"), positive=True) or float(
            trial.get("task_generated_at") or 0.0
        ) <= frozen_at:
            reasons.append(f"{trial_id}:task_not_generated_after_freeze")
        evaluation_started_at = trial.get("evaluation_started_at")
        if not _finite_number(evaluation_started_at, positive=True):
            reasons.append(f"{trial_id}:evaluation_start_invalid")
        else:
            evaluation_time = float(evaluation_started_at)
            earliest_evaluation_started_at = min(
                earliest_evaluation_started_at,
                evaluation_time,
            )
            latest_evaluation_started_at = max(
                latest_evaluation_started_at,
                evaluation_time,
            )
        _validate_trial_lineage(trial, reasons)
        task_payload_hash = str(trial.get("task_payload_sha256") or "")
        if task_payload_hash in seen_task_payloads:
            reasons.append(f"{trial_id}:duplicate_task_payload")
        seen_task_payloads.add(task_payload_hash)
        if trial.get("verifier_blinded") is not True:
            reasons.append(f"{trial_id}:blinded_verifier_unproven")
        treatment_information = trial.get("treatment_information_sha256")
        control_information = trial.get("control_information_sha256")
        if (
            not _is_sha256(treatment_information)
            or treatment_information != control_information
        ):
            reasons.append(f"{trial_id}:information_mismatch")
        if str(trial.get("treatment_tool_policy_sha256") or "") != str(
            trial.get("control_tool_policy_sha256") or ""
        ) or str(trial.get("treatment_tool_policy_sha256") or "") != str(
            prereg.get("tool_policy_sha256") or ""
        ):
            reasons.append(f"{trial_id}:tool_policy_mismatch")
        order = str(trial.get("run_order") or "")
        if order not in order_counts:
            reasons.append(f"{trial_id}:invalid_run_order")
        else:
            order_counts[order] += 1
        if type(trial.get("treatment_success")) is not bool or type(
            trial.get("control_success")
        ) is not bool:
            reasons.append(f"{trial_id}:non_boolean_outcome")
            continue
        treatment_flops, treatment_layers = _trial_compute(
            trial, "treatment", expected_estimator, reasons
        )
        control_flops, control_layers = _trial_compute(
            trial, "control", expected_estimator, reasons
        )
        if treatment_flops is not None and control_flops is not None:
            mismatch = abs(treatment_flops - control_flops) / max(1.0, control_flops)
            if mismatch > tolerance:
                reasons.append(f"{trial_id}:compute_mismatch")
        episode_id = _validate_treatment_receipt(
            trial,
            checkpoint,
            worker_boot_id,
            installed_app_build_sha256,
            reasons,
        )
        if episode_id in seen_episode_ids:
            reasons.append(f"{trial_id}:duplicate_treatment_episode")
        seen_episode_ids.add(episode_id)
        control_request_id = _validate_control_receipt(
            trial,
            prereg,
            worker_boot_id,
            installed_app_build_sha256,
            reasons,
        )
        if control_request_id in seen_control_request_ids:
            reasons.append(f"{trial_id}:duplicate_control_request")
        seen_control_request_ids.add(control_request_id)
        if _finite_number(trial.get("task_generated_at"), positive=True):
            latest_task_generated_at = max(
                latest_task_generated_at, float(trial["task_generated_at"])
            )
        paired[domain].append(
            PairedObservation(
                task_id=task_id,
                family=domain,
                treatment_success=trial["treatment_success"],
                control_success=trial["control_success"],
                treatment_layer_apps=treatment_layers,
                control_layer_apps=control_layers,
            )
        )

    for domain, observations in paired.items():
        if len(observations) < min_trials:
            reasons.append(f"{domain}:underpowered")
    if trials and abs(order_counts["treatment_first"] - order_counts["control_first"]) > max(
        1, math.ceil(len(trials) * 0.10)
    ):
        reasons.append("run_order_imbalanced")

    task_issuer_id, task_commitment_attestation_sha256 = _validate_task_commitment(
        bundle,
        prereg,
        trials,
        latest_task_generated_at=latest_task_generated_at,
        earliest_evaluation_started_at=earliest_evaluation_started_at,
        trusted_task_issuers=trusted_task_issuers,
        reasons=reasons,
    )

    comparison_kind = str(prereg.get("comparison_kind") or "")
    try:
        evidence_hash = evidence_payload_sha256(bundle)
    except (TypeError, ValueError, OverflowError, RecursionError):
        evidence_hash = ""
        reasons.append("evidence_payload_not_canonical_json")
    independent_verifier_id, independent_attestation_sha256 = (
        _validate_independent_attestation(
            bundle,
            evidence_hash=evidence_hash,
            latest_task_generated_at=max(
                latest_task_generated_at,
                latest_evaluation_started_at,
            ),
            trusted_verifiers=trusted_verifiers,
            reasons=reasons,
        )
    )
    if task_issuer_id and task_issuer_id == independent_verifier_id:
        reasons.append("independent_roles_reused")

    raw_alpha = prereg.get("alpha")
    alpha = (
        float(raw_alpha)
        if _finite_number(raw_alpha) and 0.0 < float(raw_alpha) <= 0.05
        else 0.05
    )
    raw_minimum_effect = prereg.get("minimum_effect")
    minimum_effect = (
        float(raw_minimum_effect)
        if _finite_number(raw_minimum_effect)
        and 0.0 < float(raw_minimum_effect) <= 0.5
        else 0.05
    )
    claim = grade_paired_treatment_vs_control(
        "exp6_frontier_comparison",
        "the full resident-32B Recursive Latent Cortex improves over its preregistered control",
        paired,
        alpha=alpha,
        minimum_effect=minimum_effect,
        compute_tolerance=1.0,
        require_compute=False,
    )
    positive = claim.evidence.get("positive_families") or []
    required_positive = math.ceil(len(domains) * 2 / 3) if domains else 1
    if len(positive) < required_positive:
        reasons.append("insufficient_positive_domains")
    if claim.evidence.get("regressed_families"):
        reasons.append("domain_regression_detected")
    if claim.tier != PROVEN:
        reasons.append("paired_gain_not_proven")

    accepted = not reasons
    statistical_claim = claim.to_dict()
    # Wall-clock grading time is not evidence and would make identical bundles
    # produce different certificates. The evidence hash binds the actual input.
    statistical_claim.pop("graded_at", None)
    certificate: dict[str, Any] = {
        "schema": CERTIFICATE_SCHEMA,
        "accepted": accepted,
        "claim_tier": PROVEN if accepted else CONJECTURE,
        "comparison_kind": comparison_kind,
        "evidence_payload_sha256": evidence_hash,
        "independent_verifier_id": independent_verifier_id,
        "independent_attestation_sha256": independent_attestation_sha256,
        "task_issuer_id": task_issuer_id,
        "task_commitment_attestation_sha256": task_commitment_attestation_sha256,
        "preregistration_sha256": expected_prereg_hash,
        "trial_count": len(trials),
        "domain_counts": {domain: len(items) for domain, items in paired.items()},
        "required_positive_domains": required_positive,
        "reasons": sorted(set(reasons)),
        "statistical_claim": statistical_claim,
    }
    certificate["certificate_sha256"] = canonical_sha256(certificate)
    return certificate


__all__ = [
    "CERTIFICATE_SCHEMA",
    "INDEPENDENT_ATTESTATION_SCHEMA",
    "SCHEMA",
    "TASK_COMMITMENT_ATTESTATION_SCHEMA",
    "canonical_sha256",
    "evidence_payload_sha256",
    "verify_frontier_gain_bundle",
]
