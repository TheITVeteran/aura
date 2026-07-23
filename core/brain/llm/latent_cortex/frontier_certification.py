"""Machine-checkable release gate for Recursive Latent Cortex frontier gains.

Unit tests can prove this verifier rejects weak evidence. Only a completed
resident-32B bundle can prove the capability claim itself.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from fractions import Fraction
from typing import Any

from core.brain.frontier_evidence_v5 import verify_signed_envelope
from core.brain.llm.latent_cortex.experiments import (
    CONJECTURE,
    PROVEN,
    PairedObservation,
    grade_paired_treatment_vs_control,
)
from core.brain.llm.latent_cortex.resource_accounting import (
    certify_comparison_accounting,
    validate_information_receipt,
    validate_resource_receipt,
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
    # The scoring PROGRAM must be frozen with everything else. Only a
    # per-trial scorer *config* (which legitimately embeds the trial id)
    # appeared in lineage, so the scoring rules themselves could be chosen
    # after outputs were known.
    if not _is_sha256(prereg.get("scorer_implementation_sha256")):
        reasons.append("missing_scorer_implementation_sha256")
    # Decoding is an experimental condition, not an implementation detail:
    # unpaired seeds/sampling let outcome differences come from decoding
    # rather than from the treatment.
    if not _is_sha256(prereg.get("decode_policy_sha256")):
        reasons.append("missing_decode_policy_sha256")
    # An ABSOLUTE capability floor: without one, a very weak treatment earns
    # a certificate purely by beating an even weaker control.
    floor = prereg.get("min_treatment_success_rate")
    if not _finite_number(floor) or not 0.0 < float(floor) <= 1.0:
        reasons.append("invalid_min_treatment_success_rate")
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
    # The class claim must be DERIVED from the hashed checkpoint, not merely
    # asserted alongside it: a bare integer in range is something any
    # producer can write. The parameter count is required to come from the
    # same manifest whose fingerprint is bound above, and the manifest's own
    # digest must match the declared checkpoint fingerprint.
    if resident.get("parameter_count_source") != "checkpoint_manifest":
        reasons.append("resident_parameter_count_not_manifest_derived")
    # CP126 8923b135. A count inside the 30-40B window, even a
    # manifest-derived one, does not establish WHICH 32B-class model this is:
    # quantization layout and architecture change what the weights compute,
    # and two runtimes can share a parameter count while serving materially
    # different functions. The class claim must carry that identity.
    for required_field in ("architecture", "quantization_bits", "quantization_group_size"):
        if not resident.get(required_field):
            reasons.append(f"resident_identity_incomplete:{required_field}")
    manifest_digest = str(resident.get("checkpoint_manifest_sha256") or "")
    if not _is_sha256(manifest_digest):
        reasons.append("resident_checkpoint_manifest_missing")
    elif manifest_digest != str(resident.get("checkpoint_fingerprint") or ""):
        reasons.append("resident_checkpoint_manifest_unbound")
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
) -> tuple[float | None, int | None, dict[str, Any] | None]:
    compute = trial.get(f"{arm}_compute")
    trial_id = str(trial.get("trial_id") or "unknown")
    if not isinstance(compute, dict):
        reasons.append(f"{trial_id}:{arm}_compute_missing")
        return None, None, None
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
    resource: dict[str, Any] | None = None
    try:
        resource = validate_resource_receipt(compute.get("resource_accounting"))
    except (TypeError, ValueError):
        reasons.append(f"{trial_id}:{arm}_resource_accounting_invalid")
    if resource is not None:
        if resource["accounting_complete"] is not True:
            reasons.append(f"{trial_id}:{arm}_resource_accounting_incomplete")
        if flops is not None and float(resource["estimated_flops"]) != float(flops):
            reasons.append(f"{trial_id}:{arm}_compute_receipt_mismatch")
    return float(flops) if flops is not None else None, layer_apps, resource


def _trial_information(
    trial: dict[str, Any],
    arm: str,
    reasons: list[str],
) -> dict[str, Any] | None:
    trial_id = str(trial.get("trial_id") or "unknown")
    try:
        receipt = validate_information_receipt(trial.get(f"{arm}_information"))
    except (TypeError, ValueError):
        reasons.append(f"{trial_id}:{arm}_information_accounting_invalid")
        return None
    if receipt["accounting_complete"] is not True:
        reasons.append(f"{trial_id}:{arm}_information_accounting_incomplete")
    task_payload_sha256 = trial.get("task_payload_sha256")
    if not any(
        source.get("content_sha256") == task_payload_sha256
        and source.get("kind") == "task_prompt"
        for source in receipt["sources"]
    ):
        reasons.append(f"{trial_id}:{arm}_task_information_unbound")
    if trial.get(f"{arm}_information_sha256") != receipt["receipt_sha256"]:
        reasons.append(f"{trial_id}:{arm}_information_receipt_mismatch")
        return None
    return receipt


def _validate_treatment_receipt(
    trial: dict[str, Any],
    checkpoint: str,
    worker_boot_id: str,
    installed_app_build_sha256: str,
    reasons: list[str],
    unproven_integrity_claims: list[str] | None = None,
) -> str:
    trial_id = str(trial.get("trial_id") or "unknown")
    if unproven_integrity_claims is None:
        unproven_integrity_claims = []
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
    # CP126 6090a5ae + 01e8b3c1. The control receipt already required DIGEST
    # evidence for params_unchanged; the treatment receipt — the arm whose
    # effect is being published — passed on the literal booleans above alone.
    #
    # A digest that REFUTES the claim is disqualifying: the boolean says the
    # weights were untouched / the fast weights erased, and the measurement
    # says otherwise. An ABSENT digest is a gap in the evidence chain, not a
    # refutation; producers do not emit treatment-side weight_integrity yet, so
    # rejecting on absence would make certification unachievable rather than
    # more honest. Absence is surfaced on the certificate instead of counting
    # as proof.
    for claim in ("params_unchanged", "fast_weights_erased"):
        verdict = _receipt_integrity_verdict(receipt, claim)
        if verdict == "refuted":
            reasons.append(f"{trial_id}:treatment_{claim}_refuted_by_digest")
        elif verdict != "proven":
            unproven_integrity_claims.append(f"treatment_{claim}")
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


# Components that change what a "vanilla" control actually computes. Each
# must be positively OFF in a control receipt: an absent field is treated as
# undeclared, not as disabled, because a control that never reported its
# configuration cannot be shown to have been unenhanced.
_CONTROL_MUST_BE_DISABLED: tuple[tuple[str, Any], ...] = (
    ("fast_weights_applied", False),
    ("latent_opt_applied", False),
    ("recurrence_adapter_applied", False),
    ("retrieval_applied", False),
    ("nonparametric_memory_applied", False),
    ("contrastive_decoding_applied", False),
    ("speculative_decoding_applied", False),
    ("expert_adapter_applied", False),
    ("affective_steering_active", False),
    ("prompt_cache_reused", False),
)

# Decoding parameters that must match the preregistered control spec. A
# control run hotter, wider, or with a different repetition penalty than the
# treatment is not a control of the same thing.
_CONTROL_DECODE_PARAMETERS: tuple[str, ...] = (
    "decode_temperature",
    "decode_top_p",
    "decode_repetition_penalty_applied",
)


def _receipt_integrity_verdict(receipt: Any, claim: str) -> str:
    """Verdict from the receipt's digest evidence; "unproven" when absent."""
    if not isinstance(receipt, dict):
        return "unproven"
    verdicts = receipt.get("integrity_verdicts")
    if isinstance(verdicts, dict):
        entry = verdicts.get(claim)
        if isinstance(entry, dict):
            verdict = str(entry.get("verdict") or "")
            if verdict in {"proven", "refuted", "unproven"}:
                return verdict
    try:
        from core.brain.llm.latent_cortex.types import WeightIntegrityProof

        proof = WeightIntegrityProof.from_dict(receipt.get("weight_integrity"))
    except (ImportError, AttributeError, TypeError, ValueError):
        return "unproven"
    proven = (
        proof.params_unchanged_proven
        if claim == "params_unchanged"
        else proof.fast_weights_erased_proven
    )
    if proven is None:
        return "unproven"
    return "proven" if proven else "refuted"


#: Everything that can make two arms decode differently. CP126 8a56c486: the
#: arm-parity checks covered information hashes, tool policy, compute and run
#: order but NOT the generation manifest, so an outcome difference could come
#: from sampling rather than from the latent-cortex treatment.
_ARM_GENERATION_PARITY_FIELDS: tuple[str, ...] = (
    "decode_seed",
    "decode_temperature",
    "decode_top_p",
    "decode_repetition_penalty_applied",
    "decode_max_tokens",
    "decode_stop_sequences_sha256",
    "context_window_tokens",
    "tokenizer_sha256",
    "prompt_template_sha256",
)


def _validate_arm_generation_parity(
    trial_id: str,
    treatment_receipt: Any,
    control_receipt: Any,
    reasons: list[str],
) -> list[str]:
    """Both arms must have decoded the same way.

    A DECLARED difference is disqualifying: the trial's outcome gap can come
    from sampling rather than from the treatment, so the trial cannot feed the
    paired claim.

    A field neither arm declares is a real gap in the evidence chain, but the
    producers do not emit the full generation manifest yet. Rejecting on it
    would make certification permanently unachievable rather than more honest,
    so undeclared fields are RETURNED as certificate-visible gaps (the claim
    carries what it could not verify) instead of silently passing as equal.
    Closing them for real needs the worker-side manifest emission.
    """
    gaps: list[str] = []
    if not isinstance(treatment_receipt, dict) or not isinstance(control_receipt, dict):
        reasons.append(f"{trial_id}:generation_parity_receipts_missing")
        return gaps
    for field_name in _ARM_GENERATION_PARITY_FIELDS:
        in_treatment = field_name in treatment_receipt
        in_control = field_name in control_receipt
        if not in_treatment and not in_control:
            gaps.append(field_name)
            continue
        if in_treatment != in_control:
            # Only one arm declares it. That is an incomplete manifest, not
            # proof the arms diverged — the control-side decode manifest
            # landed before the treatment-side one exists. Report it.
            gaps.append(field_name)
            continue
        if treatment_receipt.get(field_name) != control_receipt.get(field_name):
            reasons.append(f"{trial_id}:generation_parity_mismatch:{field_name}")
    return gaps


def _validate_vanilla_control_manifest(
    trial_id: str,
    receipt: dict[str, Any],
    prereg: dict[str, Any],
    reasons: list[str],
) -> None:
    """Every enhancement is off, declared, and decoding matches the spec."""
    for field_name, required in _CONTROL_MUST_BE_DISABLED:
        if field_name not in receipt:
            reasons.append(f"{trial_id}:control_component_undeclared:{field_name}")
            continue
        if receipt.get(field_name) is not required:
            reasons.append(f"{trial_id}:control_component_enabled:{field_name}")

    # Counters betray activity a boolean might not: a control that took
    # optimization steps or wrapped layers was not vanilla whatever it
    # claims.
    for counter in (
        "fast_weights_layers",
        "fast_weight_optimization_attempts",
        "latent_opt_attempts",
        "steps_taken",
    ):
        value = receipt.get(counter)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            reasons.append(f"{trial_id}:control_activity_observed:{counter}")

    declared = prereg.get("control_decode_spec")
    if not isinstance(declared, dict):
        reasons.append(f"{trial_id}:control_decode_spec_missing")
        return
    for parameter in _CONTROL_DECODE_PARAMETERS:
        if parameter not in declared:
            reasons.append(f"{trial_id}:control_decode_spec_incomplete:{parameter}")
            continue
        if parameter not in receipt:
            reasons.append(f"{trial_id}:control_decode_undeclared:{parameter}")
            continue
        expected = declared.get(parameter)
        actual = receipt.get(parameter)
        if not _numbers_match(expected, actual):
            reasons.append(f"{trial_id}:control_decode_mismatch:{parameter}")


def _numbers_match(expected: Any, actual: Any) -> bool:
    """Exact for non-numerics; tolerant of float representation otherwise."""
    if isinstance(expected, bool) or isinstance(actual, bool):
        return expected is actual
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return abs(float(expected) - float(actual)) <= 1e-9
    return expected == actual


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
        # A published capability claim is exactly where an asserted-but-
        # unmeasured integrity boolean must not pass: certification requires
        # the digest evidence, not the claim.
        if _receipt_integrity_verdict(receipt, "params_unchanged") != "proven":
            reasons.append(f"{trial_id}:control_params_unchanged_unproven")
        # CP126 869a0ce4. "Vanilla" was three fields: mode, the latent-cortex
        # switch, and params_unchanged. Nothing prohibited fast weights,
        # retrieval, extra reasoning passes, altered decoding, or a warm
        # cache carrying state from an earlier turn — so a control could be
        # materially enhanced and still certify as the baseline the
        # treatment is measured against. Every enhancement the runtime can
        # apply must be positively absent, not merely unmentioned.
        _validate_vanilla_control_manifest(trial_id, receipt, prereg, reasons)
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


def _validate_raw_artifact_receipt(
    receipt: Any,
    bundle: Mapping[str, Any],
    reasons: list[str],
) -> bool:
    """Require a real receipt from ``verify_raw_artifact_package``.

    Certification previously accepted a bundle whose raw artifacts were never
    opened: ``raw_artifact_manifest_sha256`` was checked for SHA-256 SYNTAX
    only. A syntactically valid but entirely fabricated hash therefore
    reached ``accepted=True`` / ``claim_tier=PROVEN``. The receipt must be
    produced by the artifact verifier, report acceptance, and bind the same
    manifest digest this bundle declares.
    """
    # Lazy import: frontier_artifacts imports canonical_sha256 from this
    # module, so a module-level import would be circular. Importing here
    # keeps ONE source of truth for the schema constant.
    from core.brain.llm.latent_cortex.frontier_artifacts import (
        ARTIFACT_VERIFICATION_SCHEMA,
    )

    if not isinstance(receipt, Mapping):
        reasons.append("raw_artifact_package_unverified")
        return False
    if receipt.get("schema") != ARTIFACT_VERIFICATION_SCHEMA:
        reasons.append("raw_artifact_receipt_schema_mismatch")
        return False
    if receipt.get("accepted") is not True:
        reasons.append("raw_artifact_receipt_not_accepted")
        return False
    manifest_sha256 = str(receipt.get("manifest_sha256") or "")
    if not _is_sha256(manifest_sha256) or manifest_sha256 != str(
        bundle.get("raw_artifact_manifest_sha256") or ""
    ):
        reasons.append("raw_artifact_receipt_manifest_mismatch")
        return False
    if not _is_sha256(receipt.get("receipt_sha256")):
        reasons.append("raw_artifact_receipt_unsigned")
        return False
    # The receipt must cover every trial the bundle grades on.
    trials = bundle.get("trials")
    declared_trials = len(trials) if isinstance(trials, list) else 0
    if type(receipt.get("trial_count")) is not int or receipt["trial_count"] != declared_trials:
        reasons.append("raw_artifact_receipt_trial_count_mismatch")
        return False
    return True


def verify_frontier_gain_bundle(
    bundle: Any,
    *,
    trusted_verifiers: Mapping[str, Mapping[str, str]] | None = None,
    trusted_task_issuers: Mapping[str, Mapping[str, str]] | None = None,
    raw_artifact_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a deterministic certificate; never infer missing evidence.

    ``raw_artifact_receipt`` must come from
    :func:`core.brain.llm.latent_cortex.frontier_artifacts.verify_raw_artifact_package`.
    Without it the bundle's raw artifacts are unverified and the certificate
    cannot be accepted.
    """
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
    # Generation-manifest fields neither arm declared (CP126 8a56c486). These
    # are surfaced on the certificate as unverified parity, not silently
    # assumed equal.
    generation_parity_gaps: set[str] = set()
    # Integrity claims asserted by a boolean but never measured by a digest
    # (CP126 6090a5ae / 01e8b3c1). A REFUTED digest rejects the trial; an
    # ABSENT one is carried here so the certificate says what it could not
    # verify instead of treating the assertion as proof.
    unproven_integrity_claims: set[str] = set()
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
    admitted_trial_count = 0
    rejected_trial_count = 0
    comparison_accounting: list[dict[str, Any]] = []
    for trial in trials:
        # ADMISSION ISOLATION: a trial's own integrity defects are collected
        # here and, if any exist, the trial is EXCLUDED from the paired
        # observations that produce the statistical claim. Previously every
        # per-trial defect only appended a reason while the trial still fed
        # the grader, so contaminated, unblinded, parity-failing, or
        # unauthenticated trials could produce a nested PROVEN result inside
        # a rejected certificate.
        trial_reasons: list[str] = []
        if not isinstance(trial, dict):
            reasons.append("trial_not_mapping")
            rejected_trial_count += 1
            continue
        trial_id = str(trial.get("trial_id") or "")
        task_id = str(trial.get("task_id") or "")
        domain = str(trial.get("domain") or "")
        if not trial_id or trial_id in seen_trial_ids:
            reasons.append("duplicate_or_missing_trial_id")
            rejected_trial_count += 1
            continue
        if not task_id or task_id in seen_task_ids:
            reasons.append(f"{trial_id}:duplicate_or_missing_task_id")
            rejected_trial_count += 1
            continue
        seen_trial_ids.add(trial_id)
        seen_task_ids.add(task_id)
        if domain not in paired:
            reasons.append(f"{trial_id}:unregistered_domain")
            rejected_trial_count += 1
            continue
        if trial.get("held_out") is not True or trial.get("contamination_scan_passed") is not True:
            trial_reasons.append(f"{trial_id}:heldout_or_contamination_unproven")
        if not _finite_number(trial.get("task_generated_at"), positive=True) or float(
            trial.get("task_generated_at") or 0.0
        ) <= frozen_at:
            trial_reasons.append(f"{trial_id}:task_not_generated_after_freeze")
        evaluation_started_at = trial.get("evaluation_started_at")
        if not _finite_number(evaluation_started_at, positive=True):
            trial_reasons.append(f"{trial_id}:evaluation_start_invalid")
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
        _validate_trial_lineage(trial, trial_reasons)
        # Each trial must have been scored by the PREREGISTERED scoring
        # program (its per-trial config may differ; the program may not).
        if str(trial.get("scorer_implementation_sha256") or "") != str(
            prereg.get("scorer_implementation_sha256") or ""
        ):
            trial_reasons.append(f"{trial_id}:scorer_not_preregistered")
        task_payload_hash = str(trial.get("task_payload_sha256") or "")
        if task_payload_hash in seen_task_payloads:
            trial_reasons.append(f"{trial_id}:duplicate_task_payload")
        seen_task_payloads.add(task_payload_hash)
        if trial.get("verifier_blinded") is not True:
            trial_reasons.append(f"{trial_id}:blinded_verifier_unproven")
        treatment_information = _trial_information(
            trial,
            "treatment",
            trial_reasons,
        )
        control_information = _trial_information(
            trial,
            "control",
            trial_reasons,
        )
        if (
            treatment_information is None
            or control_information is None
            or treatment_information["source_set_sha256"]
            != control_information["source_set_sha256"]
        ):
            trial_reasons.append(f"{trial_id}:information_mismatch")
        if str(trial.get("treatment_tool_policy_sha256") or "") != str(
            trial.get("control_tool_policy_sha256") or ""
        ) or str(trial.get("treatment_tool_policy_sha256") or "") != str(
            prereg.get("tool_policy_sha256") or ""
        ):
            trial_reasons.append(f"{trial_id}:tool_policy_mismatch")
        # DECODING PARITY: both arms must run the preregistered decode policy
        # (seed discipline, sampler, temperature, stop rules, context limit,
        # tokenizer/template). Without this, an outcome difference can come
        # from decoding rather than from the latent treatment.
        treatment_decode = str(trial.get("treatment_decode_policy_sha256") or "")
        control_decode = str(trial.get("control_decode_policy_sha256") or "")
        if (
            not _is_sha256(treatment_decode)
            or treatment_decode != control_decode
            or treatment_decode != str(prereg.get("decode_policy_sha256") or "")
        ):
            trial_reasons.append(f"{trial_id}:decode_policy_mismatch")
        order = str(trial.get("run_order") or "")
        if order not in order_counts:
            trial_reasons.append(f"{trial_id}:invalid_run_order")
        else:
            order_counts[order] += 1
        if type(trial.get("treatment_success")) is not bool or type(
            trial.get("control_success")
        ) is not bool:
            trial_reasons.append(f"{trial_id}:non_boolean_outcome")
            reasons.extend(trial_reasons)
            rejected_trial_count += 1
            continue
        treatment_flops, treatment_layers, treatment_resource = _trial_compute(
            trial, "treatment", expected_estimator, trial_reasons
        )
        control_flops, control_layers, control_resource = _trial_compute(
            trial, "control", expected_estimator, trial_reasons
        )
        if treatment_flops is not None and control_flops is not None:
            mismatch = abs(treatment_flops - control_flops) / max(1.0, control_flops)
            if mismatch > tolerance:
                trial_reasons.append(f"{trial_id}:compute_mismatch")
        # LAYER-APPLICATION PARITY: both arm layer counts were validated and
        # carried into the observation, but only FLOPs were compared — so a
        # treatment could apply radically more layer work while reporting
        # equal estimated FLOPs.
        if treatment_layers is not None and control_layers is not None:
            layer_mismatch = abs(treatment_layers - control_layers) / max(
                1, control_layers
            )
            if layer_mismatch > tolerance:
                trial_reasons.append(f"{trial_id}:layer_apps_mismatch")
        if (
            treatment_resource is not None
            and control_resource is not None
            and treatment_information is not None
            and control_information is not None
        ):
            tolerance_fraction = Fraction(str(tolerance)).limit_denominator(1_000_000)
            accounting = certify_comparison_accounting(
                treatment_resource=treatment_resource,
                control_resource=control_resource,
                treatment_information=treatment_information,
                control_information=control_information,
                tolerance_numerator=tolerance_fraction.numerator,
                tolerance_denominator=tolerance_fraction.denominator,
                require_compute_parity=True,
            )
            comparison_accounting.append(
                {"trial_id": trial_id, **accounting}
            )
            for reason in accounting["reasons"]:
                trial_reasons.append(
                    f"{trial_id}:comparison_accounting:{reason}"
                )
        trial_integrity_gaps: list[str] = []
        episode_id = _validate_treatment_receipt(
            trial,
            checkpoint,
            worker_boot_id,
            installed_app_build_sha256,
            trial_reasons,
            trial_integrity_gaps,
        )
        unproven_integrity_claims.update(trial_integrity_gaps)
        if episode_id in seen_episode_ids:
            trial_reasons.append(f"{trial_id}:duplicate_treatment_episode")
        seen_episode_ids.add(episode_id)
        control_request_id = _validate_control_receipt(
            trial,
            prereg,
            worker_boot_id,
            installed_app_build_sha256,
            trial_reasons,
        )
        if control_request_id in seen_control_request_ids:
            trial_reasons.append(f"{trial_id}:duplicate_control_request")
        seen_control_request_ids.add(control_request_id)
        # CP126 8a56c486: seeds and decoding settings were never paired, so a
        # difference in sampling could be reported as a latent-cortex effect.
        generation_parity_gaps.update(
            _validate_arm_generation_parity(
                trial_id,
                trial.get("treatment_receipt"),
                trial.get("control_receipt"),
                trial_reasons,
            )
        )
        if _finite_number(trial.get("task_generated_at"), positive=True):
            latest_task_generated_at = max(
                latest_task_generated_at, float(trial["task_generated_at"])
            )
        if trial_reasons:
            # Defective trial: its reasons surface on the certificate, but it
            # contributes NOTHING to the statistical claim.
            reasons.extend(trial_reasons)
            rejected_trial_count += 1
            continue
        admitted_trial_count += 1
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
    # CLAIM SCOPE: beating an ablated SELF establishes that the treatment
    # contributes; it does NOT establish frontier competitiveness. The two
    # comparison kinds therefore carry different statements and different
    # scopes, and only an external comparison can assert competitiveness.
    external_comparison = comparison_kind == "resident_32b_vs_external_frontier"
    if external_comparison:
        claim_scope = "frontier_competitiveness_vs_external_control"
        claim_statement = (
            "the full resident-32B Recursive Latent Cortex outperforms the "
            "preregistered external frontier control on the preregistered domains"
        )
    else:
        claim_scope = "treatment_contribution_vs_same_checkpoint_ablation"
        claim_statement = (
            "the full resident-32B Recursive Latent Cortex improves over its own "
            "preregistered same-checkpoint vanilla ablation (treatment contribution "
            "only — NOT evidence of frontier competitiveness)"
        )
    # Compute evidence must survive INTO the claim. The grader was previously
    # told to ignore it (tolerance 1.0, require_compute False), so a claim
    # could read PROVEN while carrying no compute validity at all.
    claim = grade_paired_treatment_vs_control(
        "exp6_frontier_comparison",
        claim_statement,
        paired,
        alpha=alpha,
        minimum_effect=minimum_effect,
        compute_tolerance=tolerance,
        require_compute=True,
    )
    positive = claim.evidence.get("positive_families") or []
    required_positive = math.ceil(len(domains) * 2 / 3) if domains else 1
    if len(positive) < required_positive:
        reasons.append("insufficient_positive_domains")
    if claim.evidence.get("regressed_families"):
        reasons.append("domain_regression_detected")
    if claim.tier != PROVEN:
        reasons.append("paired_gain_not_proven")
    # The certificate must name WHICH domains carry the gain: a two-thirds
    # rule lets a third of domains show no benefit, and an unqualified claim
    # would hide that heterogeneity.
    non_positive_domains = sorted(set(domains) - set(positive))

    # ABSOLUTE CAPABILITY FLOOR over admitted trials: a relative win over a
    # weaker control is not a capability certificate on its own.
    min_success_rate = (
        float(prereg["min_treatment_success_rate"])
        if _finite_number(prereg.get("min_treatment_success_rate"))
        else 1.1  # unreachable → fails closed when preregistration is invalid
    )
    admitted_observations = [obs for items in paired.values() for obs in items]
    treatment_success_rate = (
        sum(1 for obs in admitted_observations if obs.treatment_success)
        / len(admitted_observations)
        if admitted_observations
        else 0.0
    )
    if treatment_success_rate < min_success_rate:
        reasons.append("treatment_below_absolute_capability_floor")

    # RAW ARTIFACT VERIFICATION: the bundle's manifest hash was only
    # SYNTAX-checked here, so the core API could accept and claim PROVEN
    # without any raw artifact ever being loaded or recomputed. A receipt
    # from verify_raw_artifact_package (frontier_artifacts) is now required,
    # and it must bind this exact bundle.
    raw_artifact_verified = _validate_raw_artifact_receipt(
        raw_artifact_receipt, bundle, reasons
    )

    accepted = not reasons
    statistical_claim = claim.to_dict()
    # Wall-clock grading time is not evidence and would make identical bundles
    # produce different certificates. The evidence hash binds the actual input.
    statistical_claim.pop("graded_at", None)
    # A nested tier must never be readable as a standalone verdict: the claim
    # is only admissible when the whole certificate was accepted, and it is
    # only ever a claim about THIS comparison's scope.
    statistical_claim["admissible"] = accepted
    statistical_claim["claim_scope"] = claim_scope
    statistical_claim["graded_on_admitted_trials"] = admitted_trial_count
    certificate: dict[str, Any] = {
        "schema": CERTIFICATE_SCHEMA,
        "accepted": accepted,
        "claim_tier": PROVEN if accepted else CONJECTURE,
        "comparison_kind": comparison_kind,
        # Scope is part of the verdict: an accepted same-checkpoint bundle
        # proves treatment CONTRIBUTION, never frontier competitiveness.
        "claim_scope": claim_scope,
        "frontier_competitiveness_established": bool(accepted and external_comparison),
        "raw_artifact_verified": raw_artifact_verified,
        "admitted_trial_count": admitted_trial_count,
        "rejected_trial_count": rejected_trial_count,
        "positive_domains": sorted(positive),
        "non_positive_domains": non_positive_domains,
        "treatment_success_rate": round(treatment_success_rate, 6),
        "min_treatment_success_rate": min_success_rate,
        "evidence_payload_sha256": evidence_hash,
        "independent_verifier_id": independent_verifier_id,
        "independent_attestation_sha256": independent_attestation_sha256,
        "task_issuer_id": task_issuer_id,
        "task_commitment_attestation_sha256": task_commitment_attestation_sha256,
        "preregistration_sha256": expected_prereg_hash,
        "trial_count": len(trials),
        "domain_counts": {domain: len(items) for domain, items in paired.items()},
        "comparison_accounting": comparison_accounting,
        "required_positive_domains": required_positive,
        # CP126 8a56c486: generation-manifest fields that NEITHER arm declared.
        # Declared differences reject the trial; these are the parity claims the
        # certificate could not verify at all, carried on the certificate rather
        # than assumed equal. Closing them needs worker-side manifest emission.
        "unverified_generation_parity_fields": sorted(generation_parity_gaps),
        "generation_parity_fully_declared": not generation_parity_gaps,
        "unproven_integrity_claims": sorted(unproven_integrity_claims),
        "integrity_claims_digest_backed": not unproven_integrity_claims,
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
