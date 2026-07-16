from __future__ import annotations

import base64
import copy

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from core.brain.frontier_evidence_v5 import canonical_json_bytes
from core.brain.llm.latent_cortex.frontier_certification import (
    INDEPENDENT_ATTESTATION_SCHEMA,
    SCHEMA,
    TASK_COMMITMENT_ATTESTATION_SCHEMA,
    canonical_sha256,
    evidence_payload_sha256,
    verify_frontier_gain_bundle,
)

_VERIFIER_KEY = Ed25519PrivateKey.generate()
_TASK_ISSUER_KEY = Ed25519PrivateKey.generate()
_VERIFIER_ID = "independent-proof-kernel"
_TASK_ISSUER_ID = "independent-task-issuer"
_VERIFIER_IMPLEMENTATION = "d" * 64
_VERIFIER_RELEASE = "0" * 64
_TASK_ISSUER_IMPLEMENTATION = "2" * 64
_TASK_ISSUER_RELEASE = "3" * 64
_VERIFIER_PUBLIC_KEY = base64.b64encode(
    _VERIFIER_KEY.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
).decode("ascii")
_TRUSTED_VERIFIERS = {
    _VERIFIER_ID: {
        "public_key_b64": _VERIFIER_PUBLIC_KEY,
        "implementation_sha256": _VERIFIER_IMPLEMENTATION,
        "release_sha256": _VERIFIER_RELEASE,
    }
}
_TASK_ISSUER_PUBLIC_KEY = base64.b64encode(
    _TASK_ISSUER_KEY.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
).decode("ascii")
_TRUSTED_TASK_ISSUERS = {
    _TASK_ISSUER_ID: {
        "public_key_b64": _TASK_ISSUER_PUBLIC_KEY,
        "implementation_sha256": _TASK_ISSUER_IMPLEMENTATION,
        "release_sha256": _TASK_ISSUER_RELEASE,
    }
}


def _refresh_task_commitment(bundle: dict) -> None:
    manifest = [
        {
            "trial_id": trial["trial_id"],
            "task_id": trial["task_id"],
            "domain": trial["domain"],
            "task_payload_sha256": trial["task_payload_sha256"],
            "task_generated_at": trial["task_generated_at"],
        }
        for trial in bundle["trials"]
    ]
    manifest.sort(key=lambda row: row["trial_id"])
    manifest_sha256 = canonical_sha256(manifest)
    task_commitment_sha256 = canonical_sha256(
        {
            "architecture_freeze_sha256": bundle["preregistration"][
                "architecture_freeze_sha256"
            ],
            "preregistration_sha256": bundle["preregistration_sha256"],
            "task_count": len(manifest),
            "task_manifest_sha256": manifest_sha256,
        }
    )
    payload = {
        "architecture_freeze_sha256": bundle["preregistration"][
            "architecture_freeze_sha256"
        ],
        "preregistration_sha256": bundle["preregistration_sha256"],
        "task_commitment_sha256": task_commitment_sha256,
        "task_manifest_sha256": manifest_sha256,
        "task_count": len(manifest),
        "issuer_implementation_sha256": _TASK_ISSUER_IMPLEMENTATION,
        "issuer_release_sha256": _TASK_ISSUER_RELEASE,
        "committed_at": 1200.0,
    }
    bundle["task_commitment_sha256"] = task_commitment_sha256
    bundle["task_commitment"] = {
        "schema": TASK_COMMITMENT_ATTESTATION_SCHEMA,
        "signed_payload": payload,
        "signer": {
            "algorithm": "Ed25519",
            "signer_id": _TASK_ISSUER_ID,
            "public_key_b64": _TASK_ISSUER_PUBLIC_KEY,
            "signature_b64": base64.b64encode(
                _TASK_ISSUER_KEY.sign(canonical_json_bytes(payload))
            ).decode("ascii"),
        },
    }


def _refresh_attestation(bundle: dict) -> None:
    payload = {
        "accepted": True,
        "claim_tier": "PROVEN",
        "producer_id": bundle["producer_id"],
        "evidence_payload_sha256": evidence_payload_sha256(bundle),
        "preregistration_sha256": bundle["preregistration_sha256"],
        "implementation_sha256": _VERIFIER_IMPLEMENTATION,
        "verifier_release_sha256": _VERIFIER_RELEASE,
        "raw_artifact_manifest_sha256": bundle["raw_artifact_manifest_sha256"],
        "task_commitment_sha256": bundle["task_commitment_sha256"],
        "verified_at": 2000.0,
    }
    bundle["independent_verifier"] = {
        "schema": INDEPENDENT_ATTESTATION_SCHEMA,
        "signed_payload": payload,
        "signer": {
            "algorithm": "Ed25519",
            "signer_id": _VERIFIER_ID,
            "public_key_b64": _VERIFIER_PUBLIC_KEY,
            "signature_b64": base64.b64encode(
                _VERIFIER_KEY.sign(canonical_json_bytes(payload))
            ).decode("ascii"),
        },
    }


def _certify(bundle: dict) -> dict:
    return verify_frontier_gain_bundle(
        bundle,
        trusted_verifiers=_TRUSTED_VERIFIERS,
        trusted_task_issuers=_TRUSTED_TASK_ISSUERS,
    )


def _bundle() -> dict:
    checkpoint = "8" * 64
    app_build = "4" * 64
    prereg = {
        "protocol_id": "rlc-frontier-v1",
        "comparison_kind": "resident_32b_vs_vanilla_same_checkpoint",
        "architecture_freeze_sha256": "a" * 64,
        "tool_policy_sha256": "c" * 64,
        "compute_estimator_sha256": "e" * 64,
        "treatment_checkpoint_fingerprint": checkpoint,
        "control_checkpoint_fingerprint": checkpoint,
        "frozen_at": 1000.0,
        "domains": ["math", "coding", "science"],
        "min_trials_per_domain": 40,
        "alpha": 0.05,
        "minimum_effect": 0.05,
        "compute_tolerance": 0.05,
        "compute_metric": "estimated_flops",
    }
    trials = []
    index = 0
    for domain in prereg["domains"]:
        for cell in range(40):
            trial_id = f"{domain}-{cell}"
            trials.append(
                {
                    "trial_id": trial_id,
                    "task_id": f"heldout-{trial_id}",
                    "domain": domain,
                    "held_out": True,
                    "contamination_scan_passed": True,
                    "task_generated_at": 1001.0 + index,
                    "evaluation_started_at": 1201.0 + index,
                    "verifier_blinded": True,
                    "verifier_receipt_sha256": canonical_sha256(
                        ["verifier", trial_id]
                    ),
                    "task_payload_sha256": canonical_sha256(["task", trial_id]),
                    "treatment_output_sha256": canonical_sha256(
                        ["treatment", trial_id]
                    ),
                    "control_output_sha256": canonical_sha256(
                        ["control", trial_id]
                    ),
                    "scorer_config_sha256": "f" * 64,
                    "treatment_information_sha256": "1" * 64,
                    "control_information_sha256": "1" * 64,
                    "treatment_tool_policy_sha256": "c" * 64,
                    "control_tool_policy_sha256": "c" * 64,
                    "run_order": "treatment_first" if index % 2 == 0 else "control_first",
                    "treatment_success": True,
                    "control_success": cell % 4 == 0,
                    "treatment_compute": {
                        "estimated_flops": 1_000_000.0,
                        "layer_apps": 1000,
                        "estimator_sha256": "e" * 64,
                    },
                    "control_compute": {
                        "estimated_flops": 1_000_000.0,
                        "layer_apps": 1000,
                        "estimator_sha256": "e" * 64,
                    },
                    "treatment_receipt": {
                        "episode_id": f"episode-{trial_id}",
                        "checkpoint_fingerprint": checkpoint,
                        "checkpoint_fingerprint_method": "sha256",
                        "checkpoint_file_count": 8,
                        "worker_boot_id": "boot-live-1",
                        "installed_app_build_sha256": app_build,
                        "schedule_hash": "3" * 64,
                        "params_unchanged": True,
                        "latent_opt_applied": True,
                        "latent_opt_mode": "gradient",
                        "latent_opt_attempts": 2,
                        "latent_opt_steps": 2,
                        "latent_opt_rejected": 0,
                        "latent_opt_budget_exhausted": False,
                        "fast_weights_applied": True,
                        "fast_weights_erased": True,
                        "fast_weights_layers": 4,
                        "fast_weight_optimization_attempts": 2,
                        "fast_weight_optimized_steps": 2,
                        "fast_weight_rejected_steps": 0,
                        "fast_weight_budget_exhausted": False,
                        "n_slots": 8,
                        "n_branches": 2,
                        "steps_taken": 4,
                        "honest_flags": [],
                    },
                    "control_receipt": {
                        "request_id": f"control-{trial_id}",
                        "mode": "vanilla",
                        "latent_cortex_enabled": False,
                        "params_unchanged": True,
                        "checkpoint_fingerprint": checkpoint,
                        "checkpoint_fingerprint_method": "sha256",
                        "checkpoint_file_count": 8,
                        "worker_boot_id": "boot-live-1",
                        "installed_app_build_sha256": app_build,
                    },
                }
            )
            index += 1
    bundle = {
        "schema": SCHEMA,
        "producer_id": "aura-lab",
        "preregistration": prereg,
        "preregistration_sha256": canonical_sha256(prereg),
        "resident_model": {
            "parameter_count": 32_000_000_000,
            "checkpoint_fingerprint": checkpoint,
            "checkpoint_fingerprint_method": "sha256",
            "checkpoint_file_count": 8,
            "worker_boot_id": "boot-live-1",
            "loaded_in_installed_app": True,
            "installed_app_bundle_id": "com.aura.desktop",
            "installed_app_build_sha256": app_build,
            "worker_binary_sha256": "5" * 64,
            "build_provenance_sha256": "6" * 64,
            "source_commit": "7" * 40,
            "latent_cortex_architecture_sha256": "a" * 64,
        },
        "raw_artifact_manifest_sha256": "9" * 64,
        "trials": trials,
    }
    _refresh_task_commitment(bundle)
    _refresh_attestation(bundle)
    return bundle


def test_frontier_certificate_accepts_only_complete_replicated_gain():
    certificate = _certify(_bundle())
    assert certificate["accepted"] is True, certificate["reasons"]
    assert certificate["claim_tier"] == "PROVEN"
    assert certificate["required_positive_domains"] == 2
    assert certificate["independent_verifier_id"] == _VERIFIER_ID
    assert certificate["independent_attestation_sha256"]
    assert certificate["task_issuer_id"] == _TASK_ISSUER_ID
    assert certificate["task_commitment_attestation_sha256"]
    assert certificate["certificate_sha256"]


def test_frontier_certificate_requires_an_external_trust_root():
    certificate = verify_frontier_gain_bundle(_bundle())

    assert certificate["accepted"] is False
    assert "independent_verifier_trust_pin_missing" in certificate["reasons"]


def test_frontier_certificate_requires_a_distinct_trusted_task_issuer():
    bundle = _bundle()
    missing_task_trust = verify_frontier_gain_bundle(
        bundle,
        trusted_verifiers=_TRUSTED_VERIFIERS,
    )
    assert missing_task_trust["accepted"] is False
    assert "task_issuer_trust_pin_missing" in missing_task_trust["reasons"]

    same_as_producer = _bundle()
    same_as_producer["producer_id"] = _TASK_ISSUER_ID
    _refresh_task_commitment(same_as_producer)
    _refresh_attestation(same_as_producer)
    certificate = _certify(same_as_producer)
    assert certificate["accepted"] is False
    assert "task_commitment_invalid" in certificate["reasons"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("task_generated_at", 1200.0),
        ("evaluation_started_at", 1199.0),
    ],
)
def test_frontier_task_commitment_enforces_two_phase_chronology(field, value):
    bundle = _bundle()
    bundle["trials"][0][field] = value
    _refresh_task_commitment(bundle)
    _refresh_attestation(bundle)

    certificate = _certify(bundle)

    assert certificate["accepted"] is False
    assert "task_commitment_invalid" in certificate["reasons"]


def test_frontier_evidence_hash_and_signature_bind_the_producer():
    original = _bundle()
    tampered = copy.deepcopy(original)
    tampered["producer_id"] = "replacement-producer"

    assert evidence_payload_sha256(tampered) != evidence_payload_sha256(original)
    certificate = _certify(tampered)
    assert certificate["accepted"] is False
    assert "independent_verification_invalid" in certificate["reasons"]


def test_frontier_certificate_rejects_wrong_trusted_key_and_same_role_signer():
    wrong_key = Ed25519PrivateKey.generate()
    wrong_public = base64.b64encode(
        wrong_key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    ).decode("ascii")
    wrong_trust = copy.deepcopy(_TRUSTED_VERIFIERS)
    wrong_trust[_VERIFIER_ID]["public_key_b64"] = wrong_public
    wrong_key_certificate = verify_frontier_gain_bundle(
        _bundle(),
        trusted_verifiers=wrong_trust,
        trusted_task_issuers=_TRUSTED_TASK_ISSUERS,
    )
    assert wrong_key_certificate["accepted"] is False
    assert "independent_verifier_signature_invalid" in wrong_key_certificate["reasons"]

    same_role = _bundle()
    same_role["producer_id"] = _VERIFIER_ID
    _refresh_attestation(same_role)
    same_role_certificate = _certify(same_role)
    assert same_role_certificate["accepted"] is False
    assert "independent_verification_invalid" in same_role_certificate["reasons"]


@pytest.mark.parametrize(
    ("mutation", "expected_reason"),
    [
        (lambda b: b["trials"][0]["treatment_compute"].update(estimated_flops=2_000_000), "compute_mismatch"),
        (lambda b: b["trials"][0]["treatment_receipt"].update(fast_weights_erased=False), "fast_weights_erased_unproven"),
        (lambda b: b["trials"][0]["treatment_receipt"].update(latent_opt_steps=0), "treatment_latent_opt_steps_unproven"),
        (lambda b: b["trials"][0]["treatment_receipt"].update(fast_weight_budget_exhausted=True), "fast_weight_budget_exhausted"),
        (lambda b: b["trials"][0]["treatment_receipt"].update(latent_opt_rejected=1), "latent_opt_accounting_mismatch"),
        (lambda b: b["trials"][0].update(held_out=False), "heldout_or_contamination_unproven"),
        (lambda b: b["trials"][0]["control_receipt"].update(checkpoint_fingerprint="wrong"), "control_checkpoint_mismatch"),
        (lambda b: b["trials"][1].update(task_payload_sha256=b["trials"][0]["task_payload_sha256"]), "duplicate_task_payload"),
        (lambda b: b["trials"][1]["treatment_receipt"].update(episode_id=b["trials"][0]["treatment_receipt"]["episode_id"]), "duplicate_treatment_episode"),
        (lambda b: b["trials"][0].update(treatment_output_sha256="not-a-hash"), "treatment_output_sha256_invalid"),
        (lambda b: b["trials"][0]["control_compute"].update(estimator_sha256="1" * 64), "control_compute_estimator_mismatch"),
        (
            lambda b: b["independent_verifier"]["signed_payload"].update(
                accepted=False
            ),
            "independent_verifier_signature_invalid",
        ),
        (lambda b: b["preregistration"].update(minimum_effect=0.07), "preregistration_hash_mismatch"),
    ],
)
def test_frontier_certificate_rejects_missing_or_tampered_evidence(
    mutation, expected_reason
):
    bundle = copy.deepcopy(_bundle())
    mutation(bundle)
    certificate = _certify(bundle)
    assert certificate["accepted"] is False
    assert any(expected_reason in reason for reason in certificate["reasons"])


def test_frontier_certificate_rejects_underpowered_and_order_biased_runs():
    bundle = _bundle()
    bundle["trials"] = bundle["trials"][:20]
    for trial in bundle["trials"]:
        trial["run_order"] = "treatment_first"
    _refresh_task_commitment(bundle)
    _refresh_attestation(bundle)
    certificate = _certify(bundle)
    assert certificate["accepted"] is False
    assert any("underpowered" in reason for reason in certificate["reasons"])
    assert "run_order_imbalanced" in certificate["reasons"]


@pytest.mark.parametrize(
    ("field", "value", "expected_reason"),
    [
        ("frozen_at", "not-a-timestamp", "invalid_freeze_timestamp"),
        ("frozen_at", 10**10_000, "invalid_freeze_timestamp"),
        ("min_trials_per_domain", {}, "min_trials_per_domain_below_30"),
        ("domains", 3, "invalid_preregistered_domains"),
        (
            "domains",
            ["math", {"not": "hashable"}],
            "invalid_preregistered_domains",
        ),
        ("domains", ["math", ""], "invalid_preregistered_domains"),
        ("alpha", float("nan"), "invalid_alpha"),
        ("alpha", 0.0, "invalid_alpha"),
        ("compute_tolerance", "not-a-number", "invalid_compute_tolerance"),
    ],
    ids=[
        "timestamp-string",
        "timestamp-overflow",
        "minimum-trials-mapping",
        "domains-scalar",
        "domains-unhashable",
        "domains-empty",
        "alpha-nan",
        "alpha-zero",
        "compute-tolerance-string",
    ],
)
def test_frontier_certificate_fails_closed_for_malformed_preregistration(
    field, value, expected_reason
):
    bundle = _bundle()
    bundle["preregistration"][field] = value

    first = _certify(bundle)
    second = _certify(bundle)

    assert first == second
    assert first["accepted"] is False
    assert expected_reason in first["reasons"]


def test_frontier_certificate_fails_closed_for_non_json_preregistration():
    bundle = _bundle()
    bundle["preregistration"]["unexpected"] = {"not-json"}

    certificate = _certify(bundle)

    assert certificate["accepted"] is False
    assert "preregistration_not_canonical_json" in certificate["reasons"]
    assert "evidence_payload_not_canonical_json" in certificate["reasons"]


@pytest.mark.parametrize(
    ("mutation", "expected_reason"),
    [
        (
            lambda bundle: bundle["trials"][0]["treatment_receipt"].update(
                honest_flags=3
            ),
            "treatment_honest_flags_invalid",
        ),
        (
            lambda bundle: bundle["trials"][0]["treatment_compute"].update(
                layer_apps=10**10_000
            ),
            "treatment_layer_apps_invalid",
        ),
    ],
)
def test_frontier_certificate_fails_closed_for_malformed_trial_receipts(
    mutation, expected_reason
):
    bundle = _bundle()
    mutation(bundle)

    certificate = _certify(bundle)

    assert certificate["accepted"] is False
    assert any(expected_reason in reason for reason in certificate["reasons"])
