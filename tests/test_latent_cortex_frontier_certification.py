from __future__ import annotations

import base64
import copy

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from core.brain.llm.latent_cortex.frontier_certification import (
    canonical_sha256,
    evidence_payload_sha256,
    verify_frontier_gain_bundle,
)
from tests.fixtures.latent_frontier import (
    _TASK_ISSUER_ID,
    _TRUSTED_TASK_ISSUERS,
    _TRUSTED_VERIFIERS,
    _VERIFIER_ID,
    _bundle,
    _certify,
    _refresh_attestation,
    _refresh_task_commitment,
)


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


def test_frontier_certificate_accepts_external_frontier_comparison():
    certificate = _certify(_bundle(comparison_kind="resident_32b_vs_external_frontier"))
    assert certificate["accepted"] is True, certificate["reasons"]
    assert certificate["comparison_kind"] == "resident_32b_vs_external_frontier"
    assert certificate["claim_tier"] == "PROVEN"


@pytest.mark.parametrize(
    ("mutation", "expected_reason"),
    [
        (
            lambda b: b["preregistration"].pop("control_model_id"),
            "external_frontier_control_model_missing",
        ),
        (
            lambda b: b["preregistration"].pop("control_model_build_fingerprint"),
            "external_frontier_control_build_missing",
        ),
        (
            lambda b: b["preregistration"].pop("control_provider"),
            "external_frontier_control_provider_missing",
        ),
        (
            lambda b: b["trials"][0]["control_receipt"].update(model_id="different-model"),
            "external_control_model_mismatch",
        ),
        (
            lambda b: b["trials"][0]["control_receipt"].update(
                model_build_fingerprint="different-build"
            ),
            "external_control_build_mismatch",
        ),
        (
            lambda b: b["trials"][0]["control_receipt"].update(provider="different-provider"),
            "external_control_provider_mismatch",
        ),
        (
            lambda b: b["trials"][0]["control_receipt"].pop("provider_receipt_sha256"),
            "external_control_receipt_unproven",
        ),
    ],
)
def test_frontier_certificate_rejects_unpinned_external_control(mutation, expected_reason):
    bundle = _bundle(comparison_kind="resident_32b_vs_external_frontier")
    mutation(bundle)
    bundle["preregistration_sha256"] = canonical_sha256(bundle["preregistration"])
    _refresh_task_commitment(bundle)
    _refresh_attestation(bundle)

    certificate = _certify(bundle)

    assert certificate["accepted"] is False
    assert any(expected_reason in reason for reason in certificate["reasons"])


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
        (
            lambda b: b["trials"][0]["treatment_compute"].update(estimated_flops=2_000_000),
            "compute_mismatch",
        ),
        (
            lambda b: b["trials"][0]["treatment_receipt"].update(fast_weights_erased=False),
            "fast_weights_erased_unproven",
        ),
        (
            lambda b: b["trials"][0]["treatment_receipt"].update(latent_opt_steps=0),
            "treatment_latent_opt_steps_unproven",
        ),
        (
            lambda b: b["trials"][0]["treatment_receipt"].update(fast_weight_budget_exhausted=True),
            "fast_weight_budget_exhausted",
        ),
        (
            lambda b: b["trials"][0]["treatment_receipt"].update(latent_opt_rejected=1),
            "latent_opt_accounting_mismatch",
        ),
        (lambda b: b["trials"][0].update(held_out=False), "heldout_or_contamination_unproven"),
        (
            lambda b: b["trials"][0]["control_receipt"].update(checkpoint_fingerprint="wrong"),
            "control_checkpoint_mismatch",
        ),
        (
            lambda b: b["trials"][1].update(
                task_payload_sha256=b["trials"][0]["task_payload_sha256"]
            ),
            "duplicate_task_payload",
        ),
        (
            lambda b: b["trials"][1]["treatment_receipt"].update(
                episode_id=b["trials"][0]["treatment_receipt"]["episode_id"]
            ),
            "duplicate_treatment_episode",
        ),
        (
            lambda b: b["trials"][0].update(treatment_output_sha256="not-a-hash"),
            "treatment_output_sha256_invalid",
        ),
        (
            lambda b: b["trials"][0]["control_compute"].update(estimator_sha256="1" * 64),
            "control_compute_estimator_mismatch",
        ),
        (
            lambda b: b["independent_verifier"]["signed_payload"].update(accepted=False),
            "independent_verifier_signature_invalid",
        ),
        (
            lambda b: b["preregistration"].update(minimum_effect=0.07),
            "preregistration_hash_mismatch",
        ),
    ],
)
def test_frontier_certificate_rejects_missing_or_tampered_evidence(mutation, expected_reason):
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
            lambda bundle: bundle["trials"][0]["treatment_receipt"].update(honest_flags=3),
            "treatment_honest_flags_invalid",
        ),
        (
            lambda bundle: bundle["trials"][0]["treatment_compute"].update(layer_apps=10**10_000),
            "treatment_layer_apps_invalid",
        ),
    ],
)
def test_frontier_certificate_fails_closed_for_malformed_trial_receipts(mutation, expected_reason):
    bundle = _bundle()
    mutation(bundle)

    certificate = _certify(bundle)

    assert certificate["accepted"] is False
    assert any(expected_reason in reason for reason in certificate["reasons"])
