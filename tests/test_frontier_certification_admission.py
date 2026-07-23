"""CP126: evidence-admission contracts for the frontier gain certificate.

The certificate is a release gate for a capability claim, so the ways it can
LIE matter more than the ways it can fail. These tests pin the four lies the
semantic review found:

1. A defective trial (contaminated, unblinded, parity-failing, unauthenticated)
   still fed the paired statistical claim.
2. The statistical grader was explicitly told to ignore compute.
3. Beating an ablated SELF was certified with the same claim as beating an
   external frontier model.
4. The core API could accept — and claim PROVEN — without any raw artifact
   ever being loaded or verified.
"""
from __future__ import annotations

import copy

import pytest

from core.brain.llm.latent_cortex.resource_accounting import (
    ModelComputeProfile,
    ResourceLedger,
)
from tests.fixtures.latent_frontier import (
    _bundle,
    _certify,
    _raw_artifact_receipt,
    _refresh_attestation,
    _refresh_task_commitment,
)


def _accepted_bundle() -> dict:
    bundle = _bundle()
    certificate = _certify(bundle)
    assert certificate["accepted"] is True, certificate["reasons"]
    return bundle


# ── 1. defective trials must not feed the statistical claim ────────────────


@pytest.mark.parametrize(
    ("mutation", "expected_reason"),
    [
        ({"contamination_scan_passed": False}, "heldout_or_contamination_unproven"),
        ({"held_out": False}, "heldout_or_contamination_unproven"),
        ({"verifier_blinded": False}, "blinded_verifier_unproven"),
        ({"control_information_sha256": "b" * 64}, "information_mismatch"),
        ({"control_decode_policy_sha256": "e" * 64}, "decode_policy_mismatch"),
        ({"scorer_implementation_sha256": "0" * 64}, "scorer_not_preregistered"),
    ],
)
def test_defective_trial_is_excluded_from_the_paired_claim(mutation, expected_reason):
    """A trial that fails its own integrity checks contributes NOTHING.

    Previously each defect only appended a reason while the trial still became
    a PairedObservation, so contaminated or unblinded work could produce a
    nested PROVEN statistical result.
    """
    bundle = _accepted_bundle()
    bundle["trials"][0].update(mutation)
    _refresh_task_commitment(bundle)
    _refresh_attestation(bundle)

    certificate = _certify(bundle)

    assert certificate["accepted"] is False
    assert any(expected_reason in reason for reason in certificate["reasons"])
    # The trial was rejected, not merely complained about.
    assert certificate["rejected_trial_count"] == 1
    assert certificate["admitted_trial_count"] == len(bundle["trials"]) - 1
    # Its domain lost an observation, so the claim cannot silently stand.
    assert sum(certificate["domain_counts"].values()) == certificate["admitted_trial_count"]


def test_rejected_certificate_never_publishes_an_admissible_claim():
    """The nested statistical claim must not read as a standalone verdict."""
    bundle = _accepted_bundle()
    bundle["trials"][0]["contamination_scan_passed"] = False
    _refresh_task_commitment(bundle)
    _refresh_attestation(bundle)

    certificate = _certify(bundle)

    assert certificate["accepted"] is False
    assert certificate["claim_tier"] == "CONJECTURE"
    assert certificate["statistical_claim"]["admissible"] is False


# ── 2. compute evidence must survive into the claim ────────────────────────


def test_layer_application_parity_is_enforced_not_merely_collected():
    """Equal reported FLOPs must not license unequal layer work."""
    bundle = _accepted_bundle()
    bundle["trials"][0]["treatment_compute"]["layer_apps"] = 100_000
    _refresh_task_commitment(bundle)
    _refresh_attestation(bundle)

    certificate = _certify(bundle)

    assert certificate["accepted"] is False
    assert any("layer_apps_mismatch" in reason for reason in certificate["reasons"])
    assert certificate["rejected_trial_count"] == 1


def test_claim_carries_compute_validity():
    """The grader must be given the preregistered compute tolerance."""
    certificate = _certify(_accepted_bundle())
    families = certificate["statistical_claim"]["evidence"]["families"]
    for stats in families.values():
        assert stats["missing_compute"] is False
        assert stats["compute_mismatch_task_ids"] == []


def test_claim_rejects_rehashed_hidden_verifier_advantage():
    bundle = _accepted_bundle()
    compute = bundle["trials"][0]["treatment_compute"]
    original = compute["resource_accounting"]
    ledger = ResourceLedger(
        ModelComputeProfile.from_receipt(original["model_profile"])
    )
    for operation, counters in original["operations"].items():
        ledger.charge(operation, **counters)
    ledger.charge(
        "unmatched_private_verifier",
        verifier_calls=1,
        verifier_input_bytes=128,
        verifier_output_bytes=8,
    )
    compute["resource_accounting"] = ledger.to_receipt()
    compute["estimated_flops"] = compute["resource_accounting"]["estimated_flops"]
    _refresh_task_commitment(bundle)
    _refresh_attestation(bundle)

    certificate = _certify(bundle)

    assert certificate["accepted"] is False
    assert any(
        "resource_mismatch:verifier_calls" in reason
        for reason in certificate["reasons"]
    )
    assert certificate["rejected_trial_count"] == 1


# ── 3. claim scope must distinguish ablation from frontier ─────────────────


def test_same_checkpoint_win_is_not_a_frontier_claim():
    certificate = _certify(_accepted_bundle())

    assert certificate["accepted"] is True
    assert certificate["claim_scope"] == "treatment_contribution_vs_same_checkpoint_ablation"
    assert certificate["frontier_competitiveness_established"] is False
    assert "NOT evidence of frontier competitiveness" in (
        certificate["statistical_claim"]["statement"]
    )


def test_external_comparison_is_the_only_frontier_claim():
    bundle = _bundle(comparison_kind="resident_32b_vs_external_frontier")
    certificate = _certify(bundle)

    assert certificate["accepted"] is True, certificate["reasons"]
    assert certificate["claim_scope"] == "frontier_competitiveness_vs_external_control"
    assert certificate["frontier_competitiveness_established"] is True


def test_certificate_names_which_domains_carry_the_gain():
    """A two-thirds rule lets a third of domains show no benefit; the
    certificate must say which, instead of an unqualified claim."""
    certificate = _certify(_accepted_bundle())

    assert certificate["positive_domains"]
    assert set(certificate["positive_domains"]).isdisjoint(
        certificate["non_positive_domains"]
    )


# ── 4. raw artifacts must actually be verified ─────────────────────────────


def test_missing_raw_artifact_receipt_cannot_be_accepted():
    """A syntactically valid manifest hash is not evidence of any artifact.

    Calls the real API with NO receipt — the exact path that previously
    reached accepted=True / PROVEN without opening a single artifact.
    """
    from core.brain.llm.latent_cortex.frontier_certification import (
        verify_frontier_gain_bundle,
    )
    from tests.fixtures.latent_frontier import (
        _TRUSTED_TASK_ISSUERS,
        _TRUSTED_VERIFIERS,
    )

    certificate = verify_frontier_gain_bundle(
        _accepted_bundle(),
        trusted_verifiers=_TRUSTED_VERIFIERS,
        trusted_task_issuers=_TRUSTED_TASK_ISSUERS,
    )

    assert certificate["accepted"] is False
    assert certificate["raw_artifact_verified"] is False
    assert "raw_artifact_package_unverified" in certificate["reasons"]


@pytest.mark.parametrize(
    ("mutation", "expected_reason"),
    [
        ({"accepted": False}, "raw_artifact_receipt_not_accepted"),
        ({"manifest_sha256": "1" * 64}, "raw_artifact_receipt_manifest_mismatch"),
        ({"schema": "wrong.schema.v1"}, "raw_artifact_receipt_schema_mismatch"),
        ({"trial_count": 3}, "raw_artifact_receipt_trial_count_mismatch"),
    ],
)
def test_raw_artifact_receipt_must_bind_this_bundle(mutation, expected_reason):
    bundle = _accepted_bundle()
    receipt = _raw_artifact_receipt(bundle)
    receipt.update(mutation)

    certificate = _certify(bundle, raw_artifact_receipt=receipt)

    assert certificate["accepted"] is False
    assert certificate["raw_artifact_verified"] is False
    assert expected_reason in certificate["reasons"]


# ── 5. absolute capability floor ───────────────────────────────────────────


def test_relative_win_over_a_weak_control_is_not_enough():
    """Without a floor, a weak treatment certifies by beating a weaker control."""
    bundle = _accepted_bundle()
    # Treatment succeeds on a minority of trials but still beats the control.
    for index, trial in enumerate(bundle["trials"]):
        trial["treatment_success"] = index % 4 == 0
        trial["control_success"] = False
    _refresh_task_commitment(bundle)
    _refresh_attestation(bundle)

    certificate = _certify(bundle)

    assert certificate["accepted"] is False
    assert "treatment_below_absolute_capability_floor" in certificate["reasons"]
    assert certificate["treatment_success_rate"] < certificate["min_treatment_success_rate"]


def test_preregistration_must_pin_scorer_decode_and_floor():
    for field in (
        "scorer_implementation_sha256",
        "decode_policy_sha256",
        "min_treatment_success_rate",
    ):
        bundle = copy.deepcopy(_bundle())
        bundle["preregistration"].pop(field)
        from core.brain.llm.latent_cortex.frontier_certification import canonical_sha256

        bundle["preregistration_sha256"] = canonical_sha256(bundle["preregistration"])
        _refresh_task_commitment(bundle)
        _refresh_attestation(bundle)

        certificate = _certify(bundle)

        assert certificate["accepted"] is False, field


# ── 6. resident identity must be manifest-derived ──────────────────────────


def test_parameter_count_must_come_from_the_hashed_checkpoint_manifest():
    bundle = _accepted_bundle()
    bundle["resident_model"]["parameter_count_source"] = "self_reported"
    _refresh_attestation(bundle)

    certificate = _certify(bundle)

    assert certificate["accepted"] is False
    assert "resident_parameter_count_not_manifest_derived" in certificate["reasons"]
