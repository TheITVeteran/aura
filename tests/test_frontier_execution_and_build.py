"""CP126 2528aa7e + f17aa1b8: signatures over nothing, digests about nothing.

The independent verifier's signed payload repeated values copied straight out
of the trust pin — its own implementation hash and release hash, which the pin
already carried. Signing them proved key possession and nothing about work
done. A signer that never opened a file produced a payload indistinguishable
from one that verified every artifact.

The resident model's build identifiers had the mirror problem. Build hash,
worker binary hash, provenance hash and source commit were each accepted for
looking like a hash of the right shape, with no relationship required between
them, so any of them could name a build that never existed.

Neither gap can be closed completely here: no measured environment attests that
the pinned verifier binary is what ran, and nothing re-runs the build. What the
certificate can do is bind the signature to work only a real run produces, tie
the four build identifiers to one builder's claim, and name what that is worth.
"""
from __future__ import annotations

import copy

import pytest

from core.brain.llm.latent_cortex.frontier_certification import (
    BUILD_PROVENANCE_ATTESTATION,
    VERIFIER_EXECUTION_ATTESTATION,
    canonical_sha256,
)
from tests.fixtures.latent_frontier import (
    _bundle,
    _certify,
    _raw_artifact_receipt,
    _refresh_attestation,
)


class TestVerifierExecution:
    def test_the_certificate_names_what_the_signature_proves(self):
        certificate = _certify(_bundle())
        assert certificate["accepted"] is True, certificate["reasons"]
        assert (
            certificate["verifier_execution_attestation"]
            == VERIFIER_EXECUTION_ATTESTATION
        )

    def test_the_attestation_binds_the_receipt_the_signer_produced(self):
        bundle = _bundle()
        for attestation in bundle["independent_verifiers"]:
            assert attestation["signed_payload"][
                "raw_artifact_receipt_sha256"
            ] == canonical_sha256(_raw_artifact_receipt(bundle))

    def test_a_signature_over_the_wrong_receipt_is_refused(self):
        """Key possession without the artifact work behind it."""
        bundle = _bundle()
        attestations = copy.deepcopy(bundle["independent_verifiers"])
        attestations[0]["signed_payload"]["raw_artifact_receipt_sha256"] = "2" * 64
        bundle["independent_verifiers"] = attestations
        certificate = _certify(bundle)
        assert certificate["accepted"] is False
        assert "independent_verifier_signature_invalid" in certificate["reasons"]

    def test_an_attestation_missing_the_receipt_digest_is_refused(self):
        bundle = _bundle()
        attestations = copy.deepcopy(bundle["independent_verifiers"])
        del attestations[0]["signed_payload"]["raw_artifact_receipt_sha256"]
        bundle["independent_verifiers"] = attestations
        certificate = _certify(bundle)
        assert certificate["accepted"] is False
        assert "independent_verifier_signature_invalid" in certificate["reasons"]

    def test_changing_the_artifacts_invalidates_every_signature(self):
        """The receipt digest follows the bundle it was computed over."""
        bundle = _bundle()
        _refresh_attestation(bundle)
        assert _certify(bundle)["accepted"] is True
        tampered = copy.deepcopy(bundle)
        tampered["raw_artifact_manifest_sha256"] = "3" * 64
        certificate = _certify(tampered)
        assert certificate["accepted"] is False


class TestBuildProvenance:
    def test_the_certificate_names_how_far_the_build_claim_goes(self):
        certificate = _certify(_bundle())
        assert (
            certificate["build_provenance_attestation"]
            == BUILD_PROVENANCE_ATTESTATION
        )
        assert "unsigned" in BUILD_PROVENANCE_ATTESTATION

    def test_four_unrelated_digests_are_no_longer_enough(self):
        bundle = _bundle()
        del bundle["resident_model"]["release_attestation"]
        certificate = _certify(bundle)
        assert certificate["accepted"] is False
        assert "release_attestation_missing" in certificate["reasons"]

    def test_an_unattributed_attestation_is_refused(self):
        bundle = _bundle()
        bundle["resident_model"]["release_attestation"]["builder_id"] = "  "
        certificate = _certify(bundle)
        assert certificate["accepted"] is False
        assert "release_attestation_unattributed" in certificate["reasons"]

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("source_commit", "0" * 40),
            ("rebuild_worker_binary_sha256", "0" * 64),
            ("build_provenance_sha256", "0" * 64),
            ("installed_app_build_sha256", "0" * 64),
        ],
    )
    def test_each_identifier_must_be_the_one_that_ran(self, field, value):
        """A builder attesting to a different build is not this build's proof."""
        bundle = _bundle()
        bundle["resident_model"]["release_attestation"][field] = value
        certificate = _certify(bundle)
        assert certificate["accepted"] is False
        assert f"release_attestation_{field}_unbound" in certificate["reasons"]

    def test_an_attestation_without_a_time_is_refused(self):
        bundle = _bundle()
        del bundle["resident_model"]["release_attestation"]["attested_at"]
        certificate = _certify(bundle)
        assert certificate["accepted"] is False
        assert "release_attestation_time_missing" in certificate["reasons"]
