"""CP126 c7a9c1a5 + 48df4291: independence was a string comparison.

A verifier counted as independent when its signer id differed from
``bundle["producer_id"]`` — a field the producer wrote about itself, backed by
no key. One actor holding a trusted verifier key could name itself anything,
sign its own evidence, and the certificate would read as independently
verified.

And one signature settled it. A release-grade frontier claim rested on a single
organization's opinion, with no second verifier, no organizational separation,
and nothing to do if two verifiers had disagreed.
"""
from __future__ import annotations

import copy

import pytest

from core.brain.llm.latent_cortex.frontier_certification import (
    MINIMUM_VERIFIER_QUORUM,
    verify_frontier_gain_bundle,
)
from core.brain.llm.latent_cortex.frontier_verifier import (
    FrontierVerificationError,
    validate_trust_config,
)
from tests.fixtures.latent_frontier import (
    _PRODUCER_ID,
    _SECOND_VERIFIER_ID,
    _TRUSTED_PRODUCERS,
    _TRUSTED_TASK_ISSUERS,
    _TRUSTED_VERIFIERS,
    _VERIFIER_ID,
    _bundle,
    _certify,
    _raw_artifact_receipt,
    _refresh_producer_attestation,
    _signed_attestation,
    _trust_config,
)


class TestProducerIdentity:
    def test_the_certificate_names_a_verified_producer(self):
        certificate = _certify(_bundle())
        assert certificate["accepted"] is True, certificate["reasons"]
        assert certificate["verified_producer_id"] == _PRODUCER_ID

    def test_an_unsigned_producer_is_not_a_producer(self):
        bundle = _bundle()
        del bundle["producer_attestation"]
        certificate = _certify(bundle)
        assert certificate["accepted"] is False
        assert "producer_trust_pin_missing" in certificate["reasons"]

    def test_an_unpinned_producer_is_refused(self):
        bundle = _bundle()
        certificate = verify_frontier_gain_bundle(
            bundle,
            trusted_verifiers=_TRUSTED_VERIFIERS,
            trusted_task_issuers=_TRUSTED_TASK_ISSUERS,
            trusted_producers={},
            raw_artifact_receipt=_raw_artifact_receipt(bundle),
        )
        assert certificate["accepted"] is False
        assert "producer_trust_pin_missing" in certificate["reasons"]

    def test_the_claimed_id_must_match_the_signing_key(self):
        """Renaming yourself does not make you a different party."""
        bundle = _bundle()
        bundle["producer_id"] = "some-other-lab"
        _refresh_producer_attestation(bundle)
        certificate = _certify(bundle)
        assert certificate["accepted"] is False
        assert "producer_attestation_invalid" in certificate["reasons"]

    def test_the_producer_signature_covers_the_evidence(self):
        bundle = _bundle()
        bundle["trials"][0]["treatment_score"] = 0.95
        certificate = _certify(bundle)
        assert certificate["accepted"] is False
        assert "producer_attestation_invalid" in certificate["reasons"]


class TestVerifierQuorum:
    def test_the_fixture_meets_the_quorum(self):
        certificate = _certify(_bundle())
        assert certificate["independent_verifier_ids"] == sorted(
            [_VERIFIER_ID, _SECOND_VERIFIER_ID]
        )
        assert len(certificate["independent_verifier_organizations"]) >= (
            MINIMUM_VERIFIER_QUORUM
        )
        assert certificate["verifier_quorum_required"] == MINIMUM_VERIFIER_QUORUM

    def test_one_signature_is_one_opinion(self):
        bundle = _bundle()
        bundle["independent_verifiers"] = bundle["independent_verifiers"][:1]
        certificate = _certify(bundle)
        assert certificate["accepted"] is False
        assert "independent_verifier_quorum_not_met" in certificate["reasons"]

    def test_no_signatures_at_all_is_named_separately(self):
        bundle = _bundle(include_attestation=False)
        certificate = _certify(bundle)
        assert certificate["accepted"] is False
        assert "independent_verifier_quorum_missing" in certificate["reasons"]

    def test_the_same_verifier_twice_is_still_one_opinion(self):
        bundle = _bundle()
        bundle["independent_verifiers"] = [
            _signed_attestation(bundle, _VERIFIER_ID, 2000.0),
            _signed_attestation(bundle, _VERIFIER_ID, 2001.0),
        ]
        certificate = _certify(bundle)
        assert certificate["accepted"] is False
        assert "independent_verifier_signature_duplicated" in certificate["reasons"]
        assert "independent_verifier_quorum_not_met" in certificate["reasons"]

    def test_a_verifier_that_disagrees_does_not_count_toward_the_quorum(self):
        """Signing `accepted: False` is a rejection, not a second vote."""
        bundle = _bundle()
        dissent = copy.deepcopy(bundle["independent_verifiers"][1])
        dissent["signed_payload"]["accepted"] = False
        bundle["independent_verifiers"][1] = dissent
        certificate = _certify(bundle)
        assert certificate["accepted"] is False
        assert "independent_verifier_signature_invalid" in certificate["reasons"]
        assert "independent_verifier_quorum_not_met" in certificate["reasons"]

    def test_the_attestation_digest_covers_every_signature(self):
        """One signer's bytes must not stand in for the quorum's."""
        full = _certify(_bundle())
        bundle = _bundle()
        bundle["independent_verifiers"] = bundle["independent_verifiers"][:1]
        partial = _certify(bundle)
        assert (
            full["independent_attestation_sha256"]
            != partial["independent_attestation_sha256"]
        )


class TestOrganizationalSeparation:
    def test_two_keys_in_one_organization_are_one_party(self):
        shared = copy.deepcopy(_TRUSTED_VERIFIERS)
        shared[_SECOND_VERIFIER_ID]["organization"] = shared[_VERIFIER_ID][
            "organization"
        ]
        bundle = _bundle()
        certificate = verify_frontier_gain_bundle(
            bundle,
            trusted_verifiers=shared,
            trusted_task_issuers=_TRUSTED_TASK_ISSUERS,
            trusted_producers=_TRUSTED_PRODUCERS,
            raw_artifact_receipt=_raw_artifact_receipt(bundle),
        )
        assert certificate["accepted"] is False
        assert "independent_verifier_quorum_not_met" in certificate["reasons"]

    def test_a_verifier_inside_the_producers_organization_is_not_independent(self):
        captured = copy.deepcopy(_TRUSTED_VERIFIERS)
        captured[_VERIFIER_ID]["organization"] = _TRUSTED_PRODUCERS[_PRODUCER_ID][
            "organization"
        ]
        bundle = _bundle()
        certificate = verify_frontier_gain_bundle(
            bundle,
            trusted_verifiers=captured,
            trusted_task_issuers=_TRUSTED_TASK_ISSUERS,
            trusted_producers=_TRUSTED_PRODUCERS,
            raw_artifact_receipt=_raw_artifact_receipt(bundle),
        )
        assert certificate["accepted"] is False
        assert "verifier_shares_producer_organization" in certificate["reasons"]


class TestTrustConfig:
    def test_the_fixture_trust_config_validates(self):
        trust = validate_trust_config(_trust_config())
        assert set(trust["producers"]) == {_PRODUCER_ID}
        assert len(trust["verifiers"]) >= MINIMUM_VERIFIER_QUORUM

    def test_a_pin_without_an_organization_is_refused(self):
        config = copy.deepcopy(_trust_config())
        del config["verifiers"][_VERIFIER_ID]["organization"]
        with pytest.raises(FrontierVerificationError, match="verifiers_trust_pin_invalid"):
            validate_trust_config(config)

    def test_a_config_without_producers_is_refused(self):
        config = copy.deepcopy(_trust_config())
        del config["producers"]
        with pytest.raises(FrontierVerificationError, match="trust_config_schema_invalid"):
            validate_trust_config(config)

    def test_one_organization_holding_two_roles_is_refused(self):
        config = copy.deepcopy(_trust_config())
        config["producers"][_PRODUCER_ID]["organization"] = config["verifiers"][
            _VERIFIER_ID
        ]["organization"]
        with pytest.raises(
            FrontierVerificationError, match="trust_role_organization_reused"
        ):
            validate_trust_config(config)

    def test_a_trust_set_that_cannot_reach_quorum_is_refused(self):
        """Refuse at config time rather than at every certification."""
        config = copy.deepcopy(_trust_config())
        del config["verifiers"][_SECOND_VERIFIER_ID]
        with pytest.raises(
            FrontierVerificationError, match="trust_verifier_quorum_unreachable"
        ):
            validate_trust_config(config)
