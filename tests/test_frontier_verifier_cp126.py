"""CP126 frontier_verifier — tier coherence, claim scope, fail-closed identity."""
from __future__ import annotations

from core.brain.llm.latent_cortex import frontier_verifier as fv


class TestRejectionIsAlwaysProducible:
    """75ccf260: fingerprinting must not raise on the fail-closed path."""

    def test_unavailable_identity_instead_of_raise(self, monkeypatch):
        def _boom():
            raise OSError("verifier source relocated")

        monkeypatch.setattr(fv, "verifier_implementation_sha256", _boom)
        certificate = fv._rejection_certificate("bundle_unreadable")
        assert certificate["accepted"] is False
        assert certificate["verifier_implementation_sha256"] == "unavailable"
        assert certificate["reasons"] == ["bundle_unreadable"]
        assert certificate["certificate_sha256"]

    def test_normal_identity_is_used_when_available(self):
        certificate = fv._rejection_certificate("x")
        assert len(certificate["verifier_implementation_sha256"]) == 64


class TestRejectionTierCoherence:
    """fd223c1f: a rejected package cannot advertise a proven tier."""

    def test_rejection_certificate_is_unverified(self):
        certificate = fv._rejection_certificate("trust_config_invalid")
        assert certificate["statistical_tier"] == "UNVERIFIED"
        assert certificate["claim_tier"] == fv.CONJECTURE
        assert certificate["release_accepted"] is False


class TestReleaseGateHonesty:
    """463d0bfc: this module validates evidence; it does not run a release gate."""

    def test_release_gate_is_declared_not_evaluated(self):
        certificate = fv._rejection_certificate("x")
        assert certificate["release_gate"] == "not_evaluated_by_this_verifier"
        assert certificate["ablation_status"] == "NOT_VERIFIED"
