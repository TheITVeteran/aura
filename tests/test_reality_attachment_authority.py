from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from core.governance.capability_chain import (
    get_capability_issuer,
    get_capability_verifier,
    reset_capability_chain,
)
from core.reality_reach.attachment_authority import (
    ATTACHMENT_AUTHORITY_ACTION,
    AttachmentAuthorityError,
    AttachmentCapabilityAuthorityVerifier,
    build_attachment_authority_intent,
)


class Decision:
    outcome = "proceed"
    domain = "environment_action"
    constraints: tuple[str, ...] = ()

    def __init__(self, receipt_id: str) -> None:
        self.receipt_id = receipt_id


class ReceiptSource:
    def __init__(self) -> None:
        self.material: dict[str, dict[str, str]] = {}
        self.valid: set[str] = set()

    def add(self, receipt_id: str) -> None:
        payload = json.dumps(
            {
                "receipt_id": receipt_id,
                "outcome": "proceed",
                "domain": "environment_action",
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        self.valid.add(receipt_id)
        self.material[receipt_id] = {
            "receipt_id": receipt_id,
            "payload": payload,
            "signature": "ab" * 64,
            "signature_scheme": "ed25519",
        }

    def verify_receipt_signature(self, receipt_id: str) -> bool:
        return receipt_id in self.valid

    def get_receipt_verification_material(self, receipt_id: str) -> dict[str, str]:
        return dict(self.material.get(receipt_id, {}))


@pytest.fixture(autouse=True)
def isolated_capability_chain(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AURA_CAPABILITY_KEY_DIR", str(tmp_path / "capability-keys"))
    monkeypatch.delenv("AURA_CAPABILITY_FORCE_HMAC", raising=False)
    reset_capability_chain()
    yield
    reset_capability_chain()


def _intent() -> dict[str, object]:
    return build_attachment_authority_intent(
        request_id="reality.connect.1234",
        candidate_sha256="sha256:" + "1" * 64,
        identity_fingerprint="sha256:" + "2" * 64,
        connector_id="test.connector",
        manifest_sha256="sha256:" + "3" * 64,
        requested_access=("observe", "control"),
        persistent=True,
        grant_ttl_s=3600,
    )


def _issue(intent: dict[str, object], receipts: ReceiptSource, receipt_id: str = "will_test_1"):
    receipts.add(receipt_id)
    return get_capability_issuer().issue_from_decision(
        Decision(receipt_id),
        action=ATTACHMENT_AUTHORITY_ACTION,
        payload=intent,
        scope=str(intent["scope"]),
    )


def test_exact_capability_and_independent_will_receipt_produce_evidence() -> None:
    receipts = ReceiptSource()
    intent = _intent()
    capability = _issue(intent, receipts)
    verifier = AttachmentCapabilityAuthorityVerifier(will_receipts=receipts)

    evidence = verifier.verify(capability, intent=intent, persistent=True)

    assert evidence["action_digest"] == capability.action_digest
    assert evidence["scope"] == "reality_attachment.control"
    assert evidence["capability"]["receipt_id"] == "will_test_1"
    assert evidence["will_receipt"]["signature_scheme"] == "ed25519"
    assert verifier.validate_persisted(
        evidence,
        intent=intent,
        persistent=True,
    ) == evidence


def test_capability_is_single_use_across_attachment_attempts() -> None:
    receipts = ReceiptSource()
    intent = _intent()
    capability = _issue(intent, receipts)
    verifier = AttachmentCapabilityAuthorityVerifier(will_receipts=receipts)
    verifier.verify(capability, intent=intent, persistent=True)

    with pytest.raises(AttachmentAuthorityError, match="capability_replayed"):
        verifier.verify(capability, intent=intent, persistent=True)


def test_candidate_or_manifest_substitution_fails_action_binding() -> None:
    receipts = ReceiptSource()
    intent = _intent()
    capability = _issue(intent, receipts)
    tampered = dict(intent)
    tampered["manifest_sha256"] = "sha256:" + "f" * 64
    verifier = AttachmentCapabilityAuthorityVerifier(will_receipts=receipts)

    with pytest.raises(AttachmentAuthorityError, match="capability_action_mismatch"):
        verifier.verify(capability, intent=tampered, persistent=True)


def test_bare_receipt_string_cannot_mint_trust() -> None:
    receipts = ReceiptSource()
    receipts.add("will_test_bare")
    verifier = AttachmentCapabilityAuthorityVerifier(will_receipts=receipts)

    with pytest.raises(AttachmentAuthorityError, match="capability_malformed"):
        verifier.verify(
            {"receipt_id": "will_test_bare"},
            intent=_intent(),
            persistent=True,
        )


def test_unverified_will_receipt_fails_after_capability_signature() -> None:
    receipts = ReceiptSource()
    intent = _intent()
    capability = _issue(intent, receipts)
    receipts.valid.clear()
    verifier = AttachmentCapabilityAuthorityVerifier(will_receipts=receipts)

    with pytest.raises(AttachmentAuthorityError, match="will_receipt_unverified"):
        verifier.verify(capability, intent=intent, persistent=True)

    receipts.valid.add(capability.receipt_id)
    assert verifier.verify(capability, intent=intent, persistent=True)["capability"][
        "capability_id"
    ] == capability.capability_id


def test_tampered_persisted_evidence_is_refused() -> None:
    receipts = ReceiptSource()
    intent = _intent()
    capability = _issue(intent, receipts)
    verifier = AttachmentCapabilityAuthorityVerifier(will_receipts=receipts)
    evidence = dict(verifier.verify(capability, intent=intent, persistent=True))
    evidence["scope"] = "reality_attachment.observe"

    with pytest.raises(AttachmentAuthorityError, match="evidence_invalid"):
        verifier.validate_persisted(evidence, intent=intent, persistent=True)


def test_persistent_trust_rejects_symmetric_degraded_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AURA_CAPABILITY_FORCE_HMAC", "1")
    reset_capability_chain()
    receipts = ReceiptSource()
    intent = _intent()
    capability = _issue(intent, receipts)
    verifier = AttachmentCapabilityAuthorityVerifier(will_receipts=receipts)

    with pytest.raises(AttachmentAuthorityError, match="root_not_durable"):
        verifier.verify(capability, intent=intent, persistent=True)


def test_expired_capability_is_refused_before_receipt_acceptance() -> None:
    receipts = ReceiptSource()
    intent = _intent()
    capability = _issue(intent, receipts)
    expired = replace(capability, expires_at=capability.issued_at - 1.0)
    # Mutation invalidates the signature as well; either condition must fail at
    # the cryptographic boundary before a Will receipt can create trust.
    result = get_capability_verifier().verify(expired, consume=False)
    assert result.ok is False
