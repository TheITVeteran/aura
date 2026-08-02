from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from core.governance.capability_chain import (
    get_capability_issuer,
    get_capability_verifier,
    reset_capability_chain,
)
from core.reality_reach import attachment_authority as authority_module
from core.reality_reach.attachment_authority import (
    ATTACHMENT_AUTHORITY_ACTION,
    MANIFEST_MIGRATION_AUTHORITY_ACTION,
    MANIFEST_MIGRATION_AUTHORITY_EVIDENCE_SCHEMA,
    MANIFEST_MIGRATION_AUTHORITY_SCHEMA,
    MANIFEST_MIGRATION_AUTHORITY_SCOPE,
    AttachmentAuthorityError,
    AttachmentCapabilityAuthorityVerifier,
    ManifestMigrationAuthorityVerifier,
    build_attachment_authority_intent,
    build_manifest_migration_authority_intent,
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


def _migration_intent() -> dict[str, object]:
    return build_manifest_migration_authority_intent(
        request_id="reality.migrate.1234",
        identity_fingerprint="sha256:" + "4" * 64,
        connector_id="test.connector",
        expected_manifest_sha256="sha256:" + "5" * 64,
        new_manifest_sha256="sha256:" + "6" * 64,
        persistent=True,
    )


def _issue_migration(
    intent: dict[str, object],
    receipts: ReceiptSource,
    receipt_id: str = "will_migration_1",
):
    receipts.add(receipt_id)
    return get_capability_issuer().issue_from_decision(
        Decision(receipt_id),
        action=MANIFEST_MIGRATION_AUTHORITY_ACTION,
        payload=intent,
        scope=str(intent["scope"]),
    )


def _digest(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _rehash_evidence(evidence: dict[str, object]) -> dict[str, object]:
    body = {key: value for key, value in evidence.items() if key != "evidence_sha256"}
    evidence["evidence_sha256"] = _digest(body)
    return evidence


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


def test_persisted_authority_revalidates_current_will_material_after_restart() -> None:
    receipts = ReceiptSource()
    intent = _intent()
    verifier = AttachmentCapabilityAuthorityVerifier(will_receipts=receipts)
    evidence = dict(
        verifier.verify(_issue(intent, receipts), intent=intent, persistent=True)
    )

    restarted_verifier = AttachmentCapabilityAuthorityVerifier(will_receipts=receipts)

    assert restarted_verifier.validate_persisted(
        copy.deepcopy(evidence),
        intent=intent,
        persistent=True,
    ) == evidence


def test_persisted_authority_rejects_rehashed_will_material_substitution() -> None:
    receipts = ReceiptSource()
    intent = _intent()
    verifier = AttachmentCapabilityAuthorityVerifier(will_receipts=receipts)
    evidence = copy.deepcopy(
        dict(verifier.verify(_issue(intent, receipts), intent=intent, persistent=True))
    )
    evidence["will_receipt"]["signature"] = "ff" * 64
    material_body = {
        key: evidence["will_receipt"][key]
        for key in ("receipt_id", "payload", "signature", "signature_scheme")
    }
    evidence["will_receipt"]["material_sha256"] = _digest(material_body)
    _rehash_evidence(evidence)

    restarted_verifier = AttachmentCapabilityAuthorityVerifier(will_receipts=receipts)
    with pytest.raises(AttachmentAuthorityError, match="persisted_will_material_invalid"):
        restarted_verifier.validate_persisted(
            evidence,
            intent=intent,
            persistent=True,
        )


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


def test_manifest_migration_exact_contract_is_one_use_and_persistently_verifiable() -> None:
    receipts = ReceiptSource()
    intent = _migration_intent()
    capability = _issue_migration(intent, receipts)
    verifier = AttachmentCapabilityAuthorityVerifier(will_receipts=receipts)

    assert isinstance(verifier, ManifestMigrationAuthorityVerifier)
    assert set(intent) == {
        "schema",
        "action",
        "request_id",
        "identity_fingerprint",
        "connector_id",
        "expected_manifest_sha256",
        "new_manifest_sha256",
        "persistent",
        "scope",
    }
    assert intent["schema"] == MANIFEST_MIGRATION_AUTHORITY_SCHEMA
    assert intent["action"] == MANIFEST_MIGRATION_AUTHORITY_ACTION
    assert intent["scope"] == MANIFEST_MIGRATION_AUTHORITY_SCOPE

    evidence = verifier.verify_manifest_migration(
        capability,
        intent=intent,
        persistent=True,
    )

    assert evidence["schema"] == MANIFEST_MIGRATION_AUTHORITY_EVIDENCE_SCHEMA
    assert evidence["migration"] == {
        "request_id": "reality.migrate.1234",
        "identity_fingerprint": "sha256:" + "4" * 64,
        "connector_id": "test.connector",
        "expected_manifest_sha256": "sha256:" + "5" * 64,
        "new_manifest_sha256": "sha256:" + "6" * 64,
        "persistent": True,
    }
    assert evidence["nonce_consumption"]["consumed"] is True
    assert evidence["nonce_consumption"]["ledger_durable"] is True
    assert verifier.validate_persisted_manifest_migration(
        evidence,
        intent=intent,
        persistent=True,
    ) == evidence

    with pytest.raises(AttachmentAuthorityError, match="capability_replayed"):
        verifier.verify_manifest_migration(
            capability,
            intent=intent,
            persistent=True,
        )


@pytest.mark.parametrize(
    ("field", "replacement", "persistent"),
    [
        ("request_id", "reality.migrate.other", True),
        ("identity_fingerprint", "sha256:" + "a" * 64, True),
        ("connector_id", "other.connector", True),
        ("expected_manifest_sha256", "sha256:" + "b" * 64, True),
        ("new_manifest_sha256", "sha256:" + "c" * 64, True),
        ("persistent", False, False),
    ],
)
def test_manifest_migration_substitution_cannot_reuse_capability(
    field: str,
    replacement: object,
    persistent: bool,
) -> None:
    receipts = ReceiptSource()
    intent = _migration_intent()
    capability = _issue_migration(intent, receipts)
    attacked = dict(intent)
    attacked[field] = replacement
    verifier = AttachmentCapabilityAuthorityVerifier(will_receipts=receipts)

    with pytest.raises(AttachmentAuthorityError, match="capability_action_mismatch"):
        verifier.verify_manifest_migration(
            capability,
            intent=attacked,
            persistent=persistent,
        )

    # A rejected substitution must not burn the valid transition authority.
    assert verifier.verify_manifest_migration(
        capability,
        intent=intent,
        persistent=True,
    )["action_digest"] == capability.action_digest


def test_manifest_migration_scope_is_independently_checked() -> None:
    receipts = ReceiptSource()
    intent = _migration_intent()
    receipts.add("will_migration_scope")
    capability = get_capability_issuer().issue_from_decision(
        Decision("will_migration_scope"),
        action=MANIFEST_MIGRATION_AUTHORITY_ACTION,
        payload=intent,
        scope="reality_attachment.control",
    )
    verifier = AttachmentCapabilityAuthorityVerifier(will_receipts=receipts)

    with pytest.raises(AttachmentAuthorityError, match="capability_scope_mismatch"):
        verifier.verify_manifest_migration(
            capability,
            intent=intent,
            persistent=True,
        )


@pytest.mark.parametrize(
    "request_id",
    ["", "A-not-canonical", "x" * 129, "migration request", "migration\nrequest"],
)
def test_manifest_migration_request_identifier_is_bounded(request_id: str) -> None:
    with pytest.raises(ValueError, match="request_id"):
        build_manifest_migration_authority_intent(
            request_id=request_id,
            identity_fingerprint="sha256:" + "4" * 64,
            connector_id="test.connector",
            expected_manifest_sha256="sha256:" + "5" * 64,
            new_manifest_sha256="sha256:" + "6" * 64,
            persistent=True,
        )


def test_manifest_migration_rejects_noop_and_non_boolean_persistence() -> None:
    common = {
        "request_id": "reality.migrate.1234",
        "identity_fingerprint": "sha256:" + "4" * 64,
        "connector_id": "test.connector",
        "expected_manifest_sha256": "sha256:" + "5" * 64,
        "new_manifest_sha256": "sha256:" + "5" * 64,
    }
    with pytest.raises(ValueError, match="must change"):
        build_manifest_migration_authority_intent(**common, persistent=True)
    common["new_manifest_sha256"] = "sha256:" + "6" * 64
    with pytest.raises(TypeError, match="boolean"):
        build_manifest_migration_authority_intent(**common, persistent=1)  # type: ignore[arg-type]


def test_manifest_migration_intent_rejects_extra_fields() -> None:
    receipts = ReceiptSource()
    intent = _migration_intent()
    capability = _issue_migration(intent, receipts)
    attacked = dict(intent)
    attacked["unbound_override"] = True
    verifier = AttachmentCapabilityAuthorityVerifier(will_receipts=receipts)

    with pytest.raises(AttachmentAuthorityError, match="intent_shape_invalid"):
        verifier.verify_manifest_migration(
            capability,
            intent=attacked,
            persistent=True,
        )


def test_manifest_migration_persisted_capability_signature_tamper_is_refused() -> None:
    receipts = ReceiptSource()
    intent = _migration_intent()
    capability = _issue_migration(intent, receipts)
    verifier = AttachmentCapabilityAuthorityVerifier(will_receipts=receipts)
    evidence = copy.deepcopy(
        verifier.verify_manifest_migration(capability, intent=intent, persistent=True)
    )
    evidence["capability"]["signature"] = "00" * 64
    _rehash_evidence(evidence)

    with pytest.raises(AttachmentAuthorityError, match="persisted_capability_invalid"):
        verifier.validate_persisted_manifest_migration(
            evidence,
            intent=intent,
            persistent=True,
        )


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("action_digest", "0" * 64),
        ("scope", "reality_attachment.control"),
        ("persistent", False),
    ],
)
def test_manifest_migration_persisted_contract_tamper_is_refused(
    field: str,
    replacement: object,
) -> None:
    receipts = ReceiptSource()
    intent = _migration_intent()
    verifier = AttachmentCapabilityAuthorityVerifier(will_receipts=receipts)
    evidence = copy.deepcopy(
        verifier.verify_manifest_migration(
            _issue_migration(intent, receipts),
            intent=intent,
            persistent=True,
        )
    )
    evidence[field] = replacement
    _rehash_evidence(evidence)

    with pytest.raises(AttachmentAuthorityError, match="evidence_invalid"):
        verifier.validate_persisted_manifest_migration(
            evidence,
            intent=intent,
            persistent=True,
        )


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("identity_fingerprint", "sha256:" + "d" * 64),
        ("connector_id", "forged.connector"),
        ("expected_manifest_sha256", "sha256:" + "e" * 64),
        ("new_manifest_sha256", "sha256:" + "f" * 64),
    ],
)
def test_manifest_migration_persisted_binding_tamper_is_refused(
    field: str,
    replacement: str,
) -> None:
    receipts = ReceiptSource()
    intent = _migration_intent()
    verifier = AttachmentCapabilityAuthorityVerifier(will_receipts=receipts)
    evidence = copy.deepcopy(
        verifier.verify_manifest_migration(
            _issue_migration(intent, receipts),
            intent=intent,
            persistent=True,
        )
    )
    evidence["migration"][field] = replacement
    _rehash_evidence(evidence)

    with pytest.raises(AttachmentAuthorityError, match="evidence_invalid"):
        verifier.validate_persisted_manifest_migration(
            evidence,
            intent=intent,
            persistent=True,
        )


def test_manifest_migration_persisted_expiry_is_checked_at_verification_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipts = ReceiptSource()
    intent = _migration_intent()
    capability = _issue_migration(intent, receipts)
    verifier = AttachmentCapabilityAuthorityVerifier(will_receipts=receipts)
    evidence = copy.deepcopy(
        verifier.verify_manifest_migration(capability, intent=intent, persistent=True)
    )
    evidence["verified_at_ns"] = int(capability.expires_at * 1_000_000_000)
    evidence["capability_expires_at_ns"] = evidence["verified_at_ns"] + 1
    evidence["nonce_consumption"]["checked_at_ns"] = evidence["verified_at_ns"]
    evidence["nonce_consumption"]["expires_at_ns"] = evidence[
        "capability_expires_at_ns"
    ]
    _rehash_evidence(evidence)
    monkeypatch.setattr(
        authority_module.time,
        "time_ns",
        lambda: int(evidence["verified_at_ns"]),
    )

    with pytest.raises(AttachmentAuthorityError, match="persisted_capability_invalid"):
        verifier.validate_persisted_manifest_migration(
            evidence,
            intent=intent,
            persistent=True,
        )


def test_manifest_migration_persisted_nonce_marker_tamper_is_refused() -> None:
    receipts = ReceiptSource()
    intent = _migration_intent()
    verifier = AttachmentCapabilityAuthorityVerifier(will_receipts=receipts)
    evidence = copy.deepcopy(
        verifier.verify_manifest_migration(
            _issue_migration(intent, receipts),
            intent=intent,
            persistent=True,
        )
    )
    evidence["nonce_consumption"]["consumed"] = False
    _rehash_evidence(evidence)

    with pytest.raises(AttachmentAuthorityError, match="persisted_nonce_invalid"):
        verifier.validate_persisted_manifest_migration(
            evidence,
            intent=intent,
            persistent=True,
        )


def test_manifest_migration_persisted_validation_requires_live_nonce_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipts = ReceiptSource()
    intent = _migration_intent()
    verifier = AttachmentCapabilityAuthorityVerifier(will_receipts=receipts)
    evidence = verifier.verify_manifest_migration(
        _issue_migration(intent, receipts),
        intent=intent,
        persistent=True,
    )

    class MissingNonceLedger:
        @staticmethod
        def seen(_nonce: str) -> bool:
            return False

        @staticmethod
        def status() -> dict[str, object]:
            return {"healthy": True}

    monkeypatch.setattr(authority_module, "get_nonce_ledger", lambda: MissingNonceLedger())
    with pytest.raises(AttachmentAuthorityError, match="persisted_nonce_missing"):
        verifier.validate_persisted_manifest_migration(
            evidence,
            intent=intent,
            persistent=True,
        )


def test_manifest_migration_persisted_will_material_tamper_is_refused() -> None:
    receipts = ReceiptSource()
    intent = _migration_intent()
    verifier = AttachmentCapabilityAuthorityVerifier(will_receipts=receipts)
    evidence = copy.deepcopy(
        verifier.verify_manifest_migration(
            _issue_migration(intent, receipts),
            intent=intent,
            persistent=True,
        )
    )
    evidence["will_receipt"]["signature"] = "ff" * 64
    _rehash_evidence(evidence)

    with pytest.raises(AttachmentAuthorityError, match="persisted_will_material_invalid"):
        verifier.validate_persisted_manifest_migration(
            evidence,
            intent=intent,
            persistent=True,
        )


def test_manifest_migration_persistent_trust_requires_durable_asymmetric_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AURA_CAPABILITY_FORCE_HMAC", "1")
    reset_capability_chain()
    receipts = ReceiptSource()
    intent = _migration_intent()
    verifier = AttachmentCapabilityAuthorityVerifier(will_receipts=receipts)

    with pytest.raises(AttachmentAuthorityError, match="persistent_root_not_durable"):
        verifier.verify_manifest_migration(
            _issue_migration(intent, receipts),
            intent=intent,
            persistent=True,
        )
