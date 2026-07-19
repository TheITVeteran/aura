from __future__ import annotations

import base64
import copy
import hashlib

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from core.brain.llm.latent_cortex.campaign_journal import canonical_json_bytes
from core.brain.llm.latent_cortex.campaign_trust import (
    CAMPAIGN_RUNNER,
    CAMPAIGN_TRUST_POLICY_SCHEMA,
    CAMPAIGN_TRUST_ROLES,
    EVIDENCE_VERIFIER,
    TASK_ISSUER,
    CampaignTrustError,
    assemble_role_attestation,
    assemble_signed_campaign_policy,
    build_role_attestation,
    externally_custodied_roles,
    policy_signed_payload,
    prepare_policy_signature_request,
    prepare_role_signature_request,
    validate_campaign_trust_policy,
    verify_role_attestation,
)


def _public_raw(key: Ed25519PrivateKey) -> bytes:
    return key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )


def _public_pem(key: Ed25519PrivateKey) -> bytes:
    return key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )


def _pin(
    role: str,
    key: Ed25519PrivateKey,
    *,
    organization_id: str | None = None,
) -> dict[str, str]:
    raw = _public_raw(key)
    return {
        "signer_id": f"{role}-signer",
        "organization_id": organization_id or f"{role}-organization",
        "public_key_b64": base64.b64encode(raw).decode("ascii"),
        "key_id": hashlib.sha256(raw).hexdigest(),
        "implementation_sha256": hashlib.sha256(f"{role}:impl".encode()).hexdigest(),
        "release_sha256": hashlib.sha256(f"{role}:release".encode()).hexdigest(),
        "custody_class": "test_fixture",
        "custody_evidence_sha256": hashlib.sha256(
            f"{role}:custody".encode()
        ).hexdigest(),
    }


def _policy_fixture():
    root = Ed25519PrivateKey.generate()
    role_keys = {
        role: Ed25519PrivateKey.generate() for role in CAMPAIGN_TRUST_ROLES
    }
    body = {
        "schema": CAMPAIGN_TRUST_POLICY_SCHEMA,
        "policy_id": "resident-32b-confirmatory-2026-07",
        "policy_revision": 1,
        "campaign_name": "resident-32b-confirmatory",
        "protocol_sha256": "9" * 64,
        "previous_policy_sha256": None,
        "revoked_key_ids": [],
        "issued_at_unix": 1_800_000_000,
        "not_before_unix": 1_800_000_100,
        "expires_at_unix": 1_800_086_400,
        "roles": {role: _pin(role, role_keys[role]) for role in CAMPAIGN_TRUST_ROLES},
    }
    signed = canonical_json_bytes(body)
    root_raw = _public_raw(root)
    policy = {
        **body,
        "root_signature": {
            "algorithm": "Ed25519",
            "key_id": hashlib.sha256(root_raw).hexdigest(),
            "signature_b64": base64.b64encode(root.sign(signed)).decode("ascii"),
            "signed_payload_sha256": hashlib.sha256(signed).hexdigest(),
        },
    }
    return policy, root, role_keys


def _resign(policy: dict, root: Ed25519PrivateKey) -> None:
    signed = canonical_json_bytes(policy_signed_payload(policy))
    policy["root_signature"]["signature_b64"] = base64.b64encode(
        root.sign(signed)
    ).decode("ascii")
    policy["root_signature"]["signed_payload_sha256"] = hashlib.sha256(
        signed
    ).hexdigest()


def test_policy_requires_external_root_and_four_independent_roles():
    policy, root, _role_keys = _policy_fixture()

    verified = validate_campaign_trust_policy(
        policy,
        trusted_root_public_key_pem=_public_pem(root),
        expected_campaign_name="resident-32b-confirmatory",
        now_unix=1_800_000_200,
    )

    assert verified.document == policy
    assert verified.policy_sha256 == hashlib.sha256(
        canonical_json_bytes(policy)
    ).hexdigest()
    assert verified.root_key_id == hashlib.sha256(_public_raw(root)).hexdigest()
    assert set(verified.document["roles"]) == set(CAMPAIGN_TRUST_ROLES)
    assert externally_custodied_roles(verified) is False


def test_policy_detached_signature_request_round_trip():
    policy, root, _role_keys = _policy_fixture()
    unsigned = policy_signed_payload(policy)
    request = prepare_policy_signature_request(
        unsigned,
        trusted_root_public_key_pem=_public_pem(root),
        expected_campaign_name="resident-32b-confirmatory",
        expected_protocol_sha256="9" * 64,
        now_unix=1_800_000_200,
    )
    signature = root.sign(base64.b64decode(request["signed_payload_b64"]))

    verified = assemble_signed_campaign_policy(
        request,
        signature_b64=base64.b64encode(signature).decode("ascii"),
        trusted_root_public_key_pem=_public_pem(root),
        expected_campaign_name="resident-32b-confirmatory",
        expected_protocol_sha256="9" * 64,
        now_unix=1_800_000_200,
    )

    assert verified.document == policy
    assert request["signed_payload"] == unsigned
    assert request["signed_payload_sha256"] == hashlib.sha256(
        canonical_json_bytes(unsigned)
    ).hexdigest()


def test_policy_detached_request_rejects_tampering_and_wrong_signature():
    policy, root, _role_keys = _policy_fixture()
    request = prepare_policy_signature_request(
        policy_signed_payload(policy),
        trusted_root_public_key_pem=_public_pem(root),
        now_unix=1_800_000_200,
    )
    attacked = copy.deepcopy(request)
    attacked["signed_payload"]["campaign_name"] = "attacked"
    with pytest.raises(
        CampaignTrustError, match="campaign_signature_request_payload_mismatch"
    ):
        assemble_signed_campaign_policy(
            attacked,
            signature_b64=base64.b64encode(b"x" * 64).decode("ascii"),
            trusted_root_public_key_pem=_public_pem(root),
            now_unix=1_800_000_200,
        )

    with pytest.raises(
        CampaignTrustError, match="campaign_trust_root_signature_invalid"
    ):
        assemble_signed_campaign_policy(
            request,
            signature_b64=base64.b64encode(b"x" * 64).decode("ascii"),
            trusted_root_public_key_pem=_public_pem(root),
            now_unix=1_800_000_200,
        )


def test_policy_rejects_bundle_selected_or_tampered_root():
    policy, _root, _role_keys = _policy_fixture()
    attacker = Ed25519PrivateKey.generate()

    with pytest.raises(CampaignTrustError, match="campaign_trust_root_key_mismatch"):
        validate_campaign_trust_policy(
            policy,
            trusted_root_public_key_pem=_public_pem(attacker),
            now_unix=1_800_000_200,
        )

    attacker_raw = _public_raw(attacker)
    policy["root_signature"]["key_id"] = hashlib.sha256(attacker_raw).hexdigest()
    signed = canonical_json_bytes(policy_signed_payload(policy))
    policy["root_signature"]["signature_b64"] = base64.b64encode(
        attacker.sign(signed)
    ).decode("ascii")
    policy["root_signature"]["signed_payload_sha256"] = hashlib.sha256(
        signed
    ).hexdigest()
    with pytest.raises(CampaignTrustError, match="campaign_trust_root_key_mismatch"):
        validate_campaign_trust_policy(
            policy,
            trusted_root_public_key_pem=_public_pem(_root),
            now_unix=1_800_000_200,
        )


@pytest.mark.parametrize("collision", ["signer", "organization", "key"])
def test_policy_rejects_role_collisions_even_when_root_signed(collision: str):
    policy, root, _role_keys = _policy_fixture()
    issuer = policy["roles"][TASK_ISSUER]
    runner = policy["roles"][CAMPAIGN_RUNNER]
    if collision == "signer":
        runner["signer_id"] = issuer["signer_id"]
    elif collision == "organization":
        runner["organization_id"] = issuer["organization_id"]
    else:
        runner["public_key_b64"] = issuer["public_key_b64"]
        runner["key_id"] = issuer["key_id"]
    _resign(policy, root)

    expected = {
        "signer": "campaign_trust_signer_identity_reused",
        "organization": "campaign_trust_organization_reused",
        "key": "campaign_trust_role_key_reused",
    }[collision]
    with pytest.raises(CampaignTrustError, match=expected):
        validate_campaign_trust_policy(
            policy,
            trusted_root_public_key_pem=_public_pem(root),
            now_unix=1_800_000_200,
        )


@pytest.mark.parametrize(
    ("now_unix", "error"),
    [
        (1_800_000_099, "campaign_trust_policy_not_yet_valid"),
        (1_800_086_400, "campaign_trust_policy_expired"),
    ],
)
def test_policy_enforces_time_window(now_unix: int, error: str):
    policy, root, _role_keys = _policy_fixture()

    with pytest.raises(CampaignTrustError, match=error):
        validate_campaign_trust_policy(
            policy,
            trusted_root_public_key_pem=_public_pem(root),
            now_unix=now_unix,
        )


def test_policy_enforces_protocol_revision_pin_and_revocations():
    policy, root, _role_keys = _policy_fixture()
    with pytest.raises(CampaignTrustError, match="campaign_trust_protocol_mismatch"):
        validate_campaign_trust_policy(
            policy,
            trusted_root_public_key_pem=_public_pem(root),
            expected_protocol_sha256="8" * 64,
            now_unix=1_800_000_200,
        )
    with pytest.raises(CampaignTrustError, match="campaign_trust_policy_rollback"):
        validate_campaign_trust_policy(
            policy,
            trusted_root_public_key_pem=_public_pem(root),
            minimum_policy_revision=2,
            now_unix=1_800_000_200,
        )

    policy["revoked_key_ids"] = [policy["roles"][TASK_ISSUER]["key_id"]]
    _resign(policy, root)
    with pytest.raises(CampaignTrustError, match="campaign_trust_revoked_role_key"):
        validate_campaign_trust_policy(
            policy,
            trusted_root_public_key_pem=_public_pem(root),
            now_unix=1_800_000_200,
        )


def test_external_custody_requires_every_role_to_have_external_evidence():
    policy, root, _role_keys = _policy_fixture()
    for pin in policy["roles"].values():
        pin["custody_class"] = "remote_hsm"
    _resign(policy, root)
    verified = validate_campaign_trust_policy(
        policy,
        trusted_root_public_key_pem=_public_pem(root),
        now_unix=1_800_000_200,
    )
    assert externally_custodied_roles(verified) is True


def test_role_attestation_is_policy_bound_and_payload_exact():
    policy, root, role_keys = _policy_fixture()
    verified = validate_campaign_trust_policy(
        policy,
        trusted_root_public_key_pem=_public_pem(root),
        now_unix=1_800_000_200,
    )
    payload = {
        "campaign_name": "resident-32b-confirmatory",
        "task_manifest_sha256": "a" * 64,
        "task_commitment_sha256": "b" * 64,
    }
    attestation = build_role_attestation(
        verified,
        role=TASK_ISSUER,
        payload=payload,
        signed_at_unix=1_800_000_150,
        private_key=role_keys[TASK_ISSUER],
    )

    signed_payload = verify_role_attestation(
        verified,
        attestation,
        role=TASK_ISSUER,
        expected_payload=payload,
        not_after_unix=1_800_000_200,
    )

    assert signed_payload["payload"] == payload
    assert signed_payload["policy_sha256"] == verified.policy_sha256
    assert signed_payload["signer_id"] == "task_issuer-signer"


def test_role_detached_signature_request_round_trip():
    policy, root, role_keys = _policy_fixture()
    verified = validate_campaign_trust_policy(
        policy,
        trusted_root_public_key_pem=_public_pem(root),
        now_unix=1_800_000_200,
    )
    payload = {"task_manifest_sha256": "a" * 64}
    request = prepare_role_signature_request(
        verified,
        role=TASK_ISSUER,
        payload=payload,
        signed_at_unix=1_800_000_150,
    )
    signature = role_keys[TASK_ISSUER].sign(
        base64.b64decode(request["signed_payload_b64"])
    )

    attestation = assemble_role_attestation(
        verified,
        request,
        signature_b64=base64.b64encode(signature).decode("ascii"),
        role=TASK_ISSUER,
    )

    assert verify_role_attestation(
        verified,
        attestation,
        role=TASK_ISSUER,
        expected_payload=payload,
    )["payload"] == payload


def test_role_detached_request_rejects_policy_key_or_role_substitution():
    policy, root, role_keys = _policy_fixture()
    verified = validate_campaign_trust_policy(
        policy,
        trusted_root_public_key_pem=_public_pem(root),
        now_unix=1_800_000_200,
    )
    request = prepare_role_signature_request(
        verified,
        role=TASK_ISSUER,
        payload={"task": "x"},
        signed_at_unix=1_800_000_150,
    )
    signature = role_keys[TASK_ISSUER].sign(
        base64.b64decode(request["signed_payload_b64"])
    )
    with pytest.raises(CampaignTrustError, match="campaign_signature_request_key_mismatch"):
        assemble_role_attestation(
            verified,
            request,
            signature_b64=base64.b64encode(signature).decode("ascii"),
            role=CAMPAIGN_RUNNER,
        )


@pytest.mark.parametrize("mutation", ["payload", "role", "signature", "policy"])
def test_role_attestation_rejects_tampering(mutation: str):
    policy, root, role_keys = _policy_fixture()
    verified = validate_campaign_trust_policy(
        policy,
        trusted_root_public_key_pem=_public_pem(root),
        now_unix=1_800_000_200,
    )
    payload = {"task_manifest_sha256": "a" * 64}
    attestation = build_role_attestation(
        verified,
        role=TASK_ISSUER,
        payload=payload,
        signed_at_unix=1_800_000_150,
        private_key=role_keys[TASK_ISSUER],
    )
    attacked = copy.deepcopy(attestation)
    if mutation == "payload":
        attacked["signed_payload"]["payload"]["task_manifest_sha256"] = "b" * 64
    elif mutation == "role":
        attacked["signed_payload"]["role"] = CAMPAIGN_RUNNER
    elif mutation == "signature":
        attacked["signature_b64"] = base64.b64encode(b"x" * 64).decode("ascii")
    else:
        attacked["signed_payload"]["policy_sha256"] = "0" * 64

    with pytest.raises(CampaignTrustError):
        verify_role_attestation(
            verified,
            attacked,
            role=TASK_ISSUER,
            expected_payload=payload,
        )


def test_attestation_rejects_wrong_private_key_and_stage_time():
    policy, root, role_keys = _policy_fixture()
    verified = validate_campaign_trust_policy(
        policy,
        trusted_root_public_key_pem=_public_pem(root),
        now_unix=1_800_000_200,
    )
    with pytest.raises(
        CampaignTrustError, match="campaign_attestation_private_key_mismatch"
    ):
        build_role_attestation(
            verified,
            role=TASK_ISSUER,
            payload={"task": "x"},
            signed_at_unix=1_800_000_150,
            private_key=role_keys[CAMPAIGN_RUNNER],
        )

    attestation = build_role_attestation(
        verified,
        role=TASK_ISSUER,
        payload={"task": "x"},
        signed_at_unix=1_800_000_150,
        private_key=role_keys[TASK_ISSUER],
    )
    with pytest.raises(CampaignTrustError, match="campaign_attestation_too_late"):
        verify_role_attestation(
            verified,
            attestation,
            role=TASK_ISSUER,
            expected_payload={"task": "x"},
            not_after_unix=1_800_000_149,
        )


def test_final_verifier_attestation_binds_sealed_evidence_after_prelaunch():
    policy, root, role_keys = _policy_fixture()
    verified = validate_campaign_trust_policy(
        policy,
        trusted_root_public_key_pem=_public_pem(root),
        now_unix=1_800_000_200,
    )
    payload = {
        "schema": "aura.latent_cortex.final_verifier_payload.v1",
        "campaign_manifest_sha256": "a" * 64,
        "published_grade_sha256": "b" * 64,
        "production_grade_implementation_sha256": "c" * 64,
        "independent_scoring_implementation_sha256": "d" * 64,
    }
    attestation = build_role_attestation(
        verified,
        role=EVIDENCE_VERIFIER,
        payload=payload,
        signed_at_unix=1_800_000_300,
        private_key=role_keys[EVIDENCE_VERIFIER],
    )

    signed = verify_role_attestation(
        verified,
        attestation,
        role=EVIDENCE_VERIFIER,
        expected_payload=payload,
        not_before_unix=1_800_000_250,
    )

    assert signed["payload"] == payload
