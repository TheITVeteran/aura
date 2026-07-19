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
    TASK_ISSUER,
    CampaignTrustError,
    build_role_attestation,
    policy_signed_payload,
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
    }


def _policy_fixture():
    root = Ed25519PrivateKey.generate()
    role_keys = {
        role: Ed25519PrivateKey.generate() for role in CAMPAIGN_TRUST_ROLES
    }
    body = {
        "schema": CAMPAIGN_TRUST_POLICY_SCHEMA,
        "policy_id": "resident-32b-confirmatory-2026-07",
        "campaign_name": "resident-32b-confirmatory",
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
