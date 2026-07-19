from __future__ import annotations

import base64
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from core.brain.llm.latent_cortex.campaign_journal import canonical_json_bytes
from core.brain.llm.latent_cortex.campaign_trust import (
    CAMPAIGN_TRUST_POLICY_SCHEMA,
    CAMPAIGN_TRUST_ROLES,
    TASK_ISSUER,
    validate_campaign_trust_policy,
    verify_role_attestation,
)
from tools.manage_campaign_trust import (
    CampaignTrustToolError,
    _atomic_create_or_verify,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
TOOL = REPO_ROOT / "tools/manage_campaign_trust.py"


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


def _write_json(path: Path, value: dict) -> None:
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def _unsigned_policy(role_keys: dict[str, Ed25519PrivateKey]) -> dict:
    roles = {}
    for role, key in role_keys.items():
        raw = _public_raw(key)
        roles[role] = {
            "signer_id": f"{role}-signer",
            "organization_id": f"{role}-organization",
            "public_key_b64": base64.b64encode(raw).decode("ascii"),
            "key_id": hashlib.sha256(raw).hexdigest(),
            "implementation_sha256": hashlib.sha256(f"{role}:impl".encode()).hexdigest(),
            "release_sha256": hashlib.sha256(f"{role}:release".encode()).hexdigest(),
            "custody_class": "external_service",
            "custody_evidence_sha256": hashlib.sha256(
                f"{role}:custody".encode()
            ).hexdigest(),
        }
    return {
        "schema": CAMPAIGN_TRUST_POLICY_SCHEMA,
        "policy_id": "detached-signing-test",
        "policy_revision": 1,
        "campaign_name": "resident-32b-confirmatory",
        "protocol_sha256": "9" * 64,
        "previous_policy_sha256": None,
        "revoked_key_ids": [],
        "issued_at_unix": 1_800_000_000,
        "not_before_unix": 1_800_000_100,
        "expires_at_unix": 1_800_086_400,
        "roles": roles,
    }


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(TOOL), *args],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )


def test_detached_cli_policy_and_role_round_trip(tmp_path: Path):
    root = Ed25519PrivateKey.generate()
    role_keys = {role: Ed25519PrivateKey.generate() for role in CAMPAIGN_TRUST_ROLES}
    unsigned_path = tmp_path / "unsigned-policy.json"
    root_path = tmp_path / "root.pem"
    policy_request_path = tmp_path / "policy-request.json"
    policy_signature_path = tmp_path / "policy-signature.json"
    policy_path = tmp_path / "policy.json"
    _write_json(unsigned_path, _unsigned_policy(role_keys))
    root_path.write_bytes(_public_pem(root))

    prepared = _run(
        "policy-request",
        "--unsigned-policy",
        str(unsigned_path),
        "--root",
        str(root_path),
        "--campaign-name",
        "resident-32b-confirmatory",
        "--protocol-sha256",
        "9" * 64,
        "--observed-at",
        "1800000200",
        "--out",
        str(policy_request_path),
    )
    assert prepared.returncode == 0, prepared.stdout + prepared.stderr
    request = json.loads(policy_request_path.read_bytes())
    signature = root.sign(base64.b64decode(request["signed_payload_b64"]))
    _write_json(
        policy_signature_path,
        {"signature_b64": base64.b64encode(signature).decode("ascii")},
    )

    assembled = _run(
        "policy-assemble",
        "--request",
        str(policy_request_path),
        "--root",
        str(root_path),
        "--signature",
        str(policy_signature_path),
        "--observed-at",
        "1800000200",
        "--out",
        str(policy_path),
    )
    assert assembled.returncode == 0, assembled.stdout + assembled.stderr
    verified = validate_campaign_trust_policy(
        json.loads(policy_path.read_bytes()),
        trusted_root_public_key_pem=root_path.read_bytes(),
        now_unix=1_800_000_200,
    )

    payload_path = tmp_path / "issuer-payload.json"
    role_request_path = tmp_path / "issuer-request.json"
    role_signature_path = tmp_path / "issuer-signature"
    attestation_path = tmp_path / "issuer-attestation.json"
    payload = {"task_manifest_sha256": "a" * 64}
    _write_json(payload_path, payload)
    role_request = _run(
        "role-request",
        "--policy",
        str(policy_path),
        "--root",
        str(root_path),
        "--role",
        TASK_ISSUER,
        "--payload",
        str(payload_path),
        "--signed-at",
        "1800000150",
        "--observed-at",
        "1800000200",
        "--out",
        str(role_request_path),
    )
    assert role_request.returncode == 0, role_request.stdout + role_request.stderr
    request = json.loads(role_request_path.read_bytes())
    role_signature_path.write_bytes(
        role_keys[TASK_ISSUER].sign(
            base64.b64decode(request["signed_payload_b64"])
        )
    )
    role_assemble = _run(
        "role-assemble",
        "--policy",
        str(policy_path),
        "--root",
        str(root_path),
        "--role",
        TASK_ISSUER,
        "--request",
        str(role_request_path),
        "--signature",
        str(role_signature_path),
        "--observed-at",
        "1800000200",
        "--out",
        str(attestation_path),
    )
    assert role_assemble.returncode == 0, role_assemble.stdout + role_assemble.stderr
    signed = verify_role_attestation(
        verified,
        json.loads(attestation_path.read_bytes()),
        role=TASK_ISSUER,
        expected_payload=payload,
    )
    assert signed["payload"] == payload


def test_detached_cli_rejects_wrong_signature_without_output(tmp_path: Path):
    root = Ed25519PrivateKey.generate()
    role_keys = {role: Ed25519PrivateKey.generate() for role in CAMPAIGN_TRUST_ROLES}
    unsigned_path = tmp_path / "unsigned.json"
    root_path = tmp_path / "root.pem"
    request_path = tmp_path / "request.json"
    signature_path = tmp_path / "wrong.sig"
    output_path = tmp_path / "policy.json"
    _write_json(unsigned_path, _unsigned_policy(role_keys))
    root_path.write_bytes(_public_pem(root))
    assert _run(
        "policy-request",
        "--unsigned-policy",
        str(unsigned_path),
        "--root",
        str(root_path),
        "--out",
        str(request_path),
    ).returncode == 0
    signature_path.write_bytes(b"x" * 64)

    rejected = _run(
        "policy-assemble",
        "--request",
        str(request_path),
        "--root",
        str(root_path),
        "--signature",
        str(signature_path),
        "--observed-at",
        "1800000200",
        "--out",
        str(output_path),
    )

    assert rejected.returncode == 1
    assert json.loads(rejected.stdout)["reason"] == "campaign_trust_root_signature_invalid"
    assert not output_path.exists()


def test_artifact_publish_never_overwrites_a_racing_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    destination = tmp_path / "attestation.json"

    def racing_link(
        _source: os.PathLike[str] | str,
        target: os.PathLike[str] | str,
        *,
        follow_symlinks: bool,
    ) -> None:
        assert follow_symlinks is False
        Path(target).write_bytes(b"independent-writer\n")
        raise FileExistsError

    monkeypatch.setattr(os, "link", racing_link)
    with pytest.raises(
        CampaignTrustToolError,
        match="refusing to overwrite a concurrently created artifact",
    ):
        _atomic_create_or_verify(destination, {"payload": "ours"})

    assert destination.read_bytes() == b"independent-writer\n"
    assert not list(tmp_path.glob(".*.tmp"))
