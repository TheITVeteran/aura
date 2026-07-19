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
    build_role_attestation,
    validate_campaign_trust_policy,
)
from core.brain.llm.latent_cortex.worker_origin import (
    WORKER_KEY_CUSTODY_PRODUCER_SOFTWARE,
    ZERO_SHA256,
    WorkerOriginError,
    build_worker_authorization_payload,
    build_worker_result_origin,
    verify_worker_authorization,
    verify_worker_result_origin,
)

SIGNED_AT = 1_800_000_150
BOOT_ID = "a" * 32
CELL_ID = "cell-0001"
ATTEMPT_ID = "attempt-0001"


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


def _pin(role: str, key: Ed25519PrivateKey) -> dict[str, str]:
    raw = _public_raw(key)
    return {
        "signer_id": f"{role}-signer",
        "organization_id": f"{role}-organization",
        "public_key_b64": base64.b64encode(raw).decode("ascii"),
        "key_id": hashlib.sha256(raw).hexdigest(),
        "implementation_sha256": hashlib.sha256(f"{role}:impl".encode()).hexdigest(),
        "release_sha256": hashlib.sha256(f"{role}:release".encode()).hexdigest(),
        "custody_class": "test_fixture",
        "custody_evidence_sha256": hashlib.sha256(
            f"{role}:custody".encode()
        ).hexdigest(),
    }


def _fixture():
    root = Ed25519PrivateKey.generate()
    role_keys = {
        role: Ed25519PrivateKey.generate() for role in CAMPAIGN_TRUST_ROLES
    }
    body = {
        "schema": CAMPAIGN_TRUST_POLICY_SCHEMA,
        "policy_id": "resident-32b-worker-origin",
        "policy_revision": 1,
        "campaign_name": "resident-32b-confirmatory",
        "protocol_sha256": "1" * 64,
        "previous_policy_sha256": None,
        "revoked_key_ids": [],
        "issued_at_unix": 1_800_000_000,
        "not_before_unix": 1_800_000_100,
        "expires_at_unix": 1_800_086_400,
        "roles": {
            role: _pin(role, role_keys[role]) for role in CAMPAIGN_TRUST_ROLES
        },
    }
    signed = canonical_json_bytes(body)
    root_raw = _public_raw(root)
    policy_document = {
        **body,
        "root_signature": {
            "algorithm": "Ed25519",
            "key_id": hashlib.sha256(root_raw).hexdigest(),
            "signature_b64": base64.b64encode(root.sign(signed)).decode("ascii"),
            "signed_payload_sha256": hashlib.sha256(signed).hexdigest(),
        },
    }
    policy = validate_campaign_trust_policy(
        policy_document,
        trusted_root_public_key_pem=_public_pem(root),
        now_unix=1_800_000_200,
    )
    worker_key = Ed25519PrivateKey.generate()
    authorization = build_worker_authorization_payload(
        campaign_name="resident-32b-confirmatory",
        policy_sha256=policy.policy_sha256,
        protocol_sha256="1" * 64,
        plan_sha256="2" * 64,
        arm="adapter_rlc",
        worker_attempt_slot=1,
        worker_boot_id=BOOT_ID,
        worker_key_custody=WORKER_KEY_CUSTODY_PRODUCER_SOFTWARE,
        worker_source_sha256="3" * 64,
        worker_command=["python", "worker.py", "--worker-arm", "adapter_rlc"],
        model_identity_sha256="5" * 64,
        adapter_identity_sha256="6" * 64,
        worker_public_key_raw=_public_raw(worker_key),
    )
    attestation = build_role_attestation(
        policy,
        role=CAMPAIGN_RUNNER,
        payload=authorization,
        signed_at_unix=SIGNED_AT,
        private_key=role_keys[CAMPAIGN_RUNNER],
    )
    return policy, role_keys, worker_key, authorization, attestation


def _result_body(*, boot_id: str = BOOT_ID, text: str = "FINAL_ANSWER: 42"):
    return {
        "arm": "adapter_rlc",
        "text": text,
        "output_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "runtime_model_identity": {
            "worker_boot_id": boot_id,
            "worker_model_path": "/resident/32b",
        },
    }


def _signed_result(
    worker_key,
    authorization,
    attestation,
    *,
    body=None,
    sequence: int = 1,
    previous: str = ZERO_SHA256,
    boot_id: str = BOOT_ID,
):
    result = body or _result_body()
    origin = build_worker_result_origin(
        authorization_attestation=attestation,
        authorization_payload=authorization,
        private_key=worker_key,
        result_body=result,
        cell_id=CELL_ID,
        attempt_id=ATTEMPT_ID,
        worker_boot_id=boot_id,
        sequence=sequence,
        previous_origin_sha256=previous,
    )
    return {**result, "worker_origin": origin}


def _verify(policy, authorization, attestation, result, **overrides):
    arguments = {
        "expected_cell_id": CELL_ID,
        "expected_attempt_id": ATTEMPT_ID,
        "expected_sequence": 1,
        "expected_previous_origin_sha256": ZERO_SHA256,
        **overrides,
    }
    return verify_worker_result_origin(
        policy,
        authorization_attestation=attestation,
        expected_authorization_payload=authorization,
        result=result,
        **arguments,
    )


def _refresh_origin_digest(result: dict) -> None:
    material = dict(result["worker_origin"])
    material.pop("origin_sha256")
    result["worker_origin"]["origin_sha256"] = hashlib.sha256(
        canonical_json_bytes(material)
    ).hexdigest()


def _resign_origin(result: dict, worker_key: Ed25519PrivateKey) -> None:
    origin = result["worker_origin"]
    signed_payload = origin["signed_payload"]
    result_body = dict(result)
    result_body.pop("worker_origin")
    signed_payload["result_body_sha256"] = hashlib.sha256(
        canonical_json_bytes(result_body)
    ).hexdigest()
    signed_bytes = canonical_json_bytes(signed_payload)
    origin["signed_payload_sha256"] = hashlib.sha256(signed_bytes).hexdigest()
    origin["signature_b64"] = base64.b64encode(
        worker_key.sign(signed_bytes)
    ).decode("ascii")
    _refresh_origin_digest(result)


def test_authorized_worker_result_chain_round_trip():
    policy, _role_keys, worker_key, authorization, attestation = _fixture()
    verified = verify_worker_authorization(
        policy,
        attestation,
        expected_payload=authorization,
        not_after_unix=SIGNED_AT,
    )
    assert verified["payload"] == authorization

    first = _signed_result(worker_key, authorization, attestation)
    first_payload = _verify(policy, authorization, attestation, first)
    assert first_payload["sequence"] == 1
    assert first_payload["previous_origin_sha256"] == ZERO_SHA256
    assert first_payload["worker_attempt_slot"] == 1
    assert first_payload["worker_boot_id"] == BOOT_ID

    previous = first["worker_origin"]["origin_sha256"]
    second = _signed_result(
        worker_key,
        authorization,
        attestation,
        sequence=2,
        previous=previous,
    )
    second_payload = _verify(
        policy,
        authorization,
        attestation,
        second,
        expected_sequence=2,
        expected_previous_origin_sha256=previous,
    )
    assert second_payload["previous_origin_sha256"] == previous


@pytest.mark.parametrize(
    ("override", "value"),
    [
        ("expected_cell_id", "wrong-cell"),
        ("expected_attempt_id", "wrong-attempt"),
        ("expected_sequence", 2),
        ("expected_previous_origin_sha256", "f" * 64),
    ],
)
def test_result_rejects_wrong_chain_position(override: str, value):
    policy, _role_keys, worker_key, authorization, attestation = _fixture()
    result = _signed_result(worker_key, authorization, attestation)
    with pytest.raises(WorkerOriginError, match="worker_result_binding_invalid"):
        _verify(policy, authorization, attestation, result, **{override: value})


def test_result_rejects_boolean_expected_sequence_type_confusion():
    policy, _role_keys, worker_key, authorization, attestation = _fixture()
    result = _signed_result(worker_key, authorization, attestation)
    with pytest.raises(
        WorkerOriginError, match="worker_result_expected_sequence_invalid"
    ):
        _verify(
            policy,
            authorization,
            attestation,
            result,
            expected_sequence=True,
        )


def test_result_rejects_body_and_boot_identity_tampering():
    policy, _role_keys, worker_key, authorization, attestation = _fixture()
    result = _signed_result(worker_key, authorization, attestation)
    attacked = copy.deepcopy(result)
    attacked["text"] = "tampered"
    with pytest.raises(WorkerOriginError, match="worker_result_binding_invalid"):
        _verify(policy, authorization, attestation, attacked)

    mismatched_boot = _signed_result(
        worker_key,
        authorization,
        attestation,
        body=_result_body(boot_id="b" * 32),
        boot_id=BOOT_ID,
    )
    with pytest.raises(
        WorkerOriginError, match="worker_result_boot_identity_mismatch"
    ):
        _verify(policy, authorization, attestation, mismatched_boot)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("worker_boot_id", "not-a-128-bit-id", "worker_boot_id_invalid"),
        ("worker_attempt_slot", True, "worker_attempt_slot_invalid"),
        ("sequence", True, "worker_result_sequence_invalid"),
    ],
)
def test_result_rejects_invalid_validly_signed_field_types(
    field: str,
    value,
    error: str,
):
    policy, _role_keys, worker_key, authorization, attestation = _fixture()
    result = _signed_result(worker_key, authorization, attestation)
    result["worker_origin"]["signed_payload"][field] = value
    if field == "worker_boot_id":
        result["runtime_model_identity"]["worker_boot_id"] = value
    _resign_origin(result, worker_key)

    with pytest.raises(WorkerOriginError, match=error):
        _verify(policy, authorization, attestation, result)


def test_result_builder_rejects_noncanonical_boot_id():
    _policy, _role_keys, worker_key, authorization, attestation = _fixture()
    with pytest.raises(WorkerOriginError, match="worker_boot_id_invalid"):
        _signed_result(
            worker_key,
            authorization,
            attestation,
            boot_id="NOT-CANONICAL",
        )

    with pytest.raises(
        WorkerOriginError, match="worker_boot_id_authorization_mismatch"
    ):
        _signed_result(
            worker_key,
            authorization,
            attestation,
            body=_result_body(boot_id="b" * 32),
            boot_id="b" * 32,
        )


def test_result_rejects_origin_digest_signature_and_payload_digest_tampering():
    policy, _role_keys, worker_key, authorization, attestation = _fixture()
    result = _signed_result(worker_key, authorization, attestation)

    attacked = copy.deepcopy(result)
    attacked["worker_origin"]["origin_sha256"] = "0" * 64
    with pytest.raises(WorkerOriginError, match="worker_result_origin_digest_invalid"):
        _verify(policy, authorization, attestation, attacked)

    attacked = copy.deepcopy(result)
    attacked["worker_origin"]["signature_b64"] = base64.b64encode(
        b"x" * 63
    ).decode()
    _refresh_origin_digest(attacked)
    with pytest.raises(WorkerOriginError, match="worker_result_signature_invalid"):
        _verify(policy, authorization, attestation, attacked)

    attacked = copy.deepcopy(result)
    attacked["worker_origin"]["signed_payload_sha256"] = "0" * 64
    _refresh_origin_digest(attacked)
    with pytest.raises(
        WorkerOriginError, match="worker_result_signed_payload_digest_invalid"
    ):
        _verify(policy, authorization, attestation, attacked)


def test_result_rejects_wrong_ephemeral_key_and_authorization_key():
    policy, _role_keys, worker_key, authorization, attestation = _fixture()
    with pytest.raises(WorkerOriginError, match="worker_private_key_mismatch"):
        _signed_result(
            Ed25519PrivateKey.generate(), authorization, attestation
        )

    attacked = copy.deepcopy(authorization)
    attacked["worker_key_id"] = "0" * 64
    with pytest.raises(
        WorkerOriginError, match="worker_authorization_key_mismatch"
    ):
        verify_worker_authorization(
            policy,
            attestation,
            expected_payload=attacked,
        )


def test_worker_authorization_rejects_role_substitution_and_late_signature():
    policy, role_keys, _worker_key, authorization, attestation = _fixture()
    issuer_attestation = build_role_attestation(
        policy,
        role=TASK_ISSUER,
        payload=authorization,
        signed_at_unix=SIGNED_AT,
        private_key=role_keys[TASK_ISSUER],
    )
    with pytest.raises(
        WorkerOriginError, match="worker_authorization_attestation_invalid"
    ):
        verify_worker_authorization(
            policy,
            issuer_attestation,
            expected_payload=authorization,
        )

    result = _signed_result(
        _worker_key,
        authorization,
        attestation,
    )
    with pytest.raises(
        WorkerOriginError, match="worker_authorization_attestation_invalid"
    ):
        _verify(
            policy,
            authorization,
            attestation,
            result,
            authorization_not_after_unix=SIGNED_AT - 1,
        )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload.__setitem__("schema", "wrong"),
        lambda payload: payload.__setitem__("plan_sha256", "short"),
        lambda payload: payload.__setitem__("arm", " adapter_rlc"),
        lambda payload: payload.__setitem__(
            "worker_key_custody", "worker_process_memory_nonexported"
        ),
        lambda payload: payload.__setitem__("worker_public_key_b64", "not-base64"),
        lambda payload: payload.__setitem__("unexpected", True),
    ],
)
def test_worker_authorization_rejects_malformed_payload(mutation):
    policy, _role_keys, _worker_key, authorization, attestation = _fixture()
    attacked = copy.deepcopy(authorization)
    mutation(attacked)
    with pytest.raises(WorkerOriginError):
        verify_worker_authorization(
            policy,
            attestation,
            expected_payload=attacked,
        )
