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
    MAX_WORKER_ALLOWED_CELLS,
    MAX_WORKER_PROTOCOL_VALUE_BYTES,
    WORKER_AUTHORIZATION_PAYLOAD_SCHEMA,
    WORKER_KEY_CUSTODY_DETACHED_SUPERVISOR,
    WORKER_KEY_CUSTODY_PRODUCER_SOFTWARE,
    ZERO_SHA256,
    WorkerOriginError,
    assemble_worker_lifecycle_event_origin,
    assemble_worker_result_origin,
    build_worker_authorization_payload,
    build_worker_lifecycle_event_payload,
    build_worker_result_signed_payload,
    compute_allowed_cell_digest,
    verify_worker_authorization,
    verify_worker_lifecycle_event_origin,
    verify_worker_result_origin,
)

SIGNED_AT = 1_800_000_150
SESSION_ID = "a" * 32
CELL_ID = "cell-0001"
CELL_TYPE = "reasoning"
ATTEMPT_ID = "attempt-0001"
ALLOWED_CELLS = [
    {"cell_id": CELL_ID, "cell_type": CELL_TYPE},
    {"cell_id": "cell-0002", "cell_type": "verification"},
]


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
    authorization = _authorization(policy, worker_key)
    attestation = build_role_attestation(
        policy,
        role=CAMPAIGN_RUNNER,
        payload=authorization,
        signed_at_unix=SIGNED_AT,
        private_key=role_keys[CAMPAIGN_RUNNER],
    )
    return policy, role_keys, worker_key, authorization, attestation


def _authorization(
    policy,
    worker_key: Ed25519PrivateKey,
    *,
    custody: str = WORKER_KEY_CUSTODY_DETACHED_SUPERVISOR,
    session_id: str = SESSION_ID,
):
    return build_worker_authorization_payload(
        campaign_name="resident-32b-confirmatory",
        policy_sha256=policy.policy_sha256,
        protocol_sha256="1" * 64,
        detached_plan_sha256="2" * 64,
        broker_policy_sha256="3" * 64,
        executable_binding_sha256="4" * 64,
        environment_sha256="5" * 64,
        sandbox_sha256="6" * 64,
        source_manifest_sha256="7" * 64,
        session_id=session_id,
        supervisor_attempt=1,
        arm="adapter_rlc",
        worker_attempt_slot=1,
        allowed_cell_digest=compute_allowed_cell_digest(ALLOWED_CELLS),
        model_identity_sha256="8" * 64,
        adapter_identity_sha256="9" * 64,
        worker_key_custody=custody,
        worker_public_key_raw=_public_raw(worker_key),
    )


def _result_body(*, text: str = "FINAL_ANSWER: 42"):
    return {
        "cell_id": CELL_ID,
        "cell_type": CELL_TYPE,
        "attempt_id": ATTEMPT_ID,
        "origin_session_id": SESSION_ID,
        "arm": "adapter_rlc",
        "text": text,
    }


def _signed_result(
    worker_key,
    authorization,
    attestation,
    *,
    body=None,
    sequence: int = 1,
    previous: str = ZERO_SHA256,
):
    result = body or _result_body()
    payload = build_worker_result_signed_payload(
        authorization_attestation=attestation,
        authorization_payload=authorization,
        result_body=result,
        cell_id=result["cell_id"],
        cell_type=result["cell_type"],
        attempt_id=result["attempt_id"],
        sequence=sequence,
        previous_origin_sha256=previous,
    )
    origin = assemble_worker_result_origin(
        payload,
        signature=worker_key.sign(canonical_json_bytes(payload)),
    )
    return {**result, "worker_origin": origin}


def _verify(policy, authorization, attestation, result, **overrides):
    arguments = {
        "expected_cell_id": CELL_ID,
        "expected_cell_type": CELL_TYPE,
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


def test_v4_authorization_and_v3_result_round_trip():
    policy, _role_keys, worker_key, authorization, attestation = _fixture()
    assert authorization["schema"] == WORKER_AUTHORIZATION_PAYLOAD_SCHEMA
    verified = verify_worker_authorization(
        policy,
        attestation,
        expected_payload=authorization,
        not_before_unix=SIGNED_AT,
        not_after_unix=SIGNED_AT,
    )
    assert verified["payload"] == authorization

    first = _signed_result(worker_key, authorization, attestation)
    first_payload = _verify(policy, authorization, attestation, first)
    assert first_payload["sequence"] == 1
    assert first_payload["cell_type"] == CELL_TYPE
    assert first_payload["previous_origin_sha256"] == ZERO_SHA256

    previous = first["worker_origin"]["origin_sha256"]
    second_body = {
        **_result_body(),
        "cell_id": "cell-0002",
        "cell_type": "verification",
        "attempt_id": "attempt-0002",
    }
    second = _signed_result(
        worker_key,
        authorization,
        attestation,
        body=second_body,
        sequence=2,
        previous=previous,
    )
    payload = verify_worker_result_origin(
        policy,
        authorization_attestation=attestation,
        expected_authorization_payload=authorization,
        result=second,
        expected_cell_id="cell-0002",
        expected_cell_type="verification",
        expected_attempt_id="attempt-0002",
        expected_sequence=2,
        expected_previous_origin_sha256=previous,
    )
    assert payload["previous_origin_sha256"] == previous


def test_claim_verification_explicitly_rejects_exportable_producer_custody():
    policy, role_keys, worker_key, _existing_authorization, _attestation = _fixture()
    legacy_custody = _authorization(
        policy,
        worker_key,
        custody=WORKER_KEY_CUSTODY_PRODUCER_SOFTWARE,
    )
    attestation = build_role_attestation(
        policy,
        role=CAMPAIGN_RUNNER,
        payload=legacy_custody,
        signed_at_unix=SIGNED_AT,
        private_key=role_keys[CAMPAIGN_RUNNER],
    )
    with pytest.raises(
        WorkerOriginError, match="worker_key_custody_claim_incompatible"
    ):
        verify_worker_authorization(
            policy,
            attestation,
            expected_payload=legacy_custody,
        )

    verified = verify_worker_authorization(
        policy,
        attestation,
        expected_payload=legacy_custody,
        require_claim_proof_custody=False,
    )
    assert verified["payload"]["worker_key_custody"] == (
        WORKER_KEY_CUSTODY_PRODUCER_SOFTWARE
    )


def test_old_authorization_schema_fails_with_versioned_incompatibility():
    policy, _role_keys, _worker_key, authorization, attestation = _fixture()
    attacked = copy.deepcopy(authorization)
    attacked["schema"] = "aura.latent_cortex.worker_authorization_payload.v3"
    with pytest.raises(
        WorkerOriginError,
        match="worker_authorization_payload_version_incompatible",
    ):
        verify_worker_authorization(
            policy,
            attestation,
            expected_payload=attacked,
        )


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("supervisor_attempt", True, "worker_supervisor_attempt_invalid"),
        ("worker_attempt_slot", True, "worker_attempt_slot_invalid"),
        ("session_id", "not-hex", "worker_session_id_invalid"),
        ("environment_sha256", "short", "environment_sha256_invalid"),
    ],
)
def test_authorization_rejects_malformed_and_bool_int_confusion(
    field: str,
    value,
    error: str,
):
    policy, _role_keys, _worker_key, authorization, attestation = _fixture()
    attacked = copy.deepcopy(authorization)
    attacked[field] = value
    with pytest.raises(WorkerOriginError, match=error):
        verify_worker_authorization(
            policy,
            attestation,
            expected_payload=attacked,
        )


def test_authorization_rejects_wrong_role_policy_and_time():
    policy, role_keys, _worker_key, authorization, attestation = _fixture()
    wrong_role = build_role_attestation(
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
            wrong_role,
            expected_payload=authorization,
        )
    with pytest.raises(
        WorkerOriginError, match="worker_authorization_attestation_invalid"
    ):
        verify_worker_authorization(
            policy,
            attestation,
            expected_payload=authorization,
            not_after_unix=SIGNED_AT - 1,
        )

    attacked = copy.deepcopy(authorization)
    attacked["policy_sha256"] = "f" * 64
    with pytest.raises(WorkerOriginError, match="worker_authorization_policy_mismatch"):
        verify_worker_authorization(
            policy,
            attestation,
            expected_payload=attacked,
        )


@pytest.mark.parametrize(
    ("override", "value"),
    [
        ("expected_cell_id", "wrong-cell"),
        ("expected_cell_type", "wrong-type"),
        ("expected_attempt_id", "wrong-attempt"),
        ("expected_sequence", 2),
        ("expected_previous_origin_sha256", "f" * 64),
    ],
)
def test_result_rejects_wrong_typed_chain_position(override: str, value):
    policy, _role_keys, worker_key, authorization, attestation = _fixture()
    result = _signed_result(worker_key, authorization, attestation)
    with pytest.raises(WorkerOriginError):
        _verify(policy, authorization, attestation, result, **{override: value})


def test_result_rejects_body_signature_digest_and_schema_tampering():
    policy, _role_keys, worker_key, authorization, attestation = _fixture()
    result = _signed_result(worker_key, authorization, attestation)

    attacked = copy.deepcopy(result)
    attacked["text"] = "tampered"
    with pytest.raises(WorkerOriginError, match="worker_result_binding_invalid"):
        _verify(policy, authorization, attestation, attacked)

    attacked = copy.deepcopy(result)
    attacked["worker_origin"]["signature_b64"] = base64.b64encode(
        b"x" * 64
    ).decode()
    _refresh_origin_digest(attacked)
    with pytest.raises(WorkerOriginError, match="worker_result_signature_invalid"):
        _verify(policy, authorization, attestation, attacked)

    attacked = copy.deepcopy(result)
    attacked["worker_origin"]["schema"] = (
        "aura.latent_cortex.worker_result_origin.v1"
    )
    _refresh_origin_digest(attacked)
    with pytest.raises(
        WorkerOriginError, match="worker_result_origin_version_incompatible"
    ):
        _verify(policy, authorization, attestation, attacked)


def test_terminal_and_abandoned_lifecycle_receipts_are_exactly_bound():
    _policy, _role_keys, worker_key, authorization, attestation = _fixture()
    terminal_payload = build_worker_lifecycle_event_payload(
        authorization_attestation=attestation,
        authorization_payload=authorization,
        event_type="terminal",
        prior_state="running",
        result_count=1,
        previous_origin_sha256="a" * 64,
        completed_cell_ids=[CELL_ID],
        occurred_at_unix=SIGNED_AT + 10,
        return_code=0,
        reason=None,
    )
    terminal = assemble_worker_lifecycle_event_origin(
        terminal_payload,
        signature=worker_key.sign(canonical_json_bytes(terminal_payload)),
    )
    verified = verify_worker_lifecycle_event_origin(
        policy=_policy,
        authorization_payload=authorization,
        authorization_attestation=attestation,
        event_origin=terminal,
        expected_event_type="terminal",
        expected_prior_state="running",
        expected_result_count=1,
        expected_previous_origin_sha256="a" * 64,
        expected_completed_cell_ids=[CELL_ID],
        expected_occurred_at_unix=SIGNED_AT + 10,
        expected_return_code=0,
        expected_reason=None,
    )
    assert verified["event_type"] == "terminal"

    abandoned_payload = build_worker_lifecycle_event_payload(
        authorization_attestation=None,
        authorization_payload=authorization,
        event_type="abandoned",
        prior_state="prepared",
        result_count=0,
        previous_origin_sha256=ZERO_SHA256,
        completed_cell_ids=[],
        occurred_at_unix=SIGNED_AT,
        return_code=None,
        reason="supervisor_shutdown",
    )
    abandoned = assemble_worker_lifecycle_event_origin(
        abandoned_payload,
        signature=worker_key.sign(canonical_json_bytes(abandoned_payload)),
    )
    assert (
        verify_worker_lifecycle_event_origin(
            policy=_policy,
            authorization_payload=authorization,
            authorization_attestation=None,
            event_origin=abandoned,
            expected_event_type="abandoned",
            expected_prior_state="prepared",
            expected_result_count=0,
            expected_previous_origin_sha256=ZERO_SHA256,
            expected_completed_cell_ids=[],
            expected_occurred_at_unix=SIGNED_AT,
            expected_return_code=None,
            expected_reason="supervisor_shutdown",
        )["authorization_attestation_sha256"]
        == ZERO_SHA256
    )

    attacked = copy.deepcopy(terminal)
    attacked["signed_payload"]["return_code"] = 9
    material = dict(attacked)
    material.pop("event_sha256")
    attacked["event_sha256"] = hashlib.sha256(
        canonical_json_bytes(material)
    ).hexdigest()
    with pytest.raises(WorkerOriginError, match="worker_lifecycle_binding_invalid"):
        verify_worker_lifecycle_event_origin(
            policy=_policy,
            authorization_payload=authorization,
            authorization_attestation=attestation,
            event_origin=attacked,
            expected_event_type="terminal",
            expected_prior_state="running",
            expected_result_count=1,
            expected_previous_origin_sha256="a" * 64,
            expected_completed_cell_ids=[CELL_ID],
            expected_occurred_at_unix=SIGNED_AT + 10,
            expected_return_code=0,
            expected_reason=None,
        )


def test_lifecycle_rejects_bool_int_and_invalid_event_fields():
    _policy, _role_keys, _worker_key, authorization, attestation = _fixture()
    with pytest.raises(
        WorkerOriginError, match="worker_lifecycle_result_count_invalid"
    ):
        build_worker_lifecycle_event_payload(
            authorization_attestation=attestation,
            authorization_payload=authorization,
            event_type="terminal",
            prior_state="running",
            result_count=True,
            previous_origin_sha256=ZERO_SHA256,
            completed_cell_ids=[],
            occurred_at_unix=SIGNED_AT,
            return_code=0,
            reason=None,
        )


def test_protocol_rejects_unbounded_cells_and_result_payloads():
    policy, _role_keys, _worker_key, authorization, attestation = _fixture()
    oversized_cells = [
        {"cell_id": f"cell-{index}", "cell_type": "reasoning"}
        for index in range(MAX_WORKER_ALLOWED_CELLS + 1)
    ]
    with pytest.raises(WorkerOriginError, match="worker_allowed_cells_too_large"):
        compute_allowed_cell_digest(oversized_cells)

    oversized_result = {
        **_result_body(),
        "text": "x" * MAX_WORKER_PROTOCOL_VALUE_BYTES,
    }
    with pytest.raises(WorkerOriginError, match="worker_result_body_too_large"):
        build_worker_result_signed_payload(
            authorization_attestation=attestation,
            authorization_payload=authorization,
            result_body=oversized_result,
            cell_id=CELL_ID,
            cell_type=CELL_TYPE,
            attempt_id=ATTEMPT_ID,
            sequence=1,
        )


def test_lifecycle_with_authorization_requires_independent_policy_verification():
    policy, role_keys, worker_key, authorization, attestation = _fixture()
    payload = build_worker_lifecycle_event_payload(
        authorization_attestation=attestation,
        authorization_payload=authorization,
        event_type="terminal",
        prior_state="running",
        result_count=1,
        previous_origin_sha256="a" * 64,
        completed_cell_ids=[CELL_ID],
        occurred_at_unix=SIGNED_AT + 10,
        return_code=0,
        reason=None,
    )
    origin = assemble_worker_lifecycle_event_origin(
        payload,
        signature=worker_key.sign(canonical_json_bytes(payload)),
    )
    arguments = {
        "authorization_payload": authorization,
        "authorization_attestation": attestation,
        "event_origin": origin,
        "expected_event_type": "terminal",
        "expected_prior_state": "running",
        "expected_result_count": 1,
        "expected_previous_origin_sha256": "a" * 64,
        "expected_completed_cell_ids": [CELL_ID],
        "expected_occurred_at_unix": SIGNED_AT + 10,
        "expected_return_code": 0,
        "expected_reason": None,
    }
    with pytest.raises(
        WorkerOriginError, match="worker_lifecycle_authorization_policy_missing"
    ):
        verify_worker_lifecycle_event_origin(**arguments)

    wrong_role = build_role_attestation(
        policy,
        role=TASK_ISSUER,
        payload=authorization,
        signed_at_unix=SIGNED_AT,
        private_key=role_keys[TASK_ISSUER],
    )
    wrong_payload = build_worker_lifecycle_event_payload(
        authorization_attestation=wrong_role,
        authorization_payload=authorization,
        event_type="terminal",
        prior_state="running",
        result_count=1,
        previous_origin_sha256="a" * 64,
        completed_cell_ids=[CELL_ID],
        occurred_at_unix=SIGNED_AT + 10,
        return_code=0,
        reason=None,
    )
    wrong_origin = assemble_worker_lifecycle_event_origin(
        wrong_payload,
        signature=worker_key.sign(canonical_json_bytes(wrong_payload)),
    )
    with pytest.raises(
        WorkerOriginError, match="worker_authorization_attestation_invalid"
    ):
        verify_worker_lifecycle_event_origin(
            policy=policy,
            **{
                **arguments,
                "authorization_attestation": wrong_role,
                "event_origin": wrong_origin,
            },
        )
    with pytest.raises(
        WorkerOriginError, match="worker_lifecycle_terminal_fields_invalid"
    ):
        build_worker_lifecycle_event_payload(
            authorization_attestation=attestation,
            authorization_payload=authorization,
            event_type="terminal",
            prior_state="running",
            result_count=0,
            previous_origin_sha256=ZERO_SHA256,
            completed_cell_ids=[],
            occurred_at_unix=SIGNED_AT,
            return_code=True,
            reason=None,
        )
