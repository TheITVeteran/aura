from __future__ import annotations

import base64
import hashlib
import inspect
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from core.brain.llm.latent_cortex.campaign_journal import canonical_json_bytes
from core.brain.llm.latent_cortex.campaign_trust import (
    CAMPAIGN_RUNNER,
    CAMPAIGN_TRUST_POLICY_SCHEMA,
    CAMPAIGN_TRUST_ROLES,
    TASK_ISSUER,
    assemble_role_attestation,
    build_role_attestation,
    validate_campaign_trust_policy,
)
from core.brain.llm.latent_cortex.worker_origin import (
    WORKER_KEY_CUSTODY_DETACHED_SUPERVISOR,
    ZERO_SHA256,
    verify_worker_lifecycle_event_origin,
    verify_worker_result_origin,
)
from core.runtime.detached_worker_origin import (
    DetachedWorkerOriginAuthority,
    DetachedWorkerOriginError,
    DetachedWorkerOriginState,
)

SIGNED_AT = 1_800_000_150
SESSION_ID = "a" * 32
ALLOWED_CELLS = [
    {"cell_id": "cell-0001", "cell_type": "reasoning"},
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


@pytest.fixture
def trust_fixture():
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
    return policy, role_keys


def _authority(policy, **overrides):
    arguments = {
        "policy": policy,
        "campaign_name": "resident-32b-confirmatory",
        "protocol_sha256": "1" * 64,
        "detached_plan_sha256": "2" * 64,
        "broker_policy_sha256": "3" * 64,
        "executable_binding_sha256": "4" * 64,
        "environment_sha256": "5" * 64,
        "sandbox_sha256": "6" * 64,
        "source_manifest_sha256": "7" * 64,
        "session_id": SESSION_ID,
        "supervisor_attempt": 1,
        "arm": "adapter_rlc",
        "worker_attempt_slot": 1,
        "allowed_cells": ALLOWED_CELLS,
        "model_identity_sha256": "8" * 64,
        "adapter_identity_sha256": "9" * 64,
        "authorization_ttl_seconds": 300,
        **overrides,
    }
    return DetachedWorkerOriginAuthority(**arguments)


def _external_attestation(authority, policy, runner_key):
    request = authority.request_authorization(signed_at_unix=SIGNED_AT)
    signed_bytes = base64.b64decode(request["signed_payload_b64"], validate=True)
    return assemble_role_attestation(
        policy,
        request,
        signature_b64=base64.b64encode(runner_key.sign(signed_bytes)).decode(
            "ascii"
        ),
        role=CAMPAIGN_RUNNER,
    )


def _authorize_and_start(authority, policy, runner_key):
    attestation = _external_attestation(authority, policy, runner_key)
    authority.accept_authorization(attestation, now_unix=SIGNED_AT + 1)
    authority.start()
    return attestation


def _result(cell_id: str, cell_type: str, attempt_id: str):
    return {
        "cell_id": cell_id,
        "cell_type": cell_type,
        "attempt_id": attempt_id,
        "origin_session_id": SESSION_ID,
        "answer": "42",
    }


def test_authority_executes_exact_state_machine_and_verifiable_chain(
    trust_fixture,
):
    policy, role_keys = trust_fixture
    authority = _authority(policy)
    assert authority.state is DetachedWorkerOriginState.PREPARED
    assert authority.authorization_payload["worker_key_custody"] == (
        WORKER_KEY_CUSTODY_DETACHED_SUPERVISOR
    )

    attestation = _external_attestation(
        authority, policy, role_keys[CAMPAIGN_RUNNER]
    )
    assert authority.state is DetachedWorkerOriginState.AWAITING_EXTERNAL_SIGNATURE
    authority.accept_authorization(attestation, now_unix=SIGNED_AT + 1)
    assert authority.state is DetachedWorkerOriginState.AUTHORIZED
    authority.start()
    assert authority.state is DetachedWorkerOriginState.RUNNING

    first = authority.record_result(
        _result("cell-0001", "reasoning", "attempt-0001")
    )
    first_payload = verify_worker_result_origin(
        policy,
        authorization_attestation=attestation,
        expected_authorization_payload=authority.authorization_payload,
        result=first,
        expected_cell_id="cell-0001",
        expected_cell_type="reasoning",
        expected_attempt_id="attempt-0001",
        expected_sequence=1,
        expected_previous_origin_sha256=ZERO_SHA256,
        authorization_not_before_unix=SIGNED_AT,
        authorization_not_after_unix=SIGNED_AT,
    )
    assert first_payload["session_id"] == SESSION_ID

    second = authority.record_result(
        _result("cell-0002", "verification", "attempt-0002")
    )
    verify_worker_result_origin(
        policy,
        authorization_attestation=attestation,
        expected_authorization_payload=authority.authorization_payload,
        result=second,
        expected_cell_id="cell-0002",
        expected_cell_type="verification",
        expected_attempt_id="attempt-0002",
        expected_sequence=2,
        expected_previous_origin_sha256=first["worker_origin"]["origin_sha256"],
    )
    terminal = authority.complete(occurred_at_unix=SIGNED_AT + 10)
    assert authority.state is DetachedWorkerOriginState.TERMINAL
    assert authority.result_count == 2
    verified = verify_worker_lifecycle_event_origin(
        policy=policy,
        authorization_payload=authority.authorization_payload,
        authorization_attestation=attestation,
        event_origin=terminal,
        expected_event_type="terminal",
        expected_prior_state="running",
        expected_result_count=2,
        expected_previous_origin_sha256=second["worker_origin"]["origin_sha256"],
        expected_completed_cell_ids=["cell-0001", "cell-0002"],
        expected_occurred_at_unix=SIGNED_AT + 10,
        expected_return_code=0,
        expected_reason=None,
    )
    assert verified["result_count"] == 2

    with pytest.raises(
        DetachedWorkerOriginError, match="worker_origin_state_transition_invalid"
    ):
        authority.record_result(
            _result("cell-0002", "verification", "attempt-late")
        )


def test_authority_rejects_duplicate_wrong_out_of_order_and_wrong_type(
    trust_fixture,
):
    policy, role_keys = trust_fixture
    authority = _authority(policy)
    _authorize_and_start(authority, policy, role_keys[CAMPAIGN_RUNNER])

    with pytest.raises(
        DetachedWorkerOriginError, match="worker_result_cell_out_of_order"
    ):
        authority.record_result(
            _result("cell-0002", "verification", "attempt-early")
        )
    with pytest.raises(
        DetachedWorkerOriginError, match="worker_result_cell_not_allowed"
    ):
        authority.record_result(
            _result("unknown-cell", "reasoning", "attempt-unknown")
        )
    with pytest.raises(
        DetachedWorkerOriginError, match="worker_result_cell_type_mismatch"
    ):
        authority.record_result(
            _result("cell-0001", "verification", "attempt-wrong-type")
        )

    authority.record_result(_result("cell-0001", "reasoning", "attempt-0001"))
    with pytest.raises(
        DetachedWorkerOriginError, match="worker_result_cell_duplicate"
    ):
        authority.record_result(
            _result("cell-0001", "reasoning", "attempt-duplicate")
        )
    with pytest.raises(
        DetachedWorkerOriginError, match="worker_origin_completion_incomplete"
    ):
        authority.complete(occurred_at_unix=SIGNED_AT + 5)


def test_authority_rejects_stale_swapped_wrong_role_and_replayed_attestation(
    trust_fixture,
):
    policy, role_keys = trust_fixture
    first = _authority(policy)
    first_request = first.request_authorization(signed_at_unix=SIGNED_AT)
    signed_bytes = base64.b64decode(
        first_request["signed_payload_b64"], validate=True
    )
    first_attestation = assemble_role_attestation(
        policy,
        first_request,
        signature_b64=base64.b64encode(
            role_keys[CAMPAIGN_RUNNER].sign(signed_bytes)
        ).decode("ascii"),
        role=CAMPAIGN_RUNNER,
    )
    with pytest.raises(
        DetachedWorkerOriginError, match="worker_authorization_window_expired"
    ):
        first.accept_authorization(first_attestation, now_unix=SIGNED_AT + 301)

    second = _authority(policy, session_id="b" * 32)
    second.request_authorization(signed_at_unix=SIGNED_AT)
    with pytest.raises(
        DetachedWorkerOriginError, match="worker_authorization_attestation_invalid"
    ):
        second.accept_authorization(first_attestation, now_unix=SIGNED_AT + 1)

    third = _authority(policy, session_id="c" * 32)
    third.request_authorization(signed_at_unix=SIGNED_AT)
    wrong_role = build_role_attestation(
        policy,
        role=TASK_ISSUER,
        payload=third.authorization_payload,
        signed_at_unix=SIGNED_AT,
        private_key=role_keys[TASK_ISSUER],
    )
    with pytest.raises(
        DetachedWorkerOriginError, match="worker_authorization_attestation_invalid"
    ):
        third.accept_authorization(wrong_role, now_unix=SIGNED_AT + 1)

    fourth = _authority(policy, session_id="d" * 32)
    replay = _external_attestation(
        fourth, policy, role_keys[CAMPAIGN_RUNNER]
    )
    fourth.accept_authorization(replay, now_unix=SIGNED_AT + 1)
    with pytest.raises(
        DetachedWorkerOriginError, match="worker_origin_state_transition_invalid"
    ):
        fourth.accept_authorization(replay, now_unix=SIGNED_AT + 1)


def test_abandonment_is_signed_from_preterminal_states_and_then_sealed(
    trust_fixture,
):
    policy, _role_keys = trust_fixture
    authority = _authority(policy)
    receipt = authority.abandon(
        reason="supervisor_shutdown",
        occurred_at_unix=SIGNED_AT,
    )
    assert authority.state is DetachedWorkerOriginState.ABANDONED
    assert authority.authorization_attestation is None
    verified = verify_worker_lifecycle_event_origin(
        authorization_payload=authority.authorization_payload,
        authorization_attestation=None,
        event_origin=receipt,
        expected_event_type="abandoned",
        expected_prior_state="prepared",
        expected_result_count=0,
        expected_previous_origin_sha256=ZERO_SHA256,
        expected_completed_cell_ids=[],
        expected_occurred_at_unix=SIGNED_AT,
        expected_return_code=None,
        expected_reason="supervisor_shutdown",
    )
    assert verified["event_type"] == "abandoned"
    with pytest.raises(
        DetachedWorkerOriginError, match="worker_origin_state_transition_invalid"
    ):
        authority.request_authorization(signed_at_unix=SIGNED_AT + 1)


@pytest.mark.parametrize(
    ("override", "error"),
    [
        ({"supervisor_attempt": True}, "worker_supervisor_attempt_invalid"),
        ({"worker_attempt_slot": True}, "worker_attempt_slot_invalid"),
        (
            {"authorization_ttl_seconds": True},
            "worker_authorization_ttl_seconds_invalid",
        ),
        ({"allowed_cells": []}, "worker_allowed_cells_invalid"),
        (
            {
                "allowed_cells": [
                    {"cell_id": "duplicate", "cell_type": "reasoning"},
                    {"cell_id": "duplicate", "cell_type": "verification"},
                ]
            },
            "worker_allowed_cell_duplicate",
        ),
    ],
)
def test_authority_rejects_malformed_contracts_and_bool_int_confusion(
    trust_fixture,
    override,
    error,
):
    policy, _role_keys = trust_fixture
    with pytest.raises(DetachedWorkerOriginError, match=error):
        _authority(policy, **override)


def test_authority_has_no_public_key_export_or_generic_signing_surface(
    trust_fixture,
):
    policy, _role_keys = trust_fixture
    authority = _authority(policy)
    public_methods = {
        name
        for name, member in inspect.getmembers(
            DetachedWorkerOriginAuthority,
            predicate=inspect.isfunction,
        )
        if not name.startswith("_")
    }
    assert public_methods == {
        "abandon",
        "accept_authorization",
        "complete",
        "record_result",
        "request_authorization",
        "start",
    }
    forbidden_fragments = {"private", "export", "arbitrary", "sign_payload"}
    assert not any(
        fragment in method
        for method in public_methods
        for fragment in forbidden_fragments
    )
    assert not hasattr(authority, "sign")
    assert not hasattr(authority, "key")

    protocol_source = Path(
        "core/brain/llm/latent_cortex/worker_origin.py"
    ).read_text(encoding="utf-8")
    assert "Ed25519PrivateKey" not in protocol_source
    assert ".sign(" not in protocol_source
    assert "build_worker_result_origin" not in protocol_source
