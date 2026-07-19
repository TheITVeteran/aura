from __future__ import annotations

import base64
import copy
import hashlib
import os
from dataclasses import replace
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from core.brain.llm.latent_cortex.campaign_journal import (
    CampaignJournal,
    CampaignPlan,
    canonical_json_bytes,
)
from core.brain.llm.latent_cortex.campaign_trust import (
    CAMPAIGN_RUNNER,
    CAMPAIGN_TRUST_POLICY_SCHEMA,
    CAMPAIGN_TRUST_ROLES,
    assemble_role_attestation,
    validate_campaign_trust_policy,
)
from core.brain.llm.latent_cortex.worker_attempt_import import (
    LIFECYCLE_ARTIFACT_SCHEMA,
    PAIRED_CAMPAIGN_CELL_TYPE,
    WorkerAttemptImportError,
    import_verified_worker_stage,
    verify_terminal_worker_stage,
)
from core.runtime.detached_subprocess_broker import BrokeredProcessResult
from core.runtime.detached_worker_origin import DetachedWorkerOriginAuthority

SIGNED_AT = 1_800_000_150
CAMPAIGN_NAME = "resident-32b-confirmatory"
PROTOCOL_SHA256 = "1" * 64
DETACHED_PLAN_SHA256 = "2" * 64
BROKER_POLICY_SHA256 = "3" * 64
MODEL_IDENTITY_SHA256 = "8" * 64
ADAPTER_IDENTITY_SHA256 = "9" * 64
SESSION_ID = "a" * 32
ARM = "adapter_rlc"


def _sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


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
        "custody_evidence_sha256": hashlib.sha256(f"{role}:custody".encode()).hexdigest(),
    }


@pytest.fixture
def trust_fixture():
    root = Ed25519PrivateKey.generate()
    role_keys = {role: Ed25519PrivateKey.generate() for role in CAMPAIGN_TRUST_ROLES}
    body = {
        "schema": CAMPAIGN_TRUST_POLICY_SCHEMA,
        "policy_id": "resident-32b-worker-stage-import",
        "policy_revision": 1,
        "campaign_name": CAMPAIGN_NAME,
        "protocol_sha256": PROTOCOL_SHA256,
        "previous_policy_sha256": None,
        "revoked_key_ids": [],
        "issued_at_unix": 1_800_000_000,
        "not_before_unix": 1_800_000_100,
        "expires_at_unix": 1_800_086_400,
        "roles": {role: _pin(role, role_keys[role]) for role in CAMPAIGN_TRUST_ROLES},
    }
    signed = canonical_json_bytes(body)
    root_raw = _public_raw(root)
    document = {
        **body,
        "root_signature": {
            "algorithm": "Ed25519",
            "key_id": hashlib.sha256(root_raw).hexdigest(),
            "signature_b64": base64.b64encode(root.sign(signed)).decode("ascii"),
            "signed_payload_sha256": hashlib.sha256(signed).hexdigest(),
        },
    }
    policy = validate_campaign_trust_policy(
        document,
        trusted_root_public_key_pem=_public_pem(root),
        now_unix=1_800_000_200,
    )
    return policy, role_keys


def _plan() -> CampaignPlan:
    return CampaignPlan.build(
        CAMPAIGN_NAME,
        [
            {
                "arm": ARM,
                "domain": "mathematics",
                "seed": 101,
                "task_sha256": "d" * 64,
            },
            {
                "arm": ARM,
                "domain": "coding",
                "seed": 102,
                "task_sha256": "e" * 64,
            },
        ],
        metadata={"protocol_sha256": PROTOCOL_SHA256},
    )


def _allowed_cells(plan: CampaignPlan) -> list[dict[str, str]]:
    return [
        {"cell_id": cell_id, "cell_type": PAIRED_CAMPAIGN_CELL_TYPE} for cell_id in plan.cell_ids
    ]


def _write_private_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.parent.chmod(0o700)
    path.write_bytes(canonical_json_bytes(value) + b"\n")
    path.chmod(0o600)


def _broker_result(lifecycle_path: Path, lifecycle: dict) -> BrokeredProcessResult:
    event_origin = lifecycle["event_origin"]
    authorization = lifecycle["authorization_payload"]
    signed_lifecycle = event_origin["signed_payload"]
    return BrokeredProcessResult(
        returncode=0,
        request_id="b" * 32,
        policy_sha256=BROKER_POLICY_SHA256,
        worker_pid=1234,
        worker_process_group_id=1234,
        worker_start_token="worker-start-token",
        started_at=100.0,
        finished_at=101.0,
        duration_s=1.0,
        timed_out=False,
        containment_verified=True,
        status="passed",
        error=None,
        worker_origin_lifecycle={
            "artifact_path": str(lifecycle_path),
            "artifact_sha256": lifecycle["artifact_sha256"],
            "event_type": "terminal",
            "event_sha256": event_origin["event_sha256"],
            "result_count": signed_lifecycle["result_count"],
            "session_id": authorization["session_id"],
        },
        receipt_sha256="c" * 64,
        response_hmac_sha256="d" * 64,
    )


def _stage_fixture(
    root: Path,
    policy,
    runner_key: Ed25519PrivateKey,
    *,
    persisted_cells: int = 2,
    forge_first_result: bool = False,
):
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    root.chmod(0o700)
    plan = _plan()
    authority = DetachedWorkerOriginAuthority(
        policy=policy,
        campaign_name=CAMPAIGN_NAME,
        protocol_sha256=PROTOCOL_SHA256,
        detached_plan_sha256=DETACHED_PLAN_SHA256,
        broker_policy_sha256=BROKER_POLICY_SHA256,
        executable_binding_sha256="4" * 64,
        environment_sha256="5" * 64,
        sandbox_sha256="6" * 64,
        source_manifest_sha256="7" * 64,
        session_id=SESSION_ID,
        supervisor_attempt=1,
        arm=ARM,
        worker_attempt_slot=1,
        allowed_cells=_allowed_cells(plan),
        model_identity_sha256=MODEL_IDENTITY_SHA256,
        adapter_identity_sha256=ADAPTER_IDENTITY_SHA256,
        authorization_ttl_seconds=300,
    )
    request = authority.request_authorization(signed_at_unix=SIGNED_AT)
    signed_bytes = base64.b64decode(request["signed_payload_b64"], validate=True)
    attestation = assemble_role_attestation(
        policy,
        request,
        signature_b64=base64.b64encode(runner_key.sign(signed_bytes)).decode("ascii"),
        role=CAMPAIGN_RUNNER,
    )
    authority.accept_authorization(attestation, now_unix=SIGNED_AT + 1)
    authority.start()

    stage_path = root / "stage.jsonl"
    signed_results: list[dict] = []
    with CampaignJournal(stage_path, plan) as stage:
        for ordinal, cell_id in enumerate(plan.cell_ids):
            attempt_id = stage.start_cell(cell_id)
            signed_result = authority.record_result(
                {
                    "cell_id": cell_id,
                    "cell_type": PAIRED_CAMPAIGN_CELL_TYPE,
                    "attempt_id": attempt_id,
                    "origin_session_id": SESSION_ID,
                    "answer": f"answer-{ordinal}",
                }
            )
            signed_results.append(signed_result)
            if ordinal >= persisted_cells:
                continue
            persisted_result = copy.deepcopy(signed_result)
            if forge_first_result and ordinal == 0:
                persisted_result["worker_origin"]["signature_b64"] = base64.b64encode(
                    b"x" * 64
                ).decode("ascii")
            stage.record_arm_result(cell_id, attempt_id, persisted_result)
            stage.record_verified(cell_id, attempt_id, {"accepted": True})
            stage.commit_cell(cell_id, attempt_id, {"stage": True})
    stage_path.chmod(0o600)

    terminal = authority.complete(occurred_at_unix=SIGNED_AT + 10)
    lifecycle_body = {
        "schema": LIFECYCLE_ARTIFACT_SCHEMA,
        "broker_policy_sha256": BROKER_POLICY_SHA256,
        "authorization_payload": authority.authorization_payload,
        "authorization_request": request,
        "authorization_attestation": attestation,
        "event_origin": terminal,
        "completion_error": None,
    }
    lifecycle = {
        **lifecycle_body,
        "artifact_sha256": _sha256(lifecycle_body),
    }
    lifecycle_path = root / "worker-origin-lifecycle.json"
    _write_private_json(lifecycle_path, lifecycle)
    return (
        plan,
        stage_path,
        lifecycle_path,
        lifecycle,
        _broker_result(lifecycle_path, lifecycle),
        signed_results,
    )


def _verify(stage_fixture, policy):
    plan, stage_path, lifecycle_path, _lifecycle, broker_result, _results = stage_fixture
    return verify_terminal_worker_stage(
        stage_path=stage_path,
        lifecycle_path=lifecycle_path,
        plan=plan,
        policy=policy,
        broker_result=broker_result,
        arm=ARM,
        worker_attempt_slot=1,
        expected_protocol_sha256=PROTOCOL_SHA256,
        expected_detached_plan_sha256=DETACHED_PLAN_SHA256,
        expected_broker_policy_sha256=BROKER_POLICY_SHA256,
        expected_model_identity_sha256=MODEL_IDENTITY_SHA256,
        expected_adapter_identity_sha256=ADAPTER_IDENTITY_SHA256,
    )


def _assert_import_error(code: str, operation) -> None:
    with pytest.raises(WorkerAttemptImportError) as exc_info:
        operation()
    assert exc_info.value.code == code


def test_complete_worker_stage_verifies_and_imports_idempotently(
    tmp_path: Path,
    trust_fixture,
) -> None:
    policy, role_keys = trust_fixture
    fixture = _stage_fixture(tmp_path / "attempt", policy, role_keys[CAMPAIGN_RUNNER])
    verified = _verify(fixture, policy)
    plan = fixture[0]
    evidence = tmp_path / "canonical-evidence"
    evidence.mkdir(mode=0o700)
    journal_path = evidence / "campaign.jsonl"
    intent_path = evidence / "import-intent.json"
    receipt_path = evidence / "import-receipt.json"

    first = import_verified_worker_stage(
        canonical_journal_path=journal_path,
        intent_path=intent_path,
        receipt_path=receipt_path,
        plan=plan,
        verified_stage=verified,
    )
    second = import_verified_worker_stage(
        canonical_journal_path=journal_path,
        intent_path=intent_path,
        receipt_path=receipt_path,
        plan=plan,
        verified_stage=verified,
    )

    assert first == second
    assert [entry["cell_id"] for entry in first["imported"]] == list(plan.cell_ids)
    assert intent_path.stat().st_mode & 0o777 == 0o600
    assert receipt_path.stat().st_mode & 0o777 == 0o600
    with CampaignJournal(journal_path, plan) as journal:
        snapshot = journal.resume()
        assert snapshot.committed_cell_ids == ()
        assert snapshot.sealed_cell_ids == plan.cell_ids
        assert tuple(record["result"] for record in journal.result_records()) == tuple(
            record["result"] for record in verified.records
        )


def test_partial_stage_is_never_imported(tmp_path: Path, trust_fixture) -> None:
    policy, role_keys = trust_fixture
    fixture = _stage_fixture(
        tmp_path / "partial",
        policy,
        role_keys[CAMPAIGN_RUNNER],
        persisted_cells=1,
    )
    _assert_import_error("worker_stage_not_complete", lambda: _verify(fixture, policy))


def test_failed_or_uncontained_broker_result_is_never_imported(
    tmp_path: Path,
    trust_fixture,
) -> None:
    policy, role_keys = trust_fixture
    fixture = _stage_fixture(tmp_path / "failed", policy, role_keys[CAMPAIGN_RUNNER])
    broker_result = replace(fixture[4], status="failed", returncode=9, error="worker failed")
    failed_fixture = (*fixture[:4], broker_result, fixture[5])
    _assert_import_error(
        "worker_stage_broker_not_terminal",
        lambda: _verify(failed_fixture, policy),
    )

    uncontained = replace(fixture[4], containment_verified=False)
    uncontained_fixture = (*fixture[:4], uncontained, fixture[5])
    _assert_import_error(
        "worker_stage_broker_not_terminal",
        lambda: _verify(uncontained_fixture, policy),
    )


def test_forged_result_origin_is_rejected(tmp_path: Path, trust_fixture) -> None:
    policy, role_keys = trust_fixture
    fixture = _stage_fixture(
        tmp_path / "forged",
        policy,
        role_keys[CAMPAIGN_RUNNER],
        forge_first_result=True,
    )
    _assert_import_error(
        "worker_stage_result_origin_invalid",
        lambda: _verify(fixture, policy),
    )


def test_broker_lifecycle_substitution_is_rejected(tmp_path: Path, trust_fixture) -> None:
    policy, role_keys = trust_fixture
    fixture = _stage_fixture(tmp_path / "substitution", policy, role_keys[CAMPAIGN_RUNNER])
    bad_summary = dict(fixture[4].worker_origin_lifecycle or {})
    bad_summary["event_sha256"] = "f" * 64
    broker_result = replace(fixture[4], worker_origin_lifecycle=bad_summary)
    substituted = (*fixture[:4], broker_result, fixture[5])
    _assert_import_error(
        "worker_stage_broker_lifecycle_summary_invalid",
        lambda: _verify(substituted, policy),
    )


def test_stage_and_lifecycle_must_be_private_canonical_files(
    tmp_path: Path,
    trust_fixture,
) -> None:
    policy, role_keys = trust_fixture
    fixture = _stage_fixture(tmp_path / "storage", policy, role_keys[CAMPAIGN_RUNNER])
    fixture[1].chmod(0o644)
    _assert_import_error(
        "worker_stage_journal_storage_invalid",
        lambda: _verify(fixture, policy),
    )

    fixture[1].chmod(0o600)
    fixture[2].write_bytes(fixture[2].read_bytes() + b" ")
    fixture[2].chmod(0o600)
    _assert_import_error(
        "worker_stage_lifecycle_noncanonical",
        lambda: _verify(fixture, policy),
    )


def test_import_intent_conflict_fails_closed(tmp_path: Path, trust_fixture) -> None:
    policy, role_keys = trust_fixture
    fixture = _stage_fixture(tmp_path / "intent", policy, role_keys[CAMPAIGN_RUNNER])
    verified = _verify(fixture, policy)
    plan = fixture[0]
    evidence = tmp_path / "conflict-evidence"
    evidence.mkdir(mode=0o700)
    intent_path = evidence / "intent.json"
    receipt_path = evidence / "receipt.json"
    _write_private_json(intent_path, {"schema": "substituted"})

    _assert_import_error(
        "worker_stage_import_intent_invalid",
        lambda: import_verified_worker_stage(
            canonical_journal_path=evidence / "campaign.jsonl",
            intent_path=intent_path,
            receipt_path=receipt_path,
            plan=plan,
            verified_stage=verified,
        ),
    )
    assert not receipt_path.exists()


def test_import_replays_from_verified_boundary(tmp_path: Path, trust_fixture) -> None:
    policy, role_keys = trust_fixture
    fixture = _stage_fixture(tmp_path / "resume", policy, role_keys[CAMPAIGN_RUNNER])
    verified = _verify(fixture, policy)
    plan = fixture[0]
    evidence = tmp_path / "resume-evidence"
    evidence.mkdir(mode=0o700)
    journal_path = evidence / "campaign.jsonl"
    intent_path = evidence / "intent.json"
    receipt_path = evidence / "receipt.json"

    first = import_verified_worker_stage(
        canonical_journal_path=journal_path,
        intent_path=intent_path,
        receipt_path=receipt_path,
        plan=plan,
        verified_stage=verified,
    )
    receipt_path.unlink()
    replay = import_verified_worker_stage(
        canonical_journal_path=journal_path,
        intent_path=intent_path,
        receipt_path=receipt_path,
        plan=plan,
        verified_stage=verified,
    )

    assert replay == first
    assert os.stat(journal_path).st_mode & 0o777 == 0o600
