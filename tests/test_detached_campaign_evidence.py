from __future__ import annotations

import copy
import hashlib
import hmac
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from core.brain.llm.latent_cortex.campaign_journal import canonical_json_bytes
from core.brain.llm.latent_cortex.detached_campaign_evidence import (
    ATTEMPT_EVENT_SCHEMA,
    BROKER_RESPONSE_SCHEMA,
    PLAN_SCHEMA,
    WORKER_ORIGIN_POLICY_SCHEMA,
    WORKER_ORIGIN_QUARANTINE_RECEIPT_SCHEMA,
    DetachedCampaignEvidenceError,
    verify_detached_broker_evidence,
)
from core.brain.llm.latent_cortex.worker_origin import (
    WORKER_KEY_CUSTODY_DETACHED_SUPERVISOR,
    build_worker_authorization_payload,
    compute_allowed_cell_digest,
)
from core.runtime.detached_subprocess_broker import BrokeredProcessResult

PLAN_FILE = "detached_plan.json"
ATTEMPTS_FILE = "detached_attempts.jsonl"
POLICY_TRUST_SHA256 = "1" * 64
PROTOCOL_SHA256 = "2" * 64
MODEL_SHA256 = "3" * 64
ADAPTER_SHA256 = "4" * 64
SOURCE_SHA256 = "5" * 64
ENVIRONMENT_SHA256 = "6" * 64
EXECUTABLE_SHA256 = "7" * 64
SANDBOX_SHA256 = "8" * 64
COMMAND = ["/usr/bin/python3", "worker.py"]
ALLOWED_CELLS = [{"cell_id": "cell-0001", "cell_type": "paired_campaign_cell"}]


def _sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _hashed(body: dict[str, Any], key: str) -> dict[str, Any]:
    return {**body, key: _sha256(body)}


def _launcher_binding(path: str = "/usr/bin/python3") -> dict[str, Any]:
    return _hashed(
        {
            "schema": "aura.detached_step.launcher_binding.v1",
            "invocation_path": path,
            "invocation_kind": "file",
            "invocation_mode": 0o755,
            "symlink_target": None,
            "resolved_path": path,
            "resolved_sha256": EXECUTABLE_SHA256,
            "pyvenv": None,
        },
        "binding_sha256",
    )


def _execution_manifest() -> dict[str, Any]:
    return _hashed(
        {
            "schema": "aura.detached_step.execution_manifest.v1",
            "excluded_roots": [],
            "roots": [
                {
                    "path": "/usr/bin/python3",
                    "kind": "file",
                    "size": 1,
                    "sha256": SOURCE_SHA256,
                }
            ],
        },
        "manifest_sha256",
    )


def _worker_contract(root: Path) -> dict[str, Any]:
    body = {
        "schema": WORKER_ORIGIN_POLICY_SCHEMA,
        "campaign_name": "resident-32b-confirmatory",
        "protocol_sha256": PROTOCOL_SHA256,
        "trust_policy_path": str(root / "trust-policy.json"),
        "trust_policy_binding": {"sha256": "9" * 64},
        "trust_policy_document": {"schema": "test-trust-policy"},
        "trust_policy_sha256": POLICY_TRUST_SHA256,
        "trust_root_path": str(root / "trust-root.pem"),
        "trust_root_binding": {"sha256": "a" * 64},
        "trust_root_public_key_pem_b64": "dGVzdA==",
        "trust_root_key_id": "b" * 64,
        "artifact_dir": str(root / "worker-origins"),
        "arm": "adapter_rlc",
        "worker_attempt_slot": 1,
        "allowed_cells": ALLOWED_CELLS,
        "allowed_cell_digest": compute_allowed_cell_digest(ALLOWED_CELLS),
        "model_identity_sha256": MODEL_SHA256,
        "adapter_identity_sha256": ADAPTER_SHA256,
        "authorization_ttl_seconds": 300,
    }
    return _hashed(body, "contract_sha256")


def _plan(root: Path) -> dict[str, Any]:
    contract = _worker_contract(root)
    policy_body = {
        "command": COMMAND,
        "command_sha256": _sha256(COMMAND),
        "executable_binding": _launcher_binding(),
        "cwd": str(root),
        "stdout_path": str(root / "worker.log"),
        "timeout_s_max": 30.0,
        "max_invocations": 1,
        "execution_manifest": _execution_manifest(),
        "worker_origin": contract,
    }
    policy = _hashed(policy_body, "policy_sha256")
    environment = {"LANG": "C.UTF-8"}
    sandbox = {
        "path": "/usr/bin/sandbox-exec",
        "sha256": "c" * 64,
        "profile": "deny-fork",
        "profile_sha256": "d" * 64,
    }
    body = {
        "schema": PLAN_SCHEMA,
        "name": "detached-campaign-test",
        "command": ["/usr/bin/python3", "coordinator.py"],
        "command_sha256": _sha256(["/usr/bin/python3", "coordinator.py"]),
        "executable_sha256": EXECUTABLE_SHA256,
        "executable_binding": _launcher_binding(),
        "execution_sandbox": sandbox,
        "power_assertion": None,
        "target_execution_manifest": _execution_manifest(),
        "execution_environment": environment,
        "execution_environment_sha256": _sha256(environment),
        "resume_verifier_command": None,
        "resume_verifier_command_sha256": None,
        "resume_verifier_executable_sha256": None,
        "resume_verifier_executable_binding": None,
        "resume_verifier_execution_manifest": None,
        "broker_policy": [policy],
        "broker_policy_sha256": _sha256([policy]),
        "cwd": str(root),
        "timeout_s": 60.0,
        "restart_policy": "never",
        "resume_contract": "none",
        "session_escape_policy": "prohibited",
        "fork_policy": "kernel_denied",
        "containment_policy": "sandbox_no_fork_plus_process_identity_and_group",
        "containment_environment_key": "AURA_DETACHED_RUN_TOKEN",
        "created_at": 100.0,
    }
    return _hashed(body, "plan_sha256")


def _private_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.parent.chmod(0o700)
    path.write_bytes(payload)
    path.chmod(0o600)


def _write_plan(run_dir: Path, plan: dict[str, Any]) -> None:
    _private_write(run_dir / PLAN_FILE, canonical_json_bytes(plan) + b"\n")


def _seal_events(bodies: list[dict[str, Any]]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    previous = ""
    for sequence, event_body in enumerate(bodies, start=1):
        body = {
            "schema": ATTEMPT_EVENT_SCHEMA,
            "sequence": sequence,
            "previous_event_sha256": previous,
            **event_body,
        }
        event = _hashed(body, "event_sha256")
        events.append(event)
        previous = event["event_sha256"]
    return events


def _write_events(run_dir: Path, bodies: list[dict[str, Any]]) -> list[dict[str, Any]]:
    events = _seal_events(bodies)
    _private_write(
        run_dir / ATTEMPTS_FILE,
        b"".join(canonical_json_bytes(event) + b"\n" for event in events),
    )
    return events


def _launch_body(plan: dict[str, Any], attempt: int) -> dict[str, Any]:
    return {
        "event": "LAUNCHED",
        "attempt": attempt,
        "plan_sha256": plan["plan_sha256"],
        "supervisor_pid": 1000 + attempt,
        "supervisor_start_token": f"supervisor-{attempt}",
        "recorded_at": float(attempt * 100),
    }


def _control_body(plan: dict[str, Any], attempt: int, broker_token: str) -> dict[str, Any]:
    return {
        "event": "CONTROL_READY",
        "attempt": attempt,
        "plan_sha256": plan["plan_sha256"],
        "supervisor_pid": 1000 + attempt,
        "supervisor_start_token": f"supervisor-{attempt}",
        "socket_path": f"/tmp/control-{attempt}.sock",
        "control_token": f"{attempt:x}" * 64,
        "broker_enabled": True,
        "broker_token": broker_token,
        "recorded_at": float(attempt * 100 + 1),
    }


def _target_body(plan: dict[str, Any], attempt: int) -> dict[str, Any]:
    return {
        "event": "TARGET_STARTED",
        "attempt": attempt,
        "plan_sha256": plan["plan_sha256"],
        "supervisor_pid": 1000 + attempt,
        "supervisor_start_token": f"supervisor-{attempt}",
        "child_pid": 2000 + attempt,
        "child_process_group_id": 2000 + attempt,
        "child_start_token": f"target-{attempt}",
        "containment_token": "e" * 64,
        "recorded_at": float(attempt * 100 + 2),
    }


def _origin_paths(plan: dict[str, Any], attempt: int) -> tuple[dict[str, Any], dict[str, str]]:
    policy = plan["broker_policy"][0]
    contract = policy["worker_origin"]
    prefix = f"worker-origin-attempt-{attempt:04d}-slot-0001-{policy['policy_sha256'][:16]}"
    root = Path(contract["artifact_dir"])
    return contract, {
        "payload_path": str(root / f"{prefix}.payload.json"),
        "request_path": str(root / f"{prefix}.request.json"),
        "attestation_path": str(root / f"{prefix}.attestation.json"),
        "lifecycle_path": str(root / f"{prefix}.lifecycle.json"),
    }


def _start_body(
    plan: dict[str, Any],
    *,
    attempt: int,
    request_id: str,
    session_id: str,
) -> dict[str, Any]:
    policy = plan["broker_policy"][0]
    contract, paths = _origin_paths(plan, attempt)
    public_key = (
        Ed25519PrivateKey.generate()
        .public_key()
        .public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
    )
    authorization = build_worker_authorization_payload(
        campaign_name=contract["campaign_name"],
        policy_sha256=contract["trust_policy_sha256"],
        protocol_sha256=contract["protocol_sha256"],
        detached_plan_sha256=plan["plan_sha256"],
        broker_policy_sha256=policy["policy_sha256"],
        executable_binding_sha256=policy["executable_binding"]["binding_sha256"],
        environment_sha256=plan["execution_environment_sha256"],
        sandbox_sha256=_sha256(plan["execution_sandbox"]),
        source_manifest_sha256=policy["execution_manifest"]["manifest_sha256"],
        session_id=session_id,
        supervisor_attempt=attempt,
        arm=contract["arm"],
        worker_attempt_slot=contract["worker_attempt_slot"],
        allowed_cell_digest=contract["allowed_cell_digest"],
        model_identity_sha256=contract["model_identity_sha256"],
        adapter_identity_sha256=contract["adapter_identity_sha256"],
        worker_key_custody=WORKER_KEY_CUSTODY_DETACHED_SUPERVISOR,
        worker_public_key_raw=public_key,
    )
    return {
        "event": "BROKER_STARTED",
        "attempt": attempt,
        "plan_sha256": plan["plan_sha256"],
        "supervisor_pid": 1000 + attempt,
        "supervisor_start_token": f"supervisor-{attempt}",
        "request_id": request_id,
        "policy_sha256": policy["policy_sha256"],
        "command_sha256": policy["command_sha256"],
        "worker_pid": 3000 + attempt,
        "worker_process_group_id": 3000 + attempt,
        "worker_start_token": f"worker-{attempt}",
        "containment_token": "f" * 64,
        "reply_path": f"/tmp/reply-{request_id}.sock",
        "timeout_s": 20.0,
        "worker_origin": {
            "contract_sha256": contract["contract_sha256"],
            "session_id": session_id,
            "authorization_payload": authorization,
            "authorization_request_sha256": "a" * 64,
            "authorization_attestation_sha256": "b" * 64,
            **paths,
        },
        "recorded_at": float(attempt * 100 + 3),
    }


def _signed_response(
    *,
    start: dict[str, Any],
    policy: dict[str, Any],
    broker_token: str,
) -> dict[str, Any]:
    lifecycle = {
        "artifact_path": start["worker_origin"]["lifecycle_path"],
        "artifact_sha256": "c" * 64,
        "event_type": "terminal",
        "event_sha256": "d" * 64,
        "result_count": len(policy["worker_origin"]["allowed_cells"]),
        "session_id": start["worker_origin"]["session_id"],
    }
    body = {
        "schema": BROKER_RESPONSE_SCHEMA,
        "request_id": start["request_id"],
        "policy_sha256": policy["policy_sha256"],
        "command_sha256": policy["command_sha256"],
        "worker_pid": start["worker_pid"],
        "worker_process_group_id": start["worker_process_group_id"],
        "worker_start_token": start["worker_start_token"],
        "started_at": float(start["attempt"] * 100 + 4),
        "finished_at": float(start["attempt"] * 100 + 5),
        "duration_s": 1.0,
        "returncode": 0,
        "timed_out": False,
        "cleanup_performed": True,
        "lineage_cleanup_count": 0,
        "containment_verified": True,
        "status": "passed",
        "error": None,
        "worker_origin_lifecycle": lifecycle,
    }
    signed = _hashed(body, "receipt_sha256")
    return {
        **signed,
        "response_hmac_sha256": hmac.new(
            bytes.fromhex(broker_token),
            canonical_json_bytes(signed),
            hashlib.sha256,
        ).hexdigest(),
    }


def _resign_response(response: dict[str, Any], broker_token: str) -> None:
    body = {
        key: value
        for key, value in response.items()
        if key not in {"receipt_sha256", "response_hmac_sha256"}
    }
    response["receipt_sha256"] = _sha256(body)
    signed = {key: value for key, value in response.items() if key != "response_hmac_sha256"}
    response["response_hmac_sha256"] = hmac.new(
        bytes.fromhex(broker_token),
        canonical_json_bytes(signed),
        hashlib.sha256,
    ).hexdigest()


def _terminal_body(
    plan: dict[str, Any], start: dict[str, Any], response: dict[str, Any]
) -> dict[str, Any]:
    return {
        "event": "BROKER_TERMINAL",
        "attempt": start["attempt"],
        "plan_sha256": plan["plan_sha256"],
        "request_id": start["request_id"],
        "policy_sha256": plan["broker_policy"][0]["policy_sha256"],
        "response": response,
        "recorded_at": response["finished_at"],
    }


def _outer_terminal_body(plan: dict[str, Any], attempt: int) -> dict[str, Any]:
    receipt_body = {
        "schema": "aura.detached_step.receipt.v1",
        "plan_sha256": plan["plan_sha256"],
        "supervisor_attempt": attempt,
        "status": "passed",
    }
    return {
        "event": "TERMINAL",
        "attempt": attempt,
        "plan_sha256": plan["plan_sha256"],
        "supervisor_pid": 1000 + attempt,
        "supervisor_start_token": f"supervisor-{attempt}",
        "receipt": _hashed(receipt_body, "receipt_sha256"),
        "recorded_at": float(attempt * 100 + 6),
    }


def _quarantine_body(
    *,
    plan: dict[str, Any],
    start: dict[str, Any],
    previous_event_sha256: str,
    quarantined_at: int,
) -> dict[str, Any]:
    policy = plan["broker_policy"][0]
    origin = start["worker_origin"]
    receipt_body = {
        "schema": WORKER_ORIGIN_QUARANTINE_RECEIPT_SCHEMA,
        "plan_sha256": plan["plan_sha256"],
        "broker_policy_sha256": policy["policy_sha256"],
        "request_id": start["request_id"],
        "supervisor_attempt": start["attempt"],
        "supervisor_pid": start["supervisor_pid"],
        "supervisor_start_token": start["supervisor_start_token"],
        "worker_pid": start["worker_pid"],
        "worker_process_group_id": start["worker_process_group_id"],
        "worker_start_token": start["worker_start_token"],
        "containment_token": start["containment_token"],
        "worker_origin_contract_sha256": origin["contract_sha256"],
        "session_id": origin["session_id"],
        "authorization_request_sha256": origin["authorization_request_sha256"],
        "authorization_attestation_sha256": origin["authorization_attestation_sha256"],
        "payload_path": origin["payload_path"],
        "request_path": origin["request_path"],
        "attestation_path": origin["attestation_path"],
        "lifecycle_path": origin["lifecycle_path"],
        "lifecycle_artifact_sha256": None,
        "prior_journal_head_sha256": previous_event_sha256,
        "supervisor_identity_observed": "dead",
        "worker_identity_observed": "dead",
        "worker_process_group_empty": True,
        "cleanup_action_performed": True,
        "authority_key_recoverable": False,
        "lifecycle_recoverable": False,
        "claim_eligible": False,
        "reason": "supervisor_ephemeral_authority_lost",
        "quarantined_at_unix": quarantined_at,
    }
    return {
        "event": "BROKER_ORIGIN_QUARANTINED",
        "attempt": start["attempt"],
        "plan_sha256": plan["plan_sha256"],
        "request_id": start["request_id"],
        "policy_sha256": policy["policy_sha256"],
        "quarantine_receipt": _hashed(receipt_body, "receipt_sha256"),
        "recorded_at": float(quarantined_at),
    }


def _broker_result(response: dict[str, Any]) -> BrokeredProcessResult:
    return BrokeredProcessResult(
        returncode=response["returncode"],
        request_id=response["request_id"],
        policy_sha256=response["policy_sha256"],
        worker_pid=response["worker_pid"],
        worker_process_group_id=response["worker_process_group_id"],
        worker_start_token=response["worker_start_token"],
        started_at=response["started_at"],
        finished_at=response["finished_at"],
        duration_s=response["duration_s"],
        timed_out=response["timed_out"],
        containment_verified=response["containment_verified"],
        status=response["status"],
        error=response["error"],
        worker_origin_lifecycle=copy.deepcopy(response["worker_origin_lifecycle"]),
        receipt_sha256=response["receipt_sha256"],
        response_hmac_sha256=response["response_hmac_sha256"],
    )


def _fixture(
    root: Path, *, with_quarantine: bool = False
) -> tuple[Path, dict[str, Any], list[dict[str, Any]], BrokeredProcessResult]:
    run_dir = root / "run"
    run_dir.mkdir(parents=True, mode=0o700)
    run_dir.chmod(0o700)
    plan = _plan(root)
    _write_plan(run_dir, plan)
    bodies: list[dict[str, Any]] = []
    if with_quarantine:
        first_token = "1" * 64
        first_start = _start_body(
            plan,
            attempt=1,
            request_id="1" * 32,
            session_id="2" * 32,
        )
        bodies.extend(
            [
                _launch_body(plan, 1),
                _control_body(plan, 1, first_token),
                _target_body(plan, 1),
                first_start,
            ]
        )
        prior = _seal_events(bodies)[-1]["event_sha256"]
        bodies.append(
            _quarantine_body(
                plan=plan,
                start=first_start,
                previous_event_sha256=prior,
                quarantined_at=110,
            )
        )
        attempt = 2
        request_id = "3" * 32
        session_id = "4" * 32
        broker_token = "5" * 64
    else:
        attempt = 1
        request_id = "3" * 32
        session_id = "4" * 32
        broker_token = "5" * 64
    start = _start_body(
        plan,
        attempt=attempt,
        request_id=request_id,
        session_id=session_id,
    )
    response = _signed_response(
        start=start,
        policy=plan["broker_policy"][0],
        broker_token=broker_token,
    )
    bodies.extend(
        [
            _launch_body(plan, attempt),
            _control_body(plan, attempt, broker_token),
            _target_body(plan, attempt),
            start,
            _terminal_body(plan, start, response),
            _outer_terminal_body(plan, attempt),
        ]
    )
    _write_events(run_dir, bodies)
    return run_dir, plan, bodies, _broker_result(response)


def _assert_error(code: str, operation) -> None:
    with pytest.raises(DetachedCampaignEvidenceError) as exc_info:
        operation()
    assert exc_info.value.code == code


def test_success_replays_without_mutating_evidence(tmp_path: Path) -> None:
    run_dir, plan, _bodies, result = _fixture(tmp_path, with_quarantine=True)
    paths = [run_dir / PLAN_FILE, run_dir / ATTEMPTS_FILE]
    before = [(path.read_bytes(), path.stat().st_mtime_ns) for path in paths]

    verified = verify_detached_broker_evidence(
        run_dir=run_dir,
        broker_result=result,
    )

    after = [(path.read_bytes(), path.stat().st_mtime_ns) for path in paths]
    assert after == before
    assert verified.plan["plan_sha256"] == plan["plan_sha256"]
    assert verified.journal_head_sha256 == _seal_events(_bodies)[-1]["event_sha256"]
    assert verified.attempt == 2
    assert verified.request["request_id"] == result.request_id
    assert verified.terminal_event["response"]["receipt_sha256"] == result.receipt_sha256
    assert verified.policy["policy_sha256"] == result.policy_sha256
    assert len(verified.quarantine_summaries) == 1
    assert verified.quarantine_summaries[0].session_id == "2" * 32


def test_journal_hash_tamper_is_rejected(tmp_path: Path) -> None:
    run_dir, _plan_value, _bodies, result = _fixture(tmp_path)
    path = run_dir / ATTEMPTS_FILE
    path.write_bytes(path.read_bytes().replace(b'"status":"passed"', b'"status":"failed"'))
    path.chmod(0o600)
    _assert_error(
        "detached_attempt_event_chain_invalid",
        lambda: verify_detached_broker_evidence(run_dir=run_dir, broker_result=result),
    )


def test_orphan_terminal_is_rejected_after_valid_rehash(tmp_path: Path) -> None:
    run_dir, _plan_value, bodies, result = _fixture(tmp_path)
    bodies = [body for body in bodies if body["event"] != "BROKER_STARTED"]
    _write_events(run_dir, bodies)
    _assert_error(
        "detached_broker_classification_orphan",
        lambda: verify_detached_broker_evidence(run_dir=run_dir, broker_result=result),
    )


def test_duplicate_start_and_mixed_classification_are_rejected(tmp_path: Path) -> None:
    duplicate_dir, _plan_value, bodies, result = _fixture(tmp_path / "duplicate")
    start_index = next(
        index for index, body in enumerate(bodies) if body["event"] == "BROKER_STARTED"
    )
    bodies.insert(start_index + 1, copy.deepcopy(bodies[start_index]))
    _write_events(duplicate_dir, bodies)
    _assert_error(
        "detached_broker_start_invalid",
        lambda: verify_detached_broker_evidence(
            run_dir=duplicate_dir,
            broker_result=result,
        ),
    )

    mixed_dir, plan, bodies, mixed_result = _fixture(tmp_path / "mixed")
    start = next(body for body in bodies if body["event"] == "BROKER_STARTED")
    terminal_index = next(
        index for index, body in enumerate(bodies) if body["event"] == "BROKER_TERMINAL"
    )
    prior = _seal_events(bodies[: terminal_index + 1])[-1]["event_sha256"]
    bodies.insert(
        terminal_index + 1,
        _quarantine_body(
            plan=plan,
            start=start,
            previous_event_sha256=prior,
            quarantined_at=110,
        ),
    )
    _write_events(mixed_dir, bodies)
    _assert_error(
        "detached_broker_classification_duplicate_or_mixed",
        lambda: verify_detached_broker_evidence(
            run_dir=mixed_dir,
            broker_result=mixed_result,
        ),
    )


def test_unfinished_started_session_is_rejected(tmp_path: Path) -> None:
    run_dir, _plan_value, bodies, result = _fixture(tmp_path)
    bodies = [body for body in bodies if body["event"] not in {"BROKER_TERMINAL", "TERMINAL"}]
    _write_events(run_dir, bodies)
    _assert_error(
        "detached_worker_origin_session_unfinished",
        lambda: verify_detached_broker_evidence(run_dir=run_dir, broker_result=result),
    )


def test_bad_response_hmac_is_rejected_after_valid_event_rehash(tmp_path: Path) -> None:
    run_dir, _plan_value, bodies, result = _fixture(tmp_path)
    terminal = next(body for body in bodies if body["event"] == "BROKER_TERMINAL")
    terminal["response"]["response_hmac_sha256"] = "0" * 64
    _write_events(run_dir, bodies)
    _assert_error(
        "detached_broker_response_authentication_invalid",
        lambda: verify_detached_broker_evidence(run_dir=run_dir, broker_result=result),
    )


def test_bad_quarantine_is_rejected_even_with_recomputed_hashes(tmp_path: Path) -> None:
    run_dir, _plan_value, bodies, result = _fixture(tmp_path, with_quarantine=True)
    quarantine = next(body for body in bodies if body["event"] == "BROKER_ORIGIN_QUARANTINED")
    receipt = quarantine["quarantine_receipt"]
    receipt["claim_eligible"] = True
    receipt_body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    receipt["receipt_sha256"] = _sha256(receipt_body)
    _write_events(run_dir, bodies)
    _assert_error(
        "detached_quarantine_receipt_invalid",
        lambda: verify_detached_broker_evidence(run_dir=run_dir, broker_result=result),
    )


@pytest.mark.parametrize(
    ("attack", "expected"),
    [
        ("response", "detached_broker_response_binding_invalid"),
        ("lifecycle", "detached_broker_lifecycle_summary_invalid"),
        ("policy", "detached_broker_classification_policy_mismatch"),
    ],
)
def test_rehashed_binding_substitution_is_rejected(
    tmp_path: Path,
    attack: str,
    expected: str,
) -> None:
    run_dir, _plan_value, bodies, result = _fixture(tmp_path)
    terminal = next(body for body in bodies if body["event"] == "BROKER_TERMINAL")
    control = next(body for body in bodies if body["event"] == "CONTROL_READY")
    if attack == "response":
        terminal["response"]["worker_pid"] += 1
        _resign_response(terminal["response"], control["broker_token"])
    elif attack == "lifecycle":
        terminal["response"]["worker_origin_lifecycle"]["session_id"] = "f" * 32
        _resign_response(terminal["response"], control["broker_token"])
    else:
        terminal["policy_sha256"] = "f" * 64
    _write_events(run_dir, bodies)
    _assert_error(
        expected,
        lambda: verify_detached_broker_evidence(run_dir=run_dir, broker_result=result),
    )


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (
            b'{"schema":"aura.detached_step.plan.v2","schema":"aura.detached_step.plan.v2"}\n',
            "detached_plan_duplicate_key",
        ),
        (b'{"created_at":NaN}\n', "detached_plan_non_finite_number"),
    ],
)
def test_plan_rejects_duplicate_keys_and_nonfinite_numbers(
    tmp_path: Path,
    payload: bytes,
    expected: str,
) -> None:
    run_dir, _plan_value, _bodies, result = _fixture(tmp_path)
    _private_write(run_dir / PLAN_FILE, payload)
    _assert_error(
        expected,
        lambda: verify_detached_broker_evidence(run_dir=run_dir, broker_result=result),
    )


def test_evidence_files_must_be_single_link_owned_regular_files(tmp_path: Path) -> None:
    run_dir, _plan_value, _bodies, result = _fixture(tmp_path)
    os.link(run_dir / PLAN_FILE, tmp_path / "plan-hardlink.json")
    _assert_error(
        "detached_plan_storage_invalid",
        lambda: verify_detached_broker_evidence(run_dir=run_dir, broker_result=result),
    )


def test_supplied_result_must_match_exact_terminal_response(tmp_path: Path) -> None:
    run_dir, _plan_value, _bodies, result = _fixture(tmp_path)
    substituted = replace(result, receipt_sha256="f" * 64)
    _assert_error(
        "detached_broker_result_binding_invalid",
        lambda: verify_detached_broker_evidence(
            run_dir=run_dir,
            broker_result=substituted,
        ),
    )
