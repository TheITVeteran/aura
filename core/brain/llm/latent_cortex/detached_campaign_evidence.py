"""Pure read-only verification for detached campaign broker evidence."""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Never

from core.brain.llm.latent_cortex.campaign_journal import (
    CampaignJournalError,
    canonical_json_bytes,
)
from core.brain.llm.latent_cortex.worker_origin import (
    WORKER_KEY_CUSTODY_DETACHED_SUPERVISOR,
    WorkerOriginError,
    compute_allowed_cell_digest,
    validate_worker_authorization_payload,
)
from core.runtime.detached_subprocess_broker import BrokeredProcessResult

PLAN_SCHEMA = "aura.detached_step.plan.v2"
ATTEMPT_EVENT_SCHEMA = "aura.detached_step.attempt_event.v1"
BROKER_RESPONSE_SCHEMA = "aura.detached_step.broker_response.v1"
WORKER_ORIGIN_POLICY_SCHEMA = "aura.detached_step.worker_origin_policy.v1"
WORKER_ORIGIN_QUARANTINE_RECEIPT_SCHEMA = "aura.detached_step.worker_origin_quarantine_receipt.v1"

_PLAN_FILE = "detached_plan.json"
_ATTEMPTS_FILE = "detached_attempts.jsonl"
_MAX_PLAN_BYTES = 16 * 1024 * 1024
_MAX_ATTEMPTS_BYTES = 256 * 1024 * 1024
_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)

_PLAN_KEYS = {
    "schema",
    "name",
    "command",
    "command_sha256",
    "executable_sha256",
    "executable_binding",
    "execution_sandbox",
    "power_assertion",
    "target_execution_manifest",
    "execution_environment",
    "execution_environment_sha256",
    "resume_verifier_command",
    "resume_verifier_command_sha256",
    "resume_verifier_executable_sha256",
    "resume_verifier_executable_binding",
    "resume_verifier_execution_manifest",
    "broker_policy",
    "broker_policy_sha256",
    "cwd",
    "timeout_s",
    "restart_policy",
    "resume_contract",
    "session_escape_policy",
    "fork_policy",
    "containment_policy",
    "containment_environment_key",
    "created_at",
    "plan_sha256",
}
_BROKER_POLICY_KEYS = {
    "command",
    "command_sha256",
    "executable_binding",
    "cwd",
    "stdout_path",
    "timeout_s_max",
    "max_invocations",
    "execution_manifest",
    "worker_origin",
    "policy_sha256",
}
_WORKER_ORIGIN_CONTRACT_KEYS = {
    "schema",
    "campaign_name",
    "protocol_sha256",
    "trust_policy_path",
    "trust_policy_binding",
    "trust_policy_document",
    "trust_policy_sha256",
    "trust_root_path",
    "trust_root_binding",
    "trust_root_public_key_pem_b64",
    "trust_root_key_id",
    "artifact_dir",
    "arm",
    "worker_attempt_slot",
    "allowed_cells",
    "allowed_cell_digest",
    "model_identity_sha256",
    "adapter_identity_sha256",
    "authorization_ttl_seconds",
    "contract_sha256",
}
_WORKER_ORIGIN_START_KEYS = {
    "contract_sha256",
    "session_id",
    "authorization_payload",
    "authorization_request_sha256",
    "authorization_attestation_sha256",
    "request_path",
    "payload_path",
    "attestation_path",
    "lifecycle_path",
}
_LIFECYCLE_SUMMARY_KEYS = {
    "artifact_path",
    "artifact_sha256",
    "event_type",
    "event_sha256",
    "result_count",
    "session_id",
}
_BROKER_RESPONSE_KEYS = {
    "schema",
    "request_id",
    "policy_sha256",
    "command_sha256",
    "worker_pid",
    "worker_process_group_id",
    "worker_start_token",
    "started_at",
    "finished_at",
    "duration_s",
    "returncode",
    "timed_out",
    "cleanup_performed",
    "lineage_cleanup_count",
    "containment_verified",
    "status",
    "error",
    "worker_origin_lifecycle",
    "receipt_sha256",
    "response_hmac_sha256",
}
_QUARANTINE_RECEIPT_KEYS = {
    "schema",
    "plan_sha256",
    "broker_policy_sha256",
    "request_id",
    "supervisor_attempt",
    "supervisor_pid",
    "supervisor_start_token",
    "worker_pid",
    "worker_process_group_id",
    "worker_start_token",
    "containment_token",
    "worker_origin_contract_sha256",
    "session_id",
    "authorization_request_sha256",
    "authorization_attestation_sha256",
    "payload_path",
    "request_path",
    "attestation_path",
    "lifecycle_path",
    "lifecycle_artifact_sha256",
    "prior_journal_head_sha256",
    "supervisor_identity_observed",
    "worker_identity_observed",
    "worker_process_group_empty",
    "cleanup_action_performed",
    "authority_key_recoverable",
    "lifecycle_recoverable",
    "claim_eligible",
    "reason",
    "quarantined_at_unix",
    "receipt_sha256",
}


class DetachedCampaignEvidenceError(ValueError):
    """Stable fail-closed error raised for invalid detached evidence."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _fail(code: str) -> Never:
    raise DetachedCampaignEvidenceError(code)


def _sha256(value: Any) -> str:
    try:
        payload = canonical_json_bytes(value)
    except (CampaignJournalError, TypeError, ValueError, OverflowError):
        _fail("detached_evidence_value_not_canonical")
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_hex(value: Any, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_int(value: Any, *, minimum: int = 0) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


def _is_plain_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_finite_number(value: Any, *, minimum: float | None = None) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    parsed = float(value)
    return math.isfinite(parsed) and (minimum is None or parsed >= minimum)


def _strict_json(payload: bytes, *, role: str) -> dict[str, Any]:
    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                _fail(f"{role}_duplicate_key")
            value[key] = item
        return value

    def parse_float(raw: str) -> float:
        value = float(raw)
        if not math.isfinite(value):
            _fail(f"{role}_non_finite_number")
        return value

    try:
        value = json.loads(
            payload,
            object_pairs_hook=object_pairs,
            parse_float=parse_float,
            parse_constant=lambda _raw: _fail(f"{role}_non_finite_number"),
        )
    except DetachedCampaignEvidenceError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, OverflowError):
        _fail(f"{role}_json_invalid")
    if not isinstance(value, dict):
        _fail(f"{role}_not_object")
    return value


def _read_owned_bytes(path: Path, *, maximum: int, role: str) -> bytes:
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | _NOFOLLOW,
        )
    except OSError:
        _fail(f"{role}_unavailable")
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_uid != os.geteuid()
            or before.st_nlink != 1
            or before.st_mode & 0o022
            or before.st_size <= 0
            or before.st_size > maximum
        ):
            _fail(f"{role}_storage_invalid")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                _fail(f"{role}_short_read")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            _fail(f"{role}_changed_during_read")
        after = os.fstat(descriptor)
        try:
            current = path.lstat()
        except OSError:
            _fail(f"{role}_changed_during_read")
        before_identity = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
            before.st_nlink,
        )
        if before_identity != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
            after.st_nlink,
        ) or before_identity != (
            current.st_dev,
            current.st_ino,
            current.st_size,
            current.st_mtime_ns,
            current.st_ctime_ns,
            current.st_nlink,
        ):
            _fail(f"{role}_changed_during_read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _verified_hash_object(value: Any, *, hash_key: str, role: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(f"{role}_invalid")
    body = {key: item for key, item in value.items() if key != hash_key}
    if not _is_sha256(value.get(hash_key)) or value.get(hash_key) != _sha256(body):
        _fail(f"{role}_hash_invalid")
    return value


def _verify_execution_manifest(value: Any) -> dict[str, Any]:
    manifest = _verified_hash_object(
        value,
        hash_key="manifest_sha256",
        role="detached_execution_manifest",
    )
    if (
        manifest.get("schema") != "aura.detached_step.execution_manifest.v1"
        or not isinstance(manifest.get("excluded_roots"), list)
        or not isinstance(manifest.get("roots"), list)
        or not manifest["roots"]
    ):
        _fail("detached_execution_manifest_invalid")
    return manifest


def _verify_launcher_binding(value: Any) -> dict[str, Any]:
    binding = _verified_hash_object(
        value,
        hash_key="binding_sha256",
        role="detached_launcher_binding",
    )
    if (
        binding.get("schema") != "aura.detached_step.launcher_binding.v1"
        or not isinstance(binding.get("invocation_path"), str)
        or not Path(binding["invocation_path"]).is_absolute()
        or not _is_sha256(binding.get("resolved_sha256"))
    ):
        _fail("detached_launcher_binding_invalid")
    return binding


def _verify_worker_origin_contract(value: Any) -> dict[str, Any]:
    contract = _verified_hash_object(
        value,
        hash_key="contract_sha256",
        role="detached_worker_origin_contract",
    )
    allowed_cells = contract.get("allowed_cells")
    if (
        set(contract) != _WORKER_ORIGIN_CONTRACT_KEYS
        or contract.get("schema") != WORKER_ORIGIN_POLICY_SCHEMA
        or not isinstance(allowed_cells, list)
        or not allowed_cells
        or not _is_int(contract.get("worker_attempt_slot"), minimum=1)
        or not _is_int(contract.get("authorization_ttl_seconds"), minimum=1)
        or not isinstance(contract.get("arm"), str)
        or not contract["arm"]
        or not Path(str(contract.get("artifact_dir") or "")).is_absolute()
    ):
        _fail("detached_worker_origin_contract_invalid")
    for key in (
        "protocol_sha256",
        "trust_policy_sha256",
        "allowed_cell_digest",
        "model_identity_sha256",
        "adapter_identity_sha256",
        "contract_sha256",
    ):
        if not _is_sha256(contract.get(key)):
            _fail("detached_worker_origin_contract_invalid")
    try:
        digest = compute_allowed_cell_digest(allowed_cells)
    except WorkerOriginError as exc:
        raise DetachedCampaignEvidenceError("detached_worker_origin_contract_invalid") from exc
    if digest != contract["allowed_cell_digest"]:
        _fail("detached_worker_origin_contract_invalid")
    return contract


def _verify_plan(plan: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if set(plan) != _PLAN_KEYS or plan.get("schema") != PLAN_SCHEMA:
        _fail("detached_plan_schema_invalid")
    body = {key: value for key, value in plan.items() if key != "plan_sha256"}
    if not _is_sha256(plan.get("plan_sha256")) or plan["plan_sha256"] != _sha256(body):
        _fail("detached_plan_hash_invalid")
    command = plan.get("command")
    environment = plan.get("execution_environment")
    if (
        not isinstance(command, list)
        or not command
        or any(not isinstance(item, str) or not item for item in command)
        or plan.get("command_sha256") != _sha256(command)
        or not isinstance(environment, dict)
        or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in environment.items()
        )
        or plan.get("execution_environment_sha256") != _sha256(environment)
        or not _is_finite_number(plan.get("timeout_s"), minimum=0.000001)
        or not _is_finite_number(plan.get("created_at"))
        or plan.get("restart_policy") != "never"
        or plan.get("resume_contract") not in {"none", "target_checkpoint"}
        or plan.get("session_escape_policy") != "prohibited"
        or plan.get("fork_policy") != "kernel_denied"
        or plan.get("containment_policy") != "sandbox_no_fork_plus_process_identity_and_group"
        or plan.get("containment_environment_key") != "AURA_DETACHED_RUN_TOKEN"
    ):
        _fail("detached_plan_binding_invalid")
    launcher = _verify_launcher_binding(plan.get("executable_binding"))
    if plan.get("executable_sha256") != launcher["resolved_sha256"]:
        _fail("detached_plan_binding_invalid")
    _verify_execution_manifest(plan.get("target_execution_manifest"))

    policies = plan.get("broker_policy")
    if not isinstance(policies, list) or plan.get("broker_policy_sha256") != _sha256(policies):
        _fail("detached_plan_broker_policy_set_invalid")
    by_sha: dict[str, dict[str, Any]] = {}
    seen_commands: set[str] = set()
    for value in policies:
        if not isinstance(value, dict) or set(value) != _BROKER_POLICY_KEYS:
            _fail("detached_broker_policy_invalid")
        policy = value
        policy_body = {key: item for key, item in policy.items() if key != "policy_sha256"}
        policy_command = policy.get("command")
        if (
            not _is_sha256(policy.get("policy_sha256"))
            or policy["policy_sha256"] != _sha256(policy_body)
            or not isinstance(policy_command, list)
            or not policy_command
            or any(not isinstance(item, str) or not item for item in policy_command)
            or policy.get("command_sha256") != _sha256(policy_command)
            or policy["command_sha256"] in seen_commands
            or not Path(str(policy.get("cwd") or "")).is_absolute()
            or not Path(str(policy.get("stdout_path") or "")).is_absolute()
            or not _is_finite_number(policy.get("timeout_s_max"), minimum=0.000001)
            or not _is_int(policy.get("max_invocations"), minimum=1)
        ):
            _fail("detached_broker_policy_invalid")
        _verify_launcher_binding(policy.get("executable_binding"))
        _verify_execution_manifest(policy.get("execution_manifest"))
        if policy.get("worker_origin") is not None:
            _verify_worker_origin_contract(policy["worker_origin"])
            if policy["max_invocations"] != 1:
                _fail("detached_broker_policy_invalid")
        policy_sha = policy["policy_sha256"]
        if policy_sha in by_sha:
            _fail("detached_broker_policy_duplicate")
        by_sha[policy_sha] = policy
        seen_commands.add(policy["command_sha256"])
    if not by_sha:
        _fail("detached_plan_broker_policy_set_invalid")
    return by_sha


def _read_plan(run_dir: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    raw = _read_owned_bytes(
        run_dir / _PLAN_FILE,
        maximum=_MAX_PLAN_BYTES,
        role="detached_plan",
    )
    plan = _strict_json(raw, role="detached_plan")
    if raw != canonical_json_bytes(plan) + b"\n":
        _fail("detached_plan_noncanonical")
    return plan, _verify_plan(plan)


def _read_attempts(run_dir: Path) -> list[dict[str, Any]]:
    raw = _read_owned_bytes(
        run_dir / _ATTEMPTS_FILE,
        maximum=_MAX_ATTEMPTS_BYTES,
        role="detached_attempt_journal",
    )
    lines = raw.splitlines(keepends=True)
    if not lines or any(not line.endswith(b"\n") for line in lines):
        _fail("detached_attempt_journal_noncanonical")
    events: list[dict[str, Any]] = []
    previous = ""
    for sequence, line in enumerate(lines, start=1):
        payload = line[:-1]
        if not payload:
            _fail("detached_attempt_journal_empty_record")
        event = _strict_json(payload, role="detached_attempt_event")
        if payload != canonical_json_bytes(event):
            _fail("detached_attempt_event_noncanonical")
        body = {key: value for key, value in event.items() if key != "event_sha256"}
        if (
            event.get("schema") != ATTEMPT_EVENT_SCHEMA
            or event.get("sequence") != sequence
            or event.get("previous_event_sha256") != previous
            or not _is_sha256(event.get("event_sha256"))
            or event["event_sha256"] != _sha256(body)
        ):
            _fail("detached_attempt_event_chain_invalid")
        events.append(event)
        previous = event["event_sha256"]
    return events


def _origin_paths(
    contract: dict[str, Any],
    *,
    attempt: int,
    policy_sha256: str,
) -> dict[str, str]:
    prefix = (
        f"worker-origin-attempt-{attempt:04d}-"
        f"slot-{int(contract['worker_attempt_slot']):04d}-"
        f"{policy_sha256[:16]}"
    )
    root = Path(contract["artifact_dir"])
    return {
        "payload_path": str(root / f"{prefix}.payload.json"),
        "request_path": str(root / f"{prefix}.request.json"),
        "attestation_path": str(root / f"{prefix}.attestation.json"),
        "lifecycle_path": str(root / f"{prefix}.lifecycle.json"),
    }


def _verify_worker_origin_start(
    *,
    plan: dict[str, Any],
    policy: dict[str, Any],
    event: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    contract_value = policy.get("worker_origin")
    start = event.get("worker_origin")
    if contract_value is None:
        if start is not None:
            _fail("detached_worker_origin_without_contract")
        return {}, {}
    contract = _verify_worker_origin_contract(contract_value)
    if not isinstance(start, dict) or set(start) != _WORKER_ORIGIN_START_KEYS:
        _fail("detached_worker_origin_start_invalid")
    try:
        authorization = validate_worker_authorization_payload(start.get("authorization_payload"))
    except WorkerOriginError as exc:
        raise DetachedCampaignEvidenceError("detached_worker_origin_authorization_invalid") from exc
    expected_authorization = {
        "campaign_name": contract["campaign_name"],
        "policy_sha256": contract["trust_policy_sha256"],
        "protocol_sha256": contract["protocol_sha256"],
        "detached_plan_sha256": plan["plan_sha256"],
        "broker_policy_sha256": policy["policy_sha256"],
        "executable_binding_sha256": policy["executable_binding"]["binding_sha256"],
        "environment_sha256": plan["execution_environment_sha256"],
        "sandbox_sha256": _sha256(plan["execution_sandbox"]),
        "source_manifest_sha256": policy["execution_manifest"]["manifest_sha256"],
        "supervisor_attempt": event["attempt"],
        "arm": contract["arm"],
        "worker_attempt_slot": contract["worker_attempt_slot"],
        "allowed_cell_digest": contract["allowed_cell_digest"],
        "model_identity_sha256": contract["model_identity_sha256"],
        "adapter_identity_sha256": contract["adapter_identity_sha256"],
        "worker_key_custody": WORKER_KEY_CUSTODY_DETACHED_SUPERVISOR,
    }
    expected_paths = _origin_paths(
        contract,
        attempt=event["attempt"],
        policy_sha256=policy["policy_sha256"],
    )
    if (
        start.get("contract_sha256") != contract["contract_sha256"]
        or start.get("session_id") != authorization["session_id"]
        or any(authorization.get(key) != value for key, value in expected_authorization.items())
        or any(start.get(key) != value for key, value in expected_paths.items())
        or not _is_sha256(start.get("authorization_request_sha256"))
        or not _is_sha256(start.get("authorization_attestation_sha256"))
    ):
        _fail("detached_worker_origin_start_binding_invalid")
    return contract, authorization


def _expected_status(response: dict[str, Any]) -> str:
    if response["containment_verified"] is False:
        return "containment_failed"
    if response["timed_out"] is True:
        return "timed_out"
    return "passed" if response["returncode"] == 0 else "failed"


def _verify_lifecycle_summary(
    *,
    summary: Any,
    start_origin: dict[str, Any] | None,
    contract: dict[str, Any] | None,
    passed: bool,
) -> None:
    if contract is None:
        if summary is not None:
            _fail("detached_broker_lifecycle_orphan")
        return
    if (
        not isinstance(summary, dict)
        or set(summary) != _LIFECYCLE_SUMMARY_KEYS
        or start_origin is None
        or summary.get("artifact_path") != start_origin.get("lifecycle_path")
        or summary.get("session_id") != start_origin.get("session_id")
        or not _is_sha256(summary.get("artifact_sha256"))
        or not _is_sha256(summary.get("event_sha256"))
        or not _is_int(summary.get("result_count"))
        or summary.get("event_type") != ("terminal" if passed else "abandoned")
        or (passed and summary.get("result_count") != len(contract["allowed_cells"]))
        or summary.get("result_count", 0) > len(contract["allowed_cells"])
    ):
        _fail("detached_broker_lifecycle_summary_invalid")


def _verify_broker_response(
    *,
    response: Any,
    terminal_event: dict[str, Any],
    start_event: dict[str, Any],
    control_event: dict[str, Any],
    policy: dict[str, Any],
    contract: dict[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(response, dict) or set(response) != _BROKER_RESPONSE_KEYS:
        _fail("detached_broker_response_invalid")
    signed = {key: value for key, value in response.items() if key != "response_hmac_sha256"}
    body = {key: value for key, value in signed.items() if key != "receipt_sha256"}
    broker_token = control_event.get("broker_token")
    try:
        expected_hmac = hmac.new(
            bytes.fromhex(str(broker_token)),
            canonical_json_bytes(signed),
            hashlib.sha256,
        ).hexdigest()
    except ValueError:
        _fail("detached_broker_control_token_invalid")
    if (
        not _is_hex(broker_token, 64)
        or response.get("receipt_sha256") != _sha256(body)
        or not _is_sha256(response.get("response_hmac_sha256"))
        or not hmac.compare_digest(response["response_hmac_sha256"], expected_hmac)
    ):
        _fail("detached_broker_response_authentication_invalid")
    if (
        response.get("schema") != BROKER_RESPONSE_SCHEMA
        or response.get("request_id") != start_event.get("request_id")
        or terminal_event.get("request_id") != start_event.get("request_id")
        or terminal_event.get("policy_sha256") != policy["policy_sha256"]
        or response.get("policy_sha256") != policy["policy_sha256"]
        or response.get("command_sha256") != policy["command_sha256"]
        or start_event.get("command_sha256") != policy["command_sha256"]
        or response.get("worker_pid") != start_event.get("worker_pid")
        or response.get("worker_process_group_id") != start_event.get("worker_process_group_id")
        or response.get("worker_start_token") != start_event.get("worker_start_token")
        or not _is_int(response.get("worker_pid"), minimum=1)
        or not _is_int(response.get("worker_process_group_id"), minimum=2)
        or not isinstance(response.get("worker_start_token"), str)
        or not response["worker_start_token"]
        or not _is_plain_int(response.get("returncode"))
        or not isinstance(response.get("timed_out"), bool)
        or not isinstance(response.get("cleanup_performed"), bool)
        or not _is_int(response.get("lineage_cleanup_count"))
        or not isinstance(response.get("containment_verified"), bool)
        or response.get("status")
        not in {
            "passed",
            "failed",
            "timed_out",
            "containment_failed",
        }
        or response.get("status") != _expected_status(response)
        or (response.get("error") is not None and not isinstance(response.get("error"), str))
        or not _is_finite_number(response.get("started_at"))
        or not _is_finite_number(response.get("finished_at"))
        or float(response["finished_at"]) < float(response["started_at"])
        or not _is_finite_number(response.get("duration_s"), minimum=0.0)
    ):
        _fail("detached_broker_response_binding_invalid")
    start_origin = (
        start_event["worker_origin"] if isinstance(start_event.get("worker_origin"), dict) else None
    )
    _verify_lifecycle_summary(
        summary=response.get("worker_origin_lifecycle"),
        start_origin=start_origin,
        contract=contract,
        passed=response["status"] == "passed",
    )
    return response


def _verify_quarantine(
    *,
    event: dict[str, Any],
    start: dict[str, Any],
    plan: dict[str, Any],
    policy: dict[str, Any],
    contract: dict[str, Any] | None,
) -> VerifiedDetachedQuarantine:
    receipt = event.get("quarantine_receipt")
    start_origin = start.get("worker_origin")
    if contract is None or not isinstance(start_origin, dict) or not isinstance(receipt, dict):
        _fail("detached_quarantine_without_worker_origin")
    receipt_body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    lifecycle_sha = receipt.get("lifecycle_artifact_sha256")
    quarantined_at = receipt.get("quarantined_at_unix")
    started_at = start.get("recorded_at")
    if (
        set(receipt) != _QUARANTINE_RECEIPT_KEYS
        or receipt.get("schema") != WORKER_ORIGIN_QUARANTINE_RECEIPT_SCHEMA
        or receipt.get("receipt_sha256") != _sha256(receipt_body)
        or event.get("request_id") != start.get("request_id")
        or event.get("policy_sha256") != policy["policy_sha256"]
        or receipt.get("plan_sha256") != plan["plan_sha256"]
        or receipt.get("broker_policy_sha256") != policy["policy_sha256"]
        or receipt.get("request_id") != start["request_id"]
        or receipt.get("supervisor_attempt") != start["attempt"]
        or receipt.get("supervisor_pid") != start.get("supervisor_pid")
        or receipt.get("supervisor_start_token") != start.get("supervisor_start_token")
        or receipt.get("worker_pid") != start.get("worker_pid")
        or receipt.get("worker_process_group_id") != start.get("worker_process_group_id")
        or receipt.get("worker_start_token") != start.get("worker_start_token")
        or receipt.get("containment_token") != start.get("containment_token")
        or receipt.get("worker_origin_contract_sha256") != contract["contract_sha256"]
        or receipt.get("session_id") != start_origin.get("session_id")
        or receipt.get("authorization_request_sha256")
        != start_origin.get("authorization_request_sha256")
        or receipt.get("authorization_attestation_sha256")
        != start_origin.get("authorization_attestation_sha256")
        or receipt.get("payload_path") != start_origin.get("payload_path")
        or receipt.get("request_path") != start_origin.get("request_path")
        or receipt.get("attestation_path") != start_origin.get("attestation_path")
        or receipt.get("lifecycle_path") != start_origin.get("lifecycle_path")
        or (lifecycle_sha is not None and not _is_sha256(lifecycle_sha))
        or receipt.get("prior_journal_head_sha256") != event.get("previous_event_sha256")
        or receipt.get("supervisor_identity_observed") != "dead"
        or receipt.get("worker_identity_observed") != "dead"
        or receipt.get("worker_process_group_empty") is not True
        or not isinstance(receipt.get("cleanup_action_performed"), bool)
        or receipt.get("authority_key_recoverable") is not False
        or receipt.get("lifecycle_recoverable") is not False
        or receipt.get("claim_eligible") is not False
        or receipt.get("reason") != "supervisor_ephemeral_authority_lost"
        or not _is_int(quarantined_at)
        or not _is_finite_number(started_at)
        or quarantined_at < int(float(started_at))
        or event.get("recorded_at") != float(quarantined_at)
    ):
        _fail("detached_quarantine_receipt_invalid")
    return VerifiedDetachedQuarantine(
        attempt=start["attempt"],
        request_id=start["request_id"],
        policy_sha256=policy["policy_sha256"],
        session_id=start_origin["session_id"],
        event_sha256=event["event_sha256"],
        receipt_sha256=receipt["receipt_sha256"],
        prior_journal_head_sha256=receipt["prior_journal_head_sha256"],
    )


def _matches_broker_result(response: dict[str, Any], result: BrokeredProcessResult) -> bool:
    return all(
        (
            response["returncode"] == result.returncode,
            response["request_id"] == result.request_id,
            response["policy_sha256"] == result.policy_sha256,
            response["worker_pid"] == result.worker_pid,
            response["worker_process_group_id"] == result.worker_process_group_id,
            response["worker_start_token"] == result.worker_start_token,
            float(response["started_at"]) == result.started_at,
            float(response["finished_at"]) == result.finished_at,
            float(response["duration_s"]) == result.duration_s,
            response["timed_out"] is result.timed_out,
            response["containment_verified"] is result.containment_verified,
            response["status"] == result.status,
            response["error"] == result.error,
            response["worker_origin_lifecycle"] == result.worker_origin_lifecycle,
            response["receipt_sha256"] == result.receipt_sha256,
            response["response_hmac_sha256"] == result.response_hmac_sha256,
        )
    )


@dataclass(frozen=True)
class VerifiedDetachedQuarantine:
    attempt: int
    request_id: str
    policy_sha256: str
    session_id: str
    event_sha256: str
    receipt_sha256: str
    prior_journal_head_sha256: str


@dataclass(frozen=True)
class VerifiedDetachedBrokerEvidence:
    plan: dict[str, Any]
    journal_head_sha256: str
    attempt: int
    terminal_event: dict[str, Any]
    policy: dict[str, Any]
    request: dict[str, Any]
    quarantine_summaries: tuple[VerifiedDetachedQuarantine, ...]


def verify_detached_broker_evidence(
    *,
    run_dir: Path,
    broker_result: BrokeredProcessResult,
) -> VerifiedDetachedBrokerEvidence:
    """Replay one detached run and bind ``broker_result`` to its exact terminal event."""

    if not isinstance(run_dir, Path) or not isinstance(broker_result, BrokeredProcessResult):
        _fail("detached_evidence_arguments_invalid")
    plan, policies = _read_plan(run_dir)
    events = _read_attempts(run_dir)

    attempts: dict[int, dict[str, Any]] = {}
    starts: dict[str, dict[str, Any]] = {}
    start_contracts: dict[str, dict[str, Any] | None] = {}
    classifications: dict[str, tuple[str, dict[str, Any]]] = {}
    sessions: set[str] = set()
    invocation_counts: dict[tuple[int, str], int] = {}
    quarantines: list[VerifiedDetachedQuarantine] = []
    outer_terminal_seen = False
    target: tuple[dict[str, Any], dict[str, Any], dict[str, Any]] | None = None

    for event in events:
        if outer_terminal_seen:
            _fail("detached_attempt_continues_after_terminal")
        if event.get("plan_sha256") != plan["plan_sha256"]:
            _fail("detached_attempt_plan_mismatch")
        event_type = event.get("event")
        attempt = event.get("attempt")
        if not isinstance(event_type, str) or not _is_int(attempt, minimum=1):
            _fail("detached_attempt_event_invalid")
        state = attempts.get(attempt)

        if event_type == "LAUNCHED":
            if (
                outer_terminal_seen
                or state is not None
                or attempt != len(attempts) + 1
                or any(
                    start["attempt"] < attempt and request_id not in classifications
                    for request_id, start in starts.items()
                )
                or not _is_int(event.get("supervisor_pid"), minimum=1)
                or not isinstance(event.get("supervisor_start_token"), str)
                or not event["supervisor_start_token"]
            ):
                _fail("detached_attempt_launch_invalid")
            attempts[attempt] = {
                "launch": event,
                "control": None,
                "target": None,
                "terminal": None,
            }
            continue
        if state is None or attempt != max(attempts):
            _fail("detached_attempt_event_orphan")
        launch = state["launch"]
        if event_type == "CONTROL_READY":
            if (
                state["control"] is not None
                or state["target"] is not None
                or event.get("supervisor_pid") != launch.get("supervisor_pid")
                or event.get("supervisor_start_token") != launch.get("supervisor_start_token")
                or bool(event.get("broker_enabled")) is not bool(policies)
                or not _is_hex(event.get("control_token"), 64)
                or not _is_hex(event.get("broker_token"), 64)
            ):
                _fail("detached_attempt_control_invalid")
            state["control"] = event
        elif event_type == "TARGET_STARTED":
            if (
                state["control"] is None
                or state["target"] is not None
                or event.get("supervisor_pid") != launch.get("supervisor_pid")
                or event.get("supervisor_start_token") != launch.get("supervisor_start_token")
                or not _is_int(event.get("child_pid"), minimum=1)
                or not _is_int(event.get("child_process_group_id"), minimum=2)
                or not isinstance(event.get("child_start_token"), str)
                or not event["child_start_token"]
                or not _is_hex(event.get("containment_token"), 64)
            ):
                _fail("detached_attempt_target_invalid")
            state["target"] = event
        elif event_type == "BROKER_STARTED":
            request_id = event.get("request_id")
            policy_sha = event.get("policy_sha256")
            policy = policies.get(str(policy_sha))
            key = (attempt, str(policy_sha))
            if (
                state["target"] is None
                or not _is_hex(request_id, 32)
                or request_id in starts
                or policy is None
                or event.get("supervisor_pid") != launch.get("supervisor_pid")
                or event.get("supervisor_start_token") != launch.get("supervisor_start_token")
                or event.get("command_sha256") != policy["command_sha256"]
                or not _is_int(event.get("worker_pid"), minimum=1)
                or event.get("worker_process_group_id") != event.get("worker_pid")
                or not isinstance(event.get("worker_start_token"), str)
                or not event["worker_start_token"]
                or not _is_hex(event.get("containment_token"), 64)
                or not _is_finite_number(event.get("timeout_s"), minimum=0.000001)
                or float(event["timeout_s"]) > float(policy["timeout_s_max"])
            ):
                _fail("detached_broker_start_invalid")
            invocation_counts[key] = invocation_counts.get(key, 0) + 1
            if invocation_counts[key] > policy["max_invocations"]:
                _fail("detached_broker_invocation_bound_exceeded")
            contract, authorization = _verify_worker_origin_start(
                plan=plan,
                policy=policy,
                event=event,
            )
            session_id = authorization.get("session_id") if authorization else None
            if session_id is not None:
                if session_id in sessions:
                    _fail("detached_worker_origin_session_duplicate")
                sessions.add(session_id)
            starts[request_id] = event
            start_contracts[request_id] = contract or None
        elif event_type in {"BROKER_TERMINAL", "BROKER_ORIGIN_QUARANTINED"}:
            request_id = event.get("request_id")
            start = starts.get(str(request_id))
            if start is None:
                _fail("detached_broker_classification_orphan")
            if request_id in classifications:
                _fail("detached_broker_classification_duplicate_or_mixed")
            policy = policies.get(str(start.get("policy_sha256")))
            if policy is None or event.get("policy_sha256") != policy["policy_sha256"]:
                _fail("detached_broker_classification_policy_mismatch")
            contract = start_contracts[request_id]
            if event_type == "BROKER_TERMINAL":
                response = _verify_broker_response(
                    response=event.get("response"),
                    terminal_event=event,
                    start_event=start,
                    control_event=state["control"],
                    policy=policy,
                    contract=contract,
                )
                classifications[request_id] = ("terminal", event)
                if request_id == broker_result.request_id:
                    if target is not None or not _matches_broker_result(response, broker_result):
                        _fail("detached_broker_result_binding_invalid")
                    target = (start, event, policy)
            else:
                quarantine = _verify_quarantine(
                    event=event,
                    start=start,
                    plan=plan,
                    policy=policy,
                    contract=contract,
                )
                quarantines.append(quarantine)
                classifications[request_id] = ("quarantine", event)
        elif event_type == "TERMINAL":
            receipt = event.get("receipt")
            receipt_body = (
                {key: value for key, value in receipt.items() if key != "receipt_sha256"}
                if isinstance(receipt, dict)
                else {}
            )
            if (
                state["terminal"] is not None
                or any(
                    start["attempt"] == attempt and request_id not in classifications
                    for request_id, start in starts.items()
                )
                or not isinstance(receipt, dict)
                or receipt.get("schema") != "aura.detached_step.receipt.v1"
                or receipt.get("plan_sha256") != plan["plan_sha256"]
                or receipt.get("supervisor_attempt") != attempt
                or receipt.get("receipt_sha256") != _sha256(receipt_body)
            ):
                _fail("detached_attempt_terminal_invalid")
            state["terminal"] = event
            outer_terminal_seen = True
        else:
            _fail("detached_attempt_event_type_invalid")

    unfinished = [request_id for request_id in starts if request_id not in classifications]
    if unfinished:
        _fail("detached_worker_origin_session_unfinished")
    if target is None:
        _fail("detached_broker_result_not_found")
    start, terminal, policy = target
    response = terminal["response"]
    if (
        broker_result.policy_sha256 != policy["policy_sha256"]
        or broker_result.returncode != 0
        or broker_result.status != "passed"
        or broker_result.timed_out
        or not broker_result.containment_verified
        or broker_result.error is not None
        or response.get("worker_origin_lifecycle") is None
    ):
        _fail("detached_broker_result_not_claim_eligible")
    return VerifiedDetachedBrokerEvidence(
        plan=json.loads(canonical_json_bytes(plan)),
        journal_head_sha256=events[-1]["event_sha256"],
        attempt=start["attempt"],
        terminal_event=json.loads(canonical_json_bytes(terminal)),
        policy=json.loads(canonical_json_bytes(policy)),
        request=json.loads(canonical_json_bytes(start)),
        quarantine_summaries=tuple(quarantines),
    )


__all__ = [
    "ATTEMPT_EVENT_SCHEMA",
    "BROKER_RESPONSE_SCHEMA",
    "DetachedCampaignEvidenceError",
    "PLAN_SCHEMA",
    "VerifiedDetachedBrokerEvidence",
    "VerifiedDetachedQuarantine",
    "WORKER_ORIGIN_POLICY_SCHEMA",
    "WORKER_ORIGIN_QUARANTINE_RECEIPT_SCHEMA",
    "verify_detached_broker_evidence",
]
