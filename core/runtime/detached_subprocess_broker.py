"""Client for the detached supervisor's exact-command subprocess broker."""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import secrets
import shutil
import socket
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.governance_context import local_internal_governed_scope
from core.runtime.file_write_gateway import get_file_write_gateway

REQUEST_SCHEMA = "aura.detached_step.broker_request.v2"
RESPONSE_SCHEMA = "aura.detached_step.broker_response.v1"
BROKER_SOCKET_ENV = "AURA_DETACHED_BROKER_SOCKET"
BROKER_TOKEN_ENV = "AURA_DETACHED_BROKER_TOKEN"
_MAX_DATAGRAM_BYTES = 65_536
_MAX_REQUEST_DATAGRAM_BYTES = 2_048


class DetachedBrokerError(RuntimeError):
    """Raised when a brokered subprocess cannot be proven or completed."""


@dataclass(frozen=True)
class BrokeredProcessResult:
    returncode: int
    request_id: str
    policy_sha256: str
    worker_pid: int
    worker_process_group_id: int
    worker_start_token: str
    started_at: float
    finished_at: float
    duration_s: float
    timed_out: bool
    containment_verified: bool
    status: str
    error: str | None
    worker_origin_lifecycle: dict[str, Any] | None
    receipt_sha256: str
    response_hmac_sha256: str


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _command_sha256(command: list[str]) -> str:
    return hashlib.sha256(_canonical_bytes(command)).hexdigest()


def compute_broker_request_binding(
    command: list[str],
    *,
    cwd: str,
    stdout_path: str,
) -> str:
    """Bind a request to one exact frozen broker policy without retransmitting it."""

    return hashlib.sha256(
        _canonical_bytes(
            {
                "command": command,
                "cwd": cwd,
                "stdout_path": stdout_path,
            }
        )
    ).hexdigest()


def _normalized_command(command: list[str], cwd: Path) -> list[str]:
    executable = command[0]
    try:
        if "/" in executable:
            candidate = Path(executable).expanduser()
            if not candidate.is_absolute():
                candidate = cwd / candidate
        else:
            located = shutil.which(executable, path=os.environ.get("PATH"))
            if located is None:
                raise DetachedBrokerError(
                    f"broker command executable is unavailable: {executable}"
                )
            candidate = Path(located)
        launcher = candidate.parent.resolve(strict=True) / candidate.name
        resolved_target = launcher.resolve(strict=True)
    except OSError as exc:
        raise DetachedBrokerError(
            f"broker command executable is unavailable: {executable}"
        ) from exc
    if not resolved_target.is_file() or not os.access(launcher, os.X_OK):
        raise DetachedBrokerError(
            f"broker command executable is not executable: {launcher}"
        )
    return [str(launcher), *command[1:]]


def broker_available() -> bool:
    return bool(os.environ.get(BROKER_SOCKET_ENV) and os.environ.get(BROKER_TOKEN_ENV))


def run_brokered_process(
    command: list[str],
    *,
    cwd: Path,
    stdout_path: Path,
    timeout_s: float,
) -> BrokeredProcessResult:
    if not command or any(not isinstance(item, str) or not item for item in command):
        raise DetachedBrokerError("broker command must be a non-empty string array")
    if timeout_s <= 0.0:
        raise DetachedBrokerError("broker timeout must be positive")
    raw_socket_path = os.environ.get(BROKER_SOCKET_ENV, "")
    broker_token = os.environ.get(BROKER_TOKEN_ENV, "")
    if not raw_socket_path or len(broker_token) != 64:
        raise DetachedBrokerError("detached subprocess broker is unavailable")
    broker_socket_path = Path(raw_socket_path)
    try:
        broker_stat = broker_socket_path.lstat()
    except OSError as exc:
        raise DetachedBrokerError("detached subprocess broker socket is unavailable") from exc
    if not stat.S_ISSOCK(broker_stat.st_mode) or broker_stat.st_uid != os.geteuid():
        raise DetachedBrokerError("detached subprocess broker socket identity is invalid")

    resolved_cwd = cwd.expanduser().resolve(strict=True)
    normalized_command = _normalized_command(command, resolved_cwd)
    request_id = secrets.token_hex(16)
    reply_name = f"aura-broker-reply-{os.geteuid()}-{os.getpid()}-{request_id}.sock"
    reply_path = Path("/tmp") / reply_name
    if len(os.fsencode(reply_path)) >= 100:
        raise DetachedBrokerError("detached subprocess broker reply path is too long")
    resolved_stdout_path = str(stdout_path.expanduser().resolve(strict=False))
    request = {
        "schema": REQUEST_SCHEMA,
        "action": "run",
        "broker_token": broker_token,
        "request_id": request_id,
        "command_sha256": _command_sha256(normalized_command),
        "request_binding_sha256": compute_broker_request_binding(
            normalized_command,
            cwd=str(resolved_cwd),
            stdout_path=resolved_stdout_path,
        ),
        "timeout_s": float(timeout_s),
        "reply_path": str(reply_path),
    }
    payload = _canonical_bytes(request)
    if len(payload) > _MAX_REQUEST_DATAGRAM_BYTES:
        raise DetachedBrokerError("detached subprocess broker request is too large")

    with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as client:
        try:
            previous_umask = os.umask(0o077)
            try:
                client.bind(str(reply_path))
            finally:
                os.umask(previous_umask)
            reply_stat = reply_path.lstat()
            if reply_stat.st_uid != os.geteuid() or reply_stat.st_mode & 0o077:
                raise DetachedBrokerError("detached subprocess broker reply socket is not private")
            client.settimeout(timeout_s + 30.0)
            client.sendto(payload, str(broker_socket_path))
            response_payload = client.recv(_MAX_DATAGRAM_BYTES)
        except (OSError, TimeoutError) as exc:
            raise DetachedBrokerError("detached subprocess broker did not return a result") from exc
        finally:
            with local_internal_governed_scope(
                "runtime.detached_subprocess_broker.reply_socket",
                domain="file_write",
            ):
                get_file_write_gateway().delete_file(
                    reply_path,
                    source="runtime.detached_subprocess_broker.reply_socket",
                )
    try:
        response = json.loads(response_payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DetachedBrokerError("detached subprocess broker returned invalid JSON") from exc
    if not isinstance(response, dict):
        raise DetachedBrokerError("detached subprocess broker response must be an object")
    receipt_sha = str(response.get("receipt_sha256") or "")
    response_hmac = str(response.get("response_hmac_sha256") or "")
    signed = {key: value for key, value in response.items() if key != "response_hmac_sha256"}
    body = {key: value for key, value in signed.items() if key != "receipt_sha256"}
    common_binding_valid = (
        response.get("schema") == RESPONSE_SCHEMA
        and response.get("request_id") == request_id
        and response.get("command_sha256") == request["command_sha256"]
        and len(receipt_sha) == 64
        and hashlib.sha256(_canonical_bytes(body)).hexdigest() == receipt_sha
        and len(response_hmac) == 64
        and hmac.compare_digest(
            response_hmac,
            hmac.new(
                bytes.fromhex(broker_token),
                _canonical_bytes(signed),
                hashlib.sha256,
            ).hexdigest(),
        )
    )
    if common_binding_valid and response.get("status") == "rejected":
        raise DetachedBrokerError(str(response.get("error") or "broker request rejected"))
    if (
        not common_binding_valid
        or not isinstance(response.get("policy_sha256"), str)
        or len(response["policy_sha256"]) != 64
        or not isinstance(response.get("returncode"), int)
        or isinstance(response.get("returncode"), bool)
        or not isinstance(response.get("worker_pid"), int)
        or int(response["worker_pid"]) <= 0
        or not isinstance(response.get("worker_process_group_id"), int)
        or isinstance(response.get("worker_process_group_id"), bool)
        or int(response["worker_process_group_id"]) <= 1
        or not isinstance(response.get("worker_start_token"), str)
        or not response["worker_start_token"]
        or not isinstance(response.get("started_at"), (int, float))
        or isinstance(response.get("started_at"), bool)
        or not math.isfinite(float(response["started_at"]))
        or not isinstance(response.get("finished_at"), (int, float))
        or isinstance(response.get("finished_at"), bool)
        or not math.isfinite(float(response["finished_at"]))
        or float(response["finished_at"]) < float(response["started_at"])
        or not isinstance(response.get("duration_s"), (int, float))
        or isinstance(response.get("duration_s"), bool)
        or not math.isfinite(float(response["duration_s"]))
        or float(response["duration_s"]) < 0.0
        or not isinstance(response.get("timed_out"), bool)
        or not isinstance(response.get("containment_verified"), bool)
        or response.get("status")
        not in {"passed", "failed", "timed_out", "containment_failed"}
        or (
            response.get("error") is not None
            and not isinstance(response.get("error"), str)
        )
        or (
            response.get("worker_origin_lifecycle") is not None
            and not isinstance(response.get("worker_origin_lifecycle"), dict)
        )
    ):
        raise DetachedBrokerError("detached subprocess broker response binding is invalid")
    return BrokeredProcessResult(
        returncode=int(response["returncode"]),
        request_id=request_id,
        policy_sha256=str(response["policy_sha256"]),
        worker_pid=int(response["worker_pid"]),
        worker_process_group_id=int(response["worker_process_group_id"]),
        worker_start_token=str(response["worker_start_token"]),
        started_at=float(response["started_at"]),
        finished_at=float(response["finished_at"]),
        duration_s=float(response["duration_s"]),
        timed_out=bool(response["timed_out"]),
        containment_verified=bool(response["containment_verified"]),
        status=str(response["status"]),
        error=response.get("error"),
        worker_origin_lifecycle=(
            dict(response["worker_origin_lifecycle"])
            if isinstance(response.get("worker_origin_lifecycle"), dict)
            else None
        ),
        receipt_sha256=receipt_sha,
        response_hmac_sha256=response_hmac,
    )
