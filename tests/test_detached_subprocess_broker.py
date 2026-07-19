from __future__ import annotations

import hashlib
import hmac
import json
import os
import socket
import threading
import uuid
from pathlib import Path

import pytest

from core.runtime import detached_subprocess_broker as broker


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def test_client_rejects_forged_unkeyed_broker_response(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    socket_path = Path("/tmp") / f"aura-broker-test-{uuid.uuid4().hex}.sock"
    token = "a" * 64
    server = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    server.bind(str(socket_path))
    os.chmod(socket_path, 0o600)
    monkeypatch.setenv(broker.BROKER_SOCKET_ENV, str(socket_path))
    monkeypatch.setenv(broker.BROKER_TOKEN_ENV, token)

    def forge_response() -> None:
        payload = server.recv(65_536)
        request = json.loads(payload)
        body = {
            "schema": broker.RESPONSE_SCHEMA,
            "request_id": request["request_id"],
            "policy_sha256": "b" * 64,
            "command_sha256": request["command_sha256"],
            "worker_pid": os.getpid(),
            "worker_process_group_id": os.getpgrp(),
            "worker_start_token": "forged",
            "returncode": 0,
            "timed_out": False,
            "containment_verified": True,
            "status": "passed",
            "error": None,
        }
        signed = {
            **body,
            "receipt_sha256": hashlib.sha256(_canonical_bytes(body)).hexdigest(),
        }
        response = {**signed, "response_hmac_sha256": "0" * 64}
        server.sendto(_canonical_bytes(response), request["reply_path"])

    thread = threading.Thread(target=forge_response, daemon=True)
    thread.start()
    try:
        with pytest.raises(broker.DetachedBrokerError, match="response binding"):
            broker.run_brokered_process(
                [str(Path(os.__file__).resolve()), "unused"],
                cwd=tmp_path,
                stdout_path=tmp_path / "worker.log",
                timeout_s=2.0,
            )
    finally:
        thread.join(timeout=2.0)
        server.close()
        socket_path.unlink(missing_ok=True)


def test_client_preserves_authenticated_supervisor_lifecycle_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    socket_path = Path("/tmp") / f"aura-broker-test-{uuid.uuid4().hex}.sock"
    token = "b" * 64
    server = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    server.bind(str(socket_path))
    os.chmod(socket_path, 0o600)
    monkeypatch.setenv(broker.BROKER_SOCKET_ENV, str(socket_path))
    monkeypatch.setenv(broker.BROKER_TOKEN_ENV, token)
    lifecycle = {
        "artifact_path": str(tmp_path / "lifecycle.json"),
        "artifact_sha256": "c" * 64,
        "event_type": "terminal",
        "event_sha256": "d" * 64,
        "result_count": 3,
        "session_id": "e" * 32,
    }

    def authenticated_response() -> None:
        payload = server.recv(65_536)
        request = json.loads(payload)
        body = {
            "schema": broker.RESPONSE_SCHEMA,
            "request_id": request["request_id"],
            "policy_sha256": "f" * 64,
            "command_sha256": request["command_sha256"],
            "worker_pid": os.getpid(),
            "worker_process_group_id": os.getpgrp(),
            "worker_start_token": "authenticated",
            "started_at": 10.0,
            "finished_at": 12.5,
            "duration_s": 2.5,
            "returncode": 0,
            "timed_out": False,
            "cleanup_performed": False,
            "lineage_cleanup_count": 0,
            "containment_verified": True,
            "status": "passed",
            "error": None,
            "worker_origin_lifecycle": lifecycle,
        }
        signed = {
            **body,
            "receipt_sha256": hashlib.sha256(_canonical_bytes(body)).hexdigest(),
        }
        response = {
            **signed,
            "response_hmac_sha256": hmac.new(
                bytes.fromhex(token),
                _canonical_bytes(signed),
                hashlib.sha256,
            ).hexdigest(),
        }
        server.sendto(_canonical_bytes(response), request["reply_path"])

    thread = threading.Thread(target=authenticated_response, daemon=True)
    thread.start()
    try:
        result = broker.run_brokered_process(
            [str(Path(os.__file__).resolve()), "unused"],
            cwd=tmp_path,
            stdout_path=tmp_path / "worker.log",
            timeout_s=2.0,
        )
        assert result.returncode == 0
        assert result.policy_sha256 == "f" * 64
        assert result.worker_process_group_id == os.getpgrp()
        assert result.started_at == 10.0
        assert result.finished_at == 12.5
        assert result.duration_s == 2.5
        assert result.containment_verified is True
        assert result.status == "passed"
        assert result.worker_origin_lifecycle == lifecycle
        assert len(result.response_hmac_sha256) == 64
    finally:
        thread.join(timeout=2.0)
        server.close()
        socket_path.unlink(missing_ok=True)


def test_broker_availability_requires_both_socket_and_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(broker.BROKER_SOCKET_ENV, raising=False)
    monkeypatch.delenv(broker.BROKER_TOKEN_ENV, raising=False)
    assert broker.broker_available() is False
    monkeypatch.setenv(broker.BROKER_SOCKET_ENV, "/var/empty/not-enough.sock")
    assert broker.broker_available() is False
    monkeypatch.setenv(broker.BROKER_TOKEN_ENV, "a" * 64)
    assert broker.broker_available() is True
