from __future__ import annotations

import hashlib
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
