from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import run_campaign_role_signer_client as signer_client


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii") + b"\n"


def test_strict_input_reads_canonical_absolute_request_file(tmp_path: Path) -> None:
    request = {
        "purpose": "test:sign",
        "schema": signer_client.COMMAND_SIGNER_REQUEST_SCHEMA,
        "signature_request": {"role": "task_issuer"},
    }
    request_path = tmp_path / "request.json"
    request_path.write_bytes(_canonical(request))

    assert signer_client._strict_input(str(request_path)) == request


def test_strict_input_rejects_relative_or_symlink_request_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_path = tmp_path / "request.json"
    request_path.write_bytes(_canonical({"schema": "test"}))
    symlink_path = tmp_path / "request-link.json"
    symlink_path.symlink_to(request_path)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(signer_client.SignerClientError, match="request_file_invalid"):
        signer_client._strict_input("request.json")
    with pytest.raises(signer_client.SignerClientError, match="request_file_invalid"):
        signer_client._strict_input(str(symlink_path))


def test_strict_input_preserves_canonical_stdin_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = {"schema": "test", "value": 7}
    stdin = io.TextIOWrapper(io.BytesIO(_canonical(request)), encoding="ascii")
    monkeypatch.setattr(signer_client.sys, "stdin", stdin)

    assert signer_client._strict_input(None) == request


def test_activation_signer_resolves_transfer_packet_next_to_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    packet = tmp_path / "external-signer-packet.json"
    packet.write_bytes(_canonical({"schema": "fixture.packet"}))
    request_file = tmp_path / "external-signer-command.json"
    request_file.write_bytes(b"fixture")
    observed: dict[str, object] = {}

    def _call(_socket: Path, **kwargs: object) -> dict[str, object]:
        observed.update(kwargs)
        return {"request_sha256": "a" * 64, "signature_b64": "c2ln"}

    monkeypatch.setattr(signer_client, "_agent_call", _call)
    args = SimpleNamespace(
        socket=str(tmp_path / "signer.sock"),
        role="evidence_verifier",
        request_file=str(request_file),
    )
    document = {
        "schema": signer_client.COMMAND_SIGNER_REQUEST_SCHEMA,
        "purpose": "verified-recurrent-adapter-activation",
        "signature_request": {"schema": "fixture.request"},
        "verification_packet_path": "external-signer-packet.json",
    }

    response = signer_client._sign_role(document, args)

    assert response["request_sha256"] == "a" * 64
    assert observed["payload"]["verification_packet_path"] == str(packet.resolve())


def test_activation_signer_rejects_relative_packet_escape(tmp_path: Path) -> None:
    request_root = tmp_path / "request"
    request_root.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_bytes(_canonical({"schema": "fixture.packet"}))
    request_file = request_root / "command.json"
    request_file.write_bytes(b"fixture")
    args = SimpleNamespace(
        socket=str(tmp_path / "signer.sock"),
        role="evidence_verifier",
        request_file=str(request_file),
    )

    with pytest.raises(
        signer_client.SignerClientError,
        match="activation_verification_packet_path_escape",
    ):
        signer_client._sign_role(
            {
                "schema": signer_client.COMMAND_SIGNER_REQUEST_SCHEMA,
                "purpose": "verified-recurrent-adapter-activation",
                "signature_request": {"schema": "fixture.request"},
                "verification_packet_path": "../outside.json",
            },
            args,
        )
