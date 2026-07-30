from __future__ import annotations

import io
import json
from pathlib import Path

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
