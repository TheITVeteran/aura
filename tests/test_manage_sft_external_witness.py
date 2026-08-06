from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from core.brain.llm.latent_cortex.campaign_journal import canonical_json_bytes
from core.governance_context import local_internal_governed_scope
from core.runtime.subprocess_gateway import get_subprocess_gateway
from tools.manage_sft_external_witness import (
    ExternalWitnessToolError,
    _atomic_bytes_create_or_verify,
    _extract_uuid,
    _submit,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL = REPO_ROOT / "tools" / "manage_sft_external_witness.py"
ZERO = "0" * 64


def _run(*arguments: str) -> subprocess.CompletedProcess[str]:
    with local_internal_governed_scope(
        "tests.spark.external_witness.operator_cli",
        domain="tool_execution",
    ):
        return get_subprocess_gateway().run(
            [sys.executable, os.fspath(TOOL), *arguments],
            cwd=REPO_ROOT,
            capture_output=True,
            check=False,
            timeout=30,
            source="tests.spark.external_witness.operator_cli",
            accelerator_capability="none",
        )


def test_packet_and_raw_signing_payload_are_deterministic(tmp_path: Path):
    packet = tmp_path / "packet.json"
    statement = tmp_path / "statement.json"
    signing_payload = tmp_path / "statement.canonical.json"
    result = _run(
        "packet",
        "--source-git-commit",
        "1" * 40,
        "--out",
        os.fspath(packet),
    )
    assert result.returncode == 0, result.stdout
    result = _run(
        "statement",
        "--packet",
        os.fspath(packet),
        "--sequence",
        "1",
        "--previous-statement-sha256",
        ZERO,
        "--previous-rekor-uuid",
        "none",
        "--issued-at",
        "1785082400",
        "--signing-payload-out",
        os.fspath(signing_payload),
        "--out",
        os.fspath(statement),
    )
    assert result.returncode == 0, result.stdout
    document = json.loads(statement.read_bytes())
    assert signing_payload.read_bytes() == canonical_json_bytes(document)
    assert statement.read_bytes() == canonical_json_bytes(document) + b"\n"
    assert signing_payload.stat().st_mode & 0o777 == 0o600


def test_raw_signing_payload_refuses_different_existing_content(tmp_path: Path):
    destination = tmp_path / "payload"
    _atomic_bytes_create_or_verify(destination, b"first")
    _atomic_bytes_create_or_verify(destination, b"first")
    with pytest.raises(
        ExternalWitnessToolError,
        match="refusing to overwrite a different signing payload",
    ):
        _atomic_bytes_create_or_verify(destination, b"second")


def test_upload_uuid_is_bound_to_location_field():
    uuid = "1" * 80
    assert _extract_uuid({"Location": f"/api/v1/log/entries/{uuid}"}) == uuid
    with pytest.raises(ExternalWitnessToolError, match="rekor_upload_uuid_ambiguous"):
        _extract_uuid(
            {
                "Location": f"/api/v1/log/entries/{uuid}",
                "UUID": "2" * 80,
            }
        )


def test_submit_rejects_symlinked_client_before_network(tmp_path: Path):
    executable = tmp_path / "rekor-real"
    executable.write_bytes(b"#!/bin/sh\nexit 0\n")
    executable.chmod(0o700)
    symlink = tmp_path / "rekor-cli"
    symlink.symlink_to(executable)
    args = argparse.Namespace(rekor_cli=symlink, out=None)
    with pytest.raises(ExternalWitnessToolError, match="rekor_cli_not_regular_executable"):
        _submit(args)


def test_committed_cp403_bundle_passes_offline_operator_verification():
    result = _run(
        "verify",
        "--packet",
        os.fspath(
            REPO_ROOT
            / "artifacts/current/cp403_spark_059_production_audit_packet.json"
        ),
        "--sequence",
        "1",
        "--previous-statement-sha256",
        ZERO,
        "--previous-rekor-uuid",
        "none",
        "--bundle",
        os.fspath(REPO_ROOT / "artifacts/current/cp403_rekor_witness_bundle.json"),
        "--trusted-log-key",
        os.fspath(REPO_ROOT / "config/trust/sigstore_rekor_public_good_v1.pem"),
    )
    assert result.returncode == 0, result.stdout
    validation = json.loads(result.stdout)
    assert validation["status"] == "externally_witnessed_audit_head_verified_offline"
    assert validation["trainer_ready"] is False
