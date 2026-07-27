from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from core.learning.recurrent_sft_falsification import sha256_json
from tools import verify_recurrent_sft_falsification as verifier


def _contract(tmp_path: Path) -> tuple[dict, dict]:
    profile = tmp_path / "evaluator.sb"
    profile.write_text("(version 1)\n(deny default)\n")
    source_closure = {
        "schema": "source",
        "files": [],
        "closure_sha256": "1" * 64,
    }
    custody = {"candidate": {}, "evaluator": {}, "custody": {}}
    command = [
        "/usr/bin/sandbox-exec",
        "-f",
        str(profile),
        "/usr/bin/python3",
        "evaluator.py",
    ]
    environment = {"PYTHONHASHSEED": "0"}
    body = {
        "source_closure": source_closure,
        "authority_sha256": "2" * 64,
        "model_identity_sha256": "3" * 64,
        "execution_spec_sha256": "4" * 64,
        "custody_binding_sha256": "5" * 64,
        "custody_bindings": custody,
        "network": "kernel_denied",
        "process_fork": "kernel_denied",
        "evaluator_access": True,
        "training_write_access": False,
        "resident_checkpoint_access": False,
        "production_write_access": False,
        "resume_contract": "none",
        "profile_path": str(profile),
        "profile_sha256": hashlib.sha256(profile.read_bytes()).hexdigest(),
        "sandbox_executable_sha256": hashlib.sha256(
            Path("/usr/bin/sandbox-exec").read_bytes()
        ).hexdigest(),
        "command": command,
        "command_sha256": sha256_json(command),
        "environment": environment,
        "environment_sha256": sha256_json(environment),
    }
    contract = {**body, "contract_sha256": sha256_json(body)}
    report = {
        "containment_contract_sha256": contract["contract_sha256"],
        "source_closure": source_closure,
        "authority_sha256": "2" * 64,
        "model_identity_sha256": "3" * 64,
        "execution_spec_sha256": "4" * 64,
        "custody_binding_sha256": "5" * 64,
        "custody": custody,
    }
    return contract, report


def test_contract_replays_execution_and_custody_bindings(tmp_path: Path) -> None:
    contract, report = _contract(tmp_path)
    verifier._verify_contract(contract, report)

    report["custody_binding_sha256"] = "6" * 64
    with pytest.raises(
        verifier.RecurrentSFTFalsificationVerificationError,
        match="contract_invalid",
    ):
        verifier._verify_contract(contract, report)


def test_binding_rejects_symlink(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    artifact.write_bytes(b"bound")
    link = tmp_path / "link"
    link.symlink_to(artifact)
    with pytest.raises(
        verifier.RecurrentSFTFalsificationVerificationError,
        match="symlink_rejected",
    ):
        verifier._verify_binding(
            {
                "path": str(link),
                "sha256": hashlib.sha256(b"bound").hexdigest(),
                "size_bytes": 5,
            },
            role="test",
        )


def test_receipt_requires_exact_contained_command(tmp_path: Path) -> None:
    contract, _report = _contract(tmp_path)
    receipt = {
        "returncode": 0,
        "timed_out": False,
        "process_group_empty": True,
        "duration_s": 1.0,
        "status": "passed",
        "restart_count": 0,
        "containment_verified": True,
        "command": contract["command"],
        "executed_command": contract["command"],
        "command_sha256": contract["command_sha256"],
        "lineage_empty": True,
    }
    verifier._verify_receipt(receipt, contract=contract)

    receipt["executed_command"] = ["/bin/echo", "substituted"]
    with pytest.raises(
        verifier.RecurrentSFTFalsificationVerificationError,
        match="detached_receipt_invalid",
    ):
        verifier._verify_receipt(receipt, contract=contract)
