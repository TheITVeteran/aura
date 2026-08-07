#!/usr/bin/env python3
"""Provision same-host, process-isolated principals for a resident campaign.

This is a research launch authority, not evidence of independent organizational
custody. Four distinct signer processes generate non-exportable in-memory role
keys. A one-use root process signs the policy and exits. The resulting bundle
is sufficient to run the production trust protocol while its claim boundary
continues to require later independently administered roots.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import secrets
import socket
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.brain.llm.latent_cortex.campaign_trust import (  # noqa: E402
    CAMPAIGN_TRUST_POLICY_SCHEMA,
    CAMPAIGN_TRUST_ROLES,
    validate_campaign_trust_policy,
)
from core.runtime.file_read_gateway import read_stable_bytes  # noqa: E402

AGENT_REQUEST_SCHEMA = "aura.campaign_role_signer_agent.request.v1"
AGENT_RESPONSE_SCHEMA = "aura.campaign_role_signer_agent.response.v1"
AGENT_STATUS_SCHEMA = "aura.campaign_role_signer_agent.status.v1"
ROOT_RESPONSE_SCHEMA = "aura.campaign_root_signer_once.response.v1"
SIGNER_CONFIG_SCHEMA = "aura.verified_transition.external_signer_config.v1"
RESULT_SCHEMA = "aura.resident_campaign_host_trust_provisioning.v1"
MAX_BYTES = 16 * 1024 * 1024


class TrustProvisioningError(RuntimeError):
    """A trust principal could not be provisioned without ambiguity."""


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError, RecursionError) as exc:
        raise TrustProvisioningError("document_not_canonical") from exc


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(read_stable_bytes(path, max_bytes=512 * 1024 * 1024)).hexdigest()


def _strict_json_bytes(raw: bytes, *, role: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise TrustProvisioningError(f"{role}_json_invalid") from exc
    canonical = _canonical(value)
    if not isinstance(value, dict) or raw not in {canonical, canonical + b"\n"}:
        raise TrustProvisioningError(f"{role}_noncanonical")
    return value


def _read_json(path: Path, *, role: str) -> dict[str, Any]:
    return _strict_json_bytes(
        read_stable_bytes(path.resolve(strict=True), max_bytes=MAX_BYTES),
        role=role,
    )


def _private_directory(path: Path) -> Path:
    resolved = Path(os.path.abspath(os.fspath(path.expanduser())))
    resolved.mkdir(parents=True, exist_ok=True, mode=0o700)
    if resolved.is_symlink():
        raise TrustProvisioningError("trust_root_symlink_rejected")
    metadata = resolved.stat()
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or stat.S_IMODE(metadata.st_mode) & 0o077
    ):
        raise TrustProvisioningError("trust_root_not_private")
    return resolved


def _publish(path: Path, payload: bytes, *, mode: int = 0o600) -> Path:
    if path.is_symlink():
        raise TrustProvisioningError("trust_artifact_symlink_rejected")
    if path.exists():
        observed = read_stable_bytes(path, max_bytes=MAX_BYTES)
        if observed != payload:
            raise TrustProvisioningError("trust_artifact_exists_with_different_bytes")
        return path
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
        mode,
    )
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise TrustProvisioningError("trust_artifact_write_failed")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)
    return path


def _publish_json(path: Path, document: dict[str, Any], *, mode: int = 0o600) -> Path:
    return _publish(path, _canonical(document) + b"\n", mode=mode)


def _copy_executable(source: Path, target: Path) -> Path:
    payload = read_stable_bytes(source.resolve(strict=True), max_bytes=16 * 1024 * 1024)
    _publish(target, payload, mode=0o700)
    os.chmod(target, 0o700)
    return target


def _agent_call(
    socket_path: Path,
    *,
    role: str,
    action: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    body = {
        "schema": AGENT_REQUEST_SCHEMA,
        "action": action,
        "role": role,
        "payload": payload,
    }
    raw = _canonical({**body, "request_sha256": _digest(body)}) + b"\n"
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.settimeout(30.0)
    try:
        client.connect(str(socket_path))
        client.sendall(raw)
        client.shutdown(socket.SHUT_WR)
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = client.recv(min(64 * 1024, MAX_BYTES + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > MAX_BYTES:
                raise TrustProvisioningError("agent_response_too_large")
    finally:
        client.close()
    response = _strict_json_bytes(b"".join(chunks), role="agent_response")
    unsigned = dict(response)
    claimed = unsigned.pop("response_sha256", None)
    if (
        set(response) != {"schema", "ok", "result", "error", "response_sha256"}
        or response.get("schema") != AGENT_RESPONSE_SCHEMA
        or claimed != _digest(unsigned)
        or response.get("ok") is not True
        or not isinstance(response.get("result"), dict)
    ):
        raise TrustProvisioningError(str(response.get("error") or "agent_response_rejected"))
    return response["result"]


def _wait_for_status(path: Path, process: subprocess.Popen[bytes]) -> dict[str, Any]:
    deadline = time.monotonic() + 20.0
    while time.monotonic() < deadline:
        if path.exists():
            status = _read_json(path, role="agent_status")
            if status.get("schema") != AGENT_STATUS_SCHEMA:
                raise TrustProvisioningError("agent_status_schema_invalid")
            return status
        returncode = process.poll()
        if returncode is not None:
            raise TrustProvisioningError(f"signer_agent_exited_{returncode}")
        time.sleep(0.05)
    raise TrustProvisioningError("signer_agent_start_timeout")


def _launcher_bytes(python: Path, client: Path) -> bytes:
    if any("'" in str(path) or "\n" in str(path) for path in (python, client)):
        raise TrustProvisioningError("launcher_path_invalid")
    return (f"#!/bin/sh\nexec '{python}' '{client}' \"$@\"\n").encode("ascii")


def _recover(root: Path, contract: dict[str, Any]) -> dict[str, Any] | None:
    result_path = root / "provisioning-result.json"
    if not result_path.exists():
        return None
    result = _read_json(result_path, role="provisioning_result")
    if (
        result.get("schema") != RESULT_SCHEMA
        or result.get("campaign_id") != contract["campaign_id"]
        or result.get("protocol_sha256") != contract["contract_sha256"]
    ):
        raise TrustProvisioningError("existing_provisioning_identity_mismatch")
    for role, service in result["services"].items():
        status_path = Path(service["status_path"])
        status = _read_json(status_path, role=f"{role}_status")
        if (
            status.get("key_id") != service["key_id"]
            or not Path(service["socket_path"]).is_socket()
        ):
            raise TrustProvisioningError("existing_signer_service_unavailable")
    policy_path = Path(result["trust_policy_path"])
    root_path = Path(result["trust_root_path"])
    policy = validate_campaign_trust_policy(
        _read_json(policy_path, role="campaign_policy"),
        trusted_root_public_key_pem=read_stable_bytes(root_path, max_bytes=64 * 1024),
        expected_campaign_name=contract["campaign_id"],
        expected_protocol_sha256=contract["contract_sha256"],
        now_unix=int(time.time()),
    )
    if policy.policy_sha256 != result["policy_sha256"]:
        raise TrustProvisioningError("existing_policy_identity_mismatch")
    return result


def provision(
    *,
    preregistration_path: Path,
    output_root: Path,
    ttl_seconds: int,
    key_custody: str = "keychain",
) -> dict[str, Any]:
    if key_custody not in {"keychain", "ephemeral"}:
        raise TrustProvisioningError("key_custody_invalid")
    contract = _read_json(preregistration_path, role="preregistration")
    root = _private_directory(output_root)
    recovered = _recover(root, contract)
    if recovered is not None:
        return {**recovered, "reopened": True}
    if any(root.iterdir()):
        raise TrustProvisioningError("partial_trust_provisioning_requires_review")

    bin_dir = _private_directory(root / "bin")
    service_dir = _private_directory(root / "services")
    artifact_dir = _private_directory(root / "artifacts")
    socket_root = _private_directory(
        Path("/tmp")
        / f"aura-signers-{os.getuid()}-{hashlib.sha256(str(root).encode('utf-8')).hexdigest()[:12]}"
    )
    source_agent = REPO_ROOT / "tools/run_campaign_role_signer_agent.py"
    source_client = REPO_ROOT / "tools/run_campaign_role_signer_client.py"
    agent = _copy_executable(source_agent, bin_dir / "role-signer-agent.py")
    client = _copy_executable(source_client, bin_dir / "role-signer-client.py")
    interpreter = Path(os.path.abspath(sys.executable))
    if not interpreter.exists():
        raise TrustProvisioningError("python_interpreter_unavailable")
    agent_sha = _file_sha256(agent)
    client_sha = _file_sha256(client)
    interpreter_sha = _file_sha256(interpreter.resolve(strict=True))
    started_at = int(time.time())
    not_before = started_at
    expires_at = started_at + ttl_seconds

    services: dict[str, dict[str, Any]] = {}
    bootstrap_tokens: dict[str, str] = {}
    processes: list[subprocess.Popen[bytes]] = []
    try:
        socket_names = {
            "task_issuer": "task.sock",
            "campaign_runner": "runner.sock",
            "contamination_auditor": "audit.sock",
            "evidence_verifier": "verify.sock",
        }
        for role in CAMPAIGN_TRUST_ROLES:
            role_dir = _private_directory(service_dir / role)
            socket_path = socket_root / socket_names[role]
            status_path = role_dir / "status.json"
            journal_path = role_dir / "idempotency-journal.json"
            log_path = role_dir / "service.log"
            bootstrap = secrets.token_hex(32)
            bootstrap_tokens[role] = bootstrap
            log = open(log_path, "xb", buffering=0)
            os.chmod(log_path, 0o600)
            environment = {
                "HOME": os.environ.get("HOME", ""),
                "LANG": "C",
                "LC_ALL": "C",
                "AURA_CAMPAIGN_SIGNER_BOOTSTRAP": bootstrap,
            }
            process = subprocess.Popen(
                [
                    str(interpreter),
                    str(agent),
                    "serve",
                    "--role",
                    role,
                    "--socket",
                    str(socket_path),
                    "--status",
                    str(status_path),
                    "--journal",
                    str(journal_path),
                    "--repo-root",
                    str(REPO_ROOT),
                    "--keychain-account",
                    f"{contract['campaign_id']}:{role}",
                    *(["--ephemeral-key"] if key_custody == "ephemeral" else []),
                ],
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                close_fds=True,
                start_new_session=True,
                env=environment,
            )
            log.close()
            processes.append(process)
            status = _wait_for_status(status_path, process)
            if status.get("role") != role or status.get("pid") != process.pid:
                raise TrustProvisioningError("signer_agent_status_identity_mismatch")
            services[role] = {
                "pid": process.pid,
                "socket_path": str(socket_path),
                "status_path": str(status_path),
                "journal_path": str(journal_path),
                "status_sha256": _file_sha256(status_path),
                "public_key_b64": status["public_key_b64"],
                "key_id": status["key_id"],
            }

        role_pins: dict[str, dict[str, Any]] = {}
        signer_configs: dict[str, str] = {}
        for role in CAMPAIGN_TRUST_ROLES:
            role_dir = service_dir / role
            launcher = _publish(
                role_dir / "signer-client",
                _launcher_bytes(interpreter, client),
                mode=0o700,
            )
            os.chmod(launcher, 0o700)
            release = {
                "schema": "aura.campaign_role_signer.release.v1",
                "campaign_id": contract["campaign_id"],
                "role": role,
                "agent_sha256": agent_sha,
                "client_sha256": client_sha,
                "launcher_sha256": _file_sha256(launcher),
                "interpreter_sha256": interpreter_sha,
            }
            release_path = _publish_json(
                artifact_dir / f"{role}-release.json",
                release,
            )
            custody = {
                "schema": "aura.campaign_role_signer.custody.v1",
                "campaign_id": contract["campaign_id"],
                "role": role,
                "custody_mechanism": (
                    "same_host_macos_keychain_process_service"
                    if key_custody == "keychain"
                    else "same_host_ephemeral_process_memory_test"
                ),
                "private_key_exported": False,
                "private_key_persisted": key_custody == "keychain",
                "private_key_persistence_boundary": (
                    "same_user_macos_keychain_not_independent_custody"
                    if key_custody == "keychain"
                    else "none"
                ),
                "service_pid": services[role]["pid"],
                "service_status_sha256": services[role]["status_sha256"],
                "idempotency_journal": services[role]["journal_path"],
                "socket_mode": "0600",
                "claim_boundary": ("host_process_isolation_only_not_independent_organization"),
            }
            custody_path = _publish_json(
                artifact_dir / f"{role}-custody.json",
                custody,
            )
            signer_id = f"{contract['campaign_id']}-{role}-host-service"
            role_pins[role] = {
                "signer_id": signer_id,
                "organization_id": "same-host-aura-research-operator",
                "public_key_b64": services[role]["public_key_b64"],
                "key_id": services[role]["key_id"],
                "implementation_sha256": _file_sha256(launcher),
                "release_sha256": _file_sha256(release_path),
                "custody_class": "host_isolated_service",
                "custody_evidence_sha256": _file_sha256(custody_path),
            }
            config = {
                "schema": SIGNER_CONFIG_SCHEMA,
                "identity": signer_id,
                "executable": str(launcher),
                "executable_sha256": _file_sha256(launcher),
                "release_manifest": str(release_path),
                "custody_evidence": str(custody_path),
                "arguments": [
                    "--socket",
                    services[role]["socket_path"],
                    "--role",
                    role,
                    "--campaign",
                    contract["campaign_id"],
                    "--protocol-sha256",
                    contract["contract_sha256"],
                    "--trust-root",
                    str(artifact_dir / "root-public.pem"),
                    "--repo-root",
                    str(REPO_ROOT),
                ],
                "timeout_millis": 300000,
                "inherited_environment_names": [],
            }
            config_path = _publish_json(
                artifact_dir / f"{role}-signer-config.json",
                config,
            )
            signer_configs[role] = str(config_path)

        policy_body = {
            "schema": CAMPAIGN_TRUST_POLICY_SCHEMA,
            "policy_id": f"{contract['campaign_id']}-host-research-policy",
            "policy_revision": 1,
            "campaign_name": contract["campaign_id"],
            "protocol_sha256": contract["contract_sha256"],
            "previous_policy_sha256": None,
            "revoked_key_ids": [],
            "issued_at_unix": started_at,
            "not_before_unix": not_before,
            "expires_at_unix": expires_at,
            "roles": role_pins,
        }
        root_signer = subprocess.run(
            [str(interpreter), str(agent), "sign-policy-once"],
            input=_canonical(policy_body) + b"\n",
            capture_output=True,
            check=False,
            timeout=30,
        )
        if root_signer.returncode != 0:
            raise TrustProvisioningError("root_signer_failed")
        root_response = _strict_json_bytes(
            root_signer.stdout,
            role="root_signer_response",
        )
        unsigned_root = dict(root_response)
        claimed_root = unsigned_root.pop("response_sha256", None)
        if (
            root_response.get("schema") != ROOT_RESPONSE_SCHEMA
            or claimed_root != _digest(unsigned_root)
            or root_response.get("private_key_exported") is not False
        ):
            raise TrustProvisioningError("root_signer_response_invalid")
        root_pem = base64.b64decode(
            root_response["public_key_pem_b64"],
            validate=True,
        )
        root_path = _publish(artifact_dir / "root-public.pem", root_pem)
        policy = {
            **policy_body,
            "root_signature": {
                "algorithm": "Ed25519",
                "key_id": root_response["key_id"],
                "signature_b64": root_response["signature_b64"],
                "signed_payload_sha256": root_response["signed_payload_sha256"],
            },
        }
        policy_path = _publish_json(artifact_dir / "campaign-policy.json", policy)
        validated = validate_campaign_trust_policy(
            policy,
            trusted_root_public_key_pem=root_pem,
            expected_campaign_name=contract["campaign_id"],
            expected_protocol_sha256=contract["contract_sha256"],
            now_unix=started_at,
        )

        for role in CAMPAIGN_TRUST_ROLES:
            sealed = _agent_call(
                Path(services[role]["socket_path"]),
                role=role,
                action="seal",
                payload={
                    "bootstrap_token": bootstrap_tokens.pop(role),
                    "campaign_id": contract["campaign_id"],
                    "protocol_sha256": contract["contract_sha256"],
                    "policy_sha256": validated.policy_sha256,
                    "signer_id": role_pins[role]["signer_id"],
                    "not_before_unix": not_before,
                    "expires_at_unix": expires_at,
                },
            )
            if (
                sealed.get("sealed") is not True
                or sealed.get("key_id") != services[role]["key_id"]
                or sealed.get("policy_sha256") != validated.policy_sha256
            ):
                raise TrustProvisioningError("signer_agent_seal_failed")

        result = {
            "schema": RESULT_SCHEMA,
            "campaign_id": contract["campaign_id"],
            "protocol_sha256": contract["contract_sha256"],
            "policy_sha256": validated.policy_sha256,
            "trust_policy_path": str(policy_path),
            "trust_root_path": str(root_path),
            "role_signer_config_paths": dict(sorted(signer_configs.items())),
            "task_issuer_signer_config_path": signer_configs["task_issuer"],
            "campaign_runner_signer_config_path": signer_configs["campaign_runner"],
            "contamination_auditor_signer_config_path": signer_configs[
                "contamination_auditor"
            ],
            "evidence_verifier_signer_config_path": signer_configs["evidence_verifier"],
            "services": services,
            "issued_at_unix": started_at,
            "expires_at_unix": expires_at,
            "private_role_keys_exported": False,
            "private_role_keys_persisted_in_keychain": key_custody == "keychain",
            "root_private_key_persisted": False,
            "claim_boundary": (
                "same_host_process_isolation_not_independent_external_administration"
            ),
            "reopened": False,
        }
        _publish_json(root / "provisioning-result.json", result)
        return result
    except Exception:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--ttl-seconds", type=int, default=7 * 24 * 60 * 60)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if not 3600 <= args.ttl_seconds <= 30 * 24 * 60 * 60:
            raise TrustProvisioningError("ttl_seconds_invalid")
        result = provision(
            preregistration_path=args.preregistration,
            output_root=args.output_root,
            ttl_seconds=args.ttl_seconds,
            key_custody="keychain",
        )
        print(json.dumps(result, sort_keys=True, indent=2))
        return 0
    except (
        OSError,
        ValueError,
        subprocess.SubprocessError,
        TrustProvisioningError,
    ) as exc:
        print(f"provision_resident_campaign_trust: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
