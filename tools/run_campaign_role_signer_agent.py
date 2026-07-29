#!/usr/bin/env python3
"""Host-isolated, policy-scoped Ed25519 role signer.

Role private keys exist only inside the long-lived service process. The service
publishes public identity, accepts one authenticated policy seal, and signs only
canonical role requests for that sealed campaign. A separate one-shot mode
signs the root policy and exits without exporting the root private key.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import signal
import socket
import stat
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

ROLE_REQUEST_SCHEMA = "aura.latent_cortex.campaign_role_signature_request.v1"
ROLE_PAYLOAD_SCHEMA = "aura.latent_cortex.campaign_role_payload.v2"
AGENT_STATUS_SCHEMA = "aura.campaign_role_signer_agent.status.v1"
AGENT_REQUEST_SCHEMA = "aura.campaign_role_signer_agent.request.v1"
AGENT_RESPONSE_SCHEMA = "aura.campaign_role_signer_agent.response.v1"
ROOT_RESPONSE_SCHEMA = "aura.campaign_root_signer_once.response.v1"
MAX_REQUEST_BYTES = 16 * 1024 * 1024
IDEMPOTENCY_JOURNAL_SCHEMA = "aura.campaign_role_signer_agent.idempotency.v1"
ALLOWED_ROLES = {
    "task_issuer",
    "campaign_runner",
    "contamination_auditor",
    "evidence_verifier",
}


class SignerAgentError(RuntimeError):
    """A signer request violated the sealed service contract."""


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
        raise SignerAgentError("document_not_canonical") from exc


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha256(value: Any, *, role: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SignerAgentError(f"{role}_invalid")
    return value


def _identifier(value: Any, *, role: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > 256
        or any(
            character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._:-"
            for character in value
        )
    ):
        raise SignerAgentError(f"{role}_invalid")
    return value


def _strict_json(raw: bytes) -> dict[str, Any]:
    if len(raw) > MAX_REQUEST_BYTES:
        raise SignerAgentError("request_too_large")
    try:
        value = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise SignerAgentError("request_json_invalid") from exc
    if not isinstance(value, dict) or raw != _canonical(value) + b"\n":
        raise SignerAgentError("request_noncanonical")
    return value


def _atomic_create(path: Path, payload: bytes, *, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.is_symlink() or path.exists():
        raise SignerAgentError("status_path_exists")
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
                raise SignerAgentError("status_write_failed")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)


def _public_raw(private_key: Ed25519PrivateKey) -> bytes:
    return private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )


def _load_private_key(args: argparse.Namespace) -> tuple[Ed25519PrivateKey, str]:
    if args.ephemeral_key:
        return Ed25519PrivateKey.generate(), "process_memory_ephemeral"
    repo_root = Path(args.repo_root).expanduser().resolve(strict=True)
    if not repo_root.is_dir():
        raise SignerAgentError("repo_root_invalid")
    sys.path.insert(0, str(repo_root))
    from core.security.zenith_secrets import require_keychain_backend

    backend = require_keychain_backend()
    encoded = backend.get_password(args.keychain_service, args.keychain_account)
    if encoded is None:
        candidate = Ed25519PrivateKey.generate()
        raw = candidate.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption(),
        )
        encoded = base64.b64encode(raw).decode("ascii")
        if backend.set_password(args.keychain_service, args.keychain_account, encoded) is not True:
            raise SignerAgentError("keychain_private_key_write_failed")
        if backend.get_password(args.keychain_service, args.keychain_account) != encoded:
            raise SignerAgentError("keychain_private_key_write_unconfirmed")
    try:
        raw = base64.b64decode(encoded, validate=True)
        if len(raw) != 32:
            raise ValueError("wrong key length")
        return Ed25519PrivateKey.from_private_bytes(raw), "macos_keychain"
    except (TypeError, ValueError) as exc:
        raise SignerAgentError("keychain_private_key_invalid") from exc


def _load_idempotency_journal(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "schema": IDEMPOTENCY_JOURNAL_SCHEMA,
            "sequence": 0,
            "head_sha256": "0" * 64,
            "entries": {},
        }
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise SignerAgentError("idempotency_journal_invalid") from exc
    if (
        not isinstance(value, dict)
        or raw != _canonical(value) + b"\n"
        or value.get("schema") != IDEMPOTENCY_JOURNAL_SCHEMA
        or type(value.get("sequence")) is not int
        or not isinstance(value.get("head_sha256"), str)
        or not isinstance(value.get("entries"), dict)
    ):
        raise SignerAgentError("idempotency_journal_invalid")
    return value


def _replace_private(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise SignerAgentError("journal_write_failed")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)


def _record_idempotent_result(
    journal: dict[str, Any],
    path: Path,
    *,
    idempotency_key: str,
    request_sha256: str,
    result: Mapping[str, Any],
) -> dict[str, Any]:
    entries = journal["entries"]
    existing = entries.get(idempotency_key)
    if existing is not None:
        if existing.get("request_sha256") != request_sha256:
            raise SignerAgentError("idempotency_key_conflict")
        return dict(existing["result"])
    sequence = int(journal["sequence"]) + 1
    event = {
        "sequence": sequence,
        "previous_event_sha256": journal["head_sha256"],
        "idempotency_key": idempotency_key,
        "request_sha256": request_sha256,
        "result": dict(result),
    }
    event_sha256 = _digest(event)
    entries[idempotency_key] = {**event, "event_sha256": event_sha256}
    journal["sequence"] = sequence
    journal["head_sha256"] = event_sha256
    _replace_private(path, _canonical(journal) + b"\n")
    return dict(result)


def _response(*, ok: bool, result: Mapping[str, Any] | None = None, error: str = "") -> bytes:
    body = {
        "schema": AGENT_RESPONSE_SCHEMA,
        "ok": ok,
        "result": dict(result or {}),
        "error": error,
    }
    return _canonical({**body, "response_sha256": _digest(body)}) + b"\n"


def _read_connection(connection: socket.socket) -> bytes:
    chunks: list[bytes] = []
    size = 0
    while True:
        chunk = connection.recv(min(64 * 1024, MAX_REQUEST_BYTES + 1 - size))
        if not chunk:
            break
        chunks.append(chunk)
        size += len(chunk)
        if size > MAX_REQUEST_BYTES:
            raise SignerAgentError("request_too_large")
    return b"".join(chunks)


def _validate_signature_request(
    request: Mapping[str, Any],
    *,
    role: str,
    campaign_id: str,
    protocol_sha256: str,
    policy_sha256: str,
    signer_id: str,
    public_raw: bytes,
    not_before_unix: int,
    expires_at_unix: int,
    purpose: str,
) -> tuple[bytes, str]:
    expected_keys = {
        "schema",
        "algorithm",
        "key_id",
        "public_key_b64",
        "signed_payload",
        "signed_payload_sha256",
        "signed_payload_b64",
        "request_sha256",
    }
    unsigned = dict(request)
    claimed_request = unsigned.pop("request_sha256", None)
    if (
        set(request) != expected_keys
        or request.get("schema") != ROLE_REQUEST_SCHEMA
        or request.get("algorithm") != "Ed25519"
        or claimed_request != _digest(unsigned)
    ):
        raise SignerAgentError("signature_request_invalid")
    expected_key_id = hashlib.sha256(public_raw).hexdigest()
    expected_public = base64.b64encode(public_raw).decode("ascii")
    if request.get("key_id") != expected_key_id or request.get("public_key_b64") != expected_public:
        raise SignerAgentError("signature_request_key_mismatch")
    payload = request.get("signed_payload")
    if not isinstance(payload, dict):
        raise SignerAgentError("signature_payload_invalid")
    payload_bytes = _canonical(payload)
    if (
        request.get("signed_payload_sha256") != hashlib.sha256(payload_bytes).hexdigest()
        or request.get("signed_payload_b64") != base64.b64encode(payload_bytes).decode("ascii")
        or payload.get("schema") != ROLE_PAYLOAD_SCHEMA
        or payload.get("policy_sha256") != policy_sha256
        or payload.get("campaign_name") != campaign_id
        or payload.get("protocol_sha256") != protocol_sha256
        or payload.get("role") != role
        or payload.get("signer_id") != signer_id
        or payload.get("purpose") != purpose
    ):
        raise SignerAgentError("signature_payload_mismatch")
    operation = _identifier(payload.get("operation"), role="operation")
    idempotency_key = _sha256(payload.get("idempotency_key"), role="idempotency_key")
    signed_at = payload.get("signed_at_unix")
    if type(signed_at) is not int or not not_before_unix <= signed_at < expires_at_unix:
        raise SignerAgentError("signature_time_outside_policy")
    if role == "task_issuer":
        expected_operation = (
            "campaign_manifest"
            if purpose == f"{campaign_id}:campaign-manifest"
            else "group_lineage"
            if purpose.startswith(f"{campaign_id}:group:") and purpose.endswith(":lineage")
            else "group_manifest"
            if purpose.startswith(f"{campaign_id}:group:") and purpose.endswith(":manifest")
            else ""
        )
        if operation != expected_operation:
            raise SignerAgentError("task_issuer_purpose_rejected")
    elif role == "evidence_verifier":
        if purpose != "verified-recurrent-campaign-close" or operation != "campaign_close":
            raise SignerAgentError("evidence_verifier_purpose_rejected")
    else:
        raise SignerAgentError("inactive_campaign_role_rejected")
    return payload_bytes, idempotency_key


def _serve(args: argparse.Namespace) -> int:
    role = _identifier(args.role, role="role")
    if role not in ALLOWED_ROLES:
        raise SignerAgentError("role_invalid")
    bootstrap = os.environ.pop("AURA_CAMPAIGN_SIGNER_BOOTSTRAP", "")
    if len(bootstrap) < 64:
        raise SignerAgentError("bootstrap_token_missing")
    private_key, key_custody = _load_private_key(args)
    public_raw = _public_raw(private_key)
    key_id = hashlib.sha256(public_raw).hexdigest()
    socket_path = Path(args.socket).expanduser()
    status_path = Path(args.status).expanduser()
    if not socket_path.is_absolute() or socket_path.is_symlink() or socket_path.exists():
        raise SignerAgentError("socket_path_invalid")
    if not status_path.is_absolute():
        raise SignerAgentError("status_path_invalid")
    socket_parent = socket_path.parent
    if socket_parent.is_symlink() or not socket_parent.is_dir():
        raise SignerAgentError("socket_parent_invalid")
    socket_metadata = socket_parent.stat()
    if socket_metadata.st_uid != os.getuid() or stat.S_IMODE(socket_metadata.st_mode) & 0o077:
        raise SignerAgentError("socket_parent_not_private")

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(socket_path))
    os.chmod(socket_path, 0o600)
    server.listen(8)
    server.settimeout(2.0)
    started = int(time.time())
    status_body = {
        "schema": AGENT_STATUS_SCHEMA,
        "role": role,
        "pid": os.getpid(),
        "started_at_unix": started,
        "socket_path": str(socket_path),
        "public_key_b64": base64.b64encode(public_raw).decode("ascii"),
        "key_id": key_id,
        "private_key_exported": False,
        "private_key_custody": key_custody,
    }
    _atomic_create(
        status_path,
        _canonical({**status_body, "status_sha256": _digest(status_body)}) + b"\n",
    )

    sealed: dict[str, Any] | None = None
    journal_path = Path(args.journal).expanduser()
    if not journal_path.is_absolute():
        raise SignerAgentError("journal_path_invalid")
    journal = _load_idempotency_journal(journal_path)
    stopping = False

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    try:
        while not stopping:
            if sealed is not None and int(time.time()) >= sealed["expires_at_unix"]:
                break
            try:
                connection, _ = server.accept()
            except TimeoutError:
                continue
            with connection:
                try:
                    document = _strict_json(_read_connection(connection))
                    if (
                        set(document) != {"schema", "action", "role", "payload", "request_sha256"}
                        or document.get("schema") != AGENT_REQUEST_SCHEMA
                    ):
                        raise SignerAgentError("agent_request_invalid")
                    unsigned = dict(document)
                    claimed = unsigned.pop("request_sha256")
                    if claimed != _digest(unsigned) or document.get("role") != role:
                        raise SignerAgentError("agent_request_identity_mismatch")
                    action = document.get("action")
                    payload = document.get("payload")
                    if not isinstance(payload, dict):
                        raise SignerAgentError("agent_request_payload_invalid")
                    if action == "seal":
                        if sealed is not None or payload.pop("bootstrap_token", None) != bootstrap:
                            raise SignerAgentError("agent_seal_rejected")
                        candidate = {
                            "campaign_id": _identifier(
                                payload.get("campaign_id"), role="campaign_id"
                            ),
                            "protocol_sha256": _sha256(
                                payload.get("protocol_sha256"), role="protocol_sha256"
                            ),
                            "policy_sha256": _sha256(
                                payload.get("policy_sha256"), role="policy_sha256"
                            ),
                            "signer_id": _identifier(payload.get("signer_id"), role="signer_id"),
                            "not_before_unix": payload.get("not_before_unix"),
                            "expires_at_unix": payload.get("expires_at_unix"),
                        }
                        if (
                            type(candidate["not_before_unix"]) is not int
                            or type(candidate["expires_at_unix"]) is not int
                            or candidate["not_before_unix"] >= candidate["expires_at_unix"]
                        ):
                            raise SignerAgentError("agent_policy_window_invalid")
                        sealed = candidate
                        bootstrap = ""
                        result = {
                            "role": role,
                            "key_id": key_id,
                            "policy_sha256": sealed["policy_sha256"],
                            "sealed": True,
                        }
                    elif action == "sign":
                        if sealed is None:
                            raise SignerAgentError("agent_not_sealed")
                        purpose = _identifier(payload.get("purpose"), role="purpose")
                        signature_request = payload.get("signature_request")
                        if not isinstance(signature_request, dict):
                            raise SignerAgentError("signature_request_invalid")
                        signed_payload, idempotency_key = _validate_signature_request(
                            signature_request,
                            role=role,
                            campaign_id=sealed["campaign_id"],
                            protocol_sha256=sealed["protocol_sha256"],
                            policy_sha256=sealed["policy_sha256"],
                            signer_id=sealed["signer_id"],
                            public_raw=public_raw,
                            not_before_unix=sealed["not_before_unix"],
                            expires_at_unix=sealed["expires_at_unix"],
                            purpose=purpose,
                        )
                        if role == "evidence_verifier":
                            close_payload = signature_request["signed_payload"].get("payload")
                            if not isinstance(close_payload, dict):
                                raise SignerAgentError("evidence_close_payload_invalid")
                            close_receipt = close_payload.get(
                                "external_evidence_verification_receipt"
                            )
                            close_evidence = close_payload.get("evidence_manifest")
                            if not isinstance(close_receipt, dict) or not isinstance(
                                close_evidence, dict
                            ):
                                raise SignerAgentError("evidence_close_receipt_missing")
                            evidence_sha256 = _digest(close_evidence)
                            verified = any(
                                isinstance(entry, dict)
                                and isinstance(entry.get("result"), dict)
                                and entry["result"].get("verification_receipt")
                                == close_receipt
                                and entry["result"].get("evidence_manifest_sha256")
                                == evidence_sha256
                                for entry in journal["entries"].values()
                            )
                            if not verified:
                                raise SignerAgentError(
                                    "evidence_close_without_recorded_verification"
                                )
                        unsigned_result = {
                            "request_sha256": signature_request["request_sha256"],
                            "signature_b64": base64.b64encode(
                                private_key.sign(signed_payload)
                            ).decode("ascii"),
                        }
                        result = _record_idempotent_result(
                            journal,
                            journal_path,
                            idempotency_key=idempotency_key,
                            request_sha256=signature_request["request_sha256"],
                            result=unsigned_result,
                        )
                    elif action == "record_verification":
                        if role != "evidence_verifier" or sealed is None:
                            raise SignerAgentError("verification_record_role_rejected")
                        if payload.get("purpose") != (
                            "verified-recurrent-campaign-evidence-replay"
                        ):
                            raise SignerAgentError("verification_record_purpose_rejected")
                        receipt = payload.get("verification_receipt")
                        evidence_sha256 = _sha256(
                            payload.get("evidence_manifest_sha256"),
                            role="evidence_manifest_sha256",
                        )
                        request_sha256 = _sha256(
                            payload.get("verifier_request_sha256"),
                            role="verifier_request_sha256",
                        )
                        if not isinstance(receipt, dict):
                            raise SignerAgentError("verification_receipt_invalid")
                        idempotency_key = _digest(
                            {
                                "action": "record_verification",
                                "policy_sha256": sealed["policy_sha256"],
                                "request_sha256": request_sha256,
                                "evidence_manifest_sha256": evidence_sha256,
                            }
                        )
                        result = _record_idempotent_result(
                            journal,
                            journal_path,
                            idempotency_key=idempotency_key,
                            request_sha256=request_sha256,
                            result={
                                "recorded": True,
                                "verification_receipt": receipt,
                                "evidence_manifest_sha256": evidence_sha256,
                            },
                        )
                    elif action == "shutdown":
                        if payload.get("bootstrap_token") != bootstrap or not bootstrap:
                            raise SignerAgentError("agent_shutdown_rejected")
                        stopping = True
                        result = {"stopping": True}
                    else:
                        raise SignerAgentError("agent_action_invalid")
                    connection.sendall(_response(ok=True, result=result))
                except (OSError, SignerAgentError) as exc:
                    connection.sendall(_response(ok=False, error=str(exc) or type(exc).__name__))
    finally:
        server.close()
        try:
            socket_path.unlink()
        except OSError:
            pass
    return 0


def _sign_policy_once() -> int:
    raw = sys.stdin.buffer.read(MAX_REQUEST_BYTES + 1)
    body = _strict_json(raw)
    private_key = Ed25519PrivateKey.generate()
    public_raw = _public_raw(private_key)
    public_pem = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    signed = _canonical(body)
    response_body = {
        "schema": ROOT_RESPONSE_SCHEMA,
        "public_key_pem_b64": base64.b64encode(public_pem).decode("ascii"),
        "key_id": hashlib.sha256(public_raw).hexdigest(),
        "signed_payload_sha256": hashlib.sha256(signed).hexdigest(),
        "signature_b64": base64.b64encode(private_key.sign(signed)).decode("ascii"),
        "private_key_exported": False,
    }
    sys.stdout.buffer.write(
        _canonical({**response_body, "response_sha256": _digest(response_body)}) + b"\n"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    serve = subparsers.add_parser("serve")
    serve.add_argument("--role", required=True)
    serve.add_argument("--socket", required=True)
    serve.add_argument("--status", required=True)
    serve.add_argument("--journal", required=True)
    serve.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    serve.add_argument("--keychain-service", default="AuraCampaignRoleSigners")
    serve.add_argument("--keychain-account", required=True)
    serve.add_argument("--ephemeral-key", action="store_true")
    subparsers.add_parser("sign-policy-once")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.command == "serve":
            return _serve(args)
        return _sign_policy_once()
    except (OSError, SignerAgentError) as exc:
        print(f"campaign_role_signer_agent: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
