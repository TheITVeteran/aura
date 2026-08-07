#!/usr/bin/env python3
"""Pinned client for one host-isolated campaign role signer."""

from __future__ import annotations

import argparse
import hashlib
import json
import socket
import sys
from pathlib import Path
from typing import Any

COMMAND_SIGNER_REQUEST_SCHEMA = "aura.external_role_signer.request.v1"
COMMAND_SIGNER_RESPONSE_SCHEMA = "aura.external_role_signer.response.v1"
COMMAND_VERIFIER_REQUEST_SCHEMA = "aura.external_evidence_verifier.request.v2"
COMMAND_VERIFIER_RESPONSE_SCHEMA = "aura.external_evidence_verifier.response.v1"
AGENT_REQUEST_SCHEMA = "aura.campaign_role_signer_agent.request.v1"
AGENT_RESPONSE_SCHEMA = "aura.campaign_role_signer_agent.response.v1"
MAX_REQUEST_BYTES = 64 * 1024 * 1024


class SignerClientError(RuntimeError):
    """The command or signer agent violated the pinned protocol."""


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
        raise SignerClientError("document_not_canonical") from exc


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _strict_input(request_file: str | None) -> dict[str, Any]:
    if request_file:
        path = Path(request_file).expanduser()
        if not path.is_absolute() or path.is_symlink() or not path.is_file():
            raise SignerClientError("request_file_invalid")
        raw = path.read_bytes()
    else:
        raw = sys.stdin.buffer.read(MAX_REQUEST_BYTES + 1)
    if len(raw) > MAX_REQUEST_BYTES:
        raise SignerClientError("request_too_large")
    try:
        value = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise SignerClientError("request_json_invalid") from exc
    if not isinstance(value, dict) or raw != _canonical(value) + b"\n":
        raise SignerClientError("request_noncanonical")
    return value


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
    request = _canonical({**body, "request_sha256": _digest(body)}) + b"\n"
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.settimeout(30.0)
    try:
        client.connect(str(socket_path))
        client.sendall(request)
        client.shutdown(socket.SHUT_WR)
        chunks: list[bytes] = []
        size = 0
        while True:
            chunk = client.recv(min(64 * 1024, MAX_REQUEST_BYTES + 1 - size))
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
            if size > MAX_REQUEST_BYTES:
                raise SignerClientError("agent_response_too_large")
    finally:
        client.close()
    raw = b"".join(chunks)
    try:
        response = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise SignerClientError("agent_response_invalid") from exc
    if not isinstance(response, dict) or raw != _canonical(response) + b"\n":
        raise SignerClientError("agent_response_noncanonical")
    unsigned = dict(response)
    claimed = unsigned.pop("response_sha256", None)
    if (
        set(response) != {"schema", "ok", "result", "error", "response_sha256"}
        or response.get("schema") != AGENT_RESPONSE_SCHEMA
        or claimed != _digest(unsigned)
        or response.get("ok") is not True
        or not isinstance(response.get("result"), dict)
    ):
        raise SignerClientError(str(response.get("error") or "agent_response_rejected"))
    return response["result"]


def _sign_role(document: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    allowed_keys = {"schema", "purpose", "signature_request"}
    if "verification_packet_path" in document:
        allowed_keys.add("verification_packet_path")
    if (
        set(document) != allowed_keys
        or document.get("schema") != COMMAND_SIGNER_REQUEST_SCHEMA
        or not isinstance(document.get("purpose"), str)
        or not isinstance(document.get("signature_request"), dict)
    ):
        raise SignerClientError("command_signer_request_invalid")
    purpose = document["purpose"]
    verification_packet = document.get("verification_packet_path")
    if purpose == "verified-recurrent-adapter-activation":
        if not isinstance(verification_packet, str) or not verification_packet:
            raise SignerClientError("activation_verification_packet_required")
        candidate = Path(verification_packet).expanduser()
        if not candidate.is_absolute():
            if not args.request_file:
                raise SignerClientError("activation_verification_packet_path_invalid")
            request_path = Path(args.request_file).expanduser().resolve(strict=True)
            request_root = request_path.parent
            candidate = request_root / candidate
            resolved = candidate.resolve(strict=True)
            if resolved != request_root and not resolved.is_relative_to(request_root):
                raise SignerClientError("activation_verification_packet_path_escape")
        else:
            resolved = candidate.resolve(strict=True)
        if candidate.is_symlink() or not resolved.is_file():
            raise SignerClientError("activation_verification_packet_path_invalid")
        verification_packet = str(resolved)
    elif verification_packet is not None:
        raise SignerClientError("verification_packet_unexpected")
    payload = {
        "purpose": purpose,
        "signature_request": document["signature_request"],
    }
    if verification_packet is not None:
        payload["verification_packet_path"] = verification_packet
    result = _agent_call(
        Path(args.socket),
        role=args.role,
        action="sign",
        payload=payload,
    )
    return {
        "schema": COMMAND_SIGNER_RESPONSE_SCHEMA,
        "request_sha256": result.get("request_sha256"),
        "signature_b64": result.get("signature_b64"),
    }


def _verify_evidence(
    document: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    if args.role != "evidence_verifier":
        raise SignerClientError("evidence_verifier_role_required")
    if (
        document.get("schema") != COMMAND_VERIFIER_REQUEST_SCHEMA
        or not isinstance(document.get("evidence_manifest"), dict)
        or not isinstance(document.get("campaign_trust_policy"), dict)
        or not isinstance(document.get("verifier_identity"), str)
        or type(document.get("verified_at_unix")) is not int
        or not isinstance(document.get("request_sha256"), str)
    ):
        raise SignerClientError("evidence_verifier_request_invalid")
    unsigned = dict(document)
    claimed = unsigned.pop("request_sha256")
    if claimed != _digest(unsigned):
        raise SignerClientError("evidence_verifier_request_digest_mismatch")

    repo_root = Path(args.repo_root).expanduser().resolve(strict=True)
    if not repo_root.is_dir():
        raise SignerClientError("repo_root_invalid")
    sys.path.insert(0, str(repo_root))
    from core.brain.llm.latent_cortex.campaign_trust import (
        validate_campaign_trust_policy,
    )
    from core.learning.verified_recurrent_transition_repository import (
        VerifiedRecurrentTransitionRepositoryError,
        verify_recurrent_evidence_manifest_artifacts,
    )

    trust_material = document["campaign_trust_policy"]
    if set(trust_material) != {"document", "policy_sha256", "root_key_id"}:
        raise SignerClientError("evidence_verifier_trust_material_invalid")
    root_raw = Path(args.trust_root).read_bytes()
    policy = validate_campaign_trust_policy(
        trust_material["document"],
        trusted_root_public_key_pem=root_raw,
        expected_campaign_name=args.campaign,
        expected_policy_sha256=trust_material["policy_sha256"],
        expected_protocol_sha256=args.protocol_sha256,
        now_unix=document["verified_at_unix"],
    )
    if policy.root_key_id != trust_material["root_key_id"]:
        raise SignerClientError("evidence_verifier_root_identity_mismatch")
    try:
        receipt = verify_recurrent_evidence_manifest_artifacts(
            document["evidence_manifest"],
            campaign_trust_policy=policy,
            verifier_identity=document["verifier_identity"],
            verified_at_unix=document["verified_at_unix"],
        )
    except VerifiedRecurrentTransitionRepositoryError as exc:
        raise SignerClientError(f"evidence_verification_failed:{exc}") from exc
    recorded = _agent_call(
        Path(args.socket),
        role=args.role,
        action="record_verification",
        payload={
            "purpose": document["purpose"],
            "verifier_request_sha256": document["request_sha256"],
            "evidence_manifest_sha256": _digest(document["evidence_manifest"]),
            "verification_receipt": receipt,
        },
    )
    if recorded.get("recorded") is not True or recorded.get(
        "verification_receipt"
    ) != receipt:
        raise SignerClientError("evidence_verification_record_rejected")
    return {
        "schema": COMMAND_VERIFIER_RESPONSE_SCHEMA,
        "request_sha256": document["request_sha256"],
        "verification_receipt": receipt,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", required=True)
    parser.add_argument("--role", required=True)
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--trust-root", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--request-file")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        document = _strict_input(args.request_file)
        if document.get("schema") == COMMAND_SIGNER_REQUEST_SCHEMA:
            response = _sign_role(document, args)
        elif document.get("schema") == COMMAND_VERIFIER_REQUEST_SCHEMA:
            response = _verify_evidence(document, args)
        else:
            raise SignerClientError("command_schema_unsupported")
        sys.stdout.buffer.write(_canonical(response) + b"\n")
        return 0
    except (OSError, SignerClientError, ValueError) as exc:
        print(f"campaign_role_signer_client: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
