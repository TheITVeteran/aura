#!/usr/bin/env python
"""Prepare and assemble detached RLC campaign signatures.

This tool deliberately has no private-key or key-generation option. Production
keys remain in an independently operated signer or HSM; Aura emits canonical
bytes and accepts only a verified detached Ed25519 signature in return.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.brain.llm.latent_cortex.campaign_journal import (  # noqa: E402
    canonical_json_bytes,
)
from core.brain.llm.latent_cortex.campaign_trust import (  # noqa: E402
    CAMPAIGN_TRUST_ROLES,
    CampaignTrustError,
    assemble_role_attestation,
    assemble_signed_campaign_policy,
    externally_custodied_roles,
    prepare_policy_signature_request,
    prepare_role_signature_request,
    validate_campaign_trust_policy,
)
from core.runtime.file_read_gateway import read_stable_bytes  # noqa: E402

_MAX_JSON_BYTES = 16 * 1024 * 1024
_MAX_KEY_BYTES = 64 * 1024
_MAX_SIGNATURE_BYTES = 64 * 1024


class CampaignTrustToolError(RuntimeError):
    """Stable operator-facing trust workflow error."""


def _lexical_path(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path.expanduser())))


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, child in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = child
    return value


def _read_json(path: Path, *, role: str) -> dict[str, Any]:
    payload = read_stable_bytes(_lexical_path(path), max_bytes=_MAX_JSON_BYTES)
    try:
        value = json.loads(
            payload,
            object_pairs_hook=_strict_object,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant: {constant}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise CampaignTrustToolError(f"{role} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise CampaignTrustToolError(f"{role} must be a JSON object")
    return value


def _read_key(path: Path) -> bytes:
    return read_stable_bytes(_lexical_path(path), max_bytes=_MAX_KEY_BYTES)


def _read_signature(path: Path) -> str:
    raw = read_stable_bytes(_lexical_path(path), max_bytes=_MAX_SIGNATURE_BYTES)
    signature_b64: str
    try:
        document = json.loads(
            raw,
            object_pairs_hook=_strict_object,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant: {constant}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        document = None
    if isinstance(document, dict) and set(document) == {"signature_b64"}:
        signature_b64 = str(document["signature_b64"])
    else:
        try:
            text = raw.decode("ascii").strip()
        except UnicodeDecodeError:
            text = ""
        try:
            decoded = base64.b64decode(text, validate=True)
        except (ValueError, binascii.Error):
            decoded = b""
        if len(decoded) == 64:
            signature_b64 = text
        elif len(raw) == 64:
            signature_b64 = base64.b64encode(raw).decode("ascii")
        else:
            raise CampaignTrustToolError(
                "signature must be 64 raw Ed25519 bytes, canonical base64, or "
                "a JSON object containing only signature_b64"
            )
    try:
        decoded = base64.b64decode(signature_b64, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise CampaignTrustToolError("signature is not canonical base64") from exc
    if len(decoded) != 64 or base64.b64encode(decoded).decode("ascii") != signature_b64:
        raise CampaignTrustToolError("signature is not a canonical Ed25519 signature")
    return signature_b64


def _atomic_create_or_verify(path: Path, document: dict[str, Any]) -> None:
    payload = canonical_json_bytes(document) + b"\n"
    destination = _lexical_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        raise CampaignTrustToolError("symlink output rejected")
    if destination.exists():
        if read_stable_bytes(destination, max_bytes=_MAX_JSON_BYTES) == payload:
            return
        raise CampaignTrustToolError("refusing to overwrite a different artifact")
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    fd = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    try:
        try:
            view = memoryview(payload)
            while view:
                written = os.write(fd, view)
                if written <= 0:
                    raise CampaignTrustToolError("short artifact write")
                view = view[written:]
            os.fsync(fd)
        finally:
            os.close(fd)
        try:
            os.link(temporary, destination, follow_symlinks=False)
        except FileExistsError:
            if read_stable_bytes(destination, max_bytes=_MAX_JSON_BYTES) != payload:
                raise CampaignTrustToolError(
                    "refusing to overwrite a concurrently created artifact"
                ) from None
    finally:
        temporary.unlink(missing_ok=True)
    directory_fd = os.open(destination.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _emit(document: dict[str, Any], out: Path | None) -> None:
    if out is not None:
        _atomic_create_or_verify(out, document)
    sys.stdout.buffer.write(canonical_json_bytes(document) + b"\n")


def _observed_at(value: int | None) -> int:
    return int(time.time()) if value is None else value


def _load_policy(args: argparse.Namespace):
    return validate_campaign_trust_policy(
        _read_json(args.policy, role="campaign policy"),
        trusted_root_public_key_pem=_read_key(args.root),
        expected_campaign_name=args.campaign_name or None,
        expected_protocol_sha256=args.protocol_sha256 or None,
        minimum_policy_revision=args.minimum_revision,
        now_unix=_observed_at(args.observed_at),
    )


def _add_policy_context(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--campaign-name", default="")
    parser.add_argument("--protocol-sha256", default="")
    parser.add_argument("--minimum-revision", type=int)
    parser.add_argument("--observed-at", type=int)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    policy_request = subparsers.add_parser(
        "policy-request", help="emit exact unsigned policy bytes for the external root"
    )
    policy_request.add_argument("--unsigned-policy", type=Path, required=True)
    policy_request.add_argument("--root", type=Path, required=True)
    policy_request.add_argument("--campaign-name", default="")
    policy_request.add_argument("--protocol-sha256", default="")
    policy_request.add_argument("--minimum-revision", type=int)
    policy_request.add_argument("--observed-at", type=int)
    policy_request.add_argument("--out", type=Path)

    policy_assemble = subparsers.add_parser(
        "policy-assemble", help="verify a detached root signature and seal the policy"
    )
    policy_assemble.add_argument("--request", type=Path, required=True)
    policy_assemble.add_argument("--root", type=Path, required=True)
    policy_assemble.add_argument("--signature", type=Path, required=True)
    policy_assemble.add_argument("--campaign-name", default="")
    policy_assemble.add_argument("--protocol-sha256", default="")
    policy_assemble.add_argument("--minimum-revision", type=int)
    policy_assemble.add_argument("--observed-at", type=int)
    policy_assemble.add_argument("--out", type=Path)

    role_request = subparsers.add_parser(
        "role-request", help="emit exact role-attestation bytes for a remote signer"
    )
    _add_policy_context(role_request)
    role_request.add_argument("--role", choices=CAMPAIGN_TRUST_ROLES, required=True)
    role_request.add_argument("--payload", type=Path, required=True)
    role_request.add_argument("--signed-at", type=int, required=True)
    role_request.add_argument("--out", type=Path)

    role_assemble = subparsers.add_parser(
        "role-assemble", help="verify and seal a detached role signature"
    )
    _add_policy_context(role_assemble)
    role_assemble.add_argument("--role", choices=CAMPAIGN_TRUST_ROLES, required=True)
    role_assemble.add_argument("--request", type=Path, required=True)
    role_assemble.add_argument("--signature", type=Path, required=True)
    role_assemble.add_argument("--out", type=Path)

    inspect = subparsers.add_parser(
        "inspect", help="authenticate a policy and print its public role identities"
    )
    _add_policy_context(inspect)
    inspect.add_argument("--out", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.command == "policy-request":
            document = prepare_policy_signature_request(
                _read_json(args.unsigned_policy, role="unsigned campaign policy"),
                trusted_root_public_key_pem=_read_key(args.root),
                expected_campaign_name=args.campaign_name or None,
                expected_protocol_sha256=args.protocol_sha256 or None,
                minimum_policy_revision=args.minimum_revision,
                now_unix=args.observed_at,
            )
        elif args.command == "policy-assemble":
            verified = assemble_signed_campaign_policy(
                _read_json(args.request, role="policy signature request"),
                signature_b64=_read_signature(args.signature),
                trusted_root_public_key_pem=_read_key(args.root),
                expected_campaign_name=args.campaign_name or None,
                expected_protocol_sha256=args.protocol_sha256 or None,
                minimum_policy_revision=args.minimum_revision,
                now_unix=_observed_at(args.observed_at),
            )
            document = verified.document
        elif args.command == "role-request":
            verified = _load_policy(args)
            document = prepare_role_signature_request(
                verified,
                role=args.role,
                payload=_read_json(args.payload, role="role payload"),
                signed_at_unix=args.signed_at,
            )
        elif args.command == "role-assemble":
            verified = _load_policy(args)
            document = assemble_role_attestation(
                verified,
                _read_json(args.request, role="role signature request"),
                signature_b64=_read_signature(args.signature),
                role=args.role,
            )
        else:
            verified = _load_policy(args)
            document = {
                "schema": "aura.latent_cortex.campaign_trust_identity.v1",
                "policy_id": verified.document["policy_id"],
                "policy_revision": verified.document["policy_revision"],
                "campaign_name": verified.document["campaign_name"],
                "policy_sha256": verified.policy_sha256,
                "root_key_id": verified.root_key_id,
                "protocol_sha256": verified.document["protocol_sha256"],
                "externally_custodied": externally_custodied_roles(verified),
                "roles": verified.document["roles"],
            }
        _emit(document, args.out)
        return 0
    except (CampaignTrustError, CampaignTrustToolError, OSError, ValueError) as exc:
        error = {
            "schema": "aura.latent_cortex.campaign_trust_tool_error.v1",
            "ok": False,
            "reason": getattr(exc, "code", str(exc)) or type(exc).__name__,
        }
        sys.stdout.buffer.write(canonical_json_bytes(error) + b"\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
