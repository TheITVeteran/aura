#!/usr/bin/env python3
"""Prepare and assemble externally signed Reality Reach witness bundles."""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.reality_reach.acceptance import AcceptanceCertificateStore  # noqa: E402
from core.reality_reach.acceptance_mandate import (  # noqa: E402
    AcceptanceMandateStore,
)
from core.reality_reach.acceptance_witness import (  # noqa: E402
    ZERO_SHA256,
    AcceptanceWitnessBundle,
    AcceptanceWitnessRole,
    AcceptanceWitnessStatement,
)
from core.runtime.secure_path_custody import (  # noqa: E402
    DirectoryCustody,
    SecurePathCustodyError,
)

_MAX_DOCUMENT_BYTES = 1024 * 1024


def _read_bytes(path: Path, *, max_bytes: int) -> bytes:
    target = path.expanduser().absolute()
    try:
        with DirectoryCustody.acquire(target.parent, create=False) as custody:
            payload = custody.read_bytes(target.name, max_bytes=max_bytes)
    except SecurePathCustodyError as exc:
        raise ValueError("acceptance_witness_input_read_failed") from exc
    if not isinstance(payload, bytes):
        raise ValueError("acceptance_witness_input_invalid")
    return payload


def _read_json(path: Path) -> Mapping[str, Any]:
    payload = _read_bytes(path, max_bytes=_MAX_DOCUMENT_BYTES)

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        document: dict[str, Any] = {}
        for key, value in pairs:
            if key in document:
                raise ValueError("acceptance_witness_duplicate_json_key")
            document[key] = value
        return document

    try:
        document = json.loads(payload, object_pairs_hook=unique_object)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ValueError("acceptance_witness_json_invalid") from exc
    if not isinstance(document, Mapping):
        raise ValueError("acceptance_witness_json_invalid")
    return document


def _write_once(path: Path, payload: bytes, *, mode: int) -> bool:
    target = path.expanduser().absolute()
    try:
        with DirectoryCustody.acquire(target.parent, create=True, private=True) as custody:
            published = bool(custody.write_bytes_once(target.name, payload, mode=mode))
            existing = custody.read_bytes(target.name, max_bytes=_MAX_DOCUMENT_BYTES)
    except SecurePathCustodyError as exc:
        raise ValueError("acceptance_witness_output_custody_invalid") from exc
    if existing != payload:
        raise ValueError("acceptance_witness_output_collision")
    return published


def _statement(args: argparse.Namespace) -> int:
    mandate_store = AcceptanceMandateStore.from_system(args.mandate_state)
    try:
        mandate = mandate_store.get(args.campaign_id)
    finally:
        mandate_store.close()
    certificate = AcceptanceCertificateStore(args.root).load(args.campaign_id)
    role = AcceptanceWitnessRole(args.role)
    evidence_sha256 = (
        certificate.metrology_evidence_sha256
        if role is AcceptanceWitnessRole.METROLOGY
        else certificate.governance_evidence_sha256
    )
    if not evidence_sha256:
        raise ValueError(f"acceptance_{role.value}_evidence_missing")
    witnessed_at_ns = args.witnessed_at_ns or max(
        time.time_ns(),
        certificate.completed_at_ns + 1,
    )
    statement = AcceptanceWitnessStatement(
        role=role,
        witness_id=args.witness_id,
        campaign_id=mandate.campaign_id,
        mandate_sha256=mandate.sha256,
        certificate_sha256=certificate.sha256,
        evidence_sha256=evidence_sha256,
        sequence=args.sequence,
        previous_statement_sha256=args.previous_statement_sha256,
        witnessed_at_ns=witnessed_at_ns,
    )
    document = json.dumps(
        statement.to_dict(),
        indent=2,
        sort_keys=True,
    ).encode("utf-8")
    payload = json.dumps(
        statement.to_dict(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    _write_once(args.statement_output, document, mode=0o600)
    _write_once(args.payload_output, payload, mode=0o600)
    print(
        json.dumps(
            {
                "statement": statement.to_dict(),
                "signed_payload_b64": base64.b64encode(payload).decode("ascii"),
                "statement_output": str(args.statement_output.expanduser().absolute()),
                "payload_output": str(args.payload_output.expanduser().absolute()),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _assemble(args: argparse.Namespace) -> int:
    statement = AcceptanceWitnessStatement.from_dict(_read_json(args.statement))
    public_key = _read_bytes(args.public_key_raw, max_bytes=32)
    signature = _read_bytes(args.signature, max_bytes=64)
    if len(public_key) != 32 or len(signature) != 64:
        raise ValueError("acceptance_witness_key_or_signature_size_invalid")
    bundle = AcceptanceWitnessBundle(
        statement=statement,
        public_key_raw_b64=base64.b64encode(public_key).decode("ascii"),
        signature_b64=base64.b64encode(signature).decode("ascii"),
    )
    try:
        Ed25519PublicKey.from_public_bytes(public_key).verify(
            signature,
            bundle.signed_payload(),
        )
    except InvalidSignature as exc:
        raise ValueError("acceptance_witness_signature_invalid") from exc
    payload = json.dumps(
        bundle.to_dict(),
        indent=2,
        sort_keys=True,
    ).encode("utf-8")
    _write_once(args.output, payload, mode=0o600)
    print(json.dumps(bundle.to_dict(), indent=2, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    statement = commands.add_parser(
        "statement",
        help="emit canonical bytes for an external custodian to sign",
    )
    statement.add_argument("--root", type=Path, required=True)
    statement.add_argument("--mandate-state", type=Path, required=True)
    statement.add_argument("--campaign-id", required=True)
    statement.add_argument(
        "--role",
        required=True,
        choices=[item.value for item in AcceptanceWitnessRole],
    )
    statement.add_argument("--witness-id", required=True)
    statement.add_argument("--sequence", type=int, required=True)
    statement.add_argument(
        "--previous-statement-sha256",
        default=ZERO_SHA256,
    )
    statement.add_argument("--witnessed-at-ns", type=int, default=0)
    statement.add_argument("--statement-output", type=Path, required=True)
    statement.add_argument("--payload-output", type=Path, required=True)
    statement.set_defaults(handler=_statement)

    assemble = commands.add_parser(
        "assemble",
        help="verify a detached Ed25519 signature and create the witness bundle",
    )
    assemble.add_argument("--statement", type=Path, required=True)
    assemble.add_argument("--public-key-raw", type=Path, required=True)
    assemble.add_argument("--signature", type=Path, required=True)
    assemble.add_argument("--output", type=Path, required=True)
    assemble.set_defaults(handler=_assemble)

    args = parser.parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
