#!/usr/bin/env python3
"""Prepare, assemble, and independently verify acceptance preregistration."""

from __future__ import annotations

import argparse
import base64
import json
import os
import stat
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.reality_reach.acceptance_mandate import (  # noqa: E402
    AcceptanceMandateProvisionReceipt,
    AcceptanceMandateStore,
    AcceptanceVerificationMandate,
)
from core.reality_reach.acceptance_preregistration import (  # noqa: E402
    build_acceptance_preregistration_bundle,
    build_acceptance_preregistration_statement,
    persist_preregistered_acceptance_receipt,
    verify_acceptance_preregistration,
)
from core.reality_reach.acceptance_transparency import ZERO_SHA256  # noqa: E402
from core.runtime.secure_path_custody import (  # noqa: E402
    DirectoryCustody,
    SecurePathCustodyError,
)

_MAX_DOCUMENT_BYTES = 4 * 1024 * 1024


def _read_bytes(path: Path, *, maximum: int = _MAX_DOCUMENT_BYTES) -> bytes:
    target = path.expanduser().absolute()
    try:
        with DirectoryCustody.acquire(target.parent, create=False) as custody:
            payload = custody.read_bytes(target.name, max_bytes=maximum)
    except SecurePathCustodyError as exc:
        raise ValueError("acceptance_preregistration_input_read_failed") from exc
    if not isinstance(payload, bytes):
        raise ValueError("acceptance_preregistration_input_read_failed")
    return payload


def _read_json(path: Path) -> Mapping[str, Any]:
    payload = _read_bytes(path)

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        document: dict[str, Any] = {}
        for key, value in pairs:
            if key in document:
                raise ValueError("acceptance_preregistration_duplicate_json_key")
            document[key] = value
        return document

    try:
        document = json.loads(payload, object_pairs_hook=unique_object)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ValueError("acceptance_preregistration_json_invalid") from exc
    if not isinstance(document, Mapping):
        raise ValueError("acceptance_preregistration_json_invalid")
    return document


def _write_once(path: Path, payload: bytes) -> bool:
    target = path.expanduser().absolute()
    try:
        with DirectoryCustody.acquire(target.parent, create=True, private=True) as custody:
            published = bool(custody.write_bytes_once(target.name, payload, mode=0o600))
            fd = custody.open_file(target.name, os.O_RDONLY)
            try:
                if stat.S_IMODE(os.fstat(fd).st_mode) != 0o600:
                    raise ValueError("acceptance_preregistration_output_mode_invalid")
            finally:
                os.close(fd)
            existing = custody.read_bytes(target.name, max_bytes=_MAX_DOCUMENT_BYTES)
    except SecurePathCustodyError as exc:
        raise ValueError("acceptance_preregistration_output_custody_invalid") from exc
    if existing != payload:
        raise ValueError("acceptance_preregistration_output_collision")
    return published


def _provision_receipt(path: Path) -> AcceptanceMandateProvisionReceipt:
    document = _read_json(path)
    raw = document.get("provision_receipt", document)
    if not isinstance(raw, Mapping):
        raise ValueError("acceptance_preregistration_provision_receipt_invalid")
    return AcceptanceMandateProvisionReceipt.from_dict(raw)


def _portable_mandate(path: Path) -> AcceptanceVerificationMandate:
    document = _read_json(path)
    raw = document.get("mandate", document)
    if not isinstance(raw, Mapping):
        raise ValueError("acceptance_preregistration_mandate_invalid")
    return AcceptanceVerificationMandate.from_dict(raw)


def _system_mandate(path: Path, campaign_id: str) -> AcceptanceVerificationMandate:
    store = AcceptanceMandateStore.from_system(path)
    try:
        return store.get(campaign_id)
    finally:
        store.close()


def _statement(args: argparse.Namespace) -> int:
    mandate = _system_mandate(args.mandate_state, args.campaign_id)
    provision_receipt = _provision_receipt(args.provision_receipt)
    statement = build_acceptance_preregistration_statement(
        mandate,
        provision_receipt,
        sequence=args.sequence,
        previous_statement_sha256=args.previous_statement_sha256,
        previous_rekor_uuid=args.previous_rekor_uuid,
        issued_at_unix=args.issued_at_unix or int(time.time()),
    )
    payload = json.dumps(
        statement,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    _write_once(
        args.statement_output,
        json.dumps(statement, indent=2, sort_keys=True).encode("utf-8"),
    )
    _write_once(args.payload_output, payload)
    print(
        json.dumps(
            {
                "statement": statement,
                "signed_payload_b64": base64.b64encode(payload).decode("ascii"),
                "rekor_kind": "rekord",
                "statement_output": str(args.statement_output.expanduser().absolute()),
                "payload_output": str(args.payload_output.expanduser().absolute()),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _assemble(args: argparse.Namespace) -> int:
    bundle = build_acceptance_preregistration_bundle(
        statement=_read_json(args.statement),
        producer_signature=_read_bytes(args.producer_signature, maximum=64 * 1024),
        producer_certificate_pem=_read_bytes(
            args.producer_certificate_pem,
            maximum=64 * 1024,
        ),
        rekor_uuid=args.rekor_uuid,
        rekor_entry=_read_json(args.rekor_entry),
        trusted_log_public_key_pem=_read_bytes(
            args.trusted_log_public_key_pem,
            maximum=64 * 1024,
        ),
    )
    payload = json.dumps(bundle, indent=2, sort_keys=True).encode("utf-8")
    _write_once(args.output, payload)
    print(json.dumps(bundle, indent=2, sort_keys=True))
    return 0


def _verify(args: argparse.Namespace) -> int:
    mandate = _portable_mandate(args.mandate_document)
    provision_receipt = _provision_receipt(args.provision_receipt)
    receipt = verify_acceptance_preregistration(
        mandate,
        provision_receipt,
        transparency_bundle=_read_json(args.transparency_bundle),
        trusted_log_public_key_pem=_read_bytes(
            args.trusted_log_public_key_pem,
            maximum=64 * 1024,
        ),
        campaign_started_at_ns=args.campaign_started_at_ns,
        expected_sequence=args.sequence,
        expected_previous_statement_sha256=args.previous_statement_sha256,
        expected_previous_rekor_uuid=args.previous_rekor_uuid,
        minimum_log_index=args.minimum_log_index,
        minimum_integrated_time=args.minimum_integrated_time,
    )
    if args.receipt_output is not None:
        persist_preregistered_acceptance_receipt(receipt, args.receipt_output)
    print(json.dumps(receipt.to_dict(), indent=2, sort_keys=True))
    return 0 if receipt.accepted else 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    statement = commands.add_parser(
        "statement",
        help="emit the Keychain-bound mandate for pre-campaign public timestamping",
    )
    statement.add_argument("--mandate-state", type=Path, required=True)
    statement.add_argument("--campaign-id", required=True)
    statement.add_argument("--provision-receipt", type=Path, required=True)
    statement.add_argument("--sequence", type=int, required=True)
    statement.add_argument("--previous-statement-sha256", default=ZERO_SHA256)
    statement.add_argument("--previous-rekor-uuid")
    statement.add_argument("--issued-at-unix", type=int, default=0)
    statement.add_argument("--statement-output", type=Path, required=True)
    statement.add_argument("--payload-output", type=Path, required=True)
    statement.set_defaults(handler=_statement)

    assemble = commands.add_parser(
        "assemble",
        help="verify Rekor evidence and create a portable preregistration bundle",
    )
    assemble.add_argument("--statement", type=Path, required=True)
    assemble.add_argument("--producer-signature", type=Path, required=True)
    assemble.add_argument("--producer-certificate-pem", type=Path, required=True)
    assemble.add_argument("--rekor-uuid", required=True)
    assemble.add_argument("--rekor-entry", type=Path, required=True)
    assemble.add_argument("--trusted-log-public-key-pem", type=Path, required=True)
    assemble.add_argument("--output", type=Path, required=True)
    assemble.set_defaults(handler=_assemble)

    verify = commands.add_parser(
        "verify",
        help="independently prove the portable mandate predates campaign execution",
    )
    verify.add_argument("--mandate-document", type=Path, required=True)
    verify.add_argument("--provision-receipt", type=Path, required=True)
    verify.add_argument("--transparency-bundle", type=Path, required=True)
    verify.add_argument("--trusted-log-public-key-pem", type=Path, required=True)
    verify.add_argument("--campaign-started-at-ns", type=int, required=True)
    verify.add_argument("--sequence", type=int, required=True)
    verify.add_argument("--previous-statement-sha256", default=ZERO_SHA256)
    verify.add_argument("--previous-rekor-uuid")
    verify.add_argument("--minimum-log-index", type=int)
    verify.add_argument("--minimum-integrated-time", type=int)
    verify.add_argument("--receipt-output", type=Path)
    verify.set_defaults(handler=_verify)

    args = parser.parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
