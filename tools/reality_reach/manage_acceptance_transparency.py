#!/usr/bin/env python3
"""Prepare and assemble externally logged Reality Reach acceptance evidence."""

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

from core.reality_reach.acceptance import AcceptanceCertificateStore  # noqa: E402
from core.reality_reach.acceptance_mandate import AcceptanceMandateStore  # noqa: E402
from core.reality_reach.acceptance_transparency import (  # noqa: E402
    ZERO_SHA256 as TRANSPARENCY_ZERO_SHA256,
)
from core.reality_reach.acceptance_transparency import (  # noqa: E402
    build_acceptance_transparency_bundle,
    build_acceptance_transparency_statement,
)
from core.reality_reach.acceptance_witness import (  # noqa: E402
    ZERO_SHA256 as WITNESS_ZERO_SHA256,
)
from core.reality_reach.acceptance_witness import (  # noqa: E402  # noqa: E402
    ExternallyWitnessedAcceptanceReceipt,
    verify_acceptance_with_external_witnesses,
)
from core.reality_reach.acoustic_acceptance import (  # noqa: E402
    AcousticA1CampaignStore,
    ExternallyWitnessedAcousticA1Receipt,
    build_acoustic_a1_transparency_bundle,
    build_acoustic_a1_transparency_statement,
    verify_acoustic_a1_with_external_witnesses,
)
from core.runtime.secure_path_custody import (  # noqa: E402
    DirectoryCustody,
    SecurePathCustodyError,
)


def _read_bytes(path: Path, *, maximum: int) -> bytes:
    target = path.expanduser().absolute()
    try:
        with DirectoryCustody.acquire(target.parent, create=False) as custody:
            payload = custody.read_bytes(target.name, max_bytes=maximum)
    except SecurePathCustodyError as exc:
        raise ValueError("acceptance_transparency_input_read_failed") from exc
    if not isinstance(payload, bytes):
        raise ValueError("acceptance_transparency_input_read_failed")
    return payload


def _read_json(path: Path) -> Mapping[str, Any]:
    payload = _read_bytes(path, maximum=4 * 1024 * 1024)

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        document: dict[str, Any] = {}
        for key, value in pairs:
            if key in document:
                raise ValueError("acceptance_transparency_duplicate_json_key")
            document[key] = value
        return document

    try:
        document = json.loads(payload, object_pairs_hook=unique_object)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ValueError("acceptance_transparency_json_invalid") from exc
    if not isinstance(document, Mapping):
        raise ValueError("acceptance_transparency_json_invalid")
    return document


def _write_once(path: Path, payload: bytes, *, mode: int) -> bool:
    target = path.expanduser().absolute()
    try:
        with DirectoryCustody.acquire(target.parent, create=True, private=True) as custody:
            published = bool(custody.write_bytes_once(target.name, payload, mode=mode))
            fd = custody.open_file(target.name, os.O_RDONLY)
            try:
                if stat.S_IMODE(os.fstat(fd).st_mode) != mode:
                    raise ValueError("acceptance_transparency_output_mode_invalid")
            finally:
                os.close(fd)
            existing = custody.read_bytes(target.name, max_bytes=4 * 1024 * 1024)
    except SecurePathCustodyError as exc:
        raise ValueError("acceptance_transparency_output_custody_invalid") from exc
    if existing != payload:
        raise ValueError("acceptance_transparency_output_collision")
    return published


def _external_receipt(
    args: argparse.Namespace,
) -> ExternallyWitnessedAcceptanceReceipt | ExternallyWitnessedAcousticA1Receipt:
    mandate_store = AcceptanceMandateStore.from_system(args.mandate_state)
    try:
        mandate = mandate_store.get(args.campaign_id)
    finally:
        mandate_store.close()
    if args.artifact_kind == "acoustic-a1":
        record = AcousticA1CampaignStore(args.root).load(args.campaign_id)
        return verify_acoustic_a1_with_external_witnesses(
            record,
            mandate,
            metrology_witness_bundle=_read_json(args.metrology_witness_bundle),
            governance_witness_bundle=_read_json(args.governance_witness_bundle),
            metrology_witness_key_sha256=args.metrology_witness_key_sha256,
            governance_witness_key_sha256=args.governance_witness_key_sha256,
            metrology_sequence=args.metrology_witness_sequence,
            governance_sequence=args.governance_witness_sequence,
            metrology_previous_statement_sha256=(
                args.metrology_previous_statement_sha256
            ),
            governance_previous_statement_sha256=(
                args.governance_previous_statement_sha256
            ),
        )
    certificate_store = AcceptanceCertificateStore(args.root)
    certificate = certificate_store.load(args.campaign_id)
    evidence = certificate_store.load_evidence(certificate)
    return verify_acceptance_with_external_witnesses(
        certificate,
        evidence,
        mandate,
        metrology_witness_bundle=_read_json(args.metrology_witness_bundle),
        governance_witness_bundle=_read_json(args.governance_witness_bundle),
        metrology_witness_key_sha256=args.metrology_witness_key_sha256,
        governance_witness_key_sha256=args.governance_witness_key_sha256,
        metrology_sequence=args.metrology_witness_sequence,
        governance_sequence=args.governance_witness_sequence,
        metrology_previous_statement_sha256=(
            args.metrology_previous_statement_sha256
        ),
        governance_previous_statement_sha256=(
            args.governance_previous_statement_sha256
        ),
    )


def _statement(args: argparse.Namespace) -> int:
    receipt = _external_receipt(args)
    issued_at_unix = args.issued_at_unix or int(time.time())
    builder = (
        build_acoustic_a1_transparency_statement
        if args.artifact_kind == "acoustic-a1"
        else build_acceptance_transparency_statement
    )
    statement = builder(
        receipt,
        sequence=args.sequence,
        previous_statement_sha256=args.previous_statement_sha256,
        previous_rekor_uuid=args.previous_rekor_uuid,
        issued_at_unix=issued_at_unix,
    )
    payload = json.dumps(
        statement,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    document = json.dumps(statement, indent=2, sort_keys=True).encode("utf-8")
    _write_once(args.statement_output, document, mode=0o600)
    _write_once(args.payload_output, payload, mode=0o600)
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
    builder = (
        build_acoustic_a1_transparency_bundle
        if args.artifact_kind == "acoustic-a1"
        else build_acceptance_transparency_bundle
    )
    bundle = builder(
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
    _write_once(args.output, payload, mode=0o600)
    print(json.dumps(bundle, indent=2, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    statement = commands.add_parser(
        "statement",
        help="emit a mandate-bound verdict for external signing and Rekor submission",
    )
    statement.add_argument("--root", type=Path, required=True)
    statement.add_argument(
        "--artifact-kind",
        choices=("scalar", "acoustic-a1"),
        default="scalar",
    )
    statement.add_argument("--mandate-state", type=Path, required=True)
    statement.add_argument("--campaign-id", required=True)
    statement.add_argument("--metrology-witness-bundle", type=Path, required=True)
    statement.add_argument("--governance-witness-bundle", type=Path, required=True)
    statement.add_argument("--metrology-witness-key-sha256", required=True)
    statement.add_argument("--governance-witness-key-sha256", required=True)
    statement.add_argument("--metrology-witness-sequence", type=int, default=1)
    statement.add_argument("--governance-witness-sequence", type=int, default=1)
    statement.add_argument(
        "--metrology-previous-statement-sha256",
        default=WITNESS_ZERO_SHA256,
    )
    statement.add_argument(
        "--governance-previous-statement-sha256",
        default=WITNESS_ZERO_SHA256,
    )
    statement.add_argument("--sequence", type=int, required=True)
    statement.add_argument(
        "--previous-statement-sha256",
        default=TRANSPARENCY_ZERO_SHA256,
    )
    statement.add_argument("--previous-rekor-uuid")
    statement.add_argument("--issued-at-unix", type=int, default=0)
    statement.add_argument("--statement-output", type=Path, required=True)
    statement.add_argument("--payload-output", type=Path, required=True)
    statement.set_defaults(handler=_statement)

    assemble = commands.add_parser(
        "assemble",
        help="verify Rekor evidence and assemble an offline transparency bundle",
    )
    assemble.add_argument(
        "--artifact-kind",
        choices=("scalar", "acoustic-a1"),
        default="scalar",
    )
    assemble.add_argument("--statement", type=Path, required=True)
    assemble.add_argument("--producer-signature", type=Path, required=True)
    assemble.add_argument("--producer-certificate-pem", type=Path, required=True)
    assemble.add_argument("--rekor-uuid", required=True)
    assemble.add_argument("--rekor-entry", type=Path, required=True)
    assemble.add_argument("--trusted-log-public-key-pem", type=Path, required=True)
    assemble.add_argument("--output", type=Path, required=True)
    assemble.set_defaults(handler=_assemble)

    args = parser.parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
