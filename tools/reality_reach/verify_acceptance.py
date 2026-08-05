#!/usr/bin/env python3
"""Replay a Reality Reach acceptance campaign in an independent process."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.reality_reach.acceptance import (  # noqa: E402
    AcceptanceCertificateStore,
    AcceptanceEvidenceClass,
)
from core.reality_reach.acceptance_mandate import (  # noqa: E402
    AcceptanceMandateStore,
)
from core.reality_reach.acceptance_verifier import (  # noqa: E402
    persist_verification_receipt,
    verify_acceptance_evidence,
)
from core.reality_reach.acceptance_witness import (  # noqa: E402
    ZERO_SHA256,
    persist_externally_witnessed_acceptance_receipt,
    verify_acceptance_with_external_witnesses,
)
from core.runtime.secure_path_custody import (  # noqa: E402
    DirectoryCustody,
    SecurePathCustodyError,
)


def _read_json(path: Path) -> Mapping[str, Any]:
    target = path.expanduser().absolute()
    try:
        with DirectoryCustody.acquire(target.parent, create=False) as custody:
            payload = custody.read_bytes(target.name, max_bytes=1024 * 1024)
    except SecurePathCustodyError as exc:
        raise ValueError("acceptance_witness_bundle_read_failed") from exc

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        document: dict[str, Any] = {}
        for key, value in pairs:
            if key in document:
                raise ValueError("acceptance_witness_bundle_duplicate_json_key")
            document[key] = value
        return document

    try:
        document = json.loads(payload, object_pairs_hook=unique_object)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ValueError("acceptance_witness_bundle_json_invalid") from exc
    if not isinstance(document, Mapping):
        raise ValueError("acceptance_witness_bundle_json_invalid")
    return document


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--mandate-state", type=Path)
    parser.add_argument("--source-commit-sha256")
    parser.add_argument("--physical-identity-sha256")
    parser.add_argument(
        "--expected-evidence-class",
        choices=[item.value for item in AcceptanceEvidenceClass],
        help="Externally declared burden; simulation, HIL, and live are not interchangeable.",
    )
    parser.add_argument("--metrology-evidence-sha256", default="")
    parser.add_argument("--governance-evidence-sha256", default="")
    parser.add_argument("--metrology-witness-bundle", type=Path)
    parser.add_argument("--governance-witness-bundle", type=Path)
    parser.add_argument("--metrology-witness-key-sha256", default="")
    parser.add_argument("--governance-witness-key-sha256", default="")
    parser.add_argument("--metrology-witness-sequence", type=int, default=1)
    parser.add_argument("--governance-witness-sequence", type=int, default=1)
    parser.add_argument(
        "--metrology-previous-statement-sha256",
        default=ZERO_SHA256,
    )
    parser.add_argument(
        "--governance-previous-statement-sha256",
        default=ZERO_SHA256,
    )
    parser.add_argument("--receipt-output", type=Path)
    args = parser.parse_args(argv)

    store = AcceptanceCertificateStore(args.root)
    certificate = store.load(args.campaign_id)
    evidence = store.load_evidence(certificate)
    if args.mandate_state is not None:
        mandate_store = AcceptanceMandateStore.from_system(args.mandate_state)
        try:
            mandate = mandate_store.get(args.campaign_id)
            external_receipt = verify_acceptance_with_external_witnesses(
                certificate,
                evidence,
                mandate,
                metrology_witness_bundle=(
                    _read_json(args.metrology_witness_bundle)
                    if args.metrology_witness_bundle is not None
                    else None
                ),
                governance_witness_bundle=(
                    _read_json(args.governance_witness_bundle)
                    if args.governance_witness_bundle is not None
                    else None
                ),
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
        finally:
            mandate_store.close()
        if args.receipt_output is not None:
            persist_externally_witnessed_acceptance_receipt(
                external_receipt,
                args.receipt_output,
            )
        output = external_receipt.to_dict()
        accepted = external_receipt.accepted
    else:
        missing = [
            flag
            for flag, value in (
                ("--source-commit-sha256", args.source_commit_sha256),
                ("--physical-identity-sha256", args.physical_identity_sha256),
                ("--expected-evidence-class", args.expected_evidence_class),
            )
            if not value
        ]
        if missing:
            parser.error(
                "without --mandate-state these arguments are required: "
                + ", ".join(missing)
            )
        if args.expected_evidence_class != AcceptanceEvidenceClass.SIMULATION.value:
            parser.error(
                "physical acceptance requires --mandate-state and distinct "
                "external metrology/governance witness bundles"
            )
        legacy_receipt = verify_acceptance_evidence(
            certificate,
            evidence,
            expected_source_commit_sha256=args.source_commit_sha256,
            expected_physical_identity_sha256=args.physical_identity_sha256,
            expected_evidence_class=AcceptanceEvidenceClass(
                args.expected_evidence_class
            ),
            trusted_metrology_evidence_sha256=args.metrology_evidence_sha256,
            trusted_governance_evidence_sha256=args.governance_evidence_sha256,
        )
        if args.receipt_output is not None:
            persist_verification_receipt(legacy_receipt, args.receipt_output)
        output = legacy_receipt.to_dict()
        accepted = legacy_receipt.accepted
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0 if accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
