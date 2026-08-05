#!/usr/bin/env python3
"""Replay a Reality Reach acceptance campaign in an independent process."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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
    persist_mandated_verification_receipt,
    persist_verification_receipt,
    verify_acceptance_against_mandate,
    verify_acceptance_evidence,
)


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
    parser.add_argument("--receipt-output", type=Path)
    args = parser.parse_args(argv)

    store = AcceptanceCertificateStore(args.root)
    certificate = store.load(args.campaign_id)
    evidence = store.load_evidence(certificate)
    if args.mandate_state is not None:
        mandate_store = AcceptanceMandateStore.from_system(args.mandate_state)
        try:
            mandate = mandate_store.get(args.campaign_id)
            mandated_receipt = verify_acceptance_against_mandate(
                certificate,
                evidence,
                mandate,
                trusted_metrology_evidence_sha256=args.metrology_evidence_sha256,
                trusted_governance_evidence_sha256=args.governance_evidence_sha256,
            )
        finally:
            mandate_store.close()
        if args.receipt_output is not None:
            persist_mandated_verification_receipt(
                mandated_receipt,
                args.receipt_output,
            )
        output = mandated_receipt.to_dict()
        accepted = mandated_receipt.accepted
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
