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
from core.reality_reach.acceptance_verifier import (  # noqa: E402
    persist_verification_receipt,
    verify_acceptance_evidence,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--source-commit-sha256", required=True)
    parser.add_argument("--physical-identity-sha256", required=True)
    parser.add_argument(
        "--expected-evidence-class",
        required=True,
        choices=[item.value for item in AcceptanceEvidenceClass],
        help="Externally declared burden; simulation, HIL, and live are not interchangeable.",
    )
    parser.add_argument("--metrology-evidence-sha256", default="")
    parser.add_argument("--receipt-output", type=Path)
    args = parser.parse_args()

    store = AcceptanceCertificateStore(args.root)
    certificate = store.load(args.campaign_id)
    evidence = store.load_evidence(certificate)
    receipt = verify_acceptance_evidence(
        certificate,
        evidence,
        expected_source_commit_sha256=args.source_commit_sha256,
        expected_physical_identity_sha256=args.physical_identity_sha256,
        expected_evidence_class=AcceptanceEvidenceClass(args.expected_evidence_class),
        trusted_metrology_evidence_sha256=args.metrology_evidence_sha256,
    )
    if args.receipt_output is not None:
        persist_verification_receipt(receipt, args.receipt_output)
    print(json.dumps(receipt.to_dict(), indent=2, sort_keys=True))
    return 0 if receipt.accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
