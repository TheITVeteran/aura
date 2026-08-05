#!/usr/bin/env python3
"""Precommit one Reality Reach acceptance question under Keychain custody."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.reality_reach.acceptance import (  # noqa: E402
    REQUIRED_SCALAR_ACCEPTANCE_CASES,
    AcceptanceEvidenceClass,
)
from core.reality_reach.acceptance_mandate import (  # noqa: E402
    AcceptanceMandateStore,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-path", type=Path)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--connector-id", required=True)
    parser.add_argument("--adapter-id", required=True)
    parser.add_argument("--source-commit-sha256", required=True)
    parser.add_argument("--physical-identity-sha256", required=True)
    parser.add_argument(
        "--evidence-class",
        required=True,
        choices=[item.value for item in AcceptanceEvidenceClass],
    )
    parser.add_argument("--target", type=float, required=True)
    parser.add_argument("--target-tolerance", type=float, required=True)
    parser.add_argument("--scenario-id", default="")
    parser.add_argument(
        "--required-case",
        action="append",
        default=None,
        help="Repeat to replace the canonical scalar acceptance case set.",
    )
    args = parser.parse_args(argv)

    store = AcceptanceMandateStore.provision_system(args.state_path)
    try:
        receipt = store.provision(
            campaign_id=args.campaign_id,
            connector_id=args.connector_id,
            adapter_id=args.adapter_id,
            expected_source_commit_sha256=args.source_commit_sha256,
            expected_physical_identity_sha256=args.physical_identity_sha256,
            expected_evidence_class=AcceptanceEvidenceClass(args.evidence_class),
            target=args.target,
            target_tolerance=args.target_tolerance,
            scenario_id=args.scenario_id,
            required_cases=tuple(args.required_case or REQUIRED_SCALAR_ACCEPTANCE_CASES),
        )
        mandate = store.get(args.campaign_id)
        print(
            json.dumps(
                {
                    "mandate": mandate.to_dict(),
                    "provision_receipt": receipt.to_dict(),
                    "custody_status": store.status(),
                },
                indent=2,
                sort_keys=True,
            )
        )
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
