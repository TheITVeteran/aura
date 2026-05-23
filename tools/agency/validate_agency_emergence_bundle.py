#!/usr/bin/env python3
"""
tools/agency/validate_agency_emergence_bundle.py
Authoritative Validator for the Agency Emergence & Boxed Entity Proof Bundle.
"""

import hashlib
import json
import sys
from pathlib import Path

_VALIDATION_ERRORS = (
    KeyError,
    OSError,
    json.JSONDecodeError,
    TypeError,
    ValueError,
)


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    if not argv:
        print("Usage: python tools/agency/validate_agency_emergence_bundle.py <bundle_dir>")
        return 1

    bundle_dir = Path(argv[0]).resolve()
    print("============================================================")
    print(" VALIDATING AGENCY EMERGENCE PROOF BUNDLE AT:")
    print(f" {bundle_dir}")
    print("============================================================")

    if not bundle_dir.exists():
        print(f"[-] ERROR: Bundle directory does not exist: {bundle_dir}")
        return 1

    # Required files
    required_files = [
        "AGENCY_EMERGENCE_PROOF.json",
        "AGENCY_EMERGENCE_PROOF.md",
        "SCORECARD.json",
        "BASELINES.json",
        "ABLATIONS.json",
        "TASK_TRACE.jsonl",
        "RECEIPTS.jsonl",
        "GOVERNANCE_REPORT.json",
        "MANIFEST.json"
    ]

    failures = []

    # 1. Check file existence
    for fname in required_files:
        fpath = bundle_dir / fname
        if not fpath.exists():
            failures.append(f"Missing required artifact: {fname}")

    if failures:
        print("[-] File presence checks failed:")
        for f in failures:
            print(f"  - {f}")
        return 1

    print("[+] All required artifact files are present.")

    # 2. Check Scorecard
    try:
        scorecard = json.loads((bundle_dir / "SCORECARD.json").read_text(encoding="utf-8"))
        print(f"[+] Scorecard loaded: {scorecard.get('passed_tasks')}/{scorecard.get('total_tasks')} passed ({scorecard.get('overall_pass_rate'):.1%})")
        if scorecard.get("total_tasks", 0) < 1:
            failures.append("Scorecard reports 0 tasks attempted")
        if scorecard.get("overall_pass_rate", 0.0) <= 0.0:
            failures.append("Overall pass rate is 0.0%")
    except _VALIDATION_ERRORS as e:
        failures.append(f"Failed to load or parse SCORECARD.json: {e}")

    # 3. Check Ablations and Outperformance
    try:
        ablations = json.loads((bundle_dir / "ABLATIONS.json").read_text(encoding="utf-8"))
        required_ablations = [
            "no_persistent_memory",
            "no_volition",
            "no_will_authority",
            "no_system2",
            "no_self_repair",
            "no_affect_steering"
        ]

        full_aura_pr = ablations.get("full_aura", {}).get("pass_rate", 0.0)
        print(f"[+] Full Aura pass rate: {full_aura_pr:.1%}")

        outperformed_count = 0
        for ab in required_ablations:
            if ab not in ablations:
                failures.append(f"Required ablation '{ab}' missing from ABLATIONS.json")
            else:
                ab_data = ablations[ab]
                if ab_data.get("status") != "RUN":
                    failures.append(f"Ablation '{ab}' did not run successfully (status: {ab_data.get('status')})")
                else:
                    ab_pr = ab_data.get("pass_rate", 0.0)
                    print(f"  - {ab} pass rate: {ab_pr:.1%}")
                    if full_aura_pr > ab_pr:
                        outperformed_count += 1

        print(f"[+] Full Aura outperformed {outperformed_count}/6 required ablations.")
        if outperformed_count < 4:
            failures.append(f"Full Aura did not materially outperform at least 4 required ablations: got {outperformed_count}/6")
    except _VALIDATION_ERRORS as e:
        failures.append(f"Failed to load or parse ABLATIONS.json: {e}")

    # 4. Check Governance Receipts
    try:
        gov_report = json.loads((bundle_dir / "GOVERNANCE_REPORT.json").read_text(encoding="utf-8"))
        receipts_content = (bundle_dir / "RECEIPTS.jsonl").read_text(encoding="utf-8").strip()
        actual_receipts = len([line for line in receipts_content.splitlines() if line.strip()])

        print(f"[+] Governance report receipt_count: {gov_report.get('receipt_count')}")
        print(f"[+] RECEIPTS.jsonl actual count: {actual_receipts}")

        if gov_report.get("receipt_count", 0) <= 0:
            failures.append("Governance report indicates 0 receipts generated")
        if actual_receipts <= 0:
            failures.append("RECEIPTS.jsonl is empty")
        if gov_report.get("receipt_count") != actual_receipts:
            failures.append(f"Mismatch between GOVERNANCE_REPORT count ({gov_report.get('receipt_count')}) and RECEIPTS.jsonl actual count ({actual_receipts})")
    except _VALIDATION_ERRORS as e:
        failures.append(f"Failed to validate governance receipts: {e}")

    # 5. Validate Manifest
    try:
        manifest = json.loads((bundle_dir / "MANIFEST.json").read_text(encoding="utf-8"))
        for fname, details in manifest.get("files", {}).items():
            fpath = bundle_dir / fname
            if fpath.exists():
                actual_hash = hashlib.sha256(fpath.read_bytes()).hexdigest()
                expected_hash = details.get("sha256", "")
                if actual_hash != expected_hash:
                    failures.append(f"Hash mismatch for {fname}: expected {expected_hash}, got {actual_hash}")
            else:
                failures.append(f"File listed in manifest is missing on disk: {fname}")
        print("[+] Manifest hash verification completed successfully.")
    except _VALIDATION_ERRORS as e:
        failures.append(f"Failed to load or parse MANIFEST.json: {e}")

    # Verdict
    print("============================================================")
    if failures:
        print(f"[-] VALIDATION FAILED: {len(failures)} failures detected.")
        for f in failures:
            print(f"  - {f}")
        print("============================================================")
        return 1
    print("[+] VALIDATION SUCCESSFUL: Agency Emergence Proof Bundle is fully valid!")
    print("============================================================")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
