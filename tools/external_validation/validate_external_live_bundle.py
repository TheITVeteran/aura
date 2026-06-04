#!/usr/bin/env python3
"""Authoritative External Live Validation Bundle Validator for Aura.

Performs deep structure, manifest hash, and secure receipt verification
on the generated external validation logs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

_VALIDATION_DATA_ERRORS = (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError, KeyError)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle_path", nargs="?", default="artifacts/current/external_live_validation")
    args = parser.parse_args(argv)

    bundle_dir = Path(args.bundle_path).resolve()
    
    scorecard_path = bundle_dir / "SCORECARD.json"
    receipts_path = bundle_dir / "RECEIPTS.jsonl"
    manifest_path = bundle_dir / "MANIFEST.json"

    # 1. Structural files existence check
    if not scorecard_path.exists() or not receipts_path.exists() or not manifest_path.exists():
        print(f"Error: Missing required files in bundle: {bundle_dir}", file=sys.stderr)
        return 1

    # 2. Manifest and integrity check
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema") != "external_live_validation_manifest":
            print("Error: Invalid manifest schema.", file=sys.stderr)
            return 1
        
        expected_hashes = manifest.get("sha256", {})
        for fname, expected_hash in expected_hashes.items():
            fpath = bundle_dir / fname
            if not fpath.exists():
                print(f"Error: File '{fname}' declared in manifest does not exist.", file=sys.stderr)
                return 1
            actual_hash = hashlib.sha256(fpath.read_bytes()).hexdigest()
            if actual_hash != expected_hash:
                print(f"Error: Manifest integrity hash mismatch for {fname}: expected {expected_hash}, got {actual_hash}", file=sys.stderr)
                return 1
        print("  [OK] Manifest hash integrity verified.")
    except _VALIDATION_DATA_ERRORS as exc:
        print(f"Error validating manifest: {exc}", file=sys.stderr)
        return 1

    # 3. Scorecard validation
    try:
        scorecard = json.loads(scorecard_path.read_text(encoding="utf-8"))
        total = scorecard.get("total_attempted", 0)
        passed = scorecard.get("passed_count", 0)
        if total == 0:
            print("Error: Scorecard contains 0 attempted tasks.", file=sys.stderr)
            return 1
        print(f"  [OK] Scorecard: {passed}/{total} tasks passed (rate: {scorecard.get('pass_rate'):.1%}).")
    except _VALIDATION_DATA_ERRORS as exc:
        print(f"Error reading scorecard: {exc}", file=sys.stderr)
        return 1

    # 4. Receipt validation
    try:
        receipts = []
        with open(receipts_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    receipts.append(json.loads(line))

        if len(receipts) == 0:
            print("Error: RECEIPTS.jsonl is empty.", file=sys.stderr)
            return 1

        for r in receipts:
            if not r.get("task_id") or not r.get("receipt_id") or not r.get("domain") or not r.get("outcome"):
                print(f"Error: Malformed receipt entry: {r}", file=sys.stderr)
                return 1

        # Match tasks to receipts
        task_ids = {t["id"] for t in scorecard.get("tasks", [])}
        receipt_task_ids = {r["task_id"] for r in receipts}
        missing_task_receipts = task_ids - receipt_task_ids
        if missing_task_receipts:
            print(f"Error: Missing secure receipts for tasks: {missing_task_receipts}", file=sys.stderr)
            return 1

        print("  [OK] Secure receipt matching verified.")
    except _VALIDATION_DATA_ERRORS as exc:
        print(f"Error validating receipts: {exc}", file=sys.stderr)
        return 1

    print("External Live Validation Bundle: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
