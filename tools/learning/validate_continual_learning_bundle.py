#!/usr/bin/env python3
"""
tools/learning/validate_continual_learning_bundle.py
Validator for the Continual Learning and Adaptation proof bundle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle_path", nargs="?", default="artifacts/current/continual_learning")
    args = parser.parse_args(argv)

    bundle_dir = Path(args.bundle_path).resolve()
    
    scorecard_path = bundle_dir / "SCORECARD.json"
    receipts_path = bundle_dir / "RECEIPTS.jsonl"
    manifest_path = bundle_dir / "MANIFEST.json"

    # 1. Structural check
    if not scorecard_path.exists() or not receipts_path.exists() or not manifest_path.exists():
        print(f"Error: Missing required files in bundle: {bundle_dir}", file=sys.stderr)
        return 1

    # 2. Manifest integrity check
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema") != "continual_learning_manifest":
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
                print(f"Error: Integrity hash mismatch for {fname}: expected {expected_hash}, got {actual_hash}", file=sys.stderr)
                return 1
        print("  [OK] Manifest hash integrity verified.")
    except Exception as exc:
        print(f"Error validating manifest: {exc}", file=sys.stderr)
        return 1

    # 3. Scorecard verification
    try:
        scorecard = json.loads(scorecard_path.read_text(encoding="utf-8"))
        total = scorecard.get("total_attempted", 0)
        passed = scorecard.get("passed_count", 0)
        pass_rate = scorecard.get("pass_rate", 0.0)
        if total < 3:
            print(f"Error: Scorecard contains fewer than 3 tasks: got {total}", file=sys.stderr)
            return 1
        if pass_rate < 0.66:
            print(f"Error: Pass rate below 66% threshold: got {pass_rate:.1%}", file=sys.stderr)
            return 1
        print(f"  [OK] Scorecard: {passed}/{total} tasks passed (rate: {pass_rate:.1%}).")
    except Exception as exc:
        print(f"Error reading scorecard: {exc}", file=sys.stderr)
        return 1

    # 4. Receipt verification
    try:
        receipts = []
        with open(receipts_path, "r", encoding="utf-8") as f:
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
    except Exception as exc:
        print(f"Error validating receipts: {exc}", file=sys.stderr)
        return 1

    print("Continual Learning Bundle: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
