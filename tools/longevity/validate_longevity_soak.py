#!/usr/bin/env python3
"""Authoritative Longevity Soak Bundle Validator for Aura.

Ensures no linear memory growth or queue growth regressions occur in the soak data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle_path", nargs="?", default="artifacts/current/longevity_soak")
    args = parser.parse_args(argv)

    bundle_dir = Path(args.bundle_path).resolve()
    
    soak_metrics_path = bundle_dir / "SOAK_METRICS.json"
    manifest_path = bundle_dir / "MANIFEST.json"

    receipts_path = bundle_dir / "RECEIPTS.jsonl"
    if not soak_metrics_path.exists() or not manifest_path.exists() or not receipts_path.exists():
        print(f"Error: Missing required files in bundle: {bundle_dir}", file=sys.stderr)
        return 1

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema") != "longevity_soak_manifest":
            print("Error: Invalid longevity manifest schema.", file=sys.stderr)
            return 1
        for rel_path, expected in manifest.get("sha256", {}).items():
            path = bundle_dir / rel_path
            if not path.exists():
                print(f"Error: Manifest references missing file: {rel_path}", file=sys.stderr)
                return 1
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != expected:
                print(f"Error: Manifest hash mismatch for {rel_path}", file=sys.stderr)
                return 1
    except Exception as exc:
        print(f"Error validating manifest: {exc}", file=sys.stderr)
        return 1

    try:
        report = json.loads(soak_metrics_path.read_text(encoding="utf-8"))
        if report.get("memory_leakage_detected") is True:
            print("Error: Memory leakage detected in soak run.", file=sys.stderr)
            return 1
        if report.get("iterations_completed", 0) == 0:
            print("Error: Soak contains 0 completed iterations.", file=sys.stderr)
            return 1
        if report.get("queue_growth_stable") is not True:
            print("Error: Queue growth was not stable in soak run.", file=sys.stderr)
            return 1
        if report.get("boot_event_loop_stable") is not True:
            print("Error: Event loop did not stabilize before measured soak.", file=sys.stderr)
            return 1
        warmup = report.get("boot_event_loop_warmup")
        if not isinstance(warmup, dict) or not warmup.get("samples"):
            print("Error: Soak is missing event-loop warmup evidence.", file=sys.stderr)
            return 1
        if report.get("event_loop_lag_normal") is not True:
            print("Error: Event loop lag exceeded proof profile threshold.", file=sys.stderr)
            return 1
        if not report.get("metrics"):
            print("Error: Soak metrics are empty.", file=sys.stderr)
            return 1
    except Exception as exc:
        print(f"Error reading metrics: {exc}", file=sys.stderr)
        return 1

    if not receipts_path.read_text(encoding="utf-8").strip():
        print("Error: Soak receipts are empty.", file=sys.stderr)
        return 1

    print("Longevity Soak Validation: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
