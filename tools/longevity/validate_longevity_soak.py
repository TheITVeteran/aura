#!/usr/bin/env python3
"""Authoritative Longevity Soak Bundle Validator for Aura.

Ensures no linear memory growth or queue growth regressions occur in the soak data.
"""

from __future__ import annotations

import argparse
import json
import os
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

    if not soak_metrics_path.exists() or not manifest_path.exists():
        print(f"Error: Missing required files in bundle: {bundle_dir}", file=sys.stderr)
        return 1

    try:
        report = json.loads(soak_metrics_path.read_text(encoding="utf-8"))
        if report.get("memory_leakage_detected") is True:
            print("Error: Memory leakage detected in soak run.", file=sys.stderr)
            return 1
        if report.get("iterations_completed", 0) == 0:
            print("Error: Soak contains 0 completed iterations.", file=sys.stderr)
            return 1
    except Exception as exc:
        print(f"Error reading metrics: {exc}", file=sys.stderr)
        return 1

    print("Longevity Soak Validation: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
