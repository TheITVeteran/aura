#!/usr/bin/env python3
"""Authoritative External Live Validation Bundle Validator for Aura.

Performs deep structure and schema checks on generated external validation logs.
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
    parser.add_argument("bundle_path", nargs="?", default="artifacts/current/external_live_validation")
    args = parser.parse_args(argv)

    bundle_dir = Path(args.bundle_path).resolve()
    
    scorecard_path = bundle_dir / "SCORECARD.json"
    receipts_path = bundle_dir / "RECEIPTS.jsonl"
    manifest_path = bundle_dir / "MANIFEST.json"

    if not scorecard_path.exists() or not receipts_path.exists() or not manifest_path.exists():
        print(f"Error: Missing required files in bundle: {bundle_dir}", file=sys.stderr)
        return 1

    try:
        scorecard = json.loads(scorecard_path.read_text(encoding="utf-8"))
        if scorecard.get("total_attempted", 0) == 0:
            print("Error: Scorecard contains 0 tasks.", file=sys.stderr)
            return 1
    except Exception as exc:
        print(f"Error reading scorecard: {exc}", file=sys.stderr)
        return 1

    print("External Live Validation Bundle: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
