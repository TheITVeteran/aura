#!/usr/bin/env python3
"""Replay and verify a certified-recurrence behavioral evidence bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.run_certified_recurrence_behavioral_gate import (  # noqa: E402
    verify_behavioral_bundle,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--skip-model-identity", action="store_true")
    args = parser.parse_args()
    result = verify_behavioral_bundle(
        args.bundle,
        verify_model_identity=not args.skip_model_identity,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
