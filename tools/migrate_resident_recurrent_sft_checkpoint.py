#!/usr/bin/env python3
"""Migrate an exact resident recurrent-SFT checkpoint across a source repair."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.learning.resident_recurrent_sft_checkpoint_migration import (  # noqa: E402
    migrate_checkpoint,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repo-root", type=Path, required=True)
    parser.add_argument("--source-authority", type=Path, required=True)
    parser.add_argument("--destination-repo-root", type=Path, required=True)
    parser.add_argument("--destination-authority", type=Path, required=True)
    parser.add_argument("--allow-budget-extension", action="store_true")
    args = parser.parse_args()
    receipt = migrate_checkpoint(
        source_repo_root=args.source_repo_root,
        source_authority_path=args.source_authority,
        destination_repo_root=args.destination_repo_root,
        destination_authority_path=args.destination_authority,
        allow_budget_extension=args.allow_budget_extension,
    )
    print(json.dumps(receipt, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
