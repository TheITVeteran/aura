#!/usr/bin/env python3
"""Export Aura's verifier-derived preference pairs for local DPO/ORPO training.

The data source is ``core.learning.verifiable_preference_harness``: pairs are
created only when the same problem has a checked verified-correct candidate and
a checked verified-wrong candidate. This script makes that store inspectable and
trainer-ready without running any training by accident.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.learning.verifiable_preference_harness import VerifiablePreferenceHarness  # noqa: E402


def _default_store() -> Path:
    try:
        from core.config import config

        return Path(config.paths.data_dir) / "verifiable_preferences.jsonl"
    except (ImportError, AttributeError, TypeError, OSError):
        return Path.home() / ".aura" / "data" / "verifiable_preferences.jsonl"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, default=_default_store())
    parser.add_argument("--out", type=Path, default=Path("training/data/verifiable_preferences_dpo.jsonl"))
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--min-rows", type=int, default=1)
    parser.add_argument("--stats", action="store_true", help="print store stats and exit")
    args = parser.parse_args(argv)

    harness = VerifiablePreferenceHarness(store_path=args.store)
    rows = harness.export_dpo_rows(limit=max(1, int(args.limit)))
    if args.stats:
        print(json.dumps({**harness.stats(), "exportable_rows": len(rows)}, indent=2))
        return 0
    if len(rows) < int(args.min_rows):
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": "insufficient_verifiable_preference_rows",
                    "rows": len(rows),
                    "min_rows": int(args.min_rows),
                    "store": str(args.store),
                },
                indent=2,
            ),
            file=sys.stderr,
        )
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "ok": True,
                "rows": len(rows),
                "store": str(args.store),
                "out": str(args.out),
                "format": "jsonl(prompt, chosen, rejected)",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
