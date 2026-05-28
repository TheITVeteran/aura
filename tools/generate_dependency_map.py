#!/usr/bin/env python3
"""Generate the reviewer-facing architecture map from the canonical scanner.

`tools/arch_map.py` owns the operational dependency model used by final-proof.
This command is a stable compatibility entrypoint for people who expect a
human-readable document at docs/ARCHITECTURE_MAP.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.runtime.atomic_writer import atomic_write_text
from tools.arch_map import build_architecture_report, render_markdown_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-md",
        type=Path,
        default=ROOT / "docs" / "ARCHITECTURE_MAP.md",
        help="Markdown report path.",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Optional machine-readable report path.",
    )
    args = parser.parse_args(argv)

    report = build_architecture_report()
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(args.out_md, render_markdown_report(report), encoding="utf-8")

    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(
            args.out_json,
            json.dumps(report, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )

    print(f"Architecture map written to {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
