#!/usr/bin/env python3
"""Materialize the clean-canary mathematics memory tissue for runtime use."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from core.learning.recurrent_work_memory_training import (  # noqa: E402
    train_and_write_mathematics_memory_artifact,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canary", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        canary = json.loads(
            args.canary.expanduser().resolve(strict=True).read_text(encoding="ascii")
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("mathematics memory canary is unreadable") from exc
    if not isinstance(canary, dict):
        raise RuntimeError("mathematics memory canary is not an object")
    manifest = train_and_write_mathematics_memory_artifact(
        args.out_dir,
        canary_receipt=canary,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
