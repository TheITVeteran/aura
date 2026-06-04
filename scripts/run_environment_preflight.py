#!/usr/bin/env python
"""Run one environment-family preflight."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.runtime.subprocess_gateway import get_subprocess_gateway


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True)
    parser.add_argument("--mode", default="fixture")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.env == "terminal_grid:nethack":
        result = get_subprocess_gateway().run(
            [sys.executable, "-m", "pytest", "tests/environments/terminal_grid", "-q"],
            cwd=str(root),
            timeout=300,
            read_only=True,
            source="environment_preflight:terminal_grid:nethack",
        )
        return result.returncode
    print(f"No preflight tests registered for {args.env} in mode {args.mode}; passing empty fixture preflight.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
