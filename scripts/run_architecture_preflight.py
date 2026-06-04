#!/usr/bin/env python
# ruff: noqa: I001
"""Run the general environment architecture preflight gate."""
from __future__ import annotations

import compileall
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.runtime.subprocess_gateway import get_subprocess_gateway


TEST_TARGETS = [
    "tests/architecture/preflight",
    "tests/architecture/contracts",
    "tests/architecture/perception",
    "tests/architecture/modal",
    "tests/architecture/command",
    "tests/architecture/action_gateway",
    "tests/architecture/belief",
    "tests/architecture/planning",
    "tests/architecture/simulation",
    "tests/architecture/outcome",
    "tests/architecture/learning",
    "tests/architecture/governance",
    "tests/architecture/trace",
    "tests/architecture/resilience",
    "tests/architecture/generalization",
    "tests/architecture/benchmarks",
]


def main() -> int:
    ok = compileall.compile_dir(ROOT / "core", quiet=1) and compileall.compile_dir(ROOT / "scripts", quiet=1) and compileall.compile_dir(ROOT / "tests", quiet=1)
    if not ok:
        return 1
    existing = [target for target in TEST_TARGETS if (ROOT / target).exists()]
    if not existing:
        return 0
    result = get_subprocess_gateway().run(
        [sys.executable, "-m", "pytest", *existing, "-q"],
        cwd=str(ROOT),
        timeout=600,
        read_only=True,
        source="architecture_preflight",
    )
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
