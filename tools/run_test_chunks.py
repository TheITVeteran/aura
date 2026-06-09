#!/usr/bin/env python3
"""Run the test suite in bounded process chunks.

Running all ~7,400 tests in a single pytest process accumulates enough
module/singleton/cache state that macOS OOM-kills the runner around the
83% mark (observed twice: SIGKILL, exit 137). One process cannot give
back what thousands of heavyweight test modules pile up.

This runner splits the test files into N chunks and runs each chunk in
its own pytest process, so memory is returned to the OS between chunks.
Failure output is preserved per chunk; the exit code is non-zero if any
chunk fails, times out, or dies on a signal — a killed chunk is a loud
failure, never a silent pass.

Usage:
    python tools/run_test_chunks.py [--chunks N] [--marker "not live"]
    python tools/run_test_chunks.py --chunk-timeout 1800
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def discover_test_files(tests_dir: Path) -> list[Path]:
    return sorted(p for p in tests_dir.rglob("test_*.py") if p.is_file())


def split_chunks(files: list[Path], chunks: int) -> list[list[Path]]:
    chunks = max(1, min(chunks, len(files) or 1))
    size = (len(files) + chunks - 1) // chunks
    return [files[i : i + size] for i in range(0, len(files), size)]


def run_chunk(
    index: int,
    total: int,
    files: list[Path],
    *,
    marker: str,
    timeout_s: float,
    python: str,
    extra_args: list[str],
) -> tuple[bool, str]:
    cmd = [
        python,
        "-m",
        "pytest",
        *[str(f) for f in files],
        "-q",
        "-p",
        "no:cacheprovider",
    ]
    if marker:
        cmd.extend(["-m", marker])
    cmd.extend(extra_args)

    started = time.monotonic()
    print(f"━━ chunk {index}/{total}: {len(files)} files ━━", flush=True)
    try:
        proc = subprocess.run(cmd, cwd=ROOT, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return False, f"chunk {index}/{total} TIMEOUT after {timeout_s:.0f}s"
    elapsed = time.monotonic() - started
    if proc.returncode == 0:
        return True, f"chunk {index}/{total} passed in {elapsed:.0f}s"
    if proc.returncode < 0 or proc.returncode in (137, 139, 143):
        return False, (
            f"chunk {index}/{total} KILLED (exit {proc.returncode}) after "
            f"{elapsed:.0f}s — likely OOM; a killed chunk is a failure"
        )
    # pytest exit 5 == no tests collected in this chunk (e.g. everything
    # deselected by the marker) — that is not a failure.
    if proc.returncode == 5:
        return True, f"chunk {index}/{total} had no selected tests"
    return False, f"chunk {index}/{total} FAILED (exit {proc.returncode}) after {elapsed:.0f}s"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", type=int, default=6)
    parser.add_argument("--marker", default="not live")
    parser.add_argument("--chunk-timeout", type=float, default=2400.0)
    parser.add_argument("--tests-dir", type=Path, default=ROOT / "tests")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("extra", nargs="*", help="extra pytest args")
    args = parser.parse_args(argv)

    files = discover_test_files(args.tests_dir)
    if not files:
        print(f"no test files found under {args.tests_dir}", file=sys.stderr)
        return 2

    chunk_lists = split_chunks(files, args.chunks)
    results: list[tuple[bool, str]] = []
    for i, chunk in enumerate(chunk_lists, start=1):
        results.append(
            run_chunk(
                i,
                len(chunk_lists),
                chunk,
                marker=args.marker,
                timeout_s=args.chunk_timeout,
                python=args.python,
                extra_args=list(args.extra),
            )
        )

    print("\n━━ chunk summary ━━")
    failed = 0
    for ok, line in results:
        print(("✅ " if ok else "❌ ") + line)
        if not ok:
            failed += 1
    if failed:
        print(f"\n❌ {failed}/{len(results)} chunks failed")
        return 1
    print(f"\n✅ all {len(results)} chunks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
