#!/usr/bin/env python3
"""Coverage ratchet — the floor may rise, never fall.

An external review found no enforced line-coverage, branch-coverage, or
mutation-testing threshold, and noted why it matters here specifically: a
4,465-line function with 632 branches can carry many tests while leaving most
combinations unexplored. Line coverage alone would call that function well
tested.

A fixed target (`fail_under = 80`) is the wrong instrument for a repo this
size — it is either unreachable today or so low it never fails. A ratchet
matches the convention already used for layering and the enterprise gate:
record where we are, and refuse to go backwards.

Usage:

    # measure (long — the full suite runs in 6 chunks)
    make coverage

    # compare against the recorded floor
    python tools/coverage_ratchet.py check

    # accept a genuine improvement
    python tools/coverage_ratchet.py bless
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
BASELINE = REPO / "config" / "coverage_baseline.json"
#: How far coverage may drift below the recorded floor before failing, in
#: percentage points. Non-zero because measured coverage moves slightly with
#: test ordering and chunking; zero would make the gate flap.
TOLERANCE = 0.25


def _measure() -> dict[str, float]:
    """Read the current numbers from coverage's own JSON report."""
    try:
        raw = subprocess.run(
            [sys.executable, "-m", "coverage", "json", "-o", "-", "--quiet"],
            capture_output=True,
            text=True,
            cwd=REPO,
            timeout=600,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise SystemExit(f"could not run coverage: {exc}") from exc
    if raw.returncode != 0:
        raise SystemExit(
            "coverage produced no report. Run `make coverage` first "
            f"(exit {raw.returncode}): {raw.stderr.strip()[:400]}"
        )
    payload = json.loads(raw.stdout)
    totals = payload.get("totals", {})
    return {
        "line_percent": round(float(totals.get("percent_covered", 0.0)), 2),
        "branch_percent": round(
            100.0
            * float(totals.get("covered_branches", 0))
            / max(1.0, float(totals.get("num_branches", 0) or 1)),
            2,
        ),
        "statements": int(totals.get("num_statements", 0)),
        "branches": int(totals.get("num_branches", 0)),
    }


def _load_baseline() -> dict[str, float] | None:
    if not BASELINE.exists():
        return None
    try:
        return json.loads(BASELINE.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SystemExit(f"baseline unreadable: {exc}") from exc


def check() -> int:
    baseline = _load_baseline()
    current = _measure()
    if baseline is None:
        print("No baseline recorded yet. Establish one with:")
        print("    make coverage && python tools/coverage_ratchet.py bless")
        print(f"Current: {json.dumps(current, indent=2)}")
        return 0

    failures = []
    for key in ("line_percent", "branch_percent"):
        floor = float(baseline.get(key, 0.0))
        now = float(current.get(key, 0.0))
        status = "ok" if now >= floor - TOLERANCE else "REGRESSED"
        print(f"{key:>16}: {now:6.2f}%  floor {floor:6.2f}%  {status}")
        if status == "REGRESSED":
            failures.append(f"{key} fell {floor - now:.2f} points below the recorded floor")

    if failures:
        print("\nCoverage ratchet FAILED:")
        for line in failures:
            print(f"  - {line}")
        print("\nAdd tests, or justify and re-bless deliberately.")
        return 1
    print("\nCoverage ratchet passed.")
    return 0


def bless() -> int:
    current = _measure()
    baseline = _load_baseline() or {}
    for key in ("line_percent", "branch_percent"):
        if float(current.get(key, 0.0)) < float(baseline.get(key, 0.0)):
            print(
                f"Refusing to bless: {key} {current[key]:.2f}% is BELOW the "
                f"recorded floor {baseline[key]:.2f}%. The ratchet only rises."
            )
            return 1
    BASELINE.parent.mkdir(parents=True, exist_ok=True)
    BASELINE.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Baseline recorded at {BASELINE.relative_to(REPO)}:")
    print(json.dumps(current, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "bless"))
    args = parser.parse_args()
    return check() if args.action == "check" else bless()


if __name__ == "__main__":
    raise SystemExit(main())
