#!/usr/bin/env python3
"""Module and class size, as a ratchet that only tightens.

`interface/routes/chat.py` is 29,481 lines with 457 module-level functions.
`core/brain/llm/mlx_client.py` is 15,423 with a 165-method class.
`core/brain/inference_gate.py` is 13,416 with a 193-method class that handles
worker processes, cloud fallback, health probing, warm-up, desktop resource
guards, background deferral, PII scrubbing, PBKDF2 offloading, RAM diagnostics
and UI prompt strings. Thirty-two files are over three thousand lines.

None of that is fixable in one commit, and pretending otherwise is how it stays
unfixed. What IS fixable in one commit is the direction of travel: nothing stops
chat.py reaching forty thousand lines, and nothing stops the next God object
being created from scratch. This is the same shape as the layering, effect-
ownership, async-write and bounded-await ratchets already in this repo — a
checked-in baseline that may shrink and may not grow.

Three rules:

1. A file recorded in the baseline may not exceed its recorded size. Growth in a
   file already known to be too large is the failure this exists to stop.
2. A file NOT in the baseline may not exceed the thresholds at all. A new God
   object is never grandfathered.
3. A file that has shrunk must have its baseline refreshed. A stale entry is
   headroom nobody earned, and it is how a ratchet quietly stops ratcheting.

The thresholds come from this repository's own distribution rather than from
taste: 2,000 lines is just under the 98th percentile of file length (2,115) and
30 methods is just above the 98th percentile of class size (26). A new file
above either is an outlier by the standard of the code around it.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASELINE = ROOT / "config" / "module_size_baseline.json"
SCANNED = ("core", "interface")

#: p98 of file length across the scanned tree (2,115 lines) rounded down. A new
#: file longer than 98% of everything already here is an outlier by the
#: codebase's own measure, not by an opinion about file length.
MAX_NEW_MODULE_LINES = 2_000

#: Just above p98 of class size (26 methods). Same reasoning.
MAX_NEW_CLASS_METHODS = 30

BASELINE_SCHEMA = "aura.module_size_baseline.v1"


@dataclass(frozen=True)
class Measurement:
    path: str
    lines: int
    max_class_methods: int
    largest_class: str


def measure(path: Path) -> Measurement | None:
    try:
        source = path.read_text("utf-8")
        tree = ast.parse(source)
    except (OSError, SyntaxError, UnicodeError):
        return None
    worst_name = ""
    worst = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        count = sum(
            1 for item in node.body if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
        )
        if count > worst:
            worst = count
            worst_name = node.name
    return Measurement(
        path=str(path.relative_to(ROOT)),
        lines=len(source.splitlines()),
        max_class_methods=worst,
        largest_class=worst_name,
    )


def measure_tree(roots: tuple[str, ...] = SCANNED) -> dict[str, Measurement]:
    found: dict[str, Measurement] = {}
    for root in roots:
        base = ROOT / root
        if not base.exists():
            continue
        for path in sorted(base.rglob("*.py")):
            if "__pycache__" in str(path):
                continue
            measurement = measure(path)
            if measurement is not None:
                found[measurement.path] = measurement
    return found


def load_baseline(path: Path) -> dict[str, dict[str, int]]:
    try:
        payload = json.loads(path.read_text("utf-8"))
    except (OSError, ValueError):
        return {}
    entries = payload.get("modules")
    return entries if isinstance(entries, dict) else {}


def write_baseline(path: Path, measurements: dict[str, Measurement]) -> int:
    """Record only what exceeds a threshold. A baseline of everything is noise."""
    modules = {
        m.path: {"lines": m.lines, "max_class_methods": m.max_class_methods}
        for m in measurements.values()
        if m.lines > MAX_NEW_MODULE_LINES or m.max_class_methods > MAX_NEW_CLASS_METHODS
    }
    path.write_text(
        json.dumps(
            {
                "schema": BASELINE_SCHEMA,
                "description": (
                    "Modules already above the size thresholds. Entries may shrink "
                    "and may never grow; a file that has shrunk must be re-recorded, "
                    "because a stale entry is headroom nobody earned."
                ),
                "max_new_module_lines": MAX_NEW_MODULE_LINES,
                "max_new_class_methods": MAX_NEW_CLASS_METHODS,
                "modules": dict(sorted(modules.items())),
            },
            indent=2,
        )
        + "\n"
    )
    return len(modules)


def check(
    measurements: dict[str, Measurement], baseline: dict[str, dict[str, int]]
) -> tuple[list[str], list[str]]:
    """Returns (failures, stale_entries)."""
    failures: list[str] = []
    stale: list[str] = []

    for path, measurement in sorted(measurements.items()):
        recorded = baseline.get(path)
        if recorded is None:
            if measurement.lines > MAX_NEW_MODULE_LINES:
                failures.append(
                    f"{path}: {measurement.lines} lines exceeds the "
                    f"{MAX_NEW_MODULE_LINES}-line ceiling for a module not already "
                    "in the baseline — a new God object is never grandfathered"
                )
            if measurement.max_class_methods > MAX_NEW_CLASS_METHODS:
                failures.append(
                    f"{path}: class {measurement.largest_class} has "
                    f"{measurement.max_class_methods} methods, over the "
                    f"{MAX_NEW_CLASS_METHODS}-method ceiling for a new class"
                )
            continue

        allowed_lines = int(recorded.get("lines", 0))
        allowed_methods = int(recorded.get("max_class_methods", 0))
        if measurement.lines > allowed_lines:
            failures.append(
                f"{path}: grew to {measurement.lines} lines from a baseline of "
                f"{allowed_lines}. This file is already known to be too large; "
                "growth here is the failure this ratchet exists to stop"
            )
        if measurement.max_class_methods > allowed_methods:
            failures.append(
                f"{path}: class {measurement.largest_class} grew to "
                f"{measurement.max_class_methods} methods from a baseline of "
                f"{allowed_methods}"
            )
        if (
            measurement.lines < allowed_lines
            or measurement.max_class_methods < allowed_methods
        ):
            stale.append(
                f"{path}: now {measurement.lines} lines / "
                f"{measurement.max_class_methods} methods, baseline still says "
                f"{allowed_lines} / {allowed_methods}"
            )

    for path in sorted(baseline):
        if path not in measurements:
            stale.append(f"{path}: recorded in the baseline but no longer exists")

    return failures, stale


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default=str(DEFAULT_BASELINE))
    parser.add_argument("--write-baseline", action="store_true")
    args = parser.parse_args(argv)

    baseline_path = Path(args.baseline)
    measurements = measure_tree()

    if args.write_baseline:
        count = write_baseline(baseline_path, measurements)
        print(f"module size baseline written: {count} oversized module(s)")
        return 0

    baseline = load_baseline(baseline_path)
    if not baseline:
        print(
            f"error: no usable baseline at {baseline_path}; "
            "run with --write-baseline to record the current state",
            file=sys.stderr,
        )
        return 2

    failures, stale = check(measurements, baseline)

    oversized = sum(
        1
        for m in measurements.values()
        if m.lines > MAX_NEW_MODULE_LINES or m.max_class_methods > MAX_NEW_CLASS_METHODS
    )
    print(f"modules scanned: {len(measurements)}; over threshold: {oversized}")

    if stale:
        print(f"\n📉 {len(stale)} baseline entry/entries are stale — refresh with")
        print("   python tools/lint_module_size.py --write-baseline")
        for entry in stale[:20]:
            print(f"   {entry}")
        if len(stale) > 20:
            print(f"   … and {len(stale) - 20} more")

    if failures:
        print(f"\n❌ {len(failures)} size regression(s):")
        for failure in failures:
            print(f"   {failure}")
        return 1

    if stale:
        # A stale entry is headroom nobody earned. Refusing it is what keeps the
        # ratchet ratcheting rather than slowly becoming a record of history.
        return 1

    print("✅ no module grew past its baseline")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
