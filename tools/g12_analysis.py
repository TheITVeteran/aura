#!/usr/bin/env python3
"""G12's numbers: her against her base model, request by request, on each benchmark.

MATH-500, BIG-Bench Extra Hard Mini (and BIG-Bench Hard, when given) rows
are paired by request and compared with an exact two-sided McNemar test; IFBench is read from AllenAI's grader output as
the prompt-level loose accuracy its authors report, with strict beside it,
and paired the same way on whether every instruction was followed.

G12 is assessed on her three runs (amendment of 8 October in
docs/evidence/G12_FRONTIER_EVALUATION_2026-10-07.md); a benchmark whose base
run is not given is reported for her alone: accuracy with its 95% Wilson
interval, by group, and how many expected cases are missing.

Usage:
    g12_analysis.py --her-math DIR [--base-math DIR] --her-bbeh DIR [--base-bbeh DIR]
        --her-ifbench DIR [--base-ifbench DIR] --output FILE
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def two_sided_mcnemar(first_only: int, second_only: int) -> float:
    n = first_only + second_only
    if n == 0:
        return 1.0
    k = max(first_only, second_only)
    tail = sum(math.comb(n, i) for i in range(k, n + 1)) / 2**n
    return min(1.0, 2 * tail)


def wilson(correct: int, n: int) -> list[float]:
    """95% Wilson interval for a proportion."""
    if n == 0:
        return [0.0, 0.0]
    z, p = 1.959963984540054, correct / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(centre - half, 4), round(centre + half, 4)]


def rows(directory: Path, key: str) -> dict[str, dict[str, Any]]:
    out = {}
    for path in sorted(directory.expanduser().glob("**/*.json")):
        if path.parent.name not in {"rows", "ordinary"}:
            continue
        row = json.loads(path.read_text(encoding="utf-8"))
        identity = str(row[key])
        if identity in out:
            raise ValueError(f"duplicate result case ID {identity!r}: {path}")
        out[identity] = row
    return out


def paired(her: dict[str, dict], base: dict[str, dict], field: str, group: str | None = None,
           *, expected_keys: Iterable[str] | None = None) -> dict[str, Any]:
    expected = None
    if expected_keys is not None:
        expected_list = list(expected_keys)
        expected = set(expected_list)
        if len(expected) != len(expected_list):
            raise ValueError("expected source case IDs are not unique")
        unexpected = (set(her) | set(base)) - expected
        if unexpected:
            raise ValueError(f"results contain unexpected source case IDs: {sorted(unexpected)!r}")
    shared = sorted(set(her) & set(base))
    her_only = sum(bool(her[k][field]) and not bool(base[k][field]) for k in shared)
    base_only = sum(bool(base[k][field]) and not bool(her[k][field]) for k in shared)
    her_correct = sum(bool(her[k][field]) for k in shared)
    base_correct = sum(bool(base[k][field]) for k in shared)
    result = {"requests": len(shared), "her": her_correct, "base": base_correct,
              "her_interval": wilson(her_correct, len(shared)), "base_interval": wilson(base_correct, len(shared)),
              "her_only": her_only, "base_only": base_only, "two_sided_p": two_sided_mcnemar(her_only, base_only),
              "unpaired": {"her": len(set(base) - set(her)), "base": len(set(her) - set(base))},
              "missing": None if expected is None else
              {"her": len(expected - set(her)), "base": len(expected - set(base))},
              "coverage": {"expected": None, "complete": None} if expected is None else
              {"expected": len(expected), "her": len(her), "base": len(base),
               "complete": set(her) == set(base) == expected if expected else None}}
    if group:
        by: dict[str, list[str]] = defaultdict(list)
        for k in shared:
            by[str(her[k][group])].append(k)
        result["by_" + group] = {name: {"n": len(keys), "her": sum(bool(her[k][field]) for k in keys),
                                        "base": sum(bool(base[k][field]) for k in keys)}
                                 for name, keys in sorted(by.items())}
    return result


def alone(her: dict[str, dict], field: str, group: str | None = None,
          *, expected_keys: Iterable[str] | None = None) -> dict[str, Any]:
    """Her accuracy on one benchmark with no base run beside it."""
    expected = set(expected_keys) if expected_keys is not None else None
    if expected is not None and set(her) - expected:
        raise ValueError(f"results contain unexpected source case IDs: {sorted(set(her) - expected)!r}")
    correct = sum(bool(row[field]) for row in her.values())
    result = {"requests": len(her), "her": correct, "her_interval": wilson(correct, len(her)),
              "missing": None if expected is None else len(expected - set(her))}
    if group:
        by: dict[str, list[str]] = defaultdict(list)
        for k, row in her.items():
            by[str(row[group])].append(k)
        result["by_" + group] = {name: {"n": len(keys), "her": sum(bool(her[k][field]) for k in keys)}
                                 for name, keys in sorted(by.items())}
    return result


def ifbench(directory: Path) -> dict[str, dict[str, Any]]:
    """Per prompt, whether every instruction was followed, loose and strict."""
    out: dict[str, dict[str, Any]] = {}
    for mode in ("loose", "strict"):
        found = sorted(directory.expanduser().glob(f"**/*eval_results_{mode}.jsonl"))
        if not found:
            raise SystemExit(f"no IFBench {mode} results under {directory}")
        for line in found[-1].read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                out.setdefault(record["prompt"], {})[mode] = bool(record["follow_all_instructions"])
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("her-math", "her-ifbench"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("base-math", "base-ifbench", "her-bbeh", "base-bbeh", "her-bbh", "base-bbh"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--bbeh-mini", type=Path, default=Path("~/.aura/benchmarks/g09/bbeh/mini/data.json"),
                        help="the pinned complete Mini source, for case coverage and reference validation")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report: dict[str, Any] = {}
    her_math = rows(args.her_math, "unique_id")
    report["math500"] = (paired(her_math, rows(args.base_math, "unique_id"), "correct", "level") if args.base_math
                         else alone(her_math, "correct", "level"))
    if args.her_bbeh:
        from tools.run_g12_bbeh import load_cases, validate_saved_result

        _, catalog = load_cases(args.bbeh_mini)
        her = rows(args.her_bbeh, "id")
        base = rows(args.base_bbeh, "id") if args.base_bbeh else {}
        for row in (*her.values(), *base.values()):
            catalog.validate_row(row)
            validate_saved_result(row)
        report["bbeh_mini"] = (paired(her, base, "correct", "task", expected_keys=catalog.by_id) if args.base_bbeh
                               else alone(her, "correct", "task", expected_keys=catalog.by_id))
    if args.her_bbh and args.base_bbh:
        report["bbh"] = paired(rows(args.her_bbh, "id"), rows(args.base_bbh, "id"), "correct", "task")
    her_if = ifbench(args.her_ifbench)
    base_if = ifbench(args.base_ifbench) if args.base_ifbench else None
    report["ifbench"] = {mode: paired(her_if, base_if, mode) if base_if is not None else alone(her_if, mode)
                         for mode in ("loose", "strict")}
    args.output.write_text(json.dumps(report, indent=1, sort_keys=True), encoding="utf-8")
    print(json.dumps({name: {k: v for k, v in value.items() if not k.startswith("by_")} if "her" in value else value
                      for name, value in report.items()}, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
