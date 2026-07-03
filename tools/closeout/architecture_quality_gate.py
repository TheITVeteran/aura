#!/usr/bin/env python3
"""Run Aura's deterministic architecture-quality gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.architecture_quality.gate import ArchitectureQualityGate
from core.architecture_quality.scorer import score_codebase


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT), help="Repository root to score")
    parser.add_argument(
        "--include",
        action="append",
        default=None,
        help="Top-level package/root to include. Repeatable.",
    )
    parser.add_argument("--baseline", help="Baseline report JSON to compare against")
    parser.add_argument("--write-baseline", help="Write current report JSON to this path")
    parser.add_argument("--min-score", type=float, default=None)
    parser.add_argument("--json", action="store_true", help="Emit full JSON")
    args = parser.parse_args()

    include_roots = tuple(args.include or ("core", "interface", "infrastructure", "slo", "tools"))
    root = Path(args.root).resolve()
    current = score_codebase(root, include_roots=include_roots)

    payload: dict[str, Any] = {"current": current.to_dict(), "passed": True, "reasons": []}

    if args.min_score is not None and current.score < args.min_score:
        payload["passed"] = False
        payload["reasons"].append(
            f"score {current.score:.1f} below required minimum {args.min_score:.1f}"
        )

    if args.baseline:
        baseline_data = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
        baseline = _report_from_baseline(baseline_data)
        result = ArchitectureQualityGate(root, include_roots=include_roots).evaluate_reports(
            baseline,
            current,
        )
        payload["baseline"] = baseline.to_dict()
        payload["passed"] = bool(payload["passed"] and result.passed)
        payload["reasons"].extend(result.reasons)

    if args.write_baseline:
        output_path = Path(args.write_baseline)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(_baseline_payload(current), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(current.summary())
        for reason in payload["reasons"]:
            print(f"- {reason}")

    return 0 if payload["passed"] else 1


def _report_from_baseline(data: dict[str, Any]):
    """Rehydrate the report subset needed for comparisons."""
    from core.architecture_quality.scorer import (
        ArchitectureQualityFinding,
        ArchitectureQualityMetrics,
        ArchitectureQualityReport,
    )

    metrics_data = data["metrics"]
    metrics = ArchitectureQualityMetrics(
        module_count=int(metrics_data["module_count"]),
        dependency_edges=int(metrics_data["dependency_edges"]),
        cycle_count=int(metrics_data["cycle_count"]),
        largest_cycle_size=int(metrics_data["largest_cycle_size"]),
        god_file_count=int(metrics_data["god_file_count"]),
        max_file_lines=int(metrics_data["max_file_lines"]),
        max_out_degree=int(metrics_data["max_out_degree"]),
        max_in_degree=int(metrics_data["max_in_degree"]),
        dependency_concentration_pct=float(metrics_data["dependency_concentration_pct"]),
    )
    findings = tuple(
        ArchitectureQualityFinding(
            severity=str(item["severity"]),
            code=str(item["code"]),
            message=str(item["message"]),
            path=item.get("path"),
            modules=tuple(item.get("modules") or ()),
            value=item.get("value"),
        )
        for item in data.get("findings", ())
    )
    return ArchitectureQualityReport(
        root=str(data.get("root", "")),
        include_roots=tuple(data.get("include_roots") or ()),
        god_file_threshold=int(data.get("god_file_threshold", 1500)),
        metrics=metrics,
        score=float(data["score"]),
        line_counts={str(key): int(value) for key, value in data.get("line_counts", {}).items()},
        module_to_path={str(key): str(value) for key, value in data.get("module_to_path", {}).items()},
        graph={
            str(key): tuple(str(item) for item in value)
            for key, value in data.get("graph", {}).items()
        },
        cycles=tuple(tuple(str(item) for item in cycle) for cycle in data.get("cycles", ())),
        findings=findings,
    )


def _baseline_payload(report) -> dict[str, Any]:
    """Persist only fields needed for stable regression comparison."""
    data = report.to_dict()
    return {
        "root": data["root"],
        "include_roots": data["include_roots"],
        "god_file_threshold": data["god_file_threshold"],
        "metrics": data["metrics"],
        "score": data["score"],
        "line_counts": data["line_counts"],
        "module_to_path": data["module_to_path"],
        "cycles": data["cycles"],
        "findings": data["findings"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
