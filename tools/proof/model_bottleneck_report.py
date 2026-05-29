#!/usr/bin/env python3
"""Build the model bottleneck report for the person-in-a-box proof bundle."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

ABLATION_KEYS = (
    "raw_llm",
    "aura_full_runtime",
    "aura_without_memory",
    "aura_without_system2",
    "aura_without_tools",
    "aura_without_governance",
    "aura_without_retrieval",
    "aura_without_self_repair",
    "aura_with_weaker_model",
    "aura_with_stronger_model",
)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _completion_rate_from_scorecard(scorecard: dict[str, Any] | None) -> float | None:
    if not scorecard:
        return None
    value = scorecard.get("task_completion_rate")
    if isinstance(value, int | float):
        return float(value)
    value = scorecard.get("overall_pass_rate")
    if isinstance(value, int | float):
        return float(value)
    return None


def _normalize_run_result(name: str, payload: dict[str, Any]) -> dict[str, Any]:
    if not payload:
        return {
            "status": "NOT_RUN",
            "reason": "No live comparison artifact was supplied for this lane.",
        }
    if "success_rate" in payload:
        rate = payload.get("success_rate")
    elif "pass_rate" in payload:
        rate = payload.get("pass_rate")
    else:
        rate = payload.get("task_completion_rate")
    if not isinstance(rate, int | float):
        return {
            "status": payload.get("status", "NOT_RUN"),
            "reason": payload.get("reason", f"{name} artifact did not expose a numeric rate."),
        }
    return {
        "status": payload.get("status", "RUN"),
        "success_rate": float(rate),
        "task_count": payload.get("task_count") or payload.get("total_tasks"),
        "source": payload.get("source", "external_comparison_artifact"),
    }


def build_model_bottleneck_report(
    run_dir: Path,
    *,
    scorecard: dict[str, Any] | None = None,
    comparison_path: Path | None = None,
) -> dict[str, Any]:
    """Return a non-synthetic report about runtime lift over model-only lanes.

    The function never fabricates raw model or ablation scores. If no live A/B
    comparison artifact is present, it records NOT_RUN lanes and explicitly
    withholds the runtime-lift claim.
    """
    run_dir = Path(run_dir)
    comparison = _load_json(comparison_path) if comparison_path else {}
    if not comparison:
        comparison = _load_json(run_dir / "MODEL_COMPARISON_RESULTS.json")

    lanes = {key: _normalize_run_result(key, comparison.get(key, {})) for key in ABLATION_KEYS}
    aura_rate = _completion_rate_from_scorecard(scorecard)
    if aura_rate is not None:
        lanes["aura_full_runtime"] = {
            "status": "RUN",
            "success_rate": aura_rate,
            "task_count": scorecard.get("total_tasks") if scorecard else None,
            "source": "person_box_scorecard",
        }

    raw_rate = lanes.get("raw_llm", {}).get("success_rate")
    full_rate = lanes.get("aura_full_runtime", {}).get("success_rate")
    lift = None
    if isinstance(raw_rate, int | float) and isinstance(full_rate, int | float):
        lift = round(float(full_rate) - float(raw_rate), 6)

    report = {
        "schema": "aura.model_bottleneck_report.v1",
        "generated_at_unix": time.time(),
        "lanes": lanes,
        "raw_llm_success": raw_rate if isinstance(raw_rate, int | float) else None,
        "aura_full_runtime_success": full_rate if isinstance(full_rate, int | float) else None,
        "runtime_lift_over_raw_model": lift,
        "claim": (
            "runtime_lift_established"
            if lift is not None and lift > 0
            else "runtime_lift_not_established_without_live_raw_model_comparison"
        ),
        "non_synthetic": True,
        "notes": [
            "Raw model and ablation scores must come from live comparison artifacts.",
            "Missing lanes are reported as NOT_RUN rather than estimated.",
        ],
    }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build Aura person-box model bottleneck report")
    parser.add_argument("run_dir", help="Person-in-box proof run directory")
    parser.add_argument("--comparison", default="", help="Optional MODEL_COMPARISON_RESULTS.json path")
    parser.add_argument("--out", default="", help="Output path; defaults to run_dir/MODEL_BOTTLENECK_REPORT.json")
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir).resolve()
    scorecard = _load_json(run_dir / "SCORECARD.json")
    comparison_path = Path(args.comparison).resolve() if args.comparison else None
    report = build_model_bottleneck_report(run_dir, scorecard=scorecard, comparison_path=comparison_path)
    out = Path(args.out).resolve() if args.out else run_dir / "MODEL_BOTTLENECK_REPORT.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

