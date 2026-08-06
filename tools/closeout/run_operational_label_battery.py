#!/usr/bin/env python3
"""Run Aura's operational-label validator battery.

The label baseline file defines what each label means and which validators
exercise it. This runner turns that map into an executable proof surface so the
labels cannot quietly remain prose.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.closeout.operational_label_baselines import (
    BASELINES,
    ROOT,
    audit_evidence_integrity,
    evaluate,
)

# Bounded wall-clock ceiling for the full validator pytest run. The battery is
# a proof harness, never an unbounded background job.
_BATTERY_TIMEOUT_S = float(os.environ.get("AURA_LABEL_BATTERY_TIMEOUT_S", "5400"))
_VALIDATOR_TIMEOUT_S = float(os.environ.get("AURA_LABEL_VALIDATOR_TIMEOUT_S", "900"))


@dataclass(frozen=True)
class LabelValidatorPlan:
    key: str
    label: str
    validator_paths: tuple[str, ...]
    live_required: bool
    claim_boundary: str
    operational_definition: str
    minimum_behavioral_bar: tuple[str, ...]
    positive_controls: tuple[str, ...]
    negative_controls: tuple[str, ...]
    answer_contract: tuple[str, ...]


def build_label_plans(
    *,
    labels: set[str] | None = None,
    include_live: bool = True,
) -> list[LabelValidatorPlan]:
    selected: list[LabelValidatorPlan] = []
    for baseline in BASELINES:
        if labels and baseline.key not in labels:
            continue
        validator_paths = tuple(
            path
            for path in baseline.validator_paths
            if include_live or "/live/" not in path
        )
        selected.append(
            LabelValidatorPlan(
                key=baseline.key,
                label=baseline.label,
                validator_paths=validator_paths,
                live_required=bool(baseline.live_artifacts),
                claim_boundary=baseline.claim_boundary,
                operational_definition=baseline.operational_definition,
                minimum_behavioral_bar=baseline.minimum_behavioral_bar,
                positive_controls=baseline.positive_controls,
                negative_controls=baseline.negative_controls,
                answer_contract=baseline.answer_contract,
            )
        )
    return selected


def unique_validator_paths(plans: list[LabelValidatorPlan]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for plan in plans:
        for path in plan.validator_paths:
            if path not in seen:
                seen.add(path)
                ordered.append(path)
    return ordered


def build_pytest_command(
    plans: list[LabelValidatorPlan],
    *,
    extra_args: list[str] | None = None,
) -> list[str]:
    paths = unique_validator_paths(plans)
    return [sys.executable, "-m", "pytest", "-q", *paths, *(extra_args or [])]


def build_pytest_commands(
    plans: list[LabelValidatorPlan],
    *,
    extra_args: list[str] | None = None,
) -> list[list[str]]:
    """Build one bounded pytest command per validator file.

    The all-in-one command is useful for display and backwards compatibility,
    but closeout proof tooling should identify the exact validator that stalls
    or fails. Per-file commands make the proof battery auditable and prevent a
    single quiet subprocess from looking like progress.
    """

    return [
        [sys.executable, "-m", "pytest", "-q", path, *(extra_args or [])]
        for path in unique_validator_paths(plans)
    ]


def _existing_path(path: str) -> bool:
    return (ROOT / path).exists()


def build_report(
    plans: list[LabelValidatorPlan],
    *,
    command: list[str],
    exit_code: int | None,
    stdout: str = "",
    stderr: str = "",
    require_live: bool = False,
    validator_results: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    status_by_key = {status.key: status for status in evaluate(require_live=require_live)}
    evidence_issues = audit_evidence_integrity()
    return {
        "total_labels": len(plans),
        "validator_files": unique_validator_paths(plans),
        "command": command,
        "validator_results": validator_results or [],
        "exit_code": exit_code,
        "passed": None if exit_code is None else exit_code == 0,
        "require_live": require_live,
        "evidence_integrity": {
            "passed": not evidence_issues,
            "issues": [asdict(issue) for issue in evidence_issues],
        },
        "labels": [
            {
                **asdict(plan),
                "validator_paths_exist": all(_existing_path(path) for path in plan.validator_paths),
                "baseline_status": status_by_key.get(plan.key).status
                if status_by_key.get(plan.key)
                else "missing",
            }
            for plan in plans
        ],
        "stdout_tail": "\n".join(stdout.splitlines()[-80:]),
        "stderr_tail": "\n".join(stderr.splitlines()[-80:]),
    }


def _tail(text: str, lines: int = 80) -> str:
    return "\n".join((text or "").splitlines()[-lines:])


def _result_exit_code(results: list[dict[str, Any]]) -> int:
    for result in results:
        code = result.get("exit_code")
        if code not in (0, None):
            return int(code)
    return 0


def run_validator_commands(
    commands: list[list[str]],
    *,
    timeout_s: float,
) -> tuple[int, list[dict[str, Any]], str, str]:
    """Run validator commands with per-file timeout and progress accounting."""

    from core.runtime.subprocess_gateway import get_subprocess_gateway

    results: list[dict[str, Any]] = []
    stdout_parts: list[str] = []
    stderr_parts: list[str] = []
    deadline = time.monotonic() + max(float(_BATTERY_TIMEOUT_S), 1.0)
    for index, command in enumerate(commands, start=1):
        validator_path = command[4] if len(command) > 4 else "unknown"
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            result = {
                "validator_path": validator_path,
                "command": command,
                "exit_code": 124,
                "timed_out": True,
                "duration_s": 0.0,
                "stdout_tail": "",
                "stderr_tail": "Full operational label battery exceeded wall-clock deadline.",
            }
            results.append(result)
            stderr_parts.append(result["stderr_tail"])
            break

        effective_timeout = max(1.0, min(float(timeout_s), remaining))
        print(
            f"[label-battery] {index}/{len(commands)} {validator_path} "
            f"(timeout={effective_timeout:.0f}s)",
            flush=True,
        )
        started = time.monotonic()
        try:
            completed = get_subprocess_gateway().run(
                command,
                cwd=ROOT,
                timeout=effective_timeout,
                offline_tooling=True,
                check=False,
                source="proof_tooling:run_operational_label_battery",
                accelerator_capability="auto",
            )
            duration = time.monotonic() - started
            result = {
                "validator_path": validator_path,
                "command": command,
                "exit_code": completed.returncode,
                "timed_out": False,
                "duration_s": round(duration, 3),
                "stdout_tail": _tail(completed.stdout),
                "stderr_tail": _tail(completed.stderr),
            }
            if completed.stdout:
                stdout_parts.append(completed.stdout)
                print(completed.stdout, end="")
            if completed.stderr:
                stderr_parts.append(completed.stderr)
                print(completed.stderr, end="", file=sys.stderr)
        except subprocess.TimeoutExpired as exc:
            duration = time.monotonic() - started
            stdout = exc.stdout if isinstance(exc.stdout, str) else (exc.stdout or b"").decode("utf-8", "replace")
            stderr = exc.stderr if isinstance(exc.stderr, str) else (exc.stderr or b"").decode("utf-8", "replace")
            timeout_msg = f"Validator timed out after {effective_timeout:.0f}s: {validator_path}"
            result = {
                "validator_path": validator_path,
                "command": command,
                "exit_code": 124,
                "timed_out": True,
                "duration_s": round(duration, 3),
                "stdout_tail": _tail(stdout),
                "stderr_tail": _tail((stderr + "\n" + timeout_msg).strip()),
            }
            stdout_parts.append(stdout)
            stderr_parts.append((stderr + "\n" + timeout_msg).strip())
            print(timeout_msg, file=sys.stderr, flush=True)
        results.append(result)
        if result["exit_code"] != 0:
            break
    return _result_exit_code(results), results, "".join(stdout_parts), "".join(stderr_parts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", action="append", help="Run only this baseline key; repeatable")
    parser.add_argument(
        "--skip-live",
        action="store_true",
        help="Exclude validators under tests/agi/live for a source-only pass.",
    )
    parser.add_argument(
        "--require-live-artifacts",
        action="store_true",
        help="Also require current live artifact paths declared by label baselines.",
    )
    parser.add_argument("--list", action="store_true", help="Print the selected validator plan only")
    parser.add_argument("--json-out", type=Path, help="Write machine-readable report JSON")
    parser.add_argument(
        "--pytest-arg",
        action="append",
        default=[],
        help="Extra argument forwarded to pytest; repeatable.",
    )
    args = parser.parse_args(argv)

    labels = set(args.label or []) or None
    plans = build_label_plans(labels=labels, include_live=not args.skip_live)
    if not plans:
        print("No operational label baselines selected.", file=sys.stderr)
        return 2

    command = build_pytest_command(plans, extra_args=args.pytest_arg)
    if args.list:
        report = build_report(
            plans,
            command=command,
            exit_code=None,
            require_live=args.require_live_artifacts,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    evidence_issues = audit_evidence_integrity()
    if evidence_issues:
        report = build_report(
            plans,
            command=command,
            exit_code=2,
            stderr="Evidence integrity issues: "
            + "; ".join(f"{issue.baseline_key}:{issue.path}:{issue.reason}" for issue in evidence_issues),
            require_live=args.require_live_artifacts,
        )
        if args.json_out:
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(report["stderr_tail"], file=sys.stderr)
        return 2

    missing = [path for path in unique_validator_paths(plans) if not _existing_path(path)]
    if missing:
        report = build_report(
            plans,
            command=command,
            exit_code=2,
            stderr="Missing validator paths: " + ", ".join(missing),
            require_live=args.require_live_artifacts,
        )
        if args.json_out:
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(report["stderr_tail"], file=sys.stderr)
        return 2

    validator_commands = build_pytest_commands(plans, extra_args=args.pytest_arg)
    exit_code, validator_results, stdout, stderr = run_validator_commands(
        validator_commands,
        timeout_s=_VALIDATOR_TIMEOUT_S,
    )
    report = build_report(
        plans,
        command=command,
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        require_live=args.require_live_artifacts,
        validator_results=validator_results,
    )
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
