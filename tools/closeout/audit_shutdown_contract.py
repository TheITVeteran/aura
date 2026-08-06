#!/usr/bin/env python3
"""Fail when Aura's canonical shutdown ownership contract drifts."""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.runtime.shutdown_coordinator import SHUTDOWN_PHASES  # noqa: E402
from tools.shutdown_signal_matrix import CASE_SPECS  # noqa: E402

_EXPECTED_PHASES = (
    "output_flush",
    "memory_commit",
    "state_vault",
    "actors",
    "model_runtime",
    "event_bus",
    "task_supervisor",
)
_EXPECTED_SIGNAL_CASES = frozenset(
    {
        "launcher_bootstrap",
        "orchestrator_boot_repeated",
        "ready_repeated",
        "state_vault_repeated",
        "model_runtime_repeated",
        "container_repeated",
        "root_finalization_repeated",
        "active_foreground_repeated",
    }
)
_REQUIRED_CALLS: dict[str, dict[str, frozenset[str]]] = {
    "core/runtime/shutdown_coordinator.py": {
        "ShutdownCoordinator.shutdown": frozenset(
            {"request_shutdown", "shutdown_remaining_budget_seconds"}
        ),
        "ShutdownCoordinator._execute_shutdown": frozenset(
            {
                "shutdown_deadline_monotonic",
                "shutdown_remaining_budget_seconds",
                "_record_global_deadline_exhausted",
            }
        ),
        "ShutdownCoordinator._invoke": frozenset(
            {"shutdown_remaining_budget_seconds"}
        ),
        "publish_shutdown_verdict": frozenset(
            {"shutdown_request_snapshot", "shutdown_admission_snapshot"}
        ),
        "publish_root_exit_verdict": frozenset(
            {"shutdown_request_snapshot", "shutdown_admission_snapshot"}
        ),
        "request_shutdown": frozenset(
            {"_shutdown_budget_seconds", "shutdown_request_snapshot"}
        ),
    },
    "core/runtime/root_signal_owner.py": {
        "RootShutdownSignalOwner._handle_signal": frozenset({"request_shutdown"}),
    },
    "core/runtime/task_ownership.py": {
        "runtime_shutdown_blocks_new_work": frozenset(
            {"runtime_shutdown_requested", "record_shutdown_admission_event"}
        ),
    },
    "core/runtime/subprocess_gateway.py": {
        "_require_not_shutting_down": frozenset(
            {"is_shutdown_requested", "record_shutdown_admission_event"}
        ),
    },
    "core/runtime/runtime_hygiene.py": {
        "RuntimeHygieneManager._shutdown_blocks_resource_start": frozenset(
            {"is_shutdown_requested", "record_shutdown_admission_event"}
        ),
        "RuntimeHygieneManager._cleanup_shutdown_resources": frozenset(
            {"_close_shutdown_resource"}
        ),
    },
    "core/container.py": {
        "ServiceContainer._runtime_registration_suppressed": frozenset(
            {"is_shutdown_requested", "record_shutdown_admission_event"}
        ),
    },
}
_REQUEST_SNAPSHOT_FIELDS = frozenset(
    {
        "requested",
        "first_reason",
        "last_reason",
        "first_requested_at_unix",
        "elapsed_seconds",
        "deadline_at_unix",
        "deadline_source",
        "initial_budget_seconds",
        "remaining_budget_seconds",
        "deadline_exhausted",
        "deadline_tighten_count",
        "request_count",
    }
)


def _call_name(node: ast.Call) -> str:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def _qualified_functions(tree: ast.AST) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    functions: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
    for node in getattr(tree, "body", []):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions[node.name] = node
        elif isinstance(node, ast.ClassDef):
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    functions[f"{node.name}.{child.name}"] = child
    return functions


def _returned_literal_keys(function: ast.AST) -> set[str]:
    keys: set[str] = set()
    for node in ast.walk(function):
        if not isinstance(node, ast.Return) or not isinstance(node.value, ast.Dict):
            continue
        for key in node.value.keys:
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                keys.add(key.value)
    return keys


def audit(root: Path) -> dict[str, Any]:
    issues: list[str] = []
    checked_functions = 0

    if tuple(SHUTDOWN_PHASES) != _EXPECTED_PHASES:
        issues.append(
            f"canonical phase order drifted: expected={_EXPECTED_PHASES!r} "
            f"actual={tuple(SHUTDOWN_PHASES)!r}"
        )
    if frozenset(CASE_SPECS) != _EXPECTED_SIGNAL_CASES:
        issues.append(
            "external signal matrix coverage drifted: "
            f"expected={sorted(_EXPECTED_SIGNAL_CASES)!r} "
            f"actual={sorted(CASE_SPECS)!r}"
        )
    for name, spec in CASE_SPECS.items():
        if "repeated" in name and spec.repeat_signal is None:
            issues.append(f"signal matrix case {name} no longer injects a repeat signal")

    parsed: dict[str, ast.Module] = {}
    for relative, required_functions in _REQUIRED_CALLS.items():
        path = root / relative
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError) as exc:
            issues.append(f"cannot parse {relative}: {type(exc).__name__}: {exc}")
            continue
        parsed[relative] = tree
        functions = _qualified_functions(tree)
        for qualified_name, required_calls in required_functions.items():
            function = functions.get(qualified_name)
            if function is None:
                issues.append(f"missing shutdown contract function: {relative}:{qualified_name}")
                continue
            checked_functions += 1
            calls = {_call_name(node) for node in ast.walk(function) if isinstance(node, ast.Call)}
            missing_calls = sorted(required_calls - calls)
            if missing_calls:
                issues.append(
                    f"{relative}:{qualified_name} lost calls {missing_calls!r}"
                )

    coordinator_tree = parsed.get("core/runtime/shutdown_coordinator.py")
    if coordinator_tree is not None:
        functions = _qualified_functions(coordinator_tree)
        snapshot = functions.get("shutdown_request_snapshot")
        if snapshot is None:
            issues.append("shutdown_request_snapshot is missing")
        else:
            missing_fields = sorted(_REQUEST_SNAPSHOT_FIELDS - _returned_literal_keys(snapshot))
            if missing_fields:
                issues.append(f"shutdown request diagnostics lost fields {missing_fields!r}")

    hygiene_path = root / "core/runtime/runtime_hygiene.py"
    try:
        hygiene_source = hygiene_path.read_text(encoding="utf-8")
    except OSError as exc:
        issues.append(f"cannot read runtime hygiene source: {type(exc).__name__}: {exc}")
    else:
        if "key=lambda record: record.sequence" not in hygiene_source or "reverse=True" not in hygiene_source:
            issues.append("runtime hygiene no longer closes owned resources in reverse registration order")
        for owner_kind in ("tasks", "threads", "processes", "resources", "native_resources"):
            if f'"{owner_kind}"' not in hygiene_source:
                issues.append(f"runtime hygiene final census lost owner class {owner_kind}")

    return {
        "schema": "aura.closeout.shutdown_contract_audit.v1",
        "passed": not issues,
        "canonical_phases": list(SHUTDOWN_PHASES),
        "signal_matrix_cases": sorted(CASE_SPECS),
        "checked_functions": checked_functions,
        "request_snapshot_fields": sorted(_REQUEST_SNAPSHOT_FIELDS),
        "issues": issues,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = audit(args.root.resolve())
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        status = "passed" if report["passed"] else "failed"
        print(
            f"Shutdown contract audit {status}: "
            f"{report['checked_functions']} ownership functions, "
            f"{len(report['signal_matrix_cases'])} external cases"
        )
        for issue in report["issues"]:
            print(f"- {issue}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
