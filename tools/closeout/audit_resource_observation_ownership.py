#!/usr/bin/env python3
"""Reject unattributed host-resource observations in production Python."""

from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
EXCLUDED_PARTS = frozenset(
    {
        ".git",
        ".claude",
        ".aura",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "archive",
        "artifacts",
        "build",
        "dev_archive",
        "dist",
        "logs",
        "scratch",
        "site-packages",
        "tests",
    }
)
CANONICAL_ADAPTERS = frozenset(
    {
        "core/runtime/resource_observation.py",
        "core/runtime/process_footprint.py",
        "core/runtime/resource_psutil.py",
        "core/runtime/thermal.py",
        "core/utils/memory_monitor.py",
        # This adapter converts observed child identities into termination
        # handles only on the explicitly host-backed watchdog path.
        "core/resilience/memory_watchdog.py",
    }
)
PSUTIL_RESOURCE_CALLS = frozenset(
    {
        "boot_time",
        "cpu_count",
        "cpu_freq",
        "cpu_percent",
        "cpu_stats",
        "cpu_times",
        "disk_usage",
        "disk_io_counters",
        "getloadavg",
        "net_connections",
        "net_if_addrs",
        "net_io_counters",
        "pid_exists",
        "pids",
        "process_iter",
        "sensors_battery",
        "sensors_temperatures",
        "swap_memory",
        "virtual_memory",
    }
)
PROCESS_OBSERVATION_CALLS = frozenset(
    {
        "children",
        "cmdline",
        "connections",
        "cpu_percent",
        "cpu_times",
        "create_time",
        "cwd",
        "environ",
        "exe",
        "is_running",
        "memory_full_info",
        "memory_info",
        "memory_maps",
        "memory_percent",
        "name",
        "net_connections",
        "num_fds",
        "num_handles",
        "num_threads",
        "open_files",
        "parent",
        "parents",
        "ppid",
        "status",
        "threads",
        "username",
    }
)
ACCELERATOR_OBSERVATION_CALLS = frozenset(
    {"get_active_memory", "get_cache_memory", "get_peak_memory"}
)
OS_RESOURCE_CALLS = frozenset({"cpu_count", "getloadavg", "sysconf"})
RESOURCE_MODULE_CALLS = frozenset({"getrusage"})
PLATFORM_RESOURCE_CALLS = frozenset({"proc_listchildpids", "proc_pid_rusage"})


@dataclass(frozen=True)
class AuditFinding:
    code: str
    path: str
    line: int
    detail: str


def _repository_source_paths(root: Path) -> list[Path]:
    paths: list[Path] = []
    for directory, child_dirs, filenames in os.walk(root):
        child_dirs[:] = [name for name in child_dirs if name not in EXCLUDED_PARTS]
        base = Path(directory)
        paths.extend(
            base / name
            for name in filenames
            if name.endswith(".py") and (base / name).is_file()
        )
    return sorted(paths)


def _qualified_name(node: ast.AST) -> tuple[str, ...]:
    parts: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
    return tuple(reversed(parts))


def _assignment_targets(node: ast.AST) -> set[tuple[str, ...]]:
    if isinstance(node, (ast.Name, ast.Attribute)):
        qualified = _qualified_name(node)
        return {qualified} if qualified else set()
    if isinstance(node, (ast.Tuple, ast.List)):
        targets: set[tuple[str, ...]] = set()
        for item in node.elts:
            targets.update(_assignment_targets(item))
        return targets
    return set()


def _is_psutil_process_call(node: ast.AST | None, aliases: set[str]) -> bool:
    if not isinstance(node, ast.Call):
        return False
    qualified = _qualified_name(node.func)
    return len(qualified) == 2 and qualified[0] in aliases and qualified[1] == "Process"


def _findings_for_tree(tree: ast.Module, relative_path: str) -> list[AuditFinding]:
    if relative_path in CANONICAL_ADAPTERS:
        return []

    psutil_aliases: set[str] = set()
    shutil_aliases: set[str] = set()
    os_aliases: set[str] = set()
    resource_aliases: set[str] = set()
    direct_psutil_functions: dict[str, str] = {}
    direct_shutil_functions: set[str] = set()
    direct_os_functions: dict[str, str] = {}
    direct_resource_functions: dict[str, str] = {}
    process_handle_targets: set[tuple[str, ...]] = set()
    process_iteration_targets: set[tuple[str, ...]] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "psutil":
                    psutil_aliases.add(alias.asname or "psutil")
                elif alias.name == "shutil":
                    shutil_aliases.add(alias.asname or "shutil")
                elif alias.name == "os":
                    os_aliases.add(alias.asname or "os")
                elif alias.name == "resource":
                    resource_aliases.add(alias.asname or "resource")
        elif isinstance(node, ast.ImportFrom):
            if node.module == "psutil":
                for alias in node.names:
                    if alias.name in PSUTIL_RESOURCE_CALLS:
                        direct_psutil_functions[alias.asname or alias.name] = alias.name
            elif node.module == "shutil":
                for alias in node.names:
                    if alias.name == "disk_usage":
                        direct_shutil_functions.add(alias.asname or alias.name)
            elif node.module == "os":
                for alias in node.names:
                    if alias.name in OS_RESOURCE_CALLS:
                        direct_os_functions[alias.asname or alias.name] = alias.name
            elif node.module == "resource":
                for alias in node.names:
                    if alias.name in RESOURCE_MODULE_CALLS:
                        direct_resource_functions[alias.asname or alias.name] = alias.name

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and _is_psutil_process_call(node.value, psutil_aliases):
            for target in node.targets:
                process_handle_targets.update(_assignment_targets(target))
        elif isinstance(node, ast.AnnAssign) and _is_psutil_process_call(
            node.value,
            psutil_aliases,
        ):
            process_handle_targets.update(_assignment_targets(node.target))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for argument in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs):
                annotation = _qualified_name(argument.annotation) if argument.annotation else ()
                if (
                    len(annotation) == 2
                    and annotation[0] in psutil_aliases
                    and annotation[1] == "Process"
                ):
                    process_handle_targets.add((argument.arg,))
        elif isinstance(node, (ast.For, ast.AsyncFor)) and isinstance(node.iter, ast.Call):
            iterator = _qualified_name(node.iter.func)
            if (
                len(iterator) == 2
                and iterator[0] in psutil_aliases
                and iterator[1] == "process_iter"
            ):
                process_iteration_targets.update(_assignment_targets(node.target))

    findings: list[AuditFinding] = []
    call_function_ids = {
        id(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)
    }
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        qualified = _qualified_name(function)
        if isinstance(function, ast.Name):
            if function.id in direct_psutil_functions:
                api = direct_psutil_functions[function.id]
                findings.append(
                    AuditFinding(
                        "direct_psutil_resource_observation",
                        relative_path,
                        node.lineno,
                        f"psutil.{api} must route through ResourceObserver",
                    )
                )
            elif function.id in direct_shutil_functions:
                findings.append(
                    AuditFinding(
                        "direct_disk_observation",
                        relative_path,
                        node.lineno,
                        "shutil.disk_usage must route through ResourceObserver",
                    )
                )
            elif function.id in direct_os_functions:
                api = direct_os_functions[function.id]
                findings.append(
                    AuditFinding(
                        "direct_standard_resource_observation",
                        relative_path,
                        node.lineno,
                        f"os.{api} must route through ResourceObserver",
                    )
                )
            elif function.id in direct_resource_functions:
                api = direct_resource_functions[function.id]
                findings.append(
                    AuditFinding(
                        "direct_standard_resource_observation",
                        relative_path,
                        node.lineno,
                        f"resource.{api} must route through ResourceObserver",
                    )
                )
        if len(qualified) >= 2 and qualified[0] in psutil_aliases:
            api = qualified[-1]
            if api in PSUTIL_RESOURCE_CALLS:
                findings.append(
                    AuditFinding(
                        "direct_psutil_resource_observation",
                        relative_path,
                        node.lineno,
                        f"psutil.{api} must route through ResourceObserver",
                    )
                )
            elif api in PROCESS_OBSERVATION_CALLS and "Process" in qualified:
                findings.append(
                    AuditFinding(
                        "direct_process_observation",
                        relative_path,
                        node.lineno,
                        f"Process.{api} must route through ResourceObserver",
                    )
                )
        if isinstance(function, ast.Attribute) and function.attr in PROCESS_OBSERVATION_CALLS:
            receiver = function.value
            receiver_name = _qualified_name(receiver)
            rooted_process_call = _is_psutil_process_call(receiver, psutil_aliases)
            tracked_process_handle = receiver_name in (
                process_handle_targets | process_iteration_targets
            )
            if rooted_process_call or tracked_process_handle:
                findings.append(
                    AuditFinding(
                        "direct_process_observation",
                        relative_path,
                        node.lineno,
                        f"Process.{function.attr} must route through ResourceObserver",
                    )
                )
        if (
            len(qualified) >= 2
            and qualified[0] in shutil_aliases
            and qualified[-1] == "disk_usage"
        ):
            findings.append(
                AuditFinding(
                    "direct_disk_observation",
                    relative_path,
                    node.lineno,
                    "shutil.disk_usage must route through ResourceObserver",
                )
            )
        if (
            len(qualified) >= 2
            and qualified[0] in os_aliases
            and qualified[-1] in OS_RESOURCE_CALLS
        ):
            findings.append(
                AuditFinding(
                    "direct_standard_resource_observation",
                    relative_path,
                    node.lineno,
                    f"os.{qualified[-1]} must route through ResourceObserver",
                )
            )
        if (
            len(qualified) >= 2
            and qualified[0] in resource_aliases
            and qualified[-1] in RESOURCE_MODULE_CALLS
        ):
            findings.append(
                AuditFinding(
                    "direct_standard_resource_observation",
                    relative_path,
                    node.lineno,
                    f"resource.{qualified[-1]} must route through ResourceObserver",
                )
            )
        if qualified and qualified[-1] in ACCELERATOR_OBSERVATION_CALLS:
            findings.append(
                AuditFinding(
                    "direct_accelerator_observation",
                    relative_path,
                    node.lineno,
                    f"accelerator {qualified[-1]} must route through ResourceObserver",
                )
            )
        if qualified and qualified[-1] in PLATFORM_RESOURCE_CALLS:
            findings.append(
                AuditFinding(
                    "direct_platform_resource_observation",
                    relative_path,
                    node.lineno,
                    f"platform {qualified[-1]} must route through a canonical adapter",
                )
            )

    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute) or id(node) in call_function_ids:
            continue
        qualified = _qualified_name(node)
        if len(qualified) < 2:
            continue
        api = qualified[-1]
        if qualified[0] in psutil_aliases and api in PSUTIL_RESOURCE_CALLS:
            findings.append(
                AuditFinding(
                    "direct_psutil_resource_reference",
                    relative_path,
                    node.lineno,
                    f"psutil.{api} reference must route through ResourceObserver",
                )
            )
        elif qualified[0] in shutil_aliases and api == "disk_usage":
            findings.append(
                AuditFinding(
                    "direct_disk_observation_reference",
                    relative_path,
                    node.lineno,
                    "shutil.disk_usage reference must route through ResourceObserver",
                )
            )
        elif qualified[0] in os_aliases and api in OS_RESOURCE_CALLS:
            findings.append(
                AuditFinding(
                    "direct_standard_resource_reference",
                    relative_path,
                    node.lineno,
                    f"os.{api} reference must route through ResourceObserver",
                )
            )
        elif qualified[0] in resource_aliases and api in RESOURCE_MODULE_CALLS:
            findings.append(
                AuditFinding(
                    "direct_standard_resource_reference",
                    relative_path,
                    node.lineno,
                    f"resource.{api} reference must route through ResourceObserver",
                )
            )
    return findings


def run_audit(*, root: Path = ROOT) -> dict[str, Any]:
    findings: list[AuditFinding] = []
    scanned = 0
    parse_errors: list[dict[str, Any]] = []
    for path in _repository_source_paths(root):
        relative = path.relative_to(root).as_posix()
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, UnicodeError, SyntaxError) as exc:
            parse_errors.append(
                {
                    "path": relative,
                    "error": f"{type(exc).__name__}:{exc}",
                }
            )
            continue
        scanned += 1
        findings.extend(_findings_for_tree(tree, relative))

    findings.sort(key=lambda item: (item.path, item.line, item.code))
    return {
        "schema": "aura.resource_observation_ownership.v1",
        "passed": not findings and not parse_errors,
        "scanned_python_files": scanned,
        "canonical_adapters": sorted(CANONICAL_ADAPTERS),
        "finding_count": len(findings),
        "findings": [asdict(finding) for finding in findings],
        "parse_errors": parse_errors,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    report = run_audit(root=args.root.resolve())
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        status = "PASS" if report["passed"] else "FAIL"
        print(
            f"RESOURCE_OBSERVATION_OWNERSHIP={status} "
            f"files={report['scanned_python_files']} findings={report['finding_count']}"
        )
        for finding in report["findings"]:
            print(
                f"  {finding['path']}:{finding['line']} "
                f"{finding['code']} {finding['detail']}"
            )
        for error in report["parse_errors"]:
            print(f"  {error['path']} parse_error {error['error']}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
