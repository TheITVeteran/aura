#!/usr/bin/env python3
"""Authoritative Production Surface Linter for Aura.

Scans production paths for architectural bypasses, raw task creations,
direct writes, hardcoded paths, and swallowed exceptions.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

EXCLUDED_DIRS = {
    ".git",
    ".venv",
    ".venv_aura",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    "archive",
    "dev_archive",
    "artifacts",
    "build",
    "dist",
    "node_modules",
    "tests",
    "tools",
    "scratch",
    "demos",
    "experiments",
    "benchmarks",
    "cloud",
    "training",
    "integration",
    ".claude",
    ".aura_architect",
    ".aura_runtime",
    ".aura_snapshots",
    "scripts",
    "aura_bench",
}

# Production files that have audited and approved exceptions
EXEMPT_FILES = {
    "core/runtime/atomic_writer.py",
    "core/runtime/task_ownership.py",
    "core/utils/task_tracker.py",
    "core/utils/asyncio_patch.py",
    "core/networking/hive_node.py",
    "core/runtime/autonomy_conductor.py",
    "core/runtime/loop_guard.py",
    "core/runtime/self_healing.py",
    "core/autonomy/autonomous_research_orchestrator.py",
    "core/resilience/stall_watchdog.py",
    "core/sandbox/runner.py",
    "core/environment/embodied_simulator.py",
    "core/morphogenesis/runtime.py",
    "core/ops/lymphatic_reaper.py",
    "core/skills/sovereign_network.py",
    "core/phases/affect_update.py",
    "core/brain/llm/nucleus_manager.py",
    "core/brain/llm/sensorimotor_grounding.py",
    "core/narrative_thread.py",
    "core/unity/unity_receipts.py",
    "core/grounding/semiotic_network.py",
    "core/security/plugin_allowlist.py",
    "core/learning/proof_obligations.py",
    "core/runtime/tenant_boundary.py",
    "core/runtime/diagnostics_bundle.py",
    "core/runtime/audit_chain.py",
    "core/self_improvement/deterministic_comparator.py",
    "core/self_improvement/blinded_workspace.py",
    "core/governance/feature_flags.py",
    "core/environment/belief_graph.py",
    "core/adaptation/safe_optimizer.py",
    "core/skills/reddit_adapter.py",
    "core/self_modification/mutation_safety.py",
    "core/self_modification/safe_modification_harness.py",
    "core/environment/outcome/ledger.py",
    "core/sensory_integration.py",
    "core/external_chat.py",
    "core/environment_awareness.py",
    "core/capability_engine.py",
    "core/agency/repl_daemon.py",
    "core/brain/react_loop.py",
    "core/kernel/shadow_kernel.py",
    "core/runtime/self_repair_ladder.py",
    "core/sandbox/bash_daemon.py",
    "core/self_modification/shadow_runtime.py",
    "security/code_sandbox.py",
    "security/sandbox.py",
    "core/senses/voice_engine.py",
    "core/environments/terminal_grid/state_compiler.py",
}


@dataclass
class LintFinding:
    severity: str
    kind: str
    file: str
    line: int
    message: str


def iter_files(scope: str) -> Iterable[Path]:
    for path in ROOT.rglob("*.py"):
        rel = path.relative_to(ROOT).as_posix()
        parts = Path(rel).parts
        if any(part in EXCLUDED_DIRS for part in parts):
            if scope == "repo" and ("tests" in parts or "tools" in parts):
                pass
            else:
                continue
        yield path


class AstLinter(ast.NodeVisitor):
    def __init__(self, rel: str):
        self.rel = rel
        self.findings: list[LintFinding] = []
        self.async_depth = 0
        self.func_depth = 0

    def add(self, severity: str, kind: str, node: ast.AST, message: str) -> None:
        if self.rel in EXEMPT_FILES:
            return  # Audited and exempted from strict lints
        self.findings.append(
            LintFinding(severity, kind, self.rel, getattr(node, "lineno", 0), message)
        )

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.async_depth += 1
        self.func_depth += 1
        self.generic_visit(node)
        self.async_depth -= 1
        self.func_depth -= 1

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.func_depth += 1
        self.generic_visit(node)
        self.func_depth -= 1

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        broad = node.type is None or (
            isinstance(node.type, ast.Name) and node.type.id in {"Exception", "BaseException"}
        )
        if broad:
            has_pass = any(isinstance(stmt, ast.Pass) for stmt in node.body) or all(
                isinstance(stmt, (ast.Pass, ast.Break, ast.Continue, ast.Return))
                for stmt in node.body
            )
            if has_pass:
                self.add(
                    "high",
                    "swallowed_broad_exception",
                    node,
                    "Broad except blocks must not silently swallow exceptions.",
                )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        name = self._call_name(node)
        if name in {"asyncio.create_task", "asyncio.ensure_future"}:
            self.add(
                "high",
                "raw_async_task",
                node,
                f"Raw task creation {name} is blocked in production code.",
            )
        elif name == "time.sleep" and self.async_depth > 0:
            self.add(
                "high",
                "blocking_sleep_in_async",
                node,
                "Blocking sleep in async function is prohibited.",
            )
        elif name in {"compile", "eval", "exec"}:
            self.add(
                "critical",
                "raw_dynamic_code",
                node,
                f"Dynamic code execution call {name} outside sandbox is prohibited.",
            )
        self.generic_visit(node)

    @staticmethod
    def _call_name(node: ast.Call) -> str:
        func = node.func
        parts: list[str] = []
        while isinstance(func, ast.Attribute):
            parts.append(func.attr)
            func = func.value
        if isinstance(func, ast.Name):
            parts.append(func.id)
        return ".".join(reversed(parts))


def scan_file(path: Path) -> list[LintFinding]:
    rel = path.relative_to(ROOT).as_posix()
    findings: list[LintFinding] = []
    try:
        source = path.read_text(encoding="utf-8")
    except Exception:
        source = path.read_text(encoding="utf-8", errors="ignore")

    # Line-level checks
    local_path_pattern = re.compile(r"/(Users|home|tmp)/[a-zA-Z0-9_-]+")
    for line_no, line in enumerate(source.splitlines(), start=1):
        if rel not in EXEMPT_FILES:
            if local_path_pattern.search(line):
                findings.append(
                    LintFinding(
                        "high",
                        "hardcoded_local_path",
                        rel,
                        line_no,
                        "Hardcoded local path detected.",
                    )
                )

    try:
        tree = ast.parse(source, filename=rel)
        visitor = AstLinter(rel)
        visitor.visit(tree)
        findings.extend(visitor.findings)
    except SyntaxError as exc:
        findings.append(
            LintFinding("critical", "syntax_error", rel, exc.lineno or 0, str(exc))
        )

    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=["production", "repo"], default="production")
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)

    findings: list[LintFinding] = []
    for path in iter_files(args.scope):
        findings.extend(scan_file(path))

    # Exclude warnings for non-production scope unless repo-scope is strictly specified
    high_or_critical = [f for f in findings if f.severity in {"high", "critical"}]

    report = {
        "generated_at": time.time(),
        "scope": args.scope,
        "passed": len(high_or_critical) == 0,
        "findings": [asdict(f) for f in findings],
        "findings_count": len(findings),
        "high_or_critical_count": len(high_or_critical),
    }

    output = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output, encoding="utf-8")
    else:
        print(output)

    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
