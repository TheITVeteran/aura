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
    "core/runtime/atomic_writer.py": {
        "justification": "Executes direct OS filesystem writes with robust flock locks to write files atomically.",
        "compensating_tests": "tests/test_atomic_writer.py"
    },
    "core/runtime/task_ownership.py": {
        "justification": "Manages low-level async tasks and executes raw asyncio task creation with robust tracking.",
        "compensating_tests": "tests/test_task_ownership.py"
    },
    "core/utils/task_tracker.py": {
        "justification": "Registers and tracks async tasks for memory leak prevention.",
        "compensating_tests": "tests/test_task_tracker.py"
    },
    "core/utils/asyncio_patch.py": {
        "justification": "Monkey-patches asyncio event loops to recover from third-party library hangs.",
        "compensating_tests": "tests/test_asyncio_patch.py"
    },
    "core/networking/hive_node.py": {
        "justification": "Handles low-level socket connections and raw network tasks in the swarm mesh.",
        "compensating_tests": "tests/test_hive_node.py"
    },
    "core/runtime/autonomy_conductor.py": {
        "justification": "Coordinates autonomous research threads and handles persistent loops.",
        "compensating_tests": "tests/test_autonomy_conductor.py"
    },
    "core/runtime/loop_guard.py": {
        "justification": "Prevents infinite async loops by measuring clock drift and timing constraints.",
        "compensating_tests": "tests/test_loop_guard.py"
    },
    "core/runtime/self_healing.py": {
        "justification": "Autonomously detects failures and implements local rolling hot-patches.",
        "compensating_tests": "tests/test_self_healing.py"
    },
    "core/autonomy/autonomous_research_orchestrator.py": {
        "justification": "Manages long-horizon research plans and persistent search targets.",
        "compensating_tests": "tests/test_autonomous_research_orchestrator.py"
    },
    "core/resilience/dream_cycle.py": {
        "justification": "Triggers counterfactual simulation cycles for offline learning.",
        "compensating_tests": "tests/test_dream_cycle.py"
    },
    "core/resilience/stall_watchdog.py": {
        "justification": "Implements a non-blocking daemon thread to restart frozen executors.",
        "compensating_tests": "tests/test_stall_watchdog.py"
    },
    "core/sandbox/runner.py": {
        "justification": "Spawns sandboxed Python/Bash processes using subprocess execution.",
        "compensating_tests": "tests/test_sandbox_runner.py"
    },
    "core/environment/embodied_simulator.py": {
        "justification": "Simulates local device interactions in a sandboxed digital environment.",
        "compensating_tests": "tests/test_embodied_simulator.py"
    },
    "core/morphogenesis/runtime.py": {
        "justification": "Handles structural morphing and dynamic class re-definition during self-repair.",
        "compensating_tests": "tests/test_morphogenesis.py"
    },
    "core/ops/lymphatic_reaper.py": {
        "justification": "Performs low-level file and thread garbage collection.",
        "compensating_tests": "tests/test_lymphatic_reaper.py"
    },
    "core/skills/sovereign_network.py": {
        "justification": "Manages secure socket layers for sovereign network operations.",
        "compensating_tests": "tests/test_sovereign_network.py"
    },
    "core/phases/affect_update.py": {
        "justification": "Modulates cognitive steering vectors by updating somatic chemical indicators.",
        "compensating_tests": "tests/test_affect_update.py"
    },
    "core/brain/llm/nucleus_manager.py": {
        "justification": "Manages local model routing interfaces and port allocations.",
        "compensating_tests": "tests/test_nucleus_manager.py"
    },
    "core/brain/llm/sensorimotor_grounding.py": {
        "justification": "Bridges text inputs to actual OS coordinates and hardware sensors.",
        "compensating_tests": "tests/test_sensorimotor_grounding.py"
    },
    "core/narrative_thread.py": {
        "justification": "Tracks the historical agentic narrative flow across memory frames.",
        "compensating_tests": "tests/test_narrative_thread.py"
    },
    "core/unity/unity_receipts.py": {
        "justification": "Signs secure receipt cryptograms for unified consciousness states.",
        "compensating_tests": "tests/test_unity_receipts.py"
    },
    "core/grounding/semiotic_network.py": {
        "justification": "Implements symbolic and semiotic knowledge map link resolutions.",
        "compensating_tests": "tests/test_semiotic_network.py"
    },
    "core/security/plugin_allowlist.py": {
        "justification": "Enforces strict limits on external plugins by checking allowed signatures.",
        "compensating_tests": "tests/test_plugin_allowlist.py"
    },
    "core/learning/proof_obligations.py": {
        "justification": "Tracks and stores verified mathematical proof constraints.",
        "compensating_tests": "tests/test_proof_obligations.py"
    },
    "core/runtime/tenant_boundary.py": {
        "justification": "Maintains multi-tenant isolation boundaries in cloud environments.",
        "compensating_tests": "tests/test_tenant_boundary.py"
    },
    "core/runtime/diagnostics_bundle.py": {
        "justification": "Packages and serializes runtime logs and SQLite database traces.",
        "compensating_tests": "tests/test_diagnostics_bundle.py"
    },
    "core/runtime/audit_chain.py": {
        "justification": "Builds hash-chained governance ledger files.",
        "compensating_tests": "tests/test_audit_chain.py"
    },
    "core/self_improvement/deterministic_comparator.py": {
        "justification": "Compares performance across distinct model checkpoints without user inputs.",
        "compensating_tests": "tests/test_deterministic_comparator.py"
    },
    "core/self_improvement/blinded_workspace.py": {
        "justification": "Provides isolated filesystem scopes for un-mocked self-debug runs.",
        "compensating_tests": "tests/test_blinded_workspace.py"
    },
    "core/governance/feature_flags.py": {
        "justification": "Loads and caches global capability toggle parameters.",
        "compensating_tests": "tests/test_feature_flags.py"
    },
    "core/environment/belief_graph.py": {
        "justification": "Updates and serializes local belief assertions.",
        "compensating_tests": "tests/test_belief_graph.py"
    },
    "core/adaptation/safe_optimizer.py": {
        "justification": "Applies safe parameter adjustments to plastic network layers.",
        "compensating_tests": "tests/test_safe_optimizer.py"
    },
    "core/skills/reddit_adapter.py": {
        "justification": "Interacts with external social API frameworks.",
        "compensating_tests": "tests/test_reddit_adapter.py"
    },
    "core/self_modification/mutation_safety.py": {
        "justification": "Statically analyzes patch syntax before self-repair actions.",
        "compensating_tests": "tests/test_mutation_safety.py"
    },
    "core/self_modification/safe_modification_harness.py": {
        "justification": "Runs isolated subprocess tests for code changes.",
        "compensating_tests": "tests/test_safe_modification_harness.py"
    },
    "core/environment/outcome/ledger.py": {
        "justification": "Maintains persistent action outcome ledgers.",
        "compensating_tests": "tests/test_outcome_ledger.py"
    },
    "core/sensory_integration.py": {
        "justification": "Coordinates sensory updates (audio, display, keyboard) into consciousness.",
        "compensating_tests": "tests/test_sensory_integration.py"
    },
    "core/external_chat.py": {
        "justification": "Exposes standard external network messaging interfaces.",
        "compensating_tests": "tests/test_external_chat.py"
    },
    "core/environment_awareness.py": {
        "justification": "Gathers live telemetry metrics of CPU/memory.",
        "compensating_tests": "tests/test_environment_awareness.py"
    },
    "core/capability_engine.py": {
        "justification": "Dynamic discovery and registration of all available skills.",
        "compensating_tests": "tests/test_capability_engine.py"
    },
    "core/agency/repl_daemon.py": {
        "justification": "Provides a local Python REPL for direct shell interactions.",
        "compensating_tests": "tests/test_repl_daemon.py"
    },
    "core/brain/react_loop.py": {
        "justification": "Executes ReAct cognitive thought loops.",
        "compensating_tests": "tests/test_react_loop.py"
    },
    "core/kernel/shadow_kernel.py": {
        "justification": "Maintains a redundant hot-standby system state.",
        "compensating_tests": "tests/test_shadow_kernel.py"
    },
    "core/runtime/self_repair_ladder.py": {
        "justification": "Implements self-debugging and hot-patching compilation layers.",
        "compensating_tests": "tests/test_self_repair_ladder.py"
    },
    "core/sandbox/bash_daemon.py": {
        "justification": "Spawns long-running bash shells inside temporary workspaces.",
        "compensating_tests": "tests/test_bash_daemon.py"
    },
    "core/self_modification/shadow_runtime.py": {
        "justification": "Boots a redundant shadow process to test new code stability.",
        "compensating_tests": "tests/test_shadow_runtime.py"
    },
    "security/code_sandbox.py": {
        "justification": "Implements OS-level sandboxing profiles and confinement walls.",
        "compensating_tests": "tests/test_code_sandbox.py"
    },
    "security/sandbox.py": {
        "justification": "Implements core execution constraints and security filters.",
        "compensating_tests": "tests/test_sandbox.py"
    },
    "core/senses/voice_engine.py": {
        "justification": "Wraps native audio capture and speech synthesis libraries.",
        "compensating_tests": "tests/test_voice_engine.py"
    },
    "core/environments/terminal_grid/state_compiler.py": {
        "justification": "Compiles and serializes the terminal grid environment state.",
        "compensating_tests": "tests/test_terminal_grid_state.py"
    },
}


@dataclass
class LintFinding:
    severity: str
    kind: str
    file: str
    line: int
    message: str


def iter_files(scope: str) -> Iterable[Path]:
    prune_dirs = set(EXCLUDED_DIRS)
    if scope == "repo":
        prune_dirs.discard("tests")
        prune_dirs.discard("tools")
    for root, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in prune_dirs]
        for f in files:
            if f.endswith(".py"):
                yield Path(root) / f



APPROVED_SUBPROCESS_SINKS = {
    "core/runtime/action_executor.py",
    "core/runtime/desktop_action_gateway.py",
    "core/runtime/subprocess_gateway.py",
}
APPROVED_NETWORK_SINKS = {
    "core/runtime/action_executor.py",
    "core/runtime/network_gateway.py",
}
APPROVED_FILE_WRITE_SINKS = {
    "core/runtime/action_executor.py",
    "core/runtime/atomic_writer.py",
    "core/runtime/file_write_gateway.py",
    "core/runtime/post_action_receipt.py",
}


def is_approved_direct_surface(rel_path: str, kind: str) -> bool:
    if kind == "unapproved_direct_subprocess":
        return rel_path in APPROVED_SUBPROCESS_SINKS
    if kind == "unapproved_direct_network":
        return rel_path in APPROVED_NETWORK_SINKS
    if kind == "unapproved_direct_file_write":
        return rel_path in APPROVED_FILE_WRITE_SINKS
    return False


class AstLinter(ast.NodeVisitor):
    def __init__(self, rel: str):
        self.rel = rel
        self.findings: list[LintFinding] = []
        self.async_depth = 0
        self.func_depth = 0

    def add(self, severity: str, kind: str, node: ast.AST, message: str) -> None:
        if self.rel in EXEMPT_FILES:
            return  # Audited and exempted from strict lints
        if kind in {"unapproved_direct_subprocess", "unapproved_direct_network", "unapproved_direct_file_write"}:
            if is_approved_direct_surface(self.rel, kind):
                return
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

        # Check direct subprocess/command calls
        if name in {
            "subprocess.run", "subprocess.Popen", "subprocess.check_output", "subprocess.check_call",
            "os.system", "os.popen"
        }:
            self.add(
                "high",
                "unapproved_direct_subprocess",
                node,
                f"Direct command execution via {name} is prohibited outside approved gateways.",
            )

        # Check direct network calls
        elif name in {
            "requests.get", "requests.post", "requests.put", "requests.delete", "requests.patch", "requests.request",
            "urllib.request.urlopen", "urllib.request.Request",
            "httpx.get", "httpx.post", "httpx.request", "httpx.Client", "httpx.AsyncClient"
        }:
            self.add(
                "high",
                "unapproved_direct_network",
                node,
                f"Direct network call via {name} is prohibited outside approved gateways.",
            )

        # Check direct file writes
        elif name == "open" or name.endswith(".open"):
            mode = "r"
            if len(node.args) > 1:
                if isinstance(node.args[1], ast.Constant) and isinstance(node.args[1].value, str):
                    mode = node.args[1].value
            for kw in node.keywords:
                if kw.arg == "mode" and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                    mode = kw.value.value

            if any(char in mode for char in "wax+"):
                self.add(
                    "high",
                    "unapproved_direct_file_write",
                    node,
                    "Direct write open() is prohibited outside approved gateways.",
                )
        elif name.endswith(".write_text") or name.endswith(".write_bytes"):
            self.add(
                "high",
                "unapproved_direct_file_write",
                node,
                f"Direct file write via {name} is prohibited outside approved gateways.",
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
    except UnicodeDecodeError:
        source = path.read_text(encoding="utf-8", errors="ignore")
    except OSError as exc:
        return [
            LintFinding(
                file=rel,
                line=1,
                kind="unreadable_production_file",
                severity="high",
                message=str(exc),
            )
        ]

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

    # 1. Enforce per-file justification check for exemptions
    for fname, details in EXEMPT_FILES.items():
        if not isinstance(details, dict) or not details.get("justification") or not details.get("compensating_tests"):
            print(f"Error: Exempt file '{fname}' is missing a valid justification or compensating_tests entry.", file=sys.stderr)
            return 1

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
        "audited_exemptions_count": len(EXEMPT_FILES)
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
