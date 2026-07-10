#!/usr/bin/env python3
"""Fail on unclassified or stale cognitive candidate-gate surfaces."""

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

from core.consciousness.candidate_gate_inventory import (  # noqa: E402
    COGNITIVE_GATE_SURFACES,
    inventory_report,
    validate_inventory_contract,
)

_DISCOVERY_ROOTS = (
    "core/consciousness",
    "core/phenomenal_substrate",
    "core/brain/llm/context_gate.py",
)
_EXPLICIT_METHODS: dict[str, frozenset[str]] = {
    "GlobalWorkspace": frozenset({"submit", "run_competition", "compete"}),
    "AttentionSchema": frozenset({"set_focus"}),
    "MultipleDraftsEngine": frozenset({"submit_input", "probe"}),
    "QuorumDecisionGate": frozenset({"check_quorum"}),
    "AttentionGate": frozenset({"gate_context"}),
    "AttentionalContextGate": frozenset({"should_include_block"}),
    "ReadinessGate": frozenset({"evaluate"}),
    "ExecutiveInhibitor": frozenset({"authorize"}),
    "SomaticMarkerGate": frozenset({"evaluate"}),
    "SubstrateAuthority": frozenset({"authorize"}),
}
_GENERIC_GATE_METHODS = frozenset(
    {"authorize", "check_quorum", "evaluate", "gate_context", "should_include_block"}
)


def _files(root: Path) -> list[Path]:
    paths: list[Path] = []
    for relative in _DISCOVERY_ROOTS:
        path = root / relative
        if path.is_dir():
            paths.extend(sorted(path.rglob("*.py")))
        elif path.is_file():
            paths.append(path)
    return list(dict.fromkeys(paths))


def discover_surfaces(root: Path) -> set[tuple[str, str, str]]:
    discovered: set[tuple[str, str, str]] = set()
    for path in _files(root):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        relative = path.relative_to(root).as_posix()
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            expected = _EXPLICIT_METHODS.get(node.name)
            available = {
                child.name
                for child in node.body
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            if expected is None and node.name.endswith("Gate"):
                expected = frozenset(available & _GENERIC_GATE_METHODS)
                if not expected:
                    discovered.add((relative, node.name, "<unclassified-gate-class>"))
                    continue
            if expected is None:
                continue
            for method in expected & available:
                discovered.add((relative, node.name, method))
    return discovered


def audit(root: Path) -> dict[str, Any]:
    issues = validate_inventory_contract()
    declared = {
        (surface.owner_file, surface.owner_class, surface.callable_name)
        for surface in COGNITIVE_GATE_SURFACES
    }
    discovered = discover_surfaces(root)
    for missing in sorted(discovered - declared):
        issues.append("unclassified surface: " + ":".join(missing))
    for stale in sorted(declared - discovered):
        issues.append("stale inventory surface: " + ":".join(stale))
    report = inventory_report()
    report.update(
        {
            "passed": not issues,
            "declared_count": len(declared),
            "discovered_count": len(discovered),
            "issues": issues,
        }
    )
    return report


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
            f"Cognitive gate audit {status}: {report['declared_count']} declared, "
            f"{report['discovered_count']} discovered"
        )
        for issue in report["issues"]:
            print(f"- {issue}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
