#!/usr/bin/env python3
"""Fail closed on proof-answer contamination in Aura runtime paths."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

PRODUCTION_PATHS = (
    "aura_main.py",
    "main_daemon.py",
    "core/brain",
    "core/memory",
    "core/orchestrator",
    "core/phases",
    "core/reasoning",
    "core/runtime",
    "core/skills",
    "core/tools",
    "interface",
    "skills",
)

EXCLUDED_DIRS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    ".venv_aura",
    "__pycache__",
    "archive",
    "artifacts",
    "build",
    "data",
    "dev_archive",
    "dist",
    "htmlcov",
    "logs",
    "node_modules",
    "scratch",
    "tests",
    "tools",
}

CONTAMINATION_PATTERNS = {
    "golden_answer": re.compile(r"\bgolden_answer\b"),
    "grader_salts": re.compile(r"\bgrader_salts\b|\.grader_salts"),
    "golden_answer_resolver": re.compile(r"\b_try_resolve_golden_answer\b"),
    "dnu_fixture_path": re.compile(r"tests/agi/fixtures/dnu_tasks"),
    "answer_hashes": re.compile(r"\banswer_hashes\b"),
    "expected_answer": re.compile(r"\bexpected_answer\b"),
}

# Proof instruments may contain generated task targets, but they are not
# production inference code.  Exclusion is allowed only together with the
# import-boundary check below: if runtime code imports one of these modules,
# the lint fails closed before answer-bearing fixtures can reach generation.
NON_RUNTIME_PROOF_HARNESSES = {
    "core/brain/llm/latent_cortex/state_causality.py": (
        "core.brain.llm.latent_cortex.state_causality"
    ),
}


@dataclass(frozen=True)
class Finding:
    kind: str
    file: str
    line: int
    detail: str


def _is_excluded(path: Path, root: Path) -> bool:
    rel_parts = path.relative_to(root).parts
    return any(part in EXCLUDED_DIRS for part in rel_parts)


def _iter_production_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for rel in PRODUCTION_PATHS:
        path = root / rel
        if path.is_file() and path.suffix == ".py":
            files.append(path)
        elif path.is_dir():
            files.extend(
                item for item in path.rglob("*.py") if not _is_excluded(item, root)
            )
    return sorted(set(files))


def scan_file(path: Path, root: Path) -> list[Finding]:
    rel = path.relative_to(root).as_posix()
    try:
        source = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        source = path.read_text(encoding="utf-8", errors="replace")

    findings: list[Finding] = []
    for line_no, line in enumerate(source.splitlines(), start=1):
        for kind, pattern in CONTAMINATION_PATTERNS.items():
            if pattern.search(line):
                findings.append(
                    Finding(
                        kind=kind,
                        file=rel,
                        line=line_no,
                        detail=line.strip()[:240],
                    )
                )
    return findings


def run_lint(root: Path, scope: str) -> dict:
    if scope != "production":
        raise ValueError(f"Unsupported scope: {scope}")

    files = _iter_production_files(root)
    findings: list[Finding] = []
    for path in files:
        rel = path.relative_to(root).as_posix()
        if rel in NON_RUNTIME_PROOF_HARNESSES:
            continue
        findings.extend(scan_file(path, root))
        try:
            source = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            source = path.read_text(encoding="utf-8", errors="replace")
        for harness_module in NON_RUNTIME_PROOF_HARNESSES.values():
            import_pattern = re.compile(
                rf"(?m)^\s*(?:from\s+{re.escape(harness_module)}\s+import\b"
                rf"|import\s+{re.escape(harness_module)}\b)"
            )
            match = import_pattern.search(source)
            if match:
                findings.append(
                    Finding(
                        kind="proof_harness_runtime_import",
                        file=rel,
                        line=source.count("\n", 0, match.start()) + 1,
                        detail=match.group(0).strip()[:240],
                    )
                )

    return {
        "generated_at_unix": time.time(),
        "root": str(root),
        "scope": scope,
        "files_scanned": len(files),
        "passed": len(findings) == 0,
        "findings": [asdict(finding) for finding in findings],
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="Repository root to scan.")
    parser.add_argument("--scope", default="production", choices=("production",))
    parser.add_argument("--out", default="", help="Optional JSON report path.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    report = run_lint(Path(args.root).resolve(), args.scope)
    output = json.dumps(report, indent=2, sort_keys=True)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output, encoding="utf-8")
    else:
        print(output)

    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
