#!/usr/bin/env python3
"""Audit every direct MLX model-load reference against an ownership contract."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INVENTORY = ROOT / "config" / "model_load_ownership.json"
MODEL_MODULES = frozenset({"mlx_lm", "mlx_vlm"})
MODEL_CONSTRUCTORS = {
    ("sentence_transformers", "SentenceTransformer"): "sentence_transformers",
    ("faster_whisper", "WhisperModel"): "faster_whisper",
}
SOURCE_EXCLUDED_PARTS = frozenset(
    {
        ".git",
        ".claude",
        ".aura",
        ".aura_architect",
        ".aura_runtime",
        ".aura_snapshots",
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


@dataclass(frozen=True)
class LoadReference:
    path: str
    line: int
    module: str


@dataclass(frozen=True)
class AuditFinding:
    code: str
    path: str
    detail: str


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _references_in_tree(tree: ast.Module) -> set[tuple[int, str]]:
    direct_aliases: dict[str, str] = {}
    constructor_aliases: dict[str, str] = {}
    module_aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module_family = str(node.module or "").split(".", 1)[0]
            if module_family in MODEL_MODULES:
                for alias in node.names:
                    if alias.name == "load":
                        direct_aliases[alias.asname or alias.name] = module_family
            else:
                for alias in node.names:
                    module = MODEL_CONSTRUCTORS.get((str(node.module), alias.name))
                    if module:
                        constructor_aliases[alias.asname or alias.name] = module
        elif isinstance(node, ast.Import):
            for alias in node.names:
                module_family = alias.name.split(".", 1)[0]
                if module_family in MODEL_MODULES:
                    module_aliases[alias.asname or alias.name] = module_family

    references: set[tuple[int, str]] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        if isinstance(function, ast.Name) and function.id in direct_aliases:
            references.add((node.lineno, direct_aliases[function.id]))
        elif isinstance(function, ast.Name) and function.id in constructor_aliases:
            references.add((node.lineno, constructor_aliases[function.id]))
        elif (
            isinstance(function, ast.Attribute)
            and function.attr == "load"
            and isinstance(function.value, ast.Name)
            and function.value.id in module_aliases
        ):
            references.add((node.lineno, module_aliases[function.value.id]))
        elif isinstance(function, ast.Attribute) and function.attr == "from_pretrained":
            references.add((node.lineno, "from_pretrained"))
        elif isinstance(function, ast.Name) and function.id == "whisper_model_cls":
            references.add((node.lineno, "faster_whisper"))
        elif isinstance(function, ast.Name) and function.id == "TTS":
            references.add((node.lineno, "coqui_tts"))
        elif (
            isinstance(function, ast.Attribute)
            and function.attr == "load"
            and isinstance(function.value, ast.Name)
            and function.value.id == "PiperVoice"
        ):
            references.add((node.lineno, "piper"))
    return references


def _load_references(path: Path, relative_path: str) -> list[LoadReference]:
    tree = _parse(path)
    references = _references_in_tree(tree)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
            continue
        source = node.value
        if not any(
            marker in source
            for marker in (
                "mlx_lm",
                "mlx_vlm",
                "from_pretrained",
                "WhisperModel",
            )
        ):
            continue
        try:
            inline_tree = ast.parse(source, mode="exec")
        except SyntaxError:
            continue
        for _inner_line, module in _references_in_tree(inline_tree):
            references.add((node.lineno, module))
    return [
        LoadReference(relative_path, line, module)
        for line, module in sorted(references)
    ]


def _symbol_sites(tree: ast.Module, symbol: str) -> int:
    return sum(
        1
        for node in ast.walk(tree)
        if (
            isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and node.id == symbol
        )
        or (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and node.name == symbol
        )
        or (
            isinstance(node, ast.Attribute)
            and isinstance(node.ctx, ast.Load)
            and node.attr == symbol
        )
    )


def _module_name(path: str) -> str:
    return path.removesuffix(".py").replace("/", ".")


def _repository_source_paths(root: Path) -> list[Path]:
    paths: list[Path] = []
    for directory, child_dirs, filenames in os.walk(root):
        child_dirs[:] = [
            name for name in child_dirs if name not in SOURCE_EXCLUDED_PARTS
        ]
        base = Path(directory)
        paths.extend(
            base / name
            for name in filenames
            if name.endswith(".py") and (base / name).is_file()
        )
    return paths


def _production_importers(root: Path, target_path: str) -> set[str]:
    target_module = _module_name(target_path)
    importers: set[str] = set()
    for path in _repository_source_paths(root):
        relative = path.relative_to(root).as_posix()
        if relative == target_path:
            continue
        tree = _parse(path)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == target_module:
                importers.add(relative)
            elif isinstance(node, ast.Import):
                if any(alias.name == target_module for alias in node.names):
                    importers.add(relative)
    return importers


def run_audit(
    *,
    root: Path = ROOT,
    inventory_path: Path = DEFAULT_INVENTORY,
) -> dict[str, Any]:
    inventory_bytes = inventory_path.read_bytes()
    inventory = json.loads(inventory_bytes)
    entries = {
        str(entry["path"]): dict(entry)
        for entry in list(inventory.get("entries") or [])
    }
    references: list[LoadReference] = []
    parse_findings: list[AuditFinding] = []
    source_paths = _repository_source_paths(root)
    for source_path in source_paths:
        relative = source_path.relative_to(root).as_posix()
        try:
            references.extend(_load_references(source_path, relative))
        except (OSError, SyntaxError, UnicodeError) as exc:
            parse_findings.append(
                AuditFinding("source_unreadable", relative, f"{type(exc).__name__}:{exc}")
            )

    findings = list(parse_findings)
    by_path: dict[str, list[LoadReference]] = {}
    for reference in references:
        by_path.setdefault(reference.path, []).append(reference)
    for owned_path in sorted(set(by_path) - set(entries)):
        findings.append(
            AuditFinding(
                "unowned_model_load",
                owned_path,
                f"load references at lines {[item.line for item in by_path[owned_path]]}",
            )
        )
    for owned_path in sorted(set(entries) - set(by_path)):
        findings.append(
            AuditFinding(
                "stale_inventory_entry",
                owned_path,
                "no model-load reference",
            )
        )

    for owned_path in sorted(set(entries) & set(by_path)):
        entry = entries[owned_path]
        observed = by_path[owned_path]
        expected_count = int(entry.get("expected_load_references") or 0)
        if len(observed) != expected_count:
            findings.append(
                AuditFinding(
                    "load_reference_count_changed",
                    owned_path,
                    f"expected={expected_count} observed={len(observed)}",
                )
            )
        expected_modules = {str(item) for item in entry.get("modules") or []}
        observed_modules = {item.module for item in observed}
        if observed_modules != expected_modules:
            findings.append(
                AuditFinding(
                    "model_module_set_changed",
                    owned_path,
                    f"expected={sorted(expected_modules)} observed={sorted(observed_modules)}",
                )
            )
        guard_path = str(entry.get("guard_path") or owned_path)
        guard_file = root / guard_path
        guard_symbol = str(entry.get("guard_symbol") or "")
        try:
            guard_tree = _parse(guard_file)
            guard_sites = _symbol_sites(guard_tree, guard_symbol)
        except (OSError, SyntaxError, UnicodeError) as exc:
            findings.append(
                AuditFinding("guard_unreadable", guard_path, f"{type(exc).__name__}:{exc}")
            )
            guard_sites = 0
        minimum = int(entry.get("min_guard_sites") or 1)
        if not guard_symbol or guard_sites < minimum:
            findings.append(
                AuditFinding(
                    "ownership_guard_missing",
                    owned_path,
                    f"guard={guard_path}:{guard_symbol} expected_sites>={minimum} observed={guard_sites}",
                )
            )
        worker_entrypoint = str(entry.get("worker_entrypoint") or "")
        if worker_entrypoint:
            source_tree = _parse(root / owned_path)
            if _symbol_sites(source_tree, worker_entrypoint) < 1:
                findings.append(
                    AuditFinding(
                        "worker_entrypoint_missing",
                        owned_path,
                        worker_entrypoint,
                    )
                )
        allowed_importers = {str(item) for item in entry.get("allowed_importers") or []}
        if allowed_importers:
            observed_importers = _production_importers(root, owned_path)
            if observed_importers != allowed_importers:
                findings.append(
                    AuditFinding(
                        "worker_component_importers_changed",
                        owned_path,
                        f"expected={sorted(allowed_importers)} observed={sorted(observed_importers)}",
                    )
                )

    report = {
        "schema": "aura.model_load_ownership.audit.v1",
        "passed": not findings,
        "inventory_sha256": hashlib.sha256(inventory_bytes).hexdigest(),
        "inventory_entries": len(entries),
        "source_paths_scanned": len(source_paths),
        "load_references": len(references),
        "owned_paths": len(by_path),
        "references": [asdict(item) for item in sorted(references, key=lambda item: (item.path, item.line))],
        "findings": [asdict(item) for item in findings],
    }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    report = run_audit(root=args.root.resolve(), inventory_path=args.inventory.resolve())
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        status = "PASS" if report["passed"] else "FAIL"
        print(
            f"MODEL_LOAD_OWNERSHIP={status} paths={report['owned_paths']} "
            f"references={report['load_references']} findings={len(report['findings'])}"
        )
        for finding in report["findings"]:
            print(f"- {finding['code']}: {finding['path']}: {finding['detail']}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
