"""Deterministic architecture quality scoring.

The scorer intentionally avoids model judgment. It reads Python modules,
builds a project-local import graph, detects dependency cycles, and records
module-size/concentration risks that should not silently regress.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_INCLUDE_ROOTS = ("core", "interface", "infrastructure", "slo", "tools")
DEFAULT_EXCLUDE_PARTS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "artifacts",
    "build",
    "dist",
    "htmlcov",
    "node_modules",
    "venv",
    ".venv",
}


@dataclass(frozen=True)
class ArchitectureQualityFinding:
    """A concrete architecture-quality finding."""

    severity: str
    code: str
    message: str
    path: str | None = None
    modules: tuple[str, ...] = ()
    value: int | float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "path": self.path,
            "modules": list(self.modules),
            "value": self.value,
        }


@dataclass(frozen=True)
class ArchitectureQualityMetrics:
    """Compact metrics used by CI and self-modification gates."""

    module_count: int
    dependency_edges: int
    cycle_count: int
    largest_cycle_size: int
    god_file_count: int
    max_file_lines: int
    max_out_degree: int
    max_in_degree: int
    dependency_concentration_pct: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "module_count": self.module_count,
            "dependency_edges": self.dependency_edges,
            "cycle_count": self.cycle_count,
            "largest_cycle_size": self.largest_cycle_size,
            "god_file_count": self.god_file_count,
            "max_file_lines": self.max_file_lines,
            "max_out_degree": self.max_out_degree,
            "max_in_degree": self.max_in_degree,
            "dependency_concentration_pct": round(self.dependency_concentration_pct, 4),
        }


@dataclass(frozen=True)
class ArchitectureQualityReport:
    """Architecture-quality report for one source tree snapshot."""

    root: str
    include_roots: tuple[str, ...]
    god_file_threshold: int
    metrics: ArchitectureQualityMetrics
    score: float
    line_counts: dict[str, int] = field(default_factory=dict)
    module_to_path: dict[str, str] = field(default_factory=dict)
    graph: dict[str, tuple[str, ...]] = field(default_factory=dict)
    reverse_graph: dict[str, tuple[str, ...]] = field(default_factory=dict)
    cycles: tuple[tuple[str, ...], ...] = ()
    findings: tuple[ArchitectureQualityFinding, ...] = ()

    def changed_module_for_path(self, path: str) -> str | None:
        normalized = path.replace("\\", "/").removesuffix(".py")
        module = normalized.replace("/", ".")
        if module in self.module_to_path:
            return module
        init_suffix = ".__init__"
        if module.endswith(init_suffix):
            package = module[: -len(init_suffix)]
            if package in self.module_to_path:
                return package
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": self.root,
            "include_roots": list(self.include_roots),
            "god_file_threshold": self.god_file_threshold,
            "metrics": self.metrics.to_dict(),
            "score": round(self.score, 2),
            "line_counts": dict(sorted(self.line_counts.items())),
            "module_to_path": dict(sorted(self.module_to_path.items())),
            "graph": {key: list(value) for key, value in sorted(self.graph.items())},
            "cycles": [list(cycle) for cycle in self.cycles],
            "findings": [finding.to_dict() for finding in self.findings],
        }

    def summary(self) -> str:
        metrics = self.metrics
        return (
            f"score={self.score:.1f}; modules={metrics.module_count}; "
            f"edges={metrics.dependency_edges}; cycles={metrics.cycle_count}; "
            f"god_files={metrics.god_file_count}; max_lines={metrics.max_file_lines}"
        )


def score_codebase(
    root: str | Path,
    *,
    include_roots: Iterable[str] = DEFAULT_INCLUDE_ROOTS,
    exclude_parts: Iterable[str] = DEFAULT_EXCLUDE_PARTS,
    overlay_content: Mapping[str, str] | None = None,
    god_file_threshold: int = 1500,
    max_cycles_reported: int = 40,
) -> ArchitectureQualityReport:
    """Score a Python source tree.

    ``overlay_content`` maps repo-relative paths to candidate text. It lets the
    self-modification harness score quarantined bytes before they are promoted.
    """

    root_path = Path(root).resolve()
    include_roots_tuple = tuple(_clean_part(part) for part in include_roots if _clean_part(part))
    exclude_set = {_clean_part(part) for part in exclude_parts if _clean_part(part)}
    overlays = {
        _normalize_rel_path(path): content
        for path, content in (overlay_content or {}).items()
        if _normalize_rel_path(path).endswith(".py")
    }

    files = _collect_python_files(root_path, include_roots_tuple, exclude_set)
    for rel_path in overlays:
        if _is_included(rel_path, include_roots_tuple):
            files[rel_path] = root_path / rel_path

    module_to_path, path_to_module = _build_module_maps(files)
    known_modules = set(module_to_path)

    line_counts: dict[str, int] = {}
    raw_imports: dict[str, set[str]] = {}
    findings: list[ArchitectureQualityFinding] = []

    for rel_path in sorted(files):
        module = path_to_module[rel_path]
        content = overlays.get(rel_path)
        if content is None:
            try:
                content = files[rel_path].read_text(encoding="utf-8")
            except UnicodeDecodeError:
                content = files[rel_path].read_text(encoding="utf-8", errors="replace")
        line_counts[rel_path] = _line_count(content)
        try:
            tree = ast.parse(content, filename=rel_path)
        except SyntaxError as exc:
            findings.append(
                ArchitectureQualityFinding(
                    severity="critical",
                    code="syntax_error",
                    path=rel_path,
                    message=f"{rel_path} does not parse: {exc.msg}",
                    value=exc.lineno,
                )
            )
            raw_imports[module] = set()
            continue
        raw_imports[module] = _extract_imports(tree, module)

    graph: dict[str, set[str]] = {module: set() for module in known_modules}
    for module, imports in raw_imports.items():
        for raw_import in imports:
            target = _resolve_known_module(raw_import, known_modules)
            if target and target != module:
                graph[module].add(target)

    reverse_graph: dict[str, set[str]] = {module: set() for module in known_modules}
    for source, targets in graph.items():
        for target in targets:
            reverse_graph[target].add(source)

    cycles = _strongly_connected_components(graph)
    cycles = tuple(sorted(cycles, key=lambda item: (-len(item), item))[:max_cycles_reported])
    god_files = {
        path: count for path, count in line_counts.items() if count > god_file_threshold
    }
    findings.extend(_build_findings(graph, cycles, god_files, line_counts, god_file_threshold))

    dependency_edges = sum(len(targets) for targets in graph.values())
    out_degrees = [len(targets) for targets in graph.values()]
    in_degrees = [len(sources) for sources in reverse_graph.values()]
    concentration = _dependency_concentration(out_degrees)
    metrics = ArchitectureQualityMetrics(
        module_count=len(known_modules),
        dependency_edges=dependency_edges,
        cycle_count=len(cycles),
        largest_cycle_size=max((len(cycle) for cycle in cycles), default=0),
        god_file_count=len(god_files),
        max_file_lines=max(line_counts.values(), default=0),
        max_out_degree=max(out_degrees, default=0),
        max_in_degree=max(in_degrees, default=0),
        dependency_concentration_pct=concentration,
    )
    score = _quality_score(metrics)

    return ArchitectureQualityReport(
        root=str(root_path),
        include_roots=include_roots_tuple,
        god_file_threshold=god_file_threshold,
        metrics=metrics,
        score=score,
        line_counts=line_counts,
        module_to_path=module_to_path,
        graph={key: tuple(sorted(value)) for key, value in graph.items()},
        reverse_graph={key: tuple(sorted(value)) for key, value in reverse_graph.items()},
        cycles=cycles,
        findings=tuple(findings),
    )


def _collect_python_files(
    root: Path,
    include_roots: tuple[str, ...],
    exclude_parts: set[str],
) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for include_root in include_roots:
        start = root / include_root
        if not start.exists():
            continue
        for path in start.rglob("*.py"):
            rel_path = path.relative_to(root).as_posix()
            if any(part in exclude_parts for part in rel_path.split("/")):
                continue
            files[rel_path] = path
    return files


def _extract_imports(tree: ast.AST, current_module: str) -> set[str]:
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = _resolve_import_from_base(node, current_module)
            if base:
                imports.add(base)
                imports.update(f"{base}.{alias.name}" for alias in node.names if alias.name != "*")
    return imports


def _resolve_import_from_base(node: ast.ImportFrom, current_module: str) -> str | None:
    if node.level <= 0:
        return node.module

    package_parts = current_module.split(".")[:-1]
    if node.level > len(package_parts) + 1:
        return None
    if node.level == 1:
        base_parts = package_parts
    else:
        base_parts = package_parts[: -(node.level - 1)]
    if node.module:
        base_parts = [*base_parts, *node.module.split(".")]
    return ".".join(part for part in base_parts if part)


def _resolve_known_module(raw_import: str, known_modules: set[str]) -> str | None:
    parts = raw_import.split(".")
    for length in range(len(parts), 0, -1):
        candidate = ".".join(parts[:length])
        if candidate in known_modules:
            return candidate
    return None


def _strongly_connected_components(graph: Mapping[str, Iterable[str]]) -> tuple[tuple[str, ...], ...]:
    index = 0
    stack: list[str] = []
    indices: dict[str, int] = {}
    lowlinks: dict[str, int] = {}
    on_stack: set[str] = set()
    components: list[tuple[str, ...]] = []

    def visit(node: str) -> None:
        nonlocal index
        indices[node] = index
        lowlinks[node] = index
        index += 1
        stack.append(node)
        on_stack.add(node)

        for target in graph.get(node, ()):
            if target not in indices:
                visit(target)
                lowlinks[node] = min(lowlinks[node], lowlinks[target])
            elif target in on_stack:
                lowlinks[node] = min(lowlinks[node], indices[target])

        if lowlinks[node] == indices[node]:
            component: list[str] = []
            while True:
                target = stack.pop()
                on_stack.remove(target)
                component.append(target)
                if target == node:
                    break
            if len(component) > 1:
                components.append(tuple(sorted(component)))

    for node in sorted(graph):
        if node not in indices:
            visit(node)
    return tuple(components)


def _build_findings(
    graph: Mapping[str, Iterable[str]],
    cycles: tuple[tuple[str, ...], ...],
    god_files: Mapping[str, int],
    line_counts: Mapping[str, int],
    god_file_threshold: int,
) -> list[ArchitectureQualityFinding]:
    findings: list[ArchitectureQualityFinding] = []
    for cycle in cycles[:10]:
        findings.append(
            ArchitectureQualityFinding(
                severity="high",
                code="import_cycle",
                modules=cycle,
                message=f"Import cycle across {len(cycle)} modules",
                value=len(cycle),
            )
        )
    for path, count in sorted(god_files.items(), key=lambda item: (-item[1], item[0]))[:20]:
        findings.append(
            ArchitectureQualityFinding(
                severity="medium",
                code="oversized_module",
                path=path,
                message=f"{path} has {count} lines (threshold {god_file_threshold})",
                value=count,
            )
        )
    for module, targets in sorted(graph.items(), key=lambda item: (-len(tuple(item[1])), item[0]))[:10]:
        degree = len(tuple(targets))
        if degree >= 40:
            findings.append(
                ArchitectureQualityFinding(
                    severity="medium",
                    code="dependency_fanout",
                    path=_path_from_module(module, line_counts),
                    modules=(module,),
                    message=f"{module} imports {degree} local modules",
                    value=degree,
                )
            )
    return findings


def _quality_score(metrics: ArchitectureQualityMetrics) -> float:
    penalty = 0.0
    penalty += min(30.0, metrics.cycle_count * 2.0)
    penalty += min(20.0, metrics.god_file_count * 0.75)
    penalty += min(12.0, max(0, metrics.max_out_degree - 35) * 0.2)
    penalty += min(12.0, max(0, metrics.max_in_degree - 45) * 0.15)
    penalty += min(10.0, max(0.0, metrics.dependency_concentration_pct - 30.0) * 0.2)
    return max(0.0, 100.0 - penalty)


def _dependency_concentration(out_degrees: list[int]) -> float:
    total = sum(out_degrees)
    if total <= 0:
        return 0.0
    top_n = max(1, len(out_degrees) // 20)
    return (sum(sorted(out_degrees, reverse=True)[:top_n]) / total) * 100.0


def _module_name_from_path(rel_path: str) -> str:
    module = rel_path.removesuffix(".py").replace("/", ".")
    return module.removesuffix(".__init__")


def _build_module_maps(paths: Mapping[str, Path]) -> tuple[dict[str, str], dict[str, str]]:
    module_to_path: dict[str, str] = {}
    path_to_module: dict[str, str] = {}
    for rel_path in sorted(paths):
        module = _module_name_from_path(rel_path)
        if module in module_to_path:
            existing_path = module_to_path[module]
            if rel_path.endswith("/__init__.py"):
                file_module = f"{module}.__file__"
                module_to_path[file_module] = existing_path
                path_to_module[existing_path] = file_module
                module_to_path[module] = rel_path
                path_to_module[rel_path] = module
                continue
            module = f"{module}.__file__"
        module_to_path[module] = rel_path
        path_to_module[rel_path] = module
    return module_to_path, path_to_module


def _path_from_module(module: str, line_counts: Mapping[str, int]) -> str | None:
    candidate = f"{module.replace('.', '/')}.py"
    if candidate in line_counts:
        return candidate
    package_candidate = f"{module.replace('.', '/')}/__init__.py"
    if package_candidate in line_counts:
        return package_candidate
    return None


def _line_count(content: str) -> int:
    if not content:
        return 0
    return content.count("\n") + (0 if content.endswith("\n") else 1)


def _is_included(rel_path: str, include_roots: tuple[str, ...]) -> bool:
    return any(rel_path == root or rel_path.startswith(f"{root}/") for root in include_roots)


def _normalize_rel_path(path: str | Path) -> str:
    return str(path).replace("\\", "/").lstrip("./")


def _clean_part(part: str | Path) -> str:
    return str(part).replace("\\", "/").strip().strip("/")
