"""Regression gate for architecture-quality metrics."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .scorer import ArchitectureQualityReport, score_codebase


@dataclass(frozen=True)
class ArchitectureQualityPolicy:
    """Allowed architecture drift for one candidate change."""

    allowed_score_drop: float = 1.0
    max_new_dependency_edges: int = 20
    max_line_growth_for_large_file: int = 120
    block_new_cycles: bool = True
    block_new_god_files: bool = True
    block_growth_of_existing_large_files: bool = True


@dataclass(frozen=True)
class ArchitectureQualityResult:
    """Pass/fail result for one architecture-quality comparison."""

    passed: bool
    reasons: tuple[str, ...]
    before: ArchitectureQualityReport
    after: ArchitectureQualityReport
    changed_paths: tuple[str, ...] = field(default_factory=tuple)

    def summary(self) -> str:
        if self.passed:
            return f"architecture quality passed ({self.after.summary()})"
        return "architecture quality failed: " + "; ".join(self.reasons)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "reasons": list(self.reasons),
            "changed_paths": list(self.changed_paths),
            "before": self.before.to_dict(),
            "after": self.after.to_dict(),
        }


class ArchitectureQualityGate:
    """Compares current and candidate architecture-quality reports."""

    def __init__(
        self,
        root: str | Path,
        *,
        include_roots: Iterable[str] = ("core", "interface", "infrastructure", "slo", "tools"),
        policy: ArchitectureQualityPolicy | None = None,
        god_file_threshold: int = 1500,
    ) -> None:
        self.root = Path(root)
        self.include_roots = tuple(include_roots)
        self.policy = policy or ArchitectureQualityPolicy()
        self.god_file_threshold = god_file_threshold

    def score_current(self) -> ArchitectureQualityReport:
        return score_codebase(
            self.root,
            include_roots=self.include_roots,
            god_file_threshold=self.god_file_threshold,
        )

    def evaluate_overlay(
        self,
        overlay_content: Mapping[str, str],
        *,
        changed_paths: Iterable[str],
    ) -> ArchitectureQualityResult:
        before = self.score_current()
        after = score_codebase(
            self.root,
            include_roots=self.include_roots,
            overlay_content=overlay_content,
            god_file_threshold=self.god_file_threshold,
        )
        return self.evaluate_reports(before, after, changed_paths=changed_paths)

    def evaluate_reports(
        self,
        before: ArchitectureQualityReport,
        after: ArchitectureQualityReport,
        *,
        changed_paths: Iterable[str] = (),
    ) -> ArchitectureQualityResult:
        changed_tuple = tuple(_normalize_path(path) for path in changed_paths)
        reasons: list[str] = []
        policy = self.policy

        if after.score < before.score - policy.allowed_score_drop:
            reasons.append(
                f"quality score dropped from {before.score:.1f} to {after.score:.1f}"
            )

        edge_growth = after.metrics.dependency_edges - before.metrics.dependency_edges
        if edge_growth > policy.max_new_dependency_edges:
            reasons.append(
                f"dependency edges grew by {edge_growth} "
                f"(limit {policy.max_new_dependency_edges})"
            )

        if policy.block_new_cycles:
            new_cycles = _new_cycles(before.cycles, after.cycles)
            changed_modules = _changed_modules(after, changed_tuple)
            relevant_new_cycles = [
                cycle for cycle in new_cycles if not changed_modules or set(cycle) & changed_modules
            ]
            if relevant_new_cycles:
                sample = ", ".join(" -> ".join(cycle) for cycle in relevant_new_cycles[:3])
                reasons.append(f"new import cycle(s): {sample}")

        god_growth = after.metrics.god_file_count - before.metrics.god_file_count
        if policy.block_new_god_files and god_growth > 0:
            reasons.append(f"new oversized module count grew by {god_growth}")

        if policy.block_growth_of_existing_large_files:
            for path in changed_tuple:
                before_lines = before.line_counts.get(path, 0)
                after_lines = after.line_counts.get(path, 0)
                if after_lines <= self.god_file_threshold:
                    continue
                line_growth = after_lines - before_lines
                crossed_threshold = before_lines <= self.god_file_threshold < after_lines
                grew_too_much = line_growth > policy.max_line_growth_for_large_file
                if crossed_threshold or grew_too_much:
                    reasons.append(
                        f"{path} grew to {after_lines} lines "
                        f"(+{line_growth}; threshold {self.god_file_threshold})"
                    )

        return ArchitectureQualityResult(
            passed=not reasons,
            reasons=tuple(reasons),
            before=before,
            after=after,
            changed_paths=changed_tuple,
        )


def _new_cycles(
    before_cycles: tuple[tuple[str, ...], ...],
    after_cycles: tuple[tuple[str, ...], ...],
) -> list[tuple[str, ...]]:
    before = {tuple(sorted(cycle)) for cycle in before_cycles}
    return [cycle for cycle in after_cycles if tuple(sorted(cycle)) not in before]


def _changed_modules(report: ArchitectureQualityReport, changed_paths: tuple[str, ...]) -> set[str]:
    modules: set[str] = set()
    for path in changed_paths:
        module = report.changed_module_for_path(path)
        if module:
            modules.add(module)
    return modules


def _normalize_path(path: str | Path) -> str:
    return str(path).replace("\\", "/").lstrip("./")
