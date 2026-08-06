"""Regression gate for architecture-quality metrics."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .models import SCHEMA_VERSION, ArchitectureQualityReport, normalize_repository_path
from .scorer import score_codebase


@dataclass(frozen=True)
class ArchitectureQualityPolicy:
    """Allowed architecture drift for one candidate change."""

    allowed_score_drop: float = 1.0
    max_new_dependency_edges: int = 20
    max_new_executable_dependency_edges: int = 20
    max_line_growth_for_large_file: int = 120
    max_architecture_debt_growth: float = 1.0
    block_new_cycles: bool = True
    block_new_executable_cycles: bool = True
    block_new_god_files: bool = True
    block_growth_of_existing_large_files: bool = True
    require_complete_evidence: bool = True
    block_new_unresolved_local_imports: bool = True


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
        overlay_paths = tuple(
            sorted(normalize_repository_path(path, label="overlay path") for path in overlay_content)
        )
        changed_tuple = tuple(sorted(_normalize_path(path) for path in changed_paths))
        if set(overlay_paths) != set(changed_tuple):
            raise ValueError(
                "changed_paths must exactly match overlay paths "
                f"(overlay={overlay_paths}, changed={changed_tuple})"
            )
        before = self.score_current()
        after = score_codebase(
            self.root,
            include_roots=self.include_roots,
            overlay_content=overlay_content,
            god_file_threshold=self.god_file_threshold,
        )
        return self.evaluate_reports(before, after, changed_paths=overlay_paths)

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
        if before.include_roots != after.include_roots:
            reasons.append(
                f"analysis include roots differ ({before.include_roots!r}->{after.include_roots!r})"
            )
        if before.exclude_parts != after.exclude_parts:
            reasons.append("analysis exclusion scope differs")
        if before.god_file_threshold != after.god_file_threshold:
            reasons.append(
                "god-file threshold differs "
                f"({before.god_file_threshold}->{after.god_file_threshold})"
            )
        if before.schema_version != after.schema_version:
            reasons.append(
                "architecture report schema mismatch "
                f"({before.schema_version}->{after.schema_version}); run a reviewed baseline migration"
            )
        if after.schema_version != SCHEMA_VERSION:
            reasons.append(
                f"unsupported architecture report schema {after.schema_version} "
                f"(expected {SCHEMA_VERSION})"
            )
        if policy.require_complete_evidence and not after.findings_complete:
            reasons.append(
                f"architecture evidence is incomplete ({after.findings_omitted} findings omitted)"
            )
        if after.metrics.parse_error_count:
            syntax_paths = sorted(
                finding.path or "<unknown>"
                for finding in after.findings
                if finding.code in {"syntax_error", "source_decode_error"}
            )
            sample = ", ".join(syntax_paths[:3])
            reasons.append(
                f"candidate contains {after.metrics.parse_error_count} unparseable source module(s): {sample}"
            )
        if (
            policy.block_new_unresolved_local_imports
            and after.metrics.unresolved_local_imports
            > before.metrics.unresolved_local_imports
        ):
            new_missing = after.metrics.unresolved_local_imports - before.metrics.unresolved_local_imports
            reasons.append(f"candidate introduces {new_missing} unresolved local import(s)")

        if after.score < before.score - policy.allowed_score_drop:
            reasons.append(
                f"quality score dropped from {before.score:.1f} to {after.score:.1f}"
            )

        debt_growth = (
            after.metrics.architecture_debt - before.metrics.architecture_debt
        )
        if debt_growth > policy.max_architecture_debt_growth:
            reasons.append(
                f"architecture debt grew by {debt_growth:.2f} "
                f"(limit {policy.max_architecture_debt_growth:.2f})"
            )

        edge_growth = after.metrics.dependency_edges - before.metrics.dependency_edges
        if edge_growth > policy.max_new_dependency_edges:
            reasons.append(
                f"dependency edges grew by {edge_growth} "
                f"(limit {policy.max_new_dependency_edges})"
            )

        executable_edge_growth = (
            after.metrics.executable_dependency_edges
            - before.metrics.executable_dependency_edges
        )
        if executable_edge_growth > policy.max_new_executable_dependency_edges:
            reasons.append(
                f"executable dependency edges grew by {executable_edge_growth} "
                f"(limit {policy.max_new_executable_dependency_edges})"
            )

        if policy.block_new_cycles:
            new_cycles = _new_cycles(before.cycles, after.cycles)
            if new_cycles and not _cycles_are_decomposition(before.cycles, new_cycles):
                changed_modules = _changed_modules(after, changed_tuple)
                relevant_new_cycles = [
                    cycle for cycle in new_cycles
                    if not changed_modules or set(cycle) & changed_modules
                ]
                if relevant_new_cycles:
                    sample = ", ".join(" -> ".join(cycle) for cycle in relevant_new_cycles[:3])
                    reasons.append(f"new import cycle(s): {sample}")
                elif (
                    after.metrics.cycle_count > before.metrics.cycle_count
                    or after.metrics.largest_cycle_size > before.metrics.largest_cycle_size
                ):
                    reasons.append(
                        "import-cycle metrics regressed "
                        f"(count {before.metrics.cycle_count}->{after.metrics.cycle_count}, "
                        f"largest {before.metrics.largest_cycle_size}->{after.metrics.largest_cycle_size})"
                    )

        if policy.block_new_executable_cycles:
            new_executable_cycles = _new_cycles(
                before.executable_cycles,
                after.executable_cycles,
            )
            if new_executable_cycles and not _cycles_are_decomposition(
                before.executable_cycles,
                new_executable_cycles,
            ):
                sample = ", ".join(
                    " -> ".join(cycle) for cycle in new_executable_cycles[:3]
                )
                reasons.append(f"new executable dependency cycle(s): {sample}")

        before_oversized = _finding_paths(before, "structurally_oversized_module")
        after_oversized = _finding_paths(after, "structurally_oversized_module")
        new_oversized = sorted(after_oversized - before_oversized)
        if policy.block_new_god_files and new_oversized:
            reasons.append(
                "new oversized module(s): " + ", ".join(new_oversized[:5])
            )

        ignored_identity_codes = {
            "import_cycle",
            "executable_import_cycle",
            "structurally_oversized_module",
        }
        new_serious = sorted(
            _finding_identities(after, minimum_severity={"critical", "high"})
            - _finding_identities(before, minimum_severity={"critical", "high"})
        )
        new_serious = [item for item in new_serious if item[0] not in ignored_identity_codes]
        if new_serious:
            sample = ", ".join(
                f"{code}:{path or '/'.join(modules)}"
                for code, path, modules in new_serious[:5]
            )
            reasons.append(f"new high-severity architecture finding(s): {sample}")

        if policy.block_growth_of_existing_large_files:
            # An empty `changed_paths` means "no narrowing supplied", NOT
            # "nothing to check". It used to mean the latter by accident, and
            # the accident disabled this entire block in the only place it
            # runs: tools/closeout/architecture_quality_gate.py calls
            # evaluate_reports(baseline, current) with no changed_paths, so
            # changed_tuple was () and the loop below iterated over nothing.
            # A fully-implemented, policy-configurable growth ratchet examined
            # zero files on every CI run, and interface/routes/chat.py reached
            # 24,658 lines under a gate whose default config forbids exactly
            # that.
            #
            # Narrowing is an optimisation for diff-scoped runs. Absent it,
            # the honest scope is every file the candidate report measured.
            scope = changed_tuple or tuple(sorted(after.line_counts))
            for path in scope:
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


def _cycles_are_decomposition(
    before_cycles: tuple[tuple[str, ...], ...],
    new_cycles: Iterable[tuple[str, ...]],
) -> bool:
    before_sets = [set(cycle) for cycle in before_cycles]
    new_sets = [set(cycle) for cycle in new_cycles]
    return bool(new_sets) and all(
        any(new_cycle < before_cycle for before_cycle in before_sets)
        for new_cycle in new_sets
    )


def _finding_paths(report: ArchitectureQualityReport, code: str) -> set[str]:
    return {
        finding.path
        for finding in report.findings
        if finding.code == code and finding.path is not None
    }


def _finding_identities(
    report: ArchitectureQualityReport,
    *,
    minimum_severity: set[str],
) -> set[tuple[str, str, tuple[str, ...]]]:
    return {
        (finding.code, finding.path or "", tuple(sorted(finding.modules)))
        for finding in report.findings
        if finding.severity in minimum_severity
    }


def _changed_modules(report: ArchitectureQualityReport, changed_paths: tuple[str, ...]) -> set[str]:
    modules: set[str] = set()
    for path in changed_paths:
        module = report.changed_module_for_path(path)
        if module:
            modules.add(module)
    return modules


def _normalize_path(path: str | Path) -> str:
    return normalize_repository_path(path, label="changed path")
