"""Deadline-bounded, cached repository traversal for code-health analysis."""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.self_modification.code_health_rules import (
    analyze_python_file,
    issue,
    issue_sort_key,
)

_SKIP_DIRS = {
    ".claude",
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    ".zenflow",
    "__pycache__",
    "archive",
    "artifacts",
    "build",
    "dev_archive",
    "dist",
    "htmlcov",
    "llm_data",
    "models",
    "node_modules",
    "scratch",
    "site-packages",
    "venv",
}
_MAX_INVENTORY_FILES = 20_000


@dataclass
class _ScanState:
    target: Path
    cycle_id: int = 1
    candidates: list[Path] = field(default_factory=list)
    candidate_keys: set[str] = field(default_factory=set)
    pending_dirs: list[Path] = field(default_factory=list)
    cursor: int = 0
    inventory_complete: bool = False
    inventory_truncated: bool = False
    needs_refresh: bool = False
    fingerprints: dict[str, tuple[int, int, int, int]] = field(default_factory=dict)
    issues_by_file: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    inventory_errors: set[str] = field(default_factory=set)
    scan_errors: dict[str, str] = field(default_factory=dict)


class IncrementalCodeHealthScanner:
    """Advance a complete repository review through small, resumable batches."""

    def __init__(self, root_path: Path) -> None:
        self.root_path = root_path.expanduser().resolve()
        self._states: dict[str, _ScanState] = {}
        self._state_lock = threading.RLock()

    def resolve_target(self, path_str: str) -> Path:
        raw = Path(str(path_str or ".")).expanduser()
        target = raw.resolve() if raw.is_absolute() else (self.root_path / raw).resolve()
        try:
            target.relative_to(self.root_path)
        except ValueError as exc:
            raise ValueError("auto_refactor target must stay inside the Aura repository") from exc
        if not target.exists():
            raise ValueError(f"auto_refactor target does not exist: {target}")
        if target.is_file() and target.suffix != ".py":
            raise ValueError("auto_refactor file target must be a Python source file")
        if not target.is_file() and not target.is_dir():
            raise ValueError(f"auto_refactor target is not a file or directory: {target}")
        return target

    def scan(
        self,
        path_str: str,
        *,
        max_files: int = 100,
        time_budget_s: float = 1.5,
    ) -> dict[str, Any]:
        started = time.monotonic()
        budget_s = max(0.25, min(10.0, float(time_budget_s)))
        file_budget = max(1, min(2_000, int(max_files)))
        deadline = started + budget_s
        target = self.resolve_target(path_str)

        with self._state_lock:
            state = self._state_for(target)
            self._advance_inventory(
                state,
                deadline=min(deadline, started + max(0.05, budget_s * 0.30)),
                max_directories=max(128, file_budget),
            )
            examined, parsed, cache_hits = self._scan_candidates(
                state,
                deadline=deadline,
                file_budget=file_budget,
            )
            cycle_complete = state.inventory_complete and state.cursor >= len(state.candidates)
            if cycle_complete:
                self._prune_deleted_candidates(state)

            issues = [
                finding
                for key in state.candidate_keys
                for finding in state.issues_by_file.get(key, ())
            ]
            issues.sort(key=issue_sort_key)
            durable_errors = sorted(state.inventory_errors | set(state.scan_errors.values()))
            elapsed_ms = round((time.monotonic() - started) * 1000.0, 1)
            coverage_complete = bool(
                cycle_complete and not state.inventory_truncated and not durable_errors
            )
            coverage_percent = (
                round(100.0 * state.cursor / max(1, len(state.candidates)), 2)
                if state.inventory_complete
                else None
            )
            coverage = {
                "cycle_id": state.cycle_id,
                "coverage_complete": coverage_complete,
                "cycle_complete": cycle_complete,
                "inventory_complete": state.inventory_complete,
                "inventory_truncated": state.inventory_truncated,
                "candidate_files_discovered": len(state.candidates),
                "files_completed_this_cycle": state.cursor,
                "coverage_percent": coverage_percent,
                "files_examined_this_batch": examined,
                "files_parsed_this_batch": parsed,
                "cache_hits_this_batch": cache_hits,
                "file_budget": file_budget,
                "time_budget_s": budget_s,
                "elapsed_ms": elapsed_ms,
                "deadline_reached": time.monotonic() >= deadline and not cycle_complete,
                "batch_limit_reached": examined >= file_budget and not cycle_complete,
            }
            if cycle_complete:
                state.needs_refresh = True

        return {
            "target": str(target),
            "display_target": self.display_path(target),
            "issues": issues,
            "coverage": coverage,
            "scan_errors": durable_errors[:20],
        }

    def _scan_candidates(
        self,
        state: _ScanState,
        *,
        deadline: float,
        file_budget: int,
    ) -> tuple[int, int, int]:
        examined = 0
        parsed = 0
        cache_hits = 0
        while examined < file_budget and time.monotonic() < deadline:
            if state.cursor >= len(state.candidates):
                if state.inventory_complete:
                    break
                self._advance_inventory(state, deadline=deadline, max_directories=32)
                if state.cursor >= len(state.candidates) and not state.inventory_complete:
                    break
                continue

            file_path = state.candidates[state.cursor]
            state.cursor += 1
            examined += 1
            key = str(file_path)
            try:
                stat = file_path.stat()
                fingerprint = (
                    int(stat.st_dev),
                    int(stat.st_ino),
                    int(stat.st_mtime_ns),
                    int(stat.st_size),
                )
                if state.fingerprints.get(key) == fingerprint:
                    cache_hits += 1
                    state.scan_errors.pop(key, None)
                    continue
                state.issues_by_file[key] = self._analyze_python_file(file_path, stat)
                state.fingerprints[key] = fingerprint
                state.scan_errors.pop(key, None)
                parsed += 1
            except OSError as exc:
                state.fingerprints.pop(key, None)
                error = f"scan:{self.display_path(file_path)}:{type(exc).__name__}"
                state.scan_errors[key] = error
                state.issues_by_file[key] = [
                    issue(
                        file_path,
                        display_path=self.display_path,
                        line=0,
                        rule_id="PY-SCAN-IO",
                        severity="error",
                        issue_type="scan_io",
                        confidence=1.0,
                        message=f"Source could not be read: {type(exc).__name__}: {exc}",
                        remediation="Restore readable source access and rerun the scan cycle.",
                    )
                ]
        return examined, parsed, cache_hits

    def _analyze_python_file(
        self,
        file_path: Path,
        stat: os.stat_result,
    ) -> list[dict[str, Any]]:
        return analyze_python_file(file_path, stat, display_path=self.display_path)

    def _advance_inventory(
        self,
        state: _ScanState,
        *,
        deadline: float,
        max_directories: int,
    ) -> None:
        processed = 0
        while (
            state.pending_dirs
            and processed < max_directories
            and time.monotonic() < deadline
        ):
            current = state.pending_dirs.pop()
            processed += 1
            try:
                with os.scandir(current) as iterator:
                    entries = sorted(iterator, key=lambda entry: entry.name)
            except OSError as exc:
                state.inventory_errors.add(
                    f"inventory:{self.display_path(current)}:{type(exc).__name__}"
                )
                continue

            child_dirs: list[Path] = []
            for entry in entries:
                try:
                    if entry.is_dir(follow_symlinks=False):
                        if not self._skip_directory(entry.name):
                            child_dirs.append(Path(entry.path))
                        continue
                    if not entry.is_file(follow_symlinks=False) or not entry.name.endswith(".py"):
                        continue
                except OSError:
                    continue
                key = str(Path(entry.path).resolve())
                if key in state.candidate_keys:
                    continue
                if len(state.candidates) >= _MAX_INVENTORY_FILES:
                    state.inventory_truncated = True
                    state.pending_dirs.clear()
                    break
                state.candidate_keys.add(key)
                state.candidates.append(Path(key))
            if not state.inventory_truncated:
                state.pending_dirs.extend(reversed(child_dirs))
        state.inventory_complete = not state.pending_dirs

    def _state_for(self, target: Path) -> _ScanState:
        key = str(target)
        state = self._states.get(key)
        if state is None:
            if len(self._states) >= 32:
                self._states.pop(next(iter(self._states)))
            state = _ScanState(target=target)
            state.pending_dirs = [] if target.is_file() else [target]
            if target.is_file():
                state.candidates.append(target)
                state.candidate_keys.add(key)
                state.inventory_complete = True
            self._states[key] = state
        elif state.needs_refresh:
            self._reset_inventory(state)
        return state

    @staticmethod
    def _reset_inventory(state: _ScanState) -> None:
        state.cycle_id += 1
        state.candidates.clear()
        state.candidate_keys.clear()
        state.pending_dirs = [] if state.target.is_file() else [state.target]
        if state.target.is_file():
            state.candidates.append(state.target)
            state.candidate_keys.add(str(state.target))
        state.cursor = 0
        state.inventory_complete = state.target.is_file()
        state.inventory_truncated = False
        state.inventory_errors.clear()
        state.scan_errors.clear()
        state.needs_refresh = False

    @staticmethod
    def _skip_directory(name: str) -> bool:
        return name in _SKIP_DIRS or name.startswith(".")

    @staticmethod
    def _prune_deleted_candidates(state: _ScanState) -> None:
        live_keys = state.candidate_keys
        for stale_key in set(state.fingerprints) - live_keys:
            state.fingerprints.pop(stale_key, None)
            state.issues_by_file.pop(stale_key, None)
            state.scan_errors.pop(stale_key, None)

    def display_path(self, path: Path) -> str:
        try:
            relative = path.resolve().relative_to(self.root_path)
            return str(relative) or "."
        except (OSError, ValueError):
            return str(path)


__all__ = ["IncrementalCodeHealthScanner"]
