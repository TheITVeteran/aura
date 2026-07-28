"""How much of the locking in this codebase can lockdep actually see?

``core/runtime/lockdep.py`` finds ABBA deadlocks without the deadlock having
to happen — but only for locks it wraps. CLAUDE.md says so plainly: "it only
sees locks it wraps." That sentence is a coverage claim with no number
behind it, and an ABBA detector covering an unknown fraction of the system
gives an assurance nobody can size.

This measures it. Every ``threading.Lock()``, ``threading.RLock()``,
``asyncio.Lock()`` and friend constructed in ``core/`` and ``interface/`` is
invisible to lockdep unless it is wrapped by ``checked_lock`` /
``checked_async_lock``, or its critical section is adopted with
``instrument(name)``.

The honest first reading is 721 raw constructions. That is not a number to
fix in one pass — migrating a lock carelessly causes exactly the deadlock
the migration is meant to prevent, and 721 careless migrations would be a
catastrophe. So this exists to do two things a mass edit cannot:

* publish the coverage fraction, so "lockdep protects us" stops being an
  unquantified claim and becomes a number that can be driven up;
* support a ratchet, so the debt stops growing while it is paid down.

Reports rather than judges: a module-private lock guarding one dict is a
very different risk from a lock held across an await in the inference path,
and this cannot tell them apart. The caller decides.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

#: Constructors that create a lock primitive lockdep does not know about.
_RAW_LOCK_ATTRS = frozenset({
    "Lock", "RLock", "Condition", "Semaphore", "BoundedSemaphore",
})
_RAW_LOCK_MODULES = frozenset({"threading", "asyncio"})

#: The wrapped lane. Constructing through these IS lockdep coverage.
_CHECKED_CONSTRUCTORS = frozenset({
    "checked_lock", "checked_async_lock", "instrument",
})

_DEFAULT_ROOTS = ("core", "interface")

_SKIP_PARTS = frozenset({
    "__pycache__", ".venv", "node_modules", ".git", "build", "dist",
    ".claude", "artifacts", "tests",
})


@dataclass(frozen=True)
class RawLock:
    """One lock primitive lockdep cannot see."""

    path: str
    line: int
    kind: str
    function: str = ""

    def key(self) -> str:
        """Identity for a baseline entry, insensitive to line drift."""
        return f"{self.path}::{self.function}::{self.kind}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "line": self.line,
            "kind": self.kind,
            "function": self.function,
        }


@dataclass
class LockdepCoverageReport:
    raw: list[RawLock] = field(default_factory=list)
    checked: int = 0

    @property
    def keys(self) -> set[str]:
        return {lock.key() for lock in self.raw}

    @property
    def total(self) -> int:
        return len(self.raw) + self.checked

    @property
    def coverage(self) -> float:
        """Fraction of lock primitives lockdep can reason about.

        Zero total reports 0.0 rather than 1.0: a codebase with no locks
        found has not achieved full coverage, it has failed to measure.
        """
        return (self.checked / self.total) if self.total else 0.0

    def new_since(self, baseline: Iterable[str]) -> list[RawLock]:
        known = set(baseline)
        return sorted(
            (lock for lock in self.raw if lock.key() not in known),
            key=lambda lock: (lock.path, lock.line),
        )

    def fixed_since(self, baseline: Iterable[str]) -> list[str]:
        """Baseline entries that are gone — the half that makes it a ratchet.

        A baseline that only grows is a list of excuses; entries that have
        been migrated must leave it or the next regression hides behind a
        stale allowance.
        """
        return sorted(set(baseline) - self.keys)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "aura.lockdep_coverage.v1",
            "raw": len(self.raw),
            "checked": self.checked,
            "total": self.total,
            "coverage": round(self.coverage, 4),
        }


class _LockVisitor(ast.NodeVisitor):
    def __init__(self, rel_path: str) -> None:
        self.rel_path = rel_path
        self.raw: list[RawLock] = []
        self.checked = 0
        self._scope: list[str] = []

    def _enter(self, node: Any) -> None:
        self._scope.append(node.name)
        self.generic_visit(node)
        self._scope.pop()

    visit_FunctionDef = _enter          # noqa: N815 - ast API casing
    visit_AsyncFunctionDef = _enter     # noqa: N815
    visit_ClassDef = _enter             # noqa: N815

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802 - ast API
        func = node.func
        if isinstance(func, ast.Attribute):
            module = getattr(func.value, "id", "")
            if module in _RAW_LOCK_MODULES and func.attr in _RAW_LOCK_ATTRS:
                self.raw.append(
                    RawLock(
                        path=self.rel_path,
                        line=node.lineno,
                        kind=f"{module}.{func.attr}",
                        function=".".join(self._scope),
                    )
                )
        elif isinstance(func, ast.Name) and func.id in _CHECKED_CONSTRUCTORS:
            self.checked += 1
        self.generic_visit(node)


def scan_lock_coverage(
    *,
    roots: Iterable[str] = _DEFAULT_ROOTS,
    repo_root: Path | None = None,
) -> LockdepCoverageReport:
    """Measure how many lock primitives lockdep can see."""
    root = repo_root or Path(__file__).resolve().parent.parent.parent
    report = LockdepCoverageReport()
    for name in roots:
        base = root / name
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.py")):
            if _SKIP_PARTS.intersection(path.parts):
                continue
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except (OSError, SyntaxError, UnicodeDecodeError):
                continue
            visitor = _LockVisitor(str(path.relative_to(root)))
            visitor.visit(tree)
            report.raw.extend(visitor.raw)
            report.checked += visitor.checked
    report.raw.sort(key=lambda lock: (lock.path, lock.line))
    return report


__all__ = [
    "LockdepCoverageReport",
    "RawLock",
    "scan_lock_coverage",
]
