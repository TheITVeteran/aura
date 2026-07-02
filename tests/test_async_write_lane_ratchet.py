"""Ratchet: no NEW blocking gateway/atomic writes inside async functions.

Every one of the 12 recorded live loop-wedge crashes came from a synchronous
fsync running on the event loop. The async write lane exists for this
(`FileWriteGateway.*_async`, `core.runtime.atomic_writer.async_atomic_*`);
this test freezes the historical offenders and fails when a new one appears,
or when a fixed one lingers in the allowlist.

If this test fails on code you just wrote: use the *_async gateway methods or
async_atomic_* writers instead of calling the sync writers from async code.
If you just FIXED an entry, delete it from the allowlist below.
"""
from __future__ import annotations

import ast
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

_SYNC_WRITE_CALLS = {
    "atomic_write_bytes",
    "atomic_write_text",
    "atomic_write_json",
    "atomic_append_text",
    "write_text",
    "write_bytes",
    "append_text",
}

# (file, async function, callee) — historical debt only; DRAINED TO EMPTY
# on 2026-07-02 (66 call sites across 42 files converted to the async lane).
# Any entry appearing here again is a regression, not debt.
ALLOWED_LEGACY_OFFENDERS: frozenset[tuple[str, str, str]] = frozenset()


def _scan_offenders() -> set[tuple[str, str, str]]:
    offenders: set[tuple[str, str, str]] = set()
    for path in (PROJECT_ROOT / "core").rglob("*.py"):
        if "__pycache__" in str(path):
            continue
        try:
            tree = ast.parse(path.read_text())
        except (SyntaxError, UnicodeDecodeError, OSError):
            continue
        rel = str(path.relative_to(PROJECT_ROOT))

        class Visitor(ast.NodeVisitor):
            def __init__(self) -> None:
                self.async_stack: list[str] = []

            def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
                self.async_stack.append(node.name)
                self.generic_visit(node)
                self.async_stack.pop()

            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                # A sync def nested in an async def runs wherever it is
                # called from (thread executors included) — not our target.
                saved = self.async_stack
                self.async_stack = []
                self.generic_visit(node)
                self.async_stack = saved

            def visit_Lambda(self, node: ast.Lambda) -> None:
                # Same rule as nested sync defs: a lambda handed to
                # asyncio.to_thread / an executor runs off-loop.
                saved = self.async_stack
                self.async_stack = []
                self.generic_visit(node)
                self.async_stack = saved

            def visit_Call(self, node: ast.Call) -> None:
                if self.async_stack:
                    if isinstance(node.func, ast.Name):
                        name = node.func.id
                    elif isinstance(node.func, ast.Attribute):
                        name = node.func.attr
                    else:
                        name = ""
                    if name in _SYNC_WRITE_CALLS:
                        offenders.add((rel, self.async_stack[-1], name))
                self.generic_visit(node)

        Visitor().visit(tree)
    return offenders


def test_no_new_blocking_writes_in_async_functions():
    offenders = _scan_offenders()
    new = offenders - ALLOWED_LEGACY_OFFENDERS
    assert not new, (
        "NEW blocking write(s) inside async functions — this exact pattern "
        "froze the live event loop for 20 minutes and crash-cycled the "
        "runtime. Use FileWriteGateway.*_async or async_atomic_* instead:\n"
        + "\n".join(f"  {f}:{fn}() -> {call}" for f, fn, call in sorted(new))
    )


def test_allowlist_contains_no_fixed_entries():
    offenders = _scan_offenders()
    stale = ALLOWED_LEGACY_OFFENDERS - offenders
    assert not stale, (
        "These allowlist entries are fixed — delete them from "
        "ALLOWED_LEGACY_OFFENDERS so the ratchet only tightens:\n"
        + "\n".join(f"  {f}:{fn}() -> {call}" for f, fn, call in sorted(stale))
    )


def test_hot_paths_stay_converted():
    """The per-turn lanes must never regress to on-loop fsyncs."""
    offenders = _scan_offenders()
    hot_files = {
        "core/memory/memory_write_gateway.py",
        "core/memory/episode_store.py",
        "core/state/state_gateway.py",
        "core/self_model.py",
    }
    regressions = {o for o in offenders if o[0] in hot_files}
    assert not regressions, f"hot path regressed to blocking writes: {sorted(regressions)}"
