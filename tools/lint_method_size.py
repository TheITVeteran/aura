#!/usr/bin/env python3
"""The functions nobody can reason about, measured and held to a falling ceiling.

Four execution surfaces in this repository are large enough that the usual
tools stop working on them:

    interface/routes/chat.py::_api_chat_turn                   4,635 lines, CC ~731
    core/brain/llm/latent_cortex/engine.py::_latent_episode    4,432 lines, CC ~472
    core/brain/inference_gate.py::InferenceGate.generate       3,130 lines, CC ~630
    core/phases/response_generation_unitary.py::execute        3,084 lines, CC ~609

Cyclomatic complexity in the hundreds is not a style opinion. It is a
statement that the function has more independent paths than any test suite
will cover and more than any reader will hold at once. ``_api_chat_turn``
has 125 return points; 2,732 of this repo's 2,746 core files are smaller
than that one function.

The debt is unpaid, not unnoticed, and paying it down in one pass is the
wrong move: this is the code that serves every conversation, the branches
encode years of live incidents, and a behaviour-preserving rewrite of 4,635
lines cannot be validated by the offline suite alone. A rewrite that
silently drops one of those branches is worse than the size.

So this ratchets. Every listed function may only get smaller. That turns an
unbounded liability into a one-way one and puts the cost on whoever next
touches the function, which is the only time anyone has the context to
split it correctly.

Run: ``python tools/lint_method_size.py`` / ``--write-baseline`` / ``--top``
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASELINE = ROOT / "config" / "method_size_baseline.json"

SCAN_ROOTS = ("core", "interface", "skills")
SKIP_DIR_PARTS = {
    ".git",
    ".venv",
    "__pycache__",
    "archive",
    "node_modules",
    "tests",
}

#: Functions at or above this many lines are tracked. Set so the list is the
#: genuine outliers rather than every long function in the codebase.
TRACK_THRESHOLD_LINES = 400


def _complexity(node: ast.AST) -> int:
    branches = sum(
        1
        for child in ast.walk(node)
        if isinstance(
            child, (ast.If, ast.For, ast.AsyncFor, ast.While, ast.ExceptHandler,
                    ast.IfExp, ast.Assert, ast.With, ast.AsyncWith)
        )
    )
    booleans = sum(
        len(child.values) - 1
        for child in ast.walk(node)
        if isinstance(child, ast.BoolOp)
    )
    comprehensions = sum(
        len(child.generators)
        for child in ast.walk(node)
        if isinstance(child, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp))
    )
    return branches + booleans + comprehensions + 1


def _qualified_names(tree: ast.Module):
    """Yield (qualname, node) for every function, methods included."""

    def walk(node, prefix: str):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                # `yield from`, not a bare call: without it this generator
                # silently dropped every method in the codebase, which is how
                # a size gate comes to report that the largest methods do not
                # exist.
                yield from walk(child, f"{prefix}{child.name}.")
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                yield f"{prefix}{child.name}", child
                # Nested functions are part of the enclosing function's size;
                # counting them separately would let someone "shrink" a giant
                # by nesting more of it.
            else:
                yield from walk(child, prefix)

    yield from walk(tree, "")


def measure() -> dict[str, object]:
    tracked: dict[str, dict[str, int]] = {}
    for top in SCAN_ROOTS:
        base = ROOT / top
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.py")):
            relative = path.relative_to(ROOT)
            if SKIP_DIR_PARTS.intersection(relative.parts):
                continue
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except (OSError, SyntaxError, UnicodeDecodeError):
                continue
            for qualname, node in _qualified_names(tree):
                lines = (node.end_lineno or node.lineno) - node.lineno + 1
                if lines < TRACK_THRESHOLD_LINES:
                    continue
                tracked[f"{relative}::{qualname}"] = {
                    "lines": lines,
                    "complexity": _complexity(node),
                    "returns": sum(
                        1 for c in ast.walk(node) if isinstance(c, ast.Return)
                    ),
                }
    return {
        "threshold_lines": TRACK_THRESHOLD_LINES,
        "tracked": len(tracked),
        "total_lines": sum(v["lines"] for v in tracked.values()),
        "functions": dict(sorted(tracked.items())),
    }


def main(argv: list[str]) -> int:
    current = measure()
    functions: dict[str, dict[str, int]] = current["functions"]  # type: ignore[assignment]
    print(
        f"functions at or over {TRACK_THRESHOLD_LINES} lines: {current['tracked']} "
        f"({current['total_lines']} lines total)"
    )

    if "--top" in argv:
        ranked = sorted(functions.items(), key=lambda kv: -kv[1]["lines"])
        for name, stats in ranked[:20]:
            print(
                f"  {stats['lines']:5d} lines  CC {stats['complexity']:4d}  "
                f"{stats['returns']:3d} returns  {name}"
            )
        return 0

    if "--write-baseline" in argv:
        BASELINE.parent.mkdir(parents=True, exist_ok=True)
        BASELINE.write_text(json.dumps(current, indent=2) + "\n", encoding="utf-8")
        print(f"baseline written: {BASELINE.relative_to(ROOT)}")
        return 0

    if not BASELINE.is_file():
        print(f"❌ no baseline at {BASELINE.relative_to(ROOT)}; run --write-baseline")
        return 1

    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    previous: dict[str, dict[str, int]] = baseline.get("functions") or {}

    grew: list[str] = []
    appeared: list[str] = []
    shrank: list[str] = []
    for name, stats in functions.items():
        was = previous.get(name)
        if was is None:
            appeared.append(
                f"{name}: NEW at {stats['lines']} lines, CC {stats['complexity']}"
            )
        elif stats["lines"] > was["lines"]:
            grew.append(
                f"{name}: {was['lines']} -> {stats['lines']} lines "
                f"(CC {was['complexity']} -> {stats['complexity']})"
            )
        elif stats["lines"] < was["lines"]:
            shrank.append(f"{name}: {was['lines']} -> {stats['lines']} lines")

    for name in previous:
        if name not in functions:
            shrank.append(f"{name}: no longer over the threshold")

    if grew or appeared:
        print("\n❌ tracked functions grew, or a new one crossed the threshold:")
        for line in grew + appeared:
            print(f"    {line}")
        print(
            "\nThese are already past the point where a test suite can cover "
            "their paths or a reader can hold them. They may only shrink — "
            "extract the part you came to change."
        )
        return 1

    if shrank:
        print("\n⬇️  tracked functions shrank:")
        for line in shrank:
            print(f"    {line}")
        print("    refresh with: python tools/lint_method_size.py --write-baseline")
        return 1

    print("✅ no tracked function grew")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
