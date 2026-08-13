#!/usr/bin/env python3
"""Find and score the extractable seams in an oversized function.

Cutting `_api_chat_turn` down took one afternoon of analysis and about four
minutes of editing, and almost all of the analysis was mechanical: which names
does this block read from the enclosing scope, which does it hand back, how
many early returns, does it await, is anything read before it is stored. That
analysis is the same for all 75 tracked functions, so it belongs in a tool
rather than in whoever next opens the file.

What makes a seam safe, in the order that matters:

* **Early returns.** Zero or one is a clean extraction — one needs a single
  optional-response sentinel. Several means general control-flow surgery and
  the seam is not worth taking.
* **Narrow interface.** Few names in, few out. A block that reads twenty locals
  is not a unit, it is a paragraph.
* **Conditionally-bound escapes.** A name bound only inside the block and read
  after it is the trap: returning a default converts a path that raised
  UnboundLocalError into one that quietly proceeds, which is a behaviour change
  wearing a refactor's clothes. The extraction has to hand those back through a
  sentinel, and this tool counts them so nobody discovers it afterwards.
* **Size.** A seam that removes 40 lines from a 4,000-line function is not
  worth the interface it costs.

The scoring is deliberately conservative: seams with multiple returns are
reported and marked unsafe rather than hidden, because "this function has no
clean seam" is a real and useful answer.

Run:
    python tools/find_extraction_seam.py interface/routes/chat.py::_api_chat_turn
    python tools/find_extraction_seam.py --tracked        # every baselined function
"""

from __future__ import annotations

import argparse
import ast
import builtins
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASELINE = ROOT / "config" / "method_size_baseline.json"

_BUILTINS = set(dir(builtins))


@dataclass
class Seam:
    """One candidate block to lift out, with the contract it would need."""

    lineno: int
    end_lineno: int
    kind: str
    reads: list[str] = field(default_factory=list)
    escapes: list[str] = field(default_factory=list)
    conditional_escapes: list[str] = field(default_factory=list)
    returns: list[int] = field(default_factory=list)
    awaits: int = 0
    yields: int = 0

    @property
    def lines(self) -> int:
        return self.end_lineno - self.lineno + 1

    @property
    def safe(self) -> bool:
        return (
            len(self.returns) <= 1
            and self.yields == 0
            and len(self.reads) <= 10
            and len(self.escapes) <= 10
        )

    @property
    def blockers(self) -> list[str]:
        out = []
        if len(self.returns) > 1:
            out.append(
                f"{len(self.returns)} early returns — needs control-flow surgery, "
                "not a sentinel"
            )
        if self.yields:
            out.append("contains yield; the block is a generator body")
        if len(self.reads) > 10:
            out.append(f"reads {len(self.reads)} enclosing names — not a unit")
        if len(self.escapes) > 10:
            out.append(f"hands back {len(self.escapes)} names — not a unit")
        return out

    def to_dict(self) -> dict[str, object]:
        return {
            "lines": self.lines,
            "range": f"{self.lineno}-{self.end_lineno}",
            "kind": self.kind,
            "reads": self.reads,
            "escapes": self.escapes,
            "conditional_escapes": self.conditional_escapes,
            "returns": self.returns,
            "awaits": self.awaits,
            "safe": self.safe,
            "blockers": self.blockers,
        }


def _names(node: ast.AST, ctx: type) -> set[str]:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ctx)}


def _bound_in(node: ast.AST) -> set[str]:
    bound = _names(node, ast.Store)
    for sub in ast.walk(node):
        if isinstance(sub, (ast.Import, ast.ImportFrom)):
            for alias in sub.names:
                bound.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(sub, ast.ExceptHandler) and sub.name:
            bound.add(sub.name)
    return bound


def _module_scope(tree: ast.Module) -> set[str]:
    scope = {
        n.name
        for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                scope.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, ast.Assign):
            scope.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            scope.add(node.target.id)
    return scope


def analyse(path: Path, function: str, *, min_lines: int = 60) -> list[Seam]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    fn = next(
        (
            n
            for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            and n.name == function.split(".")[-1]
        ),
        None,
    )
    if fn is None:
        raise SystemExit(f"no function named {function!r} in {path}")

    module_scope = _module_scope(tree)
    seams: list[Seam] = []

    for stmt in fn.body:
        end = getattr(stmt, "end_lineno", stmt.lineno) or stmt.lineno
        if end - stmt.lineno + 1 < min_lines:
            continue

        bound = _bound_in(stmt)
        reads = _names(stmt, ast.Load) - bound - module_scope - _BUILTINS

        after: set[str] = set()
        for other in fn.body:
            if other.lineno > end:
                after |= _names(other, ast.Load)

        # Names bound before the seam are carried through; names bound only
        # inside it and read afterwards are the sentinel cases.
        before: set[str] = set()
        for other in fn.body:
            if other.lineno < stmt.lineno:
                before |= _bound_in(other)

        escapes = sorted(bound & after)
        conditional = sorted(n for n in escapes if n not in before)

        seams.append(
            Seam(
                lineno=stmt.lineno,
                end_lineno=end,
                kind=type(stmt).__name__,
                reads=sorted(reads),
                escapes=escapes,
                conditional_escapes=conditional,
                returns=[
                    n.lineno for n in ast.walk(stmt) if isinstance(n, ast.Return)
                ],
                awaits=sum(1 for n in ast.walk(stmt) if isinstance(n, ast.Await)),
                yields=sum(
                    1 for n in ast.walk(stmt) if isinstance(n, (ast.Yield, ast.YieldFrom))
                ),
            )
        )

    seams.sort(key=lambda s: (not s.safe, -s.lines))
    return seams


def _tracked() -> list[tuple[Path, str]]:
    data = json.loads(BASELINE.read_text(encoding="utf-8"))
    out = []
    for key in data.get("functions", {}):
        rel, _, func = key.partition("::")
        out.append((ROOT / rel, func))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", nargs="?", help="path/to/file.py::function_name")
    parser.add_argument("--tracked", action="store_true", help="every baselined function")
    parser.add_argument("--min-lines", type=int, default=60)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.tracked:
        targets = _tracked()
    elif args.target:
        rel, _, func = args.target.partition("::")
        targets = [(ROOT / rel, func)]
    else:
        parser.error("give a target or --tracked")

    report: dict[str, object] = {}
    total_safe = 0
    for path, func in targets:
        if not path.is_file():
            continue
        try:
            seams = analyse(path, func, min_lines=args.min_lines)
        except (SyntaxError, SystemExit):
            continue
        safe = [s for s in seams if s.safe]
        total_safe += len(safe)
        key = f"{path.relative_to(ROOT)}::{func}"
        report[key] = [s.to_dict() for s in seams]

        if not args.json:
            if not seams:
                continue
            print(f"\n{key}")
            for seam in seams[:4]:
                mark = "CUT " if seam.safe else "skip"
                print(
                    f"  [{mark}] {seam.lines:5d} lines  {seam.range if False else f'{seam.lineno}-{seam.end_lineno}':>13}"
                    f"  in={len(seam.reads):2d} out={len(seam.escapes):2d}"
                    f" sentinel={len(seam.conditional_escapes):2d}"
                    f" returns={len(seam.returns)} awaits={seam.awaits}"
                )
                for blocker in seam.blockers:
                    print(f"           blocked: {blocker}")

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"\n{total_safe} safe seam(s) across {len(report)} function(s)")
        print(
            "A safe seam still needs the sentinel treatment for its "
            "conditional escapes: returning a default for a name that was "
            "bound conditionally turns a path that raised into one that "
            "quietly proceeds."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
