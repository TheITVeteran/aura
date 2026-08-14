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


def _read_before_written(node: ast.AST) -> set[str]:
    """Names the block reads before it first assigns them.

    These are free variables even though the block also binds them, and
    treating any Store as "bound" misses them entirely. It is not a corner
    case: ``AuraKernel.tick`` sets ``state = self.state`` in its preamble and
    the seam both reads ``state`` and rebinds it via ``state.derive(...)``.
    Extracting on the strength of the naive analysis produced a clean
    100%-similarity move that raised ``UnboundLocalError`` on the first tick.

    Line numbers are a sound approximation here because a seam is one
    statement: a read on an earlier line than every write to the same name
    cannot be reached after that write.
    """
    # Comprehension targets bind in their own scope and are written and read
    # on the same line, so a line-based comparison flags every one of them.
    # They are never inputs to the enclosing block.
    comprehension_targets: set[str] = set()
    for sub in ast.walk(node):
        if isinstance(sub, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            for generator in sub.generators:
                comprehension_targets |= _names(generator.target, ast.Store)

    first_write: dict[str, int] = {}
    first_read: dict[str, int] = {}
    for sub in ast.walk(node):
        if not isinstance(sub, ast.Name):
            continue
        line = getattr(sub, "lineno", 0)
        if isinstance(sub.ctx, ast.Store):
            first_write.setdefault(sub.id, line)
            first_write[sub.id] = min(first_write[sub.id], line)
        elif isinstance(sub.ctx, ast.Load):
            first_read.setdefault(sub.id, line)
            first_read[sub.id] = min(first_read[sub.id], line)
    return {
        name
        for name, read_at in first_read.items()
        if name in first_write
        and read_at <= first_write[name]
        and name not in comprehension_targets
    }


def _module_scope(tree: ast.Module) -> set[str]:
    """Names bound at MODULE level only.

    Walking the whole tree was wrong and quietly destructive. ``ast.walk``
    reaches functions nested inside other functions, so a helper defined in the
    body of a large function — ``positive_int``, ``finite_number_list`` and
    four siblings inside LatentCortexService._receipt_contract_errors — was
    treated as globally available and dropped from the seam's free-variable
    set. Extracting on that analysis produced a helper referencing six names
    that do not exist in its new scope; ruff caught it as F821 only after the
    file was written.
    """
    scope = {
        n.name
        for n in tree.body
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
        # A name the block rebinds is still an input if it is read first.
        reads = (
            (_names(stmt, ast.Load) - bound) | _read_before_written(stmt)
        ) - module_scope - _BUILTINS

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


def implementation_source(owner: object, name: str, *, depth: int = 3) -> str:
    """Source of a method plus the bodies it delegates to on ``self``.

    Source-inspection tests couple to function boundaries: assert that
    ``tick``'s source mentions something, extract half of ``tick`` into
    ``_tick_body``, and the assertion fails while the behaviour is identical.
    Three tests broke that way on the first kernel extraction, and every one of
    the remaining 50 seams would break more of them.

    Following ``self.<helper>()`` calls one level at a time keeps the assertion
    about the *implementation* rather than about where its curly braces happen
    to fall, which is what those tests meant in the first place.
    """
    import inspect

    seen: set[str] = set()
    parts: list[str] = []

    def walk(method_name: str, remaining: int) -> None:
        if method_name in seen or remaining < 0:
            return
        seen.add(method_name)
        method = getattr(owner, method_name, None)
        if method is None:
            return
        try:
            source = inspect.getsource(method)
        except (OSError, TypeError):
            return
        parts.append(source)
        try:
            tree = ast.parse(textwrap_dedent(source))
        except SyntaxError:
            return
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "self"
            ):
                walk(node.func.attr, remaining - 1)

    walk(name, depth)
    return "\n".join(parts)


def textwrap_dedent(text: str) -> str:
    import textwrap

    return textwrap.dedent(text)


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
