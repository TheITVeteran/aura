"""What is built, tested, and called by nobody.

The recurring failure this exists to catch: a module that is written, internally
coherent, thoroughly tested — and not on the production path. Found by hand at
least six times (arbitration, checkpoint promotion, trace propagation, recovery
paths, health reporting, `held_out_challenge`), which means finding it by hand
is not working. The architecture is larger than anyone's ability to re-verify
end-to-end by reading, so the check has to be mechanical.

It is the same disease as an untested guard, from the other side:

    a protection nothing TESTS and a protection nothing CALLS
    are both protections that do not exist.

The difference is that an untested guard fails silently under attack, while an
uncalled one never runs at all — and a passing test suite actively conceals it,
because the tests are green and the coverage looks real.

What counts as debt here: a public symbol defined under `core/` whose only
references are its own defining module and files under `tests/`. Tests are
deliberately NOT evidence of integration. They are the thing that makes this
class invisible.

Deliberate exclusions, each because it would be a false positive rather than
because it is inconvenient:

* dunder and private (`_`-prefixed) names — not part of any contract;
* entry points invoked by name rather than by import: pytest fixtures, CLI
  `main`, framework hooks, anything decorated with a registrar (`@invariant`,
  `@register*`, `@app.*`, `@pytest.*`), since a registry call IS the wiring;
* symbols re-exported through `__all__` of a package `__init__` — the package
  is the seam, and the consumer may be external;
* `core/verify` invariants, which are called by the verifier by discovery.

The output is a ranked list, worst first: a symbol with many tests and zero
callers is more suspicious than one with neither, because someone invested in
proving it works and never connected it.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

#: Decorators that ARE the wiring. A symbol handed to a registry is reached by
#: that registry, not by an import, so absence of callers proves nothing.
REGISTRAR_MARKERS = (
    "invariant", "register", "app.", "router.", "pytest.", "fixture",
    "hookimpl", "command", "cli.", "task", "celery", "channel", "declare",
    "subscribe", "on_event", "handler", "route",
)

#: Names reached by convention rather than by reference.
CONVENTION_NAMES = {"main", "setup", "teardown", "run", "app", "cli"}

#: Symbols that EXIST for tests. Being called only by tests is their contract,
#: not their debt — a reset hook or a simulated observer that production code
#: called would be the bug. Without this filter the report is dominated by them
#: and the real findings are unreadable, which is how a check stops being used.
TEST_ONLY_PATTERNS = (
    re.compile(r"^reset_"), re.compile(r"_for_test$"), re.compile(r"^clear_.*_cache$"),
    re.compile(r"^clear_shutdown_request$"), re.compile(r"^Simulated"),
    re.compile(r"^Fake"), re.compile(r"^Mock"), re.compile(r"^Stub"),
    re.compile(r"^Recording"), re.compile(r"_snapshot_for_test$"),
)


def is_test_only_by_design(name: str) -> bool:
    return any(p.search(name) for p in TEST_ONLY_PATTERNS)

SKIP_DIRS = ("archive/", "core/verify/")


@dataclass
class Symbol:
    module: str
    name: str
    kind: str
    line: int
    test_refs: int
    prod_refs: int

    @property
    def is_debt(self) -> bool:
        return self.prod_refs == 0

    def to_dict(self) -> dict:
        return {
            "module": self.module, "name": self.name, "kind": self.kind,
            "line": self.line, "test_refs": self.test_refs,
            "prod_refs": self.prod_refs,
        }


def _has_registrar(node: ast.AST) -> bool:
    for dec in getattr(node, "decorator_list", []):
        text = ast.unparse(dec).lower()
        if any(marker in text for marker in REGISTRAR_MARKERS):
            return True
    return False


def public_symbols(path: Path) -> list[tuple[str, str, int]]:
    try:
        tree = ast.parse(path.read_text(errors="ignore"))
    except SyntaxError:
        return []
    # A package __init__ re-exports for consumers that may be external.
    if path.name == "__init__.py":
        return []
    out = []
    for node in tree.body:  # top level only; nested defs are implementation
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            name = node.name
            if name.startswith("_") or name in CONVENTION_NAMES:
                continue
            if is_test_only_by_design(name):
                continue
            if _has_registrar(node):
                continue
            kind = "class" if isinstance(node, ast.ClassDef) else "function"
            out.append((name, kind, node.lineno))
    return out


def build_reference_index() -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """One pass over the tree, not one grep per symbol.

    The naive shape — grep each symbol across the repo — is O(symbols x files)
    and did not finish in ten minutes on core/runtime alone. Tokenizing every
    file once and counting identifier occurrences is O(files) and takes
    seconds, which is the difference between a check that runs in CI and a
    check nobody runs.
    """
    prod: dict[str, set[str]] = {}
    tests: dict[str, set[str]] = {}
    roots = ("core", "interface", "tools", "scripts", "tests")
    for root in roots:
        base = REPO / root
        if not base.is_dir():
            continue
        for path in base.rglob("*.py"):
            rel = str(path.relative_to(REPO))
            if any(rel.startswith(d) for d in SKIP_DIRS):
                continue
            try:
                text = path.read_text(errors="ignore")
            except OSError:
                continue
            target = tests if rel.startswith("tests/") else prod
            for ident in set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text)):
                target.setdefault(ident, set()).add(rel)
    return prod, tests


def scan(roots: list[str]) -> list[Symbol]:
    prod_index, test_index = build_reference_index()
    found: list[Symbol] = []
    for root in roots:
        for path in sorted(Path(REPO / root).rglob("*.py")):
            rel = str(path.relative_to(REPO))
            if any(rel.startswith(d) for d in SKIP_DIRS):
                continue
            for name, kind, line in public_symbols(path):
                # A module referencing its own symbol is not integration.
                prod_refs = len(prod_index.get(name, set()) - {rel})
                test_refs = len(test_index.get(name, set()) - {rel})
                found.append(Symbol(rel, name, kind, line, test_refs, prod_refs))
    return found


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="*", default=["core"])
    ap.add_argument("--out", default="")
    ap.add_argument("--min-tests", type=int, default=1,
                    help="only report debt with at least this many test refs")
    args = ap.parse_args()

    symbols = scan(args.roots)
    debt = [s for s in symbols if s.is_debt and s.test_refs >= args.min_tests]
    # Worst first: heavily tested and never called is the sharpest signal —
    # someone proved it works and never connected it.
    debt.sort(key=lambda s: (-s.test_refs, s.module, s.name))

    print(f"scanned {len(symbols)} public symbols under {', '.join(args.roots)}")
    print(f"INTEGRATION DEBT: {len(debt)} tested but never called from production\n")
    for s in debt[:60]:
        print(f"  {s.test_refs:3d} test file(s), 0 callers   "
              f"{s.module}:{s.line}  {s.kind} {s.name}")
    if args.out:
        Path(args.out).write_text(json.dumps([s.to_dict() for s in debt], indent=2))
    print(f"\n(total {len(debt)}; full list written to {args.out or '<not saved>'})")


if __name__ == "__main__":
    main()
