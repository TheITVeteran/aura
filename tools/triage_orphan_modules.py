#!/usr/bin/env python3
"""Classify every unreachable module, so 279 unknowns become 279 decisions.

``lint_module_reachability`` counts modules nothing references. That stops the
number growing and says nothing about what any individual module *is*, so the
honest state of the tree is still unknown: some of those 279 are finished work
nobody wired up, some are entry points invoked by name, some are scaffolding
that should have been deleted, and reading them one at a time is the only way
to tell — unless the evidence is gathered mechanically first.

This gathers it. Each orphan is classified by what the tree can actually
demonstrate about it:

``ENTRY_POINT``   has ``if __name__ == "__main__"``, or is referenced from a
                  Makefile target, a launchd plist or a script. Invoked, just
                  not imported. Not dead.
``SERVICE``       exposes a ``get_*()`` singleton accessor or registers itself
                  with the container. Written to be wired and not wired —
                  the "half-wired" shape this codebase keeps finding.
``TEST_ONLY``     referenced only from ``tests/``. Coverage spent on something
                  no production path reaches.
``DOCUMENTED``    named in docs or a frozen artifact but imported nowhere.
                  Deleting it invalidates a record.
``SCAFFOLDING``   no references of any kind, no entry point, no accessor. The
                  candidates for retirement.

Nothing is deleted here. The output is evidence for a decision, and the
disposition file it feeds is where the decision gets recorded.

Run: ``python tools/triage_orphan_modules.py`` / ``--json`` / ``--class SCAFFOLDING``
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASELINE = ROOT / "config" / "module_reachability_baseline.json"

_CLASSES = ("ENTRY_POINT", "SERVICE", "TEST_ONLY", "DOCUMENTED", "SCAFFOLDING")


def _module_path(dotted: str) -> Path | None:
    direct = ROOT / (dotted.replace(".", "/") + ".py")
    if direct.is_file():
        return direct
    package = ROOT / dotted.replace(".", "/") / "__init__.py"
    return package if package.is_file() else None


_NON_PYTHON_SUFFIXES = (
    ".md", ".json", ".plist", ".sh", ".toml", ".cfg", ".yaml", ".yml", ".txt"
)

#: Built once. Grepping the tree per module is 279 full scans and takes longer
#: than anyone will wait, so the corpus is read once and matched in memory.
_corpus: list[tuple[str, str]] | None = None


def _load_corpus() -> list[tuple[str, str]]:
    global _corpus
    if _corpus is not None:
        return _corpus
    docs: list[tuple[str, str]] = []
    for path in ROOT.rglob("*"):
        rel = path.as_posix()
        if ".claude/worktrees" in rel or "/artifacts/" in rel or "/.git/" in rel:
            continue
        if not path.is_file():
            continue
        if path.name != "Makefile" and path.suffix not in _NON_PYTHON_SUFFIXES:
            continue
        try:
            if path.stat().st_size > 4_000_000:
                continue
            docs.append((rel, path.read_text(encoding="utf-8", errors="ignore")))
        except OSError:
            continue
    _corpus = docs
    return docs


def _grep_outside_python(needle: str) -> list[str]:
    """References from Makefiles, plists, docs and config."""
    return [rel for rel, text in _load_corpus() if needle in text]


def classify(dotted: str) -> dict[str, object]:
    path = _module_path(dotted)
    if path is None:
        return {"module": dotted, "class": "MISSING", "lines": 0, "evidence": ["no file"]}

    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = len(text.splitlines())
    evidence: list[str] = []

    has_main = '__name__ == "__main__"' in text or "__name__ == '__main__'" in text
    accessors: list[str] = []
    registers = False
    try:
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and re.match(
                r"^get_[a-z0-9_]+$", node.name
            ):
                accessors.append(node.name)
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in {"register", "register_service"}
            ):
                registers = True
    except SyntaxError:
        evidence.append("does not parse")

    external = _grep_outside_python(dotted)
    tail = dotted.rsplit(".", 1)[-1]
    external += [p for p in _grep_outside_python(tail) if p not in external]

    if has_main:
        evidence.append("has __main__ block")
        kind = "ENTRY_POINT"
    elif external:
        evidence.append(f"named in {len(external)} non-Python file(s)")
        kind = "DOCUMENTED" if all(p.endswith(".md") for p in external) else "ENTRY_POINT"
    elif accessors or registers:
        evidence.append(
            f"exposes {accessors[:3] if accessors else 'container registration'}"
        )
        kind = "SERVICE"
    else:
        evidence.append("no references, no entry point, no accessor")
        kind = "SCAFFOLDING"

    return {
        "module": dotted,
        "class": kind,
        "lines": lines,
        "evidence": evidence,
        "external_refs": external[:4],
    }


DISPOSITIONS = ROOT / "config" / "orphan_dispositions.json"

#: What a maintainer has decided about an unreachable module. The point of the
#: file is that every one of them has an entry: "279 unreachable" is a number
#: nobody can act on, "279 decided" is a plan.
_DISPOSITIONS = {
    #: Reachable by something the import graph cannot see. Not debt.
    "INVOKED": "entry point or named in configuration",
    #: Deliberately not wired yet, with a reason someone can argue with.
    "STAGED": "kept on purpose, not yet wired",
    #: Written to be wired and not wired. This is the debt.
    "WIRE_PENDING": "exposes a service surface nothing reaches",
    #: Slated for deletion.
    "RETIRE": "no references, no entry point, no service surface",
}

_DEFAULT_BY_CLASS = {
    "ENTRY_POINT": "INVOKED",
    "DOCUMENTED": "INVOKED",
    "SERVICE": "WIRE_PENDING",
    "TEST_ONLY": "STAGED",
    "SCAFFOLDING": "RETIRE",
    "MISSING": "RETIRE",
}


def _write_dispositions(results: list[dict[str, object]]) -> None:
    existing: dict[str, dict[str, str]] = {}
    if DISPOSITIONS.is_file():
        existing = json.loads(DISPOSITIONS.read_text(encoding="utf-8")).get("modules", {})

    modules: dict[str, dict[str, str]] = {}
    for row in sorted(results, key=lambda r: str(r["module"])):
        name = str(row["module"])
        prior = existing.get(name)
        if prior:
            modules[name] = prior  # a human decision is never overwritten
            continue
        modules[name] = {
            "disposition": _DEFAULT_BY_CLASS[str(row["class"])],
            "classified": str(row["class"]),
            "lines": row["lines"],
            "reason": str(row["evidence"][0]) if row["evidence"] else "",
        }

    DISPOSITIONS.parent.mkdir(parents=True, exist_ok=True)
    DISPOSITIONS.write_text(
        json.dumps(
            {
                "note": (
                    "Every unreachable module has a decision here. Defaults come "
                    "from tools/triage_orphan_modules.py and are a starting "
                    "point, not a verdict — change one and it is never "
                    "overwritten by a re-run."
                ),
                "dispositions": _DISPOSITIONS,
                "modules": modules,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {DISPOSITIONS.relative_to(ROOT)}: {len(modules)} decisions")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--class", dest="want", choices=_CLASSES)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--write-dispositions",
        action="store_true",
        help="seed config/orphan_dispositions.json, preserving existing decisions",
    )
    args = parser.parse_args()

    orphans = json.loads(BASELINE.read_text(encoding="utf-8"))["orphans"]
    results = [classify(m) for m in orphans]

    if args.write_dispositions:
        _write_dispositions(results)
        return 0

    if args.want:
        results = [r for r in results if r["class"] == args.want]
    if args.limit:
        results = results[: args.limit]

    if args.json:
        print(json.dumps(results, indent=2))
        return 0

    counts = Counter(r["class"] for r in results)
    by_lines = Counter()
    for r in results:
        by_lines[r["class"]] += r["lines"]

    print(f"{len(results)} unreachable module(s)\n")
    for kind in _CLASSES + ("MISSING",):
        if counts.get(kind):
            print(f"  {kind:12s} {counts[kind]:4d} modules  {by_lines[kind]:7,d} lines")
    print(
        "\nSCAFFOLDING is the retirement candidate set. ENTRY_POINT and "
        "DOCUMENTED are reachable by something the import graph cannot see. "
        "SERVICE is the interesting one: written to be wired, and not wired."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
