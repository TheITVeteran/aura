#!/usr/bin/env python3
"""Modules nothing reaches, counted and held to a falling ceiling.

292 modules under ``core/`` — 27,896 lines — are imported by nothing in the
repository. That is not a style complaint. When a large fraction of the tree is
unreachable, "is X wired?" stops having an answer you can trust, and this
codebase has already been bitten by exactly that: a second affect engine with
no construction path, a fallback that could never be reached, a vision flag with
three different defaults in four files. Dead weight is where half-wired things
hide.

Deleting 292 modules in one pass is the wrong move and this tool does not
propose it. Some are entry points invoked by name, some are loaded dynamically
through ``importlib`` or a service registry, and some are staged work. What can
be established mechanically is which ones nothing references *at all* — by
import or by name — and that the number only falls.

Reachability here means either:

* a static import from anywhere in the repo outside ``archive/``, or
* the dotted module path appearing in a string literal — how ``importlib``,
  plugin registries and config-driven factories reach a module. A module
  referenced only that way is reachable, and calling it dead would be wrong.

Run: ``python tools/lint_module_reachability.py`` / ``--write-baseline`` /
``--list``
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASELINE = ROOT / "config" / "module_reachability_baseline.json"

#: Directories whose imports do not count as reachability — they are copies.
_EXCLUDED_PREFIXES = ("archive/", ".claude/", "build/", "dist/")


def _module_name(path: Path) -> str:
    parts = list(path.relative_to(ROOT).with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _iter_sources() -> list[Path]:
    out = []
    for path in ROOT.rglob("*.py"):
        rel = path.relative_to(ROOT).as_posix()
        if "__pycache__" in rel or rel.startswith(_EXCLUDED_PREFIXES):
            continue
        out.append(path)
    return sorted(out)


def scan() -> dict[str, object]:
    sources = _iter_sources()
    core_modules = {
        _module_name(p): p
        for p in sources
        if p.relative_to(ROOT).as_posix().startswith("core/")
    }

    referenced: set[str] = set()
    importers: dict[str, set[str]] = defaultdict(set)

    for path in sources:
        try:
            text = path.read_text(encoding="utf-8")
            tree = ast.parse(text)
        except (SyntaxError, UnicodeDecodeError, OSError):
            continue
        me = _module_name(path)

        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                names = [node.module] + [
                    f"{node.module}.{a.name}" for a in node.names
                ]
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                # importlib.import_module("core.x.y"), registry tables, config
                # strings. A module reached this way is reached.
                candidate = node.value.strip()
                if candidate.startswith("core.") and " " not in candidate:
                    names = [candidate]
            for name in names:
                if name in core_modules and name != me:
                    referenced.add(name)
                    importers[name].add(me)

    orphans = sorted(set(core_modules) - referenced)
    lines = 0
    for name in orphans:
        try:
            lines += len(
                core_modules[name].read_text(encoding="utf-8", errors="ignore").splitlines()
            )
        except OSError:
            continue

    return {
        "core_modules": len(core_modules),
        "orphans": orphans,
        "orphan_count": len(orphans),
        "orphan_lines": lines,
    }


def _load_baseline() -> dict[str, object] | None:
    if not BASELINE.is_file():
        return None
    return json.loads(BASELINE.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-baseline", action="store_true")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    report = scan()
    orphans: list[str] = report["orphans"]  # type: ignore[assignment]

    if args.list:
        for name in orphans:
            print(f"  {name}")
        print()

    if args.write_baseline:
        BASELINE.parent.mkdir(parents=True, exist_ok=True)
        BASELINE.write_text(
            json.dumps(
                {
                    "orphan_count": report["orphan_count"],
                    "orphan_lines": report["orphan_lines"],
                    "orphans": orphans,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(
            f"wrote baseline: {report['orphan_count']} orphans, "
            f"{report['orphan_lines']:,} lines"
        )
        return 0

    baseline = _load_baseline()
    print(
        f"🕸  {report['orphan_count']} of {report['core_modules']} core modules are "
        f"reached by nothing ({report['orphan_lines']:,} lines)"
    )

    if baseline is None:
        print("no baseline yet — run with --write-baseline")
        return 0

    known = set(baseline.get("orphans", []))
    new = sorted(set(orphans) - known)
    closed = sorted(known - set(orphans))

    if closed:
        print(f"✅ {len(closed)} newly reachable (baseline should shrink): {closed[:8]}")

    if new:
        print(f"\n❌ {len(new)} module(s) became unreachable:")
        for name in new:
            print(f"   {name}")
        print(
            "\nEither wire it to something, or retire it. A module nothing reaches "
            "is where a half-wired subsystem hides — this repo has shipped a second "
            "affect engine and an unreachable fallback exactly this way."
        )
        return 1

    if len(orphans) > int(baseline.get("orphan_count", 0)):
        print("❌ orphan count rose without a new module name — refresh the baseline")
        return 1

    print("✅ no new unreachable modules")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
