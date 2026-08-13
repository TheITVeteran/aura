#!/usr/bin/env python3
"""Move a seam out of an oversized function, and prove it was a move.

``find_extraction_seam.py`` says where to cut. This does the cutting, and the
important half is the proof: it diffs the relocated body against the original
and refuses to write if they are not the same code. 400 lines re-entered by
hand is how a branch gets silently dropped, and the method-size gate's own
docstring warns that a rewrite losing one is worse than the size it removed.

What it will not do
-------------------
* More than one early return. That needs real control-flow surgery and a
  sentinel cannot express it.
* A seam whose conditional escapes were not declared. A name bound only inside
  the block and read after it must come back through a sentinel, because
  returning a default converts a path that raised ``UnboundLocalError`` into
  one that quietly proceeds — a behaviour change wearing a refactor's clothes.
* Anything it cannot verify byte-for-byte afterwards.

Run::

    python tools/extract_seam.py core/kernel/aura_kernel.py::AuraKernel.tick \\
        --lines 1119-1596 --name _tick_body --method \\
        --params objective priority turn_origin

Dry by default; pass ``--write`` to apply.
"""

from __future__ import annotations

import argparse
import ast
import difflib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.find_extraction_seam import analyse  # noqa: E402


def _verify_move(original: list[str], relocated: list[str]) -> tuple[float, list[str]]:
    """How much of the moved body is literally the original body."""
    a = [ln.strip() for ln in original if ln.strip()]
    b = [ln.strip() for ln in relocated if ln.strip()]
    ratio = difflib.SequenceMatcher(None, a, b).ratio()
    diff = [
        d
        for d in difflib.unified_diff(a, b, lineterm="", n=0)
        if d.startswith(("+", "-")) and not d.startswith(("+++", "---"))
    ]
    return ratio, diff


def extract(
    path: Path,
    function: str,
    *,
    lo: int,
    hi: int,
    new_name: str,
    params: list[str],
    is_method: bool,
    write: bool,
    min_similarity: float = 0.97,
) -> int:
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    block = lines[lo - 1 : hi]

    seams = analyse(path, function, min_lines=1)
    seam = next((s for s in seams if s.lineno == lo and s.end_lineno == hi), None)
    if seam is None:
        print(f"❌ no statement spans exactly {lo}-{hi}; a seam must be a whole statement")
        return 1
    if len(seam.returns) > 1:
        print(f"❌ {len(seam.returns)} early returns — not a mechanical extraction")
        return 1
    if seam.yields:
        print("❌ the block yields; it is a generator body")
        return 1
    if seam.conditional_escapes:
        print(
            "❌ conditional escapes not handled by this tool: "
            f"{seam.conditional_escapes}. These need an explicit sentinel — "
            "returning a default for a conditionally-bound name changes behaviour."
        )
        return 1
    if seam.escapes:
        print(f"❌ the block hands back {seam.escapes}; only pure-return seams here")
        return 1

    indent = "    " if not is_method else "    "
    signature_params = (["self"] if is_method else []) + list(params)
    decl = "async def" if seam.awaits else "def"

    helper: list[str] = [
        "\n",
        f"{indent if is_method else ''}{decl} {new_name}(",
        ", ".join(signature_params),
        "):\n",
        f'{indent}{indent if is_method else ""}"""Body lifted verbatim out of '
        f'``{function}``.\n\n',
        f"{indent}{indent if is_method else ''}Moved by tools/extract_seam.py, which "
        "refuses to write unless the\n",
        f"{indent}{indent if is_method else ''}relocated body diffs clean against the "
        "original. The seam was\n",
        f"{indent}{indent if is_method else ''}{len(seam.reads)} names in, "
        f"{len(seam.escapes)} out, {len(seam.returns)} early return(s), "
        f"{seam.awaits} awaits.\n",
        f'{indent}{indent if is_method else ""}"""\n',
    ]
    helper.extend(block)

    call_indent = " " * (len(block[0]) - len(block[0].lstrip()))
    await_kw = "await " if seam.awaits else ""
    args = ", ".join(params)
    call = [
        f"{call_indent}return {await_kw}"
        f"{'self.' if is_method else ''}{new_name}({args})\n"
    ]

    ratio, diff = _verify_move(block, helper[len(helper) - len(block) :])
    print(f"seam        : {hi - lo + 1} lines, {lo}-{hi}")
    print(f"contract    : in={seam.reads} out={seam.escapes} returns={seam.returns}")
    print(f"similarity  : {ratio:.4f} ({len(diff)} differing non-blank lines)")

    if ratio < min_similarity:
        print(f"❌ relocated body is only {ratio:.2%} identical; refusing to write")
        for d in diff[:10]:
            print("   ", d[:110])
        return 1

    if not write:
        print("\ndry run — pass --write to apply")
        return 0

    # Insert the helper immediately after the enclosing function so the reader
    # meets it where it was used, then replace the block with the call.
    tree = ast.parse("".join(lines))
    enclosing = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        and n.name == function.split(".")[-1]
    )
    tail = enclosing.end_lineno

    out = lines[: lo - 1] + call + lines[hi:tail] + helper + lines[tail:]
    path.write_text("".join(out), encoding="utf-8")
    print(f"✅ wrote {path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", help="path/to/file.py::Class.function")
    parser.add_argument("--lines", required=True, help="LO-HI, inclusive")
    parser.add_argument("--name", required=True, help="name for the extracted function")
    parser.add_argument("--params", nargs="*", default=[], help="names to pass in")
    parser.add_argument("--method", action="store_true", help="extract as a method")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    rel, _, func = args.target.partition("::")
    lo, _, hi = args.lines.partition("-")
    return extract(
        ROOT / rel,
        func,
        lo=int(lo),
        hi=int(hi),
        new_name=args.name,
        params=args.params,
        is_method=args.method,
        write=args.write,
    )


if __name__ == "__main__":
    raise SystemExit(main())
