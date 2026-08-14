#!/usr/bin/env python3
"""Flag the writing patterns that read as machine-generated.

The rules live in docs/WRITING_RULES.md; this file is the executable half.
Adding a pattern there without adding it here leaves the rule unenforced,
which is how the last set of writing conventions quietly stopped applying.

Scope is deliberate. Append-only ledgers and dated records are NOT checked:
they are the record, and editing a July entry in August falsifies it. See
docs/DOC_STATUS.md for which documents are which.

    python tools/lint_ai_writing.py                 # the front-facing docs
    python tools/lint_ai_writing.py --all           # every guide
    python tools/lint_ai_writing.py FILE [FILE...]  # specific files
    python tools/lint_ai_writing.py --baseline      # write the ratchet file
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASELINE = ROOT / "config" / "ai_writing_baseline.json"
#: Baselines are per-scope; an --all run is not comparable to a front-docs run.
SCOPES = ("front", "all", "adhoc")

#: Documents that are records rather than prose we maintain.
EXCLUDE_PREFIX = (
    "docs/AURA_EXECUTION_TRACKER.md",
    "docs/RLC_SPARK_EXECUTION_LEDGER.md",
    "docs/AURA_EXECUTION_PLAN.md",
    "docs/AURA_PROMPT_COVERAGE_AUDIT.md",
    "docs/evidence/",
    "docs/runbooks/",
    "scoping/",
    "aura_bench/",
    "archive/",
    "scratch/",
    "dev_archive/",
    "models/",
    "artifacts/",
    "research/",
    "specs/",
    "tests/",
    "security/",
    "proof_kernel/",
    "training/",
    "demos/",
    "challenges/",
    ".github/",
    "data/",
    "docs/FMEA.md",
    "docs/RUNTIME_CONTRACT.md",
    "docs/ARCHITECTURE_MAP.md",
    "docs/AURA_PROGRESS.md",
)
EXCLUDE_DATED = re.compile(r"_20\d\d[_-]\d\d[_-]\d\d\.md$|_2026_\d\d\.md$|RESULTS\.md$")

#: The pages a visitor actually reads.
FRONT = (
    "README.md",
    "HOW_IT_WORKS.md",
    "ARCHITECTURE.md",
    "CONTRIBUTING.md",
    "INSTALL.md",
    "TESTING.md",
    "MODEL_CARD.md",
    "CHANGELOG.md",
    "AGENTS.md",
    "docs/README.md",
    "docs/RECURSIVE_LATENT_CORTEX.md",
    "docs/INTRINSIC_RECURRENCE.md",
    "docs/COGNITIVE_ARCHITECTURE_ADOPTION.md",
    "docs/MODEL_ROSTER.md",
    "docs/ENGINEERING_ADOPTION.md",
    "docs/USER_GUIDE.md",
    "docs/OPERATOR_GUIDE.md",
)

Rule = tuple[str, re.Pattern[str], str]

RULES: list[Rule] = [
    (
        "negation-flip",
        re.compile(
            r"(?:That's|That is|This is|It's|It is)\s+not\s+[^.!?\n]{1,60}[.!?]\s+"
            r"(?:That's|That is|This is|It's|It is)\s",
        ),
        '"That\'s not X. That\'s Y." Say the second half only.',
    ),
    (
        "stapled-fragments",
        re.compile(r"(?<=[.!?:]\s)([A-Z][a-z]{2,12})\.\s([A-Z][a-z]{2,12})\.\s(?=[A-Z])"),
        "Two one-word sentences in a row. Pick one and write it as a sentence.",
    ),
    (
        "twin-images",
        re.compile(r"\bless\s+(?:an?\s+)?\w+[^.\n]{0,30}?,?\s+more\s+(?:an?\s+)?\w+", re.I),
        '"Less a hammer, more a scalpel." Say what to do instead.',
    ),
    (
        "self-applause",
        re.compile(
            r"\b(?:and that matters"
            r"|that'?s? (?:the part|what) (?:everyone|most people|nobody) (?:miss|gets?)"
            r"|which is exactly the point|that'?s (?:exactly )?the (?:whole )?point"
            r"|is the important part|worth reading|the part worth"
            r"|which is the whole point|and that'?s the thing"
            r"|cannot be overstated|it'?s worth stating)\b",
            re.I,
        ),
        "Clapping for itself. Delete it; you lose nothing.",
    ),
    (
        "borrowed-analogy",
        re.compile(r"\bit'?s the [A-Z][A-Za-z0-9-]+ of\b", re.I),
        '"It\'s the Excel of X." Only works if the reader knows both things.',
    ),
    (
        "throat-clearing",
        re.compile(
            r"(?:^|(?<=[.!?]\s)|(?<=\n))\s*(?:Here'?s the thing|Let me be clear"
            r"|The truth is|The reality is|The thing is|Here'?s what'?s"
            r"|What'?s (?:interesting|striking) (?:here )?is|Make no mistake)\b",
            re.I,
        ),
        "Warming up. Start one sentence later.",
    ),
    (
        "hedged-range",
        re.compile(
            r"\b\d+\s*(?:to|–|—|-)\s*\d+\s*"
            r"(?:seconds?|minutes?|hours?|days?|weeks?|months?|s\b|ms\b)",
            re.I,
        ),
        "A range means it was never measured. Give the number.",
    ),
    (
        "recap-ending",
        re.compile(
            r"\b(?:In short|At the end of the day|To sum up|In summary"
            r"|In conclusion|The bottom line|The upshot is|All told)\b",
            re.I,
        ),
        "The ending that repeats the post. Just stop typing.",
    ),
]


def prose_lines(text: str) -> list[tuple[int, str]]:
    """Lines that are prose: no code fences, tables, indented blocks, or links-only."""
    out: list[tuple[int, str]] = []
    fence = False
    for i, line in enumerate(text.splitlines(), 1):
        stripped = line.lstrip()
        if stripped.startswith("```"):
            fence = not fence
            continue
        if fence:
            continue
        if stripped.startswith(("|", ">", "#")) or line.startswith(("    ", "\t")):
            continue
        out.append((i, line))
    return out


def scan_file(rel: str) -> list[tuple[int, str, str]]:
    path = ROOT / rel
    if not path.exists():
        return []
    lines = prose_lines(path.read_text(encoding="utf-8", errors="replace"))
    # Join so patterns can span a hard-wrapped sentence.
    offsets: list[tuple[int, int]] = []
    pos = 0
    parts: list[str] = []
    for lineno, line in lines:
        offsets.append((pos, lineno))
        parts.append(line)
        pos += len(line) + 1
    blob = "\n".join(parts)

    def lineof(idx: int) -> int:
        best = offsets[0][1] if offsets else 0
        for start, lineno in offsets:
            if start <= idx:
                best = lineno
            else:
                break
        return best

    hits: list[tuple[int, str, str]] = []
    for name, rx, _ in RULES:
        for m in rx.finditer(blob):
            frag = " ".join(m.group(0).split())[:80]
            hits.append((lineof(m.start()), name, frag))
    return sorted(hits)


def tracked_guides() -> list[str]:
    listing = subprocess.run(
        ["git", "ls-files", "*.md"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.split()
    return [
        rel
        for rel in listing
        if not rel.startswith(EXCLUDE_PREFIX) and not EXCLUDE_DATED.search(rel)
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*")
    ap.add_argument("--all", action="store_true", help="every tracked guide")
    ap.add_argument("--baseline", action="store_true", help="rewrite the ratchet")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    scope = "adhoc" if args.files else ("all" if args.all else "front")
    targets = args.files or (tracked_guides() if args.all else list(FRONT))
    results = {rel: scan_file(rel) for rel in targets}
    results = {k: v for k, v in results.items() if v}
    total = sum(len(v) for v in results.values())

    explain = {name: why for name, _, why in RULES}
    if not args.quiet:
        for rel in sorted(results):
            print(f"\n{rel}")
            for lineno, name, frag in results[rel]:
                print(f"  {lineno:>5}  {name:<18} {frag}")
                print(f"         {'':<18} -> {explain[name]}")

    if args.baseline:
        BASELINE.parent.mkdir(parents=True, exist_ok=True)
        data = json.loads(BASELINE.read_text()) if BASELINE.exists() else {}
        data[scope] = {"total": total,
                       "per_file": {k: len(v) for k, v in results.items()}}
        BASELINE.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
        print(f"\nbaseline written for scope '{scope}': {total} findings")
        return 0

    print(f"\n{total} findings across {len(targets)} documents")

    if BASELINE.exists() and scope != "adhoc":
        stored = json.loads(BASELINE.read_text())
        if scope not in stored:
            print(f"no baseline for scope '{scope}'; run --baseline to record one")
            return 0
        prior = stored[scope]["total"]
        if total > prior:
            print(f"FAIL: {total} > baseline {prior}. The ratchet only goes down.")
            return 1
        if total < prior:
            print(f"OK: {total} < baseline {prior}. Re-run with --baseline to tighten.")
        else:
            print(f"OK: at baseline {prior}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
