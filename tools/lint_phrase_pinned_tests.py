#!/usr/bin/env python3
"""Tests that assert on production wording, counted and ratcheted.

Reading production source in a test is not automatically wrong. Three
different things use the same two calls, and only one of them is a defect:

* **structural** — "this call site exists", "this import is present", "the
  kernel reads the shared set rather than its own copy". No behavioural test
  can express those; they are the honest use, and this repository relies on
  them heavily.
* **fixture** — reading a tmp_path file the test itself wrote. Not about
  production source at all.
* **phrase pin** — asserting a sentence, a log line, a user-visible message
  or a docstring clause. This is the defect. It passes while the wording is
  frozen and fails the moment somebody improves it, which means it punishes
  exactly the work it should protect.

Two real cases: a test pinned the literal string "I'm here", which broke
when 718e46091 correctly grounded a recovery claim in current evidence; and
one pinned ``os.environ["AURA_MEDIA_SIDECAR_PROCESS"] = "1"`` in the sensory
client, which broke when child-process spawning was correctly centralised
behind the subprocess gateway. Both improvements. Both tests red.

Converting all of them at once would be a large mechanical change to the
test suite with no behavioural benefit, so this ratchets instead: the count
in ``config/phrase_pinned_test_baseline.json`` may only fall. A new test
that pins a phrase has to earn it by replacing an old one.

Run: ``python tools/lint_phrase_pinned_tests.py`` / ``--write-baseline``
"""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASELINE = ROOT / "config" / "phrase_pinned_test_baseline.json"

#: A read is of production source if it names a package or uses getsource.
_SOURCE_HINTS = ("core/", "interface/", "skills/", "getsource")
#: ...and is a fixture read if it names a temp or artifact location.
_FIXTURE_HINTS = ("tmp_path", "tmp_dir", "tmpdir", "fixture", "artifacts", "/tmp")

#: How far past the read to look for the assertions it feeds.
_WINDOW = 12

_ASSERT_LITERAL = re.compile(r'assert[^\n]*?["\']([^"\']{4,})["\']')


def _is_structural(literal: str) -> bool:
    """Is this literal a fragment of CODE rather than a sentence?

    Parsing is the discriminator, and a much better one than counting words.
    ``from core.container import ServiceContainer`` is a layering invariant
    that no behavioural test can express — it parses. ``Bryan is kin.`` and
    ``I'm here`` are wording — they do not. Word counts got the first of
    those wrong and would have had this tool demanding the removal of the
    assertions that hold the architecture together.
    """
    candidate = literal.strip()
    if not candidate:
        return True
    try:
        ast.parse(candidate)
    except SyntaxError:
        return False
    return True


def _phrase_pins(source: str) -> int:
    """Assertions on production wording in one test file."""
    lines = source.splitlines()
    pins = 0
    for index, line in enumerate(lines):
        if "getsource(" not in line and "read_text(" not in line:
            continue
        if any(hint in line for hint in _FIXTURE_HINTS):
            continue
        if not any(hint in line for hint in _SOURCE_HINTS):
            continue
        window = "\n".join(lines[index : index + _WINDOW])
        for literal in _ASSERT_LITERAL.findall(window):
            if not _is_structural(literal):
                pins += 1
    return pins


def measure() -> dict[str, object]:
    by_file: dict[str, int] = {}
    for path in sorted((ROOT / "tests").rglob("test_*.py")):
        try:
            source = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        pins = _phrase_pins(source)
        if pins:
            by_file[str(path.relative_to(ROOT))] = pins
    return {"total": sum(by_file.values()), "files": len(by_file), "by_file": by_file}


def main(argv: list[str]) -> int:
    current = measure()
    total = int(current["total"])
    print(
        f"tests asserting on production wording: {total} "
        f"across {current['files']} file(s)"
    )

    if "--write-baseline" in argv:
        BASELINE.parent.mkdir(parents=True, exist_ok=True)
        BASELINE.write_text(json.dumps(current, indent=2) + "\n", encoding="utf-8")
        print(f"baseline written: {BASELINE.relative_to(ROOT)}")
        return 0

    if "--list" in argv:
        for name, count in sorted(
            current["by_file"].items(), key=lambda kv: -kv[1]  # type: ignore[union-attr]
        )[:30]:
            print(f"  {count:3d}  {name}")
        return 0

    if not BASELINE.is_file():
        print(f"❌ no baseline at {BASELINE.relative_to(ROOT)}; run --write-baseline")
        return 1

    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    allowed = int(baseline.get("total", 0))
    if total > allowed:
        print(f"❌ phrase-pinned assertions rose: {allowed} -> {total}")
        previous = baseline.get("by_file") or {}
        for name, count in sorted(current["by_file"].items()):  # type: ignore[union-attr]
            was = int(previous.get(name, 0))
            if count > was:
                print(f"    {name}: {was} -> {count}")
        print(
            "\nAssert the behaviour instead of the wording. A test that fails "
            "when someone improves a message is a test that punishes the work "
            "it exists to protect."
        )
        return 1

    if total < allowed:
        print(f"⬇️  phrase-pinned assertions fell: {allowed} -> {total}")
        print("    refresh with: python tools/lint_phrase_pinned_tests.py --write-baseline")
        return 1

    print("✅ phrase-pinned assertions held at baseline")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
