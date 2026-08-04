"""A file named ``test_*.py`` that collects nothing is not evidence.

An external review caught ``tests/test_state_causality.py`` holding helper
definitions and no test functions, and noted it "should not count as evidence
merely because of its filename". Sweeping for the general case found twelve
such files, which between them collect one test.

That is worse than a missing test, because a missing test is visible. These
are counted in "~24,900 tests", show up green, and name subsystems
(causality, telemetry, swarm, hardening) that a reader would reasonably
believe are covered.

The baseline below follows the repo convention for grandfathered debt: it may
only ever shrink. A NEW empty test file fails this gate immediately.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent

#: Files that collected zero tests when this gate was introduced. All twelve
#: have since been given real tests, so the baseline is EMPTY and any new
#: phantom file fails immediately. It may only ever stay this way.
KNOWN_EMPTY: frozenset[str] = frozenset()


def _looks_empty(path: pathlib.Path) -> bool:
    """Cheap structural check: no test function and no test class."""
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(("def test_", "async def test_")):
            return False
        if stripped.startswith("class Test"):
            return False
    return True


def _empty_test_files() -> set[str]:
    found: set[str] = set()
    for path in sorted((REPO / "tests").glob("test_*.py")):
        if _looks_empty(path):
            found.add(str(path.relative_to(REPO)))
    return found


def test_no_new_phantom_test_files():
    """A new test_*.py with no tests must fail rather than pad the count."""
    empty = _empty_test_files()
    new = empty - KNOWN_EMPTY
    assert not new, (
        "These files are named like tests but collect nothing, so they count "
        f"as evidence purely by filename: {sorted(new)}"
    )


def test_baseline_only_shrinks():
    """Entries must be removed from KNOWN_EMPTY as they gain real tests."""
    empty = _empty_test_files()
    stale = KNOWN_EMPTY - empty
    assert not stale, (
        "These files now have tests — remove them from KNOWN_EMPTY so the "
        f"baseline keeps shrinking: {sorted(stale)}"
    )


def test_the_reviewed_file_now_has_real_tests():
    """The specific finding that prompted this gate, closed."""
    assert "tests/test_state_causality.py" not in _empty_test_files()


def test_the_baseline_is_empty():
    """Twelve files were converted. Nothing may be grandfathered back in."""
    assert KNOWN_EMPTY == frozenset()
