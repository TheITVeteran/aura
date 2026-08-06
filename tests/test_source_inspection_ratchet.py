"""Ratchet: a test that reads SOURCE STRINGS is not a test of behaviour.

    source = inspect.getsource(MLXLocalClient._run_warmup_precompile)
    assert "campaign_deadline - time.monotonic()" in source[probe_at:]

That assertion was green while the warmup exceeded its documented hard
deadline twice over — once because the probe took `max(10.0, remaining)` and
once because the retry's gc + reboot + settle ran entirely outside the budget.
The string was there. The contract was not. Replacing it with three tests that
MEASURE elapsed time against the budget found the second leak in the first
run.

This is the failure mode the CHOP series names "Tests Passing ≠ Features
Working": an agent-written test that validates the shape of the code it was
written beside, and keeps passing when the behaviour is deleted around the
string it looks for.

Source inspection is not always wrong — "this module must not import X",
"this handler must not call a blocking writer", "the docstring must state the
measurement" are honest uses, and some contracts genuinely have no runtime
surface. But 520 of them across 175 files is a large body of tests that cannot
fail for the reason they exist.

So: the count is frozen per file and MAY ONLY SHRINK. Replace an entry with a
test that exercises the contract, then lower the number in
``config/source_inspection_baseline.json``.
"""

from __future__ import annotations

import ast
import json
from collections import Counter
from functools import lru_cache
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASELINE_PATH = PROJECT_ROOT / "config" / "source_inspection_baseline.json"

_SOURCE_READERS = {"getsource", "getsourcelines"}


def _count_source_inspections(path: Path) -> int:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError, OSError):
        return 0
    total = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr in _SOURCE_READERS:
            total += 1
        elif isinstance(func, ast.Name) and func.id in _SOURCE_READERS:
            total += 1
    return total


@lru_cache(maxsize=1)
def _scan() -> Counter:
    """Parse every test module once; four assertions read the same result."""
    counts: Counter = Counter()
    for path in sorted((PROJECT_ROOT / "tests").rglob("*.py")):
        if "__pycache__" in str(path):
            continue
        found = _count_source_inspections(path)
        if found:
            counts[str(path.relative_to(PROJECT_ROOT))] = found
    return counts


@pytest.fixture(scope="module")
def baseline() -> dict:
    assert BASELINE_PATH.exists(), f"{BASELINE_PATH} is tracked in git"
    return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))


def test_no_file_grows_its_source_inspection_count(baseline):
    recorded = baseline["counts"]
    current = _scan()
    grown = {
        path: (recorded.get(path, 0), count)
        for path, count in current.items()
        if count > recorded.get(path, 0)
    }
    assert not grown, (
        "new source-string assertions — write a test that exercises the "
        "contract instead, or record why inspection is the only surface:\n  "
        + "\n  ".join(
            f"{path}: {was} -> {now}" for path, (was, now) in sorted(grown.items())
        )
    )


def test_the_total_only_shrinks(baseline):
    current = sum(_scan().values())
    recorded = int(baseline["total"])
    assert current <= recorded, (
        f"source-inspection assertions grew from {recorded} to {current}"
    )


def test_a_drained_file_is_removed_from_the_baseline(baseline):
    """A stale entry hides room the ratchet has already earned."""
    recorded = baseline["counts"]
    current = _scan()
    stale = sorted(path for path in recorded if path not in current)
    assert not stale, (
        "these no longer inspect source; delete them from "
        f"{BASELINE_PATH.name}:\n  " + "\n  ".join(stale)
    )


def test_the_baseline_total_matches_its_own_counts(baseline):
    assert int(baseline["total"]) == sum(int(v) for v in baseline["counts"].values())


def test_this_ratchet_does_not_inspect_source_itself():
    """It parses ASTs; it must never become an instance of what it counts.

    The first draft of this file ended with an ``inspect.getsource`` assertion
    about the warmup tests — and failed its own ratchet on the first run, which
    is the shortest possible demonstration of how easily these accumulate.
    """
    assert _scan().get("tests/test_source_inspection_ratchet.py", 0) == 0
