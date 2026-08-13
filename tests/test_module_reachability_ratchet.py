"""Modules nothing reaches may only get fewer.

279 of 2,839 modules under core/ — 26,060 lines — are reached by nothing: no
import anywhere in the repo, and no dotted path in a string literal either.

That number is not primarily a tidiness problem. When a large slice of the tree
is unreachable, "is X wired?" stops having a trustworthy answer, and this
codebase has been bitten by exactly that failure repeatedly: a second affect
engine with no construction path, a declared fallback that could never run, a
vision flag with three different defaults across four files. Unreachable code
is where half-wired subsystems hide, because nothing distinguishes "staged for
later" from "silently disconnected".

Bulk deletion is the wrong response and this ratchet does not ask for one.
Some orphans are entry points, some are staged work, and the frozen reqproof
and architecture artifacts reference paths that still have to resolve. What
the ratchet does is make the count visible and one-way, so the cost lands on
whoever adds the next one.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.lint_module_reachability import BASELINE, scan

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def report() -> dict[str, object]:
    return scan()


@pytest.fixture(scope="module")
def baseline() -> dict[str, object]:
    assert BASELINE.is_file(), "reachability baseline is missing"
    return json.loads(BASELINE.read_text(encoding="utf-8"))


def test_no_new_unreachable_modules(report, baseline):
    known = set(baseline["orphans"])
    now = set(report["orphans"])
    new = sorted(now - known)
    assert not new, (
        "these modules became unreachable — wire them to something or retire "
        f"them:\n  " + "\n  ".join(new)
    )


def test_the_count_does_not_rise(report, baseline):
    assert report["orphan_count"] <= baseline["orphan_count"]


def test_dynamic_references_count_as_reachable():
    """A module reached only by importlib or a registry string is not dead.

    Naive import-graph analysis called 292 modules orphaned; string-literal
    detection found 13 of those were reached by name. Deleting one of those
    would have broken a working path while the analysis reported it as dead
    weight — the same class of error the ratchet exists to prevent.
    """
    from tools.lint_module_reachability import _iter_sources

    sources = _iter_sources()
    assert sources, "the source scan found nothing at all"
    # A real dynamic reference exists somewhere in the tree; if string
    # detection regressed to zero, every such module would flip to orphaned
    # at once and the ratchet would fire on all of them.
    text = "\n".join(
        p.read_text(encoding="utf-8", errors="ignore")
        for p in sources
        if p.name.endswith(".py")
    )
    assert '"core.' in text or "'core." in text


def test_the_baseline_is_a_list_of_real_modules(baseline):
    """A stale baseline naming deleted files would silently absorb new orphans."""
    root = Path(__file__).resolve().parent.parent
    missing = [
        name
        for name in baseline["orphans"]
        if not (root / (name.replace(".", "/") + ".py")).is_file()
        and not (root / (name.replace(".", "/") + "/__init__.py")).is_file()
    ]
    assert not missing, (
        "the baseline names modules that no longer exist; refresh it with "
        f"--write-baseline: {missing[:10]}"
    )
