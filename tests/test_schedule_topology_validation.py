"""A schedule is only "valid" relative to a region that is itself real.

CP126 f905e912: prelude_end/coda_start arrive from the caller and were used
without checking their types or their relation, so ops were compared against
numbers that might describe no model at all — and an empty problem list is
this module's "safe to execute" claim.

CP126 4b6e3234: candidates were validated, the DEFAULT was not, and the
default is what gets returned whenever no candidate wins.
"""
from __future__ import annotations

import pytest

from core.brain.llm.latent_cortex.schedules import (
    LayerSchedule,
    ScheduleLibrary,
    StageOp,
)


def _sched(start=2, end=6, repeats=2):
    return LayerSchedule(ops=(StageOp(start, end, repeats),), name="s")


# --- f905e912: the region is checked before anything is measured on it ---


@pytest.mark.parametrize(
    "prelude_end,coda_start",
    [(6, 2), (4, 4), (0, 0), (10, 3)],
)
def test_an_empty_or_inverted_region_is_refused(prelude_end, coda_start):
    problems = _sched().validate(prelude_end=prelude_end, coda_start=coda_start)

    assert problems
    assert "empty or inverted" in problems[0]


@pytest.mark.parametrize("bad", ["2", 2.5, None, [2], True])
def test_non_integer_bounds_are_refused(bad):
    problems = _sched().validate(prelude_end=bad, coda_start=10)

    assert problems
    assert "must be an int" in problems[0]


def test_negative_bounds_are_refused():
    problems = _sched().validate(prelude_end=-1, coda_start=6)

    assert problems
    assert "non-negative" in problems[0]


def test_a_bad_region_short_circuits_rather_than_reporting_op_noise():
    """Comparing ops against unusable bounds is false reassurance."""
    problems = _sched().validate(prelude_end="x", coda_start="y")

    assert all("must be an int" in p for p in problems)


def test_a_valid_region_still_validates_normally():
    assert _sched(2, 6, 2).validate(prelude_end=2, coda_start=6) == []
    assert _sched(2, 7, 2).validate(prelude_end=2, coda_start=6)  # escapes region


# --- 4b6e3234: the default is validated like everything else -------------


def test_the_default_schedule_is_validated(tmp_path):
    library = ScheduleLibrary(tmp_path / "s.json")

    with pytest.raises(ValueError, match="not executable"):
        library.best_for_domain(
            "d", prelude_end=6, coda_start=2, default_repeats=1
        )


def test_a_non_positive_default_repeat_is_refused(tmp_path):
    library = ScheduleLibrary(tmp_path / "s.json")

    with pytest.raises(ValueError, match="not executable"):
        library.best_for_domain(
            "d", prelude_end=2, coda_start=6, default_repeats=0
        )


def test_a_valid_default_is_returned(tmp_path):
    library = ScheduleLibrary(tmp_path / "s.json")

    schedule = library.best_for_domain(
        "d", prelude_end=2, coda_start=6, default_repeats=2
    )

    assert schedule.validate(prelude_end=2, coda_start=6) == []


def test_the_returned_default_is_executable_for_its_topology(tmp_path):
    """The property the module promises: nothing unrunnable is handed back."""
    library = ScheduleLibrary(tmp_path / "s.json")

    for repeats in (1, 2, 8):
        schedule = library.best_for_domain(
            "d", prelude_end=0, coda_start=12, default_repeats=repeats
        )
        assert schedule.validate(prelude_end=0, coda_start=12) == []
