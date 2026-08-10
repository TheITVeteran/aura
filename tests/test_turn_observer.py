"""Measurement that is structurally incapable of changing what runs.

Nobody knew what a turn costs: PassInstrumentation.report() aggregates by pass
name across the whole process, which answers "is this pass slow" and not "what
did that turn cost". And every StuckDetector threshold is inherited from
another project's tuning, never checked against Aura.

The first test is the important one. Observe-only is not a promise about
behaviour here — it is a property of registering after-hooks and no
before-hooks, since a before-hook's False return is the only way this seam can
skip a pass.
"""
from __future__ import annotations

import pytest

from core.runtime.stuck_detector import StuckDetector, StuckPattern
from core.runtime.turn_budget import Budget
from core.observability.turn_observer import TurnObserver, install_turn_observer
from core.pipeline.pass_manager import PassInstrumentation, PassRecord

pytestmark = pytest.mark.unit


class _Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, s):
        self.now += s


def _record(name="phase", ordinal=1, *, duration=0.01, skipped=False, error="", reason=""):
    return PassRecord(
        name=name, ordinal=ordinal, duration_s=duration,
        skipped=skipped, reason=reason, error=error,
    )


def _observer(**kwargs):
    return TurnObserver(clock=kwargs.pop("clock", _Clock()), **kwargs)


# ── the safety property ────────────────────────────────────────────────────


def test_installation_registers_no_before_hook():
    """A before-hook's False return is the only way this seam skips a pass.
    Registering none is what makes observe-only structural."""
    instrumentation = PassInstrumentation()

    install_turn_observer(instrumentation)

    assert instrumentation._before_hooks == []
    assert len(instrumentation._after_hooks) == 1


def test_a_pass_still_runs_while_observed():
    instrumentation = PassInstrumentation()
    install_turn_observer(instrumentation)

    should_run, _, reason = instrumentation.should_run("phase")

    assert should_run is True
    assert reason == ""


def test_an_exploding_observer_cannot_break_a_pass():
    """after_pass wraps each hook in its own try/except."""
    instrumentation = PassInstrumentation()

    def explode(record):
        raise RuntimeError("observer bug")

    instrumentation.add_after_hook(explode)
    instrumentation.after_pass(_record())  # must not raise

    assert instrumentation.records()


def test_installation_is_idempotent():
    instrumentation = PassInstrumentation()

    install_turn_observer(instrumentation)
    install_turn_observer(instrumentation)

    assert len(instrumentation._after_hooks) == 1


def test_idempotence_does_not_block_a_second_instrumentation():
    """Idempotence is a property of the pairing. A bare 'already installed'
    flag left the second instrumentation unmetered while reporting success."""
    first, second = PassInstrumentation(), PassInstrumentation()

    install_turn_observer(first)
    install_turn_observer(second)

    assert len(second._after_hooks) == 1


# ── turn boundaries ────────────────────────────────────────────────────────


def test_ordinal_one_opens_a_turn():
    observer = _observer()

    observer.observe(_record("a", 1))
    observer.observe(_record("b", 2))

    assert observer.report()["in_flight"]["passes"] == 2


def test_a_new_ordinal_one_closes_the_previous_turn():
    observer = _observer()
    observer.observe(_record("a", 1))
    observer.observe(_record("b", 2))

    observer.observe(_record("a", 1))

    report = observer.report()
    assert report["turns_recorded"] == 1
    assert report["passes"]["max"] == 2


def test_an_ordinal_that_does_not_advance_also_opens_a_turn():
    """Covers paths that reset without going through begin_run. Without this a
    missed reset merges every later turn into one growing record."""
    observer = _observer()
    observer.observe(_record("a", 5))
    observer.observe(_record("b", 6))

    observer.observe(_record("a", 3))

    assert observer.report()["turns_recorded"] == 1


def test_close_turn_makes_the_last_turn_visible():
    observer = _observer()
    observer.observe(_record("a", 1))

    observer.close_turn()

    report = observer.report()
    assert report["in_flight"] is None
    assert report["turns_recorded"] == 1


def test_closing_with_nothing_in_flight_is_not_an_error():
    assert _observer().close_turn() is None


# ── what it counts ─────────────────────────────────────────────────────────


def test_skipped_passes_are_counted_separately_from_run_ones():
    observer = _observer()

    observer.observe(_record("a", 1))
    observer.observe(_record("b", 2, skipped=True))

    in_flight = observer.report()["in_flight"]
    assert in_flight["passes"] == 1
    assert in_flight["skipped"] == 1


def test_errors_are_counted():
    observer = _observer()

    observer.observe(_record("a", 1, error="boom"))

    assert observer.report()["in_flight"]["errors"] == 1


def test_duration_comes_from_the_clock_not_the_pass_records():
    clock = _Clock()
    observer = _observer(clock=clock)
    observer.observe(_record("a", 1))
    clock.advance(2.5)

    summary = observer.close_turn()

    assert summary.duration_s == 2.5


def test_the_report_states_tokens_are_unmeasured_rather_than_zero():
    """Absent is not zero. PassRecord carries no token count."""
    assert "not measured" in _observer().report()["tokens"]


def test_an_empty_report_is_not_an_error():
    report = _observer().report()

    assert report["turns_recorded"] == 0
    assert report["passes"] == {"count": 0}


def test_the_distribution_summarises_cost_across_turns():
    observer = _observer()
    for turn, passes in enumerate([1, 3, 5]):
        for ordinal in range(1, passes + 1):
            observer.observe(_record("p", ordinal))
    observer.close_turn()

    passes = observer.report()["passes"]

    assert passes["count"] == 3
    assert passes["min"] == 1
    assert passes["max"] == 5


def test_history_is_bounded():
    """Unbounded history is a leak with a nice name."""
    observer = _observer(history=3)
    for _ in range(10):
        observer.observe(_record("p", 1))
    observer.close_turn()

    assert observer.report()["turns_recorded"] <= 3


# ── stuck detection over phases ────────────────────────────────────────────


def test_a_phase_erroring_repeatedly_is_flagged():
    observer = _observer(detector=StuckDetector(error_threshold=3))

    for ordinal in range(1, 5):
        observer.observe(_record("recall", ordinal, error="ModuleNotFound"))

    assert observer.report()["in_flight"]["stuck"] is not None


def test_the_flagged_pattern_is_reported():
    observer = _observer(detector=StuckDetector(error_threshold=3))
    for ordinal in range(1, 5):
        observer.observe(_record("recall", ordinal, error="same"))
    observer.close_turn()

    assert str(StuckPattern.REPEATED_ACTION_ERROR) in observer.report()["stuck_patterns"]


def test_a_healthy_turn_is_not_flagged():
    observer = _observer()

    for ordinal, name in enumerate(["perceive", "recall", "reason", "answer"], start=1):
        observer.observe(_record(name, ordinal))

    assert observer.report()["in_flight"]["stuck"] is None


def test_stuck_state_does_not_leak_across_turns():
    observer = _observer(detector=StuckDetector(error_threshold=3))
    for ordinal in range(1, 5):
        observer.observe(_record("recall", ordinal, error="same"))

    observer.observe(_record("perceive", 1))

    assert observer.report()["in_flight"]["stuck"] is None


def test_being_flagged_does_not_stop_the_turn():
    """The whole point: it notices and keeps counting."""
    observer = _observer(detector=StuckDetector(error_threshold=3))
    for ordinal in range(1, 5):
        observer.observe(_record("recall", ordinal, error="same"))

    observer.observe(_record("recall", 5, error="same"))

    assert observer.report()["in_flight"]["passes"] == 5


# ── the budget stays inert ─────────────────────────────────────────────────


def test_an_unlimited_budget_never_raises_into_the_pipeline():
    observer = _observer(budget=Budget.unlimited())

    for ordinal in range(1, 500):
        observer.observe(_record("p", ordinal))

    assert observer.report()["in_flight"]["passes"] == 499


def test_a_ceiling_is_recorded_not_enforced():
    """record() rather than spend(): a metered turn must never raise."""
    observer = _observer(budget=Budget(max_steps=2))

    for ordinal in range(1, 6):
        observer.observe(_record("p", ordinal))

    assert observer.report()["in_flight"]["passes"] == 5
