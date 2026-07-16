"""A system that cannot fail hard cannot stop hurting itself.

The concern this answers: with thousands of broad excepts replaced by receipts,
Aura will never see a hard failure — it will see a slow 110GB runaway, recorded
faithfully, one polite warning at a time, until the host freezes.
autonomy_latitude.py exists because there was a runaway. It will happen again.

Level thresholds (prune at 28GB, unload at 34GB) miss this two ways, and both
are tested here:

  - Slow drift. The 4h soak measured ~242MB/h. At that rate the 28GB trigger is
    days away while the trend is obvious the entire time.
  - Ineffective mitigation. Pruning fires, RSS keeps climbing, pruning fires
    again — forever — and each cycle emits a receipt saying it handled things.
"""
from __future__ import annotations

import pytest

from core.resilience.runaway_budget import (
    RunawayDetector,
    RunawayPolicy,
    RunawayState,
    get_runaway_budget,
    reset_runaway_budget,
)


@pytest.fixture(autouse=True)
def _clean_budget():
    reset_runaway_budget()
    yield
    reset_runaway_budget()


def _policy(**kw) -> RunawayPolicy:
    base = dict(
        min_slope_per_hour=50.0,
        min_samples=5,
        min_window_s=60.0,
        ineffective_after_mitigations=3,
        ceiling=None,
        projection_horizon_s=3600.0,
    )
    base.update(kw)
    return RunawayPolicy(**base)


def _feed(det, start_value, slope_per_hour, *, minutes, step_s=60.0, t0=1000.0):
    """Feed a linear ramp. Returns the last timestamp."""
    now = t0
    end = t0 + minutes * 60
    while now <= end:
        det.observe(start_value + slope_per_hour * (now - t0) / 3600.0, now=now)
        now += step_s
    return now - step_s


# ---------------------------------------------------------------------------
# The trend itself
# ---------------------------------------------------------------------------


def test_flat_memory_is_nominal():
    det = RunawayDetector("rss", _policy())
    _feed(det, 8000.0, 0.0, minutes=30)
    verdict = det.assess(now=1000.0 + 30 * 60)
    assert verdict.state is RunawayState.NOMINAL
    assert abs(verdict.slope_per_hour) < 1.0


def test_falling_memory_is_nominal():
    det = RunawayDetector("rss", _policy())
    _feed(det, 8000.0, -300.0, minutes=30)
    assert det.assess(now=1000.0 + 30 * 60).state is RunawayState.NOMINAL


def test_the_real_soak_leak_is_detected_as_drift():
    """~242MB/h linear — the growth the 4h soak actually measured.

    A level threshold at 28GB would not fire for days. The trend is unambiguous
    within the hour.
    """
    det = RunawayDetector("rss", _policy())
    _feed(det, 8000.0, 242.0, minutes=60)
    verdict = det.assess(now=1000.0 + 60 * 60)

    assert verdict.state is RunawayState.DRIFT
    assert verdict.slope_per_hour == pytest.approx(242.0, rel=0.05)


def test_noise_below_the_slope_floor_is_not_drift():
    """A detector that cries wolf gets ignored, which is the same as absent."""
    det = RunawayDetector("rss", _policy(min_slope_per_hour=200.0))
    _feed(det, 8000.0, 20.0, minutes=60)
    assert det.assess(now=1000.0 + 60 * 60).state is RunawayState.NOMINAL


def test_insufficient_history_is_not_a_verdict():
    det = RunawayDetector("rss", _policy())
    det.observe(8000.0, now=1000.0)
    det.observe(9000.0, now=1060.0)
    verdict = det.assess(now=1060.0)
    assert verdict.state is RunawayState.NOMINAL
    assert "insufficient history" in verdict.reason


# ---------------------------------------------------------------------------
# The judgement the receipt-only system cannot make
# ---------------------------------------------------------------------------


def test_growth_despite_repeated_mitigation_is_a_runaway():
    """The headline: pruning ran three times and RSS kept climbing.

    Each of those prunes wrote a receipt saying it handled things. Three
    receipts and a rising line is not "handled" — it is proof the mitigation
    does not address this problem.
    """
    det = RunawayDetector("rss", _policy())
    t0 = 1000.0
    now = t0
    for i in range(20):
        det.observe(8000.0 + 242.0 * (now - t0) / 3600.0, now=now)
        if i in (5, 10, 15):
            det.record_mitigation(now=now)
        now += 60.0

    verdict = det.assess(now=now)
    assert verdict.is_runaway()
    assert verdict.mitigations_in_window >= 3
    assert "does not work" in verdict.reason


def test_mitigation_that_works_is_not_a_runaway():
    """Fail hard only when the fix is failing — not whenever cleanup runs."""
    det = RunawayDetector("rss", _policy())
    t0 = 1000.0
    now = t0
    value = 8000.0
    for i in range(20):
        det.observe(value, now=now)
        value += 4.0                       # creeping up
        if i in (5, 10, 15):
            det.record_mitigation(now=now)
            value -= 60.0                  # ...and the prune actually reclaims
        now += 60.0

    verdict = det.assess(now=now)
    assert not verdict.is_runaway(), (
        f"working mitigation misreported as a runaway: {verdict.reason}"
    )


def test_two_mitigations_is_not_yet_enough():
    """Give the fix its stated number of chances before declaring failure."""
    det = RunawayDetector("rss", _policy(ineffective_after_mitigations=3))
    t0 = 1000.0
    now = t0
    for i in range(20):
        det.observe(8000.0 + 242.0 * (now - t0) / 3600.0, now=now)
        if i in (5, 10):
            det.record_mitigation(now=now)
        now += 60.0

    assert det.assess(now=now).state is RunawayState.DRIFT


# ---------------------------------------------------------------------------
# Projection — the 110GB case
# ---------------------------------------------------------------------------


def test_fast_growth_toward_the_ceiling_is_a_runaway_without_waiting():
    """The 110GB incident was not slow at the end.

    When the projection says we breach the ceiling within the horizon, waiting
    for mitigation to have three chances is waiting for the host to freeze.
    """
    det = RunawayDetector(
        "rss", _policy(ceiling=48000.0, projection_horizon_s=3600.0)
    )
    _feed(det, 40000.0, 12000.0, minutes=20)   # 12GB/h, 8GB of headroom left
    verdict = det.assess(now=1000.0 + 20 * 60)

    assert verdict.is_runaway()
    assert verdict.projected_breach_s is not None
    assert verdict.projected_breach_s <= 3600.0
    assert "ceiling" in verdict.reason


def test_slow_growth_far_from_the_ceiling_is_only_drift():
    det = RunawayDetector(
        "rss", _policy(ceiling=48000.0, projection_horizon_s=3600.0)
    )
    _feed(det, 8000.0, 242.0, minutes=60)
    verdict = det.assess(now=1000.0 + 60 * 60)
    assert verdict.state is RunawayState.DRIFT
    assert verdict.projected_breach_s > 3600.0


def test_already_past_the_ceiling_projects_zero():
    det = RunawayDetector("rss", _policy(ceiling=10000.0))
    _feed(det, 12000.0, 500.0, minutes=20)
    verdict = det.assess(now=1000.0 + 20 * 60)
    assert verdict.is_runaway()
    assert verdict.projected_breach_s == 0.0


# ---------------------------------------------------------------------------
# The refusal — a runaway must cost something
# ---------------------------------------------------------------------------


def test_runaway_makes_the_budget_fail_closed():
    """Otherwise this is just a louder log line."""
    budget = get_runaway_budget()
    det = budget.detector("rss", _policy())
    t0 = 1000.0
    now = t0
    for i in range(20):
        det.observe(8000.0 + 500.0 * (now - t0) / 3600.0, now=now)
        if i in (5, 10, 15):
            det.record_mitigation(now=now)
        now += 60.0

    assert budget.is_failing_closed(now=now) is True
    reasons = budget.runaway_reasons(now=now)
    assert reasons and "rss" in reasons[0]


def test_nominal_budget_does_not_fail_closed():
    budget = get_runaway_budget()
    det = budget.detector("rss", _policy())
    _feed(det, 8000.0, 0.0, minutes=30)
    assert budget.is_failing_closed(now=1000.0 + 30 * 60) is False


def test_runaway_notifies_listeners():
    budget = get_runaway_budget()
    seen: list[tuple[str, str]] = []
    budget.on_runaway(lambda name, v: seen.append((name, v.state.value)))

    det = budget.detector("rss", _policy())
    t0 = 1000.0
    now = t0
    for i in range(20):
        det.observe(8000.0 + 500.0 * (now - t0) / 3600.0, now=now)
        if i in (5, 10, 15):
            det.record_mitigation(now=now)
        now += 60.0

    budget.assess_all(now=now)
    assert seen and seen[0][1] == "runaway"


def test_a_failing_listener_does_not_break_the_budget():
    budget = get_runaway_budget()
    budget.on_runaway(lambda name, v: (_ for _ in ()).throw(RuntimeError("boom")))

    det = budget.detector("rss", _policy())
    t0 = 1000.0
    now = t0
    for i in range(20):
        det.observe(8000.0 + 500.0 * (now - t0) / 3600.0, now=now)
        if i in (5, 10, 15):
            det.record_mitigation(now=now)
        now += 60.0

    assert budget.is_failing_closed(now=now) is True


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------


def test_memory_governor_feeds_the_detector():
    """The governor must observe its own trend, not only its thresholds."""
    import inspect

    from core.resilience.memory_governor import MemoryGovernor

    src = inspect.getsource(MemoryGovernor._enforce_policy)
    assert "_runaway.observe" in src, (
        "MemoryGovernor does not feed RSS to the runaway detector — it can only "
        "see levels, never the trend"
    )

    src = inspect.getsource(MemoryGovernor._remember_cleanup_event)
    assert "record_mitigation" in src, (
        "MemoryGovernor does not tell the detector when it mitigates — it can "
        "never learn that its own cleanup is not working"
    )
