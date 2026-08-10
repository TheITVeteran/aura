"""Ceilings on work, asked before spending rather than after.

Nothing bounded a unit of work. context_manager.token_budget bounds prompt
assembly, not the task — so a run could take four hundred steps and hold the
resident 32B for an hour, and the only thing that would stop it was something
else breaking.

The scarce resource here is not dollars (most turns are local and free) but the
wired memory the live model holds and the wall clock during which nothing else
can have it.
"""
from __future__ import annotations

import pytest

from core.runtime.turn_budget import (
    Breach,
    Budget,
    BudgetAxis,
    BudgetExceeded,
    BudgetLedger,
)

pytestmark = pytest.mark.unit


class _Clock:
    """Controllable time; a budget test that really sleeps is a slow test."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _ledger(budget=None, clock=None):
    return BudgetLedger(budget=budget or Budget(max_steps=5), clock=clock or _Clock())


# ── no magic defaults ──────────────────────────────────────────────────────


def test_an_unset_budget_is_unlimited():
    assert Budget().is_unlimited
    assert Budget.unlimited().is_unlimited


def test_an_unlimited_budget_never_stops_anything():
    ledger = BudgetLedger(budget=Budget.unlimited(), clock=_Clock())

    for _ in range(1000):
        ledger.spend(steps=1, tokens=10_000)

    assert not ledger.exhausted
    assert not ledger.winding_down


def test_a_partially_set_budget_only_binds_the_axes_given():
    ledger = BudgetLedger(budget=Budget(max_steps=2), clock=_Clock())

    ledger.spend(tokens=10**9)

    assert not ledger.exhausted


def test_a_zero_ceiling_is_refused_as_a_different_decision():
    with pytest.raises(ValueError, match="forbid the work outright"):
        Budget(max_steps=0)


@pytest.mark.parametrize("kwargs", [
    {"max_tokens": -1}, {"max_seconds": -0.5}, {"max_cost_usd": -1.0},
    {"soft_fraction": 0.0}, {"soft_fraction": 1.5},
])
def test_incoherent_budgets_are_refused(kwargs):
    with pytest.raises(ValueError):
        Budget(**kwargs)


# ── prospective: ask before you spend ──────────────────────────────────────


def test_can_afford_answers_before_the_step_runs():
    """A retrospective-only budget always overshoots by exactly one step."""
    ledger = _ledger(Budget(max_steps=2))
    ledger.spend()

    assert ledger.can_afford(steps=1)
    ledger.spend()
    assert not ledger.can_afford(steps=1)


def test_can_afford_accounts_for_the_size_of_the_request():
    ledger = _ledger(Budget(max_tokens=1000))

    assert ledger.can_afford(tokens=900)
    assert not ledger.can_afford(tokens=1001)


def test_the_prospective_breach_explains_the_projection():
    ledger = _ledger(Budget(max_tokens=100))
    ledger.spend(tokens=90)

    breach = ledger.breach(tokens=50)

    assert breach is not None
    assert "would reach 140" in breach.describe()


# ── retrospective ──────────────────────────────────────────────────────────


def test_spending_past_a_ceiling_raises():
    ledger = _ledger(Budget(max_steps=1))
    ledger.spend()

    with pytest.raises(BudgetExceeded):
        ledger.spend()


def test_the_exception_carries_the_breach():
    ledger = _ledger(Budget(max_steps=1))
    ledger.spend()

    with pytest.raises(BudgetExceeded) as caught:
        ledger.spend()

    assert caught.value.breach.axis is BudgetAxis.STEPS


def test_record_absorbs_work_that_already_happened():
    """The tokens were spent whether or not the ledger approves; refusing to
    record them would make the ledger a worse record than the log."""
    ledger = _ledger(Budget(max_tokens=100))

    breach = ledger.record(tokens=500)

    assert ledger.tokens == 500
    assert breach is not None


def test_a_raising_spend_does_not_record_the_work():
    ledger = _ledger(Budget(max_steps=1))
    ledger.spend()

    with pytest.raises(BudgetExceeded):
        ledger.spend()

    assert ledger.steps == 1


# ── each axis binds ────────────────────────────────────────────────────────


def test_steps_bind():
    ledger = _ledger(Budget(max_steps=3))
    for _ in range(3):
        ledger.spend()

    assert ledger.exhausted


def test_tokens_bind():
    ledger = _ledger(Budget(max_tokens=100))
    ledger.record(tokens=100)

    assert ledger.exhausted


def test_wall_clock_binds_without_any_step_being_taken():
    """The hour the model is held is the cost, whether or not steps ran."""
    clock = _Clock()
    ledger = BudgetLedger(budget=Budget(max_seconds=10.0), clock=clock)

    clock.advance(11.0)

    assert ledger.exhausted


def test_cost_binds():
    ledger = _ledger(Budget(max_cost_usd=1.0))
    ledger.record(cost_usd=1.5)

    assert ledger.exhausted


def test_being_exactly_at_the_limit_is_exhausted_but_not_an_overrun():
    """Two different questions. Conflating them either lets a run take one step
    past its budget or reports an overrun that never happened."""
    ledger = _ledger(Budget(max_steps=3))
    for _ in range(3):
        ledger.spend()

    assert ledger.exhausted          # no room left
    assert ledger.breach() is None   # but nothing was exceeded
    assert not ledger.can_afford(steps=1)


def test_the_first_crossed_ceiling_is_the_one_reported():
    ledger = _ledger(Budget(max_steps=1, max_tokens=10))
    ledger.record(steps=5, tokens=500)

    assert ledger.breach().axis is BudgetAxis.STEPS


# ── winding down before the guillotine ─────────────────────────────────────


def test_the_soft_threshold_signals_before_exhaustion():
    """Cut off mid-action is how a half-finished edit is left on disk."""
    ledger = _ledger(Budget(max_steps=10, soft_fraction=0.8))

    for _ in range(8):
        ledger.spend()

    assert ledger.winding_down
    assert not ledger.exhausted


def test_early_work_is_not_winding_down():
    ledger = _ledger(Budget(max_steps=10))
    ledger.spend()

    assert not ledger.winding_down


def test_the_soft_threshold_is_configurable():
    ledger = _ledger(Budget(max_steps=10, soft_fraction=0.5))
    for _ in range(5):
        ledger.spend()

    assert ledger.winding_down


def test_any_bounded_axis_can_trigger_the_wind_down():
    clock = _Clock()
    ledger = BudgetLedger(
        budget=Budget(max_steps=100, max_seconds=10.0), clock=clock
    )
    clock.advance(9.0)

    assert ledger.winding_down


# ── reporting ──────────────────────────────────────────────────────────────


def test_remaining_is_none_on_an_unbounded_axis():
    ledger = _ledger(Budget(max_steps=5))

    assert ledger.remaining(BudgetAxis.STEPS) == 5
    assert ledger.remaining(BudgetAxis.TOKENS) is None


def test_remaining_never_goes_negative():
    ledger = _ledger(Budget(max_steps=2))
    ledger.record(steps=10)

    assert ledger.remaining(BudgetAxis.STEPS) == 0


def test_utilization_omits_unbounded_axes():
    """A ratio against infinity is not a number anyone can act on."""
    ledger = _ledger(Budget(max_steps=4))
    ledger.spend()

    assert set(ledger.utilization()) == {"steps"}
    assert ledger.utilization()["steps"] == 0.25


def test_the_snapshot_explains_why_work_stopped():
    ledger = _ledger(Budget(max_steps=1))
    ledger.record(steps=2)

    snapshot = ledger.snapshot()

    assert snapshot["exhausted"] is True
    assert "steps budget exhausted" in snapshot["breach"]
    assert snapshot["steps"] == 2


def test_a_healthy_snapshot_reports_no_breach():
    ledger = _ledger(Budget(max_steps=10))
    ledger.spend()

    snapshot = ledger.snapshot()

    assert snapshot["exhausted"] is False
    assert snapshot["breach"] is None


def test_the_breach_message_names_the_axis_and_the_numbers():
    breach = Breach(axis=BudgetAxis.TOKENS, limit=100.0, used=150.0)

    assert "tokens" in breach.describe()
    assert "150" in breach.describe() and "100" in breach.describe()
