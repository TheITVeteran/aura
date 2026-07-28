"""The self-model must not be able to drive itself in a loop.

Three vectors exist, and each is closed:
  1. a probe that reaches back into the self-model (unbounded recursion),
  2. a faculty re-proposed on every tick before improvement could land
     (livelock — the same shape as the immune lane firing one remedy 247
     times),
  3. a cyclic gates declaration (unbounded graph walk).
"""
from __future__ import annotations

import threading
import time

import pytest

from core.metacognition import faculty_model as fm
from core.metacognition.faculty_model import (
    Faculty,
    FacultyRegistry,
    ImprovementMetric,
    clear_proposal_history,
    improvement_goal,
)


@pytest.fixture(autouse=True)
def _clean():
    clear_proposal_history()
    fm._assessing.active = False
    yield
    clear_proposal_history()


def _metric(mid, probe, **kw):
    return ImprovementMetric(mid, "", "higher_is_better", probe, 0.0, 0.8, 1.0, **kw)


# --- 1. probe re-entrancy ------------------------------------------------


def test_a_probe_that_reassesses_terminates_at_one_level():
    """The guarantee is BOUNDED recursion, not a refused outer probe: the
    nested assessment is the one that declines, so the probe still completes
    and the stack cannot grow."""
    registry = FacultyRegistry()
    calls = []
    nested: list = []

    def _recursive_probe():
        calls.append(1)
        nested.append(registry.assess())   # the loop, if unguarded
        return 0.5

    registry.declare(Faculty("f", "d", "o", metrics=(_metric("m", _recursive_probe),)))

    model = registry.assess()      # must return, not blow the stack

    assert len(calls) == 1                                    # invoked once
    assert model.by_id("f").readings[0].measured is True       # outer succeeds
    inner = nested[0].by_id("f").readings[0]
    assert inner.measured is False                             # nested declined
    assert "re-entered" in inner.reason


def test_a_probe_that_calls_improvement_goal_terminates():
    registry = FacultyRegistry()
    calls = []

    def _probe():
        calls.append(1)
        improvement_goal(registry.assess())
        return 0.5

    registry.declare(Faculty("f", "d", "o", metrics=(_metric("m", _probe),)))

    registry.assess()

    assert len(calls) == 1  # no re-entry multiplied the work


def test_the_guard_is_released_after_a_normal_read():
    registry = FacultyRegistry()
    registry.declare(Faculty("f", "d", "o", metrics=(_metric("m", lambda: 0.5),)))

    registry.assess()

    assert getattr(fm._assessing, "active", False) is False


def test_the_guard_is_released_when_a_probe_raises():
    """A latched guard would turn every later metric into 're-entered'."""
    registry = FacultyRegistry()

    def _boom():
        raise RuntimeError("nope")

    registry.declare(
        Faculty("f", "d", "o", metrics=(_metric("bad", _boom), _metric("good", lambda: 0.5)))
    )

    model = registry.assess()
    readings = {r.metric_id: r for r in model.by_id("f").readings}

    assert readings["bad"].measured is False
    assert readings["good"].measured is True    # not poisoned by the raiser
    assert getattr(fm._assessing, "active", False) is False


def test_a_recursive_probe_does_not_poison_its_siblings():
    registry = FacultyRegistry()

    def _recursive():
        registry.assess()
        return 0.5

    registry.declare(
        Faculty("f", "d", "o", metrics=(_metric("r", _recursive), _metric("ok", lambda: 0.4)))
    )

    readings = {r.metric_id: r for r in registry.assess().by_id("f").readings}

    assert readings["r"].measured is True
    assert readings["ok"].measured is True
    assert getattr(fm._assessing, "active", False) is False


def test_the_guard_is_per_thread():
    """Concurrent assessments on different threads must stay independent."""
    registry = FacultyRegistry()
    registry.declare(Faculty("f", "d", "o", metrics=(_metric("m", lambda: 0.5),)))
    results = []

    def _run():
        results.append(registry.assess().by_id("f").readings[0].measured)

    threads = [threading.Thread(target=_run) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)

    assert results == [True] * 4


# --- 2. re-proposal livelock ---------------------------------------------


def test_the_same_faculty_is_not_proposed_twice_in_a_row():
    registry = FacultyRegistry()
    registry.declare(Faculty("memory", "d", "o", metrics=(_metric("m", lambda: 0.1),)))

    first = improvement_goal(registry.assess())
    second = improvement_goal(registry.assess())

    assert first is not None and first["faculty"] == "memory"
    assert second is None          # suppressed, not repeated forever


def test_a_second_faculty_is_still_reachable_while_the_first_cools_down():
    registry = FacultyRegistry()
    registry.declare(
        Faculty("memory", "d", "o", metrics=(_metric("m", lambda: 0.1),), gates=("attn",))
    )
    registry.declare(Faculty("attn", "d", "o", metrics=(_metric("a", lambda: 0.2),)))

    first = improvement_goal(registry.assess())
    second = improvement_goal(registry.assess())

    assert first["faculty"] == "memory"
    assert second is not None and second["faculty"] == "attn"


def test_the_cooldown_expires():
    registry = FacultyRegistry()
    registry.declare(Faculty("memory", "d", "o", metrics=(_metric("m", lambda: 0.1),)))

    improvement_goal(registry.assess())
    again = improvement_goal(registry.assess(), cooldown_s=0.0)

    assert again is not None and again["faculty"] == "memory"


def test_blind_spot_goals_are_also_rate_limited():
    registry = FacultyRegistry()
    registry.declare(
        Faculty("temporal", "d", "o", metrics=(_metric("t", lambda: None),))
    )

    first = improvement_goal(registry.assess())
    second = improvement_goal(registry.assess())

    assert first["faculty"] == "temporal"
    assert second is None


def test_proposal_history_is_bounded():
    for index in range(400):
        fm._mark_proposed(f"f{index}")

    assert len(fm._last_proposed) <= 256


# --- 3. cyclic gates -----------------------------------------------------


def test_a_gates_cycle_terminates():
    registry = FacultyRegistry()
    registry.declare(Faculty("a", "d", "o", gates=("b",)))
    registry.declare(Faculty("b", "d", "o", gates=("c",)))
    registry.declare(Faculty("c", "d", "o", gates=("a",)))

    start = time.monotonic()
    assert registry.leverage("a") >= 1.0
    assert time.monotonic() - start < 1.0


def test_a_self_gating_faculty_terminates():
    registry = FacultyRegistry()
    registry.declare(Faculty("a", "d", "o", gates=("a",)))

    assert registry.leverage("a") == 1.0
