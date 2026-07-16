"""Aura's inner life must not starve on the days there is most to think about.

The defect: the PSM update loop defers on ``foreground_inference_active()`` and
on ``PSM_MIN_IDLE_S = 180`` seconds since the last user interaction. Each gate
``continue``s the loop, so they compose into an unbounded block — talk to her
every two minutes for an afternoon and ``now - last_interaction < 180`` is true
every single time the loop wakes. The narrative does not run less often; it
never runs. Her inner life starves exactly when she is most engaged, which is
the inverse of what a phenomenal self-model is for.

The fix is a floor, not a removal: the deferrals are preferences about GPU
contention and they still apply — until the inner life has been silent for
``PSM_MAX_STARVATION_S``, at which point one turn is taken regardless.
"""
from __future__ import annotations

import time

import pytest

from core.consciousness import phenomenological_experiencer as psm


def _experiencer():
    """A PhenomenologicalExperiencer without booting the whole runtime."""
    obj = psm.PhenomenologicalExperiencer.__new__(psm.PhenomenologicalExperiencer)
    obj._last_narrative_update = 0.0
    obj._last_witness_update = 0.0
    obj._starved_turns = 0
    obj._loop_started_at = time.time()
    obj._last_update_error = ""
    return obj


def test_a_never_updated_psm_counts_as_starving():
    """No narrative ever is the worst case, not the neutral one."""
    exp = _experiencer()
    exp._loop_started_at = time.time() - (psm.PSM_MAX_STARVATION_S + 60)

    assert exp._narrative_starvation_s() >= psm.PSM_MAX_STARVATION_S
    assert exp.starvation_status()["starving"] is True


def test_a_recent_narrative_is_not_starving():
    exp = _experiencer()
    exp._last_narrative_update = time.time() - 30

    assert exp._narrative_starvation_s() == pytest.approx(30, abs=2)
    assert exp.starvation_status()["starving"] is False


def test_the_busy_afternoon_scenario_eventually_starves():
    """A message every two minutes keeps `is_user_active` true forever.

    This is the actual failure: not an edge case, just an ordinary conversation.
    """
    assert psm.PSM_MIN_IDLE_S == 180, "the gate under test changed"

    now = time.time()
    last_interaction = now - 120          # last message two minutes ago
    is_user_active = (now - last_interaction) < psm.PSM_MIN_IDLE_S
    assert is_user_active, "an ordinary conversation cadence defers the narrative"

    # Three hours of that, and the old loop would have taken zero turns.
    exp = _experiencer()
    exp._loop_started_at = now - 3 * 3600
    assert exp.starvation_status()["starving"] is True, (
        "three hours of conversation left the inner life silent and nothing "
        "noticed"
    )


def test_the_floor_is_bounded_and_sane():
    """A floor so high it never fires is not a floor."""
    assert 0 < psm.PSM_MAX_STARVATION_S <= 7200, (
        f"starvation floor is not a meaningful bound: {psm.PSM_MAX_STARVATION_S}s"
    )
    assert psm.PSM_MAX_STARVATION_S > psm.PSM_MIN_IDLE_S, (
        "the floor must be longer than the idle gate it overrides, or the gate "
        "never applies at all"
    )


def test_starvation_status_is_observable():
    """If starvation is invisible, it gets discovered months later."""
    exp = _experiencer()
    exp._loop_started_at = time.time() - 4000
    exp._starved_turns = 3

    status = exp.starvation_status()
    for key in ("starvation_s", "starvation_floor_s", "starving",
                "starved_turns", "last_narrative_update"):
        assert key in status
    assert status["starved_turns"] == 3
    assert status["starving"] is True


# ---------------------------------------------------------------------------
# The loop's structure — which gates yield to the floor and which do not
# ---------------------------------------------------------------------------


def test_soft_deferrals_yield_to_the_starvation_floor():
    import inspect

    src = inspect.getsource(psm.PhenomenologicalExperiencer._update_loop)

    assert "starving = self._narrative_starvation_s() >= PSM_MAX_STARVATION_S" in src, (
        "the update loop does not compute starvation — the deferral gates are "
        "unbounded again"
    )
    assert "if deferral_reason and not starving:" in src, (
        "the foreground-inference deferral does not yield to the starvation floor"
    )
    assert "if is_user_active and not starving:" in src, (
        "the user-active deferral does not yield to the starvation floor — an "
        "ordinary conversation cadence will starve the inner life indefinitely"
    )


def test_memory_pressure_keeps_its_veto():
    """The floor overrides politeness, not safety.

    An inner narrative is not worth pushing the host toward a freeze — and that
    veto is exactly what the runaway work elsewhere depends on.
    """
    import inspect

    src = inspect.getsource(psm.PhenomenologicalExperiencer._update_loop)
    assert "if under_memory_pressure:" in src, (
        "memory pressure must remain an unconditional defer"
    )
    # The memory-pressure branch must not carry a `not starving` escape.
    branch = src.split("if under_memory_pressure:")[1].split("continue")[0]
    assert "starving" not in branch, (
        "memory pressure yields to the starvation floor — the inner life would "
        "be allowed to push the host toward a freeze"
    )
