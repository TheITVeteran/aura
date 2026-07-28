"""One event must not serialize the whole immune ecology three times.

CP126 df9f2a05: core observation writes state, reinforcement can write
again, and the response summary writes again. During a failure storm — when
many events arrive at once — that multiplied I/O and lock time inside the
subsystem meant to be responding to the storm.
"""
from __future__ import annotations

import time

import pytest

from core.adaptation.adaptive_immunity import get_adaptive_immune_system


@pytest.fixture()
def immune(monkeypatch, tmp_path):
    system = get_adaptive_immune_system()
    writes = []
    monkeypatch.setattr(
        system, "_write_state_payload", lambda payload: writes.append(payload), raising=False
    )
    system._writes = writes
    system._last_save_at = 0.0
    system._deferred_saves = 0
    return system


def test_a_burst_of_saves_collapses(immune, monkeypatch):
    calls = []
    monkeypatch.setattr(
        type(immune), "_save_state",
        lambda self, force=False: calls.append(force) if (
            force or (time.time() - self._last_save_at) >= self._save_min_interval_s
        ) else None,
        raising=False,
    )
    # Exercise the REAL implementation instead of a stub.
    monkeypatch.undo()

    immune._last_save_at = time.time()
    before = immune._deferred_saves
    for _ in range(5):
        immune._save_state()

    assert immune._deferred_saves == before + 5     # all deferred
    assert immune._state_dirty is True              # and remembered


def test_a_deferred_write_is_not_lost(immune):
    immune._last_save_at = time.time()
    immune._save_state()
    assert immune._state_dirty is True

    # Past the interval, the next call writes the latest state.
    immune._last_save_at = time.time() - immune._save_min_interval_s - 1
    immune._save_state()

    assert immune._state_dirty is False


def test_force_always_writes(immune):
    immune._last_save_at = time.time()

    immune._save_state(force=True)

    assert immune._state_dirty is False


def test_durability_points_force(immune):
    import inspect

    from core.adaptation import adaptive_immunity as mod

    boot = inspect.getsource(mod.AdaptiveImmuneSystem.__init__)
    assert "_save_state(force=True)" in boot
    consolidate = inspect.getsource(mod.AdaptiveImmuneSystem.dream_consolidate)
    assert "_save_state(force=True)" in consolidate


def test_the_intermediate_writes_within_an_event_coalesce(immune):
    """The finding is multiple snapshots PER EVENT, not one per event.

    _observe_core and _reinforce_after_execution defer into the single write
    at the end of the observation, so one event costs one snapshot instead of
    three — without giving up recurrence memory across a reload.
    """
    import inspect

    from core.adaptation import adaptive_immunity as mod

    for name in ("_observe_core", "_reinforce_after_execution"):
        source = inspect.getsource(getattr(mod.AdaptiveImmuneSystem, name))
        assert "_save_state(force=True)" not in source


def test_the_last_write_of_an_observation_is_durable(immune):
    import inspect

    from core.adaptation import adaptive_immunity as mod

    source = inspect.getsource(mod.AdaptiveImmuneSystem._record_response_summary)
    assert "_save_state(force=True)" in source


def test_the_coalescing_interval_is_bounded(immune):
    assert 0.0 < immune._save_min_interval_s <= 30.0
