"""Recurrence pressure must mean "this is happening again".

CP126 61cc8648: occurrences and streak were LIFETIME totals, so after six
events the count term pinned at 1.0 for the life of the process. A subsystem
that failed a lot once looked identical to one failing right now — and a
successful repair only decremented the streak by one, so the old history kept
dominating and a fixed subsystem still reported the pressure of the problem
that had been fixed.
"""
from __future__ import annotations

import time

import pytest

from core.adaptation.adaptive_immunity import (
    AdaptiveImmuneConfig,
    AdaptiveImmuneSystem,
)


@pytest.fixture()
def immune(tmp_path):
    cfg = AdaptiveImmuneConfig(population_size=2, max_population=4)
    return AdaptiveImmuneSystem(config=cfg, state_dir=tmp_path, rng_seed=11)


def _track(immune, *, occurrences, streak, age_s, verified_at=0.0, failed=0, verified=0):
    now = time.time()
    key = immune._recurrence_keys("memory", "RuntimeError")[0]
    immune._recurrence_tracker[key] = {
        "occurrences": occurrences,
        "last_seen": now - age_s,
        "interval_ewma": 0.0,
        "last_interval": None,
        "streak": streak,
        "peak_streak": streak,
        "verified_repairs": verified,
        "failed_repairs": failed,
        "last_verified_at": verified_at,
    }
    return immune._estimate_recurrence_pressure("memory", "RuntimeError")


def test_recent_recurrence_produces_pressure(immune):
    assert _track(immune, occurrences=6, streak=4, age_s=1.0) > 0.4


def test_the_same_history_ages_out(immune):
    """Six failures long ago is not six failures now."""
    window = immune.cfg.recurrence_window_s
    fresh = _track(immune, occurrences=6, streak=4, age_s=1.0)
    stale = _track(immune, occurrences=6, streak=4, age_s=window * 0.95)

    assert stale < fresh


def test_history_beyond_the_window_contributes_nothing(immune):
    window = immune.cfg.recurrence_window_s
    assert _track(immune, occurrences=99, streak=99, age_s=window * 3) == pytest.approx(
        0.0, abs=1e-6
    )


def test_the_count_term_no_longer_pins_forever(immune):
    """It used to saturate at six occurrences and stay there."""
    window = immune.cfg.recurrence_window_s
    many_old = _track(immune, occurrences=1000, streak=100, age_s=window * 2)

    assert many_old < 0.05


def test_a_verified_repair_opens_a_healthy_epoch(immune):
    now = time.time()
    without = _track(immune, occurrences=6, streak=4, age_s=10.0)
    with_repair = _track(
        immune, occurrences=6, streak=4, age_s=10.0, verified_at=now - 1.0
    )

    assert with_repair < without


def test_a_repair_older_than_the_last_failure_does_not_damp(immune):
    """The failure came AFTER the fix — the epoch is not healthy."""
    now = time.time()
    stale_fix = _track(
        immune, occurrences=6, streak=4, age_s=10.0, verified_at=now - 500.0
    )
    no_fix = _track(immune, occurrences=6, streak=4, age_s=10.0)

    assert stale_fix == pytest.approx(no_fix)


def test_failed_repairs_still_raise_pressure(immune):
    clean = _track(immune, occurrences=3, streak=2, age_s=5.0)
    failing = _track(immune, occurrences=3, streak=2, age_s=5.0, failed=4)

    assert failing > clean


def test_an_unknown_subsystem_has_no_pressure(immune):
    assert immune._estimate_recurrence_pressure("never_seen", "Whatever") == 0.0
