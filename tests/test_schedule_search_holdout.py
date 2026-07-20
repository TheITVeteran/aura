"""Held-out discipline in the live schedule search (CP235).

Anima Rationis line 138: search schedules against HELD-OUT verified tasks.
A schedule selected on the tasks it is scored on is a memorized answer key,
and in the final number that is indistinguishable from a discovery.

These properties were prototyped in a parallel module (CP232) before it was
found that ScheduleSearch already existed. They live here now, on the live
implementation, and the duplicate is gone.
"""
from __future__ import annotations

import pytest

from core.brain.llm.latent_cortex.schedules import LayerSchedule, ScheduleSearch


def _search():
    return ScheduleSearch(prelude_end=4, coda_start=12, max_repeats=6, seed=3)


def test_same_evaluator_for_search_and_holdout_is_refused():
    def scorer(schedule):
        return 1.0

    with pytest.raises(ValueError, match="answer key"):
        _search().run(scorer, holdout_evaluator=scorer, population=4, generations=2)


def test_holdout_score_and_generalization_gap_are_reported():
    def search_scorer(schedule):
        return float(schedule.total_layer_repeats)

    def holdout_scorer(schedule):
        return 1.0  # none of the search gain transfers

    result = _search().run(
        search_scorer, holdout_evaluator=holdout_scorer,
        population=6, generations=3,
    )
    assert result.holdout_score == 1.0
    assert result.generalization_gap() == result.best_score - 1.0
    assert result.overfit_warning() is True


def test_no_holdout_evaluator_leaves_the_gap_unknown_not_zero():
    """Absence of evidence must not read as evidence of generalization."""
    result = _search().run(
        lambda s: float(s.total_layer_repeats), population=4, generations=2
    )
    assert result.holdout_score is None
    assert result.generalization_gap() is None
    assert result.overfit_warning() is False


def test_compute_budget_prevents_winning_by_spending_more():
    def greedy(schedule):
        return float(schedule.total_layer_repeats)

    # Seed is single_window(4, 12, 4) = 8 layers x 4 repeats = 32.
    result = _search().run(
        greedy, population=6, generations=4, max_layer_apps=40
    )
    assert result.best.total_layer_repeats <= 40, (
        "a schedule must not win by running more compute than allowed"
    )


def test_a_budget_below_the_seed_is_refused():
    """The seed enters the pool unconditionally, so such a budget would
    silently bound nothing while appearing to cap compute."""
    with pytest.raises(ValueError, match="would bound nothing"):
        _search().run(
            lambda s: 1.0, population=4, generations=2, max_layer_apps=10
        )


def test_search_without_a_budget_still_works():
    result = _search().run(
        lambda s: 1.0 / max(s.total_layer_repeats, 1), population=4, generations=2
    )
    assert isinstance(result.best, LayerSchedule)
    assert result.evaluated >= 1
