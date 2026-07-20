"""Searching the execution schedule (CP232).

Anima Rationis line 138: do not let a model guess schedules -- search them
against HELD-OUT verified tasks. A schedule selected on the tasks it is
scored on is a memorized answer key, and in the final number that is
indistinguishable from a discovery.
"""
from __future__ import annotations

import pytest

from core.learning.schedule_search import (
    LayerSchedule,
    evolve_schedules,
    mutate,
)

TOTAL = 16


# ── Schedules describe compute honestly ─────────────────────────────────


def test_vanilla_schedule_is_the_ordinary_forward_pass():
    schedule = LayerSchedule.vanilla(TOTAL)
    assert schedule.layer_applications() == TOTAL
    assert schedule.touched_layers() == set(range(TOTAL))


def test_repeating_a_segment_costs_what_it_costs():
    """Compute must be comparable, or a 'better' schedule just ran more."""
    schedule = LayerSchedule(segments=((0, 4), (4, 12), (4, 12), (12, 16)))
    assert schedule.layer_applications() == 4 + 8 + 8 + 4
    assert schedule.touched_layers() == set(range(TOTAL))


def test_malformed_schedules_are_refused():
    with pytest.raises(ValueError, match="at least one segment"):
        LayerSchedule(segments=())
    with pytest.raises(ValueError, match="invalid segment"):
        LayerSchedule(segments=((5, 5),))
    with pytest.raises(ValueError, match="invalid segment"):
        LayerSchedule(segments=((8, 4),))
    assert not LayerSchedule(segments=((0, 99),)).is_valid_for(TOTAL)


# ── Mutation stays inside the model ─────────────────────────────────────


def test_mutation_produces_valid_schedules():
    import random

    rng = random.Random(7)
    schedule = LayerSchedule.vanilla(TOTAL)
    for _ in range(200):
        schedule = mutate(schedule, TOTAL, rng)
        assert schedule.is_valid_for(TOTAL)
        assert schedule.layer_applications() > 0


# ── The discipline that makes a result a result ─────────────────────────


def test_using_one_task_set_for_search_and_scoring_is_refused():
    """Selecting on the tasks being scored produces an answer key."""
    def scorer(schedule):
        return 1.0

    with pytest.raises(ValueError, match="answer key"):
        evolve_schedules(
            total_layers=TOTAL, search_scorer=scorer, holdout_scorer=scorer,
        )


def test_search_finds_a_schedule_that_generalizes():
    """A signal present in both sets should be found and should transfer."""
    # A climbable landscape: partial credit for compute spent in the
    # middle band. A scorer that pays only for an exact configuration is a
    # needle in a haystack, and evolution cannot climb a flat surface --
    # which says nothing about whether the search works.
    def _middle_fraction(schedule):
        inside = sum(
            len(set(range(s, e)) & set(range(4, 12))) for s, e in schedule.segments
        )
        return inside / max(schedule.layer_applications(), 1)

    def search_scorer(schedule):
        return _middle_fraction(schedule)

    def holdout_scorer(schedule):
        return 0.9 * _middle_fraction(schedule)

    result = evolve_schedules(
        total_layers=TOTAL, search_scorer=search_scorer,
        holdout_scorer=holdout_scorer, generations=6, population=12, seed=3,
    )
    assert result["best_holdout_score"] > 0
    assert result["beats_baseline"] is True
    assert result["unique_schedules_evaluated"] > 1


def test_a_schedule_that_only_wins_on_the_search_set_is_flagged():
    """The failure this module exists to make visible."""
    def search_scorer(schedule):
        return float(len(schedule.segments))  # rewards complexity

    def holdout_scorer(schedule):
        return 0.0  # none of it transfers

    result = evolve_schedules(
        total_layers=TOTAL, search_scorer=search_scorer,
        holdout_scorer=holdout_scorer, generations=6, population=12, seed=5,
    )
    assert result["beats_baseline"] is False
    assert result["overfit_warning"] is True
    assert result["generalization_gap"] > 0.1


def test_compute_budget_prevents_winning_by_spending_more():
    def search_scorer(schedule):
        return float(schedule.layer_applications())  # pure compute greed

    def holdout_scorer(schedule):
        return float(schedule.layer_applications())

    budget = TOTAL * 2
    result = evolve_schedules(
        total_layers=TOTAL, search_scorer=search_scorer,
        holdout_scorer=holdout_scorer, generations=5, population=8, seed=1,
        compute_budget=budget,
    )
    assert result["best_applications"] <= budget, (
        "a schedule must not win by running more compute than allowed"
    )


def test_baseline_is_scored_on_holdout_for_a_fair_comparison():
    def search_scorer(schedule):
        return 1.0 if len(schedule.segments) > 1 else 0.0

    def holdout_scorer(schedule):
        return 0.5

    result = evolve_schedules(
        total_layers=TOTAL, search_scorer=search_scorer,
        holdout_scorer=holdout_scorer, generations=3, population=8, seed=2,
    )
    assert result["baseline_holdout_score"] == 0.5
    assert result["improvement"] == 0.0
    assert result["beats_baseline"] is False


def test_invalid_search_settings_are_refused():
    with pytest.raises(ValueError, match="total_layers"):
        evolve_schedules(
            total_layers=1, search_scorer=lambda s: 0.0,
            holdout_scorer=lambda s: 0.0,
        )
    with pytest.raises(ValueError, match="generations"):
        evolve_schedules(
            total_layers=TOTAL, search_scorer=lambda s: 0.0,
            holdout_scorer=lambda s: 0.0, generations=0,
        )
    with pytest.raises(ValueError, match="population"):
        evolve_schedules(
            total_layers=TOTAL, search_scorer=lambda s: 0.0,
            holdout_scorer=lambda s: 0.0, population=2,
        )


# ── Execution ───────────────────────────────────────────────────────────


def test_run_schedule_executes_the_program():
    mx = pytest.importorskip("mlx.core")
    pytest.importorskip("mlx_lm")
    from mlx_lm.models.qwen2 import Model, ModelArgs

    from core.learning.schedule_search import run_schedule

    args = ModelArgs(
        model_type="qwen2", hidden_size=32, num_hidden_layers=8,
        intermediate_size=64, num_attention_heads=4, rms_norm_eps=1e-6,
        vocab_size=64, num_key_value_heads=2, max_position_embeddings=128,
        rope_theta=10000.0,
    )
    model = Model(args)
    mx.eval(model.parameters())
    hidden = model.model.embed_tokens(mx.array([[3, 9, 17]]))

    plain = run_schedule(model, hidden, LayerSchedule.vanilla(8))
    repeated = run_schedule(
        model, hidden, LayerSchedule(segments=((0, 4), (2, 6), (6, 8)))
    )
    assert plain.shape == repeated.shape
    assert not bool(mx.allclose(plain, repeated, atol=1e-5)), (
        "a different program must compute a different function"
    )
    with pytest.raises(ValueError, match="does not have"):
        run_schedule(model, hidden, LayerSchedule(segments=((0, 99),)))
