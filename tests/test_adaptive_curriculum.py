"""Sampling where learning is possible (CP237).

GRPO's failure: on tasks the model never solves, every group is all-wrong
and the run burns compute over no signal. The minimax curriculum samples
where the model is weakest but not hopeless, and escalates as it improves.
"""
from __future__ import annotations

import random

import pytest

from core.learning.adaptive_curriculum import (
    AdaptiveCurriculum,
    CellStats,
    warm_start_pass_rates,
)


def test_hopeless_and_saturated_cells_score_zero_learnability():
    """The two degenerate extremes: nothing to reinforce, nothing to
    improve."""
    hopeless = CellStats("f", 8, trials=10, reward_sum=0.0)
    saturated = CellStats("f", 1, trials=10, reward_sum=10.0)
    mid = CellStats("f", 4, trials=10, reward_sum=5.0)
    assert hopeless.learnability() == 0.0
    assert saturated.learnability() == 0.0
    assert mid.learnability() > 0.0


def test_minimax_prefers_the_weaker_of_two_learnable_cells():
    """Given two cells with reward variance, the harder one is preferred."""
    easier = CellStats("f", 2, trials=20, reward_sum=14.0)  # 0.70
    harder = CellStats("f", 4, trials=20, reward_sum=6.0)   # 0.30
    assert harder.learnability() > easier.learnability()


def test_unexplored_cells_get_an_optimistic_prior():
    """A cell tried fewer than twice must be attempted before being written
    off -- optimism under uncertainty."""
    fresh = CellStats("f", 8, trials=1, reward_sum=0.0)
    assert fresh.learnability() == pytest.approx(0.3)


def test_sampler_avoids_hopeless_cells():
    curriculum = AdaptiveCurriculum.over(["f"], [1, 4, 8])
    # 1 is saturated, 8 is hopeless, 4 is learnable.
    for _ in range(10):
        curriculum.observe("f", 1, 1.0, degenerate=True)
        curriculum.observe("f", 8, 0.0, degenerate=True)
        curriculum.observe("f", 4, 0.4, degenerate=False)
    rng = random.Random(0)
    picks = [curriculum.sample(rng) for _ in range(200)]
    assert all(difficulty == 4 for _family, difficulty in picks), (
        "the sampler must not spend steps on all-wrong or all-correct cells"
    )


def test_escalation_when_an_easy_cell_saturates():
    """A saturated cell stops being sampled; the harder cell, now reachable,
    takes over -- difficulty rises with competence."""
    curriculum = AdaptiveCurriculum.over(["f"], [1, 2])
    for _ in range(10):
        curriculum.observe("f", 1, 1.0, degenerate=True)   # mastered
        curriculum.observe("f", 2, 0.35, degenerate=False)  # now learnable
    rng = random.Random(1)
    picks = [d for _f, d in (curriculum.sample(rng) for _ in range(100))]
    assert set(picks) == {2}


def test_total_hopelessness_falls_back_to_least_explored_not_a_dead_cell():
    curriculum = AdaptiveCurriculum.over(["f"], [4, 8])
    for _ in range(10):
        curriculum.observe("f", 4, 0.0, degenerate=True)
    # cell 8 is unexplored; it must be preferred over the known-dead cell 4
    rng = random.Random(2)
    picks = [d for _f, d in (curriculum.sample(rng) for _ in range(50))]
    assert 8 in picks


def test_report_names_the_frontier_state():
    curriculum = AdaptiveCurriculum.over(["f"], [1, 4, 8])
    for _ in range(10):
        curriculum.observe("f", 1, 1.0, degenerate=True)
        curriculum.observe("f", 4, 0.4, degenerate=False)
        curriculum.observe("f", 8, 0.0, degenerate=True)
    report = curriculum.report()
    assert report["saturated"] == ["f@1"]
    assert report["learnable"] == ["f@4"]
    assert report["hopeless"] == ["f@8"]
    assert report["has_reachable_frontier"] is True


def test_exhausted_frontier_is_reported_not_hidden():
    curriculum = AdaptiveCurriculum.over(["f"], [4])
    for _ in range(10):
        curriculum.observe("f", 4, 0.0, degenerate=True)
    assert curriculum.report()["has_reachable_frontier"] is False


def test_state_round_trips_so_a_resumed_run_keeps_its_map():
    curriculum = AdaptiveCurriculum.over(["f", "g"], [2, 4])
    for _ in range(5):
        curriculum.observe("f", 2, 0.5, degenerate=False)
        curriculum.observe("g", 4, 0.1, degenerate=False)
    restored = AdaptiveCurriculum.from_state(curriculum.state())
    assert restored.cells[("f", 2)].pass_rate == pytest.approx(0.5)
    assert restored.cells[("g", 4)].trials == 5


def test_warm_start_measures_before_training():
    def measure(family, difficulty):
        return {1: 0.9, 4: 0.3, 8: 0.0}[difficulty]

    curriculum = warm_start_pass_rates(["f"], [1, 4, 8], measure, samples_per_cell=3)
    report = curriculum.report()
    assert "f@4" in report["learnable"]
    assert "f@8" in report["hopeless"]


def test_bad_reward_is_refused():
    with pytest.raises(ValueError, match="rate in"):
        CellStats("f", 1).observe(1.5, degenerate=False)
    with pytest.raises(ValueError, match="families and difficulties"):
        AdaptiveCurriculum.over([], [1])
