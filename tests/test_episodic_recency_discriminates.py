"""Episodic ranking must be able to tell recent memories from old ones.

Before this, ``EpisodicMemory._recency_score`` was
``min(1.0, max(0.0, ep.timestamp - 1774000000) / 2000000)`` — position against
a hardcoded epoch rather than elapsed time. Run on 2026-08-12 it returned
exactly 1.000000 for everything newer than 2026-04-12, so the recency half of
``_static_rank`` was a constant and the ranking was importance-only. The window
also recedes: the function degrades further every day it is left in place.

These tests assert the properties that rule out the whole class, not a repaired
constant. ``test_ranking_is_stable_under_epoch_shift`` is the direct one — it
runs the same relative history at wall-clock times years apart and requires the
same order.
"""

from __future__ import annotations

import time

import pytest

from core.memory.episodic_memory import Episode, EpisodicMemory

HOUR = 3600.0
DAY = 86400.0


def _episode(eid: str, *, age_s: float, now: float, importance: float = 0.5,
             access_count: int = 0, last_access_age_s: float | None = None) -> Episode:
    last = 0.0 if last_access_age_s is None else now - last_access_age_s
    return Episode(
        id=eid,
        timestamp=now - age_s,
        importance=importance,
        access_count=access_count,
        last_accessed=last,
    )


def _rank(episodes: list[Episode], now: float | None = None) -> list[str]:
    ranked = EpisodicMemory._static_rank(
        EpisodicMemory.__new__(EpisodicMemory), episodes, now
    )
    return [ep.episode_id for ep in ranked]


def test_recent_outranks_old_at_equal_importance():
    """The regression. Both of these previously scored recency 1.000000."""
    now = time.time()
    episodes = [
        _episode("month", age_s=30 * DAY, now=now),
        _episode("minute", age_s=60.0, now=now),
    ]
    assert _rank(episodes)[0] == "minute"


def test_ranking_is_stable_under_epoch_shift():
    """Identical relative histories must rank identically whenever they happen.

    The old scorer failed this outright: shift the same episodes far enough
    into the past and every score collapses to 0.0.
    """
    orders = []
    for now in (1_700_000_000.0, 1_786_589_186.0, 1_900_000_000.0):
        orders.append(
            _rank(
                [
                    _episode("old", age_s=90 * DAY, now=now),
                    _episode("mid", age_s=2 * DAY, now=now),
                    _episode("new", age_s=5 * 60.0, now=now),
                ],
                now,
            )
        )
    assert orders[0] == orders[1] == orders[2] == ["new", "mid", "old"]


def test_the_recency_term_is_not_a_constant():
    """Distinct ages must produce distinct activations across the live range."""
    now = time.time()
    scores = {
        label: EpisodicMemory._recency_score(_episode(label, age_s=age, now=now), now)
        for label, age in (
            ("minute", 60.0),
            ("hour", HOUR),
            ("day", DAY),
            ("week", 7 * DAY),
            ("month", 30 * DAY),
        )
    }
    assert len(set(scores.values())) == len(scores), f"ages collapsed together: {scores}"
    ordered = [scores[k] for k in ("minute", "hour", "day", "week", "month")]
    assert ordered == sorted(ordered, reverse=True), scores


def test_frequent_recall_strengthens_an_older_trace():
    """Frequency is part of activation; the old scorer could not see it at all."""
    now = time.time()
    rehearsed = _episode(
        "rehearsed", age_s=10 * DAY, now=now, access_count=12, last_access_age_s=HOUR
    )
    stored_once = _episode("stored_once", age_s=2 * DAY, now=now)
    assert EpisodicMemory._recency_score(rehearsed, now) > EpisodicMemory._recency_score(
        stored_once, now
    )


def test_importance_still_carries_its_share():
    """Activation must not swamp importance — the 0.6/0.4 blend is unchanged."""
    now = time.time()
    episodes = [
        _episode("trivial_but_fresh", age_s=30.0, now=now, importance=0.0),
        _episode("vital_but_old", age_s=30 * DAY, now=now, importance=1.0),
    ]
    # 0.6*1.0 + 0.4*0.0 = 0.60 beats 0.6*0.0 + 0.4*1.0 = 0.40
    assert _rank(episodes)[0] == "vital_but_old"


def test_single_and_empty_candidate_sets_are_returned_intact():
    now = time.time()
    one = [_episode("solo", age_s=DAY, now=now)]
    assert _rank(one) == ["solo"]
    assert _rank([]) == []


def test_identical_episodes_rank_deterministically():
    """Ties must not depend on Episode being orderable — it is not."""
    now = time.time()
    episodes = [_episode(f"e{i}", age_s=DAY, now=now, importance=0.5) for i in range(4)]
    first = _rank(episodes)
    assert first == _rank(episodes)
    assert sorted(first) == ["e0", "e1", "e2", "e3"]


@pytest.mark.parametrize("age", [0.0, -5.0])
def test_a_just_written_episode_does_not_break_ranking(age: float):
    """Zero or negative age (clock skew) must not produce inf and must rank."""
    now = time.time()
    episodes = [
        _episode("justnow", age_s=age, now=now),
        _episode("older", age_s=DAY, now=now),
    ]
    assert _rank(episodes)[0] == "justnow"
