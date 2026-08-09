"""Surface wording was deciding the cognitive lane.

`analyze_turn` picks a semantic mode from a first-match `elif` chain over
word lists. "schedule" lives in _PLANNING_PATTERNS, which is checked before
_TECHNICAL_PATTERNS, so

    "help me schedule these jobs to minimise makespan"

routed to the calendar lane — for a combinatorial optimisation question. The
failure is silent: a wrong lane still produces a fluent answer, so nobody
finds out.

CP093 fixed that word. This fixes the SHAPE, because every polysemous term
was otherwise another individual patch: an ambiguous term now votes for a
lane only when that lane's discriminator is present, and ABSTAINS when
nothing discriminates it — abstention being the important half, since the
old behaviour let an unresolved token pick the lane by list order, which is
a coin flip wearing a rule's clothes.
"""
from __future__ import annotations

import re

import pytest

from core.runtime.turn_analysis import (
    _AMBIGUOUS_TERMS,
    _PLANNING_PATTERNS,
    _resolve_ambiguous_lane,
    analyze_turn,
)


def _mode(text: str) -> str:
    result = analyze_turn(text)
    return getattr(result, "semantic_mode", None) or result["semantic_mode"]


# ────────────────────────────────── the same word, two lanes, by context


@pytest.mark.parametrize(
    "text,expected",
    [
        ("help me schedule these jobs to minimise makespan", "technical"),
        ("schedule the workers across the queue to cut latency", "technical"),
        ("can you schedule a meeting with Sam tomorrow", "planning"),
        ("put that on my calendar and schedule an invite", "planning"),
        ("prioritise the backlog for the next milestone", "planning"),
        ("the priority queue preempts the wrong thread", "technical"),
        ("the p99 latency performance regressed after the deploy", "technical"),
        ("my performance review with my manager is next week", "emotional"),
        ("the memory leak is eating 200MB an hour", "technical"),
        ("what do you actually remember, and is your memory continuous", "philosophical"),
    ],
)
def test_context_decides_the_lane_not_list_order(text, expected):
    assert _mode(text) == expected, (
        f"{text!r} routed to {_mode(text)!r}; the word matched a lane list "
        "and the surrounding evidence was ignored"
    )


def test_the_original_cp093_case():
    """The failure that motivated all of this."""
    assert _mode("help me schedule these jobs to minimise makespan") == "technical"


# ─────────────────────────────────────── an unresolved word abstains


def test_a_bare_ambiguous_word_does_not_pick_a_lane():
    """"schedule something" has no discriminator either way.

    The honest outcome is that the word contributes nothing and the rest of
    the turn decides — not that the earliest list in the chain wins.
    """
    lane, abstained = _resolve_ambiguous_lane("schedule something")

    assert lane is None
    assert abstained, "an undiscriminated ambiguous term still voted"


def test_abstention_actually_removes_the_word_from_voting():
    """Otherwise the abstention is cosmetic.

    "schedule" would still sit in _PLANNING_PATTERNS and still win the chain,
    and the register would be decoration.
    """
    assert _mode("schedule something") != "planning"


def test_a_genuinely_mixed_request_does_not_get_forced(monkeypatch):
    """Two lanes both discriminating is not a resolution.

    Picking one would be the same coin flip with extra steps.
    """
    lane, abstained = _resolve_ambiguous_lane(
        "schedule the meeting and also optimise the job queue makespan"
    )

    assert lane is None
    assert abstained


def test_an_unambiguous_turn_is_untouched():
    """The mechanism must not disturb the cases that already worked."""
    assert _mode("there is a traceback in the parser") == "technical"
    assert _mode("what is the roadmap for next quarter") == "planning"
    assert _mode("is there a security vulnerability in this") == "critical"


# ────────────────────────────────────────── the register stays honest


def test_every_ambiguous_term_declares_at_least_two_lanes():
    """A term with one lane is not ambiguous; it is a lane pattern."""
    thin = {
        term: sorted(lanes)
        for term, lanes in _AMBIGUOUS_TERMS.items()
        if len(lanes) < 2
    }

    assert not thin, f"these declare fewer than two lanes: {thin}"


def test_every_lane_has_discriminators():
    """A lane with no discriminators can never fire, so the term can only
    ever abstain — which silently removes it from routing entirely."""
    empty: list[str] = []
    for term, lanes in _AMBIGUOUS_TERMS.items():
        for lane, discriminators in lanes.items():
            if not discriminators:
                empty.append(f"{term} -> {lane}")

    assert not empty, f"lanes with no way to fire: {empty}"


def test_every_declared_pattern_compiles():
    for term, lanes in _AMBIGUOUS_TERMS.items():
        re.compile(term)
        for discriminators in lanes.values():
            for pattern in discriminators:
                re.compile(pattern)


def test_a_word_in_two_lane_lists_is_declared_ambiguous():
    """The structural guard on the class.

    If a token appears in more than one lane list and is NOT in the register,
    the elif order decides it — which is the exact defect. This catches the
    next one at the moment it is added rather than the next time someone
    notices a wrong answer.
    """
    from core.runtime import turn_analysis

    lane_lists = {
        name: getattr(turn_analysis, name)
        for name in dir(turn_analysis)
        if name.endswith("_PATTERNS") and isinstance(getattr(turn_analysis, name), tuple)
    }

    seen: dict[str, list[str]] = {}
    for name, patterns in lane_lists.items():
        for pattern in patterns:
            seen.setdefault(pattern, []).append(name)

    collisions = {
        pattern: names for pattern, names in seen.items() if len(names) > 1
    }
    undeclared = {
        pattern: names
        for pattern, names in collisions.items()
        if not any(pattern in term for term in _AMBIGUOUS_TERMS)
    }

    assert not undeclared, (
        f"these patterns appear in multiple lane lists and are not in "
        f"_AMBIGUOUS_TERMS, so the elif order decides them: {undeclared}"
    )


def test_schedule_is_still_in_the_planning_list():
    """The fix must not work by deleting the word.

    Removing "schedule" from _PLANNING_PATTERNS would make the calendar case
    fall through to casual — trading one wrong lane for another while the
    test suite went green.
    """
    assert any("schedule" in pattern for pattern in _PLANNING_PATTERNS)
