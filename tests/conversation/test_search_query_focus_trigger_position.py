"""A trailing "look it up" is an instruction, not the thing to look up.

LIVE DEFECT, 2026-08-10. Asked:

    "obscure one for you: the curator Michael T. Wright built a working
     planetarium model to show the Antikythera mechanism could have done it.
     he revised Derek de Solla Price's gear count upward — from what number to
     what number? if you dont actually know, look it up rather than
     estimating."

she autonomously dispatched web_search (correctly — capability token minted,
intention declared, 23.1s round trip) and came back with:

    "I checked live web evidence. You (TV series)
     Wikipedia: I'm not sure what you're asking. Could you please clarify…
     Source: https://en.wikipedia.org/wiki/You_(TV_series)"

The query it searched for was, verbatim from the log:

    'if you dont actually know, look it up rather than estimating'

_strip_search_preamble treats a search trigger as INTRODUCING the subject, so
it keeps the request from the trigger's clause onward and discards everything
before. That is right for "can you look up X" and exactly wrong when the
trigger is in the final clause, where it says how to answer rather than what
to find. The remaining pronoun "you" then resolved to a real Wikipedia article
and the reply cited it.
"""

from __future__ import annotations

import pytest


LIVE_MESSAGE = (
    "obscure one for you: the curator Michael T. Wright built a working "
    "planetarium model to show the Antikythera mechanism could have done it. "
    "he revised Derek de Solla Price's gear count upward - from what number "
    "to what number? if you dont actually know, look it up rather than "
    "estimating."
)


def _focus(text: str) -> str:
    from core.phases.response_contract import extract_search_query_focus

    return extract_search_query_focus(text)


def test_the_live_message_searches_for_its_subject() -> None:
    focus = _focus(LIVE_MESSAGE)

    assert "Antikythera" in focus
    # The instruction clause must not BE the query.
    assert focus.strip().lower() != (
        "if you dont actually know, look it up rather than estimating"
    )


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("what is the tsar bomba yield? look it up if you are not sure", "tsar bomba yield"),
        ("who won the 1998 world cup? search if you have to", "who won the 1998 world cup"),
    ],
)
def test_trailing_trigger_keeps_the_question(message: str, expected: str) -> None:
    assert _focus(message).lower() == expected


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("can you look up the Antikythera mechanism gear count", "the Antikythera mechanism gear count"),
        ("search for the Liskov substitution principle", "the Liskov substitution principle"),
        ("find out who invented the Antikythera mechanism", "who invented the Antikythera mechanism"),
        ("hey, quick thing: google the Kessler syndrome", "the Kessler syndrome"),
    ],
)
def test_leading_trigger_still_introduces_the_subject(
    message: str, expected: str
) -> None:
    """The behaviour this function was written for must be untouched."""
    assert _focus(message) == expected


def test_a_bare_trigger_is_left_alone() -> None:
    """"look it up" with no subject is still a search request."""
    assert _focus("look it up") == "look it up"


def test_a_short_preamble_does_not_swallow_the_trigger() -> None:
    """Guard the >=3 word floor: "ok, search X" is trigger-first, not trailing."""
    assert "quantum" in _focus("ok, search quantum tunnelling")


def test_the_trailing_clause_pattern_is_anchored() -> None:
    """It must match a whole trailing clause, never a mid-sentence mention."""
    from core.phases.response_contract import (
        _SEARCH_TRAILING_TRIGGER_CLAUSE_RE as pattern,
    )

    assert pattern.fullmatch("if you dont actually know, look it up rather than estimating.")
    assert pattern.fullmatch("look it up if you are not sure")
    # A subject that merely contains the word "research" is not an instruction.
    assert not pattern.fullmatch("research chemicals used in semiconductors")
