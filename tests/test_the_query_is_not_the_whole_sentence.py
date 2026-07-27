"""What she searches for should be the question, not the message.

Measured live 2026-07-27. Asked:

    "Now something outside yourself: look up who won the most recent Formula 1
     world championship and tell me where you got it."

what reached the search engine was that entire sentence — preamble, request and
the instruction addressed to her, verbatim. It found something, which is the
worst version of this failure: a bad query that returns results looks like it
worked, so nothing downstream ever reports a problem.

The cause is that every extraction pattern is anchored with ``.match()``. They
handle "search for X and tell me Y" perfectly and are defeated entirely by
anything in front of the trigger, at which point the extractor falls through to
using the whole message.

Two trims, both conservative. A conversational preamble ending at a clause
break before the trigger is dropped, and a trailing instruction addressed to
*her* rather than to the search engine is dropped. Neither invents terms, and a
message with no preamble is untouched.
"""
from __future__ import annotations

import pytest

from core.phases.response_contract import extract_search_query_focus


def test_the_live_failure_is_fixed() -> None:
    query = extract_search_query_focus(
        "Now something outside yourself: look up who won the most recent "
        "Formula 1 world championship and tell me where you got it."
    )
    assert query == "who won the most recent Formula 1 world championship"


@pytest.mark.parametrize(
    "preamble",
    [
        "Now something outside yourself: ",
        "Okay. ",
        "One more thing — ",
        "Right, next question. ",
        "Aura, ",
    ],
)
def test_a_preamble_never_reaches_the_engine(preamble: str) -> None:
    query = extract_search_query_focus(f"{preamble}look up the population of Tokyo")
    assert "Tokyo" in query
    for word in ("outside", "Okay", "next question"):
        assert word not in query


@pytest.mark.parametrize(
    "instruction",
    [
        "and tell me where you got it",
        "then tell me what you find",
        "and cite your sources",
        "and summarise it for me",
        "and let me know",
    ],
)
def test_an_instruction_to_her_is_not_a_search_term(instruction: str) -> None:
    """"Cite your sources" is a requirement on the answer, not a query term."""
    query = extract_search_query_focus(
        f"look up the population of Tokyo {instruction}"
    )
    assert "Tokyo" in query
    assert "cite" not in query.lower()
    assert "tell me" not in query.lower()


# ── Conservative: nothing that already worked may break ───────────────────

@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("look up the population of Tokyo", "the population of Tokyo"),
        ("what is the weather in Paris right now?", "weather in Paris"),
        (
            "can you search the web and find me a good recipe for carbonara please",
            "carbonara",
        ),
    ],
)
def test_requests_that_already_worked_still_work(message: str, expected: str) -> None:
    assert extract_search_query_focus(message) == expected


def test_a_quoted_query_is_taken_exactly() -> None:
    """An explicit quote is the user being precise; it must survive intact."""
    assert (
        extract_search_query_focus('Search for "quantum error correction 2026" and summarise it')
        == "quantum error correction 2026"
    )


def test_a_url_still_wins() -> None:
    query = extract_search_query_focus("have a look at https://example.com/page for me")
    assert query == "https://example.com/page"


def test_a_message_with_no_trigger_is_untouched_by_the_preamble_trim() -> None:
    """The trim keys on a search trigger; without one it must do nothing."""
    assert "carbonara" in extract_search_query_focus("a good recipe for carbonara")


def test_an_empty_message_yields_no_query() -> None:
    assert extract_search_query_focus("") == ""
    assert extract_search_query_focus("   ") == ""


def test_the_query_stays_bounded() -> None:
    assert len(extract_search_query_focus("look up " + "word " * 200)) <= 180
