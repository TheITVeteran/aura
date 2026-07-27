"""An answer cut off before its conclusion is not a partial answer.

Live, 2026-07-27. Asked when a second train catches the first and how far from
the station, she worked it correctly and stopped here:

    "5. Calculate where they meet.
     - The first train has been traveling for 3:00pm + 2.25 "

Recorded as ``desktop_quick_reply_midsentence_cutoff`` and trimmed back to the
last complete sentence, which is the right salvage — but the number was in the
sentence that never arrived, so the user got working and no answer. The same
budget cut a recall answer at "Probably just read".

The budget was 512 tokens because the turn came in on the quick lane. How much
room an answer needs is a property of the question: a two-part derivation is a
two-part derivation whichever lane carries it. The extended band already
existed and was gated behind ``require_full_foreground_mind_reply``, which the
quick lane does not set.

The quick lane still exists for latency, so a question whose shape asks for
room gets a middle band rather than the full one — enough to finish a
derivation, not enough to turn every two-part question into an essay.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from core.brain.cognitive_engine import _turn_wants_a_derivation

SOURCE = Path("core/brain/cognitive_engine.py")


@pytest.mark.parametrize(
    "question",
    [
        "A train leaves at 2:15pm going 60mph. Another leaves the same station "
        "at 3:00pm going 80mph on the same track. When does the second catch "
        "the first, and how far from the station?",
        "Work out how many marbles are left and show your working.",
        "Walk me through how you got that.",
        "Calculate the compound interest over five years.",
        "How long would it take, and how much would it cost?",
        "Which would you choose, and why?",
    ],
)
def test_a_question_that_needs_working_is_recognised(question: str) -> None:
    assert _turn_wants_a_derivation(question)


@pytest.mark.parametrize(
    "question",
    [
        "Morning. What's it actually like in there right now?",
        "What was the very first thing I asked you in this conversation?",
        "What is 17 times 23?",
        "hey",
        "",
    ],
)
def test_an_ordinary_turn_keeps_the_conversational_budget(question: str) -> None:
    """Over-claiming here would slow every turn to pay for a few."""
    assert not _turn_wants_a_derivation(question)


def test_a_very_long_message_is_not_treated_as_a_derivation() -> None:
    """A pasted document is not a request to derive anything."""
    assert not _turn_wants_a_derivation("and how " * 400)


def test_the_shape_is_consulted_independently_of_the_lane() -> None:
    """The bug was the extended band being reachable only from the full lane."""
    src = SOURCE.read_text(encoding="utf-8")
    assert "shape_wants_room = bool(" in src
    assert (
        'extended_full_mind_reply = bool(\n'
        '            context.get("require_full_foreground_mind_reply", False) '
        'and shape_wants_room\n'
        '        )'
    ) in src


def test_the_quick_lane_gets_a_middle_band_not_the_full_one() -> None:
    src = SOURCE.read_text(encoding="utf-8")
    assert "elif shape_wants_room:" in src
    assert "max_tokens = max(896, min(max_tokens, 1536))" in src
    # The full band stays reserved for the full lane.
    assert "max_tokens = max(1024, min(max_tokens, 2048))" in src


def test_the_conversational_floor_is_unchanged_for_everything_else() -> None:
    src = SOURCE.read_text(encoding="utf-8")
    assert "max_tokens = max(512, min(max_tokens, 1024))" in src


def test_tight_contracts_still_win() -> None:
    """Status and inventory answers stay short; they are answered, not derived."""
    src = SOURCE.read_text(encoding="utf-8")
    clamp = src[src.index("if memory_state_contract or runtime_fact_status_contract") :]
    clamp = clamp[: clamp.index("request_timeout =")]
    assert clamp.index("max_tokens = max(128, min(max_tokens, 256))") < clamp.index(
        "elif shape_wants_room:"
    )
