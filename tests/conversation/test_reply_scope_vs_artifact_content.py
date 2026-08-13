"""A file's contents are not the answer, and "reply with just X" means it.

LIVE DEFECT, 2026-08-10 — this is the turn that produced the canned fallback.

Asked to write ~/Documents/aura_selftest.md with three lines (a date, a
subsystem count, the word DONE) and to "reply with just the path you wrote",
the worker rejected six consecutive drafts:

    [WORKER] Rejected live user-surface draft reasons=numeric_answer_missing
    (x6, 21:03:15 through 21:03:35 — twenty seconds of 32B generation)
    CognitiveEngine reply carried no number for a question that can only be
    answered with one (75 chars); refusing it

then the turn collapsed:

    compact desktop generation returned no usable text
    TurnOutcomeError: retryable_failure:retryable_error_and_nothing_served
    CRITICAL SERVICE FAILURE: Subsystem 'cognitive_engine' … fail-closed
    reply_reliability_gate_failed:runtime_boilerplate,missing_requested_line_count

and the person got: "I couldn't get to an answer I'd stand behind on that one …
Ask me again in a moment and I should have it."

The gate read "line two how many subsystems are heartbeating" and concluded the
REPLY could only be a quantity. That number belonged in the file. The reply had
been specified in the same sentence as just the path — which is what she
produced, and what was thrown away six times.
"""

from __future__ import annotations

import pytest
from pathlib import Path


LIVE_MESSAGE = (
    "my mistake on that path - ~/Desktop/Aura is a symlink into your source "
    "tree and you were right to refuse it. try again at "
    "~/Documents/aura_selftest.md instead: exactly three lines, line one "
    "today's date, line two how many subsystems are heartbeating, line three "
    "the single word DONE. reply with just the path you wrote."
)


def _asks(message: str) -> bool:
    from core.conversation.response_reliability import asks_for_a_number

    return asks_for_a_number(message)


def test_the_live_turn_no_longer_demands_a_number() -> None:
    from core.conversation.response_reliability import numeric_answer_missing

    assert _asks(LIVE_MESSAGE) is False
    # The correct answer — the path — must survive the gate.
    assert (
        numeric_answer_missing(LIVE_MESSAGE, str(Path.home() / "Documents" / "aura_selftest.md"))
        is False
    )


@pytest.mark.parametrize(
    "message",
    [
        "write a file at ~/Documents/x.md containing the word DONE",
        "create a note with three lines: one, two, three. reply with just the path",
        "tell me only the filename you used",
        "save a file whose contents are line one 5 and line two 7, then reply with just the path",
    ],
)
def test_artifact_specifications_are_not_questions(message: str) -> None:
    assert _asks(message) is False


@pytest.mark.parametrize(
    "message",
    [
        "what is 17 minus 8, and then times 3",
        "what is the probability of two heads in three flips",
    ],
)
def test_genuine_numeric_questions_still_require_a_number(message: str) -> None:
    """The guard this predicate exists for must be untouched."""
    assert _asks(message) is True


def test_an_explicit_reply_constraint_defines_the_reply_scope() -> None:
    from core.conversation.requested_reply_shape import reply_scope_text as _reply_scope_text

    scope = _reply_scope_text(LIVE_MESSAGE)

    assert "path" in scope
    # The file specification is not part of what the answer must contain.
    assert "how many subsystems" not in scope


def test_a_reply_constraint_asking_for_a_number_still_requires_one() -> None:
    """The suppression must not cut the other way.

    "reply with just the number" is an explicit NUMERIC reply constraint, so
    narrowing the scope to that phrase — which drops the operands — would
    suppress the very guard the person asked for.

    The phrasing here uses operators asks_for_a_number actually recognises;
    "add 14 and 9" is not one of them, which is pre-existing narrowness in
    _NUMERIC_OPERATOR_RE and a separate matter from this suppression.
    """
    message = "what is 17 minus 8, and then times 3 — reply with just the number"

    assert _asks(message) is True


def test_messages_without_a_constraint_keep_their_whole_text() -> None:
    """Only an explicit constraint narrows the scope; nothing else is dropped."""
    from core.conversation.requested_reply_shape import reply_scope_text as _reply_scope_text

    plain = "what is 17 minus 8, and then times 3"

    assert _reply_scope_text(plain).strip() == plain
