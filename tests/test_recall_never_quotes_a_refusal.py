"""A refusal is not an acknowledgement.

Live, 2026-08-04, on the restarted runtime. Asked "what was the thing I said
I always wanted to learn?", she quoted the sentence back correctly — the
continuity fix working — and then appended:

    and I acknowledged it: "I couldn't get a clear enough answer together,
    and I'd rather say that than hand you something thin."

She had not acknowledged it. She had declined to answer that turn. The
conversation-recall path exists specifically so that what she claims about
the transcript is always true, and it was asserting a response that never
happened.
"""

from __future__ import annotations

import pytest

from interface.routes.chat import _NON_ANSWER_OPENERS, _is_non_answer_surface


@pytest.mark.parametrize("opener", _NON_ANSWER_OPENERS)
def test_every_refusal_surface_is_recognised(opener):
    text = f"{opener}, and I'd rather say that than hand you something thin."
    assert _is_non_answer_surface(text)


def test_a_real_reply_is_not_treated_as_a_refusal():
    assert not _is_non_answer_surface(
        "Physics is a good place to start — waves, then interference."
    )


def test_a_reply_merely_mentioning_difficulty_is_not_a_refusal():
    """Only the built notices count; ordinary hedging is still an answer."""
    assert not _is_non_answer_surface(
        "I couldn't tell you the exact figure, but the order of magnitude is clear."
    )


def test_empty_text_is_not_a_refusal():
    assert not _is_non_answer_surface("")
    assert not _is_non_answer_surface(None)


def test_leading_whitespace_does_not_defeat_the_guard():
    assert _is_non_answer_surface(
        "\n   I couldn't get a clear enough answer together, and I'd rather say that."
    )


def test_the_openers_match_the_notices_the_builder_produces():
    """If those sentences change, this guard must change with them."""
    import inspect

    from interface.routes import chat

    source = inspect.getsource(chat)
    for opener in _NON_ANSWER_OPENERS:
        assert opener in source, f"{opener!r} no longer appears in chat.py"
