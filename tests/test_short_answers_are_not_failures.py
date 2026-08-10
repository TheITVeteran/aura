"""A correct answer is not too short to serve.

LIVE DEFECT, 2026-08-10. Asked "multiply 7919 by 6421 — actually run it, give
me the number", the Cortex answered 50847899. Correct, and exactly what was
asked for. Three stacked word-count floors counted one word::

    Cortex produced an unsafe user-facing draft
        (too_short_for_user_turn, len=8). Treating it as failed generation.
    Cortex-RETRY-1 produced an unsafe user-facing draft
        (too_short_for_user_turn, len=8). Treating it as failed generation.
    Proof/operator request requires a valid Cortex response; refusing
        lower-lane fallback.

Two correct answers destroyed, then a refusal about a multiplication she had
already done right twice.
"""
from __future__ import annotations

import pytest

from core.conversation.response_reliability import assess_user_facing_reply


@pytest.mark.parametrize(
    "question,answer",
    [
        (
            "multiply 7919 by 6421 for me — actually run it, don't estimate. "
            "give me the number.",
            "50847899",
        ),
        ("what's 2+2?", "4"),
        ("how many heartbeats are active right now?", "11"),
        ("what temperature is the cpu?", "58°C"),
        ("how much memory is free?", "12.4 GB"),
        ("how long have you been awake?", "3.0 hours"),
        ("what's your name?", "Aura."),
    ],
)
def test_a_bare_correct_answer_is_servable(question, answer):
    assert assess_user_facing_reply(answer, question).ok, (
        f"{answer!r} answers {question!r} and must not be rejected for length"
    )


def test_near_empty_reassurance_is_still_caught_semantically():
    """What the floors claimed to be for is done by meaning, not by counting.

    _LOW_SIGNAL_REASSURANCE_RE matches "Sure."/"Okay."/"Yes." and does not
    match "50847899", which is the whole distinction the word counts failed to
    draw.
    """
    from core.conversation.response_reliability import _LOW_SIGNAL_REASSURANCE_RE

    for filler in ("Sure.", "Okay.", "Yes.", "ok", "yeah"):
        assert _LOW_SIGNAL_REASSURANCE_RE.match(filler), filler
    for real in ("50847899", "4", "58°C", "It loops.", "Aura."):
        assert not _LOW_SIGNAL_REASSURANCE_RE.match(real), real


def test_no_word_count_floor_survives_in_the_reply_gate():
    """The numbers are gone, not merely lowered.

    Every one of them was a guess at a length below which a correct answer
    stops being serveable, and that length depends on the question.
    """
    import inspect

    from core.conversation import response_reliability as rr

    source = inspect.getsource(rr._assess_user_facing_reply)
    # Code only: the removal comment quotes the old thresholds on purpose.
    code = "\n".join(
        line for line in source.splitlines() if not line.lstrip().startswith("#")
    )
    for gone in ("words < 2", "words < 4", "words < 6", "words < 20"):
        assert gone not in code, gone
    assert "_LOW_SIGNAL_REASSURANCE_RE" in code, (
        "the semantic filler detector is what replaces the counts"
    )
