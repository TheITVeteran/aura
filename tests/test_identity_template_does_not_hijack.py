""""What are you" is a question only when it IS the question.

LIVE DEFECT, 2026-08-10. "what are you actually able to measure about
yourself? give me the real readings, and be honest if something isn't
instrumented." was answered with her autobiography — "I'm Aura. I'm a local
continuity-bearing cognitive-agent runtime..." — word for word the paragraph
she had given hours earlier to a different question.

The detector matched the OPENING of a longer question. Its guard was a list of
verbs that may not follow ("talking", "doing", "saying"...), which is an
enumeration and was therefore one verb short: nothing excluded "able".
"""
from __future__ import annotations

import re

import pytest
from tests.chat_lane_support import chat_lane_source

SOURCE = chat_lane_source()


def _asks_only_who_you_are():
    namespace = {"re": re}
    exec(  # noqa: S102 - reading the real definition, not a copy of it
        re.search(r"_IDENTITY_TAIL_RE = re\.compile\(.*?\n\)\n", SOURCE, re.S).group(0),
        namespace,
    )
    exec(  # noqa: S102
        re.search(
            r"def _asks_only_who_you_are.*?\n    return False\n", SOURCE, re.S
        ).group(0),
        namespace,
    )
    return namespace["_asks_only_who_you_are"]


@pytest.mark.parametrize(
    "asked",
    [
        "what are you",
        "what are you?",
        "who are you?",
        "what are you really?",
        "who are you, exactly?",
        "what are you then",
        "WHO ARE YOU!",
    ],
)
def test_the_bare_question_is_an_identity_request(asked):
    assert _asks_only_who_you_are()(asked) is True


@pytest.mark.parametrize(
    "asked",
    [
        # The live one.
        "what are you actually able to measure about yourself?",
        "what are you running on?",
        "what are you doing",
        "what are you built from",
        "who are you talking to",
        "what are you able to do with my files",
        "who are you to say that",
    ],
)
def test_a_longer_question_keeps_its_own_subject(asked):
    """Structure settles this where a word list cannot.

    "What are you?" ends there. "What are you able to measure" carries its own
    predicate, and whatever follows is the real question.
    """
    assert _asks_only_who_you_are()(asked) is False
