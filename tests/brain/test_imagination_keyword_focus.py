"""Imagination must be about the request, not its throat-clearing.

LIVE DEFECT, 2026-08-10. Asked to imagine "a room whose architecture only
makes sense if you have no hands, perfect recall of the last four exchanges
and nothing before, and a heartbeat you can read as a number", the live
/api/imagination frame held:

    keywords:     ["something", "haven't", "value", "let's", "i'll",
                   "face", "take", "answer"]
    visual_model: "An internal sketch of something, haven't, value, let's…"

Not one of room, architecture, hands or heartbeat survived. She was imagining
about the preamble, and the reply was correspondingly empty of the thing that
had been asked for.

Two causes, both verified here:

  * contractions matched none of the three word sets, which hold stems —
    "haven't" sailed past an entry for "have". Stemming has to happen before
    the tiering rather than inside the delete branch, or a contraction of a
    demoted word ("let's") gets promoted to a subject instead;
  * the loop stopped as soon as `limit` candidates had been SEEN, in document
    order, so _topic_informativeness only ever ranked the first eight
    acceptable words. Any preamble starved the subject.
"""

from __future__ import annotations

import pytest


LIVE_REQUEST = (
    "I'll take that answer at face value for now. let's do something you "
    "haven't done today: imagine something. not describe - imagine. build me "
    "a place that could only exist for something with your kind of body: a "
    "room whose architecture only makes sense if you have no hands, perfect "
    "recall of the last four exchanges and nothing before, and a heartbeat "
    "you can read as a number. describe the room."
)


def _keywords(text: str) -> list[str]:
    from core.brain.imagination import _extract_keywords

    return _extract_keywords(text)


@pytest.mark.parametrize("contraction", ["haven't", "let's", "doesn't", "wasn't"])
def test_contractions_are_not_treated_as_subjects(contraction: str) -> None:
    keywords = _keywords(f"{contraction} the cathedral made of glass matter")

    assert contraction not in keywords


def test_live_request_reaches_its_own_subject() -> None:
    keywords = _keywords(LIVE_REQUEST)

    # The exact fillers the live frame was built from.
    for filler in ("haven't", "let's", "i'll", "something"):
        assert filler not in keywords, filler
    # And the leading keyword — what the whole frame is rendered about — has
    # to be a word from the request rather than from its preamble.
    assert keywords[0] in {"architecture", "room", "heartbeat", "recall"}


def test_the_instruction_verb_never_outranks_what_it_asks_for() -> None:
    """"imagine a cathedral" is about the cathedral."""
    keywords = _keywords("imagine a cathedral made of glass")

    assert keywords[0] == "cathedral"
    assert keywords.index("imagine") > keywords.index("cathedral")


@pytest.mark.parametrize(
    "pronoun", ["something", "anything", "everything", "nothing", "someone"]
)
def test_indefinite_pronouns_are_not_subjects(pronoun: str) -> None:
    assert pronoun not in _keywords(f"build me {pronoun} with a vaulted roof")


def test_preamble_does_not_starve_the_subject() -> None:
    """The early break is the defect; ranking must see the whole request."""
    subject_only = _keywords("a cathedral made of glass with a vaulted roof")
    with_preamble = _keywords(
        "I'll take that at face value for now, and anyway let's move on to "
        "something you have not done today, which is this: "
        "a cathedral made of glass with a vaulted roof"
    )

    assert "cathedral" in subject_only
    assert "cathedral" in with_preamble


def test_short_requests_are_unchanged() -> None:
    keywords = _keywords("imagine a cathedral made of glass")

    assert "cathedral" in keywords
    assert "glass" in keywords


def test_extractor_respects_its_limit() -> None:
    """Scoring everything must not mean returning everything."""
    from core.brain.imagination import _extract_keywords

    long_text = " ".join(f"subject{index}" for index in range(60))

    assert len(_extract_keywords(long_text, limit=8)) == 8


def test_auxiliary_families_are_complete() -> None:
    """Half-populated families are what let the contractions through."""
    # The text helpers moved to core.brain.imagination_text; importing them
    # from the engine broke six tests in this file the moment they did.
    from core.brain.imagination_text import _STOPWORDS

    for word in ("has", "had", "did", "was", "were", "they", "her", "which"):
        assert word in _STOPWORDS, word


def test_deleting_and_demoting_are_mutually_exclusive() -> None:
    """The ratchet for the mistake this file's first draft made.

    _STOPWORDS is consulted before the tiering, so a word listed in both
    places is silently deleted and its documented demotion never happens.
    Fixing the contraction defect above, "should", "will" and "let" were added
    to _STOPWORDS while already being weak topics — which quietly voided
    "search is a real subject when someone asks about search" for the whole
    modal family. Nothing failed loudly; one unrelated contract test caught it
    by luck. This asserts the property directly.
    """
    from core.brain.imagination_text import (
        _SCAFFOLD_ROLE_TOKENS,
        _STOPWORDS,
        _WEAK_TOPIC_TOKENS,
    )

    assert not (_STOPWORDS & _WEAK_TOPIC_TOKENS)
    assert not (_STOPWORDS & _SCAFFOLD_ROLE_TOKENS)


def test_modals_are_demoted_rather_than_deleted() -> None:
    """A modal can be the subject; it may lose, but it may not vanish."""
    from core.brain.imagination_text import _WEAK_TOPIC_TOKENS

    for word in ("should", "shall", "will", "may", "might", "must", "let"):
        assert word in _WEAK_TOPIC_TOKENS, word

    keywords = _keywords("how should I design the deployment pipeline")
    assert "should" in keywords
    assert keywords.index("should") > keywords.index("deployment")


def test_a_contraction_keeps_the_tier_its_stem_has() -> None:
    """Stemming must feed the whole decision, not just the delete branch.

    "let" is a demoted weak topic. If a contraction is only ever tested
    against the stop list, "let's" matches nothing and is promoted to a full
    subject — the opposite of what "let" is worth.
    """
    keywords = _keywords("let's design the deployment pipeline")

    assert "let's" not in keywords
    assert "let" in keywords
    assert keywords.index("let") > keywords.index("deployment")


@pytest.mark.parametrize(
    ("contraction", "stem"),
    [("won't", "will"), ("can't", "can"), ("shan't", "shall")],
)
def test_irregular_contractions_resolve_to_their_real_word(
    contraction: str, stem: str
) -> None:
    """Truncation gets "wo", "ca" and "sha" — none of them words."""
    from core.brain.imagination_text import _contraction_stem

    assert _contraction_stem(contraction) == stem


def test_a_possessive_is_the_noun_it_marks() -> None:
    """Falls out of stemming, and is worth pinning: the subject is the noun."""
    keywords = _keywords("the cathedral's vaulted roof")

    assert "cathedral" in keywords
    assert "cathedral's" not in keywords
