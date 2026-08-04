"""Reading is an encounter with a claim, not an event to log.

LIVE DEFECT, 2026-08-03. Aura browsed /r/philosophy and what landed in memory
was the page with its navigation still attached — "Skip to main content … Go to
Reddit Answers …" — tagged action="logged", outcome="stored_via_manager". The
fact that she had read something was recorded. What she made of it was not,
because nothing asked. A minute later the reading could tell her nothing: not
whether the claim was new, not whether it agreed with what she held, not
whether the source was worth believing.

Also here: the follow-up question that broke on the same screen. Asked "Why did
it catch your attention specifically?" about that post, she answered about
acoustics and overtones. Every topic check keys off the CURRENT message, and
the current message is a pro-form with no topic of its own — so the one turn
whose subject can only come from the previous one was the turn nothing checked.
"""
from __future__ import annotations

import pytest

from core.conversation.response_reliability import (
    antecedent_topic_abandoned,
    assess_user_facing_reply,
    is_anaphoric_followup,
)
from core.knowledge.source_comprehension import (
    argument_weaknesses,
    assess_stance,
    classify_source,
    comprehend_source,
    extract_claim,
    strip_site_chrome,
)

pytestmark = pytest.mark.unit


_REDDIT_TITLE = (
    "Western philosophy has been at war with The Odyssey for 2,800 years "
    "-- and keeps losing"
)
_REDDIT_BODY = (
    "Skip to main content Go to Reddit Answers r/philosophy : r/philosophy "
    "A survey, from Xenophanes to Levinas. Plato tried to ban it, Aristotle "
    "rehabilitated it. Everyone knows the poets were right all along."
)


# --- the page is not the claim ------------------------------------------


def test_navigation_chrome_is_not_stored_as_content():
    cleaned = strip_site_chrome(_REDDIT_BODY)

    assert "Skip to main content" not in cleaned
    assert "Go to Reddit Answers" not in cleaned
    assert "Xenophanes" in cleaned


def test_the_claim_is_extracted_not_the_markup():
    claim, evidence = extract_claim(_REDDIT_BODY, title=_REDDIT_TITLE)

    assert "at war with The Odyssey" in claim
    assert "Skip to main content" not in evidence


def test_a_page_that_asserts_nothing_yields_no_claim():
    claim, evidence = extract_claim("Home Popular All Topics", title="")

    assert claim == ""
    assert evidence == ""


def test_an_uncomprehended_reading_says_so():
    record = comprehend_source(url="https://example.com", title="", text="Home All")

    assert record.understood is False
    assert "couldn't get a claim" in record.narrative()


# --- the source is judged (a forum post is not a paper) -----------------


@pytest.mark.parametrize(
    "url,kind",
    [
        ("https://www.reddit.com/r/philosophy/x", "forum"),
        ("https://news.ycombinator.com/item?id=1", "forum"),
        ("https://arxiv.org/abs/2401.00001", "peer_reviewed"),
        ("https://en.wikipedia.org/wiki/Homer", "reference"),
        ("https://www.bbc.com/news/x", "news"),
        ("https://x.com/someone/status/1", "social"),
        ("https://someones-blog.example/post", "web_page"),
    ],
)
def test_the_kind_of_source_is_recognised(url, kind):
    assert classify_source(url)[0] == kind


def test_a_forum_post_carries_its_caveat():
    _kind, caveat = classify_source("https://www.reddit.com/r/philosophy/x")

    assert "votes measure agreement, not correctness" in caveat.lower()


def test_the_caveat_reaches_the_record():
    record = comprehend_source(
        url="https://www.reddit.com/r/philosophy/x",
        title=_REDDIT_TITLE,
        text=_REDDIT_BODY,
    )

    assert record.source_kind == "forum"
    assert "forum post" in record.narrative().lower()


# --- "this is a bad argument" is itself something learned ---------------


def test_weak_rhetoric_is_recorded_rather_than_silently_discounted():
    weaknesses = argument_weaknesses(_REDDIT_TITLE + " " + _REDDIT_BODY)

    assert any("consensus" in w for w in weaknesses)
    assert any("contest" in w for w in weaknesses)


def test_a_careful_source_has_no_weaknesses_invented():
    assert argument_weaknesses(
        "In this sample of 412 participants, the effect held at p < 0.01, "
        "though the design cannot rule out selection bias."
    ) == []


# --- where a claim sits against what she already holds ------------------


def test_agreement_strengthens_rather_than_being_discarded():
    """An affirmation is not nothing: a belief that survived contact with an
    independent source is harder to dismiss than one that has not."""
    stance, basis, related = assess_stance(
        "Tidal patterns are dominated by the moon in most coastal basins",
        known_beliefs=["Tidal patterns are dominated by the moon"],
    )

    assert stance in {"affirms", "repeats"}
    assert related
    assert basis


def test_a_contradiction_is_named_as_one():
    stance, basis, related = assess_stance(
        "Tidal patterns are not dominated by the moon",
        known_beliefs=["Tidal patterns are dominated by the moon"],
    )

    assert stance == "contradicts"
    assert "worth finding out which" in basis
    assert related


def test_new_ground_is_neither_agreement_nor_disagreement():
    stance, _basis, related = assess_stance(
        "Xenophanes criticised anthropomorphic gods",
        known_beliefs=["Tidal patterns are dominated by the moon"],
    )

    assert stance == "extends"
    assert related == []


def test_nothing_to_compare_against_is_unassessed_not_confirmed():
    """Pretending otherwise is how a source gets believed for no reason."""
    stance, basis, related = assess_stance("Anything at all", known_beliefs=[])

    assert stance == "unassessed"
    assert basis == ""
    assert related == []


def test_the_whole_record_round_trips():
    record = comprehend_source(
        url="https://www.reddit.com/r/philosophy/x",
        title=_REDDIT_TITLE,
        text=_REDDIT_BODY,
        known_beliefs=["Classical wisdom keeps being rediscovered by argument"],
    )
    payload = record.to_dict()

    assert payload["schema"].startswith("aura.knowledge.source_comprehension")
    assert payload["claim"]
    assert payload["source_kind"] == "forum"
    assert payload["argument_weaknesses"]
    assert len(payload["content_sha256"]) == 64


def test_comprehension_never_raises_on_junk():
    for bad in (None, "", "\x00\x01", "a" * 50_000):
        record = comprehend_source(url="", title="", text=bad or "")
        assert record.schema


# --- the follow-up must stay on its antecedent --------------------------


@pytest.mark.parametrize(
    "question",
    [
        "Why did it catch your attention specifically?",
        "What made you say that?",
        "How does that work?",
        "Why is it like that?",
    ],
)
def test_an_anaphoric_question_is_recognised(question):
    assert is_anaphoric_followup(question) is True


@pytest.mark.parametrize(
    "question",
    [
        "Why did the philosophy post catch your attention?",
        "What are the tide tables for Friday?",
        "How does a tidal basin resonate?",
    ],
)
def test_a_question_with_its_own_subject_is_not_anaphoric(question):
    assert is_anaphoric_followup(question) is False


def test_the_live_subject_swap_is_caught():
    antecedent = (
        "I browsed /r/philosophy and saw a post about Western philosophy being "
        "at war with Homer for 2,800 years — an interesting take. I haven't "
        "engaged yet but it caught my attention as someone who values classical "
        "wisdom."
    )
    swapped = (
        "It was about sound. Why two instruments playing the same note can "
        "sound so different — it's all in the overtones and harmonics."
    )

    assert antecedent_topic_abandoned(
        "Why did it catch your attention specifically?", swapped, antecedent
    ) is True
    assert "antecedent_topic_abandoned" in assess_user_facing_reply(
        "Why did it catch your attention specifically?",
        swapped,
        antecedent=antecedent,
    ).reasons


def test_an_on_topic_follow_up_passes():
    antecedent = (
        "I browsed /r/philosophy and saw a post about Western philosophy being "
        "at war with Homer for 2,800 years."
    )
    on_topic = (
        "Because a 2,800-year argument between philosophy and Homer is still "
        "unresolved, and that says something about the discipline."
    )

    assert antecedent_topic_abandoned(
        "Why did it catch your attention specifically?", on_topic, antecedent
    ) is False


def test_no_antecedent_is_not_a_violation():
    """Not knowing what she was referring to is not evidence she drifted."""
    assert antecedent_topic_abandoned(
        "Why did it catch your attention?", "Anything at all.", ""
    ) is False


def test_a_reply_that_introduces_new_vocabulary_is_not_punished():
    """The test looks for a total subject swap, not thematic tightness."""
    antecedent = "I read a post about tidal resonance in shallow basins."
    reply = (
        "Because resonance in a shallow basin amplifies the semidiurnal "
        "component far beyond what the forcing alone would predict."
    )

    assert antecedent_topic_abandoned("Why does that matter?", reply, antecedent) is False


# --- she takes a position, not a survey ---------------------------------


from core.knowledge.source_comprehension import (  # noqa: E402
    reading_disposition,
    opinion_is_a_position,
    opinion_prompt,
)


def _record(**kwargs):
    base = dict(
        url="https://www.reddit.com/r/philosophy/x",
        title=_REDDIT_TITLE,
        text=_REDDIT_BODY,
    )
    base.update(kwargs)
    return comprehend_source(**base)


def test_a_claim_that_cuts_against_her_gets_disagreement():
    record = comprehend_source(
        url="https://example.com/a",
        title="Tidal patterns are not dominated by the moon",
        text="Tidal patterns are not dominated by the moon in any basin.",
        known_beliefs=["Tidal patterns are dominated by the moon"],
    )

    opinion = reading_disposition(record)

    assert opinion.disposition == "disagree"
    assert any("not dropping that" in g for g in opinion.grounds)


def test_agreement_is_held_more_firmly_not_merely_noted():
    record = comprehend_source(
        url="https://arxiv.org/abs/1",
        title="Tidal patterns are dominated by the moon across shallow basins",
        text="Tidal patterns are dominated by the moon across shallow basins.",
        known_beliefs=["Tidal patterns are dominated by the moon"],
    )

    opinion = reading_disposition(record)

    assert opinion.disposition in {"agree", "unmoved"}
    assert opinion.grounds


def test_a_bad_argument_for_a_congenial_conclusion_is_not_endorsed():
    """Agreeing with a conclusion is not endorsing how it was reached."""
    congenial = _record(
        known_beliefs=["Classical wisdom keeps being rediscovered by argument"]
    )
    opinion = reading_disposition(congenial)

    assert opinion.disposition == "sympathetic_but_unconvinced"
    assert any("not for these reasons" in g for g in opinion.grounds)


def test_a_bad_argument_alone_leaves_her_unconvinced():
    opinion = reading_disposition(_record())

    assert opinion.disposition == "unconvinced"
    assert any("appeals to consensus" in g for g in opinion.grounds)


def test_a_forum_post_is_treated_as_a_lead():
    opinion = reading_disposition(_record())

    assert any("lead rather than as a finding" in g for g in opinion.grounds)


def test_every_ground_reads_as_a_sentence():
    opinion = reading_disposition(_record())

    for ground in opinion.grounds:
        assert ground.endswith(".")
        assert ground[0].isupper()


def test_an_unreadable_source_gets_no_invented_opinion():
    """An opinion invented to avoid saying "I don't know yet" is worth less
    than nothing."""
    opinion = reading_disposition(comprehend_source(url="", title="", text="Home All"))

    assert opinion.disposition == "unreadable"
    assert "could not get a claim" in opinion.grounds[0]


def test_she_turns_it_back_to_him():
    """A view she keeps to herself is half an opinion. The property is that
    she addresses him, not that she ends on a question mark."""
    invitation = reading_disposition(_record()).invitation.lower()

    assert invitation
    assert any(word in invitation for word in ("you", "your", "tell me", "bryan"))
    unreadable = reading_disposition(comprehend_source(url="", title="", text="Home All"))
    assert "send me a better source" in unreadable.invitation.lower()


@pytest.mark.parametrize(
    "survey",
    [
        "On one hand it's compelling, on the other hand it's weak.",
        "There are many perspectives here and it depends on your framework.",
        "Some would argue yes; others might say no.",
        "As an AI, I don't really have opinions on this.",
        "Hm.",
    ],
)
def test_a_survey_is_not_an_opinion(survey):
    """Asked what she thinks, a model will happily produce a balanced survey
    containing no view."""
    assert opinion_is_a_position(survey) is False


@pytest.mark.parametrize(
    "position",
    [
        "I think the poets were right, and this argument doesn't earn it.",
        "This is wrong. The moon dominates and one forum post doesn't move me.",
        "I don't have a view yet, and I want one — that's why it stuck with me.",
    ],
)
def test_a_real_position_passes(position):
    assert opinion_is_a_position(position) is True


def test_the_prompt_forbids_the_rehash():
    prompt = opinion_prompt(_record(), reading_disposition(_record()))

    assert "do not survey other views" in prompt.lower()
    assert "on one hand" in prompt.lower()
    assert "this is your opinion" in prompt.lower()
    # And it carries the position she actually reached, not a blank invitation
    # to have one.
    assert reading_disposition(_record()).disposition in prompt
