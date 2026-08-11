"""A factual question never consulted the encyclopedia sitting on disk.

LIVE, 2026-08-10. "Who was Grace Hopper and what specifically did she build?
... tell me where you got it." was answered from model weights and signed
"Source: Wikipedia". No tool ran on that turn. The corpus holds a Grace Hopper
page and returns it in ~13ms, so the attribution was generated to match what
the answer WOULD have come from if anything had looked.

Three separate faults, and only the first was known:

1. LANE, not question. The corpus was reachable only through
   assemble_cognitive_ingress, which runs on the latent lane. That turn took
   the fast path, which had no ingress, so whether she could consult an
   encyclopedia depended on which lane the turn was routed to.

2. THE QUERY. Reached at all, it was reached with the whole sentence. The
   store builds an FTS query from every word with AND semantics and falls back
   to any-term when nothing matches — which a natural question always does. So
   "Who was Grace Hopper and what specifically did she build?" fell through to
   the OR pass, where who/was/and/what/did outweigh the two words carrying the
   question, and the top three hits were "History of software", "Terminator:
   Dark Fate" and "Vassar College". Confident irrelevant retrieval is worse
   than none.

3. LATENCY. That any-term fallback over a 7M-page index is not cheap:
   "thanks, that helps" spent 3.0 SECONDS returning nothing, so a turn needing
   no reference at all was the slowest turn on the lane.

The gate is deliberately NOT a pattern over question shape. Gating on
who/what/when + is/was is the hard-coding that let "can you run code" reach
her instruments while "can you search the web" did not: it serves the examples
its author thought of. What decides here is whether a retrieved page is about
what was asked.
"""
from __future__ import annotations

import time

import pytest

from core.conversation.chat_preflight import _reference_corpus_summary
from core.knowledge.local_corpus import CONVERSATION_SEARCH_DEADLINE_S


#: A real corpus, built here rather than borrowed from the host.
#:
#: The first version of this file skipped whenever the machine had no corpus at
#: ~/.aura/knowledge — which under pytest is ALWAYS, because state_root() moves
#: to a test directory. Ten of fifteen tests skipped and the file reported
#: green while checking nothing. A test that cannot fail is the same defect as
#: the silence it was written about.
_FIXTURE_DOCS = [
    (
        "Grace Hopper",
        "Grace Brewster Hopper (December 9, 1906 - January 1, 1992) was an "
        "American computer scientist and United States Navy rear admiral. She "
        "popularised the idea of machine-independent programming languages, "
        "which led to the development of COBOL, and worked on the Harvard "
        "Mark I and the UNIVAC I compiler.",
        "wikipedia",
    ),
    (
        "Peace of Westphalia",
        "The Peace of Westphalia is the collective name for two peace treaties "
        "signed in October 1648 in Osnabruck and Munster, ending the Thirty "
        "Years War in the Holy Roman Empire.",
        "wikipedia",
    ),
    (
        "COBOL",
        "COBOL is a compiled English-like computer programming language "
        "designed for business use, derived from the FLOW-MATIC language "
        "designed by Grace Hopper.",
        "wikipedia",
    ),
    (
        "Tokamak",
        "A tokamak is a device which uses a powerful magnetic field to confine "
        "plasma in the shape of a torus, and is a leading candidate for "
        "producing controlled thermonuclear fusion power.",
        "wikipedia",
    ),
]


@pytest.fixture(autouse=True)
def corpus(tmp_path, monkeypatch):
    """A small real corpus, wired in as the process-wide store."""
    from core.knowledge import local_corpus

    store = local_corpus.LocalCorpusStore(tmp_path / "corpus.db")
    store.add_documents(_FIXTURE_DOCS)
    monkeypatch.setattr(local_corpus, "get_local_corpus_store", lambda: store)
    monkeypatch.setattr(
        "core.conversation.chat_preflight.get_local_corpus_store",
        lambda: store,
        raising=False,
    )
    return store


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("Who was Grace Hopper and what specifically did she build?", "hopper"),
        ("explain the Treaty of Westphalia", "westphalia"),
        ("tell me about COBOL", "cobol"),
    ],
)
def test_a_question_about_the_world_retrieves_the_right_page(question, expected):
    """Phrasings no interrogative pattern would agree on."""
    lines = _reference_corpus_summary(question)

    assert lines, question
    assert expected in " ".join(lines).lower()


def test_the_retrieved_passage_carries_its_text_and_provenance():
    """The field is `snippet`; reading `text` gave every hit a bare title.

    Provenance is the point: the defect was a fabricated "Source: Wikipedia".
    """
    lines = _reference_corpus_summary("Who was Grace Hopper?")

    assert lines
    assert len(lines[0]) > 60, lines[0]
    assert "[" in lines[0] and "]" in lines[0], "the source must be quoted"


@pytest.mark.parametrize(
    "not_about_the_world",
    [
        "what are you feeling right now",
        "what is on my screen",
        "how are your subsystems doing",
        "what do you think about that",
    ],
)
def test_questions_about_her_or_her_senses_retrieve_nothing(not_about_the_world):
    assert _reference_corpus_summary(not_about_the_world) == []


@pytest.mark.parametrize(
    "chatter", ["thanks, that helps", "ok cool", "yeah exactly", "sure, go on"]
)
def test_conversation_never_pays_for_a_search_it_did_not_need(chatter):
    """3.0s for nothing was the measured cost of small talk."""
    started = time.monotonic()
    lines = _reference_corpus_summary(chatter)
    elapsed = time.monotonic() - started

    assert lines == []
    # Generous against the deadline: this asserts the bound exists, not a
    # particular machine's speed.
    assert elapsed < (CONVERSATION_SEARCH_DEADLINE_S * 4), f"{chatter} took {elapsed:.2f}s"


def test_the_bound_is_a_property_of_the_search(corpus):
    """Any caller can cap a search; it is not one caller being careful.

    Asserts the contract rather than a duration. Timing a four-document
    fixture would pass however the deadline behaved, which is the kind of
    test that reports green while checking nothing.
    """
    import inspect

    from core.knowledge.local_corpus import SEARCH_DEADLINE_S, LocalCorpusStore

    signature = inspect.signature(LocalCorpusStore.search)
    assert "deadline_s" in signature.parameters
    assert signature.parameters["deadline_s"].default == SEARCH_DEADLINE_S
    # A conversation lane must ask for less than the backstop, or the split
    # between the two is decorative.
    assert CONVERSATION_SEARCH_DEADLINE_S < SEARCH_DEADLINE_S


def test_an_expired_deadline_never_raises(corpus, monkeypatch):
    """A deadline that surfaced as an exception would be a worse outage.

    Deliberately NOT asserting the result is empty. The progress handler is
    consulted every few thousand VM instructions, and a four-document fixture
    finishes before the first check — so on a corpus this size the query
    completes and returning its rows is correct. Asserting [] here would only
    be asserting that the fixture is slow, which it is not, and the assertion
    would pass for the wrong reason on a big corpus and fail on a small one.

    What must hold at every size is that an expired deadline degrades to a
    result rather than an exception on a conversation lane.
    """
    from core.knowledge import local_corpus

    real_monotonic = time.monotonic
    monkeypatch.setattr(
        local_corpus.time, "monotonic", lambda: real_monotonic() + 3600.0
    )

    assert isinstance(corpus.search("grace hopper", limit=2), list)


def test_the_fast_path_actually_calls_it():
    """Wiring: retrieval bound to the lane again would be silent."""
    import inspect

    from core.conversation import chat_preflight

    source = inspect.getsource(chat_preflight.inject_operational_self_context)
    assert "_reference_corpus_summary" in source
