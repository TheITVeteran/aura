"""Aura challenges the interlocutor when it is wrong — but only with grounds.
Pins the precision-first contradiction detection and the grounded-pushback
composition, plus the honesty rule (no grounding => no challenge)."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass

from core.capabilities.interlocutor_factcheck import (
    compose_challenge_message,
    extract_checkable_claims,
    factcheck_reply,
)


def test_extract_skips_opinion_and_hedged():
    reply = (
        "The Eiffel Tower was completed in 1912. "
        "I think transformers are the best architecture. "
        "It might be the case that memory helps."
    )
    claims = extract_checkable_claims(reply)
    assert any("Eiffel Tower" in c for c in claims)
    assert not any("I think" in c for c in claims)
    assert not any("might be" in c for c in claims)


def test_numeric_mismatch_is_flagged_with_evidence():
    def corpus(query, k):
        return [{"text": "The Eiffel Tower was completed in 1889 in Paris.",
                 "source": "wiki:Eiffel_Tower"}]

    contradictions = factcheck_reply(
        "The Eiffel Tower was completed in 1912.", corpus_search=corpus,
    )
    assert len(contradictions) == 1
    c = contradictions[0]
    assert c.signal == "numeric_mismatch"
    assert "1889" in c.counter_evidence
    assert c.source == "wiki:Eiffel_Tower"


def test_agreement_is_not_a_contradiction():
    def corpus(query, k):
        return [{"text": "The Eiffel Tower was completed in 1889.", "source": "wiki"}]

    # interlocutor says the SAME year the corpus does — no challenge
    assert factcheck_reply("The Eiffel Tower was completed in 1889.", corpus_search=corpus) == []


def test_no_grounding_means_no_challenge():
    assert factcheck_reply("The Eiffel Tower was completed in 1912.", corpus_search=lambda q, k: []) == []


def test_non_numeric_claim_does_not_search_without_adjudicator():
    calls = []

    def corpus(query, k):
        calls.append(query)
        return [{"text": "A passage that should not be read.", "source": "test"}]

    assert factcheck_reply("Memory is useful for continuity.", corpus_search=corpus) == []
    assert calls == []


def test_adjudication_seam_requires_confidence_and_evidence():
    def corpus(query, k):
        return [{"text": "Canberra is the capital of Australia.", "source": "wiki:Canberra"}]

    def confident(claim, passages):
        return True, "Canberra is the capital of Australia", 0.9

    def unsure(claim, passages):
        return True, "Canberra is the capital of Australia", 0.4  # below floor

    claim = "The capital of Australia is Sydney."
    assert len(factcheck_reply(claim, corpus_search=corpus, adjudicate=confident)) == 1
    assert factcheck_reply(claim, corpus_search=corpus, adjudicate=unsure) == []
    # no adjudicator + no numeric signal => silence, not a false accusation
    assert factcheck_reply(claim, corpus_search=corpus) == []


def test_challenge_message_cites_evidence_and_source():
    def corpus(query, k):
        return [{"text": "The Eiffel Tower was completed in 1889.", "source": "wiki:Eiffel_Tower"}]

    contradictions = factcheck_reply("The Eiffel Tower was completed in 1912.", corpus_search=corpus)
    msg = compose_challenge_message(contradictions)
    assert "1889" in msg
    assert "wiki:Eiffel_Tower" in msg
    assert "push back" in msg.lower()


@dataclass
class _Turn:
    observed_reply: str


def test_session_grounded_challenge_fires_and_records():
    from core.capabilities.web_interlocutor import WebInterlocutorSession

    def corpus(query, k):
        return [{"text": "The Eiffel Tower was completed in 1889.", "source": "wiki:Eiffel_Tower"}]

    session = WebInterlocutorSession(browser=object(), cognitive_engine=None)
    ctx: dict = {"corpus_search": corpus}
    turns = [_Turn("The Eiffel Tower was completed in 1912, a fact everyone agrees on.")]
    message = asyncio.run(session._grounded_challenge(turns, ctx))
    assert message and "1889" in message
    assert ctx.get("_challenges_issued")
    assert ctx["_challenges_issued"][0]["signal"] == "numeric_mismatch"


def test_session_no_challenge_when_corpus_silent():
    from core.capabilities.web_interlocutor import WebInterlocutorSession

    session = WebInterlocutorSession(browser=object(), cognitive_engine=None)
    ctx: dict = {"corpus_search": lambda q, k: []}
    turns = [_Turn("The Eiffel Tower was completed in 1912.")]
    assert asyncio.run(session._grounded_challenge(turns, ctx)) == ""
    assert not ctx.get("_challenges_issued")
