"""Two derivations part at a point; the point is settled by the strongest check that applies, then she answers."""

from __future__ import annotations

import pytest

from core.reasoning.where_derivations_part import (
    Crux,
    Derivation,
    kind_of,
    parse_crux,
    parse_reading,
    settle,
    side_of_value,
)

CRUX = "FIRST: the train covers 120 km\nSECOND: the train covers 150 km\nQUESTION: How far does it go in 2.5 h at 48 km/h?"


def engines(*, answers, crux=CRUX, value=None, passages=(), readings=()):
    """Fake engines; ``answers`` maps (evidence given?, sampled?) and crux questions to Derivations."""
    calls: dict[str, list] = {"derive": [], "ask": [], "compute": [], "lookup": []}
    pending_readings = list(readings)

    async def derive(question, evidence, sampled):
        calls["derive"].append((question, evidence, sampled))
        return answers[(question, bool(evidence), sampled)]

    async def ask(request):
        calls["ask"].append(request)
        if request.startswith("Two answers"):
            return crux
        return pending_readings.pop(0) if pending_readings else "NEITHER"

    async def compute(question):
        calls["compute"].append(question)
        return value

    async def lookup(query):
        calls["lookup"].append(query)
        return list(passages)

    return calls, {"derive": derive, "ask": ask, "compute": compute, "lookup": lookup, "same": lambda a, b: a == b}


Q = "A train goes 48 km/h for 2.5 hours. How far? (A) 120 km (B) 150 km"
QC = "How far does it go in 2.5 h at 48 km/h?"


@pytest.mark.asyncio
async def test_agreeing_derivations_answer_without_a_crux() -> None:
    calls, kw = engines(answers={(Q, False, False): Derivation("A", "120 km, so A", 900),
                                 (Q, False, True): Derivation("A", "It is A", 700)})
    out = await settle(Q, **kw)
    assert out.agreed and out.answer == "A" and out.crux is None and out.tokens == 1600 and not calls["ask"]


@pytest.mark.asyncio
async def test_a_quantity_is_settled_in_the_sandbox_and_her_last_derivation_is_given_it() -> None:
    answers = {(Q, False, False): Derivation("B", "150 km", 900), (Q, False, True): Derivation("A", "120 km", 800),
               (Q, True, False): Derivation("A", "Given the check, 120 km: A", 600)}
    calls, kw = engines(answers=answers, value="120")
    out = await settle(Q, **kw)
    assert out.crux.kind == "quantity" and out.crux.settled == "first" and out.answer == "A"
    assert calls["compute"] == [QC] and not calls["lookup"]
    evidence = calls["derive"][-1][1]
    assert "the train covers 150 km" not in evidence and "120 km" in evidence and "sandbox" in evidence


@pytest.mark.asyncio
async def test_a_claim_is_settled_by_her_corpus_when_the_passages_support_one_side() -> None:
    crux = "FIRST: Canberra is the capital\nSECOND: Sydney is the capital\nQUESTION: What is Australia's capital?"
    q = "What is the capital of Australia? (A) Sydney (B) Canberra"
    answers = {(q, False, False): Derivation("A", "Sydney", 300), (q, False, True): Derivation("B", "Canberra", 300),
               (q, True, False): Derivation("B", "Canberra, as checked", 200)}
    calls, kw = engines(answers=answers, crux=crux, passages=["Canberra is the capital city of Australia."],
                        readings=["FIRST"])
    out = await settle(q, **kw)
    assert out.crux.kind == "claim" and out.crux.how == "found in her corpus" and out.answer == "B"
    assert not calls["compute"] and "Canberra is the capital city" in calls["derive"][-1][1]


@pytest.mark.asyncio
async def test_failing_both_checks_the_point_alone_is_derived_twice_and_must_agree() -> None:
    crux = "FIRST: the clause modifies the noun\nSECOND: the clause modifies the verb\nQUESTION: What does it modify?"
    q = "Which reading? (A) noun (B) verb"
    qc = "What does it modify?"
    answers = {(q, False, False): Derivation("A", "noun", 300), (q, False, True): Derivation("B", "verb", 300),
               (qc, False, False): Derivation("", "It modifies the verb.", 100),
               (qc, False, True): Derivation("", "The verb.", 100),
               (q, True, False): Derivation("B", "verb", 200)}
    calls, kw = engines(answers=answers, crux=crux, readings=["SECOND", "SECOND"])
    out = await settle(q, **kw)
    assert out.crux.settled == "second" and out.crux.how.startswith("derived again") and out.answer == "B"
    assert len(out.derivations) == 4 and out.final.answer == "B"


@pytest.mark.asyncio
async def test_a_crux_nothing_settles_leaves_her_greedy_answer() -> None:
    crux = "FIRST: x\nSECOND: y\nQUESTION: which?"
    q = "pick (A) or (B)"
    answers = {(q, False, False): Derivation("A", "a", 1), (q, False, True): Derivation("B", "b", 1),
               ("which?", False, False): Derivation("", "x", 1), ("which?", False, True): Derivation("", "y", 1)}
    calls, kw = engines(answers=answers, crux=crux, readings=["FIRST", "SECOND"])
    out = await settle(q, **kw)
    assert out.answer == "A" and out.final is None and out.stages[-1] == "unsettled"


def test_the_crux_and_readings_parse_only_when_they_say_what_is_needed() -> None:
    assert parse_crux("FIRST: a\nSECOND: a\nQUESTION: q") is None
    assert parse_crux("FIRST: a\nQUESTION: q") is None
    crux = parse_crux(CRUX)
    assert crux and crux.question.startswith("How far") and kind_of(crux) == "quantity"
    assert kind_of(Crux("it is red", "it is blue", "colour?")) == "claim"
    assert parse_reading("SECOND") == "second" and parse_reading("NEITHER") is None
    assert parse_reading("FIRST or SECOND") is None
    assert side_of_value("150", crux) == "second" and side_of_value("130", crux) is None
    assert side_of_value("120 or 150", crux) is None


@pytest.mark.asyncio
async def test_a_search_told_as_a_story_is_settled_by_its_unique_solution() -> None:
    crux = "FIRST: Ann sits left of Bo\nSECOND: Bo sits left of Ann\nQUESTION: Who sits on the left?"
    q = "Who is leftmost? (A) Ann (B) Bo"
    answers = {(q, False, False): Derivation("A", "Ann", 300), (q, False, True): Derivation("B", "Bo", 300),
               (q, True, False): Derivation("B", "Bo, as solved", 200)}
    calls, kw = engines(answers=answers, crux=crux, passages=["irrelevant"], readings=["FIRST"])

    async def formalize(question):
        return "B"

    out = await settle(q, formalize=formalize, **kw)
    assert out.crux.settled == "second" and out.crux.how.startswith("solved as constraints") and out.answer == "B"
    assert not calls["lookup"] and "exactly one solution" in calls["derive"][-1][1]
