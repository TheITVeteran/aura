"""A request reaches a kept kind by what it means, is restated in the kind's form, and is answered as evidence."""

from __future__ import annotations

import types

import numpy as np
import pytest

from core.learning.kinds_by_meaning import (
    KindMeaning,
    meanings_of,
    nearest_kind,
    restatement_request,
)
from core.learning.procedures_from_solved_examples import ProcedureBook, Signature
from core.sandbox.untrusted_python import available_boundary

needs_boundary = pytest.mark.skipif(not available_boundary(), reason="no OS sandbox on this host")
# Enough kinds for one to stand out from the rest, as her book has (26 and more).
AXES = {word: axis for axis, word in enumerate(
    ("percent", "dates", "sorting", "units", "primes", "brackets", "routes", "tables", "clocks", "shapes",
     "letters", "orders"))}


def _embed(texts: list[str]) -> np.ndarray:
    """A stand-in embedding: each text points along its kind's axis, with a little of a second."""
    rows = []
    for text in texts:
        row = np.full(len(AXES), 0.05, dtype=np.float32)
        for word, axis in AXES.items():
            if word in text:
                row[axis] = 1.0
        rows.append(row)
    return np.asarray(rows)


def _meanings() -> dict[str, KindMeaning]:
    kinds = {kind: ([f"{kind} problem {i}" for i in range(4)], [f"{kind} sealed {i}" for i in range(4)])
             for kind in AXES}
    return meanings_of(kinds, documents=_embed, queries=_embed)


def test_a_request_goes_to_the_kind_it_clearly_means_and_nowhere_when_it_means_none() -> None:
    meanings = _meanings()
    assert set(meanings) == set(AXES) and all(0.9 < m.nearest_required <= 1.0 for m in meanings.values())
    assert nearest_kind(_embed(["what's 15 percent of 340"])[0], meanings)[0] == "percent"
    kind, _score, why = nearest_kind(_embed(["tell me about your day"])[0], meanings)
    assert kind is None and why
    assert nearest_kind(_embed(["percent"])[0], dict(list(meanings.items())[:2]))[2] == \
        "too_few_kinds_to_tell_meaning_apart"


def test_the_restatement_shows_the_kinds_form_and_carries_the_request() -> None:
    text = restatement_request("what's 15% of 340?", [{"problem": "Calculate 0.15*340.", "answer": "51"}])
    assert "Calculate 0.15*340." in text and text.rstrip().endswith("what's 15% of 340?") and "51" not in text


SOLVE = "def solve(problem):\n    a, b = problem.split('Calculate ')[1].rstrip('.').split('*')\n" \
        "    return str(round(float(a) * float(b), 6)).rstrip('0').rstrip('.')\n"


def _book() -> ProcedureBook:
    signatures = {kind: Signature({kind: 1.0}, 0.99) for kind in AXES}
    return ProcedureBook(signatures, {"percent": [SOLVE]}, {"percent": 40}, meanings=_meanings(),
                         examples={"percent": [{"problem": "Calculate 0.2*50.", "answer": "10"}]})


@needs_boundary
def test_a_book_answers_a_restated_request_and_keeps_its_meanings_through_saving() -> None:
    book = ProcedureBook.from_json(_book().to_json())
    assert set(book.meanings) == set(AXES) and book.examples["percent"][0]["answer"] == "10"
    receipt = book.answer_restated("percent", "Calculate 0.15*340.")
    assert receipt["answer"] == "51" and receipt["form"] == "restated" and receipt["agreed_known"] == 40
    assert book.answer_restated("no such kind", "x")["declined"] == "no_such_kind"


@needs_boundary
@pytest.mark.asyncio
async def test_her_turn_finds_the_kind_by_meaning_restates_it_and_keeps_her_words(monkeypatch) -> None:
    from core.phases import response_generation as phase_module
    from core.phases.response_generation import ResponseGenerationPhase
    from core.state.aura_state import AuraState

    monkeypatch.setattr("core.brain.reasoning_amplifier_v2.answer_from_kept_procedures", lambda *a, **k: None)
    monkeypatch.setattr("core.learning.procedures_from_solved_examples.kept_procedure_book", _book)
    monkeypatch.setattr(phase_module, "_her_query_vector", lambda text, task: _embed([text])[0])
    asked: list[dict] = []

    class Router:
        async def think(self, **kwargs):
            asked.append(kwargs)
            return "Calculate 0.15*340."

    phase, state = ResponseGenerationPhase(types.SimpleNamespace()), AuraState.default()
    assert await phase._consult_kept_procedures(state=state, objective="what's 15 percent of 340?",
                                                router=Router(), origin="desktop", tier="primary")
    planted = state.response_modifiers["kept_procedure"]
    assert planted["answer"] == "51" and planted["form"] == "restated"
    assert asked and asked[0]["clean_user_surface_contract"] is True and asked[0]["temperature"] == 0.0
    assert "Calculate 0.2*50." in asked[0]["messages"][0]["content"]
    assert "Calculate 0.15*340." in state.cognition.working_memory[-1]["content"]

    async def no_search(*_a, **_k):
        raise AssertionError("the answer was already in front of her")

    monkeypatch.setattr("core.brain.reasoning_amplifier_v2.amplify_turn", no_search)
    draft = "Fifteen percent of 340 is 52, I think."
    out = await phase._maybe_amplify_response(
        objective="what's 15 percent of 340?", draft=draft, router=None, state=state, request_timeout=20.0,
        origin="desktop", tier="primary", runtime_context={}, is_user_facing=True, is_background=False,
        proof_or_benchmark=False)
    assert str(out) == draft
    assert state.response_modifiers["reasoning_amplifier_v2_active_phase"]["voice"] == "evidence_for_a_restated_request"


@pytest.mark.asyncio
async def test_without_her_embedder_held_nothing_is_read_by_meaning(monkeypatch) -> None:
    from core.phases import response_generation as phase_module
    from core.phases.response_generation import ResponseGenerationPhase
    from core.state.aura_state import AuraState

    monkeypatch.setattr("core.brain.reasoning_amplifier_v2.answer_from_kept_procedures", lambda *a, **k: None)
    monkeypatch.setattr("core.learning.procedures_from_solved_examples.kept_procedure_book", _book)
    monkeypatch.setattr(phase_module, "_her_query_vector", lambda text, task: None)

    class Router:
        async def think(self, **kwargs):
            raise AssertionError("no restatement without a meaning")

    phase, state = ResponseGenerationPhase(types.SimpleNamespace()), AuraState.default()
    assert not await phase._consult_kept_procedures(state=state, objective="what's 15 percent of 340?",
                                                    router=Router(), origin="desktop", tier="primary")
    assert "kept_procedure" not in state.response_modifiers
