"""Stance inference must run live in the social context phase and write modifiers."""
from __future__ import annotations

import types

import pytest

from core.phases.social_context_phase import SocialContextPhase


class _StubContainer:
    def __init__(self, facade=None):
        self._facade = facade

    def get(self, name, default=None):
        if name == "memory_facade":
            return self._facade if self._facade is not None else default
        return default


def _state(recent=None):
    return types.SimpleNamespace(
        conversation=types.SimpleNamespace(history=recent or []),
    )


@pytest.mark.asyncio
async def test_phase_writes_stance_modifier_for_joke():
    phase = SocialContextPhase(_StubContainer())
    modifiers: dict = {}
    await phase._infer_communicative_stance(_state(), modifiers, "you crashed prod again lol jk")
    assert modifiers["communicative_stance"] == "joking"
    assert modifiers["take_literally"] is False


@pytest.mark.asyncio
async def test_phase_flags_false_claim_against_memory():
    class _Facade:
        async def search(self, query, limit=5):
            return [{"content": "Python is an interpreted language, not compiled.", "id": "m1"}]

    phase = SocialContextPhase(_StubContainer(_Facade()))
    modifiers: dict = {}
    await phase._infer_communicative_stance(
        _state(), modifiers, "Python is definitely a compiled language with no interpreter."
    )
    assert modifiers.get("stance_factual_conflict") is True
    assert "flagged_false_claim" in modifiers


@pytest.mark.asyncio
async def test_phase_sincere_message_taken_literally():
    phase = SocialContextPhase(_StubContainer())
    modifiers: dict = {}
    await phase._infer_communicative_stance(_state(), modifiers, "The migration finished successfully.")
    assert modifiers["communicative_stance"] == "sincere"
    assert modifiers["take_literally"] is True


@pytest.mark.asyncio
async def test_phase_uses_recent_history_for_contradiction():
    phase = SocialContextPhase(_StubContainer())
    modifiers: dict = {}
    recent = [{"role": "user", "content": "I edited the config file earlier."}]
    await phase._infer_communicative_stance(_state(recent), modifiers, "I never touched the config file.")
    # Contradiction with own recent statement should be reflected in the stance signals path.
    assert "communicative_stance" in modifiers


def _state_with_mods(mods: dict):
    return types.SimpleNamespace(cognition=types.SimpleNamespace(modifiers=mods))


HOW_IT_WAS_MEANT = "## HOW WHAT WAS SAID WAS MEANT"


def _meant(mods: dict) -> str:
    from core.phases.response_generation_unitary import UnitaryResponsePhase

    return UnitaryResponsePhase._stance_directive(_state_with_mods(mods))


def test_a_stance_read_of_the_turn_reaches_her_as_a_reading():
    read = _meant({"communicative_stance": "sarcastic"})
    assert read == f"{HOW_IT_WAS_MEANT}\n- Read as: sarcastic."


def test_a_sincere_turn_sends_no_reading():
    assert _meant({"communicative_stance": "sincere"}) == ""
    assert _meant({}) == ""


def test_a_claim_at_odds_with_what_she_holds_is_named_whatever_the_stance():
    read = _meant({"communicative_stance": "sincere", "flagged_false_claim": "claim negates known fact: X"})
    assert read == f"{HOW_IT_WAS_MEANT}\n- It conflicts with what you hold: claim negates known fact: X."
    both = _meant({"communicative_stance": "mistaken", "flagged_false_claim": "claim negates known fact: Y"})
    assert "- Read as: mistaken." in both and "known fact: Y" in both


def test_the_reading_tells_her_nothing_to_do():
    """It was a line of instruction per stance. What is read of a turn is not an order about it."""
    for stance in ("joking", "sarcastic", "hypothesizing", "mistaken", "deceptive"):
        read = _meant({"communicative_stance": stance, "flagged_false_claim": "a claim"}).lower()
        for told in ("don't", "do not", "respond", "play along", "correct it", "treat ", "engage", "verify"):
            assert told not in read, (stance, told)


def test_the_reading_is_one_her_model_is_handed():
    from core.brain.what_she_is_given import _READINGS, readings_in

    assert HOW_IT_WAS_MEANT in _READINGS
    read = _meant({"communicative_stance": "joking"})
    told = "Narrative anchor: something\n" + read
    assert readings_in(told, _READINGS) == [read]
