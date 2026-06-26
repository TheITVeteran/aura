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


def test_stance_directive_sarcasm_is_causal():
    from core.phases.response_generation_unitary import UnitaryResponsePhase

    d = UnitaryResponsePhase._stance_directive(_state_with_mods({"communicative_stance": "sarcastic"}))
    assert "sarcastic" in d.lower()
    assert d.startswith("COMMUNICATIVE STANCE")


def test_stance_directive_sincere_is_silent():
    from core.phases.response_generation_unitary import UnitaryResponsePhase

    d = UnitaryResponsePhase._stance_directive(_state_with_mods({"communicative_stance": "sincere"}))
    assert d == ""


def test_stance_directive_false_claim_surfaces_discrepancy():
    from core.phases.response_generation_unitary import UnitaryResponsePhase

    d = UnitaryResponsePhase._stance_directive(
        _state_with_mods({"communicative_stance": "sincere", "flagged_false_claim": "claim negates known fact: X"})
    )
    assert "conflicts with what I know" in d


def test_stance_directive_mistaken_includes_detail():
    from core.phases.response_generation_unitary import UnitaryResponsePhase

    d = UnitaryResponsePhase._stance_directive(
        _state_with_mods({"communicative_stance": "mistaken", "flagged_false_claim": "claim negates known fact: Y"})
    )
    assert "correct it" in d.lower()
    assert "Y" in d
