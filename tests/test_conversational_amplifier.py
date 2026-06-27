"""Tests for the conversational amplifier + personalized taste model + learning loop."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from core.brain import conversation_outcome as outcome
from core.brain.conversational_amplifier import (
    amplify_conversation,
    is_conversationally_amplifiable,
)
from core.brain.response_quality import extract_features, select_best
from core.brain.taste_model import TasteModel

_GOOD = "Honestly, Blade Runner 2049 nails it; the pacing's deliberate and that's the point."
_BAD = "It's a great movie with interesting themes. I'd be happy to discuss! What do you think?"


@pytest.fixture(autouse=True)
def _fresh_taste(tmp_path, monkeypatch):
    tm = TasteModel(tmp_path / "taste.json")
    monkeypatch.setattr("core.brain.taste_model.get_taste_model", lambda: tm)
    monkeypatch.setattr("core.brain.response_quality.get_taste_model", lambda: tm)
    monkeypatch.setattr("core.brain.conversation_outcome.get_taste_model", lambda: tm)
    monkeypatch.setenv("AURA_CONVERSATIONAL_AMPLIFIER", "1")
    return tm


# ── feature extraction / selection ──────────────────────────────────────────
def test_good_outscores_bad():
    best, ranked = select_best([_BAD, _GOOD])
    assert best == _GOOD
    assert ranked[0][1] > ranked[-1][1]


def test_penalties_fire():
    f = extract_features(_BAD)
    assert f["banned_phrase_penalty"] >= 1     # "I'd be happy to"
    assert f["prompt_farm_penalty"] >= 1       # "what do you think" + ends with ?


def test_specificity_and_casual_detected():
    f = extract_features(_GOOD)
    assert f["specificity"] > 0  # "Blade Runner"
    assert f["casual"] > 0       # contractions


def test_callback_rewards_grounding_overlap():
    f = extract_features("That reminds me of the aquarium trip you mentioned.",
                         grounding_tokens={"aquarium", "trip", "jellyfish"})
    assert f["callback"] > 0


# ── taste model online learning ─────────────────────────────────────────────
def test_taste_update_moves_weights_with_reward(_fresh_taste):
    before = _fresh_taste.weights()["specificity"]
    _fresh_taste.update({"specificity": 1.0}, reward=1.0)
    assert _fresh_taste.weights()["specificity"] > before


def test_taste_update_clamped(_fresh_taste):
    for _ in range(1000):
        _fresh_taste.update({"specificity": 1.0}, reward=1.0)
    assert _fresh_taste.weights()["specificity"] <= 4.0


def test_taste_neutral_reward_noop(_fresh_taste):
    before = _fresh_taste.weights()
    _fresh_taste.update({"specificity": 1.0}, reward=0.0)
    assert _fresh_taste.weights() == before


# ── learning loop (reaction -> taste update) ────────────────────────────────
def test_positive_reaction_updates_taste(_fresh_taste):
    outcome.reset()
    outcome.record_pending_response(_GOOD, {"specificity": 1.0, "stance": 1.0})
    before = _fresh_taste.weights()["specificity"]
    reward = outcome.register_reaction("haha love it, exactly")
    assert reward == 1.0
    assert _fresh_taste.weights()["specificity"] > before


def test_negative_reaction_updates_taste(_fresh_taste):
    outcome.reset()
    outcome.record_pending_response(_BAD, {"banned_phrase_penalty": 1.0})
    reward = outcome.register_reaction("no, that's wrong")
    assert reward == -1.0


def test_neutral_reaction_no_update(_fresh_taste):
    outcome.reset()
    outcome.record_pending_response(_GOOD, {"specificity": 1.0})
    before = _fresh_taste.weights()
    assert outcome.register_reaction("ok and then what about tuesday") is None
    assert _fresh_taste.weights() == before


# ── amplifier end to end ────────────────────────────────────────────────────
def test_amplify_selects_better_candidate_and_revises():
    async def gen(prompt, temp):
        if "Improve this reply" in prompt:
            # clearly higher taste: stance + contractions + specificity, no penalties
            return "Honestly? Blade Runner 2049 — Villeneuve's slow pace IS the argument, and that's exactly why it's the one I'd rewatch."
        return _GOOD

    res = asyncio.run(amplify_conversation(_BAD, generate=gen, user_message="thoughts on blade runner 2049?", n=3))
    assert res.answer != _BAD               # the mediocre draft was beaten
    assert "Villeneuve" in res.answer        # the revised version won
    assert res.revised is True
    assert res.n_candidates >= 2


def test_amplify_fail_open_on_generate_error():
    calls = []

    async def gen(prompt, temp):
        calls.append((prompt, temp))
        raise RuntimeError("model down")

    res = asyncio.run(amplify_conversation(_GOOD, generate=gen, user_message="hi there friend", n=3))
    assert res.answer == _GOOD  # fell back to the draft, no crash
    assert calls


def test_amplify_disabled_returns_draft(monkeypatch):
    monkeypatch.setenv("AURA_CONVERSATIONAL_AMPLIFIER", "0")

    async def gen(prompt, temp):
        return "should not be used"

    res = asyncio.run(amplify_conversation(_BAD, generate=gen, user_message="some message here", n=3))
    assert res.answer == _BAD


# ── gating ───────────────────────────────────────────────────────────────────
def test_gating_excludes_actions_and_reasoning():
    assert is_conversationally_amplifiable("what's your take on jazz?", "user") is True
    assert is_conversationally_amplifiable("open three tabs", "user") is False     # action
    assert is_conversationally_amplifiable("what is 17 to the power of 4", "user") is False  # reasoning
    assert is_conversationally_amplifiable("hey", "user") is False                  # too short
    assert is_conversationally_amplifiable("what's your take on jazz?", "background") is False


# ── primary response phase hook ─────────────────────────────────────────────
class _LLMSpy:
    def __init__(self):
        self.calls = 0

    async def think(self, prompt, **kwargs):
        self.calls += 1
        if "Improve this reply" in str(prompt):
            return "Honestly, this is the sharper revised version with one concrete detail."
        return "Honestly, this is the alternate version with one concrete detail."


def _unitary_state(origin: str = "user"):
    return SimpleNamespace(
        cognition=SimpleNamespace(current_origin=origin, rolling_summary="Bryan likes direct specific answers"),
        response_modifiers={},
        metadata={},
    )


def test_unitary_conversational_amplifier_default_does_not_spawn_extra_model_calls(monkeypatch):
    from core.phases.response_generation_unitary import UnitaryResponsePhase

    monkeypatch.delenv("AURA_CONVERSATIONAL_AMPLIFIER_LIVE", raising=False)
    phase = UnitaryResponsePhase.__new__(UnitaryResponsePhase)
    llm = _LLMSpy()

    out = asyncio.run(
        phase._maybe_amplify_conversation(
            objective="what do you think about long horizon autonomy?",
            draft=_GOOD,
            llm=llm,
            state=_unitary_state(),
            request_timeout=20.0,
            is_user_facing=True,
            is_background=False,
            proof_or_benchmark=False,
        )
    )

    assert out == _GOOD
    assert llm.calls == 0


def test_unitary_conversational_amplifier_live_opt_in_is_bounded(monkeypatch):
    from core.phases.response_generation_unitary import UnitaryResponsePhase

    monkeypatch.setenv("AURA_CONVERSATIONAL_AMPLIFIER_LIVE", "1")
    monkeypatch.setattr(
        "core.utils.memory_monitor.get_memory_pressure_snapshot",
        lambda: SimpleNamespace(refuse_heavy_local_generation=False, pressure_pct=20.0),
    )
    phase = UnitaryResponsePhase.__new__(UnitaryResponsePhase)
    llm = _LLMSpy()
    state = _unitary_state()

    out = asyncio.run(
        phase._maybe_amplify_conversation(
            objective="what do you think about long horizon autonomy?",
            draft=_BAD,
            llm=llm,
            state=state,
            request_timeout=20.0,
            is_user_facing=True,
            is_background=False,
            proof_or_benchmark=False,
        )
    )

    assert out != _BAD
    assert 1 <= llm.calls <= 2
    assert state.metadata["conversation_amplification"]["n_candidates"] == 2


def test_unitary_conversational_amplifier_blocks_under_memory_pressure(monkeypatch):
    from core.phases.response_generation_unitary import UnitaryResponsePhase

    monkeypatch.setenv("AURA_CONVERSATIONAL_AMPLIFIER_LIVE", "1")
    monkeypatch.setattr(
        "core.utils.memory_monitor.get_memory_pressure_snapshot",
        lambda: SimpleNamespace(refuse_heavy_local_generation=True, pressure_pct=94.0),
    )
    phase = UnitaryResponsePhase.__new__(UnitaryResponsePhase)
    llm = _LLMSpy()

    out = asyncio.run(
        phase._maybe_amplify_conversation(
            objective="what do you think about long horizon autonomy?",
            draft=_GOOD,
            llm=llm,
            state=_unitary_state(),
            request_timeout=20.0,
            is_user_facing=True,
            is_background=False,
            proof_or_benchmark=False,
        )
    )

    assert out == _GOOD
    assert llm.calls == 0
