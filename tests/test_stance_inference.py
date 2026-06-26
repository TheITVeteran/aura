"""Tests for communicative stance / deception inference."""
from __future__ import annotations

import pytest

from core.social.stance_inference import Stance, StanceInference, get_stance_inference


@pytest.fixture()
def si() -> StanceInference:
    return StanceInference()


def test_sincere_default(si):
    a = si.assess("The deployment finished and the tests are green.")
    assert a.primary is Stance.SINCERE
    assert a.take_literally


def test_hypothesizing(si):
    a = si.assess("Suppose we doubled the cache size, what would happen to latency?")
    assert a.primary is Stance.HYPOTHESIZING
    assert not a.take_literally


def test_pretending(si):
    a = si.assess("Let's pretend you're a pirate captain for this story.")
    assert a.primary is Stance.PRETENDING


def test_unsure(si):
    a = si.assess("I think it might be a caching issue, but I'm not really sure.")
    assert a.primary is Stance.UNSURE
    assert a.take_literally  # an honest guess is still a literal (if uncertain) claim


def test_joking(si):
    a = si.assess("you broke production again lol jk")
    assert a.primary is Stance.JOKING


def test_sarcasm_valence_incongruity(si):
    a = si.assess("Oh great, the build failed AGAIN. Just wonderful.")
    assert a.primary is Stance.SARCASTIC
    assert not a.take_literally


def test_flippant(si):
    a = si.assess("meh, whatever, don't care")
    assert a.primary is Stance.FLIPPANT


def test_factual_conflict_flags_mistaken(si):
    a = si.assess(
        "Python is definitely a compiled language with no interpreter.",
        known_facts=["Python is an interpreted language, not a compiled one."],
    )
    assert a.factual_conflict
    assert a.primary in {Stance.MISTAKEN, Stance.DECEPTIVE}
    # We do not pretend to read intent from a bare false claim.
    assert a.intent_readable is False


def test_deception_cues_lean_deceptive(si):
    a = si.assess(
        "Trust me, honestly, the server is not down, I would never lie about that.",
        known_facts=["The server is down right now."],
    )
    assert a.factual_conflict
    assert a.primary is Stance.DECEPTIVE
    assert a.intent_readable is True


def test_contradicts_recent(si):
    a = si.assess(
        "I never touched the config file.",
        recent_messages=["I edited the config file earlier today."],
    )
    assert any("recent" in s for s in a.signals)


@pytest.mark.asyncio
async def test_model_refinement_on_ambiguous(si):
    async def generate(prompt: str, temperature: float) -> str:
        return "sarcastic"

    # Force the ambiguous-refinement path: the model pass relabels and boosts.
    a = await si.assess_with_model("oh sure, fine", generate, ambiguity_threshold=0.99)
    assert "model-refined" in a.signals
    assert a.primary is Stance.SARCASTIC


def test_singleton():
    assert get_stance_inference() is get_stance_inference()
