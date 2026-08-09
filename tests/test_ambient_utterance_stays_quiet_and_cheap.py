"""An unprompted remark must never make the person's own question slower.

``consider_utterance`` runs off the ambient observation loop — every time she
notices the screen changed, on a machine whose resident 32B is the same model
answering whatever the person just typed. So the judgment has two costs and
both have to stay small: the cheap gate must reject almost everything before
any model call, and the model call it does make must be BACKGROUND.

It was neither marked background nor bounded. It passed ``mode=ThinkingMode.
FAST``, and the router's tiering reads ``is_background`` / ``priority`` /
``prefer_tier`` — never ``mode`` — so the flag was inert and an ambient
musing rode the foreground lane against a live turn.

There was no dedicated test for this module at all, which is why.
"""
from __future__ import annotations

import asyncio
import inspect

import pytest

from core.perception.ambient_utterance import (
    _COMPOSE_TIMEOUT_S,
    _compose,
    _is_refusal_or_noise,
    _worth_noticing,
    consider_utterance,
)


class _Observation:
    def __init__(self, capture: str):
        self.capture = capture

    def for_reasoning(self) -> str:
        return f"[screen] {self.capture}"


class _Router:
    """Records exactly how the ambient lane asked for cognition."""

    def __init__(self, reply: str = "The failing assertion is on line 42."):
        self.kwargs: dict = {}
        self.reply = reply

    async def think(self, prompt, **kwargs):
        self.kwargs = kwargs
        return self.reply


@pytest.fixture
def router(monkeypatch):
    instance = _Router()
    monkeypatch.setattr(
        "core.container.ServiceContainer.get",
        staticmethod(
            lambda name, default=None: instance if name == "llm_router" else default
        ),
    )
    return instance


# ─────────────────────────────────────────── it does not fight the foreground


def test_the_ambient_judgment_is_marked_background(router):
    """The flag the router actually reads.

    ``mode=ThinkingMode.FAST`` went into **kwargs and was never consulted, so
    the call it was meant to de-prioritise ran at full foreground priority.
    """
    asyncio.run(_compose(_Observation("traceback (most recent call last)"), "traceback"))

    assert router.kwargs.get("is_background") is True
    assert router.kwargs.get("priority", 1.0) <= 0.3
    assert router.kwargs.get("prefer_tier") is not None
    assert "mode" not in router.kwargs, (
        "the router does not read `mode`; passing it looks like tiering and is not"
    )


def test_composing_is_bounded(router):
    """It runs inside a loop. An unbounded call is a wedged loop."""
    source = inspect.getsource(_compose)

    assert "wait_for" in source
    assert "_COMPOSE_TIMEOUT_S" in source
    assert 0 < _COMPOSE_TIMEOUT_S <= 60


def test_a_slow_composition_is_silence_not_a_late_remark(monkeypatch):
    """A remark about a screen the person already left is wrong, not late."""

    class _Slow:
        async def think(self, prompt, **kwargs):
            await asyncio.sleep(5)
            return "too late to matter"

    monkeypatch.setattr(
        "core.container.ServiceContainer.get",
        staticmethod(
            lambda name, default=None: _Slow() if name == "llm_router" else default
        ),
    )
    monkeypatch.setattr(
        "core.perception.ambient_utterance._COMPOSE_TIMEOUT_S", 0.05
    )

    assert asyncio.run(_compose(_Observation("panic: nil map"), "panic:")) == ""


def test_no_cognition_means_no_fallback_line(monkeypatch):
    """Silence beats a canned "I noticed an error on your screen!".

    A template interrupts and carries no information, and the person cannot
    tell it from a real observation until they have already been interrupted.
    """
    monkeypatch.setattr(
        "core.container.ServiceContainer.get",
        staticmethod(lambda name, default=None: default),
    )

    assert asyncio.run(_compose(_Observation("build failed"), "build failed")) == ""


# ──────────────────────────────────── the cheap gate rejects almost everything


@pytest.mark.parametrize(
    "capture",
    [
        "Traceback (most recent call last):\n  File 'x.py'",
        "AssertionError: expected 3, got 5",
        "build failed after 12 seconds",
        "panic: runtime error: index out of range",
    ],
)
def test_a_visible_fault_is_worth_noticing(capture):
    assert _worth_noticing(capture) != ""


@pytest.mark.parametrize(
    "capture",
    [
        "def handler(): pass  # ordinary code with no fault",
        "Stack Overflow — how to fix error: undefined",
        "Read the documentation about error: codes here",
        "One error: appears once in ordinary prose",
    ],
)
def test_ordinary_screens_are_not_worth_interrupting_for(capture):
    assert _worth_noticing(capture) == ""


def test_the_cheap_gate_runs_before_any_model_call(router):
    """Most ticks must cost nothing. Otherwise the loop is unaffordable."""
    import core.perception.ambient_utterance as module

    calls: list[str] = []

    class _Memory:
        def latest(self, _kind):
            calls.append("latest")
            return _Observation("def handler(): pass  # nothing wrong here at all")

    module_memory = _Memory()
    original = module._latest_observation
    module._latest_observation = lambda: module_memory.latest(None)
    try:
        assert asyncio.run(consider_utterance(object())) == ""
    finally:
        module._latest_observation = original

    assert router.kwargs == {}, "an ordinary screen reached the model"


# ────────────────────────────────────────────── a greeting is not an insight


@pytest.mark.parametrize(
    "reply",
    [
        "NOTHING",
        "Hi! I noticed you're working on something.",
        "It looks like you're debugging.",
        "Would you like some help with that?",
        "I'm here to help!",
        "ok",
    ],
)
def test_template_replies_are_discarded(reply):
    assert _is_refusal_or_noise(reply) is True


def test_a_specific_observation_survives():
    assert (
        _is_refusal_or_noise("The assertion on line 42 contradicts the test name.")
        is False
    )
