"""A deliberation that cannot happen must say so, not hang or vanish.

CP126 (high), core/brain/deliberation.py: "Ordinary generation has no
deadline or typed fallback. The model call can hang and exceptions escape
without a decision receipt; native failures silently drop to the weaker
ordinary route."

Three defects in one line of code. ``await self.llm.generate(...)`` had no
timeout, so a wedged model stalled the decision and everything waiting on
it. It had no exception handling, so a provider error escaped to a caller
that asked for a Decision. And when native System 2 declined, the drop to
the weaker route left no trace.

The last one is the quiet one. ``_parse`` falls back to ``actions[0]`` when
it cannot find an answer, so a failed deliberation returns a well-formed
Decision naming a real action — indistinguishable from a considered choice.
Downstream had no way to know the action was picked by position rather than
by reasoning.
"""
from __future__ import annotations

import asyncio

import pytest

from core.brain.deliberation import DeliberationController


class _LLM:
    def __init__(self, behaviour):
        self._behaviour = behaviour

    async def generate(self, _prompt, **_kwargs):
        return await self._behaviour()


def _deliberator(behaviour) -> DeliberationController:
    d = DeliberationController.__new__(DeliberationController)
    d.llm = _LLM(behaviour)
    d.trace = None
    return d


async def _hang():
    await asyncio.sleep(3600)


async def _boom():
    raise RuntimeError("provider exploded")


async def _good():
    return "Action: 2\nReason: it is safer\nConfidence: 0.8"


ACTIONS = ["restart the worker", "wait and retry"]


class TestTheCallIsBounded:
    @pytest.mark.asyncio
    async def test_a_hanging_model_does_not_hang_the_decision(self):
        d = _deliberator(_hang)
        decision = await asyncio.wait_for(
            d.deliberate("ctx", ACTIONS, use_native_system2=False,
                         deliberation_timeout_s=1.0),
            timeout=10.0,
        )
        assert decision.action in ACTIONS

    @pytest.mark.asyncio
    async def test_a_timeout_is_marked_degraded(self):
        d = _deliberator(_hang)
        decision = await d.deliberate(
            "ctx", ACTIONS, use_native_system2=False, deliberation_timeout_s=1.0,
        )
        assert "timeout" in decision.metadata["degraded"]
        assert decision.metadata["deliberated"] is False


class TestExceptionsBecomeDecisions:
    @pytest.mark.asyncio
    async def test_a_provider_error_does_not_escape(self):
        d = _deliberator(_boom)
        decision = await d.deliberate("ctx", ACTIONS, use_native_system2=False)
        assert decision.action in ACTIONS

    @pytest.mark.asyncio
    async def test_the_error_is_named_in_the_receipt(self):
        d = _deliberator(_boom)
        decision = await d.deliberate("ctx", ACTIONS, use_native_system2=False)
        assert "RuntimeError" in decision.metadata["degraded"]

    @pytest.mark.asyncio
    async def test_cancellation_still_propagates(self):
        """Cancellation is the caller's decision; absorbing it would turn a
        shutdown into a hung task."""

        async def _cancelled():
            raise asyncio.CancelledError()

        with pytest.raises(asyncio.CancelledError):
            await _deliberator(_cancelled).deliberate(
                "ctx", ACTIONS, use_native_system2=False,
            )


class TestAFallbackCannotPassAsAChoice:
    """_parse falls back to actions[0], so a failed deliberation returns a
    well-formed Decision naming a real action. Without a receipt that is
    indistinguishable from reasoning."""

    @pytest.mark.asyncio
    async def test_a_degraded_decision_has_zero_confidence(self):
        d = _deliberator(_boom)
        decision = await d.deliberate("ctx", ACTIONS, use_native_system2=False)
        assert decision.confidence == 0.0

    @pytest.mark.asyncio
    async def test_a_degraded_decision_explains_itself(self):
        d = _deliberator(_boom)
        decision = await d.deliberate("ctx", ACTIONS, use_native_system2=False)
        assert decision.reason

    @pytest.mark.asyncio
    async def test_a_real_deliberation_is_not_marked_degraded(self):
        d = _deliberator(_good)
        decision = await d.deliberate("ctx", ACTIONS, use_native_system2=False)
        assert decision.action == "wait and retry"
        assert decision.confidence == pytest.approx(0.8)
        assert not decision.metadata.get("degraded")


class TestTheDowngradeIsRecorded:
    @pytest.mark.asyncio
    async def test_declining_native_system2_is_noted(self, monkeypatch):
        d = _deliberator(_good)

        async def _declined(*_a, **_kw):
            return None

        monkeypatch.setattr(d, "_native_system2_deliberate", _declined)
        decision = await d.deliberate("ctx", ACTIONS, use_native_system2=True)
        assert decision.metadata["native_system2_declined"] == "native_system2_unavailable"

    @pytest.mark.asyncio
    async def test_a_native_decision_carries_no_downgrade_note(self, monkeypatch):
        from core.brain.deliberation import Decision

        d = _deliberator(_good)

        async def _decided(*_a, **_kw):
            return Decision(action=ACTIONS[0], reason="native", raw="", confidence=0.9, metadata={})

        monkeypatch.setattr(d, "_native_system2_deliberate", _decided)
        decision = await d.deliberate("ctx", ACTIONS, use_native_system2=True)
        assert not decision.metadata.get("native_system2_declined")
