"""The reasoning amplifier must fire on the dominant phase response lane.

These call UnitaryResponsePhase._maybe_amplify_response directly (with stubs) so we
prove the live wiring without standing up the whole phase graph.
"""
from __future__ import annotations

import types

import pytest

from core.phases.response_generation import ResponseGenerationPhase
from core.phases.response_generation_unitary import UnitaryResponsePhase
from core.state.aura_state import AuraState


class _StubLLM:
    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.calls = 0

    async def think(self, prompt: str, **kwargs) -> str:
        self.calls += 1
        return self.answer


class _StubRouter:
    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.calls = 0
        self.kwargs = []

    async def think(self, **kwargs) -> str:
        self.calls += 1
        self.kwargs.append(kwargs)
        return self.answer


def _self_stub():
    # The method only touches self._last_reasoning_receipt.
    return types.SimpleNamespace(_last_reasoning_receipt=None)


def _state_stub():
    return types.SimpleNamespace(metadata={})


def _phase_stub() -> ResponseGenerationPhase:
    return ResponseGenerationPhase(types.SimpleNamespace())


@pytest.mark.asyncio
async def test_phase_amplifies_verified_math_turn():
    llm = _StubLLM("The product: 12 * 12 = 144")
    state = _state_stub()
    me = _self_stub()
    out = await UnitaryResponsePhase._maybe_amplify_response(
        me,
        objective="compute the product of the two given values",
        draft="The product is 12 * 12 = 150",  # wrong first draft
        llm=llm,
        state=state,
        request_timeout=20.0,
        is_user_facing=True,
        is_background=False,
        proof_or_benchmark=False,
    )
    assert "144" in out
    assert me._last_reasoning_receipt is not None
    assert state.metadata.get("reasoning_receipt", {}).get("task_type") == "math"
    assert llm.calls >= 1


@pytest.mark.asyncio
async def test_phase_keeps_draft_when_amplified_unverified():
    # Amplifier generates an arithmetic error → not verified → keep the original draft.
    llm = _StubLLM("The product: 12 * 12 = 150")
    me = _self_stub()
    out = await UnitaryResponsePhase._maybe_amplify_response(
        me,
        objective="compute the product of the two given values",
        draft="ORIGINAL DRAFT",
        llm=llm,
        state=_state_stub(),
        request_timeout=20.0,
        is_user_facing=True,
        is_background=False,
        proof_or_benchmark=False,
    )
    assert out == "ORIGINAL DRAFT"


@pytest.mark.asyncio
async def test_phase_skips_casual_turn():
    llm = _StubLLM("hello there")
    out = await UnitaryResponsePhase._maybe_amplify_response(
        _self_stub(),
        objective="hey how are you doing today",
        draft="DRAFT",
        llm=llm,
        state=_state_stub(),
        request_timeout=20.0,
        is_user_facing=True,
        is_background=False,
        proof_or_benchmark=False,
    )
    assert out == "DRAFT"
    assert llm.calls == 0


@pytest.mark.asyncio
async def test_phase_skips_background_and_proof():
    llm = _StubLLM("12 * 12 = 144")
    for kwargs in ({"is_background": True}, {"proof_or_benchmark": True}, {"is_user_facing": False}):
        base = dict(
            objective="compute the product of the two values",
            draft="DRAFT",
            llm=llm,
            state=_state_stub(),
            request_timeout=20.0,
            is_user_facing=True,
            is_background=False,
            proof_or_benchmark=False,
        )
        base.update(kwargs)
        out = await UnitaryResponsePhase._maybe_amplify_response(_self_stub(), **base)
        assert out == "DRAFT"
    assert llm.calls == 0


@pytest.mark.asyncio
async def test_phase_respects_env_disable(monkeypatch):
    monkeypatch.setenv("AURA_REASONING_AMPLIFIER_V2", "0")
    llm = _StubLLM("12 * 12 = 144")
    out = await UnitaryResponsePhase._maybe_amplify_response(
        _self_stub(),
        objective="compute the product of the two values",
        draft="DRAFT",
        llm=llm,
        state=_state_stub(),
        request_timeout=20.0,
        is_user_facing=True,
        is_background=False,
        proof_or_benchmark=False,
    )
    assert out == "DRAFT"
    assert llm.calls == 0


@pytest.mark.asyncio
async def test_active_response_generation_phase_amplifies_verified_math_turn():
    router = _StubRouter("The product: 12 * 12 = 144")
    state = AuraState.default()
    phase = _phase_stub()

    out = await phase._maybe_amplify_response(
        objective="compute the product of the two given values",
        draft="The product is 12 * 12 = 150",
        router=router,
        state=state,
        request_timeout=20.0,
        origin="desktop",
        tier="primary",
        runtime_context={"desktop_cognitive_engine_required": True},
        is_user_facing=True,
        is_background=False,
        proof_or_benchmark=False,
    )

    assert "144" in out
    assert phase._last_reasoning_receipt is not None
    assert state.response_modifiers["reasoning_receipt"]["task_type"] == "math"
    assert state.response_modifiers["reasoning_amplifier_v2_active_phase"]["adopted"] is True
    assert router.calls >= 1
    assert router.kwargs[0]["desktop_cognitive_engine_required"] is True


@pytest.mark.asyncio
async def test_active_response_generation_phase_skips_casual_turn():
    router = _StubRouter("hello there")
    state = AuraState.default()
    phase = _phase_stub()

    out = await phase._maybe_amplify_response(
        objective="hey how are you doing today",
        draft="DRAFT",
        router=router,
        state=state,
        request_timeout=20.0,
        origin="desktop",
        tier="primary",
        runtime_context={"desktop_cognitive_engine_required": True},
        is_user_facing=True,
        is_background=False,
        proof_or_benchmark=False,
    )

    assert out == "DRAFT"
    assert router.calls == 0
