import asyncio
from types import SimpleNamespace

import pytest

from core.brain.llm import autonomous_brain_integration as brain_module
from core.brain.llm.autonomous_brain_integration import (
    AutonomousCognitiveEngine,
    LLMEndpoint,
    LLMTier,
)


def _engine_with_router(router):
    engine = AutonomousCognitiveEngine.__new__(AutonomousCognitiveEngine)
    engine.llm_router = router
    engine._last_think_error_time = 0.0
    engine._agentic_semaphore = asyncio.Semaphore(1)

    async def _no_state():
        return None

    engine._get_live_state = _no_state
    return engine


@pytest.mark.asyncio
async def test_think_uses_reflex_response_after_autonomous_and_router_failures(monkeypatch):
    recorded = []
    monkeypatch.setattr(
        brain_module,
        "_record_brain_degradation",
        lambda error, **kwargs: recorded.append((error, kwargs)),
    )

    class FailingRouter:
        def __init__(self):
            self.endpoints = {}
            self.health_monitor = SimpleNamespace(is_healthy=lambda _name: False)
            self.calls = 0

        async def think(self, *_args, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("primary routing failed")
            raise OSError("fallback routing failed")

    result = await _engine_with_router(FailingRouter()).think("Explain continuity.")

    assert result["fallback"] == "reflex"
    assert result["confidence"] == 0.1
    assert "Absolute failure" not in result["content"]
    assert [entry[1]["action"] for entry in recorded] == [
        "entered standard router fallback after autonomous thinking path failed",
        "served emergency reflex response after router fallback failed",
    ]


@pytest.mark.asyncio
async def test_agentic_path_continues_with_empty_tool_map_when_tool_discovery_fails(monkeypatch):
    recorded = []
    monkeypatch.setattr(
        brain_module,
        "_record_brain_degradation",
        lambda error, **kwargs: recorded.append((error, kwargs)),
    )
    monkeypatch.setattr(
        brain_module,
        "build_agentic_tool_map",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("tool registry offline")),
    )

    class AgenticClient:
        async def think_and_act(self, _objective, _system_prompt, **kwargs):
            return {"content": "acted", "tools": kwargs.get("tools")}

    # CP126 af6a3b7d: agentic execution is restricted to the declared LOCAL
    # agentic tiers, so this stands up the endpoint under a name that is on
    # that allowlist rather than relying on dictionary order.
    from core.brain.llm.model_registry import DEEP_ENDPOINT

    endpoint = LLMEndpoint(
        name=DEEP_ENDPOINT,
        tier=LLMTier.SECONDARY,
        model_name="agentic-test",
        client=AgenticClient(),
    )
    router = SimpleNamespace(
        endpoints={DEEP_ENDPOINT: endpoint},
        health_monitor=SimpleNamespace(is_healthy=lambda _name: True),
    )

    result = await _engine_with_router(router).think(
        "Research and act.", tools={"declared": True},
    )

    assert result["content"] == "acted"
    # The caller's tools reach the client; a discovery failure would leave {}.
    assert result["tools"] == {"declared": True}
    assert result["route"] == "agentic"
    assert result["endpoint"] == DEEP_ENDPOINT
    # CP126 480a31a0: the map's provenance travels with the result.
    assert result["tool_authority"]["source"] == "caller_supplied"


@pytest.mark.asyncio
async def test_agentic_tool_discovery_failure_yields_an_empty_map(monkeypatch):
    """A tool registry that is offline must not stop the turn — and the
    degradation is recorded as a capability loss, not a safe fallback."""
    recorded = []
    monkeypatch.setattr(
        brain_module,
        "_record_brain_degradation",
        lambda error, **kwargs: recorded.append((error, kwargs)),
    )
    monkeypatch.setattr(
        brain_module,
        "build_agentic_tool_map",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("tool registry offline")),
    )

    class AgenticClient:
        async def think_and_act(self, _objective, _system_prompt, **kwargs):
            return {"content": "acted", "tools": kwargs.get("tools")}

    from core.brain.llm.model_registry import DEEP_ENDPOINT

    endpoint = LLMEndpoint(
        name=DEEP_ENDPOINT,
        tier=LLMTier.SECONDARY,
        model_name="agentic-test",
        client=AgenticClient(),
    )
    router = SimpleNamespace(
        endpoints={DEEP_ENDPOINT: endpoint},
        health_monitor=SimpleNamespace(is_healthy=lambda _name: True),
    )

    result = await _engine_with_router(router).think(
        "Research and act.", context={"require_tools": True},
    )

    assert result["content"] == "acted"
    assert result["tools"] == {}
    assert result["tool_authority"]["source"] == "unavailable"
    assert recorded[0][1]["action"] == (
        "continued agentic reasoning without dynamically built tool map"
    )
    assert recorded[0][1]["kind"] == "capability_loss"
