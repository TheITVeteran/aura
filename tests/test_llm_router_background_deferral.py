import pytest

from core.brain.llm.llm_router import IntelligentLLMRouter, LLMEndpoint, LLMTier


class EndpointClientFixture:
    def __init__(self):
        self.calls = []

    async def think(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return True, "7B response", {}


class InferenceGateFixture:
    def _background_local_deferral_reason(self):
        return "cortex_startup_quiet"


@pytest.mark.asyncio
async def test_legacy_llm_router_defers_background_inference_when_gate_is_guarded(monkeypatch):
    router = IntelligentLLMRouter()
    tertiary = EndpointClientFixture()
    router.register_endpoint(
        LLMEndpoint(
            name="Brainstem",
            tier=LLMTier.TERTIARY,
            model_name="brainstem-7b",
            client=tertiary,
        )
    )

    fake_gate = InferenceGateFixture()
    monkeypatch.setattr(
        "core.container.ServiceContainer.get",
        lambda name, default=None: fake_gate if name == "inference_gate" else default,
    )
    result = await router.think(
        "Idle thought",
        prefer_tier="tertiary",
        origin="system",
        is_background=True,
    )

    assert result == ""
    assert tertiary.calls == []
