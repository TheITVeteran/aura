import pytest

from core.brain.local_llm import LocalBrain
from core.brain.unified_inference import UnifiedInferenceEngine
from core.runtime.errors import get_degradation_tracker


def test_the_messages_it_builds_are_the_ones_it_was_handed():
    """No identity text is added on the way to the model."""
    built = UnifiedInferenceEngine._build_messages(
        prompt="ignored",
        system_prompt=None,
        messages=[{"role": "user", "content": "What tide is it?"}],
    )
    assert built == [{"role": "user", "content": "What tide is it?"}]
    assert UnifiedInferenceEngine._build_messages(
        prompt="What tide is it?", system_prompt=None, messages=None
    ) == [{"role": "user", "content": "What tide is it?"}]
    assert not hasattr(UnifiedInferenceEngine, "_ensure_identity_anchor")


@pytest.mark.asyncio
async def test_generate_unified_routes_through_internal_mlx(monkeypatch):
    engine = UnifiedInferenceEngine()
    import core.brain.llm.mlx_client as mlx_client
    import core.brain.llm.model_registry as model_registry

    monkeypatch.setattr(model_registry, "get_lane_runtime_model_path", lambda name: "/models/aura-mlx")
    monkeypatch.setattr(model_registry, "get_lane_context_window", lambda name: 8192)

    calls = []

    class MLXProbe:
        async def generate_text_async(self, prompt, **kwargs):
            calls.append((prompt, kwargs))
            return "<think>thinking process</think>final answer"

    monkeypatch.setattr(mlx_client, "get_mlx_client", lambda **_kwargs: MLXProbe())

    res = await engine.generate_unified(prompt="test prompt", options={"num_predict": 96})

    assert res["response"] == "final answer"
    assert res["thought"] == "thinking process"
    assert calls
    assert calls[0][1]["max_tokens"] == 96
    assert calls[0][1]["messages"] == [{"role": "user", "content": "test prompt"}]


@pytest.mark.asyncio
async def test_local_brain_records_unified_generate_fallback(monkeypatch):
    import core.brain.unified_inference as unified_inference

    get_degradation_tracker().reset()

    class FailingUnifiedInference:
        async def generate_unified(self, **_kwargs):
            self.was_called = True
            raise RuntimeError("unified unavailable")

    monkeypatch.setattr(
        unified_inference,
        "UnifiedInferenceEngine",
        FailingUnifiedInference,
    )

    brain = LocalBrain(model_name="test-model")

    result = await brain.generate("hello")

    assert result["response"] == ""
    assert result["error"] == "internal_mlx_unified_inference_failed"
    assert any(
        "refused raw external generation fallback" in record.action
        for record in get_degradation_tracker().recent(
            subsystem="local_llm_unified_fallback", limit=5
        )
    )


@pytest.mark.asyncio
async def test_local_brain_stream_failure_yields_visible_marker(monkeypatch):
    import core.brain.unified_inference as unified_inference

    get_degradation_tracker().reset()

    class FailingUnifiedInference:
        async def generate_unified(self, **_kwargs):
            self.called = True
            raise ConnectionError("internal MLX stream offline")

    monkeypatch.setattr(
        unified_inference,
        "UnifiedInferenceEngine",
        FailingUnifiedInference,
    )
    brain = LocalBrain(model_name="test-model")

    chunks = [chunk async for chunk in brain.generate_text_stream_async("hello")]

    assert chunks and "Sovereign stream interrupted" in chunks[-1]
    assert any(
        "refused raw external generation fallback" in record.action
        for record in get_degradation_tracker().recent(
            subsystem="local_llm_unified_fallback", limit=5
        )
    )
