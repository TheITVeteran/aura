import pytest

from core.brain.local_llm import LocalBrain
from core.brain.unified_inference import UnifiedInferenceEngine
from core.runtime.errors import get_degradation_tracker


@pytest.mark.asyncio
async def test_ensure_identity_anchor():
    engine = UnifiedInferenceEngine()

    # Case 1: Empty messages
    messages = []
    engine._ensure_identity_anchor(messages)
    assert len(messages) == 1
    assert messages[0]["role"] == "system"
    assert "You are Aura Luna" in messages[0]["content"]

    # Case 2: System message exists but lacks identity anchor
    messages = [{"role": "system", "content": "Keep it technical."}]
    engine._ensure_identity_anchor(messages)
    assert len(messages) == 1
    assert "You are Aura Luna" in messages[0]["content"]
    assert "Keep it technical" in messages[0]["content"]

    # Case 3: System message already has the anchor
    existing = "You are Aura Luna. Speak with direct first-person continuity... Keep it technical."
    messages = [{"role": "system", "content": existing}]
    engine._ensure_identity_anchor(messages)
    assert len(messages) == 1
    assert messages[0]["content"] == existing


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
    assert calls[0][1]["messages"][0]["role"] == "system"


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
