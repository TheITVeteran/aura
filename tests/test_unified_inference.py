import numpy as np
import pytest

from core.brain.homeostatic_modulator import InferenceModulation
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
async def test_run_ollama_fallback(monkeypatch):
    engine = UnifiedInferenceEngine()
    modulation = InferenceModulation(
        temperature=0.8,
        top_p=0.95,
        repetition_penalty=1.1,
        logit_bias={123: 1.0},
        head_weights=np.ones(32),
        urgency=0.5,
    )

    import core.brain.local_llm as local_llm

    class LocalBrainProbe:
        instances = []

        def __init__(self, model_name: str):
            self.model_name = model_name
            self.chat_calls = []
            LocalBrainProbe.instances.append(self)

        async def __aenter__(self):
            self.entered = True
            return self

        async def __aexit__(self, *_args):
            self.exited = True
            return None

        async def chat(self, **kwargs):
            self.chat_calls.append(kwargs)
            return {
                "response": "This is a response from Ollama fallback",
                "thought": "thought",
            }

    monkeypatch.setattr(local_llm, "LocalBrain", LocalBrainProbe)

    result = await engine._run_ollama_fallback(
        messages=[{"role": "user", "content": "hello"}],
        model_name="default_model",
        modulation=modulation,
        options=None,
    )

    assert result["response"] == "This is a response from Ollama fallback"
    assert result["thought"] == "thought"

    brain = LocalBrainProbe.instances[-1]
    assert brain.model_name == "default_model"
    assert brain.entered is True
    assert brain.exited is True
    called_options = brain.chat_calls[0]["options"]
    assert called_options["temperature"] == 0.8
    assert called_options["top_p"] == 0.95
    assert called_options["repeat_penalty"] == 1.1


@pytest.mark.asyncio
async def test_generate_unified_routing(monkeypatch):
    engine = UnifiedInferenceEngine()
    import core.brain.llm.model_registry as model_registry

    monkeypatch.setattr(model_registry, "get_local_backend", lambda: "llama_cpp")
    monkeypatch.setattr(model_registry, "get_lane_model_name", lambda name: "model_name")
    monkeypatch.setattr(
        model_registry,
        "get_lane_runtime_model_path",
        lambda name: "model_path.gguf",
    )
    monkeypatch.setattr(model_registry, "get_lane_context_window", lambda name: 8192)

    class LlamaProbe:
        def __init__(self):
            self.chat_calls = []

        def create_chat_completion(self, **kwargs):
            self.chat_calls.append(kwargs)
            return {
                "choices": [
                    {
                        "message": {"content": "<think>thinking process</think>final answer"},
                        "logprobs": {"content": []},
                    }
                ]
            }

    llama_probe = LlamaProbe()
    load_calls = []

    def get_llama_instance(model_path, context_size):
        load_calls.append((model_path, context_size))
        return llama_probe

    engine._get_llama_instance = get_llama_instance

    res = await engine.generate_unified(prompt="test prompt")
    assert res["response"] == "final answer"
    assert res["thought"] == "thinking process"
    assert load_calls == [("model_path.gguf", 8192)]
    assert llama_probe.chat_calls

    fallback_calls = []

    def missing_llama_instance(model_path, context_size):
        load_calls.append((model_path, context_size, "missing"))
        return None

    async def fallback_probe(*args, **kwargs):
        fallback_calls.append((args, kwargs))
        return {"response": "fallback answer", "thought": ""}

    engine._get_llama_instance = missing_llama_instance
    engine._run_ollama_fallback = fallback_probe

    res = await engine.generate_unified(prompt="test prompt")
    assert res["response"] == "fallback answer"
    assert fallback_calls


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

    async def _fake_request_json(*_args, **_kwargs):
        return 200, {"response": "<think>private scratch</think>visible answer"}, ""

    monkeypatch.setattr(brain, "_request_json", _fake_request_json)

    result = await brain.generate("hello")

    assert result == {"response": "visible answer", "thought": "private scratch"}
    assert any(
        "fell back to raw Ollama generation" in record.action
        for record in get_degradation_tracker().recent(
            subsystem="local_llm_unified_fallback", limit=5
        )
    )


@pytest.mark.asyncio
async def test_local_brain_stream_failure_yields_visible_marker(monkeypatch):
    get_degradation_tracker().reset()

    brain = LocalBrain(model_name="test-model")

    async def _failing_request_json(*_args, **_kwargs):
        raise ConnectionError("stream offline")

    monkeypatch.setattr(brain, "_request_json", _failing_request_json)

    chunks = [chunk async for chunk in brain.generate_text_stream_async("hello")]

    assert chunks and "Sovereign stream interrupted" in chunks[-1]
    assert any(
        "yielded stream interruption marker" in record.action
        for record in get_degradation_tracker().recent(subsystem="local_llm", limit=5)
    )
