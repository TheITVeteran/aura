import time

import pytest

from core.brain.llm.llm_router import IntelligentLLMRouter, LLMEndpoint, LLMTier
from core.brain.llm.mlx_client import MLXLocalClient


class ProcessProbe:
    def __init__(self):
        self.kill_calls = 0
        self.join_calls = []

    def is_alive(self):
        return True

    def kill(self):
        self.kill_calls += 1

    def join(self, timeout=None):
        self.join_calls.append(timeout)


class MetalFailingAdapter:
    def __init__(self):
        self.reboot_calls = []

    async def reboot_worker(self, **kwargs):
        self.reboot_calls.append(dict(kwargs))

    async def call(self, prompt, **kwargs):
        return False, "", {"error": "RESOURCE_EXHAUSTED: [metal::Device] error 3 - No such process"}


@pytest.mark.asyncio
async def test_neural_reboot_purges_worker():
    client = MLXLocalClient(model_path="test-model")
    process = ProcessProbe()
    client._process = process
    client._init_done = True

    await client.reboot_worker()

    assert client._process is None
    assert client._init_done is False
    assert process.kill_calls == 1
    assert process.join_calls == [2.0]


@pytest.mark.asyncio
async def test_router_metal_recovery(monkeypatch):
    import core.container as container

    monkeypatch.setattr(container.ServiceContainer, "get", lambda *args, **kwargs: None)
    router = IntelligentLLMRouter()
    adapter = MetalFailingAdapter()

    endpoint = LLMEndpoint(name="MLX-Test", tier=LLMTier.PRIMARY)
    router.register_endpoint(endpoint)
    router.adapters["MLX-Test"] = adapter

    failure = None
    try:
        result = await router.think("test prompt", prefer_tier=LLMTier.PRIMARY)
    except RuntimeError as exc:
        failure = exc
    else:
        assert result is not None

    assert adapter.reboot_calls == [
        {
            "reason": "router_backend_failure:RESOURCE_EXHAUSTED",
            "mark_failed": False,
        }
    ]
    assert router._recovery_states["MLX-Test"] < time.time() + 20
    if failure is not None:
        assert "MLX-Test" in str(failure) or "endpoint" in str(failure).lower()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
