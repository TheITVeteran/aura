from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.brain.compute_router import CloudConfig, ComputeBackend, ComputeRouter, InferenceTask
from core.container import ServiceContainer


class _Brain:
    async def think(self, prompt: str):
        return SimpleNamespace(content=f"local:{prompt}")


class _BrokenBrain:
    async def think(self, _prompt: str):
        return ""


class _CloudProvider:
    def __init__(self) -> None:
        self.calls = 0

    async def infer(self, task: InferenceTask, config: CloudConfig):
        self.calls += 1
        assert config.enabled is True
        return {"content": f"cloud:{task.prompt}", "success": True, "estimated_cost_usd": 0.002}


@pytest.fixture(autouse=True)
def _clear_container():
    ServiceContainer.clear()
    yield
    ServiceContainer.clear()


@pytest.mark.asyncio
async def test_compute_router_prefers_local_cognitive_engine():
    ServiceContainer.register_instance("cognitive_engine", _Brain(), required=False)
    router = ComputeRouter()

    result = await router.route(InferenceTask(prompt="hello"))

    assert result.success is True
    assert result.backend_used == ComputeBackend.LOCAL
    assert result.content == "local:hello"


@pytest.mark.asyncio
async def test_compute_router_does_not_use_cloud_without_opt_in():
    ServiceContainer.register_instance("cognitive_engine", _BrokenBrain(), required=False)
    ServiceContainer.register_instance("cloud_inference_runpod", _CloudProvider(), required=False)
    router = ComputeRouter(CloudConfig(enabled=False, provider="runpod"))

    result = await router.route(InferenceTask(prompt="hello"))

    assert result.success is False
    assert result.backend_used == ComputeBackend.LOCAL
    assert "returned no text" in result.error


@pytest.mark.asyncio
async def test_compute_router_uses_registered_cloud_provider_after_local_failure():
    ServiceContainer.register_instance("cognitive_engine", _BrokenBrain(), required=False)
    provider = _CloudProvider()
    ServiceContainer.register_instance("cloud_inference_runpod", provider, required=False)
    router = ComputeRouter(CloudConfig(enabled=True, provider="runpod"))

    result = await router.route(InferenceTask(prompt="hello", max_tokens=100))

    assert result.success is True
    assert result.backend_used == ComputeBackend.CLOUD
    assert result.content == "cloud:hello"
    assert result.estimated_cost_usd == 0.002
    assert provider.calls == 1


@pytest.mark.asyncio
async def test_compute_router_cloud_missing_provider_fails_closed():
    ServiceContainer.register_instance("cognitive_engine", _BrokenBrain(), required=False)
    router = ComputeRouter(CloudConfig(enabled=True, provider="runpod"))

    result = await router.route(InferenceTask(prompt="hello"))

    assert result.success is False
    assert result.backend_used == ComputeBackend.CLOUD
    assert result.error == "cloud_provider_plugin_missing:runpod"
