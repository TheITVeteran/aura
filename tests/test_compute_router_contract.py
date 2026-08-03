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

    result = await router.route(
        InferenceTask(
            prompt="hello",
            max_tokens=100,
            cloud_authorized=True,
            authorized_by="test_compute_router_contract",
        )
    )

    assert result.success is True
    assert result.backend_used == ComputeBackend.CLOUD
    assert result.content == "cloud:hello"
    assert result.estimated_cost_usd == 0.002
    assert provider.calls == 1


@pytest.mark.asyncio
async def test_compute_router_cloud_missing_provider_fails_closed():
    ServiceContainer.register_instance("cognitive_engine", _BrokenBrain(), required=False)
    router = ComputeRouter(CloudConfig(enabled=True, provider="runpod"))

    result = await router.route(
        InferenceTask(
            prompt="hello",
            cloud_authorized=True,
            authorized_by="test_compute_router_contract",
        )
    )

    assert result.success is False
    # CP126 9e684e4b: the terminal result used to carry ONLY the cloud-side
    # error, so a dual failure was indistinguishable from a cloud-only one
    # and the original local outage could not be diagnosed. Both are named.
    assert "local:" in result.error
    assert "cloud_provider_plugin_missing:runpod" in result.error


# ── CP126: a process-global switch is not authority to disclose ──────────


@pytest.mark.asyncio
async def test_the_global_flag_alone_no_longer_sends_content_to_cloud():
    """CP126 a6f7851c: the exact behaviour this test file used to encode.

    Cloud disclosure was authorised by mutable process configuration alone —
    no principal, no sensitivity, no data-residency, no scoped authority. Once
    an operator enabled cloud for one purpose, EVERY caller in the process
    could send prompts and metadata off the host.
    """
    ServiceContainer.register_instance("cognitive_engine", _BrokenBrain(), required=False)
    provider = _CloudProvider()
    ServiceContainer.register_instance("cloud_inference_runpod", provider, required=False)
    router = ComputeRouter(CloudConfig(enabled=True, provider="runpod"))

    # Cloud is ON, but this task never said its content may leave.
    result = await router.route(InferenceTask(prompt="something private"))

    assert provider.calls == 0, "content left the host on a global flag alone"
    assert result.success is False


@pytest.mark.asyncio
async def test_an_unattributable_authorisation_is_refused():
    """A disclosure nobody signed cannot be audited or withdrawn."""
    ServiceContainer.register_instance("cognitive_engine", _BrokenBrain(), required=False)
    provider = _CloudProvider()
    ServiceContainer.register_instance("cloud_inference_runpod", provider, required=False)
    router = ComputeRouter(CloudConfig(enabled=True, provider="runpod"))

    result = await router.route(
        InferenceTask(prompt="hello", cloud_authorized=True, authorized_by="  ")
    )

    assert provider.calls == 0
    assert result.success is False


@pytest.mark.asyncio
async def test_an_unlisted_provider_never_receives_the_task_or_the_key():
    """CP126 5c3beeaa: any string selected which object got the prompt and key."""
    ServiceContainer.register_instance("cognitive_engine", _BrokenBrain(), required=False)
    impostor = _CloudProvider()
    ServiceContainer.register_instance("cloud_inference_evilcorp", impostor, required=False)
    router = ComputeRouter(
        CloudConfig(enabled=True, provider="evilcorp", api_key="sk-secret")
    )

    result = await router.route(
        InferenceTask(prompt="hello", cloud_authorized=True, authorized_by="test")
    )

    assert impostor.calls == 0
    assert result.success is False


@pytest.mark.asyncio
async def test_a_provider_cannot_declare_its_own_success():
    """CP126 80561bf5: success=True with empty content was recorded as success."""

    class _LyingProvider:
        async def infer(self, task, config=None):
            return {"success": True, "content": "", "error": None}

    ServiceContainer.register_instance("cognitive_engine", _BrokenBrain(), required=False)
    ServiceContainer.register_instance("cloud_inference_runpod", _LyingProvider(), required=False)
    router = ComputeRouter(CloudConfig(enabled=True, provider="runpod"))

    result = await router.route(
        InferenceTask(prompt="hello", cloud_authorized=True, authorized_by="test")
    )

    assert result.success is False


@pytest.mark.asyncio
async def test_a_provider_is_never_called_twice_by_a_typeerror_probe():
    """CP126 181f8490: a TypeError from INSIDE the provider re-invoked it."""

    class _ChargesThenRaises:
        def __init__(self):
            self.calls = 0

        async def infer(self, task, config=None):
            self.calls += 1
            raise TypeError("a defect inside the provider, not a signature mismatch")

    provider = _ChargesThenRaises()
    ServiceContainer.register_instance("cognitive_engine", _BrokenBrain(), required=False)
    ServiceContainer.register_instance("cloud_inference_runpod", provider, required=False)
    router = ComputeRouter(CloudConfig(enabled=True, provider="runpod"))

    await router.route(
        InferenceTask(prompt="hello", cloud_authorized=True, authorized_by="test")
    )

    assert provider.calls == 1, "the provider ran twice; a charge would double"


@pytest.mark.asyncio
async def test_concurrent_routes_cannot_all_spend_the_same_budget():
    """CP126 f1c0e39a: check-call-charge had no lock, so the cap was a suggestion."""
    import asyncio as _asyncio

    class _SlowProvider:
        def __init__(self):
            self.calls = 0

        async def infer(self, task, config=None):
            self.calls += 1
            await _asyncio.sleep(0.05)
            return {"success": True, "content": "cloud", "estimated_cost_usd": 1.0}

    provider = _SlowProvider()
    ServiceContainer.register_instance("cognitive_engine", _BrokenBrain(), required=False)
    ServiceContainer.register_instance("cloud_inference_runpod", provider, required=False)
    router = ComputeRouter(
        CloudConfig(
            enabled=True, provider="runpod", max_monthly_budget_usd=2.0,
            cost_per_token=0.01,
        )
    )

    tasks = [
        router.route(
            InferenceTask(
                prompt="hello", max_tokens=100,
                cloud_authorized=True, authorized_by="test",
            )
        )
        for _ in range(10)
    ]
    await _asyncio.gather(*tasks)

    # Each request reserves 100 * 0.01 = $1.00 against a $2.00 budget.
    assert provider.calls <= 2, (
        f"{provider.calls} concurrent calls all saw the same headroom; the "
        "budget only bound when requests arrived one at a time"
    )


@pytest.mark.asyncio
async def test_a_dual_failure_reports_both_errors():
    """CP126 9e684e4b: the cloud error overwrote the local one."""

    class _FailingProvider:
        async def infer(self, task, config=None):
            return {"success": False, "error": "cloud exploded"}

    ServiceContainer.register_instance("cognitive_engine", _BrokenBrain(), required=False)
    ServiceContainer.register_instance("cloud_inference_runpod", _FailingProvider(), required=False)
    router = ComputeRouter(CloudConfig(enabled=True, provider="runpod"))

    result = await router.route(
        InferenceTask(prompt="hello", cloud_authorized=True, authorized_by="test")
    )

    assert "local:" in result.error and "cloud:" in result.error


@pytest.mark.asyncio
async def test_a_nonsense_provider_cost_never_corrupts_the_ledger():
    """CP126 e07da2b4: any post-hoc cost was added to spend unchecked."""

    class _NaNProvider:
        async def infer(self, task, config=None):
            return {
                "success": True,
                "content": "cloud",
                "estimated_cost_usd": float("nan"),
            }

    ServiceContainer.register_instance("cognitive_engine", _BrokenBrain(), required=False)
    ServiceContainer.register_instance("cloud_inference_runpod", _NaNProvider(), required=False)
    router = ComputeRouter(CloudConfig(enabled=True, provider="runpod"))

    result = await router.route(
        InferenceTask(
            prompt="hello", max_tokens=10,
            cloud_authorized=True, authorized_by="test",
        )
    )

    import math

    assert math.isfinite(result.estimated_cost_usd)
    assert math.isfinite(router._monthly_spend_usd)
