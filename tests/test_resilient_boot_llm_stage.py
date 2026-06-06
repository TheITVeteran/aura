import asyncio
from types import SimpleNamespace

import pytest

from core.ops.resilient_boot import BootStatus, ResilientBoot


class AsyncCallRecorder:
    def __init__(self, result=None):
        self.result = result
        self.calls = []

    async def __call__(self, *args, **kwargs):
        self.calls.append(SimpleNamespace(args=args, kwargs=kwargs))
        return self.result


class CallRecorder:
    def __init__(self, result=None):
        self.result = result
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append(SimpleNamespace(args=args, kwargs=kwargs))
        return self.result


class _OrchestratorShell:
    __slots__ = ()


async def _failing_llm_stage():
    await asyncio.sleep(0)
    raise RuntimeError("llama_server_missing")


def _install_boot_dependencies(monkeypatch):
    immunity = SimpleNamespace(
        hook_system=lambda: None,
        registry=SimpleNamespace(
            match_and_repair=lambda *_args, **_kwargs: None,
            log_sieve=lambda *_args, **_kwargs: [],
        ),
        audit_error=lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr("core.resilience.immunity_hyphae.get_immunity", lambda: immunity)
    monkeypatch.setattr("core.resilience.stall_watchdog.start_watchdog", lambda: SimpleNamespace())
    monkeypatch.setattr("core.resilience.diagnostic_hub.get_diagnostic_hub", lambda: SimpleNamespace())
    monkeypatch.setattr("core.reaper.register_reaper_pid", lambda *_args, **_kwargs: None)
    return immunity


@pytest.mark.asyncio
async def test_stage_llm_prepares_client_without_warmup(monkeypatch):
    boot = ResilientBoot(_OrchestratorShell())
    client = SimpleNamespace(warmup=AsyncCallRecorder())
    get_client = CallRecorder(result=client)

    monkeypatch.setattr("core.brain.llm.mlx_client.get_mlx_client", get_client)
    monkeypatch.setattr("core.brain.llm.model_registry.get_local_backend", lambda: "mlx")
    model_path_lookup = CallRecorder(result="/models/active")
    monkeypatch.setattr("core.brain.llm.model_registry.get_runtime_model_path", model_path_lookup)
    monkeypatch.setattr("core.brain.llm.model_registry.ACTIVE_MODEL", "ACTIVE")
    await boot._stage_llm()

    assert len(model_path_lookup.calls) == 1
    assert model_path_lookup.calls[0].args == ("ACTIVE",)
    assert len(get_client.calls) == 1
    assert get_client.calls[0].kwargs == {"model_path": "/models/active"}
    assert client.warmup.calls == []


@pytest.mark.asyncio
async def test_resilient_boot_strict_runtime_fails_closed_on_llm_stage_error(service_container, monkeypatch):
    _install_boot_dependencies(monkeypatch)
    monkeypatch.setenv("AURA_STRICT_RUNTIME", "1")

    orchestrator = SimpleNamespace(status=SimpleNamespace(initialized=False, health_metrics={}))
    boot = ResilientBoot(orchestrator)

    boot.stages = [("LLM Infrastructure", _failing_llm_stage)]

    with pytest.raises(RuntimeError, match="Strict runtime critical boot stage failed: LLM Infrastructure"):
        await boot.ignite()


@pytest.mark.asyncio
async def test_resilient_boot_non_strict_runtime_degrades_on_llm_stage_error(service_container, monkeypatch):
    _install_boot_dependencies(monkeypatch)
    monkeypatch.delenv("AURA_STRICT_RUNTIME", raising=False)

    orchestrator = SimpleNamespace(status=SimpleNamespace(initialized=False, health_metrics={}))
    boot = ResilientBoot(orchestrator)

    boot.stages = [("LLM Infrastructure", _failing_llm_stage)]

    status = await boot.ignite()

    assert status is BootStatus.DEGRADED
    assert boot.results["LLM Infrastructure"].success is False
