import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.ops.resilient_boot import BootStatus, ResilientBoot


class _DummyOrchestrator:
    __slots__ = ()


async def _failing_llm_stage():
    await asyncio.sleep(0)
    raise RuntimeError("llama_server_missing")


def _stub_boot_dependencies(monkeypatch):
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
async def test_stage_llm_prepares_client_without_warmup():
    boot = ResilientBoot(_DummyOrchestrator())
    client = MagicMock()
    client.warmup = AsyncMock()

    with patch("core.brain.llm.mlx_client.get_mlx_client", return_value=client) as get_client:
        with patch("core.brain.llm.model_registry.get_local_backend", return_value="mlx"):
            with patch("core.brain.llm.model_registry.get_runtime_model_path", return_value="/models/active"):
                with patch("core.brain.llm.model_registry.ACTIVE_MODEL", "ACTIVE"):
                    await boot._stage_llm()

    get_client.assert_called_once_with(model_path="/models/active")
    client.warmup.assert_not_awaited()


@pytest.mark.asyncio
async def test_resilient_boot_strict_runtime_fails_closed_on_llm_stage_error(service_container, monkeypatch):
    _stub_boot_dependencies(monkeypatch)
    monkeypatch.setenv("AURA_STRICT_RUNTIME", "1")

    orchestrator = SimpleNamespace(status=SimpleNamespace(initialized=False, health_metrics={}))
    boot = ResilientBoot(orchestrator)

    boot.stages = [("LLM Infrastructure", _failing_llm_stage)]

    with pytest.raises(RuntimeError, match="Strict runtime critical boot stage failed: LLM Infrastructure"):
        await boot.ignite()


@pytest.mark.asyncio
async def test_resilient_boot_non_strict_runtime_degrades_on_llm_stage_error(service_container, monkeypatch):
    _stub_boot_dependencies(monkeypatch)
    monkeypatch.delenv("AURA_STRICT_RUNTIME", raising=False)

    orchestrator = SimpleNamespace(status=SimpleNamespace(initialized=False, health_metrics={}))
    boot = ResilientBoot(orchestrator)

    boot.stages = [("LLM Infrastructure", _failing_llm_stage)]

    status = await boot.ignite()

    assert status is BootStatus.DEGRADED
    assert boot.results["LLM Infrastructure"].success is False
