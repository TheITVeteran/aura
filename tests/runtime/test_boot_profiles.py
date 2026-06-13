"""tests/runtime/test_boot_profiles.py — Assertions for boot profiles including minimal boot.
"""
from __future__ import annotations

import os
import pytest
import aura_main
from aura_main import boot_aura_runtime, _stop_orchestrator_once
from core.config import config


@pytest.mark.asyncio
async def test_minimal_boot_profile(monkeypatch):
    """Assert that minimal profile boot disables heavy devices/subsystems and forces deterministic LLM."""
    # Patch bootstrap_lock to skip locking when running tests
    monkeypatch.setattr(aura_main, "bootstrap_lock", lambda skip_lock=False: None)

    # Replace orchestration with deterministic lightweight services for this boot contract.
    class DeterministicRouter:
        async def think(self, prompt: str) -> str:
            return "deterministic thinking response"

    class DeterministicOrchestrator:
        def stop(self):
            return None

    async def boot_deterministic_orchestrator(*args, **kwargs):
        from core.container import ServiceContainer
        # Unlock registry if locked to allow registering the lightweight llm_router.
        ServiceContainer._locked = False
        ServiceContainer.register_instance("llm_router", DeterministicRouter())
        return DeterministicOrchestrator()

    monkeypatch.setattr(aura_main, "_boot_runtime_orchestrator", boot_deterministic_orchestrator)

    # Run the boot routine with minimal profile
    orchestrator = await boot_aura_runtime(profile="minimal")
    
    try:
        # Assert env vars are populated correctly
        assert os.environ.get("AURA_BOOT_PROFILE") == "minimal"
        assert os.environ.get("AURA_USE_MOCK_LLM") == "1"
        assert os.environ.get("AURA_ENABLE_CAMERA") == "0"
        assert os.environ.get("AURA_ENABLE_MIC") == "0"
        assert os.environ.get("AURA_ENABLE_DESKTOP") == "0"
        assert os.environ.get("AURA_DISABLE_CLOUD") == "1"
        
        # Assert configuration properties are disabled
        assert config.skeletal_mode is True
        assert config.features.camera_enabled is False
        assert config.features.voice_enabled is False
        assert config.security.allow_network_access is False
        
        # Assert deterministic LLM response is returned rather than executing heavy model routes.
        from core.container import ServiceContainer
        router = ServiceContainer.get("llm_router")
        assert router is not None
        
        response = await router.think("hello minimal aura")
        assert len(response) > 0
    finally:
        # Clean shutdown of orchestrator to avoid hanging during pytest teardown
        await _stop_orchestrator_once(orchestrator, reason="test_exit")
        
        # Reset the event bus ready signal to allow subsequent tests to run
        from core.event_bus import reset_event_bus
        await reset_event_bus()
        # Reset ServiceContainer locked state
        from core.container import ServiceContainer
        ServiceContainer._locked = False


@pytest.mark.asyncio
async def test_stop_orchestrator_can_flush_before_global_shutdown_flag():
    """Desktop shutdown must let the runtime owner flush state before API teardown."""

    from core.runtime.shutdown_coordinator import clear_shutdown_request, is_shutdown_requested

    clear_shutdown_request()
    calls: list[str] = []

    class FakeOrchestrator:
        async def stop(self):
            calls.append("stop")

    try:
        await _stop_orchestrator_once(
            FakeOrchestrator(),
            reason="desktop_exit",
            request_global_shutdown=False,
        )

        assert calls == ["stop"]
        assert is_shutdown_requested() is False
    finally:
        clear_shutdown_request()
