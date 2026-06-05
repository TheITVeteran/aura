import asyncio

import pytest

from core.container import ContainerError, ServiceContainer
from core.resilience.resource_arbitrator import get_resource_arbitrator


@pytest.mark.asyncio
async def test_arbitrator_blocks_evolution_while_inference_active(tmp_path):
    arbitrator = get_resource_arbitrator()
    original_lock_path = arbitrator._lock_path
    original_inference_active = arbitrator._inference_active
    original_evolution_active = arbitrator._evolution_active
    original_mp_fd = arbitrator._mp_fd
    arbitrator._lock_path = str(tmp_path / "vram.lock")
    arbitrator._inference_active = False
    arbitrator._evolution_active = False
    arbitrator._mp_fd = None

    try:
        async with arbitrator.inference_context(timeout=0.25):
            evolution_result = await arbitrator.acquire_evolution(timeout=0.05)
            assert evolution_result is False

        async with arbitrator.evolution_context(timeout=0.25) as acquired:
            assert acquired is True
    finally:
        if arbitrator._evolution_active:
            arbitrator.release_evolution()
        arbitrator._lock_path = original_lock_path
        arbitrator._inference_active = original_inference_active
        arbitrator._evolution_active = original_evolution_active
        arbitrator._mp_fd = original_mp_fd


def test_container_registration_lock_rejects_new_factories():
    saved_services = dict(ServiceContainer._services)
    saved_aliases = dict(ServiceContainer._aliases)
    saved_locked = ServiceContainer._registration_locked
    try:
        ServiceContainer.clear()
        ServiceContainer.register("test_service", lambda: {"id": 1})
        ServiceContainer.lock_registration()

        with pytest.raises(ContainerError):
            ServiceContainer.register("rogue_service", lambda: {"id": 666})

        assert "rogue_service" not in ServiceContainer._services
        assert ServiceContainer.get("test_service") == {"id": 1}
    finally:
        ServiceContainer._services = saved_services
        ServiceContainer._aliases = saved_aliases
        ServiceContainer._registration_locked = saved_locked


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
