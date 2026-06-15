import time
from types import SimpleNamespace

import pytest

from core.autonomic.core_monitor import AutonomicCore
from core.container import ServiceContainer


class _LoadedCortex:
    def __init__(self):
        self.reboot_reasons = []

    def is_alive(self):
        return True

    async def reboot_worker(self, *, reason):
        self.reboot_reasons.append(reason)


class _DeferredBrainstem:
    async def warmup(self):
        return False

    def get_lane_status(self):
        return {
            "state": "recovering",
            "last_error": "warmup_deferred",
            "conversation_ready": False,
        }


@pytest.mark.asyncio
async def test_idle_swap_reports_brainstem_incomplete_when_warmup_is_deferred(monkeypatch):
    cortex = _LoadedCortex()
    brainstem = _DeferredBrainstem()
    services = {
        "mlx_client": cortex,
        "brainstem_client": brainstem,
    }
    monkeypatch.setattr(
        ServiceContainer,
        "get",
        classmethod(lambda _cls, name, default=None: services.get(name, default)),
    )
    monkeypatch.setattr(
        "core.autonomic.core_monitor.psutil.virtual_memory",
        lambda: SimpleNamespace(percent=80.0),
    )

    monitor = AutonomicCore.__new__(AutonomicCore)
    monitor.orchestrator = SimpleNamespace(
        _last_user_interaction_time=time.time() - 600.0
    )
    monitor._idle_swap_done = False
    emitted = []

    async def _capture_status(message):
        emitted.append(message)

    monitor._emit_status = _capture_status

    await monitor._check_idle_model_swap()

    assert cortex.reboot_reasons == ["idle_budget_swap"]
    assert monitor._idle_swap_done is True
    assert emitted == [
        "Cortex hibernated (idle). Brainstem warmup incomplete; "
        "foreground demand will restore the Cortex."
    ]
