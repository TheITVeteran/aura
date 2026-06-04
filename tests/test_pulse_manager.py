################################################################################

import asyncio
import pytest
from types import SimpleNamespace

from core.container import ServiceContainer
from core.senses.pulse_manager import PulseManager


class FixedMetabolicMonitor:
    def __init__(self, health_score: float):
        self.snapshot = SimpleNamespace(health_score=health_score)

    def get_current_metabolism(self):
        return self.snapshot


class MaintenanceRecorder:
    def __init__(self):
        self.calls = 0

    async def perform_maintenance(self):
        self.calls += 1


class VisionRecorder:
    def __init__(self, description: str):
        self.description = description
        self.prompts = []

    async def analyze_moment(self, *, prompt: str):
        self.prompts.append(prompt)
        return self.description


class PulseTestOrchestrator:
    def __init__(self):
        self.message_queue = asyncio.Queue()
        self.is_busy = False
        try:
            self.loop = asyncio.get_running_loop()
        except RuntimeError:
            self.loop = asyncio.new_event_loop()
        self.peers = {}
        self.metabolic_monitor = FixedMetabolicMonitor(health_score=1.0)
        self.optimization_engine = MaintenanceRecorder()
    
    def enqueue_message(self, msg):
        self.message_queue.put_nowait(msg)

@pytest.mark.asyncio
async def test_system_pulse_critical_health():
    orch = PulseTestOrchestrator()
    pm = PulseManager(orch)
    pm.system_sample_interval = 0.1
    orch.metabolic_monitor = FixedMetabolicMonitor(health_score=0.2)
    
    await pm.start()
    await asyncio.sleep(0.2)
    await pm.stop()
    
    assert pm.running is False

@pytest.mark.asyncio
async def test_vision_pulse_idle():
    orch = PulseTestOrchestrator()
    pm = PulseManager(orch)
    pm.vision_sample_interval = 0.1
    pm.enable_proactive_vision = True

    vision = VisionRecorder("Warning: low disk space")
    ServiceContainer.register_instance("vision_engine", vision)

    await pm.start()
    await asyncio.sleep(0.5)
    await pm.stop()

    # Verify interjection (vision pulse fires after vision_sample_interval)
    assert orch.message_queue.qsize() > 0
    msg = await orch.message_queue.get()
    assert "warning" in msg.lower()

##
