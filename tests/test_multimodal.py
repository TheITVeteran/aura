import pytest
import asyncio
from types import SimpleNamespace

from core.brain.multimodal_orchestrator import MultimodalOrchestrator
from core.container import ServiceContainer
from core.utils.task_tracker import get_task_tracker


class AsyncCallRecorder:
    def __init__(self, result=None):
        self.result = result
        self.calls = []

    async def __call__(self, *args, **kwargs):
        self.calls.append(SimpleNamespace(args=args, kwargs=kwargs))
        return self.result

@pytest.mark.asyncio
async def test_multimodal_render_sync():
    voice_speak = AsyncCallRecorder()
    bus_publish = AsyncCallRecorder()

    ServiceContainer.clear()
    ServiceContainer.register_instance("voice_engine", SimpleNamespace(speak=voice_speak))
    ServiceContainer.register_instance("input_bus", SimpleNamespace(publish=bus_publish))

    try:
        orchestrator = MultimodalOrchestrator()

        await orchestrator.render("I am so happy to see you!", metadata={"voice": True})

        await asyncio.sleep(0.1)
        await get_task_tracker().shutdown(timeout=1.0)

        assert len(voice_speak.calls) == 1
        assert voice_speak.calls[0].args == ("I am so happy to see you!",)

        assert len(bus_publish.calls) == 1
        assert bus_publish.calls[0].args[0] == "aura/expression"
        assert bus_publish.calls[0].args[1]["expression"] == "joy"
    finally:
        ServiceContainer.clear()

@pytest.mark.asyncio
async def test_heuristic_expressions():
    orchestrator = MultimodalOrchestrator()
    assert orchestrator._heuristic_expression("I am happy") == "joy"
    assert orchestrator._heuristic_expression("This is an error") == "alert"
    assert orchestrator._heuristic_expression("I am sorry") == "sad"
    assert orchestrator._heuristic_expression("I am neutral") == "neutral"
