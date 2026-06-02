import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.asyncio
async def test_model_manager_evict_loop_stops_at_capacity(monkeypatch):
    from core.utils import model_manager
    from core.utils.model_manager import ModelManager

    closed: list[str] = []

    class ManagedModel:
        def __init__(self, name: str) -> None:
            self.name = name

        def close(self) -> None:
            closed.append(self.name)

    manager = ModelManager(lambda name, _opts: ManagedModel(name), max_models=1)
    manager._models["a"] = ManagedModel("a")
    manager._models["b"] = ManagedModel("b")
    manager._meta["a"] = {}
    manager._meta["b"] = {}
    manager._last_used["a"] = 1.0
    manager._last_used["b"] = 2.0

    monkeypatch.setattr(
        model_manager.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(percent=0.0),
    )

    await manager.evict_if_needed()

    assert list(manager._models) == ["b"]
    assert closed == ["a"]


@pytest.mark.asyncio
async def test_streaming_coordinator_subscriber_exits_on_close() -> None:
    from core.multimodal.coordinator import StreamingCoordinator

    coordinator = StreamingCoordinator()
    timeline = await coordinator.open_turn(turn_id="turn-test")

    async def collect_events() -> list[str]:
        events: list[str] = []
        async for event in coordinator.subscribe(timeline.turn_id):
            events.append(event.kind)
        return events

    task = asyncio.create_task(collect_events())
    await asyncio.sleep(0)
    await coordinator.emit(timeline.turn_id, "text_token", {"text": "hello"})
    await coordinator.close(timeline.turn_id)

    assert await asyncio.wait_for(task, timeout=1.0) == ["text_token"]


def test_singleton_process_metadata_records_psutil_probe_failure(monkeypatch) -> None:
    from core.utils import singleton

    calls: list[int] = []

    def fail_process(pid: int):
        calls.append(pid)
        raise OSError("process table unavailable")

    monkeypatch.setitem(sys.modules, "psutil", SimpleNamespace(Process=fail_process))

    metadata = singleton._current_process_metadata("unit-test", 123)

    assert calls == [123]
    assert metadata["identity_error"] == "OSError: process table unavailable"


def test_runtime_loop_sources_have_explicit_boundaries() -> None:
    source_expectations = {
        "core/agency/repl_daemon.py": ["while True", "except BaseException"],
        "core/utils/model_manager.py": ["while True", "pass  # no-op"],
        "core/utils/singleton.py": ["except Exception"],
        "core/multimodal/coordinator.py": ["while True", "pass  # no-op"],
    }

    for source_path, forbidden_fragments in source_expectations.items():
        source = Path(source_path).read_text(encoding="utf-8")
        for forbidden in forbidden_fragments:
            assert forbidden not in source
