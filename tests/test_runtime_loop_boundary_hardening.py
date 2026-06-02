import asyncio
import queue
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest


def _content_item(title: str):
    from core.autonomy.curated_media_loader import ContentItem

    return ContentItem(
        category="Science education",
        title=title,
        creator=None,
        url=None,
        description="runtime contract topic",
    )


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
        "core/resilience/immune_system.py": ["while True"],
        "core/sandbox/bash_daemon.py": ["while True"],
        "core/senses/sensory_worker.py": ["while True", "except Exception"],
    }

    for source_path, forbidden_fragments in source_expectations.items():
        source = Path(source_path).read_text(encoding="utf-8")
        for forbidden in forbidden_fragments:
            assert forbidden not in source


def test_research_trigger_payload_is_json_safe(tmp_path) -> None:
    from core.autonomy import research_triggers

    class NonSerializable:
        def __repr__(self) -> str:
            return "<non-serializable-marker>"

    trigger_path = tmp_path / "research-triggers.jsonl"
    research_triggers.emit_research_trigger(
        topic="runtime payload hygiene",
        source_intent_id="intent-runtime",
        payload_hint={"object": NonSerializable(), "nested": {"set": {1, 2}}},
        path=trigger_path,
    )

    [trigger] = research_triggers.drain_pending_triggers(path=trigger_path)
    assert trigger.payload_hint["object"] == "<non-serializable-marker>"
    assert trigger.payload_hint["nested"]["set"] == "{1, 2}"


def test_curiosity_scheduler_degrades_known_reader_errors_but_surfaces_invariants() -> None:
    from core.autonomy.content_progress_tracker import ProgressLog
    from core.autonomy.curiosity_scheduler import CuriosityScheduler

    def typed_failure_reader() -> dict[str, float]:
        observed = "substrate unavailable"
        raise RuntimeError(observed)

    scheduler = CuriosityScheduler(
        corpus_loader=lambda: [_content_item("bounded curiosity")],
        progress_loader=lambda: ProgressLog(),
        substrate_reader=typed_failure_reader,
        trigger_drainer=lambda: [],
    )

    assert scheduler.pick_next() is not None

    invariant_calls: list[str] = []

    def invariant_failure_reader() -> dict[str, float]:
        invariant_calls.append("called")
        raise AssertionError("substrate invariant broken")

    invariant_scheduler = CuriosityScheduler(
        corpus_loader=lambda: [_content_item("surfaced invariant")],
        progress_loader=lambda: ProgressLog(),
        substrate_reader=invariant_failure_reader,
        trigger_drainer=lambda: [],
    )

    with pytest.raises(AssertionError, match="substrate invariant broken"):
        invariant_scheduler.pick_next()
    assert invariant_calls == ["called"]


@pytest.mark.asyncio
async def test_process_tcell_patrol_has_stop_contract_and_cancels_stale_named_tasks() -> None:
    from core.resilience.immune_system import ProcessTCell

    async def sleeper() -> None:
        await asyncio.sleep(30)

    task = asyncio.create_task(sleeper(), name="runtime_stale_task")
    tcell = ProcessTCell(max_lifespan_seconds=0.1, patrol_interval=0.001)
    tcell._first_seen[id(task)] = time.monotonic() - 1.0

    assert tcell.patrol_once(current_task=asyncio.current_task()) == 1
    with pytest.raises(asyncio.CancelledError):
        await task

    await asyncio.wait_for(tcell.patrol_bloodstream(max_cycles=1), timeout=0.5)


@pytest.mark.asyncio
async def test_bash_daemon_reader_exits_on_delimiter_without_forever_loop() -> None:
    from core.sandbox.bash_daemon import PersistentBashSession

    class FakeStdout:
        def __init__(self, lines: list[bytes]) -> None:
            self._items = queue.SimpleQueue()
            for line in lines:
                self._items.put(line)

        async def readline(self) -> bytes:
            if self._items.empty():
                return b""
            return self._items.get()

    session = PersistentBashSession(cwd="/tmp")
    session._process = SimpleNamespace(
        stdout=FakeStdout([b"hello\n", f"{session._delimiter}:0\n".encode()])
    )

    assert await session._read_until_delimiter() == ("hello", 0)


@pytest.mark.asyncio
async def test_bash_daemon_execute_preserves_legacy_timeout_keyword(monkeypatch) -> None:
    from core.sandbox.bash_daemon import PersistentBashSession

    class FakeStdin:
        def __init__(self) -> None:
            self.writes: list[bytes] = []

        def write(self, payload: bytes) -> None:
            self.writes.append(payload)

        async def drain(self) -> None:
            return None

    process = SimpleNamespace(stdin=FakeStdin(), returncode=None)
    session = PersistentBashSession(cwd="/tmp")
    session._process = process
    monkeypatch.setattr(session, "_read_until_delimiter", lambda: asyncio.sleep(0, result=("ok", 0)))

    assert await session.execute("echo ok", timeout=0.1) == (True, "ok")
    assert process.stdin.writes == [b"echo ok\n"]


def test_sensory_worker_ping_and_exit_contract() -> None:
    from core.senses.sensory_worker import sensory_worker_loop

    request_queue: queue.Queue[dict[str, str]] = queue.Queue()
    response_queue: queue.Queue[dict[str, str]] = queue.Queue()
    request_queue.put({"command": "ping"})
    request_queue.put({"command": "exit"})

    sensory_worker_loop(request_queue, response_queue)

    assert response_queue.get(timeout=0.1) == {"status": "ok", "msg": "pong"}
