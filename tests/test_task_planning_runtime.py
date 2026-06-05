from __future__ import annotations

import pytest

from core.container import ServiceContainer
from core.planning.mission_state import MissionState
from core.planning.task_decomposer import TaskDecomposer
from core.planning.task_graph import TaskGraph, TaskNode, TaskStatus


@pytest.mark.asyncio
async def test_task_decomposer_builds_general_graph_without_llm() -> None:
    ServiceContainer.clear()
    graph = await TaskDecomposer().decompose(
        "Create a journal note, export it as a PDF, then search web articles about climate news."
    )

    actions = [node.action for node in graph.nodes.values()]

    assert "create_folder" in actions
    assert "create_text_file" in actions
    assert "create_pdf" in actions
    assert "search_web" in actions
    assert graph.validate() == []
    assert all(node.verification for node in graph.nodes.values())


def test_task_graph_dependency_readiness_and_completion() -> None:
    graph = TaskGraph("mission-test", "Open app and write")
    graph.add_node(TaskNode(task_id="t1", action="launch_app"))
    graph.add_node(TaskNode(task_id="t2", action="type_text", preconditions=["t1"]))

    assert [node.task_id for node in graph.get_ready_nodes()] == ["t1"]
    graph.mark_succeeded("t1", result={"ok": True}, receipt_id="r1")

    ready = graph.get_ready_nodes()

    assert [node.task_id for node in ready] == ["t2"]
    graph.mark_running("t2")
    assert graph.nodes["t2"].status == TaskStatus.RUNNING
    graph.mark_succeeded("t2")
    assert graph.is_successful is True


def test_task_graph_persist_uses_file_write_gateway(monkeypatch, tmp_path) -> None:
    from core.planning import task_graph as module

    writes: list[dict[str, object]] = []

    class FakeGateway:
        def write_text(self, path, text, *, encoding="utf-8", source="unknown"):
            writes.append(
                {
                    "path": path,
                    "text": text,
                    "encoding": encoding,
                    "source": source,
                }
            )

    monkeypatch.setattr(module, "get_file_write_gateway", lambda: FakeGateway())
    graph = TaskGraph("mission-persist", "Persist")
    graph.add_node(TaskNode(task_id="t1", action="observe"))
    target = tmp_path / "state" / "mission.json"

    graph.persist(target)

    assert writes
    assert writes[0]["path"] == target
    assert writes[0]["encoding"] == "utf-8"
    assert writes[0]["source"] == "task_graph.persist"
    assert '"mission_id": "mission-persist"' in str(writes[0]["text"])


@pytest.mark.asyncio
async def test_mission_state_create_text_file_uses_file_write_gateway(monkeypatch, tmp_path) -> None:
    from core.planning import mission_state as module

    writes: list[dict[str, object]] = []

    class FakeGateway:
        def write_text(self, path, text, *, encoding="utf-8", source="unknown"):
            writes.append(
                {
                    "path": path,
                    "text": text,
                    "encoding": encoding,
                    "source": source,
                }
            )
            path.write_text(text, encoding=encoding)

    monkeypatch.setattr(module, "get_file_write_gateway", lambda: FakeGateway())
    node = TaskNode(
        task_id="t1",
        action="create_text_file",
        params={"path": str(tmp_path / "note.txt"), "content": "hello"},
    )

    result = await MissionState(data_dir=str(tmp_path / "missions"))._execute_node(node)

    assert result["success"] is True
    assert writes[0]["source"] == "mission_state.create_text_file"
    assert writes[0]["text"] == "hello"


@pytest.mark.asyncio
async def test_mission_state_verifier_failure_blocks_success(tmp_path) -> None:
    ServiceContainer.clear()

    class FailingVerifier:
        def __init__(self) -> None:
            self.calls = 0

        async def verify(self, *_args, **_kwargs):
            self.calls += 1
            raise RuntimeError("verifier offline")

    verifier = FailingVerifier()
    ServiceContainer.register_instance("post_action_verifier", verifier, required=False)
    node = TaskNode(task_id="t1", action="launch_app", verification="app_is_frontmost")

    try:
        ok = await MissionState(data_dir=str(tmp_path / "missions"))._verify_node(node)
    finally:
        ServiceContainer.clear()

    assert ok is False
    assert verifier.calls == 1
    assert node.verification_result is not None
    assert node.verification_result["success"] is False
    assert "verifier offline" in str(node.verification_result["evidence"])
