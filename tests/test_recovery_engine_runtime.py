import pytest

from core.container import ServiceContainer
from core.planning.mission_state import Mission
from core.planning.recovery_engine import RecoveryEngine
from core.planning.task_graph import TaskGraph, TaskNode, TaskStatus


async def _no_sleep(_seconds: float) -> None:
    return None


class FakeMissionState:
    def __init__(self, *, execute_success: bool = True, verify_success: bool = True) -> None:
        self.execute_success = execute_success
        self.verify_success = verify_success
        self.execute_calls = 0
        self.verify_calls = 0

    async def _execute_node(self, _node: TaskNode) -> dict:
        self.execute_calls += 1
        return {"success": self.execute_success, "receipt_id": f"r-{self.execute_calls}"}

    async def _verify_node(self, _node: TaskNode) -> bool:
        self.verify_calls += 1
        return self.verify_success


def _mission_with_node(node: TaskNode) -> Mission:
    graph = TaskGraph("mission-1", "test mission")
    graph.add_node(node)
    return Mission(mission_id="mission-1", objective="test mission", graph=graph)


@pytest.mark.asyncio
async def test_recovery_retry_success_requires_post_action_verification() -> None:
    ServiceContainer.clear()
    fake_state = FakeMissionState(execute_success=True, verify_success=True)
    ServiceContainer.register_instance("mission_state", fake_state, required=False)
    node = TaskNode(task_id="n1", action="open_url", retry_count=2)
    mission = _mission_with_node(node)

    try:
        ok = await RecoveryEngine(sleep=_no_sleep).recover(mission, node, "timeout")
    finally:
        ServiceContainer.clear()

    assert ok is True
    assert fake_state.execute_calls == 1
    assert fake_state.verify_calls == 1
    assert node.retries_used == 1
    assert node.status == TaskStatus.SUCCEEDED


@pytest.mark.asyncio
async def test_backoff_recovery_rejects_unverified_success() -> None:
    ServiceContainer.clear()
    fake_state = FakeMissionState(execute_success=True, verify_success=False)
    ServiceContainer.register_instance("mission_state", fake_state, required=False)
    node = TaskNode(task_id="n1", action="open_url", retry_count=3)
    mission = _mission_with_node(node)

    try:
        ok = await RecoveryEngine(sleep=_no_sleep).recover(mission, node, "network failure")
    finally:
        ServiceContainer.clear()

    assert ok is False
    assert fake_state.execute_calls == 3
    assert fake_state.verify_calls == 3
    assert node.status != TaskStatus.SUCCEEDED


@pytest.mark.asyncio
async def test_noncritical_permission_recovery_marks_skipped_not_succeeded() -> None:
    ServiceContainer.clear()
    node = TaskNode(
        task_id="n1",
        action="set_wallpaper",
        critical=False,
        description="set wallpaper",
    )
    mission = _mission_with_node(node)

    ok = await RecoveryEngine(sleep=_no_sleep).recover(mission, node, "permission denied")

    assert ok is True
    assert node.status == TaskStatus.SKIPPED
    assert "Permission needed" in node.error


@pytest.mark.asyncio
async def test_screenshot_recovery_failure_stays_visible() -> None:
    ServiceContainer.clear()

    class FailingHost:
        def __init__(self) -> None:
            self.screen_capture_blocked = True

        async def take_screenshot(self) -> None:
            if self.screen_capture_blocked:
                raise RuntimeError("screen capture blocked")
            return None

    ServiceContainer.register_instance("host_automation", FailingHost(), required=False)
    node = TaskNode(task_id="n1", action="click")
    mission = _mission_with_node(node)
    engine = RecoveryEngine(sleep=_no_sleep)

    try:
        ok = await engine.recover(mission, node, "click failed")
    finally:
        ServiceContainer.clear()

    assert ok is False
    attempts = engine.get_recent_attempts()
    assert attempts[0]["success"] is False
    assert "Screenshot+retry failed" in attempts[0]["result"]
