"""Self-repair backlog — defect registers become approval-gated repair goals.

Guarantees pinned here: read-only ingestion, idempotency, approval-gated +
shadow plan creation, and fail-closed behavior when no engine is present.
"""
from __future__ import annotations

import asyncio
import json

from core.agency.self_repair_backlog import SelfRepairBacklog


def _register(tmp_path, order=(), real=()):
    path = tmp_path / "defect_register.json"
    path.write_text(
        json.dumps(
            {
                "schema": "aura.test_defect_register.v1",
                "order_dependent": list(order),
                "real_failures": list(real),
            }
        ),
        encoding="utf-8",
    )
    return path


def test_parse_register_is_pure(tmp_path):
    reg = _register(tmp_path, order=["tests/test_a.py::test_x"], real=["tests/test_b.py::test_y"])
    backlog = SelfRepairBacklog(seen_path=tmp_path / "seen.json")
    items = backlog.parse_register(reg)
    kinds = {it.kind for it in items}
    assert kinds == {"order_dependence", "real_failure"}
    order_item = next(it for it in items if it.kind == "order_dependence")
    assert "ORDER-DEPENDENCE" in order_item.goal
    assert "Do not weaken the assertion" in order_item.goal
    assert not (tmp_path / "seen.json").exists()  # pure — nothing written


def test_new_items_are_idempotent(tmp_path):
    reg = _register(tmp_path, order=["tests/test_a.py::test_x"])
    backlog = SelfRepairBacklog(seen_path=tmp_path / "seen.json")
    first = backlog.new_items(reg)
    assert len(first) == 1
    # dry-run enqueue marks them seen
    asyncio.run(backlog.enqueue_repairs(reg, dry_run=True))
    assert backlog.new_items(reg) == []


def test_stable_defect_ids(tmp_path):
    reg = _register(tmp_path, real=["tests/test_b.py::test_y"])
    b1 = SelfRepairBacklog(seen_path=tmp_path / "s1.json").parse_register(reg)
    b2 = SelfRepairBacklog(seen_path=tmp_path / "s2.json").parse_register(reg)
    assert b1[0].defect_id == b2[0].defect_id


def test_enqueue_creates_approval_gated_shadow_plans(tmp_path):
    reg = _register(tmp_path, order=["tests/test_a.py::test_x"])
    backlog = SelfRepairBacklog(seen_path=tmp_path / "seen.json")

    calls: list[dict] = []

    class _Engine:
        async def execute_goal(self, goal, context=None, is_shadow=False):
            calls.append({"goal": goal, "context": context, "is_shadow": is_shadow})
            from types import SimpleNamespace

            return SimpleNamespace(status="waiting_for_approval")

    results = asyncio.run(
        backlog.enqueue_repairs(reg, task_engine=_Engine(), auto_execute=False)
    )
    assert len(results) == 1
    assert results[0]["created"] is True
    assert calls[0]["is_shadow"] is True
    assert calls[0]["context"]["requires_approval"] is True
    assert calls[0]["context"]["origin"] == "self_repair_backlog"


def test_no_engine_marks_seen_without_creating(tmp_path):
    reg = _register(tmp_path, real=["tests/test_b.py::test_y"])
    backlog = SelfRepairBacklog(seen_path=tmp_path / "seen.json")
    results = asyncio.run(backlog.enqueue_repairs(reg, task_engine=None, dry_run=True))
    assert results and all(r["created"] is False for r in results)
    # Marked seen — no duplicate goals on the next run.
    assert backlog.new_items(reg) == []


def test_enqueue_auto_executes_safe_repair_requests(tmp_path):
    from core.resilience.autonomous_repair_executor import (
        set_autonomous_repair_executor_for_tests,
    )

    reg = _register(tmp_path, real=["tests/test_b.py::test_y"])
    backlog = SelfRepairBacklog(seen_path=tmp_path / "seen.json")

    class _Executor:
        def __init__(self):
            self.requests = []

        def enqueue_background(self, request):
            self.requests.append(request)
            return {"status": "scheduled", "fingerprint": request.fingerprint}

    executor = _Executor()
    set_autonomous_repair_executor_for_tests(executor)
    try:
        results = asyncio.run(backlog.enqueue_repairs(reg))
    finally:
        set_autonomous_repair_executor_for_tests(None)

    assert len(results) == 1
    assert results[0]["created"] is True
    assert results[0]["auto_execute"] is True
    assert executor.requests[0].subsystem == "self_repair_backlog"
    assert executor.requests[0].context["origin"] == "self_repair_backlog"
    assert backlog.new_items(reg) == []


def test_runner_emits_register_contract():
    """The chunk runner writes the register the ingestor consumes."""
    import inspect

    import tools.run_test_chunks as runner

    src = inspect.getsource(runner)
    assert "--defect-register" in src
    assert '"order_dependent": order_dependent' in src
    assert '"real_failures": real_failures' in src
    assert "aura.test_defect_register.v1" in src
