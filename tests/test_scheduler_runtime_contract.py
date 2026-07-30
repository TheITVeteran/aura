from __future__ import annotations

import asyncio
import time

import pytest

from core.scheduler import Lifecycle, Scheduler, TaskSpec


@pytest.fixture
def isolated_scheduler():
    Scheduler._instance = None
    instance = Scheduler()
    yield instance
    Scheduler._instance = None


@pytest.mark.asyncio
async def test_scheduler_enforces_task_deadline_and_records_failure(isolated_scheduler):
    async def _stuck() -> None:
        await asyncio.Event().wait()

    spec = TaskSpec(
        name="critical_reconciler",
        coro=_stuck,
        tick_interval=5.0,
        timeout_s=0.01,
        critical=True,
    )

    await isolated_scheduler.register(spec)
    await isolated_scheduler._run_task(spec)

    detail = isolated_scheduler.get_health()["task_details"][spec.name]
    assert detail["status"] == "error: TimeoutError"
    assert detail["failure_count"] == 1
    assert detail["run_count"] == 1
    assert detail["last_duration_s"] < 0.5
    assert "TimeoutError" in detail["last_error"]
    assert isolated_scheduler.state is Lifecycle.RECOVERING


@pytest.mark.asyncio
async def test_scheduler_isolates_unexpected_periodic_exception(isolated_scheduler):
    async def _fails() -> None:
        raise OSError("transient telemetry failure")

    spec = TaskSpec(name="telemetry", coro=_fails, tick_interval=1.0)
    await isolated_scheduler.register(spec)

    await isolated_scheduler._run_task(spec)

    detail = isolated_scheduler.get_health()["task_details"][spec.name]
    assert detail["status"] == "error: OSError"
    assert detail["last_error"] == "OSError: transient telemetry failure"
    assert isolated_scheduler.state is Lifecycle.INITIALIZING


@pytest.mark.asyncio
async def test_scheduler_reports_task_freshness_without_breaking_legacy_status(
    isolated_scheduler,
):
    async def _complete() -> None:
        return None

    spec = TaskSpec(name="heartbeat", coro=_complete, tick_interval=1.0, timeout_s=1.0)
    await isolated_scheduler.register(spec)
    await isolated_scheduler._run_task(spec)

    health = isolated_scheduler.get_health()
    assert health["tasks"][spec.name] == "ok"
    assert health["task_details"][spec.name]["freshness"] == "fresh"

    spec.last_completed_at = time.time() - 10.0
    assert isolated_scheduler.get_health()["task_details"][spec.name]["freshness"] == "stale"


def test_task_spec_rejects_invalid_deadline():
    with pytest.raises(ValueError, match="timeout must be positive"):
        TaskSpec(name="invalid", coro=lambda: None, timeout_s=0.0)


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("tick_interval", float("nan"), "interval must be a finite number"),
        ("tick_interval", float("inf"), "interval must be a finite number"),
        ("timeout_s", float("nan"), "timeout must be a finite number"),
        ("timeout_s", float("-inf"), "timeout must be a finite number"),
    ],
)
def test_task_spec_rejects_non_finite_timing(field, value, match):
    with pytest.raises(ValueError, match=match):
        TaskSpec(name="invalid", coro=lambda: None, **{field: value})


def test_zero_interval_remains_a_bounded_every_tick_contract():
    spec = TaskSpec(name="every_tick", coro=lambda: None, tick_interval=0.0)
    assert spec.tick_interval == 0.0
