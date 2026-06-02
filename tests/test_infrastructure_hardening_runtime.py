import time
from pathlib import Path

import pytest

from infrastructure import hardening


@pytest.mark.asyncio
async def test_retry_policy_does_not_retry_programmer_assertion() -> None:
    attempts = 0
    policy = hardening.RetryPolicy(max_retries=3, base_delay=0.0)

    async def fail_with_programmer_error():
        nonlocal attempts
        attempts += 1
        raise AssertionError("invariant broken")

    with pytest.raises(AssertionError, match="invariant broken"):
        await policy.execute(fail_with_programmer_error)

    assert attempts == 1


@pytest.mark.asyncio
async def test_retry_policy_retries_recoverable_operational_error() -> None:
    attempts = 0
    policy = hardening.RetryPolicy(max_retries=3, base_delay=0.0)

    async def fail_then_recover():
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise OSError("temporary filesystem pressure")
        return "ok"

    assert await policy.execute(fail_then_recover) == "ok"
    assert attempts == 3


def test_state_manager_checkpoint_thread_runs_and_stops(tmp_path: Path) -> None:
    manager = hardening.StateManager(checkpoint_dir=str(tmp_path))
    manager.checkpoint_interval = 0.01
    manager.last_checkpoint = time.time()
    manager.set("boot", "ok")

    manager.start_auto_checkpoint()
    time.sleep(0.05)
    manager.stop_auto_checkpoint()

    assert manager.checkpoint_thread is not None
    assert not manager.checkpoint_thread.is_alive()
    assert list(tmp_path.glob("checkpoint_*.json"))


def test_health_monitor_thread_runs_probe_and_stops() -> None:
    monitor = hardening.HealthMonitor()
    monitor.check_interval = 0.01
    probes = 0

    def health_probe() -> bool:
        nonlocal probes
        probes += 1
        return True

    monitor.register_component("kernel", health_probe)
    monitor.start_monitoring()
    time.sleep(0.05)
    monitor.stop_monitoring()

    assert monitor.monitor_thread is not None
    assert not monitor.monitor_thread.is_alive()
    assert probes > 0
    assert monitor.components["kernel"] is hardening.ComponentState.HEALTHY


def test_health_monitor_marks_assertion_probe_failed() -> None:
    monitor = hardening.HealthMonitor()
    monitor.register_component(
        "tool_governance",
        lambda: (_ for _ in ()).throw(AssertionError("governance probe failed")),
    )

    result = monitor.check_health("tool_governance")

    assert result.state is hardening.ComponentState.FAILED
    assert result.error == "governance probe failed"


def test_sync_resilient_wrapper_does_not_rerun_raw_function_after_failure() -> None:
    system = hardening.InfrastructureHardeningSystem()
    hardening.set_global_hardening_system(system)
    calls = 0

    @hardening.resilient("sync-runtime", retry=False, circuit_breaker=False)
    def failing_sync_call():
        nonlocal calls
        calls += 1
        raise OSError("runtime unavailable")

    try:
        with pytest.raises(OSError, match="runtime unavailable"):
            failing_sync_call()
    finally:
        hardening.set_global_hardening_system(None)  # type: ignore[arg-type]

    assert calls == 1


def test_infrastructure_hardening_has_no_broad_exception_boundaries() -> None:
    source = Path("infrastructure/hardening.py").read_text(encoding="utf-8")

    assert "except Exception" not in source
    assert "raise Exception" not in source
    assert "async def _checkpoint_loop" not in source
    assert "async def _monitor_loop" not in source
