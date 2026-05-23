from __future__ import annotations

import asyncio
import os
from types import SimpleNamespace

import pytest

from core.container import ServiceContainer
from core.orchestrator.initializers import hardening
from core.runtime.errors import get_degradation_tracker
from core.runtime.health_contract import RUNTIME_CONTRACT, ServiceTier


class _Validator:
    async def run_all(self) -> bool:
        return True


class _Supervisor:
    def __init__(
        self,
        *,
        error: BaseException | None = None,
        alive_after_start: bool = True,
    ) -> None:
        self.error = error
        self.alive_after_start = alive_after_start
        self.start_calls = 0
        self.stop_calls = 0
        self._alive = False

    async def start(self) -> None:
        self.start_calls += 1
        if self.error is not None:
            raise self.error
        self._alive = self.alive_after_start

    async def stop(self) -> None:
        self.stop_calls += 1
        self._alive = False

    def is_alive(self) -> bool:
        return self._alive


class _EventLoopMonitor:
    threshold = 0.25

    def __init__(
        self,
        *,
        error: BaseException | None = None,
        alive_after_start: bool = True,
    ) -> None:
        self.error = error
        self.alive_after_start = alive_after_start
        self.start_calls = 0
        self._alive = False

    def start(self) -> None:
        self.start_calls += 1
        if self.error is not None:
            raise self.error
        self._alive = self.alive_after_start

    def is_alive(self) -> bool:
        return self._alive


@pytest.fixture(autouse=True)
def isolated_contract_state():
    ServiceContainer.clear()
    get_degradation_tracker().reset()
    yield
    ServiceContainer.clear()
    get_degradation_tracker().reset()


def _patch_dependencies(monkeypatch, *, reaper, hypervisor, monitor) -> None:
    import core.ops.hypervisor as hypervisor_module
    import core.ops.lymphatic_reaper as reaper_module
    import core.startup.validator as validator_module
    import core.utils.concurrency as concurrency_module

    monkeypatch.setattr(validator_module, "get_validator", lambda: _Validator())
    monkeypatch.setattr(reaper_module, "get_reaper", lambda: reaper)
    monkeypatch.setattr(hypervisor_module, "get_hypervisor", lambda: hypervisor)
    monkeypatch.setattr(concurrency_module, "EventLoopMonitor", lambda: monitor)


def test_hardening_initializer_leaves_failed_supervisor_unregistered_in_dev(monkeypatch):
    monkeypatch.setattr(hardening.config, "env", hardening.Environment.DEV)
    reaper = _Supervisor(error=RuntimeError("reaper spawn failed"))
    hypervisor = _Supervisor()
    monitor = _EventLoopMonitor()
    _patch_dependencies(monkeypatch, reaper=reaper, hypervisor=hypervisor, monitor=monitor)

    orchestrator = SimpleNamespace()
    asyncio.run(hardening.init_hardening_layer(orchestrator))

    assert ServiceContainer.get("reaper", default=None) is None
    assert ServiceContainer.get("hypervisor", default=None) is hypervisor
    assert ServiceContainer.get("event_loop_monitor", default=None) is monitor
    assert orchestrator.hardening_status["reaper"]["state"] == "failed"
    assert orchestrator.hardening_status["hypervisor"]["state"] == "online"
    assert "unregistered" in get_degradation_tracker().recent(subsystem="hardening")[-1].action


def test_hardening_initializer_fails_closed_on_dead_monitor_in_prod(monkeypatch):
    monkeypatch.setattr(hardening.config, "env", hardening.Environment.PROD)
    reaper = _Supervisor()
    hypervisor = _Supervisor()
    monitor = _EventLoopMonitor(alive_after_start=False)
    _patch_dependencies(monkeypatch, reaper=reaper, hypervisor=hypervisor, monitor=monitor)

    with pytest.raises(RuntimeError, match="event_loop_monitor"):
        asyncio.run(hardening.init_hardening_layer(SimpleNamespace()))

    assert ServiceContainer.get("reaper", default=None) is reaper
    assert ServiceContainer.get("hypervisor", default=None) is hypervisor
    assert ServiceContainer.get("event_loop_monitor", default=None) is None
    assert get_degradation_tracker().recent(subsystem="hardening")[-1].severity == "critical"


def test_long_run_supervisors_are_part_of_runtime_health_contract():
    required = {
        requirement.container_key: requirement
        for requirement in RUNTIME_CONTRACT
        if requirement.container_key in {"reaper", "hypervisor", "event_loop_monitor"}
    }

    assert set(required) == {"reaper", "hypervisor", "event_loop_monitor"}
    assert all(requirement.tier == ServiceTier.IMPORTANT for requirement in required.values())
    assert all(requirement.liveness_check == "is_alive" for requirement in required.values())


class _ChildProcess:
    def __init__(self, status: str, *, pid: int = 1001) -> None:
        self._status = status
        self.pid = pid
        self.terminated = False
        self.waited = False
        self.wait_timeout = object()

    def status(self) -> str:
        return self._status

    def create_time(self) -> float:
        return 0.0

    def terminate(self) -> None:
        self.terminated = True

    def wait(self, timeout: float | None = None) -> None:
        self.waited = True
        self.wait_timeout = timeout


class _CurrentProcess:
    def __init__(self, children: list[_ChildProcess]) -> None:
        self._children = children

    def children(self, *, recursive: bool = False) -> list[_ChildProcess]:
        assert recursive is True
        return list(self._children)


def test_lymphatic_reaper_retains_long_lived_children_without_opt_in(monkeypatch, tmp_path):
    import core.ops.lymphatic_reaper as reaper_module

    monkeypatch.delenv("AURA_REAPER_TERMINATE_LONG_CHILDREN", raising=False)
    child = _ChildProcess("sleeping")
    current = _CurrentProcess([child])
    monkeypatch.setattr(reaper_module.psutil, "Process", lambda: current)
    monkeypatch.setattr(reaper_module.time, "time", lambda: reaper_module.LONG_CHILD_AGE_S + 60.0)

    reaper = reaper_module.LymphaticReaper(data_dir=tmp_path)

    assert reaper._hunt_orphans() == 0
    assert child.terminated is False
    assert child.waited is False


def test_lymphatic_reaper_reaps_zombie_children(monkeypatch, tmp_path):
    import core.ops.lymphatic_reaper as reaper_module

    child = _ChildProcess(reaper_module.psutil.STATUS_ZOMBIE)
    current = _CurrentProcess([child])
    monkeypatch.setattr(reaper_module.psutil, "Process", lambda: current)

    reaper = reaper_module.LymphaticReaper(data_dir=tmp_path)

    assert reaper._hunt_orphans() == 1
    assert child.waited is True
    assert child.wait_timeout == 0
    assert child.terminated is False


def test_lymphatic_reaper_unlinks_stale_symlink_without_touching_target(monkeypatch, tmp_path):
    import core.ops.lymphatic_reaper as reaper_module

    tmp_dir = tmp_path / "tmp"
    tmp_dir.mkdir()
    target_dir = tmp_path / "target"
    target_dir.mkdir()
    link_path = tmp_dir / "old-target-link"
    link_path.symlink_to(target_dir, target_is_directory=True)
    os.utime(link_path, (1.0, 1.0), follow_symlinks=False)
    monkeypatch.setattr(reaper_module.time, "time", lambda: reaper_module.STALE_TMP_AGE_S + 2.0)

    reaper = reaper_module.LymphaticReaper(data_dir=tmp_path)

    assert reaper._filesystem_sweep() >= 0
    assert not link_path.exists()
    assert target_dir.exists()


def test_lymphatic_reaper_sweep_preserves_step_errors(monkeypatch, tmp_path):
    import core.ops.lymphatic_reaper as reaper_module

    marker = {"called": False}

    def failing_hunt() -> int:
        marker["called"] = True
        assert marker["called"] is True
        raise RuntimeError("process scan unavailable")

    reaper = reaper_module.LymphaticReaper(data_dir=tmp_path)
    monkeypatch.setattr(reaper, "_hunt_orphans", failing_hunt)
    monkeypatch.setattr(reaper, "_filesystem_sweep", lambda: 128)
    monkeypatch.setattr(reaper, "_defragment_memory", lambda: True)

    result = asyncio.run(reaper.sweep())

    assert result["processes_reaped"] == 0
    assert result["storage_reclaimed_bytes"] == 128
    assert "hunt_orphans" in result["step_errors"]
    assert reaper.get_status()["last_sweep_status"]["storage_reclaimed_bytes"] == 128


def test_metabolic_monitor_dispatches_pressure_mitigation(monkeypatch):
    import core.ops.metabolic_monitor as metabolic_module
    import core.resource.resource_governor as governor_module

    actions = []

    class _Process:
        def cpu_percent(self) -> float:
            return 145.0

        def memory_info(self) -> SimpleNamespace:
            return SimpleNamespace(rss=2048 * 1024 * 1024)

    class _Governor:
        def execute_eviction(self, tier) -> int:
            actions.append(tier.value)
            return 1

    monkeypatch.setattr(metabolic_module.psutil, "Process", lambda *_args, **_kwargs: _Process())
    monkeypatch.setattr(
        metabolic_module.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(percent=96.5, total=64 * 1024 * 1024 * 1024),
    )
    monkeypatch.setattr(metabolic_module.psutil, "disk_usage", lambda _path: SimpleNamespace(percent=72.0))
    monkeypatch.setattr(governor_module, "get_resource_governor", lambda: _Governor())

    monitor = metabolic_module.MetabolicMonitor(ram_threshold_mb=1024, cpu_threshold=80.0)
    monkeypatch.setattr(monitor, "_sync_registry", lambda _snapshot: None)

    snapshot = monitor.get_current_metabolism()

    assert snapshot.pressure_state == "critical"
    assert actions == ["aggressive"]
    assert monitor.get_status_report()["pressure_actions_total"] == 1


def test_metabolic_monitor_loop_records_failure_without_exiting_silently(monkeypatch):
    import core.ops.metabolic_monitor as metabolic_module

    monitor = metabolic_module.MetabolicMonitor()
    calls = {"count": 0}

    def failing_sample():
        calls["count"] += 1
        monitor._running = False
        assert calls["count"] == 1
        raise RuntimeError("sample unavailable")

    monkeypatch.setattr(monitor, "get_current_metabolism", failing_sample)
    monitor._interval = 0.01
    monitor._running = True

    monitor._run_loop()

    assert monitor._consecutive_failures == 1
    recent = get_degradation_tracker().recent(subsystem="metabolic_monitor")
    assert recent[-1].severity == "degraded"
    assert "backed off" in recent[-1].action


def test_compute_cost_tracker_handles_corrupt_state_and_clamps_invalid_inputs(tmp_path):
    import core.ops.metabolic_monitor as metabolic_module

    state_path = tmp_path / "metabolic_state.json"
    state_path.write_text("{not json")

    tracker = metabolic_module.PersistentComputeCostTracker(state_path=state_path)
    ergs = tracker.record_operation("invalid", -10, float("nan"))

    assert tracker.total_ergs == 0.0
    assert ergs == 0.0
    assert tracker.cost_history[-1]["tokens"] == 0
    assert tracker.cost_history[-1]["duration"] == 0.0
    assert get_degradation_tracker().count("metabolic_monitor") >= 2


def test_aegis_pulse_restores_missing_mycelial_network(monkeypatch):
    import core.mycelium as mycelium_module
    from core.orchestrator.handlers import aegis

    class _MycelialNetwork:
        def __init__(self) -> None:
            self._aegis_locked = True

    monkeypatch.setattr(mycelium_module, "MycelialNetwork", _MycelialNetwork)
    orchestrator = SimpleNamespace()

    status = asyncio.run(aegis._aegis_pulse(orchestrator))

    assert status["state"] == "restored"
    assert status["locked"] is True
    assert ServiceContainer.get("mycelial_network")._aegis_locked is True


def test_aegis_pulse_marks_integrity_failure_when_lock_restore_fails(monkeypatch):
    import core.mycelium as mycelium_module
    from core.orchestrator.handlers import aegis

    class _MycelialNetwork:
        @classmethod
        async def restore_from_vault(cls) -> bool:
            return False

    monkeypatch.setattr(mycelium_module, "MycelialNetwork", _MycelialNetwork)
    ServiceContainer.register_instance("mycelial_network", SimpleNamespace(_aegis_locked=False))
    orchestrator = SimpleNamespace()

    status = asyncio.run(aegis._aegis_pulse(orchestrator))

    assert status["state"] == "failed_closed"
    assert status["reason"] == "true_lock_restore_failed"
    assert orchestrator._aegis_integrity_failed is True
    recent = get_degradation_tracker().recent(subsystem="aegis")
    assert recent[-1].severity == "critical"


def test_aegis_loop_records_degraded_pulse_and_exits_on_stop(monkeypatch):
    from core.orchestrator.handlers import aegis

    class _StopEvent:
        def __init__(self) -> None:
            self._set = False

        def is_set(self) -> bool:
            return self._set

        def set(self) -> None:
            self._set = True

    orchestrator = SimpleNamespace(_stop_event=_StopEvent())

    async def failing_pulse(_orch, *, vault_sync_interval_s: float) -> dict:
        assert vault_sync_interval_s == 60.0
        orchestrator._stop_event.set()
        raise RuntimeError("aegis pulse unavailable")

    monkeypatch.setattr(aegis, "_aegis_pulse", failing_pulse)

    asyncio.run(aegis.aegis_sentinel_loop(orchestrator, interval_s=0.01))

    assert orchestrator.aegis_status["state"] == "degraded"
    assert orchestrator.aegis_status["consecutive_failures"] == 1
    recent = get_degradation_tracker().recent(subsystem="aegis")
    assert recent[-1].severity == "degraded"
    assert "backed off" in recent[-1].action
