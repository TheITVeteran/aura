import asyncio
from pathlib import Path
from types import SimpleNamespace

from core.managers.boot_manager import BootManager
from core.orchestrator.types import SystemStatus
from core.runtime.errors import get_degradation_tracker
from tools.audit_degradation import analyze_file


def _manager():
    orchestrator = SimpleNamespace(status=SystemStatus())
    return BootManager(orchestrator)


def test_boot_manager_degradation_audit_is_clean():
    assert analyze_file(Path("core/managers/boot_manager.py")) == []


def test_boot_degradation_is_reflected_in_status_health_metrics():
    tracker = get_degradation_tracker()
    tracker.reset()
    manager = _manager()

    manager._record_degradation(
        RuntimeError("watchdog unavailable"),
        component="system_watchdog",
        action="continued boot without system watchdog registration",
        severity="degraded",
    )

    degraded = manager.orchestrator.status.health_metrics["boot_degraded_components"]
    assert degraded[-1]["component"] == "system_watchdog"
    assert degraded[-1]["severity"] == "degraded"
    assert degraded[-1]["action"] == "continued boot without system watchdog registration"
    recent = tracker.recent(subsystem="boot_manager", limit=1)
    assert recent
    assert recent[0].action == "continued boot without system watchdog registration"


def test_boot_sequence_failure_fails_closed(monkeypatch):
    manager = _manager()
    marker = {"called": False}

    def fail_threading():
        marker["called"] = True
        raise RuntimeError("event loop unavailable")

    monkeypatch.setattr(manager, "_async_init_threading", fail_threading)

    result = asyncio.run(manager._async_init_subsystems())

    assert marker["called"] is True
    assert result is False
    assert manager.orchestrator.status.initialized is False
    assert manager.orchestrator.status.dependencies_ok is False
    assert "boot_sequence" in str(manager.orchestrator.status.last_error)
    degraded = manager.orchestrator.status.health_metrics["boot_degraded_components"]
    assert degraded[-1]["component"] == "boot_sequence"
    assert degraded[-1]["severity"] == "critical"


def test_boot_manager_starts_quarantine_monitor_without_runtime_promotion(monkeypatch):
    monkeypatch.delenv("AURA_FOREGROUND_ONLY", raising=False)
    monkeypatch.delenv("AURA_ALLOW_RUNTIME_SELF_MODIFICATION", raising=False)
    monkeypatch.delenv("AURA_ALLOW_AUTONOMOUS_PATCH_PROMOTION", raising=False)
    manager = _manager()
    manager.orchestrator.cognitive_engine = object()
    manager.orchestrator.auto_fix_enabled = True

    class FakeSelfModifier:
        def __init__(self, *_args, **_kwargs):
            self.started = False

        def runtime_promotion_enabled(self):
            return False

        def start_monitoring(self):
            self.started = True

    monkeypatch.setattr(
        "core.self_modification.self_modification_engine.AutonomousSelfModificationEngine",
        FakeSelfModifier,
    )

    manager._init_autonomous_evolution()

    assert isinstance(manager.orchestrator.self_modifier, FakeSelfModifier)
    assert manager.orchestrator.self_modifier.started is True


def test_boot_manager_keeps_monitor_active_after_runtime_promotion_opt_in(monkeypatch):
    monkeypatch.delenv("AURA_FOREGROUND_ONLY", raising=False)
    manager = _manager()
    manager.orchestrator.cognitive_engine = object()
    manager.orchestrator.auto_fix_enabled = True

    class FakeSelfModifier:
        def __init__(self, *_args, **_kwargs):
            self.started = False

        def runtime_promotion_enabled(self):
            return True

        def start_monitoring(self):
            self.started = True

    monkeypatch.setattr(
        "core.self_modification.self_modification_engine.AutonomousSelfModificationEngine",
        FakeSelfModifier,
    )

    manager._init_autonomous_evolution()

    assert isinstance(manager.orchestrator.self_modifier, FakeSelfModifier)
    assert manager.orchestrator.self_modifier.started is True


def test_boot_manager_disables_self_modification_engine_for_foreground_only_boot(monkeypatch):
    monkeypatch.setenv("AURA_FOREGROUND_ONLY", "1")
    manager = _manager()
    manager.orchestrator.cognitive_engine = object()
    manager.orchestrator.auto_fix_enabled = True

    class FakeSelfModifier:
        def __init__(self, *_args, **_kwargs):
            self.construct_attempted = True
            raise AssertionError("foreground-only boot must not construct self-modification engine")

    monkeypatch.setattr(
        "core.self_modification.self_modification_engine.AutonomousSelfModificationEngine",
        FakeSelfModifier,
    )

    manager._init_autonomous_evolution()

    assert manager.orchestrator.self_modifier is None
