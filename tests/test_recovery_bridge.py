"""Recovery bridge — faults actuate their cataloged strategy.

The RecoveryStrategy column was descriptive; now qualifying fault
records flow to the immune system (auto lane) or become operator
recommendations with runbook links (operator lane). Recording stays
O(1): the listener only enqueues.
"""
from __future__ import annotations

import time

from core.resilience.fault_taxonomy import FaultRegistry
from core.resilience.recovery_bridge import RecoveryBridge


def _wait(predicate, timeout_s: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def _started_bridge(monkeypatch) -> RecoveryBridge:
    monkeypatch.setenv("AURA_RECOVERY_BRIDGE", "1")
    bridge = RecoveryBridge()
    assert bridge.start() is True
    return bridge


class TestRouting:
    def test_auto_strategy_reaches_immune_and_marks_recovered(self, monkeypatch):
        calls = []

        class _Immune:
            def assess_and_respond(self, *, source, description):
                calls.append((source, description))
                return type("R", (), {"action": "observed"})()

        monkeypatch.setattr(
            "core.security.immune_system.get_immune_system", lambda: _Immune(),
        )
        bridge = _started_bridge(monkeypatch)
        registry = FaultRegistry()
        monkeypatch.setattr(
            "core.resilience.fault_taxonomy.get_fault_registry", lambda: registry,
        )
        registry.add_listener(bridge.on_fault)

        # F07 (resource exhaustion) → graceful_degradation → AUTO lane.
        record = registry.record_fault("F07", "test.resource", details="RAM pressure")
        assert _wait(lambda: calls), "immune system never consulted"
        assert "F07" in calls[0][1]
        assert _wait(lambda: record.recovered), "record not marked recovered"

    def test_operator_strategy_recommends_never_executes(self, monkeypatch):
        immune_calls = []
        monkeypatch.setattr(
            "core.security.immune_system.get_immune_system",
            lambda: (_ for _ in ()).throw(AssertionError("must not be called")),
        )
        bridge = _started_bridge(monkeypatch)
        registry = FaultRegistry()
        monkeypatch.setattr(
            "core.resilience.fault_taxonomy.get_fault_registry", lambda: registry,
        )
        registry.add_listener(bridge.on_fault)

        # F03 (memory corruption) → stem_cell_revert → OPERATOR lane.
        registry.record_fault("F03", "test.storage", details="integrity check failed")
        assert _wait(
            lambda: bridge.status()["operator_recommendations"] >= 1
        ), "no operator recommendation surfaced"
        assert not immune_calls

    def test_negligible_and_ignore_never_enter(self, monkeypatch):
        bridge = _started_bridge(monkeypatch)
        registry = FaultRegistry()
        monkeypatch.setattr(
            "core.resilience.fault_taxonomy.get_fault_registry", lambda: registry,
        )
        registry.add_listener(bridge.on_fault)

        registry.record_fault("WILL-REFUSE", "will.test", recovered=True)  # NEGLIGIBLE
        registry.record_fault("F13", "obs.logs")  # IGNORE strategy
        time.sleep(0.1)
        status = bridge.status()
        assert status["enqueued"] == 0

    def test_cooldown_prevents_response_storms(self, monkeypatch):
        monkeypatch.setattr(
            "core.security.immune_system.get_immune_system",
            lambda: type("I", (), {"assess_and_respond": lambda self, **k: None})(),
        )
        bridge = _started_bridge(monkeypatch)
        registry = FaultRegistry()
        monkeypatch.setattr(
            "core.resilience.fault_taxonomy.get_fault_registry", lambda: registry,
        )
        registry.add_listener(bridge.on_fault)

        for _ in range(10):
            registry.record_fault("F07", "test.resource", details="storm")
        assert _wait(lambda: bridge.status()["enqueued"] >= 1)
        status = bridge.status()
        assert status["enqueued"] == 1, "cooldown must collapse the storm"
        assert status["cooldown_skips"] == 9

    def test_kill_switch(self, monkeypatch):
        monkeypatch.setenv("AURA_RECOVERY_BRIDGE", "0")
        bridge = RecoveryBridge()
        assert bridge.start() is False
        assert bridge.status()["started"] is False


class TestListenerSeam:
    def test_listener_bounded_and_deduped(self):
        registry = FaultRegistry()
        fn = lambda record: None  # noqa: E731
        assert registry.add_listener(fn) is True
        assert registry.add_listener(fn) is False  # dedupe
        for i in range(10):
            registry.add_listener(lambda r, _i=i: None)
        assert registry.add_listener(lambda r: None) is False  # bounded at 8

    def test_failing_listener_never_breaks_recording(self):
        registry = FaultRegistry()

        def boom(record):
            raise RuntimeError("listener failure")

        registry.add_listener(boom)
        record = registry.record_fault("F01", "test")  # must not raise
        assert record.fault_id == "F01"
